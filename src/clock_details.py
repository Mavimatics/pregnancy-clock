"""Deeper look at one modality's clock: what happens after birth, and what it reads.

    python src/clock_details.py immune_system

Postpartum: each woman's after-birth sample is scored by a clock trained only on the
other 16 women's pregnancy samples. Features: how often each feature is kept across
the 17 leave-one-woman-out fits, and its average (standardized) weight.
Writes results/details_<modality>.json.
"""
from pathlib import Path
import json
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import N_JOBS, OUT_DIR, fit_predict, prepare  # noqa: E402
from load_data import load  # noqa: E402

TOP_N = 12


def main(modality):
    d = load()
    X_all, keep = prepare(d[modality], modality)
    names = d[f"{modality}__features"][keep]
    preg = d["timepoint"] < 4
    post = ~preg

    X, y, groups = X_all[preg], d["weeks"][preg], d["patient"][preg]
    women = np.unique(groups)

    def one_woman(g):
        train = np.where(groups != g)[0]
        test = np.where(groups == g)[0]
        _, _, model = fit_predict(X, y, groups, train, test)
        x_post = X_all[post & (d["patient"] == g)]
        return g, float(model.predict(x_post)[0]), model[-1].coef_

    results = Parallel(n_jobs=N_JOBS)(delayed(one_woman)(g) for g in women)

    coefs = np.array([c for _, _, c in results])
    selected = (coefs != 0).mean(axis=0)
    mean_coef = coefs.mean(axis=0)
    order = np.lexsort((-np.abs(mean_coef), -selected))[:TOP_N]

    out = {
        "modality": modality,
        "postpartum": [
            {"patient": g, "predicted_weeks": p,
             "actual_weeks_after_delivery": float(d["weeks"][post & (d["patient"] == g)][0])}
            for g, p, _ in results
        ],
        "n_features_selected_median": int(np.median((coefs != 0).sum(axis=1))),
        "top_features": [
            {"name": str(names[i]), "selected_fraction": float(selected[i]), "mean_coef": float(mean_coef[i])}
            for i in order
        ],
    }
    (OUT_DIR / f"details_{modality}.json").write_text(json.dumps(out, indent=1))
    pp = np.array([r["predicted_weeks"] for r in out["postpartum"]])
    print(f"postpartum predicted weeks: median {np.median(pp):.1f}, range {pp.min():.1f} to {pp.max():.1f}")
    print(f"features kept per fit (median): {out['n_features_selected_median']}")
    for f in out["top_features"]:
        print(f"  {f['name']:45s} kept {f['selected_fraction']:.0%}  weight {f['mean_coef']:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "immune_system")
