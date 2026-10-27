"""Two follow-ups to analysis.py (run that first).

1. Learned combination (stacking). For each held-out woman, the seven clocks' predictions
   are combined with non-negative weights learned only from the other 16 women.
2. Identity leak demo. Each woman gets a random "fake age", the same at all her visits.
   No blood signal can predict a random number, so any success means the model is
   recognizing the person, not reading the pregnancy.

Writes results/extras.json.
"""
from pathlib import Path
import json
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import N_JOBS, OUT_DIR, SEED, fit_predict, outer_cv, prepare, scores  # noqa: E402
from load_data import MODALITIES, load  # noqa: E402

N_STACK_FOLDS = 4
LEAK_MODALITIES = ["plasma_somalogic", "immune_system"]
LEAK_REPEATS = 5


def inner_oof(X, y, groups, train):
    """Out-of-fold predictions for the training women only, by grouped CV."""
    pred = np.empty(len(train))
    for tr, va in GroupKFold(n_splits=N_STACK_FOLDS).split(train, groups=groups[train]):
        _, p, _ = fit_predict(X, y, groups, train[tr], train[va])
        pred[va] = p
    return pred


def stacking(Xs, y, groups, test_preds):
    women = np.unique(groups)

    def one_woman(g):
        train = np.where(groups != g)[0]
        test = np.where(groups == g)[0]
        Z = np.column_stack([inner_oof(Xs[m], y, groups, train) for m in MODALITIES])
        combiner = LinearRegression(positive=True).fit(Z, y[train])
        Z_test = np.column_stack([test_preds[m][test] for m in MODALITIES])
        return test, combiner.predict(Z_test), combiner.coef_

    results = Parallel(n_jobs=N_JOBS)(delayed(one_woman)(g) for g in women)
    pred = np.empty(len(y))
    weights = []
    for test, p, w in results:
        pred[test] = p
        weights.append(w)
    weights = np.array(weights)
    return pred, dict(zip(MODALITIES, weights.mean(axis=0).tolist()))


def identity_leak(Xs, groups):
    rng = np.random.default_rng(SEED)
    women = np.unique(groups)
    out = {}
    for m in LEAK_MODALITIES:
        rs = {"woman": [], "sample": []}
        # Error relative to always guessing the average fake age: 1.0 = no better than guessing.
        rel = {"woman": [], "sample": []}
        guess_mae = []
        for _ in range(LEAK_REPEATS):
            fake = dict(zip(women, rng.uniform(0, 40, size=len(women))))
            y_fake = np.array([fake[g] for g in groups])
            guess = np.abs(y_fake - y_fake.mean()).mean()
            guess_mae.append(float(guess))
            for scheme in rs:
                pred = outer_cv(Xs[m], y_fake, groups, scheme)
                rs[scheme].append(float(pearsonr(y_fake, pred)[0]))
                rel[scheme].append(float(np.abs(y_fake - pred).mean() / guess))
        out[m] = {"pearson_r": rs, "relative_mae": rel, "guess_mae": guess_mae}
        print(f"identity leak {m:18s} woman r={np.mean(rs['woman']):+.2f} rel={np.mean(rel['woman']):.2f}  "
              f"sample r={np.mean(rs['sample']):+.2f} rel={np.mean(rel['sample']):.2f}", flush=True)
    return out


def main():
    d = load()
    preg = d["timepoint"] < 4
    y, groups = d["weeks"][preg], d["patient"][preg]
    Xs = {m: prepare(d[m][preg], m)[0] for m in MODALITIES}
    results = json.loads((OUT_DIR / "results.json").read_text())
    test_preds = {m: np.array(results["modalities"][m]["pred_by_woman"]) for m in MODALITIES}

    stacked, weights = stacking(Xs, y, groups, test_preds)
    s = scores(y, stacked)
    print(f"stacked  MAE={s['mae']:.2f} r={s['pearson_r']:.2f}  weights="
          + ", ".join(f"{m}:{w:.2f}" for m, w in weights.items()), flush=True)

    out = {
        "stacked": {"by_woman": s, "pred_by_woman": stacked.tolist(), "mean_weights": weights},
        "identity_leak": identity_leak(Xs, groups),
    }
    (OUT_DIR / "extras.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
