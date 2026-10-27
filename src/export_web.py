"""Pack the results into one small JSON file for the portfolio page.

    python src/export_web.py [output path]

Women are relabeled W1..W17; study IDs are not published.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import OUT_DIR  # noqa: E402
from load_data import MODALITIES  # noqa: E402

DEFAULT_OUT = Path.home() / "Documents/Development/main_portfolio_website/static/data/pregnancy_clock.json"
BEST = "plasma_somalogic"
FEATURE_MODALITIES = ["plasma_somalogic", "immune_system"]


def r2(x):
    return round(float(x), 2)


def main(out_path):
    res = json.loads((OUT_DIR / "results.json").read_text())
    extras = json.loads((OUT_DIR / "extras.json").read_text())
    details = {m: json.loads((OUT_DIR / f"details_{m}.json").read_text()) for m in FEATURE_MODALITIES}

    patients = res["samples"]["patient"]
    alias = {p: f"W{i + 1}" for i, p in enumerate(dict.fromkeys(patients))}

    modalities = sorted(
        ({"key": m, "label": res["modalities"][m]["label"],
          "n_features": res["modalities"][m]["n_features"],
          "mae": r2(res["modalities"][m]["by_woman"]["mae"]),
          "r": r2(res["modalities"][m]["by_woman"]["pearson_r"])} for m in MODALITIES),
        key=lambda x: x["mae"])

    best_pred = res["modalities"][BEST]["pred_by_woman"]
    postpartum = {alias[p["patient"]]: r2(p["predicted_weeks"]) for p in details[BEST]["postpartum"]}
    women = {}
    for p, w, t, pred in zip(patients, res["samples"]["weeks"], res["samples"]["timepoint"], best_pred):
        women.setdefault(alias[p], []).append({"visit": t, "actual": r2(w), "predicted": r2(pred)})

    leak = extras["identity_leak"]
    out = {
        "n_women": res["n_women"],
        "n_samples": res["n_samples"],
        "baseline_mae": r2(res["baseline"]["mae"]),
        "modalities": modalities,
        "best": {
            "key": BEST,
            "label": res["modalities"][BEST]["label"],
            "mae": r2(res["modalities"][BEST]["by_woman"]["mae"]),
            "r": r2(res["modalities"][BEST]["by_woman"]["pearson_r"]),
            "leaky_mae": r2(res["modalities"][BEST]["by_sample"]["mae"]),
        },
        "women": [{"id": k, "visits": v, "postpartum_reading": postpartum[k]} for k, v in women.items()],
        "combine": {
            "best_single": r2(res["modalities"][BEST]["by_woman"]["mae"]),
            "simple_average": r2(res["combined"]["by_woman"]["mae"]),
            "learned_weights": r2(extras["stacked"]["by_woman"]["mae"]),
            "weights": {res["modalities"][m]["label"]: r2(w) for m, w in extras["stacked"]["mean_weights"].items()},
        },
        "identity_leak": {
            res["modalities"][m]["label"]: {
                "honest": r2(np.mean(v["relative_mae"]["woman"])),
                "leaky": r2(np.mean(v["relative_mae"]["sample"])),
            } for m, v in leak.items()
        },
        "features": {
            m: {"label": res["modalities"][m]["label"],
                "n_selected_median": details[m]["n_features_selected_median"],
                "top": [{"name": f["name"], "weight": r2(f["mean_coef"]),
                         "kept": r2(f["selected_fraction"])} for f in details[m]["top_features"]]}
            for m in FEATURE_MODALITIES
        },
    }
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
