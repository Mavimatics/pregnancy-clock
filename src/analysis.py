"""Can a blood sample tell how far along a pregnancy is?

Every model here is an elastic net (as in Ghaemi et al., 2019), evaluated with
cross-validation that holds out whole women, so a model is always scored on a
person it has never seen. Writes results/results.json for the website.
"""
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import MODALITIES, load  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
L1_RATIOS = [0.2, 0.5, 1.0]
N_INNER_FOLDS = 4
SEED = 0
# Cores to use. Kept low so the laptop stays quiet; raise with N_JOBS=4 for speed.
N_JOBS = int(os.environ.get("N_JOBS", 2))
# Count-like modalities are log-transformed; very sparse features are dropped.
LOG_MODALITIES = {"cellfree_rna", "plasma_somalogic"}
MIN_NONZERO_FRACTION = 0.5

LABELS = {
    "cellfree_rna": "Cell-free RNA",
    "plasma_luminex": "Plasma cytokines",
    "serum_luminex": "Serum cytokines",
    "microbiome": "Microbiome",
    "immune_system": "Immune cells",
    "metabolomics": "Metabolites",
    "plasma_somalogic": "Plasma proteins",
}


def make_model(groups_train):
    """Elastic net whose penalty is tuned by grouped inner CV (never splits a woman)."""
    n_groups = len(np.unique(groups_train))
    inner = GroupKFold(n_splits=min(N_INNER_FOLDS, n_groups))
    return inner, ElasticNetCV(l1_ratio=L1_RATIOS, alphas=30, max_iter=20000, tol=1e-3)


def fit_predict(X, y, groups, train, test):
    inner, enet = make_model(groups[train])
    # ElasticNetCV takes an explicit list of (train, val) index pairs for the inner loop.
    enet.cv = list(inner.split(X[train], y[train], groups[train]))
    model = make_pipeline(StandardScaler(), enet)
    model.fit(X[train], y[train])
    return test, model.predict(X[test]), model


def outer_cv(X, y, groups, scheme):
    """Return out-of-fold predictions.

    'woman':  leave-one-woman-out, 17 folds; the test woman is never seen in training.
    'sample': 17 random folds of samples, so a test woman's other visits are usually
              in training. Same number of folds, so the only difference is the leak.
    """
    if scheme == "woman":
        folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)]
    else:
        n_folds = len(np.unique(groups))
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=SEED).split(X))
    results = Parallel(n_jobs=N_JOBS)(delayed(fit_predict)(X, y, groups, tr, te) for tr, te in folds)
    pred = np.empty(len(y))
    for test, p, _ in results:
        pred[test] = p
    return pred


def scores(y, pred):
    return {
        "mae": float(np.mean(np.abs(y - pred))),
        "pearson_r": float(pearsonr(y, pred)[0]),
        "pearson_p": float(pearsonr(y, pred)[1]),
        "spearman_r": float(spearmanr(y, pred)[0]),
    }


def prepare(X, modality):
    """Label-free cleanup, identical for every fold, so it cannot leak the answer.

    Counts span 0 to ~300,000, so they are log-transformed. Features that are zero in
    most samples are dropped: within a training fold they can be almost constant, and
    standardizing them then divides by ~0 and blows up the held-out prediction.

    Returns the cleaned matrix and a boolean mask of the kept columns.
    """
    keep = X.std(axis=0) > 0
    if modality in LOG_MODALITIES:
        X = np.log1p(np.clip(X, 0, None))
        keep &= (X > 0).mean(axis=0) >= MIN_NONZERO_FRACTION
    return X[:, keep], keep


def main():
    t0 = time.time()
    d = load()
    preg = d["timepoint"] < 4  # train only on pregnancy samples, as in the paper
    y = d["weeks"][preg]
    groups = d["patient"][preg]
    timepoint = d["timepoint"][preg]

    out = {
        "n_women": int(len(np.unique(groups))),
        "n_samples": int(preg.sum()),
        "samples": {
            "patient": groups.tolist(),
            "weeks": y.tolist(),
            "timepoint": timepoint.tolist(),
        },
        "modalities": {},
    }

    # Baseline: predict the average gestational age of the training women.
    base = np.empty(len(y))
    for g in np.unique(groups):
        base[groups == g] = y[groups != g].mean()
    out["baseline"] = {"mae": float(np.mean(np.abs(y - base)))}

    preds_woman = {}
    for m in MODALITIES:
        X, _ = prepare(d[m][preg], m)
        pw = outer_cv(X, y, groups, "woman")
        ps = outer_cv(X, y, groups, "sample")
        preds_woman[m] = pw
        out["modalities"][m] = {
            "label": LABELS[m],
            "n_features": int(X.shape[1]),
            "by_woman": scores(y, pw),
            "by_sample": scores(y, ps),
            "pred_by_woman": pw.tolist(),
        }
        print(f"{m:18s} feats={X.shape[1]:6d}  "
              f"woman MAE={out['modalities'][m]['by_woman']['mae']:.2f} r={out['modalities'][m]['by_woman']['pearson_r']:.2f}  "
              f"sample MAE={out['modalities'][m]['by_sample']['mae']:.2f} r={out['modalities'][m]['by_sample']['pearson_r']:.2f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    # Combine: average the seven held-out predictions. Uses no information from the test woman.
    combined = np.mean([preds_woman[m] for m in MODALITIES], axis=0)
    out["combined"] = {"by_woman": scores(y, combined), "pred_by_woman": combined.tolist()}
    print(f"combined           woman MAE={out['combined']['by_woman']['mae']:.2f} "
          f"r={out['combined']['by_woman']['pearson_r']:.2f}", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
