"""Convert the Stanford NA lab's Data.Rda into a fast-loading .npz cache.

Source: https://nalab.stanford.edu/multiomics-pregnancy/ (Ghaemi et al., 2019).
The raw file is not redistributed in this repository; run `python src/load_data.py`
after downloading it to data/Data.Rda.
"""
from pathlib import Path
import warnings

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RDA_PATH = DATA_DIR / "Data.Rda"
CACHE_PATH = DATA_DIR / "multiomics.npz"

# Order of the seven blocks inside InputData, as documented in the lab's readme.
MODALITIES = [
    "cellfree_rna",
    "plasma_luminex",
    "serum_luminex",
    "microbiome",
    "immune_system",
    "metabolomics",
    "plasma_somalogic",
]


def convert():
    import rdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(RDA_PATH)
        raw = rdata.conversion.convert(parsed, default_encoding="latin1")

    arrays = {
        "patient": np.asarray(raw["featurepatients"]).astype(str).ravel(),
        "timepoint": np.asarray(raw["featuretimes"], dtype=int).ravel(),
        "weeks": np.asarray(raw["featureweeks"], dtype=float).ravel(),
    }
    for name, block in zip(MODALITIES, raw["InputData"]):
        arrays[name] = np.asarray(block, dtype=float)
        arrays[f"{name}__features"] = np.asarray(block.coords["dim_1"]).astype(str) \
            if "dim_1" in block.coords else np.array([], dtype=str)
    np.savez_compressed(CACHE_PATH, **arrays)
    return arrays


def load():
    """Return a dict with patient, timepoint, weeks and one matrix per modality."""
    if not CACHE_PATH.exists():
        convert()
    with np.load(CACHE_PATH, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    data = convert()
    print(f"Saved {CACHE_PATH}")
    for name in MODALITIES:
        print(f"  {name:18s} {data[name].shape}")
