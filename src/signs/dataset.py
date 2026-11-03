"""Load the ASL Citizen 200-sign landmark set into cleaned clips (cached).

Download it with:  uv run python src/signs/download.py
"""

import csv
import glob
import json
import os

import numpy as np

from .features import dominant_is_left, fill_gaps, mirror

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "asl_citizen_200")
CACHE = os.path.join(DATA, "clean_cache.npz")


def labels():
    with open(os.path.join(DATA, "metadata", "id_to_label.json")) as f:
        m = json.load(f)
    return [m[k] for k in sorted(m, key=int)]


def _load_split(split):
    feats = sorted(glob.glob(os.path.join(DATA, "shards", f"{split}_features_*.npy")))
    labs = sorted(glob.glob(os.path.join(DATA, "shards", f"{split}_labels_*.npy")))
    X = np.concatenate([np.load(f)[:, :, :225] for f in feats]).reshape(-1, 200, 75, 3)
    y = np.concatenate([np.load(f) for f in labs]).astype(np.int64)
    with open(os.path.join(DATA, "metadata", f"{split}_manifest.csv")) as f:
        names = [r["filename"] for r in csv.DictReader(f)]
    return X, y, names


# The mirror's "test" split duplicates train/val clips (checked: every test
# video id also occurs in train or val), so it must not be used as a test set.


def load():
    """-> dict split -> (clips float16 (N,200,75,3), labels (N,), filenames).
    Clips are mirrored so the dominant hand is the right hand, and short
    tracking gaps are filled."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {s: (z[f"{s}_X"], z[f"{s}_y"], list(z[f"{s}_names"])) for s in ("train", "val", "test")}
    out = {}
    for split in ("train", "val", "test"):
        X, y, names = _load_split(split)
        clean = np.empty_like(X, dtype=np.float16)
        for i, s in enumerate(X.astype(np.float32)):
            s = mirror(s) if dominant_is_left(s) else s
            clean[i] = fill_gaps(s)
        out[split] = (clean, y, names)
    np.savez(CACHE, **{f"{s}_{k}": v for s, (X, y, n) in out.items() for k, v in (("X", X), ("y", y), ("names", np.array(n)))})
    return out
