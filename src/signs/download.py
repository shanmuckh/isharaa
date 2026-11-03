"""Download the ASL Citizen 200-sign landmark set (about 1.9 GB).

    uv run python src/signs/download.py

Source: huggingface.co/datasets/SharoonArshad/asl-citizen-processed-200
(ASL Citizen, Desai et al. 2023; CC BY-NC-SA 4.0: cite it, non-commercial use).
Files are checked against their published sizes and re-fetched if incomplete.
"""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "SharoonArshad/asl-citizen-processed-200"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "asl_citizen_200")
META = ["id_to_label.json", "label_to_id.json", "preprocessing_config.json",
        "train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]


def listing(path):
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{path}") as r:
        return json.load(r)


def fetch(rel, size=None, tries=4):
    dest = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for _ in range(tries):
        if size is not None and os.path.exists(dest) and os.path.getsize(dest) == size:
            return
        urllib.request.urlretrieve(BASE + rel, dest)
        if size is None or os.path.getsize(dest) == size:
            return
    raise RuntimeError(f"incomplete download: {rel}")


def main():
    for name in META:
        fetch(f"metadata/{name}")
    shards = [(f["path"], f.get("lfs", {}).get("size", f["size"])) for f in listing("shards")]
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda s: fetch(*s), shards))
    print(f"{len(shards)} shards in {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
