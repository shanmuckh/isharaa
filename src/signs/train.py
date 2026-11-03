"""Train the 200-sign word recogniser on ASL Citizen landmarks.

    uv run python -m signs.train            (from src/)
    uv run python src/signs/train.py --epochs 150

Model selection uses the validation split. NOTE: in this mirror of the data
the "test" split is NOT held out - every test clip also appears in train or
val - so it is not used. Report the validation score, and the cross-dataset
score on WLASL (src/signs/evaluate_videos.py), which uses different signers,
cameras and rooms. The best model is exported to pretrained/signs200.pt.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signs.dataset import ROOT, labels, load  # noqa: E402
from signs.features import SEQ_LEN, feature_dim, frame_features, resample  # noqa: E402
from signs.model import SignClassifier  # noqa: E402


def augment(seq, rng):
    """Geometric + temporal augmentation on a (T, 75, 3) clip."""
    seq = seq.copy()
    xy = seq[..., :2]
    present_cols = np.abs(xy).sum(-1) > 0
    a = np.deg2rad(rng.uniform(-15, 15))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    scale = rng.uniform(0.8, 1.2) * np.array([1.0, rng.uniform(0.8, 1.25)], np.float32)  # aspect ratio
    xy = (xy @ rot.T) * scale + rng.normal(0, 0.08, 2).astype(np.float32)
    xy += rng.normal(0, 0.01, xy.shape).astype(np.float32)                                # jitter
    seq[..., :2] = np.where(present_cols[..., None], xy, seq[..., :2])
    # temporal crop / stretch
    T = len(seq)
    keep = rng.uniform(0.8, 1.0)
    start = rng.integers(0, int(T * (1 - keep)) + 1)
    seq = seq[start:start + max(8, int(T * keep))]
    # drop the non-dominant hand sometimes (tracking fails on webcams)
    if rng.random() < 0.15:
        seq[:, 33:54] = 0
    return seq


def to_features(seq):
    return resample(frame_features(seq)[:, None], SEQ_LEN)[:, 0]


class Clips(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y, train
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        seq = self.X[i].astype(np.float32)
        if self.train:
            seq = augment(seq, self.rng)
        return torch.from_numpy(to_features(seq)), int(self.y[i])


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    model.eval()
    top1 = top5 = n = 0
    per_class = np.zeros((num_classes, 2))
    for x, y in loader:
        logits = model(x.to(device)).float().cpu()
        pred = logits.topk(5, -1).indices
        hit1 = pred[:, 0] == y
        top1 += int(hit1.sum())
        top5 += int((pred == y[:, None]).any(-1).sum())
        n += len(y)
        for c, h in zip(y.tolist(), hit1.tolist()):
            per_class[c] += (h, 1)
    return {"top1": top1 / n, "top5": top5 / n, "n": n, "per_class": per_class[:, 0] / np.maximum(per_class[:, 1], 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "pretrained", "signs200.pt"))
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    names = labels()
    data = load()
    kw = dict(batch_size=args.batch, num_workers=args.workers, worker_init_fn=worker_init, persistent_workers=True)
    train = DataLoader(Clips(*data["train"][:2], True), shuffle=True, drop_last=True, **kw)
    val = DataLoader(Clips(*data["val"][:2], False), **kw)

    model = SignClassifier(feature_dim(), len(names)).to(device)
    print(f"device {device} | params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | "
          f"train {len(data['train'][1])} val {len(data['val'][1])} | {len(names)} signs", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train), pct_start=0.08)
    best, t0 = -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        line = f"epoch {epoch:3d}  loss {total / len(train):.3f}"
        if epoch % 5 == 0 or epoch == args.epochs:
            res = evaluate(model, val, device, len(names))
            line += f"  val top1 {res['top1'] * 100:5.1f}%  top5 {res['top5'] * 100:5.1f}%"
            if res["top1"] > best:
                best = res["top1"]
                torch.save({"state": model.state_dict(), "labels": names, "in_dim": feature_dim(), "seq_len": SEQ_LEN}, args.out)
                line += "  * best"
        print(f"{line}  [{(time.time() - t0) / 60:.1f} min]", flush=True)

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state"])
    res = evaluate(model, val, device, len(names))
    print(f"\nVALIDATION (clips not used for training, n={res['n']}): top-1 {res['top1'] * 100:.1f}%  top-5 {res['top5'] * 100:.1f}%")
    worst = np.argsort(res["per_class"])[:10]
    print("hardest signs:", ", ".join(f"{names[i]} {res['per_class'][i] * 100:.0f}%" for i in worst))


if __name__ == "__main__":
    main()
