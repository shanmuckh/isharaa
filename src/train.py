"""Fine-tune the DETR sign detector.

    uv run python src/train.py                         # sensible defaults, ~10-15 min on an M-series Mac
    uv run python src/train.py --epochs 200 --batch 8

* trains on Apple-silicon GPU (MPS) / CUDA when available, else CPU
* starts from pretrained/4426_model.pt and adapts the class head to config.json
* holds out a stratified validation split of data/train (data/test stays
  untouched for the final, reportable score: src/evaluate_detector.py)
* keeps the checkpoint with the best validation accuracy (ties -> mean IoU) and
  exports it to pretrained/ishaara_5class.pt, which the app loads automatically
"""

import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DETRData  # noqa: E402
from evaluate_detector import evaluate, pick_device  # noqa: E402
from loss import DETRLoss, HungarianMatcher  # noqa: E402
from model import DETR  # noqa: E402
from utils.boxes import stacker  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.setup import get_classes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}


def stratified_split(dataset, per_class, seed):
    """Hold out `per_class` images of every class for validation."""
    by_class = {}
    for i, name in enumerate(dataset.labels):
        with open(os.path.join(dataset.labels_path, name)) as f:
            by_class.setdefault(int(f.readline().split()[0]), []).append(i)
    rng = random.Random(seed)
    val = []
    for idx in by_class.values():
        val += rng.sample(sorted(idx), min(per_class, len(idx) // 3))
    train = sorted(set(range(len(dataset))) - set(val))
    return train, sorted(val)


def audit_labels(dataset, classes, logger):
    """Warn when label ids disagree with the class named in the file name
    (catches a class list that is not in the annotation tool's id order)."""
    off = []
    for name in dataset.labels:
        tag = next((c for c in classes if f"-{c}-" in name), None)
        with open(os.path.join(dataset.labels_path, name)) as f:
            ids = {int(line.split()[0]) for line in f if line.strip()}
        if tag is not None and ids != {classes.index(tag)}:
            off.append(name)
    if len(off) > len(dataset.labels) // 4:
        logger.warning(f"{len(off)}/{len(dataset.labels)} label files disagree with their file names: "
                       "is the class order in src/config.json the same as the annotation export?")
    elif off:
        logger.info(f"{len(off)} label file(s) disagree with their file name (usually a mis-named capture): "
                    + ", ".join(n[:8] for n in off))


def to_device(y, device):
    return [{k: v.to(device) for k, v in t.items()} for t in y]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4, help='transformer + heads')
    ap.add_argument('--lr-backbone', type=float, default=1e-5, help='ResNet-50 backbone')
    ap.add_argument('--val-per-class', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--init', default=os.path.join(ROOT, 'pretrained', '4426_model.pt'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'checkpoints'))
    ap.add_argument('--export', default=os.path.join(ROOT, 'pretrained', 'ishaara_5class.pt'))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = get_logger("training")
    device = pick_device(args.device)
    classes = get_classes()
    os.makedirs(args.out, exist_ok=True)

    train_aug = DETRData(os.path.join(ROOT, 'data', 'train'), train=True)
    train_plain = DETRData(os.path.join(ROOT, 'data', 'train'), train=False)
    audit_labels(train_plain, classes, logger)
    train_idx, val_idx = stratified_split(train_plain, args.val_per_class, args.seed)
    train_ds, val_ds = Subset(train_aug, train_idx), Subset(train_plain, val_idx)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=stacker, drop_last=True)

    model = DETR(num_classes=len(classes))
    model.load_pretrained(args.init)
    model.to(device)

    backbone = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    opt = optim.AdamW([{'params': backbone, 'lr': args.lr_backbone},
                       {'params': rest, 'lr': args.lr}], weight_decay=1e-4)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_backbone, args.lr],
                                          total_steps=args.epochs * len(loader), pct_start=0.1)
    criterion = DETRLoss(num_classes=len(classes), matcher=HungarianMatcher(WEIGHTS),
                         weight_dict=WEIGHTS, eos_coef=0.1).to(device)

    logger.info(f"device {device} | train {len(train_ds)} imgs, val {len(val_ds)} imgs | "
                f"{args.epochs} epochs x {len(loader)} batches of {args.batch}")
    best, best_path = None, os.path.join(args.out, 'best_model.pt')
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in loader:
            X, y = X.to(device), to_device(y, device)
            ld = criterion(model(X), y)
            loss = (ld['labels']['loss_ce'] * WEIGHTS['class_weighting']
                    + ld['boxes']['loss_bbox'] * WEIGHTS['bbox_weighting']
                    + ld['boxes']['loss_giou'] * WEIGHTS['giou_weighting'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            total += loss.item()

        line = f"epoch {epoch:3d}/{args.epochs}  loss {total / len(loader):.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            res = evaluate(model, val_ds, device, classes)
            score = (res['accuracy'], res['mean_iou'])
            line += f"  val acc {res['accuracy'] * 100:5.1f}%  IoU {res['mean_iou']:.3f}"
            if best is None or score > best:
                best = score
                torch.save(model.state_dict(), best_path)
                line += "  * best"
        print(f"{line}  [{(time.time() - t_start) / 60:.1f} min]", flush=True)

    torch.save(model.state_dict(), os.path.join(args.out, 'last_model.pt'))
    shutil.copyfile(best_path, args.export)
    logger.info(f"best val acc {best[0] * 100:.1f}% IoU {best[1]:.3f} -> exported {os.path.relpath(args.export, ROOT)}")
    logger.info("final held-out score: uv run python src/evaluate_detector.py --weights " + os.path.relpath(args.export, ROOT))


if __name__ == '__main__':
    main()
