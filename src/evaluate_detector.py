"""Evaluate a DETR sign detector on the test set.

    uv run python src/evaluate_detector.py                       # default weights
    uv run python src/evaluate_detector.py --weights checkpoints/best_model.pt

For every test image the most confident non-background query is taken as the
prediction (each image contains one sign). Reports:
  * accuracy        predicted class == labelled class
  * mAP-style hit   correct class AND box IoU >= 0.5
  * mean IoU        of the predicted box against the best-matching labelled box
  * per-class accuracy and a confusion matrix
  * latency         mean / p95 forward-pass time on the chosen device
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DETRData  # noqa: E402
from model import DETR  # noqa: E402
from utils.boxes import box_cxcywh_to_xyxy, box_iou  # noqa: E402
from utils.setup import get_classes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def evaluate(model, dataset, device, classes):
    n = len(classes)
    confusion = [[0] * n for _ in range(n)]
    hits = correct = 0
    ious, times = [], []
    for i in range(len(dataset)):
        img, target = dataset[i]
        x = img.unsqueeze(0).to(device)
        t0 = time.perf_counter()
        out = model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        probs = out["pred_logits"][0].softmax(-1)[:, :-1]
        score, cls = probs.max(-1)
        q = int(score.argmax())
        pred_cls = int(cls[q])
        gt_cls = int(target["labels"][0])
        # best overlap with any labelled box (many images show the sign with both hands)
        iou = float(box_iou(box_cxcywh_to_xyxy(out["pred_boxes"][0, q:q + 1].cpu()),
                            box_cxcywh_to_xyxy(target["boxes"]))[0].max())
        confusion[gt_cls][pred_cls] += 1
        correct += pred_cls == gt_cls
        hits += pred_cls == gt_cls and iou >= 0.5
        ious.append(iou)
    times = sorted(times[1:]) or times            # drop warm-up
    total = len(dataset)
    return {
        "accuracy": correct / total,
        "hit@0.5": hits / total,
        "mean_iou": sum(ious) / total,
        "latency_ms": sum(times) / len(times),
        "latency_p95_ms": times[int(0.95 * (len(times) - 1))],
        "per_class": {c: confusion[k][k] / max(sum(confusion[k]), 1) for k, c in enumerate(classes)},
        "confusion": confusion,
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.path.join(ROOT, "pretrained", "4426_model.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "test"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    classes = get_classes()
    device = pick_device(args.device)
    model = DETR(num_classes=len(classes))
    model.load_pretrained(args.weights)
    model.eval().to(device)
    res = evaluate(model, DETRData(args.data, train=False), device, classes)

    print(f"\nweights   {os.path.relpath(args.weights, ROOT)}   device {device}   n={res['n']}")
    print(f"accuracy  {res['accuracy'] * 100:6.1f}%   hit@IoU0.5 {res['hit@0.5'] * 100:6.1f}%   mean IoU {res['mean_iou']:.3f}")
    print(f"latency   {res['latency_ms']:6.1f} ms mean   {res['latency_p95_ms']:6.1f} ms p95")
    print("per class " + "  ".join(f"{c} {v * 100:.0f}%" for c, v in res["per_class"].items()))
    print("confusion (rows = true, cols = predicted): " + " ".join(c[:5] for c in classes))
    for c, row in zip(classes, res["confusion"]):
        print(f"  {c:<9} {row}")


if __name__ == "__main__":
    main()
