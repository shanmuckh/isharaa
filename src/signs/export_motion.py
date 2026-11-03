"""Export one real signing performance per word for the avatar.

    uv run python src/signs/export_motion.py        -> data/asl_citizen_200/motion_points.json
    node static/text2sign/tools/retarget.mjs        -> src/text2sign/data/clips.json

For every sign the most typical training clip is chosen (the medoid in model
feature space, among clips where the dominant hand is well tracked). It is
trimmed to the part where the hands are up, resampled to 15 fps, smoothed, and
converted to the avatar's world frame (metres; +x = image right = signer's
left, +y up, +z towards the viewer). Hands are already mirrored so the
dominant hand is the right hand.
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signs.dataset import DATA, labels, load  # noqa: E402
from signs.features import L_HAND, R_HAND, hand_present, prepare  # noqa: E402
from signs.recognizer import english  # noqa: E402

SHOULDER_W = 0.35        # avatar shoulder width (m)
SHOULDER_Y = 1.425       # avatar shoulder height (m)
DEPTH_SCALE = 0.12       # MediaPipe pose depth is ~3x exaggerated: median wrist ends ~0.35 m in front
FPS = 15


def medoid(feats):
    flat = feats.reshape(len(feats), -1)
    d = ((flat[:, None] - flat[None]) ** 2).sum(-1)
    return int(d.mean(1).argmin())


def to_world(seq, hand, wrist_pose):
    """(T, 75, 3) normalised clip -> (T, 21, 3) world points of one hand."""
    pts = seq[:, hand]
    sc_z = (seq[:, 11, 2] + seq[:, 12, 2]) / 2
    depth = np.clip(-(seq[:, wrist_pose, 2] - sc_z) * DEPTH_SCALE, 0.15, 0.5)      # wrist depth (m)
    out = np.empty_like(pts)
    out[..., 0] = pts[..., 0] * SHOULDER_W
    out[..., 1] = SHOULDER_Y - pts[..., 1] * SHOULDER_W
    out[..., 2] = depth[:, None] - (pts[..., 2] - pts[:, :1, 2]) * SHOULDER_W        # in-hand depth relative to wrist
    return out


def despike(x, present, jump=0.08):
    """Replace single-frame tracking glitches: a frame that jumps away (m) from
    both neighbours while the neighbours agree with each other."""
    out = x.copy()
    idx = np.flatnonzero(present)
    for a, t, b in zip(idx[:-2], idx[1:-1], idx[2:]):
        wa, wt, wb = out[a, 0], out[t, 0], out[b, 0]
        if min(np.linalg.norm(wt - wa), np.linalg.norm(wt - wb)) > jump and np.linalg.norm(wa - wb) < jump:
            out[t] = (out[a] + out[b]) / 2
    return out


def smooth(x, present, k=2):
    """Moving average over present frames only."""
    out = x.copy()
    for t in np.flatnonzero(present):
        lo, hi = max(0, t - k), min(len(x), t + k + 1)
        w = present[lo:hi]
        out[t] = x[lo:hi][w].mean(0)
    return out


def main():
    names = labels()
    X, y, files = load()["train"]
    with open(os.path.join(DATA, "metadata", "train_manifest.csv")) as f:
        orig = {r["filename"]: int(r["original_frames"]) for r in csv.DictReader(f)}
    motion = {}
    for c, gloss in enumerate(names):
        idx = [i for i in np.flatnonzero(y == c) if hand_present(X[i].astype(np.float32), R_HAND).mean() > 0.3]
        if not idx:
            continue
        feats = np.stack([prepare(X[i].astype(np.float32)) for i in idx])
        i = idx[medoid(feats)]
        seq = X[i].astype(np.float32)
        pres = {"r": hand_present(seq, R_HAND), "l": hand_present(seq, L_HAND)}
        on = np.flatnonzero(pres["r"] | pres["l"])
        a, b = max(0, on[0] - 3), min(len(seq), on[-1] + 4)
        seconds = (b - a) / len(seq) * orig.get(files[i], 90) / 30.0
        n = max(6, round(seconds * FPS))
        pick = np.round(np.linspace(a, b - 1, n)).astype(int)
        frames = {}
        for side, hand, wp in (("r", R_HAND, 16), ("l", L_HAND, 15)):
            pts = smooth(despike(to_world(seq, hand, wp), pres[side]), pres[side])
            frames[side] = [np.round(pts[t], 4).tolist() if pres[side][t] else None for t in pick]
        motion[gloss] = {"word": english(gloss), "fps": FPS, "source": files[i], "r": frames["r"], "l": frames["l"]}
        if (c + 1) % 40 == 0:
            print(f"{c + 1}/{len(names)}", flush=True)
    out = os.path.join(DATA, "motion_points.json")
    with open(out, "w") as f:
        json.dump(motion, f)
    lens = [len(m["r"]) for m in motion.values()]
    print(f"exported {len(motion)} signs, {min(lens)}-{max(lens)} frames each -> {out}")


if __name__ == "__main__":
    main()
