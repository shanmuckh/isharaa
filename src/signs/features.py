"""Landmark sequence -> model features. Used for training AND for the live camera,
so the two can never drift apart.

Input: (T, 75, 3) MediaPipe landmarks, ASL Citizen layout
    0-32  pose (BlazePose), 33-53 left hand, 54-74 right hand
normalised like the dataset: shoulder centre subtracted, divided by shoulder
width. A missing hand is any constant value (all 21 points identical).

Note on the dataset: clips were resampled to 200 frames *after* missing hands
were stored as a constant, so frames next to a gap are blends of a real hand
and that constant (shrunken, displaced "ghost" hands). A hand therefore only
counts as present when it has its normal size for the clip.
"""

import numpy as np

POSE_POINTS = [0, 9, 10, 11, 12, 13, 14, 15, 16]      # nose, mouth, shoulders, elbows, wrists
L_HAND, R_HAND = slice(33, 54), slice(54, 75)
SEQ_LEN = 64
# mirroring swaps left/right body points
POSE_MIRROR = {9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}


def normalize_raw(frames):
    """Raw image-normalised landmarks (T, 75, 3), missing points = 0, ->
    the dataset's shoulder-centred, shoulder-width-scaled coordinates."""
    out = np.array(frames, dtype=np.float32)
    ls, rs = out[:, 11], out[:, 12]
    centre = (ls + rs) / 2
    width = np.linalg.norm(ls - rs, axis=-1)
    ok = width > 1e-6
    if not ok.any():
        raise ValueError("no shoulders in view")
    width = np.where(ok, width, np.median(width[ok]))
    centre = np.where(ok[:, None], centre, np.median(centre[ok], axis=0))
    return (out - centre[:, None]) / width[:, None, None]


MAX_GAP = 6     # frames; shorter tracking gaps are interpolated


def hand_present(seq, hand):
    pts = seq[:, hand, :2]
    size = np.linalg.norm(pts[:, 9] - pts[:, 0], axis=-1)      # wrist -> middle MCP
    seen = size > 1e-4
    if seen.sum() < 3:
        return seen
    return size > 0.75 * np.median(size[seen])


def fill_gaps(seq):
    """Interpolate a hand across short tracking gaps; long gaps stay missing."""
    seq = seq.copy()
    for hand in (L_HAND, R_HAND):
        present = hand_present(seq, hand)
        idx = np.flatnonzero(present)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= MAX_GAP:
                for t in range(a + 1, b):
                    w = (t - a) / (b - a)
                    seq[t, hand] = seq[a, hand] * (1 - w) + seq[b, hand] * w
    return seq


def dominant_is_left(seq):
    """The dominant hand is the one that is visible more and moves more."""
    score = []
    for hand in (L_HAND, R_HAND):
        present = hand_present(seq, hand)
        wrist = seq[:, hand][:, 0, :2]
        motion = np.abs(np.diff(wrist[present], axis=0)).sum() if present.sum() > 1 else 0.0
        score.append(present.mean() + motion)
    return score[0] > score[1]


def mirror(seq):
    """Mirror left<->right: x -> -x and swap the hands and paired body points."""
    m = seq.copy()
    m[..., 0] *= -1
    m[:, L_HAND], m[:, R_HAND] = m[:, R_HAND].copy(), m[:, L_HAND].copy()
    for a, b in POSE_MIRROR.items():
        m[:, a] = seq[:, b] * [-1, 1, 1]
    return m


def resample(seq, length=SEQ_LEN):
    t = np.linspace(0, len(seq) - 1, length)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(seq) - 1)
    w = (t - i0).astype(np.float32)[:, None, None]
    return (seq[i0] * (1 - w) + seq[i1] * w).astype(np.float32)


def frame_features(seq, use_z=False):
    """(T, 75, 3) -> (T, F) per-frame features; a missing hand is all zeros."""
    d = 3 if use_z else 2
    feats = []
    for hand in (R_HAND, L_HAND):                         # dominant (right) first
        pts = seq[:, hand, :d]
        present = hand_present(seq, hand)
        wrist = pts[:, :1]
        size = np.linalg.norm(pts[:, 9] - pts[:, 0], axis=-1)[:, None, None] + 1e-6   # wrist -> middle MCP
        shape = ((pts - wrist) / size).reshape(len(seq), -1)
        shape[~present] = 0
        pos = wrist[:, 0].copy()
        pos[~present] = 0
        feats += [shape, pos, present[:, None].astype(np.float32)]
    feats.append(seq[:, POSE_POINTS, :2].reshape(len(seq), -1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def prepare(seq, use_z=False, length=SEQ_LEN):
    """Full pipeline for one clip of normalised landmarks -> (length, F).
    Features are computed per frame and then resampled, so a missing hand is
    never blended with a real one."""
    seq = np.asarray(seq, dtype=np.float32)
    if dominant_is_left(seq):
        seq = mirror(seq)
    feats = frame_features(fill_gaps(seq), use_z)
    return resample(feats[:, None], length)[:, 0]


def feature_dim(use_z=False):
    return frame_features(np.zeros((2, 75, 3), np.float32), use_z).shape[1]
