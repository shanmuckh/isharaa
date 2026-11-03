"""Tests for the 200-word recogniser (src/signs)."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from signs.features import L_HAND, R_HAND, SEQ_LEN, feature_dim, mirror, normalize_raw, prepare  # noqa: E402
from signs.recognizer import WEIGHTS, english, recognizer  # noqa: E402


def fake_clip(T=40, seed=0, hand=R_HAND):
    """A plausible clip: shoulders at +-0.5, one hand moving in front of the chest."""
    rng = np.random.default_rng(seed)
    seq = np.zeros((T, 75, 3), np.float32)
    seq[:, 11, 0], seq[:, 12, 0] = 0.5, -0.5
    seq[:, :33, :2] += rng.normal(0, 0.01, (T, 33, 2))
    shape = rng.normal(0, 0.15, (21, 2))
    shape[9] = (0, -0.5)                      # wrist -> middle MCP keeps a normal hand size
    path = np.stack([np.linspace(-0.3, 0.3, T), np.linspace(0.4, -0.2, T)], 1)
    seq[:, hand, :2] = path[:, None] + shape[None]
    return seq


class Features(unittest.TestCase):
    def test_shape_and_finite(self):
        f = prepare(fake_clip())
        self.assertEqual(f.shape, (SEQ_LEN, feature_dim()))
        self.assertTrue(np.isfinite(f).all())

    def test_mirror_is_an_involution(self):
        s = fake_clip()
        np.testing.assert_allclose(mirror(mirror(s)), s)

    def test_left_handed_signer_matches_right_handed(self):
        right = fake_clip(hand=R_HAND)
        left = mirror(right)                  # the same sign, signed left-handed
        np.testing.assert_allclose(prepare(left), prepare(right), atol=1e-5)

    def test_missing_hand_is_never_blended(self):
        s = fake_clip(T=60)
        s[20:40, R_HAND] = s[20, 11]          # long tracking gap: hand stored as a constant
        f = prepare(s)
        shape_cols = f[:, :42]
        size = np.abs(shape_cols).sum(1)
        # every frame is either a full-size hand or exactly empty (resampling edges aside)
        self.assertTrue(((size == 0) | (size > 0.5 * np.median(size[size > 0]))).mean() > 0.9)

    def test_normalize_raw_centres_and_scales_shoulders(self):
        raw = np.zeros((3, 75, 3), np.float32)
        raw[:, 11, :2], raw[:, 12, :2] = (0.7, 0.5), (0.3, 0.5)
        n = normalize_raw(raw)
        np.testing.assert_allclose(n[0, 11, :2], (0.5, 0.0), atol=1e-6)
        np.testing.assert_allclose(n[0, 12, :2], (-0.5, 0.0), atol=1e-6)


class Vocabulary(unittest.TestCase):
    def test_english_names(self):
        self.assertEqual(english("BREAKFAST1"), "breakfast")
        self.assertEqual(english("TAKECARE"), "take care")
        self.assertEqual(english("EGYPT"), "Egypt")
        self.assertEqual(english("CALLTTY"), "TTY call")


@unittest.skipUnless(os.path.exists(WEIGHTS) and os.path.exists(os.path.join(ROOT, "data", "asl_citizen_200", "clean_cache.npz")),
                     "needs pretrained/signs200.pt and the downloaded dataset")
class Recognizer(unittest.TestCase):
    def test_recognizes_held_out_clips(self):
        from signs.dataset import labels, load
        X, y, _ = load()["val"]               # clips not used for training
        names = labels()
        hits = 0
        for i in range(0, 400, 20):
            # undo the dataset normalisation so the clip looks like raw browser input
            clip = X[i].astype(np.float32) * 0.2 + 0.5
            res = recognizer.recognize(clip.reshape(len(clip), -1))
            hits += english(names[y[i]]) in [c["word"] for c in res["candidates"]]
        self.assertGreaterEqual(hits, 16)     # validation top-5 is ~96%


if __name__ == "__main__":
    unittest.main()
