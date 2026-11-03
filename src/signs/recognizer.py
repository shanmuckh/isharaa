"""Live word recogniser used by the web app (POST /api/sign).

The browser tracks body and hands with MediaPipe and sends one landmark clip
per sign (from raising the hands to lowering them). The clip goes through the
same feature code as training (signs/features.py), then the sequence model.
"""

import os
import re
import threading
import time

import numpy as np

from .features import normalize_raw, prepare

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(ROOT, "pretrained", "signs200.pt")

# ASL Citizen gloss ids -> plain English (ids end in a variant number)
SPECIAL = {
    "BACKOUT": "back out", "BASEBALLCAP": "baseball cap", "CALLTTY": "TTY call", "COCACOLA": "Coca-Cola",
    "COVERUP": "cover up", "DEPENDON": "depend on", "EGGBEATER": "egg beater", "FALLINGASLEEP": "falling asleep",
    "HURDLE-TRIP": "trip", "LOOKAPPEARANCE": "looks", "ROCKINGCHAIR": "rocking chair", "SSHH": "shh",
    "TAKECARE": "take care", "TAKEOFF": "take off", "WHATFOR": "what for", "ZOOMOFF": "zoom off",
    "VIDEOPHONE": "videophone", "WRISTWATCH": "wristwatch", "DODO": "dodo",
    "CHRISTMAS": "Christmas", "EGYPT": "Egypt", "EUROPE": "Europe", "GREECE": "Greece",
    "HALLOWEEN": "Halloween", "SWITZERLAND": "Switzerland",
}


def english(gloss):
    base = re.sub(r"\d+$", "", gloss)
    return SPECIAL.get(base, base.lower())


class WordRecognizer:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.labels = None
        self.error = None
        self._mtime = None

    def load(self):
        mtime = os.path.getmtime(WEIGHTS) if os.path.exists(WEIGHTS) else None
        if self.model is not None and mtime == self._mtime:
            return
        with self._lock:
            if self.model is not None and mtime == self._mtime:
                return
            if mtime is None:
                self.error = "No word model yet. Train it with: uv run python src/signs/train.py"
                raise RuntimeError(self.error)
            import torch
            from .model import SignClassifier
            ckpt = torch.load(WEIGHTS, map_location="cpu")
            model = SignClassifier(ckpt["in_dim"], len(ckpt["labels"]), seq_len=ckpt["seq_len"])
            model.load_state_dict(ckpt["state"])
            # tiny model: the CPU is fast and avoids contending with the GPU detector
            self.model, self.labels, self._torch = model.eval(), ckpt["labels"], torch
            self._mtime, self.error = mtime, None

    def info(self):
        return {"loaded": self.model is not None, "signs": len(self.labels) if self.labels else 0, "error": self.error}

    def vocabulary(self):
        self.load()
        return [{"gloss": g, "word": english(g)} for g in self.labels]

    def recognize(self, frames, top=5):
        """frames: (T, 225) or (T, 75, 3) raw MediaPipe landmarks, missing = 0."""
        self.load()
        raw = np.asarray(frames, dtype=np.float32).reshape(-1, 75, 3)
        if len(raw) < 6:
            raise ValueError("clip too short")
        t0 = time.perf_counter()
        feats = prepare(normalize_raw(raw))
        torch = self._torch
        with self._lock, torch.inference_mode():
            probs = self.model(torch.from_numpy(feats)[None]).softmax(-1)[0].numpy()
        order = np.argsort(-probs)[:top]
        return {
            "candidates": [{"gloss": self.labels[i], "word": english(self.labels[i]), "p": round(float(probs[i]), 4)} for i in order],
            "frames": len(raw),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


recognizer = WordRecognizer()
