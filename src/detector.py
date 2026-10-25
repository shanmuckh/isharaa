"""In-process sign detector used by the web app (POST /api/detect).

The browser captures the camera, squashes each frame to the model's 224x224
input (exactly what training does to its 640x480 images) and posts it as a
small JPEG. The model is loaded once, kept warm on the fastest device, and
calls are serialised with a lock so concurrent requests cannot thrash it.
"""

import os
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_CANDIDATES = [
    os.path.join(ROOT, "pretrained", "ishaara_5class.pt"),   # fine-tuned by src/train.py
    os.path.join(ROOT, "pretrained", "4426_model.pt"),
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


class SignDetector:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.device = None
        self.weights = None
        self.classes = None
        self.error = None
        self._loaded = None

    # ------------------------------------------------------------ loading
    def _preferred_weights(self):
        path = next((w for w in WEIGHT_CANDIDATES if os.path.exists(w)), None)
        return path, (os.path.getmtime(path) if path else None)

    def load(self):
        """Load once (and again if better weights appear, e.g. after training);
        safe to call from many request threads."""
        if self.model is not None and self._preferred_weights() == self._loaded:
            return
        with self._lock:
            if self.model is not None and self._preferred_weights() == self._loaded:
                return
            try:
                import torch
                from model import DETR
                from utils.setup import get_classes
            except ImportError as e:
                self.error = (f"PyTorch is not installed in this Python ({e.name}). "
                              "Start the server with: uv run python app.py")
                raise RuntimeError(self.error) from e
            self.classes = get_classes()
            self.weights, mtime = self._preferred_weights()
            if self.weights is None:
                self.error = "No model weights found in pretrained/"
                raise RuntimeError(self.error)
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            model = DETR(num_classes=len(self.classes))
            model.load_pretrained(self.weights)
            model.eval().to(self.device)
            self._torch = torch
            self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
            self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
            with torch.inference_mode():                       # warm-up: compile kernels now,
                for _ in range(3):                                 # not on the first camera frame
                    model(torch.randn(1, 3, SIZE, SIZE, device=self.device))
                    if self.device.type == "mps":
                        torch.mps.synchronize()
            self.model = model
            self._loaded = (self.weights, mtime)
            self.error = None

    def info(self):
        return {
            "loaded": self.model is not None,
            "device": str(self.device) if self.device else None,
            "weights": os.path.relpath(self.weights, ROOT) if self.weights else None,
            "classes": self.classes,
            "error": self.error,
        }

    # ------------------------------------------------------------ inference
    def detect(self, image_bytes: bytes, threshold: float = 0.5) -> dict:
        import cv2

        self.load()
        t0 = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode image")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            # training data is RGB
        if rgb.shape[:2] != (SIZE, SIZE):
            rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        torch = self._torch
        with self._lock, torch.inference_mode():
            x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255)
            x = (x - self._mean) / self._std
            out = self.model(x)
            probs = out["pred_logits"][0].softmax(-1)[:, :-1]   # drop "no object"
            boxes = out["pred_boxes"][0]
            score, cls = probs.max(-1)
            order = score.argsort(descending=True)
            probs, boxes, score, cls, order = (t.cpu() for t in (probs, boxes, score, cls, order))
        infer_ms = (time.perf_counter() - t0) * 1000

        detections, kept = [], []
        for q in order.tolist():
            s = float(score[q])
            if s < threshold:
                break
            cx, cy, w, h = boxes[q].tolist()
            box = [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]
            if any(_iou(box, k) > 0.5 for k in kept):          # class-agnostic de-duplication
                continue
            kept.append(box)
            detections.append({"class": self.classes[int(cls[q])], "score": round(s, 4),
                               "box": [round(v, 4) for v in box]})
        best = int(order[0])
        return {
            "detections": detections,
            # class distribution of the most confident query: lets the client
            # smooth over time instead of trusting single frames
            "probs": [round(float(p), 4) for p in probs[best]],
            "classes": self.classes,
            "latency_ms": round(infer_ms, 1),
            "device": str(self.device),
        }


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


detector = SignDetector()
