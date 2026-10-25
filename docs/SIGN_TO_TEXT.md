# Ishaara Sign → Text: method and implementation

## Pipeline

```
browser camera (getUserMedia, 640×480)
  └─► 224×224 canvas ─► JPEG q0.85 (~10 KB) ─► POST /api/detect
        server: decode ─► BGR→RGB ─► normalise (ImageNet) ─► DETR on MPS/CUDA/CPU
              ◄─ boxes, scores, class distribution of the most confident query
  └─► overlay + temporal smoothing ─► transcript ─► "Sign it back" (Text → Sign)
```

* **The camera runs in the browser.** The browser handles camera permission
  itself, so the model process never needs operating-system camera access. That
  was the cause of the earlier macOS failure (`not authorized to capture video`).
* **The frame is resized in the browser**, to the 224×224 input the model was
  trained on. The upload is small and the server does no resizing.
* **The model is loaded once** (`src/detector.py`), warmed up with dummy frames,
  and protected by a lock. It reloads automatically when
  `pretrained/ishaara_5class.pt` changes.
* **Two requests are in flight at once.** Encoding and upload of frame N+1
  overlap inference on frame N. Results carry sequence numbers, so stale
  answers are dropped.
* **Temporal smoothing.** The class distribution `p_t` of the best query feeds an
  exponential moving average `s_t = 0.55·s_{t−1} + 0.45·p_t`. A sign is
  committed when `max s_t ≥ 0.70` for 6 consecutive frames. It can be committed
  again only after `max s_t < 0.35`, i.e. after the hand is lowered. This removes
  single-frame flicker and duplicate words.

## Model and training

* DETR with a ResNet-50 backbone (ImageNet), 1 encoder layer, 1 decoder layer
  and 25 object queries. Hungarian matching; loss is cross-entropy + 5·L1 + 2·GIoU.
* **Data:** 150 training and 25 test webcam images (30 + 5 per class) of 5 ASL
  signs: hello, I love you, one, thank you, zero. 79 images contain two boxes
  (two hands).
* **Fine-tuning** (`src/train.py`):
  * starts from `pretrained/4426_model.pt`, a 3-class checkpoint whose class
    head is extended to 5 classes;
  * AdamW with learning rate 1e-5 for the backbone and 1e-4 for the transformer
    and heads, one-cycle schedule, gradient clipping at 0.1, batch 8, 120 epochs
    on an Apple M2 GPU;
  * augmentation: random crop, horizontal flip, mild colour jitter. Hue jitter
    was reduced from 0.5 to 0.04, because ±0.5 hue turns skin blue or green.
  * a stratified validation split (4 per class) is held out of `data/train`
    and used for model selection; `data/test` is used only for the final score.

### Bugs found and fixed

1. **Class order.** Label Studio exports class ids in alphabetical order
   (`hello, iloveyou, one, thankyou, zero`). The project's class list was in a
   different order, so three of the five signs were displayed under the wrong
   name. The config is now in export order, and `train.py` audits every label
   against its file name.
2. **Colour order.** The OpenCV live demo fed BGR frames to a model trained on RGB
   images. It now converts the frames to RGB first.
3. **Untrained classes.** The shipped checkpoint had only 3 trained classes; the
   other two class outputs were random. Fine-tuning on all 5 classes fixes this.
4. **Mis-named captures.** 6 images (2 train, 4 test) have a file name that
   disagrees with their annotation. Visual inspection showed that the
   annotations are correct in all 6 cases; the capture script named the file
   after the prompted sign while the signer was still changing signs.

## Results (held-out test set, n = 25)

| Model | Accuracy | Hit (class + IoU ≥ 0.5) | Mean IoU | Latency (M2 GPU) |
|---|---|---|---|---|
| Shipped checkpoint (`4426_model.pt`) | 24–28 %* | 12 % | 0.55 | 19 ms |
| Fine-tuned (`ishaara_5class.pt`) | **100 %** (25/25) | **96 %** | **0.80** | 19 ms |

\* The shipped checkpoint's two untrained class outputs are randomly initialised
on every load, so its score varies between runs.

IoU is measured against the best-matching labelled box, because many images show
the sign with both hands. Validation accuracy reached 100 % at epoch 10. The best
checkpoint (by validation IoU) is from epoch 30.

Reproduce with `uv run python src/evaluate_detector.py --weights <file>`.
End-to-end server time per frame (JPEG decode + inference + post-processing) is
about 25 ms.

**Limitations.**

* The test set is small (25 images) and comes from one signer, one room and one
  camera, so the numbers measure fit to this setup, not general signer
  independence. Report them that way.
* These are static handshape signs. Signs whose meaning lies in movement, such
  as THANK-YOU's forward motion, are recognised from a single pose.
* Recording more signers and backgrounds is the most valuable next step.
