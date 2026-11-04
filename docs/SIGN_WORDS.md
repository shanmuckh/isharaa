# Ishaara word recognition and captured avatar signs (200 ASL words)

## Data

The data is a 200-sign subset of **ASL Citizen** (Desai et al., NeurIPS 2023), recorded
by Deaf and fluent signers on their own webcams. We use the keypoint release on
Hugging Face (`SharoonArshad/asl-citizen-processed-200`, CC BY-NC-SA 4.0: cite the
dataset, academic/non-commercial use only). Download it with
`uv run python src/signs/download.py`; the files go to `data/asl_citizen_200/` (git-ignored).

Each clip is a MediaPipe landmark sequence: 33 body points, 21 points for each
hand, and 3 coordinates per point. Clips are resampled to 200 frames and
normalised: shoulder centre at the origin, shoulder width = 1.

**Problems found in this release, and how they are handled:**

* **"Ghost" hands.** Missing hands are stored as a constant point, and the
  release resampled the clips *after* that. Frames next to a gap therefore blend
  a real hand with the constant, producing shrunken, misplaced hands. A hand only
  counts as present when it has its normal size for the clip.
* **The "test" split is not held out.** Every one of its 3,552 clips also appears
  in train or validation (identical video IDs and landmarks). Using it gave an
  invalid 98.6 %. It is not used. Results below are validation, plus a
  cross-dataset test.

## Model (`src/signs/`)

**Per-frame features** (`features.py`, shared by training and the live app):

* **Hand shape.** Each hand's 21 points relative to its wrist, scaled by hand size.
* **Position.** The wrist positions, plus 9 upper-body points: nose, mouth,
  shoulders, elbows and wrists.
* **Presence.** A flag per hand saying whether it is visible.
* **Handedness.** The dominant hand is detected (the one that is visible more and
  moves more) and mirrored to the right. This matched the visibly active hand on
  96 % of one-handed clips.
* **Gaps.** Tracking gaps of up to 6 frames are interpolated.

That gives 108 features per frame, resampled to 64 frames.

**Classifier** (`model.py`): a linear embedding, two grouped 1-D convolutions,
then 3 pre-norm transformer layers (d = 192), and mean + max pooling.
1.5 M parameters; inference takes about 3 ms on CPU.

**Training** (`train.py`): AdamW, one-cycle learning rate 1e-3, label smoothing
0.1, 150 epochs, about 25 min on an M2 GPU. Augmentation:

* rotation ±15°, scale 0.8–1.2, aspect ratio 0.8–1.25, shift, jitter;
* temporal crop and stretch;
* dropping the non-dominant hand in 15 % of clips.

## Results

| Evaluation | Clips | Top-1 | Top-5 |
|---|---|---|---|
| ASL Citizen validation (clips not used for training) | 654 | 89.0 % | 96.2 % |
| **WLASL cross-dataset** (other signers, cameras, rooms) | 441 clips, 149 words | **74.8 %** | **89.1 %** |

For WLASL, landmarks come from the same MediaPipe models the browser uses
(`video_landmarks.py`), and the clips are scored by the live recogniser
(`evaluate_videos.py`). WLASL has its own sign variants for some words, which
lowers this score a little.

**The validation split may share signers with training**, because the release
has no signer IDs. Report the WLASL number as the generalisation result.

## Live use (`static/sign2text/words.js`)

* MediaPipe Pose (lite) and Hand Landmarker run in the browser. The model files
  are kept in `static/vendor/mediapipe/`, so this works offline.
* Each hand is assigned to the nearest body wrist. MediaPipe's own left/right
  labels assume a mirrored selfie image, so they are not used.
* **A word = raise the hands, sign, lower them.** Signing starts after 3 frames
  with a hand above chest level, and ends after 9 frames without. A 10-frame lead
  is kept so clips look like the dataset's.
* The top word is added when its probability is at least 35 %. Otherwise the
  user picks from the top 5; clicking a guess also corrects a wrong word.

**Verified.** Real tracked WLASL signing (hello, eat, want, understand) was
replayed through the app's own segmenter and server. It produced exactly 4 clips;
3 were correct first guesses and "want" came second. Headless Chrome cannot
supply live video frames, so this replay, together with a check that the browser
hand model works on a still image, replaces a fully automated camera test. Test
with a real webcam before relying on it.

## Captured signs for the avatar (Text → Sign)

`src/signs/export_motion.py` and `static/text2sign/tools/retarget.mjs`:

1. For each word, pick the medoid training clip.
2. Trim it to where the hands are up, resample to 15 fps, remove spikes, smooth.
3. Retarget it onto the avatar:
   * wrist position (depth rescaled to about 0.35 m in front of the body);
   * palm rotation;
   * finger flexion and abduction from bone angles;
   * the thumb, solved by the contact IK used for handshapes.

This adds **180 words** to the avatar (20 already had hand-authored signs, which are
kept), so most words the camera recognises can also be signed back.

**Quality:** a sheet of 12 signs looks plausible, but nobody has checked them
sign by sign, and a few are rough. MOVIE's non-dominant hand drifts upward, and
fast signs (AXE, SKATE) peak above 4 m/s. Ask an ASL user to review them before
claiming accuracy.

## Reproduce

```bash
uv run python src/signs/download.py                  # ~1.9 GB, once
uv run python src/signs/train.py
uv run --no-project --with mediapipe python src/signs/video_landmarks.py data/wlasl_eval/pick.json data/wlasl_eval/lm.npz
uv run python src/signs/evaluate_videos.py data/wlasl_eval/pick.json data/wlasl_eval/lm.npz
uv run python src/signs/export_motion.py && node static/text2sign/tools/retarget.mjs
```
