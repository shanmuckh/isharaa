"""Extract MediaPipe landmarks from video files, exactly like the browser does.

Runs in its own environment (MediaPipe pins NumPy/OpenCV versions that would
clash with the app's):

    uv run --no-project --with mediapipe python src/signs/video_landmarks.py \
        data/wlasl_eval/pick.json data/wlasl_eval/landmarks.npz [from to]

Uses the same .task models as static/sign2text/words.js and the same rule for
assigning each hand to the nearest body wrist. Output: one (T, 75, 3) array per
clip in ASL Citizen layout, missing points = 0.
"""

import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "static", "vendor", "mediapipe", "models")


def make():
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    return pose, hands


def clip_landmarks(path, pose, hands, ts0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, t = [], ts0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t += int(1000 / fps)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        p = pose.detect_for_video(image, t).pose_landmarks
        h = hands.detect_for_video(image, t).hand_landmarks
        raw = np.zeros((75, 3), np.float32)
        if p:
            p = p[0]
            raw[:33] = [(q.x, q.y, q.z) for q in p]
            used = set()
            for lm in h:
                d = lambda k: np.hypot(lm[0].x - p[k].x, lm[0].y - p[k].y)
                slot = 0 if d(15) <= d(16) else 1
                if slot in used:
                    slot = 1 - slot
                used.add(slot)
                raw[33 + slot * 21: 54 + slot * 21] = [(q.x, q.y, q.z) for q in lm]
        frames.append(raw)
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 75, 3), np.float32), t


def main():
    items_path, out = sys.argv[1], sys.argv[2]
    base = os.path.dirname(items_path)
    items = json.load(open(items_path))
    if len(sys.argv) > 4:                       # optional slice: from to
        items = items[int(sys.argv[3]):int(sys.argv[4])]
    pose, hands = make()
    clips, ts = {}, 0
    for i, it in enumerate(items):
        path = os.path.join(base, it["file"])
        if not os.path.exists(path):
            continue
        clips[it["file"]], ts = clip_landmarks(path, pose, hands, ts + 1000)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(items)}", flush=True)
    np.savez_compressed(out, **{k.replace("/", "|"): v for k, v in clips.items()})
    print(f"saved {len(clips)} clips -> {out}")


if __name__ == "__main__":
    main()
