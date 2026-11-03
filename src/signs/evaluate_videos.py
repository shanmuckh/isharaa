"""Score the word recogniser on landmarks extracted from video clips.

    uv run --no-project --with mediapipe python src/signs/video_landmarks.py data/wlasl_eval/pick.json data/wlasl_eval/lm.npz
    uv run python src/signs/evaluate_videos.py data/wlasl_eval/pick.json data/wlasl_eval/lm*.npz

Used for the cross-dataset test on WLASL (different signers, cameras and
recording conditions from ASL Citizen). Each clip goes through exactly the
same recogniser as the live app.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signs.recognizer import recognizer  # noqa: E402


def main():
    items = {it["file"]: it for it in json.load(open(sys.argv[1]))}
    clips = {}
    for path in sys.argv[2:]:
        z = np.load(path)
        clips.update({k.replace("|", "/"): z[k] for k in z.files})
    top1 = top5 = n = skipped = 0
    per_word = collections.defaultdict(lambda: [0, 0])
    for f, clip in clips.items():
        word = items[f]["word"]
        if len(clip) < 6 or not np.abs(clip[:, 33:]).sum():
            skipped += 1                       # no hands tracked at all
            continue
        try:
            words = [c["word"].lower() for c in recognizer.recognize(clip.reshape(len(clip), -1))["candidates"]]
        except ValueError:
            skipped += 1                       # no body/shoulders tracked
            continue
        n += 1
        top1 += words[0] == word
        top5 += word in words
        per_word[word][0] += words[0] == word
        per_word[word][1] += 1
    print(f"clips {n} (skipped {skipped} without usable tracking), words {len(per_word)}")
    print(f"top-1 {100 * top1 / n:.1f}%   top-5 {100 * top5 / n:.1f}%")
    best = sorted(per_word.items(), key=lambda kv: -kv[1][0] / kv[1][1])
    print("always right:", ", ".join(w for w, (h, t) in best if h == t)[:600])


if __name__ == "__main__":
    main()
