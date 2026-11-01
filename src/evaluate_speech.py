"""Evaluate Speech -> Sign using synthetic speech (macOS `say`).

    uv run python src/evaluate_speech.py

Speaks every English sentence of the gloss evaluation set, transcribes it with
local Whisper and reports word error rate, speed, and how often the resulting
ASL gloss equals the gloss of the typed sentence. Synthetic voices are cleaner
than real speakers, so treat this as an upper bound and as a regression check.
"""

import os
import re
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speech import SAMPLE_RATE, recognizer  # noqa: E402
from text2sign import load_json, translate  # noqa: E402
from text2sign.evaluate import edit_distance  # noqa: E402


def words(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def synthesize(text, voice, path):
    subprocess.run(["say", "-v", voice, "-o", path, f"--data-format=LEI16@{SAMPLE_RATE}",
                    "--file-format=WAVE", text], check=True)
    with wave.open(path) as w:
        return w.readframes(w.getnframes())


def gloss(text):
    return [g for s in translate(text)["sentences"] for g in s["gloss"]]


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else "Samantha"
    items = load_json("eval_set.json")["items"]
    edits = ref_words = same_gloss = 0
    audio_s = busy_ms = 0.0
    misses = []
    with tempfile.TemporaryDirectory() as tmp:
        for it in items:
            res = recognizer.transcribe(synthesize(it["en"], voice, os.path.join(tmp, "s.wav")))
            ref, hyp = words(it["en"]), words(res["text"])
            edits += edit_distance(hyp, ref)
            ref_words += len(ref)
            audio_s += res["audio_s"]
            busy_ms += res["latency_ms"]
            if gloss(res["text"]) == gloss(it["en"]):
                same_gloss += 1
            else:
                misses.append((it["en"], res["text"]))
    n = len(items)
    print(f"voice {voice}  n={n}")
    print(f"WER {100 * edits / ref_words:.1f}%   same gloss as typed {100 * same_gloss / n:.0f}%")
    print(f"speed {busy_ms / n:.0f} ms per sentence   {audio_s / (busy_ms / 1000):.1f}x real time")
    for en, heard in misses:
        print(f"  said {en!r}  heard {heard!r}")


if __name__ == "__main__":
    main()
