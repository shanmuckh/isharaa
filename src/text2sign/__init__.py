"""Isharaa Text -> Sign: English text to ASL gloss to animated avatar.

    from text2sign import translate
    result = translate("What is your name?")
    result["sentences"][0]["gloss"]   # -> ['YOUR', 'NAME', 'WHAT']
"""

import json
import os
from functools import lru_cache

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_json(name: str) -> dict:
    """Load a data file; cached, but reloaded when the file changes on disk."""
    path = os.path.join(DATA_DIR, name)
    return _load_json(path, os.path.getmtime(path))


@lru_cache(maxsize=16)
def _load_json(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Grammatical role of captured signs, for the translator's reordering rules
# (everything else is treated as an ordinary content word).
CLIP_CATS = {
    "NOON1": "time", "RECENT1": "time", "WHATFOR1": "wh", "THAT": "pron",
    "EGYPT": "proper", "EUROPE": "proper", "GREECE": "proper", "SWITZERLAND": "proper",
    "CHRISTMAS1": "proper", "HALLOWEEN1": "proper",
}


def clip_signs() -> dict:
    """Signs captured from real signers (src/signs/export_motion.py +
    static/text2sign/tools/retarget.mjs), as lexicon entries. A word that
    already has a hand-authored sign keeps it."""
    path = os.path.join(DATA_DIR, "clips.json")
    if not os.path.exists(path):
        return {}
    lex = load_json("lexicon.json")["signs"]
    taken = {w.lower() for e in lex.values() for w in e["words"]}
    out = {}
    for gloss, clip in load_json("clips.json")["clips"].items():
        word = clip["word"]
        if gloss in lex or word.lower() in taken or not clip.get("r"):
            continue
        out[gloss] = {"words": [word], "cat": CLIP_CATS.get(gloss, "noun"), "clip": True,
                      "desc": "Captured from a real signer (ASL Citizen) and retargeted to the avatar."}
    return out


def sign_data() -> dict:
    """Everything the browser renderer needs: handshapes, the hand-authored
    lexicon, and the captured signs with their motion."""
    clips = clip_signs()
    motion = load_json("clips.json")["clips"] if clips else {}
    return {
        "handshapes": load_json("handshapes.json"),
        "lexicon": load_json("lexicon.json"),
        "captured": {g: {**e, "fps": motion[g]["fps"], "r": motion[g]["r"], "l": motion[g]["l"]} for g, e in clips.items()},
    }


from .translator import translate  # noqa: E402

__all__ = ["translate", "sign_data", "load_json", "clip_signs"]
