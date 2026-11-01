"""Tests for the speech service that need no model download."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from speech import SpeechRecognizer  # noqa: E402


class Transcribe(unittest.TestCase):
    def setUp(self):
        self.rec = SpeechRecognizer()
        self.rec.model = object()    # loaded, but must not be reached for silence

    def test_silence_returns_no_text(self):
        res = self.rec.transcribe(b"\x00\x00" * 16000)
        self.assertEqual(res["text"], "")
        self.assertEqual(res["audio_s"], 1.0)

    def test_odd_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self.rec.transcribe(b"\x00\x00\x00")


if __name__ == "__main__":
    unittest.main()
