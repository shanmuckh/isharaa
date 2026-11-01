"""On-device English speech recognition used by the web app (POST /api/transcribe).

Runs faster-whisper (Whisper via CTranslate2) on the CPU. The model is
downloaded once on first use and then works offline; audio never leaves this
computer. The browser sends mono 16-bit PCM at 16 kHz, so no audio codec is
needed on the server.
"""

import threading
import time

import numpy as np

SAMPLE_RATE = 16000
MODEL_NAME = "base.en"
# Whisper's usual "nothing was said" guesses on near-silent audio
MIN_SPEECH_RMS = 0.002


class SpeechRecognizer:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self.error = None

    def load(self):
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                self.error = ("Speech recognition is not installed in this Python. "
                              "Start the server with: uv run python app.py")
                raise RuntimeError(self.error) from e
            try:
                self.model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
            except Exception as e:
                self.error = f"Could not load the speech model ({e}). The first run needs internet to download it."
                raise RuntimeError(self.error) from e
            self.error = None

    def info(self):
        return {"loaded": self.model is not None, "model": MODEL_NAME, "error": self.error}

    def transcribe(self, pcm16: bytes) -> dict:
        """pcm16: little-endian mono 16-bit samples at 16 kHz."""
        if len(pcm16) % 2:
            raise ValueError("expected 16-bit PCM audio")
        self.load()
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        seconds = len(audio) / SAMPLE_RATE
        if not len(audio) or float(np.sqrt(np.mean(audio ** 2))) < MIN_SPEECH_RMS:
            return {"text": "", "audio_s": round(seconds, 2), "latency_ms": 0.0}
        t0 = time.perf_counter()
        with self._lock:
            segments, _ = self.model.transcribe(
                audio, language="en", beam_size=1, vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()   # segments are lazy
        return {"text": text, "audio_s": round(seconds, 2),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


recognizer = SpeechRecognizer()
