#!/usr/bin/env python3
"""
Ishaara web server.

    uv run python app.py            # opens http://localhost:8000
    uv run python app.py --no-open

Serves the single-page app and these engines:
  * Sign -> Text, words: the browser tracks hands and body with MediaPipe and
    posts one landmark clip per sign to POST /api/sign (src/signs/, 200 signs).
  * Sign -> Text, handshapes: camera frames go to POST /api/detect, where the
    DETR detector (src/detector.py) runs on the GPU (MPS/CUDA) or CPU.
  * Text -> Sign: POST /api/translate turns English into an ASL gloss plan
    (src/text2sign), which the browser animates on the 3D avatar.
  * Speech -> Sign: POST /api/transcribe runs local Whisper (src/speech.py) on
    microphone audio; the text then goes through Text -> Sign.

The command-line OpenCV demo is still available: uv run python src/realtime.py
"""

import json
import os
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_FRAME_BYTES = 2_000_000

sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import text2sign  # noqa: E402  (pure Python)
from detector import detector  # noqa: E402  (imports PyTorch lazily)
from speech import SAMPLE_RATE, recognizer  # noqa: E402  (imports Whisper lazily)
from signs.recognizer import recognizer as words  # noqa: E402  (200-sign word model)

MAX_AUDIO_BYTES = 2 * SAMPLE_RATE * 60   # one minute of 16-bit PCM


class IshaaraHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, fmt, *args):
        # the camera posts many frames per second; only log page/static requests
        if not self.path.startswith("/api/"):
            super().log_message(fmt, *args)

    def _send_json(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self, limit):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= limit:
            raise ValueError(f"request body must be 1..{limit} bytes")
        return self.rfile.read(length)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("", "/"):
            self.path = "/index.html"
            return super().do_GET()
        if path == "/api/health":
            return self._send_json({"ok": True, "detector": detector.info(), "speech": recognizer.info(), "words": words.info()})
        if path == "/api/sign-data":
            return self._send_json(text2sign.sign_data())
        if path == "/api/detector":
            # ?load=1 loads the model now so the first camera frame is fast
            if parse_qs(parsed.query).get("load") == ["1"]:
                try:
                    detector.load()
                except Exception:
                    pass   # reported through info()["error"]
            return self._send_json(detector.info())
        if path == "/api/signs":
            try:
                return self._send_json({"signs": words.vocabulary()})
            except Exception as e:
                return self._send_json({"error": str(e)}, status_code=503)
        if path.startswith("/api/"):
            return self._send_json({"error": "Not Found"}, status_code=404)
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/detect":
            try:
                frame = self._read_body(MAX_FRAME_BYTES)
            except ValueError as e:
                return self._send_json({"error": str(e)}, status_code=400)
            try:
                return self._send_json(detector.detect(frame))
            except Exception as e:
                return self._send_json({"error": str(e)}, status_code=503 if detector.error else 400)

        if path == "/api/sign":
            try:
                body = json.loads(self._read_body(5_000_000))
                return self._send_json(words.recognize(body["frames"]))
            except (ValueError, KeyError, TypeError) as e:
                return self._send_json({"error": str(e)}, status_code=400)
            except Exception as e:
                return self._send_json({"error": str(e)}, status_code=503)

        if path == "/api/transcribe":
            try:
                audio = self._read_body(MAX_AUDIO_BYTES)
            except ValueError as e:
                return self._send_json({"error": str(e)}, status_code=400)
            try:
                return self._send_json(recognizer.transcribe(audio))
            except ValueError as e:
                return self._send_json({"error": str(e)}, status_code=400)
            except Exception as e:
                return self._send_json({"error": str(e)}, status_code=503)

        if path == "/api/translate":
            try:
                body = json.loads(self._read_body(100_000))
                return self._send_json(text2sign.translate(str(body.get("text", ""))[:2000]))
            except Exception as e:
                return self._send_json({"error": str(e)}, status_code=400)

        return self._send_json({"error": "Not Found"}, status_code=404)


def preload_models():
    """Load the models in the background so the first frame and first sentence are fast."""
    try:
        detector.load()
        info = detector.info()
        print(f"[ishaara] sign detector ready: {info['weights']} on {info['device']}")
    except Exception as e:
        print(f"[ishaara] sign detector unavailable: {e}")
    try:
        words.load()
        print(f"[ishaara] word recogniser ready: {words.info()['signs']} signs")
    except Exception as e:
        print(f"[ishaara] word recogniser unavailable: {e}")
    try:
        recognizer.load()
        print(f"[ishaara] speech recogniser ready: whisper {recognizer.info()['model']}")
    except Exception as e:
        print(f"[ishaara] speech recogniser unavailable: {e}")


def run_server(port=PORT, auto_open=True):
    try:
        httpd = ThreadingHTTPServer(("", port), IshaaraHandler)
    except OSError as e:
        if e.errno != 48:   # address already in use -> try the next port
            raise
        port += 1
        httpd = ThreadingHTTPServer(("", port), IshaaraHandler)
    httpd.daemon_threads = True

    url = f"http://localhost:{port}"
    print("=" * 60)
    print(f"🤟 Ishaara running at {url}   (Ctrl+C to stop)")
    print("=" * 60)
    threading.Thread(target=preload_models, daemon=True).start()
    if auto_open:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        print("\n[ishaara] server stopped.")


if __name__ == "__main__":
    run_server(PORT, auto_open="--no-open" not in sys.argv)
