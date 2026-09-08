#!/usr/bin/env python3
"""
Isharaa Web Bridge Server
Serves the web application and provides API endpoints to trigger real-time
sign language detection (src/realtime.py) and stream logs to the browser.
"""

import os
import sys
import json
import time
import signal
import shutil
import threading
import subprocess
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from collections import deque
from urllib.parse import urlparse

PORT = 8000
HOST = "0.0.0.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_python_command():
    """
    Find the best Python command to run src/realtime.py.
    Prioritizes project virtual environment (.venv) containing torch/cv2,
    or uv run python, before falling back to sys.executable.
    """
    # 1. Project local virtual environment (.venv)
    venv_py = os.path.join(BASE_DIR, ".venv", "bin", "python")
    if os.path.exists(venv_py) and os.access(venv_py, os.X_OK):
        return [venv_py]

    venv_py_win = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py_win):
        return [venv_py_win]

    # 2. Check if current sys.executable has cv2 and torch
    try:
        import cv2
        import torch
        return [sys.executable]
    except ImportError:
        pass

    # 3. Check if uv is available
    uv_bin = shutil.which("uv")
    if uv_bin:
        return [uv_bin, "run", "python"]

    return [sys.executable]


# Global process management
process_lock = threading.RLock()
realtime_proc = None
log_buffer = deque(maxlen=200)


def append_log(line: str):
    with process_lock:
        log_buffer.append({
            "time": time.strftime("%H:%M:%S"),
            "message": line.strip()
        })


def stream_reader(proc):
    """Read subprocess stdout and pipe into log_buffer."""
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            append_log(line)
    except Exception as e:
        append_log(f"[bridge] Stream reader exception: {e}")
    finally:
        proc.stdout.close()
        proc.wait()
        append_log(f"[bridge] Detection process terminated (code {proc.returncode})")


class IsharaaHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving index.html and handling detection APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Cache-Control, Accept")
        self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, data, status_code=200):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("", "/"):
            self.path = "/index.html"
            return super().do_GET()

        if path == "/api/status":
            global realtime_proc
            is_running = False
            pid = None
            with process_lock:
                if realtime_proc is not None:
                    if realtime_proc.poll() is None:
                        is_running = True
                        pid = realtime_proc.pid
                    else:
                        realtime_proc = None

            classes = ["hello", "iloveyou", "thankyou", "zero", "one"]
            colors = [[41, 171, 202], [240, 172, 95], [131, 193, 103], [235, 87, 87], [155, 89, 182]]
            config_path = os.path.join(BASE_DIR, "src", "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                        classes = cfg.get("classes", classes)
                        colors = cfg.get("colors", colors)
                except Exception:
                    pass

            py_cmd = resolve_python_command()
            return self._send_json({
                "running": is_running,
                "pid": pid,
                "classes": classes,
                "colors": colors,
                "python_cmd": " ".join(py_cmd),
                "server_time": time.time()
            })

        if path == "/api/logs":
            with process_lock:
                logs_snapshot = list(log_buffer)
            return self._send_json({"logs": logs_snapshot})

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        global realtime_proc

        if path == "/api/start":
            with process_lock:
                if realtime_proc is not None and realtime_proc.poll() is None:
                    return self._send_json({
                        "status": "already_running",
                        "pid": realtime_proc.pid,
                        "message": "Detection is already active."
                    })

            realtime_script = os.path.join(BASE_DIR, "src", "realtime.py")
            if not os.path.exists(realtime_script):
                return self._send_json({
                    "status": "error",
                    "message": f"Cannot find script at {realtime_script}"
                }, status_code=404)

            env = os.environ.copy()
            # Ensure src is in PYTHONPATH
            src_dir = os.path.join(BASE_DIR, "src")
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_dir}:{existing_pythonpath}" if existing_pythonpath else src_dir
            env["PYTHONUNBUFFERED"] = "1"

            py_cmd = resolve_python_command()
            full_cmd = py_cmd + [realtime_script]
            cmd_str = " ".join(full_cmd)

            try:
                append_log(f"[bridge] Spawning real-time detection: {cmd_str}")
                proc = subprocess.Popen(
                    full_cmd,
                    cwd=BASE_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Quick health check: did the process immediately crash?
                time.sleep(0.35)
                if proc.poll() is not None:
                    exit_code = proc.returncode
                    crash_lines = []
                    try:
                        for _ in range(20):
                            line = proc.stdout.readline()
                            if not line:
                                break
                            crash_lines.append(line.strip())
                    except Exception:
                        pass
                    crash_msg = " \n".join(crash_lines) or f"Process terminated with exit code {exit_code}"
                    append_log(f"[bridge] Immediate crash detected: {crash_msg}")
                    return self._send_json({
                        "status": "error",
                        "message": f"Failed to start model: {crash_msg}"
                    }, status_code=500)

                with process_lock:
                    realtime_proc = proc

                threading.Thread(target=stream_reader, args=(proc,), daemon=True).start()
                append_log(f"[bridge] Process started successfully with PID {proc.pid}")

                return self._send_json({
                    "status": "started",
                    "pid": proc.pid,
                    "message": "Real-time sign detection initiated. Camera window should open."
                })
            except Exception as e:
                append_log(f"[bridge] Error launching process: {e}")
                return self._send_json({
                    "status": "error",
                    "message": str(e)
                }, status_code=500)

        if path == "/api/stop":
            with process_lock:
                if realtime_proc is None or realtime_proc.poll() is not None:
                    realtime_proc = None
                    return self._send_json({
                        "status": "not_running",
                        "message": "No active detection process found."
                    })

                pid = realtime_proc.pid
                append_log(f"[bridge] Sending SIGINT to process {pid}...")
                try:
                    realtime_proc.send_signal(signal.SIGINT)
                    # Allow up to 3 seconds for graceful cleanup
                    for _ in range(30):
                        if realtime_proc.poll() is not None:
                            break
                        time.sleep(0.1)

                    if realtime_proc.poll() is None:
                        append_log(f"[bridge] Process {pid} did not exit gracefully, terminating...")
                        realtime_proc.terminate()
                        realtime_proc.wait(timeout=2)
                except Exception as e:
                    append_log(f"[bridge] Error stopping process {pid}: {e}")
                    try:
                        realtime_proc.kill()
                    except Exception:
                        pass
                finally:
                    realtime_proc = None

                return self._send_json({
                    "status": "stopped",
                    "message": "Real-time sign detection stopped."
                })

        return self._send_json({"error": "Not Found"}, status_code=404)


def shutdown_handler(signum, frame):
    print("\n[bridge] Shutting down bridge server...")
    global realtime_proc
    if realtime_proc and realtime_proc.poll() is None:
        try:
            realtime_proc.send_signal(signal.SIGINT)
            realtime_proc.wait(timeout=2)
        except Exception:
            realtime_proc.kill()
    sys.exit(0)


def run_server(port=PORT, auto_open=True):
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    server_address = ("", port)
    try:
        httpd = HTTPServer(server_address, IsharaaHandler)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            port = port + 1
            server_address = ("", port)
            httpd = HTTPServer(server_address, IsharaaHandler)
        else:
            raise e

    url = f"http://localhost:{port}"
    print("=" * 60)
    print(f"🤟 Isharaa Web Bridge running at: {url}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    if auto_open:
        def open_browser():
            time.sleep(0.8)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        print("[bridge] Server stopped.")


if __name__ == "__main__":
    no_open = "--no-open" in sys.argv
    run_server(PORT, auto_open=not no_open)
