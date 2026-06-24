from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STREAMLIT = ROOT / ".venv" / "Scripts" / "streamlit.exe"
BACKEND_URL = "http://127.0.0.1:8000/health"
FRONTEND_URL = "http://127.0.0.1:8501"


def wait_for_url(url: str, name: str, timeout: int = 90) -> None:
    start = time.time()
    last_error = ""
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status < 500:
                    print(f"[ok] {name} is ready: {url}", flush=True)
                    return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"{name} did not become ready at {url}. Last error: {last_error}")


def start_process(name: str, command: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=None,
        stderr=None,
    )
    print(f"[start] {name} pid={process.pid}", flush=True)
    return process


def stop_process(process: subprocess.Popen, name: str) -> None:
    if process.poll() is not None:
        return
    print(f"[stop] stopping {name}...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    if not PYTHON.exists():
        print(f"Missing virtual environment Python: {PYTHON}", flush=True)
        print("Run .\\run_app.ps1 so the environment can be created first.", flush=True)
        return 1

    backend = start_process(
        "backend",
        [str(PYTHON), "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
    )
    try:
        wait_for_url(BACKEND_URL, "FastAPI backend")
        frontend = start_process(
            "frontend",
            [
                str(STREAMLIT),
                "run",
                "ui/app.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8501",
            ],
        )
        try:
            wait_for_url(FRONTEND_URL, "Streamlit frontend")
            print("\nDemo is running.", flush=True)
            print(f"Frontend: {FRONTEND_URL}", flush=True)
            print("Press Ctrl+C here when you are done presenting.\n", flush=True)
            webbrowser.open(FRONTEND_URL)

            while True:
                if backend.poll() is not None:
                    raise RuntimeError("Backend stopped unexpectedly.")
                if frontend.poll() is not None:
                    raise RuntimeError("Frontend stopped unexpectedly.")
                time.sleep(2)
        finally:
            stop_process(frontend, "frontend")
    except KeyboardInterrupt:
        print("\nShutting down demo...", flush=True)
    finally:
        stop_process(backend, "backend")
    return 0


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise SystemExit(main())
