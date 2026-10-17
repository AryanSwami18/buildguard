"""Start the BuildGuard API and Streamlit frontend together."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from app.config.settings import PROJECT_ROOT, Settings, load_project_environment


def main() -> int:
    load_project_environment()
    settings = Settings()
    api_port = int(os.getenv("BUILDGUARD_API_PORT", str(settings.webhook_port)))
    # Keep the backend private by default. ngrok can still tunnel localhost:8080
    # for the signed GitHub webhook without exposing the comparison API broadly.
    api_host = os.getenv("BUILDGUARD_API_HOST", "127.0.0.1")
    ui_host = os.getenv("STREAMLIT_HOST", "127.0.0.1")
    ui_port = int(os.getenv("STREAMLIT_PORT", "8501"))
    api_url = os.getenv("BUILDGUARD_API_URL", f"http://127.0.0.1:{api_port}")
    ngrok_domain = os.getenv("NGROK_DOMAIN", "").strip()
    ngrok_binary = os.getenv("NGROK_BIN", "ngrok").strip() or "ngrok"

    ngrok_url = ""
    if ngrok_domain:
        ngrok_url = (
            ngrok_domain
            if ngrok_domain.startswith(("http://", "https://"))
            else f"https://{ngrok_domain}"
        )
        ngrok_path = shutil.which(ngrok_binary)
        if ngrok_path is None:
            raise SystemExit(
                f"NGROK_DOMAIN is configured, but '{ngrok_binary}' was not found. "
                "Install ngrok or set NGROK_BIN to its executable path."
            )

    environment = os.environ.copy()
    environment["BUILDGUARD_API_URL"] = api_url

    process_commands = [
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:app",
            "--host",
            api_host,
            "--port",
            str(api_port),
        ],
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "streamlit_app.py"),
            "--server.address",
            ui_host,
            "--server.port",
            str(ui_port),
        ],
    ]
    if ngrok_url:
        process_commands.append(
            [
                ngrok_path,
                "http",
                str(api_port),
                "--url",
                ngrok_url,
            ]
        )

    processes = [
        subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
        )
        for command in process_commands
    ]

    stopping = False

    def stop_processes(signum: int | None = None, frame: object | None = None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 8
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()

    signal.signal(signal.SIGINT, stop_processes)
    signal.signal(signal.SIGTERM, stop_processes)
    try:
        while not stopping:
            for process in processes:
                code = process.poll()
                if code is not None:
                    stop_processes()
                    return code or 0
            time.sleep(0.5)
    finally:
        stop_processes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
