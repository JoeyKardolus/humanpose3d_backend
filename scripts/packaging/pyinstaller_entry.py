"""PyInstaller entry point for HumanPose3D standalone executable.

This module serves as the entry point when the application is bundled with PyInstaller.
It configures the Django environment and starts the development server.

Features:
- Detects frozen (bundled) vs development environment
- Automatically starts Django server on 127.0.0.1:8000 when no arguments provided
- Opens default web browser automatically after server starts
- Supports CLI commands: ./HumanPose3D-linux run_pipeline --video ...
"""
from __future__ import annotations

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path


def _repo_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parents[2]


def _wait_and_open_browser(url: str, max_wait: int = 30) -> None:
    """Wait for server to be ready and open browser."""
    import urllib.request
    import urllib.error

    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            urllib.request.urlopen(url, timeout=1)
            # Server is ready, open browser
            time.sleep(0.5)  # Small delay for stability
            webbrowser.open(url)
            return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)


def main() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "humanpose3d.settings")
    import humanpose3d.settings  # noqa: F401

    from django.core.management import execute_from_command_line

    if len(sys.argv) == 1:
        sys.argv += ["runserver", "127.0.0.1:8000", "--noreload"]

        # Print startup banner
        print("\n" + "=" * 70)
        print("  HumanPose3D - 3D Human Pose Estimation")
        print("=" * 70)
        print("\n  Starting Django server...")
        print("  Server URL: http://127.0.0.1:8000")
        print("\n  Your browser will open automatically in a few seconds.")
        print("  If it doesn't open, visit: http://127.0.0.1:8000")
        print("\n  To stop the server: Press Ctrl+C in this terminal")
        print("=" * 70 + "\n")

        # Launch browser opener in background thread
        url = "http://127.0.0.1:8000"
        browser_thread = threading.Thread(
            target=_wait_and_open_browser,
            args=(url,),
            daemon=True
        )
        browser_thread.start()

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
