#!/usr/bin/env python3
"""Install prerequisites and run the HumanPose3D web app."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Iterable


def resolve_repo_root() -> Path:
    """Resolve the repository root based on the current working directory."""
    cwd = Path.cwd()
    if (cwd / "manage.py").exists():
        return cwd

    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "manage.py").exists():
        return candidate

    print("Could not find manage.py. Run this from the repo root.")
    sys.exit(1)


REPO_ROOT = resolve_repo_root()
SERVER_URL = "http://127.0.0.1:8000/"
PYTHON_VERSION = "3.12"


def run_command(command: Iterable[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command and return the completed process."""
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        check=check,
    )


def capture_command(command: Iterable[str], *, cwd: Path) -> str:
    """Run a command and return stdout."""
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout or ""


def uv_installed() -> bool:
    """Return True if uv is available on PATH."""
    return shutil.which("uv") is not None


def uv_python_installed() -> bool:
    """Return True if uv has the required Python version installed."""
    output = capture_command(["uv", "python", "list", "--installed"], cwd=REPO_ROOT)
    return PYTHON_VERSION in output


def venv_ready() -> bool:
    """Return True if the virtual environment appears to be created."""
    venv_cfg = REPO_ROOT / ".venv" / "pyvenv.cfg"
    return venv_cfg.exists()


def wait_for_port(host: str, port: int, timeout_seconds: int = 30) -> bool:
    """Wait for a TCP port to open."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def install_uv_windows() -> bool:
    """Install uv on Windows using the official installer."""
    print("Installing uv...")
    try:
        result = subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "irm https://astral.sh/uv/install.ps1 | iex",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to install uv: {result.stderr}")
            return False

        # Refresh PATH to pick up the new installation
        refresh_path_windows()
        return uv_installed()
    except Exception as e:
        print(f"Error installing uv: {e}")
        return False


def refresh_path_windows() -> None:
    """Refresh PATH environment variable on Windows to pick up new installations."""
    # uv installs to %USERPROFILE%\.local\bin on Windows
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.exists():
        current_path = os.environ.get("PATH", "")
        if str(local_bin) not in current_path:
            os.environ["PATH"] = f"{local_bin}{os.pathsep}{current_path}"

    # Also check %USERPROFILE%\.cargo\bin (alternative location)
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists():
        current_path = os.environ.get("PATH", "")
        if str(cargo_bin) not in current_path:
            os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{current_path}"


def ensure_uv() -> None:
    """Ensure uv is installed, auto-installing on Windows if needed."""
    if uv_installed():
        print("uv is already installed. Skipping.")
        return

    if sys.platform == "win32":
        print("uv is not installed. Attempting automatic installation...")
        if install_uv_windows():
            print("uv installed successfully.")
            return
        print("Automatic installation failed.")

    print("uv is not installed.")
    print("Install it first:")
    if sys.platform == "win32":
        print("  powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
    else:
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  source ~/.local/bin/env")
    sys.exit(1)


def ensure_python() -> None:
    """Ensure the required Python version is installed via uv."""
    if uv_python_installed():
        print(f"Python {PYTHON_VERSION} already installed via uv. Skipping.")
        return

    print(f"Installing Python {PYTHON_VERSION} via uv...")
    run_command(["uv", "python", "install", PYTHON_VERSION], cwd=REPO_ROOT)


def ensure_dependencies() -> None:
    """Ensure the virtual environment and dependencies are installed."""
    if venv_ready():
        print("Virtual environment already exists. Skipping uv sync.")
        return

    print("Creating virtual environment and installing dependencies...")
    print("This may take several minutes on first run (downloading ~2GB of packages).")
    print("")
    run_command(["uv", "sync"], cwd=REPO_ROOT)
    print("")
    print("Dependencies installed successfully.")


def start_server() -> subprocess.Popen[str]:
    """Start the Django development server."""
    return subprocess.Popen(
        ["uv", "run", "python", "manage.py", "runserver"],
        cwd=REPO_ROOT,
    )


def main() -> int:
    """Run setup checks, start the server, and open the browser."""
    print(f"Working directory: {REPO_ROOT}")

    if os.environ.get("MPLBACKEND") is None:
        print("Optional: set MPLBACKEND=Agg if running headless/WSL.")

    ensure_uv()
    ensure_python()
    ensure_dependencies()

    print("Starting server...")
    server_process = start_server()

    try:
        if wait_for_port("127.0.0.1", 8000, timeout_seconds=30):
            print(f"Opening browser: {SERVER_URL}")
            webbrowser.open(SERVER_URL)
        else:
            print("Server did not open port 8000 within 30 seconds.")

        return server_process.wait()
    except KeyboardInterrupt:
        print("Stopping server...")
        server_process.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
