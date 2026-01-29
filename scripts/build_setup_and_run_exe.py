#!/usr/bin/env python3
"""Build a PyInstaller executable for scripts/setup_and_run.py."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "build" / "pyinstaller"
WORK_DIR = DIST_DIR / "work"


def run_command(command: Iterable[str], *, cwd: Path) -> None:
    """Run a command and raise on failure."""
    subprocess.run(list(command), cwd=cwd, check=True)


def main() -> int:
    """Build the executable."""
    if shutil.which("pyinstaller") is None:
        print("pyinstaller is not installed.")
        print("Install it with:")
        print("  uv sync --group dev")
        return 1

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "pyinstaller",
            "--onefile",
            "--name",
            "humanpose3d-setup-run",
            "--distpath",
            str(DIST_DIR),
            "--workpath",
            str(WORK_DIR),
            "--specpath",
            str(DIST_DIR),
            str(REPO_ROOT / "scripts" / "setup_and_run.py"),
        ],
        cwd=REPO_ROOT,
    )
    print(f"Executable created at: {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
