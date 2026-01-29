#!/usr/bin/env python3
"""Build the Windows installer for HumanPose3D.

This script:
1. Runs PyInstaller to create the bootstrapper executable
2. Runs Inno Setup Compiler to create the final installer

Prerequisites:
- PyInstaller: `uv sync --group dev`
- Inno Setup 6: Download from https://jrsoftware.org/isdl.php
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
BUILD_DIR = REPO_ROOT / "build"
PYINSTALLER_DIR = BUILD_DIR / "pyinstaller"
INSTALLER_DIR = BUILD_DIR / "installer"

# Common Inno Setup Compiler locations on Windows
ISCC_PATHS = [
    Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
]


def find_iscc() -> Path | None:
    """Find the Inno Setup Compiler executable."""
    # Check if it's on PATH
    iscc_path = shutil.which("ISCC")
    if iscc_path:
        return Path(iscc_path)

    # Check common installation locations
    for path in ISCC_PATHS:
        if path.exists():
            return path

    return None


def run_command(command: list[str], *, cwd: Path) -> bool:
    """Run a command and return True on success."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, check=False)
    return result.returncode == 0


def build_pyinstaller_exe() -> bool:
    """Build the bootstrapper executable with PyInstaller."""
    print("\n=== Step 1: Building bootstrapper with PyInstaller ===\n")

    if shutil.which("pyinstaller") is None:
        print("Error: pyinstaller is not installed.")
        print("Install it with: uv sync --group dev")
        return False

    PYINSTALLER_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = PYINSTALLER_DIR / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    return run_command(
        [
            "pyinstaller",
            "--onefile",
            "--name",
            "humanpose3d-setup-run",
            "--distpath",
            str(PYINSTALLER_DIR),
            "--workpath",
            str(work_dir),
            "--specpath",
            str(PYINSTALLER_DIR),
            str(SCRIPTS_DIR / "setup_and_run.py"),
        ],
        cwd=REPO_ROOT,
    )


def build_inno_installer() -> bool:
    """Build the Windows installer with Inno Setup."""
    print("\n=== Step 2: Building installer with Inno Setup ===\n")

    iscc = find_iscc()
    if iscc is None:
        print("Error: Inno Setup Compiler (ISCC.exe) not found.")
        print("Download Inno Setup 6 from: https://jrsoftware.org/isdl.php")
        return False

    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)

    iss_file = SCRIPTS_DIR / "installer.iss"
    return run_command([str(iscc), str(iss_file)], cwd=REPO_ROOT)


def main() -> int:
    """Build the complete Windows installer."""
    print("=" * 60)
    print("HumanPose3D Windows Installer Build")
    print("=" * 60)

    # Check platform
    if sys.platform != "win32":
        print("Warning: This script is designed for Windows.")
        print("Cross-compilation may not work correctly.")

    # Step 1: Build PyInstaller exe
    if not build_pyinstaller_exe():
        print("\nError: PyInstaller build failed.")
        return 1

    exe_path = PYINSTALLER_DIR / "humanpose3d-setup-run.exe"
    if not exe_path.exists():
        print(f"\nError: Expected executable not found at {exe_path}")
        return 1

    print(f"\nBootstrapper created: {exe_path}")

    # Step 2: Build Inno Setup installer
    if not build_inno_installer():
        print("\nError: Inno Setup build failed.")
        print("\nYou can still use the bootstrapper executable directly:")
        print(f"  {exe_path}")
        return 1

    # Find the output installer
    installers = list(INSTALLER_DIR.glob("HumanPose3D-Setup-*.exe"))
    if installers:
        print("\n" + "=" * 60)
        print("Build complete!")
        print("=" * 60)
        for installer in installers:
            print(f"\nInstaller: {installer}")
    else:
        print("\nWarning: Installer file not found in expected location.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
