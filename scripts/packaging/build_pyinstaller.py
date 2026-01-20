"""PyInstaller build script for HumanPose3D.

NOTE: This script dynamically generates PyInstaller arguments. For production builds,
prefer using the platform-specific .spec files (HumanPose3D-linux.spec, etc.) which
provide reproducible builds and can be version-controlled.

Usage:
    # Generate a new spec file (one-time):
    uv run pyinstaller [options] scripts/packaging/pyinstaller_entry.py --name HumanPose3D-linux

    # Build from existing spec file (recommended):
    uv run pyinstaller HumanPose3D-linux.spec
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _add_data_arg(source: Path, dest: str) -> str:
    sep = ";" if os.name == "nt" else ":"
    return f"{source}{sep}{dest}"


def _discover_modules(repo_root: Path, module_dirs: list[str]) -> list[str]:
    """Discover all Python modules in specified directories."""
    modules = []
    for module_dir in module_dirs:
        target_dir = repo_root / module_dir
        if not target_dir.exists():
            continue
        for py_file in target_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
            rel_path = py_file.relative_to(repo_root)
            module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
            modules.append(module_path)
    return modules


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    entry_script = repo_root / "scripts" / "packaging" / "pyinstaller_entry.py"

    if not entry_script.exists():
        raise FileNotFoundError(entry_script)

    parser = argparse.ArgumentParser(description="Build PyInstaller bundle.")
    parser.add_argument(
        "--dist-dir",
        default="",
        help="Output directory root for the bundled app (default: ./bin)",
    )
    args = parser.parse_args()

    platform_name = "windows" if os.name == "nt" else sys.platform
    if platform_name == "darwin":
        platform_name = "macos"
    app_name = f"HumanPose3D-{platform_name}"

    if args.dist_dir:
        dist_dir = Path(args.dist_dir).resolve()
    else:
        dist_dir = (repo_root / "bin").resolve()

    datas = [
        _add_data_arg(repo_root / "src" / "application" / "templates", "src/application/templates"),
        _add_data_arg(repo_root / "src" / "application" / "static", "src/application/static"),
        _add_data_arg(repo_root / "models", "models"),
    ]

    # Discover all modules for hidden imports
    all_modules = _discover_modules(repo_root, ["src", "humanpose3d"])

    os.chdir(repo_root)
    args = [
        "--name",
        app_name,
        "--noconfirm",
        "--clean",
        "--onedir",
        "--distpath",
        str(dist_dir),
        "--paths",
        str(repo_root),
        "--add-data",
        datas[0],
        "--add-data",
        datas[1],
        "--add-data",
        datas[2],
    ]

    # Add all discovered modules as hidden imports
    for module in all_modules:
        args.extend(["--hidden-import", module])

    args.append(str(entry_script))

    from PyInstaller.__main__ import run

    run(args)

    build_dir = repo_root / "build"
    if build_dir.exists():
        import shutil

        try:
            shutil.rmtree(build_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
