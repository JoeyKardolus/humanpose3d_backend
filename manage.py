#!/usr/bin/env python
"""Command-line utility for managing the Django project."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "humanpose3d.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
