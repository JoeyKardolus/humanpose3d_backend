#!/usr/bin/env python
import os
import sys
from pathlib import Path


def main() -> None:
    """Run administrative tasks for the Django project."""
    app_root = Path(__file__).resolve().parent
    repo_root = app_root.parents[1]
    for path in (repo_root, app_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
