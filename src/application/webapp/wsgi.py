"""WSGI entrypoint for the webapp application."""

import os
import sys
from pathlib import Path

from django.core.wsgi import get_wsgi_application

APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parents[1]
for path in (REPO_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")

application = get_wsgi_application()
