"""WSGI entrypoint for the HumanPose3D project."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "humanpose3d.settings")

application = get_wsgi_application()
