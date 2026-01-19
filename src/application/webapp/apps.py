"""Django app configuration for the HumanPose3D web UI."""

from django.apps import AppConfig


class WebappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.application.webapp"
    verbose_name = "HumanPose3D Webapp"
