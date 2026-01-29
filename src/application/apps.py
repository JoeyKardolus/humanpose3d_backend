"""Django app configuration for the HumanPose3D web UI."""

from django.apps import AppConfig


class ApplicationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.application"
    verbose_name = "HumanPose3D Application"
