"""Django app configuration for the API module."""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.api"
    verbose_name = "HumanPose3D API"
