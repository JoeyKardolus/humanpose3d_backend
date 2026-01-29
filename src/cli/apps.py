"""Django app configuration for CLI commands."""

from django.apps import AppConfig


class CliConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.cli"
    verbose_name = "HumanPose3D CLI"
