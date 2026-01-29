"""Django app configuration for the postprocessing module."""

from django.apps import AppConfig


class PostprocessingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.postprocessing"
    verbose_name = "Postprocessing"
