"""Django app configuration for the visualize data module."""

from django.apps import AppConfig


class VisualizedataConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.visualizedata"
    verbose_name = "Visualize Data"
