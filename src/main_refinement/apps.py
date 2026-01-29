"""Django app configuration for the main refiner module."""

from django.apps import AppConfig


class MainRefinementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.main_refinement"
    verbose_name = "Main Refinement"
