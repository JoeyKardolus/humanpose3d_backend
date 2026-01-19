"""Django app configuration for the depth refinement module."""

from django.apps import AppConfig


class DepthRefinementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.depth_refinement"
    verbose_name = "Depth Refinement"
