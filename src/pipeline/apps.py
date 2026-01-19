"""Django app configuration for the pipeline module."""

from django.apps import AppConfig


class PipelineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.pipeline"
    verbose_name = "Pipeline"
