"""Django app configuration for the mediastream module."""

from django.apps import AppConfig


class MediastreamConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.mediastream"
    verbose_name = "Media Stream"
