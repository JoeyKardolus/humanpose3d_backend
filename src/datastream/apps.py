"""Django app configuration for the datastream module."""

from django.apps import AppConfig


class DatastreamConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.datastream"
    verbose_name = "Datastream"
