"""Django app configuration for the marker augmentation module."""

from django.apps import AppConfig


class MarkerAugmentationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.markeraugmentation"
    verbose_name = "Marker Augmentation"
