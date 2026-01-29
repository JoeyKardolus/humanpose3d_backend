"""Django app configuration for the pose detector module."""

from django.apps import AppConfig


class PosedetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.posedetector"
    verbose_name = "Pose Detector"
