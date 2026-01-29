"""Django app configuration for the kinematics module."""

from django.apps import AppConfig


class KinematicsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.kinematics"
    verbose_name = "Kinematics"
