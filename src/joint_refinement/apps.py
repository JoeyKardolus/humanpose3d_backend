"""Django app configuration for the joint refinement module."""

from django.apps import AppConfig


class JointRefinementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.joint_refinement"
    verbose_name = "Joint Refinement"
