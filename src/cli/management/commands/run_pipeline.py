"""Management command to run the HumanPose3D pipeline."""

from __future__ import annotations

import argparse

from django.core.management.base import BaseCommand

from src.pipeline.runner import add_pipeline_arguments, run_pipeline


class Command(BaseCommand):
    help = "Run the HumanPose3D pipeline from Django."  # noqa: A003

    def add_arguments(self, parser) -> None:  # type: ignore[override]
        add_pipeline_arguments(parser)

    def handle(self, *args, **options) -> None:  # type: ignore[override]
        namespace = argparse.Namespace(**options)
        run_pipeline(namespace)
