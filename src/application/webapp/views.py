"""Public view exports for Django URL routing."""

from src.application.webapp.controllers.pipeline_views import (
    DeleteRunView,
    DownloadAllView,
    DownloadView,
    HomeView,
    MediaView,
    PipelineProgressView,
    ResultsView,
    RunPipelineView,
    StatisticsView,
    UploadMediaView,
)

__all__ = [
    "DeleteRunView",
    "DownloadAllView",
    "DownloadView",
    "HomeView",
    "MediaView",
    "PipelineProgressView",
    "ResultsView",
    "RunPipelineView",
    "StatisticsView",
    "UploadMediaView",
]
