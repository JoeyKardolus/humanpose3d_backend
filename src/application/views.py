"""Public view exports for Django URL routing."""

from src.application.controllers.model_views import (
    CheckModelsView,
    DownloadModelsView,
)
from src.application.controllers.pipeline_views import (
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
    "CheckModelsView",
    "DeleteRunView",
    "DownloadAllView",
    "DownloadModelsView",
    "DownloadView",
    "HomeView",
    "MediaView",
    "PipelineProgressView",
    "ResultsView",
    "RunPipelineView",
    "StatisticsView",
    "UploadMediaView",
]
