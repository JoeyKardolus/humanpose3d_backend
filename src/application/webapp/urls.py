"""URL routing for the webapp domain."""

from django.urls import path

from .views import (
    DownloadAllView,
    DownloadView,
    DeleteRunView,
    HomeView,
    MediaView,
    PipelineProgressView,
    ResultsView,
    RunPipelineView,
    StatisticsView,
    UploadMediaView,
)

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("run/", RunPipelineView.as_view(), name="run"),
    path("progress/<path:run_key>/", PipelineProgressView.as_view(), name="progress"),
    path(
        "results/<path:run_key>/download-all/",
        DownloadAllView.as_view(),
        name="download_all",
    ),
    path(
        "results/<path:run_key>/download/<path:file_path>/",
        DownloadView.as_view(),
        name="download",
    ),
    path(
        "results/<path:run_key>/media/<path:file_path>/",
        MediaView.as_view(),
        name="media",
    ),
    path(
        "results/<path:run_key>/upload/<path:file_path>/",
        UploadMediaView.as_view(),
        name="upload_media",
    ),
    path(
        "results/<path:run_key>/statistics/",
        StatisticsView.as_view(),
        name="statistics",
    ),
    path("results/<path:run_key>/delete/", DeleteRunView.as_view(), name="delete_run"),
    path("results/<path:run_key>/", ResultsView.as_view(), name="results"),
]
