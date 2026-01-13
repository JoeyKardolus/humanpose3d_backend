from django.urls import path

from .views import DownloadView, HomeView, MediaView, ResultsView, StatisticsView, UploadMediaView

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("results/<path:run_key>/download/<path:file_path>/", DownloadView.as_view(), name="download"),
    path("results/<path:run_key>/media/<path:file_path>/", MediaView.as_view(), name="media"),
    path("results/<path:run_key>/upload/<path:file_path>/", UploadMediaView.as_view(), name="upload_media"),
    path("results/<path:run_key>/statistics/", StatisticsView.as_view(), name="statistics"),
    path("results/<path:run_key>/", ResultsView.as_view(), name="results"),
]
