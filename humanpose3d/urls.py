"""Root URL configuration for the HumanPose3D Django project."""

from django.urls import include, path

urlpatterns = [
    path("api/", include("src.api.urls")),
    path("", include("src.application.urls")),
]
