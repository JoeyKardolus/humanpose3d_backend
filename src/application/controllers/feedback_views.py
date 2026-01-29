"""Django views for user feedback and bug reports."""

from __future__ import annotations

import json
from pathlib import Path

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator

from src.application.config.paths import StoragePaths
from src.application.repositories.bug_report_repository import BugReportRepository
from src.application.services.bug_report_service import BugReportService


# Module-level wiring
_STORAGE_PATHS = StoragePaths.load()
_BUG_REPORT_REPO = BugReportRepository(_STORAGE_PATHS.logs_root)
_BUG_REPORT_SERVICE = BugReportService(_BUG_REPORT_REPO)


@method_decorator(csrf_protect, name="dispatch")
class SubmitBugReportView(View):
    """Handle bug report submissions."""

    def post(self, request: HttpRequest) -> HttpResponse:
        """Submit a bug report."""
        # Parse JSON body or form data
        content_type = request.content_type or ""

        if "application/json" in content_type:
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({"error": "Invalid JSON"}, status=400)
        else:
            data = request.POST

        run_key = data.get("run_key")
        error_summary = data.get("error_summary", "").strip()
        stderr_snippet = data.get("stderr_snippet")
        user_comment = data.get("user_comment", "").strip() or None
        contact_email = data.get("contact_email", "").strip() or None
        pipeline_settings_raw = data.get("pipeline_settings")

        # Validate required fields
        if not error_summary:
            return JsonResponse(
                {"error": "Error summary is required"},
                status=400,
            )

        # Parse pipeline settings if provided as JSON string
        pipeline_settings = None
        if pipeline_settings_raw:
            if isinstance(pipeline_settings_raw, str):
                try:
                    pipeline_settings = json.loads(pipeline_settings_raw)
                except json.JSONDecodeError:
                    pass
            elif isinstance(pipeline_settings_raw, dict):
                pipeline_settings = pipeline_settings_raw

        # Set base URL from request for email links
        base_url = f"{request.scheme}://{request.get_host()}"
        _BUG_REPORT_SERVICE._base_url = base_url

        try:
            report = _BUG_REPORT_SERVICE.submit_report(
                run_key=run_key,
                error_summary=error_summary,
                stderr_snippet=stderr_snippet,
                user_comment=user_comment,
                contact_email=contact_email,
                pipeline_settings=pipeline_settings,
            )

            return JsonResponse({
                "success": True,
                "message": "Bug report submitted successfully",
                "timestamp": report.timestamp.isoformat(),
            })
        except Exception as e:
            return JsonResponse(
                {"error": f"Failed to submit bug report: {str(e)}"},
                status=500,
            )


class RecentBugReportsView(View):
    """List recent bug reports (for admin/debugging)."""

    def get(self, request: HttpRequest) -> HttpResponse:
        """Get recent bug reports."""
        limit = int(request.GET.get("limit", 20))
        reports = _BUG_REPORT_SERVICE.get_recent_reports(limit=limit)

        return JsonResponse({
            "reports": [r.to_dict() for r in reports],
            "total_count": _BUG_REPORT_SERVICE.get_report_count(),
        })


def get_bug_report_service() -> BugReportService:
    """Get the module-level bug report service instance."""
    return _BUG_REPORT_SERVICE
