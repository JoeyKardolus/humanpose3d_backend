"""Service for handling bug reports with email notifications."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.mail import send_mail

from src.application.dto.bug_report import BugReport
from src.application.repositories.bug_report_repository import BugReportRepository

logger = logging.getLogger(__name__)


class BugReportService:
    """Handle bug report submission and email notifications."""

    def __init__(self, repository: BugReportRepository, base_url: str = "") -> None:
        self._repository = repository
        self._base_url = base_url

    def submit_report(
        self,
        run_key: str | None,
        error_summary: str,
        stderr_snippet: str | None = None,
        user_comment: str | None = None,
        contact_email: str | None = None,
        pipeline_settings: dict[str, Any] | None = None,
    ) -> BugReport:
        """Submit a bug report and send email notification."""
        report = BugReport(
            timestamp=datetime.utcnow(),
            run_key=run_key,
            error_summary=error_summary,
            stderr_snippet=stderr_snippet,
            user_comment=user_comment,
            contact_email=contact_email,
            pipeline_settings=pipeline_settings,
        )

        # Save report to file
        file_path = self._repository.save_report(report)
        logger.info(f"Bug report saved to {file_path}")

        # Send email notification
        self._send_email_notification(report)

        return report

    def _send_email_notification(self, report: BugReport) -> bool:
        """Send email notification for a bug report."""
        recipient = getattr(settings, "BUG_REPORT_RECIPIENT", None)
        if not recipient:
            recipient = os.environ.get("BUG_REPORT_EMAIL")
        if not recipient:
            logger.warning("No BUG_REPORT_RECIPIENT configured, skipping email")
            return False

        subject = f"[KinetIQ Bug Report] {report.error_summary[:50]}"

        # Build email body
        body_parts = [
            f"Bug Report - {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            f"Error Summary: {report.error_summary}",
            "",
        ]

        if report.run_key:
            run_url = f"{self._base_url}/results/{report.run_key}/" if self._base_url else ""
            body_parts.append(f"Run Key: {report.run_key}")
            if run_url:
                body_parts.append(f"Results URL: {run_url}")
            body_parts.append("")

        if report.user_comment:
            body_parts.extend([
                "User Comment:",
                report.user_comment,
                "",
            ])

        if report.contact_email:
            body_parts.append(f"Contact Email: {report.contact_email}")
            body_parts.append("")

        if report.pipeline_settings:
            body_parts.append("Pipeline Settings:")
            for key, value in report.pipeline_settings.items():
                body_parts.append(f"  {key}: {value}")
            body_parts.append("")

        if report.stderr_snippet:
            body_parts.extend([
                "Error Details:",
                "```",
                report.stderr_snippet[-1000:],  # Limit to last 1000 chars
                "```",
            ])

        body = "\n".join(body_parts)

        try:
            send_mail(
                subject=subject,
                message=body,
                from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None),
                recipient_list=[recipient],
                fail_silently=False,
            )
            logger.info(f"Bug report email sent to {recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send bug report email: {e}")
            return False

    def get_recent_reports(self, limit: int = 20) -> list[BugReport]:
        """Get recent bug reports."""
        return self._repository.list_reports(limit=limit)

    def get_report_count(self) -> int:
        """Get total number of bug reports."""
        return self._repository.get_report_count()
