"""Small standard-library client used by the Streamlit frontend."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config.settings import load_project_environment
from app.models.report import ComparisonReport
from app.models.repair import ScriptFixProposal


class BuildGuardAPIError(RuntimeError):
    """Raised when the BuildGuard backend cannot complete a request."""


class BuildGuardAPIClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        load_project_environment()
        self.base_url = (
            base_url or os.getenv("BUILDGUARD_API_URL", "http://127.0.0.1:8080")
        ).rstrip("/")
        self.api_secret = os.getenv(
            "BUILDGUARD_API_SECRET",
            os.getenv("GITHUB_WEBHOOK_SECRET", ""),
        )
        self.timeout = timeout or float(
            os.getenv("BUILDGUARD_API_TIMEOUT_SECONDS", "900")
        )

    def list_packages(self) -> list[str]:
        payload = self._request("GET", "/api/packages")
        packages = payload.get("packages", [])
        if not isinstance(packages, list) or not all(
            isinstance(package, str) for package in packages
        ):
            raise BuildGuardAPIError("BuildGuard API returned an invalid package list")
        return packages

    def list_reports(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/reports")
        reports = payload.get("reports", [])
        if not isinstance(reports, list) or not all(
            isinstance(report, dict) for report in reports
        ):
            raise BuildGuardAPIError("BuildGuard API returned an invalid report list")
        return reports

    def compare(
        self,
        package: str,
        *,
        force_comparison: bool = False,
    ) -> ComparisonReport:
        payload = self._request(
            "POST",
            "/api/compare",
            {"package": package, "force_comparison": force_comparison},
        )
        return ComparisonReport.model_validate(payload)

    def start_comparison(
        self,
        package: str,
        *,
        force_comparison: bool = False,
    ) -> str:
        payload = self._request(
            "POST",
            "/api/compare-jobs",
            {"package": package, "force_comparison": force_comparison},
        )
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise BuildGuardAPIError("BuildGuard API did not return a comparison job ID")
        return job_id

    def get_comparison(
        self,
        job_id: str,
    ) -> tuple[str, str | None, int, list[str], ComparisonReport | None, str | None]:
        payload = self._request("GET", f"/api/compare-jobs/{job_id}")
        status = payload.get("status")
        if not isinstance(status, str):
            raise BuildGuardAPIError("BuildGuard API returned an invalid comparison status")
        progress = payload.get("progress")
        if progress is not None and not isinstance(progress, str):
            raise BuildGuardAPIError("BuildGuard API returned invalid comparison progress")
        progress_percent = payload.get("progress_percent", 0)
        if not isinstance(progress_percent, int) or not 0 <= progress_percent <= 100:
            raise BuildGuardAPIError("BuildGuard API returned invalid comparison percentage")
        progress_log = payload.get("progress_log", [])
        if not isinstance(progress_log, list) or not all(
            isinstance(message, str) for message in progress_log
        ):
            raise BuildGuardAPIError("BuildGuard API returned invalid comparison log")
        error = payload.get("error")
        if error is not None and not isinstance(error, str):
            error = str(error)
        if status != "complete":
            return status, progress, progress_percent, progress_log, None, error
        report_data = payload.get("report")
        if not isinstance(report_data, dict):
            raise BuildGuardAPIError("BuildGuard API returned no comparison report")
        return (
            status,
            progress,
            progress_percent,
            progress_log,
            ComparisonReport.model_validate(report_data),
            error,
        )

    def script_source(self, report: ComparisonReport) -> str | None:
        payload = self._request(
            "POST",
            "/api/script-source",
            {"report": report.model_dump(mode="json")},
        )
        content = payload.get("content")
        if content is not None and not isinstance(content, str):
            raise BuildGuardAPIError("BuildGuard API returned invalid script content")
        return content

    def start_fix_proposal(self, report: ComparisonReport) -> str:
        payload = self._request(
            "POST",
            "/api/fix-proposals",
            {"report": report.model_dump(mode="json")},
        )
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise BuildGuardAPIError("BuildGuard API did not return a fix job ID")
        return job_id

    def get_fix_proposal(self, job_id: str) -> tuple[str, tuple[str, ScriptFixProposal] | None, str | None]:
        payload = self._request("GET", f"/api/fix-proposals/{job_id}")
        status = payload.get("status")
        if not isinstance(status, str):
            raise BuildGuardAPIError("BuildGuard API returned an invalid fix status")
        if status != "complete":
            error = payload.get("error")
            return status, None, error if isinstance(error, str) else None
        original = payload.get("original_script")
        proposal_data = payload.get("proposal")
        if not isinstance(original, str) or not isinstance(proposal_data, dict):
            raise BuildGuardAPIError("BuildGuard API returned an invalid fix proposal")
        return status, (original, ScriptFixProposal.model_validate(proposal_data)), None

    def create_pull_request(
        self,
        report: ComparisonReport,
        original_script: str,
        proposal: ScriptFixProposal,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/pull-requests",
            {
                "report": report.model_dump(mode="json"),
                "original_script": original_script,
                "proposal": proposal.model_dump(mode="json"),
            },
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if self.api_secret:
            headers["X-BuildGuard-API-Key"] = self.api_secret
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (TypeError, json.JSONDecodeError):
                pass
            raise BuildGuardAPIError(
                f"BuildGuard API request failed ({error.code}): {detail or error.reason}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise BuildGuardAPIError(
                f"BuildGuard backend is unavailable at {self.base_url}: {error}"
            ) from error
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise BuildGuardAPIError("BuildGuard API returned invalid JSON") from error
        if not isinstance(data, dict):
            raise BuildGuardAPIError("BuildGuard API returned a non-object response")
        return data
