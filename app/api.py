"""FastAPI backend for the BuildGuard UI and GitHub webhook."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import hmac
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.config.settings import Settings, load_project_environment
from app.models.report import ComparisonReport
from app.models.repair import ScriptFixProposal
from app.service import (
    create_script_fix_pull_request,
    get_script_fix_proposal,
    get_package_comparison,
    list_available_packages,
    list_saved_comparison_reports,
    load_script_source_content,
    run_package_comparison,
    shutdown_background_jobs,
    start_package_comparison,
    start_script_fix_proposal,
)
from app.webhook import WebhookProcessor


class PackageComparisonRequest(BaseModel):
    package: str
    force_comparison: bool = False


class ScriptSourceRequest(BaseModel):
    report: ComparisonReport


class FixProposalRequest(BaseModel):
    report: ComparisonReport


class PullRequestRequest(BaseModel):
    report: ComparisonReport
    original_script: str
    proposal: ScriptFixProposal


@asynccontextmanager
async def _lifespan(application: FastAPI):
    load_project_environment()
    settings = Settings()
    processor = WebhookProcessor(settings)
    application.state.settings = settings
    application.state.webhook_processor = processor
    try:
        yield
    finally:
        processor.executor.shutdown(wait=True)
        shutdown_background_jobs()


app = FastAPI(
    title="BuildGuard API",
    version="1.0",
    lifespan=_lifespan,
)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _require_internal_api(request: Request) -> Settings:
    """Protect backend operations when the webhook port is tunnelled publicly."""
    settings = _settings(request)
    secret = settings.buildguard_api_secret
    if secret and not hmac.compare_digest(
        request.headers.get("X-BuildGuard-API-Key", ""),
        secret,
    ):
        raise HTTPException(status_code=401, detail="invalid BuildGuard API key")
    return settings


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/packages")
def packages(request: Request) -> dict[str, list[str]]:
    try:
        return {"packages": list_available_packages(_require_internal_api(request))}
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/reports")
def saved_reports(request: Request) -> dict[str, list[dict[str, Any]]]:
    try:
        entries = list_saved_comparison_reports(_require_internal_api(request))
        return {
            "reports": [
                {
                    "id": entry.report_id,
                    "package": entry.report.package,
                    "version": entry.report.version,
                    "updated_at": entry.updated_at,
                    "status": entry.report.status,
                    "findings": entry.report.counts.findings,
                    "report": entry.report.model_dump(mode="json"),
                }
                for entry in entries
            ]
        }
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/compare", response_model=ComparisonReport)
def compare(
    payload: PackageComparisonRequest,
    request: Request,
) -> ComparisonReport:
    try:
        return run_package_comparison(
            payload.package,
            _require_internal_api(request),
            force_comparison=payload.force_comparison,
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/compare-jobs")
def create_comparison_job(
    payload: PackageComparisonRequest,
    request: Request,
) -> dict[str, str]:
    """Start a comparison without holding the UI request open."""
    try:
        job_id = start_package_comparison(
            payload.package,
            _require_internal_api(request),
            force_comparison=payload.force_comparison,
        )
        return {"job_id": job_id}
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/compare-jobs/{job_id}")
def comparison_job(job_id: str, request: Request) -> dict[str, Any]:
    _require_internal_api(request)
    status, progress, progress_percent, progress_log, report, error = get_package_comparison(job_id)
    response: dict[str, Any] = {
        "status": status,
        "progress": progress,
        "progress_percent": progress_percent,
        "progress_log": progress_log,
        "error": error,
    }
    if report is not None:
        response["report"] = report.model_dump(mode="json")
    return response


@app.post("/api/script-source")
def script_source(
    payload: ScriptSourceRequest,
    request: Request,
) -> dict[str, str | None]:
    try:
        content = load_script_source_content(
            payload.report,
            _require_internal_api(request),
        )
        return {"content": content}
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/fix-proposals")
def create_fix_proposal(
    payload: FixProposalRequest,
    request: Request,
) -> dict[str, str]:
    try:
        job_id = start_script_fix_proposal(
            payload.report,
            _require_internal_api(request),
        )
        return {"job_id": job_id}
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/fix-proposals/{job_id}")
def fix_proposal(job_id: str, request: Request) -> dict[str, Any]:
    _require_internal_api(request)
    status, result, error = get_script_fix_proposal(job_id)
    if result is None:
        return {"status": status, "error": error}
    original, proposal = result
    return {
        "status": status,
        "original_script": original,
        "proposal": proposal.model_dump(mode="json"),
        "error": error,
    }


@app.post("/api/pull-requests")
def pull_request(
    payload: PullRequestRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        return create_script_fix_pull_request(
            payload.report,
            payload.original_script,
            payload.proposal,
            _require_internal_api(request),
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/github/webhook")
async def github_webhook(request: Request) -> JSONResponse:
    settings = _settings(request)
    if not settings.github_webhook_secret:
        return JSONResponse(
            status_code=503,
            content={"error": "GITHUB_WEBHOOK_SECRET is not configured"},
        )

    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    expected = "sha256=" + hmac.new(
        settings.github_webhook_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    if not signature or not hmac.compare_digest(expected, signature):
        return JSONResponse(
            status_code=401,
            content={"error": "invalid signature"},
        )

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid JSON"},
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "JSON object expected"},
        )

    event = request.headers.get("X-GitHub-Event", "")
    jobs = request.app.state.webhook_processor.submit_event(event, payload)
    return JSONResponse(status_code=202, content={"accepted": True, "jobs": jobs})
