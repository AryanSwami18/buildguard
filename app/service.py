"""Application service for one package's Wiki/script comparison."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import re
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config.settings import Settings
from app.github.client import GitHubClient, GitHubResourceNotFound
from app.intelligence.factory import create_provider
from app.models.build import BuildInstruction
from app.models.repair import ScriptFixProposal
from app.models.report import ComparisonReport
from app.parsers.bash_semantic_parser import parse_bash_steps
from app.parsers.version_parser import extract_version, product_from_module
from app.package_mapping import canonical_wiki_package, script_target_for_package
from app.pipeline.build_pipeline import BuildPipeline
from app.reporting.reporter import build_comparison_report
from app.repair import generate_script_fix_proposal
from app.sources.github import GitHubSource
from app.storage.repository import (
    ComparisonCacheKey,
    ParsedSourceRepository,
    StoredComparisonReport,
)
from app.notifications import send_report_notification

ProgressCallback = Callable[[str], None]

WIKI_PARSER_VERSION = "wiki-buildsteps-v1"
BASH_PARSER_VERSION = "bash-semantic-v3"
COMPARISON_ANALYSIS_VERSION = "comparison-v1"


class SourcePending(RuntimeError):
    """A source has not arrived yet and should be retried by a later event."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ComparisonRun:
    """A comparison result together with its exact cache identity."""

    report: ComparisonReport
    cache_key: ComparisonCacheKey | None
    stored_report: StoredComparisonReport | None
    reused: bool


@dataclass(frozen=True)
class AutomationOutcome:
    """Result of one webhook-triggered package/version review."""

    status: str
    report: ComparisonReport | None = None
    pr_url: str | None = None
    message: str | None = None

_FIX_PROPOSAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="buildguard-fix",
)
_FIX_PROPOSAL_JOBS: dict[str, Future[tuple[str, ScriptFixProposal]]] = {}
_FIX_PROPOSAL_JOBS_LOCK = Lock()


@dataclass
class _ComparisonJob:
    """State for a comparison running outside the Streamlit request cycle."""

    future: Future[ComparisonReport] | None = None
    progress: str = "Queued for comparison..."
    progress_percent: int = 0
    progress_history: list[str] = field(
        default_factory=lambda: ["Queued for comparison..."]
    )


_COMPARISON_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="buildguard-compare",
)
_COMPARISON_JOBS: dict[str, _ComparisonJob] = {}
_COMPARISON_JOBS_LOCK = Lock()


def _wiki_parser_version(settings: Settings) -> str:
    model = {
        "ollama": settings.ollama_model,
        "gemini": settings.gemini_model,
        "openrouter": settings.openrouter_model,
    }.get(settings.ai_provider, settings.ai_provider)
    return f"{WIKI_PARSER_VERSION}:{settings.ai_provider}:{model}"


def run_package_comparison(
    package: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> ComparisonReport:
    """Fetch, parse, and compare one package's documentation and script."""
    return _run_package_comparison(package, settings, progress).report


def _run_package_comparison(
    package: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
    *,
    force_wiki: bool = False,
    force_bash: bool = False,
    version_hint: str | None = None,
) -> ComparisonRun:
    """Run one comparison, optionally targeting a script version from a webhook."""
    notify = progress or (lambda _message: None)
    module = wiki_module_for_package(package)
    github = GitHubClient.from_settings(settings)
    cache = ParsedSourceRepository(settings.buildguard_database_path)
    pipeline = BuildPipeline(
        create_provider(settings),
        progress=notify,
    )
    product = product_from_module(module)

    if version_hint is not None:
        # A script event can arrive before the corresponding Wiki update. Keep
        # the exact script version cached, then wait if the Wiki has not caught
        # up instead of comparing it with an older version.
        bash_instruction = _load_bash_instruction(
            github=github,
            cache=cache,
            settings=settings,
            product=product,
            version=version_hint,
            notify=notify,
            # The webhook is a change signal, not proof that the payload is
            # newer than the cache. Re-check the remote SHA so duplicate
            # deliveries do not trigger another parse.
            force_refresh=False,
        )
        wiki_instruction = _load_wiki_instruction(
            github=github,
            pipeline=pipeline,
            cache=cache,
            settings=settings,
            module=module,
            notify=notify,
            force_refresh=force_wiki,
        )
        version = wiki_instruction.version
        if version != version_hint:
            raise SourcePending(
                "waiting_for_wiki",
                f"Wiki for {product} has version {version or 'unknown'}; waiting for version {version_hint}.",
            )
    else:
        wiki_instruction = _load_wiki_instruction(
            github=github,
            pipeline=pipeline,
            cache=cache,
            settings=settings,
            module=module,
            notify=notify,
            force_refresh=force_wiki,
        )
        version = wiki_instruction.version
        if version is None:
            raise ValueError(
                f"Could not determine the product version from Wiki module '{module}'."
            )
        bash_instruction = _load_bash_instruction(
            github=github,
            cache=cache,
            settings=settings,
            product=product,
            version=version,
            notify=notify,
            force_refresh=force_bash,
        )

    if version is None:
        raise ValueError(
            f"Could not determine the product version from Wiki module '{module}'."
        )

    cache_key = _comparison_cache_key(
        cache,
        settings,
        wiki_instruction,
        bash_instruction,
        package=product,
        version=version,
    )
    if cache_key is not None:
        notify("Checking for a saved comparison report...")
        cached_report = cache.get_comparison_report(cache_key)
        if cached_report is not None:
            notify("Using the saved comparison report; no source changes were found.")
            return ComparisonRun(
                report=cached_report.report,
                cache_key=cache_key,
                stored_report=cached_report,
                reused=True,
            )

    notify("Comparing Wiki instructions with the build script using AI...")
    comparison = pipeline.compare(wiki_instruction, bash_instruction)
    report = build_comparison_report(
        comparison,
        wiki_instruction,
        bash_instruction,
        package=product,
        version=version,
    )
    stored_report = None
    if cache_key is not None:
        stored_report = cache.save_comparison_report(cache_key, report)
    return ComparisonRun(
        report=report,
        cache_key=cache_key,
        stored_report=stored_report,
        reused=False,
    )


def list_saved_comparison_reports(
    settings: Settings,
) -> list[StoredComparisonReport]:
    """Return all persisted reports for the report browser."""
    return ParsedSourceRepository(settings.buildguard_database_path).list_comparison_reports()


def _comparison_cache_key(
    cache: ParsedSourceRepository,
    settings: Settings,
    wiki: BuildInstruction,
    bash: BuildInstruction,
    *,
    package: str,
    version: str,
) -> ComparisonCacheKey | None:
    wiki_source = wiki.source_metadata
    script_source = bash.source_metadata
    if wiki_source is None or script_source is None:
        return None

    wiki_revision = cache.get_remote_revision(
        source_kind="wiki",
        repository=wiki_source.repository,
        path=wiki_source.path,
        ref=wiki_source.ref,
        parser_version=_wiki_parser_version(settings),
    )
    script_revision = cache.get_remote_revision(
        source_kind="bash",
        repository=script_source.repository,
        path=script_source.path,
        ref=script_source.ref,
        parser_version=BASH_PARSER_VERSION,
    )
    if not wiki_revision:
        wiki_revision = cache.get_content_sha256(
            source_kind="wiki",
            repository=wiki_source.repository,
            path=wiki_source.path,
            ref=wiki_source.ref,
            parser_version=_wiki_parser_version(settings),
        )
    if not script_revision:
        script_revision = cache.get_content_sha256(
            source_kind="bash",
            repository=script_source.repository,
            path=script_source.path,
            ref=script_source.ref,
            parser_version=BASH_PARSER_VERSION,
        )
    if not wiki_revision or not script_revision:
        # A report is only reusable when both sources have stable identities.
        return None

    return ComparisonCacheKey(
        package=package,
        version=version,
        wiki_repository=wiki_source.repository,
        wiki_path=wiki_source.path,
        wiki_ref=wiki_source.ref,
        wiki_revision=wiki_revision,
        script_repository=script_source.repository,
        script_path=script_source.path,
        script_ref=script_source.ref,
        script_revision=script_revision,
        wiki_parser_version=_wiki_parser_version(settings),
        bash_parser_version=BASH_PARSER_VERSION,
        analysis_version=(
            f"{COMPARISON_ANALYSIS_VERSION}:"
            f"{settings.ai_provider}:"
            f"{settings.ollama_model if settings.ai_provider == 'ollama' else settings.gemini_model}"
        ),
    )


def _comparison_run_key(key: ComparisonCacheKey) -> str:
    """Create a stable idempotency key from all comparison inputs."""
    payload = {
        field: getattr(key, field)
        for field in key.__dataclass_fields__
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_automated_package_review(
    package: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
    *,
    version: str | None = None,
    refresh_wiki: bool = False,
    refresh_bash: bool = False,
) -> AutomationOutcome:
    """Process a webhook target through comparison, PR, and notification."""
    notify = progress or (lambda _message: None)
    try:
        run = _run_package_comparison(
            package,
            settings,
            notify,
            # Source loaders verify the current remote revision themselves.
            # The flags remain part of the durable job for auditability, but
            # must not bypass revision checks on duplicate webhook deliveries.
            force_wiki=False,
            force_bash=False,
            version_hint=version,
        )
    except ValueError as error:
        message = str(error)
        if "No shell script found" in message:
            notify("Wiki parsed and stored; waiting for the matching build script.")
            return AutomationOutcome("waiting_for_script", message=message)
        if "GitHub Wiki page was not found" in message:
            notify("Build script parsed and stored; waiting for the matching Wiki page.")
            return AutomationOutcome("waiting_for_wiki", message=message)
        raise
    except SourcePending as pending:
        notify(str(pending))
        return AutomationOutcome(pending.status, message=str(pending))

    report = run.report
    if run.cache_key is None:
        raise RuntimeError(
            "The completed comparison has no stable source revisions and cannot be automated safely"
        )

    cache = ParsedSourceRepository(settings.buildguard_database_path)
    run_key = _comparison_run_key(run.cache_key)
    previous = cache.get_automation_run(run_key)
    report_id = run.stored_report.report_id if run.stored_report else None

    pr_url = previous.pr_url if previous is not None else None
    action_error: str | None = None

    if not report.discrepancies:
        cache.save_automation_run(
            run_key=run_key,
            package=report.package,
            version=report.version or "unknown",
            report_id=report_id,
            status="no_findings",
            pr_url=pr_url,
        )
        notify("Comparison complete; no actionable findings were found.")
    elif settings.buildguard_auto_pr and not pr_url:
        try:
            notify("Generating an automatic draft PR proposal...")
            original = load_script_source_content(report, settings, progress=notify)
            if original is None:
                raise RuntimeError("The complete build script is unavailable")
            proposal = propose_script_fix(report, original, settings, progress=notify)
            pull_request = create_script_fix_pull_request(
                report,
                original,
                proposal,
                settings,
                branch_name=_automation_branch_name(report, run_key),
            )
            pr_url = pull_request.get("html_url")
            if not isinstance(pr_url, str):
                pr_url = None
            cache.save_automation_run(
                run_key=run_key,
                package=report.package,
                version=report.version or "unknown",
                report_id=report_id,
                status="pr_created",
                pr_url=pr_url,
            )
            notify("Automatic draft PR created.")
        except Exception as error:
            action_error = str(error) or error.__class__.__name__
            cache.save_automation_run(
                run_key=run_key,
                package=report.package,
                version=report.version or "unknown",
                report_id=report_id,
                status="pr_failed",
                pr_url=pr_url,
                last_error=action_error,
            )
            notify(f"Automatic PR could not be created: {action_error}")
    elif not settings.buildguard_auto_pr:
        cache.save_automation_run(
            run_key=run_key,
            package=report.package,
            version=report.version or "unknown",
            report_id=report_id,
            status="report_ready",
            pr_url=pr_url,
        )

    notification_sent = previous.notification_sent if previous is not None else False
    if settings.notification_email_to and not notification_sent:
        try:
            notify("Sending the BuildGuard report notification...")
            send_report_notification(
                settings,
                report,
                pr_url=pr_url,
                action_error=action_error,
            )
            cache.save_automation_run(
                run_key=run_key,
                package=report.package,
                version=report.version or "unknown",
                report_id=report_id,
                status="notification_sent",
                pr_url=pr_url,
                notification_sent=True,
                last_error=action_error,
            )
            notify("BuildGuard report notification sent.")
        except Exception as error:
            notification_error = str(error) or error.__class__.__name__
            action_error = action_error or notification_error
            cache.save_automation_run(
                run_key=run_key,
                package=report.package,
                version=report.version or "unknown",
                report_id=report_id,
                status="notification_failed",
                pr_url=pr_url,
                last_error=action_error,
            )
            notify(f"Report notification could not be sent: {notification_error}")

    return AutomationOutcome(
        "followup_failed" if action_error else "complete",
        report=report,
        pr_url=pr_url,
        message=action_error,
    )


def _automation_branch_name(report: ComparisonReport, run_key: str) -> str:
    safe_package = re.sub(r"[^a-zA-Z0-9._-]+", "-", report.package).strip("-")
    safe_version = re.sub(r"[^a-zA-Z0-9._-]+", "-", report.version or "unknown")
    return f"buildguard/auto/{safe_package}-{safe_version}-{run_key[:12]}"


def start_package_comparison(package: str, settings: Settings) -> str:
    """Start a comparison job and return its ID for progress polling."""
    job_id = uuid4().hex
    job = _ComparisonJob()
    with _COMPARISON_JOBS_LOCK:
        _COMPARISON_JOBS[job_id] = job

    def notify(message: str) -> None:
        with _COMPARISON_JOBS_LOCK:
            job.progress = message
            job.progress_percent = max(
                job.progress_percent,
                _comparison_progress_percent(message),
            )
            if not job.progress_history or job.progress_history[-1] != message:
                job.progress_history.append(message)
                del job.progress_history[:-200]

    job.future = _COMPARISON_EXECUTOR.submit(
        run_package_comparison,
        package,
        settings,
        notify,
    )
    return job_id


def get_package_comparison(
    job_id: str,
) -> tuple[str, str, int, list[str], ComparisonReport | None, str | None]:
    """Return comparison status, progress, result, and error."""
    with _COMPARISON_JOBS_LOCK:
        job = _COMPARISON_JOBS.get(job_id)
        if job is None:
            return (
                "missing",
                "Comparison job is no longer available.",
                0,
                [],
                None,
                "The comparison job is no longer available.",
            )
        future = job.future
        progress = job.progress
        progress_percent = job.progress_percent
        progress_history = list(job.progress_history)

    if future is None or not future.done():
        return "running", progress, progress_percent, progress_history, None, None

    try:
        report = future.result()
    except Exception as error:  # Provider and source failures are shown in the UI.
        return (
            "failed",
            progress,
            progress_percent,
            progress_history,
            None,
            str(error) or error.__class__.__name__,
        )
    return "complete", "Comparison complete.", 100, progress_history, report, None


def _comparison_progress_percent(message: str) -> int:
    """Convert internal pipeline updates into a user-facing overall estimate."""
    text = message.casefold()
    if "queued" in text:
        return 0
    if "fetching wiki module" in text:
        return 10
    if "extracting wiki chunk" in text:
        match = re.search(r"chunk\s+(\d+)\s*/\s*(\d+)", text)
        if match:
            current, total = (int(value) for value in match.groups())
            return 15 + round(25 * current / max(total, 1))
        return 25
    if "extracting wiki build instructions" in text:
        return 20
    if "using cached parsed wiki" in text:
        return 40
    if "fetching " in text and "build script" in text:
        return 45
    if "using cached parsed build script" in text:
        return 60
    if "comparing wiki batch" in text:
        match = re.search(r"batch\s+(\d+)\s*/\s*(\d+)", text)
        if match:
            current, total = (int(value) for value in match.groups())
            return 65 + round(30 * current / max(total, 1))
        return 75
    if "comparing wiki instructions" in text:
        return 65
    if "retrying comparison batch" in text:
        return 75
    return 0


def shutdown_background_jobs() -> None:
    """Stop service executors during an orderly backend shutdown."""
    _COMPARISON_EXECUTOR.shutdown(wait=True)
    _FIX_PROPOSAL_EXECUTOR.shutdown(wait=True)


def load_script_source_content(
    report: ComparisonReport,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> str | None:
    """Load the complete script for the UI code view without reparsing it."""
    source = report.script_source
    if source is None:
        return None

    cache = ParsedSourceRepository(settings.buildguard_database_path)
    content = cache.get_source_content(
        source_kind="bash",
        repository=source.repository,
        path=source.path,
        ref=source.ref,
        parser_version=BASH_PARSER_VERSION,
    )
    if content is not None:
        return content

    notify = progress or (lambda _message: None)
    notify("Loading the complete build script for the Code view...")
    script_file = GitHubClient.from_settings(settings).fetch_file(
        repository=source.repository,
        path=source.path,
        ref=source.ref,
    )
    cache.save_source_content(
        source_kind="bash",
        repository=source.repository,
        path=source.path,
        ref=source.ref,
        parser_version=BASH_PARSER_VERSION,
        content=script_file.content,
    )
    return script_file.content


def propose_script_fix(
    report: ComparisonReport,
    original_script: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> ScriptFixProposal:
    """Generate a complete script proposal for the report's findings."""
    return generate_script_fix_proposal(
        create_provider(settings),
        report,
        original_script,
        progress=progress,
    )


def _prepare_script_fix_proposal(
    report: ComparisonReport,
    settings: Settings,
) -> tuple[str, ScriptFixProposal]:
    """Load the script and generate a proposal outside Streamlit's run cycle."""
    original = load_script_source_content(report, settings)
    if original is None:
        raise RuntimeError("The complete build script is unavailable")
    return original, propose_script_fix(report, original, settings)


def start_script_fix_proposal(
    report: ComparisonReport,
    settings: Settings,
) -> str:
    """Start a proposal job that survives Streamlit widget-triggered reruns."""
    job_id = uuid4().hex
    future = _FIX_PROPOSAL_EXECUTOR.submit(
        _prepare_script_fix_proposal,
        report,
        settings,
    )
    with _FIX_PROPOSAL_JOBS_LOCK:
        _FIX_PROPOSAL_JOBS[job_id] = future
    return job_id


def get_script_fix_proposal(
    job_id: str,
) -> tuple[str, tuple[str, ScriptFixProposal] | None, str | None]:
    """Return a background proposal's state without blocking the UI."""
    with _FIX_PROPOSAL_JOBS_LOCK:
        future = _FIX_PROPOSAL_JOBS.get(job_id)
    if future is None:
        return "missing", None, "The fix proposal job is no longer available."
    if not future.done():
        return "running", None, None
    try:
        return "complete", future.result(), None
    except Exception as error:  # Provider and source errors are shown in the UI.
        return "failed", None, str(error) or error.__class__.__name__


def _pr_text(value: str | None, max_chars: int = 1200) -> str:
    """Keep generated PR sections readable and safe to render as Markdown."""
    if not value:
        return ""
    clipped = value.strip()
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 1].rstrip() + "…"
    # A source excerpt or command can otherwise terminate a Markdown code span
    # or fence in the PR body.
    return clipped.replace("```", "'''").replace("`", "'")


def _pr_human_text(value: str | None, max_chars: int = 1200) -> str:
    """Remove internal parser labels from prose shown to PR reviewers."""
    text = _pr_text(value, max_chars)
    text = re.sub(
        r"\bWiki\s+step\s+wiki-\d+\b",
        "the documented build instruction",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bBash\s+step\s+bash-\d+\b",
        "the script step",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bwiki-\d+\b", "the documented instruction", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbash-\d+\b", "the script step", text, flags=re.IGNORECASE)
    return text


def _pr_commands(commands: list[str]) -> str:
    rendered = [_pr_text(command, 400) for command in commands[:4] if command.strip()]
    if not rendered:
        return ""
    suffix = "" if len(commands) <= 4 else f" (+{len(commands) - 4} more)"
    return "; ".join(f"`{command}`" for command in rendered) + suffix


def _pr_evidence_summary(evidence: Any) -> str:
    """Turn structured evidence into a short explanation for a human reviewer."""
    if evidence is None:
        return "No matching evidence was linked."

    parts: list[str] = []
    if evidence.description:
        parts.append(_pr_human_text(evidence.description, 600))
    commands = _pr_commands(evidence.commands)
    if commands:
        parts.append(f"Command(s): {commands}")
    if evidence.packages:
        packages = ", ".join(
            f"`{_pr_text(item, 160)}`" for item in evidence.packages[:12]
        )
        if len(evidence.packages) > 12:
            packages += f" (+{len(evidence.packages) - 12} more)"
        parts.append(f"Packages involved: {packages}")
    if evidence.repository:
        repository = f"Repository: `{_pr_text(evidence.repository, 300)}`"
        if evidence.ref:
            repository += f" at `{_pr_text(evidence.ref, 200)}`"
        parts.append(repository)
    if evidence.commit:
        parts.append(f"Commit: `{_pr_text(evidence.commit, 200)}`")
    if evidence.version:
        parts.append(f"Version: `{_pr_text(evidence.version, 200)}`")
    if evidence.environment:
        values = ", ".join(
            f"`{_pr_text(name, 120)}={_pr_text(value, 200)}`"
            for name, value in list(evidence.environment.items())[:8]
        )
        if len(evidence.environment) > 8:
            values += f" (+{len(evidence.environment) - 8} more)"
        parts.append(f"Configuration: {values}")
    return " ".join(part for part in parts if part)


def _pr_issue_label(finding: Any) -> str:
    """Use a plain-English issue heading instead of an internal category name."""
    category = getattr(finding.category, "value", finding.category)
    labels = {
        "missing_step": "A documented build action is missing from the script",
        "undocumented_step": "The script performs an undocumented build action",
        "command_mismatch": "The script command behaves differently from the documented command",
        "configuration_mismatch": "The script uses different configuration from the documented process",
        "parameter_value_mismatch": "The script uses a different parameter value",
        "version_inconsistency": "The script uses a different version",
        "dependency_inconsistency": "The script uses different build dependencies",
        "unused_configuration": "The script defines a setting that is not used",
        "other_deviation": "The script behavior differs from the documented process",
    }
    return labels.get(str(category), "The script behavior needs review")


def _build_pull_request_body(
    report: ComparisonReport,
    proposal: ScriptFixProposal,
    source: Any,
    settings: Settings,
) -> str:
    """Build a reviewer-friendly explanation for a generated script PR."""
    findings = report.discrepancies
    addressed_ids = set(proposal.affected_finding_ids)
    finding_count = report.counts.findings or len(findings)
    version = report.version or "unspecified version"
    script_url = source.url if source is not None else ""

    lines = [
        "## Summary",
        "",
        (
            "This PR updates the automation script so its build behavior is "
            "consistent with the documented build process. BuildGuard compared "
            "the build instructions with the script and prepared the change "
            "shown in the diff."
        ),
        "",
        f"- **Package:** `{_pr_text(report.package, 160)}`",
        f"- **Version:** `{_pr_text(version, 160)}`",
        f"- **Script:** `{_pr_text(source.path, 400)}`",
        f"- **Base branch:** `{_pr_text(settings.scripts_branch, 160)}`",
        f"- **Issues found:** `{finding_count}`",
        (
            f"- **Issues addressed by this PR:** `{len(proposal.affected_finding_ids)}`"
            if proposal.affected_finding_ids
            else "- **Issues addressed by this PR:** The generated proposal did not identify any specific issues; review the diff carefully."
        ),
        "",
        "## Why this change is needed",
        "",
        (
            f"BuildGuard found {finding_count} actionable difference(s) between "
            "the documented build process and the automation script. The sections "
            "below explain what should happen, what happens today, and why the "
            "proposed change is needed."
        ),
        "",
        "## What the proposed change does",
        "",
        f"**Summary**  \n{_pr_human_text(proposal.summary, 1800)}",
        "",
        f"**Rationale**  \n{_pr_human_text(proposal.rationale, 2200)}",
        "",
        (
            "BuildGuard validated the proposed file as Bash and checked that the "
            "generated edits are tied to reported findings. The complete diff is "
            "the authoritative description of what will change."
        ),
        "",
        "## Problems explained",
        "",
    ]

    if not findings:
        lines.append("No actionable findings were present when this proposal was generated.")
    else:
        for index, finding in enumerate(findings, start=1):
            addressed = finding.id in addressed_ids
            lines.extend(
                [
                    f"### Problem {index}: {_pr_issue_label(finding)}",
                    "",
                    (
                        f"**In this PR:** {'Addressed by the proposed script change.' if addressed else 'Not changed by this proposal; review separately.'}"
                    ),
                    "",
                    "**What the documented process expects**",
                    "",
                ]
            )
            if finding.wiki is not None:
                lines.append(_pr_evidence_summary(finding.wiki))
            else:
                lines.append("The documentation does not contain a matching action.")
            lines.extend(["", "**What the script currently does**", ""])
            if finding.script:
                for evidence in finding.script[:5]:
                    lines.append(f"- {_pr_evidence_summary(evidence)}")
                if len(finding.script) > 5:
                    lines.append(
                        f"- Plus {len(finding.script) - 5} related script action(s)."
                    )
            else:
                lines.append("The script does not contain a matching action.")
            lines.extend(
                [
                    "",
                    "**Why this matters**",
                    "",
                    _pr_human_text(finding.explanation, 1600)
                    or "The two sources describe different build behavior.",
                    "",
                    "**Recommended resolution**",
                    "",
                    _pr_human_text(finding.recommendation, 1600)
                    or "Review the diff and confirm the resulting behavior.",
                    "",
                ]
            )

    lines.extend(["## Sources", ""])
    if report.wiki_source is not None:
        wiki = report.wiki_source
        wiki_label = _pr_text(f"{wiki.name} ({wiki.repository} @ {wiki.ref})", 500)
        lines.append(f"- Wiki: [{wiki_label}]({wiki.url})")
    if source is not None:
        script_label = _pr_text(f"{source.path} ({source.repository} @ {source.ref})", 500)
        if script_url:
            lines.append(f"- Script: [{script_label}]({script_url})")
        else:
            lines.append(f"- Script: {script_label}")

    lines.extend(
        [
            "",
            "## Reviewer checklist",
            "",
            "- Review the complete diff and confirm that only finding-related lines changed.",
            "- Confirm the resulting script behavior matches the Wiki instructions.",
            "- Run the package build or relevant tests before merging.",
            "- If a finding is not addressed by this proposal, resolve it in a follow-up change or explain why it is intentionally deferred.",
            "",
            "_This PR was generated by BuildGuard and requires human review before merging._",
        ]
    )
    return "\n".join(lines)


def create_script_fix_pull_request(
    report: ComparisonReport,
    original_script: str,
    proposal: ScriptFixProposal,
    settings: Settings,
    *,
    branch_name: str | None = None,
) -> dict[str, Any]:
    """Commit a reviewed proposal and open a PR against the scripts repository."""
    source = report.script_source
    if source is None:
        raise ValueError("This report has no script source to update")
    if source.repository != settings.scripts_repo:
        raise ValueError(
            "The report script repository differs from the configured scripts repository"
        )

    title = (
        f"BuildGuard: align {report.package} "
        f"{report.version or 'build script'} with the documented build process"
    )
    body = _build_pull_request_body(report, proposal, source, settings)
    return GitHubClient.from_settings(settings).create_file_pull_request(
        repository=settings.scripts_repo,
        head_repository=settings.github_pr_head_repo,
        base_branch=settings.scripts_branch,
        path=source.path,
        original_content=original_script,
        fixed_content=proposal.fixed_script,
        title=title,
        body=body,
        draft=settings.github_pr_draft,
        branch_name=branch_name,
    )


def synchronize_package_cache(
    package: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
    *,
    refresh_wiki: bool = False,
    refresh_bash: bool = False,
) -> None:
    """Refresh cached parsed sources for one package without running comparison."""
    notify = progress or (lambda _message: None)
    module = wiki_module_for_package(package)
    github = GitHubClient.from_settings(settings)
    cache = ParsedSourceRepository(settings.buildguard_database_path)
    pipeline = BuildPipeline(
        create_provider(settings),
        progress=notify,
    )
    wiki_instruction = _load_wiki_instruction(
        github=github,
        pipeline=pipeline,
        cache=cache,
        settings=settings,
        module=module,
        notify=notify,
        force_refresh=refresh_wiki,
    )
    product = product_from_module(module)
    version = wiki_instruction.version
    if version is None:
        raise ValueError(
            f"Could not determine the product version from Wiki module '{module}'."
        )
    _load_bash_instruction(
        github=github,
        cache=cache,
        settings=settings,
        product=product,
        version=version,
        notify=notify,
        force_refresh=refresh_bash,
    )


def _load_wiki_instruction(
    *,
    github: GitHubClient,
    pipeline: BuildPipeline,
    cache: ParsedSourceRepository,
    settings: Settings,
    module: str,
    notify: ProgressCallback,
    force_refresh: bool = False,
) -> BuildInstruction:
    if not settings.github_repo:
        raise ValueError("GITHUB_REPO is required to fetch Wiki modules")

    path = module if module.lower().endswith(".md") else f"{module}.md"
    parser_version = _wiki_parser_version(settings)
    remote_revision: str | None = None
    cached: tuple[BuildInstruction, str | None] | None = None
    if not force_refresh:
        try:
            remote_revision = github.wiki_revision(settings.github_branch)
        except RuntimeError:
            notify("Could not verify the Wiki cache; fetching the current page...")

        cached = cache.get_instruction(
            source_kind="wiki",
            repository=settings.github_repo,
            path=path,
            ref=settings.github_branch,
            parser_version=parser_version,
        )
        if (
            cached is not None
            and remote_revision is not None
            and cached[1] == remote_revision
            and cached[0].version is not None
        ):
            notify(f"Using cached parsed Wiki instructions for '{module}'.")
            return cached[0]

    notify(f"Fetching Wiki module '{module}'...")
    try:
        page = github.fetch_module_wiki(module)
    except GitHubResourceNotFound as error:
        raise SourcePending(
            "waiting_for_wiki",
            f"Wiki page for {module} is not available yet.",
        ) from error
    product = product_from_module(module)
    if (
        cached is not None
        and cached[0].version is not None
        and remote_revision is None
        and cache.get_content_sha256(
            source_kind="wiki",
            repository=settings.github_repo,
            path=page.path,
            ref=settings.github_branch,
            parser_version=parser_version,
        )
        == _content_sha256(page.content)
    ):
        notify(
            "Wiki content is unchanged; using the cached parsed Wiki instructions."
        )
        cache.save_source_content(
            source_kind="wiki",
            repository=settings.github_repo,
            path=page.path,
            ref=settings.github_branch,
            parser_version=parser_version,
            content=page.content,
        )
        return cached[0]

    version = extract_version(page.content, product)
    if version is None:
        raise ValueError(
            f"Could not determine the product version from Wiki module '{module}'."
        )

    notify("Extracting Wiki build instructions with AI...")
    instruction = pipeline.from_github_wiki(page, version=version)
    if remote_revision is None:
        try:
            remote_revision = github.wiki_revision(settings.github_branch)
        except RuntimeError:
            remote_revision = None
    cache_revision = remote_revision or _content_sha256(page.content)
    if cache_revision is not None:
        cache.save_instruction(
            source_kind="wiki",
            repository=settings.github_repo,
            path=page.path,
            ref=settings.github_branch,
            parser_version=parser_version,
            remote_revision=cache_revision,
            content=page.content,
            instruction=instruction,
        )
    return instruction


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _load_bash_instruction(
    *,
    github: GitHubClient,
    cache: ParsedSourceRepository,
    settings: Settings,
    product: str,
    version: str,
    notify: ProgressCallback,
    force_refresh: bool = False,
) -> BuildInstruction:
    script_target = script_target_for_package(product, version)
    try:
        script_path, remote_revision = github.find_build_script_reference(
            repository=settings.scripts_repo,
            product=script_target.directory,
            version=version,
            ref=settings.scripts_branch,
            script_filename=script_target.filename,
        )
    except GitHubResourceNotFound as error:
        raise SourcePending(
            "waiting_for_script",
            f"Build script for {product} {version} is not available yet.",
        ) from error
    parser_version = BASH_PARSER_VERSION
    if not force_refresh and remote_revision is not None:
        cached = cache.get_instruction(
            source_kind="bash",
            repository=settings.scripts_repo,
            path=script_path,
            ref=settings.scripts_branch,
            parser_version=parser_version,
        )
        if cached is not None and cached[1] == remote_revision:
            notify(f"Using cached parsed build script for {product} {version}.")
            return cached[0]

    notify(f"Fetching {product} {version} build script...")
    script_file = github.fetch_file(
        repository=settings.scripts_repo,
        path=script_path,
        ref=settings.scripts_branch,
    )
    instruction = BuildInstruction(
        name=script_file.filename,
        source="bash",
        version=version,
        source_metadata=GitHubSource.model_validate(script_file.model_dump()),
        steps=parse_bash_steps(script_file.content, script_file.filename),
    )
    if remote_revision is not None:
        cache.save_instruction(
            source_kind="bash",
            repository=settings.scripts_repo,
            path=script_path,
            ref=settings.scripts_branch,
            parser_version=parser_version,
            remote_revision=remote_revision,
            content=script_file.content,
            instruction=instruction,
        )
    return instruction


def list_available_packages(settings: Settings) -> list[str]:
    """List package names represented by BuildGuard Wiki modules."""
    github = GitHubClient.from_settings(settings)
    packages = {
        product_from_module(module)
        for module in github.list_wiki_modules()
    }
    return sorted(packages, key=str.casefold)


def wiki_module_for_package(package: str) -> str:
    """Accept either a package name or its explicit Wiki module name."""
    value = canonical_wiki_package(package)
    if not value:
        raise ValueError("Package name cannot be empty")
    if value.lower().startswith("building-") or value.lower().endswith(".md"):
        return value
    return f"Building-{value}"
