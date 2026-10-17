"""Application service for one package's Wiki/script comparison."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import re
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config.settings import Settings
from app.github.client import GitHubClient
from app.intelligence.factory import create_provider
from app.models.build import BuildInstruction
from app.models.repair import ScriptFixProposal
from app.models.report import ComparisonReport
from app.parsers.bash_semantic_parser import parse_bash_steps
from app.parsers.version_parser import extract_version, product_from_module
from app.pipeline.build_pipeline import BuildPipeline
from app.reporting.reporter import build_comparison_report
from app.repair import generate_script_fix_proposal
from app.sources.github import GitHubSource
from app.storage.repository import (
    ComparisonCacheKey,
    ParsedSourceRepository,
    StoredComparisonReport,
)

ProgressCallback = Callable[[str], None]

WIKI_PARSER_VERSION = "wiki-buildsteps-v1"
BASH_PARSER_VERSION = "bash-semantic-v2"
COMPARISON_ANALYSIS_VERSION = "comparison-v1"

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
    )
    product = product_from_module(module)
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
            return cached_report.report

    notify("Comparing Wiki instructions with the build script using AI...")
    comparison = pipeline.compare(wiki_instruction, bash_instruction)
    report = build_comparison_report(
        comparison,
        wiki_instruction,
        bash_instruction,
        package=product,
        version=version,
    )
    if cache_key is not None:
        cache.save_comparison_report(cache_key, report)
    return report


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
    if not wiki_revision or not script_revision:
        # A report is only reusable when both remote revisions were verified.
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

    job.future = _COMPARISON_EXECUTOR.submit(
        run_package_comparison,
        package,
        settings,
        notify,
    )
    return job_id


def get_package_comparison(
    job_id: str,
) -> tuple[str, str, int, ComparisonReport | None, str | None]:
    """Return comparison status, progress, result, and error."""
    with _COMPARISON_JOBS_LOCK:
        job = _COMPARISON_JOBS.get(job_id)
        if job is None:
            return "missing", "Comparison job is no longer available.", 0, None, (
                "The comparison job is no longer available."
            )
        future = job.future
        progress = job.progress
        progress_percent = job.progress_percent

    if future is None or not future.done():
        return "running", progress, progress_percent, None, None

    try:
        report = future.result()
    except Exception as error:  # Provider and source failures are shown in the UI.
        return "failed", progress, progress_percent, None, str(error) or error.__class__.__name__
    return "complete", "Comparison complete.", 100, report, None


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


def create_script_fix_pull_request(
    report: ComparisonReport,
    original_script: str,
    proposal: ScriptFixProposal,
    settings: Settings,
) -> dict[str, Any]:
    """Commit a reviewed proposal and open a PR against the scripts repository."""
    source = report.script_source
    if source is None:
        raise ValueError("This report has no script source to update")
    if source.repository != settings.scripts_repo:
        raise ValueError(
            "The report script repository differs from the configured scripts repository"
        )

    title = f"BuildGuard: fix {report.package} {report.version or 'build script'}"
    findings = "\n".join(
        f"- `{finding.id}`: {finding.title}"
        for finding in report.discrepancies
    )
    body = (
        "## BuildGuard proposed build-script fix\n\n"
        f"Package: `{report.package}`\n\n"
        f"Script: `{source.path}`\n\n"
        "### Findings addressed\n"
        f"{findings}\n\n"
        f"### Summary\n{proposal.summary}\n\n"
        f"### Rationale\n{proposal.rationale}\n\n"
        "This change was generated by BuildGuard and should be reviewed before merging."
    )
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
    page = github.fetch_module_wiki(module)
    product = product_from_module(module)
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
    if remote_revision is not None:
        cache.save_instruction(
            source_kind="wiki",
            repository=settings.github_repo,
            path=page.path,
            ref=settings.github_branch,
            parser_version=parser_version,
            remote_revision=remote_revision,
            content=page.content,
            instruction=instruction,
        )
    return instruction


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
    script_path, remote_revision = github.find_build_script_reference(
        repository=settings.scripts_repo,
        product=product,
        version=version,
        ref=settings.scripts_branch,
    )
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
    value = package.strip()
    if not value:
        raise ValueError("Package name cannot be empty")
    if value.lower().startswith("building-") or value.lower().endswith(".md"):
        return value
    return f"Building-{value}"
