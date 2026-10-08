"""Application service for one package's Wiki/script comparison."""

from __future__ import annotations

from collections.abc import Callable

from app.config.settings import Settings
from app.github.client import GitHubClient
from app.intelligence.factory import create_provider
from app.models.build import BuildInstruction
from app.models.report import ComparisonReport
from app.parsers.bash_semantic_parser import parse_bash_steps
from app.parsers.version_parser import extract_version, product_from_module
from app.pipeline.build_pipeline import BuildPipeline
from app.reporting.reporter import build_comparison_report
from app.sources.github import GitHubSource
from app.storage.repository import ParsedSourceRepository

ProgressCallback = Callable[[str], None]

WIKI_PARSER_VERSION = "wiki-buildsteps-v1"
BASH_PARSER_VERSION = "bash-semantic-v1"


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

    notify("Comparing Wiki instructions with the build script using AI...")
    comparison = pipeline.compare(wiki_instruction, bash_instruction)
    return build_comparison_report(
        comparison,
        wiki_instruction,
        bash_instruction,
        package=product,
        version=version,
    )


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
