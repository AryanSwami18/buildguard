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

ProgressCallback = Callable[[str], None]


def run_package_comparison(
    package: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> ComparisonReport:
    """Fetch, parse, and compare one package's documentation and script."""
    notify = progress or (lambda _message: None)
    module = wiki_module_for_package(package)
    github = GitHubClient.from_settings(settings)

    notify(f"Fetching Wiki module '{module}'...")
    page = github.fetch_module_wiki(module)

    product = product_from_module(module)
    version = extract_version(page.content, product)
    if version is None:
        raise ValueError(
            f"Could not determine the product version from Wiki module '{module}'."
        )

    pipeline = BuildPipeline(
        create_provider(settings),
        progress=notify,
    )
    notify("Extracting Wiki build instructions with AI...")
    wiki_instruction = pipeline.from_github_wiki(page, version=version)

    notify(f"Fetching {product} {version} build script...")
    script_file = github.fetch_build_script(
        repository=settings.scripts_repo,
        product=product,
        version=version,
        ref=settings.scripts_branch,
    )
    bash_instruction = BuildInstruction(
        name=script_file.filename,
        source="bash",
        version=version,
        source_metadata=GitHubSource.model_validate(script_file.model_dump()),
        steps=parse_bash_steps(script_file.content, script_file.filename),
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


def wiki_module_for_package(package: str) -> str:
    """Accept either a package name or its explicit Wiki module name."""
    value = package.strip()
    if not value:
        raise ValueError("Package name cannot be empty")
    if value.lower().startswith("building-") or value.lower().endswith(".md"):
        return value
    return f"Building-{value}"
