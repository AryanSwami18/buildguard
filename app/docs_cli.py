"""BuildGuard AI command line pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from app.config.settings import Settings
from app.github.client import GitHubClient
from app.intelligence.factory import create_provider
from app.models.build import BuildInstruction
from app.parsers.bash_semantic_parser import parse_bash_steps
from app.parsers.version_parser import extract_version, product_from_module
from app.pipeline.build_pipeline import BuildPipeline
from app.reporting.reporter import render_report
from app.sources.github import GitHubSource


def main(argv: list[str] | None = None) -> int:
    """Run the BuildGuard Wiki + Bash parsing pipeline."""
    arguments = _arguments(argv)

    load_dotenv()

    try:
        settings = Settings()
        return _run_pipeline(
            module=arguments.module,
            settings=settings,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"buildguard: {error}", file=sys.stderr)
        return 1


def _run_pipeline(
    module: str,
    settings: Settings,
) -> int:
    """Fetch, parse, and display the Wiki and matching Bash script."""

    github = GitHubClient.from_settings(settings)
    print(f"Fetching Wiki module '{module}'...", file=sys.stderr)

    # ------------------------------------------------------------------
    # 1. Fetch and parse the Build Instruction Wiki
    # ------------------------------------------------------------------

    page = github.fetch_module_wiki(module)

    # ------------------------------------------------------------------
    # 2. Determine product and version from the Wiki
    # ------------------------------------------------------------------

    product = product_from_module(module)

    version = extract_version(
        page.content,
        product,
    )

    if version is None:
        raise ValueError(
            f"Could not determine the product version from Wiki module "
            f"'{module}'."
        )

    provider = create_provider(settings)
    pipeline = BuildPipeline(
        provider,
        progress=lambda message: print(message, file=sys.stderr),
    )

    # ------------------------------------------------------------------
    # 3. Convert the Wiki into a BuildInstruction
    # ------------------------------------------------------------------

    print("Extracting Wiki build instructions with AI...", file=sys.stderr)
    wiki_instruction = pipeline.from_github_wiki(
        page,
        version=version,
    )

    # ------------------------------------------------------------------
    # 4. Find the matching versioned Bash script
    # ------------------------------------------------------------------

    print(f"Fetching {product} {version} build script...", file=sys.stderr)
    script_file = github.fetch_build_script(
        repository=settings.scripts_repo,
        product=product,
        version=version,
        ref=settings.scripts_branch,
    )

    bash_steps = parse_bash_steps(
        script_file.content,
        script_file.filename,
    )

    bash_instruction = BuildInstruction(
        name=script_file.filename,
        source="bash",
        version=version,
        source_metadata=GitHubSource.model_validate(
            script_file.model_dump()
        ),
        steps=bash_steps,
    )

    print(
        "Comparing Wiki instructions with the build script using AI...",
        file=sys.stderr,
    )
    comparison = pipeline.compare(wiki_instruction, bash_instruction)
    print(
        render_report(
            comparison,
            f"{product} {version}",
            script_name=script_file.filename,
        )
    )

    return 0


def _arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Parse a BuildGuard Wiki instruction and its "
            "matching Bash automation script."
        )
    )

    parser.add_argument(
        "module",
        nargs="?",
        default="Building-Alfresco",
        help=(
            "Build Instruction Wiki module, "
            "for example Building-Alfresco "
            "(default: Building-Alfresco)"
        ),
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
