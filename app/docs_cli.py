"""BuildGuard AI command line pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config.settings import Settings, load_project_environment
from app.reporting.reporter import render_report
from app.service import run_package_comparison


def main(argv: list[str] | None = None) -> int:
    """Run the BuildGuard Wiki + Bash parsing pipeline."""
    arguments = _arguments(argv)

    load_project_environment()

    try:
        settings = Settings()
        return _run_pipeline(
            package=arguments.package,
            settings=settings,
            output_format=arguments.format,
            output_path=arguments.output,
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
    package: str,
    settings: Settings,
    output_format: str = "markdown",
    output_path: str | None = None,
) -> int:
    """Run and render one package's Build Instructions/script comparison."""
    report = run_package_comparison(
        package,
        settings,
        progress=lambda message: print(message, file=sys.stderr),
    )
    rendered_report = render_report(report, output_format)
    if output_path:
        destination = Path(output_path).expanduser()
        destination.write_text(rendered_report + "\n", encoding="utf-8")
        print(f"Wrote {output_format} discrepancy report to {destination}", file=sys.stderr)
    else:
        print(rendered_report)

    return 0


def _arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare a package's Build Instructions with its versioned Bash "
            "automation script and report discrepancies."
        )
    )

    parser.add_argument(
        "package",
        nargs="?",
        default="Alfresco",
        help=(
            "package name (for example Alfresco) or Wiki module name "
            "(for example Building-Alfresco; default: Alfresco)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "html"),
        default="markdown",
        help="report format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the report to this file instead of stdout",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
