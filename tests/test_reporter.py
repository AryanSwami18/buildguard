from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.report import ComparisonReport
from app.models.step import ActionType, BuildStep
from app.models.validation import (
    DiscrepancyCategory,
    MatchStatus,
    StepMatch,
    ValidationResult,
)
from app.reporting.reporter import build_comparison_report, render_report


def _step(step_id: str, source_type: str) -> BuildStep:
    return BuildStep(
        id=step_id,
        order=1,
        action=ActionType.RUN_COMMAND,
        description="Build the component",
        commands=["build-component"],
        context=SourceContext(
            source_type=source_type,
            source_name=source_type,
            line_start=1,
            line_end=1,
        ),
    )


def _report(status: MatchStatus) -> ComparisonReport:
    wiki_step = _step("wiki-1", "Wiki")
    script_step = _step("bash-1", "build.sh")
    result = ValidationResult(
        matches=[
            StepMatch(
                wiki_step=wiki_step,
                automation_step=script_step,
                status=status,
                category=(
                    DiscrepancyCategory.COMMAND_MISMATCH
                    if status == MatchStatus.CONFLICT
                    else None
                ),
                explanation="The sources differ.",
            )
        ],
        valid=status == MatchStatus.MATCHED,
        summary="",
    )
    return build_comparison_report(
        result,
        BuildInstruction(name="Wiki", source="wiki", steps=[wiki_step]),
        BuildInstruction(name="build.sh", source="script", steps=[script_step]),
        "Demo",
        "1.0",
    )


def test_report_leads_with_plain_language_and_actionable_findings() -> None:
    report = _report(MatchStatus.CONFLICT)

    markdown = render_report(report, "markdown")
    html = render_report(report, "html")

    assert "1 actionable item needs review" in report.overview
    assert "At a glance" in markdown
    assert "What needs attention" in markdown
    assert "Why this needs attention" in markdown
    assert "How to read this report" in html
    assert "What the Wiki says" in html
    assert "Technical classification" in html


def test_report_explains_when_there_are_no_actionable_findings() -> None:
    report = _report(MatchStatus.MATCHED)

    assert report.counts.findings == 0
    assert "No actionable differences were found" in report.overview
    assert "No actionable issues found" in render_report(report, "markdown")
