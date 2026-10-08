from app.models.report import (
    ComparisonCounts,
    ComparisonReport,
    Discrepancy,
    StepEvidence,
)
from app.models.step import ActionType
from app.models.validation import DiscrepancyCategory, MatchStatus
from app.reporting.code_view import annotations_by_line


def test_code_annotations_cover_flagged_lines_and_comment_only_once() -> None:
    report = ComparisonReport(
        package="Bazel",
        version="9.2.0",
        status="review_required",
        overview="One finding.",
        counts=ComparisonCounts(
            wiki_steps_compared=1,
            script_steps_checked=4,
            missing_steps=0,
            conflicting_steps=0,
            undocumented_script_steps=1,
            out_of_scope_steps=0,
            findings=1,
        ),
        discrepancies=[
            Discrepancy(
                id="D-001",
                status=MatchStatus.EXTRA,
                category=DiscrepancyCategory.UNDOCUMENTED_STEP,
                title="Bazel clone",
                explanation="This clone is not represented in the Wiki.",
                recommendation="Document it.",
                script=[
                    StepEvidence(
                        id="bash-22",
                        description="Clone Bazel",
                        action=ActionType.CLONE_REPOSITORY,
                        location="build.sh | line 2-3",
                        line_start=2,
                        line_end=3,
                    )
                ],
            )
        ],
    )

    annotations = annotations_by_line(report, line_count=5)

    assert set(annotations) == {2, 3}
    assert annotations[2][0].comment_line == 2
    assert annotations[3][0].comment_line == 2
    assert annotations[2][0].css_class == "undocumented"
    assert annotations[2][0].comment.startswith("# BuildGuard D-001")
