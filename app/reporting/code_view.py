"""Line annotations used by the Streamlit source-code view."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.models.report import ComparisonReport, StepEvidence
from app.models.validation import MatchStatus


_LOCATION_LINES = re.compile(r"\blines?\s+(\d+)(?:-(\d+))?\b")


@dataclass(frozen=True)
class CodeAnnotation:
    finding_id: str
    reason: str
    comment: str
    css_class: str
    comment_line: int


def annotations_by_line(
    report: ComparisonReport,
    line_count: int,
) -> dict[int, list[CodeAnnotation]]:
    """Map flagged script lines to display-only BuildGuard annotations."""
    annotations: dict[int, list[CodeAnnotation]] = {}
    for finding in report.discrepancies:
        if finding.status == MatchStatus.MISSING:
            continue
        reason, css_class = _finding_reason(finding.status)
        comment = _comment(finding.id, reason, finding.explanation)
        for evidence in finding.script:
            start, end = _evidence_lines(evidence)
            if start is None:
                continue
            start = max(1, start)
            end = min(line_count, max(start, end or start))
            for line_number in range(start, end + 1):
                annotations.setdefault(line_number, []).append(
                    CodeAnnotation(
                        finding_id=finding.id,
                        reason=reason,
                        comment=comment,
                        css_class=css_class,
                        comment_line=start,
                    )
                )
    return annotations


def _evidence_lines(evidence: StepEvidence) -> tuple[int | None, int | None]:
    if evidence.line_start is not None:
        return evidence.line_start, evidence.line_end
    match = _LOCATION_LINES.search(evidence.location)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2) or match.group(1))


def _finding_reason(status: MatchStatus) -> tuple[str, str]:
    if status == MatchStatus.CONFLICT:
        return "Potential conflict", "conflict"
    if status == MatchStatus.UNUSED_CONFIGURATION:
        return "Unused / dead configuration", "unused"
    if status == MatchStatus.EXTRA:
        return "Undocumented script action", "undocumented"
    return status.value.replace("_", " ").title(), "other"


def _comment(finding_id: str, reason: str, explanation: str) -> str:
    detail = " ".join(explanation.split())
    return f"# BuildGuard {finding_id} | {reason}: {detail}"
