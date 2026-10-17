"""Generate and validate human-reviewable build-script fix proposals."""

from __future__ import annotations

import json
import difflib
from pathlib import Path
from typing import Any, Callable

from app.intelligence.interface import LLMProvider
from app.intelligence.request import LLMRequest
from app.models.repair import ScriptFixProposal
from app.models.report import ComparisonReport
from app.parsers.bash_parser import parse_script


PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "script_fix.txt"
ProgressCallback = Callable[[str], None]


def generate_script_fix_proposal(
    provider: LLMProvider,
    report: ComparisonReport,
    original_script: str,
    progress: ProgressCallback | None = None,
) -> ScriptFixProposal:
    """Ask the configured model for a minimal complete-script correction."""
    if not report.discrepancies:
        raise ValueError("A fix proposal can only be generated for actionable findings")
    if not original_script.strip():
        raise ValueError("The build script is empty; a fix proposal cannot be generated")

    deterministic = _deterministic_unused_configuration_proposal(
        report,
        original_script,
    )
    if deterministic is not None:
        # This path cannot introduce unrelated model edits because it removes
        # only the exact declaration lines linked by the report.
        return deterministic

    notify = progress or (lambda _message: None)
    notify("Generating a proposed script fix with AI...")
    request = LLMRequest(
        system_prompt=PROMPT.read_text(encoding="utf-8"),
        user_prompt=json.dumps(
            {
                "original_script": original_script,
                "findings": [
                    finding.model_dump(mode="json", exclude_none=True)
                    for finding in report.discrepancies
                ],
                "expected_output": ScriptFixProposal.model_json_schema(),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    response = provider.generate(request).content
    proposal = ScriptFixProposal.model_validate(_json_object(response))
    proposal.fixed_script = _preserve_eof_style(
        _strip_code_fence(proposal.fixed_script),
        original_script,
    )
    if not proposal.fixed_script.strip():
        raise RuntimeError("The AI returned an empty fixed script")
    try:
        parse_script(proposal.fixed_script)
    except Exception as error:  # bashlex uses several exception types.
        raise RuntimeError(f"The proposed script is not valid Bash syntax: {error}") from error
    _validate_proposal_scope(proposal, report, original_script)
    return proposal


def _deterministic_unused_configuration_proposal(
    report: ComparisonReport,
    original_script: str,
) -> ScriptFixProposal | None:
    """Remove only exact, source-linked unused declarations.

    An LLM is unnecessary for this finding type. More importantly, asking it
    to reproduce an entire script creates opportunities for unrelated
    whitespace, indentation, or EOF changes. The report already contains the
    declaration's line and variable name, so this edit can be exact.
    """
    if not report.discrepancies or any(
        finding.category.value != "unused_configuration"
        for finding in report.discrepancies
    ):
        return None

    lines = original_script.splitlines(keepends=True)
    removals: list[tuple[int, str]] = []
    for finding in report.discrepancies:
        evidence = finding.script
        if len(evidence) != 1:
            return None
        item = evidence[0]
        name = next(iter(item.environment), None)
        if not name or item.line_start is None or item.line_end != item.line_start:
            return None
        index = item.line_start - 1
        if not 0 <= index < len(lines):
            return None
        line = lines[index]
        assignment = _assignment_name(line)
        if assignment != name:
            return None
        if item.commands and line.strip() not in {
            command.strip() for command in item.commands
        }:
            return None
        removals.append((index, finding.id))

    if not removals:
        return None
    for index, _ in sorted(removals, reverse=True):
        del lines[index]
    removed_ids = [finding_id for _, finding_id in removals]
    return ScriptFixProposal(
        fixed_script="".join(lines),
        summary=(
            "Removed only the unused configuration declaration(s) identified "
            "by static analysis."
        ),
        rationale=(
            "The affected variable declaration has no static references in the "
            "script, so the edit is limited to its source-linked line."
        ),
        affected_finding_ids=removed_ids,
    )


def _assignment_name(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    if "=" not in stripped:
        return None
    name = stripped.split("=", 1)[0].strip()
    if name and all(character.isalnum() or character == "_" for character in name):
        return name
    return None


def _preserve_eof_style(fixed_script: str, original_script: str) -> str:
    """Normalize model wrappers without changing the source's EOF convention."""
    if original_script.endswith(("\n", "\r")):
        if not fixed_script.endswith("\n"):
            fixed_script += "\n"
        return fixed_script
    return fixed_script.rstrip("\r\n")


def _changed_line_ranges(original: str, fixed: str) -> list[tuple[int, int, int, int]]:
    original_lines = original.splitlines()
    fixed_lines = fixed.splitlines()
    matcher = difflib.SequenceMatcher(a=original_lines, b=fixed_lines, autojunk=False)
    return [
        (old_start + 1, old_end, new_start + 1, new_end)
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes()
        if tag != "equal"
    ]


def _validate_proposal_scope(
    proposal: ScriptFixProposal,
    report: ComparisonReport,
    original_script: str,
) -> None:
    """Reject complete-script responses that modify unrelated source lines."""
    finding_ids = {finding.id for finding in report.discrepancies}
    affected_ids = set(proposal.affected_finding_ids)
    if not affected_ids.issubset(finding_ids):
        unknown = ", ".join(sorted(affected_ids - finding_ids))
        raise RuntimeError(f"The proposal references unknown finding IDs: {unknown}")

    changes = _changed_line_ranges(original_script, proposal.fixed_script)
    if not changes:
        if affected_ids:
            raise RuntimeError("The proposal claims findings but does not change the script")
        return

    allowed_ranges: list[tuple[int, int]] = []
    unsupported_findings: list[str] = []
    for finding in report.discrepancies:
        if finding.id not in affected_ids:
            continue
        if finding.script and all(
            evidence.line_start is not None and evidence.line_end is not None
            for evidence in finding.script
        ):
            allowed_ranges.extend(
                (evidence.line_start, evidence.line_end)
                for evidence in finding.script
            )
        else:
            unsupported_findings.append(finding.id)

    if unsupported_findings:
        raise RuntimeError(
            "The proposal changes findings without source-linked script lines: "
            + ", ".join(unsupported_findings)
        )

    def in_allowed_range(start: int, end: int) -> bool:
        # Insertions have start > end; anchor them to the preceding source line.
        anchor = start if start <= end else max(1, start - 1)
        return any(
            allowed_start <= anchor <= allowed_end
            for allowed_start, allowed_end in allowed_ranges
        )

    unrelated = [change for change in changes if not in_allowed_range(change[0], change[1])]
    if unrelated:
        rendered = ", ".join(
            f"original lines {start}-{end or start}"
            for start, end, _, _ in unrelated
        )
        raise RuntimeError(
            "The proposed fix changes unrelated script lines ("
            f"{rendered}). Generate a smaller finding-backed fix."
        )


def _json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        preview = " ".join(content.split())[:240]
        raise RuntimeError(f"AI returned invalid fix JSON: {preview}") from error
    if not isinstance(data, dict):
        raise RuntimeError("AI returned a non-object fix proposal")
    return data


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped
