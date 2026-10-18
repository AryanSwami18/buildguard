"""Generate and validate human-reviewable build-script fix proposals."""

from __future__ import annotations

import json
import difflib
from pathlib import Path
import re
from typing import Any, Callable

from app.intelligence.interface import LLMProvider
from app.intelligence.request import LLMRequest
from app.models.repair import ScriptFixEdit, ScriptFixPlan, ScriptFixProposal
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
    """Ask the configured model for a minimal source-linked correction."""
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
                "expected_output": ScriptFixPlan.model_json_schema(),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    response = provider.generate(request).content
    response_data = _json_object(response)
    if "edits" in response_data or "fixed_script" not in response_data:
        try:
            plan = ScriptFixPlan.model_validate(response_data)
        except ValueError as error:
            raise RuntimeError(
                "AI returned an incomplete fix plan. The response must include "
                "edits, summary, rationale, and affected_finding_ids."
            ) from error
        return _apply_fix_plan(plan, report, original_script)

    # Accept the previous complete-script format for already configured/custom
    # providers, but new requests use ScriptFixPlan so unrelated lines never
    # depend on how accurately the model reproduces the whole file.
    proposal = ScriptFixProposal.model_validate(response_data)
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


def _variable_assignment_ranges(
    original_script: str,
    variable_names: set[str],
) -> list[tuple[int, int]]:
    """Find exact source lines defining variables cited by finding evidence.

    Semantic comparison evidence often links the command that consumes a
    variable rather than the earlier declaration that supplies its value. A
    repair may safely update that declaration when the variable name is part
    of the finding's structured evidence.
    """
    if not variable_names:
        return []
    assignment_pattern = re.compile(
        r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*="
    )
    ranges: list[tuple[int, int]] = []
    for line_number, line in enumerate(original_script.splitlines(), start=1):
        match = assignment_pattern.match(line)
        if match and match.group(1) in variable_names:
            ranges.append((line_number, line_number))
    return ranges


def _finding_source_ranges(
    finding: Any,
    original_script: str,
) -> list[tuple[int, int]]:
    """Return direct evidence ranges plus referenced variable declarations."""
    ranges: list[tuple[int, int]] = []
    variable_names: set[str] = set()
    for evidence in finding.script:
        if evidence.line_start is not None and evidence.line_end is not None:
            ranges.append((evidence.line_start, evidence.line_end))
        variable_names.update(evidence.environment)
    ranges.extend(_variable_assignment_ranges(original_script, variable_names))
    return ranges


def _apply_fix_plan(
    plan: ScriptFixPlan,
    report: ComparisonReport,
    original_script: str,
) -> ScriptFixProposal:
    """Apply validated source-linked edits to the untouched original script."""
    findings = {finding.id: finding for finding in report.discrepancies}
    edit_ids = {edit.finding_id for edit in plan.edits}
    declared_ids = set(plan.affected_finding_ids)
    if declared_ids != edit_ids:
        raise RuntimeError(
            "The fix plan's affected findings do not match its edits. "
            "Generate a plan with one finding ID per source-linked edit."
        )

    source_lines = original_script.splitlines(keepends=True)
    if not source_lines:
        raise RuntimeError("The build script is empty; a fix plan cannot be applied")

    normalized_source = original_script.replace("\r\n", "\n").replace("\r", "\n")
    line_ending = "\r\n" if "\r\n" in original_script else "\n"
    resolved_edits: list[tuple[ScriptFixEdit, int, int, list[str]]] = []
    for edit in plan.edits:
        if edit.finding_id not in findings:
            raise RuntimeError(
                f"The fix plan references unknown finding ID: {edit.finding_id}"
            )

        finding = findings[edit.finding_id]
        evidence_ranges = _finding_source_ranges(finding, original_script)
        if not evidence_ranges:
            raise RuntimeError(
                f"Finding {edit.finding_id} has no source-linked Bash lines; "
                "BuildGuard cannot safely create an edit for it."
            )
        original_text = edit.original_text.replace("\r\n", "\n").replace("\r", "\n")
        if not original_text:
            raise RuntimeError(
                f"Edit for {edit.finding_id} has no original_text patch anchor."
            )
        occurrences: list[tuple[int, int, int]] = []
        search_from = 0
        while True:
            match_start = normalized_source.find(original_text, search_from)
            if match_start < 0:
                break
            match_end = match_start + len(original_text)
            starts_at_line = match_start == 0 or normalized_source[match_start - 1] == "\n"
            ends_at_line = (
                match_end == len(normalized_source)
                or normalized_source[match_end - 1] == "\n"
                or normalized_source[match_end] == "\n"
            )
            if starts_at_line and ends_at_line:
                actual_start = normalized_source.count("\n", 0, match_start) + 1
                old_lines = normalized_source[match_start:match_end].splitlines()
                actual_end = actual_start + max(1, len(old_lines)) - 1
                if any(
                    actual_start <= end and actual_end >= start
                    for start, end in evidence_ranges
                ):
                    occurrences.append((match_start, actual_start, actual_end))
            search_from = match_start + 1
        if len(occurrences) != 1:
            raise RuntimeError(
                f"The original block for {edit.finding_id} was not found exactly "
                f"once within its linked Bash evidence; found {len(occurrences)} matches."
            )

        match_start, actual_start, actual_end = occurrences[0]
        match_end = match_start + len(original_text)

        replacement_normalized = edit.replacement.replace("\r\n", "\n").replace("\r", "\n")
        local_changes = _changed_line_ranges(original_text, replacement_normalized)
        for old_start, old_end, _, _ in local_changes:
            if old_start <= old_end:
                changed_start = actual_start + old_start - 1
                changed_end = actual_start + old_end - 1
            else:
                changed_start = changed_end = actual_start + old_start - 1
            if not any(
                changed_start <= end and changed_end >= start
                for start, end in evidence_ranges
            ):
                raise RuntimeError(
                    f"Edit for {edit.finding_id} changes lines "
                    f"{changed_start}-{changed_end} outside its linked Bash evidence."
                )

        replacement_lines = _edit_replacement_lines(
            edit.replacement,
            line_ending,
            actual_end,
            len(source_lines),
            original_script.endswith(("\n", "\r")),
        )
        resolved_edits.append((edit, actual_start, actual_end, replacement_lines))

    ordered_edits = sorted(resolved_edits, key=lambda item: (item[1], item[2]))
    previous_end = 0
    for _, actual_start, actual_end, _ in ordered_edits:
        if actual_start <= previous_end:
            raise RuntimeError("The fix plan contains overlapping script edits")
        previous_end = actual_end

    fixed_lines = list(source_lines)
    changed = False
    for edit, actual_start, actual_end, replacement_lines in reversed(ordered_edits):
        old_text = "".join(fixed_lines[actual_start - 1 : actual_end])
        new_text = "".join(replacement_lines)
        changed = changed or old_text != new_text
        fixed_lines[actual_start - 1 : actual_end] = replacement_lines

    if plan.edits and not changed:
        raise RuntimeError("The fix plan does not change the original script")

    fixed_script = _preserve_eof_style("".join(fixed_lines), original_script)
    try:
        parse_script(fixed_script)
    except Exception as error:  # bashlex uses several exception types.
        raise RuntimeError(f"The proposed script is not valid Bash syntax: {error}") from error
    return ScriptFixProposal(
        fixed_script=fixed_script,
        summary=plan.summary,
        rationale=plan.rationale,
        affected_finding_ids=plan.affected_finding_ids,
    )


def _edit_replacement_lines(
    replacement: str,
    line_ending: str,
    end_line: int,
    source_line_count: int,
    source_has_final_newline: bool,
) -> list[str]:
    """Normalize one model edit to the source file's line-ending convention."""
    normalized = replacement.replace("\r\n", "\n").replace("\r", "\n")
    if line_ending == "\r\n":
        normalized = normalized.replace("\n", line_ending)
    replacement_lines = normalized.splitlines(keepends=True)
    if normalized and not normalized.endswith("\n"):
        if end_line < source_line_count or source_has_final_newline:
            replacement_lines.append(line_ending)
    return replacement_lines


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
            allowed_ranges.extend(_finding_source_ranges(finding, original_script))
        else:
            unsupported_findings.append(finding.id)

    if unsupported_findings:
        raise RuntimeError(
            "The proposal changes findings without source-linked script lines: "
            + ", ".join(unsupported_findings)
        )

    # Models sometimes re-indent otherwise unchanged lines while reproducing a
    # complete script.  Keep the source text for those harmless formatting
    # changes so the proposal remains a minimal, reviewable edit.  Substantive
    # edits outside a finding's source range are still rejected below.
    proposal.fixed_script = _preserve_only_finding_changes(
        original_script,
        proposal.fixed_script,
        allowed_ranges,
    )
    changes = _changed_line_ranges(original_script, proposal.fixed_script)

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


def _preserve_only_finding_changes(
    original_script: str,
    proposed_script: str,
    allowed_ranges: list[tuple[int, int]],
) -> str:
    """Discard formatting-only model edits outside finding-linked lines.

    The repair prompt asks the model to return a complete script.  Complete
    script reproduction is useful for review, but models can change indentation
    or whitespace on unrelated lines.  This function restores those lines from
    the original source while leaving changes in allowed ranges untouched.
    It deliberately raises for substantive out-of-scope edits.
    """
    original_lines = original_script.splitlines(keepends=True)
    proposed_lines = proposed_script.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(
        a=original_lines,
        b=proposed_lines,
        autojunk=False,
    )

    def allowed(line_number: int) -> bool:
        return any(start <= line_number <= end for start, end in allowed_ranges)

    def formatting_only(left: str, right: str) -> bool:
        return left.rstrip("\r\n").strip() == right.rstrip("\r\n").strip()

    def non_behavioral_log_line(line: str) -> bool:
        """Identify a plain status message, not a build/configuration command."""
        stripped = line.strip()
        if not stripped or (stripped.startswith("#") and not stripped.startswith("#!")):
            return True
        if not (stripped.startswith("printf ") or stripped.startswith("echo ")):
            return False
        # A redirected message can write an environment/configuration file, so
        # leave those lines under the normal strict scope validation. Piping a
        # status message to tee is still logging, not a build/configuration
        # operation.
        if "|& tee" in stripped or "| tee" in stripped:
            return True
        return not any(operator in stripped for operator in (">", "|", "$(", "`"))

    repaired: list[str] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        old_block = original_lines[old_start:old_end]
        new_block = proposed_lines[new_start:new_end]
        if tag == "equal":
            repaired.extend(old_block)
            continue

        if tag == "replace" and len(old_block) == len(new_block):
            for offset, (old_line, new_line) in enumerate(zip(old_block, new_block)):
                line_number = old_start + offset + 1
                if allowed(line_number):
                    repaired.append(new_line)
                elif formatting_only(old_line, new_line) or non_behavioral_log_line(old_line):
                    repaired.append(old_line)
                else:
                    raise RuntimeError(
                        "The proposed fix changes unrelated script lines ("
                        f"original lines {line_number}-{line_number}). Generate "
                        "a smaller finding-backed fix."
                    )
            continue

        anchor = old_start + 1 if old_start < old_end else max(1, old_start)
        if any(allowed(old_start + offset + 1) for offset in range(len(old_block))):
            repaired.extend(new_block)
            continue

        # Adding/removing blank lines is formatting-only.  Preserve the
        # original block so the proposal cannot silently reorder source code.
        if all(not line.rstrip("\r\n").strip() for line in old_block + new_block):
            repaired.extend(old_block)
            continue

        raise RuntimeError(
            "The proposed fix changes unrelated script lines ("
            f"original lines {anchor}-{max(anchor, old_end)}). Generate a "
            "smaller finding-backed fix."
        )

    return "".join(repaired)


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
