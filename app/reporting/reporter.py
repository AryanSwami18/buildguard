"""Human-readable Wiki versus build-script discrepancy reports."""

from textwrap import fill

from app.models.step import BuildStep
from app.models.validation import MatchStatus, StepMatch, ValidationResult

MAX_SOURCE_EXCERPT_CHARS = 1200


def render_report(
    result: ValidationResult,
    title: str | None = None,
    script_name: str | None = None,
) -> str:
    """Render only findings that need review, with source locations and evidence."""
    findings = {
        MatchStatus.MISSING: [
            match for match in result.matches if match.status == MatchStatus.MISSING
        ],
        MatchStatus.CONFLICT: [
            match for match in result.matches if match.status == MatchStatus.CONFLICT
        ],
        MatchStatus.EXTRA: [
            match for match in result.matches if match.status == MatchStatus.EXTRA
        ],
    }
    finding_count = sum(len(items) for items in findings.values())
    status = "REVIEW REQUIRED" if finding_count else "NO MISMATCHES FOUND"

    lines = ["BUILD COMPARISON — MISMATCHES ONLY"]
    if title:
        lines.append(title)
    lines.extend(["", f"Status: {status}"])

    if not finding_count:
        return "\n".join(lines + ["", "No unmatched Wiki or Bash steps were reported."])

    lines.extend(
        [
            f"Findings: {finding_count}",
            (
                "  Wiki steps with no linked Bash implementation: "
                f"{len(findings[MatchStatus.MISSING])}"
            ),
            f"  Wiki/Bash differences: {len(findings[MatchStatus.CONFLICT])}",
            f"  Bash steps with no linked Wiki instruction: {len(findings[MatchStatus.EXTRA])}",
        ]
    )

    if findings[MatchStatus.MISSING]:
        lines.extend(["", "WIKI STEPS WITH NO BASH MATCH"])
        for index, match in enumerate(findings[MatchStatus.MISSING], start=1):
            wiki = match.wiki_step
            if wiki is None:
                continue
            lines.extend(["", f"{index}. Wiki step {wiki.id}"])
            lines.append(_wrap(f"Wiki location: {_location(wiki)}", "   "))
            lines.extend(_render_step_facts(wiki, "Expected"))
            lines.extend(_render_source_excerpt(wiki, "Wiki source"))
            if script_name:
                lines.append(_wrap(f"Script checked: {script_name}", "   "))
            lines.append("   No Bash step was linked to this Wiki instruction.")

    if findings[MatchStatus.CONFLICT]:
        lines.extend(["", "WIKI AND SCRIPT DIFFER"])
        for index, match in enumerate(findings[MatchStatus.CONFLICT], start=1):
            wiki = match.wiki_step
            if wiki is None:
                continue
            bash_steps = _bash_steps(match)
            bash_ids = ", ".join(step.id for step in bash_steps) or "unavailable"
            lines.extend(["", f"{index}. Wiki step {wiki.id} <> Bash {bash_ids}"])
            lines.append(_wrap(f"Wiki location: {_location(wiki)}", "   "))
            lines.extend(_render_step_facts(wiki, "Wiki expects"))
            lines.extend(_render_source_excerpt(wiki, "Wiki source"))
            for step in bash_steps:
                lines.append(_wrap(f"Bash location: {_location(step)}", "   "))
                lines.extend(_render_step_facts(step, "Bash does"))
                lines.extend(_render_source_excerpt(step, "Bash source"))

    if findings[MatchStatus.EXTRA]:
        lines.extend(["", "BASH STEPS WITHOUT A WIKI COUNTERPART"])
        for index, match in enumerate(findings[MatchStatus.EXTRA], start=1):
            bash = match.automation_step
            if bash is None:
                continue
            lines.extend(["", f"{index}. Bash step {bash.id}"])
            lines.append(_wrap(f"Bash location: {_location(bash)}", "   "))
            lines.extend(_render_step_facts(bash, "Bash does"))
            lines.extend(_render_source_excerpt(bash, "Bash source"))

    return "\n".join(lines)


def _bash_steps(match: StepMatch) -> list[BuildStep]:
    steps = match.automation_steps or (
        [match.automation_step] if match.automation_step is not None else []
    )
    unique: dict[str, BuildStep] = {step.id: step for step in steps}
    return list(unique.values())


def _location(step: BuildStep) -> str:
    context = step.context
    parts = [context.source_name or context.source_type]
    if context.heading_path:
        parts.append(" > ".join(context.heading_path))
    elif context.subsection:
        parts.append(f"function {context.subsection}")

    if context.line_start is not None and context.line_end is not None:
        if context.line_start == context.line_end:
            parts.append(f"line {context.line_start}")
        else:
            parts.append(f"lines {context.line_start}-{context.line_end}")
    elif context.location:
        parts.append(context.location)
    else:
        parts.append(step.id)
    return " | ".join(parts)


def _render_step_facts(step: BuildStep, label: str) -> list[str]:
    facts: list[tuple[str, str]] = [(label, step.description)]
    extracted = (
        ("Action", step.action.value),
        ("Packages", ", ".join(step.packages)),
        ("Commands", " ; ".join(step.commands)),
        ("Paths", ", ".join(step.paths)),
        ("URLs", ", ".join(step.urls)),
        ("Repository", step.repository or ""),
        ("Ref", step.ref or ""),
        ("Commit", step.commit or ""),
        ("Version", step.version or ""),
        ("Docker image", step.docker_image or ""),
        ("Dockerfile", step.dockerfile or ""),
        (
            "Build arguments",
            ", ".join(f"{key}={value}" for key, value in step.build_args.items()),
        ),
        (
            "Environment",
            ", ".join(f"{key}={value}" for key, value in step.environment.items()),
        ),
        ("Patches", ", ".join(step.patches)),
        ("Build tool", step.build_tool or ""),
        ("Conditions", "; ".join(step.context.conditions)),
    )
    facts.extend((key, value) for key, value in extracted if value)
    facts.extend(
        (f"Supported {name} options", ", ".join(options))
        for name, options in step.context.runtime_options.items()
        if options
    )
    facts.extend(
        (f"Default {name}", value)
        for name, value in step.context.runtime_defaults.items()
        if value
    )
    return [_wrap(f"{key}: {value}", "   ") for key, value in facts]


def _render_source_excerpt(step: BuildStep, label: str) -> list[str]:
    source = (
        step.context.source_excerpt
        if step.context.source_type == "wiki"
        else step.context.surrounding_text
    )
    if not source:
        return []
    truncated = len(source) > MAX_SOURCE_EXCERPT_CHARS
    if truncated:
        source = source[:MAX_SOURCE_EXCERPT_CHARS].rstrip() + "…"
    lines = [f"   {label}:", *(f"     {line}" for line in source.splitlines())]
    if truncated:
        lines.append("     [excerpt truncated]")
    return lines


def _wrap(text: str, indent: str) -> str:
    return fill(
        " ".join(text.split()),
        width=88,
        initial_indent=indent,
        subsequent_indent=" " * len(indent),
    )
