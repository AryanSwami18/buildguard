"""Build structured discrepancy data and render JSON, Markdown, or HTML."""

from __future__ import annotations

import html
import json
from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.report import (
    ComparisonCounts,
    ComparisonReport,
    Discrepancy,
    ReportSource,
    StepEvidence,
)
from app.models.step import BuildStep
from app.models.validation import (
    DiscrepancyCategory,
    MatchStatus,
    StepMatch,
    ValidationResult,
)


def build_comparison_report(
    result: ValidationResult,
    wiki: BuildInstruction,
    script: BuildInstruction,
    package: str,
    version: str | None,
) -> ComparisonReport:
    """Convert validated matches into a source-linked mismatch-only report."""
    discrepancies: list[Discrepancy] = []
    for match in result.matches:
        if match.status not in {
            MatchStatus.MISSING,
            MatchStatus.CONFLICT,
            MatchStatus.EXTRA,
        }:
            continue

        wiki_evidence = _step_evidence(match.wiki_step) if match.wiki_step else None
        script_steps = _script_steps(match)
        category = _category(match)
        recommendation = match.recommendation or _recommendation(match.status, category)
        discrepancies.append(
            Discrepancy(
                id=f"D-{len(discrepancies) + 1:03d}",
                status=match.status,
                category=category,
                title=_discrepancy_title(match, category),
                explanation=match.explanation,
                recommendation=recommendation,
                wiki=wiki_evidence,
                script=[_step_evidence(step) for step in script_steps],
            )
        )

    by_category: dict[str, int] = {}
    for discrepancy in discrepancies:
        key = discrepancy.category.value
        by_category[key] = by_category.get(key, 0) + 1

    missing_count = sum(
        item.status == MatchStatus.MISSING for item in discrepancies
    )
    conflict_count = sum(
        item.status == MatchStatus.CONFLICT for item in discrepancies
    )
    extra_count = sum(item.status == MatchStatus.EXTRA for item in discrepancies)
    out_of_scope_count = sum(
        match.status == MatchStatus.OUT_OF_SCOPE for match in result.matches
    )
    status = "review_required" if discrepancies else "no_mismatches_found"
    overview = (
        f"Identified {len(discrepancies)} potential discrepancies between the "
        f"{len(wiki.steps)} Wiki steps and {len(script.steps)} script steps."
        if discrepancies
        else "No unmatched or conflicting steps were reported."
    )

    return ComparisonReport(
        package=package,
        version=version,
        status=status,
        overview=overview,
        counts=ComparisonCounts(
            wiki_steps_compared=len(wiki.steps),
            script_steps_checked=len(script.steps),
            missing_steps=missing_count,
            conflicting_steps=conflict_count,
            undocumented_script_steps=extra_count,
            out_of_scope_steps=out_of_scope_count,
            findings=len(discrepancies),
            by_category=by_category,
        ),
        wiki_source=_source_reference(wiki),
        script_source=_source_reference(script),
        discrepancies=discrepancies,
    )


def render_report(
    report: ComparisonReport,
    output_format: str = "markdown",
) -> str:
    """Render a structured report without including successful matches."""
    normalized_format = output_format.lower().strip()
    if normalized_format == "json":
        return json.dumps(
            report.model_dump(mode="json", exclude_none=True),
            indent=2,
            ensure_ascii=False,
        )
    if normalized_format in {"markdown", "md"}:
        return _render_markdown(report)
    if normalized_format == "html":
        return _render_html(report)
    raise ValueError(
        f"Unsupported report format '{output_format}'. Use markdown, json, or html."
    )


def _category(match: StepMatch) -> DiscrepancyCategory:
    if match.status == MatchStatus.MISSING:
        return DiscrepancyCategory.MISSING_STEP
    if match.status == MatchStatus.EXTRA:
        return DiscrepancyCategory.UNDOCUMENTED_STEP
    return match.category or DiscrepancyCategory.OTHER_DEVIATION


def _recommendation(
    status: MatchStatus,
    category: DiscrepancyCategory,
) -> str:
    if status == MatchStatus.MISSING:
        return (
            "Implement the operation documented in the Wiki, preserving its stated "
            "commands, values, and platform conditions."
        )
    if status == MatchStatus.EXTRA:
        return (
            "Confirm whether this operation is required. If it is, document it in the "
            "Build Instructions; otherwise remove or disable it only after checking "
            "its effects."
        )
    category_name = _category_label(category).lower()
    return (
        f"Resolve the {category_name} using the source evidence. Update the script to "
        "match the Wiki, or revise the Wiki if it is no longer authoritative."
    )


def _discrepancy_title(
    match: StepMatch,
    category: DiscrepancyCategory,
) -> str:
    if match.status == MatchStatus.MISSING and match.wiki_step:
        return f"Wiki step {match.wiki_step.id}: {match.wiki_step.description}"
    if match.status == MatchStatus.EXTRA and match.automation_step:
        return f"Bash step {match.automation_step.id}: {match.automation_step.description}"
    return _category_label(category)


def _script_steps(match: StepMatch) -> list[BuildStep]:
    steps = match.automation_steps or (
        [match.automation_step] if match.automation_step is not None else []
    )
    unique: dict[str, BuildStep] = {step.id: step for step in steps}
    return list(unique.values())


def _source_reference(instruction: BuildInstruction) -> ReportSource | None:
    source = instruction.source_metadata
    if source is None:
        return None
    return ReportSource(
        name=source.filename,
        repository=source.repository,
        ref=source.ref,
        path=source.path,
        url=source.github_url,
    )


def _step_evidence(step: BuildStep) -> StepEvidence:
    context: SourceContext = step.context
    excerpt = (
        context.source_excerpt
        if context.source_type == "wiki"
        else context.surrounding_text
    )
    return StepEvidence(
        id=step.id,
        description=step.description,
        action=step.action,
        location=_location(context, step.id),
        component=step.component,
        commands=step.commands,
        packages=step.packages,
        paths=step.paths,
        urls=step.urls,
        repository=step.repository,
        ref=step.ref,
        commit=step.commit,
        version=step.version,
        docker_image=step.docker_image,
        dockerfile=step.dockerfile,
        build_args=step.build_args,
        environment=step.environment,
        dependencies=step.depends_on,
        conditions=context.conditions,
        source_excerpt=excerpt,
    )


def _location(context: SourceContext, step_id: str) -> str:
    parts = [context.source_name or context.source_type]
    if context.heading_path:
        parts.append(" > ".join(context.heading_path))
    elif context.subsection:
        parts.append(f"function {context.subsection}")

    if context.line_start is not None and context.line_end is not None:
        location = (
            f"line {context.line_start}"
            if context.line_start == context.line_end
            else f"lines {context.line_start}-{context.line_end}"
        )
        parts.append(location)
    elif context.location:
        parts.append(context.location)
    else:
        parts.append(step_id)
    return " | ".join(parts)


def _render_markdown(report: ComparisonReport) -> str:
    lines = [
        "# Build comparison discrepancy report",
        "",
        f"**Package:** {report.package}",
    ]
    if report.version:
        lines.append(f"**Version:** {report.version}")
    lines.extend(
        [
            f"**Status:** {_status_label(report.status)}",
            "",
            "## Summary",
            "",
            f"{report.overview}",
            "",
            f"- Wiki steps compared: {report.counts.wiki_steps_compared}",
            f"- Script steps checked: {report.counts.script_steps_checked}",
            f"- Missing Wiki steps: {report.counts.missing_steps}",
            f"- Wiki/script conflicts: {report.counts.conflicting_steps}",
            f"- Undocumented script steps: {report.counts.undocumented_script_steps}",
            f"- Findings: {report.counts.findings}",
        ]
    )
    if report.counts.by_category:
        category_counts = ", ".join(
            f"{_category_label(DiscrepancyCategory(category))}: {count}"
            for category, count in sorted(report.counts.by_category.items())
        )
        lines.extend([f"- By category: {category_counts}"])
    if report.counts.out_of_scope_steps:
        lines.append(
            f"- Bootstrap steps outside comparison scope: {report.counts.out_of_scope_steps}"
        )
    if report.wiki_source or report.script_source:
        lines.extend(["", "## Sources", ""])
        if report.wiki_source:
            lines.append(_markdown_source("Build Instructions", report.wiki_source))
        if report.script_source:
            lines.append(_markdown_source("Automation script", report.script_source))

    if not report.discrepancies:
        lines.extend(["", "## Discrepancies", "", "None found."])
        return "\n".join(lines)

    lines.extend(["", "## Discrepancies"])
    for item in report.discrepancies:
        lines.extend(
            [
                "",
                f"### {item.id} — {_category_label(item.category)}",
                "",
                f"**Finding:** {_markdown_inline(item.title)}",
                f"**Type:** `{item.status.value}` / `{item.category.value}`",
                f"**Why flagged:** {_markdown_inline(item.explanation)}",
            ]
        )
        if item.wiki:
            lines.extend(["", "**Build Instructions evidence**", ""])
            lines.extend(_render_evidence_markdown(item.wiki))
        else:
            lines.extend(["", "**Build Instructions evidence:** No Wiki step was linked."])

        if item.script:
            lines.extend(["", "**Automation script evidence**", ""])
            for evidence in item.script:
                lines.extend(_render_evidence_markdown(evidence))
                lines.append("")
        elif item.status == MatchStatus.MISSING:
            lines.extend(["", "**Automation script evidence:** No Bash step was linked."])

        lines.extend(
            [
                "",
                f"**Suggested correction:** {_markdown_inline(item.recommendation)}",
            ]
        )

    return "\n".join(lines)


def _render_evidence_markdown(evidence: StepEvidence) -> list[str]:
    lines = [
        f"- Step: `{evidence.id}` — {_markdown_inline(evidence.description)}",
        f"- Location: {_markdown_inline(evidence.location)}",
        f"- Action: `{evidence.action.value}`",
    ]
    for label, values in (
        ("Commands", evidence.commands),
        ("Packages", evidence.packages),
        ("Paths", evidence.paths),
        ("URLs", evidence.urls),
        ("Dependencies", evidence.dependencies),
        ("Conditions", evidence.conditions),
    ):
        if values:
            lines.append(f"- {label}: {_markdown_inline('; '.join(values))}")
    for label, value in (
        ("Repository", evidence.repository),
        ("Ref", evidence.ref),
        ("Commit", evidence.commit),
        ("Version", evidence.version),
        ("Docker image", evidence.docker_image),
        ("Dockerfile", evidence.dockerfile),
    ):
        if value:
            lines.append(f"- {label}: `{_markdown_inline(value)}`")
    if evidence.build_args:
        values = ", ".join(
            f"{key}={value}" for key, value in evidence.build_args.items()
        )
        lines.append(f"- Build arguments: `{_markdown_inline(values)}`")
    if evidence.environment:
        values = ", ".join(
            f"{key}={value}" for key, value in evidence.environment.items()
        )
        lines.append(f"- Environment: `{_markdown_inline(values)}`")
    if evidence.source_excerpt:
        excerpt = _truncate(evidence.source_excerpt)
        lines.extend(["", "  ```text", *[f"  {line}" for line in excerpt.splitlines()], "  ```"])
    return lines


def _markdown_source(label: str, source: ReportSource) -> str:
    return (
        f"- {label}: [{source.name}]({source.url}) "
        f"(`{source.repository}` @ `{source.ref}`, `{source.path}`)"
    )


def _render_html(report: ComparisonReport) -> str:
    esc = html.escape
    version_line = (
        f"<p><strong>Version:</strong> {esc(report.version)}</p>"
        if report.version
        else ""
    )
    summary_rows = [
        ("Wiki steps compared", report.counts.wiki_steps_compared),
        ("Script steps checked", report.counts.script_steps_checked),
        ("Missing Wiki steps", report.counts.missing_steps),
        ("Wiki/script conflicts", report.counts.conflicting_steps),
        ("Undocumented script steps", report.counts.undocumented_script_steps),
        ("Findings", report.counts.findings),
    ]
    if report.counts.out_of_scope_steps:
        summary_rows.append(
            ("Bootstrap steps outside comparison scope", report.counts.out_of_scope_steps)
        )
    summary_html = "\n".join(
        f"<tr><th>{esc(label)}</th><td>{value}</td></tr>"
        for label, value in summary_rows
    )
    if report.counts.by_category:
        category_counts = ", ".join(
            f"{_category_label(DiscrepancyCategory(category))}: {count}"
            for category, count in sorted(report.counts.by_category.items())
        )
        summary_html += (
            f"<tr><th>Findings by category</th><td>{esc(category_counts)}</td></tr>"
        )
    sources_html = "".join(
        _source_html(label, source)
        for label, source in (
            ("Build Instructions", report.wiki_source),
            ("Automation script", report.script_source),
        )
        if source is not None
    )
    findings_html = "".join(_discrepancy_html(item) for item in report.discrepancies)
    if not findings_html:
        findings_html = "<p>No discrepancies found.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Build comparison — {esc(report.package)}</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #17202a; }}
    h1, h2, h3 {{ line-height: 1.25; }}
    .status {{ padding: .7rem 1rem; background: #fff2cc; border-left: 4px solid #d89b00; }}
    table {{ border-collapse: collapse; }} th, td {{ border: 1px solid #ccd1d1; padding: .4rem .7rem; text-align: left; }}
    article {{ border-top: 1px solid #ccd1d1; padding: 1rem 0; }}
    pre {{ overflow-x: auto; white-space: pre-wrap; background: #f4f6f7; padding: .8rem; }}
    code {{ overflow-wrap: anywhere; }}
    .recommendation {{ background: #eafaf1; padding: .7rem 1rem; }}
  </style>
</head>
<body>
  <h1>Build comparison discrepancy report</h1>
  <p><strong>Package:</strong> {esc(report.package)}</p>
  {version_line}
  <p class="status"><strong>Status:</strong> {esc(_status_label(report.status))} — {esc(report.overview)}</p>
  <h2>Summary</h2>
  <table><tbody>{summary_html}</tbody></table>
  {f'<h2>Sources</h2>{sources_html}' if sources_html else ''}
  <h2>Discrepancies</h2>
  {findings_html}
</body>
</html>
"""


def _discrepancy_html(item: Discrepancy) -> str:
    esc = html.escape
    evidence_html = ""
    if item.wiki:
        evidence_html += _evidence_html("Build Instructions evidence", item.wiki)
    if item.script:
        evidence_html += "<h4>Automation script evidence</h4>"
        evidence_html += "".join(_evidence_html("", evidence) for evidence in item.script)
    elif item.status == MatchStatus.MISSING:
        evidence_html += "<p>No Bash step was linked.</p>"
    return f"""<article>
  <h3>{esc(item.id)} — {esc(_category_label(item.category))}</h3>
  <p><strong>Finding:</strong> {esc(item.title)}</p>
  <p><strong>Type:</strong> {esc(item.status.value)} / {esc(item.category.value)}</p>
  <p><strong>Why flagged:</strong> {esc(item.explanation)}</p>
  {evidence_html}
  <p class="recommendation"><strong>Suggested correction:</strong> {esc(item.recommendation)}</p>
</article>"""


def _evidence_html(heading: str, evidence: StepEvidence) -> str:
    esc = html.escape
    fields: list[tuple[str, str]] = [
        ("Step", f"{evidence.id}: {evidence.description}"),
        ("Location", evidence.location),
        ("Action", evidence.action.value),
    ]
    for label, values in (
        ("Commands", evidence.commands),
        ("Packages", evidence.packages),
        ("Paths", evidence.paths),
        ("URLs", evidence.urls),
        ("Dependencies", evidence.dependencies),
        ("Conditions", evidence.conditions),
    ):
        if values:
            fields.append((label, "; ".join(values)))
    for label, value in (
        ("Repository", evidence.repository),
        ("Ref", evidence.ref),
        ("Commit", evidence.commit),
        ("Version", evidence.version),
        ("Docker image", evidence.docker_image),
        ("Dockerfile", evidence.dockerfile),
    ):
        if value:
            fields.append((label, value))
    if evidence.build_args:
        fields.append(
            ("Build arguments", ", ".join(f"{key}={value}" for key, value in evidence.build_args.items()))
        )
    if evidence.environment:
        fields.append(
            ("Environment", ", ".join(f"{key}={value}" for key, value in evidence.environment.items()))
        )
    heading_html = f"<h4>{esc(heading)}</h4>" if heading else ""
    fields_html = "".join(
        f"<li><strong>{esc(label)}:</strong> {esc(value)}</li>"
        for label, value in fields
    )
    excerpt_html = (
        f"<pre>{esc(_truncate(evidence.source_excerpt))}</pre>"
        if evidence.source_excerpt
        else ""
    )
    return f"{heading_html}<ul>{fields_html}</ul>{excerpt_html}"


def _source_html(label: str, source: ReportSource) -> str:
    esc = html.escape
    return (
        f'<p>{esc(label)}: <a href="{esc(source.url, quote=True)}">'
        f"{esc(source.name)}</a> ({esc(source.repository)} @ {esc(source.ref)}, "
        f"{esc(source.path)})</p>"
    )


def _category_label(category: DiscrepancyCategory) -> str:
    labels = {
        DiscrepancyCategory.MISSING_STEP: "Missing Wiki step",
        DiscrepancyCategory.UNDOCUMENTED_STEP: "Undocumented script step",
        DiscrepancyCategory.PARAMETER_VALUE_MISMATCH: "Parameter/value mismatch",
        DiscrepancyCategory.VERSION_INCONSISTENCY: "Version inconsistency",
        DiscrepancyCategory.DEPENDENCY_INCONSISTENCY: "Dependency inconsistency",
        DiscrepancyCategory.COMMAND_MISMATCH: "Command mismatch",
        DiscrepancyCategory.CONFIGURATION_MISMATCH: "Configuration mismatch",
        DiscrepancyCategory.OTHER_DEVIATION: "Other discrepancy",
    }
    return labels[category]


def _status_label(status: str) -> str:
    return status.replace("_", " ").upper()


def _markdown_inline(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _truncate(value: str, limit: int = 1200) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "… [excerpt truncated]"
