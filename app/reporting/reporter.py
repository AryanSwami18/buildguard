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
            MatchStatus.UNUSED_CONFIGURATION,
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
    semantic_match_count = sum(
        match.status == MatchStatus.MATCHED for match in result.matches
    )
    unused_configuration_count = sum(
        match.status == MatchStatus.UNUSED_CONFIGURATION for match in result.matches
    )
    implementation_detail_count = sum(
        match.status == MatchStatus.IMPLEMENTATION_DETAIL for match in result.matches
    )
    out_of_scope_count = sum(
        match.status == MatchStatus.OUT_OF_SCOPE for match in result.matches
    )
    status = "review_required" if discrepancies else "no_mismatches_found"
    overview = _build_overview(
        findings=len(discrepancies),
        wiki_actions=len(wiki.steps),
        script_operations=len(script.steps),
        semantic_matches=semantic_match_count,
    )

    return ComparisonReport(
        package=package,
        version=version,
        status=status,
        overview=overview,
        counts=ComparisonCounts(
            wiki_steps_compared=len(wiki.steps),
            script_steps_checked=len(script.steps),
            wiki_build_actions_analyzed=len(wiki.steps),
            script_build_actions_analyzed=(
                len(script.steps)
                - implementation_detail_count
                - unused_configuration_count
            ),
            missing_steps=missing_count,
            conflicting_steps=conflict_count,
            undocumented_script_steps=extra_count,
            out_of_scope_steps=out_of_scope_count,
            findings=len(discrepancies),
            by_category=by_category,
            semantic_matches=semantic_match_count,
            true_documentation_gaps=missing_count + extra_count,
            potential_conflicts=conflict_count,
            unused_configuration=unused_configuration_count,
            implementation_details_ignored=implementation_detail_count,
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
    if match.status == MatchStatus.UNUSED_CONFIGURATION:
        return DiscrepancyCategory.UNUSED_CONFIGURATION
    return match.category or DiscrepancyCategory.OTHER_DEVIATION


def _build_overview(
    *,
    findings: int,
    wiki_actions: int,
    script_operations: int,
    semantic_matches: int,
) -> str:
    """Give the report a plain-language conclusion before its raw details."""
    if findings:
        issue_word = "item needs" if findings == 1 else "items need"
        return (
            f"{findings} actionable {issue_word} review. BuildGuard compared {wiki_actions} documented "
            f"Wiki build actions with {script_operations} script operations, "
            f"grouping related commands into semantic actions. "
            f"{semantic_matches} Wiki action{'' if semantic_matches == 1 else 's'} "
            "matched the script; implementation details were not treated as findings."
        )
    return (
        f"All {wiki_actions} documented Wiki build actions matched the script or "
        "were represented by harmless implementation details. No actionable "
        "differences were found."
    )


def _finding_label(item: Discrepancy) -> str:
    """Translate an internal status into language a report reader can act on."""
    if item.status == MatchStatus.MISSING:
        return "Wiki instruction not found in the script"
    if item.status == MatchStatus.EXTRA:
        return "Script action is not represented in the Wiki"
    if item.status == MatchStatus.CONFLICT:
        return "Wiki and script prescribe different behavior"
    if item.status == MatchStatus.UNUSED_CONFIGURATION:
        return "Configuration is defined but not used"
    return _category_label(item.category)


def _finding_priority(item: Discrepancy) -> str:
    if item.status == MatchStatus.CONFLICT:
        return "High attention"
    if item.status in {MatchStatus.MISSING, MatchStatus.EXTRA}:
        return "Needs review"
    if item.status == MatchStatus.UNUSED_CONFIGURATION:
        return "Cleanup candidate"
    return "Review"


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
    if status == MatchStatus.UNUSED_CONFIGURATION:
        return (
            "Remove the unused configuration or wire it into the build operation "
            "that it was intended to control."
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
    if match.status == MatchStatus.UNUSED_CONFIGURATION and match.automation_step:
        return (
            f"Unused configuration in Bash step {match.automation_step.id}: "
            f"{match.automation_step.description}"
        )
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
        line_start=context.line_start,
        line_end=context.line_end,
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
        "# BuildGuard comparison report",
        "",
        f"**Package:** {report.package}",
    ]
    if report.version:
        lines.append(f"**Version:** {report.version}")
    lines.extend(
        [
            f"**Status:** {_status_label(report.status)}",
            "",
            "## At a glance",
            "",
            f"{report.overview}",
            "",
            "| Result | Count | Meaning |",
            "| --- | ---: | --- |",
            f"| **Actionable findings** | **{report.counts.findings}** | Items that need a person to review |",
            f"| Semantic matches | {report.counts.semantic_matches} | Wiki actions implemented by one or more related script commands |",
            f"| Unused configuration | {report.counts.unused_configuration} | Values defined in the script but not referenced |",
            f"| Implementation details ignored | {report.counts.implementation_details_ignored} | Internal setup, cleanup, or helper commands not shown as problems |",
            "",
            "### How to read this",
            "",
            "- A **semantic match** means the Wiki describes one build action and the script may implement it with several commands, variables, or functions.",
            "- **Implementation details** are shown as ignored because they help automation but do not normally need to be documented as user-facing build steps.",
            "- **Actionable findings** are the only items listed below. They represent a possible missing instruction, disagreement, undocumented behavior, or unused configuration.",
            "",
            "### Comparison scope",
            "",
            f"- Wiki build actions reviewed: {report.counts.wiki_build_actions_analyzed or report.counts.wiki_steps_compared}",
            f"- Script operations reviewed: {report.counts.script_steps_checked}",
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
        lines.extend(["", "## What needs attention", "", "None found. The Wiki and script are aligned for the actions that were compared."])
        return "\n".join(lines)

    lines.extend(["", "## What needs attention"])
    for item in report.discrepancies:
        lines.extend(
            [
                "",
                f"### {item.id} — {_finding_label(item)}",
                "",
                f"**Finding:** {_markdown_inline(item.title)}",
                f"**Priority:** {_finding_priority(item)}",
                f"**Why this needs attention:** {_markdown_inline(item.explanation)}",
            ]
        )
        if item.wiki:
            lines.extend(["", "**What the Wiki says**", ""])
            lines.extend(_render_evidence_markdown(item.wiki))
        else:
            lines.extend(["", "**What the Wiki says:** No matching Wiki action was linked."])

        if item.script:
            lines.extend(["", "**What the script does**", ""])
            for evidence in item.script:
                lines.extend(_render_evidence_markdown(evidence))
                lines.append("")
        elif item.status == MatchStatus.MISSING:
            lines.extend(["", "**What the script does:** No matching Bash action was found."])

        lines.extend(
            [
                "",
                f"**Recommended next step:** {_markdown_inline(item.recommendation)}",
                f"**Technical classification:** `{item.status.value}` / `{item.category.value}`",
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
    status_class = "warning" if report.counts.findings else "success"
    summary_cards = "".join(
        _summary_card(label, value, description, tone)
        for label, value, description, tone in (
            (
                "Actionable findings",
                report.counts.findings,
                "Items that need a person to review",
                "danger" if report.counts.findings else "success",
            ),
            (
                "Semantic matches",
                report.counts.semantic_matches,
                "Wiki actions implemented by related script commands",
                "success",
            ),
            (
                "Unused configuration",
                report.counts.unused_configuration,
                "Defined values that are not referenced",
                "warning" if report.counts.unused_configuration else "muted",
            ),
            (
                "Implementation details ignored",
                report.counts.implementation_details_ignored,
                "Internal helper and cleanup commands",
                "muted",
            ),
        )
    )
    summary_rows = [
        ("Wiki build actions reviewed", report.counts.wiki_build_actions_analyzed or report.counts.wiki_steps_compared),
        ("Script operations reviewed", report.counts.script_steps_checked),
        ("Missing Wiki instructions", report.counts.missing_steps),
        ("Script actions not in Wiki", report.counts.undocumented_script_steps),
        ("Wiki/script conflicts", report.counts.conflicting_steps),
        ("Unused configuration", report.counts.unused_configuration),
        ("Implementation details ignored", report.counts.implementation_details_ignored),
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
        findings_html = "<p class=\"empty-state\">No actionable differences found. The Wiki and script are aligned for the actions that were compared.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BuildGuard report — {esc(report.package)}</title>
  <style>
    :root {{ color-scheme: light; --ink: #17202a; --muted: #5c6770; --line: #dfe5e8; --blue: #155eef; --blue-soft: #eef4ff; --green: #137333; --green-soft: #edf8f0; --amber: #9a6700; --amber-soft: #fff8e1; --red: #b42318; --red-soft: #fff1f0; }}
    body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 1080px; margin: 2rem auto; padding: 0 1rem; color: var(--ink); background: #fbfcfd; }}
    h1, h2, h3 {{ line-height: 1.25; }}
    .status {{ padding: 1rem 1.1rem; border-radius: .55rem; border-left: 5px solid; }}
    .status.warning {{ background: var(--amber-soft); border-color: #d89b00; }}
    .status.success {{ background: var(--green-soft); border-color: #2e8b57; }}
    .lead {{ font-size: 1.08rem; margin: .3rem 0 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: 1rem 0 1.4rem; }}
    .card {{ border: 1px solid var(--line); border-top: 4px solid; border-radius: .55rem; padding: .85rem 1rem; background: #fff; }}
    .card.danger {{ border-top-color: var(--red); }} .card.success {{ border-top-color: var(--green); }} .card.warning {{ border-top-color: var(--amber); }} .card.muted {{ border-top-color: #87909a; }}
    .card-label {{ color: var(--muted); font-size: .9rem; }} .card-value {{ display: block; font-size: 1.8rem; font-weight: 700; line-height: 1.15; margin: .2rem 0; }} .card-description {{ color: var(--muted); font-size: .85rem; }}
    .guide {{ background: var(--blue-soft); border: 1px solid #c9d9ff; border-radius: .55rem; padding: .8rem 1rem; }}
    .guide ul {{ margin: .4rem 0 0; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }} th, td {{ border: 1px solid var(--line); padding: .55rem .7rem; text-align: left; }} th {{ background: #f4f6f8; }}
    article {{ border: 1px solid var(--line); border-radius: .6rem; padding: 1rem 1.1rem; margin: 1rem 0; background: #fff; }}
    article h3 {{ margin-top: 0; }}
    .finding-kind {{ color: var(--red); font-weight: 650; }}
    .priority {{ display: inline-block; padding: .2rem .55rem; border-radius: 999px; background: var(--amber-soft); color: var(--amber); font-size: .85rem; font-weight: 650; }}
    .evidence {{ border-left: 3px solid var(--line); padding-left: .9rem; margin: .8rem 0; }}
    pre {{ overflow-x: auto; white-space: pre-wrap; background: #f4f6f7; padding: .8rem; border-radius: .35rem; }}
    code {{ overflow-wrap: anywhere; }}
    .recommendation {{ background: var(--green-soft); padding: .7rem 1rem; border-radius: .35rem; }}
    .technical {{ color: var(--muted); font-size: .85rem; }}
    .empty-state {{ background: var(--green-soft); padding: .9rem 1rem; border-radius: .4rem; }}
    @media (max-width: 760px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 460px) {{ .cards {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>BuildGuard comparison report</h1>
  <p><strong>Package:</strong> {esc(report.package)}</p>
  {version_line}
  <div class="status {status_class}"><strong>{esc(_status_label(report.status))}</strong><p class="lead">{esc(report.overview)}</p></div>
  <h2>At a glance</h2>
  <div class="cards">{summary_cards}</div>
  <div class="guide">
    <strong>How to read this report</strong>
    <ul>
      <li>A semantic match may represent one Wiki action implemented by several script commands, variables, or functions.</li>
      <li>Implementation details such as setup, cleanup, and helper commands are intentionally not presented as problems.</li>
      <li>Only the items under <strong>What needs attention</strong> are actionable findings.</li>
    </ul>
  </div>
  <h2>Comparison details</h2>
  <table><tbody>{summary_html}</tbody></table>
  {f'<h2>Sources</h2>{sources_html}' if sources_html else ''}
  <h2>What needs attention</h2>
  {findings_html}
</body>
</html>
"""


def _discrepancy_html(item: Discrepancy) -> str:
    esc = html.escape
    evidence_html = ""
    if item.wiki:
        evidence_html += _evidence_html("What the Wiki says", item.wiki)
    if item.script:
        evidence_html += "<h4>What the script does</h4>"
        evidence_html += "".join(_evidence_html("", evidence) for evidence in item.script)
    elif item.status == MatchStatus.MISSING:
        evidence_html += "<p>No matching Bash action was found.</p>"
    return f"""<article>
  <h3>{esc(item.id)} — {esc(_finding_label(item))}</h3>
  <p><strong>Finding:</strong> {esc(item.title)}</p>
  <p><span class="priority">{esc(_finding_priority(item))}</span></p>
  <p><strong>Why this needs attention:</strong> {esc(item.explanation)}</p>
  {evidence_html}
  <p class="recommendation"><strong>Recommended next step:</strong> {esc(item.recommendation)}</p>
  <p class="technical">Technical classification: {esc(item.status.value)} / {esc(item.category.value)}</p>
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
    return f'<div class="evidence">{heading_html}<ul>{fields_html}</ul>{excerpt_html}</div>'


def _summary_card(
    label: str,
    value: int,
    description: str,
    tone: str,
) -> str:
    esc = html.escape
    return (
        f'<div class="card {esc(tone)}">'
        f'<span class="card-label">{esc(label)}</span>'
        f'<span class="card-value">{value}</span>'
        f'<span class="card-description">{esc(description)}</span>'
        "</div>"
    )


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
        DiscrepancyCategory.UNUSED_CONFIGURATION: "Unused / dead configuration",
        DiscrepancyCategory.OTHER_DEVIATION: "Other discrepancy",
    }
    return labels[category]


def _status_label(status: str) -> str:
    labels = {
        "review_required": "Review needed",
        "no_mismatches_found": "No actionable issues found",
    }
    return labels.get(status, status.replace("_", " ").capitalize())


def _markdown_inline(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _truncate(value: str, limit: int = 1200) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "… [excerpt truncated]"
