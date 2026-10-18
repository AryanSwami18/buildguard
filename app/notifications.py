"""Notifications for webhook-triggered BuildGuard reviews."""

from __future__ import annotations

from email.message import EmailMessage
from html import escape
import re
import smtplib

from app.config.settings import Settings
from app.models.report import ComparisonReport
from app.time_utils import local_timestamp


def send_report_notification(
    settings: Settings,
    report: ComparisonReport,
    *,
    pr_url: str | None = None,
    action_error: str | None = None,
) -> None:
    """Send one completed comparison report through configured SMTP."""
    recipients = [
        address.strip()
        for address in (settings.notification_email_to or "").split(",")
        if address.strip()
    ]
    if not recipients:
        raise ValueError("BUILDGUARD_NOTIFICATION_EMAIL_TO is not configured")
    if not settings.smtp_host:
        raise ValueError("BUILDGUARD_SMTP_HOST is not configured")

    sender = settings.smtp_from or settings.smtp_username
    if not sender:
        raise ValueError(
            "BUILDGUARD_SMTP_FROM or BUILDGUARD_SMTP_USERNAME is required"
        )

    message = _build_report_notification_message(
        report,
        sender=sender,
        recipients=recipients,
        pr_url=pr_url,
        action_error=action_error,
    )

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
        if settings.smtp_use_tls:
            smtp.starttls()
        if settings.smtp_username:
            smtp.login(settings.smtp_username, settings.smtp_password or "")
        smtp.send_message(message)


def _build_report_notification_message(
    report: ComparisonReport,
    *,
    sender: str,
    recipients: list[str],
    pr_url: str | None = None,
    action_error: str | None = None,
) -> EmailMessage:
    """Build the email without opening an SMTP connection."""
    findings = report.discrepancies
    version = report.version or "unknown version"
    sent_at = local_timestamp()
    subject_status = (
        "No issues found" if not findings else f"{len(findings)} item(s) need review"
    )

    message = EmailMessage()
    message["Subject"] = (
        f"BuildGuard · {report.package} {version} · {subject_status} · {sent_at}"
    )
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(
        _plain_text_body(report, pr_url=pr_url, action_error=action_error)
    )
    message.add_alternative(
        _html_body(report, pr_url=pr_url, action_error=action_error),
        subtype="html",
    )
    return message


def _plain_text_body(
    report: ComparisonReport,
    *,
    pr_url: str | None,
    action_error: str | None,
) -> str:
    """Create a readable fallback for text-only email clients."""
    counts = report.counts
    lines = [
        f"BuildGuard review: {report.package} {report.version or 'unknown version'}",
        f"Sent: {local_timestamp()}",
        "",
        "RESULT: No actionable findings were found."
        if not report.discrepancies
        else f"RESULT: {len(report.discrepancies)} actionable finding(s) need review.",
        report.overview,
        "",
        "Summary:",
        f"- Wiki build actions reviewed: {counts.wiki_build_actions_analyzed or counts.wiki_steps_compared}",
        f"- Script build actions reviewed: {counts.script_build_actions_analyzed or counts.script_steps_checked}",
        f"- Semantic matches: {counts.semantic_matches}",
        f"- Actionable findings: {counts.findings}",
    ]
    if report.discrepancies:
        lines.extend(["", "What needs attention:"])
        for finding in report.discrepancies:
            lines.extend(
                [
                    f"\n{finding.id} · {_category_label(finding.category)}",
                    finding.title,
                    f"Why: {finding.explanation}",
                    *_plain_evidence_lines("Wiki", finding.wiki),
                    *[
                        line
                        for evidence in finding.script
                        for line in _plain_evidence_lines("Script", evidence)
                    ],
                    f"Recommended next step: {finding.recommendation}",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "The Wiki and automation script are aligned for this revision.",
                "No pull request was created because there is nothing actionable to fix.",
            ]
        )
    if pr_url:
        lines.extend(["", f"Draft pull request: {pr_url}"])
    if action_error:
        lines.extend(["", f"BuildGuard follow-up warning: {action_error}"])
    lines.extend(["", "Sources:"])
    if report.wiki_source:
        lines.append(f"- Wiki: {report.wiki_source.url}")
    if report.script_source:
        lines.append(f"- Script: {report.script_source.url}")
    return "\n".join(lines)


def _html_body(
    report: ComparisonReport,
    *,
    pr_url: str | None,
    action_error: str | None,
) -> str:
    """Create a restrained, email-client-friendly engineering report."""
    counts = report.counts
    findings = report.discrepancies
    clean = not findings
    accent = "#16794a" if clean else "#b42318"
    result_label = "No actionable findings" if clean else f"{len(findings)} finding(s) need review"
    version = report.version or "unknown version"

    summary_rows = [
        (
            "Actionable findings",
            counts.findings,
            accent if counts.findings else "#16794a",
        ),
        ("Semantic matches", counts.semantic_matches, "#1f2937"),
        (
            "Wiki build actions reviewed",
            counts.wiki_build_actions_analyzed or counts.wiki_steps_compared,
            "#1f2937",
        ),
        (
            "Script build actions reviewed",
            counts.script_build_actions_analyzed or counts.script_steps_checked,
            "#1f2937",
        ),
    ]
    summary_html = "".join(
        f"""
        <tr>
          <td style=\"padding:8px 0;border-bottom:1px solid #e5e7eb;font-size:13px;color:#4b5563;\">{escape(label)}</td>
          <td align=\"right\" style=\"padding:8px 0;border-bottom:1px solid #e5e7eb;font-size:13px;font-weight:700;color:{color};\">{value}</td>
        </tr>
        """
        for label, value, color in summary_rows
    )

    if findings:
        finding_html = "".join(_finding_html(finding) for finding in findings)
        attention_section = f"""
        <h2 style=\"margin:28px 0 10px;font-size:16px;color:#111827;\">Findings</h2>
        {finding_html}
        """
    else:
        attention_section = f"""
        <div style=\"margin-top:24px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:13px;line-height:1.6;color:#374151;\">
          <strong style=\"color:{accent};\">{escape(result_label)}.</strong>
          The documented Wiki build actions are represented by the automation script.
        </div>
        """

    follow_up = ""
    if pr_url:
        safe_pr_url = escape(pr_url, quote=True)
        follow_up += f"""
        <div style=\"margin-top:20px;padding-top:14px;border-top:1px solid #e5e7eb;font-size:13px;color:#374151;\">
          <strong>Pull request:</strong>
          <a href=\"{safe_pr_url}\" style=\"color:#155eef;\">Open the draft pull request</a>
        </div>
        """
    if action_error:
        follow_up += f"""
        <div style=\"margin-top:12px;padding-top:12px;border-top:1px solid #e5e7eb;font-size:13px;color:#92400e;\">
          <strong>Follow-up warning:</strong> {escape(action_error)}
        </div>
        """

    sources = []
    if report.wiki_source:
        source = report.wiki_source
        sources.append(
            f'<a href="{escape(source.url, quote=True)}" style="color:#2563eb;">Wiki · {escape(source.name)}</a>'
        )
    if report.script_source:
        source = report.script_source
        sources.append(
            f'<a href="{escape(source.url, quote=True)}" style="color:#2563eb;">Script · {escape(source.name)}</a>'
        )
    source_html = "<br>".join(sources) if sources else "Source links unavailable"

    return f"""<!doctype html>
<html>
  <body style=\"margin:0;padding:0;background:#f5f6f8;font-family:Arial,Helvetica,sans-serif;color:#111827;\">
    <table role=\"presentation\" align=\"center\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" style=\"width:100%;background:#f5f6f8;\">
      <tr><td align=\"center\" style=\"padding:24px 12px;\">
      <table role=\"presentation\" align=\"center\" cellpadding=\"0\" cellspacing=\"0\" width=\"660\" style=\"width:100%;max-width:660px;margin:0 auto;background:#ffffff;border:1px solid #d9dee7;text-align:left;\">
        <tr>
          <td style=\"border-top:4px solid #1f2937;padding:22px 28px 18px;\">
            <div style=\"font-size:12px;font-weight:700;letter-spacing:.04em;color:#6b7280;\">BUILDGUARD REPORT</div>
            <div style=\"margin-top:8px;font-size:22px;line-height:1.25;font-weight:700;color:#111827;\">{escape(report.package)}</div>
            <div style=\"margin-top:4px;font-size:13px;color:#6b7280;\">Version {escape(version)}</div>
          </td>
        </tr>
        <tr>
          <td style=\"padding:22px 28px 28px;\">
            <div style=\"font-size:15px;font-weight:700;color:{accent};\">{escape(result_label)}</div>
            <div style=\"margin-top:9px;font-size:13px;line-height:1.6;color:#374151;\">{escape(report.overview)}</div>
            <h2 style=\"margin:24px 0 4px;font-size:16px;color:#111827;\">Summary</h2>
            <table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" style=\"width:100%;border-top:1px solid #e5e7eb;\">{summary_html}</table>
            {attention_section}
            {follow_up}
            <div style=\"margin-top:26px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:12px;line-height:1.8;color:#6b7280;\">
              <strong style=\"color:#374151;\">Sources</strong><br>{source_html}
            </div>
          </td>
        </tr>
        <tr>
          <td style=\"padding:14px 28px;border-top:1px solid #e5e7eb;background:#fafafa;font-size:11px;color:#9ca3af;\">
            Automated BuildGuard notification
          </td>
        </tr>
      </table>
      </td></tr>
    </table>
  </body>
</html>"""


def _finding_html(finding) -> str:
    """Render one actionable finding in the HTML email."""
    category = _category_label(finding.category)
    evidence_html = ""
    sections = [("Wiki", finding.wiki)]
    sections.extend(("Script", evidence) for evidence in finding.script)
    for source_label, evidence in sections:
        if evidence is None:
            continue
        line_label = _evidence_line_label(evidence)
        commands = evidence.commands
        if not line_label and not commands:
            continue
        line_html = f"<span style=\"color:#6b7280;\">{escape(line_label)}</span><br>" if line_label else ""
        command_html = (
            f'<pre style="margin:5px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 monospace;color:#1f2937;">{escape(chr(10).join(commands))}</pre>'
            if commands
            else ""
        )
        evidence_html += (
            f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid #f0f2f5;font-size:11px;line-height:1.5;color:#4b5563;">'
            f'<strong>{source_label}</strong><br>{line_html}{command_html}</div>'
        )
    return f"""
    <div style=\"margin-top:16px;padding-top:14px;border-top:2px solid #e5e7eb;\">
      <div style=\"font-size:11px;font-weight:700;letter-spacing:.04em;color:#6b7280;\">{escape(finding.id)} · {escape(category)}</div>
      <div style=\"margin-top:6px;font-size:15px;font-weight:700;color:#111827;\">{escape(finding.title)}</div>
      <div style=\"margin-top:7px;font-size:13px;line-height:1.55;color:#374151;\">{escape(finding.explanation)}</div>
      {evidence_html}
      <div style=\"margin-top:12px;font-size:13px;line-height:1.55;color:#374151;\"><strong>Next step:</strong> {escape(finding.recommendation)}</div>
    </div>
    """


def _evidence_line_label(evidence) -> str:
    if evidence.line_start is not None:
        end = evidence.line_end or evidence.line_start
        return (
            f"Line {end}"
            if end == evidence.line_start
            else f"Lines {evidence.line_start}–{end}"
        )
    match = re.search(r"\blines?\s+(\d+)(?:\s*[-–]\s*(\d+))?", evidence.location, re.I)
    if not match:
        return ""
    start, end = match.groups()
    return f"Line {start}" if not end or end == start else f"Lines {start}–{end}"


def _plain_evidence_lines(label: str, evidence) -> list[str]:
    if evidence is None:
        return []
    line_label = _evidence_line_label(evidence)
    commands = evidence.commands
    result = [f"{label} · {line_label}" if line_label else label]
    result.extend(f"  {command}" for command in commands)
    return result


def _category_label(category) -> str:
    value = getattr(category, "value", category)
    return str(value).replace("_", " ").title()
