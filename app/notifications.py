"""Notifications for webhook-triggered BuildGuard reviews."""

from __future__ import annotations

from email.message import EmailMessage
from html import escape
import smtplib

from app.config.settings import Settings
from app.models.report import ComparisonReport


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
    subject_status = (
        "No issues found" if not findings else f"{len(findings)} item(s) need review"
    )

    message = EmailMessage()
    message["Subject"] = f"BuildGuard · {report.package} {version} · {subject_status}"
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
        f"- Implementation details ignored: {counts.implementation_details_ignored}",
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
    """Create a self-contained, email-client-friendly HTML summary."""
    counts = report.counts
    findings = report.discrepancies
    clean = not findings
    accent = "#16a34a" if clean else "#dc2626"
    accent_soft = "#f0fdf4" if clean else "#fef2f2"
    result_label = (
        "Review complete · no issues found"
        if clean
        else f"{len(findings)} item(s) need review"
    )
    version = report.version or "unknown version"

    stats = [
        ("Actionable", counts.findings, "#dc2626" if counts.findings else "#16a34a"),
        ("Semantic matches", counts.semantic_matches, "#2563eb"),
        (
            "Wiki actions",
            counts.wiki_build_actions_analyzed or counts.wiki_steps_compared,
            "#475569",
        ),
        ("Details ignored", counts.implementation_details_ignored, "#64748b"),
    ]
    stats_html = "".join(
        f"""
        <td style=\"width:25%;padding:0 4px 0 0;vertical-align:top;\">
          <div style=\"border:1px solid #e2e8f0;border-radius:10px;padding:14px 12px;background:#ffffff;\">
            <div style=\"font-size:25px;line-height:1;font-weight:700;color:{color};\">{value}</div>
            <div style=\"padding-top:7px;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#64748b;\">{escape(label)}</div>
          </div>
        </td>
        """
        for label, value, color in stats
    )

    if findings:
        finding_html = "".join(_finding_html(finding) for finding in findings)
        attention_section = f"""
        <h2 style=\"margin:28px 0 12px;font-size:17px;color:#0f172a;\">What needs attention</h2>
        {finding_html}
        """
    else:
        attention_section = f"""
        <div style=\"margin-top:24px;padding:16px 18px;border-radius:10px;background:{accent_soft};border:1px solid #bbf7d0;\">
          <div style=\"font-size:15px;font-weight:700;color:{accent};\">Everything matched for this revision</div>
          <div style=\"margin-top:6px;font-size:13px;line-height:1.5;color:#334155;\">
            The documented Wiki build actions are represented by the automation script.
            Internal setup and cleanup details were not treated as problems.
          </div>
        </div>
        """

    follow_up = ""
    if pr_url:
        safe_pr_url = escape(pr_url, quote=True)
        follow_up += f"""
        <div style=\"margin-top:20px;padding:14px 16px;border-radius:9px;background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a;\">
          <strong>Draft pull request created.</strong>
          <a href=\"{safe_pr_url}\" style=\"color:#1d4ed8;\">Open the pull request</a>
        </div>
        """
    if action_error:
        follow_up += f"""
        <div style=\"margin-top:12px;padding:14px 16px;border-radius:9px;background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;\">
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
  <body style=\"margin:0;padding:0;background:#f1f5f9;font-family:Arial,Helvetica,sans-serif;color:#0f172a;\">
    <div style=\"padding:28px 12px;text-align:center;\">
      <table role=\"presentation\" align=\"center\" cellpadding=\"0\" cellspacing=\"0\" width=\"680\" style=\"width:100%;max-width:680px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden;text-align:left;\">
        <tr>
          <td style=\"padding:24px 28px;background:#111827;\">
            <div style=\"font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#93c5fd;\">BuildGuard</div>
            <div style=\"margin-top:10px;font-size:24px;line-height:1.2;font-weight:700;color:#ffffff;\">{escape(report.package)} <span style=\"font-weight:400;color:#cbd5e1;\">{escape(version)}</span></div>
            <div style=\"margin-top:14px;display:inline-block;padding:6px 10px;border-radius:999px;background:{accent};font-size:12px;font-weight:700;color:#ffffff;\">{escape(result_label)}</div>
          </td>
        </tr>
        <tr>
          <td style=\"padding:26px 28px 30px;\">
            <div style=\"font-size:15px;line-height:1.6;color:#334155;\">{escape(report.overview)}</div>
            <table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" style=\"width:100%;margin-top:22px;\"><tr>{stats_html}</tr></table>
            {attention_section}
            {follow_up}
            <div style=\"margin-top:28px;padding-top:18px;border-top:1px solid #e2e8f0;font-size:12px;line-height:1.7;color:#64748b;\">
              <strong style=\"color:#475569;\">Compared sources</strong><br>{source_html}
            </div>
          </td>
        </tr>
        <tr>
          <td style=\"padding:16px 28px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;\">
            This notification was generated automatically by BuildGuard.
          </td>
        </tr>
      </table>
    </div>
  </body>
</html>"""


def _finding_html(finding) -> str:
    """Render one actionable finding for the HTML email."""
    category = _category_label(finding.category)
    locations = [evidence.location for evidence in finding.script if evidence.location]
    if finding.wiki and finding.wiki.location:
        locations.insert(0, f"Wiki: {finding.wiki.location}")
    location_html = "<br>".join(escape(location) for location in locations[:4])
    if len(locations) > 4:
        location_html += f"<br>+ {len(locations) - 4} more location(s)"
    location_block = (
        f'<div style="margin-top:10px;font-size:11px;line-height:1.5;color:#64748b;">{location_html}</div>'
        if location_html
        else ""
    )
    return f"""
    <div style=\"margin-top:12px;padding:16px 18px;border:1px solid #fecaca;border-left:4px solid #dc2626;border-radius:9px;background:#fffafa;\">
      <div style=\"font-size:11px;letter-spacing:.05em;text-transform:uppercase;font-weight:700;color:#b91c1c;\">{escape(finding.id)} · {escape(category)}</div>
      <div style=\"margin-top:7px;font-size:15px;font-weight:700;color:#0f172a;\">{escape(finding.title)}</div>
      <div style=\"margin-top:8px;font-size:13px;line-height:1.55;color:#334155;\">{escape(finding.explanation)}</div>
      <div style=\"margin-top:10px;padding:10px 12px;border-radius:6px;background:#fff1f2;font-size:13px;line-height:1.5;color:#881337;\"><strong>Recommended next step:</strong> {escape(finding.recommendation)}</div>
      {location_block}
    </div>
    """


def _category_label(category) -> str:
    value = getattr(category, "value", category)
    return str(value).replace("_", " ").title()
