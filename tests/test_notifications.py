from email.message import EmailMessage

from app.models.report import ComparisonCounts, ComparisonReport
from app.notifications import _build_report_notification_message


def _clean_report() -> ComparisonReport:
    return ComparisonReport(
        package="Alfresco",
        version="25.3.0",
        status="no_mismatches_found",
        overview="All documented actions matched the script.",
        counts=ComparisonCounts(
            wiki_steps_compared=30,
            script_steps_checked=113,
            wiki_build_actions_analyzed=30,
            script_build_actions_analyzed=98,
            missing_steps=0,
            conflicting_steps=0,
            undocumented_script_steps=0,
            out_of_scope_steps=0,
            findings=0,
            semantic_matches=30,
            implementation_details_ignored=15,
        ),
    )


def test_notification_contains_plain_text_and_html_parts() -> None:
    message = _build_report_notification_message(
        _clean_report(),
        sender="buildguard@example.com",
        recipients=["owner@example.com"],
    )

    assert isinstance(message, EmailMessage)
    assert message["Subject"] == "BuildGuard · Alfresco 25.3.0 · No issues found"
    assert message.get_body(preferencelist=("plain",)).get_content().startswith(
        "BuildGuard review: Alfresco 25.3.0"
    )
    html = message.get_body(preferencelist=("html",)).get_content()
    assert "Everything matched for this revision" in html
    assert 'align="center"' in html
    assert "30" in html
    assert "No actionable findings were found" in message.get_body(
        preferencelist=("plain",)
    ).get_content()
