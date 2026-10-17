from app.models.report import (
    ComparisonCounts,
    ComparisonReport,
    Discrepancy,
    StepEvidence,
)
from app.models.repair import ScriptFixProposal
from app.models.step import ActionType
from app.models.validation import DiscrepancyCategory, MatchStatus
from app.service import ComparisonRun, _comparison_run_key, run_automated_package_review
from app.storage.repository import ComparisonCacheKey, ParsedSourceRepository
from app.config.settings import Settings


def _key() -> ComparisonCacheKey:
    return ComparisonCacheKey(
        package="Alfresco",
        version="26.0.0",
        wiki_repository="example/docs",
        wiki_path="Building-Alfresco.md",
        wiki_ref="HEAD",
        wiki_revision="wiki-sha",
        script_repository="example/scripts",
        script_path="Alfresco/26.0.0/build_alfresco.sh",
        script_ref="master",
        script_revision="script-sha",
        wiki_parser_version="wiki-v1",
        bash_parser_version="bash-v1",
        analysis_version="comparison-v1:ollama:test",
    )


def _report(*, findings: bool) -> ComparisonReport:
    discrepancy = []
    if findings:
        discrepancy = [
            Discrepancy(
                id="D-001",
                status=MatchStatus.EXTRA,
                category=DiscrepancyCategory.UNDOCUMENTED_STEP,
                title="Undocumented script action",
                explanation="The script action is not in the Wiki.",
                recommendation="Review the action.",
                script=[
                    StepEvidence(
                        id="bash-1",
                        description="Run command",
                        action=ActionType.RUN_COMMAND,
                        location="build.sh | line 1",
                        line_start=1,
                        line_end=1,
                        commands=["echo extra"],
                    )
                ],
            )
        ]
    return ComparisonReport(
        package="Alfresco",
        version="26.0.0",
        status="review_required" if findings else "no_mismatches_found",
        overview="Automatic test report",
        counts=ComparisonCounts(
            wiki_steps_compared=1,
            script_steps_checked=1,
            missing_steps=0,
            conflicting_steps=0,
            undocumented_script_steps=int(findings),
            out_of_scope_steps=0,
            findings=int(findings),
        ),
        discrepancies=discrepancy,
    )


def test_automation_saves_a_clean_report_result(tmp_path, monkeypatch) -> None:
    settings = Settings(buildguard_database_path=str(tmp_path / "buildguard.sqlite3"))
    report = _report(findings=False)
    key = _key()
    repository = ParsedSourceRepository(settings.buildguard_database_path)
    stored = repository.save_comparison_report(key, report)

    monkeypatch.setattr(
        "app.service._run_package_comparison",
        lambda *args, **kwargs: ComparisonRun(report, key, stored, True),
    )

    outcome = run_automated_package_review("Alfresco", settings)

    assert outcome.status == "complete"
    assert repository.get_automation_run(_comparison_run_key(key)).status == "no_findings"
    assert len(repository.list_comparison_reports()) == 1


def test_automation_emails_a_clean_report(tmp_path, monkeypatch) -> None:
    settings = Settings(
        buildguard_database_path=str(tmp_path / "buildguard.sqlite3"),
        notification_email_to="owner@example.com",
        smtp_host="smtp.example.com",
        smtp_from="buildguard@example.com",
    )
    report = _report(findings=False)
    key = _key()
    repository = ParsedSourceRepository(settings.buildguard_database_path)
    stored = repository.save_comparison_report(key, report)
    monkeypatch.setattr(
        "app.service._run_package_comparison",
        lambda *args, **kwargs: ComparisonRun(report, key, stored, True),
    )
    notifications: list[ComparisonReport] = []
    monkeypatch.setattr(
        "app.service.send_report_notification",
        lambda _settings, notification_report, **_kwargs: notifications.append(
            notification_report
        ),
    )

    outcome = run_automated_package_review("Alfresco", settings)

    assert outcome.status == "complete"
    assert notifications == [report]
    automation_run = repository.get_automation_run(_comparison_run_key(key))
    assert automation_run.status == "notification_sent"
    assert automation_run.notification_sent


def test_automation_does_not_create_duplicate_pr_for_same_report(tmp_path, monkeypatch) -> None:
    settings = Settings(
        buildguard_database_path=str(tmp_path / "buildguard.sqlite3"),
        buildguard_auto_pr=True,
    )
    report = _report(findings=True)
    key = _key()
    repository = ParsedSourceRepository(settings.buildguard_database_path)
    stored = repository.save_comparison_report(key, report)
    monkeypatch.setattr(
        "app.service._run_package_comparison",
        lambda *args, **kwargs: ComparisonRun(report, key, stored, True),
    )
    monkeypatch.setattr(
        "app.service.load_script_source_content",
        lambda *args, **kwargs: "echo extra\n",
    )
    monkeypatch.setattr(
        "app.service.propose_script_fix",
        lambda *args, **kwargs: ScriptFixProposal(
            fixed_script="echo fixed\n",
            summary="Fix",
            rationale="Fix",
            affected_finding_ids=["D-001"],
        ),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "app.service.create_script_fix_pull_request",
        lambda *args, **kwargs: calls.append(kwargs["branch_name"]) or {
            "html_url": "https://github.com/example/scripts/pull/1"
        },
    )

    first = run_automated_package_review("Alfresco", settings)
    second = run_automated_package_review("Alfresco", settings)

    assert first.pr_url == second.pr_url
    assert len(calls) == 1
    assert repository.get_automation_run(_comparison_run_key(key)).pr_url.endswith("/pull/1")
