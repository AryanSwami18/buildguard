from app.storage.repository import ParsedSourceRepository
from app.config.settings import Settings
from app.service import AutomationOutcome
from app.webhook import WebhookProcessor
from threading import Event


def test_webhook_jobs_merge_events_while_a_refresh_is_running(tmp_path) -> None:
    repository = ParsedSourceRepository(tmp_path / "buildguard.sqlite3")
    job = repository.enqueue_webhook_job(
        package="Alfresco",
        version="26.0.0",
        refresh_wiki=True,
        refresh_bash=False,
    )
    claimed = repository.claim_webhook_job(job.job_id)

    assert claimed is not None
    assert claimed.refresh_wiki
    assert not claimed.refresh_bash

    repository.enqueue_webhook_job(
        package="Alfresco",
        version="26.0.0",
        refresh_wiki=False,
        refresh_bash=True,
    )
    assert repository.finish_webhook_job(
        job.job_id,
        status="complete",
    )

    pending = repository.get_webhook_job(job.job_id)
    assert pending is not None
    assert pending.status == "pending"
    assert pending.refresh_bash
    assert not pending.refresh_wiki

    next_claim = repository.claim_webhook_job(job.job_id)
    assert next_claim is not None
    assert not next_claim.refresh_wiki
    assert next_claim.refresh_bash


def test_interrupted_webhook_jobs_are_recovered(tmp_path) -> None:
    repository = ParsedSourceRepository(tmp_path / "buildguard.sqlite3")
    job = repository.enqueue_webhook_job(
        package="Bazel",
        version=None,
        refresh_wiki=True,
        refresh_bash=False,
    )
    assert repository.claim_webhook_job(job.job_id) is not None

    repository.recover_webhook_jobs()

    recovered = repository.get_webhook_job(job.job_id)
    assert recovered is not None
    assert recovered.status == "pending"


def test_automation_run_is_idempotently_updated(tmp_path) -> None:
    repository = ParsedSourceRepository(tmp_path / "buildguard.sqlite3")

    first = repository.save_automation_run(
        run_key="run-1",
        package="Alfresco",
        version="26.0.0",
        report_id=4,
        status="pr_created",
        pr_url="https://github.com/example/scripts/pull/4",
    )
    second = repository.save_automation_run(
        run_key="run-1",
        package="Alfresco",
        version="26.0.0",
        report_id=4,
        status="notification_sent",
        notification_sent=True,
    )

    assert first.run_key == second.run_key
    assert second.pr_url == "https://github.com/example/scripts/pull/4"
    assert second.notification_sent
    assert repository.get_automation_run("run-1").status == "notification_sent"


def test_webhook_processor_schedules_versioned_work(monkeypatch, tmp_path) -> None:
    settings = Settings(
        github_repo="example/docs",
        scripts_repo="example/scripts",
        buildguard_database_path=str(tmp_path / "buildguard.sqlite3"),
    )
    completed = Event()
    calls: list[tuple[str, str | None, bool, bool]] = []

    def fake_review(package, settings, progress=None, **kwargs):
        calls.append(
            (
                package,
                kwargs["version"],
                kwargs["refresh_wiki"],
                kwargs["refresh_bash"],
            )
        )
        completed.set()
        return AutomationOutcome("complete")

    monkeypatch.setattr("app.webhook.run_automated_package_review", fake_review)
    processor = WebhookProcessor(settings)
    try:
        accepted = processor.submit_event(
            "push",
            {
                "repository": {"full_name": "example/scripts"},
                "ref": "refs/heads/master",
                "commits": [{"added": ["Alfresco/26.0.0/build_alfresco.sh"]}],
            },
        )
        assert accepted == 1
        assert completed.wait(timeout=2)
        assert calls == [("Alfresco", "26.0.0", False, True)]
    finally:
        processor.executor.shutdown(wait=True)
