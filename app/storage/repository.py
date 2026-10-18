"""SQLite persistence for validated source interpretations."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from collections.abc import Iterator

from app.models.build import BuildInstruction
from app.models.report import ComparisonReport


@dataclass(frozen=True)
class ComparisonCacheKey:
    """Immutable identity of one source revision comparison."""

    package: str
    version: str
    wiki_repository: str
    wiki_path: str
    wiki_ref: str
    wiki_revision: str
    script_repository: str
    script_path: str
    script_ref: str
    script_revision: str
    wiki_parser_version: str
    bash_parser_version: str
    analysis_version: str


@dataclass(frozen=True)
class StoredComparisonReport:
    """A persisted report together with its cache metadata."""

    report_id: int
    report: ComparisonReport
    updated_at: str
    key: ComparisonCacheKey


@dataclass(frozen=True)
class WebhookJob:
    """Persisted package/version work scheduled by a GitHub event."""

    job_id: int
    package: str
    version: str | None
    refresh_wiki: bool
    refresh_bash: bool
    status: str
    attempts: int
    last_error: str | None
    updated_at: str
    trigger_id: str | None = None


@dataclass(frozen=True)
class AutomationRun:
    """Persisted post-comparison actions for one exact comparison key."""

    run_key: str
    package: str
    version: str
    report_id: int | None
    status: str
    pr_url: str | None
    notification_sent: bool
    last_error: str | None
    updated_at: str


class ParsedSourceRepository:
    """Store parsed Wiki/Bash instructions keyed by their remote revision."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get_instruction(
        self,
        *,
        source_kind: str,
        repository: str,
        path: str,
        ref: str,
        parser_version: str,
    ) -> tuple[BuildInstruction, str | None] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT remote_revision, parsed_json
                FROM parsed_sources
                WHERE source_kind = ?
                  AND repository = ?
                  AND path = ?
                  AND ref = ?
                  AND parser_version = ?
                """,
                (source_kind, repository, path, ref, parser_version),
            ).fetchone()
        if row is None:
            return None
        try:
            instruction = BuildInstruction.model_validate(json.loads(row[1]))
        except (TypeError, ValueError, json.JSONDecodeError):
            # A malformed or old cache entry must never prevent a fresh parse.
            return None
        return instruction, row[0]

    def save_instruction(
        self,
        *,
        source_kind: str,
        repository: str,
        path: str,
        ref: str,
        parser_version: str,
        remote_revision: str | None,
        content: str,
        instruction: BuildInstruction,
    ) -> None:
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        updated_at = datetime.now(UTC).isoformat()
        parsed_json = json.dumps(
            instruction.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO parsed_sources (
                    source_kind,
                    repository,
                    path,
                    ref,
                    parser_version,
                    remote_revision,
                    content_sha256,
                    source_content,
                    parsed_json,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_kind, repository, path, ref, parser_version)
                DO UPDATE SET
                    remote_revision = excluded.remote_revision,
                    content_sha256 = excluded.content_sha256,
                    source_content = excluded.source_content,
                    parsed_json = excluded.parsed_json,
                    updated_at = excluded.updated_at
                """,
                (
                    source_kind,
                    repository,
                    path,
                    ref,
                    parser_version,
                    remote_revision,
                    content_sha256,
                    content,
                    parsed_json,
                    updated_at,
                ),
            )

    def get_source_content(
        self,
        *,
        source_kind: str,
        repository: str,
        path: str,
        ref: str,
        parser_version: str,
    ) -> str | None:
        """Return the original source text stored with a parsed entry."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_content
                FROM parsed_sources
                WHERE source_kind = ?
                  AND repository = ?
                  AND path = ?
                  AND ref = ?
                  AND parser_version = ?
                """,
                (source_kind, repository, path, ref, parser_version),
            ).fetchone()
        return None if row is None else row[0]

    def save_source_content(
        self,
        *,
        source_kind: str,
        repository: str,
        path: str,
        ref: str,
        parser_version: str,
        content: str,
    ) -> None:
        """Backfill original source text for an existing parsed entry."""
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        updated_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE parsed_sources
                SET content_sha256 = ?,
                    source_content = ?,
                    updated_at = ?
                WHERE source_kind = ?
                  AND repository = ?
                  AND path = ?
                  AND ref = ?
                  AND parser_version = ?
                """,
                (
                    content_sha256,
                    content,
                    updated_at,
                    source_kind,
                    repository,
                    path,
                    ref,
                    parser_version,
                ),
            )

    def get_remote_revision(
        self,
        *,
        source_kind: str,
        repository: str,
        path: str,
        ref: str,
        parser_version: str,
    ) -> str | None:
        """Return the revision associated with a parsed source cache entry."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT remote_revision
                FROM parsed_sources
                WHERE source_kind = ?
                  AND repository = ?
                  AND path = ?
                  AND ref = ?
                  AND parser_version = ?
                """,
                (source_kind, repository, path, ref, parser_version),
            ).fetchone()
        return None if row is None else row[0]

    def get_content_sha256(
        self,
        *,
        source_kind: str,
        repository: str,
        path: str,
        ref: str,
        parser_version: str,
    ) -> str | None:
        """Return the content fingerprint stored with a parsed source."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT content_sha256
                FROM parsed_sources
                WHERE source_kind = ?
                  AND repository = ?
                  AND path = ?
                  AND ref = ?
                  AND parser_version = ?
                """,
                (source_kind, repository, path, ref, parser_version),
            ).fetchone()
        return None if row is None else row[0]

    def get_comparison_report(
        self,
        key: ComparisonCacheKey,
    ) -> StoredComparisonReport | None:
        """Load a report only when every source and analysis key is unchanged."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, updated_at, report_json
                FROM comparison_reports
                WHERE package = ?
                  AND version = ?
                  AND wiki_repository = ?
                  AND wiki_path = ?
                  AND wiki_ref = ?
                  AND wiki_revision = ?
                  AND script_repository = ?
                  AND script_path = ?
                  AND script_ref = ?
                  AND script_revision = ?
                  AND wiki_parser_version = ?
                  AND bash_parser_version = ?
                  AND analysis_version = ?
                """,
                _comparison_key_values(key),
            ).fetchone()
        return self._stored_comparison_report(row, key) if row else None

    def get_latest_comparison_report_for_sources(
        self,
        key: ComparisonCacheKey,
    ) -> StoredComparisonReport | None:
        """Load the newest report for an exact source revision pair.

        Parser and analysis versions are deliberately excluded here. The UI's
        normal compare action should show the last result when neither input
        changed; users can explicitly request a fresh analysis when desired.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, updated_at, report_json,
                       wiki_parser_version, bash_parser_version, analysis_version
                FROM comparison_reports
                WHERE package = ?
                  AND version = ?
                  AND wiki_repository = ?
                  AND wiki_path = ?
                  AND wiki_ref = ?
                  AND wiki_revision = ?
                  AND script_repository = ?
                  AND script_path = ?
                  AND script_ref = ?
                  AND script_revision = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (
                    key.package,
                    key.version,
                    key.wiki_repository,
                    key.wiki_path,
                    key.wiki_ref,
                    key.wiki_revision,
                    key.script_repository,
                    key.script_path,
                    key.script_ref,
                    key.script_revision,
                ),
            ).fetchone()
        if row is None:
            return None

        stored_key = ComparisonCacheKey(
            package=key.package,
            version=key.version,
            wiki_repository=key.wiki_repository,
            wiki_path=key.wiki_path,
            wiki_ref=key.wiki_ref,
            wiki_revision=key.wiki_revision,
            script_repository=key.script_repository,
            script_path=key.script_path,
            script_ref=key.script_ref,
            script_revision=key.script_revision,
            wiki_parser_version=row[3],
            bash_parser_version=row[4],
            analysis_version=row[5],
        )
        return self._stored_comparison_report((row[0], row[1], row[2]), stored_key)

    def save_comparison_report(
        self,
        key: ComparisonCacheKey,
        report: ComparisonReport,
    ) -> StoredComparisonReport:
        """Persist or replace the report for one exact source revision pair."""
        updated_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO comparison_reports (
                    package, version,
                    wiki_repository, wiki_path, wiki_ref, wiki_revision,
                    script_repository, script_path, script_ref, script_revision,
                    wiki_parser_version, bash_parser_version, analysis_version,
                    report_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    package, version,
                    wiki_repository, wiki_path, wiki_ref, wiki_revision,
                    script_repository, script_path, script_ref, script_revision,
                    wiki_parser_version, bash_parser_version, analysis_version
                ) DO UPDATE SET
                    report_json = excluded.report_json,
                    updated_at = excluded.updated_at
                """,
                (
                    *_comparison_key_values(key),
                    json.dumps(report.model_dump(mode="json"), ensure_ascii=False),
                    updated_at,
                ),
            )
            row = connection.execute(
                """
                SELECT id, updated_at, report_json
                FROM comparison_reports
                WHERE package = ?
                  AND version = ?
                  AND wiki_repository = ?
                  AND wiki_path = ?
                  AND wiki_ref = ?
                  AND wiki_revision = ?
                  AND script_repository = ?
                  AND script_path = ?
                  AND script_ref = ?
                  AND script_revision = ?
                  AND wiki_parser_version = ?
                  AND bash_parser_version = ?
                  AND analysis_version = ?
                """,
                _comparison_key_values(key),
            ).fetchone()
        if row is None:
            raise RuntimeError("Saved comparison report could not be reloaded")
        stored = self._stored_comparison_report(row, key)
        if stored is None:
            raise RuntimeError("Saved comparison report is invalid")
        return stored

    def list_comparison_reports(self) -> list[StoredComparisonReport]:
        """List all saved reports, grouped by package and newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, package, version,
                       wiki_repository, wiki_path, wiki_ref, wiki_revision,
                       script_repository, script_path, script_ref, script_revision,
                       wiki_parser_version, bash_parser_version, analysis_version,
                       report_json, updated_at
                FROM comparison_reports
                ORDER BY package COLLATE NOCASE, updated_at DESC
                """
            ).fetchall()

        reports: list[StoredComparisonReport] = []
        for row in rows:
            key = ComparisonCacheKey(
                package=row[1],
                version=row[2],
                wiki_repository=row[3],
                wiki_path=row[4],
                wiki_ref=row[5],
                wiki_revision=row[6],
                script_repository=row[7],
                script_path=row[8],
                script_ref=row[9],
                script_revision=row[10],
                wiki_parser_version=row[11],
                bash_parser_version=row[12],
                analysis_version=row[13],
            )
            stored = self._stored_comparison_report(
                (row[0], row[15], row[14]),
                key,
            )
            if stored is not None:
                reports.append(stored)
        return reports

    def enqueue_webhook_job(
        self,
        *,
        package: str,
        version: str | None,
        refresh_wiki: bool,
        refresh_bash: bool,
        trigger_id: str | None = None,
    ) -> WebhookJob:
        """Create or merge pending work for one package/version/event."""
        updated_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, refresh_wiki, refresh_bash, trigger_id
                FROM webhook_jobs
                WHERE package = ?
                  AND version IS ?
                  AND status IN ('pending', 'running')
                ORDER BY id DESC
                LIMIT 1
                """,
                (package, version),
            ).fetchone()
            if row is None:
                existing_event = None
                if trigger_id:
                    existing_event = connection.execute(
                        """
                        SELECT id, status
                        FROM webhook_jobs
                        WHERE package = ?
                          AND version IS ?
                          AND trigger_id = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (package, version, trigger_id),
                    ).fetchone()
                if existing_event is not None:
                    job_id = int(existing_event[0])
                    if existing_event[1] != "complete":
                        connection.execute(
                            """
                            UPDATE webhook_jobs
                            SET status = 'pending', attempts = 0,
                                last_error = NULL, updated_at = ?
                            WHERE id = ?
                            """,
                            (updated_at, job_id),
                        )
                else:
                    cursor = connection.execute(
                        """
                        INSERT INTO webhook_jobs (
                            package, version, refresh_wiki, refresh_bash,
                            status, attempts, last_error, updated_at, trigger_id
                        ) VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, ?)
                        """,
                        (package, version, refresh_wiki, refresh_bash, updated_at, trigger_id),
                    )
                    job_id = int(cursor.lastrowid)
            else:
                job_id = int(row[0])
                connection.execute(
                    """
                    UPDATE webhook_jobs
                    SET refresh_wiki = ?,
                        refresh_bash = ?,
                        trigger_id = COALESCE(?, trigger_id),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        bool(row[1]) or refresh_wiki,
                        bool(row[2]) or refresh_bash,
                        trigger_id,
                        updated_at,
                        job_id,
                    ),
                )
        job = self.get_webhook_job(job_id)
        if job is None:
            raise RuntimeError("Webhook job could not be reloaded")
        return job

    def recover_webhook_jobs(self) -> None:
        """Return jobs interrupted by a previous backend process to the queue."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE webhook_jobs
                SET status = 'pending', updated_at = ?
                WHERE status = 'running'
                """,
                (datetime.now(UTC).isoformat(),),
            )

    def get_webhook_job(self, job_id: int) -> WebhookJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, package, version, refresh_wiki, refresh_bash,
                       status, attempts, last_error, updated_at, trigger_id
                FROM webhook_jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        return self._webhook_job(row)

    def get_active_webhook_job(
        self,
        *,
        package: str,
        version: str | None,
    ) -> WebhookJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, package, version, refresh_wiki, refresh_bash,
                       status, attempts, last_error, updated_at, trigger_id
                FROM webhook_jobs
                WHERE package = ?
                  AND version IS ?
                  AND status IN ('pending', 'running')
                ORDER BY id DESC
                LIMIT 1
                """,
                (package, version),
            ).fetchone()
        return self._webhook_job(row)

    def list_pending_webhook_jobs(self) -> list[WebhookJob]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, package, version, refresh_wiki, refresh_bash,
                       status, attempts, last_error, updated_at, trigger_id
                FROM webhook_jobs
                WHERE status = 'pending'
                ORDER BY id
                """
            ).fetchall()
        return [job for row in rows if (job := self._webhook_job(row)) is not None]

    def claim_webhook_job(self, job_id: int) -> WebhookJob | None:
        """Atomically claim pending work and clear its queued refresh flags."""
        with self._connect() as connection:
            current = connection.execute(
                """
                SELECT id, package, version, refresh_wiki, refresh_bash,
                       status, attempts, last_error, updated_at, trigger_id
                FROM webhook_jobs
                WHERE id = ? AND status = 'pending'
                """,
                (job_id,),
            ).fetchone()
            if current is None:
                return None
            connection.execute(
                """
                UPDATE webhook_jobs
                SET status = 'running',
                    refresh_wiki = 0,
                    refresh_bash = 0,
                    attempts = attempts + 1,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (datetime.now(UTC).isoformat(), job_id),
            )
        return WebhookJob(
            job_id=int(current[0]),
            package=str(current[1]),
            version=current[2],
            refresh_wiki=bool(current[3]),
            refresh_bash=bool(current[4]),
            status="running",
            attempts=int(current[6]) + 1,
            last_error=None,
            updated_at=datetime.now(UTC).isoformat(),
            trigger_id=current[9],
        )

    def finish_webhook_job(
        self,
        job_id: int,
        *,
        status: str,
        last_error: str | None = None,
        retryable: bool = False,
    ) -> bool:
        """Finish a job, retaining any new flags merged while it was running."""
        if status not in {"complete", "waiting", "failed"}:
            raise ValueError(f"Unsupported webhook job status: {status}")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT refresh_wiki, refresh_bash
                FROM webhook_jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                return False
            pending = bool(row[0]) or bool(row[1])
            next_status = "pending" if pending or retryable else status
            connection.execute(
                """
                UPDATE webhook_jobs
                SET status = ?, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (next_status, last_error, datetime.now(UTC).isoformat(), job_id),
            )
            return pending

    def get_automation_run(self, run_key: str) -> AutomationRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_key, package, version, report_id, status,
                       pr_url, notification_sent, last_error, updated_at
                FROM automation_runs
                WHERE run_key = ?
                """,
                (run_key,),
            ).fetchone()
        return self._automation_run(row)

    def save_automation_run(
        self,
        *,
        run_key: str,
        package: str,
        version: str,
        report_id: int | None,
        status: str,
        pr_url: str | None = None,
        notification_sent: bool = False,
        last_error: str | None = None,
    ) -> AutomationRun:
        """Persist an idempotent record of automatic PR/notification work."""
        updated_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO automation_runs (
                    run_key, package, version, report_id, status,
                    pr_url, notification_sent, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_key) DO UPDATE SET
                    report_id = excluded.report_id,
                    status = excluded.status,
                    pr_url = COALESCE(excluded.pr_url, automation_runs.pr_url),
                    notification_sent = (
                        automation_runs.notification_sent OR excluded.notification_sent
                    ),
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (
                    run_key,
                    package,
                    version,
                    report_id,
                    status,
                    pr_url,
                    notification_sent,
                    last_error,
                    updated_at,
                ),
            )
        result = self.get_automation_run(run_key)
        if result is None:
            raise RuntimeError("Automation run could not be reloaded")
        return result

    @staticmethod
    def _stored_comparison_report(
        row: sqlite3.Row | tuple[object, object, object] | None,
        key: ComparisonCacheKey,
    ) -> StoredComparisonReport | None:
        if row is None:
            return None
        try:
            report = ComparisonReport.model_validate(json.loads(row[2]))
            return StoredComparisonReport(
                report_id=int(row[0]),
                report=report,
                updated_at=str(row[1]),
                key=key,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _webhook_job(row: sqlite3.Row | None) -> WebhookJob | None:
        if row is None:
            return None
        return WebhookJob(
            job_id=int(row[0]),
            package=str(row[1]),
            version=row[2],
            refresh_wiki=bool(row[3]),
            refresh_bash=bool(row[4]),
            status=str(row[5]),
            attempts=int(row[6]),
            last_error=row[7],
            updated_at=str(row[8]),
            trigger_id=row[9],
        )

    @staticmethod
    def _automation_run(row: sqlite3.Row | None) -> AutomationRun | None:
        if row is None:
            return None
        return AutomationRun(
            run_key=str(row[0]),
            package=str(row[1]),
            version=str(row[2]),
            report_id=int(row[3]) if row[3] is not None else None,
            status=str(row[4]),
            pr_url=row[5],
            notification_sent=bool(row[6]),
            last_error=row[7],
            updated_at=str(row[8]),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS parsed_sources (
                    source_kind TEXT NOT NULL,
                    repository TEXT NOT NULL,
                    path TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    remote_revision TEXT,
                    content_sha256 TEXT NOT NULL,
                    source_content TEXT,
                    parsed_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        source_kind,
                        repository,
                        path,
                        ref,
                        parser_version
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS comparison_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    package TEXT NOT NULL,
                    version TEXT NOT NULL,
                    wiki_repository TEXT NOT NULL,
                    wiki_path TEXT NOT NULL,
                    wiki_ref TEXT NOT NULL,
                    wiki_revision TEXT NOT NULL,
                    script_repository TEXT NOT NULL,
                    script_path TEXT NOT NULL,
                    script_ref TEXT NOT NULL,
                    script_revision TEXT NOT NULL,
                    wiki_parser_version TEXT NOT NULL,
                    bash_parser_version TEXT NOT NULL,
                    analysis_version TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (
                        package, version,
                        wiki_repository, wiki_path, wiki_ref, wiki_revision,
                        script_repository, script_path, script_ref, script_revision,
                        wiki_parser_version, bash_parser_version, analysis_version
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    package TEXT NOT NULL,
                    version TEXT,
                    refresh_wiki INTEGER NOT NULL DEFAULT 0,
                    refresh_bash INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL,
                    trigger_id TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhook_jobs_active
                ON webhook_jobs(package, version, status)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS automation_runs (
                    run_key TEXT PRIMARY KEY,
                    package TEXT NOT NULL,
                    version TEXT NOT NULL,
                    report_id INTEGER,
                    status TEXT NOT NULL,
                    pr_url TEXT,
                    notification_sent INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(parsed_sources)")
            }
            if "source_content" not in columns:
                connection.execute(
                    "ALTER TABLE parsed_sources ADD COLUMN source_content TEXT"
                )
            webhook_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(webhook_jobs)")
            }
            if "trigger_id" not in webhook_columns:
                connection.execute(
                    "ALTER TABLE webhook_jobs ADD COLUMN trigger_id TEXT"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=30,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _comparison_key_values(key: ComparisonCacheKey) -> tuple[str, ...]:
    return (
        key.package,
        key.version,
        key.wiki_repository,
        key.wiki_path,
        key.wiki_ref,
        key.wiki_revision,
        key.script_repository,
        key.script_path,
        key.script_ref,
        key.script_revision,
        key.wiki_parser_version,
        key.bash_parser_version,
        key.analysis_version,
    )
