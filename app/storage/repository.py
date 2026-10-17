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
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(parsed_sources)")
            }
            if "source_content" not in columns:
                connection.execute(
                    "ALTER TABLE parsed_sources ADD COLUMN source_content TEXT"
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
