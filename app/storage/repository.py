"""SQLite persistence for validated source interpretations."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from collections.abc import Iterator

from app.models.build import BuildInstruction


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
