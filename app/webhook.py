"""Small GitHub webhook listener for proactive cache synchronization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from threading import Lock
from typing import Any

from app.config.settings import Settings, load_project_environment
from app.package_mapping import canonical_wiki_package, wiki_package_for_script_path
from app.parsers.version_parser import product_from_module
from app.service import run_automated_package_review
from app.storage.repository import ParsedSourceRepository, WebhookJob

LOGGER = logging.getLogger("buildguard.webhook")
WEBHOOK_PATH = "/github/webhook"


@dataclass(frozen=True)
class WebhookTarget:
    """A package/version affected by one GitHub event."""

    package: str
    version: str | None
    refresh_wiki: bool
    refresh_bash: bool
    trigger_id: str | None = None


def verify_signature(secret: str, body: bytes, signature: str | None) -> bool:
    """Verify GitHub's HMAC-SHA256 webhook signature."""
    if not signature or not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


class WebhookProcessor:
    """Turn GitHub events into durable, deduplicated review jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = Lock()
        self.cache = ParsedSourceRepository(settings.buildguard_database_path)
        self.cache.recover_webhook_jobs()
        self.scheduled: set[tuple[str, str | None]] = set()
        self.package_locks: dict[str, Lock] = {}
        for job in self.cache.list_pending_webhook_jobs():
            self._schedule(job)

    def submit_event(self, event: str, payload: dict[str, Any]) -> int:
        targets = _webhook_targets(event, payload, self.settings)
        LOGGER.info(
            "Accepted GitHub %s event for %s; affected targets=%d trigger=%s",
            event,
            (payload.get("repository") or {}).get("full_name", "unknown repository"),
            len(targets),
            _event_trigger_id(event, payload) or "none",
        )
        for target in targets:
            job = self.cache.enqueue_webhook_job(
                package=target.package,
                version=target.version,
                refresh_wiki=target.refresh_wiki,
                refresh_bash=target.refresh_bash,
                trigger_id=target.trigger_id,
            )
            LOGGER.info(
                "Queued BuildGuard webhook job %s for %s %s (wiki=%s bash=%s trigger=%s)",
                job.job_id,
                target.package,
                target.version or "latest",
                target.refresh_wiki,
                target.refresh_bash,
                target.trigger_id or "none",
            )
            self._schedule(job)
        return len(targets)

    def _schedule(self, job: WebhookJob) -> None:
        key = (job.package, job.version)
        with self.lock:
            if key in self.scheduled:
                return
            self.scheduled.add(key)
        self.executor.submit(self._refresh, job.job_id, key)

    def _refresh(self, job_id: int, key: tuple[str, str | None]) -> None:
        # A docs job (version unknown) and a script job (version known) can be
        # created by the same commit. Serialize them per package so they do
        # not both start an expensive AI comparison before the first report is
        # saved.
        with self.lock:
            package_lock = self.package_locks.setdefault(key[0], Lock())
        with package_lock:
            self._refresh_locked(job_id, key)

    def _refresh_locked(self, job_id: int, key: tuple[str, str | None]) -> None:
        try:
            while True:
                job = self.cache.claim_webhook_job(job_id)
                if job is None:
                    return
                try:
                    LOGGER.info(
                        "Starting BuildGuard webhook job %s for %s %s (trigger=%s)",
                        job.job_id,
                        job.package,
                        job.version or "latest",
                        job.trigger_id or "none",
                    )
                    outcome = run_automated_package_review(
                        job.package,
                        self.settings,
                        progress=lambda message: LOGGER.info(
                            "%s %s: %s",
                            job.package,
                            job.version or "latest",
                            message,
                        ),
                        version=job.version,
                        refresh_wiki=job.refresh_wiki,
                        refresh_bash=job.refresh_bash,
                        trigger_id=job.trigger_id,
                        force_comparison=bool(job.trigger_id),
                    )
                    pending = self.cache.finish_webhook_job(
                        job.job_id,
                        status=(
                            "waiting"
                            if outcome.status.startswith("waiting_")
                            else "failed"
                            if outcome.status == "followup_failed"
                            else "complete"
                        ),
                        last_error=outcome.message,
                        retryable=(
                            outcome.status == "followup_failed"
                            and job.attempts < 3
                        ),
                    )
                    if (
                        pending
                        and outcome.status == "followup_failed"
                        and job.attempts < 3
                    ):
                        time.sleep(min(2 ** job.attempts, 8))
                    else:
                        LOGGER.info(
                            "Finished BuildGuard webhook job %s for %s %s: %s",
                            job.job_id,
                            job.package,
                            job.version or "latest",
                            outcome.status,
                        )
                except Exception as error:
                    LOGGER.exception(
                        "Unable to process BuildGuard webhook job for %s",
                        job.package,
                    )
                    pending = self.cache.finish_webhook_job(
                        job.job_id,
                        status="failed",
                        last_error=str(error) or error.__class__.__name__,
                        retryable=job.attempts < 3,
                    )
                    if pending and job.attempts < 3:
                        time.sleep(min(2 ** job.attempts, 8))
                if not pending:
                    return
        except Exception:
            LOGGER.exception("Unable to finalize BuildGuard webhook job for %s", key[0])
        finally:
            with self.lock:
                self.scheduled.discard(key)
            # An event can arrive between the final database update and the
            # in-memory scheduling cleanup. Re-check durable state so it is
            # never stranded without a worker.
            pending = self.cache.get_active_webhook_job(
                package=key[0],
                version=key[1],
            )
            if pending is not None:
                self._schedule(pending)


def _cache_jobs(
    event: str,
    payload: dict[str, Any],
    settings: Settings,
) -> set[tuple[str, bool, bool]]:
    """Compatibility view used by callers that only need package-level flags."""
    jobs: dict[str, tuple[bool, bool]] = {}
    for target in _webhook_targets(event, payload, settings):
        existing = jobs.get(target.package, (False, False))
        jobs[target.package] = (
            existing[0] or target.refresh_wiki,
            existing[1] or target.refresh_bash,
        )
    return {
        (package, refresh_wiki, refresh_bash)
        for package, (refresh_wiki, refresh_bash) in jobs.items()
    }


def _webhook_targets(
    event: str,
    payload: dict[str, Any],
    settings: Settings,
) -> set[WebhookTarget]:
    repository = payload.get("repository") or {}
    repository_name = repository.get("full_name")
    if not isinstance(repository_name, str):
        return set()

    trigger_id = _event_trigger_id(event, payload)

    if event == "gollum" and repository_name == settings.github_repo:
        jobs: set[WebhookTarget] = set()
        for page in payload.get("pages") or []:
            if not isinstance(page, dict):
                continue
            page_name = page.get("page_name")
            if not isinstance(page_name, str):
                continue
            if page_name.casefold().startswith("building-"):
                jobs.add(
                    WebhookTarget(
                        canonical_wiki_package(product_from_module(page_name)),
                        None,
                        True,
                        False,
                        trigger_id,
                    )
                )
        return jobs

    if event != "push":
        return set()

    if not _push_matches_branch(payload, settings, repository_name):
        return set()

    changed_paths = _changed_paths(payload)
    if repository_name == settings.github_repo:
        return {
            WebhookTarget(
                canonical_wiki_package(product_from_module(path.name)),
                None,
                True,
                False,
                trigger_id,
            )
            for path in map(PurePosixPath, changed_paths)
            if path.suffix.casefold() == ".md"
            and path.name.casefold().startswith("building-")
        }

    if repository_name == settings.scripts_repo:
        jobs: set[WebhookTarget] = set()
        for raw_path in changed_paths:
            path = PurePosixPath(raw_path)
            parts = path.parts
            if (
                len(parts) >= 3
                and parts[1]
                and path.suffix.casefold() == ".sh"
            ):
                jobs.add(
                    WebhookTarget(
                        wiki_package_for_script_path(raw_path),
                        parts[1],
                        False,
                        True,
                        trigger_id,
                    )
                )
        return jobs

    return set()


def _event_trigger_id(event: str, payload: dict[str, Any]) -> str | None:
    """Return a stable identifier for one GitHub source-change event."""
    if event == "push":
        value = payload.get("after")
        if isinstance(value, str) and value:
            return f"push:{value}"
        head_commit = payload.get("head_commit") or {}
        if isinstance(head_commit, dict):
            value = head_commit.get("id")
            if isinstance(value, str) and value:
                return f"push:{value}"
    if event == "gollum":
        pages = payload.get("pages") or []
        page_revisions = sorted(
            (page.get("page_name"), page.get("sha"))
            for page in pages
            if isinstance(page, dict)
            and isinstance(page.get("page_name"), str)
            and isinstance(page.get("sha"), str)
        )
        if page_revisions:
            digest = hashlib.sha256(
                json.dumps(page_revisions, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            return f"gollum:{digest}"
    return None


def _push_matches_branch(
    payload: dict[str, Any],
    settings: Settings,
    repository_name: str,
) -> bool:
    ref = payload.get("ref")
    configured = settings.scripts_branch if repository_name == settings.scripts_repo else settings.github_branch
    if not isinstance(ref, str):
        return True
    if configured == "HEAD":
        repository = payload.get("repository") or {}
        default_branch = repository.get("default_branch")
        if isinstance(default_branch, str) and default_branch:
            return ref == f"refs/heads/{default_branch}"
        return True
    return ref == f"refs/heads/{configured}"


def _changed_paths(payload: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for commit in payload.get("commits") or []:
        if not isinstance(commit, dict):
            continue
        for field in ("added", "modified", "removed"):
            values = commit.get(field) or []
            paths.update(value for value in values if isinstance(value, str))
    return paths


class _WebhookServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], settings: Settings) -> None:
        super().__init__(address, _WebhookHandler)
        self.settings = settings
        self.processor = WebhookProcessor(settings)


class _WebhookHandler(BaseHTTPRequestHandler):
    server: _WebhookServer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if self.path.rstrip("/") != "/healthz":
            self._respond(404, {"error": "not found"})
            return
        self._respond(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        if self.path.rstrip("/") != WEBHOOK_PATH:
            self._respond(404, {"error": "not found"})
            return

        if not self.server.settings.github_webhook_secret:
            self._respond(503, {"error": "GITHUB_WEBHOOK_SECRET is not configured"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 10_000_000:
            self._respond(400, {"error": "invalid request body"})
            return

        body = self.rfile.read(content_length)
        if not verify_signature(
            self.server.settings.github_webhook_secret,
            body,
            self.headers.get("X-Hub-Signature-256"),
        ):
            self._respond(401, {"error": "invalid signature"})
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._respond(400, {"error": "invalid JSON"})
            return
        if not isinstance(payload, dict):
            self._respond(400, {"error": "JSON object expected"})
            return

        event = self.headers.get("X-GitHub-Event", "")
        jobs = self.server.processor.submit_event(event, payload)
        self._respond(202, {"accepted": True, "jobs": jobs})

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    """Run the standalone GitHub webhook listener."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_project_environment()
    settings = Settings()
    if not settings.github_webhook_secret:
        raise SystemExit("Set GITHUB_WEBHOOK_SECRET before starting the webhook listener")

    server = _WebhookServer(
        (settings.webhook_host, settings.webhook_port),
        settings,
    )
    LOGGER.info(
        "BuildGuard webhook listener running on %s:%s%s",
        settings.webhook_host,
        settings.webhook_port,
        WEBHOOK_PATH,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping BuildGuard webhook listener")
    finally:
        server.shutdown()
        server.server_close()
        server.processor.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
