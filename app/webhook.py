"""Small GitHub webhook listener for proactive cache synchronization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from threading import Lock
from typing import Any

from app.config.settings import Settings
from app.parsers.version_parser import product_from_module
from app.service import synchronize_package_cache

LOGGER = logging.getLogger("buildguard.webhook")
WEBHOOK_PATH = "/github/webhook"


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
    """Turn GitHub events into deduplicated background cache refreshes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = Lock()
        self.pending: dict[str, tuple[bool, bool]] = {}

    def submit_event(self, event: str, payload: dict[str, Any]) -> int:
        package_jobs: dict[str, tuple[bool, bool]] = {}
        for package, refresh_wiki, refresh_bash in _cache_jobs(
            event,
            payload,
            self.settings,
        ):
            existing = package_jobs.get(package, (False, False))
            package_jobs[package] = (
                existing[0] or refresh_wiki,
                existing[1] or refresh_bash,
            )

        for package, (refresh_wiki, refresh_bash) in package_jobs.items():
            self._enqueue(package, refresh_wiki, refresh_bash)
        return len(package_jobs)

    def _enqueue(
        self,
        package: str,
        refresh_wiki: bool,
        refresh_bash: bool,
    ) -> None:
        with self.lock:
            existing = self.pending.get(package)
            if existing is not None:
                self.pending[package] = (
                    existing[0] or refresh_wiki,
                    existing[1] or refresh_bash,
                )
                return
            self.pending[package] = (refresh_wiki, refresh_bash)

        self.executor.submit(self._refresh, package)

    def _refresh(self, package: str) -> None:
        try:
            with self.lock:
                refresh_wiki, refresh_bash = self.pending.get(package, (False, False))
            synchronize_package_cache(
                package,
                self.settings,
                refresh_wiki=refresh_wiki,
                refresh_bash=refresh_bash,
            )
        except Exception:
            LOGGER.exception("Unable to refresh BuildGuard cache for %s", package)
        finally:
            with self.lock:
                self.pending.pop(package, None)


def _cache_jobs(
    event: str,
    payload: dict[str, Any],
    settings: Settings,
) -> set[tuple[str, bool, bool]]:
    repository = payload.get("repository") or {}
    repository_name = repository.get("full_name")
    if not isinstance(repository_name, str):
        return set()

    if event == "gollum" and repository_name == settings.github_repo:
        jobs: set[tuple[str, bool, bool]] = set()
        for page in payload.get("pages") or []:
            if not isinstance(page, dict):
                continue
            page_name = page.get("page_name")
            if not isinstance(page_name, str):
                continue
            if page_name.casefold().startswith("building-"):
                jobs.add((product_from_module(page_name), True, False))
        return jobs

    if event != "push":
        return set()

    changed_paths = _changed_paths(payload)
    if repository_name == settings.github_repo:
        return {
            (product_from_module(path.name), True, False)
            for path in map(PurePosixPath, changed_paths)
            if path.suffix.casefold() == ".md"
            and path.name.casefold().startswith("building-")
        }

    if repository_name == settings.scripts_repo:
        jobs = set()
        for raw_path in changed_paths:
            path = PurePosixPath(raw_path)
            parts = path.parts
            if (
                len(parts) >= 3
                and parts[1]
                and path.suffix.casefold() == ".sh"
            ):
                jobs.add((parts[0], False, True))
        return jobs

    return set()


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
