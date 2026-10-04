"""Small GitHub contents client used to retrieve build documentation and files."""

import base64
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.config.settings import Settings
from app.sources.github import GitHubDirectoryEntry, GitHubFile, GitHubWikiPage

GITHUB_REQUEST_TIMEOUT_SECONDS = 30
WIKI_CLONE_TIMEOUT_SECONDS = 60


class GitHubClient:
    """Retrieve GitHub sources without making assumptions about their contents."""

    def __init__(
        self,
        token: str | None = None,
        api_base_url: str = "https://api.github.com",
        repository: str | None = None,
        default_ref: str = "HEAD",
    ) -> None:
        self.token = token
        self.api_base_url = api_base_url.rstrip("/")
        self.repository = repository
        self.default_ref = default_ref

    @classmethod
    def from_settings(cls, settings: Settings) -> "GitHubClient":
        """Create a client using the GitHub values from application settings."""
        return cls(
            token=settings.github_token,
            repository=settings.github_repo,
            default_ref=settings.github_branch,
        )

    def fetch_module_wiki(self, module_name: str) -> GitHubWikiPage:
        """Resolve a user-provided module name to a page in the configured Wiki."""
        if not self.repository:
            raise ValueError("GITHUB_REPO is required to fetch a module Wiki page")
        return self.fetch_wiki_markdown(self.repository, module_name, self.default_ref)

    def fetch_wiki_markdown(
        self, repository: str, page: str, ref: str | None = None
    ) -> GitHubWikiPage:
        """Fetch one Markdown page from the Git repository backing a GitHub Wiki.

        GitHub's Contents API does not expose ``<repository>.wiki`` repositories,
        so using it produces a 404 even for a public Wiki. Wikis are Git
        repositories, however, and can be retrieved through their Git endpoint.
        """
        resolved_ref = self._resolve_ref(ref)
        path = page if page.endswith(".md") else f"{page}.md"
        content = self._fetch_wiki_content(repository, path, resolved_ref)
        page_name = path.removesuffix(".md")
        return GitHubWikiPage(
            repository=repository,
            ref=resolved_ref,
            path=path,
            filename=PurePosixPath(path).name,
            github_url=f"https://github.com/{repository}/wiki/{quote(page_name)}",
            content=content,
        )

    def _fetch_wiki_content(self, repository: str, path: str, ref: str) -> str:
        """Clone a Wiki shallowly and return one Markdown file from it."""
        relative_path = PurePosixPath(path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Wiki page paths must stay within the Wiki repository")

        with TemporaryDirectory(prefix="buildguard-wiki-") as directory:
            checkout = Path(directory) / "wiki"
            command = ["git", "clone", "--depth", "1"]
            if ref != "HEAD":
                command.extend(["--branch", ref])
            command.extend([f"https://github.com/{repository}.wiki.git", str(checkout)])

            environment = os.environ.copy()
            if self.token:
                environment.update(
                    {
                        "GIT_CONFIG_COUNT": "1",
                        "GIT_CONFIG_KEY_0": "http.extraHeader",
                        "GIT_CONFIG_VALUE_0": f"Authorization: Bearer {self.token}",
                    }
                )
            try:
                self._run_wiki_clone(command, environment)
            except (OSError, subprocess.CalledProcessError) as error:
                # A stale token should not prevent access to a public Wiki. Retry
                # anonymously before reporting the clone failure.
                if self.token and isinstance(error, subprocess.CalledProcessError):
                    anonymous_environment = environment.copy()
                    for key in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
                        anonymous_environment.pop(key, None)
                    try:
                        self._run_wiki_clone(command, anonymous_environment)
                    except (OSError, subprocess.CalledProcessError) as retry_error:
                        error = retry_error
                    else:
                        error = None
                if error is None:
                    pass
                else:
                    detail = getattr(error, "stderr", "") or str(error)
                    raise RuntimeError(f"Unable to retrieve GitHub Wiki '{repository}': {detail.strip()}") from error

            wiki_path = checkout / relative_path
            try:
                return wiki_path.read_text(encoding="utf-8")
            except FileNotFoundError as error:
                raise ValueError(f"GitHub Wiki page was not found: {path}") from error

    def list_path(
        self, repository: str, path: str = "", ref: str | None = None
    ) -> list[GitHubDirectoryEntry]:
        """List files and directories at a repository path for a given ref."""
        resolved_ref = self._resolve_ref(ref)
        payload = self._get_json(self._contents_url(repository, path, resolved_ref))
        if not isinstance(payload, list):
            raise ValueError(f"GitHub path is not a directory: {path}")
        return [self._directory_entry(repository, resolved_ref, item) for item in payload]

    def fetch_file(self, repository: str, path: str, ref: str | None = None) -> GitHubFile:
        """Fetch text content for one repository file at a given ref."""
        resolved_ref = self._resolve_ref(ref)
        payload = self._get_json(self._contents_url(repository, path, resolved_ref))
        if not isinstance(payload, dict) or payload.get("type") != "file":
            raise ValueError(f"GitHub path is not a file: {path}")

        content = payload.get("content")
        if not isinstance(content, str) or payload.get("encoding") != "base64":
            raise ValueError(f"GitHub did not return base64 file content for: {path}")

        return GitHubFile(
            repository=repository,
            ref=resolved_ref,
            path=payload.get("path", path),
            filename=payload.get("name", PurePosixPath(path).name),
            github_url=payload.get("html_url") or self._file_url(repository, resolved_ref, path),
            content=base64.b64decode(content).decode("utf-8"),
        )

    def _get_json(self, url: str) -> Any:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(url, headers=headers)
        try:
            with urlopen(
                request,
                timeout=GITHUB_REQUEST_TIMEOUT_SECONDS,
            ) as response:  # noqa: S310 - URL is constructed from caller input.
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as error:
            raise RuntimeError(
                f"GitHub request timed out after {GITHUB_REQUEST_TIMEOUT_SECONDS} seconds"
            ) from error

    @staticmethod
    def _run_wiki_clone(command: list[str], environment: dict[str, str]) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                timeout=WIKI_CLONE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"GitHub Wiki clone timed out after {WIKI_CLONE_TIMEOUT_SECONDS} seconds"
            ) from error

    def _contents_url(self, repository: str, path: str, ref: str) -> str:
        repository_path = quote(repository, safe="/")
        encoded_path = quote(path.strip("/"), safe="/")
        suffix = f"/{encoded_path}" if encoded_path else ""
        # `HEAD` is useful provenance for callers, but it is not a branch name
        # accepted by the Contents API for GitHub Wiki repositories.  Omitting
        # `ref` asks GitHub for that repository's default branch (normally
        # `master` for a Wiki).
        ref_query = "" if ref == "HEAD" else f"?ref={quote(ref, safe='')}"
        return f"{self.api_base_url}/repos/{repository_path}/contents{suffix}{ref_query}"

    def _resolve_ref(self, ref: str | None) -> str:
        return ref or self.default_ref

    @staticmethod
    def _file_url(repository: str, ref: str, path: str) -> str:
        return f"https://github.com/{repository}/blob/{quote(ref, safe='')}/{quote(path, safe='/')}"

    @staticmethod
    def _directory_entry(repository: str, ref: str, payload: dict[str, Any]) -> GitHubDirectoryEntry:
        entry_type = payload.get("type")
        if entry_type not in {"file", "dir"}:
            raise ValueError(f"Unsupported GitHub directory entry type: {entry_type}")
        path = payload["path"]
        return GitHubDirectoryEntry(
            repository=repository,
            ref=ref,
            path=path,
            filename=payload.get("name", PurePosixPath(path).name),
            github_url=payload.get("html_url") or (
                GitHubClient._file_url(repository, ref, path)
                if entry_type == "file"
                else f"https://github.com/{repository}/tree/{quote(ref, safe='')}/{quote(path, safe='/')}"
            ),
            kind="directory" if entry_type == "dir" else "file",
        )

    def fetch_build_script(
        self,
        repository: str,
        product: str,
        version: str,
        ref: str | None = None,
    ) -> GitHubFile:
        """Find and fetch the build shell script for a product/version."""
        resolved_ref = self._resolve_ref(ref)
        product_path = product.strip().strip("/")
        if not product_path:
            raise ValueError("Script product name cannot be empty")

        version_path = f"{product_path}/{version}"
        entries = self.list_path(repository, version_path, resolved_ref)
        shell_files = [
            entry
            for entry in entries
            if entry.kind == "file" and entry.filename.lower().endswith(".sh")
        ]

        if not shell_files:
            raise ValueError(
                f"No shell script found in {repository}/{version_path}"
            )

        expected_name = f"build_{PurePosixPath(product_path).name.lower()}.sh"
        matching = [
            entry for entry in shell_files
            if entry.filename.lower() == expected_name
        ]
        if len(matching) == 1:
            selected = matching[0]
        elif len(shell_files) == 1:
            selected = shell_files[0]
        else:
            names = ", ".join(entry.filename for entry in shell_files)
            raise ValueError(
                f"Multiple shell scripts found in {repository}/{version_path}: {names}"
            )

        return self.fetch_file(repository, selected.path, resolved_ref)
