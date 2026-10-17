"""Small GitHub contents client used to retrieve build documentation and files."""

import base64
from datetime import UTC, datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.config.settings import Settings
from app.package_mapping import normalize_package_name
from app.sources.github import GitHubDirectoryEntry, GitHubFile, GitHubWikiPage

GITHUB_REQUEST_TIMEOUT_SECONDS = 30
WIKI_CLONE_TIMEOUT_SECONDS = 60
WIKI_REVISION_TIMEOUT_SECONDS = 30


class GitHubResourceNotFound(RuntimeError):
    """A requested GitHub file, directory, or Wiki page does not exist."""


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
            self._clone_wiki_repository(repository, checkout, ref)

            wiki_path = checkout / relative_path
            try:
                return wiki_path.read_text(encoding="utf-8")
            except FileNotFoundError as error:
                raise ValueError(f"GitHub Wiki page was not found: {path}") from error

    def list_wiki_modules(self) -> list[str]:
        """Return the package build pages available in the configured Wiki."""
        if not self.repository:
            raise ValueError("GITHUB_REPO is required to list Wiki modules")

        with TemporaryDirectory(prefix="buildguard-wiki-list-") as directory:
            checkout = Path(directory) / "wiki"
            self._clone_wiki_repository(
                self.repository,
                checkout,
                self._resolve_ref(None),
            )
            return sorted(
                (
                    path.stem
                    for path in checkout.iterdir()
                    if path.is_file()
                    and path.suffix.lower() == ".md"
                    and path.stem.casefold().startswith("building-")
                ),
                key=str.casefold,
            )

    def wiki_revision(self, ref: str | None = None) -> str:
        """Return the current remote Wiki revision without cloning its files."""
        if not self.repository:
            raise ValueError("GITHUB_REPO is required to check Wiki revisions")

        resolved_ref = self._resolve_ref(ref)
        remote_ref = "HEAD" if resolved_ref == "HEAD" else f"refs/heads/{resolved_ref}"
        command = [
            "git",
            "ls-remote",
            f"https://github.com/{self.repository}.wiki.git",
            remote_ref,
        ]
        environment = self._wiki_environment()
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                timeout=WIKI_REVISION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            # A public Wiki can still be readable when a configured token is
            # stale, scoped for the API only, or rejected by Git. Mirror the
            # clone fallback so cache verification does not fail merely
            # because authenticated Git access failed.
            if self.token and isinstance(error, subprocess.CalledProcessError):
                anonymous_environment = self._anonymous_wiki_environment(environment)
                try:
                    result = subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=anonymous_environment,
                        timeout=WIKI_REVISION_TIMEOUT_SECONDS,
                    )
                except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as retry_error:
                    error = retry_error
                else:
                    revision = result.stdout.strip().split(maxsplit=1)
                    if not revision or not revision[0]:
                        raise RuntimeError(
                            f"GitHub Wiki returned no revision for '{self.repository}'"
                        )
                    return revision[0]
            raise RuntimeError(
                f"Unable to check GitHub Wiki revision for '{self.repository}'"
            ) from error

        revision = result.stdout.strip().split(maxsplit=1)
        if not revision or not revision[0]:
            raise RuntimeError(
                f"GitHub Wiki returned no revision for '{self.repository}'"
            )
        return revision[0]

    def _clone_wiki_repository(
        self,
        repository: str,
        checkout: Path,
        ref: str,
    ) -> None:
        command = ["git", "clone", "--depth", "1"]
        if ref != "HEAD":
            command.extend(["--branch", ref])
        command.extend([f"https://github.com/{repository}.wiki.git", str(checkout)])

        environment = self._wiki_environment()
        try:
            self._run_wiki_clone(command, environment)
        except (OSError, subprocess.CalledProcessError) as error:
            # A stale token should not prevent access to a public Wiki. Retry
            # anonymously before reporting the clone failure.
            if self.token and isinstance(error, subprocess.CalledProcessError):
                anonymous_environment = self._anonymous_wiki_environment(environment)
                try:
                    self._run_wiki_clone(command, anonymous_environment)
                except (OSError, subprocess.CalledProcessError) as retry_error:
                    error = retry_error
                else:
                    return
            detail = getattr(error, "stderr", "") or str(error)
            raise RuntimeError(
                f"Unable to retrieve GitHub Wiki '{repository}': {detail.strip()}"
            ) from error

    @staticmethod
    def _anonymous_wiki_environment(environment: dict[str, str]) -> dict[str, str]:
        """Remove the Git auth override while preserving the process env."""
        anonymous_environment = environment.copy()
        for key in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
            anonymous_environment.pop(key, None)
        return anonymous_environment

    def _wiki_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        if self.token:
            environment.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.extraHeader",
                    "GIT_CONFIG_VALUE_0": f"Authorization: Bearer {self.token}",
                }
            )
        return environment

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
            sha=payload.get("sha") if isinstance(payload.get("sha"), str) else None,
        )

    def create_file_pull_request(
        self,
        *,
        repository: str,
        head_repository: str | None,
        base_branch: str,
        path: str,
        original_content: str,
        fixed_content: str,
        title: str,
        body: str,
        draft: bool = True,
        branch_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a branch, commit one file, and open a pull request.

        The current base file is re-read and compared with ``original_content``
        before any branch or commit is created. This prevents a stale report from
        overwriting a newer script revision.
        """
        if not self.token:
            raise ValueError("GITHUB_TOKEN is required to create a pull request")
        if base_branch == "HEAD":
            raise ValueError(
                "SCRIPTS_BRANCH must be a concrete branch name when creating a PR"
            )
        if fixed_content == original_content:
            raise ValueError("The proposed script has no changes")

        head_repository = head_repository or repository
        branch = branch_name or self._fix_branch_name(path)
        if branch_name:
            existing = self.find_open_pull_request(
                repository=repository,
                head_repository=head_repository,
                branch=branch,
                base_branch=base_branch,
            )
            if existing is not None:
                existing["head_repository"] = head_repository
                existing["branch"] = branch
                return existing

        base_file = self.fetch_file(repository, path, base_branch)
        if base_file.content != original_content:
            raise RuntimeError(
                "The build script changed on GitHub after this report was generated. "
                "Run the comparison again before creating a PR."
            )
        if not base_file.sha:
            raise RuntimeError("GitHub did not return the current file SHA")

        base_ref = self._get_json(
            self._git_ref_url(repository, base_branch)
        )
        try:
            base_sha = base_ref["object"]["sha"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("GitHub did not return the base branch commit SHA") from error

        self._request_json(
            "POST",
            self._git_refs_url(head_repository),
            {
                "ref": f"refs/heads/{branch}",
                "sha": base_sha,
            },
        )
        self._request_json(
            "PUT",
            self._contents_url(head_repository, path, branch),
            {
                "message": title,
                "content": base64.b64encode(fixed_content.encode("utf-8")).decode("ascii"),
                "branch": branch,
                "sha": base_file.sha,
            },
        )

        head = branch
        if head_repository != repository:
            head = f"{head_repository.split('/', 1)[0]}:{branch}"
        pull_request = self._request_json(
            "POST",
            self._pulls_url(repository),
            {
                "title": title,
                "head": head,
                "base": base_branch,
                "body": body,
                "draft": draft,
            },
        )
        if not isinstance(pull_request, dict):
            raise RuntimeError("GitHub returned an invalid pull request response")
        pull_request["head_repository"] = head_repository
        pull_request["branch"] = branch
        return pull_request

    def find_open_pull_request(
        self,
        *,
        repository: str,
        head_repository: str,
        branch: str,
        base_branch: str,
    ) -> dict[str, Any] | None:
        """Find an existing open PR for a deterministic automation branch."""
        head = f"{head_repository.split('/', 1)[0]}:{branch}"
        url = (
            f"{self._pulls_url(repository)}?state=open"
            f"&base={quote(base_branch, safe='')}"
            f"&head={quote(head, safe='')}"
        )
        payload = self._get_json(url)
        if not isinstance(payload, list):
            return None
        for item in payload:
            if isinstance(item, dict):
                return item
        return None

    def _get_json(self, url: str) -> Any:
        return self._request_json("GET", url)

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/vnd.github+json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(
                request,
                timeout=GITHUB_REQUEST_TIMEOUT_SECONDS,
            ) as response:  # noqa: S310 - URL is constructed from caller input.
                body = response.read().decode("utf-8")
        except TimeoutError as error:
            raise RuntimeError(
                f"GitHub request timed out after {GITHUB_REQUEST_TIMEOUT_SECONDS} seconds"
            ) from error
        except Exception as error:
            if getattr(error, "code", None) == 404:
                raise GitHubResourceNotFound(
                    f"GitHub resource was not found: {url}"
                ) from error
            detail = getattr(error, "read", lambda: b"")()
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            if detail:
                try:
                    detail = json.loads(detail).get("message", detail)
                except (TypeError, json.JSONDecodeError):
                    pass
            raise RuntimeError(
                f"GitHub API request failed for {method} {url}: {detail or error}"
            ) from error
        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError("GitHub returned invalid JSON") from error

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

    def _git_ref_url(self, repository: str, branch: str) -> str:
        repository_path = quote(repository, safe="/")
        encoded_branch = quote(branch, safe="")
        return f"{self.api_base_url}/repos/{repository_path}/git/ref/heads/{encoded_branch}"

    def _git_refs_url(self, repository: str) -> str:
        repository_path = quote(repository, safe="/")
        return f"{self.api_base_url}/repos/{repository_path}/git/refs"

    def _pulls_url(self, repository: str) -> str:
        repository_path = quote(repository, safe="/")
        return f"{self.api_base_url}/repos/{repository_path}/pulls"

    @staticmethod
    def _fix_branch_name(path: str) -> str:
        file_name = PurePosixPath(path).name.removesuffix(".sh")
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", file_name).strip("-")
        return f"buildguard/fix/{safe_name or 'build-script'}-{stamp}"

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
        script_filename: str | None = None,
    ) -> GitHubFile:
        """Find and fetch the build shell script for a product/version."""
        selected = self._select_build_script(
            repository,
            product,
            version,
            ref,
            script_filename,
        )
        resolved_ref = self._resolve_ref(ref)
        return self.fetch_file(repository, selected.path, resolved_ref)

    def find_build_script_reference(
        self,
        repository: str,
        product: str,
        version: str,
        ref: str | None = None,
        script_filename: str | None = None,
    ) -> tuple[str, str | None]:
        """Find a build script and return its path plus its GitHub blob SHA."""
        resolved_ref = self._resolve_ref(ref)
        selected = self._select_build_script(
            repository,
            product,
            version,
            ref,
            script_filename,
        )
        metadata = self._get_json(
            self._contents_url(repository, selected.path, resolved_ref)
        )
        remote_revision = (
            metadata.get("sha")
            if isinstance(metadata, dict) and isinstance(metadata.get("sha"), str)
            else None
        )
        return selected.path, remote_revision

    def _select_build_script(
        self,
        repository: str,
        product: str,
        version: str,
        ref: str | None,
        script_filename: str | None = None,
    ) -> GitHubDirectoryEntry:
        resolved_ref = self._resolve_ref(ref)
        product_path = product.strip().strip("/")
        if not product_path:
            raise ValueError("Script product name cannot be empty")

        version_path = f"{product_path}/{version}"
        try:
            entries = self.list_path(repository, version_path, resolved_ref)
        except GitHubResourceNotFound as original_error:
            product_path = self._find_normalized_product_path(
                repository,
                product_path,
                resolved_ref,
            )
            if product_path is None:
                raise original_error
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

        if script_filename:
            matching = [
                entry
                for entry in shell_files
                if entry.filename.casefold() == script_filename.casefold()
            ]
            if len(matching) == 1:
                return matching[0]
            names = ", ".join(entry.filename for entry in shell_files)
            raise ValueError(
                f"Script '{script_filename}' was not found in "
                f"{repository}/{version_path}; available scripts: {names}"
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
        return selected

    def _find_normalized_product_path(
        self,
        repository: str,
        product: str,
        ref: str,
    ) -> str | None:
        """Find a package directory differing only by case or separators."""
        entries = self.list_path(repository, "", ref)
        candidates = [
            entry.path
            for entry in entries
            if entry.kind == "directory"
            and normalize_package_name(entry.filename)
            == normalize_package_name(product)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None
