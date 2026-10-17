"""Source references returned by GitHub retrieval operations."""

from typing import Literal

from pydantic import BaseModel


class GitHubSource(BaseModel):
    """Location metadata needed to trace content back to GitHub."""

    repository: str
    ref: str
    path: str
    filename: str
    github_url: str


class GitHubDirectoryEntry(GitHubSource):
    """A file or directory listed by GitHub's contents API."""

    kind: Literal["file", "directory"]


class GitHubFile(GitHubSource):
    """Text content and its GitHub location metadata."""

    content: str
    sha: str | None = None


class GitHubWikiPage(GitHubFile):
    """Markdown content fetched from a repository's GitHub Wiki."""
