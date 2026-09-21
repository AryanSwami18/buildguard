"""Environment-backed application settings."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Configuration needed by source ingestion and future LLM adapters."""

    github_token: str | None = field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    github_repo: str | None = field(
        default_factory=lambda: os.getenv("GITHUB_REPO", "AryanSwami18/docs")
    )
    github_branch: str = field(default_factory=lambda: os.getenv("GITHUB_BRANCH", "HEAD"))
    openrouter_api_key: str | None = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free")
    )
