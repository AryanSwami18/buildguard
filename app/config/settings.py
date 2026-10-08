"""Environment-backed application settings."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Configuration for BuildGuard source ingestion."""

    ai_provider: str = field(
        default_factory=lambda: os.getenv("AI_PROVIDER", "ollama").strip().lower()
    )

    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        ).rstrip("/")
    )

    ollama_model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_MODEL",
            "qwen3.5:4b-q4_K_M",
        )
    )

    ollama_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
    )

    ollama_max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "16384"))
    )

    ollama_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_CONTEXT_TOKENS", "32768"))
    )

    github_token: str | None = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN")
    )

    github_repo: str | None = field(
        default_factory=lambda: os.getenv(
            "GITHUB_REPO",
            "AryanSwami18/docs",
        )
    )

    github_branch: str = field(
        default_factory=lambda: os.getenv(
            "GITHUB_BRANCH",
            "HEAD",
        )
    )

    scripts_repo: str = field(
        default_factory=lambda: os.getenv(
            "SCRIPTS_REPO",
            "AryanSwami18/scripts",
        )
    )

    scripts_branch: str = field(
        default_factory=lambda: os.getenv(
            "SCRIPTS_BRANCH",
            "master",
        )
    )

    buildguard_database_path: str = field(
        default_factory=lambda: os.getenv(
            "BUILDGUARD_DATABASE_PATH",
            ".buildguard/buildguard.sqlite3",
        )
    )

    github_webhook_secret: str | None = field(
        default_factory=lambda: os.getenv("GITHUB_WEBHOOK_SECRET")
    )

    webhook_host: str = field(
        default_factory=lambda: os.getenv("BUILDGUARD_WEBHOOK_HOST", "0.0.0.0")
    )

    webhook_port: int = field(
        default_factory=lambda: int(os.getenv("BUILDGUARD_WEBHOOK_PORT", "8080"))
    )

    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )

    gemini_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )
    )

    gemini_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))
    )

    gemini_max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768"))
    )

    # Legacy settings remain readable for callers using the old provider directly.
    openrouter_api_key: str | None = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_API_KEY"
        )
    )

    openrouter_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL",
            "qwen/qwen3-coder:free",
        )
    )

    openrouter_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90")
        )
    )

    openrouter_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("OPENROUTER_MAX_TOKENS", "32768")
        )
    )
