"""Compatibility helpers for callers that need a JSON-ready Bash AST."""

from app.models.bash import BashScript


def normalize_bash(script: BashScript) -> dict[str, object]:
    """Return the structural representation sent to an LLM provider."""
    return script.model_dump(mode="json")
