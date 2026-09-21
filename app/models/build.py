from pydantic import BaseModel, Field

from .step import BuildStep
from app.sources.github import GitHubSource


class BuildInstruction(BaseModel):
    """An ordered build procedure from one documentation or automation source."""

    name: str
    source: str
    version: str | None = None
    source_metadata: GitHubSource | None = None

    steps: list[BuildStep] = Field(default_factory=list)
