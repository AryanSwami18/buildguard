"""Structured AI output for a proposed build-script correction."""

from pydantic import BaseModel, Field


class ScriptFixProposal(BaseModel):
    """A complete, reviewable replacement for one build script."""

    fixed_script: str
    summary: str
    rationale: str
    affected_finding_ids: list[str] = Field(default_factory=list)
