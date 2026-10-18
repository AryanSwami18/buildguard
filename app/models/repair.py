"""Structured AI output for a proposed build-script correction."""

from pydantic import BaseModel, Field


class ScriptFixEdit(BaseModel):
    """One source-linked replacement in the original Bash script."""

    finding_id: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    original_text: str
    replacement: str


class ScriptFixPlan(BaseModel):
    """Small edit plan returned by the model before local application."""

    edits: list[ScriptFixEdit]
    summary: str
    rationale: str
    affected_finding_ids: list[str]


class ScriptFixProposal(BaseModel):
    """A complete, reviewable replacement for one build script."""

    fixed_script: str
    summary: str
    rationale: str
    affected_finding_ids: list[str] = Field(default_factory=list)
