from enum import Enum

from pydantic import BaseModel, Field

from .step import BuildStep


class MatchStatus(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"
    EXTRA = "extra"
    CONFLICT = "conflict"
    OUT_OF_SCOPE = "out_of_scope"


class StepMatch(BaseModel):
    wiki_step: BuildStep | None = None
    automation_step: BuildStep | None = None
    automation_steps: list[BuildStep] = Field(default_factory=list)

    status: MatchStatus

    explanation: str


class ValidationResult(BaseModel):
    matches: list[StepMatch] = Field(default_factory=list)

    valid: bool

    summary: str


class AIComparisonMatch(BaseModel):
    """A Wiki step and all Bash steps that implement it."""

    status: MatchStatus
    explanation: str
    wiki_step_id: str | None
    bash_step_id: str | None
    bash_step_ids: list[str] = Field(default_factory=list)


class AIComparisonResult(BaseModel):
    """Compact, provider-facing representation of a source comparison."""

    summary: str
    matches: list[AIComparisonMatch] = Field(default_factory=list)
