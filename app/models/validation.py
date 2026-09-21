from enum import Enum

from pydantic import BaseModel, Field

from .step import BuildStep


class MatchStatus(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"
    EXTRA = "extra"
    CONFLICT = "conflict"


class StepMatch(BaseModel):
    wiki_step: BuildStep | None = None
    automation_step: BuildStep | None = None

    status: MatchStatus

    explanation: str


class ValidationResult(BaseModel):
    matches: list[StepMatch] = Field(default_factory=list)

    valid: bool

    summary: str
