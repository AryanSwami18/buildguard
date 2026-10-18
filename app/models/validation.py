from enum import Enum

from pydantic import BaseModel, Field

from .step import BuildStep


class MatchStatus(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"
    EXTRA = "extra"
    CONFLICT = "conflict"
    OUT_OF_SCOPE = "out_of_scope"
    IMPLEMENTATION_DETAIL = "implementation_detail"
    UNUSED_CONFIGURATION = "unused_configuration"


class DiscrepancyCategory(str, Enum):
    """Kinds of actionable differences in a Wiki/script comparison."""

    MISSING_STEP = "missing_step"
    UNDOCUMENTED_STEP = "undocumented_step"
    PARAMETER_VALUE_MISMATCH = "parameter_value_mismatch"
    VERSION_INCONSISTENCY = "version_inconsistency"
    DEPENDENCY_INCONSISTENCY = "dependency_inconsistency"
    COMMAND_MISMATCH = "command_mismatch"
    CONFIGURATION_MISMATCH = "configuration_mismatch"
    UNUSED_CONFIGURATION = "unused_configuration"
    OTHER_DEVIATION = "other_deviation"


class StepMatch(BaseModel):
    wiki_step: BuildStep | None = None
    automation_step: BuildStep | None = None
    automation_steps: list[BuildStep] = Field(default_factory=list)

    status: MatchStatus

    category: DiscrepancyCategory | None = None

    recommendation: str | None = None

    explanation: str


class ValidationResult(BaseModel):
    matches: list[StepMatch] = Field(default_factory=list)

    valid: bool

    summary: str


class AIValueDifference(BaseModel):
    """Source values the AI says materially differ for one matched action."""

    field: str
    wiki_value: str
    bash_step_id: str
    bash_value: str


class AIComparisonMatch(BaseModel):
    """A Wiki step and all Bash steps that implement it."""

    status: MatchStatus
    explanation: str
    wiki_step_id: str | None
    bash_step_id: str | None
    bash_step_ids: list[str] = Field(default_factory=list)
    category: DiscrepancyCategory | None = None
    recommendation: str | None = None
    value_differences: list[AIValueDifference] = Field(default_factory=list)


class AIComparisonResult(BaseModel):
    """Compact, provider-facing representation of a source comparison."""

    summary: str
    matches: list[AIComparisonMatch] = Field(default_factory=list)
