"""Structured, mismatch-only comparison report models."""

from pydantic import BaseModel, Field

from .step import ActionType
from .validation import DiscrepancyCategory, MatchStatus


class ReportSource(BaseModel):
    name: str
    repository: str
    ref: str
    path: str
    url: str


class ComparisonCounts(BaseModel):
    wiki_steps_compared: int
    script_steps_checked: int
    missing_steps: int
    conflicting_steps: int
    undocumented_script_steps: int
    out_of_scope_steps: int
    findings: int
    by_category: dict[str, int] = Field(default_factory=dict)


class StepEvidence(BaseModel):
    id: str
    description: str
    action: ActionType
    location: str
    component: str | None = None
    commands: list[str] = Field(default_factory=list)
    packages: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    repository: str | None = None
    ref: str | None = None
    commit: str | None = None
    version: str | None = None
    docker_image: str | None = None
    dockerfile: str | None = None
    build_args: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    source_excerpt: str | None = None


class Discrepancy(BaseModel):
    id: str
    status: MatchStatus
    category: DiscrepancyCategory
    title: str
    explanation: str
    recommendation: str
    wiki: StepEvidence | None = None
    script: list[StepEvidence] = Field(default_factory=list)


class ComparisonReport(BaseModel):
    schema_version: str = "1.0"
    package: str
    version: str | None = None
    status: str
    overview: str
    counts: ComparisonCounts
    wiki_source: ReportSource | None = None
    script_source: ReportSource | None = None
    discrepancies: list[Discrepancy] = Field(default_factory=list)
