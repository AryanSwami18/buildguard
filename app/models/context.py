from pydantic import BaseModel, Field


class SourceContext(BaseModel):
    source_type: str
    source_name: str

    section: str | None = None
    subsection: str | None = None

    # Full Markdown heading hierarchy.
    # Example:
    # ["Build Alfresco", "3. Build component docker images", "3.1 base-java"]
    heading_path: list[str] = Field(default_factory=list)

    location: str | None = None

    line_start: int | None = None
    line_end: int | None = None

    surrounding_text: str | None = None

    # Exact source block used to validate a Wiki interpretation.
    source_excerpt: str | None = None

    # Explicit source qualifiers such as "Only if building for RHEL".
    conditions: list[str] = Field(default_factory=list)

    # Script-level options/defaults relevant to parameterized operations.
    runtime_options: dict[str, list[str]] = Field(default_factory=dict)
    runtime_defaults: dict[str, str] = Field(default_factory=dict)

    previous_step_id: str | None = None
    next_step_id: str | None = None

    related_step_ids: list[str] = Field(default_factory=list)
