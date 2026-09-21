from pydantic import BaseModel, Field


class SourceContext(BaseModel):
    """Where a step came from and the nearby information needed to read it."""

    source_type: str
    source_name: str
    section: str | None = None
    subsection: str | None = None
    location: str | None = None
    line_start: int | None = None
    line_end: int | None = None

    surrounding_text: str | None = None
    previous_step_id: str | None = None
    next_step_id: str | None = None
    related_step_ids: list[str] = Field(default_factory=list)
