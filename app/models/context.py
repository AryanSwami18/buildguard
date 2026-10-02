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

    previous_step_id: str | None = None
    next_step_id: str | None = None

    related_step_ids: list[str] = Field(default_factory=list)

