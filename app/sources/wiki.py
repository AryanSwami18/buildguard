from pydantic import BaseModel, Field


class WikiHeading(BaseModel):
    level: int
    text: str
    line: int



class WikiBlock(BaseModel):
    kind: str
    text: str

    # Raw Markdown hierarchy. These fields intentionally describe the
    # document exactly as authored.
    section: str | None = None
    subsection: str | None = None
    heading_path: list[str] = Field(default_factory=list)

    # Logical build hierarchy. This is separate from raw Markdown hierarchy
    # because the source can contain H1 headings inside a numbered section.
    logical_section: str | None = None
    logical_subsection: str | None = None
    logical_heading_path: list[str] = Field(default_factory=list)

    # Contextual headings/comments that qualify the current operation.
    context_labels: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)

    line_start: int
    line_end: int


class WikiDocument(BaseModel):
    name: str
    content: str

    headings: list[WikiHeading] = Field(default_factory=list)
    blocks: list[WikiBlock] = Field(default_factory=list)
