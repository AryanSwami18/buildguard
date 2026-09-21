from pydantic import BaseModel, Field


class WikiHeading(BaseModel):
    """A Markdown heading and its position in the source document."""

    level: int
    text: str
    line: int


class WikiBlock(BaseModel):
    kind: str
    text: str
    section: str | None = None
    subsection: str | None = None
    line_start: int
    line_end: int


class WikiDocument(BaseModel):
    name: str
    content: str
    headings: list[WikiHeading] = Field(default_factory=list)
    blocks: list[WikiBlock] = Field(default_factory=list)
