from app.models.context import SourceContext


def neighbouring_text(items: list[str], index: int) -> str | None:
    """Return a compact view of the immediately adjacent source items."""
    neighbours = items[max(0, index - 1):index] + items[index + 1:index + 2]
    return "\n".join(neighbours) or None


def step_context(
    *,
    source_type: str,
    source_name: str,
    step_ids: list[str],
    index: int,
    location: str,
    line_start: int | None = None,
    line_end: int | None = None,
    section: str | None = None,
    subsection: str | None = None,
    surrounding_text: str | None = None,
) -> SourceContext:
    return SourceContext(
        source_type=source_type,
        source_name=source_name,
        section=section,
        subsection=subsection,
        location=location,
        line_start=line_start,
        line_end=line_end,
        surrounding_text=surrounding_text,
        previous_step_id=step_ids[index - 1] if index else None,
        next_step_id=step_ids[index + 1] if index + 1 < len(step_ids) else None,
        related_step_ids=[step_id for step_id in (step_ids[index - 1:index] + step_ids[index + 1:index + 2])],
    )
