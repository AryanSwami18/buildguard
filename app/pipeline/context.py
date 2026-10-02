from app.models.context import SourceContext


def neighbouring_text(
    items: list[str],
    index: int,
    window: int = 1,
) -> str:
    start = max(0, index - window)
    end = min(len(items), index + window + 1)

    return "\n".join(
        item
        for item in items[start:end]
        if item
    )


def step_context(
    *,
    source_type: str,
    source_name: str,
    step_ids: list[str] | None = None,
    index: int | None = None,
    section: str | None = None,
    subsection: str | None = None,
    heading_path: list[str] | None = None,
    logical_section: str | None = None,
    logical_subsection: str | None = None,
    logical_heading_path: list[str] | None = None,
    context_labels: list[str] | None = None,
    conditions: list[str] | None = None,
    location: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
    surrounding_text: str | None = None,
    previous_step_id: str | None = None,
    next_step_id: str | None = None,
    related_step_ids: list[str] | None = None,
) -> SourceContext:

    if step_ids is not None and index is not None:
        if previous_step_id is None and index > 0:
            previous_step_id = step_ids[index - 1]

        if next_step_id is None and index + 1 < len(step_ids):
            next_step_id = step_ids[index + 1]

    if related_step_ids is None and step_ids is not None and index is not None:
        related_step_ids = (
            step_ids[max(0, index - 1):index]
            + step_ids[index + 1:index + 2]
        )

    return SourceContext(
        source_type=source_type,
        source_name=source_name,
        section=section,
        subsection=subsection,
        heading_path=heading_path or [],
        logical_section=logical_section,
        logical_subsection=logical_subsection,
        logical_heading_path=logical_heading_path or [],
        context_labels=context_labels or [],
        conditions=conditions or [],
        location=location,
        line_start=line_start,
        line_end=line_end,
        surrounding_text=surrounding_text,
        previous_step_id=previous_step_id,
        next_step_id=next_step_id,
        related_step_ids=related_step_ids or [],
    )

