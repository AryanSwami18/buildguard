import re

from app.sources.wiki import WikiBlock, WikiDocument, WikiHeading


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_ITEM = re.compile(r"^\s*\d+[.)]\s+.*$")
_BULLET_ITEM = re.compile(r"^\s*[-+*]\s+.*$")
_CODE_FENCE = re.compile(r"^ {0,3}```")
_INDENTED_CODE = re.compile(r"^(?: {4}|\t)")

# Numbered Build Instruction headings such as:
#   3. Build component docker images
#   3.1 base-java
#   5. Start Alfresco service
#
# We deliberately require a leading numeric label. Ordinary H1/H2 headings
# such as "Execute the Build Command" therefore remain contextual headings.
_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+?)\s*$"
)

# Explicit conditions commonly found in Build Instructions. This is kept
# intentionally conservative so we do not turn arbitrary prose into a
# condition.
_CONDITION_HEADING = re.compile(
    r"^(?:only\s+if|if|when|unless|for)\b",
    re.IGNORECASE,
)

_CONDITION_COMMENT = re.compile(
    r"^\s*#\s*(?P<condition>"
    r"(?:only\s+if|if|when|unless|for)\b.+?)\s*$",
    re.IGNORECASE,
)


def _heading_path(headings: list[WikiHeading], line: int) -> list[str]:
    path: list[str] = []

    for heading in headings:
        if heading.line > line:
            break

        while len(path) >= heading.level:
            path.pop()

        path.append(heading.text)

    return path


def _current_section(
    headings: list[WikiHeading],
    line: int,
) -> tuple[str | None, str | None]:
    path = _heading_path(headings, line)

    if not path:
        return None, None

    section = path[0]
    subsection = path[1] if len(path) > 1 else None

    return section, subsection


def _numbered_headings_before(
    headings: list[WikiHeading],
    line: int,
) -> list[WikiHeading]:
    return [
        heading
        for heading in headings
        if heading.line <= line
        and _NUMBERED_HEADING.match(heading.text)
    ]


def _logical_heading_context(
    headings: list[WikiHeading],
    line: int,
) -> tuple[
    str | None,
    str | None,
    list[str],
    list[str],
]:
    """
    Return logical numbered hierarchy plus contextual headings.

    The raw Markdown hierarchy is not used here. A contextual H1 such as
    "Execute the Build Command" must not erase the logical 3.1 component that
    it belongs to.
    """
    numbered = _numbered_headings_before(headings, line)

    if not numbered:
        return None, None, [], []

    # The nearest numbered heading is the most specific logical location.
    current = numbered[-1]
    match = _NUMBERED_HEADING.match(current.text)
    assert match is not None

    number = match.group("number")
    title = match.group("title").strip()

    parts = number.split(".")
    logical_section = f"{parts[0]}. {title}" if len(parts) == 1 else None

    logical_subsection: str | None = None

    if len(parts) >= 2:
        # The nearest parent section is useful even when the current source
        # heading itself is deeply nested.
        parent = next(
            (
                heading
                for heading in reversed(numbered[:-1])
                if _NUMBERED_HEADING.match(heading.text)
                and _NUMBERED_HEADING.match(heading.text).group("number")
                == parts[0]
            ),
            None,
        )
        logical_section = parent.text.strip() if parent else None
        logical_subsection = current.text.strip()

    logical_path: list[str] = []
    for heading in numbered:
        heading_match = _NUMBERED_HEADING.match(heading.text)
        if not heading_match:
            continue

        heading_number = heading_match.group("number")
        heading_parts = heading_number.split(".")

        if len(heading_parts) == 1:
            logical_path = [heading.text.strip()]
        elif len(heading_parts) == 2:
            if logical_path:
                logical_path = [logical_path[0], heading.text.strip()]
            else:
                logical_path = [heading.text.strip()]
        else:
            logical_path.append(heading.text.strip())

    # Context headings belong only to the current numbered section. Once a
    # new numbered heading starts, old contextual labels are discarded.
    current_number = number
    current_number_prefix = current_number.split(".")[0]

    current_index = next(
        (
            index
            for index in range(len(headings) - 1, -1, -1)
            if headings[index] is current
        ),
        -1,
    )

    context_labels: list[str] = []
    if current_index >= 0:
        for heading in headings[current_index + 1:]:
            if heading.line > line:
                break
            if _NUMBERED_HEADING.match(heading.text):
                continue
            context_labels.append(heading.text.strip())

    # If there was no contextual heading after the current numbered heading,
    # return an empty list. This also prevents labels from previous sections
    # leaking into later steps.
    del current_number_prefix

    return logical_section, logical_subsection, logical_path, context_labels


def _extract_conditions(
    text: str,
    context_labels: list[str],
) -> list[str]:
    conditions: list[str] = []

    for label in context_labels:
        if _CONDITION_HEADING.match(label):
            if label not in conditions:
                conditions.append(label)

    for line in text.splitlines():
        match = _CONDITION_COMMENT.match(line)
        if match:
            condition = match.group("condition").strip()
            if condition not in conditions:
                conditions.append(condition)

    return conditions


def _make_block(
    *,
    kind: str,
    lines: list[str],
    start_line: int,
    end_line: int,
    headings: list[WikiHeading],
) -> WikiBlock | None:
    text = "\n".join(lines).strip()

    if not text:
        return None

    section, subsection = _current_section(headings, start_line)
    (
        logical_section,
        logical_subsection,
        logical_heading_path,
        context_labels,
    ) = _logical_heading_context(headings, start_line)

    return WikiBlock(
        kind=kind,
        text=text,
        section=section,
        subsection=subsection,
        heading_path=_heading_path(headings, start_line),
        logical_section=logical_section,
        logical_subsection=logical_subsection,
        logical_heading_path=logical_heading_path,
        context_labels=context_labels,
        conditions=_extract_conditions(text, context_labels),
        line_start=start_line,
        line_end=end_line,
    )


def parse_wiki(markdown: str, name: str) -> WikiDocument:
    lines = markdown.splitlines()

    headings: list[WikiHeading] = []

    for index, line in enumerate(lines, start=1):
        match = _HEADING.match(line)

        if not match:
            continue

        text = match.group(2).strip()

        headings.append(
            WikiHeading(
                level=len(match.group(1)),
                text=text,
                line=index,
            )
        )

    blocks: list[WikiBlock] = []

    paragraph_lines: list[str] = []
    paragraph_start: int | None = None

    in_code_block = False
    code_lines: list[str] = []
    code_start: int | None = None
    in_indented_code = False
    indented_code_lines: list[str] = []
    indented_code_start: int | None = None
    indented_code_end: int | None = None

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph_lines, paragraph_start

        if not paragraph_lines or paragraph_start is None:
            paragraph_lines = []
            paragraph_start = None
            return

        block = _make_block(
            kind="paragraph",
            lines=paragraph_lines,
            start_line=paragraph_start,
            end_line=end_line,
            headings=headings,
        )

        if block is not None:
            blocks.append(block)

        paragraph_lines = []
        paragraph_start = None

    def add_list_item(kind: str, line: str, line_number: int) -> None:
        block = _make_block(
            kind=kind,
            lines=[line],
            start_line=line_number,
            end_line=line_number,
            headings=headings,
        )

        if block is not None:
            blocks.append(block)

    def flush_indented_code() -> None:
        nonlocal in_indented_code
        nonlocal indented_code_lines
        nonlocal indented_code_start
        nonlocal indented_code_end

        if indented_code_start is not None and indented_code_end is not None:
            block = _make_block(
                kind="code_block",
                lines=indented_code_lines,
                start_line=indented_code_start,
                end_line=indented_code_end,
                headings=headings,
            )

            if block is not None:
                blocks.append(block)

        in_indented_code = False
        indented_code_lines = []
        indented_code_start = None
        indented_code_end = None

    for index, line in enumerate(lines, start=1):
        if in_indented_code:
            if _INDENTED_CODE.match(line):
                indented_code_lines.append(line)
                indented_code_end = index
                continue

            if not line.strip():
                next_line_is_indented = (
                    index < len(lines)
                    and _INDENTED_CODE.match(lines[index]) is not None
                )
                if next_line_is_indented:
                    indented_code_lines.append(line)
                    continue

                flush_indented_code()
                continue

            flush_indented_code()

        if _CODE_FENCE.match(line):
            if not in_code_block:
                flush_paragraph(index - 1)
                in_code_block = True
                code_start = index
                code_lines = [line]
            else:
                code_lines.append(line)

                block = _make_block(
                    kind="code_block",
                    lines=code_lines,
                    start_line=code_start or index,
                    end_line=index,
                    headings=headings,
                )

                if block is not None:
                    blocks.append(block)

                in_code_block = False
                code_lines = []
                code_start = None

            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if _INDENTED_CODE.match(line):
            flush_paragraph(index - 1)
            in_indented_code = True
            indented_code_start = index
            indented_code_end = index
            indented_code_lines = [line]
            continue

        heading_match = _HEADING.match(line)

        if heading_match:
            flush_paragraph(index - 1)
            continue

        if not line.strip():
            flush_paragraph(index - 1)
            continue

        if _NUMBERED_ITEM.match(line):
            flush_paragraph(index - 1)
            add_list_item("numbered_step", line, index)
            continue

        if _BULLET_ITEM.match(line):
            flush_paragraph(index - 1)
            add_list_item("bulleted_step", line, index)
            continue

        if paragraph_start is None:
            paragraph_start = index

        paragraph_lines.append(line)

    if in_code_block and code_start is not None:
        block = _make_block(
            kind="code_block",
            lines=code_lines,
            start_line=code_start,
            end_line=len(lines),
            headings=headings,
        )

        if block is not None:
            blocks.append(block)
    elif in_indented_code:
        flush_indented_code()
    else:
        flush_paragraph(len(lines))

    return WikiDocument(
        name=name,
        content=markdown,
        headings=headings,
        blocks=blocks,
    )
