"""Small Markdown structure parser; semantic meaning is left to the LLM."""

import re

from app.sources.wiki import WikiBlock, WikiDocument, WikiHeading

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM = re.compile(r"^\s*(?:(\d+)[.)]|([-+*]))\s+.*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def parse_wiki(content: str, name: str = "wiki") -> WikiDocument:
    lines = content.splitlines()
    blocks: list[WikiBlock] = []
    headings: list[WikiHeading] = []
    section: str | None = None
    subsection: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        heading = _HEADING.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            headings.append(WikiHeading(level=level, text=title, line=index + 1))
            if level == 1:
                section, subsection = title, None
            else:
                subsection = title
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        start = index
        fence = _FENCE.match(line)
        if fence:
            fence_marker = fence.group(1)
            block_lines = [line]
            index += 1
            while index < len(lines):
                block_lines.append(lines[index])
                if lines[index].lstrip().startswith(fence_marker):
                    index += 1
                    break
                index += 1
            kind = "code_block"
        elif list_item := _LIST_ITEM.match(line):
            block_lines = [line]
            kind = "numbered_step" if list_item.group(1) else "bulleted_step"
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if (
                    not next_line.strip()
                    or _HEADING.match(next_line)
                    or _FENCE.match(next_line)
                    or _LIST_ITEM.match(next_line)
                ):
                    break
                block_lines.append(next_line)
                index += 1
        else:
            block_lines = []
            while index < len(lines) and lines[index].strip() and not _FENCE.match(lines[index]):
                if _HEADING.match(lines[index]) or _LIST_ITEM.match(lines[index]):
                    break
                block_lines.append(lines[index])
                index += 1
            kind = "paragraph"
        if block_lines:
            blocks.append(WikiBlock(kind=kind, text="\n".join(block_lines), section=section,
                subsection=subsection, line_start=start + 1, line_end=index))
    return WikiDocument(name=name, content=content, headings=headings, blocks=blocks)
