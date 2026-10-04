"""Bash syntax parsing using bashlex.

The parser keeps the Bash structure intact and intentionally does not
interpret individual commands such as package managers or download tools.
"""

import re

import bashlex

from app.models.bash import BashNode, BashScript


class BashParser:
    def parse(self, script: str) -> BashScript:
        """Parse a script into a small, JSON-friendly view of the bashlex AST."""
        parser_source, conditional_tests = _mask_double_bracket_tests(script)
        parser_source = _mask_arithmetic_expansions(parser_source)
        parser_source = _mask_case_dispatch(parser_source)
        return BashScript(
            source=script,
            nodes=[
                self._convert(node, script, conditional_tests)
                for node in bashlex.parse(parser_source)
            ],
        )

    def build_tree(self, script: str) -> list[BashNode]:
        """Return root nodes for callers that only need the AST tree."""
        return self.parse(script).nodes

    def _convert(
        self,
        node: object,
        script: str,
        conditional_tests: set[int],
    ) -> BashNode:
        position_start, position_end = node.pos
        children = [
            self._convert(child, script, conditional_tests)
            for child in self._child_nodes(node)
        ]
        words = [
            part.word
            for part in getattr(node, "parts", [])
            if part.kind == "word"
        ]
        kind = node.kind
        if kind == "command" and position_start in conditional_tests:
            # bashlex does not accept Bash's [[ ... ]] test syntax when its
            # operands contain quotes. The shadow source uses a same-length
            # ':' placeholder, while consumers see the original text/offsets.
            kind = "conditional_test"
        return BashNode(
            kind=kind,
            position_start=position_start,
            position_end=position_end,
            text=script[position_start:position_end],
            operator=getattr(node, "op", None) or getattr(node, "pipe", None),
            words=words,
            children=children,
        )

    @staticmethod
    def _child_nodes(node: object) -> list[object]:
        """Read bashlex child containers without assigning them semantics."""
        children: list[object] = []
        for attribute in ("parts", "list", "command", "output", "heredoc"):
            value = getattr(node, attribute, None)
            if isinstance(value, list):
                children.extend(item for item in value if hasattr(item, "kind"))
            elif hasattr(value, "kind"):
                children.append(value)
        return children


_CONTROL_CONDITION = re.compile(r"^\s*(?:if|elif|while|until)\b")
_DOUBLE_BRACKET_TEST = re.compile(r"\[\[.*?\]\]")
_ARITHMETIC_EXPANSION = re.compile(r"\$\(\(.*?\)\)")
_CASE_START = re.compile(r"^\s*case\b")


def _mask_double_bracket_tests(source: str) -> tuple[str, set[int]]:
    """Hide [[ tests from bashlex without shifting original source positions."""
    lines = source.splitlines(keepends=True)
    parser_lines: list[str] = []
    conditional_tests: set[int] = set()
    offset = 0

    for line in lines:
        parser_line = line
        if _CONTROL_CONDITION.match(line):
            for match in reversed(list(_DOUBLE_BRACKET_TEST.finditer(line))):
                start, end = match.span()
                conditional_tests.add(offset + start)
                original = parser_line[start:end]
                placeholder = ":" + "".join(
                    "\n" if character == "\n" else " "
                    for character in original[1:]
                )
                parser_line = parser_line[:start] + placeholder + parser_line[end:]
        parser_lines.append(parser_line)
        offset += len(line)

    return "".join(parser_lines), conditional_tests


def _mask_arithmetic_expansions(source: str) -> str:
    """Replace arithmetic expansions with a same-length assignment literal."""
    return _ARITHMETIC_EXPANSION.sub(
        lambda match: "0" + " " * (len(match.group()) - 1),
        source,
    )


def _mask_case_dispatch(source: str) -> str:
    """Remove case patterns and separators while leaving branch commands parseable.

    bashlex raises NotImplementedError for Bash's case-pattern grammar. Build
    analysis needs the commands in each arm, but not the dispatch pattern, so
    turn each pattern into whitespace and each arm terminator into a command
    separator. The same-length shadow source keeps every original offset valid.
    """
    characters = list(source)
    case_bodies: list[bool] = []
    index = 0

    while index < len(characters):
        line_start = source.rfind("\n", 0, index) + 1
        line_end = source.find("\n", index)
        if line_end < 0:
            line_end = len(source)

        if not case_bodies:
            line = source[line_start:line_end]
            if not _CASE_START.match(line):
                index = line_end + (line_end < len(source))
                continue

            in_token = _find_case_in(line)
            if in_token is None:
                index = line_end + (line_end < len(source))
                continue

            index = line_start + in_token
            _blank(characters, line_start, index + 2)
            case_bodies.append(False)
            continue

        if case_bodies[-1]:
            line = source[line_start:line_end]
            if _CASE_START.match(line):
                in_token = _find_case_in(line)
                if in_token is not None:
                    index = line_start + in_token
                    _blank(characters, line_start, index + 2)
                    case_bodies.append(False)
                    continue

        if not case_bodies[-1]:
            case_end = _find_case_end(source, index, line_end)
            if case_end is not None:
                start, end = case_end
                _blank(characters, start, end)
                case_bodies.pop()
                index = end
                continue

            pattern_end = _find_case_pattern_end(source, index, line_end)
            if pattern_end is None:
                # Case patterns may span lines. Hide this part and keep looking
                # for the closing ')' on the next line.
                _blank(characters, index, line_end)
                index = line_end + (line_end < len(source))
                continue

            _blank(characters, index, pattern_end + 1)
            index = pattern_end + 1
            case_bodies[-1] = True
            continue

        separator = _find_case_separator(source, index, line_end)
        if separator is not None:
            start, end = separator
            if any(
                not characters[position].isspace()
                for position in range(line_start, start)
            ):
                characters[start] = ";"
                for position in range(start + 1, end):
                    characters[position] = " "
            else:
                _blank(characters, start, end)
            index = end
            case_bodies[-1] = False
            continue

        case_end = _find_case_end(source, index, line_end)
        if case_end is not None:
            start, end = case_end
            _blank(characters, start, end)
            case_bodies.pop()
            index = end
            continue

        index = line_end + (line_end < len(source))

    return "".join(characters)


def _find_case_in(line: str) -> int | None:
    """Return the start of the unquoted ``in`` that opens a case statement."""
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote:
            if character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
            continue
        if (
            line.startswith("in", index)
            and (index == 0 or line[index - 1].isspace())
            and (index + 2 == len(line) or line[index + 2].isspace())
        ):
            return index
    return None


def _find_case_end(
    source: str,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Find ``esac`` at the start of the next case arm or statement line."""
    token_start = start
    while token_start < end and source[token_start] in " \t":
        token_start += 1
    token_end = token_start + 4
    if (
        source.startswith("esac", token_start)
        and (token_end == end or source[token_end].isspace() or source[token_end] == ";")
    ):
        return token_start, token_end
    return None


def _find_case_pattern_end(source: str, start: int, end: int) -> int | None:
    """Find a case arm's unquoted closing ')' on the current line."""
    quote: str | None = None
    escaped = False
    in_bracket_expression = False
    for index in range(start, end):
        character = source[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote:
            if character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
            continue
        if character == "[":
            in_bracket_expression = True
        elif character == "]":
            in_bracket_expression = False
        elif character == ")" and not in_bracket_expression:
            following = index + 1
            while following < end and source[following] in " \t":
                following += 1
            if following == end or source[following] not in "|)":
                return index
    return None


def _find_case_separator(
    source: str,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Find an unquoted ``;;``, ``;&`` or ``;;&`` ending a case arm."""
    quote: str | None = None
    escaped = False
    index = start
    while index < end:
        character = source[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\":
            escaped = True
            index += 1
            continue
        if quote:
            if character == quote:
                quote = None
            index += 1
            continue
        if character in "'\"":
            quote = character
            index += 1
            continue
        if source.startswith(";;&", index):
            return index, index + 3
        if source.startswith((";;", ";&"), index):
            return index, index + 2
        index += 1
    return None


def _blank(characters: list[str], start: int, end: int) -> None:
    """Replace source syntax with spaces without changing line positions."""
    for position in range(start, end):
        if characters[position] != "\n":
            characters[position] = " "


def parse_script(script: str) -> BashScript:
    return BashParser().parse(script)
    
