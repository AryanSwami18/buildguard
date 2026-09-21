"""Bash syntax parsing using bashlex.

The parser keeps the Bash structure intact and intentionally does not
interpret individual commands such as package managers or download tools.
"""

import bashlex

from app.models.bash import BashNode, BashScript


class BashParser:
    def parse(self, script: str) -> BashScript:
        """Parse a script into a small, JSON-friendly view of the bashlex AST."""
        return BashScript(
            source=script,
            nodes=[self._convert(node, script) for node in bashlex.parse(script)],
        )

    def build_tree(self, script: str) -> list[BashNode]:
        """Return root nodes for callers that only need the AST tree."""
        return self.parse(script).nodes

    def _convert(self, node: object, script: str) -> BashNode:
        position_start, position_end = node.pos
        children = [self._convert(child, script) for child in self._child_nodes(node)]
        words = [
            part.word
            for part in getattr(node, "parts", [])
            if part.kind == "word"
        ]
        return BashNode(
            kind=node.kind,
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


def parse_script(script: str) -> BashScript:
    return BashParser().parse(script)
    
