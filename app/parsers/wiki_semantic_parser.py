"""Deterministic completeness checks for AI-extracted Wiki instructions.

The LLM is useful for describing a Wiki action, but it must not be the only
source of coverage. A model can omit a fenced command block—especially when a
page contains several distro variants or optional sections—and that omission
would turn valid script operations into undocumented-step findings.

This module adds one semantic step for every meaningful, uncovered Wiki code
block. Related Java blocks are intentionally collapsed into one configurable
Java workflow. Existing AI steps are preserved and are not duplicated.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
import textwrap

from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep
from app.parsers.bash_semantic_parser import parse_bash_steps
from app.parsers.wiki_parser import parse_wiki
from app.sources.wiki import WikiBlock


_JAVA_OPTION = re.compile(r"\b(?:IBMSemeru|Semeru|Temurin|OpenJDK)\d+\b")
_NOISE_PACKAGES = {"", "\\", "\n", "|&", "tee", "$log_file"}


def complete_wiki_instruction(
    instruction: BuildInstruction,
    markdown: str,
    name: str | None = None,
) -> BuildInstruction:
    """Add source-grounded steps omitted by the Wiki extraction provider.

    The function is idempotent. It can therefore be called for both freshly
    parsed and previously cached instructions without producing duplicate
    steps. The original AI interpretation remains authoritative wherever its
    source range already covers a code block.
    """
    document = parse_wiki(markdown, name or instruction.name)
    code_blocks = [block for block in document.blocks if block.kind == "code_block"]
    if not code_blocks:
        return instruction

    existing = list(instruction.steps)
    supplements: list[BuildStep] = []

    java_blocks = [block for block in code_blocks if _is_java_block(block)]
    if java_blocks and not any(_step_covers_java(step) for step in existing):
        java_step = _aggregate_blocks(
            java_blocks,
            name=name or instruction.name,
            description=_java_description(markdown),
            action=ActionType.INSTALL_PACKAGE,
            component="java",
            runtime_options=_java_options(markdown),
        )
        if java_step is not None:
            supplements.append(java_step)

    for block in code_blocks:
        if block in java_blocks or _block_is_covered(block, existing):
            continue

        candidate = _aggregate_blocks(
            [block],
            name=name or instruction.name,
        )
        if candidate is None or _duplicate_semantic_step(candidate, existing, supplements):
            continue
        supplements.append(candidate)

    if not supplements:
        return instruction

    return instruction.model_copy(
        update={
            "steps": _reindex_steps([*existing, *supplements]),
        }
    )


def _is_java_block(block: WikiBlock) -> bool:
    return any("install java" in heading.lower() for heading in block.heading_path)


def _step_covers_java(step: BuildStep) -> bool:
    description = step.description.lower()
    heading = " ".join(step.context.heading_path).lower()
    if step.component and step.component.lower() in {"java", "jdk", "jre"}:
        return True
    if "install java" in heading or "download and install java" in heading:
        return True
    if not any(marker in description for marker in ("install java", "configure java")):
        return False
    values = " ".join(
        value
        for value in (
            step.description,
            step.component,
            step.context.subsection,
            " ".join(step.context.heading_path),
        )
        if value
    ).lower()
    return any(marker in values for marker in ("java", "jdk", "jre", "temurin", "semeru"))


def _block_is_covered(block: WikiBlock, steps: Iterable[BuildStep]) -> bool:
    for step in steps:
        start = step.context.line_start
        end = step.context.line_end
        if start is None or end is None:
            continue
        if start <= block.line_end and end >= block.line_start:
            return True
    return False


def _aggregate_blocks(
    blocks: list[WikiBlock],
    *,
    name: str,
    description: str | None = None,
    action: ActionType | None = None,
    component: str | None = None,
    runtime_options: list[str] | None = None,
) -> BuildStep | None:
    if not blocks:
        return None

    shell_sources = [_shell_source(block.text) for block in blocks]
    parsed_steps: list[BuildStep] = []
    for source in shell_sources:
        if not source.strip():
            continue
        try:
            parsed_steps.extend(parse_bash_steps(source, name))
        except Exception:
            # The source remains available in the Wiki context. A malformed
            # code block must not make the whole document unavailable.
            continue

    commands = [command for step in parsed_steps for command in step.commands]
    packages = _unique(
        package
        for step in parsed_steps
        for package in step.packages
        if package.strip() not in _NOISE_PACKAGES
    )
    urls = _unique(url for step in parsed_steps for url in step.urls)
    paths = _unique(path for step in parsed_steps for path in step.paths)
    environment = {
        key: value
        for step in parsed_steps
        for key, value in step.environment.items()
    }
    build_tools = _unique(
        step.build_tool for step in parsed_steps if step.build_tool
    )

    heading_text = " ".join(
        [*blocks[0].heading_path, *(block.subsection or "" for block in blocks)]
    )
    source_text = "\n\n".join(block.text for block in blocks)
    lowered = f"{heading_text} {source_text}".lower()

    resolved_action = action or _action_for_block(blocks, parsed_steps)
    if resolved_action is None:
        return None

    resolved_component = component or (
        None
        if resolved_action == ActionType.INSTALL_PACKAGE
        else _component_for_text(lowered)
    )
    first = blocks[0]
    last = blocks[-1]
    context = SourceContext(
        source_type="wiki",
        source_name=name,
        section=first.section,
        subsection=first.subsection,
        heading_path=first.heading_path,
        location=f"lines {first.line_start}-{last.line_end}",
        line_start=first.line_start,
        line_end=last.line_end,
        surrounding_text=source_text[:1200],
        source_excerpt=source_text,
        conditions=_unique(condition for block in blocks for condition in block.conditions),
        runtime_options={"JAVA_PROVIDED": runtime_options or []}
        if runtime_options
        else {},
    )
    return BuildStep(
        id="wiki-supplement",
        order=0,
        action=resolved_action,
        description=description or _description_for_block(resolved_action, lowered),
        packages=packages,
        commands=commands or [line.strip() for line in source_text.splitlines() if line.strip()],
        paths=paths,
        urls=urls,
        component=resolved_component,
        environment=environment,
        build_tool=build_tools[0] if build_tools else None,
        context=context,
    )


def _shell_source(text: str) -> str:
    lines = text.splitlines()
    fenced_lines: list[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            fenced_lines.append(line)
    if fenced_lines:
        lines = fenced_lines
    else:
        lines = [
            line
            for line in lines
            if line.strip()
            and not line.lstrip().startswith(("*", "-", ">"))
        ]
    return textwrap.dedent("\n".join(lines))


def _action_for_block(
    blocks: list[WikiBlock],
    parsed_steps: list[BuildStep],
) -> ActionType | None:
    if any(_is_test_block(block) for block in blocks):
        return ActionType.RUN_TEST

    priority = (
        ActionType.INSTALL_PACKAGE,
        ActionType.CLONE_REPOSITORY,
        ActionType.DOWNLOAD,
        ActionType.EXTRACT,
        ActionType.BUILD_IMAGE,
        ActionType.RUN_TEST,
        ActionType.CONFIGURE,
        ActionType.RUN_COMMAND,
    )
    actions = {step.action for step in parsed_steps}
    for candidate in priority:
        if candidate in actions:
            return candidate
    return ActionType.RUN_COMMAND if parsed_steps else None


def _is_test_block(block: WikiBlock) -> bool:
    heading = (block.subsection or "").lower()
    return bool(re.search(r"\btest\b|\bvalidate\b", heading))


def _component_for_text(text: str) -> str | None:
    if any(marker in text for marker in ("golang", "go version", "go_url", " go ")):
        return "go"
    if "node" in text:
        return "nodejs"
    if "maven" in text or "mvn" in text:
        return "maven"
    if "antlr" in text:
        return "antlr"
    return None


def _description_for_block(action: ActionType, text: str) -> str:
    if action == ActionType.INSTALL_PACKAGE:
        return "Install the documented build dependencies."
    if action == ActionType.DOWNLOAD:
        return "Download the documented build resource."
    if action == ActionType.RUN_TEST:
        return "Run the documented build/runtime test."
    if action == ActionType.EXTRACT:
        return "Extract the documented build archive."
    return "Run the documented build operation."


def _java_description(markdown: str) -> str:
    options = ", ".join(_java_options(markdown))
    if options:
        return f"Install and configure supported Java runtimes ({options})."
    return "Install and configure the documented Java runtime."


def _java_options(markdown: str) -> list[str]:
    return _unique(match.group(0) for match in _JAVA_OPTION.finditer(markdown))


def _duplicate_semantic_step(
    candidate: BuildStep,
    existing: list[BuildStep],
    supplements: list[BuildStep],
) -> bool:
    for step in [*existing, *supplements]:
        if step.context.line_start != candidate.context.line_start:
            continue
        if step.action == candidate.action and (
            set(step.urls) & set(candidate.urls)
            or set(step.packages) & set(candidate.packages)
            or step.component == candidate.component
        ):
            return True
    return False


def _reindex_steps(steps: list[BuildStep]) -> list[BuildStep]:
    steps = sorted(
        steps,
        key=lambda step: (
            step.context.line_start or 0,
            step.order,
        ),
    )
    id_map = {step.id: f"wiki-{index}" for index, step in enumerate(steps, start=1)}
    result: list[BuildStep] = []
    for index, step in enumerate(steps, start=1):
        context = step.context.model_copy(
            update={
                "previous_step_id": f"wiki-{index - 1}" if index > 1 else None,
                "next_step_id": f"wiki-{index + 1}" if index < len(steps) else None,
                "related_step_ids": (
                    ([f"wiki-{index - 1}"] if index > 1 else [])
                    + ([f"wiki-{index + 1}"] if index < len(steps) else [])
                ),
            }
        )
        result.append(
            step.model_copy(
                update={
                    "id": f"wiki-{index}",
                    "order": index,
                    "depends_on": [id_map.get(value, value) for value in step.depends_on],
                    "context": context,
                }
            )
        )
    return result


def _unique(values: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
