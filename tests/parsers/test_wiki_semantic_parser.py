from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep
from app.parsers.wiki_semantic_parser import complete_wiki_instruction


def _context(line_start: int, line_end: int) -> SourceContext:
    return SourceContext(
        source_type="wiki",
        source_name="Building-Example.md",
        line_start=line_start,
        line_end=line_end,
    )


def test_wiki_completeness_adds_uncovered_dependency_and_java_workflows() -> None:
    markdown = """# Build Example

## 1: Install dependencies

```bash
sudo yum install -y wget tar curl
```

## 2: Install Java

```bash
sudo yum install -y java-17-openjdk-devel
```

## 3: Build

```bash
mvn install -DskipTests=true
```
"""
    initial = BuildInstruction(
        name="Building-Example.md",
        source="wiki",
        steps=[
            BuildStep(
                id="wiki-1",
                order=1,
                action=ActionType.RUN_COMMAND,
                description="Build Example.",
                commands=["mvn install -DskipTests=true"],
                context=_context(15, 17),
            )
        ],
    )

    completed = complete_wiki_instruction(initial, markdown, "Building-Example.md")

    assert len(completed.steps) == 3
    assert [step.action for step in completed.steps] == [
        ActionType.INSTALL_PACKAGE,
        ActionType.INSTALL_PACKAGE,
        ActionType.RUN_COMMAND,
    ]
    assert completed.steps[0].packages == ["wget", "tar", "curl"]
    assert completed.steps[1].component == "java"
    assert completed.steps[1].context.runtime_options == {}
    assert completed.steps[2].commands == ["mvn install -DskipTests=true"]


def test_wiki_completeness_is_idempotent() -> None:
    markdown = """# Build

```bash
sudo apt-get update
```
"""
    initial = BuildInstruction(
        name="guide.md",
        source="wiki",
        steps=[],
    )

    once = complete_wiki_instruction(initial, markdown, "guide.md")
    twice = complete_wiki_instruction(once, markdown, "guide.md")

    assert twice.model_dump() == once.model_dump()
