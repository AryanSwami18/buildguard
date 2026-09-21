from app.matching.matcher import BuildMatcher
from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep


def _step(step_id: str, order: int, dependency: list[str] | None = None) -> BuildStep:
    return BuildStep(
        id=step_id,
        order=order,
        action=ActionType.INSTALL_PACKAGE,
        description="Install git.",
        packages=["git"],
        depends_on=dependency or [],
        context=SourceContext(source_type="test", source_name="test"),
    )


def test_matcher_uses_semantics_and_dependency_mapping() -> None:
    wiki = BuildInstruction(name="guide", source="wiki", steps=[_step("wiki-1", 1), _step("wiki-2", 2, ["wiki-1"])])
    bash = BuildInstruction(name="script", source="bash", steps=[_step("bash-1", 1), _step("bash-2", 2, ["bash-1"])])

    result = BuildMatcher().compare(wiki, bash)

    assert result.valid is True
    assert [match.status.value for match in result.matches] == ["matched", "matched"]
