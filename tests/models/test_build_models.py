from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep


def test_build_step_keeps_source_and_neighbouring_step_context() -> None:
    step = BuildStep(
        id="wiki-2",
        order=2,
        action=ActionType.DOWNLOAD,
        description="Download the distribution as the Alfresco user.",
        urls=["https://example.com/app.tar.gz"],
        user="alfresco",
        depends_on=["wiki-1"],
        context=SourceContext(
            source_type="wiki",
            source_name="Alfresco build guide",
            section="Installation",
            subsection="Download",
            line_start=12,
            line_end=13,
            surrounding_text="Create the Alfresco user. Configure the service.",
            previous_step_id="wiki-1",
            next_step_id="wiki-3",
            related_step_ids=["wiki-1", "wiki-3"],
        ),
    )

    assert step.depends_on == ["wiki-1"]
    assert step.context.previous_step_id == "wiki-1"
    assert step.context.next_step_id == "wiki-3"
    assert step.user == "alfresco"


def test_build_instruction_defaults_to_an_empty_step_list() -> None:
    instruction = BuildInstruction(name="Example guide", source="wiki")

    assert instruction.steps == []
