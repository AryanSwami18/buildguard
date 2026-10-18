import json

from app.intelligence.response import LLMResponse
from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep
from app.models.validation import (
    AIComparisonMatch,
    AIComparisonResult,
    AIComparisonMatch,
    AIValueDifference,
    MatchStatus,
)
from app.pipeline.build_pipeline import (
    BuildPipeline,
    _comparison_step,
    _alias_bash_candidates,
    _resolve_bash_candidate_ids,
    _is_reusable_parameterized_step,
    _select_comparison_bash_candidates,
)


def _step(
    step_id: str,
    source_type: str,
    *,
    version: str | None = None,
    commands: list[str] | None = None,
    runtime_options: dict[str, list[str]] | None = None,
    source_excerpt: str | None = None,
) -> BuildStep:
    return BuildStep(
        id=step_id,
        order=1,
        action=ActionType.RUN_COMMAND,
        description="Build the component.",
        commands=commands or [],
        version=version,
        context=SourceContext(
            source_type=source_type,
            source_name="guide.md" if source_type == "wiki" else "build.sh",
            runtime_options=runtime_options or {},
            source_excerpt=source_excerpt,
        ),
    )


def _comparison(value_differences: list[AIValueDifference]) -> tuple:
    wiki_step = _step("wiki-1", "wiki", version="3.9.11")
    bash_step = _step("bash-1", "bash", version="3.8.6")
    wiki = BuildInstruction(name="guide.md", source="wiki", steps=[wiki_step])
    bash = BuildInstruction(name="build.sh", source="bash", steps=[bash_step])
    result = AIComparisonResult(
        summary="Compared Maven build versions.",
        matches=[
            AIComparisonMatch(
                status=MatchStatus.CONFLICT,
                explanation="Both sources describe the Maven installation.",
                wiki_step_id="wiki-1",
                bash_step_id="bash-1",
                value_differences=value_differences,
            )
        ],
    )
    return BuildPipeline._validate_comparison(result, wiki, bash).matches[0]


def test_comparison_keeps_conflict_only_with_exact_source_value_evidence() -> None:
    match = _comparison(
        [
            AIValueDifference(
                field="version",
                wiki_value="3.9.11",
                bash_step_id="bash-1",
                bash_value="3.8.6",
            )
        ]
    )

    assert match.status == MatchStatus.CONFLICT
    assert "Wiki has 3.9.11" in match.explanation
    assert "Bash bash-1 has 3.8.6" in match.explanation


def test_comparison_downgrades_unsubstantiated_ai_conflict() -> None:
    match = _comparison([])

    assert match.status == MatchStatus.MATCHED
    assert match.category is None
    assert match.recommendation is None
    assert "does not report it as a confirmed conflict" in match.explanation


def test_comparison_does_not_accept_substrings_as_conflict_evidence() -> None:
    match = _comparison(
        [
            AIValueDifference(
                field="version",
                wiki_value="3.9",
                bash_step_id="bash-1",
                bash_value="3.8",
            )
        ]
    )

    assert match.status == MatchStatus.MATCHED


def test_comparison_context_includes_bounded_source_excerpt() -> None:
    step = _step(
        "wiki-1",
        "wiki",
        source_excerpt="Maven 3.9.11 is required.",
    )

    payload = _comparison_step(step)

    assert payload["context"]["source_excerpt"] == "Maven 3.9.11 is required."


def test_full_comparison_request_includes_source_context() -> None:
    class RecordingProvider:
        request = None

        def generate(self, request):
            self.request = request
            return LLMResponse(
                content=(
                    '{"summary":"Maven versions match.","matches":[{'
                    '"status":"matched","explanation":"Both use Maven 3.9.11.",'
                    '"wiki_step_id":"wiki-1","bash_step_id":"bash-1",'
                    '"bash_step_ids":["bash-1"]}]}'
                )
            )

    provider = RecordingProvider()
    wiki = BuildInstruction(
        name="guide.md",
        source="wiki",
        steps=[_step("wiki-1", "wiki", version="3.9.11", source_excerpt="Maven 3.9.11")],
    )
    bash = BuildInstruction(
        name="build.sh",
        source="bash",
        steps=[_step("bash-1", "bash", version="3.9.11", source_excerpt="MAVEN_VERSION=3.9.11")],
    )

    BuildPipeline(provider).compare(wiki, bash)

    request_data = json.loads(provider.request.user_prompt)
    assert request_data["wiki_instruction"]["steps"][0]["context"]["source_excerpt"] == "Maven 3.9.11"
    assert request_data["buildscript_instruction"]["steps"][0]["context"]["source_excerpt"] == "MAVEN_VERSION=3.9.11"


def test_parameterized_step_reuse_uses_generic_runtime_options() -> None:
    step = _step(
        "bash-1",
        "bash",
        commands=["install-tool --version \"$TOOL_VERSION\""],
        runtime_options={"TOOL_VERSION": ["1.0", "2.0"]},
    )

    assert _is_reusable_parameterized_step(step)


def test_configuration_declaration_can_support_multiple_semantic_actions() -> None:
    config_step = _step(
        "bash-1",
        "bash",
        commands=['PATCH_URL="https://example.test/patch"'],
    ).model_copy(
        update={
            "action": ActionType.SET_VARIABLE,
            "context": SourceContext(
                source_type="bash",
                source_name="build.sh",
                context_labels=["configuration"],
            ),
        }
    )
    wiki_steps = [_step("wiki-1", "wiki"), _step("wiki-2", "wiki")]
    result = AIComparisonResult(
        summary="The shared configuration supports both actions.",
        matches=[
            AIComparisonMatch(
                status=MatchStatus.MATCHED,
                explanation="The shared configuration supports this action.",
                wiki_step_id=wiki_step.id,
                bash_step_id="bash-1",
            )
            for wiki_step in wiki_steps
        ],
    )

    wiki = BuildInstruction(name="guide.md", source="wiki", steps=wiki_steps)
    bash = BuildInstruction(name="build.sh", source="bash", steps=[config_step])

    validated = BuildPipeline._validate_comparison(result, wiki, bash)

    assert validated.valid
    assert len(validated.matches) == 2


def test_chunk_repair_retries_unavailable_bash_id_with_allowed_candidates() -> None:
    class SequenceProvider:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return LLMResponse(content=json.dumps(next(self.responses)))

    wiki_steps = [_step(f"wiki-{index}", "wiki") for index in range(1, 10)]
    bash_ids = [*(f"bash-{index}" for index in range(1, 8)), "bash-60", "bash-61"]
    bash_steps = [_step(step_id, "bash") for step_id in bash_ids]

    def matched(wiki_id: str, bash_id: str) -> dict[str, str]:
        return {
            "status": "matched",
            "explanation": "The Bash action implements this Wiki action.",
            "wiki_step_id": wiki_id,
            "bash_step_id": bash_id,
            "bash_step_ids": [bash_id],
        }

    batch_one = {
        "summary": "First Wiki batch.",
        "matches": [
            matched(f"wiki-{index}", f"candidate-{index}")
            for index in range(1, 9)
        ],
    }
    batch_two_bad = {
        "summary": "Second Wiki batch.",
        "matches": [matched("wiki-9", "bash-60")],
    }
    repair_bad = {
        "summary": "Repair.",
        "matches": [matched("wiki-9", "bash-60")],
    }
    repair_good = {
        "summary": "Repair.",
        "matches": [matched("wiki-9", "candidate-1")],
    }
    provider = SequenceProvider([batch_one, batch_two_bad, repair_bad, repair_good])
    pipeline = BuildPipeline(provider)
    wiki = BuildInstruction(name="guide.md", source="wiki", steps=wiki_steps)
    bash = BuildInstruction(name="build.sh", source="bash", steps=bash_steps)

    result = pipeline._compare_in_chunks(
        wiki,
        bash,
        [_comparison_step(step) for step in wiki_steps],
        [_comparison_step(step) for step in bash_steps],
    )

    assert len(provider.requests) == 4
    assert result.valid
    assert result.matches[-1].automation_step.id == "bash-61"
    repair_payload = json.loads(provider.requests[-1].user_prompt)
    assert "not available in this batch" in repair_payload["validation_feedback"]
    assert [step["id"] for step in repair_payload["available_buildscript_steps"]] == [
        "candidate-1"
    ]


def test_chunk_comparison_rematches_duplicate_bash_assignment() -> None:
    class DuplicateThenRepairProvider:
        def __init__(self):
            self.requests = []
            self.responses = iter(
                [
                    {
                        "summary": "The same Bash step was selected twice.",
                        "matches": [
                            {
                                "status": "matched",
                                "explanation": "The Bash action implements this Wiki action.",
                                "wiki_step_id": "wiki-1",
                                "bash_step_id": "candidate-1",
                            },
                            {
                                "status": "matched",
                                "explanation": "The Bash action implements this Wiki action.",
                                "wiki_step_id": "wiki-2",
                                "bash_step_id": "candidate-1",
                            },
                        ],
                    },
                    {
                        "summary": "The second Wiki action uses another Bash step.",
                        "matches": [
                            {
                                "status": "matched",
                                "explanation": "A different Bash action implements this Wiki action.",
                                "wiki_step_id": "wiki-2",
                                "bash_step_id": "candidate-1",
                            }
                        ],
                    },
                ]
            )

        def generate(self, request):
            self.requests.append(request)
            return LLMResponse(content=json.dumps(next(self.responses)))

    wiki_steps = [_step("wiki-1", "wiki"), _step("wiki-2", "wiki")]
    bash_steps = [
        _step("bash-1", "bash", commands=["prepare first action"]),
        _step("bash-2", "bash", commands=["prepare second action"]),
    ]
    provider = DuplicateThenRepairProvider()

    result = BuildPipeline(provider)._compare_in_chunks(
        BuildInstruction(name="guide.md", source="wiki", steps=wiki_steps),
        BuildInstruction(name="build.sh", source="bash", steps=bash_steps),
        [_comparison_step(step) for step in wiki_steps],
        [_comparison_step(step) for step in bash_steps],
    )

    assert result.valid
    assert len(provider.requests) == 2
    assert [match.automation_step.id for match in result.matches] == [
        "bash-1",
        "bash-2",
    ]
    repair_payload = json.loads(provider.requests[1].user_prompt)
    assert repair_payload["wiki_steps"][0]["id"] == "wiki-2"
    assert [step["id"] for step in repair_payload["available_buildscript_steps"]] == [
        "candidate-1"
    ]


def test_bash_candidates_use_local_ids_and_drop_unavailable_dependency_ids() -> None:
    candidate_steps, candidate_to_source = _alias_bash_candidates(
        [
            {"id": "bash-61", "depends_on": ["bash-60", "bash-62"]},
            {"id": "bash-62", "depends_on": ["bash-61"]},
        ]
    )

    assert [step["id"] for step in candidate_steps] == ["candidate-1", "candidate-2"]
    assert candidate_steps[0]["depends_on"] == ["candidate-2"]
    assert candidate_steps[1]["depends_on"] == ["candidate-1"]
    assert candidate_to_source == {"candidate-1": "bash-61", "candidate-2": "bash-62"}


def test_chunk_comparison_splits_and_retries_after_output_limit() -> None:
    class OutputLimitThenSequenceProvider:
        def __init__(self):
            self.requests = []
            self.responses = iter(
                [
                    {"summary": "First action matched.", "matches": [
                        {
                            "status": "matched",
                            "explanation": "The first action is implemented.",
                            "wiki_step_id": "wiki-1",
                            "bash_step_id": "candidate-1",
                        }
                    ]},
                    {"summary": "Second action matched.", "matches": [
                        {
                            "status": "matched",
                            "explanation": "The second action is implemented.",
                            "wiki_step_id": "wiki-2",
                            "bash_step_id": "candidate-1",
                        }
                    ]},
                ]
            )
            self.fail_first = True

        def generate(self, request):
            self.requests.append(request)
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError(
                    "Ollama response exceeded the output limit (16384 tokens)"
                )
            return LLMResponse(content=json.dumps(next(self.responses)))

    wiki_steps = [_step("wiki-1", "wiki"), _step("wiki-2", "wiki")]
    bash_steps = [_step("bash-1", "bash"), _step("bash-2", "bash")]
    provider = OutputLimitThenSequenceProvider()
    pipeline = BuildPipeline(provider)
    wiki = BuildInstruction(name="guide.md", source="wiki", steps=wiki_steps)
    bash = BuildInstruction(name="build.sh", source="bash", steps=bash_steps)

    result = pipeline._compare_in_chunks(
        wiki,
        bash,
        [_comparison_step(step) for step in wiki_steps],
        [_comparison_step(step) for step in bash_steps],
    )

    assert len(provider.requests) == 3
    assert result.valid
    assert [match.automation_step.id for match in result.matches] == ["bash-1", "bash-2"]


def test_candidate_aliases_are_mapped_back_in_matches_and_conflict_evidence() -> None:
    result = AIComparisonResult(
        summary="A value differs.",
        matches=[
            AIComparisonMatch(
                status=MatchStatus.CONFLICT,
                explanation="The component versions differ.",
                wiki_step_id="wiki-1",
                bash_step_id="candidate-1",
                bash_step_ids=["candidate-1", "candidate-2"],
                value_differences=[
                    AIValueDifference(
                        field="version",
                        wiki_value="2.0",
                        bash_step_id="candidate-2",
                        bash_value="1.0",
                    )
                ],
            )
        ],
    )

    mapped = _resolve_bash_candidate_ids(
        result,
        {"candidate-1": "bash-61", "candidate-2": "bash-62"},
    )

    assert mapped.matches[0].bash_step_id == "bash-61"
    assert mapped.matches[0].bash_step_ids == ["bash-61", "bash-62"]
    assert mapped.matches[0].value_differences[0].bash_step_id == "bash-62"


def test_candidate_retrieval_keeps_relevant_actions_in_a_bounded_set() -> None:
    wiki_step = _step(
        "wiki-1",
        "wiki",
        commands=["wget https://archive.apache.org/maven-3.9.11.tar.gz"],
    )
    bash_steps = [
        _step(f"bash-{index}", "bash", commands=[f"cleanup-helper-{index}"])
        for index in range(1, 51)
    ]
    bash_steps.append(
        _step(
            "bash-maven",
            "bash",
            commands=[
                "wget https://archive.apache.org/maven-${MAVEN_VERSION}.tar.gz"
            ],
        )
    )

    selected = _select_comparison_bash_candidates([wiki_step], bash_steps, set())

    assert len(selected) <= 40
    assert "bash-maven" in {step.id for step in selected}


def test_candidate_retrieval_keeps_a_small_function_group_together() -> None:
    wiki_step = _step("wiki-1", "wiki")
    wiki_step.context = wiki_step.context.model_copy(update={"subsection": "installTool"})
    bash_steps = [
        _step(f"bash-{index}", "bash")
        for index in range(1, 51)
    ]
    for index in range(51, 54):
        grouped = _step(f"bash-{index}", "bash")
        grouped.context = grouped.context.model_copy(update={"subsection": "installTool"})
        bash_steps.append(grouped)

    selected = _select_comparison_bash_candidates([wiki_step], bash_steps, set())

    assert {"bash-51", "bash-52", "bash-53"}.issubset(
        {step.id for step in selected}
    )

