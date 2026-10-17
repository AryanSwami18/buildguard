import json

import pytest

from app.intelligence.request import LLMRequest
from app.intelligence.response import LLMResponse
from app.models.report import (
    ComparisonCounts,
    ComparisonReport,
    Discrepancy,
    StepEvidence,
)
from app.models.step import ActionType
from app.models.repair import ScriptFixProposal
from app.models.validation import DiscrepancyCategory, MatchStatus
from app.repair import generate_script_fix_proposal


def _report() -> ComparisonReport:
    return ComparisonReport(
        package="Bazel",
        version="9.2.0",
        status="review_required",
        overview="One finding.",
        counts=ComparisonCounts(
            wiki_steps_compared=1,
            script_steps_checked=1,
            missing_steps=0,
            conflicting_steps=0,
            undocumented_script_steps=1,
            out_of_scope_steps=0,
            findings=1,
        ),
        discrepancies=[
            Discrepancy(
                id="D-001",
                status=MatchStatus.EXTRA,
                category=DiscrepancyCategory.UNDOCUMENTED_STEP,
                title="Undocumented command",
                explanation="The command is not represented in the Wiki.",
                recommendation="Review the command.",
                script=[
                    StepEvidence(
                        id="bash-1",
                        description="Run original command.",
                        action=ActionType.RUN_COMMAND,
                        location="build.sh | line 2",
                        line_start=2,
                        line_end=2,
                        commands=["echo original"],
                    )
                ],
            )
        ],
    )


class _FakeProvider:
    def __init__(self) -> None:
        self.request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.request = request
        return LLMResponse(
            content=json.dumps(
                ScriptFixProposal(
                    fixed_script="#!/usr/bin/env bash\necho fixed\n",
                    summary="Removed the undocumented command.",
                    rationale="The finding identified the command as unnecessary.",
                    affected_finding_ids=["D-001"],
                ).model_dump()
            )
        )


def test_fix_proposal_is_structured_and_bash_validated() -> None:
    provider = _FakeProvider()
    proposal = generate_script_fix_proposal(
        provider,
        _report(),
        "#!/usr/bin/env bash\necho original\n",
    )

    assert proposal.affected_finding_ids == ["D-001"]
    assert proposal.fixed_script == "#!/usr/bin/env bash\necho fixed\n"
    assert provider.request is not None
    assert "original_script" in provider.request.user_prompt


def _unused_configuration_report() -> ComparisonReport:
    return ComparisonReport(
        package="Alfresco",
        version="25.3.0",
        status="review_required",
        overview="One unused configuration finding.",
        counts=ComparisonCounts(
            wiki_steps_compared=1,
            script_steps_checked=2,
            missing_steps=0,
            conflicting_steps=0,
            undocumented_script_steps=0,
            out_of_scope_steps=0,
            unused_configuration=1,
            findings=1,
        ),
        discrepancies=[
            Discrepancy(
                id="D-001",
                status=MatchStatus.UNUSED_CONFIGURATION,
                category=DiscrepancyCategory.UNUSED_CONFIGURATION,
                title="Unused configuration",
                explanation="ACS_NGINX_VERSION is not referenced.",
                recommendation="Remove the unused declaration.",
                script=[
                    StepEvidence(
                        id="bash-2",
                        description='ACS_NGINX_VERSION="3.4.2"',
                        action=ActionType.SET_VARIABLE,
                        location="build_alfresco.sh | line 2",
                        line_start=2,
                        line_end=2,
                        commands=['ACS_NGINX_VERSION="3.4.2"'],
                        environment={"ACS_NGINX_VERSION": "3.4.2"},
                    )
                ],
            )
        ],
    )


def test_unused_configuration_fix_is_deterministic_and_minimal() -> None:
    class ProviderThatMustNotBeCalled:
        def generate(self, request: LLMRequest) -> LLMResponse:
            raise AssertionError("unused configuration fixes must not use the LLM")

    original = '#!/usr/bin/env bash\nACS_NGINX_VERSION="3.4.2"\necho build\n'
    proposal = generate_script_fix_proposal(
        ProviderThatMustNotBeCalled(),
        _unused_configuration_report(),
        original,
    )

    assert proposal.fixed_script == '#!/usr/bin/env bash\necho build\n'
    assert proposal.affected_finding_ids == ["D-001"]
    assert proposal.fixed_script.endswith("\n")


def test_ai_fix_rejects_unrelated_indentation_or_eof_changes() -> None:
    class BadProvider:
        def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                content=json.dumps(
                    ScriptFixProposal(
                        fixed_script=(
                            "#!/usr/bin/env bash -x\n"
                            "echo original\n"
                        ),
                        summary="Changed the command.",
                        rationale="The finding requires it.",
                        affected_finding_ids=["D-001"],
                    ).model_dump()
                )
            )

    report = _report()
    with pytest.raises(RuntimeError, match="unrelated script lines"):
        generate_script_fix_proposal(
            BadProvider(),
            report,
            "#!/usr/bin/env bash\necho original\n",
        )
