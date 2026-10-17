import json

import pytest

from app.intelligence.request import LLMRequest
from app.intelligence.response import LLMResponse
from app.config.settings import Settings
from app.models.report import (
    ComparisonCounts,
    ComparisonReport,
    Discrepancy,
    ReportSource,
    StepEvidence,
)
from app.models.step import ActionType
from app.models.repair import ScriptFixProposal
from app.models.validation import DiscrepancyCategory, MatchStatus
from app.repair import generate_script_fix_proposal
from app.service import _build_pull_request_body


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


def test_generated_pr_body_explains_the_change_and_supporting_evidence() -> None:
    report = _report()
    report.overview = "The script has one action that is not represented in the Wiki."
    report.wiki_source = ReportSource(
        name="Build Instructions",
        repository="example/docs",
        ref="main",
        path="Building-Bazel.md",
        url="https://github.com/example/docs/blob/main/Building-Bazel.md",
    )
    report.script_source = ReportSource(
        name="build_bazel.sh",
        repository="example/scripts",
        ref="master",
        path="Bazel/9.2.0/build_bazel.sh",
        url="https://github.com/example/scripts/blob/master/Bazel/9.2.0/build_bazel.sh",
    )
    report.discrepancies[0].wiki = StepEvidence(
        id="wiki-1",
        description="Install Bazel dependencies.",
        action=ActionType.INSTALL_PACKAGE,
        location="Building-Bazel.md | line 20",
        line_start=20,
        line_end=20,
        commands=["sudo apt-get install -y build-essential"],
    )

    proposal = ScriptFixProposal(
        fixed_script="#!/usr/bin/env bash\necho fixed\n",
        summary="Replace the untracked command with the documented dependency install.",
        rationale="The Wiki documents the dependency installation, so the script should use it.",
        affected_finding_ids=["D-001"],
    )
    body = _build_pull_request_body(
        report,
        proposal,
        report.script_source,
        Settings(scripts_repo="example/scripts", scripts_branch="master"),
    )

    assert "## Summary" in body
    assert "Bazel" in body and "9.2.0" in body
    assert "## Why this change is needed" in body
    assert "## What the proposed change does" in body
    assert proposal.summary in body
    assert proposal.rationale in body
    assert "### Problem 1: The script performs an undocumented build action" in body
    assert "D-001" not in body
    assert "wiki-1" not in body
    assert "Wiki step" not in body
    assert "Bash step" not in body
    assert "Install Bazel dependencies." in body
    assert "sudo apt-get install -y build-essential" in body
    assert "## Sources" in body
    assert "https://github.com/example/docs/blob/main/Building-Bazel.md" in body
    assert "## Reviewer checklist" in body
