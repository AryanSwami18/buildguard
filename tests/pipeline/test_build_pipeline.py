import json

import pytest

from app.intelligence.request import LLMRequest
from app.intelligence.response import LLMResponse
from app.models.step import BuildStep
from app.pipeline.build_pipeline import BuildPipeline


class SequenceProvider:
    """A test double proving the pipeline does not need a named LLM vendor."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=json.dumps(self.responses.pop(0)))


def test_bash_pipeline_sends_ast_and_neighbour_context_to_generic_provider() -> None:
    provider = SequenceProvider([
        {"action": "install_package", "description": "Install git.", "packages": ["git"]},
        {"action": "create_directory", "description": "Create application directory.", "paths": ["/opt/alfresco"], "depends_on": ["bash-1"]},
    ])

    instruction = BuildPipeline(provider).from_bash(
        "sudo dnf install -y git\nmkdir -p /opt/alfresco\n", "install.sh"
    )

    assert [step.id for step in instruction.steps] == ["bash-1", "bash-2"]
    assert instruction.steps[1].depends_on == ["bash-1"]
    assert instruction.steps[0].context.next_step_id == "bash-2"
    request_payload = json.loads(provider.requests[0].user_prompt)
    assert "Interpret the supplied Bash syntax" in provider.requests[0].system_prompt
    assert request_payload["source"]["current_node"]["kind"] == "command"
    assert request_payload["source"]["script_ast"]["nodes"][0]["words"][0] == "sudo"
    assert request_payload["context"]["next_step_id"] == "bash-2"
    assert request_payload["expected_output"] == BuildStep.model_json_schema()


def test_wiki_pipeline_preserves_document_relationships_for_the_provider() -> None:
    provider = SequenceProvider([
        {"action": "install_package", "description": "Install Java.", "packages": ["java"]},
        {"action": "create_user", "description": "Create the Alfresco user.", "depends_on": ["wiki-1"]},
    ])

    instruction = BuildPipeline(provider).from_wiki(
        "# Setup\n\nInstall Java.\n\nCreate the Alfresco user.\n", "guide.md"
    )

    assert instruction.steps[1].context.previous_step_id == "wiki-1"
    assert instruction.steps[1].depends_on == ["wiki-1"]
    request_payload = json.loads(provider.requests[1].user_prompt)
    assert "Interpret the supplied Markdown block" in provider.requests[1].system_prompt
    assert request_payload["source"]["document"]["headings"] == [
        {"level": 1, "text": "Setup", "line": 1}
    ]
    assert request_payload["source"]["document"]["blocks"][0]["section"] == "Setup"
    assert request_payload["context"]["known_step_ids"] == ["wiki-1"]


def test_wiki_pipeline_sends_one_complete_document_and_returns_instruction() -> None:
    provider = SequenceProvider([
        {
            "name": "ignored-by-source-validation",
            "source": "ignored-by-source-validation",
            "version": None,
            "steps": [
                {
                    "id": "model-chosen-id",
                    "order": 99,
                    "action": "download",
                    "description": "Download the archive.",
                    "urls": ["https://example.com/archive.tar.gz"],
                    "context": {
                        "source_type": "wiki",
                        "source_name": "wrong-name",
                        "section": "Install",
                        "location": "lines 3-5",
                        "line_start": 3,
                        "line_end": 5,
                    },
                }
            ],
        }
    ])

    instruction = BuildPipeline(provider).from_wiki(
        "# Install\n\nDownload the archive from https://example.com/archive.tar.gz.\n",
        "guide.md",
    )

    assert len(provider.requests) == 1
    request_payload = json.loads(provider.requests[0].user_prompt)
    assert request_payload["source"] == "wiki"
    assert request_payload["document"]["name"] == "guide.md"
    assert request_payload["document"]["blocks"][0]["line_start"] == 3
    assert request_payload["expected_output"]
    assert instruction.name == "guide.md"
    assert instruction.source == "wiki"
    assert instruction.steps[0].id == "wiki-1"
    assert instruction.steps[0].order == 1
    assert instruction.steps[0].context.source_name == "guide.md"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "LLM returned empty JSON for step wiki-1"),
        ("not json", "LLM returned invalid JSON for step wiki-1: not json"),
    ],
)
def test_wiki_pipeline_reports_empty_or_invalid_provider_json(content: str, message: str) -> None:
    class InvalidProvider:
        def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content=content)

    with pytest.raises(RuntimeError, match=message):
        BuildPipeline(InvalidProvider()).from_wiki("# Setup\n\nInstall Java.\n", "guide.md")
