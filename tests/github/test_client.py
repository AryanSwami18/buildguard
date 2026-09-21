import base64
import json

from app.github.client import GitHubClient
from app.config.settings import Settings
from app.intelligence.request import LLMRequest
from app.intelligence.response import LLMResponse
from app.pipeline.build_pipeline import BuildPipeline


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def mock_github_responses(monkeypatch, responses: dict[str, object]) -> list[str]:
    requested_urls: list[str] = []

    def fake_urlopen(request):
        requested_urls.append(request.full_url)
        return FakeResponse(responses[request.full_url])

    monkeypatch.setattr("app.github.client.urlopen", fake_urlopen)
    return requested_urls


def test_client_lists_paths_and_fetches_file_content(monkeypatch) -> None:
    directory_url = "https://api.github.com/repos/example/automation/contents/scripts?ref=release"
    file_url = "https://api.github.com/repos/example/automation/contents/scripts/build.txt?ref=release"
    requested_urls = mock_github_responses(
        monkeypatch,
        {
            directory_url: [
                {
                    "type": "dir",
                    "name": "patches",
                    "path": "scripts/patches",
                    "html_url": "https://github.com/example/automation/tree/release/scripts/patches",
                },
                {
                    "type": "file",
                    "name": "build.txt",
                    "path": "scripts/build.txt",
                    "html_url": "https://github.com/example/automation/blob/release/scripts/build.txt",
                },
            ],
            file_url: {
                "type": "file",
                "name": "build.txt",
                "path": "scripts/build.txt",
                "encoding": "base64",
                "content": base64.b64encode(b"build instructions\n").decode("ascii"),
                "html_url": "https://github.com/example/automation/blob/release/scripts/build.txt",
            },
        },
    )

    client = GitHubClient()
    entries = client.list_path("example/automation", "scripts", "release")
    source_file = client.fetch_file("example/automation", "scripts/build.txt", "release")

    assert [(entry.kind, entry.path) for entry in entries] == [
        ("directory", "scripts/patches"),
        ("file", "scripts/build.txt"),
    ]
    assert source_file.content == "build instructions\n"
    assert source_file.repository == "example/automation"
    assert source_file.ref == "release"
    assert source_file.filename == "build.txt"
    assert source_file.github_url.endswith("/scripts/build.txt")
    assert requested_urls == [directory_url, file_url]


def test_github_wiki_markdown_flows_into_the_wiki_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        GitHubClient,
        "_fetch_wiki_content",
        lambda self, repository, path, ref: "# Build\n\nPrepare the source.\n",
    )

    class FakeProvider:
        def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content='{"action": "unknown", "description": "Prepare the source."}')

    page = GitHubClient().fetch_wiki_markdown("example/docs", "Build-guide", "v2")
    instruction = BuildPipeline(FakeProvider()).from_github_wiki(page)

    assert instruction.name == "Build-guide.md"
    assert instruction.steps[0].description == "Prepare the source."
    assert instruction.source_metadata is not None
    assert instruction.source_metadata.model_dump() == {
        "repository": "example/docs",
        "ref": "v2",
        "path": "Build-guide.md",
        "filename": "Build-guide.md",
        "github_url": "https://github.com/example/docs/wiki/Build-guide",
    }


def test_configured_client_resolves_a_module_name_to_a_wiki_page(monkeypatch) -> None:
    monkeypatch.setattr(
        GitHubClient,
        "_fetch_wiki_content",
        lambda self, repository, path, ref: "# Module guide\n",
    )

    client = GitHubClient.from_settings(
        Settings(github_repo="example/docs", github_branch="release")
    )
    page = client.fetch_module_wiki("Module guide")

    assert page.repository == "example/docs"
    assert page.ref == "release"
    assert page.path == "Module guide.md"
    assert page.github_url == "https://github.com/example/docs/wiki/Module%20guide"


def test_default_wiki_ref_uses_the_wiki_default_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        GitHubClient,
        "_fetch_wiki_content",
        lambda self, repository, path, ref: "# Module\n",
    )

    page = GitHubClient(repository="example/docs").fetch_module_wiki("Module")

    assert page.ref == "HEAD"
    assert page.path == "Module.md"
