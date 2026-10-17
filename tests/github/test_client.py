import base64
import json
from pathlib import Path
import subprocess

from app.github.client import GitHubClient
from app.config.settings import Settings
from app.intelligence.request import LLMRequest
from app.intelligence.response import LLMResponse
from app.pipeline.build_pipeline import BuildPipeline
from app.sources.github import GitHubFile


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


def test_wiki_revision_retries_without_auth_when_authenticated_git_fails(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                128,
                args[0],
                stderr="authentication failed",
            )
        return type("Result", (), {"stdout": "wiki-sha\tHEAD\n"})()

    monkeypatch.setattr("app.github.client.subprocess.run", fake_run)

    revision = GitHubClient(token="stale-token", repository="example/docs").wiki_revision()

    assert revision == "wiki-sha"
    assert len(calls) == 2
    assert "GIT_CONFIG_VALUE_0" in calls[0]["env"]
    assert "GIT_CONFIG_VALUE_0" not in calls[1]["env"]


def test_client_lists_building_wiki_modules(monkeypatch) -> None:
    def fake_clone(self, repository: str, checkout: Path, ref: str):
        checkout.mkdir(parents=True)
        (checkout / "Building-Alfresco.md").write_text("# Alfresco", encoding="utf-8")
        (checkout / "Building-Share.md").write_text("# Share", encoding="utf-8")
        (checkout / "Home.md").write_text("# Home", encoding="utf-8")

    monkeypatch.setattr(GitHubClient, "_clone_wiki_repository", fake_clone)

    client = GitHubClient(repository="example/docs")

    assert client.list_wiki_modules() == ["Building-Alfresco", "Building-Share"]


def test_client_creates_file_pull_request_from_current_base(monkeypatch) -> None:
    client = GitHubClient(token="write-token")
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    monkeypatch.setattr(
        client,
        "fetch_file",
        lambda repository, path, ref: GitHubFile(
            repository=repository,
            ref=ref or "master",
            path=path,
            filename="build.sh",
            github_url="https://github.com/example/scripts/blob/master/build.sh",
            content="echo original\n",
            sha="file-sha",
        ),
    )

    def fake_get(url: str):
        assert url.endswith("/git/ref/heads/master")
        return {"object": {"sha": "base-sha"}}

    def fake_request(method: str, url: str, payload=None):
        requests.append((method, url, payload))
        if url.endswith("/pulls"):
            return {"number": 7, "html_url": "https://github.com/example/scripts/pull/7"}
        return {}

    monkeypatch.setattr(client, "_get_json", fake_get)
    monkeypatch.setattr(client, "_request_json", fake_request)

    result = client.create_file_pull_request(
        repository="example/scripts",
        head_repository=None,
        base_branch="master",
        path="Bazel/9.2.0/build_bazel.sh",
        original_content="echo original\n",
        fixed_content="echo fixed\n",
        title="BuildGuard: fix Bazel build script",
        body="Generated fix.",
    )

    assert result["number"] == 7
    assert result["head_repository"] == "example/scripts"
    assert [request[0] for request in requests] == ["POST", "PUT", "POST"]
    assert requests[0][2]["sha"] == "base-sha"
    assert requests[1][2]["sha"] == "file-sha"
    assert requests[1][2]["content"]
