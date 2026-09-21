import json

from app.docs_cli import main
from app.intelligence.response import LLMResponse
from app.sources.github import GitHubWikiPage


def test_docs_command_prints_full_structural_parse(monkeypatch, capsys) -> None:
    page = GitHubWikiPage(
        repository="example/docs",
        ref="main",
        path="Module.md",
        filename="Module.md",
        github_url="https://github.com/example/docs/wiki/Module",
        content="# Install\n\n1. Install Java.\n",
    )

    monkeypatch.setattr(
        "app.docs_cli.GitHubClient.from_settings",
        lambda settings: type("Client", (), {"fetch_module_wiki": lambda self, module: page})(),
    )

    assert main(["Module"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source"]["path"] == "Module.md"
    assert output["document"]["headings"] == [{"level": 1, "text": "Install", "line": 1}]
    assert output["document"]["blocks"][0]["kind"] == "numbered_step"


def test_docs_command_uses_building_alfresco_by_default(monkeypatch, capsys) -> None:
    page = GitHubWikiPage(
        repository="AryanSwami18/docs",
        ref="HEAD",
        path="Building-Alfresco.md",
        filename="Building-Alfresco.md",
        github_url="https://github.com/AryanSwami18/docs/wiki/Building-Alfresco",
        content="# Building Alfresco\n",
    )
    requested_modules: list[str] = []

    def fetch_module_wiki(self, module: str) -> GitHubWikiPage:
        requested_modules.append(module)
        return page

    monkeypatch.setattr(
        "app.docs_cli.GitHubClient.from_settings",
        lambda settings: type("Client", (), {"fetch_module_wiki": fetch_module_wiki})(),
    )

    assert main([]) == 0
    capsys.readouterr()
    assert requested_modules == ["Building-Alfresco"]


def test_docs_command_shows_the_ai_instruction_when_requested(monkeypatch, capsys) -> None:
    page = GitHubWikiPage(
        repository="example/docs",
        ref="main",
        path="Module.md",
        filename="Module.md",
        github_url="https://github.com/example/docs/wiki/Module",
        content="# Install\n\nInstall Java.\n",
    )

    monkeypatch.setattr(
        "app.docs_cli.GitHubClient.from_settings",
        lambda settings: type("Client", (), {"fetch_module_wiki": lambda self, module: page})(),
    )

    class FakeProvider:
        def generate(self, request):
            return LLMResponse(
                content=json.dumps(
                    {
                        "name": "Module.md",
                        "source": "wiki",
                        "version": None,
                        "steps": [],
                    }
                )
            )

    monkeypatch.setattr("app.docs_cli.create_provider", lambda settings: FakeProvider())

    assert main(["Module", "--with-ai"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["extracted_steps"]["name"] == "Module.md"
    assert output["extracted_steps"]["steps"] == []
