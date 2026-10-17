import hashlib
import hmac
import json

from app.config.settings import Settings
from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep
from app.service import (
    BASH_PARSER_VERSION,
    _load_bash_instruction,
    _load_wiki_instruction,
    _wiki_parser_version,
)
from app.sources.github import GitHubWikiPage
from app.storage.repository import ParsedSourceRepository
from app.webhook import _cache_jobs, _webhook_targets, verify_signature, WebhookTarget


def _instruction(source: str, name: str, version: str | None = None) -> BuildInstruction:
    step = BuildStep(
        id=f"{source}-1",
        order=1,
        action=ActionType.RUN_COMMAND,
        description="Run the build",
        context=SourceContext(source_type=source, source_name=name),
    )
    return BuildInstruction(
        name=name,
        source=source,
        version=version,
        steps=[step],
    )


class _CachedWikiClient:
    def wiki_revision(self, ref: str) -> str:
        return "wiki-sha"

    def fetch_module_wiki(self, module: str):
        raise AssertionError("cached Wiki output should avoid fetching the page")


class _CachedScriptClient:
    def find_build_script_reference(self, **kwargs):
        return "Alfresco/25.3.0/build_alfresco.sh", "script-sha"

    def fetch_file(self, **kwargs):
        raise AssertionError("cached Bash output should avoid fetching the script")


class _UnavailableRevisionWikiClient:
    def wiki_revision(self, ref: str) -> str:
        raise RuntimeError("revision service unavailable")

    def fetch_module_wiki(self, module: str) -> GitHubWikiPage:
        return GitHubWikiPage(
            repository="example/docs",
            ref="HEAD",
            path="Building-Alfresco.md",
            filename="Building-Alfresco.md",
            github_url="https://example.test/wiki/Building-Alfresco",
            content="# cached",
        )


def test_cached_source_is_reused_when_remote_revision_matches(tmp_path) -> None:
    settings = Settings(
        github_repo="example/docs",
        scripts_repo="example/scripts",
        buildguard_database_path=str(tmp_path / "buildguard.sqlite3"),
    )
    cache = ParsedSourceRepository(settings.buildguard_database_path)
    wiki = _instruction("wiki", "Building-Alfresco.md", "25.3.0")
    bash = _instruction("bash", "build_alfresco.sh", "25.3.0")
    cache.save_instruction(
        source_kind="wiki",
        repository="example/docs",
        path="Building-Alfresco.md",
        ref="HEAD",
        parser_version=_wiki_parser_version(settings),
        remote_revision="wiki-sha",
        content="# cached",
        instruction=wiki,
    )
    cache.save_instruction(
        source_kind="bash",
        repository="example/scripts",
        path="Alfresco/25.3.0/build_alfresco.sh",
        ref="master",
        parser_version=BASH_PARSER_VERSION,
        remote_revision="script-sha",
        content="echo cached",
        instruction=bash,
    )

    class NoParsePipeline:
        def from_github_wiki(self, *args, **kwargs):
            raise AssertionError("cached Wiki output should avoid AI parsing")

    wiki_result = _load_wiki_instruction(
        github=_CachedWikiClient(),
        pipeline=NoParsePipeline(),
        cache=cache,
        settings=settings,
        module="Building-Alfresco",
        notify=lambda message: None,
    )
    bash_result = _load_bash_instruction(
        github=_CachedScriptClient(),
        cache=cache,
        settings=settings,
        product="Alfresco",
        version="25.3.0",
        notify=lambda message: None,
    )

    assert wiki_result.model_dump() == wiki.model_dump()
    assert bash_result.model_dump() == bash.model_dump()
    assert cache.get_source_content(
        source_kind="bash",
        repository="example/scripts",
        path="Alfresco/25.3.0/build_alfresco.sh",
        ref="master",
        parser_version=BASH_PARSER_VERSION,
    ) == "echo cached"


def test_cached_wiki_is_reused_when_revision_check_fails_but_content_is_unchanged(
    tmp_path,
) -> None:
    settings = Settings(
        github_repo="example/docs",
        buildguard_database_path=str(tmp_path / "buildguard.sqlite3"),
    )
    cache = ParsedSourceRepository(settings.buildguard_database_path)
    wiki = _instruction("wiki", "Building-Alfresco.md", "25.3.0")
    cache.save_instruction(
        source_kind="wiki",
        repository="example/docs",
        path="Building-Alfresco.md",
        ref="HEAD",
        parser_version=_wiki_parser_version(settings),
        remote_revision="old-wiki-sha",
        content="# cached",
        instruction=wiki,
    )

    class NoParsePipeline:
        def from_github_wiki(self, *args, **kwargs):
            raise AssertionError("unchanged Wiki content should avoid AI parsing")

    result = _load_wiki_instruction(
        github=_UnavailableRevisionWikiClient(),
        pipeline=NoParsePipeline(),
        cache=cache,
        settings=settings,
        module="Building-Alfresco",
        notify=lambda message: None,
    )

    assert result.model_dump() == wiki.model_dump()


def test_webhook_jobs_cover_wiki_changes_and_new_script_files() -> None:
    settings = Settings(
        github_repo="example/docs",
        scripts_repo="example/scripts",
    )
    wiki_payload = {
        "repository": {"full_name": "example/docs"},
        "pages": [{"page_name": "Building-Alfresco", "action": "created"}],
    }
    script_payload = {
        "repository": {"full_name": "example/scripts"},
        "commits": [{"added": ["Alfresco/25.3.0/build_alfresco.sh"]}],
    }

    assert _cache_jobs("gollum", wiki_payload, settings) == {
        ("Alfresco", True, False)
    }
    assert _cache_jobs("push", script_payload, settings) == {
        ("Alfresco", False, True)
    }


def test_script_webhook_target_preserves_package_version_and_branch() -> None:
    settings = Settings(
        github_repo="example/docs",
        scripts_repo="example/scripts",
        scripts_branch="master",
    )
    payload = {
        "repository": {"full_name": "example/scripts"},
        "ref": "refs/heads/master",
        "commits": [
            {
                "modified": ["Alfresco/26.0.0/build_alfresco.sh"],
                "added": ["Bazel/9.2.0/build_bazel.sh"],
            }
        ],
    }

    assert _webhook_targets("push", payload, settings) == {
        WebhookTarget("Alfresco", "26.0.0", False, True),
        WebhookTarget("Bazel", "9.2.0", False, True),
    }
    payload["ref"] = "refs/heads/feature"
    assert _webhook_targets("push", payload, settings) == set()


def test_script_webhook_maps_repository_alias_to_wiki_package() -> None:
    settings = Settings(
        github_repo="example/docs",
        scripts_repo="example/scripts",
        scripts_branch="master",
    )
    payload = {
        "repository": {"full_name": "example/scripts"},
        "ref": "refs/heads/master",
        "commits": [
            {"modified": ["AntLR/4.13.2/build_antlr4.sh"]},
        ],
    }

    assert _webhook_targets("push", payload, settings) == {
        WebhookTarget("ANTLR-4", "4.13.2", False, True),
    }


def test_webhook_signature_verification() -> None:
    secret = "test-secret"
    body = json.dumps({"zen": "keep it simple"}).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert verify_signature(secret, body, signature)
    assert not verify_signature(secret, body + b"!", signature)
