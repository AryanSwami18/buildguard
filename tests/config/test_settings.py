from app.config.settings import Settings


def test_settings_reads_source_and_future_provider_values(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPO", "example/docs")
    monkeypatch.setenv("GITHUB_BRANCH", "release")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "example/model")

    settings = Settings()

    assert settings.github_token == "test-token"
    assert settings.github_repo == "example/docs"
    assert settings.github_branch == "release"
    assert settings.openrouter_api_key == "test-key"
    assert settings.openrouter_model == "example/model"


def test_settings_uses_the_requested_openrouter_model_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    assert Settings().openrouter_model == "qwen/qwen3-coder:free"


def test_settings_uses_the_building_alfresco_repository_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_REPO", raising=False)

    assert Settings().github_repo == "AryanSwami18/docs"
