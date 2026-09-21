import json

import pytest

from app.intelligence.openrouter import OpenRouterProvider
from app.intelligence.request import LLMRequest


def test_openrouter_provider_sends_request_and_unwraps_json_fence(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "```json\n{\"action\": \"unknown\"}\n```"}}]
            }).encode()

    def fake_urlopen(request):
        captured["request"] = request
        return FakeResponse()

    monkeypatch.setattr("app.intelligence.openrouter.urlopen", fake_urlopen)
    provider = OpenRouterProvider("test-key", "example/model")
    response = provider.generate(LLMRequest(system_prompt="system", user_prompt="user"))

    sent = json.loads(captured["request"].data.decode())
    assert captured["request"].get_header("Authorization") == "Bearer test-key"
    assert sent["model"] == "example/model"
    assert sent["response_format"] == {"type": "json_object"}
    assert response.content == '{"action": "unknown"}'


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"", "OpenRouter returned an empty HTTP response"),
        (b"not json", "OpenRouter returned invalid JSON: not json"),
    ],
)
def test_openrouter_provider_reports_invalid_http_json(monkeypatch, body: bytes, message: str) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return body

    monkeypatch.setattr("app.intelligence.openrouter.urlopen", lambda request: FakeResponse())

    with pytest.raises(RuntimeError, match=message):
        OpenRouterProvider("test-key", "example/model").generate(
            LLMRequest(system_prompt="system", user_prompt="user")
        )


def test_openrouter_provider_rejects_empty_completion(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "   "}}]}).encode()

    monkeypatch.setattr("app.intelligence.openrouter.urlopen", lambda request: FakeResponse())

    with pytest.raises(RuntimeError, match="OpenRouter returned empty chat-completion content"):
        OpenRouterProvider("test-key", "example/model").generate(
            LLMRequest(system_prompt="system", user_prompt="user")
        )
