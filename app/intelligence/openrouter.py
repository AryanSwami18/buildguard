"""OpenRouter implementation of the provider-neutral LLM interface."""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .request import LLMRequest
from .response import LLMResponse


class OpenRouterProvider:
    """Send structured extraction requests to an OpenRouter chat-completions model."""

    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        system_prompt = request.system_prompt
        if request.structured_output_instructions:
            system_prompt = f"{system_prompt}\n\n{request.structured_output_instructions}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        raw_response = self._post(payload)
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenRouter returned no chat-completion content") from error
        normalized_content = _message_content_text(content)
        if not normalized_content:
            raise RuntimeError("OpenRouter returned empty chat-completion content")
        return LLMResponse(content=normalized_content)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request) as response:  # noqa: S310 - endpoint is a constant.
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach OpenRouter: {error.reason}") from error

        if not body.strip():
            raise RuntimeError("OpenRouter returned an empty HTTP response")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as error:
            preview = " ".join(body.split())[:200]
            raise RuntimeError(f"OpenRouter returned invalid JSON: {preview}") from error
        if not isinstance(data, dict):
            raise RuntimeError("OpenRouter returned an unexpected response")
        return data


def _json_object_text(content: str) -> str:
    """Accept JSON returned inside an occasional Markdown code fence."""
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def _message_content_text(content: Any) -> str:
    """Normalize string and text-part chat completion content."""
    if isinstance(content, str):
        return _json_object_text(content)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return _json_object_text(content["text"])
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return _json_object_text("".join(text_parts))
    raise RuntimeError("OpenRouter returned non-text chat-completion content")
