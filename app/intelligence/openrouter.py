"""OpenRouter implementation of the provider-neutral LLM interface."""

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .request import LLMRequest
from .response import LLMResponse

FREE_MODEL_FALLBACKS = (
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
)
RATE_LIMIT_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)
MAX_RETRY_AFTER_SECONDS = 30.0


class _UpstreamRateLimitError(RuntimeError):
    def __init__(self, detail: str, retry_after: float | None) -> None:
        super().__init__(detail)
        self.retry_after = retry_after


class OpenRouterProvider:
    """Send structured extraction requests to an OpenRouter chat-completions model."""

    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 90.0,
        max_tokens: int = 32768,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def generate(self, request: LLMRequest) -> LLMResponse:
        system_prompt = request.system_prompt
        if request.structured_output_instructions:
            system_prompt = f"{system_prompt}\n\n{request.structured_output_instructions}"
        payload = {
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.model.endswith(":free"):
            payload["models"] = _free_model_fallbacks(self.model)
            payload["reasoning_effort"] = "minimal"
        else:
            payload["model"] = self.model
        for attempt in range(len(RATE_LIMIT_RETRY_DELAYS_SECONDS) + 1):
            try:
                raw_response = self._post(payload)
                break
            except _UpstreamRateLimitError as error:
                if attempt == len(RATE_LIMIT_RETRY_DELAYS_SECONDS):
                    raise RuntimeError(
                        "OpenRouter request remained rate-limited after "
                        f"{attempt + 1} attempts: {error}"
                    ) from error
                delay = error.retry_after
                if delay is None:
                    delay = RATE_LIMIT_RETRY_DELAYS_SECONDS[attempt]
                time.sleep(min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS))
        try:
            choice = raw_response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenRouter returned no chat-completion content") from error
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "OpenRouter response exceeded the completion limit "
                f"({self.max_tokens} tokens); increase OPENROUTER_MAX_TOKENS"
            )
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
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - endpoint is a constant.
                body = response.read().decode("utf-8", errors="replace")
        except TimeoutError as error:
            raise RuntimeError(
                f"OpenRouter request timed out after {self.timeout_seconds:g} seconds"
            ) from error
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 429:
                raise _UpstreamRateLimitError(
                    detail,
                    _retry_after_seconds(error.headers.get("Retry-After")),
                ) from error
            raise RuntimeError(f"OpenRouter request failed ({error.code}): {detail}") from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise RuntimeError(
                    f"OpenRouter request timed out after {self.timeout_seconds:g} seconds"
                ) from error
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


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _free_model_fallbacks(primary: str) -> list[str]:
    """Keep free requests moving when one model's shared pool is rate-limited."""
    return list(dict.fromkeys((primary, *FREE_MODEL_FALLBACKS)))


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
