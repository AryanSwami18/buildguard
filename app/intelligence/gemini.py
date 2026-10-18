"""Google Gemini implementation of the provider-neutral LLM interface."""

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .request import LLMRequest
from .response import LLMResponse

TRANSIENT_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)
MAX_RETRY_AFTER_SECONDS = 30.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class _RetryableGeminiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        detail: str,
        retry_after: float | None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.retry_after = retry_after


class GeminiProvider:
    """Send JSON-mode content generation requests to the Gemini API."""

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 90.0,
        max_output_tokens: int = 32768,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens

    def generate(self, request: LLMRequest) -> LLMResponse:
        output_tokens = request.max_output_tokens or self.max_output_tokens
        system_prompt = request.system_prompt
        if request.structured_output_instructions:
            system_prompt = (
                f"{system_prompt}\n\n{request.structured_output_instructions}"
            )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": output_tokens,
                "responseMimeType": "application/json",
            },
        }

        for attempt in range(len(TRANSIENT_RETRY_DELAYS_SECONDS) + 1):
            try:
                raw_response = self._post(payload)
            except _RetryableGeminiError as error:
                if attempt == len(TRANSIENT_RETRY_DELAYS_SECONDS):
                    raise RuntimeError(
                        "Gemini request remained temporarily unavailable after "
                        f"{attempt + 1} attempts ({error.status_code}): {error}"
                    ) from error
                delay = error.retry_after
                if delay is None:
                    delay = TRANSIENT_RETRY_DELAYS_SECONDS[attempt]
                time.sleep(min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS))
            else:
                content = self._generated_text(raw_response, request.max_output_tokens)
                try:
                    json.loads(content)
                except json.JSONDecodeError as error:
                    if attempt == len(TRANSIENT_RETRY_DELAYS_SECONDS):
                        excerpt_start = max(0, error.pos - 100)
                        excerpt_end = min(len(content), error.pos + 100)
                        excerpt = " ".join(
                            content[excerpt_start:excerpt_end].split()
                        )
                        if excerpt_start:
                            excerpt = "..." + excerpt
                        if excerpt_end < len(content):
                            excerpt += "..."
                        raise RuntimeError(
                            "Gemini returned invalid JSON after "
                            f"{attempt + 1} attempts (line {error.lineno}, "
                            f"column {error.colno}; {error.msg}) near: {excerpt}"
                        ) from error
                    time.sleep(TRANSIENT_RETRY_DELAYS_SECONDS[attempt])
                else:
                    return LLMResponse(content=content)

        raise RuntimeError("Gemini request failed without a response")

    def _generated_text(
        self,
        raw_response: dict[str, Any],
        output_tokens: int | None = None,
    ) -> str:
        try:
            candidate = raw_response["candidates"][0]
        except (KeyError, IndexError, TypeError) as error:
            feedback = raw_response.get("promptFeedback", {})
            block_reason = (
                feedback.get("blockReason") if isinstance(feedback, dict) else None
            )
            if block_reason:
                raise RuntimeError(
                    f"Gemini blocked the prompt (reason: {block_reason})"
                ) from error
            raise RuntimeError("Gemini returned no generated candidate") from error

        finish_reason = candidate.get("finishReason")
        if finish_reason == "MAX_TOKENS":
            limit = output_tokens or self.max_output_tokens
            limit_hint = (
                "increase the comparison output budget"
                if output_tokens is not None
                else "increase GEMINI_MAX_OUTPUT_TOKENS"
            )
            raise RuntimeError(
                "Gemini response exceeded the output limit "
                f"({limit} tokens); {limit_hint}"
            )
        if finish_reason and finish_reason not in {
            "STOP",
            "FINISH_REASON_UNSPECIFIED",
        }:
            raise RuntimeError(f"Gemini generation ended early (reason: {finish_reason})")

        try:
            parts = candidate["content"]["parts"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Gemini returned no generated text") from error
        if not isinstance(parts, list):
            raise RuntimeError("Gemini returned malformed generated text")
        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and part.get("thought") is not True
            and isinstance(part.get("text"), str)
        ]
        content = _strip_json_fence("".join(text_parts))
        if not content:
            raise RuntimeError("Gemini returned empty generated text")
        return content

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = quote(self.model.removeprefix("models/"), safe="-_.")
        request = Request(
            f"{self.endpoint}/{model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": self.api_key,
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
                f"Gemini request timed out after {self.timeout_seconds:g} seconds"
            ) from error
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in RETRYABLE_STATUS_CODES:
                raise _RetryableGeminiError(
                    error.code,
                    detail,
                    _retry_after_seconds(error.headers.get("Retry-After")),
                ) from error
            if error.code in {401, 403}:
                raise RuntimeError(
                    f"Gemini request failed ({error.code}); check GEMINI_API_KEY "
                    f"and Gemini API access: {detail}"
                ) from error
            raise RuntimeError(f"Gemini request failed ({error.code}): {detail}") from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise RuntimeError(
                    f"Gemini request timed out after {self.timeout_seconds:g} seconds"
                ) from error
            raise RuntimeError(f"Could not reach Gemini: {error.reason}") from error

        if not body.strip():
            raise RuntimeError("Gemini returned an empty HTTP response")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as error:
            preview = " ".join(body.split())[:200]
            raise RuntimeError(f"Gemini returned invalid JSON: {preview}") from error
        if not isinstance(data, dict):
            raise RuntimeError("Gemini returned an unexpected response")
        return data


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _strip_json_fence(content: str) -> str:
    """Accept JSON returned inside an occasional Markdown code fence."""
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped
