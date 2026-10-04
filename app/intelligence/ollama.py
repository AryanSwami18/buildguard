"""Ollama implementation of the provider-neutral LLM interface."""

import json
from copy import deepcopy
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .request import LLMRequest
from .response import LLMResponse


class OllamaProvider:
    """Send schema-constrained generation requests to a local Ollama server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 600.0,
        max_output_tokens: int = 8192,
        context_tokens: int = 16384,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.context_tokens = context_tokens

    def generate(self, request: LLMRequest) -> LLMResponse:
        system_prompt = request.system_prompt
        if request.structured_output_instructions:
            system_prompt = (
                f"{system_prompt}\n\n{request.structured_output_instructions}"
            )

        format_schema = _ollama_schema(
            _expected_output_schema(request.user_prompt)
        )
        if format_schema is not None and _is_chunked_comparison(request.user_prompt):
            _require_comparison_ids(format_schema)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "format": format_schema or "json",
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": self.context_tokens,
                "num_predict": self.max_output_tokens,
            },
        }
        raw_response = self._post(payload)

        try:
            message = raw_response["message"]
            content = message["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned no chat message content") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned empty chat message content")

        if raw_response.get("done_reason") == "length":
            generated_tokens = raw_response.get("eval_count")
            raise RuntimeError(
                "Ollama response exceeded the output limit "
                f"({self.max_output_tokens} tokens); increase "
                f"OLLAMA_MAX_OUTPUT_TOKENS. Ollama generated {generated_tokens} tokens."
            )

        try:
            json.loads(content)
        except json.JSONDecodeError as error:
            excerpt_start = max(0, error.pos - 100)
            excerpt_end = min(len(content), error.pos + 100)
            excerpt = " ".join(content[excerpt_start:excerpt_end].split())
            if excerpt_start:
                excerpt = "..." + excerpt
            if excerpt_end < len(content):
                excerpt += "..."
            raise RuntimeError(
                "Ollama returned invalid JSON "
                f"(line {error.lineno}, column {error.colno}; {error.msg}) "
                f"near: {excerpt}"
            ) from error
        return LLMResponse(content=content)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = f"{self.base_url}/api/chat"
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - endpoint comes from local settings.
                body = response.read().decode("utf-8", errors="replace")
        except TimeoutError as error:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout_seconds:g} seconds"
            ) from error
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 404 and "model" in detail.lower():
                raise RuntimeError(
                    f"Ollama model '{self.model}' is not installed. Run "
                    f"'ollama pull {self.model}'."
                ) from error
            raise RuntimeError(f"Ollama request failed ({error.code}): {detail}") from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise RuntimeError(
                    f"Ollama request timed out after {self.timeout_seconds:g} seconds"
                ) from error
            raise RuntimeError(
                f"Could not reach Ollama at {endpoint}. Start the local server with "
                f"'ollama serve'. Details: {error.reason}"
            ) from error

        if not body.strip():
            raise RuntimeError("Ollama returned an empty HTTP response")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as error:
            preview = " ".join(body.split())[:200]
            raise RuntimeError(f"Ollama returned invalid response JSON: {preview}") from error
        if not isinstance(data, dict):
            raise RuntimeError("Ollama returned an unexpected response")
        return data


def _expected_output_schema(user_prompt: str) -> dict[str, Any] | None:
    """Use the JSON Schema already embedded in BuildGuard's prompt, if available."""
    try:
        prompt_data = json.loads(user_prompt)
    except json.JSONDecodeError:
        return None
    if not isinstance(prompt_data, dict):
        return None
    schema = prompt_data.get("expected_output")
    return schema if isinstance(schema, dict) else None


def _ollama_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Hoist nested Pydantic definitions for Ollama's root-level JSON Schema refs."""
    if schema is None:
        return None
    normalized = deepcopy(schema)
    definitions: dict[str, Any] = {}
    pending: list[Any] = [normalized]

    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            nested_definitions = node.pop("$defs", None)
            if isinstance(nested_definitions, dict):
                definitions.update(nested_definitions)
                pending.extend(nested_definitions.values())
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)

    if definitions:
        normalized["$defs"] = definitions
    return normalized


def _is_chunked_comparison(user_prompt: str) -> bool:
    try:
        prompt_data = json.loads(user_prompt)
    except json.JSONDecodeError:
        return False
    return isinstance(prompt_data, dict) and {
        "wiki_steps",
        "available_buildscript_steps",
    }.issubset(prompt_data)


def _require_comparison_ids(schema: dict[str, Any]) -> None:
    """Make every batched Wiki row carry an ID, even when Bash is missing."""
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return
    match_schema = definitions.get("AIComparisonMatch")
    if not isinstance(match_schema, dict):
        return
    properties = match_schema.get("properties")
    if not isinstance(properties, dict):
        return
    properties["wiki_step_id"] = {"type": "string"}
    # Missing and out-of-scope rows use an empty string; the pipeline converts
    # that sentinel back to None before validating the shared model.
    properties["bash_step_id"] = {"type": "string"}
