import json
from pathlib import Path
from typing import Any

from app.intelligence.interface import LLMProvider
from app.intelligence.request import LLMRequest
from app.models.build import BuildInstruction
from app.models.step import BuildStep
from app.parsers.bash_normalizer import normalize_bash
from app.parsers.bash_parser import parse_script
from app.parsers.wiki_parser import parse_wiki
from app.pipeline.context import neighbouring_text, step_context
from app.sources.github import GitHubSource, GitHubWikiPage

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


class BuildPipeline:
    """Turns source structure plus context into validated common build steps."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def from_bash(self, script: str, name: str = "script") -> BuildInstruction:
        parsed = parse_script(script)
        node_texts = [node.text for node in parsed.nodes]
        step_ids = [f"bash-{index + 1}" for index in range(len(parsed.nodes))]
        steps: list[BuildStep] = []
        for index, node in enumerate(parsed.nodes):
            context = step_context(
                source_type="bash",
                source_name=name,
                step_ids=step_ids,
                index=index,
                location=f"characters {node.position_start}-{node.position_end}",
                line_start=_line_number(script, node.position_start),
                line_end=_line_number(script, node.position_end),
                surrounding_text=neighbouring_text(node_texts, index),
            )
            request = self._request(
                prompt_name="step_extraction.txt",
                source={"script_ast": normalize_bash(parsed), "current_node": node.model_dump()},
                context=context.model_dump(mode="json") | {
                    "current_step_id": step_ids[index],
                    "known_step_ids": step_ids[:index],
                },
            )
            response_data = _provider_json(self.llm.generate(request).content, step_ids[index])
            steps.append(self._validate_step(response_data, step_ids[index], index + 1, context))
        return BuildInstruction(name=name, source="bash", steps=steps)

    def from_wiki(self, markdown: str, name: str = "wiki") -> BuildInstruction:
        document = parse_wiki(markdown, name)
        request = self._request(
            prompt_name="wiki_extraction.txt",
            source={"source": "wiki", "document": document.model_dump(mode="json")},
            context={
                "source_type": "wiki",
                "source_name": name,
                "document_name": document.name,
                "known_step_ids": [],
            },
            expected_output=BuildInstruction.model_json_schema(),
            flat_source=True,
        )
        response_data = _provider_json(self.llm.generate(request).content, "wiki-1")

        # The Wiki contract is one document in and one BuildInstruction out.
        # Keep a narrow compatibility path for providers written against the
        # original block-by-block API; valid new responses always take the
        # branch above and therefore make exactly one provider call.
        if "steps" in response_data:
            return self._validate_wiki_instruction(response_data, document.name)
        if {"action", "description"}.issubset(response_data):
            return self._from_wiki_blocks_legacy(document, response_data)
        raise RuntimeError("LLM returned an object without a BuildInstruction steps field for wiki")

    def from_github_wiki(self, page: GitHubWikiPage) -> BuildInstruction:
        """Turn GitHub Wiki Markdown into steps while retaining its source location."""
        instruction = self.from_wiki(page.content, page.filename)
        instruction.source_metadata = GitHubSource.model_validate(page.model_dump())
        return instruction

    def _from_wiki_blocks_legacy(
        self, document: Any, first_response: dict[str, Any]
    ) -> BuildInstruction:
        """Support callers that still return one BuildStep per Wiki block.

        This is intentionally not the normal extraction path. It lets older
        provider adapters continue working while they adopt the complete Wiki
        document contract.
        """
        texts = [block.text for block in document.blocks]
        step_ids = [f"wiki-{index + 1}" for index in range(len(document.blocks))]
        steps: list[BuildStep] = []
        for index, block in enumerate(document.blocks):
            context = step_context(
                source_type="wiki",
                source_name=document.name,
                step_ids=step_ids,
                index=index,
                section=block.section,
                subsection=block.subsection,
                location=f"lines {block.line_start}-{block.line_end}",
                line_start=block.line_start,
                line_end=block.line_end,
                surrounding_text=neighbouring_text(texts, index),
            )
            response_data = first_response if index == 0 else _provider_json(
                self.llm.generate(
                    self._request(
                        prompt_name="wiki_extraction.txt",
                        source={
                            "source": "wiki",
                            "document": document.model_dump(mode="json"),
                            "current_block": block.model_dump(mode="json"),
                        },
                        context=context.model_dump(mode="json")
                        | {
                            "current_step_id": step_ids[index],
                            "known_step_ids": step_ids[:index],
                        },
                    )
                ).content,
                step_ids[index],
            )
            steps.append(self._validate_step(response_data, step_ids[index], index + 1, context))
        return BuildInstruction(name=document.name, source="wiki", steps=steps)

    @staticmethod
    def _validate_wiki_instruction(data: dict[str, Any], document_name: str) -> BuildInstruction:
        """Validate one whole-document response and keep source-owned fields trusted."""
        payload = dict(data)
        payload["name"] = document_name
        payload["source"] = "wiki"
        payload.pop("source_metadata", None)

        raw_steps = payload.get("steps", [])
        original_ids = {
            raw_step.get("id"): f"wiki-{index}"
            for index, raw_step in enumerate(raw_steps, start=1)
            if isinstance(raw_step, dict) and isinstance(raw_step.get("id"), str)
        }
        normalized_steps: list[dict[str, Any]] = []
        for index, raw_step in enumerate(raw_steps, start=1):
            step = dict(raw_step)
            step["id"] = f"wiki-{index}"
            step["order"] = index
            step["depends_on"] = [
                original_ids.get(dependency, dependency)
                for dependency in step.get("depends_on", [])
            ]
            context = dict(step.get("context") or {})
            context["source_type"] = "wiki"
            context["source_name"] = document_name
            step["context"] = context
            normalized_steps.append(step)
        payload["steps"] = normalized_steps
        return BuildInstruction.model_validate(payload)

    @staticmethod
    def _validate_step(data: dict[str, Any], step_id: str, order: int, context: Any) -> BuildStep:
        """Validate LLM semantics while keeping source-owned metadata accurate."""
        payload = dict(data)
        payload.update(id=step_id, order=order, context=context.model_dump(mode="json"))
        return BuildStep.model_validate(payload)

    @staticmethod
    def _request(
        prompt_name: str,
        source: dict[str, Any],
        context: dict[str, Any],
        expected_output: dict[str, Any] | None = None,
        flat_source: bool = False,
    ) -> LLMRequest:
        system_prompt = (PROMPTS / prompt_name).read_text(encoding="utf-8")
        if flat_source:
            request_source: dict[str, Any] = {
                "source": source["source"],
                "document": source["document"],
            }
        else:
            request_source = {"source": source, "context": context}
        user_prompt = json.dumps(
            request_source
            | {"context": context, "expected_output": expected_output or BuildStep.model_json_schema()},
            indent=2,
        )
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


def _line_number(text: str, character_position: int) -> int:
    return text.count("\n", 0, character_position) + 1


def _provider_json(content: str, step_id: str) -> dict[str, Any]:
    """Decode one provider response with an actionable error message."""
    if not content.strip():
        raise RuntimeError(f"LLM returned empty JSON for step {step_id}")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        preview = " ".join(content.split())[:200]
        raise RuntimeError(f"LLM returned invalid JSON for step {step_id}: {preview}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"LLM returned a JSON value instead of an object for step {step_id}")
    return data
