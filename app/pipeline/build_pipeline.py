import json
from pathlib import Path
from typing import Any, Callable

from app.intelligence.interface import LLMProvider
from app.intelligence.request import LLMRequest
from app.models.build import BuildInstruction
from app.models.step import ActionType, BuildStep
from app.models.validation import (
    AIComparisonMatch,
    AIComparisonResult,
    DiscrepancyCategory,
    MatchStatus,
    StepMatch,
    ValidationResult,
)
from app.matching.semantic_matcher import normalize_validation_result
from app.parsers.bash_normalizer import normalize_bash
from app.parsers.bash_parser import parse_script
from app.parsers.semantic_parser import parse_semantic_facts
from app.parsers.wiki_parser import parse_wiki
from app.pipeline.context import neighbouring_text, step_context
from app.sources.github import GitHubSource, GitHubWikiPage

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
WIKI_EXTRACTION_CHUNK_CHARS = 6000
COMPARISON_CHUNK_STEPS = 8
COMPARISON_SINGLE_REQUEST_MAX_STEPS = 24
COMPARISON_SINGLE_REQUEST_MAX_CHARS = 18000


def _comparison_bash_ids(match: AIComparisonMatch) -> list[str]:
    """Normalize the legacy primary ID and the multi-step ID list."""
    ids = list(match.bash_step_ids)
    if match.bash_step_id is not None:
        ids = [match.bash_step_id] + [
            bash_id
            for bash_id in ids
            if bash_id != match.bash_step_id
        ]
    return ids


def _is_reusable_parameterized_step(step: BuildStep) -> bool:
    """Allow one parameterized package/config step to cover Wiki variants."""
    commands = " ".join(step.commands).lower()
    if step.action == ActionType.INSTALL_PACKAGE:
        values = " ".join(step.packages).lower() + " " + commands
        return any(
            marker in values
            for marker in ("${jver}", "$jver", "${allowerasing}")
        )
    if step.action == ActionType.SET_VARIABLE:
        return "java_home" in commands and any(
            marker in commands for marker in ("${jver}", "$jver")
        )
    if step.action == ActionType.DOWNLOAD:
        return (
            (step.context.subsection or "").lower() == "installjava"
            and bool(step.context.runtime_options.get("JAVA_PROVIDED"))
        )
    return False


def _comparison_batch_issue(
    result: AIComparisonResult,
    expected_wiki_ids: set[str],
    available_bash_ids: set[str],
    used_bash_ids: set[str],
    bash_by_id: dict[str, BuildStep],
) -> tuple[str | None, str] | None:
    """Return a source-ID contract violation before accepting a batch."""
    seen_wiki_ids: set[str] = set()
    seen_bash_ids: set[str] = set()
    malformed_wiki_rows = False

    for match in result.matches:
        wiki_id = match.wiki_step_id
        if wiki_id not in expected_wiki_ids:
            malformed_wiki_rows = True
            continue
        if wiki_id in seen_wiki_ids:
            malformed_wiki_rows = True
            continue
        if match.status == MatchStatus.EXTRA:
            return (
                None,
                "the batch included an EXTRA entry; extras are derived by the app",
            )

        bash_ids = _comparison_bash_ids(match)
        if len(bash_ids) != len(set(bash_ids)):
            return wiki_id, f"Wiki step {wiki_id} repeats a Bash ID"
        if match.status in {MatchStatus.MATCHED, MatchStatus.CONFLICT} and not bash_ids:
            return wiki_id, f"Wiki step {wiki_id} is {match.status.value} without a Bash ID"
        if match.status in {MatchStatus.MISSING, MatchStatus.OUT_OF_SCOPE} and bash_ids:
            return wiki_id, f"Wiki step {wiki_id} is {match.status.value} but includes Bash IDs"

        for bash_id in bash_ids:
            if bash_id not in available_bash_ids:
                return wiki_id, f"Bash ID {bash_id} is not available in this batch"
            if (
                bash_id in used_bash_ids or bash_id in seen_bash_ids
            ) and not _is_reusable_parameterized_step(bash_by_id[bash_id]):
                return wiki_id, f"Bash ID {bash_id} is assigned to more than one Wiki step"
            seen_bash_ids.add(bash_id)
        seen_wiki_ids.add(wiki_id)

    omitted_wiki_ids = expected_wiki_ids - seen_wiki_ids
    if omitted_wiki_ids:
        omitted = ", ".join(sorted(omitted_wiki_ids))
        return sorted(omitted_wiki_ids)[0], f"the response omitted Wiki step(s): {omitted}"
    if malformed_wiki_rows:
        return None, "the response used an unknown or duplicate Wiki step ID"
    return None


def _comparison_batch_result(content: str, batch_index: int) -> AIComparisonResult:
    """Decode and validate a comparison response before checking its source IDs."""
    response = _provider_json_value(content, f"comparison-batch-{batch_index}")
    if isinstance(response, list):
        response = {"summary": "", "matches": response}
    if not isinstance(response, dict):
        raise RuntimeError(
            f"LLM returned an invalid result for comparison batch {batch_index}"
        )
    match_payloads = response.get("matches")
    if isinstance(match_payloads, list):
        for match_payload in match_payloads:
            if not isinstance(match_payload, dict):
                continue
            if match_payload.get("bash_step_id") == "":
                match_payload["bash_step_id"] = None
            if not match_payload.get("explanation"):
                match_payload["explanation"] = (
                    "The comparison provider returned no explanation; "
                    "classification was completed from the supplied source fields."
                )
    response.setdefault("summary", "")
    try:
        return AIComparisonResult.model_validate(response)
    except ValueError as error:
        raise RuntimeError(
            f"LLM returned an invalid result for comparison batch {batch_index}: {error}"
        ) from error


class BuildPipeline:
    """Turns source structure plus context into validated common build steps."""

    def __init__(
        self,
        llm: LLMProvider,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.progress = progress

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
        if sum(len(block.text) for block in document.blocks) > WIKI_EXTRACTION_CHUNK_CHARS:
            return self._from_wiki_chunks(document)
        semantic_facts = parse_semantic_facts(markdown)

        request = self._request(
            prompt_name="wiki_extraction.txt",
            source={
                "source": "wiki",
                "document": document.model_dump(
                    mode="json",
                    exclude={"content"},
                ),
                "semantic_facts": semantic_facts.model_dump(mode="json"),
            },
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
            return self._validate_wiki_instruction(response_data, document)
        if {"action", "description"}.issubset(response_data):
            return self._from_wiki_blocks_legacy(document, response_data)
        raise RuntimeError("LLM returned an object without a BuildInstruction steps field for wiki")

    def _from_wiki_chunks(self, document: Any) -> BuildInstruction:
        """Extract large Wiki pages in bounded requests and merge their steps."""
        chunks = _wiki_block_chunks(document, WIKI_EXTRACTION_CHUNK_CHARS)
        all_steps: list[dict[str, Any]] = []
        output_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": BuildStep.model_json_schema(),
                }
            },
            "required": ["steps"],
        }

        for chunk_index, blocks in enumerate(chunks, start=1):
            if self.progress is not None:
                self.progress(
                    f"Extracting Wiki chunk {chunk_index}/{len(chunks)} with AI..."
                )
            first_line = min(block["line_start"] for block in blocks)
            last_line = max(block["line_end"] for block in blocks)
            headings = [
                heading.model_dump(mode="json")
                for heading in document.headings
                if first_line <= heading.line <= last_line
            ]
            chunk_text = "\n".join(block["text"] for block in blocks)
            chunk_facts = parse_semantic_facts(chunk_text)
            request = self._request(
                prompt_name="wiki_extraction.txt",
                source={
                    "source": "wiki",
                    "document": {
                        "name": document.name,
                        "headings": headings,
                        "blocks": blocks,
                    },
                    "semantic_facts": chunk_facts.model_dump(mode="json"),
                },
                context={
                    "source_type": "wiki",
                    "source_name": document.name,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "known_step_ids": [
                        f"wiki-{index}"
                        for index in range(1, len(all_steps) + 1)
                    ],
                },
                expected_output=output_schema,
                flat_source=True,
            )
            response_value = _provider_json_value(
                self.llm.generate(request).content,
                f"wiki-chunk-{chunk_index}",
            )
            if isinstance(response_value, list):
                raw_steps = response_value
            elif isinstance(response_value, dict):
                raw_steps = response_value.get("steps")
                if raw_steps is None and {"action", "description"}.issubset(response_value):
                    raw_steps = [response_value]
            else:
                raw_steps = None
            if not isinstance(raw_steps, list):
                raise RuntimeError(
                    f"LLM returned no steps array for Wiki chunk {chunk_index}"
                )

            generated_ids = {
                step.get("id"): f"chunk-{chunk_index}-step-{step_index}"
                for step_index, step in enumerate(raw_steps, start=1)
                if isinstance(step, dict) and isinstance(step.get("id"), str)
            }
            for step_index, raw_step in enumerate(raw_steps, start=1):
                if not isinstance(raw_step, dict):
                    raise RuntimeError(
                        f"LLM returned an invalid step in Wiki chunk {chunk_index}"
                    )
                step = dict(raw_step)
                step["id"] = f"chunk-{chunk_index}-step-{step_index}"
                step["depends_on"] = [
                    generated_ids.get(dependency, dependency)
                    for dependency in (step.get("depends_on") or [])
                ]
                all_steps.append(step)

        return self._validate_wiki_instruction(
            {"steps": all_steps},
            document,
        )

    def from_github_wiki(
        self,
        page: GitHubWikiPage,
        version: str | None = None,
    ) -> BuildInstruction:
        """Parse a GitHub Wiki page into a BuildInstruction."""

        instruction = self.from_wiki(
            page.content,
            page.filename,
        )

        instruction.source_metadata = (
            GitHubSource.model_validate(
                page.model_dump()
            )
        )

        if version is not None:
            instruction.version = version

        return instruction

    def compare(
        self,
        wiki: BuildInstruction,
        bash: BuildInstruction,
    ) -> ValidationResult:
        """Ask the configured AI provider to compare extracted instructions."""
        wiki_steps = [_comparison_step(step) for step in wiki.steps]
        bash_steps = [_comparison_step(step) for step in bash.steps]
        estimated_size = len(json.dumps({"wiki": wiki_steps, "bash": bash_steps}))
        if (
            len(wiki_steps) + len(bash_steps) > COMPARISON_SINGLE_REQUEST_MAX_STEPS
            or estimated_size > COMPARISON_SINGLE_REQUEST_MAX_CHARS
        ):
            return self._compare_in_chunks(wiki, bash, wiki_steps, bash_steps)

        request = LLMRequest(
            system_prompt=(PROMPTS / "comparison.txt").read_text(encoding="utf-8"),
            user_prompt=json.dumps(
                {
                    "wiki_instruction": wiki.model_dump(mode="json"),
                    "buildscript_instruction": bash.model_dump(mode="json"),
                    "expected_output": AIComparisonResult.model_json_schema(),
                },
                indent=2,
            ),
        )
        response_data = _provider_json(self.llm.generate(request).content, "comparison")
        ai_result = AIComparisonResult.model_validate(response_data)
        return normalize_validation_result(
            self._validate_comparison(ai_result, wiki, bash),
            wiki,
            bash,
        )

    def _compare_in_chunks(
        self,
        wiki: BuildInstruction,
        bash: BuildInstruction,
        wiki_data: list[dict[str, Any]],
        bash_data: list[dict[str, Any]],
    ) -> ValidationResult:
        """Compare Wiki batches against available Bash steps and derive extras."""
        matches_by_wiki: dict[str, Any] = {}
        used_bash_ids: set[str] = set()
        bash_by_id = {step.id: step for step in bash.steps}
        batch_count = (
            len(wiki.steps) + COMPARISON_CHUNK_STEPS - 1
        ) // COMPARISON_CHUNK_STEPS
        comparison_prompt = (PROMPTS / "comparison_chunk.txt").read_text(
            encoding="utf-8"
        )

        for batch_index, start in enumerate(
            range(0, len(wiki.steps), COMPARISON_CHUNK_STEPS),
            start=1,
        ):
            wiki_batch = wiki_data[start:start + COMPARISON_CHUNK_STEPS]
            available_bash = [
                step
                for step in bash_data
                if step["id"] not in used_bash_ids
                or _is_reusable_parameterized_step(bash_by_id[step["id"]])
            ]
            available_bash_ids = {step["id"] for step in available_bash}
            if self.progress is not None:
                self.progress(
                    f"Comparing Wiki batch {batch_index}/{batch_count} with AI..."
                )
            request = LLMRequest(
                system_prompt=comparison_prompt,
                user_prompt=json.dumps(
                    {
                        "wiki_steps": wiki_batch,
                        "available_buildscript_steps": available_bash,
                        "expected_output": AIComparisonResult.model_json_schema(),
                    },
                    indent=2,
                ),
            )
            expected_wiki_ids = {
                step.id for step in wiki.steps[start:start + COMPARISON_CHUNK_STEPS]
            }
            batch_result = _comparison_batch_result(
                self.llm.generate(request).content,
                batch_index,
            )
            repaired_wiki_ids: set[str] = set()
            while True:
                validation_issue = _comparison_batch_issue(
                    batch_result,
                    expected_wiki_ids,
                    available_bash_ids,
                    used_bash_ids,
                    bash_by_id,
                )
                if validation_issue is None:
                    break

                repair_wiki_id, issue_detail = validation_issue
                if repair_wiki_id is None or repair_wiki_id in repaired_wiki_ids:
                    raise RuntimeError(
                        f"AI comparison batch {batch_index} failed validation: "
                        f"{issue_detail}"
                    )
                if self.progress is not None:
                    self.progress(
                        f"Retrying comparison batch {batch_index} for "
                        f"{repair_wiki_id}: repairing its step IDs..."
                    )

                retained_by_wiki = {
                    match.wiki_step_id: match
                    for match in batch_result.matches
                    if match.wiki_step_id in expected_wiki_ids
                    and match.wiki_step_id != repair_wiki_id
                }
                retained_bash_ids = {
                    bash_id
                    for match in retained_by_wiki.values()
                    for bash_id in _comparison_bash_ids(match)
                }
                excluded_bash_ids = used_bash_ids | retained_bash_ids
                repair_candidates = [
                    step
                    for step in bash_data
                    if step["id"] not in excluded_bash_ids
                    or _is_reusable_parameterized_step(bash_by_id[step["id"]])
                ]
                target_wiki_step = next(
                    step for step in wiki_batch if step["id"] == repair_wiki_id
                )
                repair_request = LLMRequest(
                    system_prompt=comparison_prompt,
                    user_prompt=json.dumps(
                        {
                            "wiki_steps": [target_wiki_step],
                            "available_buildscript_steps": repair_candidates,
                            "expected_output": AIComparisonResult.model_json_schema(),
                            "validation_feedback": (
                                f"Repair only Wiki step {repair_wiki_id}. The previous "
                                f"batch had this source-ID issue: {issue_detail}. "
                                "Return exactly one match for that Wiki ID. Choose "
                                "only from the available Bash IDs; use matched or "
                                "conflict only when you can name at least one exact "
                                "Bash ID. Otherwise use missing with empty Bash ID "
                                "fields."
                            ),
                        },
                        indent=2,
                    ),
                )
                repair_result = _comparison_batch_result(
                    self.llm.generate(repair_request).content,
                    batch_index,
                )
                if (
                    len(repair_result.matches) != 1
                    or repair_result.matches[0].wiki_step_id != repair_wiki_id
                ):
                    raise RuntimeError(
                        f"AI comparison could not repair Wiki step {repair_wiki_id} "
                        "with exactly one correctly identified result."
                    )
                retained_by_wiki[repair_wiki_id] = repair_result.matches[0]
                batch_result = AIComparisonResult(
                    summary=batch_result.summary,
                    matches=[
                        retained_by_wiki[step["id"]]
                        for step in wiki_batch
                        if step["id"] in retained_by_wiki
                    ],
                )
                repaired_wiki_ids.add(repair_wiki_id)

            batch_wiki_ids: set[str] = set()
            for match in batch_result.matches:
                wiki_id = match.wiki_step_id
                if wiki_id not in expected_wiki_ids:
                    raise RuntimeError(
                        f"AI comparison batch {batch_index} referenced an unexpected Wiki step {wiki_id}"
                    )
                if wiki_id in batch_wiki_ids:
                    raise RuntimeError(
                        f"AI comparison batch {batch_index} repeated Wiki step {wiki_id}"
                    )
                if match.status == MatchStatus.EXTRA:
                    raise RuntimeError(
                        f"AI comparison batch {batch_index} returned an extra Bash step"
                    )
                bash_ids = _comparison_bash_ids(match)
                if (
                    len(bash_ids) != len(set(bash_ids))
                    or len(match.bash_step_ids) != len(set(match.bash_step_ids))
                ):
                    raise RuntimeError(
                        f"AI comparison batch {batch_index} repeated a Bash step ID"
                    )
                for bash_id in bash_ids:
                    if bash_id not in available_bash_ids:
                        raise RuntimeError(
                            f"AI comparison batch {batch_index} referenced unavailable Bash step {bash_id}"
                        )
                    if (
                        bash_id in used_bash_ids
                        and not _is_reusable_parameterized_step(bash_by_id[bash_id])
                    ):
                        raise RuntimeError(
                            f"AI comparison reused Bash step {bash_id}"
                        )
                    used_bash_ids.add(bash_id)
                batch_wiki_ids.add(wiki_id)
                matches_by_wiki[wiki_id] = match

            missing_wiki_ids = expected_wiki_ids - batch_wiki_ids
            if missing_wiki_ids:
                raise RuntimeError(
                    "AI comparison omitted Wiki steps in batch "
                    f"{batch_index}: {', '.join(sorted(missing_wiki_ids))}"
                )

        all_matches = [matches_by_wiki[step.id] for step in wiki.steps]
        all_matches.extend(
            AIComparisonMatch(
                wiki_step_id=None,
                bash_step_id=step.id,
                status=MatchStatus.EXTRA,
                explanation="This build-script step has no matching Wiki instruction.",
            )
            for step in bash.steps
            if step.id not in used_bash_ids
        )
        status_counts = {
            status: sum(match.status == status for match in all_matches)
            for status in MatchStatus
        }
        summary = (
            f"Compared {len(wiki.steps)} Wiki steps with {len(bash.steps)} "
            "build-script steps: "
            f"{status_counts[MatchStatus.MATCHED]} matched, "
            f"{status_counts[MatchStatus.CONFLICT]} conflicting, "
            f"{status_counts[MatchStatus.MISSING]} missing, and "
            f"{status_counts[MatchStatus.EXTRA]} extra; "
            f"{status_counts[MatchStatus.OUT_OF_SCOPE]} bootstrap steps outside scope."
        )
        return normalize_validation_result(
            self._validate_comparison(
                AIComparisonResult(summary=summary, matches=all_matches),
                wiki,
                bash,
            ),
            wiki,
            bash,
        )

    @staticmethod
    def _validate_comparison(
        result: AIComparisonResult,
        wiki: BuildInstruction,
        bash: BuildInstruction,
    ) -> ValidationResult:
        """Resolve AI step IDs against source data and require full coverage."""
        wiki_steps = {step.id: step for step in wiki.steps}
        bash_steps = {step.id: step for step in bash.steps}
        seen_wiki: set[str] = set()
        seen_bash: set[str] = set()
        extra_bash_ids: set[str] = set()
        matches: list[StepMatch] = []

        for item in result.matches:
            if item.wiki_step_id is not None:
                if item.wiki_step_id not in wiki_steps:
                    raise RuntimeError(
                        f"AI comparison referenced unknown Wiki step {item.wiki_step_id}"
                    )
                if item.wiki_step_id in seen_wiki:
                    raise RuntimeError(
                        f"AI comparison used Wiki step {item.wiki_step_id} more than once"
                    )
                seen_wiki.add(item.wiki_step_id)

            bash_ids = _comparison_bash_ids(item)
            if (
                len(bash_ids) != len(set(bash_ids))
                or len(item.bash_step_ids) != len(set(item.bash_step_ids))
            ):
                raise RuntimeError("AI comparison repeated a Bash step ID")
            if item.status == MatchStatus.EXTRA and any(
                bash_id in seen_bash for bash_id in bash_ids
            ):
                raise RuntimeError("AI comparison marked a matched Bash step as extra")
            if item.status != MatchStatus.EXTRA and any(
                bash_id in extra_bash_ids for bash_id in bash_ids
            ):
                raise RuntimeError("AI comparison matched a Bash step also marked extra")
            for bash_id in bash_ids:
                if bash_id not in bash_steps:
                    raise RuntimeError(
                        f"AI comparison referenced unknown Bash step {bash_id}"
                    )
                if (
                    bash_id in seen_bash
                    and not _is_reusable_parameterized_step(bash_steps[bash_id])
                ):
                    raise RuntimeError(
                        f"AI comparison reused Bash step {bash_id}"
                    )
                seen_bash.add(bash_id)
                if item.status == MatchStatus.EXTRA:
                    extra_bash_ids.add(bash_id)

            if item.status == MatchStatus.MISSING and (
                item.wiki_step_id is None or bash_ids
            ):
                raise RuntimeError("AI comparison returned an invalid missing-step entry")
            if item.status == MatchStatus.EXTRA and (
                item.wiki_step_id is not None or len(bash_ids) != 1
            ):
                raise RuntimeError("AI comparison returned an invalid extra-step entry")
            if item.status == MatchStatus.OUT_OF_SCOPE and (
                item.wiki_step_id is None or bash_ids
            ):
                raise RuntimeError("AI comparison returned an invalid out-of-scope entry")
            if item.status in {MatchStatus.MATCHED, MatchStatus.CONFLICT} and (
                item.wiki_step_id is None or not bash_ids
            ):
                raise RuntimeError(
                    f"AI comparison marked Wiki step {item.wiki_step_id} as "
                    f"{item.status.value} without identifying a Bash step."
                )

            automation_steps = [bash_steps[bash_id] for bash_id in bash_ids]
            category = _normalized_discrepancy_category(
                item.status,
                item.category,
                wiki_steps.get(item.wiki_step_id) if item.wiki_step_id else None,
                automation_steps,
            )
            recommendation = item.recommendation
            if item.status in {
                MatchStatus.MISSING,
                MatchStatus.CONFLICT,
                MatchStatus.EXTRA,
            } and not (recommendation and recommendation.strip()):
                recommendation = _default_recommendation(
                    item.status,
                    category,
                    wiki_steps.get(item.wiki_step_id) if item.wiki_step_id else None,
                )
            matches.append(
                StepMatch(
                    wiki_step=(wiki_steps[item.wiki_step_id] if item.wiki_step_id else None),
                    automation_step=(automation_steps[0] if automation_steps else None),
                    automation_steps=automation_steps,
                    status=item.status,
                    category=category,
                    recommendation=recommendation,
                    explanation=item.explanation,
                )
            )

        if seen_wiki != set(wiki_steps):
            missing = ", ".join(sorted(set(wiki_steps) - seen_wiki))
            raise RuntimeError(f"AI comparison omitted Wiki steps: {missing}")
        if seen_bash != set(bash_steps):
            missing = ", ".join(sorted(set(bash_steps) - seen_bash))
            raise RuntimeError(f"AI comparison omitted Bash steps: {missing}")

        return ValidationResult(
            matches=matches,
            valid=all(
                match.status in {MatchStatus.MATCHED, MatchStatus.OUT_OF_SCOPE}
                for match in matches
            ),
            summary=result.summary,
        )

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
                heading_path=block.heading_path,
                logical_section=block.logical_section,
                logical_subsection=block.logical_subsection,
                logical_heading_path=block.logical_heading_path,
                context_labels=block.context_labels,
                conditions=block.conditions,
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
    def _validate_wiki_instruction(data: dict[str, Any], document: Any) -> BuildInstruction:
        """Validate one whole-document response and keep source-owned fields trusted."""
        document_name = document.name
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
        step_ids = [f"wiki-{index}" for index in range(1, len(raw_steps) + 1)]
        block_texts = [block.text for block in document.blocks]
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
            line_start = context.get("line_start")
            line_end = context.get("line_end")
            if (
                not isinstance(line_start, int)
                or isinstance(line_start, bool)
                or not isinstance(line_end, int)
                or isinstance(line_end, bool)
                or line_start < 1
                or line_end < line_start
            ):
                raise RuntimeError(
                    f"LLM omitted a valid source line range for Wiki step wiki-{index}"
                )

            matching_blocks = [
                block_index
                for block_index, block in enumerate(document.blocks)
                if block.line_start <= line_end and block.line_end >= line_start
            ]
            if matching_blocks:
                block_index = matching_blocks[0]
                block = document.blocks[block_index]
                surrounding = neighbouring_text(block_texts, block_index)
                if len(surrounding) > 1200:
                    surrounding = surrounding[:1200].rstrip() + "…"
                source_excerpt = "\n\n".join(
                    document.blocks[index].text for index in matching_blocks
                )
                context.update(
                    section=block.section,
                    subsection=block.subsection,
                    heading_path=block.heading_path,
                    location=f"lines {line_start}-{line_end}",
                    surrounding_text=surrounding,
                    source_excerpt=source_excerpt,
                    conditions=block.conditions,
                )
            else:
                context.update(
                    section=None,
                    subsection=None,
                    heading_path=[],
                    location=f"lines {line_start}-{line_end}",
                    surrounding_text=None,
                    source_excerpt=None,
                    conditions=[],
                )
            context.update(
                line_start=line_start,
                line_end=line_end,
                previous_step_id=step_ids[index - 2] if index > 1 else None,
                next_step_id=step_ids[index] if index < len(step_ids) else None,
                related_step_ids=(
                    ([step_ids[index - 2]] if index > 1 else [])
                    + ([step_ids[index]] if index < len(step_ids) else [])
                ),
            )
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
            if "semantic_facts" in source:
                request_source["semantic_facts"] = source["semantic_facts"]
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


def _comparison_step(step: BuildStep) -> dict[str, Any]:
    """Keep comparison-relevant fields while omitting duplicated source text."""
    data = step.model_dump(
        mode="json",
        exclude_none=True,
        exclude_defaults=True,
    )
    context = data.get("context")
    if isinstance(context, dict):
        data["context"] = {
            key: context[key]
            for key in (
                "section",
                "subsection",
                "heading_path",
                "line_start",
                "line_end",
                "conditions",
                "runtime_options",
                "runtime_defaults",
            )
            if key in context and context[key] not in (None, [])
        }
    return data


def _normalized_discrepancy_category(
    status: MatchStatus,
    category: DiscrepancyCategory | None,
    wiki_step: BuildStep | None,
    bash_steps: list[BuildStep],
) -> DiscrepancyCategory | None:
    """Use fixed categories for missing/extra entries and validate AI categories."""
    if status == MatchStatus.MISSING:
        return DiscrepancyCategory.MISSING_STEP
    if status == MatchStatus.EXTRA:
        return DiscrepancyCategory.UNDOCUMENTED_STEP
    if status != MatchStatus.CONFLICT:
        return None

    if category in {
        DiscrepancyCategory.MISSING_STEP,
        DiscrepancyCategory.UNDOCUMENTED_STEP,
    }:
        category = None
    if category is not None:
        return category
    return _infer_conflict_category(wiki_step, bash_steps)


def _infer_conflict_category(
    wiki_step: BuildStep | None,
    bash_steps: list[BuildStep],
) -> DiscrepancyCategory:
    """Classify an unlabelled conflict from explicit structured source values."""
    if wiki_step is None:
        return DiscrepancyCategory.OTHER_DEVIATION

    if wiki_step.version and any(
        step.version and step.version != wiki_step.version for step in bash_steps
    ):
        return DiscrepancyCategory.VERSION_INCONSISTENCY

    wiki_packages = set(wiki_step.packages)
    bash_packages = {package for step in bash_steps for package in step.packages}
    if wiki_packages and bash_packages and wiki_packages != bash_packages:
        return DiscrepancyCategory.DEPENDENCY_INCONSISTENCY

    value_fields = (
        "repository",
        "ref",
        "commit",
        "urls",
        "paths",
        "docker_image",
        "dockerfile",
        "build_args",
        "environment",
        "patches",
    )
    for field_name in value_fields:
        wiki_value = getattr(wiki_step, field_name)
        if not wiki_value:
            continue
        bash_values = [
            getattr(step, field_name)
            for step in bash_steps
            if getattr(step, field_name)
        ]
        if bash_values and all(value != wiki_value for value in bash_values):
            return DiscrepancyCategory.PARAMETER_VALUE_MISMATCH

    if wiki_step.action == ActionType.INSTALL_PACKAGE:
        return DiscrepancyCategory.DEPENDENCY_INCONSISTENCY
    if wiki_step.action == ActionType.CONFIGURE:
        return DiscrepancyCategory.CONFIGURATION_MISMATCH
    return DiscrepancyCategory.COMMAND_MISMATCH


def _default_recommendation(
    status: MatchStatus,
    category: DiscrepancyCategory | None,
    wiki_step: BuildStep | None,
) -> str:
    """Provide a safe fallback when a provider omits a correction suggestion."""
    if status == MatchStatus.MISSING:
        condition = " Preserve any documented platform or runtime conditions."
        if wiki_step is not None and not wiki_step.context.conditions:
            condition = ""
        return (
            "Implement the documented Wiki operation in the build script, using its "
            "stated commands and values." + condition
        )
    if status == MatchStatus.EXTRA:
        return (
            "Confirm this operation is required for the build. If it is, document it "
            "in the Build Instructions; otherwise remove or disable it only after "
            "checking its effects."
        )
    category_name = category.value.replace("_", " ") if category else "reported"
    return (
        f"Resolve the {category_name} using the source evidence above. Update the "
        "script to match the Wiki, or revise the Wiki if it is no longer authoritative."
    )


def _wiki_block_chunks(document: Any, max_chars: int) -> list[list[dict[str, Any]]]:
    """Group ordered Wiki blocks into bounded chunks, splitting oversized blocks."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    def add_fragment(fragment: dict[str, Any]) -> None:
        nonlocal current, current_chars
        fragment_chars = len(fragment["text"])
        if current and current_chars + fragment_chars > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(fragment)
        current_chars += fragment_chars

    for block in document.blocks:
        block_data = block.model_dump(mode="json")
        if len(block.text) <= max_chars:
            add_fragment(block_data)
            continue

        lines = block.text.splitlines()
        fragment_lines: list[str] = []
        fragment_chars = 0
        fragment_start = block.line_start
        fragment_end = block.line_start
        line_number = block.line_start

        def flush_fragment() -> None:
            nonlocal fragment_lines, fragment_chars, fragment_start
            if not fragment_lines:
                return
            fragment = dict(block_data)
            fragment.update(
                text="\n".join(fragment_lines),
                line_start=fragment_start,
                line_end=fragment_end,
            )
            add_fragment(fragment)
            fragment_lines = []
            fragment_chars = 0

        for line in lines:
            if len(line) > max_chars:
                flush_fragment()
                for start in range(0, len(line), max_chars):
                    fragment = dict(block_data)
                    fragment.update(
                        text=line[start:start + max_chars],
                        line_start=line_number,
                        line_end=line_number,
                    )
                    add_fragment(fragment)
                line_number += 1
                fragment_start = line_number
                continue

            line_chars = len(line) + (1 if fragment_lines else 0)
            if fragment_lines and fragment_chars + line_chars > max_chars:
                flush_fragment()
                fragment_start = line_number
            elif not fragment_lines:
                fragment_start = line_number
            fragment_lines.append(line)
            fragment_chars += line_chars
            fragment_end = line_number
            line_number += 1

        flush_fragment()

    if current:
        chunks.append(current)
    return chunks


def _provider_json(content: str, step_id: str) -> dict[str, Any]:
    """Decode one provider response with an actionable error message."""
    data = _provider_json_value(content, step_id)
    if not isinstance(data, dict):
        raise RuntimeError(f"LLM returned a JSON value instead of an object for step {step_id}")
    return data


def _provider_json_value(content: str, step_id: str) -> Any:
    """Decode a provider response while allowing chunk callers to accept arrays."""
    if not content.strip():
        raise RuntimeError(f"LLM returned empty JSON for step {step_id}")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        preview = " ".join(content.split())[:200]
        raise RuntimeError(f"LLM returned invalid JSON for step {step_id}: {preview}") from error
    return data
