import json
import re
from pathlib import Path
from typing import Any, Callable

from app.intelligence.interface import LLMProvider
from app.intelligence.request import LLMRequest
from app.models.build import BuildInstruction
from app.models.step import ActionType, BuildStep
from app.models.validation import (
    AIComparisonMatch,
    AIComparisonResult,
    AIValueDifference,
    DiscrepancyCategory,
    MatchStatus,
    StepMatch,
    ValidationResult,
)
from app.matching.semantic_matcher import (
    _versioned_package_disagreement,
    normalize_validation_result,
)
from app.parsers.bash_normalizer import normalize_bash
from app.parsers.bash_parser import parse_script
from app.parsers.semantic_parser import parse_semantic_facts
from app.parsers.wiki_parser import parse_wiki
from app.parsers.wiki_semantic_parser import complete_wiki_instruction
from app.pipeline.context import neighbouring_text, step_context
from app.sources.github import GitHubSource, GitHubWikiPage

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
WIKI_EXTRACTION_CHUNK_CHARS = 6000
COMPARISON_CHUNK_STEPS = 8
COMPARISON_SINGLE_REQUEST_MAX_STEPS = 24
COMPARISON_SINGLE_REQUEST_MAX_CHARS = 18000
COMPARISON_BATCH_REPAIR_ATTEMPTS = 2
COMPARISON_CANDIDATES_PER_WIKI_STEP = 8
COMPARISON_MAX_BASH_CANDIDATES = 40
COMPARISON_OUTPUT_TOKEN_BUDGET = 4096

# These words occur in almost every shell operation and do not help retrieve
# the Bash implementation of a Wiki action.  Keeping retrieval deterministic
# means the LLM still makes the semantic decision, but does not have to scan
# the entire script for every Wiki batch.
_COMPARISON_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "command",
    "commands",
    "component",
    "description",
    "do",
    "does",
    "dockerfile",
    "environment",
    "excerpt",
    "file",
    "from",
    "for",
    "if",
    "in",
    "install",
    "into",
    "is",
    "it",
    "line",
    "lines",
    "of",
    "on",
    "package",
    "packages",
    "path",
    "paths",
    "repository",
    "run",
    "script",
    "section",
    "source",
    "or",
    "step",
    "steps",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "url",
    "urls",
    "version",
    "with",
}

_COMPARISON_PACKAGE_MANAGERS = {
    "apt": "apt",
    "apt-get": "apt",
    "dnf": "dnf",
    "yum": "yum",
    "zypper": "zypper",
    "apk": "apk",
    "pacman": "pacman",
}


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
    """Allow explicitly parameterized operations to cover supported variants.

    Reuse is intentionally tied to runtime options/defaults supplied by source
    analysis, rather than a package name or a particular runtime such as Java.
    """
    if step.action not in {
        ActionType.INSTALL_PACKAGE,
        ActionType.DOWNLOAD,
        ActionType.SET_VARIABLE,
        ActionType.CONFIGURE,
        ActionType.RUN_COMMAND,
    }:
        return False
    if not (step.context.runtime_options or step.context.runtime_defaults):
        return False

    payload = json.dumps(
        {
            "packages": step.packages,
            "commands": step.commands,
            "environment": step.environment,
            "version": step.version,
            "urls": step.urls,
        },
        ensure_ascii=False,
    )
    return bool(re.search(r"\$\{?[A-Za-z_][A-Za-z0-9_]*", payload))


def _is_shareable_configuration_step(step: BuildStep) -> bool:
    """Identify declarations that can support several semantic Wiki rows."""
    return (
        step.action == ActionType.SET_VARIABLE
        and "configuration" in step.context.context_labels
    )


def _is_reusable_comparison_step(step: BuildStep) -> bool:
    """Return whether one Bash node may support more than one Wiki action."""
    return _is_reusable_parameterized_step(step) or _is_shareable_configuration_step(step)


def _comparison_search_tokens(value: Any) -> set[str]:
    """Return meaningful tokens used to retrieve semantic Bash candidates."""
    if value is None:
        return set()
    if isinstance(value, dict):
        tokens: set[str] = set()
        for item in value.values():
            tokens.update(_comparison_search_tokens(item))
        return tokens
    if isinstance(value, (list, tuple, set)):
        tokens: set[str] = set()
        for item in value:
            tokens.update(_comparison_search_tokens(item))
        return tokens

    raw = str(value).casefold()
    # Split camel-case names as well as shell punctuation.  For example,
    # "communityRepo" and "community-repo" should retrieve the same action.
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
    tokens = set(re.findall(r"[a-z0-9]+", raw))
    return {
        token
        for token in tokens
        if len(token) > 1 and token not in _COMPARISON_SEARCH_STOPWORDS
    }


def _comparison_step_search_text(step: BuildStep) -> dict[str, Any]:
    """Expose source-owned semantic fields to deterministic candidate search."""
    context = step.context
    return {
        "action": step.action.value,
        "description": step.description,
        "packages": step.packages,
        "commands": step.commands,
        "paths": step.paths,
        "urls": step.urls,
        "component": step.component,
        "version": step.version,
        "repository": step.repository,
        "ref": step.ref,
        "commit": step.commit,
        "docker_image": step.docker_image,
        "dockerfile": step.dockerfile,
        "build_args": step.build_args,
        "build_arg_names": list(step.build_args),
        "environment": step.environment,
        "environment_names": list(step.environment),
        "patches": step.patches,
        "build_tool": step.build_tool,
        "section": context.section,
        "subsection": context.subsection,
        "heading_path": context.heading_path,
        "source_excerpt": context.source_excerpt,
    }


def _comparison_package_managers(step: BuildStep) -> set[str]:
    """Return normalized package-manager commands used by one source step."""
    command_text = "\n".join(step.commands).casefold()
    return {
        normalized
        for command, normalized in _COMPARISON_PACKAGE_MANAGERS.items()
        if re.search(rf"(?<![a-z0-9_-]){re.escape(command)}(?![a-z0-9_-])", command_text)
    }


def _comparison_candidate_score(wiki_step: BuildStep, bash_step: BuildStep) -> float:
    """Score one Bash step for a Wiki action without making a match decision."""
    wiki_fields = _comparison_step_search_text(wiki_step)
    bash_fields = _comparison_step_search_text(bash_step)
    wiki_tokens = _comparison_search_tokens(wiki_fields)
    bash_tokens = _comparison_search_tokens(bash_fields)
    shared_tokens = wiki_tokens & bash_tokens
    score = float(min(len(shared_tokens), 12))

    # Commands, URLs, packages, and paths carry the actual operation.  Give
    # their overlap more weight than headings or parser context, which often
    # repeat the same words across an entire document.
    for field in (
        "commands",
        "packages",
        "urls",
        "paths",
        "description",
        "component",
        "build_tool",
    ):
        field_overlap = _comparison_search_tokens(wiki_fields[field]) & _comparison_search_tokens(
            bash_fields[field]
        )
        score += min(len(field_overlap), 6) * 2

    wiki_component = _comparison_search_tokens(wiki_step.component)
    bash_component = _comparison_search_tokens(bash_step.component)
    if wiki_component and bash_component:
        if wiki_component == bash_component:
            score += 32
        else:
            score += 8 * len(wiki_component & bash_component)

    if wiki_step.action == bash_step.action:
        score += 5
    elif wiki_step.action in {
        ActionType.RUN_COMMAND,
        ActionType.CONFIGURE,
    } and bash_step.action in {
        ActionType.RUN_COMMAND,
        ActionType.CONFIGURE,
        ActionType.BUILD_IMAGE,
        ActionType.RUN_TEST,
    }:
        score += 2

    # A package command's manager identifies its platform branch.  Without
    # this signal, yum, zypper, and apt dependency rows with mostly identical
    # package names can all look interchangeable to retrieval and to a small
    # local model.  Prefer the same manager and penalize a different manager;
    # this does not decide the comparison, it only orders repair candidates.
    wiki_managers = _comparison_package_managers(wiki_step)
    bash_managers = _comparison_package_managers(bash_step)
    if wiki_managers and bash_managers:
        if wiki_managers & bash_managers:
            score += 20
        else:
            score -= 12

    # A parameterized Bash operation is a strong candidate for a concrete
    # Wiki variant when its static package/tool tokens overlap.  The final AI
    # comparison still decides whether the parameter really covers the Wiki
    # value; this only keeps the operation in the evidence set.
    if (bash_step.context.runtime_options or bash_step.context.runtime_defaults) and len(
        shared_tokens
    ) >= 2:
        score += 10

    # Exact identity fields are much stronger than a coincidental token
    # overlap.  This is important for versions, commits, repositories, and
    # Docker images, where a related component can still have different data.
    for field in (
        "repository",
        "version",
        "ref",
        "commit",
        "docker_image",
        "build_tool",
    ):
        wiki_value = getattr(wiki_step, field)
        bash_value = getattr(bash_step, field)
        if wiki_value and bash_value:
            wiki_normalized = " ".join(str(wiki_value).split()).casefold()
            bash_normalized = " ".join(str(bash_value).split()).casefold()
            if wiki_normalized == bash_normalized:
                score += 18
            elif wiki_normalized in bash_normalized or bash_normalized in wiki_normalized:
                score += 6

    wiki_context = _comparison_search_tokens(
        [wiki_step.context.section, wiki_step.context.subsection, wiki_step.context.heading_path]
    )
    bash_context = _comparison_search_tokens(
        [bash_step.context.section, bash_step.context.subsection, bash_step.context.heading_path]
    )
    score += min(len(wiki_context & bash_context), 4) * 2
    return score


def _select_comparison_bash_candidates(
    wiki_steps: list[BuildStep],
    bash_steps: list[BuildStep],
    used_bash_ids: set[str],
) -> list[BuildStep]:
    """Select relevant Bash actions for a Wiki batch before invoking the LLM.

    This is retrieval, not classification.  The complete source is still
    used for the final EXTRA calculation, while the model receives a bounded
    set of likely implementations plus all exact component peers.  Parameter
    substitution and grouped actions therefore remain available without
    repeatedly sending cleanup and helper statements to the model.
    """
    available = [
        step
        for step in bash_steps
        if step.id not in used_bash_ids or _is_reusable_comparison_step(step)
    ]
    if len(available) <= COMPARISON_MAX_BASH_CANDIDATES:
        return available

    selected_ids: set[str] = set()
    scores: dict[str, float] = {}
    for wiki_step in wiki_steps:
        ranked = sorted(
            [
                (
                    _comparison_candidate_score(wiki_step, bash_step),
                    bash_step,
                )
                for bash_step in available
            ],
            key=lambda item: (-item[0], item[1].order),
        )
        for score, bash_step in ranked[:COMPARISON_CANDIDATES_PER_WIKI_STEP]:
            selected_ids.add(bash_step.id)
            scores[bash_step.id] = max(scores.get(bash_step.id, 0.0), score)

            # When a strong operation is found inside a large helper, retain
            # its nearby shell operations too.  Clone/fetch/checkout/patch/
            # build sequences are often represented by separate AST nodes but
            # are one semantic build action to the comparison model.
            if score >= 25 and bash_step.context.subsection:
                for neighbour in available:
                    if (
                        neighbour.context.subsection == bash_step.context.subsection
                        and abs(neighbour.order - bash_step.order) <= 2
                    ):
                        selected_ids.add(neighbour.id)
                        scores[neighbour.id] = max(
                            scores.get(neighbour.id, 0.0),
                            score - 1,
                        )

        # A component is intentionally grouped across clone, checkout, patch,
        # and build commands.  Keep all currently available operations with
        # the exact component even when one of them has weak lexical overlap.
        component_tokens = _comparison_search_tokens(wiki_step.component)
        if component_tokens:
            for bash_step in available:
                if _comparison_search_tokens(bash_step.component) == component_tokens:
                    selected_ids.add(bash_step.id)
                    scores[bash_step.id] = max(
                        scores.get(bash_step.id, 0.0),
                        28.0,
                    )

        # Keep a small function-level group together.  This is what allows a
        # Wiki Java or dependency action to be matched to the install,
        # configuration, and PATH statements in one Bash helper.  Large
        # orchestration functions are deliberately excluded; their exact
        # component/repository candidates are selected by the normal ranking.
        subsection = wiki_step.context.subsection
        if subsection:
            subsection_peers = [
                bash_step
                for bash_step in available
                if bash_step.context.subsection == subsection
            ]
            if 1 < len(subsection_peers) <= 16:
                for bash_step in subsection_peers:
                    selected_ids.add(bash_step.id)
                    scores[bash_step.id] = max(
                        scores.get(bash_step.id, 0.0),
                        20.0,
                    )

    if len(selected_ids) > COMPARISON_MAX_BASH_CANDIDATES:
        selected_ids = {
            step.id
            for step in sorted(
                (step for step in available if step.id in selected_ids),
                key=lambda step: (-scores.get(step.id, 0.0), step.order),
            )[:COMPARISON_MAX_BASH_CANDIDATES]
        }
    return [step for step in available if step.id in selected_ids]


def _comparison_batch_issue(
    result: AIComparisonResult,
    expected_wiki_ids: set[str],
    available_bash_ids: set[str],
    used_bash_ids: set[str],
    bash_by_id: dict[str, BuildStep],
) -> tuple[str | None, str, str | None] | None:
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
                None,
            )

        bash_ids = _comparison_bash_ids(match)
        if len(bash_ids) != len(set(bash_ids)):
            return wiki_id, f"Wiki step {wiki_id} repeats a Bash ID", None
        if match.status in {MatchStatus.MATCHED, MatchStatus.CONFLICT} and not bash_ids:
            return (
                wiki_id,
                f"Wiki step {wiki_id} is {match.status.value} without a Bash ID",
                None,
            )
        if match.status in {MatchStatus.MISSING, MatchStatus.OUT_OF_SCOPE} and bash_ids:
            return (
                wiki_id,
                f"Wiki step {wiki_id} is {match.status.value} but includes Bash IDs",
                None,
            )

        for bash_id in bash_ids:
            if bash_id not in available_bash_ids:
                return (
                    wiki_id,
                    f"Bash ID {bash_id} is not available in this batch",
                    None,
                )
            if (
                bash_id in used_bash_ids or bash_id in seen_bash_ids
            ) and not _is_reusable_comparison_step(bash_by_id[bash_id]):
                return (
                    wiki_id,
                    f"Bash ID {bash_id} is assigned to more than one Wiki step",
                    bash_id,
                )
            seen_bash_ids.add(bash_id)
        seen_wiki_ids.add(wiki_id)

    omitted_wiki_ids = expected_wiki_ids - seen_wiki_ids
    if omitted_wiki_ids:
        omitted = ", ".join(sorted(omitted_wiki_ids))
        return (
            sorted(omitted_wiki_ids)[0],
            f"the response omitted Wiki step(s): {omitted}",
            None,
        )
    if malformed_wiki_rows:
        return None, "the response used an unknown or duplicate Wiki step ID", None
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


def _is_output_limit_error(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return "output limit" in message or "maximum output tokens" in message


def _alias_bash_candidates(
    steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Give the model only batch-local Bash IDs and retain their source mapping."""
    source_to_alias = {
        step["id"]: f"candidate-{index}"
        for index, step in enumerate(steps, start=1)
    }
    alias_to_source = {alias: source for source, alias in source_to_alias.items()}
    aliased_steps: list[dict[str, Any]] = []
    for step in steps:
        aliased = dict(step)
        aliased["id"] = source_to_alias[step["id"]]
        dependencies = step.get("depends_on")
        if isinstance(dependencies, list):
            aliased["depends_on"] = [
                source_to_alias[dependency]
                for dependency in dependencies
                if dependency in source_to_alias
            ]
        aliased_steps.append(aliased)
    return aliased_steps, alias_to_source


def _resolve_bash_candidate_ids(
    result: AIComparisonResult,
    alias_to_source: dict[str, str],
) -> AIComparisonResult:
    """Translate model-facing candidate tokens back to original Bash IDs."""
    source_ids = set(alias_to_source.values())

    def resolve(value: str) -> str:
        # Accept a source ID only when it is one of this request's candidates.
        # This keeps compatibility with providers that return original IDs,
        # while still rejecting IDs copied from another batch or invented by
        # the model.
        if value in alias_to_source:
            return alias_to_source[value]
        if value in source_ids:
            return value
        return f"__invalid_candidate__:{value}"

    resolved_matches: list[AIComparisonMatch] = []
    for match in result.matches:
        bash_ids = [resolve(value) for value in match.bash_step_ids]
        bash_step_id = (
            resolve(match.bash_step_id)
            if match.bash_step_id is not None
            else None
        )
        value_differences = [
            difference.model_copy(
                update={
                    "bash_step_id": resolve(difference.bash_step_id)
                }
            )
            for difference in match.value_differences
        ]
        resolved_matches.append(
            match.model_copy(
                update={
                    "bash_step_id": bash_step_id,
                    "bash_step_ids": bash_ids,
                    "value_differences": value_differences,
                }
            )
        )
    return result.model_copy(update={"matches": resolved_matches})


def _verified_value_differences(
    differences: list[AIValueDifference],
    wiki_step: BuildStep,
    bash_ids: set[str],
    bash_steps: dict[str, BuildStep],
) -> list[AIValueDifference]:
    """Retain only different values present in the cited source fields."""
    def field_values(value: Any) -> set[str]:
        if isinstance(value, str):
            return {" ".join(value.split()).casefold()}
        if isinstance(value, dict):
            return set().union(*(field_values(item) for item in value.values())) if value else set()
        if isinstance(value, (list, tuple, set)):
            return set().union(*(field_values(item) for item in value)) if value else set()
        if value is None:
            return set()
        return {str(value).casefold()}

    verified: list[AIValueDifference] = []
    for difference in differences:
        if (
            difference.field not in BuildStep.model_fields
            or difference.bash_step_id not in bash_ids
            or difference.wiki_value.strip().casefold()
            == difference.bash_value.strip().casefold()
        ):
            continue
        bash_step = bash_steps[difference.bash_step_id]
        wiki_field = field_values(getattr(wiki_step, difference.field))
        bash_field = field_values(getattr(bash_step, difference.field))
        if (
            " ".join(difference.wiki_value.split()).casefold() in wiki_field
            and " ".join(difference.bash_value.split()).casefold() in bash_field
        ):
            verified.append(difference)
    return verified


def _comparison_scalar_values(value: Any) -> list[str]:
    """Flatten a BuildStep field to source values suitable for evidence."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _comparison_scalar_values(nested)]
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _comparison_scalar_values(nested)]
    if value is None:
        return []
    return [str(value)]


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
            return complete_wiki_instruction(
                self._from_wiki_chunks(document),
                markdown,
                name,
            )
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
            return complete_wiki_instruction(
                self._validate_wiki_instruction(response_data, document),
                markdown,
                name,
            )
        if {"action", "description"}.issubset(response_data):
            return complete_wiki_instruction(
                self._from_wiki_blocks_legacy(document, response_data),
                markdown,
                name,
            )
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
                    "wiki_instruction": {
                        "name": wiki.name,
                        "source": wiki.source,
                        "version": wiki.version,
                        "steps": wiki_steps,
                    },
                    "buildscript_instruction": {
                        "name": bash.name,
                        "source": bash.source,
                        "version": bash.version,
                        "steps": bash_steps,
                    },
                    "expected_output": AIComparisonResult.model_json_schema(),
                },
                indent=2,
            ),
            max_output_tokens=COMPARISON_OUTPUT_TOKEN_BUDGET,
        )
        try:
            response_data = _provider_json(
                self.llm.generate(request).content,
                "comparison",
            )
        except RuntimeError as error:
            if not _is_output_limit_error(error):
                raise
            if self.progress is not None:
                self.progress(
                    "The comparison response exceeded the output budget; "
                    "switching to smaller semantic batches..."
                )
            return self._compare_in_chunks(wiki, bash, wiki_steps, bash_steps)
        ai_result = AIComparisonResult.model_validate(response_data)
        try:
            validation = self._validate_comparison(ai_result, wiki, bash)
        except RuntimeError as error:
            # The single-request path has the same ownership contract as the
            # chunked path.  If a provider assigns one ordinary AST node to
            # multiple Wiki rows, retry through the duplicate-aware chunk
            # validator instead of surfacing a provider formatting error.
            if "reused Bash step" not in str(error) and "repeated a Bash step" not in str(error):
                raise
            if self.progress is not None:
                self.progress(
                    "The comparison reused a Bash action across Wiki rows; "
                    "rematching with semantic batches..."
                )
            return self._compare_in_chunks(wiki, bash, wiki_steps, bash_steps)
        return normalize_validation_result(validation, wiki, bash)

    def _repair_duplicate_wiki_match(
        self,
        wiki_step: BuildStep,
        wiki_data: dict[str, Any],
        bash: BuildInstruction,
        bash_data_by_id: dict[str, dict[str, Any]],
        used_bash_ids: set[str],
        claimed_bash_ids: set[str],
        duplicate_bash_id: str | None,
        bash_by_id: dict[str, BuildStep],
        comparison_prompt: str,
        batch_index: int,
    ) -> tuple[AIComparisonMatch | None, str | None]:
        """Rematch only a Wiki row whose Bash candidate was already claimed.

        A model can correctly identify a shared function but assign one AST
        node to several Wiki rows.  Repeating the whole batch leaves that
        ambiguity in the prompt and often reproduces the same invalid answer.
        Rematching the affected row with claimed non-reusable steps removed
        preserves the valid rows and gives the model a clean ownership set.
        """
        duplicate_ids = {duplicate_bash_id} if duplicate_bash_id is not None else set()
        blocked_bash_ids = {
            bash_id
            for bash_id in used_bash_ids | claimed_bash_ids | duplicate_ids
            if not _is_reusable_comparison_step(bash_by_id[bash_id])
        }
        candidates = _select_comparison_bash_candidates(
            [wiki_step],
            bash.steps,
            blocked_bash_ids,
        )
        # Put the most semantically relevant unclaimed candidate first.  The
        # aliases are intentionally local, and small local models frequently
        # choose candidate-1 when several candidates are otherwise plausible.
        # Ordering candidate-1 by evidence makes that fallback deterministic.
        candidates.sort(
            key=lambda step: (
                -_comparison_candidate_score(wiki_step, step),
                step.order,
            )
        )
        candidate_data = [bash_data_by_id[step.id] for step in candidates]
        model_bash, candidate_to_source = _alias_bash_candidates(candidate_data)
        request = LLMRequest(
            system_prompt=comparison_prompt,
            user_prompt=json.dumps(
                {
                    "wiki_steps": [wiki_data],
                    "available_buildscript_steps": model_bash,
                    "expected_output": AIComparisonResult.model_json_schema(),
                    "validation_feedback": (
                        "Repair only the supplied Wiki step. Earlier Wiki rows "
                        "already own their Bash candidates. Do not use any Bash "
                        "step that is absent from this candidate list; choose a "
                        "different candidate or return missing if no supplied "
                        "candidate implements this Wiki step. Return exactly one "
                        "result for the supplied Wiki step and do not return extras."
                    ),
                },
                indent=2,
            ),
            max_output_tokens=COMPARISON_OUTPUT_TOKEN_BUDGET,
        )
        try:
            result = _comparison_batch_result(
                self.llm.generate(request).content,
                batch_index,
            )
        except RuntimeError as error:
            return None, str(error)
        result = _resolve_bash_candidate_ids(result, candidate_to_source)
        issue = _comparison_batch_issue(
            result,
            {wiki_step.id},
            {step.id for step in candidates},
            used_bash_ids | claimed_bash_ids,
            bash_by_id,
        )
        if issue is not None:
            return None, issue[1]
        if len(result.matches) != 1:
            return None, "the duplicate-assignment repair did not return one Wiki result"
        return result.matches[0], None

    @staticmethod
    def _fallback_missing_wiki_match(
        wiki_step: BuildStep,
        bash: BuildInstruction,
        used_bash_ids: set[str],
        claimed_bash_ids: set[str],
        bash_by_id: dict[str, BuildStep],
    ) -> AIComparisonMatch:
        """Create a conservative result when a one-row repair is malformed."""
        blocked_bash_ids = {
            bash_id
            for bash_id in used_bash_ids | claimed_bash_ids
            if not _is_reusable_comparison_step(bash_by_id[bash_id])
        }
        candidates = _select_comparison_bash_candidates(
            [wiki_step],
            bash.steps,
            blocked_bash_ids,
        )
        candidates.sort(
            key=lambda step: (
                -_comparison_candidate_score(wiki_step, step),
                step.order,
            )
        )
        candidate = candidates[0] if candidates else None
        if candidate is not None and _comparison_candidate_score(wiki_step, candidate) >= 30:
            return AIComparisonMatch(
                status=MatchStatus.MATCHED,
                explanation=(
                    "The comparison response omitted this Wiki action; BuildGuard "
                    f"recovered the strongest unclaimed semantic candidate ({candidate.id})."
                ),
                wiki_step_id=wiki_step.id,
                bash_step_id=candidate.id,
                bash_step_ids=[candidate.id],
            )
        return AIComparisonMatch(
            status=MatchStatus.MISSING,
            explanation=(
                "The comparison response omitted this Wiki action and no unclaimed "
                "Bash action had enough source evidence to recover it."
            ),
            wiki_step_id=wiki_step.id,
            bash_step_id=None,
            bash_step_ids=[],
            category=DiscrepancyCategory.MISSING_STEP,
            recommendation=(
                "Review whether this Wiki action is implemented by an unclaimed "
                "Bash operation."
            ),
        )

    @staticmethod
    def _fallback_duplicate_wiki_match(
        original_match: AIComparisonMatch,
        wiki_step: BuildStep,
        bash: BuildInstruction,
        used_bash_ids: set[str],
        claimed_bash_ids: set[str],
        duplicate_bash_id: str,
        bash_by_id: dict[str, BuildStep],
    ) -> AIComparisonMatch:
        """Recover one duplicate row without re-offering its claimed node.

        This path is deliberately last-resort.  It prevents a malformed local
        model response from aborting an otherwise useful comparison, while
        still refusing to invent a semantic match: a candidate must have
        meaningful source overlap.  If no such candidate exists, the Wiki row
        remains missing and is reported for review.
        """
        blocked_bash_ids = {
            bash_id
            for bash_id in used_bash_ids | claimed_bash_ids | {duplicate_bash_id}
            if not _is_reusable_comparison_step(bash_by_id[bash_id])
        }
        candidates = _select_comparison_bash_candidates(
            [wiki_step],
            bash.steps,
            blocked_bash_ids,
        )
        candidates.sort(
            key=lambda step: (
                -_comparison_candidate_score(wiki_step, step),
                step.order,
            )
        )
        candidate = candidates[0] if candidates else None
        score = (
            _comparison_candidate_score(wiki_step, candidate)
            if candidate is not None
            else 0.0
        )
        if candidate is None or score < 30:
            return AIComparisonMatch(
                status=MatchStatus.MISSING,
                explanation=(
                    "The comparison provider reused a Bash step and no unclaimed "
                    "Bash action had enough source evidence to repair this row."
                ),
                wiki_step_id=wiki_step.id,
                bash_step_id=None,
                bash_step_ids=[],
                category=DiscrepancyCategory.MISSING_STEP,
                recommendation=(
                    "Review whether this Wiki action is implemented by an "
                    "unclaimed Bash operation."
                ),
            )

        original_ids = _comparison_bash_ids(original_match)
        if candidate.id in original_ids and candidate.id != duplicate_bash_id:
            return AIComparisonMatch(
                status=MatchStatus.MISSING,
                explanation=(
                    "The comparison provider reused a Bash step and the available "
                    "candidates could not be assigned without another duplicate."
                ),
                wiki_step_id=wiki_step.id,
                bash_step_id=None,
                bash_step_ids=[],
                category=DiscrepancyCategory.MISSING_STEP,
                recommendation=(
                    "Review whether this Wiki action is implemented by an "
                    "unclaimed Bash operation."
                ),
            )
        replacement_ids = [
            candidate.id if bash_id == duplicate_bash_id else bash_id
            for bash_id in original_ids
        ]

        repaired_differences: list[AIValueDifference] = []
        for difference in original_match.value_differences:
            difference_bash_id = (
                candidate.id
                if difference.bash_step_id == duplicate_bash_id
                else difference.bash_step_id
            )
            bash_value = difference.bash_value
            if difference.bash_step_id == duplicate_bash_id:
                source_value = getattr(candidate, difference.field, None)
                if source_value in (None, [], {}, ""):
                    continue
                source_values = _comparison_scalar_values(source_value)
                source_value_by_key = {
                    " ".join(value.split()).casefold(): value
                    for value in source_values
                }
                original_value_key = " ".join(difference.bash_value.split()).casefold()
                if original_value_key in source_value_by_key:
                    bash_value = source_value_by_key[original_value_key]
                else:
                    wiki_values = {
                        " ".join(value.split()).casefold()
                        for value in _comparison_scalar_values(
                            getattr(wiki_step, difference.field, None)
                        )
                    }
                    changed_values = [
                        value
                        for value in source_values
                        if " ".join(value.split()).casefold() not in wiki_values
                    ]
                    if not changed_values:
                        continue
                    bash_value = changed_values[0]
            repaired_differences.append(
                difference.model_copy(
                    update={
                        "bash_step_id": difference_bash_id,
                        "bash_value": bash_value,
                    }
                )
            )

        status = original_match.status
        category = original_match.category
        recommendation = original_match.recommendation
        if status == MatchStatus.CONFLICT and not _verified_value_differences(
            repaired_differences,
            wiki_step,
            set(replacement_ids),
            bash_by_id | {candidate.id: candidate},
        ):
            # Do not preserve a conflict whose old evidence belonged to the
            # duplicate node.  The final validator will otherwise expose a
            # conflict against the wrong Bash command.
            status = MatchStatus.MATCHED
            category = None
            recommendation = None

        primary_id = replacement_ids[0] if replacement_ids else candidate.id
        return original_match.model_copy(
            update={
                "status": status,
                "category": category,
                "recommendation": recommendation,
                "bash_step_id": primary_id,
                "bash_step_ids": replacement_ids,
                "value_differences": repaired_differences,
                "explanation": (
                    f"{original_match.explanation.strip()} BuildGuard reassigned "
                    f"the duplicate Bash step to the strongest unclaimed semantic "
                    f"candidate ({candidate.id})."
                ),
            }
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
        pending_batches = [
            wiki.steps[start:start + COMPARISON_CHUNK_STEPS]
            for start in range(0, len(wiki.steps), COMPARISON_CHUNK_STEPS)
        ]
        wiki_data_by_id = {step["id"]: step for step in wiki_data}
        bash_data_by_id = {step["id"]: step for step in bash_data}
        completed_batches = 0
        comparison_prompt = (PROMPTS / "comparison_chunk.txt").read_text(
            encoding="utf-8"
        )

        while pending_batches:
            wiki_batch_steps = pending_batches.pop(0)
            wiki_batch = [
                wiki_data_by_id[step.id]
                for step in wiki_batch_steps
            ]
            batch_index = completed_batches + 1
            batch_count = completed_batches + 1 + len(pending_batches)
            available_bash_steps = _select_comparison_bash_candidates(
                wiki_batch_steps,
                bash.steps,
                used_bash_ids,
            )
            available_bash = [
                bash_data_by_id[step.id]
                for step in available_bash_steps
            ]
            available_bash_ids = {step["id"] for step in available_bash}
            model_bash, candidate_to_source = _alias_bash_candidates(available_bash)
            if self.progress is not None:
                self.progress(
                    f"Comparing Wiki batch {batch_index}/{batch_count} with AI..."
                )
            request = LLMRequest(
                system_prompt=comparison_prompt,
                user_prompt=json.dumps(
                    {
                        "wiki_steps": wiki_batch,
                        "available_buildscript_steps": model_bash,
                        "expected_output": AIComparisonResult.model_json_schema(),
                    },
                    indent=2,
                ),
                max_output_tokens=COMPARISON_OUTPUT_TOKEN_BUDGET,
            )
            expected_wiki_ids = {step.id for step in wiki_batch_steps}
            try:
                batch_result = _comparison_batch_result(
                    self.llm.generate(request).content,
                    batch_index,
                )
            except RuntimeError as error:
                if not _is_output_limit_error(error) or len(wiki_batch_steps) == 1:
                    raise
                split_at = len(wiki_batch_steps) // 2
                pending_batches = [
                    wiki_batch_steps[:split_at],
                    wiki_batch_steps[split_at:],
                    *pending_batches,
                ]
                if self.progress is not None:
                    self.progress(
                        f"Comparison batch {batch_index} exceeded the AI output "
                        f"limit; splitting {len(wiki_batch_steps)} Wiki actions "
                        "into smaller batches and retrying..."
                    )
                continue
            batch_result = _resolve_bash_candidate_ids(
                batch_result,
                candidate_to_source,
            )
            repair_attempts = 0
            split_after_repair_limit = False
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

                repair_wiki_id, issue_detail, duplicate_bash_id = validation_issue
                if (
                    repair_wiki_id is not None
                    and duplicate_bash_id is None
                    and issue_detail.startswith("the response omitted Wiki step(s):")
                ):
                    # A partial response is not a reason to throw away the
                    # valid rows already returned by the model. Repair each
                    # omitted Wiki action with its own small candidate set so
                    # the provider cannot keep reproducing the same omission
                    # from the full batch prompt.
                    returned_wiki_ids = {
                        match.wiki_step_id
                        for match in batch_result.matches
                        if match.wiki_step_id in expected_wiki_ids
                    }
                    omitted_steps = [
                        step
                        for step in wiki_batch_steps
                        if step.id not in returned_wiki_ids
                    ]
                    for omitted_step in omitted_steps:
                        claimed_bash_ids = {
                            claimed_id
                            for match in batch_result.matches
                            if match.wiki_step_id in expected_wiki_ids
                            for claimed_id in _comparison_bash_ids(match)
                        }
                        repaired_match, repair_error = self._repair_duplicate_wiki_match(
                            omitted_step,
                            wiki_data_by_id[omitted_step.id],
                            bash,
                            bash_data_by_id,
                            used_bash_ids,
                            claimed_bash_ids,
                            None,
                            bash_by_id,
                            comparison_prompt,
                            batch_index,
                        )
                        if repaired_match is None:
                            repaired_match = self._fallback_missing_wiki_match(
                                omitted_step,
                                bash,
                                used_bash_ids,
                                claimed_bash_ids,
                                bash_by_id,
                            )
                            if self.progress is not None:
                                self.progress(
                                    f"Recovered omitted Wiki step {omitted_step.id} "
                                    "without another full-batch retry."
                                )
                        batch_result = batch_result.model_copy(
                            update={
                                "matches": [
                                    *batch_result.matches,
                                    repaired_match,
                                ]
                            }
                        )
                    continue

                if repair_attempts >= COMPARISON_BATCH_REPAIR_ATTEMPTS:
                    if duplicate_bash_id is not None and repair_wiki_id is not None:
                        original_match = next(
                            match
                            for match in batch_result.matches
                            if match.wiki_step_id == repair_wiki_id
                        )
                        claimed_bash_ids = {
                            claimed_id
                            for match in batch_result.matches
                            if match.wiki_step_id != repair_wiki_id
                            for claimed_id in _comparison_bash_ids(match)
                        }
                        fallback_match = self._fallback_duplicate_wiki_match(
                            original_match,
                            next(
                                step
                                for step in wiki_batch_steps
                                if step.id == repair_wiki_id
                            ),
                            bash,
                            used_bash_ids,
                            claimed_bash_ids,
                            duplicate_bash_id,
                            bash_by_id,
                        )
                        batch_result = batch_result.model_copy(
                            update={
                                "matches": [
                                    fallback_match
                                    if match.wiki_step_id == repair_wiki_id
                                    else match
                                    for match in batch_result.matches
                                ]
                            }
                        )
                        if self.progress is not None:
                            self.progress(
                                f"Recovered comparison batch {batch_index} after "
                                f"duplicate Bash assignment for {repair_wiki_id}."
                            )
                        continue
                    target = f"Wiki step {repair_wiki_id}" if repair_wiki_id else "the batch"
                    raise RuntimeError(
                        f"AI comparison batch {batch_index} could not validate "
                        f"{target} after {repair_attempts} repair "
                        f"attempts: {issue_detail}"
                    )
                repair_attempts += 1
                if self.progress is not None:
                    self.progress(
                        f"Retrying comparison batch {batch_index} "
                        f"(attempt {repair_attempts}/"
                        f"{COMPARISON_BATCH_REPAIR_ATTEMPTS}): {issue_detail}"
                    )

                if duplicate_bash_id is not None and repair_wiki_id is not None:
                    claimed_bash_ids = {
                        claimed_id
                        for match in batch_result.matches
                        if match.wiki_step_id != repair_wiki_id
                        for claimed_id in _comparison_bash_ids(match)
                    }
                    repaired_match, repair_error = self._repair_duplicate_wiki_match(
                        next(
                            step
                            for step in wiki_batch_steps
                            if step.id == repair_wiki_id
                        ),
                        wiki_data_by_id[repair_wiki_id],
                        bash,
                        bash_data_by_id,
                        used_bash_ids,
                        claimed_bash_ids,
                        duplicate_bash_id,
                        bash_by_id,
                        comparison_prompt,
                        batch_index,
                    )
                    if repaired_match is not None:
                        batch_result = batch_result.model_copy(
                            update={
                                "matches": [
                                    repaired_match
                                    if match.wiki_step_id == repair_wiki_id
                                    else match
                                    for match in batch_result.matches
                                ]
                            }
                        )
                        continue
                    if repair_error:
                        issue_detail = repair_error

                    # Do not send the original full batch again: it still
                    # contains the duplicate candidate and invites the model
                    # to reproduce the same ownership error.  The next loop
                    # iteration retries only the affected Wiki row with the
                    # claimed candidate removed.
                    continue

                repair_request = LLMRequest(
                    system_prompt=comparison_prompt,
                    user_prompt=json.dumps(
                        {
                            "wiki_steps": wiki_batch,
                            "available_buildscript_steps": model_bash,
                            "expected_output": AIComparisonResult.model_json_schema(),
                            "validation_feedback": (
                                f"The previous response failed validation: {issue_detail}. "
                                "Re-evaluate this entire Wiki batch and return exactly "
                                "one result per supplied Wiki step. Use only the "
                                "batch-local candidate IDs shown in the available "
                                "Bash steps. For an unmatched Wiki step, use missing "
                                "with empty Bash ID fields. Do not return extras."
                            ),
                        },
                        indent=2,
                    ),
                    max_output_tokens=COMPARISON_OUTPUT_TOKEN_BUDGET,
                )
                try:
                    batch_result = _comparison_batch_result(
                        self.llm.generate(repair_request).content,
                        batch_index,
                    )
                except RuntimeError as error:
                    if not _is_output_limit_error(error) or len(wiki_batch_steps) == 1:
                        raise
                    split_at = len(wiki_batch_steps) // 2
                    pending_batches = [
                        wiki_batch_steps[:split_at],
                        wiki_batch_steps[split_at:],
                        *pending_batches,
                    ]
                    split_after_repair_limit = True
                    if self.progress is not None:
                        self.progress(
                            f"Repair response for batch {batch_index} exceeded the "
                            "AI output limit; splitting the Wiki actions and retrying..."
                        )
                    break
                batch_result = _resolve_bash_candidate_ids(
                    batch_result,
                    candidate_to_source,
                )

            if split_after_repair_limit:
                continue

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
                        and not _is_reusable_comparison_step(bash_by_id[bash_id])
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
            completed_batches += 1

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
                    and not _is_reusable_comparison_step(bash_steps[bash_id])
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
            status = item.status
            explanation = item.explanation
            recommendation = item.recommendation
            if status == MatchStatus.CONFLICT:
                verified_differences = _verified_value_differences(
                    item.value_differences,
                    wiki_steps[item.wiki_step_id],
                    set(bash_ids),
                    bash_steps,
                )
                if not verified_differences:
                    status = MatchStatus.MATCHED
                    recommendation = None
                    explanation = (
                        "The AI linked this as the same build action but did not "
                        "provide a verifiable source-value difference. BuildGuard "
                        "does not report it as a confirmed conflict."
                    )
                else:
                    evidence = "; ".join(
                        f"{difference.field}: Wiki has {difference.wiki_value}, "
                        f"Bash {difference.bash_step_id} has {difference.bash_value}"
                        for difference in verified_differences
                    )
                    explanation = f"{explanation.strip()} Evidence: {evidence}."
            category = _normalized_discrepancy_category(
                status,
                item.category,
                wiki_steps.get(item.wiki_step_id) if item.wiki_step_id else None,
                automation_steps,
            )
            if status in {
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
                    status=status,
                    category=category,
                    recommendation=recommendation,
                    explanation=explanation,
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
    """Keep semantic fields plus only the source context not duplicated by them."""
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
                "context_labels",
            )
            if key in context and context[key] not in (None, [])
        }
        context = data["context"]
        source_type = step.context.source_type.casefold()
        context_limits = (
            {"surrounding_text": 600, "source_excerpt": 1200}
            if source_type == "wiki"
            else {"source_excerpt": 280}
        )
        for key, limit in context_limits.items():
            value = step.context.model_dump(mode="json").get(key)
            if isinstance(value, str) and value.strip():
                context[key] = value[:limit]
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
    normalized_wiki_packages = {
        package.lower() for package in wiki_packages
    }
    normalized_bash_packages = {
        package.lower() for package in bash_packages
    }
    if normalized_wiki_packages and normalized_bash_packages and (
        normalized_wiki_packages.isdisjoint(normalized_bash_packages)
        or _versioned_package_disagreement(
            normalized_wiki_packages,
            normalized_bash_packages,
        )
    ):
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
