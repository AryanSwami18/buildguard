import re

from app.models.step import BuildStep


def similarity(wiki_step: BuildStep, automation_step: BuildStep, matched_ids: dict[str, str]) -> float:
    """Compare validated semantics and their order/dependency context.

    This stays deliberately generic: it compares fields supplied by the LLM,
    rather than learning special rules for shell command names.
    """
    score = 0.0
    if wiki_step.action == automation_step.action:
        score += 0.40
    if _overlap(wiki_step.packages, automation_step.packages):
        score += 0.15
    if _overlap(wiki_step.paths, automation_step.paths):
        score += 0.15
    if _overlap(wiki_step.urls, automation_step.urls):
        score += 0.15
    if _overlap(_words(wiki_step.description), _words(automation_step.description)):
        score += 0.10
    if _dependencies_agree(wiki_step, automation_step, matched_ids):
        score += 0.05
    return score


def _dependencies_agree(wiki_step: BuildStep, automation_step: BuildStep, matched_ids: dict[str, str]) -> bool:
    if not wiki_step.depends_on and not automation_step.depends_on:
        return True
    return {matched_ids.get(step_id) for step_id in wiki_step.depends_on} == set(automation_step.depends_on)


def _overlap(left: list[str], right: list[str]) -> bool:
    return bool(set(item.lower() for item in left) & set(item.lower() for item in right))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_/-]+", text.lower())
