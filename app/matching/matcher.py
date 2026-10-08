from app.matching.context_matcher import similarity
from app.matching.semantic_matcher import normalize_validation_result
from app.models.build import BuildInstruction
from app.models.validation import MatchStatus, StepMatch, ValidationResult


class BuildMatcher:
    """Match Wiki and automation steps using their validated semantic fields."""

    def compare(self, wiki: BuildInstruction, automation: BuildInstruction) -> ValidationResult:
        unmatched = list(automation.steps)
        matches: list[StepMatch] = []
        matched_ids: dict[str, str] = {}

        for wiki_step in wiki.steps:
            candidate, score = self._best_candidate(wiki_step, unmatched, matched_ids)
            if candidate is None or score < 0.40:
                matches.append(StepMatch(
                    wiki_step=wiki_step,
                    status=MatchStatus.MISSING,
                    explanation="No automation step has enough matching semantic context.",
                ))
                continue
            unmatched.remove(candidate)
            matched_ids[wiki_step.id] = candidate.id
            status = MatchStatus.MATCHED if score >= 0.55 else MatchStatus.CONFLICT
            matches.append(StepMatch(
                wiki_step=wiki_step,
                automation_step=candidate,
                status=status,
                explanation=f"Semantic and contextual similarity score: {score:.2f}.",
            ))

        matches.extend(StepMatch(
            automation_step=step,
            status=MatchStatus.EXTRA,
            explanation="This automation step has no corresponding Wiki step.",
        ) for step in unmatched)
        result = ValidationResult(
            matches=matches,
            valid=all(match.status == MatchStatus.MATCHED for match in matches),
            summary=self._summary(matches),
        )
        return normalize_validation_result(result, wiki, automation)

    @staticmethod
    def _best_candidate(wiki_step, candidates, matched_ids):
        if not candidates:
            return None, 0.0
        scored = [(candidate, similarity(wiki_step, candidate, matched_ids)) for candidate in candidates]
        return max(scored, key=lambda item: item[1])

    @staticmethod
    def _summary(matches: list[StepMatch]) -> str:
        counts = {status: sum(match.status == status for match in matches) for status in MatchStatus}
        return ", ".join(f"{counts[status]} {status.value}" for status in MatchStatus)
