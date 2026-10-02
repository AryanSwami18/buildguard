from app.models.step import BuildStep


def _overlap(
    left: list[str],
    right: list[str],
) -> float:
    if not left or not right:
        return 0.0

    left_set = {
        value.lower()
        for value in left
        if value
    }

    right_set = {
        value.lower()
        for value in right
        if value
    }

    if not left_set or not right_set:
        return 0.0

    intersection = left_set & right_set

    if not intersection:
        return 0.0

    union = left_set | right_set

    return len(intersection) / len(union)


def _description_similarity(
    left: str,
    right: str,
) -> float:
    left_words = {
        word.lower()
        for word in left.split()
        if len(word) > 2
    }

    right_words = {
        word.lower()
        for word in right.split()
        if len(word) > 2
    }

    if not left_words or not right_words:
        return 0.0

    intersection = left_words & right_words

    return len(intersection) / max(
        len(left_words),
        len(right_words),
    )


def similarity(
    wiki_step: BuildStep,
    automation_step: BuildStep,
    matched_ids: dict | None = None,
) -> float:
    """
    Calculate similarity between a Wiki step and an automation step.

    matched_ids is retained for backwards compatibility with
    the existing BuildMatcher implementation.
    """

    score = 0.0

    # ---------------------------------------------------------
    # Existing matching signals
    # ---------------------------------------------------------

    if wiki_step.action == automation_step.action:
        score += 0.40

    score += (
        _overlap(
            wiki_step.packages,
            automation_step.packages,
        )
        * 0.15
    )

    score += (
        _overlap(
            wiki_step.paths,
            automation_step.paths,
        )
        * 0.15
    )

    score += (
        _overlap(
            wiki_step.urls,
            automation_step.urls,
        )
        * 0.15
    )

    score += (
        _description_similarity(
            wiki_step.description,
            automation_step.description,
        )
        * 0.10
    )

    score += (
        _overlap(
            wiki_step.depends_on,
            automation_step.depends_on,
        )
        * 0.05
    )

    # ---------------------------------------------------------
    # Build-specific semantic signals
    # ---------------------------------------------------------

    if (
        wiki_step.component
        and automation_step.component
        and wiki_step.component.lower()
        == automation_step.component.lower()
    ):
        score += 0.10

    if (
        wiki_step.repository
        and automation_step.repository
        and wiki_step.repository.lower()
        == automation_step.repository.lower()
    ):
        score += 0.15

    if (
        wiki_step.commit
        and automation_step.commit
        and wiki_step.commit.lower()
        == automation_step.commit.lower()
    ):
        score += 0.15

    if (
        wiki_step.ref
        and automation_step.ref
        and wiki_step.ref.lower()
        == automation_step.ref.lower()
    ):
        score += 0.10

    if (
        wiki_step.version
        and automation_step.version
        and wiki_step.version.lower()
        == automation_step.version.lower()
    ):
        score += 0.10

    if (
        wiki_step.docker_image
        and automation_step.docker_image
        and wiki_step.docker_image.lower()
        == automation_step.docker_image.lower()
    ):
        score += 0.15

    if (
        wiki_step.build_tool
        and automation_step.build_tool
        and wiki_step.build_tool.lower()
        == automation_step.build_tool.lower()
    ):
        score += 0.05

    # ---------------------------------------------------------
    # Build arguments
    # ---------------------------------------------------------

    if (
        wiki_step.build_args
        and automation_step.build_args
    ):
        matching_args = 0

        for key, value in wiki_step.build_args.items():
            if automation_step.build_args.get(key) == value:
                matching_args += 1

        if matching_args:
            score += min(
                0.10,
                matching_args * 0.03,
            )

    return min(score, 1.0)
