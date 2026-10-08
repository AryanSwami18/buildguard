"""Source-grounded grouping for Wiki/build-script comparisons.

The shell parser deliberately keeps useful implementation evidence at command
granularity.  This module is the boundary where that evidence becomes build
actions: related commands are considered together, and shell mechanics are
not promoted to documentation gaps merely because the Wiki describes the
result rather than every implementation command.
"""

from __future__ import annotations

import re
import shlex
from collections import defaultdict

from app.matching.context_matcher import similarity
from app.models.build import BuildInstruction
from app.models.step import ActionType, BuildStep
from app.models.validation import (
    DiscrepancyCategory,
    MatchStatus,
    StepMatch,
    ValidationResult,
)


_VARIABLE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
_WORD = re.compile(r"[a-z0-9][a-z0-9+_.-]{2,}")
_VERSION = re.compile(r"(?<![a-z])(?:v)?\d+(?:\.\d+){1,3}(?![a-z])", re.I)
_GITHUB_REPOSITORY = re.compile(
    r"github\.com/[^/\s\"']+/([A-Za-z0-9_.-]+)",
    re.I,
)

_IMPLEMENTATION_ACTIONS = {
    ActionType.SET_VARIABLE,
    ActionType.CREATE_DIRECTORY,
    ActionType.DELETE,
    ActionType.COPY,
    ActionType.MOVE,
}

_JAVA_WORDS = {
    "java", "jdk", "jre", "openjdk", "temurin", "semeru", "jver",
    "java_home", "java_provided",
}
_COMPONENT_ALIASES = {
    "activemq": {"activemq", "artemis"},
    "qemu": {"qemu"},
    "maven": {"maven", "mvn"},
    "tomcat": {"tomcat"},
    "community-repo": {"community_repo", "community-repo", "communityrepo"},
    "community-share": {"community_share", "community-share", "communityshare"},
    "search-services": {"search_services", "search-services", "searchservices"},
    "transform-core": {"transform_core", "transform-core", "transformcore"},
    "content-app": {"content_app", "content-app", "contentapp"},
    "docker-compose": {"docker_compose", "docker-compose", "compose"},
}


def normalize_command(command: str) -> str:
    """Return a stable intent token for equivalent shell spellings."""
    command = re.sub(r"\\\s*\n\s*", " ", command).strip().lower()
    command = _VARIABLE.sub("<var>", command)
    command = command.replace('"', "").replace("'", "")
    try:
        words = shlex.split(command, comments=False)
    except ValueError:
        words = command.split()

    while words and (words[0] in {"sudo", "retry"} or "=" in words[0]):
        words.pop(0)
    if not words:
        return ""

    executable = words[0]
    args = [word for word in words[1:] if word not in {"-y", "--yes", "--quiet"}]
    if executable in {"apt", "apt-get"} and "update" in args:
        return "package_index_update"
    if executable in {"apt", "apt-get", "dnf", "yum", "zypper"} and "install" in args:
        packages = sorted(
            word for word in args[args.index("install") + 1:]
            if not word.startswith("-")
        )
        return "package_install:" + ",".join(packages)
    if executable == "git" and any(arg in {"fetch", "checkout", "switch"} for arg in args):
        return "git_source"
    if executable in {"patch", "git"} and (executable == "patch" or "apply" in args):
        return "patch"
    if executable == "update-alternatives":
        return "java_configuration" if "java" in command else "configuration"
    if executable in {"export", "sed", "chown"}:
        return "configuration"
    return executable + (":" + " ".join(args[:2]) if args else "")


def _text(step: BuildStep) -> str:
    context = step.context
    return " ".join(
        value
        for value in (
            step.description,
            " ".join(step.commands),
            " ".join(step.packages),
            step.component,
            step.repository,
            step.ref,
            step.commit,
            step.version,
            step.docker_image,
            step.dockerfile,
            " ".join(step.environment),
            " ".join(step.environment.values()),
            " ".join(step.patches),
            context.subsection,
            " ".join(context.heading_path),
            context.source_excerpt,
            " ".join(context.runtime_options),
            " ".join(
                option
                for values in context.runtime_options.values()
                for option in values
            ),
        )
        if value
    ).lower()


def _families(step: BuildStep) -> set[str]:
    text = _text(step).replace("_", "-")
    identity_text = " ".join(
        value
        for value in (
            step.description,
            step.component,
            step.repository,
            step.context.subsection,
            " ".join(step.context.heading_path),
        )
        if value
    ).lower().replace("_", "-")
    families: set[str] = set()
    java_identity = any(
        word in identity_text
        for word in ("java", "jdk", "jre", "temurin", "semeru", "openjdk", "jver")
    )
    if step.component and not any(
        word in step.component.lower()
        for word in ("java", "jdk", "jre", "temurin", "semeru", "openjdk")
    ):
        # Component build tags often contain jre/java build arguments. Those
        # do not turn an Activemq, Tomcat, Search Services, or Content App
        # operation into the Java installation workflow.
        java_identity = any(
            word in " ".join(
                value for value in (
                    step.context.subsection,
                    " ".join(step.context.heading_path),
                ) if value
            ).lower()
            for word in ("java", "jdk", "jre", "temurin", "semeru", "openjdk", "jver")
        )
    java_environment = any(
        name.upper() in {"JAVA_HOME", "JAVA_PROVIDED", "JAVA_MAJOR"}
        for name in step.environment
    )
    if java_identity or java_environment:
        families.add("java")
    for family, aliases in _COMPONENT_ALIASES.items():
        if any(alias in text for alias in aliases):
            families.add(family)
    if "java" in families and families & {
        "activemq", "tomcat", "search-services", "content-app",
        "community-repo", "community-share", "transform-core", "qemu",
    }:
        families.remove("java")
    if step.component:
        families.add(re.sub(r"[^a-z0-9]+", "-", step.component.lower()).strip("-"))
    if step.repository:
        repository = step.repository.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        families.add(re.sub(r"[^a-z0-9]+", "-", repository.lower()).strip("-"))
    # Wiki extraction may keep a GitHub clone URL in ``commands`` without
    # populating the structured ``repository`` field. Recover the repository
    # family from that URL so the action can be grouped with the corresponding
    # Bash clone/build workflow.
    for repository in _GITHUB_REPOSITORY.findall(" ".join(step.commands)):
        repository = repository.removesuffix(".git")
        families.add(re.sub(r"[^a-z0-9]+", "-", repository.lower()).strip("-"))
    return {family for family in families if family}


def _words(step: BuildStep) -> set[str]:
    return set(_WORD.findall(_text(step)))


def _is_unused_configuration(step: BuildStep) -> bool:
    return "unused_configuration" in step.context.context_labels


def _is_implementation_detail(step: BuildStep) -> bool:
    if _is_unused_configuration(step):
        return False
    if step.action in _IMPLEMENTATION_ACTIONS:
        return True
    if (step.context.subsection or "").lower() == "cleanup":
        return True
    command = " ".join(step.commands).lower()
    if step.action == ActionType.CONFIGURE:
        return True
    if step.action == ActionType.RUN_COMMAND and (
        normalize_command(command) in {"package_index_update", "configuration"}
        or re.search(r"\bchown\b", command)
        or re.search(r"\bdocker\s+pull\b", command)
    ):
        return True
    return False


def _is_workflow_contributor(step: BuildStep) -> bool:
    return step.action in {
        ActionType.CLONE_REPOSITORY,
        ActionType.CHECKOUT_SOURCE,
        ActionType.APPLY_PATCH,
        ActionType.BUILD_IMAGE,
        ActionType.DOWNLOAD,
        ActionType.EXTRACT,
        ActionType.CONFIGURE,
        ActionType.CREATE_DIRECTORY,
        ActionType.RUN_COMMAND,
        ActionType.RUN_TEST,
    }


def _action_compatible(left: BuildStep, right: BuildStep) -> bool:
    if left.action == right.action:
        return True
    compatible = {
        ActionType.INSTALL_PACKAGE: {
            ActionType.DOWNLOAD, ActionType.EXTRACT, ActionType.CONFIGURE,
            ActionType.SET_VARIABLE, ActionType.RUN_COMMAND,
        },
        ActionType.DOWNLOAD: {
            ActionType.EXTRACT, ActionType.INSTALL_PACKAGE, ActionType.CONFIGURE,
            ActionType.SET_VARIABLE, ActionType.CLONE_REPOSITORY,
        },
        ActionType.CLONE_REPOSITORY: {
            ActionType.CHECKOUT_SOURCE, ActionType.APPLY_PATCH, ActionType.BUILD_IMAGE,
            ActionType.RUN_COMMAND, ActionType.CONFIGURE,
        },
        ActionType.CHECKOUT_SOURCE: {
            ActionType.CLONE_REPOSITORY, ActionType.APPLY_PATCH, ActionType.BUILD_IMAGE,
            ActionType.CONFIGURE,
        },
        ActionType.APPLY_PATCH: {
            ActionType.CLONE_REPOSITORY, ActionType.CHECKOUT_SOURCE, ActionType.BUILD_IMAGE,
        },
        ActionType.BUILD_IMAGE: {
            ActionType.CLONE_REPOSITORY, ActionType.CHECKOUT_SOURCE, ActionType.APPLY_PATCH,
            ActionType.CONFIGURE, ActionType.RUN_COMMAND,
        },
        ActionType.CONFIGURE: {
            ActionType.INSTALL_PACKAGE, ActionType.DOWNLOAD, ActionType.BUILD_IMAGE,
            ActionType.SET_VARIABLE, ActionType.RUN_COMMAND,
        },
    }
    return right.action in compatible.get(left.action, set())


def _value_overlap(left: BuildStep, right: BuildStep) -> float:
    score = 0.0
    if left.component and right.component and left.component.lower() == right.component.lower():
        score += 0.30
    if left.repository and right.repository and left.repository.lower() == right.repository.lower():
        score += 0.35
    if left.commit and right.commit and left.commit.lower() == right.commit.lower():
        score += 0.25
    if left.ref and right.ref and left.ref.lower() == right.ref.lower():
        score += 0.15
    if left.docker_image and right.docker_image and left.docker_image.lower() == right.docker_image.lower():
        score += 0.30
    if left.version and right.version and left.version.lower() == right.version.lower():
        score += 0.20
    if left.packages and right.packages:
        left_packages = {_normalize_package(value) for value in left.packages}
        right_packages = {_normalize_package(value) for value in right.packages}
        if left_packages & right_packages:
            score += 0.20
    if set(left.environment) & set(right.environment):
        score += 0.10
    return score


def _normalize_package(value: str) -> str:
    value = value.lower().replace("${jver}", "<java>").replace("$jver", "<java>")
    value = value.strip("'\"")
    if "openjdk" in value and any(marker in value for marker in ("jdk", "devel")):
        # Debian/Ubuntu use openjdk-17-jdk while RHEL/SLES use
        # java-17-openjdk-devel. They describe the same JDK capability for
        # documentation comparison, even though the package spelling differs.
        return "openjdk-jdk"
    if value.startswith("openjdk-") and value.endswith("-jdk"):
        return "openjdk-jdk"
    if value.startswith("openjdk") and value.endswith("jdk"):
        return "openjdk-jdk"
    return value


def _semantic_score(wiki: BuildStep, script: BuildStep) -> float:
    score = similarity(wiki, script)
    wiki_families = _families(wiki)
    script_families = _families(script)
    family_overlap = wiki_families & script_families
    if family_overlap:
        # A shared component/repository family is stronger evidence than
        # generic command-word similarity. This matters for Wiki actions that
        # are represented as one RUN_COMMAND containing a complete workflow;
        # otherwise an unrelated Maven command can outrank the matching clone
        # merely because both descriptions contain common build words.
        score += 0.45
    if _action_compatible(wiki, script):
        score += 0.12
    score += min(_value_overlap(wiki, script), 0.35)
    common_words = _words(wiki) & _words(script)
    if len(common_words) >= 2:
        score += min(0.12, len(common_words) * 0.02)
    if normalize_command(" ".join(wiki.commands)) == normalize_command(" ".join(script.commands)):
        score += 0.25
    return min(score, 1.0)


def _same_workflow(wiki: BuildStep, primary: BuildStep, candidate: BuildStep) -> bool:
    primary_families = _families(primary)
    candidate_families = _families(candidate)
    candidate_function = (candidate.context.subsection or "").lower()
    wiki_text = _text(wiki)

    if candidate_function == "cleanup":
        return False
    if "java" in _families(wiki):
        wiki_component = (wiki.component or "").lower()
        if "base-java" in wiki_component or "base java" in wiki_text:
            return (
                "basejava" in candidate_function
                or "base-java" in (candidate.component or "").lower()
                or "base-java" in (candidate.repository or "").lower()
            )
        return (
            candidate_function == "installjava"
            or candidate.action == ActionType.SET_VARIABLE
            and any(
                name.upper() in {"JAVA_HOME", "JAVA_PROVIDED"}
                for name in candidate.environment
            )
            )

    if (
        primary.component
        and candidate.component
        and primary.component.lower() != candidate.component.lower()
        and not (primary_families & candidate_families)
    ):
        return False

    if primary_families & candidate_families:
        if (
            primary.action in {
                ActionType.DOWNLOAD,
                ActionType.EXTRACT,
                ActionType.INSTALL_PACKAGE,
            }
            and candidate.action in {ActionType.RUN_COMMAND, ActionType.RUN_TEST}
            and candidate.order - primary.order > 4
        ):
            return False
        if (
            primary.context.subsection
            and candidate.context.subsection
            and primary.context.subsection.lower() != candidate.context.subsection.lower()
        ):
            return False
        return True
    if primary.component and candidate.component and primary.component.lower() == candidate.component.lower():
        return True
    if primary.repository and candidate.repository and primary.repository.lower() == candidate.repository.lower():
        return True
    primary_function = primary.context.subsection
    if not primary_function or primary_function != candidate.context.subsection:
        return False

    # A helper such as installJava is itself a workflow boundary. A generic
    # configure/install function contains many unrelated components, so it
    # only groups commands that are close together in source order.
    function_name = primary_function.lower()
    if function_name not in {"configureandinstall", "runtest", "installubuntu", "installrhel", "installsles"}:
        return True
    distance = candidate.order - primary.order
    if _is_implementation_detail(candidate):
        return -4 <= distance <= 1
    if primary.action in {
        ActionType.CLONE_REPOSITORY,
        ActionType.CHECKOUT_SOURCE,
        ActionType.APPLY_PATCH,
        ActionType.DOWNLOAD,
        ActionType.EXTRACT,
    }:
        return -4 <= distance <= 4
    return -4 <= distance <= 1


def _group_for(wiki: BuildStep, script_steps: list[BuildStep]) -> tuple[list[BuildStep], float]:
    candidates = [
        step for step in script_steps
        if not _is_unused_configuration(step)
    ]
    scored = sorted(
        ((_semantic_score(wiki, step), step) for step in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    scored.sort(
        key=lambda item: (
            item[0],
            not _is_implementation_detail(item[1]),
        ),
        reverse=True,
    )
    if not scored:
        return [], 0.0

    score, primary = scored[0]
    wiki_families = _families(wiki)
    threshold = 0.42 if "java" in wiki_families else 0.52
    if score < threshold:
        return [], score

    group = [primary]
    for candidate_score, candidate in scored[1:]:
        if not _same_workflow(wiki, primary, candidate):
            continue
        if (
            candidate_score >= max(0.30, threshold - 0.18)
            or _is_implementation_detail(candidate)
            or _is_workflow_contributor(candidate)
            and candidate.context.subsection
            and candidate.context.subsection == primary.context.subsection
            and abs(candidate.order - primary.order) <= 4
        ):
            group.append(candidate)

    # When a component has a repository clone, commands immediately before
    # that clone belong to the preceding component even if they share a
    # generic helper function. Keep the component's own clone as the workflow
    # boundary; Java and QEMU are the two intentional pre-build exceptions.
    anchor_candidates = [
        candidate
        for _, candidate in scored
        if candidate.order <= primary.order
        and (candidate.component or candidate.repository)
        and _families(candidate) & _families(primary)
    ]
    if anchor_candidates and not (
        "qemu" in _families(wiki)
        or (
            "java" in _families(wiki)
            and "base-java" not in _text(wiki)
        )
    ):
        anchor_order = min(candidate.order for candidate in anchor_candidates)
        group = [
            candidate
            for candidate in group
            if candidate.order >= anchor_order
            or candidate.component
            or candidate.repository
        ]
    return sorted(group, key=lambda step: step.order), score


def _group_conflicts(wiki: BuildStep, group: list[BuildStep]) -> bool:
    """Detect concrete value disagreement after operation identity is known."""
    if not group:
        return False
    script = group
    if wiki.version and _is_concrete(wiki.version):
        values = {
            step.version
            for step in script
            if step.version
        }
        values.update(
            value
            for step in script
            for name, value in step.environment.items()
            if "VERSION" in name.upper()
        )
        concrete_values = {value for value in values if _is_concrete(value)}
        if concrete_values and wiki.version not in concrete_values:
            # A Java build image may carry the product version while its Java
            # installation is parameterized; do not compare those unrelated
            # tags as if they were one value.
            if "java" not in _families(wiki) or any(
                "JAVA" in name.upper()
                for step in script
                for name in step.environment
            ):
                return True

    if wiki.commit and _is_concrete(wiki.commit):
        values = {
            step.commit for step in script
            if step.commit and _is_concrete(step.commit)
        }
        if values and wiki.commit not in values:
            return True
    if wiki.repository and _is_concrete(wiki.repository):
        values = {
            step.repository for step in script
            if step.repository and _is_concrete(step.repository)
        }
        if values and wiki.repository not in values:
            return True
    if wiki.docker_image and _is_concrete(wiki.docker_image):
        values = {
            step.docker_image for step in script
            if step.docker_image and _is_concrete(step.docker_image)
        }
        if values and wiki.docker_image not in values:
            return True

    if wiki.packages:
        wiki_packages = {_normalize_package(value) for value in wiki.packages}
        script_packages = {
            _normalize_package(package)
            for step in script
            for package in step.packages
        }
        if script_packages and wiki_packages.isdisjoint(script_packages):
            return True
    return False


def _is_concrete(value: str) -> bool:
    return not re.search(r"\$|\{|\}|<var>|\$\(", value)


def _dedupe(steps: list[BuildStep]) -> list[BuildStep]:
    seen: set[str] = set()
    result: list[BuildStep] = []
    for step in steps:
        if step.id not in seen:
            seen.add(step.id)
            result.append(step)
    return sorted(result, key=lambda step: step.order)


def normalize_validation_result(
    result: ValidationResult,
    wiki: BuildInstruction,
    bash: BuildInstruction,
) -> ValidationResult:
    """Replace command-level extras with grouped semantic classifications."""
    matches = [match.model_copy(deep=True) for match in result.matches]
    for match in matches:
        if match.automation_step is not None and not match.automation_steps:
            match.automation_steps = [match.automation_step]
    by_wiki = {
        match.wiki_step.id: match
        for match in matches
        if match.wiki_step is not None
    }

    # First recover Wiki steps that the provider called missing but that are
    # implemented by a parameterized or multi-command script workflow.
    for match in by_wiki.values():
        if match.status != MatchStatus.MISSING or match.wiki_step is None:
            continue
        group, score = _group_for(match.wiki_step, bash.steps)
        if not group:
            continue
        match.automation_steps = group
        match.automation_step = group[0]
        if _group_conflicts(match.wiki_step, group):
            match.status = MatchStatus.CONFLICT
            match.category = DiscrepancyCategory.VERSION_INCONSISTENCY
            match.explanation = (
                "The script implements this grouped build action, but a concrete "
                "version, commit, repository, image, or package value differs."
            )
        else:
            match.status = MatchStatus.MATCHED
            match.category = None
            match.explanation = (
                f"Grouped semantic implementation matched the Wiki action "
                f"(score {score:.2f}); individual shell commands are implementation details."
            )

    referenced: set[str] = {
        step.id
        for match in matches
        if match.status not in {MatchStatus.EXTRA, MatchStatus.IMPLEMENTATION_DETAIL, MatchStatus.UNUSED_CONFIGURATION}
        for step in match.automation_steps or ([match.automation_step] if match.automation_step else [])
    }

    # Rehome unused meaningful Bash commands under an existing Wiki action when
    # they are part of the same semantic group. This is what collapses a Git /
    # patch / Docker sequence into one component build action.
    for step in bash.steps:
        if step.id in referenced or _is_unused_configuration(step):
            continue
        candidates: list[tuple[float, StepMatch, list[BuildStep]]] = []
        for match in by_wiki.values():
            if match.wiki_step is None or match.status not in {
                MatchStatus.MATCHED,
                MatchStatus.CONFLICT,
            }:
                continue
            group, score = _group_for(match.wiki_step, bash.steps)
            if step.id in {item.id for item in group}:
                candidates.append((score, match, group))
        if candidates:
            _, target, group = max(candidates, key=lambda item: item[0])
            target.automation_steps = _dedupe(target.automation_steps + group)
            target.automation_step = target.automation_steps[0]
            referenced.update(item.id for item in group)

    # The original provider result may still contain one EXTRA row per command
    # even after those commands have been attached to a Wiki action. Remove
    # those stale rows before classifying genuine leftovers.
    matches = [
        match
        for match in matches
        if not (
            match.status == MatchStatus.EXTRA
            and match.automation_step is not None
            and match.automation_step.id in referenced
        )
    ]

    # Finally classify remaining command-level records. Only EXTRA remains an
    # actionable documentation gap; shell details and dead config have their
    # own non-gap statuses and are excluded from findings by the reporter.
    for match in matches:
        if match.status != MatchStatus.EXTRA or match.automation_step is None:
            continue
        step = match.automation_step
        if _is_unused_configuration(step):
            match.status = MatchStatus.UNUSED_CONFIGURATION
            match.category = DiscrepancyCategory.UNUSED_CONFIGURATION
            configuration_name = next(iter(step.environment), step.id)
            match.explanation = (
                f"Configuration {configuration_name} "
                "is defined but has no static reference in the script."
            )
        elif _is_implementation_detail(step):
            match.status = MatchStatus.IMPLEMENTATION_DETAIL
            match.category = None
            match.explanation = (
                "Internal shell plumbing for a build action; it does not need a "
                "separate Wiki instruction."
            )

    return ValidationResult(
        matches=matches,
        valid=all(
            match.status in {
                MatchStatus.MATCHED,
                MatchStatus.OUT_OF_SCOPE,
                MatchStatus.IMPLEMENTATION_DETAIL,
            }
            for match in matches
        ),
        summary=_summary(matches),
    )


def _summary(matches: list[StepMatch]) -> str:
    counts = defaultdict(int)
    for match in matches:
        counts[match.status.value] += 1
    return ", ".join(
        f"{counts[status.value]} {status.value}"
        for status in MatchStatus
        if counts[status.value]
    )
