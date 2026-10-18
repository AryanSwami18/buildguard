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
from urllib.parse import unquote, urlparse

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
_VERSION = re.compile(
    r"(?<![a-z0-9.])(?:v)?\d+(?:\.\d+){1,3}(?![a-z0-9.])",
    re.I,
)
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
    ActionType.REMOVE_PACKAGE,
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
_TOOL_FAMILY_PATTERNS = {
    "go": re.compile(r"\b(?:go|golang|goroot|go[-_]url|golang[-_]version)\b"),
    "nodejs": re.compile(r"\b(?:node|nodejs|node[-_]js|node[-_]version)\b"),
    "maven": re.compile(r"\b(?:maven|mvn|maven[-_]version|maven[-_]url)\b"),
    "antlr": re.compile(r"\bantlr(?:4)?\b"),
}
_EXPLICIT_FAILURE_HANDLER = re.compile(
    r"\s+\|\|\s*(?:error|fail|die|fatal)\b.*$",
    re.IGNORECASE | re.DOTALL,
)


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
    if executable == "make" and any(
        arg == "-j" or arg.startswith("-j") or arg == "--jobs"
        for arg in args
    ):
        targets = [
            arg for arg in args
            if not arg.startswith("-") and not re.fullmatch(r"\d+", arg)
        ]
        return "make:parallel" + (":" + " ".join(targets[:2]) if targets else "")
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
    if step.context.source_type == "wiki":
        # Do not use the full source excerpt here. A single Markdown code block
        # can contain Node, Maven, and ANTLR commands, while each extracted Wiki
        # step represents only one of those actions.
        family_signal_text = " ".join(
            value
            for value in (
                step.description,
                step.component,
                step.repository,
                step.context.subsection
                if step.action != ActionType.INSTALL_PACKAGE
                else None,
                " ".join(step.commands),
                " ".join(step.packages),
            )
            if value
        )
    else:
        family_signal_text = text
    for family, aliases in _COMPONENT_ALIASES.items():
        if any(alias in family_signal_text for alias in aliases):
            families.add(family)
    if "java" in families and families & {
        "activemq", "tomcat", "search-services", "content-app",
        "community-repo", "community-share", "transform-core", "qemu",
    }:
        families.remove("java")
    if step.component:
        families.add(re.sub(r"[^a-z0-9]+", "-", step.component.lower()).strip("-"))
    for family, pattern in _TOOL_FAMILY_PATTERNS.items():
        if pattern.search(family_signal_text):
            families.add(family)
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


def _identity_families(step: BuildStep) -> set[str]:
    """Return families named by the action identity, excluding command context.

    A test action can mention Go, JavaScript, and Maven in its commands while
    still describing the ANTLR product. Version checks must use the product
    identity, not every tool word in the code block.
    """
    # A structured component label is the strongest identity. Descriptions of
    # tests often mention runtime names as well (for example "ANTLR tests for
    # Go"); those are tool requirements, not the product being versioned.
    identity_parts = (
        (step.component, step.repository)
        if step.component
        else (
            step.description,
            step.repository,
        )
    )
    identity = " ".join(value for value in identity_parts if value).lower().replace("_", "-")
    families: set[str] = set()
    if any(word in identity for word in _JAVA_WORDS):
        families.add("java")
    for family, aliases in _COMPONENT_ALIASES.items():
        if any(alias in identity for alias in aliases):
            families.add(family)
    for family, pattern in _TOOL_FAMILY_PATTERNS.items():
        if pattern.search(identity):
            families.add(family)
    if step.component:
        families.add(re.sub(r"[^a-z0-9]+", "-", step.component.lower()).strip("-"))
    return families


def _component_tool_family(component: str | None) -> str | None:
    """Map a structured component label to a comparison tool family."""
    if not component:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", component.lower())
    if "java" in normalized or normalized in {"jdk", "jre"}:
        return "java"
    if "node" in normalized:
        return "nodejs"
    if "maven" in normalized or normalized == "mvn":
        return "maven"
    if "antlr" in normalized:
        return "antlr"
    if normalized in {"go", "golang"}:
        return "go"
    return None


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
        ActionType.SET_VARIABLE: {
            ActionType.CONFIGURE, ActionType.RUN_COMMAND,
        },
        ActionType.RUN_COMMAND: {
            ActionType.CONFIGURE, ActionType.SET_VARIABLE, ActionType.RUN_TEST,
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
    value = value.strip().lower().replace("${jver}", "<java>").replace("$jver", "<java>")
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


_PACKAGE_NOISE = {
    "|&",
    "tee",
    "$log_file",
    "<var>",
}

_GENERIC_PACKAGES = {
    "build-essential",
    "clang",
    "cmake",
    "curl",
    "diffutils",
    "gawk",
    "gcc",
    "gcc-c++",
    "git",
    "gnupg",
    "libssl-dev",
    "lld",
    "make",
    "ninja",
    "ninja-build",
    "patch",
    "pkg-config",
    "protobuf",
    "protobuf-devel",
    "python3",
    "python3-dev",
    "python3-devel",
    "python3-pip",
    "python3-venv",
    "software-properties-common",
    "unzip",
    "wget",
    "which",
}

_VERSIONED_PACKAGE_BASES = (
    "clang",
    "gcc",
    "g++",
    "python",
    "openjdk",
    "java",
)

_PACKAGE_MANAGER_ALIASES = {
    "apt": "apt",
    "apt-get": "apt",
    "dnf": "dnf",
    "yum": "yum",
    "zypper": "zypper",
}

_WORKFLOW_ACTIONS = {
    ActionType.DOWNLOAD,
    ActionType.EXTRACT,
    ActionType.INSTALL_PACKAGE,
    ActionType.CLONE_REPOSITORY,
    ActionType.CHECKOUT_SOURCE,
    ActionType.APPLY_PATCH,
    ActionType.BUILD_IMAGE,
    ActionType.RUN_TEST,
}


def _meaningful_packages(step: BuildStep) -> set[str]:
    packages = set(step.packages)
    for command in step.commands:
        packages.update(_packages_from_command(command))
    return {
        _normalize_package(package)
        for package in packages
        if _normalize_package(package)
        and _normalize_package(package) not in _PACKAGE_NOISE
    }


def _packages_from_command(command: str) -> set[str]:
    """Extract package names from package-manager commands and bare package lines."""
    try:
        words = shlex.split(command, comments=False)
    except ValueError:
        words = command.split()
    if not words:
        return set()

    manager_index = next(
        (
            index
            for index, word in enumerate(words[:-1])
            if word in {"apt", "apt-get", "dnf", "yum", "zypper"}
            and words[index + 1] in {"install", "remove"}
        ),
        None,
    )
    if manager_index is not None:
        packages: set[str] = set()
        for word in words[manager_index + 2:]:
            if word in {"|", "|&", "&&", ";"} or word.startswith((">", "<")):
                break
            if not word.startswith("-"):
                packages.add(word)
        return packages

    if (
        len(words) >= 2
        and not any(
            word in {
                "sudo",
                "apt",
                "apt-get",
                "dnf",
                "yum",
                "zypper",
                "update",
                "install",
                "remove",
                "source",
                "cd",
                "wget",
                "curl",
                "git",
                "chmod",
                "rm",
                "ln",
                "sed",
            }
            for word in words
        )
        and all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+:-]*", word)
        for word in words
        )
    ):
        return set(words)
    return set()


def _package_managers(step: BuildStep) -> set[str]:
    """Return package-manager families explicitly used by one step."""
    managers: set[str] = set()
    for command in step.commands:
        try:
            words = shlex.split(command, comments=False)
        except ValueError:
            words = command.split()
        managers.update(
            _PACKAGE_MANAGER_ALIASES[word]
            for word in words
            if word in _PACKAGE_MANAGER_ALIASES
        )
    return managers


def _versioned_package_disagreement(
    wiki_packages: set[str],
    script_packages: set[str],
) -> bool:
    """Detect explicit package-family version differences.

    Wiki pages commonly list alternatives for several distributions in one
    logical step. A different package *set* is therefore not automatically a
    conflict. A concrete family difference such as ``clang7`` versus
    ``clang13`` is meaningful and should remain actionable.
    """
    for base in _VERSIONED_PACKAGE_BASES:
        wiki_versions = {
            package[len(base):]
            for package in wiki_packages
            if package.startswith(base)
            and package[len(base):].isdigit()
        }
        script_versions = {
            package[len(base):]
            for package in script_packages
            if package.startswith(base)
            and package[len(base):].isdigit()
        }
        if wiki_versions and script_versions and wiki_versions.isdisjoint(script_versions):
            return True
    return False


def _canonical_repository(value: str) -> str:
    return value.strip().lower().rstrip("/").removesuffix(".git")


def _step_urls(step: BuildStep) -> set[str]:
    urls = {url.strip().lower().rstrip("/") for url in step.urls if url}
    urls.update(
        url.lower().rstrip("/")
        for url in re.findall(r"https?://[^\s\"']+", " ".join(step.commands))
    )
    resolved_environment = _resolved_environment(step.environment)
    urls.update(
        url.lower().rstrip("/\"'")
        for value in resolved_environment.values()
        for url in re.findall(r"https?://[^\s\"']+", value)
    )
    return urls


def _resolved_environment(environment: dict[str, str]) -> dict[str, str]:
    """Resolve simple Bash variable references in step-local values."""
    resolved = dict(environment)
    variable = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)")
    for _ in range(len(resolved) + 1):
        changed = False
        for name, value in resolved.items():
            updated = variable.sub(
                lambda match: resolved.get(
                    match.group("braced") or match.group("plain"),
                    match.group(0),
                ),
                value,
            )
            if updated != value:
                resolved[name] = updated
                changed = True
        if not changed:
            break
    return resolved


def _strong_identity_candidates(
    wiki_step: BuildStep,
    bash_steps: list[BuildStep],
) -> list[BuildStep]:
    """Find Bash steps whose source identity is stronger than text similarity."""
    if wiki_step.repository:
        repository = _canonical_repository(wiki_step.repository)
        candidates = [
            step
            for step in bash_steps
            if step.repository
            and _canonical_repository(step.repository) == repository
        ]
        if candidates:
            return candidates

    wiki_urls = _step_urls(wiki_step)
    if wiki_urls:
        candidates = [
            step for step in bash_steps if wiki_urls & _step_urls(step)
        ]
        if candidates:
            return candidates

    if wiki_step.component:
        component = re.sub(r"[^a-z0-9]+", "-", wiki_step.component.lower()).strip("-")
        candidates = [
            step
            for step in bash_steps
            if step.component
            and re.sub(r"[^a-z0-9]+", "-", step.component.lower()).strip("-")
            == component
        ]
        if candidates:
            return candidates
        # A documented tool name can be present in the Wiki description while
        # the Bash step exposes it only through a variable such as GO_URL or
        # MAVEN_VERSION. Let semantic grouping use that context; do not fall
        # through to broad package overlap and attach an unrelated dependency
        # command to the tool workflow.
        return []

    wiki_packages = _meaningful_packages(wiki_step)
    if not wiki_packages:
        return []
    candidates: list[BuildStep] = []
    for step in bash_steps:
        if not _action_compatible(wiki_step, step):
            continue
        common = wiki_packages & _meaningful_packages(step)
        if not common:
            continue
        script_packages = _meaningful_packages(step)
        specific_common = {
            package
            for package in common
            if package not in _GENERIC_PACKAGES
            or re.search(r"\d{2,}", package)
        }
        if (
            common == wiki_packages
            or common == script_packages
            or len(common) >= 2 and specific_common
        ):
            candidates.append(step)
    return candidates


def _identity_strength(wiki_step: BuildStep, script_step: BuildStep) -> float:
    """Rank exact source identities so broad package subsets do not win."""
    if (
        wiki_step.repository
        and script_step.repository
        and _canonical_repository(wiki_step.repository)
        == _canonical_repository(script_step.repository)
    ):
        return 1.0
    if _step_urls(wiki_step) & _step_urls(script_step):
        return 0.95
    if (
        wiki_step.component
        and script_step.component
        and wiki_step.component.lower() == script_step.component.lower()
    ):
        return 0.90
    wiki_packages = _meaningful_packages(wiki_step)
    script_packages = _meaningful_packages(script_step)
    if not wiki_packages or not script_packages:
        return 0.0
    common = wiki_packages & script_packages
    return (
        0.50 + 0.40 * len(common) / max(len(wiki_packages), len(script_packages))
        if common
        else 0.0
    )


def _remove_automation_step(match: StepMatch, step_id: str) -> None:
    steps = [
        step
        for step in (match.automation_steps or ([match.automation_step] if match.automation_step else []))
        if step.id != step_id
    ]
    match.automation_steps = steps
    match.automation_step = steps[0] if steps else None
    if not steps and match.status in {MatchStatus.MATCHED, MatchStatus.CONFLICT}:
        match.status = MatchStatus.MISSING
        match.category = DiscrepancyCategory.MISSING_STEP


def _reconcile_strong_matches(
    matches: list[StepMatch],
    wiki: BuildInstruction,
    bash: BuildInstruction,
) -> None:
    """Correct provider assignments when structured source identity disagrees."""
    by_wiki = {
        match.wiki_step.id: match
        for match in matches
        if match.wiki_step is not None
    }

    # The Wiki may begin by telling the operator to fetch and invoke the same
    # shell script that BuildGuard is analyzing. That bootstrap step cannot be
    # an operation implemented inside the downloaded script.
    for match in by_wiki.values():
        if match.wiki_step is None or not _is_bootstrap_script_instruction(
            match.wiki_step, bash
        ):
            continue
        match.status = MatchStatus.OUT_OF_SCOPE
        match.category = None
        match.recommendation = None
        match.automation_step = None
        match.automation_steps = []
        match.explanation = (
            "This Wiki instruction downloads the automation script being analyzed. "
            "It is a prerequisite for the operator, so it is outside the script's "
            "own build workflow."
        )

    for wiki_step in wiki.steps:
        target = by_wiki.get(wiki_step.id)
        if target is None:
            continue
        candidates = _strong_identity_candidates(wiki_step, bash.steps)
        if not candidates:
            continue
        if not (wiki_step.repository or _step_urls(wiki_step) or wiki_step.component):
            exact_command_candidates = [
                candidate
                for candidate in candidates
                if _command_intents(wiki_step)
                & _command_intents(candidate)
            ]
            if exact_command_candidates:
                candidates = exact_command_candidates
            else:
                strongest = max(
                    _identity_strength(wiki_step, candidate)
                    for candidate in candidates
                )
                candidates = [
                    candidate
                    for candidate in candidates
                    if _identity_strength(wiki_step, candidate)
                    >= strongest - 0.03
                ]
        # Package overlap is useful for grouping platform dependency blocks,
        # but it is weaker than an exact repository or URL. Do not steal a
        # package command that is already attached to another meaningful Wiki
        # action; only reclaim unassigned/extra commands in that case.
        if not (wiki_step.repository or _step_urls(wiki_step) or wiki_step.component):
            available_candidates: list[BuildStep] = []
            for candidate in candidates:
                owner = next(
                    (
                        match
                        for match in matches
                        if candidate.id in {
                            step.id
                            for step in (
                                match.automation_steps
                                or ([match.automation_step] if match.automation_step else [])
                            )
                        }
                    ),
                    None,
                )
                if owner is None or owner is target or owner.status in {
                    MatchStatus.EXTRA,
                    MatchStatus.IMPLEMENTATION_DETAIL,
                }:
                    available_candidates.append(candidate)
            candidates = available_candidates
            if not candidates:
                continue
        target_steps = target.automation_steps or (
            [target.automation_step] if target.automation_step else []
        )
        candidate_ids = {candidate.id for candidate in candidates}
        if candidate_ids.issubset({step.id for step in target_steps}):
            continue

        for candidate in candidates:
            owner = next(
                (
                    match
                    for match in matches
                    if candidate.id in {
                        step.id
                        for step in (
                            match.automation_steps
                            or ([match.automation_step] if match.automation_step else [])
                        )
                    }
                ),
                None,
            )
            if owner is not None and owner is not target:
                _remove_automation_step(owner, candidate.id)

        # A wrong repository/component assignment should not survive beside
        # the exact source identity. The normal grouping pass can add back
        # related patch/build commands after the correct anchor is installed.
        target.automation_steps = candidates
        target.automation_step = candidates[0]
        if _group_conflicts(wiki_step, candidates):
            target.status = MatchStatus.CONFLICT
            target.category = _group_conflict_category(wiki_step, candidates)
            candidates = _conflicting_value_steps(wiki_step, candidates)
            target.automation_steps = candidates
            target.automation_step = candidates[0]
        else:
            target.status = MatchStatus.MATCHED
            target.category = None
        target.explanation = (
            _group_conflict_explanation(wiki_step, candidates)
            if target.status == MatchStatus.CONFLICT
            else (
                "Matched by exact repository, URL, component, or package identity; "
                "the provider's weaker text-based assignment was corrected."
            )
        )


def _semantic_score(wiki: BuildStep, script: BuildStep) -> float:
    score = similarity(wiki, script)
    command_overlap = _command_intents(wiki) & _command_intents(script)
    if command_overlap:
        # A Wiki block may contain several commands while Bash exposes them
        # as separate steps. Matching the normalized command intents lets
        # mkdir/cp/extract/configure operations form one semantic action.
        score += min(0.56, 0.46 + len(command_overlap) * 0.08)
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


def _command_intents(step: BuildStep) -> set[str]:
    return {
        normalized
        for command in step.commands
        if (normalized := normalize_command(command))
    }


def _command_intents_without_failure_handler(step: BuildStep) -> set[str]:
    """Normalize commands after removing explicit shell error-reporting syntax."""
    return {
        normalized
        for command in step.commands
        if (
            normalized := normalize_command(
                _EXPLICIT_FAILURE_HANDLER.sub("", command).strip()
            )
        )
    }


def _workflow_actions(step: BuildStep) -> set[ActionType | str]:
    """Extract substantive operations from a step and its command text.

    A parser may represent a whole Wiki code block as one ``download`` step,
    while the Bash parser exposes the download and extraction as separate
    steps.  Comparing these coarse action labels alone loses that distinction,
    so the command text contributes normalized workflow operations as well.
    """
    actions: set[ActionType | str] = set()
    if step.action in _WORKFLOW_ACTIONS:
        actions.add(step.action)

    command_text = "\n".join(step.commands).lower()
    if re.search(r"(?:^|[;&|])\s*(?:curl|wget|fetch)\b", command_text):
        actions.add(ActionType.DOWNLOAD)
    if re.search(r"\b(?:tar|unzip|7z)\s+(?:-[a-z]*x|x(?:[a-z]+)?\b)", command_text):
        actions.add(ActionType.EXTRACT)
    if re.search(r"\bgit\s+clone\b", command_text):
        actions.add(ActionType.CLONE_REPOSITORY)
    if re.search(r"\bgit\s+(?:checkout|switch|reset)\b", command_text):
        actions.add(ActionType.CHECKOUT_SOURCE)
    if re.search(r"\b(?:git\s+apply|patch\s+)", command_text):
        actions.add(ActionType.APPLY_PATCH)
    if re.search(r"\bdocker\s+build\b", command_text):
        actions.add(ActionType.BUILD_IMAGE)
    if re.search(r"\b(?:apt|apt-get|dnf|yum|zypper)\s+[^\n;&|]*\binstall\b", command_text):
        actions.add(ActionType.INSTALL_PACKAGE)
    if re.search(r"\b(?:apt|apt-get)\s+update\b", command_text):
        actions.add("package_index_update")
    if re.search(r"\bmake\b[^\n;&|]*(?:\s-j(?:\s|\d|\$)|\s--jobs(?:=|\s))", command_text):
        actions.add("parallel_build")
    return actions


def _is_bootstrap_script_instruction(
    wiki_step: BuildStep,
    bash: BuildInstruction,
) -> bool:
    """Identify Wiki instructions that download the very script being checked.

    These are operator bootstrap directions, not operations the script can
    perform on itself. Matching the downloaded filename to the script source
    identity keeps this rule applicable across packages and repositories.
    """
    if wiki_step.action not in {ActionType.DOWNLOAD, ActionType.RUN_COMMAND}:
        return False
    labels = " ".join(
        value
        for value in (
            wiki_step.description,
            wiki_step.context.subsection,
            " ".join(wiki_step.context.heading_path),
        )
        if value
    ).lower()
    if not re.search(r"\b(?:build\s+)?script\b", labels):
        return False

    script_names: set[str] = set()
    if bash.source_metadata is not None:
        script_names.add(bash.source_metadata.filename.lower())
        script_names.add(bash.source_metadata.path.rsplit("/", 1)[-1].lower())
    if bash.name:
        script_names.add(bash.name.rsplit("/", 1)[-1].lower())
    script_names.update(
        step.context.source_name.rsplit("/", 1)[-1].lower()
        for step in bash.steps
        if step.context.source_name
    )
    script_names.discard("")
    if not script_names:
        return False

    urls = set(wiki_step.urls)
    urls.update(re.findall(r"https?://[^\s\"'<>]+", " ".join(wiki_step.commands)))
    for url in urls:
        filename = unquote(urlparse(url.rstrip(".,);]")).path.rsplit("/", 1)[-1]).lower()
        if filename in script_names:
            return True
    return False


def _parallelism_values(steps: list[BuildStep]) -> set[tuple[str, str]]:
    """Return concrete/dynamic make job settings found in grouped commands."""
    settings: set[tuple[str, str]] = set()
    for step in steps:
        for command in step.commands:
            if not re.search(r"\bmake\b", command):
                continue
            match = re.search(
                r"(?:^|\s)(?:-j\s*(?P<short>[^\s;&|]+)?|--jobs(?:=\s*|\s+)(?P<long>[^\s;&|]+))",
                command,
            )
            if not match:
                continue
            value = match.group("short") or match.group("long") or "unbounded"
            value = value.strip().strip("'\"")
            if value.isdigit():
                settings.add(("fixed", value))
            elif re.search(r"nproc|sysctl|cpu", value, re.I):
                settings.add(("host-dependent", value))
            elif value.startswith("$") or "$" in value:
                settings.add(("variable", value))
            else:
                settings.add(("symbolic", value))
    return settings


def _parallelism_conflict(
    wiki: BuildStep,
    group: list[BuildStep],
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    wiki_values = _parallelism_values([wiki])
    script_values = _parallelism_values(group)
    for wiki_value in sorted(wiki_values):
        for script_value in sorted(script_values):
            if wiki_value != script_value and wiki_value[0] == "fixed":
                return wiki_value, script_value
            if wiki_value != script_value and script_value[0] == "fixed":
                return wiki_value, script_value
    return None


def _configuration_supports_workflow(wiki: BuildStep, script: BuildStep) -> bool:
    """Whether a configuration assignment supplies evidence for a Wiki action.

    Version/ref/url assignments are not build actions by themselves, but they
    are essential evidence when the consuming download/checkout command is in
    another Bash AST node.  This deliberately requires a component/family or
    configuration-name relationship so unrelated global settings are not
    pulled into every match.
    """
    if script.action != ActionType.SET_VARIABLE:
        return False
    shared_families = _identity_families(wiki) & _families(script) - {"docker"}
    if shared_families:
        return True
    wiki_text = re.sub(
        r"[^a-z0-9]+",
        "",
        " ".join(
            value
            for value in (
                wiki.description,
                wiki.component,
                wiki.repository,
                *wiki.environment.keys(),
                *wiki.commands,
            )
            if value
        ).lower(),
    )
    generic_families = {
        "package", "source", "build", "version", "path", "root", "log",
        "temp", "home", "file", "name",
    }
    for name in script.environment:
        normalized_name = re.sub(r"[^a-z0-9]+", "", name.lower())
        if not normalized_name.endswith(("version", "commit", "ref", "url", "tar", "image")):
            continue
        variable_family = re.sub(
            r"(?:version|commit|ref|url|tar|image)$", "", normalized_name
        )
        if (
            variable_family
            and variable_family not in generic_families
            and variable_family in wiki_text
        ):
            return True
    return False


def _workflow_missing_actions(
    wiki: BuildStep,
    group: list[BuildStep],
) -> list[str]:
    """Find substantive operations documented in a Wiki action but absent from
    its grouped Bash implementation.

    Installation has two valid implementation forms: package installation or
    download/extract/configure.  Therefore a Wiki package-install action does
    not require a literal package-manager command when the script implements a
    complete archive-based installation workflow.
    """
    if not group:
        return []
    required = _workflow_actions(wiki)
    implemented = {
        action
        for step in group
        for action in _workflow_actions(step)
    }
    missing: list[str] = []

    if ActionType.INSTALL_PACKAGE in required:
        if (
            ActionType.INSTALL_PACKAGE not in implemented
            and not {
                ActionType.DOWNLOAD,
                ActionType.EXTRACT,
            }.issubset(implemented)
        ):
            missing.append("package installation")

    for action, label in (
        (ActionType.DOWNLOAD, "download"),
        (ActionType.EXTRACT, "archive extraction"),
        (ActionType.CLONE_REPOSITORY, "repository clone"),
        (ActionType.CHECKOUT_SOURCE, "source checkout"),
        (ActionType.APPLY_PATCH, "patch application"),
        (ActionType.BUILD_IMAGE, "Docker image build"),
        (ActionType.RUN_TEST, "test execution"),
    ):
        if action in required and action not in implemented:
            missing.append(label)

    if "package_index_update" in required and ActionType.INSTALL_PACKAGE in implemented:
        wiki_managers = _package_managers(wiki)
        script_managers = {
            manager
            for step in group
            for manager in _package_managers(step)
        }
        if (
            not wiki_managers
            or not script_managers
            or wiki_managers & script_managers
        ) and "package_index_update" not in implemented:
            missing.append("package-index refresh")
    return missing


def _package_difference(
    wiki: BuildStep,
    group: list[BuildStep],
) -> tuple[set[str], set[str]]:
    """Return Wiki-missing and script-only packages for a matching branch."""
    wiki_packages = _meaningful_packages(wiki)
    if not wiki_packages:
        return set(), set()
    relevant_steps = _relevant_package_steps(wiki, group)
    if not relevant_steps:
        return set(), set()
    script_packages = set().union(
        *(_meaningful_packages(step) for step in relevant_steps)
    )
    return wiki_packages - script_packages, script_packages - wiki_packages


def _relevant_package_steps(
    wiki: BuildStep,
    group: list[BuildStep],
) -> list[BuildStep]:
    """Select script dependency commands from the same distro branch."""
    wiki_managers = _package_managers(wiki)
    return [
        step
        for step in group
        if ActionType.INSTALL_PACKAGE in _workflow_actions(step)
        and (
            not wiki_managers
            or not _package_managers(step)
            or wiki_managers & _package_managers(step)
        )
    ]


def _workflow_explanation(missing: list[str]) -> str:
    labels = ", ".join(missing)
    return (
        f"The Wiki action includes {labels}, but no corresponding operation was "
        "found in the grouped Bash workflow. The other related commands are "
        "shown as evidence, so this is a missing part of one documented action "
        "rather than a separate finding for every shell statement."
    )


def _has_global_pipeline_failure_protection(step: BuildStep) -> bool:
    labels = set(step.context.context_labels)
    return {"shell_errexit", "shell_pipefail"}.issubset(labels)


def _patch_targets(step: BuildStep) -> set[str]:
    values = list(step.patches) + list(step.paths)
    values.extend(
        re.findall(r"[A-Za-z0-9_.-]+\.patch\b", " ".join(step.commands), re.I)
    )
    return {
        value.rsplit("/", 1)[-1].lower()
        for value in values
        if value.lower().endswith(".patch")
    }


def _is_failure_handling_only_conflict(match: StepMatch) -> bool:
    """Treat explicit Wiki error reporting as equivalent to safe Bash defaults.

    A Wiki often writes ``command || error \"step\"`` while a script enables
    ``set -e -o pipefail`` once at the top. The latter is not a missing build
    operation; it is a different way to propagate the same failure. This
    exception is deliberately narrow: the underlying normalized operations
    must match, all relevant script steps must carry both shell options, and
    concrete patch targets must not disagree.
    """
    if match.status != MatchStatus.CONFLICT:
        return False
    wiki = match.wiki_step
    script_steps = match.automation_steps or (
        [match.automation_step] if match.automation_step else []
    )
    if wiki is None or not script_steps:
        return False
    if not any(_EXPLICIT_FAILURE_HANDLER.search(command) for command in wiki.commands):
        return False
    if not all(_has_global_pipeline_failure_protection(step) for step in script_steps):
        return False

    wiki_core = _command_intents_without_failure_handler(wiki)
    script_core = {
        intent
        for step in script_steps
        for intent in _command_intents_without_failure_handler(step)
    }
    if not wiki_core or not wiki_core.issubset(script_core):
        return False

    wiki_patches = _patch_targets(wiki)
    script_patches = {
        patch
        for step in script_steps
        for patch in _patch_targets(step)
    }
    if wiki_patches and script_patches and wiki_patches.isdisjoint(script_patches):
        return False
    return not _group_conflicts(wiki, script_steps)


def _same_workflow(wiki: BuildStep, primary: BuildStep, candidate: BuildStep) -> bool:
    primary_families = _families(primary)
    candidate_families = _families(candidate)
    candidate_function = (candidate.context.subsection or "").lower()
    wiki_text = _text(wiki)

    if candidate_function == "cleanup":
        return False
    wiki_commands = _command_intents(wiki)
    primary_commands = _command_intents(primary)
    tool_families = {"java", "go", "nodejs", "maven", "antlr"}
    shared_tool_families = (
        _families(primary) & _families(candidate) & tool_families
    )
    if shared_tool_families and (
        primary.context.subsection != candidate.context.subsection
    ):
        return False
    if (
        wiki.action == ActionType.INSTALL_PACKAGE
    ):
        if "package_index_update" in _workflow_actions(candidate):
            wiki_managers = _package_managers(wiki)
            candidate_managers = _package_managers(candidate)
            return bool(
                (not wiki_managers or not candidate_managers or wiki_managers & candidate_managers)
                and (
                    not primary.context.subsection
                    or not candidate.context.subsection
                    or primary.context.subsection == candidate.context.subsection
                    or abs(candidate.order - primary.order) <= 12
                )
            )
        if (
            "java" in _families(wiki)
            and "java" in (primary_families & candidate_families)
            and primary.context.subsection
            and primary.context.subsection == candidate.context.subsection
        ):
            return True
        if candidate.action == ActionType.INSTALL_PACKAGE:
            wiki_packages = _meaningful_packages(wiki)
            candidate_packages = _meaningful_packages(candidate)
            common_packages = wiki_packages & candidate_packages
            if not common_packages:
                return False
            overlap = len(common_packages) / max(
                len(wiki_packages),
                len(candidate_packages),
                1,
            )
            # Package-manager commands from different distro branches are not
            # one workflow merely because they share generic tools such as
            # curl and make. Keep a close package match, allowing a concrete
            # version difference to be reported as a conflict.
            return overlap >= 0.75
        if candidate.action == ActionType.CONFIGURE:
            candidate_command = " ".join(candidate.commands).lower()
            wiki_command = " ".join(wiki.commands).lower()
            # Keep platform-specific compiler alternatives with the SLES
            # dependency block that documents them. Do not attach those
            # commands to unrelated yum/apt branches just because they share
            # a helper function.
            if "update-alternatives" in candidate_command:
                return "update-alternatives" in wiki_command
        return bool(
            primary.context.subsection
            and primary.context.subsection == candidate.context.subsection
            and candidate.action
            in {
                ActionType.CONFIGURE,
                ActionType.CREATE_DIRECTORY,
                ActionType.SET_VARIABLE,
            }
        )
    if wiki_commands & _command_intents(candidate):
        return True
    if wiki_commands & primary_commands:
        # The Wiki action already has a concrete command anchor. Do not pull
        # unrelated nearby commands from the same Bash helper into it.
        if (
            primary.context.subsection
            and primary.context.subsection == candidate.context.subsection
            and primary.action in {
                ActionType.DOWNLOAD,
                ActionType.EXTRACT,
                ActionType.RUN_COMMAND,
                ActionType.RUN_TEST,
            }
            and candidate.action in {
                ActionType.EXTRACT,
                ActionType.CONFIGURE,
                ActionType.CREATE_DIRECTORY,
                ActionType.RUN_COMMAND,
                ActionType.RUN_TEST,
                ActionType.SET_VARIABLE,
            }
            and abs(candidate.order - primary.order) <= 4
        ):
            return True
        return False
    if "java" in _families(wiki):
        wiki_component = (wiki.component or "").lower()
        if "base-java" in wiki_component or "base java" in wiki_text:
            return (
                "basejava" in candidate_function
                or "base-java" in (candidate.component or "").lower()
                or "base-java" in (candidate.repository or "").lower()
            )
        if (
            wiki.action == ActionType.RUN_TEST
            and primary.context.subsection
            and primary.context.subsection == candidate.context.subsection
            and candidate.action == ActionType.RUN_TEST
        ):
            return True
        return (
            "installjava" in candidate_function
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

    # A repository, URL, component, or package identity can anchor a larger
    # documented platform/workflow block even when the shell parser has no
    # function name for the surrounding commands. This covers, for example,
    # the Ubuntu LLVM download followed by lld installation and compiler
    # configuration.
    if (
        not primary_families
        and not primary.context.subsection
        and not candidate.context.subsection
        and -3 <= candidate.order - primary.order <= 4
        and _strong_identity_candidates(wiki, [primary])
        and (
            _action_compatible(primary, candidate)
            or _is_implementation_detail(candidate)
        )
    ):
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
    wiki_families = _families(wiki)
    component_family = _component_tool_family(wiki.component)
    family_options: list[tuple[bool, bool, float, list[BuildStep]]] = []
    use_tool_filter = wiki.action in {
        ActionType.INSTALL_PACKAGE,
        ActionType.DOWNLOAD,
        ActionType.EXTRACT,
        ActionType.RUN_TEST,
    }
    for tool_family in ("java", "go", "nodejs", "maven", "antlr"):
        if not use_tool_filter:
            break
        if tool_family not in wiki_families:
            continue
        family_candidates = [
            step for step in candidates
            if tool_family in _families(step)
        ]
        workflow_candidates = [
            step for step in family_candidates
            if _is_workflow_contributor(step)
        ]
        if workflow_candidates:
            family_candidates = workflow_candidates
        if family_candidates:
            command_anchor = any(
                _command_intents(wiki) & _command_intents(step)
                for step in family_candidates
            )
            family_options.append(
                (
                    command_anchor,
                    tool_family == component_family,
                    max(_semantic_score(wiki, step) for step in family_candidates),
                    family_candidates,
                )
            )
    if family_options:
        candidates = max(family_options, key=lambda item: item[:3])[-1]
    scored = sorted(
        ((_semantic_score(wiki, step), step) for step in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    identity_ids = {
        step.id
        for step in _strong_identity_candidates(wiki, candidates)
    }
    scored.sort(
        key=lambda item: (
            item[1].id in identity_ids,
            _identity_strength(wiki, item[1]),
            item[0],
            not _is_implementation_detail(item[1]),
        ),
        reverse=True,
    )
    if not scored:
        return [], 0.0

    score, primary = scored[0]
    threshold = 0.42 if "java" in wiki_families else 0.52
    if score < threshold and primary.id not in identity_ids:
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

    # Version/ref/url assignments are evidence for the grouped action, not
    # independent Bash operations. Keep them with their consuming workflow so
    # concrete values can be compared across AST nodes.
    for candidate in script_steps:
        if candidate in group or not _configuration_supports_workflow(wiki, candidate):
            continue
        group.append(candidate)
    return sorted(group, key=lambda step: step.order), score


def _group_conflicts(wiki: BuildStep, group: list[BuildStep]) -> bool:
    """Detect concrete value disagreement after operation identity is known."""
    if not group:
        return False
    script = group
    if _parallelism_conflict(wiki, script):
        return True
    if wiki.version and _is_concrete(wiki.version):
        values: set[str] = set()
        for step in script:
            shared_families = _families(wiki) & _families(step) - {"docker"}
            if step.version and (
                _same_component_identity(wiki, step)
                or bool(shared_families)
            ):
                values.add(step.version)
            values.update(_relevant_version_values(wiki, step, script))
        concrete_values = {value for value in values if _is_concrete(value)}
        if concrete_values and _version_key(wiki.version) not in {
            _version_key(value) for value in concrete_values
        }:
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
        commit_keys = _wiki_config_keys(wiki, "COMMIT")
        values = {
            step.environment[name]
            for step in script
            for name in commit_keys
            if name in step.environment and _is_concrete(step.environment[name])
        }
        if not commit_keys:
            values = {
                step.commit
                for step in script
                if step.commit
                and _is_concrete(step.commit)
                and _related_value_step(wiki, step)
            }
        if values and wiki.commit not in values:
            return True
    if wiki.repository and _is_concrete(wiki.repository):
        values = {
            step.repository for step in script
            if step.repository
            and _is_concrete(step.repository)
            and _related_value_step(wiki, step)
        }
        if values and wiki.repository not in values:
            return True
    if wiki.docker_image and _is_concrete(wiki.docker_image):
        values = {
            step.docker_image for step in script
            if step.docker_image
            and _is_concrete(step.docker_image)
            and _related_value_step(wiki, step)
        }
        if values and wiki.docker_image not in values:
            return True

    if wiki.packages:
        missing_packages, extra_packages = _package_difference(wiki, script)
        wiki_packages = _meaningful_packages(wiki)
        relevant_package_steps = _relevant_package_steps(wiki, script)
        script_packages = {
            _normalize_package(package)
            for step in relevant_package_steps
            for package in _meaningful_packages(step)
        }
        if (
            missing_packages
            or extra_packages
            or script_packages and (
                wiki_packages.isdisjoint(script_packages)
                or _versioned_package_disagreement(wiki_packages, script_packages)
            )
        ):
            return True
    return False


def _group_conflict_category(
    wiki: BuildStep, group: list[BuildStep]
) -> DiscrepancyCategory:
    """Use a category that describes the concrete values that disagree."""
    if _parallelism_conflict(wiki, group):
        return DiscrepancyCategory.CONFIGURATION_MISMATCH
    if wiki.packages:
        missing_packages, extra_packages = _package_difference(wiki, group)
        wiki_packages = _meaningful_packages(wiki)
        relevant_package_steps = _relevant_package_steps(wiki, group)
        script_packages = set().union(
            *(_meaningful_packages(step) for step in relevant_package_steps)
        ) if relevant_package_steps else set()
        if (
            missing_packages
            or extra_packages
            or _versioned_package_disagreement(wiki_packages, script_packages)
        ):
            return DiscrepancyCategory.DEPENDENCY_INCONSISTENCY
    return DiscrepancyCategory.VERSION_INCONSISTENCY


def _group_conflict_explanation(wiki: BuildStep, group: list[BuildStep]) -> str:
    """Explain a recovered conflict with the differing values when available."""
    parallelism = _parallelism_conflict(wiki, group)
    if parallelism:
        wiki_setting, script_setting = parallelism
        wiki_jobs = (
            wiki_setting[1]
            if wiki_setting[0] == "fixed"
            else f"{wiki_setting[1]} ({wiki_setting[0]})"
        )
        script_jobs = (
            script_setting[1]
            if script_setting[0] == "fixed"
            else f"{script_setting[1]} ({script_setting[0]})"
        )
        return (
            f"The Wiki runs make with {wiki_jobs} parallel job(s), while the script "
            f"uses {script_jobs}. This changes build parallelism and resource use; "
            "confirm whether the fixed limit or host-dependent setting is intended."
        )
    wiki_packages = _meaningful_packages(wiki)
    relevant_package_steps = _relevant_package_steps(wiki, group)
    script_packages = set().union(
        *(_meaningful_packages(step) for step in relevant_package_steps)
    ) if relevant_package_steps else set()
    for base in _VERSIONED_PACKAGE_BASES:
        wiki_values = sorted(
            package for package in wiki_packages
            if package.startswith(base) and package[len(base):].isdigit()
        )
        script_values = sorted(
            package for package in script_packages
            if package.startswith(base) and package[len(base):].isdigit()
        )
        if wiki_values and script_values and set(wiki_values).isdisjoint(script_values):
            package_family = base.rstrip("-_")
            return (
                f"The Wiki specifies {', '.join(wiki_values)}, but the script installs "
                f"{', '.join(script_values)}. These different {package_family} major "
                "versions mean the documented and automated build environments differ, "
                "which may affect compatibility or build results. BuildGuard cannot determine which "
                "version is intended; confirm the required version and align both sources."
            )
    if wiki.version and _is_concrete(wiki.version):
        script_versions = {
            step.version
            for step in group
            if step.version
            and _is_concrete(step.version)
            and (
                _same_component_identity(wiki, step)
                or bool(_families(wiki) & _families(step) - {"docker"})
            )
        }
        for step in group:
            script_versions.update(
                value
                for value in _relevant_version_values(wiki, step, group)
                if _is_concrete(value)
            )
        differing_versions = sorted(
            value
            for value in script_versions
            if _version_key(value) != _version_key(wiki.version)
        )
        if differing_versions:
            return (
                f"The Wiki specifies version {wiki.version}, while the script uses "
                f"{', '.join(differing_versions)} for the corresponding component. "
                "Confirm which version is intended and align both sources."
                + _missing_workflow_suffix(wiki, group)
            )
    if wiki.commit and _is_concrete(wiki.commit):
        commit_keys = _wiki_config_keys(wiki, "COMMIT")
        script_commits = {
            step.environment[name]
            for step in group
            for name in commit_keys
            if name in step.environment and _is_concrete(step.environment[name])
        }
        if not commit_keys:
            script_commits = {
                step.commit
                for step in group
                if step.commit
                and _is_concrete(step.commit)
                and _related_value_step(wiki, step)
            }
        differing_commits = sorted(script_commits - {wiki.commit})
        if differing_commits:
            return (
                f"The Wiki specifies commit {wiki.commit}, while the script uses "
                f"{', '.join(differing_commits)} for the corresponding source. "
                "Confirm the intended revision and align both sources."
            )
    if wiki.repository and _is_concrete(wiki.repository):
        script_repositories = {
            step.repository
            for step in group
            if step.repository
            and _is_concrete(step.repository)
            and _related_value_step(wiki, step)
        }
        differing_repositories = sorted(script_repositories - {wiki.repository})
        if differing_repositories:
            return (
                f"The Wiki points to {wiki.repository}, while the script points to "
                f"{', '.join(differing_repositories)}. Confirm which repository is "
                "intended and align both sources."
            )
    if wiki.docker_image and _is_concrete(wiki.docker_image):
        script_images = {
            step.docker_image
            for step in group
            if step.docker_image
            and _is_concrete(step.docker_image)
            and _related_value_step(wiki, step)
        }
        differing_images = sorted(script_images - {wiki.docker_image})
        if differing_images:
            return (
                f"The Wiki specifies image {wiki.docker_image}, while the script uses "
                f"{', '.join(differing_images)}. Confirm the intended image and align "
                "both sources."
            )
    missing_packages, extra_packages = _package_difference(wiki, group)
    if missing_packages or extra_packages:
        details: list[str] = []
        if missing_packages:
            details.append(f"the Wiki lists {', '.join(sorted(missing_packages))}")
        if extra_packages:
            details.append(
                f"the script additionally installs {', '.join(sorted(extra_packages))}"
            )
        return (
            "The dependency lists differ for the same package-manager branch: "
            + "; ".join(details)
            + ". Confirm whether the dependency list or the source is outdated."
            + _missing_workflow_suffix(wiki, group)
        )
    return (
        "The script implements this grouped build action, but a concrete version, "
        "commit, repository, image, or package value differs."
    )


def _relevant_version_values(
    wiki: BuildStep,
    script: BuildStep,
    related_steps: list[BuildStep] | None = None,
) -> set[str]:
    """Return script version values belonging to the Wiki action's component."""
    identity_parts = (
        (wiki.component, wiki.repository)
        if wiki.component
        else (
            wiki.description,
            wiki.repository,
        )
    )
    identity_text = " ".join(value for value in identity_parts if value).lower()
    identity_families = _identity_families(wiki)
    mentioned_variables = set(_VARIABLE.findall(" ".join(wiki.commands)))
    mentioned_variables = {
        value.strip("${}") for value in mentioned_variables
    }
    values: set[str] = set()
    local_names = set(script.environment)
    environment: dict[str, str] = {}
    for related_step in [*(related_steps or []), script]:
        environment.update(related_step.environment)
    resolved_environment = _resolved_environment(environment)
    for name, value in resolved_environment.items():
        if name not in local_names:
            continue
        upper_name = name.upper()
        if not any(
            marker in upper_name
            for marker in ("VERSION", "RELEASE", "TAG", "REF", "COMMIT")
        ):
            continue
        value_families = _configuration_name_families(name)
        if (
            name in wiki.environment
            or name in mentioned_variables
            or bool(identity_families & value_families)
        ):
            values.add(value)
    # Some parsers retain the version only inside a URL/TAR variable. Resolve
    # those values too when the variable belongs to the Wiki component.
    for name, value in resolved_environment.items():
        if name not in local_names:
            continue
        normalized_name = re.sub(r"[^a-z0-9]+", "", name.lower())
        variable_family = re.sub(
            r"(?:version|release|tag|ref|commit|url|tar)$", "", normalized_name
        )
        if not variable_family:
            continue
        value_families = _configuration_name_families(name)
        if not (
            name in wiki.environment
            or name in mentioned_variables
            or identity_families & value_families
        ):
            continue
        values.update(_VERSION.findall(value))
    return values


def _configuration_name_families(name: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
    families: set[str] = set()
    for family, aliases in {
        "go": ("go", "golang", "goroot"),
        "nodejs": ("node", "nodejs"),
        "maven": ("maven", "mvn"),
        "antlr": ("antlr",),
        "java": ("java", "jdk", "jre", "temurin", "semeru", "openjdk"),
        "activemq": ("activemq",),
        "qemu": ("qemu",),
        "tomcat": ("tomcat",),
        "community-repo": ("communityrepo",),
        "community-share": ("communityshare",),
        "search-services": ("searchservices",),
        "transform-core": ("transformcore",),
        "content-app": ("contentapp",),
        "docker-compose": ("dockercompose",),
    }.items():
        if any(alias in normalized for alias in aliases):
            families.add(family)
    return families


def _missing_workflow_suffix(wiki: BuildStep, group: list[BuildStep]) -> str:
    missing = _workflow_missing_actions(wiki, group)
    return f" The grouped workflow also omits: {', '.join(missing)}." if missing else ""


def _wiki_config_keys(wiki: BuildStep, suffix: str) -> set[str]:
    pattern = re.compile(rf"\b[A-Z][A-Z0-9_]*{suffix}\b")
    text = " ".join(
        [wiki.description, *wiki.commands, *wiki.environment.keys()]
    )
    return {
        name for name in wiki.environment if name.upper().endswith(suffix)
    } | set(pattern.findall(text))


def _related_value_step(wiki: BuildStep, script: BuildStep) -> bool:
    if _same_component_identity(wiki, script):
        return True
    if _families(wiki) & _families(script) - {"docker"}:
        return True
    return (
        wiki.action in {ActionType.CLONE_REPOSITORY, ActionType.CHECKOUT_SOURCE}
        and script.action in {ActionType.CLONE_REPOSITORY, ActionType.CHECKOUT_SOURCE}
    )


def _same_component_identity(left: BuildStep, right: BuildStep) -> bool:
    left_component = re.sub(r"[^a-z0-9]+", "", (left.component or "").lower())
    right_component = re.sub(r"[^a-z0-9]+", "", (right.component or "").lower())
    return bool(
        left_component
        and right_component
        and (
            left_component == right_component
            or left_component in right_component
            or right_component in left_component
        )
    )


def _conflicting_value_steps(
    wiki: BuildStep,
    group: list[BuildStep],
) -> list[BuildStep]:
    """Keep only Bash evidence that carries the concrete conflicting value."""
    conflicting: list[BuildStep] = []
    if wiki.version and _is_concrete(wiki.version):
        for step in group:
            values = set(_relevant_version_values(wiki, step, group))
            shared_families = _families(wiki) & _families(step) - {"docker"}
            if step.version and (
                _same_component_identity(wiki, step) or shared_families
            ):
                values.add(step.version)
            if any(
                _is_concrete(value)
                and _version_key(value) != _version_key(wiki.version)
                for value in values
            ):
                conflicting.append(step)
    if not conflicting and wiki.packages:
        _missing_packages, extra_packages = _package_difference(wiki, group)
        if _missing_packages or extra_packages:
            wiki_managers = _package_managers(wiki)
            conflicting.extend(
                step
                for step in group
                if ActionType.INSTALL_PACKAGE in _workflow_actions(step)
                and (
                    not wiki_managers
                    or not _package_managers(step)
                    or wiki_managers & _package_managers(step)
                )
            )
    if not conflicting and _parallelism_conflict(wiki, group):
        conflicting.extend(
            step
            for step in group
            if _parallelism_values([step])
        )
    if not conflicting and wiki.commit and _is_concrete(wiki.commit):
        commit_keys = _wiki_config_keys(wiki, "COMMIT")
        for step in group:
            values = (
                {step.environment[name] for name in commit_keys if name in step.environment}
                if commit_keys
                else {step.commit} if step.commit and _related_value_step(wiki, step) else set()
            )
            if any(_is_concrete(value) and value != wiki.commit for value in values):
                conflicting.append(step)
    if not conflicting and wiki.repository and _is_concrete(wiki.repository):
        conflicting.extend(
            step for step in group
            if step.repository
            and step.repository != wiki.repository
            and _related_value_step(wiki, step)
        )
    if not conflicting and wiki.docker_image and _is_concrete(wiki.docker_image):
        conflicting.extend(
            step for step in group
            if step.docker_image
            and step.docker_image != wiki.docker_image
            and _related_value_step(wiki, step)
        )
    return conflicting or group


def _is_concrete(value: str) -> bool:
    return not re.search(r"\$|\{|\}|<var>|\$\(", value)


def _version_key(value: str) -> str:
    """Normalize harmless version spelling differences such as ``v1.2.3``."""
    return value.strip().strip("'\"").lower().removeprefix("v")


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
    _reconcile_strong_matches(matches, wiki, bash)
    matches = [
        match
        for match in matches
        if not (
            match.status == MatchStatus.EXTRA
            and match.automation_step is None
            and not match.automation_steps
        )
    ]
    for match in matches:
        if match.automation_step is not None and not match.automation_steps:
            match.automation_steps = [match.automation_step]
    by_wiki = {
        match.wiki_step.id: match
        for match in matches
        if match.wiki_step is not None
    }

    # Do not turn a shell error-reporting style difference into a build
    # discrepancy. For example, ``git apply || error ...`` in the Wiki and
    # ``set -e -o pipefail`` plus ``git apply`` in Bash have the same build
    # failure semantics.
    for match in matches:
        if not _is_failure_handling_only_conflict(match):
            continue
        match.status = MatchStatus.MATCHED
        match.category = None
        match.recommendation = None
        match.explanation = (
            "The Wiki and script perform the same operation. The Wiki adds an "
            "explicit error handler, while the script's global 'set -e -o "
            "pipefail' already propagates command and pipeline failures."
        )

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
            match.category = _group_conflict_category(match.wiki_step, group)
            group = _conflicting_value_steps(match.wiki_step, group)
            match.automation_steps = group
            match.automation_step = group[0]
            match.explanation = _group_conflict_explanation(match.wiki_step, group)
        else:
            missing_actions = _workflow_missing_actions(match.wiki_step, group)
            if missing_actions:
                match.status = MatchStatus.MISSING
                match.category = DiscrepancyCategory.MISSING_STEP
                match.explanation = _workflow_explanation(missing_actions)
            else:
                match.status = MatchStatus.MATCHED
                match.category = None
                match.explanation = (
                    f"Grouped semantic implementation matched the Wiki action "
                    f"(score {score:.2f}); individual shell commands are implementation details."
                )

    # The AI can call an operation a match while overlooking a concrete value
    # difference (for example, Wiki clang7 versus script clang13). Audit those
    # claimed matches against the same semantic group used for missing-step
    # recovery so a confident-but-incorrect AI label cannot erase a conflict.
    for match in by_wiki.values():
        if match.status != MatchStatus.MATCHED or match.wiki_step is None:
            continue
        group, _score = _group_for(match.wiki_step, bash.steps)
        if not group:
            continue
        missing_actions = _workflow_missing_actions(match.wiki_step, group)
        if missing_actions and not _group_conflicts(match.wiki_step, group):
            match.automation_steps = group
            match.automation_step = group[0]
            match.status = MatchStatus.MISSING
            match.category = DiscrepancyCategory.MISSING_STEP
            match.explanation = _workflow_explanation(missing_actions)
            continue
        if not _group_conflicts(match.wiki_step, group):
            continue
        match.automation_steps = group
        match.automation_step = group[0]
        match.status = MatchStatus.CONFLICT
        match.category = _group_conflict_category(match.wiki_step, group)
        group = _conflicting_value_steps(match.wiki_step, group)
        match.automation_steps = group
        match.automation_step = group[0]
        match.explanation = _group_conflict_explanation(match.wiki_step, group)

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
