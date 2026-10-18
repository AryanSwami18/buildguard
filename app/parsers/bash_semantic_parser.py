"""Deterministic Bash-to-BuildStep semantic extraction.

This layer sits after the bashlex syntax parser. It extracts facts from each
actual Bash command and normalizes them into the same BuildStep model used by
the Wiki side. It does not ask an LLM to identify basic command semantics.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from app.models.bash import BashNode
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep
from app.parsers.bash_parser import parse_script
from app.parsers.semantic_parser import parse_semantic_facts


_ASSIGNMENT = re.compile(
    r"^(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$"
)

_URL = re.compile(r"https?://[^\s<>'\"`]+")

_PATH = re.compile(
    r"(?<![A-Za-z0-9_:])/"
    r"(?:[A-Za-z0-9._~${}:+,@%-]+/)*"
    r"[A-Za-z0-9._~${}:+,@%-]+"
)

_GIT_CLONE = re.compile(
    r"""\bgit\s+clone\b
    (?P<options>.*?)\s+
    (?P<url>https?://[^\s;&|"']+|git@[^\s:]+:[^\s;&|"']+)""",
    re.VERBOSE,
)

_GIT_BRANCH = re.compile(
    r"\b(?:-b|--branch)\s+(?P<ref>[^\s;&|]+)"
)

_GIT_CHECKOUT = re.compile(
    r"\bgit\s+(?:checkout|switch)\s+(?P<ref>[^\s;&|]+)"
)

_DOCKER_TAG = re.compile(
    r"(?:-t|--tag)\s+(?P<image>[^\s;&|]+)"
)

_BUILD_ARG = re.compile(
    r"(?:--build-arg|-build-arg)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>[^\s;&|]+)"
)

_PATCH_FILE = re.compile(
    r"([^\s;&|]+\.patch)\b"
)

_JAVA_RUNTIME_OPTION = re.compile(r"\b(?:IBMSemeru|Temurin|OpenJDK)\d+\b")
_JAVA_PROVIDED_ASSIGNMENT = re.compile(
    r"(?m)^\s*(?:export\s+)?JAVA_PROVIDED\s*=\s*(?P<value>[^\r\n]+)"
)
_VARIABLE_REFERENCE = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)"
)

_ENVIRONMENT_EFFECT_NAMES = {
    "PATH",
    "JAVA_HOME",
    "JAVA_PROVIDED",
    "SOURCE_ROOT",
    "ALLOWERASING",
}


@dataclass(frozen=True)
class _FunctionRange:
    name: str
    start: int
    end: int


class BashSemanticParser:
    """Convert Bash command nodes into deterministic BuildStep objects."""

    def parse(
        self,
        script: str,
        name: str = "script",
    ) -> list[BuildStep]:
        parsed = parse_script(script)

        all_commands = self._pipeline_groups(
            script,
            self._command_nodes(parsed.nodes),
        )
        commands = [
            node
            for node in all_commands
            if self._is_build_action(node.text)
        ]
        functions = self._function_ranges(script)
        java_runtime_options = self._java_runtime_options(script, functions)
        java_runtime_default = self._java_runtime_default(script)

        # Global facts provide information that may be spread across
        # multiple commands, for example:
        #
        #   JAVA_BASE_COMMIT="..."
        #
        # followed later by:
        #
        #   git checkout "$JAVA_BASE_COMMIT"
        facts = parse_semantic_facts(script)
        unused_configuration = self._unused_configuration_names(script, facts)
        shell_failure_labels = self._shell_failure_labels(script)

        steps: list[BuildStep] = []

        for index, node in enumerate(commands, start=1):
            function_name = self._function_for_position(
                functions,
                node.position_start,
            )

            previous_id = (
                f"bash-{index - 1}"
                if index > 1
                else None
            )

            next_id = (
                f"bash-{index + 1}"
                if index < len(commands)
                else None
            )

            context = SourceContext(
                source_type="bash",
                source_name=name,
                subsection=function_name,
                location=(
                    f"characters "
                    f"{node.position_start}-"
                    f"{node.position_end}"
                ),
                line_start=self._line_number(
                    script,
                    node.position_start,
                ),
                line_end=self._line_number(
                    script,
                    node.position_end,
                ),
                surrounding_text=node.text,
                context_labels=self._context_labels(
                    node.text,
                    unused_configuration,
                    shell_failure_labels,
                ),
                previous_step_id=previous_id,
                next_step_id=next_id,
            )
            if java_runtime_options and self._uses_java_runtime(
                node.text,
                function_name,
            ):
                context.runtime_options = {
                    "JAVA_PROVIDED": java_runtime_options,
                }
                if java_runtime_default:
                    context.runtime_defaults = {
                        "JAVA_PROVIDED": java_runtime_default,
                    }

            steps.append(
                self._to_step(
                    node=node,
                    index=index,
                    context=context,
                    facts=facts,
                    script=script,
                )
            )

        return steps

    @staticmethod
    def _command_nodes(
        nodes: list[BashNode],
    ) -> list[BashNode]:
        """Return command nodes once, ordered by source position."""

        found: dict[tuple[int, int], BashNode] = {}

        def visit(node: BashNode) -> None:
            if node.kind == "command":
                found[
                    (
                        node.position_start,
                        node.position_end,
                    )
                ] = node

            for child in node.children:
                visit(child)

        for root in nodes:
            visit(root)

        return [
            found[position]
            for position in sorted(found)
        ]

    @staticmethod
    def _pipeline_groups(
        source: str,
        commands: list[BashNode],
    ) -> list[BashNode]:
        """Represent a pipe-connected operation as one command step."""
        groups: list[BashNode] = []
        index = 0

        while index < len(commands):
            first = commands[index]
            last = first
            index += 1

            while index < len(commands):
                next_node = commands[index]
                separator = source[last.position_end:next_node.position_start]
                separator = re.sub(r"\\[ \t]*\r?\n[ \t]*", " ", separator)
                if not re.fullmatch(r"\s*\|&?\s*", separator):
                    break
                last = next_node
                index += 1

            if last is first:
                groups.append(first)
                continue

            start = first.position_start
            end = last.position_end
            groups.append(
                first.model_copy(
                    update={"position_end": end, "text": source[start:end]}
                )
            )

        return groups

    @staticmethod
    def _function_ranges(
        source: str,
    ) -> list[_FunctionRange]:
        """Find Bash function bodies."""

        pattern = re.compile(
            r"(?:function\s+)?"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
            r"\s*(?:\(\s*\))?\s*\{"
        )

        ranges: list[_FunctionRange] = []

        for match in pattern.finditer(source):
            depth = 1
            index = match.end()

            while index < len(source) and depth:
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1

                index += 1

            if depth == 0:
                ranges.append(
                    _FunctionRange(
                        name=match.group("name"),
                        start=match.start(),
                        end=index,
                    )
                )

        return ranges

    @staticmethod
    def _java_runtime_options(
        source: str,
        functions: list[_FunctionRange],
    ) -> list[str]:
        """Read supported Java choices from the script's validation helpers."""
        option_functions = {
            "getjavaurl",
            "isvalidjavaprovided",
            "printhelp",
        }
        support_text = "\n".join(
            source[function.start:function.end]
            for function in functions
            if function.name.lower() in option_functions
        )
        return list(dict.fromkeys(_JAVA_RUNTIME_OPTION.findall(support_text)))

    @staticmethod
    def _java_runtime_default(source: str) -> str | None:
        """Read the first literal default for JAVA_PROVIDED, if one is set."""
        for match in _JAVA_PROVIDED_ASSIGNMENT.finditer(source):
            value = match.group("value").strip().strip("'\"")
            if value and "$" not in value and "`" not in value:
                return value
        return None

    @staticmethod
    def _uses_java_runtime(command: str, function_name: str | None) -> bool:
        if function_name and function_name.lower() in {
            "installjava",
            "getjavaurl",
            "buildalfrescobasejavaimage",
            "buildalfrescobasetomcatimage",
        }:
            return True
        return bool(re.search(r"java|jre|\bjver\b", command, re.IGNORECASE))

    @staticmethod
    def _function_for_position(
        functions: list[_FunctionRange],
        position: int,
    ) -> str | None:
        containing = [
            function
            for function in functions
            if function.start <= position < function.end
        ]

        if not containing:
            return None

        return min(
            containing,
            key=lambda item: item.end - item.start,
        ).name

    @staticmethod
    def _line_number(
        source: str,
        character_position: int,
    ) -> int:
        """Return the one-based source line containing a character offset."""
        return source.count("\n", 0, character_position) + 1

    def _to_step(
        self,
        node: BashNode,
        index: int,
        context: SourceContext,
        facts,
        script: str,
    ) -> BuildStep:
        command = node.text.strip()

        words = self._words(command)
        base_words = self._effective_words(words)

        action = (
            ActionType.SET_VARIABLE
            if not base_words and _ASSIGNMENT.match(command)
            else self._classify(base_words, command)
        )

        packages = self._packages_from_command(
            base_words
        )

        urls = self._urls(command)

        paths = self._paths(command)

        environment = self._assignments(
            words
        )
        environment.update(
            self._referenced_configurations(command, facts)
        )

        build_args = self._build_args(command)

        docker_image = self._docker_image(
            command
        )

        repository, ref, commit = self._git_details(
            command,
            facts,
            script,
            node.position_start,
        )

        patches = self._patches(command)

        working_directory = self._working_directory(
            base_words
        )

        component = self._component(
            command=command,
            docker_image=docker_image,
            repository=repository,
        )

        version = self._version(
            component=component,
            docker_image=docker_image,
            repository=repository,
            facts=facts,
        )

        build_tool = self._build_tool(
            base_words
        )

        description = self._description(
            action=action,
            command=command,
            packages=packages,
            docker_image=docker_image,
            repository=repository,
            component=component,
        )

        return BuildStep(
            id=f"bash-{index}",
            order=index,
            action=action,
            description=description,
            packages=packages,
            commands=[command],
            paths=paths,
            urls=urls,
            user=self._user(command),
            working_directory=working_directory,
            component=component,
            version=version,
            repository=repository,
            ref=ref,
            commit=commit,
            docker_image=docker_image,
            dockerfile=self._dockerfile(command),
            build_args=build_args,
            environment=environment,
            patches=patches,
            build_tool=build_tool,
            context=context,
        )

    @staticmethod
    def _words(command: str) -> list[str]:
        try:
            return shlex.split(
                command,
                comments=False,
            )
        except ValueError:
            return command.replace(
                "\\\n",
                " ",
            ).split()

    @staticmethod
    def _classify(
        words: list[str],
        command: str,
    ) -> ActionType:
        if re.search(r"\|\s*(?:sudo\s+)?git\s+apply\b", command):
            return ActionType.APPLY_PATCH

        if re.search(r"\|\s*xargs\s+(?:sudo\s+)?sed\b", command):
            return ActionType.CONFIGURE

        if not words:
            return ActionType.UNKNOWN

        executable = words[0]
        args = words[1:]
        lowered = [
            item.lower()
            for item in args
        ]

        if executable in {
            "apt",
            "apt-get",
            "dnf",
            "yum",
            "zypper",
        }:
            if "install" in lowered:
                return ActionType.INSTALL_PACKAGE

            if (
                "remove" in lowered
                or "erase" in lowered
            ):
                return ActionType.REMOVE_PACKAGE

        if executable == "git":
            if "clone" in lowered:
                return ActionType.CLONE_REPOSITORY

            if (
                "checkout" in lowered
                or "switch" in lowered
                or "fetch" in lowered
            ):
                return ActionType.CHECKOUT_SOURCE

            if "apply" in lowered:
                return ActionType.APPLY_PATCH

        if executable == "docker":
            if "build" in lowered:
                return ActionType.BUILD_IMAGE

            if any(
                subcommand in lowered
                for subcommand in ("pull", "run", "tag", "compose")
            ):
                return ActionType.RUN_COMMAND

        if executable in {
            "mvn",
            "mvnw",
            "./mvnw",
        }:
            if any(
                arg in {"test", "verify"}
                or arg.endswith(":test")
                for arg in args
            ):
                return ActionType.RUN_TEST

        if executable in {"javac", "diff"}:
            return ActionType.RUN_TEST

        if executable == "java" and any(
            marker in args
            for marker in ("-cp", "-classpath", "org.antlr.v4.Tool", "TestRig")
        ):
            return ActionType.RUN_TEST

        if executable == "patch":
            return ActionType.APPLY_PATCH

        if executable in {
            "curl",
            "wget",
        }:
            return ActionType.DOWNLOAD

        if executable == "mkdir":
            return ActionType.CREATE_DIRECTORY

        if executable == "rm":
            return ActionType.DELETE

        if executable == "cp":
            return ActionType.COPY

        if executable == "mv":
            return ActionType.MOVE

        if executable in {
            "tar",
            "unzip",
        } and any(
            flag in command
            for flag in (
                "-x",
                " x",
                "--extract",
                " -xf",
                " -xzf",
            )
        ):
            return ActionType.EXTRACT

        if executable == "sed":
            return ActionType.CONFIGURE

        if executable == "update-alternatives":
            return ActionType.CONFIGURE

        if executable == "cd":
            return ActionType.RUN_COMMAND

        if executable == "export":
            return ActionType.SET_VARIABLE

        if _ASSIGNMENT.match(command):
            return ActionType.SET_VARIABLE

        return ActionType.RUN_COMMAND

    @staticmethod
    def _effective_words(words: list[str]) -> list[str]:
        """Remove shell prefixes so classification sees the actual command."""
        result = list(words)

        while result:
            if result[0] == "sudo":
                result.pop(0)
                continue

            if _ASSIGNMENT.match(result[0]):
                result.pop(0)
                continue

            if result[0] == "retry":
                result.pop(0)
                continue

            break

        return result

    @classmethod
    def _is_build_action(cls, command: str) -> bool:
        """Keep build operations and useful configuration, not shell plumbing."""
        words = cls._words(command)
        effective = cls._effective_words(words)

        if not effective:
            return cls._is_relevant_assignment(command)

        action = cls._classify(effective, command)
        if action not in {
            ActionType.RUN_COMMAND,
            ActionType.SET_VARIABLE,
            ActionType.UNKNOWN,
        }:
            return True

        if action == ActionType.SET_VARIABLE:
            return cls._is_relevant_assignment(command)

        executable = effective[0]
        args = effective[1:]

        # Package index refreshes and build/test tools are meaningful actions
        # even when they do not fit one of the more specific action types.
        if executable in {"apt", "apt-get", "dnf", "yum", "zypper"}:
            return "update" in args

        if executable in {"mvn", "mvnw", "./mvnw", "gradle", "gradlew", "./gradlew", "make"}:
            return not any(arg in {"--version", "-version", "-v"} for arg in args)

        if executable in {"javac", "diff"}:
            return True

        if executable == "java":
            return any(
                marker in args
                for marker in ("-cp", "-classpath", "org.antlr.v4.Tool", "TestRig")
            )

        if executable in {"npm", "npx", "yarn", "pnpm"}:
            return bool(args) and args[0] not in {"--version", "-v"}

        if executable == "docker":
            return any(
                subcommand in args
                for subcommand in ("pull", "run", "tag", "compose")
            )

        if executable == "chown":
            return True

        # These commands carry out image, source, or Java setup operations.
        # Logging, condition checks, navigation, and helper calls fall through.
        return False

    @staticmethod
    def _is_relevant_assignment(command: str) -> bool:
        match = _ASSIGNMENT.match(command.strip())
        if not match:
            return False

        name = match.group("name").upper()
        return (
            name in {
                "SOURCE_ROOT",
                "JAVA_HOME",
                "JAVA_PROVIDED",
                "PATH",
                "ALLOWERASING",
            }
            or name.endswith(("_VERSION", "_COMMIT", "_URL", "_REF", "_BRANCH"))
        )

    @staticmethod
    def _context_labels(
        command: str,
        unused_configuration: set[str],
        shell_failure_labels: list[str] | None = None,
    ) -> list[str]:
        """Attach script-wide shell semantics and configuration labels.

        ``set -e -o pipefail`` is not a build action by itself, but it changes
        the failure behavior of every later command. Keeping that fact on the
        parsed operations lets the comparison layer distinguish an explicit
        Wiki error handler from equivalent script-wide Bash behavior.
        """
        labels = list(shell_failure_labels or [])
        match = _ASSIGNMENT.match(command.strip())
        if not match:
            return labels

        labels.append("configuration")
        if match.group("name") in unused_configuration:
            labels.append("unused_configuration")
        return labels

    @staticmethod
    def _shell_failure_labels(script: str) -> list[str]:
        """Return global Bash options that affect command failure behavior."""
        set_commands = re.findall(
            r"(?m)^\s*set\b([^\n#]*)",
            script,
        )
        options = " ".join(set_commands).lower()
        labels: list[str] = []
        if re.search(r"(?:^|\s)(?:-e|--errexit)(?:\s|$)", options) or re.search(
            r"(?:^|\s)-o\s+errexit(?:\s|$)",
            options,
        ):
            labels.append("shell_errexit")
        if re.search(r"(?:^|\s)(?:-o\s+)?pipefail(?:\s|$)", options):
            labels.append("shell_pipefail")
        return labels

    @staticmethod
    def _referenced_configurations(command: str, facts) -> dict[str, str]:
        """Attach resolved values to the operation that actually consumes them."""
        values: dict[str, str] = {}
        for match in _VARIABLE_REFERENCE.finditer(command):
            name = match.group("braced") or match.group("plain")
            if name in facts.variables:
                values[name] = facts.variables[name]
        return values

    @staticmethod
    def _unused_configuration_names(script: str, facts) -> set[str]:
        """Find source-defined configuration never read by the script.

        A declaration itself is excluded from the reference scan. Environment
        assignments are treated as used because their effect is observable by
        child processes even without a literal ``$NAME`` expansion.
        """
        used: set[str] = set(_ENVIRONMENT_EFFECT_NAMES)
        for match in _VARIABLE_REFERENCE.finditer(script):
            used.add(match.group("braced") or match.group("plain"))

        return {
            name
            for name in facts.variables
            if name not in used
            and (
                name.upper().endswith(
                    ("_VERSION", "_COMMIT", "_URL", "_REF", "_BRANCH")
                )
                or name.upper() in {"JAVA_PROVIDED", "MAVEN_VERSION"}
            )
        }

    @staticmethod
    def _packages_from_command(
        words: list[str],
    ) -> list[str]:
        if not words or words[0] not in {
            "apt",
            "apt-get",
            "dnf",
            "yum",
            "zypper",
        }:
            return []

        try:
            operation_index = next(
                index
                for index, word in enumerate(words)
                if word in {
                    "install",
                    "remove",
                    "erase",
                }
            )
        except StopIteration:
            return []

        result: list[str] = []

        for token in words[
            operation_index + 1:
        ]:
            if token.startswith("-"):
                continue

            result.append(
                token.rstrip("\\")
            )

        return result

    @staticmethod
    def _urls(command: str) -> list[str]:
        return list(
            dict.fromkeys(
                url.rstrip(".,);]")
                for url in _URL.findall(command)
            )
        )

    @staticmethod
    def _paths(command: str) -> list[str]:
        urls = list(
            _URL.finditer(command)
        )

        paths: list[str] = []

        for match in _PATH.finditer(command):
            if any(
                item.start()
                <= match.start()
                < item.end()
                for item in urls
            ):
                continue

            value = match.group(0).rstrip(
                ".,);]"
            )

            if value not in paths:
                paths.append(value)

        return paths

    @staticmethod
    def _assignments(
        words: list[str],
    ) -> dict[str, str]:
        values: dict[str, str] = {}

        for word in words:
            match = _ASSIGNMENT.match(word)

            if match:
                values[
                    match.group("name")
                ] = match.group("value").strip(
                    "'\""
                )

        return values

    @staticmethod
    def _build_args(
        command: str,
    ) -> dict[str, str]:
        return {
            match.group("name"): match.group(
                "value"
            ).strip("'\"")
            for match in _BUILD_ARG.finditer(
                command
            )
        }

    @staticmethod
    def _docker_image(
        command: str,
    ) -> str | None:
        if not re.search(
            r"\bdocker\s+build\b",
            command,
        ):
            return None

        match = _DOCKER_TAG.search(command)

        if not match:
            return None

        return match.group(
            "image"
        ).strip("'\"")

    @staticmethod
    def _dockerfile(
        command: str,
    ) -> str | None:
        match = re.search(
            r"(?:^|\s)-f\s+([^\s]+)",
            command,
        )

        return (
            match.group(1)
            if match
            else None
        )

    @staticmethod
    def _git_details(
        command: str,
        facts,
        script: str,
        position: int,
    ) -> tuple[
        str | None,
        str | None,
        str | None,
    ]:
        clone = _GIT_CLONE.search(command)

        if clone:
            url = clone.group("url").rstrip(".,);]")

            ref_match = _GIT_BRANCH.search(
                command
            )

            ref = (
                BashSemanticParser._resolve_git_value(
                    ref_match.group("ref"),
                    facts,
                )
                if ref_match
                else None
            )

            repository = next(
                (
                    item
                    for item in facts.repositories
                    if item.url == url
                ),
                None,
            )

            return (
                url,
                ref or (
                    repository.ref
                    if repository
                    else None
                ),
                repository.commit
                if repository
                else None,
            )

        if re.search(
            r"\bgit\s+(?:checkout|switch|fetch)\b",
            command,
        ):
            clone_matches = list(_GIT_CLONE.finditer(script, 0, position))
            item = None
            if clone_matches:
                url = clone_matches[-1].group("url").rstrip(".,);]")
                item = next(
                    (reference for reference in facts.repositories if reference.url == url),
                    None,
                )
            elif len(facts.repositories) == 1:
                item = facts.repositories[0]

            if item is not None:

                checkout = _GIT_CHECKOUT.search(
                    command
                )

                value = (
                    BashSemanticParser._resolve_git_value(
                        checkout.group("ref"),
                        facts,
                    )
                    if checkout
                    else None
                )

                if value and re.fullmatch(
                    r"[0-9a-fA-F]{7,40}",
                    value,
                ):
                    return (
                        item.url,
                        item.ref,
                        value,
                    )

                return (
                    item.url,
                    value or item.ref,
                    item.commit,
                )

        return None, None, None

    @staticmethod
    def _resolve_git_value(value: str, facts) -> str:
        """Remove shell quotes and resolve a simple source-defined variable."""
        value = value.strip("'\"")
        variable = re.fullmatch(
            r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
            r"\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)",
            value,
        )
        if variable:
            name = variable.group("braced") or variable.group("plain")
            return facts.variables.get(name, value)
        return value

    @staticmethod
    def _patches(
        command: str,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                match.group(1)
                for match in _PATCH_FILE.finditer(
                    command
                )
            )
        )

    @staticmethod
    def _working_directory(
        words: list[str],
    ) -> str | None:
        if (
            words
            and words[0] == "cd"
            and len(words) > 1
        ):
            return words[1]

        return None

    @staticmethod
    def _user(
        command: str,
    ) -> str | None:
        match = re.search(
            r"\bsudo\s+-u\s+([^\s]+)",
            command,
        )

        return (
            match.group(1)
            if match
            else None
        )

    @staticmethod
    def _component(
        command: str,
        docker_image: str | None,
        repository: str | None,
    ) -> str | None:
        if docker_image:
            name = docker_image.split(
                "/"
            )[-1]

            return name.split(":")[0]

        if repository:
            return (
                repository
                .rstrip("/")
                .split("/")[-1]
                .removesuffix(".git")
            )

        if command.startswith("cd "):
            value = command.split(
                maxsplit=1
            )[1]

            return value.rstrip(
                "/"
            ).split("/")[-1]

        return None

    @staticmethod
    def _version(
        component: str | None,
        docker_image: str | None,
        repository: str | None,
        facts,
    ) -> str | None:
        if (
            docker_image
            and ":" in docker_image.rsplit(
                "/",
                1,
            )[-1]
        ):
            return docker_image.rsplit(
                ":",
                1,
            )[1]

        if (
            component
            and component.upper()
            in facts.variables
        ):
            return facts.variables[
                component.upper()
            ]

        return None

    @staticmethod
    def _build_tool(
        words: list[str],
    ) -> str | None:
        if not words:
            return None

        if words[0] in {
            "mvn",
            "mvnw",
            "./mvnw",
        }:
            return "maven"

        if words[0] in {
            "gradle",
            "gradlew",
            "./gradlew",
        }:
            return "gradle"

        if words[0] == "make":
            return "make"

        return None

    @staticmethod
    def _description(
        action: ActionType,
        command: str,
        packages: list[str],
        docker_image: str | None,
        repository: str | None,
        component: str | None,
    ) -> str:
        if action == ActionType.INSTALL_PACKAGE:
            return (
                "Install packages: "
                f"{', '.join(packages)}."
            )

        if action == ActionType.CLONE_REPOSITORY:
            return (
                f"Clone repository {repository}."
            )

        if (
            action == ActionType.BUILD_IMAGE
            and docker_image
        ):
            return (
                f"Build Docker image "
                f"{docker_image}."
            )

        if action == ActionType.CREATE_DIRECTORY:
            return "Create directory."

        if action == ActionType.DOWNLOAD:
            return "Download a file or resource."

        if action == ActionType.APPLY_PATCH:
            return "Apply a patch."

        if action == ActionType.RUN_TEST:
            return "Run build tests."

        return command


def parse_bash_steps(
    script: str,
    name: str = "script",
) -> list[BuildStep]:
    return BashSemanticParser().parse(
        script,
        name,
    )
