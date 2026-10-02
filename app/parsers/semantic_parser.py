import re
import shlex

from app.models.semantic import DockerImage, RepositoryReference, SemanticFacts

_VARIABLE_ASSIGNMENT = re.compile(
    r"""
    (?<![\w-])
    (?:export\s+)?
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*=\s*
    (?P<value>
        "(?:[^"\\]|\\.)*"
        |
        '(?:[^'\\]|\\.)*'
        |
        [^\s;&|]+
    )
    """,
    re.VERBOSE,
)

_GIT_CLONE = re.compile(
    r"""
    git\s+clone
    (?P<options>.*?)
    \s+
    (?P<url>
        https?://[^\s'"]+
        |
        git@[^\s:]+:[^\s'"]+
    )
    """,
    re.VERBOSE,
)

_GIT_CLONE_REF = re.compile(
    r"""git\s+clone(?P<options>.*?)\s+(?:-b|--branch)\s+(?P<ref>[^\s;&|]+)""",
    re.VERBOSE,
)

_GIT_CHECKOUT = re.compile(
    r"""git\s+(?:checkout|switch)\s+(?P<ref>[^\s;&|]+)""",
    re.VERBOSE,
)

_GIT_COMMIT = re.compile(
    r"""git\s+(?:checkout|reset|show)\s+(?P<commit>[0-9a-fA-F]{7,40})""",
    re.VERBOSE,
)

_DOCKER_BUILD = re.compile(
    r"""
    docker\s+build
    (?P<options>.*?)
    \s+(?:-t|--tag)\s+(?P<tagged_image>[^\s;&|]+)
    \s+(?P<context>\.|[^\s;&|]+)
    """,
    re.VERBOSE,
)

_DOCKER_TAG = re.compile(
    r"""
    docker\s+(?:build|pull|push|run)
    .*?
    (?:-t|--tag)
    \s+(?P<image>[^\s;&|]+)
    """,
    re.VERBOSE,
)

_DOCKER_BUILD_ARG = re.compile(
    r"""(?:--build-arg|-build-arg)\s+
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s;&|]+)""",
    re.VERBOSE,
)

_PACKAGE_MANAGER = re.compile(
    r"""
    (?:
        (?:sudo\s+)?
        (?P<manager>apt-get|apt|dnf|yum|zypper)
        \s+
        (?P<operation>install|remove|erase)
        \s+
        (?P<arguments>[^\n]+)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PATH = re.compile(
    r"""
    (?<![A-Za-z0-9_:])
    /
    (?!/)
    (?:[A-Za-z0-9._~${}:+,@%-]+/)*
    [A-Za-z0-9._~${}:+,@%-]+
    """,
    re.VERBOSE,
)

_MAVEN = re.compile(r"(?<![\w-])(?:mvn|mvnw)(?=\s|$)", re.IGNORECASE)
_GRADLE = re.compile(r"(?<![\w-])(?:gradle|gradlew)(?=\s|$)", re.IGNORECASE)
_MAKE = re.compile(r"(?<![\w-])make(?=\s|$)", re.IGNORECASE)

_PATCH = re.compile(
    r"""
    (?:
        \bpatch\b.*?(?:<|--input=|\s)(?P<file>[^\s;&|]+\.patch)
        |
        \b(?:apply|git\s+apply)\b.*?(?P<apply>[^\s;&|]+\.patch)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_URL = re.compile("https?://[^\\s<>'\"`]+")
_CONDITION = re.compile(
    r"^\s*#\s*((?:only\s+if|if|when|unless|for)\b.+?)\s*$",
    re.IGNORECASE,
)


class SemanticParser:
    """Extract explicit, deterministic facts from build source text."""

    def parse(self, source: str) -> SemanticFacts:
        facts = SemanticFacts()
        logical_source = self._join_continuations(source)

        self._extract_variables(source, facts)
        self._extract_repositories(source, facts)
        self._extract_git_refs(source, facts)
        self._extract_docker_images(logical_source, facts)
        self._extract_docker_build_args(logical_source, facts)
        self._extract_build_tools(source, facts)
        self._extract_patches(source, facts)
        self._extract_urls(source, facts)
        self._extract_packages(source, facts)
        self._extract_paths(source, facts)
        self._extract_environment(source, facts)
        self._extract_conditions(source, facts)
        self._extract_commands(logical_source, facts)

        return facts

    @staticmethod
    def _join_continuations(source: str) -> str:
        return re.sub(r"\\\s*\n\s*", " ", source)

    @staticmethod
    def _clean_value(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

    def _extract_variables(self, source: str, facts: SemanticFacts) -> None:
        for match in _VARIABLE_ASSIGNMENT.finditer(source):
            facts.variables[match.group("name")] = self._clean_value(match.group("value"))

    def _extract_repositories(self, source: str, facts: SemanticFacts) -> None:
        seen: set[str] = set()
        for match in _GIT_CLONE.finditer(source):
            url = match.group("url").rstrip(".,);]")
            if url not in seen:
                seen.add(url)
                facts.repositories.append(RepositoryReference(url=url))

    def _extract_git_refs(self, source: str, facts: SemanticFacts) -> None:
        clone_matches = list(_GIT_CLONE.finditer(source))

        for clone_match in clone_matches:
            ref_match = _GIT_CLONE_REF.search(clone_match.group(0))
            if ref_match:
                url = clone_match.group("url").rstrip(".,);]")
                repository = next(
                    (repo for repo in facts.repositories if repo.url == url),
                    None,
                )
                if repository is not None:
                    repository.ref = SemanticParser._clean_value(ref_match.group("ref"))

        for match in _GIT_CHECKOUT.finditer(source):
            value = match.group("ref")
            repository = self._nearest_repository(
                clone_matches, facts.repositories, match.start()
            )
            if repository is not None:
                if re.fullmatch(r"[0-9a-fA-F]{7,40}", value):
                    repository.commit = value
                else:
                    repository.ref = value

        for match in _GIT_COMMIT.finditer(source):
            repository = self._nearest_repository(
                clone_matches, facts.repositories, match.start()
            )
            if repository is not None:
                repository.commit = match.group("commit")
            elif not any(item.commit == match.group("commit") for item in facts.repositories):
                facts.repositories.append(
                    RepositoryReference(url="", commit=match.group("commit"))
                )

    @staticmethod
    def _nearest_repository(
        clone_matches: list[re.Match[str]],
        repositories: list[RepositoryReference],
        position: int,
    ) -> RepositoryReference | None:
        preceding = [match for match in clone_matches if match.start() <= position]
        if not preceding:
            return None

        url = preceding[-1].group("url").rstrip(".,);]")
        return next((repo for repo in repositories if repo.url == url), None)

    def _extract_docker_images(self, source: str, facts: SemanticFacts) -> None:
        seen: set[tuple[str, str | None]] = set()
        for pattern in (_DOCKER_BUILD, _DOCKER_TAG):
            for match in pattern.finditer(source):
                image = match.groupdict().get("tagged_image") or match.groupdict().get("image")
                if not image:
                    continue

                name, tag = self._split_image(image)
                key = (name, tag)
                if key not in seen:
                    seen.add(key)
                    facts.docker_images.append(DockerImage(name=name, tag=tag))

    def _extract_docker_build_args(self, source: str, facts: SemanticFacts) -> None:
        for match in _DOCKER_BUILD_ARG.finditer(source):
            facts.build_args[match.group("name")] = self._clean_value(match.group("value"))

    @staticmethod
    def _split_image(image: str) -> tuple[str, str | None]:
        image = image.strip().rstrip(".,);]")
        last_slash = image.rfind("/")
        last_colon = image.rfind(":")
        if last_colon > last_slash:
            return image[:last_colon], image[last_colon + 1:]
        return image, None

    def _extract_build_tools(self, source: str, facts: SemanticFacts) -> None:
        if _MAVEN.search(source):
            facts.build_tools.append("maven")
        if _GRADLE.search(source):
            facts.build_tools.append("gradle")
        if _MAKE.search(source):
            facts.build_tools.append("make")

    def _extract_patches(self, source: str, facts: SemanticFacts) -> None:
        seen: set[str] = set()
        for match in _PATCH.finditer(source):
            patch = match.group("file") or match.group("apply")
            if patch and patch not in seen:
                seen.add(patch)
                facts.patches.append(patch)

    def _extract_urls(self, source: str, facts: SemanticFacts) -> None:
        seen: set[str] = set()
        for url in _URL.findall(source):
            url = url.rstrip(".,);]")
            if url not in seen:
                seen.add(url)
                facts.urls.append(url)

    def _extract_packages(self, source: str, facts: SemanticFacts) -> None:
        seen: set[str] = set()
        for match in _PACKAGE_MANAGER.finditer(source):
            try:
                tokens = shlex.split(match.group("arguments"), comments=False)
            except ValueError:
                tokens = match.group("arguments").split()

            for token in tokens:
                token = token.rstrip("\\")
                if not token or token.startswith("-") or token in {"&&", "||", ";"}:
                    continue
                if token not in seen:
                    seen.add(token)
                    facts.packages.append(token)

    def _extract_paths(self, source: str, facts: SemanticFacts) -> None:
        seen: set[str] = set()
        url_spans = [(match.start(), match.end()) for match in _URL.finditer(source)]

        for match in _PATH.finditer(source):
            if any(start <= match.start() < end for start, end in url_spans):
                continue

            path = match.group(0).rstrip(".,);]")
            if path not in seen:
                seen.add(path)
                facts.paths.append(path)

    def _extract_environment(self, source: str, facts: SemanticFacts) -> None:
        for match in _VARIABLE_ASSIGNMENT.finditer(source):
            facts.environment[match.group("name")] = self._clean_value(match.group("value"))

    def _extract_conditions(self, source: str, facts: SemanticFacts) -> None:
        seen: set[str] = set()
        for line in source.splitlines():
            match = _CONDITION.match(line)
            if match:
                condition = match.group(1).strip()
                if condition not in seen:
                    seen.add(condition)
                    facts.conditions.append(condition)

    def _extract_commands(self, source: str, facts: SemanticFacts) -> None:
        prefixes = (
            "git ", "docker ", "mvn ", "./mvnw ", "mvnw ", "gradle ",
            "./gradlew ", "make ", "patch ", "sed ", "curl ", "wget ",
            "cd ", "mkdir ", "cp ", "mv ", "rm ", "tar ", "unzip ",
        )

        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("```") or stripped.startswith("#"):
                continue

            if stripped.startswith("sudo "):
                command = stripped[5:].lstrip()
                if command.startswith(prefixes):
                    facts.commands.append(stripped)
            elif stripped.startswith(prefixes):
                facts.commands.append(stripped)


def parse_semantic_facts(source: str) -> SemanticFacts:
    return SemanticParser().parse(source)
