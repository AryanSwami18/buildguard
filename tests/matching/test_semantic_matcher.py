from app.matching.semantic_matcher import normalize_command, normalize_validation_result
from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep
from app.models.validation import MatchStatus, StepMatch, ValidationResult
from app.parsers.bash_semantic_parser import parse_bash_steps


def _context(source: str = "test") -> SourceContext:
    return SourceContext(source_type=source, source_name=source)


def _step(
    step_id: str,
    action: ActionType,
    description: str,
    *,
    commands: list[str] | None = None,
    packages: list[str] | None = None,
    version: str | None = None,
    component: str | None = None,
) -> BuildStep:
    return BuildStep(
        id=step_id,
        order=int(step_id.split("-")[-1]),
        action=action,
        description=description,
        commands=commands or [],
        packages=packages or [],
        version=version,
        component=component,
        context=_context(),
    )


def test_command_normalization_ignores_sudo_for_package_index_refresh() -> None:
    assert normalize_command("sudo apt-get update") == "package_index_update"
    assert normalize_command("apt-get update") == "package_index_update"


def test_bash_parser_marks_only_unreferenced_version_configuration_as_dead() -> None:
    steps = parse_bash_steps(
        'MAVEN_VERSION="3.9.11"\n'
        'ACS_NGINX_VERSION="3.4.2"\n'
        'wget "https://example.test/apache-maven-${MAVEN_VERSION}.tar.gz"\n',
        "build.sh",
    )

    assignments = {
        next(iter(step.environment)): step
        for step in steps
        if step.action == ActionType.SET_VARIABLE
    }
    assert "unused_configuration" not in assignments["MAVEN_VERSION"].context.context_labels
    assert "unused_configuration" in assignments["ACS_NGINX_VERSION"].context.context_labels
    download = next(step for step in steps if step.action == ActionType.DOWNLOAD)
    assert download.environment["MAVEN_VERSION"] == "3.9.11"


def test_grouped_java_workflow_is_not_reported_as_many_script_extras() -> None:
    wiki_step = _step(
        "wiki-1",
        ActionType.INSTALL_PACKAGE,
        "Install and configure Java",
        packages=["openjdk-${jver}-jdk"],
        version="17",
    )
    script_steps = parse_bash_steps(
        'JAVA_PROVIDED="Temurin17"\n'
        "installJava() {\n"
        "  wget https://example.test/jdk-${JAVA_PROVIDED}.tar.gz\n"
        "  tar -xzf jdk-${JAVA_PROVIDED}.tar.gz -C /opt/java\n"
        "  update-alternatives --install /usr/bin/java java /opt/java/bin/java 1\n"
        "  export JAVA_HOME=/opt/java\n"
        "}\n",
        "build.sh",
    )
    wiki = BuildInstruction(name="wiki", source="wiki", steps=[wiki_step])
    bash = BuildInstruction(name="build.sh", source="bash", steps=script_steps)
    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(wiki_step=wiki_step, status=MatchStatus.MISSING, explanation="missing"),
                *[
                    StepMatch(automation_step=step, status=MatchStatus.EXTRA, explanation="extra")
                    for step in script_steps
                ],
            ],
            valid=False,
            summary="",
        ),
        wiki,
        bash,
    )

    assert [match.status for match in result.matches] == [MatchStatus.MATCHED]
    assert len(result.matches[0].automation_steps) == len(script_steps)


def test_unrelated_meaningful_script_action_remains_a_documentation_gap() -> None:
    wiki_step = _step("wiki-1", ActionType.INSTALL_PACKAGE, "Install Java", packages=["java"])
    extra = _step(
        "bash-1",
        ActionType.RUN_TEST,
        "Run a security scan",
        commands=["security-scan --full"],
    )
    wiki = BuildInstruction(name="wiki", source="wiki", steps=[wiki_step])
    bash = BuildInstruction(name="build.sh", source="bash", steps=[extra])
    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(wiki_step=wiki_step, status=MatchStatus.MISSING, explanation="missing"),
                StepMatch(automation_step=extra, status=MatchStatus.EXTRA, explanation="extra"),
            ],
            valid=False,
            summary="",
        ),
        wiki,
        bash,
    )

    assert [match.status for match in result.matches] == [MatchStatus.MISSING, MatchStatus.EXTRA]


def test_concrete_version_disagreement_remains_a_conflict() -> None:
    wiki_step = _step(
        "wiki-1",
        ActionType.DOWNLOAD,
        "Install Maven 3.9.11",
        version="3.9.11",
    )
    script_steps = parse_bash_steps(
        'MAVEN_VERSION="3.8.8"\n'
        'wget "https://example.test/apache-maven-${MAVEN_VERSION}.tar.gz"\n',
        "build.sh",
    )
    wiki = BuildInstruction(name="wiki", source="wiki", steps=[wiki_step])
    bash = BuildInstruction(name="build.sh", source="bash", steps=script_steps)
    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(wiki_step=wiki_step, status=MatchStatus.MISSING, explanation="missing"),
                *[
                    StepMatch(automation_step=step, status=MatchStatus.EXTRA, explanation="extra")
                    for step in script_steps
                ],
            ],
            valid=False,
            summary="",
        ),
        wiki,
        bash,
    )

    assert result.matches[0].status == MatchStatus.CONFLICT
