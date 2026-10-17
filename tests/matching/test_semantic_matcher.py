from app.matching.semantic_matcher import normalize_command, normalize_validation_result
from app.models.build import BuildInstruction
from app.models.context import SourceContext
from app.models.step import ActionType, BuildStep
from app.models.validation import (
    DiscrepancyCategory,
    MatchStatus,
    StepMatch,
    ValidationResult,
)
from app.parsers.bash_semantic_parser import parse_bash_steps


def _context(
    source: str = "test",
    *,
    context_labels: list[str] | None = None,
) -> SourceContext:
    return SourceContext(
        source_type=source,
        source_name=source,
        context_labels=context_labels or [],
    )


def _step(
    step_id: str,
    action: ActionType,
    description: str,
    *,
    commands: list[str] | None = None,
    packages: list[str] | None = None,
    version: str | None = None,
    component: str | None = None,
    repository: str | None = None,
    urls: list[str] | None = None,
    patches: list[str] | None = None,
    context_labels: list[str] | None = None,
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
        repository=repository,
        urls=urls or [],
        patches=patches or [],
        context=_context(context_labels=context_labels),
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


def test_bash_parser_attaches_global_failure_options_to_operations() -> None:
    steps = parse_bash_steps(
        "set -e -o pipefail\n"
        "curl -sSL https://example.test/influxdb.patch | git apply\n",
        "build.sh",
    )

    patch_step = next(step for step in steps if step.action == ActionType.APPLY_PATCH)
    assert patch_step.context.context_labels == [
        "shell_errexit",
        "shell_pipefail",
    ]


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


def test_wiki_github_clone_url_groups_with_script_repository_workflow() -> None:
    wiki_step = _step(
        "wiki-1",
        ActionType.RUN_COMMAND,
        "Clone Bazel source code and apply patch",
        commands=[
            "git clone --depth 1 -b 9.2.0 https://github.com/bazelbuild/bazel.git",
            "curl -sSLO $PATCH_URL/bazel.patch",
            "patch -p1 < bazel.patch",
        ],
    )
    script_steps = parse_bash_steps(
        "configureAndInstall() {\n"
        "  git clone --depth 1 -b $PACKAGE_VERSION https://github.com/bazelbuild/bazel.git\n"
        "  curl -sSLO $PATCH_URL/bazel.patch\n"
        "  patch -p1 < bazel.patch\n"
        "  rm -f bazel.patch\n"
        "}\n",
        "build_bazel.sh",
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

    matched = next(match for match in result.matches if match.status == MatchStatus.MATCHED)
    assert {step.id for step in matched.automation_steps} == {
        step.id for step in script_steps
        if step.action != ActionType.DELETE
    }
    assert any(match.status == MatchStatus.IMPLEMENTATION_DETAIL for match in result.matches)


def test_exact_repository_identity_corrects_crossed_clone_matches() -> None:
    tombi_wiki = _step(
        "wiki-1",
        ActionType.CLONE_REPOSITORY,
        "Clone Tombi repository",
        commands=["git clone https://github.com/tombi-toml/tombi.git"],
        repository="https://github.com/tombi-toml/tombi.git",
        urls=["https://github.com/tombi-toml/tombi.git"],
    )
    influx_wiki = _step(
        "wiki-2",
        ActionType.CLONE_REPOSITORY,
        "Clone InfluxDB repository",
        commands=["git clone --depth 1 -b v3.10.0 https://github.com/influxdata/influxdb.git"],
        repository="https://github.com/influxdata/influxdb.git",
        urls=["https://github.com/influxdata/influxdb.git"],
    )
    tombi_bash = _step(
        "bash-1",
        ActionType.CLONE_REPOSITORY,
        "Clone repository https://github.com/tombi-toml/tombi.git",
        commands=["git clone https://github.com/tombi-toml/tombi.git"],
        repository="https://github.com/tombi-toml/tombi.git",
        urls=["https://github.com/tombi-toml/tombi.git"],
    )
    influx_bash = _step(
        "bash-2",
        ActionType.CLONE_REPOSITORY,
        "Clone repository https://github.com/influxdata/influxdb.git",
        commands=["git clone --depth 1 -b $PACKAGE_VERSION https://github.com/influxdata/influxdb.git"],
        repository="https://github.com/influxdata/influxdb.git",
        urls=["https://github.com/influxdata/influxdb.git"],
    )

    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(
                    wiki_step=tombi_wiki,
                    automation_step=influx_bash,
                    status=MatchStatus.CONFLICT,
                    explanation="provider crossed the repositories",
                ),
                StepMatch(
                    wiki_step=influx_wiki,
                    automation_step=tombi_bash,
                    status=MatchStatus.MATCHED,
                    explanation="provider crossed the repositories",
                ),
            ],
            valid=False,
            summary="",
        ),
        BuildInstruction(name="wiki", source="wiki", steps=[tombi_wiki, influx_wiki]),
        BuildInstruction(name="build.sh", source="bash", steps=[tombi_bash, influx_bash]),
    )

    pairs = {
        match.wiki_step.id: match.automation_step.id
        for match in result.matches
        if match.wiki_step is not None and match.automation_step is not None
    }
    assert pairs == {"wiki-1": "bash-1", "wiki-2": "bash-2"}
    assert all(match.status == MatchStatus.MATCHED for match in result.matches)


def test_grouped_wiki_artifact_merge_matches_mkdir_and_copy_steps() -> None:
    wiki_step = _step(
        "wiki-1",
        ActionType.RUN_COMMAND,
        "Merge build artifacts into dist-my-platform directory",
        commands=[
            "mkdir dist-my-platform",
            "cp build/toolchain/dist/* build/sysroot/dist/* dist-my-platform",
            "./ci/merge-artifacts.sh",
        ],
    )
    create_directory = _step(
        "bash-1",
        ActionType.CREATE_DIRECTORY,
        "Create directory.",
        commands=["mkdir dist-my-platform"],
    )
    copy_artifacts = _step(
        "bash-2",
        ActionType.COPY,
        "Copy build artifacts.",
        commands=["cp build/toolchain/dist/* build/sysroot/dist/* dist-my-platform"],
    )

    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(wiki_step=wiki_step, status=MatchStatus.MISSING, explanation="missing"),
                StepMatch(automation_step=create_directory, status=MatchStatus.EXTRA, explanation="extra"),
                StepMatch(automation_step=copy_artifacts, status=MatchStatus.EXTRA, explanation="extra"),
            ],
            valid=False,
            summary="",
        ),
        BuildInstruction(name="wiki", source="wiki", steps=[wiki_step]),
        BuildInstruction(name="build.sh", source="bash", steps=[create_directory, copy_artifacts]),
    )

    assert len(result.matches) == 1
    assert result.matches[0].status == MatchStatus.MATCHED
    assert {step.id for step in result.matches[0].automation_steps} == {"bash-1", "bash-2"}


def test_remove_package_is_classified_as_implementation_detail() -> None:
    wiki_step = _step(
        "wiki-1",
        ActionType.INSTALL_PACKAGE,
        "Install CMake",
        packages=["cmake"],
    )
    remove_cmake = _step(
        "bash-1",
        ActionType.REMOVE_PACKAGE,
        "Remove the system CMake package before installing the required version",
        commands=["sudo apt-get remove --purge -y cmake"],
        packages=["cmake"],
    )
    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(wiki_step=wiki_step, status=MatchStatus.MISSING, explanation="missing"),
                StepMatch(automation_step=remove_cmake, status=MatchStatus.EXTRA, explanation="extra"),
            ],
            valid=False,
            summary="",
        ),
        BuildInstruction(name="wiki", source="wiki", steps=[wiki_step]),
        BuildInstruction(name="build.sh", source="bash", steps=[remove_cmake]),
    )

    assert result.matches[-1].status == MatchStatus.IMPLEMENTATION_DETAIL


def test_global_bash_failure_options_match_explicit_wiki_error_handler() -> None:
    wiki_step = _step(
        "wiki-1",
        ActionType.APPLY_PATCH,
        "Apply influxdb.patch",
        commands=[
            'curl -sSL ${PATCH_URL}/influxdb.patch | git apply || error "influxdb.patch"',
        ],
        patches=["influxdb.patch"],
    )
    script_step = _step(
        "bash-1",
        ActionType.APPLY_PATCH,
        "Apply a patch.",
        commands=["curl -sSL ${PATCH_URL}/influxdb.patch | git apply"],
        patches=["influxdb.patch"],
        context_labels=["shell_errexit", "shell_pipefail"],
    )

    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(
                    wiki_step=wiki_step,
                    automation_step=script_step,
                    status=MatchStatus.CONFLICT,
                    category=DiscrepancyCategory.COMMAND_MISMATCH,
                    explanation="The Wiki includes an explicit error handler.",
                ),
            ],
            valid=False,
            summary="",
        ),
        BuildInstruction(name="wiki", source="wiki", steps=[wiki_step]),
        BuildInstruction(name="build.sh", source="bash", steps=[script_step]),
    )

    assert result.matches[0].status == MatchStatus.MATCHED
    assert result.matches[0].category is None
    assert "set -e -o pipefail" in result.matches[0].explanation


def test_explicit_wiki_error_handler_stays_actionable_without_pipefail() -> None:
    wiki_step = _step(
        "wiki-1",
        ActionType.APPLY_PATCH,
        "Apply influxdb.patch",
        commands=[
            'curl -sSL ${PATCH_URL}/influxdb.patch | git apply || error "influxdb.patch"',
        ],
        patches=["influxdb.patch"],
    )
    script_step = _step(
        "bash-1",
        ActionType.APPLY_PATCH,
        "Apply a patch.",
        commands=["curl -sSL ${PATCH_URL}/influxdb.patch | git apply"],
        patches=["influxdb.patch"],
        context_labels=["shell_errexit"],
    )

    result = normalize_validation_result(
        ValidationResult(
            matches=[
                StepMatch(
                    wiki_step=wiki_step,
                    automation_step=script_step,
                    status=MatchStatus.CONFLICT,
                    category=DiscrepancyCategory.COMMAND_MISMATCH,
                    explanation="The Wiki includes an explicit error handler.",
                ),
            ],
            valid=False,
            summary="",
        ),
        BuildInstruction(name="wiki", source="wiki", steps=[wiki_step]),
        BuildInstruction(name="build.sh", source="bash", steps=[script_step]),
    )

    assert result.matches[0].status == MatchStatus.CONFLICT


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
