from app.package_mapping import (
    canonical_wiki_package,
    script_target_for_package,
    wiki_package_for_script_path,
)


def test_antlr_alias_resolves_in_both_directions() -> None:
    target = script_target_for_package("ANTLR-4", "4.13.2")

    assert target.directory == "AntLR"
    assert target.filename is None
    assert canonical_wiki_package("AntLR") == "ANTLR-4"
    assert wiki_package_for_script_path(
        "AntLR/4.13.2/build_antlr4.sh"
    ) == "ANTLR-4"


def test_version_family_aliases_resolve_webhook_paths() -> None:
    assert script_target_for_package("Apache-Cassandra-4.x", "4.1.10").directory == "ApacheCassandra"
    assert script_target_for_package("Zabbix-agent", "7.0.30").filename == "build_zabbixagent.sh"
    assert wiki_package_for_script_path(
        "ApacheCassandra/4.1.10/build_cassandra.sh"
    ) == "Apache-Cassandra-4.x"
    assert wiki_package_for_script_path(
        "Zabbix/7.0.30/build_zabbixserver.sh"
    ) == "Zabbix-server"


def test_custom_alias_can_be_added_without_code_change(monkeypatch) -> None:
    monkeypatch.setenv(
        "BUILDGUARD_PACKAGE_ALIASES",
        '{"Example-Wiki": {"directory": "example_scripts", "filename": "run.sh"}}',
    )

    target = script_target_for_package("Example-Wiki", "1.0.0")

    assert target.directory == "example_scripts"
    assert target.filename == "run.sh"
    assert wiki_package_for_script_path("example_scripts/1.0.0/run.sh") == "Example-Wiki"
