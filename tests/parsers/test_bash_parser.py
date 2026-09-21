from app.parsers.bash_parser import parse_script


def test_bashlex_structure_keeps_operators_nested_commands_and_positions() -> None:
    script = "cd /opt/alfresco && wget https://example.com/app.tar.gz\n"

    parsed = parse_script(script)

    root = parsed.nodes[0]
    assert root.kind == "list"
    assert root.text == script.strip()
    assert [child.kind for child in root.children] == ["command", "operator", "command"]
    assert root.children[1].operator == "&&"
    assert root.children[2].words == ["wget", "https://example.com/app.tar.gz"]
    assert root.children[2].position_start > root.children[0].position_end


def test_bash_parser_does_not_assign_command_specific_semantics() -> None:
    command = parse_script("sudo dnf install -y git curl wget").nodes[0]

    assert command.kind == "command"
    assert command.words == ["sudo", "dnf", "install", "-y", "git", "curl", "wget"]
    assert not hasattr(command, "action")
