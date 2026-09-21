from app.parsers.wiki_parser import parse_wiki


def test_wiki_parser_keeps_headings_order_and_code_block_locations() -> None:
    markdown = """# Install\n\nInstall Java.\n\n## Application\n\nCreate the Alfresco user.\n\n```bash\nmkdir -p /opt/alfresco\n```\n"""

    document = parse_wiki(markdown, "guide")

    assert [block.kind for block in document.blocks] == ["paragraph", "paragraph", "code_block"]
    assert document.blocks[1].section == "Install"
    assert document.blocks[1].subsection == "Application"
    assert document.blocks[2].text == "```bash\nmkdir -p /opt/alfresco\n```"
    assert document.blocks[2].line_start == 9


def test_wiki_parser_keeps_list_steps_and_heading_structure() -> None:
    markdown = """# Prepare

Introductory text.

- Create a directory.
- Add the configuration.

## Verify

1. Start the service.
2. Check its status.
"""

    document = parse_wiki(markdown, "guide")

    assert [(heading.level, heading.text, heading.line) for heading in document.headings] == [
        (1, "Prepare", 1),
        (2, "Verify", 8),
    ]
    assert [block.kind for block in document.blocks] == [
        "paragraph",
        "bulleted_step",
        "bulleted_step",
        "numbered_step",
        "numbered_step",
    ]
    assert document.blocks[1].line_start == document.blocks[1].line_end == 5
    assert document.blocks[3].section == "Prepare"
    assert document.blocks[3].subsection == "Verify"
