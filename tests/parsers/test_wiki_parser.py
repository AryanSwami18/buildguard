from app.parsers.wiki_parser import parse_wiki


def test_wiki_parser_keeps_headings_order_and_code_block_locations() -> None:
    markdown = """# Install

Install Java.

## Application

Create the Alfresco user.

```bash
mkdir -p /opt/alfresco
```
"""

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


def test_wiki_parser_preserves_logical_build_context_across_context_h1_headings() -> None:
    markdown = """# Building Alfresco

### 3. Build component docker images

#### 3.1 base-java

# Execute the Build Command

```bash
docker build -t alfresco/base-java:17 .
```

#### 3.2 base-tomcat

# Only if Building for Rhel Distributions

```bash
# For RHEL only
yum install -y gcc
```
"""

    document = parse_wiki(markdown, "Building-Alfresco.md")

    first = document.blocks[0]
    second = document.blocks[1]

    assert first.heading_path == ["Execute the Build Command"]
    assert first.logical_section == "3. Build component docker images"
    assert first.logical_subsection == "3.1 base-java"
    assert first.logical_heading_path == [
        "3. Build component docker images",
        "3.1 base-java",
    ]
    assert first.context_labels == ["Execute the Build Command"]
    assert first.conditions == []

    assert second.logical_section == "3. Build component docker images"
    assert second.logical_subsection == "3.2 base-tomcat"
    assert second.context_labels == ["Only if Building for Rhel Distributions"]
    assert second.conditions == [
        "Only if Building for Rhel Distributions",
        "For RHEL only",
    ]

