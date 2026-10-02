from app.parsers.semantic_parser import parse_semantic_facts


def test_semantic_parser_extracts_build_facts_without_interpreting_them() -> None:
    source = r'''export JAVA_MAJOR="17"
DISTRO="rhel"

git clone -b "${VERSION}" https://github.com/alfresco/repo.git
git checkout 1234567890abcdef1234567890abcdef12345678

sudo yum install -y git gcc gcc-c++

docker build \
  --build-arg JAVA_MAJOR=17 \
  --build-arg DISTRO=rhel \
  -t alfresco/base-java:17 .

# For RHEL only
patch < ${PATCH_URL}/base-java.patch
wget https://example.com/file.tar.gz
mkdir -p /opt/alfresco
'''

    facts = parse_semantic_facts(source)

    assert facts.variables == {"JAVA_MAJOR": "17", "DISTRO": "rhel"}
    assert facts.repositories[0].url == "https://github.com/alfresco/repo.git"
    assert facts.repositories[0].ref == "${VERSION}"
    assert facts.repositories[0].commit == "1234567890abcdef1234567890abcdef12345678"
    assert [image.model_dump(mode="json") for image in facts.docker_images] == [
        {"name": "alfresco/base-java", "tag": "17"}
    ]
    assert facts.build_args == {"JAVA_MAJOR": "17", "DISTRO": "rhel"}
    assert facts.packages == ["git", "gcc", "gcc-c++"]
    assert "/opt/alfresco" in facts.paths
    assert facts.patches == ["${PATCH_URL}/base-java.patch"]
    assert facts.conditions == ["For RHEL only"]


def test_semantic_parser_does_not_turn_urls_into_filesystem_paths() -> None:
    facts = parse_semantic_facts(
        "wget https://example.com/file.tar.gz\n"
        "git clone https://github.com/example/repo.git\n"
    )

    assert "/example.com/file.tar.gz" not in facts.paths
    assert "/github.com/example/repo.git" not in facts.paths

