from pydantic import BaseModel, Field


class RepositoryReference(BaseModel):
    url: str
    ref: str | None = None
    commit: str | None = None


class DockerImage(BaseModel):
    name: str
    tag: str | None = None


class SemanticFacts(BaseModel):
    """
    Deterministic facts extracted directly from source text.

    These facts are intentionally source-oriented. They describe what is
    explicitly present in the source; semantic interpretation happens later.
    """

    variables: dict[str, str] = Field(default_factory=dict)

    repositories: list[RepositoryReference] = Field(default_factory=list)

    docker_images: list[DockerImage] = Field(default_factory=list)

    build_tools: list[str] = Field(default_factory=list)

    patches: list[str] = Field(default_factory=list)

    urls: list[str] = Field(default_factory=list)

    commands: list[str] = Field(default_factory=list)

    packages: list[str] = Field(default_factory=list)

    paths: list[str] = Field(default_factory=list)

    build_args: dict[str, str] = Field(default_factory=dict)

    environment: dict[str, str] = Field(default_factory=dict)

    conditions: list[str] = Field(default_factory=list)
