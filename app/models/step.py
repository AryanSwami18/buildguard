from enum import Enum

from pydantic import BaseModel, Field

from app.models.context import SourceContext


class ActionType(str, Enum):
    INSTALL_PACKAGE = "install_package"
    REMOVE_PACKAGE = "remove_package"

    CREATE_DIRECTORY = "create_directory"
    CREATE_FILE = "create_file"

    DOWNLOAD = "download"
    EXTRACT = "extract"

    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"

    RUN_COMMAND = "run_command"

    CREATE_USER = "create_user"
    MODIFY_USER = "modify_user"

    SET_VARIABLE = "set_variable"

    START_SERVICE = "start_service"
    STOP_SERVICE = "stop_service"
    ENABLE_SERVICE = "enable_service"
    DISABLE_SERVICE = "disable_service"

    CONFIGURE = "configure"

    UNKNOWN = "unknown"


class BuildStep(BaseModel):
    id: str
    order: int

    action: ActionType
    description: str

    packages: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)

    user: str | None = None
    working_directory: str | None = None

    depends_on: list[str] = Field(default_factory=list)

    # ---------------------------------------------------------
    # Build-specific semantic information
    # ---------------------------------------------------------

    # Component being built.
    # Example: "search-services", "base-java"
    component: str | None = None

    # Build/package version.
    # Example: "25.3.0"
    version: str | None = None

    # Git repository URL.
    repository: str | None = None

    # Git branch/tag/ref.
    ref: str | None = None

    # Exact Git commit.
    commit: str | None = None

    # Docker image being produced/used.
    # Example: "alfresco/search-services:25.3.0"
    docker_image: str | None = None

    # Dockerfile path.
    dockerfile: str | None = None

    # Docker build arguments.
    # Example:
    # {
    #     "JAVA_MAJOR": "17",
    #     "DISTRO": "rhel"
    # }
    build_args: dict[str, str] = Field(default_factory=dict)

    # Environment variables relevant to the step.
    environment: dict[str, str] = Field(default_factory=dict)

    # Patches applied by the build instruction.
    patches: list[str] = Field(default_factory=list)

    # Build tool used.
    # Examples: "maven", "gradle", "make"
    build_tool: str | None = None

    context: SourceContext
