from enum import Enum

from pydantic import BaseModel, Field

from .context import SourceContext


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
    """A provider-validated meaning shared by wiki and Bash sources."""

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

    context: SourceContext
