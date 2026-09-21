from pydantic import BaseModel, Field


class BashNode(BaseModel):
    """A serializable syntax node produced from bashlex.

    This deliberately describes syntax only. It does not decide what a
    command means; that decision belongs to the LLM interpretation stage.
    """

    kind: str
    position_start: int
    position_end: int
    text: str
    operator: str | None = None
    words: list[str] = Field(default_factory=list)
    children: list["BashNode"] = Field(default_factory=list)


class BashScript(BaseModel):
    source: str
    nodes: list[BashNode] = Field(default_factory=list)


BashNode.model_rebuild()
