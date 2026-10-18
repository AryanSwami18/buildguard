from pydantic import BaseModel


class LLMRequest(BaseModel):
    """A provider-independent request for AI interpretation."""

    system_prompt: str
    user_prompt: str
    structured_output_instructions: str | None = None
    max_output_tokens: int | None = None
