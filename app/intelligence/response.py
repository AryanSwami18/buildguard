from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Text or structured JSON returned by an AI provider."""

    content: str
