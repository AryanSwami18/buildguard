from typing import Protocol

from .request import LLMRequest
from .response import LLMResponse


class LLMProvider(Protocol):
    """Provider-independent interface used by the BuildGuard pipeline."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        ...
