from app.config.settings import Settings

from .interface import LLMProvider
from .openrouter import OpenRouterProvider


def create_provider(settings: Settings | None = None) -> LLMProvider:
    """Create the configured OpenRouter-backed LLM provider."""
    configured = settings or Settings()
    if not configured.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required when AI extraction is enabled")
    return OpenRouterProvider(
        api_key=configured.openrouter_api_key,
        model=configured.openrouter_model,
    )
