from app.config.settings import Settings

from .interface import LLMProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider


def create_provider(settings: Settings | None = None) -> LLMProvider:
    """Create the configured local or hosted LLM provider."""
    configured = settings or Settings()
    if configured.ai_provider == "ollama":
        return OllamaProvider(
            model=configured.ollama_model,
            base_url=configured.ollama_base_url,
            timeout_seconds=configured.ollama_timeout_seconds,
            max_output_tokens=configured.ollama_max_output_tokens,
            context_tokens=configured.ollama_context_tokens,
        )
    if configured.ai_provider == "gemini":
        if not configured.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when AI_PROVIDER=gemini"
            )
        return GeminiProvider(
            api_key=configured.gemini_api_key,
            model=configured.gemini_model,
            timeout_seconds=configured.gemini_timeout_seconds,
            max_output_tokens=configured.gemini_max_output_tokens,
        )
    raise ValueError(
        f"Unsupported AI_PROVIDER '{configured.ai_provider}'. Use 'ollama' or 'gemini'."
    )
