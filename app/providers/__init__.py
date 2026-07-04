"""Provider registry / factory."""
from __future__ import annotations

from app.providers.base import LLMProvider
from app.providers.anthropic import AnthropicProvider
from app.providers.openai_compat import OpenAICompatProvider

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def get_provider(name: str, *, base_url: str | None, apikey: str) -> LLMProvider:
    """Build a provider by name.

    * ``openai_compat`` — any OpenAI-compatible endpoint (requires base_url).
    * ``anthropic`` — Claude (defaults to the public API if base_url omitted).
    * ``local`` — a local OpenAI-compatible server (defaults to Ollama).
    """
    if name == "openai_compat":
        if not base_url:
            raise ValueError("openai_compat provider requires a base_url")
        return OpenAICompatProvider(base_url=base_url, apikey=apikey)
    if name == "anthropic":
        return AnthropicProvider(base_url=base_url or "https://api.anthropic.com", apikey=apikey)
    if name == "local":
        return OpenAICompatProvider(base_url=base_url or OLLAMA_DEFAULT_BASE_URL, apikey=apikey or "local")
    raise ValueError(f"unknown provider: {name}")
