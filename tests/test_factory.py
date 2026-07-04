import pytest

from app.providers import get_provider
from app.providers.anthropic import AnthropicProvider
from app.providers.openai_compat import OpenAICompatProvider


def test_openai_compat():
    p = get_provider("openai_compat", base_url="http://x/v1", apikey="k")
    assert isinstance(p, OpenAICompatProvider)
    assert p.base_url == "http://x/v1"


def test_anthropic():
    p = get_provider("anthropic", base_url="https://api.anthropic.com", apikey="k")
    assert isinstance(p, AnthropicProvider)


def test_local_defaults_to_ollama_base_url():
    p = get_provider("local", base_url=None, apikey="ignored")
    assert isinstance(p, OpenAICompatProvider)
    assert "11434" in p.base_url


def test_unknown_raises():
    with pytest.raises(ValueError):
        get_provider("nope", base_url="http://x", apikey="k")
