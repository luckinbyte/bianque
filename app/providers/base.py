"""Provider abstraction.

The agent loop speaks a normalized event stream regardless of the underlying
LLM. Adapters (:mod:`app.providers.openai_compat`, :mod:`app.providers.anthropic`)
turn each provider's streaming response into these events.

Messages are exchanged in **OpenAI chat format** (the lingua franca):

    {"role": "system"|"user"|"assistant"|"tool", "content": str | None,
     "tool_calls": [{"id","name","arguments":<json-str>}, ...] | None,   # assistant
     "tool_call_id": str | None, "name": str | None}                     # tool

Adapters convert as needed for their provider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, TypedDict, Any


@dataclass
class ContentDelta:
    """A chunk of assistant text (streamed)."""
    text: str


@dataclass
class ToolCall:
    """A completed tool call with fully-assembled, parsed JSON arguments."""
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Finish:
    """End of one model turn. reason: 'stop' | 'tool_use' | 'length' | ..."""
    reason: str


LLMEvent = ContentDelta | ToolCall | Finish


class LLMProvider(Protocol):
    name: str

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> AsyncIterator[LLMEvent]:
        """Yield normalized events for one model turn."""
        ...


# Canonical OpenAI-format message shape (for documentation / type checkers).
class Message(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    name: str
