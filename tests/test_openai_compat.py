"""Tests for the OpenAI-compatible provider adapter.

Focus: normalizing a streamed /chat/completions response into LLMEvents,
including the tricky part — reassembling fragmented tool-call arguments.
Uses httpx.MockTransport so no network is involved.
"""
import json

import httpx
import pytest

from app.providers.base import ContentDelta, Finish, ToolCall
from app.providers.openai_compat import OpenAICompatProvider, build_payload


def _sse(chunks: list[dict]) -> bytes:
    body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _provider(chunks: list[dict], *, base_url: str = "http://x/v1") -> OpenAICompatProvider:
    payload = _sse(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider(base_url=base_url, apikey="k", client=client)


async def _collect(p: OpenAICompatProvider) -> list:
    out: list = []
    async for ev in p.stream(messages=[], tools=[], model="m"):
        out.append(ev)
    return out


# ---------- payload ----------

def test_build_payload_sets_stream_and_forwards_inputs():
    msgs = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "grep"}}]
    payload = build_payload(msgs, tools, "model-x")
    assert payload["model"] == "model-x"
    assert payload["stream"] is True
    assert payload["messages"] == msgs
    assert payload["tools"] == tools


# ---------- stream parsing ----------

async def test_text_only_turn():
    events = await _collect(_provider([
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]))
    assert events == [ContentDelta("Hello"), ContentDelta(" world"), Finish(reason="stop")]


async def test_reassembles_fragmented_tool_call_args():
    events = await _collect(_provider([
        {"choices": [{"delta": {"content": "Let me search."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                       "function": {"name": "grep", "arguments": "{\"pa"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0,
                       "function": {"arguments": "ttern\":\"foo\"}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]))
    assert events == [
        ContentDelta("Let me search."),
        ToolCall(id="call_1", name="grep", args={"pattern": "foo"}),
        Finish(reason="tool_calls"),
    ]


async def test_multiple_tool_calls_by_index():
    events = await _collect(_provider([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c0", "function": {"name": "grep", "arguments": "{\"pattern\":\"a\"}"}},
            {"index": 1, "id": "c1", "function": {"name": "list_dir", "arguments": "{\"path\":\".\"}"}},
        ]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]))
    assert events == [
        ToolCall(id="c0", name="grep", args={"pattern": "a"}),
        ToolCall(id="c1", name="list_dir", args={"path": "."}),
        Finish(reason="tool_calls"),
    ]


async def test_authorization_header_sent():
    """The provider must send the user's apikey as a Bearer token."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, content=_sse([{"choices": [{"delta": {}, "finish_reason": "stop"}]}]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    p = OpenAICompatProvider(base_url="http://x/v1", apikey="secret-key", client=client)
    await _collect(p)
    assert seen["auth"] == "Bearer secret-key"
    assert seen["url"] == "http://x/v1/chat/completions"
