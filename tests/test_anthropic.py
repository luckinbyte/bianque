"""Tests for the Anthropic provider adapter.

Focus: normalizing Anthropic's SSE event stream (text_delta / input_json_delta)
into LLMEvents, plus request conversion from OpenAI-format messages.
"""
import json

import httpx

from app.providers.anthropic import AnthropicProvider, build_request
from app.providers.base import ContentDelta, Finish, ToolCall


def _sse(events: list[tuple[str, dict]]) -> bytes:
    out = ""
    for name, payload in events:
        out += f"event: {name}\ndata: {json.dumps(payload)}\n\n"
    return out.encode()


def _provider(events: list[tuple[str, dict]]) -> AnthropicProvider:
    payload = _sse(events)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AnthropicProvider(apikey="k", client=client)


async def _collect(p: AnthropicProvider) -> list:
    out: list = []
    async for ev in p.stream(messages=[], tools=[], model="m"):
        out.append(ev)
    return out


# ---------- request conversion ----------

def test_build_request_lifts_system_and_converts_tools():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "grep", "description": "search", "parameters": {"type": "object"}}}]
    req = build_request(messages, tools, "claude-x", max_tokens=1024)
    assert req["model"] == "claude-x"
    assert req["max_tokens"] == 1024
    assert req["system"] == "you are helpful"
    assert req["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert req["tools"] == [{"name": "grep", "description": "search", "input_schema": {"type": "object"}}]


def test_build_request_converts_tool_roundtrip():
    """assistant tool_calls -> tool_use block; tool results -> user tool_result block."""
    messages = [
        {"role": "user", "content": "find it"},
        {"role": "assistant", "content": "searching",
         "tool_calls": [{"id": "c1", "function": {"name": "grep", "arguments": "{\"pattern\":\"x\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "grep", "content": "matched: 1"},
        {"role": "assistant", "content": "done"},
    ]
    req = build_request(messages, [], "claude-x")
    msgs = req["messages"]
    assert msgs[0] == {"role": "user", "content": [{"type": "text", "text": "find it"}]}
    assert msgs[1]["role"] == "assistant"
    assert {"type": "text", "text": "searching"} in msgs[1]["content"]
    assert {"type": "tool_use", "id": "c1", "name": "grep", "input": {"pattern": "x"}} in msgs[1]["content"]
    # tool result is expressed as a user-role tool_result block
    assert msgs[2]["role"] == "user"
    assert {"type": "tool_result", "tool_use_id": "c1", "content": "matched: 1"} in msgs[2]["content"]
    assert msgs[3] == {"role": "assistant", "content": [{"type": "text", "text": "done"}]}


# ---------- stream parsing ----------

async def test_text_and_tool_use():
    events = await _collect(_provider([
        ("message_start", {"type": "message_start", "message": {"id": "msg_1"}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "grep", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"pat"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "tern\":\"foo\"}"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        ("message_stop", {"type": "message_stop"}),
    ]))
    assert events == [
        ContentDelta("Hello"),
        ToolCall(id="toolu_1", name="grep", args={"pattern": "foo"}),
        Finish(reason="tool_use"),
    ]


async def test_text_only_finish():
    events = await _collect(_provider([
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi there"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
        ("message_stop", {"type": "message_stop"}),
    ]))
    assert events == [ContentDelta("Hi there"), Finish(reason="end_turn")]


async def test_auth_headers_and_url():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["apikey"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        seen["url"] = str(request.url)
        return httpx.Response(200, content=_sse([("message_stop", {"type": "message_stop"})]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    p = AnthropicProvider(apikey="sk-ant", client=client)
    out = [ev async for ev in p.stream(messages=[], tools=[], model="m")]
    assert any(isinstance(ev, Finish) for ev in out)
    assert seen["apikey"] == "sk-ant"
    assert seen["version"] is not None
    assert seen["url"].endswith("/v1/messages")
