"""Anthropic (Claude) provider adapter.

POSTs to ``{base_url}/v1/messages`` with ``stream=true`` and normalizes the SSE
event stream into :class:`~app.providers.base.LLMEvent`. Tool-use ``input_json``
deltas are reassembled per content-block index and emitted as a single
:class:`ToolCall`.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.providers.base import ContentDelta, Finish, LLMEvent, ToolCall

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


def build_request(
    messages: list[dict],
    tools: list[dict],
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Convert OpenAI-format messages + tools to Anthropic's request shape.

    Anthropic uses content *blocks* (text / tool_use / tool_result) and requires
    strictly alternating user/assistant turns. Consecutive same-role messages
    (e.g. an assistant tool_use turn followed by tool results) are merged.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]

    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            continue
        role = m["role"]
        if role == "tool":
            _append(out, "user", {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": m.get("content", ""),
            })
        elif role == "assistant" and m.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments") or "{}"
                try:
                    inp = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    inp = {}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": inp})
            _append(out, "assistant", *blocks)
        else:
            content = m.get("content", "")
            if isinstance(content, list):
                _append(out, role, *content)
            elif content:
                _append(out, role, {"type": "text", "text": content})

    anthropic_tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
        if t.get("type") == "function"
    ]
    req: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": out,
        "stream": True,
    }
    if system_parts:
        req["system"] = "\n\n".join(system_parts)
    if anthropic_tools:
        req["tools"] = anthropic_tools
    return req


def _append(out: list[dict[str, Any]], role: str, *blocks: dict[str, Any]) -> None:
    """Append content blocks, merging into the previous message if same role."""
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": list(blocks)})


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, base_url: str = "https://api.anthropic.com", apikey: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.apikey = apikey
        self._client = client

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> AsyncIterator[LLMEvent]:
        payload = build_request(messages, tools, model)
        headers = {
            "x-api-key": self.apikey,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=payload,
                headers=headers,
            ) as resp:
                ctype = resp.headers.get("content-type", "")
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace").strip()
                    raise RuntimeError(f"upstream HTTP {resp.status_code}: {body[:500]}")
                if ctype and "event-stream" not in ctype:
                    body = (await resp.aread()).decode("utf-8", "replace").strip()
                    raise RuntimeError(
                        f"upstream did not return an SSE stream "
                        f"(content-type={ctype!r}, status={resp.status_code}): "
                        f"{body[:300]}"
                    )
                async for ev in _parse_stream(resp):
                    yield ev
        finally:
            if owns_client:
                await client.aclose()


async def _parse_stream(resp: httpx.Response) -> AsyncIterator[LLMEvent]:
    blocks: dict[int, dict[str, Any]] = {}
    block_order: list[int] = []
    stop_reason = "end_turn"

    async for line in resp.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        etype = obj.get("type")
        if etype == "content_block_start":
            idx = obj.get("index", 0)
            cb = obj.get("content_block", {})
            blocks[idx] = {
                "type": cb.get("type"),
                "id": cb.get("id"),
                "name": cb.get("name"),
                "args": "",
                "flushed": False,
            }
            block_order.append(idx)
        elif etype == "content_block_delta":
            idx = obj.get("index", 0)
            delta = obj.get("delta", {})
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield ContentDelta(delta["text"])
            elif delta.get("type") == "input_json_delta" and idx in blocks:
                blocks[idx]["args"] += delta.get("partial_json", "")
        elif etype == "content_block_stop":
            idx = obj.get("index", 0)
            block = blocks.get(idx)
            if block and block["type"] == "tool_use" and not block["flushed"]:
                yield _flush_tool(block)
                block["flushed"] = True
        elif etype == "message_delta":
            stop_reason = obj.get("delta", {}).get("stop_reason", stop_reason)
        elif etype == "message_stop":
            break

    # Flush any tool block that never got a content_block_stop.
    for idx in block_order:
        block = blocks.get(idx)
        if block and block["type"] == "tool_use" and not block["flushed"]:
            yield _flush_tool(block)
            block["flushed"] = True

    yield Finish(reason=stop_reason or "end_turn")


def _flush_tool(block: dict[str, Any]) -> ToolCall:
    raw = block["args"] or "{}"
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        args = {}
    return ToolCall(id=block["id"] or "tool_use", name=block["name"] or "", args=args)
