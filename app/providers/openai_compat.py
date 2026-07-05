"""OpenAI-compatible provider (OpenAI, DeepSeek, Moonshot, GLM, Ollama, vLLM, ...).

POSTs to ``{base_url}/chat/completions`` with ``stream=true`` and normalizes the
SSE chunk stream into :class:`~app.providers.base.LLMEvent`. Fragmented tool-call
argument deltas are reassembled per ``index`` and emitted as a single
:class:`ToolCall` once complete.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.providers.base import ContentDelta, Finish, LLMEvent, ToolCall


def build_payload(messages: list[dict], tools: list[dict], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
    }


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, *, base_url: str, apikey: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.apikey = apikey
        self._client = client

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> AsyncIterator[LLMEvent]:
        payload = build_payload(messages, tools, model)
        headers = {"Authorization": f"Bearer {self.apikey}"}
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                ctype = resp.headers.get("content-type", "")
                if resp.status_code >= 400:
                    # Surface the upstream error body: new-api/one-api gateways
                    # put the real reason (model not allowed, no quota, banned…)
                    # in the JSON body, not the status line.
                    body = (await resp.aread()).decode("utf-8", "replace").strip()
                    raise RuntimeError(f"upstream HTTP {resp.status_code}: {body[:500]}")
                if ctype and "event-stream" not in ctype:
                    # Upstream returned a non-SSE body — often a JSON error
                    # wrapped in HTTP 200 by a gateway (e.g. a wrong base_url).
                    # Surface it instead of silently producing an empty answer.
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
    # Per-index accumulator for fragmented tool calls.
    tool_order: list[int] = []
    tools: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None

    async for line in resp.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield ContentDelta(content)
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                slot = tools.setdefault(idx, {"id": None, "name": None, "args": ""})
                if idx not in tool_order:
                    tool_order.append(idx)
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if "arguments" in fn and fn["arguments"] is not None:
                    slot["args"] += fn["arguments"]
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

    for idx in tool_order:
        slot = tools[idx]
        raw_args = slot["args"] or "{}"
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args = {}
        yield ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"] or "", args=args)

    yield Finish(reason=finish_reason or "stop")
