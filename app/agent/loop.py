"""The agent engine loop.

Provider-agnostic: drives any :class:`~app.providers.base.LLMProvider`, streams
its events to an ``emit`` callback, dispatches the read-only tools, pauses on
``ask_user`` clarifications, enforces evidence, and handles cancellation.

``emit`` receives event dicts (see the SSE protocol in the plan). The HTTP layer
wires ``emit`` to the session's event queue.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

from app.agent.prompts import DEFAULT_SYSTEM_PROMPT
from app.agent.tools import ASK_USER_SCHEMA, call_tool, tool_schemas
from app.providers.base import ContentDelta, Finish, LLMEvent, ToolCall
from app.sessions import Session

Emit = Callable[[dict[str, Any]], Awaitable[None]]

_EVIDENCE_RE = re.compile(r"([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+):(\d+(?:-\d+)?)")


def extract_evidence(text: str) -> list[dict[str, str]]:
    """Best-effort extraction of `path:line` citations from answer text."""
    return [{"file": m.group(1), "line": m.group(2)} for m in _EVIDENCE_RE.finditer(text)]


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for the progress bar (~chars/4).

    Good enough to show context fill across providers without a tokenizer; not
    used for billing or cutoff. Counts message text + assembled tool-call args.
    """
    n = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            n += len(content)
        for tc in m.get("tool_calls") or []:
            n += len((tc.get("function") or {}).get("arguments") or "")
    return n // 4


def _tools() -> list[dict[str, Any]]:
    """Full tool list shown to the model: read-only filesystem tools + ask_user."""
    return [*tool_schemas(), ASK_USER_SCHEMA]


async def run_turn(
    session: Session,
    provider: Any,
    question: str,
    *,
    emit: Emit,
    max_steps: int = 20,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context_window: int = 200_000,
) -> None:
    """Run one user question to completion (answer / clarification / cancel).

    Appends the system prompt once and the user message each call, then loops:
    model turn -> tools/ask_user -> model turn, until a final answer or a limit.
    """
    if not session.messages:
        session.messages.append({"role": "system", "content": system_prompt})
    session.messages.append({"role": "user", "content": question})
    session.touch()
    await emit({"type": "context", "used": estimate_tokens(session.messages), "window": context_window})

    try:
        for _ in range(max_steps):
            content_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            async for ev in provider.stream(session.messages, _tools(), session.model):
                if isinstance(ev, ContentDelta):
                    content_parts.append(ev.text)
                    await emit({"type": "step", "delta": ev.text})
                elif isinstance(ev, ToolCall):
                    tool_calls.append(ev)
                    await emit({"type": "tool_call", "call_id": ev.id, "tool": ev.name, "args": ev.args})
                elif isinstance(ev, Finish):
                    pass  # turn boundary; handled below

            assistant_text = "".join(content_parts)

            if not tool_calls:
                # No tool calls => this turn is the final answer.
                session.messages.append({"role": "assistant", "content": assistant_text})
                await emit({"type": "context", "used": estimate_tokens(session.messages), "window": context_window})
                await emit({
                    "type": "answer",
                    "text": assistant_text,
                    "evidence": extract_evidence(assistant_text),
                })
                return

            # Record the assistant's tool-call turn.
            session.messages.append({
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": [
                    {"id": tc.id, "function": {"name": tc.name,
                     "arguments": json.dumps(tc.args, ensure_ascii=False)}}
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                if tc.name == "ask_user":
                    await _handle_ask_user(session, tc, emit)
                else:
                    await _handle_tool(session, tc, emit)
                session.touch()
            await emit({"type": "context", "used": estimate_tokens(session.messages), "window": context_window})
        else:
            await emit({"type": "error", "message": "max reasoning steps reached without an answer"})

    except asyncio.CancelledError:
        # User hit Stop. Emit a terminal event, then let cancellation propagate.
        try:
            await emit({"type": "cancelled"})
        except Exception:
            pass
        raise
    except Exception as e:  # noqa: BLE001 - surface any failure to the client
        await emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


async def _handle_ask_user(session: Session, tc: ToolCall, emit: Emit) -> None:
    question = str(tc.args.get("question", ""))
    await emit({"type": "clarification", "call_id": tc.id, "question": question})
    future = asyncio.get_running_loop().create_future()
    session.pending[tc.id] = future
    try:
        answer = await future
    finally:
        session.pending.pop(tc.id, None)
    session.messages.append({
        "role": "tool", "tool_call_id": tc.id, "name": "ask_user", "content": str(answer),
    })


async def _handle_tool(session: Session, tc: ToolCall, emit: Emit) -> None:
    result = call_tool(tc.name, tc.args, repo_root=session.repo_root, roots=session.roots)
    summary = result.content if result.ok else f"ERROR: {result.error}"
    await emit({
        "type": "tool_result",
        "call_id": tc.id,
        "ok": result.ok,
        "summary": summary,
        "truncated": result.truncated,
    })
    session.messages.append({
        "role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": summary,
    })
