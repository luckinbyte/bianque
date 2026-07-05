"""The agent engine loop.

Provider-agnostic: drives any :class:`~app.providers.base.LLMProvider`, streams
its events to an ``emit`` callback, dispatches the read-only tools, pauses on
``ask_user`` clarifications, delegates broad exploration to an isolated
sub-agent (``explore``), enforces evidence, and handles cancellation.

``emit`` receives event dicts (see the SSE protocol in the plan). The HTTP layer
wires ``emit`` to the session's event queue.

Context-isolation invariant: the ``explore`` sub-agent runs on its OWN local
message list and emits through :func:`_subagent_emit`, which (a) renames its
events to ``subagent_*`` and (b) drops any ``context`` event. Only the main
:func:`run_turn` ever emits ``context``, computed from ``session.messages``; the
sub-agent appends exactly one ``tool`` message (its conclusion) to that list. So
the frontend context meter reflects the main agent only.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    EXPLORER_SYSTEM_PROMPT,
    with_project_guide,
)
from app.agent.tools import ASK_USER_SCHEMA, EXPLORE_SCHEMA, call_tool, tool_schemas
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
    """Full tool list shown to the main model: read-only filesystem tools,
    ``ask_user`` (clarification), and ``explore`` (delegation to a sub-agent)."""
    return [*tool_schemas(), ASK_USER_SCHEMA, EXPLORE_SCHEMA]


async def _stream_turn(
    provider: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    *,
    emit: Emit | None = None,
) -> tuple[str, list[ToolCall]]:
    """Drive one model turn to completion, accumulating streamed text + tool calls.

    Shared by the main loop and the sub-agent loop. When ``emit`` is given, each
    text delta is forwarded as a ``step`` event and each tool call as a
    ``tool_call`` event *as they arrive* (preserving live streaming for the UI).
    The sub-agent passes its filtered wrapper so these surface as ``subagent_*``.
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    async for ev in provider.stream(messages, tools, model):
        if isinstance(ev, ContentDelta):
            content_parts.append(ev.text)
            if emit is not None:
                await emit({"type": "step", "delta": ev.text})
        elif isinstance(ev, ToolCall):
            tool_calls.append(ev)
            if emit is not None:
                await emit({"type": "tool_call", "call_id": ev.id, "tool": ev.name, "args": ev.args})
        elif isinstance(ev, Finish):
            pass  # turn boundary
    return "".join(content_parts), tool_calls


def _subagent_emit(parent_call_id: str, emit: Emit) -> Emit:
    """Wrap ``emit`` for a sub-agent.

    Renames the sub-agent's ``step`` / ``tool_call`` / ``tool_result`` events to
    ``subagent_*`` (tagged with the parent explore ``call_id``; the inner tool
    id becomes ``sub_call_id``) and **drops ``context`` and anything else**. This
    is the seam that guarantees the sub-agent can never touch the main context
    meter — even if its inner loop tried to emit ``context``, it would be eaten.
    """
    async def sub(ev: dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "step":
            await emit({"type": "subagent_step", "call_id": parent_call_id, "delta": ev.get("delta", "")})
        elif t == "tool_call":
            await emit({
                "type": "subagent_tool_call", "call_id": parent_call_id,
                "sub_call_id": ev.get("call_id"), "tool": ev.get("tool"), "args": ev.get("args"),
            })
        elif t == "tool_result":
            await emit({
                "type": "subagent_tool_result", "call_id": parent_call_id,
                "sub_call_id": ev.get("call_id"), "ok": ev.get("ok", False),
                "summary": ev.get("summary", ""), "truncated": ev.get("truncated", False),
            })
        # context and any other type: intentionally dropped

    return sub


async def run_turn(
    session: Session,
    provider: Any,
    question: str,
    *,
    emit: Emit,
    max_steps: int = 20,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context_window: int = 200_000,
    spawn_provider: Callable[[], Any] | None = None,
    project_guide: str | None = None,
) -> None:
    """Run one user question to completion (answer / clarification / cancel).

    Appends the system prompt once and the user message each call, then loops:
    model turn -> tools/ask_user/explore -> model turn, until a final answer or
    a step limit.

    ``spawn_provider`` builds a fresh provider for each ``explore`` sub-agent; if
    omitted, the sub-agent reuses ``provider`` (fine in production — providers are
    stateless). It exists so tests can inject a scripted provider per agent.
    """
    if not session.messages:
        session.messages.append({
            "role": "system",
            "content": with_project_guide(system_prompt, project_guide),
        })
    session.messages.append({"role": "user", "content": question})
    session.touch()
    await emit({"type": "context", "used": estimate_tokens(session.messages), "window": context_window})

    try:
        for _ in range(max_steps):
            assistant_text, tool_calls = await _stream_turn(
                provider, session.messages, _tools(), session.model, emit=emit
            )

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
                elif tc.name == "explore":
                    await _handle_explore(
                        session, provider, tc, emit,
                        spawn_provider=spawn_provider, project_guide=project_guide,
                    )
                else:
                    await _handle_tool(session, tc, emit)
                session.touch()
            await emit({"type": "context", "used": estimate_tokens(session.messages), "window": context_window})
        else:
            await emit({"type": "error", "message": "max reasoning steps reached without an answer"})

    except asyncio.CancelledError:
        # User hit Stop. Emit a terminal event, then let cancellation propagate
        # (it also cancels any in-flight sub-agent).
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


async def _handle_explore(
    session: Session,
    provider: Any,
    tc: ToolCall,
    emit: Emit,
    *,
    spawn_provider: Callable[[], Any] | None,
    project_guide: str | None = None,
) -> None:
    """Delegate a broad exploration to an isolated sub-agent.

    The sub-agent runs on its own message list; only its final conclusion is
    appended to ``session.messages`` (as a normal ``tool`` message). Its events
    surface as ``subagent_*`` and never as ``context``. A failed exploration is
    turned into an error conclusion rather than crashing the main turn.
    """
    task = str(tc.args.get("task", ""))
    await emit({"type": "subagent_started", "call_id": tc.id, "task": task})
    sub_provider = spawn_provider() if spawn_provider else provider
    sub_emit = _subagent_emit(tc.id, emit)
    ok = True
    try:
        conclusion = await run_subagent(
            task=task,
            provider=sub_provider,
            model=session.model,
            repo_root=session.repo_root,
            roots=session.roots,
            emit=sub_emit,
            project_guide=project_guide,
        )
    except asyncio.CancelledError:
        raise  # Stop must kill the whole tree
    except Exception as e:  # noqa: BLE001 - don't let a sub-agent failure kill the main turn
        conclusion = f"exploration failed: {type(e).__name__}: {e}"
        ok = False
    await emit({"type": "subagent_finished", "call_id": tc.id, "ok": ok})
    # Standard tool_result so the explore tool block renders like any other tool;
    # its body shows the conclusion, the subagent_* events form the nested journey.
    await emit({
        "type": "tool_result", "call_id": tc.id, "ok": ok,
        "summary": conclusion, "truncated": False,
    })
    session.messages.append({
        "role": "tool", "tool_call_id": tc.id, "name": "explore", "content": conclusion,
    })


async def run_subagent(
    *,
    task: str,
    provider: Any,
    model: str,
    repo_root: Path,
    roots: list[Path],
    emit: Emit,
    max_steps: int = 15,
    project_guide: str | None = None,
) -> str:
    """Run an isolated exploration sub-agent to a single conclusion.

    Owns its message list (never the session's). Tools are the 4 read-only
    filesystem tools only — no ``ask_user``, no ``explore`` (no recursion, no
    clarifications). ``emit`` should be a :func:`_subagent_emit` wrapper so its
    events surface as ``subagent_*`` and never as ``context``.

    Returns the final conclusion text. Never raises on exhaustion or tool error
    — returns a partial / explicit "incomplete" string instead — so the caller
    can feed the conclusion back as a tool result. ``CancelledError`` propagates.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": with_project_guide(EXPLORER_SYSTEM_PROMPT, project_guide)},
        {"role": "user", "content": task},
    ]
    tools = tool_schemas()  # read-only FS only — no ask_user, no explore
    last_text = ""
    for _ in range(max_steps):
        assistant_text, tool_calls = await _stream_turn(
            provider, messages, tools, model, emit=emit
        )
        last_text = assistant_text

        if not tool_calls:
            return assistant_text  # the conclusion

        messages.append({
            "role": "assistant",
            "content": assistant_text or None,
            "tool_calls": [
                {"id": tc.id, "function": {"name": tc.name,
                 "arguments": json.dumps(tc.args, ensure_ascii=False)}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            result = call_tool(tc.name, tc.args, repo_root=repo_root, roots=roots)
            summary = result.content if result.ok else f"ERROR: {result.error}"
            await emit({
                "type": "tool_result", "call_id": tc.id, "ok": result.ok,
                "summary": summary, "truncated": result.truncated,
            })
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": summary,
            })

    return last_text or "exploration incomplete: no conclusion reached"
