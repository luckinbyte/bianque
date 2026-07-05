"""Tests for the agent engine loop, using scripted fake providers.

Covers: direct answer, tool-use round trip, ask_user clarification pause+resume,
cancellation, evidence extraction, and path-escape rejection via tools.
"""
import asyncio
from pathlib import Path

import pytest

from app.agent.loop import extract_evidence, run_turn
from app.providers.base import ContentDelta, Finish, LLMEvent, ToolCall
from app.sessions import Session


# ---------- fakes ----------

class FakeProvider:
    """Yields scripted turns. Each stream() call consumes the next turn."""
    name = "fake"

    def __init__(self, turns: list[list[LLMEvent]]):
        self.turns = list(turns)

    async def stream(self, messages, tools, model):
        for ev in self.turns.pop(0):
            yield ev


class BlockingProvider:
    """Yields one delta, then blocks forever (to test cancellation)."""
    name = "fake"

    async def stream(self, messages, tools, model):
        yield ContentDelta("starting...")
        await asyncio.Event().wait()  # never set


def _session(repo_root: Path, roots: list[Path]) -> Session:
    return Session(
        id="t", provider="openai_compat", base_url="http://x/v1", apikey="k",
        model="m", repo_root=repo_root, roots=roots,
    )


async def _drive(session, provider, question, events):
    async def emit(ev):
        events.append(ev)
    await run_turn(session, provider, question, emit=emit)


def _types(events):
    # context (progress-meter) events fire throughout the turn and are checked
    # separately; exclude them so the business-event sequence stays stable.
    return [e["type"] for e in events if e["type"] != "context"]


# ---------- scenarios ----------

async def test_direct_answer():
    provider = FakeProvider([[ContentDelta("The value is 42"), Finish("stop")]])
    events: list = []
    await _drive(_session(Path("/tmp"), []), provider, "what?", events)
    assert _types(events) == ["step", "answer"]
    assert events[-1] == {"type": "answer", "text": "The value is 42", "evidence": []}
    assert any(e["type"] == "context" for e in events)


async def test_tool_use_then_answer(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "a.py").write_text("ANSWER=42\n", encoding="utf-8")
    session = _session(repo, [tmp_path.resolve()])
    provider = FakeProvider([
        [ToolCall(id="c1", name="read_file", args={"path": "a.py"}), Finish("tool_calls")],
        [ContentDelta("It is 42."), Finish("stop")],
    ])
    events: list = []
    await _drive(session, provider, "read a.py", events)

    types = _types(events)
    assert types == ["tool_call", "tool_result", "step", "answer"]
    tool_call = next(e for e in events if e["type"] == "tool_call")
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_call["tool"] == "read_file"
    assert tool_result["ok"] is True
    assert "ANSWER=42" in tool_result["summary"]
    assert events[-1]["text"] == "It is 42."
    assert any(e["type"] == "context" for e in events)
    # history recorded the assistant tool call + tool result
    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


async def test_clarification_pauses_and_resumes(tmp_path):
    session = _session(tmp_path, [tmp_path.resolve()])
    provider = FakeProvider([
        [ToolCall(id="c1", name="ask_user", args={"question": "which scope?"}), Finish("tool_calls")],
        [ContentDelta("ok, based on your answer"), Finish("stop")],
    ])
    events: list = []
    task = asyncio.create_task(_drive(session, provider, "explain", events))

    # wait until the loop pauses on the clarification
    for _ in range(200):
        if any(e["type"] == "clarification" for e in events):
            break
        await asyncio.sleep(0.01)
    assert any(e["type"] == "clarification" for e in events)
    assert session.pending.get("c1") is not None  # loop is parked on a future

    # the user answers
    session.pending["c1"].set_result("the auth flow")
    await asyncio.wait_for(task, timeout=5)

    types = _types(events)
    assert types == ["tool_call", "clarification", "step", "answer"]
    assert events[-1]["text"] == "ok, based on your answer"
    assert any(e["type"] == "context" for e in events)


async def test_cancel_emits_cancelled(tmp_path):
    session = _session(tmp_path, [tmp_path.resolve()])
    events: list = []
    task = asyncio.create_task(_drive(session, BlockingProvider(), "q", events))
    await asyncio.sleep(0.2)  # let it emit the step and reach the blocking await
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(e["type"] == "cancelled" for e in events)


async def test_path_escape_tool_returns_error_result(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    session = _session(repo, [tmp_path.resolve()])
    provider = FakeProvider([
        [ToolCall(id="c1", name="read_file", args={"path": "../../../etc/passwd"}), Finish("tool_calls")],
        [ContentDelta("could not read"), Finish("stop")],
    ])
    events: list = []
    await _drive(session, provider, "q", events)
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["ok"] is False


# ---------- evidence extraction ----------

def test_extract_evidence_finds_file_line_citations():
    text = "The entry is src/main.py:42 and config in app/config.py:10-12."
    ev = extract_evidence(text)
    assert {"file": "src/main.py", "line": "42"} in ev
    assert {"file": "app/config.py", "line": "10-12"} in ev


def test_extract_evidence_empty_when_none():
    assert extract_evidence("no citations here") == []
