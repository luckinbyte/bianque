"""Tests for the ``explore`` sub-agent.

Covers the three guarantees the feature rests on:
- context isolation — the sub-agent's exploratory reads stay out of the main
  session's message list; only the conclusion is appended;
- the context meter reflects the main agent only — the sub-agent's filtered
  emit never produces a ``context`` event, and the last ``context.used`` equals
  ``estimate_tokens(session.messages)``;
- scoping & robustness — the sub-agent gets only the 4 read-only tools (no
  ``explore``/``ask_user``), degrades gracefully on step exhaustion, and lets
  cancellation propagate through the whole tree.
"""
import asyncio
from pathlib import Path

import pytest

from app.agent.loop import _subagent_emit, estimate_tokens, run_turn
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


class RecordingProvider(FakeProvider):
    """FakeProvider that records the tool names passed to each stream() call."""

    def __init__(self, turns):
        super().__init__(turns)
        self.seen_tools: list[list[str]] = []

    async def stream(self, messages, tools, model):
        self.seen_tools.append([t["function"]["name"] for t in tools])
        async for ev in super().stream(messages, tools, model):
            yield ev


class MessageCapturingProvider(FakeProvider):
    """FakeProvider that captures the messages handed to its first stream() call,
    so a test can assert what system prompt an agent actually ran with."""

    def __init__(self, turns):
        super().__init__(turns)
        self.captured: list[dict] | None = None

    async def stream(self, messages, tools, model):
        if self.captured is None:
            self.captured = messages
        async for ev in super().stream(messages, tools, model):
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


async def _drive(session, provider, question, events, *, spawn_provider=None, project_guide=None):
    async def emit(ev):
        events.append(ev)
    await run_turn(
        session, provider, question, emit=emit,
        spawn_provider=spawn_provider, project_guide=project_guide,
    )


def _types(events):
    return [e["type"] for e in events if e["type"] != "context"]


# ---------- the context-isolation guarantee ----------

async def test_explore_returns_conclusion_only(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "a.py").write_text("SECRET=42\n", encoding="utf-8")
    session = _session(repo, [tmp_path.resolve()])

    main = FakeProvider([
        [ToolCall(id="c1", name="explore", args={"task": "find the value"}), Finish("tool_calls")],
        [ContentDelta("answered"), Finish("stop")],
    ])
    sub = FakeProvider([
        [ToolCall(id="s1", name="read_file", args={"path": "a.py"}), Finish("tool_calls")],
        [ContentDelta("The value lives at a.py:1"), Finish("stop")],
    ])
    events: list = []
    await _drive(session, main, "q", events, spawn_provider=lambda: sub)

    # Only the conclusion is recorded — not the sub-agent's internal read/turns.
    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = next(m for m in session.messages if m["role"] == "tool")
    assert tool_msg["name"] == "explore"
    assert tool_msg["content"] == "The value lives at a.py:1"
    # the sub-agent's raw file read did NOT leak into the main message history
    assert all("SECRET=42" not in (m.get("content") or "") for m in session.messages)

    types = _types(events)
    assert "subagent_started" in types and "subagent_finished" in types
    # the sub-agent's read_file surfaces as a nested subagent_tool_call
    sub_calls = [e for e in events if e["type"] == "subagent_tool_call"]
    assert any(c["tool"] == "read_file" and c["call_id"] == "c1" for c in sub_calls)
    # the explore call itself still produces a normal tool_result (the conclusion)
    explore_result = next(e for e in events if e["type"] == "tool_result")
    assert explore_result["summary"] == "The value lives at a.py:1"


# ---------- the context-meter guarantee ----------

async def test_subagent_emit_drops_context():
    """The filter wrapper is the seam that guarantees no sub-agent context leak."""
    out: list = []

    async def sink(ev):
        out.append(ev)

    sub = _subagent_emit("parent", sink)
    await sub({"type": "context", "used": 999, "window": 200_000})   # must be dropped
    await sub({"type": "step", "delta": "thinking"})
    await sub({"type": "tool_call", "call_id": "s1", "tool": "read_file", "args": {}})
    await sub({"type": "tool_result", "call_id": "s1", "ok": True, "summary": "x", "truncated": False})
    await sub({"type": "answer", "text": "noop"})                     # unrelated types dropped too

    assert all(e["type"] != "context" for e in out)
    assert [e["type"] for e in out] == [
        "subagent_step", "subagent_tool_call", "subagent_tool_result",
    ]
    assert out[0]["call_id"] == "parent" and out[0]["delta"] == "thinking"
    assert out[1]["sub_call_id"] == "s1" and out[1]["tool"] == "read_file"
    assert out[2]["summary"] == "x"


async def test_context_meter_uses_only_main_messages(tmp_path):
    """Hard guarantee: the last context.used == estimate_tokens(session.messages),
    even though the sub-agent read a large file internally."""
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "a.py").write_text("DATA=" + "x" * 5000 + "\n", encoding="utf-8")
    session = _session(repo, [tmp_path.resolve()])
    main = FakeProvider([
        [ToolCall(id="c1", name="explore", args={"task": "read it"}), Finish("tool_calls")],
        [ContentDelta("answered at a.py:1"), Finish("stop")],
    ])
    sub = FakeProvider([
        [ToolCall(id="s1", name="read_file", args={"path": "a.py"}), Finish("tool_calls")],
        [ContentDelta("found it at a.py:1"), Finish("stop")],
    ])
    events: list = []
    await _drive(session, main, "q", events, spawn_provider=lambda: sub)

    last_ctx = [e for e in events if e["type"] == "context"][-1]
    assert last_ctx["used"] == estimate_tokens(session.messages)
    # the big file body never entered the main messages
    assert all("DATA=" not in (m.get("content") or "") for m in session.messages)


# ---------- scoping & robustness ----------

async def test_subagent_tools_exclude_explore_and_ask_user(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "a.py").write_text("hi\n", encoding="utf-8")
    session = _session(repo, [tmp_path.resolve()])
    main = FakeProvider([
        [ToolCall(id="c1", name="explore", args={"task": "x"}), Finish("tool_calls")],
        [ContentDelta("ok"), Finish("stop")],
    ])
    sub = RecordingProvider([
        [ToolCall(id="s1", name="read_file", args={"path": "a.py"}), Finish("tool_calls")],
        [ContentDelta("done at a.py:1"), Finish("stop")],
    ])
    events: list = []
    await _drive(session, main, "q", events, spawn_provider=lambda: sub)

    assert sub.seen_tools  # the sub-agent drove at least one turn
    for names in sub.seen_tools:
        assert "explore" not in names and "ask_user" not in names
    assert set(sub.seen_tools[0]) == {"read_file", "list_dir", "grep", "find_files"}


async def test_subagent_maxsteps_returns_partial(tmp_path):
    """A sub-agent that never stops calling tools must not crash — it returns a
    partial conclusion and the main agent still answers."""
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "a.py").write_text("v=1\n", encoding="utf-8")
    session = _session(repo, [tmp_path.resolve()])
    main = FakeProvider([
        [ToolCall(id="c1", name="explore", args={"task": "x"}), Finish("tool_calls")],
        [ContentDelta("final answer"), Finish("stop")],
    ])
    looping = [ToolCall(id="s1", name="read_file", args={"path": "a.py"}), Finish("tool_calls")]
    sub = FakeProvider([looping] * 100)  # max_steps=15 caps it
    events: list = []
    await _drive(session, main, "q", events, spawn_provider=lambda: sub)

    assert any(e["type"] == "answer" for e in events)
    explore_result = next(e for e in events if e["type"] == "tool_result")
    assert explore_result["ok"] is True
    assert explore_result["summary"]  # non-empty partial conclusion


async def test_subagent_cancellation_propagates(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "a.py").write_text("x\n", encoding="utf-8")
    session = _session(repo, [tmp_path.resolve()])
    main = FakeProvider([
        [ToolCall(id="c1", name="explore", args={"task": "x"}), Finish("tool_calls")],
        [ContentDelta("never reached"), Finish("stop")],
    ])
    sub = BlockingProvider()  # blocks inside the sub-agent
    events: list = []
    task = asyncio.create_task(_drive(session, main, "q", events, spawn_provider=lambda: sub))
    await asyncio.sleep(0.2)  # reach the blocking sub-agent stream
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(e["type"] == "cancelled" for e in events)


# ---------- project guide injection ----------

async def test_project_guide_reaches_main_and_subagent(tmp_path):
    """A configured PROJECT_GUIDE is appended to BOTH the main agent's system
    message and the explore sub-agent's (local) system message."""
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "a.py").write_text("v=1\n", encoding="utf-8")
    session = _session(repo, [tmp_path.resolve()])

    guide = "MASTER-NAV-MARKER-XYZ"
    main = FakeProvider([
        [ToolCall(id="c1", name="explore", args={"task": "x"}), Finish("tool_calls")],
        [ContentDelta("done at a.py:1"), Finish("stop")],
    ])
    sub = MessageCapturingProvider([
        [ContentDelta("conclusion at a.py:1"), Finish("stop")],
    ])
    events: list = []
    await _drive(
        session, main, "q", events,
        spawn_provider=lambda: sub, project_guide=guide,
    )

    # main agent: the system message (appended once at conversation start) carries it
    assert session.messages[0]["role"] == "system"
    assert guide in session.messages[0]["content"]
    # sub-agent: its own (isolated) system message carries it too
    assert sub.captured is not None
    assert sub.captured[0]["role"] == "system"
    assert guide in sub.captured[0]["content"]


async def test_no_guide_leaves_system_prompt_unchanged(tmp_path):
    """With no guide configured, the system prompt passes through untouched
    (no empty 'Project guide' section, base prompt intact)."""
    repo = tmp_path / "repo"; repo.mkdir()
    session = _session(repo, [tmp_path.resolve()])
    main = FakeProvider([[ContentDelta("ans"), Finish("stop")]])
    events: list = []
    await _drive(session, main, "q", events)

    assert session.messages[0]["role"] == "system"
    assert "Project guide" not in session.messages[0]["content"]
