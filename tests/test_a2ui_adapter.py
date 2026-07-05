"""Tests for the A2UI adapter: it must turn the agent loop's event stream into a
correct stream of A2UI envelopes (createSurface / updateComponents /
updateDataModel) while passing app-chrome events (context, clarification) and
terminal signals through untouched.
"""
import pytest

from app.agent.a2ui import A2UIAdapter


# ---------- helpers ----------

def _capturing():
    out: list = []

    async def emit(ev):
        out.append(ev)

    return out, emit


async def _drive(adapter, events):
    for ev in events:
        await adapter(ev)


def _components(out):
    """All component defs across every updateComponents message."""
    comps = []
    for e in out:
        if isinstance(e, dict) and "updateComponents" in e:
            comps.extend(e["updateComponents"]["components"])
    return comps


def _data(out):
    return [e["updateDataModel"] for e in out if isinstance(e, dict) and "updateDataModel" in e]


def _native(out):
    return [e for e in out if isinstance(e, dict) and "version" not in e]


def _has_surface(out):
    return any(isinstance(e, dict) and "createSurface" in e for e in out)


# ---------- surface lifecycle ----------

async def test_creates_surface_on_first_block_event():
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "step", "delta": "hi"})

    surf = [e for e in out if isinstance(e, dict) and "createSurface" in e]
    assert len(surf) == 1
    assert surf[0]["createSurface"]["surfaceId"] == "t1"
    assert surf[0]["createSurface"]["catalogId"] == "https://bianque.local/a2ui/v1"
    # root Column must exist
    roots = [c for c in _components(out) if c["id"] == "root"]
    assert roots and roots[0]["component"] == "Column"


async def test_passthrough_events_do_not_create_surface():
    """context/clarification are app chrome — they must not spawn a surface."""
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "context", "used": 10, "window": 200000})
    await a({"type": "clarification", "call_id": "c1", "question": "q?"})
    assert not _has_surface(out)
    # and they pass through unchanged
    assert {"type": "context", "used": 10, "window": 200000} in _native(out)
    assert {"type": "clarification", "call_id": "c1", "question": "q?"} in _native(out)


# ---------- reasoning streaming ----------

async def test_reasoning_streams_via_update_data_model():
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    for d in ("a", "b", "c"):
        await a({"type": "step", "delta": d})

    # the reasoning Text block is declared exactly once (structure), then only
    # data updates follow — no re-sending of the component per token.
    reasoning = [c for c in _components(out)
                 if c.get("component") == "Text" and c.get("variant") == "reasoning"]
    assert len(reasoning) == 1

    data = _data(out)
    text_updates = [d for d in data if d["path"].endswith("/text")]
    assert len(text_updates) == 3
    assert text_updates[-1]["value"] == "abc"


# ---------- tool blocks ----------

async def test_tool_block_structure():
    # find_files returns a path list — not source content — so it passes through
    # the redaction layer unchanged (read_file/grep results are rebuilt as
    # file:line references; covered in test_redact.py).
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "tool_call", "call_id": "c1", "tool": "find_files", "args": {"glob": "*.py"}})
    await a({"type": "tool_result", "call_id": "c1", "ok": True,
             "summary": "a.py\nb.py", "truncated": False})

    cards = [c for c in _components(out) if c.get("component") == "Card"]
    assert len(cards) == 1
    card = cards[0]
    assert card["title"] == "find_files"
    assert card["icon"] == "🔧"
    assert card["tone"] == "tool"
    assert "glob: *.py" in card["subtitle"]
    assert card["collapsible"] is True

    # body Text binds to the block's result path
    bid = card["id"]
    bodies = [c for c in _components(out)
              if c.get("component") == "Text" and c.get("text", {}).get("path", "").endswith("/result")]
    assert bodies and bodies[0]["text"]["path"] == f"/blocks/{bid}/result"

    data = _data(out)
    assert any(d["path"] == f"/blocks/{bid}/result" and d["value"] == "a.py\nb.py" for d in data)
    assert any(d["path"] == f"/blocks/{bid}/status" and d["value"] == "done" for d in data)


# ---------- explore sub-agent block ----------

async def test_explore_is_unified_no_tabs_conclusion_appended():
    """The explore card has NO 结论/探索过程 tabs — one unified journey. The
    sub-agent's conclusion is appended as the final item in that same journey,
    not routed to a separate tab."""
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "tool_call", "call_id": "c1", "tool": "explore", "args": {"task": "find x"}})
    await a({"type": "subagent_started", "call_id": "c1", "task": "find x"})
    await a({"type": "subagent_tool_call", "call_id": "c1", "sub_call_id": "s1",
             "tool": "find_files", "args": {"glob": "*.py"}})
    await a({"type": "subagent_tool_result", "call_id": "c1", "sub_call_id": "s1",
             "ok": True, "summary": "a.py", "truncated": False})
    await a({"type": "subagent_finished", "call_id": "c1", "ok": True})
    # the explore tool's own result carries the conclusion
    await a({"type": "tool_result", "call_id": "c1", "ok": True,
             "summary": "CONCLUSION", "truncated": False})

    cards = [c for c in _components(out) if c.get("component") == "Card"]
    explore = next(c for c in cards if c.get("tone") == "explore")
    assert explore["icon"] == "🔍"
    assert explore["subtitle"] == "find x"
    bid = explore["id"]

    # No Tabs component anywhere — the 结论/探索过程 split is gone.
    assert not any(c.get("component") == "Tabs" for c in _components(out))
    # The card body is the journey Column directly.
    assert explore["child"] == f"{bid}_journey"

    # a sub-step card exists for the sub-agent's find_files (path list — not redacted)
    substeps = [c for c in cards if c.get("tone") == "substep"]
    assert len(substeps) == 1
    assert substeps[0]["title"] == "find_files"
    sid = substeps[0]["id"]
    assert any(d["path"] == f"/blocks/{bid}/steps/{sid}/result" and d["value"] == "a.py"
               for d in _data(out))

    # the conclusion is appended as the final journey child and populated,
    # never written to the tool /result path. (The journey Column is re-sent on
    # each change; take the last version — the frontend overwrites by id.)
    journeys = [c for c in _components(out)
                if c.get("component") == "Column" and c.get("id") == f"{bid}_journey"]
    journey = journeys[-1]
    concl_id = f"{bid}_concl"
    assert journey["children"][-1] == concl_id
    assert any(c.get("id") == concl_id and c.get("component") == "Text" for c in _components(out))
    data = _data(out)
    assert any(d["path"] == f"/blocks/{bid}/conclusion" and d["value"] == "CONCLUSION" for d in data)
    assert all(d["path"] != f"/blocks/{bid}/result" for d in data)


# ---------- answer ----------

async def test_answer_renders_markdown_and_evidence():
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "answer", "text": "**hi**",
             "evidence": [{"file": "a.py", "line": "1"}, {"file": "b.py", "line": "2-3"}]})

    cards = [c for c in _components(out) if c.get("component") == "Card"]
    ans = next(c for c in cards if c.get("tone") == "answer")
    bid = ans["id"]
    # body Text bound to /blocks/{bid}/answer, populated via data
    assert any(d["path"] == f"/blocks/{bid}/answer" and d["value"] == "**hi**" for d in _data(out))
    # evidence chips, computed server-side
    chips = [c for c in _components(out) if c.get("component") == "Chips"]
    assert chips and chips[0]["items"] == ["a.py:1", "b.py:2-3"]
    # the answer is still passed through as a terminal signal
    assert any(e.get("type") == "answer" for e in _native(out))


# ---------- terminal passthrough ----------

async def test_error_and_cancelled_pass_through():
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "error", "message": "boom"})
    assert {"type": "error", "message": "boom"} in _native(out)

    out2, emit2 = _capturing()
    a2 = A2UIAdapter(emit2, surface_id="t2")
    await a2({"type": "cancelled"})
    assert {"type": "cancelled"} in _native(out2)


# ---------- de-duplication: final turn streams the answer/conclusion live,
# then it is emitted again as the answer/conclusion block. The adapter must drop
# the streamed duplicate so it isn't shown twice. ----------

async def test_answer_drops_duplicate_final_reasoning_block():
    """The final turn's text is streamed into a reasoning block AND returned as
    the answer. The matching reasoning block is dropped; earlier reasoning and
    the answer card remain."""
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    # earlier turn: reasoning, then a tool call
    await a({"type": "step", "delta": "Let me check. "})
    await a({"type": "tool_call", "call_id": "c1", "tool": "find_files", "args": {"glob": "*.py"}})
    await a({"type": "tool_result", "call_id": "c1", "ok": True, "summary": "a.py", "truncated": False})
    # final turn: its text IS the answer — streamed into a reasoning block, then emitted as the answer
    final_text = "The auth check is at auth.py:42."
    await a({"type": "step", "delta": final_text})
    await a({"type": "answer", "text": final_text, "evidence": [{"file": "auth.py", "line": "42"}]})

    comps = _components(out)
    roots = [c for c in comps if c.get("id") == "root"]
    final_children = roots[-1]["children"]
    rids = [c["id"] for c in comps
            if c.get("component") == "Text" and c.get("variant") == "reasoning"]
    aids = [c["id"] for c in comps
            if c.get("component") == "Card" and c.get("tone") == "answer"]

    assert rids[0] in final_children      # earlier reasoning survives
    assert rids[-1] not in final_children  # final-turn reasoning (== answer) dropped
    assert aids[-1] in final_children      # answer card present
    # and the answer text is still shown once, in the answer block
    assert any(d["path"].endswith("/answer") and d["value"] == final_text for d in _data(out))


async def test_answer_keeps_reasoning_when_it_differs_from_answer():
    """If the streamed reasoning does not verbatim match the answer (not the
    normal flow), it is kept rather than dropped — never hide genuine reasoning."""
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "step", "delta": "Some genuine reasoning here."})
    await a({"type": "answer", "text": "A different answer.", "evidence": []})

    comps = _components(out)
    roots = [c for c in comps if c.get("id") == "root"]
    rids = [c["id"] for c in comps
            if c.get("component") == "Text" and c.get("variant") == "reasoning"]
    assert rids[-1] in roots[-1]["children"]  # reasoning kept (no verbatim match)


async def test_explore_conclusion_streams_at_bottom_not_duplicated():
    """Each sub-agent turn's reasoning is its own journey segment appended in
    arrival order, so the final turn — the conclusion — streams at the BOTTOM
    and is promoted in place. It is never pinned to the top, never duplicated."""
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "tool_call", "call_id": "c1", "tool": "explore", "args": {"task": "find x"}})
    await a({"type": "subagent_started", "call_id": "c1", "task": "find x"})
    # earlier turn: reasoning, then a tool call (turn boundary)
    await a({"type": "subagent_step", "call_id": "c1", "delta": "Searching..."})
    await a({"type": "subagent_tool_call", "call_id": "c1", "sub_call_id": "s1",
             "tool": "find_files", "args": {"glob": "*.py"}})
    await a({"type": "subagent_tool_result", "call_id": "c1", "sub_call_id": "s1",
             "ok": True, "summary": "a.py", "truncated": False})
    # final turn: the conclusion, streamed live as a new bottom segment
    await a({"type": "subagent_step", "call_id": "c1", "delta": "The value is at a.py:1."})
    await a({"type": "subagent_finished", "call_id": "c1", "ok": True})
    await a({"type": "tool_result", "call_id": "c1", "ok": True,
             "summary": "The value is at a.py:1.", "truncated": False})

    comps = _components(out)
    bid = next(c["id"] for c in comps if c.get("tone") == "explore")
    final_children = [c for c in comps if c.get("id") == f"{bid}_journey"][-1]["children"]

    # the conclusion is the LAST journey child and was promoted to body styling.
    # (The segment was re-emitted with a new variant; the frontend overwrites by
    # id, so the last def for that id is the live one.)
    last_id = final_children[-1]
    last = [c for c in comps if c.get("id") == last_id][-1]
    assert last.get("variant") == "body"
    assert last["text"]["path"] == f"/blocks/{bid}/conclusion"

    # the conclusion text is written exactly once
    assert [d["value"] for d in _data(out) if d["path"].endswith("/conclusion")] == ["The value is at a.py:1."]

    # earlier reasoning survives as its own (reasoning-styled) segment, ahead of
    # the step, and never carried the conclusion text
    reasoning = [c for c in comps
                 if c.get("component") == "Text" and c.get("variant") == "reasoning"]
    assert reasoning, "earlier-turn reasoning segment should remain"
    seg_paths = [c["text"]["path"] for c in reasoning]
    assert all("/seg/" in p for p in seg_paths)
    assert any(d["path"] in seg_paths and d["value"] == "Searching..." for d in _data(out))


async def test_explore_empty_conclusion_appends_dedicated_block():
    """If the sub-agent produced no final-turn text (e.g. 'exploration
    incomplete'), there is no live segment to promote, so a dedicated conclusion
    block is appended at the bottom instead."""
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t1")
    await a({"type": "tool_call", "call_id": "c1", "tool": "explore", "args": {"task": "find x"}})
    await a({"type": "subagent_started", "call_id": "c1", "task": "find x"})
    await a({"type": "subagent_finished", "call_id": "c1", "ok": True})
    await a({"type": "tool_result", "call_id": "c1", "ok": True,
             "summary": "exploration incomplete", "truncated": False})

    comps = _components(out)
    bid = next(c["id"] for c in comps if c.get("tone") == "explore")
    final_children = [c for c in comps if c.get("id") == f"{bid}_journey"][-1]["children"]
    assert final_children[-1] == f"{bid}_concl"
    assert [d["value"] for d in _data(out) if d["path"].endswith("/conclusion")] == ["exploration incomplete"]
