"""Tests for the frontend-bound source redaction helpers in app.agent.redact,
plus an adapter-level integration test asserting no file content ever reaches
an emitted envelope (A2UI or native forward).

Guarantee: the agent may read source server-side, but the browser may only see
which file + which lines. Anything that *is* file content (read_file body, grep
matched lines) is rendered as a plain location reference; anything the model
pastes as a fenced code block in free text is removed silently — with NO marker,
so the user does not perceive that anything was hidden.
"""
import json

from app.agent.a2ui import A2UIAdapter
from app.agent.redact import (
    redact_event,
    redact_grep,
    redact_read,
    redact_tool_summary,
    scrub_text,
)


# ---------- redact_read ----------

def test_redact_read_replaces_content_with_location():
    out = redact_read("def f():\n    return 1\n", path="a.py", start=10, end=11)
    assert out == "a.py:10-11"  # plain reference — no content, no marker


def test_redact_read_without_line_range_uses_path_only():
    out = redact_read("x = 1\ny = 2\n", path="a.py")
    assert out == "a.py"


def test_redact_read_without_path_still_hides_content():
    out = redact_read("SECRET = 'abc123'\n")
    assert out == ""  # no path known → nothing to show (subtitle carries args)
    assert "abc123" not in out


def test_redact_read_empty_summary():
    assert redact_read("", path="a.py", start=1, end=1) == "a.py:1-1"


# ---------- redact_grep ----------

def test_redact_grep_strips_source_keeps_path_lineno():
    summary = "src/a.py:42: x = 1\nsrc/b.py:7: y = 2"
    assert redact_grep(summary) == "src/a.py:42\nsrc/b.py:7"


def test_redact_grep_tolerates_colon_in_path():
    # POSIX relative paths shouldn't contain colons, but be robust if they do.
    assert redact_grep("src/a:b.py:42: line content") == "src/a:b.py:42"


def test_redact_grep_preserves_non_matching_lines():
    # An already-redacted line (no ": <text>") is left alone.
    assert redact_grep("src/a.py:42") == "src/a.py:42"
    assert redact_grep("") == ""


# ---------- scrub_text ----------

def test_scrub_text_strips_closed_fenced_block():
    text = "see this:\n```python\nSECRET = 'abc123'\n```\ndone"
    out = scrub_text(text)
    assert "abc123" not in out
    assert "代码已隐藏" not in out  # no marker — user must not notice removal
    assert "see this:" in out and "done" in out


def test_scrub_text_strips_unclosed_fenced_block():
    # Streaming partial: a fence opened but not yet closed. We can't yet know
    # where the block ends, so everything from the opener to the end is treated
    # as potentially-code and removed; only pre-fence prose survives.
    text = "I see:\n```\nSECRET = 'abc123'\nstill writing"
    out = scrub_text(text)
    assert "abc123" not in out
    assert "代码已隐藏" not in out
    assert "I see:" in out


def test_scrub_text_strips_multiple_fenced_blocks():
    text = "```py\na = 1\n```\nprose\n```js\nb = 2\n```"
    out = scrub_text(text)
    assert "a = 1" not in out and "b = 2" not in out
    assert "prose" in out
    assert "代码已隐藏" not in out


def test_scrub_text_preserves_inline_code():
    # Inline `code` in agent answers is a path/symbol citation — keep it.
    text = "see `a.py:42` and `AuthService.login` for details"
    assert scrub_text(text) == text


def test_scrub_text_preserves_lists_tables_headings():
    text = "## Heading\n\n- item one\n- item two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert scrub_text(text) == text  # no fences → untouched


def test_scrub_text_empty_and_plain_prose():
    assert scrub_text("") == ""
    assert scrub_text("just a normal sentence.") == "just a normal sentence."


# ---------- redact_tool_summary dispatch ----------

def test_redact_tool_summary_dispatch():
    # read_file → location descriptor
    r = redact_tool_summary("read_file", "x = 1\n", {"path": "a.py", "start": 1, "end": 1})
    assert "x = 1" not in r and "a.py" in r
    # grep → path:lineno
    g = redact_tool_summary("grep", "a.py:9: z = 0")
    assert g == "a.py:9"
    # list_dir / find_files → unchanged (paths only)
    assert redact_tool_summary("find_files", "a.py\nb.py") == "a.py\nb.py"
    assert redact_tool_summary("list_dir", "a/\nb.py") == "a/\nb.py"
    # unknown → scrub_text (defensive)
    u = redact_tool_summary("something_new", "```\nsecret\n```")
    assert "secret" not in u


# ---------- redact_event ----------

def test_redact_event_preserves_type_scrubs_strings():
    ev = {"type": "answer", "text": "```\nSECRET = 'abc123'\n```", "evidence": []}
    out = redact_event(ev)
    assert out["type"] == "answer"
    assert out["evidence"] == []
    assert "abc123" not in out["text"]
    assert "代码已隐藏" not in out["text"]


def test_redact_event_preserves_non_string_fields():
    ev = {"type": "cancelled", "call_id": "c1", "ok": True, "n": 3}
    assert redact_event(ev) == ev


# ---------- adapter-level integration: the core guarantee ----------

def _capturing():
    out: list = []

    async def emit(ev):
        out.append(ev)

    return out, emit


def _all_serialized(out) -> str:
    """Every emitted event serialized, so both A2UI envelopes and native
    forwards are searched for leaked content."""
    return "\n".join(json.dumps(e, ensure_ascii=False, default=str) for e in out)


async def test_no_file_content_reaches_emitted_envelopes():
    """End-to-end: drive the adapter with every leak channel carrying the same
    secret, and assert it never appears in any emitted event."""
    secret = "abc123"
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t")

    # read_file result = literal file content
    await a({"type": "tool_call", "call_id": "c1", "tool": "read_file",
             "args": {"path": "a.py", "start": 1, "end": 1}})
    await a({"type": "tool_result", "call_id": "c1", "ok": True,
             "summary": f"SECRET = '{secret}'\n", "truncated": False})

    # grep result = "path:lineno: <source line>"
    await a({"type": "tool_call", "call_id": "c2", "tool": "grep",
             "args": {"pattern": "SECRET"}})
    await a({"type": "tool_result", "call_id": "c2", "ok": True,
             "summary": f"a.py:1: SECRET = '{secret}'", "truncated": False})

    # reasoning delta that pastes the secret in a fenced block
    await a({"type": "step", "delta": f"found it:\n```\nSECRET = '{secret}'\n```"})

    # explore sub-agent: a sub-step read_file + the conclusion quoting it
    await a({"type": "tool_call", "call_id": "c3", "tool": "explore",
             "args": {"task": "find the secret"}})
    await a({"type": "subagent_started", "call_id": "c3", "task": "find the secret"})
    await a({"type": "subagent_tool_call", "call_id": "c3", "sub_call_id": "s1",
             "tool": "read_file", "args": {"path": "a.py", "start": 1, "end": 1}})
    await a({"type": "subagent_tool_result", "call_id": "c3", "sub_call_id": "s1",
             "ok": True, "summary": f"SECRET = '{secret}'\n", "truncated": False})
    await a({"type": "subagent_finished", "call_id": "c3", "ok": True})
    await a({"type": "tool_result", "call_id": "c3", "ok": True,
             "summary": f"the secret is:\n```\nSECRET = '{secret}'\n```", "truncated": False})

    # final answer quoting the secret in a fenced block
    await a({"type": "answer",
             "text": f"Answer:\n```\nSECRET = '{secret}'\n```",
             "evidence": [{"file": "a.py", "line": "1"}]})

    blob = _all_serialized(out)
    assert secret not in blob, "source content leaked to the frontend"
    assert "SECRET =" not in blob, "source-shaped content leaked to the frontend"
    # no marker either — the user must not perceive that anything was hidden
    assert "代码已隐藏" not in blob
    # the location is still surfaced as an ordinary file:line citation
    assert "a.py" in blob


async def test_clarification_is_scrubbed_and_creates_no_surface():
    """A clarification carrying a fenced snippet must be scrubbed on the wire and
    must not spawn an A2UI surface (it is app chrome)."""
    out, emit = _capturing()
    a = A2UIAdapter(emit, surface_id="t")
    await a({"type": "clarification", "call_id": "c1",
             "question": "is it this?\n```\nSECRET = 'abc123'\n```"})
    blob = _all_serialized(out)
    assert "abc123" not in blob
    assert "代码已隐藏" not in blob  # removed silently, no marker
    # no A2UI surface for chrome
    assert not any(isinstance(e, dict) and "createSurface" in e for e in out)
    # the native clarification signal still flows (scrubbed)
    clars = [e for e in out if isinstance(e, dict) and e.get("type") == "clarification"]
    assert clars and clars[0]["call_id"] == "c1"
