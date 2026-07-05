"""Frontend-bound source redaction — pure helpers, no I/O, no state.

Security invariant: the agent may read source server-side to reason, but no
source-code content may cross the SSE boundary to the browser. The frontend may
only learn *which file and which lines* (plus symbol names and prose). These
helpers are applied inside :class:`app.agent.a2ui.A2UIAdapter` — *after* the
agent loop — so ``session.messages`` retains raw content for reasoning while
only the frontend-bound stream is redacted.

Two layers:

* **Structural, per-tool** redaction for tool results that are *definitionally*
  file content (``read_file`` body, ``grep`` matched lines). These are rebuilt
  as a location descriptor / a ``path:lineno`` list and never echo content.
* **Free-text scrubbing** for agent-authored strings (answers, conclusions,
  reasoning, errors, clarifications). Strips code blocks the model might paste
  despite the prompt instructing it not to.

What is deliberately **not** scrubbed:

* ``list_dir`` / ``find_files`` results (path lists only).
* Tool-call arg subtitles (``path`` / ``start`` / ``end`` / ``pattern`` /
  ``glob`` — agent-authored references, not file content; and path+lines is
  exactly the surface we want exposed).
* Inline `` `code` `` in free text — in agent answers this is overwhelmingly a
  path / symbol / identifier (the desired citation surface), so stripping it
  would gut readability.
* Indented (4-space) code blocks — markdown nested lists use the same
  indentation, so stripping them would mangle legitimate list-heavy answers.
  The prompt + structural redaction + fenced-block scrubbing cover the
  realistic threats; revisit if indented-code leaks are observed.
"""
from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# structural, per-tool
# --------------------------------------------------------------------------- #

def redact_read(
    summary: str,
    path: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> str:
    """Render a ``read_file`` result as a plain location reference.

    Returns just *which file and which lines* were read (e.g. ``a.py:10-12``) —
    no content, and crucially no "redacted" marker: the user should perceive an
    ordinary file reference, unaware the body was withheld. Falls back to an
    empty string when no path is known (the tool-call subtitle already carries
    the args).
    """
    if path and start is not None and end is not None:
        return f"{path}:{start}-{end}"
    if path and start is not None:
        return f"{path}:{start}-"
    if path:
        return path
    return ""


# Each grep line is "<relpath>:<lineno>: <source line>". Keep "<relpath>:<lineno>".
# The non-greedy path capture tolerates a colon inside a POSIX path (src/a:b.py:42).
_GREP_LINE = re.compile(r"^(.*?):(\d+): .*$", re.MULTILINE)


def redact_grep(summary: str) -> str:
    """Strip the trailing ``: <source line>`` from each ``path:lineno: line``.

    Lines that do not match the grep format (e.g. an empty result) are left
    untouched.
    """
    return _GREP_LINE.sub(r"\1:\2", summary)


# --------------------------------------------------------------------------- #
# free-text scrubbing
# --------------------------------------------------------------------------- #

# A closed fence: ```[lang]\n ... ``` . Non-greedy so multiple blocks are handled.
_FENCED_CLOSED = re.compile(r"```[^\n]*\n?[\s\S]*?```")
# An opening fence with no closing fence — a streaming partial or a stray fence.
# Matched only after closed fences are removed, so any remaining ``` is unclosed.
_FENCED_UNCLOSED = re.compile(r"```[^\n]*[\s\S]*$")


def scrub_text(text: str) -> str:
    """Silently remove pasted fenced code blocks from agent-authored free text.

    Each block (closed, and unclosed/streaming-partial) is removed with *no*
    replacement marker — the user must not perceive that anything was withheld.
    The unclosed case matters because reasoning text is scrubbed incrementally
    as it streams; without it a half-written fence would leak its content until
    closed. The prompt already steers the model away from pasting code, so this
    is a defense-in-depth backstop.

    Preserves prose, headings, lists, tables, and inline ``code`` (see module
    docstring for why).
    """
    if not text:
        return text
    text = _FENCED_CLOSED.sub("", text)
    text = _FENCED_UNCLOSED.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # tidy gaps left by removed blocks
    return text


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

def redact_tool_summary(
    tool: str, summary: str, args: dict[str, Any] | None = None
) -> str:
    """Dispatch redaction by tool name for a tool-result summary.

    ``read_file`` → location descriptor; ``grep`` → ``path:lineno`` list;
    ``list_dir`` / ``find_files`` → unchanged (paths only); anything else →
    :func:`scrub_text` (defensive — should not normally carry source).
    """
    args = args or {}
    if tool == "read_file":
        return redact_read(
            summary,
            path=args.get("path"),
            start=args.get("start"),
            end=args.get("end"),
        )
    if tool == "grep":
        return redact_grep(summary)
    if tool in ("list_dir", "find_files"):
        return summary
    return scrub_text(summary)


def redact_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-cloned event with every string value scrubbed.

    Used for native forwards (``answer`` / ``error`` / ``cancelled`` / unknown)
    so the SSE wire payload carries no source even though the frontend renders
    from the A2UI data model rather than these events. Non-string fields
    (``type``, ``call_id``, ``evidence`` lists, booleans, numbers) are preserved
    unchanged; we do not recurse into nested structures (``evidence`` is already
    ``file:line`` only).
    """
    out = dict(event)
    for k, v in out.items():
        if isinstance(v, str):
            out[k] = scrub_text(v)
    return out
