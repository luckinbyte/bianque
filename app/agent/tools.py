"""Read-only source-analysis tools.

Hard guarantees:
  * Strictly read-only — there is no write/edit/delete/shell tool anywhere.
  * Every path is sandboxed via :func:`app.security.resolve_allowed` against
    ALLOWED_ROOTS, so escapes (``..``, absolute, symlinks) are rejected.
  * Output is capped to bound context size and runtime.

No subprocess is used. ``grep``/``find_files`` walk the tree in Python and
prune a fixed set of junk directories (``.git``, ``node_modules``, ...).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.security import PathEscapeError, resolve_allowed

# Directories never descended into during grep/find.
JUNK_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".eggs", "dist", "build", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", ".cache",
}

MAX_LIST = 500
MAX_READ_CHARS = 20_000
MAX_GREP_RESULTS = 200
MAX_FIND_RESULTS = 500


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: str | None = None
    truncated: bool = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _resolve(raw: str, repo_root: Path, roots: list[Path]) -> Path:
    return resolve_allowed(raw, roots, base=repo_root)


def _rel(p: Path, repo_root: Path) -> str:
    try:
        return str(p.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(p.resolve())


def _glob_to_regex(glob: str) -> re.Pattern:
    """Translate a glob to a regex. ``**/`` matches zero or more dir segments."""
    s = glob.replace("\\", "/")
    s = s.replace("**/", "\x00")
    s = s.replace("**", "\x00")
    parts = []
    for ch in s:
        if ch == "*":
            parts.append("[^/]*")
        elif ch == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(ch))
    pattern = "".join(parts).replace("\x00", "(?:[^/]+/)*")
    return re.compile("^" + pattern + "$")


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #

def list_dir(path: str, *, repo_root: Path, roots: list[Path], max_entries: int = MAX_LIST) -> ToolResult:
    try:
        p = _resolve(path, repo_root, roots)
    except PathEscapeError as e:
        return ToolResult(ok=False, error=str(e))
    if not p.exists():
        return ToolResult(ok=False, error=f"not found: {path}")
    if not p.is_dir():
        return ToolResult(ok=False, error=f"not a directory: {path}")
    entries: list[str] = []
    with os.scandir(p) as it:
        for e in it:
            entries.append(e.name + "/" if e.is_dir(follow_symlinks=False) else e.name)
    entries.sort()
    truncated = False
    if len(entries) > max_entries:
        entries = entries[:max_entries]
        truncated = True
    return ToolResult(ok=True, content="\n".join(entries), truncated=truncated)


def read_file(
    path: str,
    *,
    start: int | None = None,
    end: int | None = None,
    repo_root: Path,
    roots: list[Path],
    max_chars: int = MAX_READ_CHARS,
) -> ToolResult:
    try:
        p = _resolve(path, repo_root, roots)
    except PathEscapeError as e:
        return ToolResult(ok=False, error=str(e))
    if not p.exists():
        return ToolResult(ok=False, error=f"not found: {path}")
    if p.is_dir():
        return ToolResult(ok=False, error=f"is a directory: {path}")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(ok=False, error=f"read failed: {e}")
    lines = text.splitlines()
    if start is not None or end is not None:
        s = max(1, start or 1)
        e = end or len(lines)
        lines = lines[s - 1:e]
    content = "\n".join(lines)
    truncated = False
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True
    return ToolResult(ok=True, content=content, truncated=truncated)


def grep(
    pattern: str,
    *,
    glob: str | None = None,
    repo_root: Path,
    roots: list[Path],
    max_results: int = MAX_GREP_RESULTS,
) -> ToolResult:
    try:
        rgx = re.compile(pattern)
    except re.error as e:
        return ToolResult(ok=False, error=f"invalid regex: {e}")
    glob_re = _glob_to_regex(glob) if glob else None
    glob_has_slash = bool(glob and "/" in glob)
    out: list[str] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in JUNK_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if glob_re is not None:
                target = _rel(p, repo_root) if glob_has_slash else p.name
                if not glob_re.match(target):
                    continue
            try:
                data = p.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:  # skip binary
                continue
            text = data.decode("utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if rgx.search(line):
                    if len(out) < max_results:
                        out.append(f"{_rel(p, repo_root)}:{i}: {line}")
                    else:
                        truncated = True
                        break
            if truncated:
                break
        if truncated:
            break
    return ToolResult(ok=True, content="\n".join(out), truncated=truncated)


def find_files(
    glob: str,
    *,
    repo_root: Path,
    roots: list[Path],
    max_results: int = MAX_FIND_RESULTS,
) -> ToolResult:
    glob_re = _glob_to_regex(glob)
    glob_has_slash = "/" in glob
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in JUNK_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            target = _rel(p, repo_root) if glob_has_slash else p.name
            if glob_re.match(target):
                out.append(_rel(p, repo_root))
                if len(out) > max_results:
                    out = out[:max_results]
                    return ToolResult(ok=True, content="\n".join(out), truncated=True)
    return ToolResult(ok=True, content="\n".join(out), truncated=False)


# --------------------------------------------------------------------------- #
# dispatch + schemas
# --------------------------------------------------------------------------- #

def call_tool(name: str, args: dict[str, Any], *, repo_root: Path, roots: list[Path]) -> ToolResult:
    """Dispatch a filesystem tool by name. Read-only tools only."""
    try:
        if name == "read_file":
            return read_file(args["path"], start=args.get("start"), end=args.get("end"),
                             repo_root=repo_root, roots=roots)
        if name == "list_dir":
            return list_dir(args["path"], repo_root=repo_root, roots=roots)
        if name == "grep":
            return grep(args["pattern"], glob=args.get("glob"), repo_root=repo_root, roots=roots)
        if name == "find_files":
            return find_files(args["glob"], repo_root=repo_root, roots=roots)
        return ToolResult(ok=False, error=f"unknown tool: {name}")
    except KeyError as e:
        return ToolResult(ok=False, error=f"missing argument: {e}")


def tool_schemas() -> list[dict]:
    """OpenAI-style function schemas for the read-only filesystem tools.

    Note: ``ask_user`` (clarification) is added by the agent loop, not here —
    it is interactive, not a read-only filesystem tool.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file under the repo, optionally a 1-indexed inclusive line range [start, end].",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repo-relative or absolute path within allowed roots."},
                        "start": {"type": "integer", "description": "First line (1-indexed, inclusive)."},
                        "end": {"type": "integer", "description": "Last line (1-indexed, inclusive)."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List entries in a directory (dirs get a trailing slash).",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Recursively search file contents with a Python regex. Skips .git/node_modules/etc and binary files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Python regular expression."},
                        "glob": {"type": "string", "description": "Optional filename/path glob filter, e.g. '*.py'."},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_files",
                "description": "Find files whose path matches a glob (e.g. '**/*.py').",
                "parameters": {
                    "type": "object",
                    "properties": {"glob": {"type": "string"}},
                    "required": ["glob"],
                },
            },
        },
    ]


ASK_USER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user a clarifying question when the request is ambiguous. "
            "Call this instead of guessing. The user's answer is returned to you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "A specific, focused question for the user."},
            },
            "required": ["question"],
        },
    },
}
