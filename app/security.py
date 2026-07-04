"""Security primitives: token check + path sandboxing for read-only source access.

All filesystem access from agent tools MUST go through ``resolve_allowed`` so a
path can never escape the configured ``ALLOWED_ROOTS``.
"""
from __future__ import annotations

import hmac
from pathlib import Path


class PathEscapeError(PermissionError):
    """Raised when a requested path falls outside ALLOWED_ROOTS."""


def load_roots(raw: str | None) -> list[Path]:
    """Parse a colon-separated list of allowed root directories.

    Entries are stripped, expanded, resolved, de-duplicated. Empty/blank input
    yields an empty list (the caller decides whether that is fatal).
    """
    if not raw:
        return []
    seen: set[str] = set()
    roots: list[Path] = []
    for part in raw.split(":"):
        token = part.strip()
        if not token:
            continue
        p = Path(token).expanduser().resolve(strict=False)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        roots.append(p)
    return roots


def verify_token(provided: str | None, expected: str) -> bool:
    """Constant-time comparison of an access token. None/empty never matches."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


def _is_within(p: Path, roots: list[Path]) -> bool:
    for r in roots:
        if p == r or r in p.parents:
            return True
    return False


def resolve_allowed(raw: str, roots: list[Path], base: Path | None = None) -> Path:
    """Resolve ``raw`` to a canonical path that MUST stay within ``roots``.

    * Absolute paths are used as-is (then checked).
    * Relative paths require ``base`` (the session repo root); without it they
      are ambiguous and therefore rejected.
    * Symlinks are followed (via :meth:`Path.resolve`), so a link that points
      outside the roots is rejected.
    """
    p = Path(raw).expanduser()
    if not p.is_absolute():
        if base is None:
            raise PathEscapeError("relative path requires a base directory")
        p = base / p
    p = p.resolve(strict=False)
    if not _is_within(p, roots):
        raise PathEscapeError(f"path outside ALLOWED_ROOTS: {p}")
    return p
