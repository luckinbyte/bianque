"""SSE event serialization."""
from __future__ import annotations

import json
from typing import Any


def sse_event(payload: dict[str, Any]) -> str:
    """Serialize a payload dict as one Server-Sent Event line group."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def format_event(type: str, /, **fields: Any) -> str:
    """Build a typed SSE event: ``{"type": type, ...fields}``."""
    payload = {"type": type, **fields}
    return sse_event(payload)
