"""In-memory session store.

A session is one conversation: provider config + the user's apikey + the running
agent turn + pending clarification futures. The store enforces a concurrency
cap and reaps idle sessions (which also clears their in-memory apikey).
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SessionLimitError(Exception):
    """Raised when the max concurrent sessions cap has been reached."""


@dataclass
class Session:
    id: str
    provider: str
    base_url: str
    apikey: str
    model: str
    repo_root: Path
    roots: list[Path]
    messages: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task | None = None
    queue: asyncio.Queue | None = None
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    last_active: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_active = time.monotonic()


class SessionStore:
    def __init__(self, *, max_sessions: int, idle_timeout: int):
        self._sessions: dict[str, Session] = {}
        self._max = max_sessions
        self._idle_timeout = idle_timeout
        self._lock = asyncio.Lock()

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    @property
    def is_full(self) -> bool:
        return len(self._sessions) >= self._max

    async def create(self, **kwargs: Any) -> Session:
        async with self._lock:
            if len(self._sessions) >= self._max:
                raise SessionLimitError(
                    f"max concurrent sessions ({self._max}) reached; try again later"
                )
            session = Session(id=secrets.token_urlsafe(16), **kwargs)
            self._sessions[session.id] = session
            return session

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            _cancel_task(session)
            return True

    async def reap_idle(self, *, now: float | None = None) -> list[str]:
        """Remove sessions idle longer than idle_timeout. Returns removed ids."""
        now = now if now is not None else time.monotonic()
        removed: list[str] = []
        async with self._lock:
            for sid, session in list(self._sessions.items()):
                if now - session.last_active > self._idle_timeout:
                    _cancel_task(session)
                    self._sessions.pop(sid, None)
                    removed.append(sid)
        return removed


def _cancel_task(session: Session) -> None:
    task = session.task
    if task is not None and not task.done():
        task.cancel()
