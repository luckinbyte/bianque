"""Tests for the in-memory session store (no real agent tasks needed)."""
from pathlib import Path

import pytest

from app.sessions import Session, SessionLimitError, SessionStore


def _kwargs():
    return dict(provider="openai_compat", base_url="http://x/v1", apikey="k",
                model="m", repo_root=Path("/tmp"), roots=[])


async def test_create_and_get():
    store = SessionStore(max_sessions=3, idle_timeout=100)
    s = await store.create(**_kwargs())
    assert isinstance(s, Session)
    assert store.get(s.id) is s


async def test_unique_ids():
    store = SessionStore(max_sessions=10, idle_timeout=100)
    a = await store.create(**_kwargs())
    b = await store.create(**_kwargs())
    assert a.id != b.id


async def test_limit_exceeded_raises():
    store = SessionStore(max_sessions=1, idle_timeout=100)
    await store.create(**_kwargs())
    with pytest.raises(SessionLimitError):
        await store.create(**_kwargs())


async def test_get_unknown_returns_none():
    store = SessionStore(max_sessions=3, idle_timeout=100)
    assert store.get("nope") is None


async def test_delete_removes():
    store = SessionStore(max_sessions=3, idle_timeout=100)
    s = await store.create(**_kwargs())
    await store.delete(s.id)
    assert store.get(s.id) is None


async def test_reap_idle_removes_old_sessions():
    store = SessionStore(max_sessions=3, idle_timeout=10)
    s = await store.create(**_kwargs())
    s.last_active = 0.0  # artificially aged
    removed = await store.reap_idle(now=100.0)
    assert s.id in removed
    assert store.get(s.id) is None


async def test_reap_idle_keeps_active_sessions():
    store = SessionStore(max_sessions=3, idle_timeout=100)
    s = await store.create(**_kwargs())
    removed = await store.reap_idle(now=s.last_active + 5)
    assert removed == []
    assert store.get(s.id) is s


async def test_create_stores_apikey_in_session():
    store = SessionStore(max_sessions=3, idle_timeout=100)
    s = await store.create(**_kwargs())
    assert s.apikey == "k"
    assert s.provider == "openai_compat"
