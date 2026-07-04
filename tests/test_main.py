"""HTTP integration tests (auth, routing, SSE) via FastAPI TestClient.

Uses an injectable provider factory so no real LLM is contacted.
"""
import asyncio

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.providers.base import ContentDelta, Finish, ToolCall

TOKEN = "pw"
HDR = {"X-App-Token": TOKEN}


class FakeProvider:
    name = "fake"

    def __init__(self, turns):
        self.turns = list(turns)

    async def stream(self, messages, tools, model):
        for ev in self.turns.pop(0):
            yield ev


class BlockingProvider:
    name = "fake"

    async def stream(self, messages, tools, model):
        yield ContentDelta("starting...")
        await asyncio.Event().wait()


def _app(tmp_path, provider_factory=None):
    settings = load_settings({"APP_PASSWORD": TOKEN, "ALLOWED_ROOTS": str(tmp_path)})
    return create_app(settings, provider_factory=provider_factory)


def _create(client, repo_path):
    return client.post(
        "/api/sessions",
        headers=HDR,
        json={"provider": "openai_compat", "base_url": "http://x/v1",
              "apikey": "k", "model": "m", "repo_path": str(repo_path)},
    )


# ---------- auth ----------

def test_missing_token_is_401(tmp_path):
    c = TestClient(_app(tmp_path))
    r = c.post("/api/sessions", json={"provider": "x", "apikey": "k", "model": "m", "repo_path": str(tmp_path)})
    assert r.status_code == 401


def test_wrong_token_is_401(tmp_path):
    c = TestClient(_app(tmp_path))
    r = c.post("/api/sessions", headers={"X-App-Token": "nope"}, json={})
    assert r.status_code == 401


# ---------- session creation ----------

def test_create_session_ok(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    c = TestClient(_app(tmp_path))
    r = _create(c, repo)
    assert r.status_code == 200
    assert "session_id" in r.json()


def test_create_session_rejects_repo_outside_roots(tmp_path):
    c = TestClient(_app(tmp_path))
    r = _create(c, "/etc")
    assert r.status_code == 400


# ---------- message + stream ----------

def test_message_then_stream_simple_answer(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    factory = lambda session: FakeProvider([[ContentDelta("Hello "), ContentDelta("world."), Finish("stop")]])
    c = TestClient(_app(tmp_path, provider_factory=factory))
    sid = _create(c, repo).json()["session_id"]

    m = c.post(f"/api/sessions/{sid}/message", headers=HDR, json={"question": "hi"})
    assert m.status_code == 200

    s = c.get(f"/api/sessions/{sid}/stream", headers=HDR)
    assert s.status_code == 200
    assert "step" in s.text
    assert "answer" in s.text
    assert "Hello " in s.text


def test_stream_unknown_session_is_404(tmp_path):
    c = TestClient(_app(tmp_path))
    s = c.get("/api/sessions/nope/stream", headers=HDR)
    assert s.status_code == 404


def test_answer_without_pending_is_404(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    c = TestClient(_app(tmp_path))
    sid = _create(c, repo).json()["session_id"]
    r = c.post(f"/api/sessions/{sid}/answer", headers=HDR, json={"call_id": "c1", "text": "x"})
    assert r.status_code == 404


def test_cancel_emits_cancelled(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    factory = lambda session: BlockingProvider()
    c = TestClient(_app(tmp_path, provider_factory=factory))
    sid = _create(c, repo).json()["session_id"]

    c.post(f"/api/sessions/{sid}/message", headers=HDR, json={"question": "q"})
    can = c.post(f"/api/sessions/{sid}/cancel", headers=HDR)
    assert can.status_code == 200

    s = c.get(f"/api/sessions/{sid}/stream", headers=HDR)
    assert "cancelled" in s.text
