"""HTTP integration tests (routing, SSE) via FastAPI TestClient.

Uses an injectable provider factory so no real LLM is contacted.
"""
import asyncio

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.providers.base import ContentDelta, Finish


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
    settings = load_settings({
        "REPO_ROOT": str(tmp_path),
        "MODEL": "m",
        "BASE_URL": "http://x/v1",
    })
    return create_app(settings, provider_factory=provider_factory)


def _create(client):
    return client.post("/api/sessions", json={"apikey": "k"})


# ---------- config ----------

def test_config_endpoint(tmp_path):
    c = TestClient(_app(tmp_path))
    r = c.get("/api/config")
    assert r.status_code == 200
    j = r.json()
    assert j["repo_root"] == str(tmp_path.resolve())
    assert j["provider"] == "openai_compat"
    assert j["base_url"] == "http://x/v1"
    assert j["model"] == "m"
    assert j["context_window"] == 200_000


# ---------- session creation ----------

def test_create_session_ok(tmp_path):
    c = TestClient(_app(tmp_path))
    r = _create(c)
    assert r.status_code == 200
    assert "session_id" in r.json()


# ---------- message + stream ----------

def test_message_then_stream_simple_answer(tmp_path):
    factory = lambda session: FakeProvider([[ContentDelta("Hello "), ContentDelta("world."), Finish("stop")]])
    c = TestClient(_app(tmp_path, provider_factory=factory))
    sid = _create(c).json()["session_id"]

    m = c.post(f"/api/sessions/{sid}/message", json={"question": "hi"})
    assert m.status_code == 200

    s = c.get(f"/api/sessions/{sid}/stream")
    assert s.status_code == 200
    assert "step" in s.text
    assert "answer" in s.text
    assert "Hello " in s.text


def test_stream_emits_context_event(tmp_path):
    factory = lambda session: FakeProvider([[ContentDelta("hi"), Finish("stop")]])
    c = TestClient(_app(tmp_path, provider_factory=factory))
    sid = _create(c).json()["session_id"]
    c.post(f"/api/sessions/{sid}/message", json={"question": "q"})
    s = c.get(f"/api/sessions/{sid}/stream")
    assert '"type": "context"' in s.text
    assert '"window": 200000' in s.text


def test_stream_unknown_session_is_404(tmp_path):
    c = TestClient(_app(tmp_path))
    s = c.get("/api/sessions/nope/stream")
    assert s.status_code == 404


def test_answer_without_pending_is_404(tmp_path):
    c = TestClient(_app(tmp_path))
    sid = _create(c).json()["session_id"]
    r = c.post(f"/api/sessions/{sid}/answer", json={"call_id": "c1", "text": "x"})
    assert r.status_code == 404


def test_cancel_emits_cancelled(tmp_path):
    factory = lambda session: BlockingProvider()
    c = TestClient(_app(tmp_path, provider_factory=factory))
    sid = _create(c).json()["session_id"]

    c.post(f"/api/sessions/{sid}/message", json={"question": "q"})
    can = c.post(f"/api/sessions/{sid}/cancel")
    assert can.status_code == 200

    s = c.get(f"/api/sessions/{sid}/stream")
    assert "cancelled" in s.text
