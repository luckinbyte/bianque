"""FastAPI application: HTTP/SSE routes for the source-analysis agent.

Auth is a single shared ``APP_PASSWORD`` checked via the ``X-App-Token`` header
on every endpoint. The browser uses fetch-based SSE streaming so it can send
that header (EventSource cannot set headers). No secret ever appears in a URL.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.loop import run_turn
from app.config import Settings, load_settings
from app.providers import get_provider
from app.security import PathEscapeError, resolve_allowed, verify_token
from app.sessions import Session, SessionLimitError, SessionStore
from app.sse import sse_event

STATIC_DIR = Path(__file__).parent.parent / "static"
TERMINAL_EVENTS = {"answer", "cancelled", "error"}


# ---------- request bodies ----------

class SessionCreate(BaseModel):
    provider: str
    base_url: str | None = None
    apikey: str
    model: str
    repo_path: str


class MessageBody(BaseModel):
    question: str


class AnswerBody(BaseModel):
    call_id: str
    text: str


def default_provider_factory(session: Session):
    return get_provider(session.provider, base_url=session.base_url, apikey=session.apikey)


async def _reap_loop(store: "SessionStore", interval: int) -> None:
    """Periodically drop idle sessions (which clears their in-memory apikey)."""
    while True:
        await asyncio.sleep(interval)
        try:
            await store.reap_idle()
        except Exception:  # noqa: BLE001 - never let the reaper kill the loop
            pass


def create_app(
    settings: Settings,
    *,
    store: SessionStore | None = None,
    provider_factory: Callable[[Session], Any] | None = None,
) -> FastAPI:
    store = store or SessionStore(
        max_sessions=settings.max_concurrent_sessions,
        idle_timeout=settings.session_idle_timeout,
    )
    reap_interval = max(30, min(settings.session_idle_timeout, 300))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        reaper = asyncio.create_task(_reap_loop(store, reap_interval))
        try:
            yield
        finally:
            reaper.cancel()

    app = FastAPI(title="Bianque", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.provider_factory = provider_factory

    def require_token(
        request: Request,
        x_app_token: str | None = Header(default=None, alias="X-App-Token"),
    ) -> None:
        if not verify_token(x_app_token, request.app.state.settings.app_password):
            raise HTTPException(status_code=401, detail="invalid or missing X-App-Token")

    def _session(request: Request, session_id: str) -> Session:
        s = request.app.state.store.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="session not found")
        return s

    @app.post("/api/sessions", dependencies=[Depends(require_token)])
    async def create_session(request: Request, body: SessionCreate):
        cfg = request.app.state.settings
        try:
            repo_root = resolve_allowed(body.repo_path, cfg.allowed_roots)
        except PathEscapeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not repo_root.exists() or not repo_root.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {body.repo_path}")
        try:
            session = await request.app.state.store.create(
                provider=body.provider, base_url=body.base_url, apikey=body.apikey,
                model=body.model, repo_root=repo_root, roots=cfg.allowed_roots,
            )
        except SessionLimitError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"session_id": session.id}

    @app.post("/api/sessions/{session_id}/message", dependencies=[Depends(require_token)])
    async def post_message(request: Request, session_id: str, body: MessageBody):
        session = _session(request, session_id)
        if session.task is not None and not session.task.done():
            raise HTTPException(status_code=409, detail="a turn is already running for this session")
        factory = request.app.state.provider_factory or default_provider_factory
        provider = factory(session)
        session.queue = asyncio.Queue()
        session.task = asyncio.create_task(_runner(session, provider, body.question))
        session.touch()
        return {"stream_url": f"/api/sessions/{session_id}/stream"}

    @app.get("/api/sessions/{session_id}/stream", dependencies=[Depends(require_token)])
    async def stream(request: Request, session_id: str):
        session = _session(request, session_id)
        if session.queue is None:
            raise HTTPException(status_code=409, detail="no active turn; POST a message first")

        async def gen():
            q = session.queue
            while True:
                ev = await q.get()
                yield sse_event(ev)
                if ev.get("type") in TERMINAL_EVENTS:
                    break

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/answer", dependencies=[Depends(require_token)])
    async def post_answer(request: Request, session_id: str, body: AnswerBody):
        session = _session(request, session_id)
        fut = session.pending.get(body.call_id)
        if fut is None or fut.done():
            raise HTTPException(status_code=404, detail="no pending clarification for that call_id")
        fut.set_result(body.text)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/cancel", dependencies=[Depends(require_token)])
    async def cancel(request: Request, session_id: str):
        session = _session(request, session_id)
        if session.task is not None and not session.task.done():
            session.task.cancel()
        return {"ok": True}

    @app.get("/")
    async def index():
        idx = STATIC_DIR / "index.html"
        if idx.exists():
            return FileResponse(idx)
        return JSONResponse({"status": "ok", "note": "UI not built yet"})

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


async def _runner(session: Session, provider: Any, question: str) -> None:
    """Run one agent turn, pushing events into the session queue."""
    queue = session.queue

    async def emit(ev: dict[str, Any]) -> None:
        await queue.put(ev)

    try:
        await run_turn(session, provider, question, emit=emit)
    except asyncio.CancelledError:
        raise
    finally:
        session.task = None


# Module-level `app` for `uvicorn app.main:app`. Requires APP_PASSWORD in env;
# if unset, `app` is None and uvicorn will refuse to start (by design).
try:
    app = create_app(load_settings())
except ValueError:
    app = None  # type: ignore[assignment]
