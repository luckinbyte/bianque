"""Runtime settings loaded from environment variables.

Everything the deployment owns — the analysis directory (``REPO_ROOT``), the LLM
provider/base_url/model, and the context-window cap — is configured here and
shared by every session. The browser only supplies each user's own API key, so
no provider config or access password is entered client-side.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    provider: str
    base_url: str
    model: str
    context_window: int = 200_000
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_sessions: int = 8
    session_idle_timeout: int = 1800
    tls_cert: str | None = None
    tls_key: str | None = None
    project_guide: str | None = None

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert and self.tls_key)


def _get(env: dict[str, str], key: str, default: str = "") -> str:
    return env.get(key, default).strip()


def _get_int(env: dict[str, str], key: str, default: int) -> int:
    raw = _get(env, key, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _load_project_guide(env: dict[str, str]) -> str | None:
    """Read the optional admin-written project guide markdown.

    ``PROJECT_GUIDE`` points at a markdown file the deployment author pre-wrote
    with an intro / summary / navigation for the analyzed repo; its text is
    injected into the agents' system prompts to help them scope user questions.
    Optional: unset means no guide is injected. If set but the path is missing or
    unreadable, we warn and skip (the server still starts) rather than fail hard.
    """
    raw = _get(env, "PROJECT_GUIDE")
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("PROJECT_GUIDE %s could not be read (%s); skipping.", path, e)
        return None


def load_settings(env: dict[str, str] | None = None) -> Settings:
    e = env if env is not None else os.environ

    repo_raw = _get(e, "REPO_ROOT")
    if not repo_raw:
        raise ValueError(
            "REPO_ROOT must be set (the single analysis directory shared by all sessions)."
        )
    repo_root = Path(repo_raw).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"REPO_ROOT must be an existing directory: {repo_root}")

    model = _get(e, "MODEL")
    if not model:
        raise ValueError("MODEL must be set (the LLM model id used for every session).")

    provider = _get(e, "PROVIDER", "openai_compat")
    base_url = _get(e, "BASE_URL")
    if provider == "openai_compat" and not base_url:
        raise ValueError("openai_compat provider requires BASE_URL")

    return Settings(
        repo_root=repo_root,
        provider=provider,
        base_url=base_url,
        model=model,
        context_window=_get_int(e, "CONTEXT_WINDOW", 200_000),
        host=_get(e, "HOST", "0.0.0.0"),
        port=_get_int(e, "PORT", 8000),
        max_concurrent_sessions=_get_int(e, "MAX_CONCURRENT_SESSIONS", 8),
        session_idle_timeout=_get_int(e, "SESSION_IDLE_TIMEOUT", 1800),
        tls_cert=_get(e, "TLS_CERT") or None,
        tls_key=_get(e, "TLS_KEY") or None,
        project_guide=_load_project_guide(e),
    )
