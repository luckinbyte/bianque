"""Runtime settings loaded from environment variables.

A missing/blank ``APP_PASSWORD`` is fatal: it is the only access gate for the
LAN-shared service, so we refuse to start without one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.security import load_roots


@dataclass(frozen=True)
class Settings:
    app_password: str
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_roots: list[Path] = field(default_factory=list)
    max_concurrent_sessions: int = 8
    session_idle_timeout: int = 1800
    default_provider: str = "openai_compat"
    default_model: str = ""
    tls_cert: str | None = None
    tls_key: str | None = None

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


def load_settings(env: dict[str, str] | None = None) -> Settings:
    e = env if env is not None else os.environ
    password = _get(e, "APP_PASSWORD")
    if not password:
        raise ValueError(
            "APP_PASSWORD must be set (the shared access token for the service)."
        )
    return Settings(
        app_password=password,
        host=_get(e, "HOST", "0.0.0.0"),
        port=_get_int(e, "PORT", 8000),
        allowed_roots=load_roots(e.get("ALLOWED_ROOTS")),
        max_concurrent_sessions=_get_int(e, "MAX_CONCURRENT_SESSIONS", 8),
        session_idle_timeout=_get_int(e, "SESSION_IDLE_TIMEOUT", 1800),
        default_provider=_get(e, "DEFAULT_PROVIDER", "openai_compat"),
        default_model=_get(e, "DEFAULT_MODEL", ""),
        tls_cert=_get(e, "TLS_CERT") or None,
        tls_key=_get(e, "TLS_KEY") or None,
    )
