import pytest

from app.config import load_settings


def test_requires_app_password():
    with pytest.raises(ValueError):
        load_settings({})


def test_defaults_when_only_password_set():
    s = load_settings({"APP_PASSWORD": "x"})
    assert s.app_password == "x"
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.max_concurrent_sessions == 8
    assert s.session_idle_timeout == 1800
    assert s.default_provider == "openai_compat"
    assert s.default_model == ""
    assert s.allowed_roots == []
    assert s.tls_cert is None
    assert s.tls_key is None


def test_parses_env(tmp_path):
    s = load_settings({
        "APP_PASSWORD": "pw",
        "HOST": "127.0.0.1",
        "PORT": "9000",
        "ALLOWED_ROOTS": str(tmp_path),
        "MAX_CONCURRENT_SESSIONS": "4",
        "SESSION_IDLE_TIMEOUT": "600",
        "DEFAULT_PROVIDER": "anthropic",
        "DEFAULT_MODEL": "claude-sonnet-4-6",
        "TLS_CERT": "/c.pem",
        "TLS_KEY": "/k.pem",
    })
    assert s.host == "127.0.0.1"
    assert s.port == 9000
    assert s.allowed_roots == [tmp_path.resolve()]
    assert s.max_concurrent_sessions == 4
    assert s.session_idle_timeout == 600
    assert s.default_provider == "anthropic"
    assert s.default_model == "claude-sonnet-4-6"
    assert s.tls_cert == "/c.pem"
    assert s.tls_key == "/k.pem"


def test_blank_password_is_rejected():
    with pytest.raises(ValueError):
        load_settings({"APP_PASSWORD": "   "})
