import pytest

from app.config import load_settings


def test_requires_repo_root():
    with pytest.raises(ValueError):
        load_settings({"MODEL": "m"})


def test_requires_model(tmp_path):
    with pytest.raises(ValueError):
        load_settings({"REPO_ROOT": str(tmp_path)})


def test_rejects_missing_repo_dir(tmp_path):
    with pytest.raises(ValueError):
        load_settings({
            "REPO_ROOT": str(tmp_path / "nope"),
            "MODEL": "m",
            "BASE_URL": "http://x/v1",
        })


def test_openai_compat_requires_base_url(tmp_path):
    with pytest.raises(ValueError):
        load_settings({"REPO_ROOT": str(tmp_path), "MODEL": "m"})


def test_defaults(tmp_path):
    s = load_settings({
        "REPO_ROOT": str(tmp_path),
        "MODEL": "glm-4.5",
        "BASE_URL": "http://x/v1",
    })
    assert s.repo_root == tmp_path.resolve()
    assert s.provider == "openai_compat"
    assert s.base_url == "http://x/v1"
    assert s.model == "glm-4.5"
    assert s.context_window == 200_000
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.max_concurrent_sessions == 8
    assert s.session_idle_timeout == 1800
    assert s.tls_cert is None
    assert s.tls_key is None


def test_parses_env(tmp_path):
    s = load_settings({
        "REPO_ROOT": str(tmp_path),
        "PROVIDER": "anthropic",
        "BASE_URL": "https://api.anthropic.com",
        "MODEL": "claude-sonnet-4-6",
        "CONTEXT_WINDOW": "100000",
        "HOST": "127.0.0.1",
        "PORT": "9000",
        "MAX_CONCURRENT_SESSIONS": "4",
        "SESSION_IDLE_TIMEOUT": "600",
        "TLS_CERT": "/c.pem",
        "TLS_KEY": "/k.pem",
    })
    assert s.repo_root == tmp_path.resolve()
    assert s.provider == "anthropic"
    assert s.base_url == "https://api.anthropic.com"
    assert s.model == "claude-sonnet-4-6"
    assert s.context_window == 100000
    assert s.host == "127.0.0.1"
    assert s.port == 9000
    assert s.max_concurrent_sessions == 4
    assert s.session_idle_timeout == 600
    assert s.tls_cert == "/c.pem"
    assert s.tls_key == "/k.pem"
