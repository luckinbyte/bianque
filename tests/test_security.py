import pytest

from app.security import (
    PathEscapeError,
    load_roots,
    resolve_allowed,
    verify_token,
)


# ---------- load_roots ----------

def test_load_roots_parses_colon_separated(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    assert load_roots(f"{a}:{b}") == [a.resolve(), b.resolve()]


def test_load_roots_strips_whitespace_and_ignores_empty_parts(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    assert load_roots(f"  {a}  ::   ") == [a.resolve()]


def test_load_roots_dedups(tmp_path):
    assert load_roots(f"{tmp_path}:{tmp_path}") == [tmp_path.resolve()]


def test_load_roots_none_and_empty_return_empty_list():
    assert load_roots(None) == []
    assert load_roots("") == []


# ---------- verify_token ----------

def test_verify_token_accepts_correct_secret():
    assert verify_token("s3cret", "s3cret") is True


def test_verify_token_rejects_wrong_secret():
    assert verify_token("nope", "s3cret") is False


def test_verify_token_rejects_none_or_empty():
    assert verify_token(None, "s3cret") is False
    assert verify_token("", "s3cret") is False
    assert verify_token("s3cret", "") is False
    assert verify_token(None, "") is False


# ---------- resolve_allowed: happy paths ----------

def test_resolve_relative_under_base(tmp_path):
    repo = tmp_path / "repo"; (repo / "src").mkdir(parents=True)
    roots = load_roots(str(tmp_path))
    assert resolve_allowed("src/foo.py", roots, base=repo) == (repo / "src" / "foo.py").resolve()


def test_resolve_absolute_under_roots(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    roots = load_roots(str(tmp_path))
    assert resolve_allowed(str(repo / "x.py"), roots) == (repo / "x.py").resolve()


def test_resolve_root_boundary_itself_is_allowed(tmp_path):
    roots = load_roots(str(tmp_path))
    assert resolve_allowed(str(tmp_path), roots) == tmp_path.resolve()


# ---------- resolve_allowed: rejections ----------

def test_resolve_rejects_dotdot_escape(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    roots = load_roots(str(repo))  # narrow root: only repo
    with pytest.raises(PathEscapeError):
        resolve_allowed("../../etc/passwd", roots, base=repo)


def test_resolve_rejects_absolute_outside_roots(tmp_path):
    roots = load_roots(str(tmp_path))
    with pytest.raises(PathEscapeError):
        resolve_allowed("/etc/passwd", roots)


def test_resolve_rejects_relative_without_base(tmp_path):
    # relative path with no base is ambiguous / unsafe -> reject
    roots = load_roots(str(tmp_path))
    with pytest.raises(PathEscapeError):
        resolve_allowed("foo.py", roots)


def test_resolve_rejects_symlink_escape(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (repo / "link").symlink_to(outside)
    roots = load_roots(str(repo))
    with pytest.raises(PathEscapeError):
        resolve_allowed("link/secret", roots, base=repo)
