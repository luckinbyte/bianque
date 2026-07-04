"""Tests for the read-only source tools.

These tools must be: strictly read-only, path-sandboxed to ALLOWED_ROOTS, and
capped (to bound context / runtime). No subprocess.
"""
from app.agent.tools import ToolResult, call_tool, find_files, grep, list_dir, read_file

ROOT_MARK = lambda tmp: [tmp.resolve()]  # noqa: E731


def _make_repo(tmp_path):
    """Build a small sample repo under tmp_path and return (repo_root, roots)."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("needle here\ndef foo():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "b.md").write_text("# Title\nneedle in markdown\n", encoding="utf-8")
    (repo / "src" / "sub").mkdir()
    (repo / "src" / "sub" / "c.py").write_text("x = 2  # needle\n", encoding="utf-8")
    # junk dir that must be skipped by grep/find
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("needle should be ignored\n", encoding="utf-8")
    return repo, [tmp_path.resolve()]


# ---------- list_dir ----------

def test_list_dir_lists_entries_sorted_with_types(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = list_dir("src", repo_root=repo, roots=roots)
    assert res.ok
    # dirs get a trailing slash; files don't
    assert "sub/" in res.content
    assert "a.py" in res.content
    assert "b.md" in res.content
    # parent dir entry is not shown
    assert ".." not in res.content


def test_list_dir_on_file_is_error(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = list_dir("src/a.py", repo_root=repo, roots=roots)
    assert not res.ok
    assert res.error


def test_list_dir_rejects_escape(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = list_dir("../outside", repo_root=repo, roots=roots)
    assert not res.ok


# ---------- read_file ----------

def test_read_file_returns_contents(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = read_file("src/a.py", repo_root=repo, roots=roots)
    assert res.ok
    assert "def foo()" in res.content
    assert res.truncated is False


def test_read_file_range_slices_lines(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = read_file("src/a.py", start=2, end=3, repo_root=repo, roots=roots)
    assert res.ok
    assert "def foo()" in res.content
    assert "needle here" not in res.content  # line 1 excluded


def test_read_file_truncates_large_file(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    roots = [tmp_path.resolve()]
    big = repo / "big.txt"
    big.write_text("L\n" * 5000, encoding="utf-8")  # ~10k lines
    res = read_file("big.txt", repo_root=repo, roots=roots, max_chars=200)
    assert res.ok
    assert res.truncated is True


def test_read_file_on_dir_is_error(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = read_file("src", repo_root=repo, roots=roots)
    assert not res.ok


def test_read_file_rejects_escape(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = read_file("../../etc/passwd", repo_root=repo, roots=roots)
    assert not res.ok


# ---------- grep ----------

def test_grep_finds_matches_with_file_line(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = grep("needle", repo_root=repo, roots=roots)
    assert res.ok
    assert "src/a.py:1:" in res.content
    assert "src/sub/c.py:1:" in res.content
    assert "src/b.md:2:" in res.content


def test_grep_respects_glob_filter(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = grep("needle", glob="*.py", repo_root=repo, roots=roots)
    assert res.ok
    assert "src/a.py" in res.content
    assert "src/b.md" not in res.content  # filtered out


def test_grep_skips_junk_dirs(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = grep("needle", repo_root=repo, roots=roots)
    assert ".git/config" not in res.content


def test_grep_caps_results_and_marks_truncated(tmp_path):
    repo = tmp_path / "repo"; (repo / "src").mkdir(parents=True)
    roots = [tmp_path.resolve()]
    (repo / "src" / "f.py").write_text("\n".join(f"# match{i} needle" for i in range(50)), encoding="utf-8")
    res = grep("needle", repo_root=repo, roots=roots, max_results=5)
    assert res.ok
    assert res.truncated is True
    # 5 result lines at most
    assert res.content.count("\n") < 5


def test_grep_invalid_regex_is_error(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = grep("(unclosed", repo_root=repo, roots=roots)
    assert not res.ok


def test_grep_no_matches_returns_empty_ok(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = grep("zzz_no_such_thing", repo_root=repo, roots=roots)
    assert res.ok
    assert res.content == ""


# ---------- find_files ----------

def test_find_files_matches_glob(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = find_files("**/*.py", repo_root=repo, roots=roots)
    assert res.ok
    assert "src/a.py" in res.content
    assert "src/sub/c.py" in res.content
    assert "b.md" not in res.content


def test_find_files_skips_junk_dirs(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = find_files("**/*", repo_root=repo, roots=roots)
    assert ".git/config" not in res.content


# ---------- call_tool dispatch ----------

def test_call_tool_dispatches_read_file(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = call_tool("read_file", {"path": "src/a.py"}, repo_root=repo, roots=roots)
    assert isinstance(res, ToolResult)
    assert res.ok
    assert "def foo()" in res.content


def test_call_tool_unknown_name_is_error(tmp_path):
    repo, roots = _make_repo(tmp_path)
    res = call_tool("rm_rf", {"path": "."}, repo_root=repo, roots=roots)
    assert not res.ok
    # a write tool must never exist in the registry
    assert "rm_rf" not in {s["function"]["name"] for s in __import__("app.agent.tools", fromlist=["tool_schemas"]).tool_schemas()}
