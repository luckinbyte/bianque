from app.sse import format_event, sse_event


def test_sse_event_has_data_prefix_and_two_newlines():
    assert sse_event({"type": "step", "delta": "x"}) == 'data: {"type": "step", "delta": "x"}\n\n'


def test_sse_event_keeps_unicode_readable():
    ev = sse_event({"type": "step", "delta": "你好"})
    assert "你好" in ev  # ensure_ascii=False
    assert ev.endswith("\n\n")


def test_format_event_wraps_type():
    assert format_event("step", delta="x") == sse_event({"type": "step", "delta": "x"})
