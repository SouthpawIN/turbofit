from turbofit_runtime.request_normalisation import apply_stream_usage_default


def test_stream_default_adds_include_usage_only_for_streams():
    out = apply_stream_usage_default({"stream": True})
    assert out["stream_options"] == {"include_usage": True}


def test_stream_explicit_false_survives():
    payload = {"stream": True, "stream_options": {"include_usage": False}}
    out = apply_stream_usage_default(payload)
    assert out["stream_options"]["include_usage"] is False


def test_non_stream_unchanged():
    payload = {"stream": False, "messages": []}
    assert apply_stream_usage_default(payload) == payload
    assert "stream_options" not in payload


def test_existing_stream_options_preserved():
    payload = {"stream": True, "stream_options": {"foo": 1}}
    out = apply_stream_usage_default(payload)
    assert out["stream_options"] == {"foo": 1, "include_usage": True}
