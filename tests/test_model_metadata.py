"""TF8 catalogue observation; fixtures prove control paths, not model health."""
import importlib.util
import json
from pathlib import Path

import pytest
from turbofit_runtime import native_lifecycle

SPEC = importlib.util.spec_from_file_location(
    "gateway_metadata", Path(__file__).resolve().parents[1] / "scripts/turbofit-gateway.py")
assert SPEC and SPEC.loader
GATEWAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATEWAY)


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    state = tmp_path / "runtime-state.json"
    routes = {"main": {"kind": "local", "alias": "Exact/Main:Q4", "context_length": 4096},
              "aux": {"kind": "local", "alias": "Exact/Aux:Q8", "context_length": 2048}}
    state.write_text(json.dumps({"active": "test", "routes": routes}))
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.delenv("TURBOFIT_LIFECYCLE_REQUIRED", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("catalogue must not resolve/probe/wake models")
    monkeypatch.setattr(GATEWAY, "resolve_main", forbidden)
    monkeypatch.setattr(GATEWAY, "resolve_aux", forbidden)
    monkeypatch.setattr(GATEWAY, "backend_state", forbidden)
    return state, routes


def test_raw_ids_and_unloaded_backing_identity(catalogue):
    models = GATEWAY.provider_models()
    assert models[0]["id"] == "auto"
    assert models[0]["metadata"]["role"] == "auto"
    assert models[0]["metadata"]["backing_model"] == "Exact/Main:Q4"
    assert models[0]["metadata"]["residency"] == "unknown"
    assert models[0]["context_length"] == 4096
    # Local/remote discovered models follow auto (none in test env)
    assert all(m["id"] != "active:main" for m in models)
    assert all(m["id"] != "active:aux" for m in models)


def test_shared_main_uses_main_backing_and_status_once(catalogue, monkeypatch):
    state, routes = catalogue
    routes["aux"] = {"kind": "shared-main", "alias": "stale-aux"}
    state.write_text(json.dumps({"active": "test", "routes": routes}))
    monkeypatch.setenv("TURBOFIT_LIFECYCLE_REQUIRED", "1")
    monkeypatch.setattr(native_lifecycle, "load_endpoint", lambda _: {})
    calls = []
    def status(endpoint, payload, timeout):
        calls.append((payload, timeout))
        return {"orphaned": False, "roles": {"main": {"residency": "loading",
            "backing_model": "Exact/Main:Q4", "context_length": 4096, "observed_at": 100,
            "freshness": {"age_s": 0, "max_age_s": 15, "stale": False}}}}
    monkeypatch.setattr(native_lifecycle, "lifecycle_request", status)
    models = GATEWAY.provider_models()
    assert models[0]["metadata"]["backing_model"] == "Exact/Main:Q4"
    assert models[0]["metadata"]["residency"] == "loading"
    assert calls == [({"action": "status"}, 0.5)]


@pytest.mark.parametrize("status", [
    {}, {"orphaned": True, "roles": {"main": {"waking": True}}},
    {"orphaned": False, "roles": {"main": {"waking": False, "failed": True}}},
    {"orphaned": False, "roles": {"main": {"waking": "true"}}},
    {"orphaned": False, "roles": []}, None,
])
def test_insufficient_or_failed_status_is_never_idle(catalogue, monkeypatch, status):
    monkeypatch.setenv("TURBOFIT_LIFECYCLE_REQUIRED", "true")
    monkeypatch.setattr(native_lifecycle, "load_endpoint", lambda _: {})
    monkeypatch.setattr(native_lifecycle, "lifecycle_request", lambda *a, **kw: status)
    assert all(m["metadata"]["residency"] == "unknown" for m in GATEWAY.provider_models())


def test_unreachable_controller_preserves_identity(catalogue, monkeypatch):
    monkeypatch.setenv("TURBOFIT_LIFECYCLE_REQUIRED", "1")
    def unavailable(_):
        raise native_lifecycle.LifecycleUnavailable("stale endpoint")
    monkeypatch.setattr(native_lifecycle, "load_endpoint", unavailable)
    assert GATEWAY.provider_models()[0]["metadata"]["backing_model"] == "Exact/Main:Q4"
    assert GATEWAY.provider_models()[0]["metadata"]["residency"] == "unknown"


@pytest.mark.parametrize("content", ['[]', '{', '{}', '{"active":"x","routes":[]}',
    '{"active":"x","routes":{"main":{"kind":"api-policy","alias":"not-physical"}}}'])
def test_missing_or_invalid_route_has_no_invented_identity(catalogue, monkeypatch, content):
    state, _ = catalogue
    state.write_text(content)
    # Existing context fallback is independent of metadata and can consult health.
    monkeypatch.setattr(GATEWAY, "active_context_length", lambda **kw: 65536)
    assert all(m["metadata"]["backing_model"] is None for m in GATEWAY.provider_models())


def test_http_catalogue_and_detail_observe_real_lifecycle_without_wake(catalogue, tmp_path, monkeypatch):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    lifecycle = native_lifecycle.IdleLifecycle(state_dir=tmp_path / "native", clock=lambda: 0, wall=lambda: 100)
    lifecycle.observe({"main": {"backing_model": "Exact/Main:Q4",
                                "context_length": 4096, "residency": "unknown"}})
    wakes = []
    endpoint = native_lifecycle.LifecycleEndpoint(lifecycle, ensure_ready=wakes.append)
    endpoint.start()
    monkeypatch.setenv("TURBOFIT_LIFECYCLE_REQUIRED", "1")
    monkeypatch.setenv("TURBOFIT_NATIVE_STATE", str(lifecycle.state_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), GATEWAY.GatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # A failed real wake must never be presented as intentional idle.
        def fail():
            raise RuntimeError("failed launch")
        with pytest.raises(RuntimeError):
            lifecycle.wake("main", fail)
        before = lifecycle.status()
        for path in ("/v1/models", "/v1/models/auto"):
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                data = json.loads(response.read())
                assert response.status == 200
                model = data["data"][0] if path == "/v1/models" else data
                assert model["id"] == "auto"
                assert model["metadata"]["residency"] == "error"
                assert model["metadata"]["backing_model"] == "Exact/Main:Q4"
            finally:
                connection.close()
        assert lifecycle.status() == before
        assert wakes == []
        assert not (lifecycle.state_dir / native_lifecycle.STATE_NAME).exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        endpoint.stop()
