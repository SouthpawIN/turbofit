"""Opt-in controller-owned residency: bounded leases, atomic idle release and IPC.

No process signalling lives here. The existing native backend remains the only
execution authority. A controller crash with outstanding leases fails closed;
active requests must be reconciled by the operator before reusing that state.
"""
from __future__ import annotations

from contextlib import contextmanager
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
from typing import Callable

ENDPOINT_NAME = "lifecycle-endpoint.json"
STATE_NAME = "lifecycle-state.json"
MAX_BODY_BYTES = 65536
# Leases older than this are treated as abandoned by a dead holder instead of
# orphaning every future owner. Recent leases still fail closed: the holder may
# be alive and its residency must be reconciled by the operator.
STALE_LEASE_WALL_S = 24 * 3600


def _lease_parts(value):
    """Split a persisted lease entry into (role, acquired_wall|None).

    Current entries are {"role": ..., "acquired_wall": ...}; entries written
    before acquisition stamps existed are bare role strings and are treated
    as fresh so old crashes still fail closed.
    """
    if isinstance(value, dict):
        return value.get("role"), value.get("acquired_wall")
    return value, None


class LifecycleUnavailable(RuntimeError):
    pass


class LifecycleError(RuntimeError):
    pass


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8))
    try:
        with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
            json.dump(data, handle)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class IdleLifecycle:
    def __init__(self, *, state_dir, idle_timeout_s=120.0, wake_timeout_s=900.0,
                 clock=time.monotonic, wall=time.time):
        if not math.isfinite(idle_timeout_s) or idle_timeout_s <= 0:
            raise ValueError("idle timeout must be finite and positive")
        if not math.isfinite(wake_timeout_s) or wake_timeout_s <= 0:
            raise ValueError("wake timeout must be finite and positive")
        self.state_dir = Path(state_dir)
        self.roles = ("main", "aux")
        self.idle_timeout_s, self.wake_timeout_s = idle_timeout_s, wake_timeout_s
        self.clock, self.wall = clock, wall
        self._lock = threading.RLock()
        self._leases = {}
        self._last_activity = dict.fromkeys(self.roles, clock())
        self._wake_gates = {}
        self._wake_retry_after = {}
        self._observations = {}
        self._lock_file = None
        self._orphaned = False
        try:
            data = json.loads((self.state_dir / STATE_NAME).read_text())
            if not isinstance(data, dict) or data.get("schema") != 2:
                raise ValueError("unrecognised lifecycle state")
            self._orphaned = self._has_live_leases(data.get("leases"), wall())
            for role, stamp in data["last_activity_wall"].items():
                if role in self.roles and type(stamp) in (int, float) and math.isfinite(stamp):
                    self._last_activity[role] = clock() - max(0.0, wall() - stamp)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, KeyError):
            # Do not reclaim potentially busy residents based on corrupt history.
            self._orphaned = True

    @staticmethod
    def _has_live_leases(leases, now_wall):
        if not isinstance(leases, dict):
            return False
        for value in leases.values():
            _, acquired = _lease_parts(value)
            if not isinstance(acquired, (int, float)) or not math.isfinite(acquired):
                return True
            if now_wall - acquired <= STALE_LEASE_WALL_S:
                return True
        return False

    def _persist(self):
        _atomic_write(self.state_dir / STATE_NAME, {
            "schema": 2,
            "leases": self._leases,
            "last_activity_wall": {r: self.wall() - max(0, self.clock() - t)
                                   for r, t in self._last_activity.items()},
        })

    def acquire_singleton(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        handle = (self.state_dir / "lifecycle.lock").open("a+")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                handle.write("0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise LifecycleError("another lifecycle owner holds the state directory") from exc
        self._lock_file = handle

    def release_singleton(self):
        if self._lock_file:
            self._lock_file.close()
            self._lock_file = None

    def acquire(self, role):
        with self._lock:
            if self._orphaned:
                raise LifecycleError("unsettled leases or corrupt state require operator reconciliation")
            if role not in self.roles or len(self._leases) >= 256:
                raise LifecycleError("invalid role or admission limit reached")
            token = secrets.token_hex(24)
            self._leases[token] = {"role": role, "acquired_wall": self.wall()}
            self._last_activity[role] = self.clock()
            self._persist()
            return token

    def release(self, token):
        with self._lock:
            role, _ = _lease_parts(self._leases.pop(token, None))
            if role:
                self._last_activity[role] = self.clock()
                self._persist()

    def lease_count(self, role):
        with self._lock:
            return sum(_lease_parts(value)[0] == role for value in self._leases.values())

    def wake(self, role, start: Callable[[], None]):
        with self._lock:
            if role not in self.roles:
                raise LifecycleError("invalid role")
            flight = self._wake_gates.get(role)
            leader = flight is None
            if leader:
                if self.clock() < self._wake_retry_after.get(role, 0):
                    raise LifecycleError("recent wake failed; retry after cooldown")
                flight = {"event": threading.Event(), "error": None}
                self._wake_gates[role] = flight
                self._record_state(role, "loading")
        if not leader:
            # Never wait holding the lock needed by the leader to finish.
            if not flight["event"].wait(self.wake_timeout_s):
                raise LifecycleError("wake timed out")
            if flight["error"]:
                raise LifecycleError("coalesced wake failed") from flight["error"]
            return
        try:
            start()  # Native backend bounds health verification and cleans failed starts.
            with self._lock:
                self._last_activity[role] = self.clock()
                self._persist()
                self._record_state(role, "ready")
        except Exception as exc:
            flight["error"] = exc
            with self._lock:
                self._wake_retry_after[role] = self.clock() + 5.0
                self._record_state(role, "error")
            raise
        finally:
            with self._lock:
                self._wake_gates.pop(role, None)
                flight["event"].set()

    def idle_roles(self):
        with self._lock:
            if self._orphaned:
                return ()
            return tuple(r for r in self.roles if not self.lease_count(r)
                         and r not in self._wake_gates
                         and self.clock() - self._last_activity[r] >= self.idle_timeout_s)

    def release_idle(self, stop):
        with self._lock:
            for role in self.idle_roles():
                try:
                    outcome = stop(role)
                    if outcome is False or outcome == "error":
                        raise LifecycleError(f"could not release idle {role}")
                except Exception:
                    self._record_state(role, "error")
                    raise
                # Cleanup after failure is not a successful idle transition.
                previous = self._observations.get(role, {}).get("residency")
                if previous != "error" and (outcome == "idle" or (outcome is True and previous in {"ready", "idle"})):
                    self._record_state(role, "idle")

    @contextmanager
    def controller_tick(self):
        # Selection changes, pressure transitions and idle release must not race
        # gateway admission or an in-flight wake. Requests do not hold this lock
        # during inference: the persisted lease protects their entire lifetime.
        with self._lock:
            yield not (self._leases or self._wake_gates or self._orphaned)

    def _record_state(self, role, state):
        previous = self._observations.get(role, {})
        self._observations[role] = dict(previous, residency=state,
                                       observed_at=self.wall(), _monotonic=self.clock())

    def observe(self, snapshots):
        """Publish owner observations, never restoring telemetry from disk.

        Readers cannot refresh age; restart requires new ownership verification.
        An absent process only preserves idle/error for the same bound recipe.
        """
        with self._lock:
            for role in self.roles:
                if role in self._wake_gates:
                    self._record_state(role, "loading")
                    continue
                item = snapshots.get(role)
                if not item:
                    self._observations.pop(role, None)
                    continue
                old = self._observations.get(role, {})
                state = item["residency"]
                same = all(old.get(k) == item.get(k) for k in ("backing_model", "context_length"))
                if same and state == "unknown":
                    if old.get("residency") in {"idle", "error"}:
                        state = old["residency"]
                    elif old.get("residency") == "ready":
                        state = "error"
                self._observations[role] = dict(item)
                self._record_state(role, state)

    def status(self):
        with self._lock:
            roles = {}
            for role in self.roles:
                item = dict(self._observations.get(role, {}))
                stamp = item.pop("_monotonic", None)
                age = max(0.0, self.clock() - stamp) if stamp is not None else None
                stale = age is None or age > 15.0 or self._orphaned
                item.update(leases=self.lease_count(role), waking=role in self._wake_gates,
                            freshness={"age_s": age, "max_age_s": 15.0, "stale": stale})
                if stale:
                    item["residency"] = "unknown"
                roles[role] = item
            return {"orphaned": self._orphaned, "roles": roles}


class LifecycleEndpoint:
    def __init__(self, lifecycle, *, ensure_ready=lambda role: None,
                 resolve_role=lambda role: role, host="127.0.0.1"):
        if host != "127.0.0.1":
            raise ValueError("lifecycle endpoint must bind loopback")
        self.lifecycle, self.ensure_ready, self.resolve_role = lifecycle, ensure_ready, resolve_role
        self.token = secrets.token_hex(32)
        self._server = self._thread = None

    @property
    def port(self):
        if self._server is None:
            raise LifecycleError("endpoint not started")
        return self._server.server_address[1]

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                self.connection.settimeout(5)
                lease = None
                try:
                    if self.path != "/lifecycle" or not hmac.compare_digest(
                        self.headers.get("Authorization", ""), "Bearer " + owner.token
                    ):
                        self.respond(403, {"error": "unauthorised"})
                        return
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= MAX_BODY_BYTES:
                        raise ValueError("invalid body size")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("invalid payload")
                    action = payload.get("action")
                    if action == "acquire":
                        requested = payload.get("role")
                        if requested not in owner.lifecycle.roles:
                            raise ValueError("invalid role")
                        with owner.lifecycle._lock:
                            role = owner.resolve_role(requested)
                            lease = owner.lifecycle.acquire(role) if role else None
                        if role:
                            owner.lifecycle.wake(role, lambda: owner.ensure_ready(role))
                        self.respond(200, {"token": lease, "role": role})
                        lease = None  # Transfer ownership to the requesting gateway.
                    elif action == "release":
                        token = payload.get("token")
                        if not isinstance(token, str):
                            raise ValueError("invalid lease token")
                        owner.lifecycle.release(token)
                        self.respond(200, {"ok": True})
                    elif action == "status":
                        self.respond(200, owner.lifecycle.status())
                    else:
                        raise ValueError("invalid action")
                except Exception as exc:
                    if lease:
                        owner.lifecycle.release(lease)
                        lease = None
                    try:
                        status = 400 if isinstance(exc, (ValueError, TypeError)) else 503
                        self.respond(status, {"error": "lifecycle_unavailable"})
                    except OSError:
                        pass
                finally:
                    if lease:
                        owner.lifecycle.release(lease)

            def respond(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        _atomic_write(self.lifecycle.state_dir / ENDPOINT_NAME,
                      {"host": "127.0.0.1", "port": self.port, "token": self.token})
        return self.port

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._server = None
        (self.lifecycle.state_dir / ENDPOINT_NAME).unlink(missing_ok=True)


def load_endpoint(state_dir):
    try:
        data = json.loads((Path(state_dir) / ENDPOINT_NAME).read_text())
        if (data["host"] != "127.0.0.1" or type(data["port"]) is not int
                or not 0 < data["port"] < 65536 or not isinstance(data["token"], str)
                or len(data["token"]) != 64):
            raise ValueError("invalid endpoint")
        return data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LifecycleUnavailable("configured lifecycle endpoint unavailable") from exc


def lifecycle_request(endpoint, payload, timeout=905.0):
    request = urllib.request.Request(
        f"http://127.0.0.1:{endpoint['port']}/lifecycle", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + endpoint["token"]},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read(MAX_BODY_BYTES + 1))
            if not isinstance(data, dict):
                raise ValueError("invalid lifecycle response")
            return data
    except (OSError, ValueError) as exc:
        raise LifecycleUnavailable("lifecycle request refused or unavailable") from exc


def leased_request(role):
    """Hold the real gateway request, including SSE, inside a controller lease."""
    from functools import wraps
    import logging

    def decorate(handler):
        @wraps(handler)
        def wrapped(self, *args, **kwargs):
            if os.getenv("TURBOFIT_LIFECYCLE_REQUIRED", "0").lower() not in {"1", "true", "yes"}:
                return handler(self, *args, **kwargs)
            try:
                endpoint = load_endpoint(os.environ.get(
                    "TURBOFIT_NATIVE_STATE", Path.home() / ".local/state/turbofit/native"))
                lease = lifecycle_request(endpoint, {"action": "acquire", "role": role})
            except LifecycleUnavailable:
                self._send_503("Local residency controller unavailable", tried=None)
                return
            try:
                return handler(self, *args, **kwargs)
            finally:
                if lease.get("token"):
                    try:
                        lifecycle_request(endpoint, {"action": "release", "token": lease["token"]}, timeout=5)
                    except LifecycleUnavailable:
                        logging.getLogger(__name__).error("residency lease release failed; owner must reconcile")
        return wrapped
    return decorate
