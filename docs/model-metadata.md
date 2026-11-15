# Optional model catalogue metadata

`GET /v1/models` and `GET /v1/models/{id}` retain raw request IDs
`auto`, `active:main`, `active:aux`, descriptions and per-role context lengths.
Clients must not persist presentation labels instead of those IDs.

```json
{
  "id": "active:aux",
  "metadata": {
    "role": "aux",
    "backing_model": "selected-main-alias",
    "mode": "shared-main",
    "residency": "idle",
    "observed_at": 1788861600.0,
    "freshness": {"age_s": 0.3, "max_age_s": 15.0, "stale": false}
  }
}
```

The example is illustrative, not a live observation.

- `role` is routing identity. Auto follows main; shared-main Aux retains role
  aux but follows main's backing and residency.
- `backing_model` preserves the exact published local alias even when unloaded,
  stale or the controller is unavailable. Missing/malformed routes and unresolved
  API policies yield null. This alias is not a file hash or proof of residency.
- `mode` is the published route kind (`local`, `shared-main`, `api-policy`) or null.
- `ready` means the owner verified process identity, exact selected command,
  alias and endpoint, plus health and the served model alias. This is a bounded
  observation, not a guarantee that a process cannot subsequently fail.
- `loading` means an owner-controlled wake is in flight. Controller ticks renew
  that observation; metadata reads do not. No global lifecycle lock is held over
  the potentially long native wake. A stalled owner heartbeat becomes unknown.
- `idle` means a verified owned process was released intentionally. No-op cleanup,
  missing processes, ownership loss and failed wake cleanup cannot establish idle.
- `error` means verified health/identity failed, an owned resident disappeared,
  or wake/unload failed. No exception text or secrets are exported. Error survives
  backend reconstruction; a successful exact verification clears it.
- `unknown` covers missing, stale, malformed, orphaned, unbound or unavailable
  observations. Restart does not reload ready/idle telemetry from persisted leases.

Owner observations are bound to the published alias and exact integer context.
The owner measures `age_s` using its monotonic clock and supplies `observed_at`
as a Unix timestamp. `max_age_s` is 15 seconds. Catalogue reads preserve this age;
clients add their own monotonic time since receipt and must not rejuvenate an old
observation on reopen. Invalid numbers, booleans, NaN, infinity and oversized
integers fail closed. A stale observation keeps backing identity but is unknown.
A long wake stays loading only while controller ticks confirm its in-flight
record. Controller lock contention or failed/stalled ticks can cause unknown.

Metadata makes one authenticated read-only lifecycle status request per response,
with a 0.5-second timeout when `TURBOFIT_LIFECYCLE_REQUIRED` is enabled. Status does
not acquire leases, launch/stop/repair models, persist state, refresh observation
age, or change routes. Periodic owner ticks—not metadata requests—check native
health. Existing context-length fallback probes remain read-only and unchanged.

## Verification boundary

`tests/test_model_metadata.py` covers catalogue/detail HTTP and raw identity.
`tests/test_residency_telemetry.py` exercises lifecycle state transitions, expiry,
malformed freshness, actual native child health/failure/unload, real service
synchronization/reconstruction, exact recipe/endpoint rejection and idle races.
Child servers are synthetic HTTP fixtures, not models or GPU inference.
Physical residency, deployment and a complete release-suite pass are separate
acceptance gates. Clients must not disable selection solely on idle/unknown.
