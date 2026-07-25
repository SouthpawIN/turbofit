from __future__ import annotations

from dataclasses import dataclass, field

from turbofit_runtime.reconciler import ReconcilerState, transition
from turbofit_runtime.runtime_profile import Turbofile

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64


def adaptive_profile() -> Turbofile:
    data = {
        "schema": "turbofit.runtime/v1", "id": "adaptive-48gb", "revision": 1,
        "hardware": {"class_vram_gb": 48, "min_devices": 2, "total_vram_gb": 48, "per_device_min_gb": 24, "accelerator": "nvidia-cuda", "topology": "2x24"},
        "policy": {"recommendation": "quality-first", "external_gpu_priority": "absolute", "contraction_dwell_s": 0, "expansion_dwell_s": 0, "expansion_margin_gb_per_card": 0, "cooldown_s": 0},
        "roles": {"main": "active:main", "auxiliary": "active:aux", "fallback": "api:auto"},
    }
    data["rungs"] = [
        {"id": "dedicated", "context": 262144, "aux_mode": "dedicated", "evidence": DIGEST_A, "main_manifest": DIGEST_A, "aux_manifest": DIGEST_B},
        {"id": "shared", "context": 262144, "aux_mode": "shared-main", "evidence": DIGEST_A, "main_manifest": DIGEST_A},
        {"id": "contracted", "context": 131072, "aux_mode": "shared-main", "evidence": DIGEST_B, "main_manifest": DIGEST_C},
        {"id": "api", "context": 131072, "aux_mode": "api", "evidence": DIGEST_C, "main_api_policy": "api:auto", "aux_api_policy": "api:auto"},
    ]
    return Turbofile.from_mapping(data)


@dataclass
class Backend:
    events: list[object] = field(default_factory=list)
    unrelated_pid: int = 99999
    current_index: int = 0
    restore_ok: bool = True

    def block_aux_admission(self): self.events.append("block")
    def drain_aux(self, timeout_s): self.events.append(("drain", timeout_s)); return True
    def clean_unload_aux(self): self.events.append("unload-aux"); return True
    def owned_pids(self): return (101, 102)
    def escalate_owned(self, pids): self.events.append(("escalate", tuple(pids)))
    def activate_local(self, rung_id: str): self.events.append(("activate-local", rung_id))
    def activate_api(self, main, aux): self.events.append(("activate-api", main, aux))
    def route_aux_to_main(self): self.events.append("route-shared")
    def route_aux_dedicated(self): self.events.append("route-dedicated")
    def verify_rung(self, rung_id: str) -> bool: self.events.append(("verify", rung_id)); return True
    def publish_routes(self, state): self.events.append(("publish", state.rung_index)); self.current_index = state.rung_index
    def restore(self, state): self.events.append(("restore", state.rung_index)); self.current_index = state.rung_index
    def verify_restore(self, state): return self.restore_ok


def test_full_contraction_and_one_rung_self_healing_never_targets_external_pid() -> None:
    profile = adaptive_profile()
    backend = Backend()
    state = ReconcilerState(profile.id, 0, "main", "aux")

    for target in (1, 2, 3):
        state = transition(state, target, profile, backend)
    assert state.rung_index == 3
    assert "unload-aux" in backend.events
    assert "route-shared" in backend.events
    assert ("activate-local", "contracted") in backend.events
    assert ("activate-api", "api:auto", "api:auto") in backend.events

    for target in (2, 1, 0):
        state = transition(state, target, profile, backend)
    assert state.rung_index == 0
    assert backend.unrelated_pid not in {
        pid for event in backend.events if isinstance(event, tuple) and event[0] == "escalate" for pid in event[1]
    }
    publishes = [event[1] for event in backend.events if isinstance(event, tuple) and event[0] == "publish"]
    assert publishes == [1, 2, 3, 2, 1, 0]


def test_failed_activation_rolls_back_without_publishing_target() -> None:
    profile = adaptive_profile()

    class Failing(Backend):
        def verify_rung(self, rung_id: str) -> bool:
            self.events.append(("verify", rung_id))
            return False

    backend = Failing()
    state = ReconcilerState(profile.id, 1, "main", "main")
    try:
        transition(state, 2, profile, backend)
    except Exception:
        pass
    assert ("restore", 1) in backend.events
    assert ("publish", 2) not in backend.events
