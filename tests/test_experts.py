from turbofit_runtime.experts import OffloadPlan, available_ram_bytes, plan_expert_offload

GIB = 2**30


def test_small_moe_fits_entirely_in_vram():
    plan = plan_expert_offload(total_bytes=20 * GIB, expert_bytes=12 * GIB, expert_layers=24,
                               vram_bytes=48 * GIB, ram_bytes=128 * GIB)
    assert plan.feasible and plan.n_cpu_moe == 0


def test_oversized_moe_offloads_expert_layers():
    # 180 GB model on 48 GB of VRAM: most experts must live in RAM.
    plan = plan_expert_offload(total_bytes=180 * GIB, expert_bytes=150 * GIB, expert_layers=60,
                               vram_bytes=48 * GIB, ram_bytes=512 * GIB)
    assert plan.feasible
    assert 0 < plan.n_cpu_moe < 60
    assert plan.vram_bytes <= 48 * GIB


def test_infeasible_when_ram_cannot_hold_the_experts():
    plan = plan_expert_offload(total_bytes=600 * GIB, expert_bytes=560 * GIB, expert_layers=60,
                               vram_bytes=48 * GIB, ram_bytes=64 * GIB)
    assert plan.feasible is False
    assert "RAM" in plan.reason


def test_infeasible_when_non_expert_weights_exceed_vram():
    plan = plan_expert_offload(total_bytes=80 * GIB, expert_bytes=40 * GIB, expert_layers=40,
                               vram_bytes=16 * GIB, ram_bytes=256 * GIB)
    assert plan.feasible is False
    assert "non-expert" in plan.reason


def test_unmeasurable_ram_is_not_reported_as_feasible(tmp_path):
    assert available_ram_bytes(tmp_path / "nope") is None
    plan = plan_expert_offload(total_bytes=180 * GIB, expert_bytes=150 * GIB, expert_layers=60,
                               vram_bytes=48 * GIB, ram_bytes=None)
    assert plan.feasible is False and "could not be measured" in plan.reason


def test_reads_memavailable(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:       1024000 kB\nMemAvailable:    512000 kB\n")
    assert available_ram_bytes(f) == 512000 * 1024
