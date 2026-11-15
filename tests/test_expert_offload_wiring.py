"""`--n-cpu-moe` comes from the model header and this host's free capacity.

These are pure-logic tests: probes are injected, so no GPU and no 70 GiB model
is needed to prove the decision.
"""
from __future__ import annotations

import pytest

from turbofit_runtime.experts import OffloadPlan, plan_expert_offload
from turbofit_runtime.gguf_footprint import ExpertFootprint
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.recipes import ExpertOffloadInfeasible, RecipeBook

GIB = 2**30


def footprint(total_gib: float, expert_gib: float, layers: int) -> ExpertFootprint:
    total = int(total_gib * GIB)
    expert = int(expert_gib * GIB)
    return ExpertFootprint(
        total_bytes=total, expert_bytes=expert, expert_layers=layers,
        dense_bytes=total - expert, tensor_count=layers * 3,
        expert_tensor_count=layers * 3,
        expert_tensor_names=tuple(f"blk.{i}.ffn_gate_exps.weight" for i in range(layers)),
        shards=("model.gguf",),
    )


def book(entry: dict, *, vram: int | None = 48 * GIB, ram: int | None = 512 * GIB,
         fp: ExpertFootprint | None = None, calls: dict | None = None,
         gpu: str = "0,1") -> RecipeBook:
    calls = calls if calls is not None else {}

    def vram_probe():
        calls["vram"] = calls.get("vram", 0) + 1
        return vram

    def ram_probe():
        calls["ram"] = calls.get("ram", 0) + 1
        return ram

    def footprint_probe(path: str):
        calls["footprint"] = calls.get("footprint", 0) + 1
        calls["path"] = path
        if fp is None:
            raise AssertionError("footprint probe should not be consulted")
        return fp

    data = {
        "schema_version": 1,
        "atomic_binary": "/bin/true",
        "models": {"Test MoE": {"alias": "test-moe", "kind": "process",
                                "model": "model.gguf", "port": 11605,
                                "methods": {"65536": "baseline"}, **entry}},
        "variants": {},
    }
    fingerprint = HardwareFingerprint(
        os="linux", architecture="x86_64", system_ram_mb=524288,
        devices=tuple(
            AcceleratorDevice(index=int(index), uuid=f"GPU-{index}", name="Test",
                              vendor="nvidia", backend="cuda", memory_total_mb=24576,
                              compute_capability="8.6", bus_id=None)
            for index in gpu.split(",") if index
        ),
    )
    return RecipeBook(data, backend_name="cuda", platform_name="linux",
                      hardware=fingerprint, vram_probe=vram_probe, ram_probe=ram_probe,
                      footprint_probe=footprint_probe)


def component(book_: RecipeBook):
    return book_.resolve_named("row", "Test MoE", "auto", 65_536).components[0]


def test_model_that_fits_gets_zero_offload_and_no_flag():
    small = footprint(20, 12, 24)
    comp = component(book({"auto_cpu_moe": True}, fp=small))

    assert comp.n_cpu_moe == 0
    assert "--n-cpu-moe" not in comp.command


def test_oversized_model_gets_the_planners_value():
    big = footprint(180, 150, 60)
    comp = component(book({"auto_cpu_moe": True}, fp=big))

    expected = plan_expert_offload(total_bytes=big.total_bytes, expert_bytes=big.expert_bytes,
                                   expert_layers=big.expert_layers, vram_bytes=48 * GIB,
                                   ram_bytes=512 * GIB)
    assert expected.feasible
    assert 0 < comp.n_cpu_moe < 60
    assert comp.n_cpu_moe == expected.n_cpu_moe
    assert comp.command[comp.command.index("--n-cpu-moe") + 1] == str(expected.n_cpu_moe)
    assert comp.offload_reason


def test_offload_scales_with_available_vram():
    big = footprint(180, 150, 60)
    entry = {"auto_cpu_moe": True}
    tight = component(book(entry, vram=40 * GIB, fp=big)).n_cpu_moe
    roomy = component(book(entry, vram=96 * GIB, fp=big)).n_cpu_moe

    assert tight > roomy


def test_infeasible_plan_raises_with_the_planners_reason():
    # 560 GiB of experts cannot live in 64 GiB of RAM.
    hopeless = footprint(600, 560, 60)
    with pytest.raises(ExpertOffloadInfeasible) as raised:
        component(book({"auto_cpu_moe": True}, ram=64 * GIB, fp=hopeless))

    assert "RAM" in str(raised.value)
    assert "GiB" in str(raised.value)
    assert raised.value.decision.plan.feasible is False


def test_non_expert_weights_too_big_for_vram_explains_itself():
    attention_heavy = footprint(80, 40, 40)
    with pytest.raises(ExpertOffloadInfeasible) as raised:
        component(book({"auto_cpu_moe": True}, vram=16 * GIB, ram=256 * GIB,
                       fp=attention_heavy))

    assert "non-expert" in str(raised.value)


def test_explicit_recipe_value_wins_and_skips_measurement():
    calls: dict = {}
    comp = component(book({"auto_cpu_moe": True, "n_cpu_moe": 5}, fp=None, calls=calls))

    assert comp.n_cpu_moe == 5
    assert comp.command[comp.command.index("--n-cpu-moe") + 1] == "5"
    assert calls == {}


def test_context_override_value_wins_over_auto():
    data = {
        "schema_version": 1, "atomic_binary": "/bin/true",
        "models": {"Test MoE": {"alias": "test-moe", "kind": "process",
                                "model": "model.gguf", "port": 11605,
                                "auto_cpu_moe": True,
                                "methods": {"65536": "baseline"},
                                "context_overrides": {"65536": {"n_cpu_moe": 7}}}},
        "variants": {},
    }
    comp = RecipeBook(data, backend_name="cuda", platform_name="linux",
                      vram_probe=lambda: 0, ram_probe=lambda: 0,
                      footprint_probe=lambda path: None).resolve_named(
        "row", "Test MoE", "auto", 65_536).components[0]
    assert comp.n_cpu_moe == 7


def test_strata_flavor_never_receives_llama_cpp_flags():
    calls: dict = {}
    comp = component(book({"auto_cpu_moe": True, "runtime_flavor": "strata",
                           "strata_pack": "/packs/x"}, fp=None, calls=calls))

    assert "--n-cpu-moe" not in comp.command
    assert "--cpu-moe" not in comp.command
    assert comp.n_cpu_moe is None
    assert calls == {}, "a non-llama engine must not even be measured"


def test_strata_flavor_rejects_an_explicit_llama_cpp_flag():
    with pytest.raises(ValueError, match="cannot take n_cpu_moe"):
        component(book({"runtime_flavor": "strata", "strata_pack": "/packs/x",
                        "n_cpu_moe": 4}))


def test_vllm_flavor_never_receives_llama_cpp_flags():
    data = {
        "schema_version": 1, "atomic_binary": "/bin/true",
        "models": {"Test MoE": {"alias": "test-moe", "kind": "process",
                                "model": "checkpoint-dir", "port": 11605,
                                "runtime_flavor": "vllm", "auto_cpu_moe": True,
                                "methods": {"65536": "baseline"}}},
        "variants": {},
    }
    comp = RecipeBook(data, backend_name="cuda", platform_name="linux",
                      vram_probe=lambda: 32 * GIB, ram_probe=lambda: 512 * GIB,
                      footprint_probe=lambda path: footprint(80, 60, 40)).resolve_named(
        "row", "Test MoE", "auto", 65_536).components[0]

    assert "--n-cpu-moe" not in comp.command
    assert comp.n_cpu_moe is None


def test_unavailable_capacity_skips_auto_without_breaking_resolution():
    comp = component(book({"auto_cpu_moe": True}, vram=None, fp=footprint(20, 12, 24)))

    assert comp.n_cpu_moe is None
    assert "--n-cpu-moe" not in comp.command


def test_cpu_backend_ignores_auto_offload():
    data = {
        "schema_version": 1, "atomic_binary": "/bin/true",
        "models": {"Test MoE": {"alias": "test-moe", "kind": "process",
                                "model": "model.gguf", "port": 11605,
                                "auto_cpu_moe": True,
                                "methods": {"65536": "baseline"}}},
        "variants": {},
    }
    comp = RecipeBook(data, backend_name="cpu", platform_name="linux",
                      vram_probe=lambda: 48 * GIB, ram_probe=lambda: 512 * GIB,
                      footprint_probe=lambda path: footprint(180, 150, 60)).resolve_named(
        "row", "Test MoE", "auto", 65_536).components[0]

    assert "--n-cpu-moe" not in comp.command


def test_auto_offload_reads_the_model_file_through_the_planner(tmp_path):
    """The seam is optional: with no probe injected the real header is read."""
    from test_gguf_footprint import write_gguf

    model = write_gguf(tmp_path / "model.gguf", [
        ("blk.0.attn_q.weight", 1 * GIB),
        ("blk.0.ffn_gate_exps.weight", 6 * GIB),
        ("blk.1.attn_q.weight", 1 * GIB),
        ("blk.1.ffn_gate_exps.weight", 6 * GIB),
    ], {"general.block_count": 2})
    data = {
        "schema_version": 1, "atomic_binary": "/bin/true",
        "models": {"Test MoE": {"alias": "test-moe", "kind": "process",
                                "model": str(model), "port": 11605,
                                "auto_cpu_moe": True,
                                "methods": {"65536": "baseline"}}},
        "variants": {},
    }
    comp = RecipeBook(data, backend_name="cuda", platform_name="linux",
                      vram_probe=lambda: 4 * GIB, ram_probe=lambda: 64 * GIB).resolve_named(
        "row", "Test MoE", "auto", 65_536).components[0]

    # 2 GiB dense + experts forced into 2.5 GiB of usable VRAM -> both layers.
    assert comp.n_cpu_moe == 2
    assert comp.command[comp.command.index("--n-cpu-moe") + 1] == "2"


def test_hardware_fingerprint_is_the_last_resort_vram_source():
    from turbofit_runtime.recipes import host_free_vram_bytes

    fingerprint = HardwareFingerprint(
        os="linux", architecture="x86_64", system_ram_mb=131072,
        devices=(AcceleratorDevice(index=0, uuid="GPU-0", name="Test", vendor="nvidia",
                                   backend="cuda", memory_total_mb=24576,
                                   compute_capability="8.6", bus_id=None),),
    )
    assert host_free_vram_bytes(hardware=fingerprint, probe=lambda: ()) == 24 * 1024 * 2**20
    assert host_free_vram_bytes(hardware=None, probe=lambda: ()) is None


def layered_footprint(total_gib, layer_expert_gib, layer_count, other_gib=0.1):
    from turbofit_runtime.gguf_footprint import LayerFootprint
    layers = tuple(LayerFootprint(index=i, expert_bytes=int(layer_expert_gib * GIB),
                                  other_bytes=int(other_gib * GIB))
                   for i in range(layer_count))
    expert = int(layer_expert_gib * layer_count * GIB)
    return ExpertFootprint(
        total_bytes=int(total_gib * GIB), expert_bytes=expert, expert_layers=layer_count,
        dense_bytes=int(total_gib * GIB) - expert, tensor_count=layer_count * 3,
        expert_tensor_count=layer_count * 3,
        expert_tensor_names=tuple(f"blk.{i}.ffn_gate_exps.weight" for i in range(layer_count)),
        shards=("model.gguf",), layers=layers,
    )


def test_multi_gpu_oversized_model_gets_a_balanced_layer_split():
    fp = layered_footprint(180, 2.5, 60)
    comp = component(book({"auto_cpu_moe": True, "large": True}, fp=fp))

    assert comp.n_cpu_moe > 0
    assert comp.gpu == "0,1"
    split = comp.command[comp.command.index("--tensor-split") + 1]
    assert sum(int(part) for part in split.split(",")) == 60


def test_single_gpu_never_gets_a_tensor_split():
    fp = layered_footprint(180, 2.5, 60)
    comp = component(book({"auto_cpu_moe": True}, fp=fp, gpu="0"))

    assert "--tensor-split" not in comp.command


def test_model_that_fits_gets_no_tensor_split():
    fp = layered_footprint(20, 0.5, 24)
    comp = component(book({"auto_cpu_moe": True}, fp=fp))

    assert comp.n_cpu_moe == 0
    assert "--tensor-split" not in comp.command
