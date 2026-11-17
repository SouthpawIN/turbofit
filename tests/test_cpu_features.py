from turbofit_runtime.cpu_features import (
    STRATA_ACCELERATOR, STRATA_REQUIRED, cpu_flags, missing_flags,
    strata_accelerated, strata_supported,
)


def test_avx2_is_the_gate_not_avx512(tmp_path):
    """Strata's docs require AVX2; AVX-512 is an accelerator only."""
    f = tmp_path / "cpuinfo"
    f.write_text("processor\t: 0\nflags\t\t: fpu avx avx2 sse4_2\n")
    flags = cpu_flags(f)
    assert flags == frozenset({"fpu", "avx", "avx2", "sse4_2"})
    assert missing_flags(STRATA_REQUIRED, flags) == ()
    assert strata_supported(flags) is True, "an AVX2-only CPU runs Strata"
    assert strata_accelerated(flags) is False, "no AVX-512: slower prefill, not a refusal"
    assert "avx512f" in missing_flags(STRATA_ACCELERATOR, flags)


def test_pre_avx2_cpu_is_deferred_with_the_flag_named(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("model name\t: Intel(R) Xeon(R) CPU E5-2620 v4\nflags\t\t: fpu avx sse4_2\n")
    flags = cpu_flags(f)
    assert strata_supported(flags) is False
    assert missing_flags(STRATA_REQUIRED, flags) == ("avx2",)


def test_avx512_is_reported_as_an_accelerator(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("flags\t\t: " + " ".join((*STRATA_REQUIRED, *STRATA_ACCELERATOR)) + "\n")
    flags = cpu_flags(f)
    assert strata_supported(flags) is True
    assert strata_accelerated(flags) is True


def test_missing_proc_is_unknown_not_unsupported(tmp_path):
    assert cpu_flags(tmp_path / "nope") is None
    assert strata_supported(None) is None      # macOS/Windows: do not block
    assert strata_accelerated(None) is None
    assert missing_flags(STRATA_REQUIRED, None) == ()
