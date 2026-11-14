from turbofit_runtime.cpu_features import (
    STRATA_REQUIRED, cpu_flags, missing_flags, strata_supported,
)


def test_reads_flags_from_cpuinfo(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("processor\t: 0\nflags\t\t: fpu avx2 avx512f avx512_vnni\n")
    assert cpu_flags(f) == frozenset({"fpu", "avx2", "avx512f", "avx512_vnni"})
    assert missing_flags(STRATA_REQUIRED, cpu_flags(f)) != ()
    assert strata_supported(cpu_flags(f)) is False


def test_full_support_detected(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("flags\t\t: " + " ".join(STRATA_REQUIRED) + "\n")
    assert strata_supported(cpu_flags(f)) is True


def test_missing_proc_is_unknown_not_unsupported(tmp_path):
    assert cpu_flags(tmp_path / "nope") is None
    assert strata_supported(None) is None      # macOS/Windows: do not block
    assert missing_flags(STRATA_REQUIRED, None) == ()


def test_broadwell_has_no_avx512(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("model name\t: Intel(R) Xeon(R) CPU E5-2620 v4\nflags\t\t: fpu avx avx2 sse4_2\n")
    flags = cpu_flags(f)
    assert strata_supported(flags) is False
    assert "avx512_vnni" in missing_flags(STRATA_REQUIRED, flags)
