"""Quantization policy: a Q4-class *ceiling*, not a target.

The product rule is a ceiling, not a floor: nothing above the Q4 class is a
default answer.  Low quants are first-class, not fallbacks — the GSQ-RCO repos
ship genuinely good Q3 and Q2 builds, and IQ2/IQ3 rungs are frequently the
right answer for the memory band.  This module never ranks, scores, or warns on
quant width, and nothing here rewards a wider build.

Three classes, one classifier, one place to change it::

    class               examples                               default?
    --------------------------------------------------------------------------
    allowed             IQ2_XXS, Q3_K_S, Q4_K_M, UD-Q4_K_XL,    yes
                        TQ2_0, PTQ1_0, PQ2_0, native sub-24GB
    allowed-exception   Q8_0, F16, BF16, FP16, FP32              yes, with a
                                                                 visible reason
    excluded-by-default Q5_K_M, UD-Q5_K_XL, Q6_K, ...            no — only with a
                                                                 recorded
                                                                 justification

Exceptions are Q8-class and full precision.  They are legitimate when the model
is small enough to live fully in the tier's memory (reason
``fits-fully-in-vram``) or when the point of the entry is accuracy (reason
``accuracy-tier``, carried by the catalog's ``accuracy_tier`` flag).  Every
surface that offers an exception must print why.

An excluded entry is admitted only when it carries an explicit justification —
``quantization_justification`` on the catalog entry — and that text is echoed in
the output.  Nothing is inferred, nothing is silently upgraded.

Classification reads the *first* quant token in the label; mixed builds such as
``Maya-M (IQ2_S routed / Q6_K attention)`` are classified by their routing quant
(IQ2_S, allowed), not by a component quant.  Labels without a GGUF-style token
(``2.788 bpw``, ``native-24GB``) describe sub-24-GB native builds and are
allowed by construction; ``token`` is None so they stay auditable.

Selection is driven by fit (memory, the 64K context floor, measured speed and
intelligence) — never by quant rank.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

ALLOWED = "allowed"
ALLOWED_EXCEPTION = "allowed-exception"
EXCLUDED_BY_DEFAULT = "excluded-by-default"

# The ceiling. Everything at or below this width class is allowed, including
# 1/2/3-bit builds; the ceiling does not privilege 4-bit.
QUANT_CEILING_BITS = 4
# 8-bit class is the only *intermediate* width that is a legitimate exception;
# 5/6/7-bit builds are exactly what the policy rules out by default.
EXCEPTION_BITS = frozenset({8})
FULL_PRECISION_TOKENS = frozenset({"F16", "BF16", "FP16", "F32", "FP32"})

REASON_FITS_FULLY_IN_VRAM = "fits-fully-in-vram"
REASON_ACCURACY_TIER = "accuracy-tier"
REASON_JUSTIFIED = "justified"

_QUANT_TOKEN = re.compile(
    r"(?<![A-Z0-9])("
    r"I?Q\d+(?:_[A-Z0-9]+)*"      # Q4_K_M, UD-Q4_K_XL, IQ3_S, Q8_0
    r"|BF16|FP16|F16|F32|FP32"    # full precision
    r"|TQ\d+(?:_[A-Z0-9]+)*"      # ternary TQ2_0
    r"|PTQ\d+(?:_[A-Z0-9]+)*"     # PTQ1_0
    r"|PQ\d+(?:_[A-Z0-9]+)*"      # PQ2_0
    r")"
)
_BITS = re.compile(r"^(?:I?Q|TQ|PTQ|PQ)(\d+)")


@dataclass(frozen=True)
class QuantClassification:
    """One label's policy verdict, plus the evidence for it."""

    label: str
    token: str | None
    bits: int | None
    quant_class: str
    reason: str

    @property
    def is_allowed(self) -> bool:
        return self.quant_class == ALLOWED

    @property
    def is_exception(self) -> bool:
        return self.quant_class == ALLOWED_EXCEPTION

    @property
    def is_excluded(self) -> bool:
        return self.quant_class == EXCLUDED_BY_DEFAULT


def quant_token(label: str) -> str | None:
    """First quant token in a label, upper-cased, or None for native builds."""
    match = _QUANT_TOKEN.search(str(label or "").upper())
    return match.group(1) if match else None


def classify(label: str) -> QuantClassification:
    """Classify one catalog quantization label against the Q4 ceiling."""
    text = str(label or "")
    token = quant_token(text)
    if token is None:
        return QuantClassification(
            label=text,
            token=None,
            bits=None,
            quant_class=ALLOWED,
            reason="no GGUF quant token: native/sub-24GB build, at or below the ceiling",
        )
    if token in FULL_PRECISION_TOKENS:
        return QuantClassification(
            label=text,
            token=token,
            bits=None,
            quant_class=ALLOWED_EXCEPTION,
            reason="full precision is an allowed accuracy exception",
        )
    bits_match = _BITS.match(token)
    bits = int(bits_match.group(1)) if bits_match else None
    if bits is None:
        return QuantClassification(
            label=text, token=token, bits=None, quant_class=EXCLUDED_BY_DEFAULT,
            reason=f"unreadable width for quant token {token}",
        )
    if bits <= QUANT_CEILING_BITS:
        return QuantClassification(
            label=text, token=token, bits=bits, quant_class=ALLOWED,
            reason=f"{bits}-bit class is at or below the Q{QUANT_CEILING_BITS} ceiling",
        )
    if bits in EXCEPTION_BITS:
        return QuantClassification(
            label=text, token=token, bits=bits, quant_class=ALLOWED_EXCEPTION,
            reason="Q8-class is an allowed accuracy exception",
        )
    return QuantClassification(
        label=text, token=token, bits=bits, quant_class=EXCLUDED_BY_DEFAULT,
        reason=(
            f"{bits}-bit class is above the Q{QUANT_CEILING_BITS} ceiling and is "
            "not a Q8/full-precision exception"
        ),
    )


def _field(entry: Any, name: str) -> Any:
    """Read a catalog field from a mapping or a ModelVariant-like object."""
    if isinstance(entry, Mapping):
        return entry.get(name)
    return getattr(entry, name, None)


def _bits_for(token: str) -> int | None:
    match = _BITS.match(token)
    return int(match.group(1)) if match else None


def _token_is_excluded(token: str) -> bool:
    if token in FULL_PRECISION_TOKENS:
        return False
    bits = _bits_for(token)
    return bits is not None and bits > QUANT_CEILING_BITS and bits not in EXCEPTION_BITS


def entry_quantization(entry: Any) -> str:
    return str(_field(entry, "quantization") or "")


def classify_entry(entry: Any) -> QuantClassification:
    """Classify a whole catalog entry, not just its ``quantization`` label.

    The declared ``quantization`` is authoritative, but a catalog id, display
    name, or artifact that *names* an above-ceiling build (for example an
    ``...-ud-q5-k-xl`` alias pointing at a Q4 artifact) is still an above-ceiling
    entry for policy purposes.  This only ever escalates a verdict, never softens
    one, so an alias cannot smuggle a Q5 past the ceiling.
    """
    declared = classify(entry_quantization(entry))
    if declared.is_excluded:
        return declared
    for field in ("id", "name", "artifact"):
        token = quant_token(str(_field(entry, field) or ""))
        if token and _token_is_excluded(token):
            return QuantClassification(
                label=declared.label,
                token=token,
                bits=_bits_for(token),
                quant_class=EXCLUDED_BY_DEFAULT,
                reason=(
                    f"{field} names above-ceiling {token}; the declared "
                    f"{declared.label or 'label'} does not relax the ceiling"
                ),
            )
    return declared


def entry_justification(entry: Any) -> str | None:
    """The catalog's recorded justification for an above-ceiling quant."""
    value = _field(entry, "quantization_justification")
    text = str(value).strip() if value is not None else ""
    return text or None


def entry_accuracy_tier(entry: Any) -> bool:
    return bool(_field(entry, "accuracy_tier"))


def exception_reason(
    entry: Any,
    *,
    fits_fully_in_vram: bool = False,
    accuracy_point: bool = False,
) -> str | None:
    """Why an allowed-exception entry is being offered, or None.

    A Q8/full-precision entry is offered only when it fully fits the memory
    band (no offload) or the catalog marks it as an accuracy-tier entry.  The
    recorded justification is appended so the output carries the actual reason,
    not just a class label.
    """
    verdict = classify_entry(entry)
    if not verdict.is_exception:
        return None
    recorded = entry_justification(entry)
    if fits_fully_in_vram:
        detail = f"weights + KV fit fully in the tier's memory ({verdict.token})"
        return f"{REASON_FITS_FULLY_IN_VRAM}: {detail}" + (f"; {recorded}" if recorded else "")
    if entry_accuracy_tier(entry) or accuracy_point:
        detail = (
            f"catalog marks this entry accuracy-tier ({verdict.token})"
            if entry_accuracy_tier(entry)
            else f"{verdict.token} accuracy build measured on purpose"
        )
        return f"{REASON_ACCURACY_TIER}: {detail}" + (f"; {recorded}" if recorded else "")
    if recorded:
        return f"{REASON_JUSTIFIED}: {recorded}"
    return None


def default_admits(
    entry: Any,
    *,
    fits_fully_in_vram: bool = False,
    accuracy_point: bool = False,
) -> tuple[bool, str]:
    """Whether an entry may be selected by default, and the visible reason.

    Returns ``(True, reason)`` or ``(False, reason)``.  The reason is never
    empty, so every surface can explain a skip as easily as an inclusion.
    """
    verdict = classify_entry(entry)
    if verdict.is_allowed:
        return True, verdict.reason
    if verdict.is_exception:
        reason = exception_reason(
            entry, fits_fully_in_vram=fits_fully_in_vram, accuracy_point=accuracy_point,
        )
        if reason is not None:
            return True, reason
        return False, (
            f"{verdict.token} offered without a fitting-band or accuracy justification"
        )
    recorded = entry_justification(entry)
    if recorded:
        return True, f"{REASON_JUSTIFIED}: {recorded}"
    return False, (
        f"{verdict.token} is above the Q{QUANT_CEILING_BITS} ceiling "
        "(quantization_justification required)"
    )


def policy_table() -> list[dict[str, str]]:
    """The documented classification table, for docs and PR descriptions."""
    return [
        {
            "class": ALLOWED,
            "rule": (
                f"at or below Q{QUANT_CEILING_BITS} class — includes Q1/Q2/Q3/IQ2/IQ3/IQ4, "
                "UD-/AD-/GSQ-RCO variants, IQ4_XS, TQ/PTQ/PQ and native sub-24GB builds"
            ),
            "default": "yes — low quants are first-class, not fallbacks",
        },
        {
            "class": ALLOWED_EXCEPTION,
            "rule": "Q8_0-class and F16/BF16/FP16/FP32",
            "default": "yes, with a printed reason (fits fully in memory, or accuracy-tier)",
        },
        {
            "class": EXCLUDED_BY_DEFAULT,
            "rule": "Q5/Q6/Q7 class and equivalents",
            "default": "no, unless quantization_justification is recorded",
        },
    ]
