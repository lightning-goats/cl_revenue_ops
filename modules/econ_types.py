"""Checked economic domain types (refactor Phase 1, Workstream J2).

Normative rules: docs/refactor/phase0/wire-contract-spec.md.

- Money is integer millisatoshi. Unsigned quantities live in
  [0, 2**63 - 1] (checked u64 in a future Rust port); signed P&L/deltas
  in [-(2**63), 2**63 - 1].
- Every bound violation, overflow, or invalid unit operation raises
  EconArithmeticError — authorization-relevant arithmetic fails closed,
  never wraps or coerces.
- Explicit methods only (add/sub/diff/...): incompatible units cannot be
  mixed silently with plain ints or each other.
- msat->sat conversions are direction-specific: CEIL for fees, budgets,
  costs, and revenue reporting; FLOOR for capacity and balances;
  TOWARD-ZERO for signed deltas (mirrors modules/utils.py rules).

These types are additive Phase 1 foundations: nothing in the live plugin
imports them yet.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

U63_MAX = 2**63 - 1
I64_MIN = -(2**63)

_CHANNEL_ID_RE = re.compile(r"^[0-9]+x[0-9]+x[0-9]+$")
_PEER_ID_RE = re.compile(r"^0[23][0-9a-f]{64}$")
_INTENT_ID_RE = re.compile(r"^[a-z0-9-]{1,64}$")


class EconArithmeticError(ArithmeticError):
    """A checked domain-type operation failed. Callers must treat this as
    a hard stop for the affected decision (reason code
    ARITHMETIC_OVERFLOW / SCHEMA_INVALID), never as zero."""


def _check_int(value, low, high, kind):
    if isinstance(value, bool) or not isinstance(value, int):
        raise EconArithmeticError(f"{kind} requires int, got {value!r}")
    if not (low <= value <= high):
        raise EconArithmeticError(
            f"{kind} out of range [{low}, {high}]: {value!r}")
    return value


@dataclass(frozen=True, order=True)
class Msat:
    """Unsigned millisatoshi amount."""

    value: int

    def __post_init__(self):
        _check_int(self.value, 0, U63_MAX, "Msat")

    def add(self, other: "Msat") -> "Msat":
        if not isinstance(other, Msat):
            raise EconArithmeticError(f"Msat.add requires Msat, got {other!r}")
        return Msat(_check_int(self.value + other.value, 0, U63_MAX, "Msat"))

    def sub(self, other: "Msat") -> "Msat":
        if not isinstance(other, Msat):
            raise EconArithmeticError(f"Msat.sub requires Msat, got {other!r}")
        result = self.value - other.value
        if result < 0:
            raise EconArithmeticError(
                f"Msat.sub underflow: {self.value} - {other.value}")
        return Msat(result)

    def diff(self, other: "Msat") -> "SignedMsat":
        if not isinstance(other, Msat):
            raise EconArithmeticError(f"Msat.diff requires Msat, got {other!r}")
        return SignedMsat(self.value - other.value)

    def to_sats_ceil(self) -> "Sat":
        """Fees, budgets, costs, revenue reporting: round UP."""
        return Sat(-(-self.value // 1000))

    def to_sats_floor(self) -> "Sat":
        """Capacity and balances: round DOWN."""
        return Sat(self.value // 1000)

    @classmethod
    def from_sats(cls, sats: int) -> "Msat":
        _check_int(sats, 0, U63_MAX, "Msat.from_sats")
        return cls(_check_int(sats * 1000, 0, U63_MAX, "Msat"))


@dataclass(frozen=True, order=True)
class SignedMsat:
    """Signed millisatoshi delta / P&L."""

    value: int

    def __post_init__(self):
        _check_int(self.value, I64_MIN, U63_MAX, "SignedMsat")

    def to_sats_toward_zero(self) -> int:
        """Signed deltas: round toward zero."""
        if self.value >= 0:
            return self.value // 1000
        return -((-self.value) // 1000)


@dataclass(frozen=True, order=True)
class Sat:
    """Unsigned satoshi amount."""

    value: int

    def __post_init__(self):
        _check_int(self.value, 0, U63_MAX, "Sat")

    def to_msat(self) -> Msat:
        return Msat(_check_int(self.value * 1000, 0, U63_MAX, "Msat"))


@dataclass(frozen=True, order=True)
class Ppm:
    """Parts-per-million rate (fee rates, budget shares)."""

    value: int

    def __post_init__(self):
        _check_int(self.value, 0, 10_000_000, "Ppm")

    def fee_ceil(self, amount: Msat) -> Msat:
        if not isinstance(amount, Msat):
            raise EconArithmeticError(f"Ppm.fee_ceil requires Msat: {amount!r}")
        return Msat(-(-(amount.value * self.value) // 1_000_000))

    def fee_floor(self, amount: Msat) -> Msat:
        if not isinstance(amount, Msat):
            raise EconArithmeticError(f"Ppm.fee_floor requires Msat: {amount!r}")
        return Msat((amount.value * self.value) // 1_000_000)


@dataclass(frozen=True, order=True)
class Micro:
    """Fixed-point ratio in [0, 1] scaled by 1_000_000 (confidence)."""

    value: int

    def __post_init__(self):
        _check_int(self.value, 0, 1_000_000, "Micro")

    @classmethod
    def from_float_clamped(cls, f: float) -> "Micro":
        """Non-authoritative ingestion helper: clamp to [0, 1], then round
        half-up. Authoritative values must be produced as ints directly."""
        if not isinstance(f, (int, float)) or isinstance(f, bool) \
                or not math.isfinite(float(f)):
            raise EconArithmeticError(f"Micro.from_float_clamped: {f!r}")
        clamped = min(1.0, max(0.0, float(f)))
        return cls(int(clamped * 1_000_000 + 0.5))


@dataclass(frozen=True, order=True)
class UnixTime:
    """Integer unix seconds, UTC."""

    value: int

    def __post_init__(self):
        _check_int(self.value, 0, U63_MAX, "UnixTime")

    def plus_seconds(self, seconds: int) -> "UnixTime":
        _check_int(seconds, 0, U63_MAX, "UnixTime.plus_seconds")
        return UnixTime(_check_int(
            self.value + seconds, 0, U63_MAX, "UnixTime"))


def _check_str(value, pattern: re.Pattern, kind: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise EconArithmeticError(f"{kind} invalid: {value!r}")
    return value


@dataclass(frozen=True, order=True)
class ChannelId:
    """Short channel id, NNNxNNNxN."""

    value: str

    def __post_init__(self):
        _check_str(self.value, _CHANNEL_ID_RE, "ChannelId")


@dataclass(frozen=True, order=True)
class PeerId:
    """Compressed node pubkey, 66 lowercase hex chars starting 02/03."""

    value: str

    def __post_init__(self):
        _check_str(self.value, _PEER_ID_RE, "PeerId")


@dataclass(frozen=True, order=True)
class IntentId:
    """Deterministic intent identifier (lowercase, digits, dashes)."""

    value: str

    def __post_init__(self):
        _check_str(self.value, _INTENT_ID_RE, "IntentId")
