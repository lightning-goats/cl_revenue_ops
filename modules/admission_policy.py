"""Admission-control policy (refactor Phase 3A, Workstream F3).

Owns the dynamic htlc_max valve: how large an HTLC each channel
advertises it will accept, based on flow role, spendable liquidity, and
the churn deadband. Extracted VERBATIM from FeeController (which now
delegates here) so admission control is a policy of its own rather than
a branch of the fee formula — the spec's F3 requirement.

Pure functions of (cfg, channel_info, flow_state): no RPC, no DB, no
clock. The Phase 0 golden fixtures (tests/golden/fixtures/htlcmax/)
pin these semantics; the FeeController shims keep that goldened seam
byte-identical.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .utils import sats_to_base

# E-1 valve constants (moved from FeeController, values unchanged).
DEPLETION_SPENDABLE_FRACTION = 0.85
FLOOR_MSAT = 10_000_000  # 10k sats — existing valve floor
UPDATE_DEADBAND_FRAC = 0.10


def _strict_nonnegative_int(value: Any, *, msat_suffix: bool = False) -> Optional[int]:
    """Parse authoritative capacity evidence without converting bad data to zero."""
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "millisatoshis"):
        value = value.millisatoshis
    if isinstance(value, str):
        value = value.strip()
        if msat_suffix and value.endswith("msat"):
            value = value[:-4]
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != parsed:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _strict_fraction(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if 0.0 < parsed <= 1.0 else None


def compute_htlcmax_msat(cfg: Any, channel_info: Dict[str, Any],
                         flow_state: str) -> Optional[int]:
    """Return the valve's target htlc_max (msat), or None when disabled.

    Keeps the operator's flow-class pct knobs as the UPPER shape and
    applies the live-depletion cap whenever the valve is enabled.
    """
    enabled = getattr(cfg, 'enable_dynamic_htlcmax', False)
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("true", "1", "yes")
    else:
        enabled = enabled is True
    if not enabled:
        return None
    capacity_sats = _strict_nonnegative_int(channel_info.get("capacity"))
    spendable_msat = _strict_nonnegative_int(
        channel_info.get("spendable_msat"), msat_suffix=True
    )
    if not capacity_sats or spendable_msat is None:
        return None
    capacity_msat = sats_to_base(capacity_sats)
    if flow_state == "source":
        fraction = _strict_fraction(getattr(cfg, "htlcmax_source_pct", None))
    elif flow_state == "sink":
        fraction = _strict_fraction(getattr(cfg, "htlcmax_sink_pct", None))
    else:
        fraction = _strict_fraction(getattr(cfg, "htlcmax_balanced_pct", None))
    if fraction is None:
        return None
    target_msat = int(capacity_msat * fraction)

    # E-1: live-depletion cap — spendable outbound is what can actually
    # forward; advertising more invites doomed HTLCs.
    depletion_cap_msat = int(spendable_msat * DEPLETION_SPENDABLE_FRACTION)
    target_msat = min(target_msat, depletion_cap_msat)

    # Safety bounds: never below 10,000 sats or above capacity.
    return max(FLOOR_MSAT, min(target_msat, capacity_msat))


def delta_exceeds_deadband(new_msat: int, current_msat: int) -> bool:
    """True when the htlcmax move is big enough to justify a broadcast
    on its own (E-1 churn guard)."""
    new_msat = int(new_msat)
    current_msat = int(current_msat)
    if new_msat == current_msat:
        return False
    if current_msat <= 0:
        return True  # unset/zero on chain: always advertise the valve
    return abs(new_msat - current_msat) > current_msat * UPDATE_DEADBAND_FRAC
