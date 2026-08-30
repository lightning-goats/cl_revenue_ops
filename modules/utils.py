"""
Small shared helpers used across cl-revenue-ops modules.

Keep this module dependency-light to avoid import cycles.
"""

import logging
import math
import re
from typing import Any, Optional

_log = logging.getLogger("cl-revenue-ops.utils")


def normalize_scid(scid: Optional[str]) -> str:
    """
    Normalize a short_channel_id to the plugin's internal 'x' separator format.

    CLN APIs may return SCIDs with either 'x' or ':' separators depending on context/version.
    """
    return (scid or "").replace(":", "x")


def parse_msat(msat_val: Any) -> int:
    """
    Safely convert msat values to an integer number of millisatoshis.

    Handles:
    - ints/floats
    - strings like '1000msat' or '1000'
    - pyln Millisatoshi-like objects (has .millisatoshis)
    """
    if msat_val is None:
        return 0
    if hasattr(msat_val, "millisatoshis"):
        try:
            return int(msat_val.millisatoshis)
        except Exception as e:
            _log.debug("parse_msat: failed to convert .millisatoshis %r: %s", msat_val, e)
            return 0
    if isinstance(msat_val, bool):
        # U-1 FIX: bool is never a valid msat value (True→1 would be wrong)
        return 0
    if isinstance(msat_val, (int, float)):
        try:
            return int(msat_val)
        except Exception as e:
            _log.debug("parse_msat: failed to convert numeric %r: %s", msat_val, e)
            return 0
    if isinstance(msat_val, str):
        s = msat_val.strip()
        if s.endswith("msat"):
            s = s[:-4]
        try:
            return int(s)
        except Exception as e:
            _log.debug("parse_msat: failed to convert string %r: %s", msat_val, e)
            return 0
    try:
        return int(msat_val)
    except Exception as e:
        _log.debug("parse_msat: failed to convert %r (type %s): %s", msat_val, type(msat_val).__name__, e)
        return 0


_NONNEGATIVE_MSAT_RE = re.compile(r"^[0-9]+(?:msat)?$")


def _optional_nonnegative_msat(value: Any) -> Optional[int]:
    """Strict optional parser used when zero and malformed mean different things."""
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "millisatoshis"):
        value = value.millisatoshis
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not _NONNEGATIVE_MSAT_RE.fullmatch(value):
            return None
        if value.endswith("msat"):
            value = value[:-4]
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def optional_channel_local_balance_msat(channel: Any) -> Optional[int]:
    """Return true local channel balance, or ``None`` when no valid value exists.

    ``spendable_msat`` is the maximum currently sendable amount and may be
    limited by reserves, HTLC policy, or the peer's single-HTLC ceiling.  It is
    therefore not a reliable inventory balance on large channels.  Modern CLN
    exposes the commitment balance as ``to_us_msat``; listfunds-compatible
    inputs use ``our_amount_msat``.
    """
    if not isinstance(channel, dict):
        return None
    total = _optional_nonnegative_msat(
        channel.get("total_msat", channel.get("capacity_msat"))
    )
    for key in ("our_amount_msat", "to_us_msat"):
        if key not in channel:
            continue
        local = _optional_nonnegative_msat(channel.get(key))
        if local is not None and (total is None or total == 0 or local <= total):
            return local
    spendable = _optional_nonnegative_msat(channel.get("spendable_msat"))
    if spendable is None:
        return None
    if total is not None and total > 0:
        return min(spendable, total)
    return spendable


def channel_local_balance_msat(channel: Any) -> int:
    """Neutral-zero wrapper around :func:`optional_channel_local_balance_msat`."""
    value = optional_channel_local_balance_msat(channel)
    return 0 if value is None else value


# ---------------------------------------------------------------------------
# Lightning Network base unit configuration
# ---------------------------------------------------------------------------
# Today: millisatoshi (msat). Future: microsatoshi (usat) or smaller.
# Change BASE_UNITS_PER_SAT when the network adopts a smaller unit.
BASE_UNITS_PER_SAT = 1000
BASE_UNIT_NAME = "msat"


def base_to_sats_ceil(base: int) -> int:
    """Convert base units to sats, rounding UP.

    Use for: fees, budgets, costs — never undercharge or underbudget.
    """
    return -(-base // BASE_UNITS_PER_SAT)


def base_to_sats_floor(base: int) -> int:
    """Convert base units to sats, rounding DOWN.

    Use for: capacity and balances — never overstate what is spendable.
    Revenue reporting uses base_to_sats_ceil so sub-sat earnings stay
    visible instead of truncating to zero.
    """
    return base // BASE_UNITS_PER_SAT


def base_delta_to_sats_toward_zero(base: int) -> int:
    """Convert signed base-unit deltas to sats, rounding toward zero."""
    if base >= 0:
        return base // BASE_UNITS_PER_SAT
    return -((-base) // BASE_UNITS_PER_SAT)


def sats_to_base(sats: int) -> int:
    """Convert sats to base units (msat today)."""
    return sats * BASE_UNITS_PER_SAT


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
# Existing code uses parse_msat and MSAT_PER_SAT. New code should prefer
# the generic names (base_to_sats_ceil, parse_base_unit, etc.) for
# future-proofing, but the aliases are permanent and safe to use.
MSAT_PER_SAT = BASE_UNITS_PER_SAT
parse_base_unit = parse_msat
msat_to_sats_ceil = base_to_sats_ceil
msat_to_sats_floor = base_to_sats_floor
sats_to_msat = sats_to_base
msat_delta_to_sats_toward_zero = base_delta_to_sats_toward_zero
