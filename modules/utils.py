"""
Small shared helpers used across cl-revenue-ops modules.

Keep this module dependency-light to avoid import cycles.
"""

import logging
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

    Use for: capacity, balance, revenue reporting — never overstate.
    """
    return base // BASE_UNITS_PER_SAT


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

