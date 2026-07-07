"""Bounded dynamic budget math for growth-spend experiments.

The helper in this module is intentionally pure: it does not read runtime
state, mutate config, or authorize actions. Callers remain responsible for the
atomic reservation rail that enforces the returned local ceiling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


_MIN_PRIOR_SAMPLES = 3
_MIN_BENEFICIAL_RATIO = 0.50


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number


def _non_negative_int(value: Any) -> int:
    return max(0, _safe_int(value, 0))


def _fraction(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _fleet_prior_status(fleet_prior: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "present": isinstance(fleet_prior, dict) and bool(fleet_prior),
        "usable": False,
        "used": False,
        "reason": "missing",
        "sample_count": 0,
        "beneficial_ratio": None,
    }
    if not isinstance(fleet_prior, dict) or not fleet_prior:
        return status

    status["reason"] = "unusable"
    if not bool(fleet_prior.get("usable")):
        return status

    sample_count = _non_negative_int(fleet_prior.get("sample_count"))
    status["sample_count"] = sample_count
    if sample_count < _MIN_PRIOR_SAMPLES:
        status["reason"] = "insufficient_samples"
        return status

    if isinstance(fleet_prior.get("beneficial_ratio"), bool):
        status["reason"] = "malformed_ratio"
        return status
    try:
        ratio = float(fleet_prior.get("beneficial_ratio"))
    except (TypeError, ValueError):
        status["reason"] = "malformed_ratio"
        return status
    if not math.isfinite(ratio):
        status["reason"] = "malformed_ratio"
        return status

    ratio = max(0.0, min(1.0, ratio))
    status["beneficial_ratio"] = round(ratio, 4)
    status["usable"] = True
    if ratio <= _MIN_BENEFICIAL_RATIO:
        status["reason"] = "non_positive_prior"
        return status

    status["used"] = True
    status["reason"] = "positive_prior"
    return status


def compute_growth_budget_status(
    *,
    base_budget_sats: int,
    net_profit_sats: int,
    actual_spent_sats: int,
    reserved_sats: int,
    enabled: bool,
    earned_fraction: float,
    growth_fraction: float,
    growth_max_extra_sats: int,
    hard_ceiling_sats: int,
    fleet_prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return local dynamic-growth budget diagnostics.

    Fleet priors can only unlock a bounded growth credit. They never own budget
    authority, bypass the local ceiling, or reduce the fixed base floor.
    """
    base_budget = _non_negative_int(base_budget_sats)
    actual_spent = _non_negative_int(actual_spent_sats)
    reserved = _non_negative_int(reserved_sats)
    hard_ceiling = max(base_budget, _non_negative_int(hard_ceiling_sats))
    prior_status = _fleet_prior_status(fleet_prior)

    if not bool(enabled):
        effective = base_budget
        return {
            "mode": "fixed",
            "authority": "local",
            "advisory_only": True,
            "fleet_prior_budget_authority": False,
            "base_budget_sats": base_budget,
            "local_hard_ceiling_sats": hard_ceiling,
            "earned_credit_sats": 0,
            "growth_credit_sats": 0,
            "growth_credit_cap_sats": max(0, _non_negative_int(growth_max_extra_sats)),
            "effective_budget_sats": effective,
            "actual_spent_sats": actual_spent,
            "reserved_sats": reserved,
            "remaining_sats": max(0, effective - actual_spent - reserved),
            "capped_by_hard_ceiling": False,
            "fleet_prior": prior_status,
        }

    positive_profit = max(0, _safe_int(net_profit_sats, 0))
    earned_credit = int(math.floor(positive_profit * _fraction(earned_fraction)))

    growth_credit = 0
    if prior_status["used"]:
        growth_credit = int(math.floor(positive_profit * _fraction(growth_fraction)))
        growth_credit = min(growth_credit, _non_negative_int(growth_max_extra_sats))

    uncapped = base_budget + earned_credit + growth_credit
    effective = min(max(base_budget, uncapped), hard_ceiling)
    return {
        "mode": "dynamic_growth",
        "authority": "local",
        "advisory_only": True,
        "fleet_prior_budget_authority": False,
        "base_budget_sats": base_budget,
        "local_hard_ceiling_sats": hard_ceiling,
        "earned_credit_sats": earned_credit,
        "growth_credit_sats": growth_credit,
        "growth_credit_cap_sats": _non_negative_int(growth_max_extra_sats),
        "effective_budget_sats": effective,
        "actual_spent_sats": actual_spent,
        "reserved_sats": reserved,
        "remaining_sats": max(0, effective - actual_spent - reserved),
        "capped_by_hard_ceiling": effective < uncapped,
        "fleet_prior": prior_status,
    }
