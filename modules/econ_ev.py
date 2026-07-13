"""Common expected-value contract helpers (Workstream F1, PR 6).

The spec's contract:

    expected_value = expected_incremental_revenue
                   - expected_execution_cost
                   - expected_capital_cost
                   - risk_premium

in checked integer msat (wire-contract numeric rules). Population of
intent envelopes is gated by ``econ_ev_populated`` (default off = the
pre-population zeros). Missing data fails CONSERVATIVELY: absent or
unparseable benefit/confidence resolves to zero — exactly the
pre-population state — never to an optimistic estimate. See
docs/refactor/phase0/ev-coverage-matrix.md for the per-path mapping.
"""
from __future__ import annotations

import math

from .econ_types import EconArithmeticError, Micro, SignedMsat

MICRO_ONE = 1_000_000


def expected_value_msat(*, revenue_msat, execution_cost_msat,
                        capital_cost_msat, risk_premium_msat) -> SignedMsat:
    """The common contract, checked. Every term must be a real int
    (bools rejected); costs SUBTRACT — a higher cost or premium can
    never raise the result."""
    terms = {
        "revenue_msat": revenue_msat,
        "execution_cost_msat": execution_cost_msat,
        "capital_cost_msat": capital_cost_msat,
        "risk_premium_msat": risk_premium_msat,
    }
    for name, value in terms.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise EconArithmeticError(
                f"EV term {name}={value!r} must be a checked int")
    return SignedMsat(revenue_msat - execution_cost_msat
                      - capital_cost_msat - risk_premium_msat)


def benefit_msat_from_sats(value_sats) -> SignedMsat:
    """Sats (int/float) -> SignedMsat. None/invalid/NaN -> 0
    (conservative missing-data rule)."""
    try:
        value = float(value_sats)
        if not math.isfinite(value):
            return SignedMsat(0)
        return SignedMsat(int(round(value * 1000)))
    except (TypeError, ValueError):
        return SignedMsat(0)


def confidence_micro(fraction) -> Micro:
    """Unit-interval fraction -> Micro fixed-point, clamped to
    [0, 1_000_000]. None/invalid/NaN -> 0 (conservative)."""
    try:
        value = float(fraction)
        if not math.isfinite(value):
            return Micro(0)
    except (TypeError, ValueError):
        return Micro(0)
    return Micro(int(round(max(0.0, min(1.0, value)) * MICRO_ONE)))
