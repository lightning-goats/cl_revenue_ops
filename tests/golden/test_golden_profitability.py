"""Golden: ChannelProfitability derived signals (marginal_roi, role_30d)
and ChannelProfitabilityAnalyzer._classify_channel.

_classify_channel reads the wall clock (portability hazard §1) — frozen
here to FROZEN_NOW so day-based inactivity math is deterministic. ROI is
fractional (0.10 = 10%), matching the class thresholds.
"""
from unittest.mock import MagicMock

import pytest

from modules.profitability_analyzer import (
    ChannelCosts,
    ChannelProfitability,
    ChannelProfitabilityAnalyzer,
    ChannelRevenue,
    ProfitabilityClass,
)
from tests.golden.util import golden_check

FROZEN_NOW = 1_752_400_000  # 2026-07-13 UTC, near fixture authoring time


@pytest.fixture(autouse=True)
def _freeze_time(monkeypatch):
    monkeypatch.setattr(
        "modules.profitability_analyzer.time.time", lambda: FROZEN_NOW)


def _make_prof(fees_earned_sats=2000, rebalance_cost_sats=1000,
               sourced_fee_contribution_sats=0):
    costs = ChannelCosts(
        channel_id="111x222x0", peer_id="02" + "a" * 64,
        open_cost_sats=500, rebalance_cost_sats=rebalance_cost_sats,
        effective_rebalance_cost_sats=0,
    )
    revenue = ChannelRevenue(
        channel_id="111x222x0",
        fees_earned_msat=fees_earned_sats * 1000,
        volume_routed_msat=1_000_000 * 1000,
        forward_count=100,
        sourced_fee_contribution_msat=sourced_fee_contribution_sats * 1000,
    )
    return ChannelProfitability(
        channel_id="111x222x0", peer_id="02" + "a" * 64,
        capacity_sats=2_000_000, costs=costs, revenue=revenue,
        net_profit_sats=fees_earned_sats - costs.total_cost_sats,
        roi_percent=10.0, classification=ProfitabilityClass.PROFITABLE,
        cost_per_sat_routed=0.001, fee_per_sat_routed=0.002,
        days_open=30, last_routed=None,
    )


ROLE_SCENARIOS = [
    # (name, window_avail, fwd_30d, sourced_fwd_30d)
    # total_forward_count_30d is a derived property (fwd + sourced).
    ("no_window_falls_back_to_lifetime", False, 0, 0),
    ("gateway_30d_dominant_sourced", True, 2, 40),
    ("exit_30d_dominant_direct", True, 40, 2),
    ("balanced_30d", True, 20, 22),
    ("too_few_forwards_30d_dormant", True, 3, 4),
]


@pytest.mark.parametrize("name,avail,fwd,sourced", ROLE_SCENARIOS)
def test_golden_role_30d(name, avail, fwd, sourced):
    prof = _make_prof()
    prof.window_30d_available = avail
    prof.forward_count_30d = fwd
    prof.sourced_forward_count_30d = sourced
    golden_check(f"profitability/role30d_{name}", {
        "inputs": {"window_30d_available": avail, "forward_count_30d": fwd,
                   "sourced_forward_count_30d": sourced,
                   "total_forward_count_30d": prof.total_forward_count_30d,
                   "lifetime_role": prof.channel_role},
        "role_30d": prof.role_30d,
    })


MARGINAL_ROI_SCENARIOS = [
    ("profit_over_cost", 500, 200),
    ("negative_profit", -300, 600),
    ("zero_cost_positive_profit", 500, 0),
    ("zero_cost_zero_profit", 0, 0),
]


@pytest.mark.parametrize("name,profit_30d,cost_30d", MARGINAL_ROI_SCENARIOS)
def test_golden_marginal_roi(name, profit_30d, cost_30d):
    prof = _make_prof()
    prof.marginal_profit_30d_sats = profit_30d
    prof.rebalance_cost_30d_sats = cost_30d
    golden_check(f"profitability/marginal_roi_{name}", {
        "inputs": {"marginal_profit_30d_sats": profit_30d,
                   "rebalance_cost_30d_sats": cost_30d},
        "marginal_roi": prof.marginal_roi,
    })


def _analyzer(diag_stats=None):
    db = MagicMock()
    db.get_diagnostic_rebalance_stats.return_value = diag_stats or {
        "attempt_count": 0, "last_success_time": None,
    }
    db.get_fee_strategy_state.return_value = None
    return ChannelProfitabilityAnalyzer(MagicMock(), MagicMock(), db)


CLASSIFY_SCENARIOS = [
    # (name, roi_fraction, net_profit_sats, last_routed_ts, days_open, fwd)
    ("young_profitable", 0.25, 5000, FROZEN_NOW - 86_400, 10, 200),
    ("old_loser_stagnant", -0.40, -8000, FROZEN_NOW - 86_400 * 143, 400, 15),
    ("never_routed_mature", 0.0, -500, None, 120, 0),
    ("breakeven_active", 0.005, 10, FROZEN_NOW - 3_600, 90, 500),
]


@pytest.mark.parametrize(
    "name,roi,net,last,days,fwd", CLASSIFY_SCENARIOS)
def test_golden_classify(name, roi, net, last, days, fwd):
    analyzer = _analyzer()
    result = analyzer._classify_channel(
        roi, net, last, days, channel_id="111x222x0",
        peer_id="02" + "a" * 64, forward_count=fwd,
    )
    golden_check(f"profitability/classify_{name}", {
        "inputs": {"roi": roi, "net_profit": net, "last_routed": last,
                   "days_open": days, "forward_count": fwd},
        "classification": result,
    })


def test_marginal_roi_hand_computed_anchor():
    prof = _make_prof()
    prof.marginal_profit_30d_sats = 500
    prof.rebalance_cost_30d_sats = 200
    assert abs(prof.marginal_roi - 2.5) < 0.01
