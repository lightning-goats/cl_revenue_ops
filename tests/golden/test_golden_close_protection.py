"""Golden: CapacityPlanner close-protection gates.

_close_protection_reason inputs are duck-typed (prof, flow_metrics) —
SimpleNamespace stands in, matching how production passes
ChannelProfitability + FlowMetrics. These fixtures pin the 30d-window
semantics fixed at baseline commit 5e8f747.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.capacity_planner import CapacityPlanner
from modules.profitability_analyzer import ChannelRole
from tests.golden.util import golden_check

SCID = "111x222x0"


@pytest.fixture
def planner():
    return CapacityPlanner(MagicMock(), MagicMock(), MagicMock())


def _prof(role_30d=ChannelRole.BALANCED, marginal_roi_percent=0.0,
          window_30d_available=True, sourced_fee_30d_msat=0,
          lifetime_sourced_fee_sats=0, days_open=100):
    return SimpleNamespace(
        role_30d=role_30d,
        channel_role=ChannelRole.BALANCED,
        marginal_roi_percent=marginal_roi_percent,
        window_30d_available=window_30d_available,
        sourced_fee_30d_msat=sourced_fee_30d_msat,
        days_open=days_open,
        revenue=SimpleNamespace(
            sourced_fee_contribution_sats=lifetime_sourced_fee_sats),
    )


def _flow(confidence=0.9, forward_count=50):
    return SimpleNamespace(confidence=confidence, forward_count=forward_count)


SCENARIOS = {
    # name: (prof, flow_metrics, route_pair_channels)
    "unprotected_dead_channel": (
        _prof(role_30d=ChannelRole.BALANCED, marginal_roi_percent=-90.0),
        _flow(0.9), set()),
    "gateway_30d_protected": (
        _prof(role_30d=ChannelRole.INBOUND_GATEWAY,
              marginal_roi_percent=-10.0),
        _flow(0.9), set()),
    "gateway_30d_but_deep_loser_unprotected": (
        _prof(role_30d=ChannelRole.INBOUND_GATEWAY,
              marginal_roi_percent=-60.0),
        _flow(0.9), set()),
    "sourced_fee_30d_protected": (
        _prof(window_30d_available=True, sourced_fee_30d_msat=500_000,
              marginal_roi_percent=-20.0),
        _flow(0.9), set()),
    "sourced_fee_lifetime_fallback_protected": (
        _prof(window_30d_available=False, lifetime_sourced_fee_sats=500,
              marginal_roi_percent=-20.0),
        _flow(0.9), set()),
    "stale_lifetime_sourcing_not_used_when_window_present": (
        # THE 5e8f747 fix: lifetime sourcing must not protect when the
        # 30d window exists and shows nothing.
        _prof(window_30d_available=True, sourced_fee_30d_msat=0,
              lifetime_sourced_fee_sats=500, marginal_roi_percent=-90.0),
        _flow(0.9), set()),
    "low_confidence_active_channel_gated": (
        # Kalman estimate is load-bearing: recent activity, low confidence.
        _prof(marginal_roi_percent=-90.0),
        _flow(confidence=0.2, forward_count=5), set()),
    "low_confidence_dead_mature_inactivity_is_signal": (
        # F3: zero forwards over a full window on a mature channel IS the
        # signal — the confidence gate must not block closure.
        _prof(marginal_roi_percent=-90.0, days_open=100),
        _flow(confidence=0.2, forward_count=0), set()),
    "revenue_route_pair_protected": (
        _prof(marginal_roi_percent=-10.0),
        _flow(0.9), {SCID}),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_close_protection_reason(planner, name):
    prof, flow, route_pairs = SCENARIOS[name]
    reason = planner._close_protection_reason(SCID, prof, flow, route_pairs)
    golden_check(f"close_protection/{name}", {"reason": reason})


def test_windowed_fix_anchor(planner):
    """Non-golden anchor for commit 5e8f747: empty 30d window + rich
    lifetime sourcing must NOT protect; missing window must fail safe
    to the lifetime figure and protect."""
    stale = _prof(window_30d_available=True, sourced_fee_30d_msat=0,
                  lifetime_sourced_fee_sats=500, marginal_roi_percent=-90.0)
    assert planner._close_protection_reason(SCID, stale, _flow(0.9), set()) is None

    no_window = _prof(window_30d_available=False,
                      lifetime_sourced_fee_sats=500,
                      marginal_roi_percent=-20.0)
    assert planner._close_protection_reason(
        SCID, no_window, _flow(0.9), set()) == "SOURCED_FEE_CONTRIBUTION"


def _policy(strategy="dynamic", tags=()):
    policy = MagicMock()
    policy.strategy = MagicMock()
    policy.strategy.value = strategy
    policy.has_tag.side_effect = lambda t: t in tags
    return policy


CLOSE_ALLOWED_SCENARIOS = {
    "dynamic_no_tags_allowed": _policy("dynamic"),
    "static_policy_blocks": _policy("static"),
    "passive_policy_blocks": _policy("passive"),
    "no_close_tag_blocks": _policy("dynamic", tags=("no_close",)),
    "protect_tag_blocks": _policy("dynamic", tags=("protect",)),
}


@pytest.mark.parametrize("name", sorted(CLOSE_ALLOWED_SCENARIOS))
def test_golden_check_close_allowed(planner, name):
    pm = MagicMock()
    pm.get_policy.return_value = CLOSE_ALLOWED_SCENARIOS[name]
    planner.policy_manager = pm
    allowed, reason = planner._check_close_allowed("02" + "a" * 64)
    golden_check(f"close_protection/allowed_{name}", {
        "allowed": allowed, "reason": reason,
    })


def test_close_allowed_fails_closed(planner):
    """Non-golden anchors: no policy manager, or a broken one, must both
    block closes (fail closed)."""
    planner.policy_manager = None
    allowed, reason = planner._check_close_allowed("02" + "a" * 64)
    assert allowed is False

    pm = MagicMock()
    pm.get_policy.side_effect = RuntimeError("db gone")
    planner.policy_manager = pm
    allowed, reason = planner._check_close_allowed("02" + "a" * 64)
    assert allowed is False
