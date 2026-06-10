"""Bounded fee discount on stagnant over-local channels."""

import os
import random
import sys
import time
from unittest.mock import MagicMock

import pytest

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fee_controller import FeeController as Controller
from modules.config import Config


def test_drain_discount_zero_when_disabled():
    assert Controller._drain_fee_multiplier(
        local_ratio=0.97, forward_count=0,
        high_threshold=0.80, discount_max=0.0,
    ) == 1.0


def test_drain_discount_scales_with_excess():
    m = Controller._drain_fee_multiplier(
        local_ratio=0.90, forward_count=0,
        high_threshold=0.80, discount_max=0.10,
    )
    assert m == pytest.approx(0.95, abs=0.005)
    m_full = Controller._drain_fee_multiplier(
        local_ratio=1.0, forward_count=0,
        high_threshold=0.80, discount_max=0.10,
    )
    assert m_full == pytest.approx(0.90, abs=0.005)


def test_no_discount_for_channels_with_recent_flow():
    assert Controller._drain_fee_multiplier(
        local_ratio=0.97, forward_count=3,
        high_threshold=0.80, discount_max=0.10,
    ) == 1.0


def test_no_discount_inside_band():
    assert Controller._drain_fee_multiplier(
        local_ratio=0.60, forward_count=0,
        high_threshold=0.80, discount_max=0.10,
    ) == 1.0


def test_degenerate_threshold_no_discount():
    assert Controller._drain_fee_multiplier(
        local_ratio=0.99, forward_count=0,
        high_threshold=1.0, discount_max=0.10,
    ) == 1.0


# =============================================================================
# Integration: discount wired into the per-channel fee cycle
# =============================================================================

PEER_ID = "02" + "a" * 64
CHANNEL_ID = "100x1x0"


def _make_config_snapshot(**overrides):
    """Mirror of tests/test_fee_cycle_optimizations.py snapshot helper."""
    defaults = {
        "paused": False,
        "min_fee_ppm": 10,
        "max_fee_ppm": 5000,
        "fee_interval": 1800,
        "fee_profile": "active",
        "enable_vegas_reflex": False,
        "enable_dynamic_htlcmax": False,
        "market_fee_mode": "undercut",
        "neighbor_median_min_competitors": 3,
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 1,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "inbound_fee_estimate_ppm": 200,
        "thompson_prior_std_fee": 200.0,
        "routing_intelligence_enabled": False,
        "high_liquidity_threshold": 0.80,
        "drain_fee_discount_max": 0.0,
    }
    defaults.update(overrides)
    snap = MagicMock()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _setup_db(mock_database, now):
    """Stagnant over-local channel: zero forwards/volume in a 2h window."""
    mock_database.get_channel_probe.return_value = None
    mock_database.get_last_rebalance_cost.return_value = None
    mock_database.get_volume_since.return_value = 0
    mock_database.get_forward_count_since.return_value = 0
    mock_database.get_peer_uptime_percent.return_value = 99.5
    mock_database.get_fee_strategy_state.side_effect = lambda *a, **kw: {
        "last_revenue_rate": 0.0,
        "last_fee_ppm": 150,
        "trend_direction": 1,
        "step_ppm": 50,
        "last_update": now - 7200,
        "consecutive_same_direction": 0,
        "is_sleeping": 0,
        "sleep_until": 0,
        "stable_cycles": 0,
        "forward_count_since_update": 0,
        "last_volume_sats": 0,
        "v2_state_json": None,
    }
    mock_database.get_last_forward_time.return_value = now - 86400
    mock_database.get_failure_count.return_value = (0, 0)
    mock_database.get_channel_cost_history.return_value = []
    mock_database.get_channel_rebalance_success_rate.return_value = None
    mock_database.get_channel_age.return_value = 30
    mock_database.get_historical_inbound_fee_ppm.return_value = None
    mock_database.get_peer_latency_stats.return_value = {
        "avg": 0.0, "std": 0.0, "count": 0,
    }
    mock_database.get_all_channel_states.return_value = [
        {
            "channel_id": CHANNEL_ID,
            "peer_id": PEER_ID,
            "state": "balanced",
            "flow_ratio": 0.5,
            "sats_in": 0,
            "sats_out": 0,
            "capacity": 2_000_000,
            "updated_at": now,
            "kalman_flow_ratio": 0.0,
            "kalman_velocity": 0.0,
        }
    ]


def _make_drain_fc(mock_plugin, mock_database, *, discount_max):
    """FeeController with ONE channel at ~97% local and zero forwards."""
    config = MagicMock(spec=Config)
    fc = Controller(mock_plugin, config, mock_database)
    cfg = _make_config_snapshot(drain_fee_discount_max=discount_max)
    fc.config.snapshot.return_value = cfg
    fc.config.dry_run = True  # set_channel_fee succeeds without RPC

    _setup_db(mock_database, now=int(time.time()))

    channels_info = {
        CHANNEL_ID: {
            "channel_id": CHANNEL_ID,
            "short_channel_id": CHANNEL_ID,
            "peer_id": PEER_ID,
            "capacity": 2_000_000,
            "spendable_msat": 1_940_000_000,   # 97% local
            "receivable_msat": 54_000_000,
            "fee_base_msat": 0,
            "fee_proportional_millionths": 150,
            "htlc_minimum_msat": 0,
            "htlc_min_msat": 0,
            "htlc_maximum_msat": 0,
            "htlc_max_msat": 0,
            "opener": "local",
        }
    }
    fc._get_channels_info = lambda: channels_info
    fc._get_dynamic_chain_costs = lambda: None
    mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}
    return fc, cfg


def _run_cycle_target(fc):
    """Run a real fee cycle; return (post-bias target ppm, applied fee ppm)."""
    random.seed(20260610)  # identical DTS sampling across runs
    adjustments = fc.adjust_all_fees()
    assert adjustments, "cycle should produce an adjustment for the channel"
    adj = adjustments[0]
    return adj.algorithm_values["post_pid_target_ppm"], adj.new_fee_ppm


def test_drain_discount_lowers_cycle_target_within_rails(mock_plugin, mock_database):
    """With drain_fee_discount_max enabled, a stagnant ~97%-local channel gets
    a strictly lower fee target than the disabled baseline, and the applied
    fee still respects min_fee_ppm rails."""
    fc_base, _ = _make_drain_fc(mock_plugin, mock_database, discount_max=0.0)
    baseline_target, baseline_fee = _run_cycle_target(fc_base)

    fc_disc, cfg = _make_drain_fc(mock_plugin, mock_database, discount_max=0.10)
    discounted_target, discounted_fee = _run_cycle_target(fc_disc)

    assert discounted_target < baseline_target, (
        f"discounted target {discounted_target} should be below "
        f"baseline {baseline_target}"
    )
    assert discounted_fee >= cfg.min_fee_ppm
    assert baseline_fee >= cfg.min_fee_ppm
