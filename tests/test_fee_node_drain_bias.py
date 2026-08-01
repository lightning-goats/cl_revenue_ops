import pytest

from modules.fee_controller import compute_node_receivable_ratio, node_drain_pressure

def _ch(to_us_msat, total_msat, state="CHANNELD_NORMAL"):
    return {"to_us_msat": to_us_msat, "total_msat": total_msat, "state": state}

def test_receivable_ratio_source_heavy():
    # 90% local across two channels -> receivable ratio 0.10
    chans = [_ch(900_000_000, 1_000_000_000), _ch(900_000_000, 1_000_000_000)]
    assert abs(compute_node_receivable_ratio(chans) - 0.10) < 1e-6

def test_receivable_ratio_balanced():
    chans = [_ch(500_000_000, 1_000_000_000)]
    assert abs(compute_node_receivable_ratio(chans) - 0.50) < 1e-6

def test_receivable_ratio_skips_non_normal_and_bad_entries():
    chans = [_ch(900_000_000, 1_000_000_000), _ch(0, 1_000_000_000, state="CHANNELD_AWAITING_LOCKIN"), "garbage", {}]
    # only the first (normal) channel counts -> 0.10
    assert abs(compute_node_receivable_ratio(chans) - 0.10) < 1e-6

def test_receivable_ratio_zero_capacity_safe():
    assert compute_node_receivable_ratio([]) == 1.0

def test_drain_pressure_ramp():
    # target 0.30, floor 0.20
    assert node_drain_pressure(0.35, 0.30, 0.20) == 0.0     # healthy
    assert node_drain_pressure(0.30, 0.30, 0.20) == 0.0     # at target
    assert node_drain_pressure(0.20, 0.30, 0.20) == 1.0     # at floor -> full
    assert node_drain_pressure(0.10, 0.30, 0.20) == 1.0     # below floor -> clamped 1
    assert abs(node_drain_pressure(0.25, 0.30, 0.20) - 0.5) < 1e-6  # midpoint

def test_drain_pressure_degenerate_target_le_floor():
    assert node_drain_pressure(0.15, 0.20, 0.20) == 1.0
    assert node_drain_pressure(0.25, 0.20, 0.20) == 0.0


# ---------------------------------------------------------------------------
# Task 2: config knobs wired end-to-end + runtime-settable
# ---------------------------------------------------------------------------

def test_node_drain_bias_defaults():
    from modules.config import Config
    c = Config()
    assert c.node_drain_bias_enabled is False
    assert c.node_drain_bias_max == 0.3


def test_node_drain_bias_runtime_settable():
    from modules.config import PUBLIC_RUNTIME_KEYS
    assert 'node_drain_bias_enabled' in PUBLIC_RUNTIME_KEYS
    assert 'node_drain_bias_max' in PUBLIC_RUNTIME_KEYS


def test_node_drain_bias_field_types_and_ranges():
    from modules.config import CONFIG_FIELD_TYPES, CONFIG_FIELD_RANGES
    assert CONFIG_FIELD_TYPES['node_drain_bias_enabled'] is bool
    assert CONFIG_FIELD_TYPES['node_drain_bias_max'] is float
    assert CONFIG_FIELD_RANGES['node_drain_bias_max'] == (0.0, 0.5)
    assert 'node_drain_bias_enabled' not in CONFIG_FIELD_RANGES


def test_node_drain_bias_snapshot_mirrors_config():
    from modules.config import Config
    c = Config()
    snap = c.snapshot()
    assert snap.node_drain_bias_enabled == c.node_drain_bias_enabled
    assert snap.node_drain_bias_max == c.node_drain_bias_max


class TestNodeDrainBiasOptionsRegistered:
    """P6-002 regression guard: a config knob added to the Config dataclass
    without a matching plugin.add_option(...) registration is a silent
    no-op at startup (the operator-supplied value is never read). Verify
    both new node-drain-bias knobs are (a) present in CONFIG_FIELD_TYPES
    and (b) actually registered as a plugin option with a default that
    matches the Config dataclass default, using the same AST-based
    extraction as TestUpstreamPatternOptionsRegistered in
    tests/test_rebalancer_module.py (avoids importing the plugin module,
    which would register the whole plugin surface)."""

    FIELD_TO_OPTION = {
        "node_drain_bias_enabled": "revenue-ops-node-drain-bias-enabled",
        "node_drain_bias_max": "revenue-ops-node-drain-bias-max",
    }

    def _plugin_option_default(self, name):
        import ast
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tree = ast.parse(open(os.path.join(root, "cl-revenue-ops.py")).read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_option"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            name_node = kw.get("name")
            if isinstance(name_node, ast.Constant) and name_node.value == name:
                default_node = kw.get("default")
                if isinstance(default_node, ast.Constant):
                    return default_node.value
        raise AssertionError(f"option {name} not found")

    def test_both_knobs_in_config_field_types(self):
        from modules.config import CONFIG_FIELD_TYPES
        for field in self.FIELD_TO_OPTION:
            assert field in CONFIG_FIELD_TYPES, f"{field} missing from CONFIG_FIELD_TYPES"

    def test_both_knobs_registered_as_plugin_options_with_matching_defaults(self):
        from modules.config import Config
        cfg = Config()
        for field, option_name in self.FIELD_TO_OPTION.items():
            option_default = self._plugin_option_default(option_name)
            field_default = getattr(cfg, field)
            if isinstance(field_default, bool):
                parsed_bool = str(option_default).lower() in ("true", "1", "yes")
                assert parsed_bool is field_default, (
                    f"{option_name} default {option_default!r} does not match "
                    f"{field}={field_default!r}"
                )
            else:
                assert type(field_default)(option_default) == field_default, (
                    f"{option_name} default {option_default!r} does not match "
                    f"{field}={field_default!r}"
                )


# ---------------------------------------------------------------------------
# Task 3: node-scaled effective drain discount cap (pure helper)
# ---------------------------------------------------------------------------

class _CfgLike:
    """Minimal cfg-like stand-in exposing only the attributes the pure
    helper reads. Deliberately NOT a MagicMock: a bare MagicMock returns a
    truthy, non-numeric value for any unset attribute, which would silently
    mask bugs in the helper's attribute access (getattr with real defaults)."""

    def __init__(self, *, drain_fee_discount_max, node_drain_bias_enabled,
                 node_drain_bias_max):
        self.drain_fee_discount_max = drain_fee_discount_max
        self.node_drain_bias_enabled = node_drain_bias_enabled
        self.node_drain_bias_max = node_drain_bias_max


def test_effective_drain_discount_enabled_full_pressure_scales_up_from_zero():
    from modules.fee_controller import effective_drain_discount_max
    cfg = _CfgLike(drain_fee_discount_max=0.0, node_drain_bias_enabled=True,
                   node_drain_bias_max=0.3)
    assert effective_drain_discount_max(cfg, 1.0) == pytest.approx(0.3)


def test_effective_drain_discount_enabled_zero_pressure_equals_static():
    from modules.fee_controller import effective_drain_discount_max
    cfg = _CfgLike(drain_fee_discount_max=0.0, node_drain_bias_enabled=True,
                   node_drain_bias_max=0.3)
    assert effective_drain_discount_max(cfg, 0.0) == pytest.approx(0.0)


def test_effective_drain_discount_enabled_takes_max_of_static_and_scaled():
    from modules.fee_controller import effective_drain_discount_max
    cfg = _CfgLike(drain_fee_discount_max=0.1, node_drain_bias_enabled=True,
                   node_drain_bias_max=0.3)
    # max(0.1, 0.3*0.5=0.15) = 0.15
    assert effective_drain_discount_max(cfg, 0.5) == pytest.approx(0.15)


def test_effective_drain_discount_enabled_static_wins_when_higher():
    from modules.fee_controller import effective_drain_discount_max
    cfg = _CfgLike(drain_fee_discount_max=0.25, node_drain_bias_enabled=True,
                   node_drain_bias_max=0.3)
    # max(0.25, 0.3*0.1=0.03) = 0.25 (static already above the scaled term)
    assert effective_drain_discount_max(cfg, 0.1) == pytest.approx(0.25)


def test_effective_drain_discount_disabled_equals_static_regardless_of_pressure():
    from modules.fee_controller import effective_drain_discount_max
    cfg = _CfgLike(drain_fee_discount_max=0.0, node_drain_bias_enabled=False,
                   node_drain_bias_max=0.3)
    assert effective_drain_discount_max(cfg, 1.0) == 0.0
    assert effective_drain_discount_max(cfg, 0.0) == 0.0

    cfg2 = _CfgLike(drain_fee_discount_max=0.1, node_drain_bias_enabled=False,
                     node_drain_bias_max=0.3)
    assert effective_drain_discount_max(cfg2, 1.0) == 0.1


# ---------------------------------------------------------------------------
# Task 3: integration — node-drain-bias wired into a real fee cycle
# ---------------------------------------------------------------------------
#
# Mirrors the harness in tests/test_drain_fee_pressure.py (per-channel drain
# discount already wired). Here the SAME over-local, zero-forward channel is
# scored on a NODE that is source-heavy overall (low receivable ratio), with
# node_drain_bias_enabled toggled on/off, to verify:
#   - OFF -> byte-identical to the pre-Task-3 static-discount-only path
#   - ON + starved node -> strictly lower target than OFF (auto-activation)

import time as _time
import random as _random
from unittest.mock import MagicMock as _MagicMock

from modules.fee_controller import FeeController as _Controller
from modules.fee_authority import FeeAuthorityGate
from modules.config import Config as _Config

_PEER_ID = "02" + "b" * 64
_CHANNEL_ID = "200x2x0"


def _node_bias_config_snapshot(**overrides):
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
        "inbound_fee_estimate_ppm": 200,
        "thompson_prior_std_fee": 200.0,
        "routing_intelligence_enabled": False,
        "high_liquidity_threshold": 0.80,
        "drain_fee_discount_max": 0.0,
        "node_drain_bias_enabled": False,
        "node_drain_bias_max": 0.3,
        "receivable_ratio_target": 0.30,
        "receivable_ratio_floor": 0.20,
    }
    defaults.update(overrides)
    snap = _MagicMock()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _setup_node_bias_db(mock_database, now):
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
            "channel_id": _CHANNEL_ID,
            "peer_id": _PEER_ID,
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


def _make_node_bias_fc(mock_plugin, mock_database, *, node_drain_bias_enabled,
                        node_receivable_ratio):
    """FeeController with one ~97%-local zero-forward channel, on a node
    whose AGGREGATE (raw listpeerchannels) receivable ratio is controlled
    via `node_receivable_ratio` — independent of the per-channel loop's
    reshaped channel dict, mirroring how compute_node_receivable_ratio
    consumes raw to_us_msat/total_msat/state fields."""
    config = _MagicMock(spec=_Config)
    fc = _Controller(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())
    cfg = _node_bias_config_snapshot(node_drain_bias_enabled=node_drain_bias_enabled)
    fc.config.snapshot.return_value = cfg

    _setup_node_bias_db(mock_database, now=int(_time.time()))

    channels_info = {
        _CHANNEL_ID: {
            "channel_id": _CHANNEL_ID,
            "short_channel_id": _CHANNEL_ID,
            "peer_id": _PEER_ID,
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

    # Raw node-wide channel list for compute_node_receivable_ratio: capacity
    # 10_000_000 msat total, local balance sized so remote/total ==
    # node_receivable_ratio.
    total_msat = 10_000_000
    to_us_msat = int(total_msat * (1.0 - node_receivable_ratio))
    mock_plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {"to_us_msat": to_us_msat, "total_msat": total_msat, "state": "CHANNELD_NORMAL"},
        ]
    }
    return fc, cfg


def _run_node_bias_cycle(fc):
    rng_state = _random.getstate()
    try:
        _random.seed(20260610)
        adjustments = fc.adjust_all_fees()
    finally:
        _random.setstate(rng_state)
    assert adjustments, "cycle should produce an adjustment for the channel"
    adj = adjustments[0]
    return adj.algorithm_values["post_pid_target_ppm"], adj.new_fee_ppm


def test_node_drain_bias_disabled_matches_pre_feature_baseline(mock_plugin, mock_database):
    """node_drain_bias_enabled=False must be byte-identical to the static
    drain_fee_discount_max=0.0 path regardless of how starved the node is."""
    fc_off, _ = _make_node_bias_fc(
        mock_plugin, mock_database,
        node_drain_bias_enabled=False, node_receivable_ratio=0.05,
    )
    off_target, off_fee = _run_node_bias_cycle(fc_off)

    # Second controller/mocks pair with the SAME static config and a
    # perfectly balanced (non-starved) node — since the feature is
    # disabled, node state must not matter at all.
    fc_off_balanced, _ = _make_node_bias_fc(
        mock_plugin, mock_database,
        node_drain_bias_enabled=False, node_receivable_ratio=0.50,
    )
    off_balanced_target, off_balanced_fee = _run_node_bias_cycle(fc_off_balanced)

    assert off_target == off_balanced_target
    assert off_fee == off_balanced_fee


def test_node_drain_bias_enabled_starved_node_lowers_target_below_disabled(mock_plugin, mock_database):
    """enabled + source-heavy node (receivable ratio at/below floor) must
    produce a strictly lower post-PID target than the disabled baseline,
    auto-activating the drain discount even though the static
    drain_fee_discount_max is 0.0."""
    fc_off, _ = _make_node_bias_fc(
        mock_plugin, mock_database,
        node_drain_bias_enabled=False, node_receivable_ratio=0.05,
    )
    off_target, off_fee = _run_node_bias_cycle(fc_off)

    fc_on, cfg_on = _make_node_bias_fc(
        mock_plugin, mock_database,
        node_drain_bias_enabled=True, node_receivable_ratio=0.05,
    )
    on_target, on_fee = _run_node_bias_cycle(fc_on)

    assert on_target < off_target, (
        f"node-drain-bias-enabled target {on_target} should be below "
        f"disabled baseline {off_target}"
    )
    assert on_fee >= cfg_on.min_fee_ppm
    assert off_fee >= cfg_on.min_fee_ppm


def test_node_drain_bias_enabled_balanced_node_matches_disabled(mock_plugin, mock_database):
    """enabled + a perfectly balanced node (receivable ratio >= target ->
    zero pressure) must produce the SAME target as disabled: pressure=0
    means effective cap == static cap == 0.0, so no discount applies."""
    fc_off, _ = _make_node_bias_fc(
        mock_plugin, mock_database,
        node_drain_bias_enabled=False, node_receivable_ratio=0.50,
    )
    off_target, off_fee = _run_node_bias_cycle(fc_off)

    fc_on, _ = _make_node_bias_fc(
        mock_plugin, mock_database,
        node_drain_bias_enabled=True, node_receivable_ratio=0.50,
    )
    on_target, on_fee = _run_node_bias_cycle(fc_on)

    assert on_target == off_target
    assert on_fee == off_fee
