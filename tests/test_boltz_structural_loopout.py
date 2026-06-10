"""Structural loop-out: config, scarcity factor, credit, and envelope gate."""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


def test_config_defaults():
    from modules.config import Config
    cfg = Config()
    assert cfg.receivable_ratio_target == 0.30
    assert cfg.receivable_ratio_floor == 0.20
    assert cfg.boltz_structural_budget_sats_per_day == 0  # off by default
    assert cfg.drain_fee_discount_max == 0.0               # off by default


def test_config_validation_ranges():
    from modules.config import Config
    with pytest.raises(ValueError):
        Config(receivable_ratio_target=1.5)
    with pytest.raises(ValueError):
        Config(drain_fee_discount_max=0.9)


def test_config_floor_must_not_exceed_target():
    from modules.config import Config
    with pytest.raises(ValueError):
        Config(receivable_ratio_floor=0.4, receivable_ratio_target=0.3)


def test_plugin_options_registered():
    mod = load_plugin_module()
    for name in (
        "revenue-ops-receivable-ratio-target",
        "revenue-ops-receivable-ratio-floor",
        "revenue-ops-boltz-structural-budget-sats",
        "revenue-ops-drain-fee-discount-max",
    ):
        assert name in mod.plugin.options, name


def _channels_info(entries):
    """entries: list of (spendable_sats, receivable_sats)."""
    return {
        f"{i}x1x0": {
            "peer_id": "02" + ("%02d" % i) * 33,
            "spendable_msat": s * 1000,
            "receivable_msat": r * 1000,
            "capacity": s + r,
        }
        for i, (s, r) in enumerate(entries, start=100)
    }


def _scarcity_module(entries):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.fee_controller = MagicMock()
    mod.fee_controller._get_channels_info.return_value = _channels_info(entries)
    from modules.config import Config
    mod.config = Config()
    return mod


def test_receivable_status_starved_node():
    # 97% local everywhere => receivable_ratio 0.03, fully starved
    mod = _scarcity_module([(970_000, 30_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.03, abs=0.001)
    assert status["scarcity"] == pytest.approx(1.0)  # clamped at floor


def test_receivable_status_healthy_node():
    mod = _scarcity_module([(500_000, 500_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.5, abs=0.001)
    assert status["scarcity"] == 0.0


def test_receivable_status_scales_between_floor_and_target():
    # ratio 0.25 sits halfway between floor 0.20 and target 0.30
    mod = _scarcity_module([(750_000, 250_000)] * 4)
    status = mod._node_receivable_status()
    assert status["scarcity"] == pytest.approx(0.5, abs=0.01)


def test_receivable_status_safe_on_error():
    mod = _scarcity_module([])
    mod.fee_controller._get_channels_info.side_effect = RuntimeError("boom")
    status = mod._node_receivable_status()
    assert status == {"receivable_ratio": None, "scarcity": 0.0,
                      "total_capacity_sats": 0, "total_receivable_sats": 0}
