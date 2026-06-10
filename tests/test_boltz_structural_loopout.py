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
