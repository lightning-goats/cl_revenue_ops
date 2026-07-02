"""P1-008: init numeric options range-validated.

rpc-timeout-seconds must be > 0; daily/weekly budgets >= 0; reputation_decay
and htlc_congestion_threshold clamped to [0,1]. Out-of-range values are
clamped with a warning rather than silently accepted.
"""

import pytest

from tests.plugin_test_utils import load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


def _val(mod, **kw):
    return mod._validate_numeric_config_options(dict(kw), log=None)


def test_rpc_timeout_zero_rejected(mod):
    assert _val(mod, rpc_timeout_seconds=0)["rpc_timeout_seconds"] >= 1


def test_rpc_timeout_valid_unchanged(mod):
    assert _val(mod, rpc_timeout_seconds=30)["rpc_timeout_seconds"] == 30


def test_daily_budget_negative_clamped(mod):
    assert _val(mod, daily_budget_sats=-100)["daily_budget_sats"] >= 0


def test_weekly_budget_negative_clamped(mod):
    assert _val(mod, weekly_budget_sats=-100)["weekly_budget_sats"] >= 0


def test_reputation_decay_above_one_clamped(mod):
    assert _val(mod, reputation_decay=1.5)["reputation_decay"] == 1.0


def test_reputation_decay_below_zero_clamped(mod):
    assert _val(mod, reputation_decay=-0.5)["reputation_decay"] == 0.0


def test_reputation_decay_valid_unchanged(mod):
    assert _val(mod, reputation_decay=0.95)["reputation_decay"] == 0.95


def test_htlc_congestion_threshold_clamped(mod):
    assert _val(mod, htlc_congestion_threshold=2.0)["htlc_congestion_threshold"] == 1.0
