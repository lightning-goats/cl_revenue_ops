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


# ---------------------------------------------------------------------------
# Fix 1: ConfigSnapshot carries the four new drain-demand fields
# ---------------------------------------------------------------------------

def test_snapshot_carries_new_drain_fields():
    """ConfigSnapshot.from_config (via Config.snapshot) must include all four
    structural loop-out fields with the correct values."""
    from modules.config import Config, ConfigSnapshot

    cfg = Config(boltz_structural_budget_sats_per_day=500)
    snap = cfg.snapshot()

    assert snap.boltz_structural_budget_sats_per_day == 500
    assert snap.receivable_ratio_target == pytest.approx(0.30)
    assert snap.receivable_ratio_floor == pytest.approx(0.20)
    assert snap.drain_fee_discount_max == pytest.approx(0.0)


def test_snapshot_drain_fields_propagate_non_defaults():
    """Non-default values for all four fields survive the snapshot round-trip."""
    from modules.config import Config

    cfg = Config(
        receivable_ratio_target=0.40,
        receivable_ratio_floor=0.25,
        boltz_structural_budget_sats_per_day=1_000,
        drain_fee_discount_max=0.05,
    )
    snap = cfg.snapshot()

    assert snap.receivable_ratio_target == pytest.approx(0.40)
    assert snap.receivable_ratio_floor == pytest.approx(0.25)
    assert snap.boltz_structural_budget_sats_per_day == 1_000
    assert snap.drain_fee_discount_max == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Fix 2: Negative msat values are clamped to zero — scarcity stays fail-open
# ---------------------------------------------------------------------------

def _channels_info_raw(entries):
    """Like _channels_info but entries are (spendable_msat, receivable_msat) raw msat."""
    return {
        f"{i}x1x0": {
            "peer_id": "02" + ("%02d" % i) * 33,
            "spendable_msat": s,
            "receivable_msat": r,
            "capacity": (max(0, s) + max(0, r)) // 1000,
        }
        for i, (s, r) in enumerate(entries, start=200)
    }


def test_negative_receivable_msat_clamped_to_zero():
    """A single channel with receivable_msat=-900_000_000 mixed with healthy
    channels must not drive scarcity to 1.0 — the bad value is treated as 0."""
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.fee_controller = MagicMock()
    # Three healthy channels: 500k/500k sats each → ratio 0.50 normally
    # Plus one poisoned channel: spendable=500k sats, receivable=-900k sats (msat)
    healthy = [(500_000 * 1000, 500_000 * 1000)] * 3
    poisoned = [(500_000 * 1000, -900_000_000)]
    mod.fee_controller._get_channels_info.return_value = _channels_info_raw(
        healthy + poisoned
    )
    from modules.config import Config
    mod.config = Config()

    status = mod._node_receivable_status()
    # Negative receivable is zeroed; total_cap = (500k+500k)*3 + (500k+0)*1 = 3_500_000 sats
    # total_recv = 500k*3 + 0 = 1_500_000 sats; ratio ~ 0.4286 → above target → scarcity 0.0
    assert status["scarcity"] != 1.0, "negative msat must not force scarcity=1.0"
    assert status["scarcity"] == pytest.approx(0.0)
    assert status["receivable_ratio"] is not None
    assert status["receivable_ratio"] > 0.0


# ---------------------------------------------------------------------------
# Fix 3a: Zero-capacity channels dict → safe dict returned
# ---------------------------------------------------------------------------

def test_zero_capacity_returns_safe_dict():
    """All channels with spendable=0 and receivable=0 yields total_cap=0,
    triggering the safe-dict early return."""
    mod = _scarcity_module([(0, 0)] * 4)
    status = mod._node_receivable_status()
    assert status == {"receivable_ratio": None, "scarcity": 0.0,
                      "total_capacity_sats": 0, "total_receivable_sats": 0}


# ---------------------------------------------------------------------------
# Fix 3b: Exact boundary tests — ratio==target → scarcity 0.0, ratio==floor → scarcity 1.0
# ---------------------------------------------------------------------------

def test_scarcity_zero_at_exact_target():
    """When receivable_ratio == receivable_ratio_target (0.30), scarcity must be 0.0."""
    # target=0.30: recv / (spend+recv) = 0.30 → recv=3, spend=7 (sats ratio 3:7)
    # Use 300k receivable + 700k spendable per channel.
    mod = _scarcity_module([(700_000, 300_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.30, abs=1e-6)
    assert status["scarcity"] == pytest.approx(0.0, abs=1e-6)


def test_scarcity_one_at_exact_floor():
    """When receivable_ratio == receivable_ratio_floor (0.20), scarcity must be 1.0."""
    # floor=0.20: recv / (spend+recv) = 0.20 → recv=2, spend=8 (ratio 2:8)
    # Use 200k receivable + 800k spendable per channel.
    mod = _scarcity_module([(800_000, 200_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.20, abs=1e-6)
    assert status["scarcity"] == pytest.approx(1.0, abs=1e-6)
