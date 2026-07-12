"""Golden: _compute_dynamic_htlcmax_msat + _htlcmax_delta_exceeds_deadband.
Pure functions of (cfg, channel_info, flow_state) — deterministic."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.fee_controller import FeeController
from tests.golden.util import golden_check


@pytest.fixture
def fc():
    return FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())


def _cfg(**over):
    base = dict(
        enable_dynamic_htlcmax=True,
        htlcmax_source_pct=0.85,
        htlcmax_sink_pct=0.25,
        htlcmax_balanced_pct=0.50,
    )
    base.update(over)
    return SimpleNamespace(**base)


SCENARIOS = [
    # (name, cfg_overrides, channel_info, flow_state)
    ("disabled_returns_none", {"enable_dynamic_htlcmax": False},
     {"capacity": 2_000_000, "spendable_msat": 1_000_000_000}, "source"),
    ("string_true_enables", {"enable_dynamic_htlcmax": "true"},
     {"capacity": 2_000_000, "spendable_msat": 1_000_000_000}, "source"),
    ("source_ample_spendable", {},
     {"capacity": 2_000_000, "spendable_msat": 1_900_000_000}, "source"),
    ("sink_small_share", {},
     {"capacity": 2_000_000, "spendable_msat": 1_900_000_000}, "sink"),
    ("balanced_mid_share", {},
     {"capacity": 2_000_000, "spendable_msat": 1_900_000_000}, "balanced"),
    ("depletion_caps_target", {},
     {"capacity": 2_000_000, "spendable_msat": 50_000_000}, "source"),
    ("floor_wins_when_depleted", {},
     {"capacity": 2_000_000, "spendable_msat": 1_000}, "source"),
    ("zero_capacity_none", {}, {"capacity": 0, "spendable_msat": 0}, "source"),
]


@pytest.mark.parametrize("name,over,chan,state", SCENARIOS)
def test_golden_htlcmax(fc, name, over, chan, state):
    result = fc._compute_dynamic_htlcmax_msat(_cfg(**over), chan, state)
    golden_check(f"htlcmax/{name}", {
        "inputs": {"cfg_overrides": over, "channel_info": chan,
                   "flow_state": state},
        "htlcmax_msat": result,
    })


def test_htlcmax_hand_computed_anchor(fc):
    """sink pct 0.25 of 2M sats capacity, ample spendable: result is
    min(uncapped, depletion cap) bounded below by the floor."""
    result = fc._compute_dynamic_htlcmax_msat(
        _cfg(), {"capacity": 2_000_000, "spendable_msat": 1_900_000_000},
        "sink",
    )
    expected_uncapped = int(2_000_000 * 1000 * 0.25)
    depletion_cap = int(1_900_000_000 * fc.HTLCMAX_DEPLETION_SPENDABLE_FRACTION)
    assert result == max(fc.HTLCMAX_FLOOR_MSAT,
                         min(expected_uncapped, depletion_cap))


DEADBAND_CASES = [
    ("equal_no_broadcast", 500_000_000, 500_000_000),
    ("zero_current_always", 500_000_000, 0),
    ("tiny_move", 501_000_000, 500_000_000),
    ("big_move", 900_000_000, 500_000_000),
]


@pytest.mark.parametrize("name,new,current", DEADBAND_CASES)
def test_golden_htlcmax_deadband(fc, name, new, current):
    golden_check(f"htlcmax/deadband_{name}", {
        "inputs": {"new_msat": new, "current_msat": current},
        "exceeds": fc._htlcmax_delta_exceeds_deadband(new, current),
    })
