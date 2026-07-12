"""Golden: fee damping (_apply_damped_fee_target/_get_fee_step_cap) and
economic floor (_calculate_floor). Pure constraint math — deterministic."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.fee_controller import FeeController
from tests.golden.util import golden_check


PROFILE = SimpleNamespace(
    wake_cycle_max_delta_ratio=0.50,
    normal_cycle_max_delta_ratio=0.15,
    wake_cycle_min_delta_ppm=25,
    normal_cycle_min_delta_ppm=10,
)


@pytest.fixture
def fc():
    controller = FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())
    # Pin the fee profile so damping math is exercised with fixed inputs.
    controller._resolve_fee_profile = lambda cfg=None: ("golden", PROFILE)
    return controller


DAMPING_SCENARIOS = [
    # (name, current_ppm, target_ppm, woke_from_sleep)
    ("small_raise_within_cap", 1000, 1100, False),
    ("large_raise_clamped", 1000, 5000, False),
    ("large_cut_clamped", 2000, 100, False),
    ("wake_cycle_wider_cap", 1000, 5000, True),
    ("low_fee_min_delta_floor", 10, 500, False),
    ("no_change", 750, 750, False),
]


@pytest.mark.parametrize("name,current,target,woke", DAMPING_SCENARIOS)
def test_golden_damped_fee_target(fc, name, current, target, woke):
    applied, diag = fc._apply_damped_fee_target(current, target, woke)
    golden_check(f"fee/damping_{name}", {
        "inputs": {"current": current, "target": target, "woke": woke},
        "applied_fee_ppm": applied,
        "diag": diag,
    })


def test_damping_hand_computed_anchor(fc):
    """Non-golden anchor: 1000 -> 5000 normal cycle caps at
    1000 + max(10, ceil(1000*0.15)) = 1150."""
    applied, diag = fc._apply_damped_fee_target(1000, 5000, False)
    assert applied == 1150
    assert diag["cap_applied"] is True
    assert diag["cap_reason"] == "normal_cycle_delta_cap"


FLOOR_SCENARIOS = [
    ("defaults_no_chain_costs", 2_000_000, None, "local"),
    ("live_chain_costs_local", 2_000_000,
     {"open_cost_sats": 2500, "close_cost_sats": 1500}, "local"),
    ("remote_opener_cheaper", 2_000_000,
     {"open_cost_sats": 2500, "close_cost_sats": 1500}, "remote"),
    ("small_channel", 200_000,
     {"open_cost_sats": 2500, "close_cost_sats": 1500}, "local"),
]


@pytest.mark.parametrize("name,capacity,chain_costs,opener", FLOOR_SCENARIOS)
def test_golden_calculate_floor(fc, name, capacity, chain_costs, opener):
    floor_ppm = fc._calculate_floor(
        capacity, chain_costs=chain_costs, peer_id=None, opener=opener
    )
    golden_check(f"fee/floor_{name}", {
        "inputs": {"capacity_sats": capacity, "chain_costs": chain_costs,
                   "opener": opener},
        "floor_ppm": floor_ppm,
    })
