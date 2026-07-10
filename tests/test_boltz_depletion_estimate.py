"""Depletion rewire: the balance plan uses flow_analysis.estimate_depletion_hours.

The old in-plan contrib/fee proxy was 7-8x off (it backed volume out of
revenue and an assumed fee). The plan now calls the flow module's pure
helper (correct day-fraction units) via a getattr fallback, keeping the
legacy proxy only when the helper is absent. The gate also drops from
kalman_ratio > 0.1 to > 0.02 so mild but persistent drains are anticipated.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.test_boltz_balance_plan_bias import _make_planner_module
from tests.test_boltz_structural_loopout import _make_prof_mock


def _depletion_module(kalman_ratio, kalman_velocity=0.001, contrib=30_000):
    mod = _make_planner_module()
    from modules.config import Config
    mod.config = Config()
    pa = MagicMock()
    pa.analyze_all_channels.return_value = None
    prof = _make_prof_mock()
    prof.revenue.total_contribution_sats = contrib
    pa.get_profitability.return_value = prof
    mod.profitability_analyzer = pa
    # 97% local => loop_out candidate
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "b" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        }
    }
    mod.database.get_all_channel_states.return_value = [
        {
            "channel_id": "100x1x0",
            "state": "source",
            "kalman_flow_ratio": kalman_ratio,
            "kalman_velocity": kalman_velocity,
        }
    ]
    return mod


def test_helper_used_when_present():
    mod = _depletion_module(kalman_ratio=0.12, kalman_velocity=0.002)
    calls = []

    def fake_helper(local_sats, capacity_sats, kalman_ratio, kalman_velocity):
        calls.append((local_sats, capacity_sats, kalman_ratio, kalman_velocity))
        return 4.2

    mod.flow_analysis_mod = SimpleNamespace(estimate_depletion_hours=fake_helper)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert calls == [(9_700_000, 10_000_000, 0.12, 0.002)]
    rec = plan["recommendations"][0]
    assert rec["score"]["predicted_depletion_hours"] == 4.2


def test_gate_lowered_to_0_02():
    """ratio 0.05 was below the old 0.1 gate; the helper must now run."""
    mod = _depletion_module(kalman_ratio=0.05)
    mod.flow_analysis_mod = SimpleNamespace(
        estimate_depletion_hours=MagicMock(return_value=12.0))

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    mod.flow_analysis_mod.estimate_depletion_hours.assert_called_once()
    assert plan["recommendations"][0]["score"]["predicted_depletion_hours"] == 12.0


def test_gate_still_blocks_noise():
    mod = _depletion_module(kalman_ratio=0.01)
    mod.flow_analysis_mod = SimpleNamespace(
        estimate_depletion_hours=MagicMock(return_value=12.0))

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    mod.flow_analysis_mod.estimate_depletion_hours.assert_not_called()
    assert plan["recommendations"][0]["score"]["predicted_depletion_hours"] is None


def test_legacy_proxy_kept_when_helper_absent():
    """No helper attr: the old contrib/fee proxy still runs (1000 sats/day
    revenue / 50ppm floor => 20M sats/day volume; x ratio 0.5 => 10M/day
    drain; 9.7M local => 23.28h -> 23.3 rounded)."""
    mod = _depletion_module(kalman_ratio=0.5)
    mod.flow_analysis_mod = SimpleNamespace()  # helper absent

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    rec = plan["recommendations"][0]
    assert rec["score"]["predicted_depletion_hours"] == pytest.approx(23.3, abs=0.05)


def test_real_flow_module_wired():
    """With the real flow_analysis module (helper landed), the audit example
    holds: 10M capacity, ratio 0.12, 9.7M local => ~190h, not the proxy's
    7-8x-off figure."""
    mod = _depletion_module(kalman_ratio=0.12, kalman_velocity=0.0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    rec = plan["recommendations"][0]
    hours = rec["score"]["predicted_depletion_hours"]
    # 9.7M / (0.12 * 10M) * 24 = 194h
    assert hours == pytest.approx(194.0, abs=1.0)
