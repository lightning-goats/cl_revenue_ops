"""Workstream H (planner): close-list batch arbitration — dedup,
selection-time conflict arming, legacy order preservation, fail-open."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.capacity_planner import CapacityPlanner

NOW = 1_752_400_000


def _planner(enabled=True, shadow=None):
    planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock())
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_cycle_planner_enabled=enabled)
    planner.config = cfg
    planner.econ_shadow = shadow
    return planner


def _loser(scid, roi=-90.0, reason="dead"):
    return {"scid": scid, "peer_id": "02" + "a" * 64,
            "marginal_roi": roi, "reason": reason}


def _summary():
    return {"skipped_reasons": []}


def test_flag_off_untouched():
    planner = _planner(enabled=False)
    losers = [_loser("100x1x0"), _loser("200x1x0")]
    assert planner._arbitrate_close_list(losers, _summary()) is losers


def test_legacy_order_preserved_among_survivors():
    planner = _planner()
    # Legacy order: worst ROI first — deliberately NOT target-sorted.
    losers = [_loser("900x1x0", roi=-95.0), _loser("100x1x0", roi=-40.0)]
    survivors = planner._arbitrate_close_list(list(losers), _summary())
    assert [l["scid"] for l in survivors] == ["900x1x0", "100x1x0"]


def test_duplicates_deduped_with_reason():
    planner = _planner()
    losers = [_loser("100x1x0", roi=-90.0), _loser("100x1x0", roi=-90.0)]
    summary = _summary()
    survivors = planner._arbitrate_close_list(losers, summary)
    assert len(survivors) == 1
    assert any("INTENT_SUPERSEDED" in r for r in summary["skipped_reasons"])


def test_selection_arms_rebalance_conflict(tmp_path):
    """The spec rule made real: once the planner SELECTS a channel for
    closure, a rebalance authorization into it is rejected — before any
    close reservation exists."""
    from modules.econ_shadow import EconShadow
    from modules.governor_facade import GovernorFacade
    from modules.econ_intents import Explanation, make_intent
    from modules.econ_types import Micro, Msat, SignedMsat, UnixTime

    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=True, econ_arbiter_enabled=True)
    cfg.db_path = str(tmp_path / "revenue_ops.db")
    shadow = EconShadow(MagicMock(), cfg,
                        ledger_path=str(tmp_path / "l.db"))
    planner = _planner(shadow=shadow)

    planner._arbitrate_close_list([_loser("111x222x0")], _summary())

    facade = GovernorFacade(
        reserve_spend=MagicMock(return_value=True),
        release_spend=MagicMock(return_value=True),
        is_paused=lambda: False,
        registry=shadow.arbitration_registry())
    import time as _time
    real_now = int(_time.time())
    reb = make_intent(
        intent_type="REBALANCE", snapshot_id="s1",
        created_at=UnixTime(real_now), expires_at=UnixTime(real_now + 600),
        target="111x222x0", amount_msat=Msat(500_000_000),
        expected_benefit_msat=SignedMsat(0), max_cost_msat=Msat(3_000),
        capital_committed_msat=Msat(500_000_000),
        confidence_micro=Micro(0), reason_codes=(),
        explanation=Explanation("t", (("x", 1),)), preconditions=(),
        priority=50, budget_bucket="rebalance", origin_policy="test",
        reversible=False)
    decision = facade.authorize(reb, real_now)
    assert decision.authorized is False
    assert decision.reason_code == "CONFLICT_CLOSE_REBALANCE"


def test_fail_open_on_internal_error(monkeypatch):
    planner = _planner()
    losers = [_loser("100x1x0")]
    monkeypatch.setattr("modules.econ_arbiter.arbitrate",
                        MagicMock(side_effect=RuntimeError("boom")))
    assert planner._arbitrate_close_list(losers, _summary()) is losers


def test_seam_is_wired_in_execute_cycle():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "modules" / "capacity_planner.py").read_text()
    seam = source.find("sorted_closeable = self._arbitrate_close_list(")
    loop = source.find("for loser in sorted_closeable:", seam)
    assert 0 < seam < loop, \
        "close arbitration must precede the close-execution loop"
