"""Workstream H cutover: batch arbitration inside the live rebalance
execution path, flag-gated and fail-open."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.rebalance_engine_v2 import CycleResult, RebalanceEngine


def _pair(src="100x1x0", dst="200x1x0", amount=500_000, budget=100):
    return SimpleNamespace(source_channel_id=src, dest_channel_id=dst,
                           amount_sats=amount, pair_budget_sats=budget,
                           score=0.298)


def _engine(cycle_enabled=True):
    engine = RebalanceEngine(plugin=MagicMock(),
                             config=MagicMock(spec=Config),
                             database=MagicMock())
    engine._config_snapshot = lambda: SimpleNamespace(
        econ_cycle_rebalance_enabled=cycle_enabled)
    return engine


def test_flag_off_returns_list_untouched():
    engine = _engine(cycle_enabled=False)
    pairs = [_pair(dst="900x1x0"), _pair(dst="100x1x0")]
    result = CycleResult()
    assert engine._arbitrate_execution_list(pairs, result) is pairs


def test_flag_check_is_strict():
    engine = RebalanceEngine(plugin=MagicMock(),
                             config=MagicMock(spec=Config),
                             database=MagicMock())
    engine._config_snapshot = lambda: MagicMock()  # truthy attrs
    assert engine._cycle_arbitration_enabled() is False


def test_ordering_applied_deterministically():
    engine = _engine()
    pairs = [_pair(dst="900x1x0"), _pair(dst="100x1x0"),
             _pair(dst="500x1x0")]
    result = CycleResult()
    ordered = engine._arbitrate_execution_list(list(pairs), result)
    assert [p.dest_channel_id for p in ordered] == \
        ["100x1x0", "500x1x0", "900x1x0"]
    assert result.audit_records == []


def test_duplicates_rejected_with_skip_records():
    engine = _engine()
    pairs = [_pair(), _pair()]  # identical -> identical intent keys
    result = CycleResult()
    ordered = engine._arbitrate_execution_list(pairs, result)
    assert len(ordered) == 1
    assert len(result.audit_records) == 1
    assert result.audit_records[0].reason == "arbitration:INTENT_SUPERSEDED"


def test_rejections_ledgered(tmp_path):
    from modules.econ_ledger import EconLedger
    from modules.econ_shadow import EconShadow
    engine = _engine()
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=True)
    cfg.db_path = str(tmp_path / "revenue_ops.db")
    engine.econ_shadow = EconShadow(
        MagicMock(), cfg, ledger_path=str(tmp_path / "l.db"))
    engine._arbitrate_execution_list([_pair(), _pair()], CycleResult())
    events = EconLedger(str(tmp_path / "l.db")).events()
    assert len(events) == 1
    assert events[0]["event_type"] == "intent_rejected"
    assert events[0]["details"]["batch"] is True


def test_fail_open_on_internal_error(monkeypatch):
    engine = _engine()
    pairs = [_pair()]
    monkeypatch.setattr(
        "modules.econ_cycle.rebalance_intent_pairs",
        MagicMock(side_effect=RuntimeError("boom")))
    result = CycleResult()
    assert engine._arbitrate_execution_list(pairs, result) is pairs


def test_seam_is_wired_in_run_cycle():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "modules" / "rebalance_engine_v2.py").read_text()
    seam = source.find("live_candidates = self._arbitrate_execution_list(")
    assert seam > 0, "arbitration seam missing from run_cycle"
    executor_after_seam = source.find(
        "executor = self._make_executor()", seam)
    assert executor_after_seam > seam, \
        "arbitration stage must precede run_cycle's executor dispatch"
