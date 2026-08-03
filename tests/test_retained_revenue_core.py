"""Pin reporting and ordinary governed-rebalance behavior during retirement."""

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.database import Database
from modules.rebalance_engine_v2 import RebalanceEngine
from tests.plugin_test_utils import load_plugin_module


ROOT = Path(__file__).resolve().parents[1]
RETAINED_RPCS = {
    "revenue-status", "revenue-profitability", "revenue-history",
    "revenue-report", "revenue-dashboard", "revenue-health",
    "revenue-budget", "revenue-rebalance-debug", "revenue-rebalance-cycle",
    "revenue-rebalance",
}


def _database(tmp_path):
    plugin = MagicMock()
    db = Database(str(tmp_path / "revenue.db"), plugin)
    db.initialize()
    return db


def test_registered_method_inventory_contains_retained_core():
    source = (ROOT / "cl-revenue-ops.py").read_text(encoding="utf-8")
    actual = set(re.findall(r'@plugin\.method\(\s*"([a-z-]+)"', source))
    assert RETAINED_RPCS <= actual


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("revenue_status", {}), ("revenue_profitability", {}),
        ("revenue_history", {}), ("revenue_report", {}),
        ("revenue_dashboard", {}), ("revenue_health", {}),
        ("revenue_budget", {"section": "ledger"}),
        ("revenue_rebalance_debug", {}),
    ],
)
def test_read_only_surfaces_degrade_without_mutation(name, kwargs, monkeypatch):
    mod = load_plugin_module()
    for attr in (
        "database", "config", "fee_controller", "rebalancer",
        "profitability_analyzer", "policy_manager", "capex_engine",
        "data_service", "capacity_planner",
    ):
        monkeypatch.setattr(mod, attr, None)
    mutation_names = (
        "fundchannel", "close", "pay", "withdraw", "connect",
        "signmessage", "sendpay", "waitsendpay",
    )
    for mutation in mutation_names:
        getattr(mod.plugin.rpc, mutation).side_effect = AssertionError(
            f"read-only surface called mutation {mutation}"
        )
    result = getattr(mod, name)(mod.plugin, **kwargs)
    assert isinstance(result, dict)
    for mutation in mutation_names:
        getattr(mod.plugin.rpc, mutation).assert_not_called()


def test_generic_no_close_tag_survives_database_reload(tmp_path):
    db = _database(tmp_path)
    peer = "02" + "a" * 64
    db.upsert_policy(
        peer, "dynamic", "enabled", None, json.dumps(["no_close"]),
        int(time.time()), None, None, None,
    )
    db.close_all_connections()
    reopened = Database(db.db_path, MagicMock())
    reopened.initialize()
    row = reopened.get_policy(peer)
    assert row is not None
    assert "no_close" in json.loads(row["tags"])


def test_historical_boltz_spend_remains_in_generic_total(tmp_path):
    db = _database(tmp_path)
    db.record_spend_event(
        event_id="historical-boltz", category="boltz", amount_sats=321,
        source="historical-test",
    )
    summary = db.get_spend_ledger_summary(window_hours=24)
    assert summary["spent_24h_sats"] == 321
    assert summary["spent_by_category"]["boltz"] == 321


def _governed_engine(*, paused=False, reserve=(True, 900)):
    cfg = MagicMock(spec=Config)
    database = MagicMock()
    database.reserve_budget.return_value = reserve
    engine = RebalanceEngine(plugin=MagicMock(), config=cfg, database=database)
    engine._config_snapshot = lambda: SimpleNamespace(
        total_cost_budget_window_hours=24,
        weekly_budget_sats=35_000,
        allow_zero_cost_auto_rebalance_when_budget_zero=False,
        paused=paused,
        econ_governor_rebalance_enabled=True,
    )
    engine._get_global_budget_limit = lambda _cfg: 1_000
    engine._pair_max_fee_sats = lambda _pair: 100
    engine._executor_mode = lambda: "native"
    return engine


def _pair():
    return SimpleNamespace(
        source_channel_id="100x1x0", dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64, dest_peer_id="02" + "b" * 64,
        amount_sats=50_000, score=1.0,
    )


def test_ordinary_rebalance_pause_and_atomic_reservation_fail_closed():
    paused = _governed_engine(paused=True)
    ok, result = paused._reserve_execution_budget(_pair(), reservation_id="r-paused")
    assert ok is False and "PAUSED" in result.error
    paused.database.reserve_budget.assert_not_called()
    exhausted = _governed_engine(reserve=(False, 0))
    ok, result = exhausted._reserve_execution_budget(_pair(), reservation_id="r-budget")
    assert ok is False and "budget" in result.error
    exhausted.database.reserve_budget.assert_called_once()


def test_ordinary_rebalance_reserves_daily_weekly_and_global_rails_atomically():
    engine = _governed_engine()
    ok, result = engine._reserve_execution_budget(_pair(), reservation_id="r-ok")
    assert ok is True and result is None
    kwargs = engine.database.reserve_budget.call_args.kwargs
    assert kwargs["reservation_id"] == "r-ok"
    assert kwargs["amount_sats"] == 100
    assert kwargs["budget_limit"] == 1_000
    assert kwargs["weekly_budget_limit"] == 35_000
    assert kwargs["channel_id"] == "200x1x0"


def test_ordinary_rebalance_policy_gate_fails_closed():
    engine = _governed_engine()
    engine._policy_manager = None
    allowed, reason = engine._pair_policy_allowed(_pair())
    assert allowed is False and "fail closed" in reason

    passive = SimpleNamespace(strategy="passive", rebalance_mode="enabled")
    engine._policy_manager = MagicMock()
    engine._policy_manager.get_policy.return_value = passive
    allowed, reason = engine._pair_policy_allowed(_pair())
    assert allowed is False and "passive" in reason


def test_ordinary_rebalance_settles_success_and_releases_failure():
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _governed_engine()
    success = ExecutionResult(success=True, fee_sats=7)
    engine._finish_execution_budget(
        reservation_id="r-success", reserved_budget=True, result=success,
    )
    engine.database.mark_budget_spent.assert_called_once_with("r-success", 7)
    engine.database.release_budget_reservation.assert_not_called()

    engine.database.reset_mock()
    failure = ExecutionResult(success=False, error="no route")
    engine._finish_execution_budget(
        reservation_id="r-failure", reserved_budget=True, result=failure,
    )
    engine.database.release_budget_reservation.assert_called_once_with("r-failure")
    engine.database.mark_budget_spent.assert_not_called()


def test_rebalance_cycle_dispatch_cannot_drop_authority_or_budget_flags():
    source = (ROOT / "modules/rebalance_engine_v2.py").read_text(encoding="utf-8")
    assert "self._pair_policy_allowed(pair)" in source
    submit = source[source.index("future = self._pool.submit("):]
    submit = submit[:submit.index(")\n            futures[future]")]
    assert "reserve_budget=True" in submit
    assert "account_costs=True" in submit
