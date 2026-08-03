"""Route-pricing failure handling (2026-07-16 audit).

Route-pricing failure-path coverage for ordinary rebalance execution
shocks (every attempt recorded as ``native_route_invalid: missing_route``):

1. ``RebalanceEngine._execute_candidate_locked`` logged a failed route
   pricing and then FELL THROUGH to ``_execute_pair`` with ``pair.route=None``.
   The executor rejected the empty route as ``missing_route``, masking the
   real getroutes error (askrene 206 "excessive delays"), burning a budget
   reservation, and misclassifying the failure for futility/cooldown logic.
   Fix: return the pricing error immediately; the executor must never run.

2. ``diagnostic_rebalance`` (defibrillator) picked its shock source purely by
   highest spendable sats with no fallback. When that single source is
   unroutable (Tallship advertises 600/1201/2016 CLTV deltas, so askrene can
   never build a route from it), every daily shock failed forever.
   Fix: on a route-availability failure, retry with the next-best source
   (bounded number of attempts). Non-route failures must NOT retry — a
   payment that may have gone out must never be repeated blindly.
"""

import os
import sys
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402
from modules.rebalance_execution import ExecutionResult  # noqa: E402
from modules.rebalance_router_v2 import RouteResult  # noqa: E402


def _make_db(tmp_path, name="pricing.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _make_engine(db, budget=1000):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    cfg = Config(dry_run=True, daily_budget_sats=budget)
    engine = RebalanceEngine(plugin=plugin, config=cfg, database=db)
    engine.global_budget_limit_provider = lambda: {"effective_budget_sats": budget}
    engine.external_liquidity_cost_provider = lambda: {
        "spent_24h_sats": 0, "reserved_24h_sats": 0,
    }
    return engine


def _candidate():
    candidate = MagicMock()
    candidate.from_channel = "100x1x0"
    candidate.to_channel = "200x1x0"
    candidate.from_peer_id = "02" + "a" * 64
    candidate.to_peer_id = "03" + "b" * 64
    candidate.amount_sats = 50_000
    candidate.max_budget_sats = 400
    candidate.reason_code = "route_pricing_test"
    candidate.route_decision = None
    return candidate


# ---------------------------------------------------------------------------
# Fix 1: execute_candidate returns the pricing error; executor never runs.
# ---------------------------------------------------------------------------
def test_execute_candidate_returns_pricing_error_without_executing(tmp_path):
    db = _make_db(tmp_path)
    engine = _make_engine(db)

    pricing_error = (
        "no_route: RPC call failed: method: getroutes, error: "
        "{'code': 206, 'message': 'Could not find route without excessive delays'}"
    )
    engine._route_pair = MagicMock(
        return_value=(RouteResult(success=False, error=pricing_error), "market")
    )
    executor = MagicMock()
    engine._make_executor = MagicMock(return_value=executor)

    result = engine.execute_candidate(_candidate())

    assert result is not None
    assert result.success is False
    executor.execute.assert_not_called()
    # The real pricing error must survive, not a synthetic missing_route.
    assert "route_pricing_failed" in (result.error or "")
    assert "excessive delays" in (result.error or "")
    assert "missing_route" not in (result.error or "")


# ---------------------------------------------------------------------------
