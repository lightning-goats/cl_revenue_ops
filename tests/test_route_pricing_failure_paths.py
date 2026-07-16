"""Route-pricing failure handling (2026-07-16 audit).

Two production defects found while diagnosing a week of failed defibrillator
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
    candidate.reason_code = "defibrillator"
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
# Fix 2: the defibrillator retries the shock from the next-best source when
# route pricing says the current source cannot route.
# ---------------------------------------------------------------------------
def _make_rebalancer(engine_side_effects, sources):
    """sources: list of (scid, spendable_sats) best-first."""
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    plugin = MagicMock()
    cfg = Config(dry_run=True, daily_budget_sats=1000,
                 diagnostic_rebalance_max_fee_sats=400)
    db = MagicMock()
    db.record_rebalance.return_value = 42
    r = EVRebalancer(plugin, cfg, db)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.execute_candidate.side_effect = engine_side_effects
    channels = {
        "200x1x0": {"peer_id": "03" + "b" * 64, "spendable_sats": 10_000},
    }
    for i, (scid, spendable) in enumerate(sources):
        channels[scid] = {
            "peer_id": "02" + format(i, "x").rjust(2, "0") * 32,
            "spendable_sats": spendable,
            "fee_ppm": 100,
        }
    r._get_channels_with_balances = MagicMock(return_value=channels)
    r._check_capital_controls = MagicMock(return_value=True)
    r._estimate_inbound_fee = MagicMock(return_value=100)
    r._record_successful_rebalance_fee = MagicMock(return_value=120)
    return r


def _route_failure():
    return ExecutionResult(
        success=False,
        error=("route_pricing_failed: no_route: RPC call failed: getroutes "
               "code 206 Could not find route without excessive delays (market)"),
        amount_sats=50_000,
    )


def test_defibrillator_retries_next_source_on_route_failure():
    r = _make_rebalancer(
        engine_side_effects=[
            _route_failure(),
            ExecutionResult(success=True, fee_sats=120, fee_msat=120_000,
                            amount_sats=50_000),
        ],
        sources=[("101x1x0", 500_000), ("102x1x0", 400_000)],
    )

    res = r.diagnostic_rebalance("200x1x0")

    assert res.get("shock_status") == "completed"
    calls = r.rebalance_engine_v2.execute_candidate.call_args_list
    assert len(calls) == 2
    assert calls[0].args[0].from_channel == "101x1x0"
    assert calls[1].args[0].from_channel == "102x1x0"


def test_defibrillator_does_not_retry_on_non_route_failure():
    r = _make_rebalancer(
        engine_side_effects=[
            ExecutionResult(
                success=False,
                error="local_execution_failed: sendpay crashed",
                amount_sats=50_000,
            ),
        ],
        sources=[("101x1x0", 500_000), ("102x1x0", 400_000)],
    )

    res = r.diagnostic_rebalance("200x1x0")

    assert res.get("shock_status") == "failed"
    assert r.rebalance_engine_v2.execute_candidate.call_count == 1


def test_defibrillator_source_attempts_are_capped():
    r = _make_rebalancer(
        engine_side_effects=[_route_failure(), _route_failure(), _route_failure()],
        sources=[
            ("101x1x0", 500_000),
            ("102x1x0", 400_000),
            ("103x1x0", 300_000),
            ("104x1x0", 200_000),
            ("105x1x0", 150_000),
        ],
    )

    res = r.diagnostic_rebalance("200x1x0")

    assert res.get("shock_status") == "failed"
    # Bounded: 3 source attempts max even with 5 valid sources.
    assert r.rebalance_engine_v2.execute_candidate.call_count == 3
