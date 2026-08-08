"""Audit 2026-08-01 wave2 FIX 2 + FIX 3.

FIX 2: pair-level failure/success bookkeeping (in-memory futility tracker +
persisted cooldowns) must run in the execution WORKER itself, not only in
the cycle thread's consume_future_result — a worker still running at the
120s as_completed ceiling is orphaned and its bookkeeping never happened,
so a chronically slow-failing pair was re-selected every cycle forever.
The recording is idempotent per ExecutionResult so the cycle thread's own
call for non-orphaned futures cannot double-count.

FIX 3: governor rejections (``governor_block: ...``) never attempted a
route. They must be classified like budget blocks: history status
'skipped', no futility strike, no persisted cooldown — and the
defibrillator in rebalancer.py must not rewrite such rows to 'failed'.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rebalance_execution import ExecutionResult  # noqa: E402


OUR_ID = "03" + "u" * 64
SRC_PEER = "02" + "b" * 64
DST_PEER = "02" + "c" * 64


def _make_engine(plugin=None, database=None):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = plugin or MagicMock()
    database = database if database is not None else MagicMock()
    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    database.record_rebalance.return_value = 77
    database.reserve_budget.return_value = (True, 9_999)
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def _pair():
    from modules.rebalance_types_v2 import PairCandidate

    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        reason_code="ev_positive",
        route=None,
    )


KEY = ("100x1x0", "200x1x0")


# ---------------------------------------------------------------------------
# FIX 2: worker-side recording
# ---------------------------------------------------------------------------


def test_worker_records_pair_failure_without_cycle_thread():
    """_execute_pair(record_pair_outcome=True) does the bookkeeping itself:
    an orphaned worker (whose consume_future_result never runs) still
    charges the futility strike and the persisted cooldown."""
    engine = _make_engine()
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=False, error="retriable_failure: NoRoutes",
    )

    engine._execute_pair(_pair(), executor, record_pair_outcome=True)

    assert len(engine._pair_failures.get(KEY, [])) == 1
    engine.database.record_pair_rebalance_failure.assert_called_once()


def test_worker_records_pair_success_and_clears_persisted():
    engine = _make_engine()
    engine._record_pair_failure(*KEY)
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, amount_sats=50_000, fee_sats=1, fee_msat=1_000,
    )

    engine._execute_pair(_pair(), executor, record_pair_outcome=True)

    assert KEY not in engine._pair_failures
    engine.database.clear_pair_rebalance_failure.assert_called_once()


def test_worker_recording_default_off_for_manual_paths():
    """Manual execute_candidate callers never recorded pair futility;
    the default keeps that behavior."""
    engine = _make_engine()
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=False, error="retriable_failure: NoRoutes",
    )

    engine._execute_pair(_pair(), executor)

    assert KEY not in engine._pair_failures
    engine.database.record_pair_rebalance_failure.assert_not_called()


def test_record_pair_outcome_is_idempotent_per_result():
    """Worker records first; the cycle thread's later call for the same
    (non-orphaned) result must not double-count."""
    engine = _make_engine()
    result = ExecutionResult(success=False, error="no_route")
    pair = _pair()

    engine._record_pair_outcome(pair, result)   # worker
    engine._record_pair_outcome(pair, result)   # consume_future_result

    assert len(engine._pair_failures.get(KEY, [])) == 1
    assert engine.database.record_pair_rebalance_failure.call_count == 1


def test_run_cycle_records_exactly_one_strike_per_failed_pair():
    """End-to-end: worker + consume both run; only one strike lands."""
    engine = _make_engine()
    pair = _pair()
    pair.route = [{"channel": "999x1x0"}]
    engine.find_candidates = MagicMock(return_value=[pair])
    executor = MagicMock()
    executor.is_available.return_value = True
    executor.execute.return_value = ExecutionResult(
        success=False, error="retriable_failure: NoRoutes",
    )
    engine._make_executor = MagicMock(return_value=executor)
    engine.reconcile_pending_settlements = MagicMock(return_value=0)

    engine.run_cycle()

    assert len(engine._pair_failures.get(KEY, [])) == 1
    assert engine.database.record_pair_rebalance_failure.call_count == 1


def test_worker_records_failure_even_when_worker_raises():
    """An exception escaping the inner execution still charges the strike
    in the worker's finally (the orphan case cannot rely on the cycle
    thread seeing the exception)."""
    engine = _make_engine()
    executor = MagicMock()
    executor.is_available.return_value = True
    engine._pair_policy_allowed = MagicMock(
        side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        engine._execute_pair(_pair(), executor, record_pair_outcome=True)

    assert len(engine._pair_failures.get(KEY, [])) == 1


def test_budget_block_still_exempt_from_strikes():
    engine = _make_engine()
    pair = _pair()
    result = ExecutionResult(
        success=False,
        error="local_budget_block: 0 sats remaining of 100 unified budget",
    )

    engine._record_pair_outcome(pair, result)

    assert KEY not in engine._pair_failures
    engine.database.record_pair_rebalance_failure.assert_not_called()


# ---------------------------------------------------------------------------
# FIX 3: governor rejections classified like budget blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error", [
    "governor_block: PAUSED",
    "governor_block: INTENT_STALE",
    "governor_block: AUTHORITY_LEVEL_BLOCKED",
    "governor_block: INTENT_SUPERSEDED",
    "governor_block: CONFLICT_SAME_CHANNEL",
    "governor_block: internal_error (boom)",
])
def test_governor_block_is_a_budget_style_block(error):
    from modules.rebalance_engine_v2 import RebalanceEngine

    result = ExecutionResult(success=False, error=error)
    assert RebalanceEngine._is_budget_block(result) is True


def test_routing_failures_are_not_governor_blocks():
    from modules.rebalance_engine_v2 import RebalanceEngine

    for error in ("no_route", "retriable_failure: NoRoutes",
                  "executor_error: boom"):
        result = ExecutionResult(success=False, error=error)
        assert RebalanceEngine._is_budget_block(result) is False


def test_governor_block_charges_no_strike_or_cooldown():
    engine = _make_engine()
    result = ExecutionResult(success=False, error="governor_block: PAUSED")

    engine._record_pair_outcome(_pair(), result)

    assert KEY not in engine._pair_failures
    engine.database.record_pair_rebalance_failure.assert_not_called()


def test_governor_block_recorded_as_skipped_in_history():
    engine = _make_engine()
    result = ExecutionResult(success=False, error="governor_block: INTENT_STALE")

    engine._record_rebalance_result(7, result, pair=_pair(),
                                    account_costs=False)

    args, _kwargs = engine.database.update_rebalance_result.call_args
    assert args[0] == 7
    assert args[1] == "skipped"


# ---------------------------------------------------------------------------
# FIX 3(b): the defibrillator must not rewrite 'skipped' rows to 'failed'
# ---------------------------------------------------------------------------
