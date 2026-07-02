"""Phase-4 rebalance hardening regression tests.

Covers:
- P4-007: engine _finish_execution_budget must not strand an 'active' budget
  reservation when a payment is pending but carries NO payment_hash (the
  history row is marked 'failed' and the reconcile sweep — which only scans
  'pending_settlement' — will never release it). Release-on-failure.
- P4-008: an orphaned execution worker (still in _execute_pair after the
  cycle timeout released the cycle lock) must not let the next cycle
  re-select the same destination -> double-pay. In-flight-destination guard.
- P4-009: rebalancer.execute_rebalance must mirror the engine's
  hold-on-pending: do NOT release the budget reservation while a payment is
  pending settlement (sweepable), but DO release when terminal/non-sweepable.
- P4-011: the PRIORITY-2 gossip base-fee ppm-equivalent must be capped at
  1_000_000 like the PRIORITY-1 peer path.
- P4-012: with cfg.max_fee_ppm == 0 the historical-fee cap must zero the
  historical EV benefit term rather than leaving it uncapped.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Engine builder (mirrors tests/test_rebalance_engine_guards.py::_make_engine)
# ---------------------------------------------------------------------------

def _make_engine(plugin, database, *, dry_run=True):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=dry_run, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    database.record_rebalance.return_value = 1
    database.reserve_budget.return_value = (True, 9999)
    database.mark_budget_spent.return_value = True
    database.release_budget_reservation.return_value = True
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def _exec_result(**kwargs):
    from modules.rebalance_executor_v2 import ExecutionResult

    return ExecutionResult(**kwargs)


# =========================================================================
# P4-007: no-payment_hash pending edge must release, not strand
# =========================================================================

class TestP4007PendingWithoutHashReleases:
    def test_pending_without_payment_hash_releases_reservation(
        self, mock_plugin, mock_database
    ):
        engine = _make_engine(mock_plugin, mock_database)
        # payment_pending but NO payment_hash in failure_data.
        result = _exec_result(
            success=False,
            payment_pending=True,
            failure_data={},
            error="payment_pending_timeout",
        )

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )

        # The row is marked 'failed' (not 'pending_settlement'), so reconcile
        # never sweeps it: the reservation MUST be released here or it strands
        # 'active' forever.
        mock_database.release_budget_reservation.assert_called_once_with("777")
        mock_database.mark_budget_spent.assert_not_called()

    def test_pending_with_payment_hash_holds_reservation(
        self, mock_plugin, mock_database
    ):
        engine = _make_engine(mock_plugin, mock_database)
        # payment_pending WITH a payment_hash -> sweepable -> keep held.
        result = _exec_result(
            success=False,
            payment_pending=True,
            failure_data={"payment_hash": "ab" * 32},
            error="payment_pending_timeout",
        )

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )

        # Sweepable pending: neither released nor spent — the reconcile sweep
        # owns the terminal transition.
        mock_database.release_budget_reservation.assert_not_called()
        mock_database.mark_budget_spent.assert_not_called()

    def test_terminal_failure_still_releases(self, mock_plugin, mock_database):
        engine = _make_engine(mock_plugin, mock_database)
        result = _exec_result(success=False, error="no_route")

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )
        mock_database.release_budget_reservation.assert_called_once_with("777")

    def test_success_marks_spent(self, mock_plugin, mock_database):
        engine = _make_engine(mock_plugin, mock_database)
        result = _exec_result(success=True, fee_sats=3)

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )
        mock_database.mark_budget_spent.assert_called_once_with("777", 3)
        mock_database.release_budget_reservation.assert_not_called()
