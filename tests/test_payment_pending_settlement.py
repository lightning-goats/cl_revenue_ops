"""Tests for in-flight payment (pending settlement) handling.

A waitsendpay timeout — whether CLN's code 200 or the RPC proxy's deadline —
means "payment unresolved", not "payment failed". The executor must mark the
result payment_pending instead of cleaning up, the engine must keep the
budget reservation held and skip retries, and a reconciliation sweep must
settle the row once listsendpays reports a terminal state.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules.rebalance_execution import ExecutionResult
from modules.rebalance_native_executor_v2 import NativeRouteExecutor


OUR_ID = "03" + "a" * 64
SRC_PEER = "02" + "b" * 64


class FakeRpc:
    """Legacy-style rpc.call(method, params) — no timeout kwarg support."""

    def __init__(self, responses=None, failures=None):
        self.responses = responses or {}
        self.failures = failures or {}
        self.calls = []

    def call(self, method, params=None):
        params = params or {}
        self.calls.append((method, params))
        if method in self.failures:
            raise self.failures[method]
        value = self.responses.get(method, {})
        return value(params) if callable(value) else value


class TimeoutAwareFakeRpc(FakeRpc):
    """Proxy-style rpc.call(method, params, timeout=...) recording kwargs."""

    def __init__(self, responses=None, failures=None):
        super().__init__(responses, failures)
        self.timeouts = {}

    def call(self, method, params=None, *, timeout=None):
        self.timeouts[method] = timeout
        return super().call(method, params)


class FakePlugin:
    def __init__(self, rpc):
        self.rpc = rpc
        self.logs = []

    def log(self, message, level="info"):
        self.logs.append((level, message))


class FakeRpcError(Exception):
    def __init__(self, error):
        super().__init__(error.get("message", "rpc error"))
        self.error = error


class RPCTimeoutError(Exception):
    """Mirrors the proxy timeout class in cl-revenue-ops.py (matched by name)."""

    def __init__(self, method):
        super().__init__(f"RPC timeout for method: {method}")


def _route(amount_sats=100, first_msat=101_000):
    return [
        {
            "id": SRC_PEER,
            "channel": "100x1x0",
            "direction": 0,
            "amount_msat": first_msat,
            "delay": 40,
            "style": "tlv",
        },
        {
            "id": OUR_ID,
            "channel": "200x1x0",
            "direction": 1,
            "amount_msat": amount_sats * 1000,
            "delay": 18,
            "style": "tlv",
        },
    ]


def _base_responses():
    return {
        "getinfo": {"id": OUR_ID},
        "invoice": {
            "payment_hash": "hash-1",
            "bolt11": "lnbc-test",
            "payment_secret": "secret-1",
        },
        "sendpay": {"status": "pending"},
        "delpay": {"ok": True},
        "delinvoice": {"ok": True},
    }


def _execute(executor):
    return executor.execute(
        route=_route(),
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=1,
    )


# ---------------------------------------------------------------------------
# Executor: pending classification
# ---------------------------------------------------------------------------


def test_waitsendpay_cln_code_200_marks_payment_pending():
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={
            "waitsendpay": FakeRpcError(
                {"code": 200, "message": "Timed out while waiting"}
            )
        },
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.success is False
    assert result.payment_pending is True
    assert "payment_pending_timeout" in result.error
    assert result.excluded_channels == []
    assert result.failure_data.get("payment_hash") == "hash-1"
    # The HTLC may still settle: invoice and payment must NOT be deleted.
    methods = [method for method, _ in rpc.calls]
    assert "delpay" not in methods
    assert "delinvoice" not in methods


def test_waitsendpay_proxy_timeout_marks_payment_pending():
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"waitsendpay": RPCTimeoutError("waitsendpay")},
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.payment_pending is True
    assert "payment_pending_timeout" in result.error
    methods = [method for method, _ in rpc.calls]
    assert "delpay" not in methods
    assert "delinvoice" not in methods


def test_sendpay_proxy_timeout_marks_payment_pending():
    # A proxy timeout on sendpay itself leaves the payment state unknown —
    # the safe interpretation is "possibly in flight".
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"sendpay": RPCTimeoutError("sendpay")},
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.payment_pending is True
    methods = [method for method, _ in rpc.calls]
    assert "delpay" not in methods
    assert "delinvoice" not in methods


def test_pending_result_carries_planned_fee_for_budget_hold():
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={
            "waitsendpay": FakeRpcError(
                {"code": 200, "message": "Timed out while waiting"}
            )
        },
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.fee_msat == 1000  # planned route fee
    assert result.fee_sats == 1


def test_invoice_failure_is_not_payment_pending():
    # Failure before sendpay means no payment can be in flight.
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"invoice": RPCTimeoutError("invoice")},
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.success is False
    assert result.payment_pending is False


def test_terminal_waitsendpay_failure_still_cleans_up():
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={
            "waitsendpay": FakeRpcError(
                {
                    "code": 204,
                    "message": "failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
                    "data": {
                        "erring_channel": "100x1x0",
                        "erring_direction": 0,
                    },
                }
            )
        },
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.success is False
    assert result.payment_pending is False
    methods = [method for method, _ in rpc.calls]
    assert "delpay" in methods
    assert "delinvoice" in methods


# ---------------------------------------------------------------------------
# Executor: proxy deadline must cover the waitsendpay window
# ---------------------------------------------------------------------------


def test_waitsendpay_passes_timeout_kwarg_to_proxy():
    responses = _base_responses()
    responses["waitsendpay"] = {
        "status": "complete",
        "amount_sent_msat": "101000msat",
    }
    rpc = TimeoutAwareFakeRpc(responses=responses)
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.success is True
    assert rpc.timeouts.get("waitsendpay") == NativeRouteExecutor.SENDPAY_TIMEOUT_SEC


def test_waitsendpay_timeout_kwarg_falls_back_for_legacy_rpc():
    # Plain rpc.call(method, params) without a timeout kwarg must still work.
    responses = _base_responses()
    responses["waitsendpay"] = {
        "status": "complete",
        "amount_sent_msat": "101000msat",
    }
    rpc = FakeRpc(responses=responses)
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = _execute(executor)

    assert result.success is True
    methods = [method for method, _ in rpc.calls]
    assert methods == ["getinfo", "invoice", "sendpay", "waitsendpay"]


# ---------------------------------------------------------------------------
# Engine: budget hold, retry guards, cooldown, reconciliation
# ---------------------------------------------------------------------------


def _make_engine(plugin, database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": OUR_ID}
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


def _pending_result(**overrides):
    fields = dict(
        success=False,
        payment_pending=True,
        amount_sats=100_000,
        fee_sats=5,
        fee_msat=5000,
        error="payment_pending_timeout: RPC timeout for method: waitsendpay",
        failure_data={"payment_hash": "hash-1"},
    )
    fields.update(overrides)
    return ExecutionResult(**fields)


def test_finish_execution_budget_keeps_reservation_on_pending(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)

    engine._finish_execution_budget(
        reservation_id="42",
        reserved_budget=True,
        result=_pending_result(),
    )

    mock_database.release_budget_reservation.assert_not_called()
    mock_database.mark_budget_spent.assert_not_called()


def test_record_rebalance_result_marks_pending_settlement(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=OUR_ID,
        amount_sats=100_000,
        pair_budget_sats=100,
    )

    engine._record_rebalance_result(7, _pending_result(), pair=pair, account_costs=True)

    args, kwargs = mock_database.update_rebalance_result.call_args
    assert args[0] == 7
    assert args[1] == "pending_settlement"
    assert kwargs.get("payment_hash") == "hash-1"
    # No cost is recorded until the payment actually settles.
    mock_database.record_rebalance_cost.assert_not_called()


def test_retries_skipped_when_payment_pending(mock_plugin, mock_database):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=OUR_ID,
        amount_sats=100_000,
        pair_budget_sats=100,
    )
    executor = MagicMock()
    # Exclusions present would normally allow an exclusion retry.
    pending = _pending_result(excluded_channels=["150x1x0/1"])

    out1 = engine._retry_native_pair_with_exclusions(pair, executor, pending)
    out2 = engine._retry_native_pair_with_partial_amounts(pair, executor, pending)

    assert out1 is pending
    assert out2 is pending
    executor.execute.assert_not_called()


def test_payment_pending_cooldown_is_at_least_an_hour(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    assert engine._pair_failure_cooldowns["payment_pending_timeout"] >= 3600


def test_run_cycle_invokes_pending_settlement_reconciliation(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine.reconcile_pending_settlements = MagicMock(return_value=0)
    engine.find_candidates = MagicMock(return_value=[])

    engine.run_cycle()

    engine.reconcile_pending_settlements.assert_called_once()


def _reconcile_row(**overrides):
    row = {
        "id": 42,
        "from_channel": "100x1x0",
        "to_channel": "200x1x0",
        "amount_sats": 100_000,
        "payment_hash": "hash-1",
        "timestamp": int(time.time()) - 900,
    }
    row.update(overrides)
    return row


def test_reconcile_settled_payment_records_cost_and_marks_spent(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_pending_settlement_rebalances.return_value = [_reconcile_row()]
    mock_plugin.rpc.call.return_value = {
        "payments": [
            {
                "status": "complete",
                "amount_msat": 100_000_000,
                "amount_sent_msat": 100_005_000,
            }
        ]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 1
    args, kwargs = mock_database.update_rebalance_result.call_args
    assert args[0] == 42
    assert args[1] == "success"
    assert kwargs.get("actual_fee_msat") == 5000
    cost_kwargs = mock_database.record_rebalance_cost.call_args.kwargs
    assert cost_kwargs["channel_id"] == "200x1x0"
    assert cost_kwargs["cost_msat"] == 5000
    mock_database.mark_budget_spent.assert_called_once_with("42", 5)


def test_reconcile_partial_fill_settles_with_actual_amount(
    mock_plugin, mock_database
):
    # A partial-amount retry parked with the ORIGINAL planned amount
    # (1,000,000 sats) settles for only 250,000 sats. The reconcile sweep
    # must correct both the history row and the cost record to the settled
    # amount, or cost-ppm stats and the fill-fraction destination cooldown
    # anchor are computed against an inflated amount.
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_pending_settlement_rebalances.return_value = [
        _reconcile_row(amount_sats=1_000_000)
    ]
    mock_plugin.rpc.call.return_value = {
        "payments": [
            {
                "status": "complete",
                "amount_msat": 250_000_000,
                "amount_sent_msat": 250_005_000,
            }
        ]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 1
    args, kwargs = mock_database.update_rebalance_result.call_args
    assert args[0] == 42
    assert args[1] == "success"
    assert kwargs.get("amount_sats") == 250_000
    cost_kwargs = mock_database.record_rebalance_cost.call_args.kwargs
    assert cost_kwargs["amount_sats"] == 250_000
    mock_database.mark_budget_spent.assert_called_once_with("42", 5)


def test_reconcile_settled_amount_missing_falls_back_to_planned(
    mock_plugin, mock_database
):
    # If listsendpays lacks a parseable amount_msat, keep the original
    # planned amount rather than zeroing the anchor (and never crash).
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_pending_settlement_rebalances.return_value = [
        _reconcile_row(amount_sats=1_000_000)
    ]
    mock_plugin.rpc.call.return_value = {
        "payments": [
            {
                "status": "complete",
                "amount_msat": "garbage",
                "amount_sent_msat": 1_000_005_000,
            }
        ]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 1
    args, kwargs = mock_database.update_rebalance_result.call_args
    assert args[1] == "success"
    assert kwargs.get("amount_sats") == 1_000_000
    cost_kwargs = mock_database.record_rebalance_cost.call_args.kwargs
    assert cost_kwargs["amount_sats"] == 1_000_000


def test_reconcile_failed_payment_releases_reservation(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_pending_settlement_rebalances.return_value = [_reconcile_row()]
    mock_plugin.rpc.call.return_value = {
        "payments": [{"status": "failed", "amount_msat": 100_000_000}]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 1
    args, _kwargs = mock_database.update_rebalance_result.call_args
    assert args[0] == 42
    assert args[1] == "failed"
    mock_database.release_budget_reservation.assert_called_once_with("42")
    mock_database.record_rebalance_cost.assert_not_called()


def test_reconcile_missing_payment_treated_as_failed(mock_plugin, mock_database):
    # No payment with this hash exists: sendpay never went through.
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_pending_settlement_rebalances.return_value = [_reconcile_row()]
    mock_plugin.rpc.call.return_value = {"payments": []}

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 1
    args, _kwargs = mock_database.update_rebalance_result.call_args
    assert args[1] == "failed"
    mock_database.release_budget_reservation.assert_called_once_with("42")


def test_reconcile_still_pending_payment_left_untouched(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_pending_settlement_rebalances.return_value = [_reconcile_row()]
    mock_plugin.rpc.call.return_value = {
        "payments": [{"status": "pending", "amount_msat": 100_000_000}]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 0
    mock_database.update_rebalance_result.assert_not_called()
    mock_database.release_budget_reservation.assert_not_called()
    mock_database.mark_budget_spent.assert_not_called()


# ---------------------------------------------------------------------------
# Database: pending_settlement persistence
# ---------------------------------------------------------------------------


@pytest.fixture
def real_database(tmp_path):
    from modules.database import Database

    plugin = MagicMock()
    plugin.log = MagicMock()
    db = Database(str(tmp_path / "test.db"), plugin)
    db.initialize()
    return db


def test_database_stores_and_lists_pending_settlement(real_database):
    rebalance_id = real_database.record_rebalance(
        from_channel="100x1x0",
        to_channel="200x1x0",
        amount_sats=100_000,
        max_fee_sats=100,
        expected_profit_sats=0,
        status="pending",
    )

    real_database.update_rebalance_result(
        rebalance_id,
        "pending_settlement",
        error_message="payment_pending_timeout",
        payment_hash="hash-abc",
    )

    rows = real_database.get_pending_settlement_rebalances()
    assert len(rows) == 1
    assert rows[0]["id"] == rebalance_id
    assert rows[0]["payment_hash"] == "hash-abc"
    assert rows[0]["to_channel"] == "200x1x0"

    # Terminal update clears it from the pending queue.
    real_database.update_rebalance_result(rebalance_id, "success", actual_fee_msat=5000)
    assert real_database.get_pending_settlement_rebalances() == []


def test_database_late_settlement_corrects_amount_in_history_row(real_database):
    # Parked with the planned 1,000,000 sats; the sweep later reports the
    # payment settled for only 250,000. The history row must reflect the
    # settled amount, not the planned one.
    rebalance_id = real_database.record_rebalance(
        from_channel="100x1x0",
        to_channel="200x1x0",
        amount_sats=1_000_000,
        max_fee_sats=100,
        expected_profit_sats=0,
        status="pending",
    )
    real_database.update_rebalance_result(
        rebalance_id,
        "pending_settlement",
        error_message="payment_pending_timeout",
        payment_hash="hash-partial",
    )

    real_database.update_rebalance_result(
        rebalance_id,
        "success",
        actual_fee_msat=5000,
        amount_sats=250_000,
    )

    row = real_database._get_connection().execute(
        "SELECT status, amount_sats FROM rebalance_history WHERE id = ?",
        (rebalance_id,),
    ).fetchone()
    assert row["status"] == "success"
    assert row["amount_sats"] == 250_000


# ---------------------------------------------------------------------------
# Task 26/78: the pending_settlement hold must be BOUNDED in practice.
# The sweep exemption (never blind-release a sweepable in-flight payment) is
# correct, but head-of-line blocking and silence made the hold effectively
# infinite: ORDER BY timestamp ASC LIMIT 50 starves newer rows behind 50
# persistently-pending older ones, and nothing ever escalated.
# ---------------------------------------------------------------------------

def test_sweep_rotates_past_persistently_pending_rows(mock_plugin, mock_database):
    """Newer rows must reconcile even when a full page of older rows keeps
    answering 'pending' — the sweep pages forward instead of re-scanning
    the same starved prefix forever."""
    engine = _make_engine(mock_plugin, mock_database)
    engine._pending_settlement_page = 2

    old_a = _reconcile_row(id=1, payment_hash="hash-a")
    old_b = _reconcile_row(id=2, payment_hash="hash-b")
    fresh = _reconcile_row(id=3, payment_hash="hash-c")

    def getter(limit=50, offset=0):
        rows = [old_a, old_b, fresh]
        return rows[offset:offset + limit]

    mock_database.get_pending_settlement_rebalances.side_effect = getter

    def rpc_call(method, params=None):
        assert method == "listsendpays"
        if params["payment_hash"] == "hash-c":
            return {"payments": [{
                "status": "complete",
                "amount_msat": 100_000_000,
                "amount_sent_msat": 100_005_000,
            }]}
        return {"payments": [{"status": "pending"}]}

    mock_plugin.rpc.call.side_effect = rpc_call

    # Sweep 1 visits the starved prefix (ids 1,2): nothing resolves.
    assert engine.reconcile_pending_settlements() == 0
    # Sweep 2 must move PAST the full unresolved page and reach id 3.
    resolved_second = engine.reconcile_pending_settlements()
    assert resolved_second == 1, (
        "a full page of persistently-pending rows must not starve newer rows")
    args, _ = mock_database.update_rebalance_result.call_args
    assert args[0] == 3


def test_sweep_offset_resets_after_reaching_the_end(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    engine._pending_settlement_page = 2
    rows = [_reconcile_row(id=i, payment_hash=f"h{i}") for i in (1, 2, 3)]

    def getter(limit=50, offset=0):
        return rows[offset:offset + limit]

    mock_database.get_pending_settlement_rebalances.side_effect = getter
    mock_plugin.rpc.call.return_value = {"payments": [{"status": "pending"}]}

    engine.reconcile_pending_settlements()   # page 1 (offset 0)
    engine.reconcile_pending_settlements()   # page 2 (offset 2, short page)
    engine.reconcile_pending_settlements()   # must be back at offset 0
    offsets = [c.kwargs.get("offset", 0)
               for c in mock_database.get_pending_settlement_rebalances.call_args_list]
    assert offsets == [0, 2, 0], offsets


def test_stale_pending_settlement_escalates_loudly(mock_plugin, mock_database):
    """A row older than the max hold age that STILL answers 'pending' is an
    anomaly (max CLTV is ~2 weeks) — it must escalate at error level with
    the row id, not stay a silent warn-forever."""
    engine = _make_engine(mock_plugin, mock_database)
    stale = _reconcile_row(id=7, timestamp=int(time.time()) - 20 * 86400)
    mock_database.get_pending_settlement_rebalances.return_value = [stale]
    mock_plugin.rpc.call.return_value = {"payments": [{"status": "pending"}]}

    engine.reconcile_pending_settlements()

    errors = [c for c in mock_plugin.log.call_args_list
              if (c.kwargs.get("level") or (c.args[1] if len(c.args) > 1 else "")) == "error"
              and "pending_settlement" in str(c.args[0])
              and "7" in str(c.args[0])]
    assert errors, "a stale held reservation must be operator-visible at error level"
    # The reservation itself must STILL be held — escalation is visibility,
    # never an automatic release of a possibly-live payment.
    mock_database.release_budget_reservation.assert_not_called()
