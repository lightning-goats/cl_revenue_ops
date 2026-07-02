"""P8-001: partial-amount retry must not double-pay on payment_pending.

`_retry_native_pair_with_partial_amounts` walks a descending list of partial
fill amounts, dispatching one ``executor.execute`` per amount. The original
code only checked ``retry_result.success`` per iteration and treated ANY
non-success — including ``payment_pending`` (an in-flight HTLC whose
sendpay/waitsendpay merely timed out) — as a definitive failure, then
dispatched ANOTHER payment at a smaller amount against the same budget
reservation. That is a double-fill.

The sibling ``_retry_native_pair_with_exclusions`` already guards the prior
result ("The first payment may still settle — never pay again on top"). Once
an intermediate partial attempt returns ``payment_pending`` the loop must
BREAK and return the pending result — no further ``executor.execute``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.rebalance_execution import ExecutionResult
from modules.rebalance_types_v2 import PairCandidate


OUR_ID = "03" + "a" * 64
SRC_PEER = "02" + "b" * 64


def _make_engine(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    mock_database.record_rebalance.return_value = 1
    mock_database.reserve_budget.return_value = (True, 9999)
    return RebalanceEngine(plugin=mock_plugin, config=cfg, database=mock_database)


def _pair():
    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=OUR_ID,
        amount_sats=100_000,
        pair_budget_sats=1_000,
    )


def _liquidity_failure():
    # Genuine liquidity failure that permits partial-amount retries.
    return ExecutionResult(
        success=False,
        payment_pending=False,
        amount_sats=100_000,
        route_type="native",
        error="temporary_channel_failure at hop 1",
        attempts=1,
    )


def _prime_partial_retry(engine):
    """Wire the engine so the partial-retry loop reaches ``executor.execute``."""
    engine._executor_mode = lambda: "native"
    engine._cycle_router = MagicMock()
    engine._hive_router = None
    engine._route_decision_for_pair = lambda pair: SimpleNamespace(policy=None)
    # Route always prices successfully and cheaply so the budget gate passes.
    route_result = SimpleNamespace(
        success=True,
        route=[{"id": OUR_ID, "channel": "200x1x0"}],
        route_cost_sats=1,
        probability_ppm=1_000_000,
        error="",
    )
    engine._route_pair = lambda **kw: (route_result, "test-route")
    engine._probability_adjusted_budget = lambda budget, prob: 10_000
    engine._per_attempt_fee_ceiling = lambda amount, budget: 10_000


def test_partial_retry_pending_does_not_double_pay(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    _prime_partial_retry(engine)

    executor = MagicMock()
    # The very first partial attempt comes back payment_pending (HTLC in flight).
    pending = ExecutionResult(
        success=False,
        payment_pending=True,
        amount_sats=50_000,
        route_type="native",
        error="payment_pending_timeout: RPC timeout for method: waitsendpay",
        failure_data={"payment_hash": "hash-1"},
        attempts=1,
    )
    executor.execute.return_value = pending

    result = engine._retry_native_pair_with_partial_amounts(
        _pair(), executor, _liquidity_failure()
    )

    # Exactly one payment was dispatched — no second fill on top of the
    # in-flight HTLC.
    assert executor.execute.call_count == 1
    # The pending result is returned so the reservation stays held.
    assert result is pending
    assert getattr(result, "payment_pending", False) is True


def test_partial_retry_genuine_failure_still_retries(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    _prime_partial_retry(engine)

    executor = MagicMock()
    # Every partial attempt genuinely fails (not pending) — the loop must keep
    # trying smaller amounts, exactly as before.
    executor.execute.return_value = ExecutionResult(
        success=False,
        payment_pending=False,
        amount_sats=50_000,
        route_type="native",
        error="temporary_channel_failure",
        attempts=1,
    )

    engine._retry_native_pair_with_partial_amounts(
        _pair(), executor, _liquidity_failure()
    )

    # Healthy retry path unchanged: more than one partial amount attempted.
    assert executor.execute.call_count > 1
