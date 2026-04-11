"""Tests for exclude-aware retry in RebalanceEngine._execute_pair.

When the executor reports a retriable failure with excluded_channels
populated, the engine re-prices the pair with those channels excluded
and attempts one retry. This unblocks the recurring pattern where a
stale-gossip WIRE_FEE_INSUFFICIENT failure on the same intermediate
channel would otherwise loop every 15 minutes until the futility
tracker stops it — instead, the retry picks a different path and
the pair can actually succeed.

Origin: Phase B Task #5 on nexus-01 2026-04-10 18:39Z. Executor
correctly reports erring_channel=934667x311x8 after the RPC proxy
fix (PR #80), but without this retry the next cycle re-picks the
same failing route.
"""

from unittest.mock import MagicMock

import pytest


def _make_engine_with_mocked_router():
    """Build a minimal engine whose _cycle_router is a MagicMock."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_router_v2 import RouteResult

    plugin = MagicMock()
    plugin.rpc.call.side_effect = Exception("askrene unavailable")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}

    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    config.capex_probability_budget_bonus = 0.0
    del config.snapshot

    database = MagicMock()
    engine = RebalanceEngine(plugin=plugin, config=config, database=database)

    # Install a mock router as the cycle router so _execute_pair can call it
    mock_router = MagicMock()
    engine._cycle_router = mock_router
    return engine, mock_router


def _make_pair(
    source_scid="100x1x0",
    dest_scid="200x2x0",
    pair_budget_sats=500,
    amount_sats=1000,
):
    from modules.rebalance_types_v2 import PairCandidate
    return PairCandidate(
        source_channel_id=source_scid,
        dest_channel_id=dest_scid,
        source_peer_id="03" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=pair_budget_sats,
        score=0.5,
        source_local_ratio=0.9,
        dest_local_ratio=0.2,
        route=[{"id": "03" + "a" * 64, "channel": source_scid, "amount_msat": 1005000, "delay": 100}],
        route_cost_sats=5,
    )


def _make_execution_result(success: bool, error: str = "", excluded: list = None):
    from modules.rebalance_executor_v2 import ExecutionResult
    return ExecutionResult(
        success=success,
        attempts=1,
        fee_sats=5 if success else 0,
        fee_msat=5000 if success else 0,
        amount_sats=1000,
        error=error,
        excluded_channels=excluded or [],
    )


# ---------------------------------------------------------------------------
# Retry happy path
# ---------------------------------------------------------------------------


def test_execute_pair_retries_on_retriable_failure_with_exclude():
    """Retriable failure with excluded_channels → router re-priced with exclude,
    new route installed, second execution attempt made."""
    from modules.rebalance_router_v2 import RouteResult

    engine, mock_router = _make_engine_with_mocked_router()

    pair = _make_pair(pair_budget_sats=500)

    # First execute: fails retriably with an erring channel
    first_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["934667x311x8"],
    )
    # Second execute (after re-price): success
    second_success = _make_execution_result(success=True)

    executor = MagicMock()
    executor.execute.side_effect = [first_fail, second_success]

    # Re-price returns a cheaper route that fits the budget
    mock_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=7,
        final_hop_fee_ppm=100,
        hops=2,
        route=[
            {"id": "03" + "a" * 64, "channel": "100x1x0", "amount_msat": 1007000, "delay": 100},
            {"id": "03" + "u" * 64, "channel": "200x2x0", "amount_msat": 1000000, "delay": 40},
        ],
        probability_ppm=980000,
    )

    result = engine._execute_pair(pair, executor)

    # Two execute() calls: original + retry
    assert executor.execute.call_count == 2
    # Router was re-priced with the exclude list
    mock_router.price_pair.assert_called_once()
    call_kwargs = mock_router.price_pair.call_args.kwargs
    assert call_kwargs["exclude"] == ["934667x311x8"]
    assert call_kwargs["source_channel_id"] == "100x1x0"
    # Final result is the retry success
    assert result.success is True


def test_execute_pair_does_not_retry_on_permanent_failure():
    """Permanent failures (channel disabled, etc.) must not trigger retry."""
    engine, mock_router = _make_engine_with_mocked_router()
    pair = _make_pair()

    perm_fail = _make_execution_result(
        success=False,
        error="permanent_failure: WIRE_CHANNEL_DISABLED",
        excluded=["111x1x1"],
    )
    executor = MagicMock()
    executor.execute.return_value = perm_fail

    result = engine._execute_pair(pair, executor)

    assert executor.execute.call_count == 1
    mock_router.price_pair.assert_not_called()
    assert result.success is False
    assert "permanent_failure" in result.error


def test_execute_pair_does_not_retry_when_no_exclude_reported():
    """Retriable failure without excluded_channels can't be usefully retried."""
    engine, mock_router = _make_engine_with_mocked_router()
    pair = _make_pair()

    fail_no_exclude = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_TEMPORARY_NODE_FAILURE",
        excluded=[],
    )
    executor = MagicMock()
    executor.execute.return_value = fail_no_exclude

    result = engine._execute_pair(pair, executor)

    assert executor.execute.call_count == 1
    mock_router.price_pair.assert_not_called()
    assert result.success is False


def test_execute_pair_retry_fails_when_reprice_returns_no_route():
    """If the router can't find an alternative path with the exclude, the
    original failure stands."""
    from modules.rebalance_router_v2 import RouteResult

    engine, mock_router = _make_engine_with_mocked_router()
    pair = _make_pair()

    first_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["934667x311x8"],
    )
    executor = MagicMock()
    executor.execute.return_value = first_fail

    mock_router.price_pair.return_value = RouteResult(
        success=False, error="no_route: all alternatives excluded"
    )

    result = engine._execute_pair(pair, executor)

    assert executor.execute.call_count == 1  # no retry submitted
    mock_router.price_pair.assert_called_once()
    assert result.success is False
    # Original failure preserved
    assert "WIRE_FEE_INSUFFICIENT" in result.error


def test_execute_pair_retry_fails_when_reprice_over_budget():
    """If the alternative route exceeds the pair budget (even probability-adjusted),
    no retry is attempted."""
    from modules.rebalance_router_v2 import RouteResult

    engine, mock_router = _make_engine_with_mocked_router()
    pair = _make_pair(pair_budget_sats=10)  # very tight budget

    first_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["934667x311x8"],
    )
    executor = MagicMock()
    executor.execute.return_value = first_fail

    # Alternative route costs way more than budget
    mock_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=500,  # 50x the budget
        final_hop_fee_ppm=100,
        hops=2,
        route=[],
    )

    result = engine._execute_pair(pair, executor)

    assert executor.execute.call_count == 1  # no retry
    assert result.success is False


def test_execute_pair_retry_second_failure_returned():
    """If both original and retry fail, the retry's failure is returned
    (the retry is what the operator actually ran)."""
    from modules.rebalance_router_v2 import RouteResult

    engine, mock_router = _make_engine_with_mocked_router()
    pair = _make_pair()

    first_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["934667x311x8"],
    )
    second_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
        excluded=["222x2x0"],
    )

    executor = MagicMock()
    executor.execute.side_effect = [first_fail, second_fail]

    mock_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=7,
        final_hop_fee_ppm=100,
        hops=2,
        route=[
            {"id": "03" + "a" * 64, "channel": "100x1x0", "amount_msat": 1007000, "delay": 100},
            {"id": "03" + "u" * 64, "channel": "200x2x0", "amount_msat": 1000000, "delay": 40},
        ],
    )

    result = engine._execute_pair(pair, executor)

    assert executor.execute.call_count == 2
    assert result.success is False
    # The returned result should be the retry attempt's failure
    assert "WIRE_TEMPORARY_CHANNEL_FAILURE" in result.error


def test_execute_pair_retry_does_not_loop_infinitely():
    """Maximum one retry per pair per cycle — even if the second failure
    also has excluded_channels, no third attempt is made."""
    from modules.rebalance_router_v2 import RouteResult

    engine, mock_router = _make_engine_with_mocked_router()
    pair = _make_pair()

    first_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["aaa"],
    )
    second_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["bbb"],
    )

    executor = MagicMock()
    executor.execute.side_effect = [first_fail, second_fail]

    mock_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=7,
        final_hop_fee_ppm=100,
        hops=2,
        route=[
            {"id": "03" + "a" * 64, "channel": "100x1x0", "amount_msat": 1007000, "delay": 100},
            {"id": "03" + "u" * 64, "channel": "200x2x0", "amount_msat": 1000000, "delay": 40},
        ],
    )

    result = engine._execute_pair(pair, executor)

    # Exactly 2 execute calls (original + one retry), NOT 3
    assert executor.execute.call_count == 2
    # Exactly 1 price_pair call for the retry
    assert mock_router.price_pair.call_count == 1


def test_execute_pair_retry_merges_remembered_excludes_with_new_failure():
    """Cross-cycle remembered excludes should be merged into retry re-pricing."""
    from modules.rebalance_router_v2 import RouteResult

    engine, mock_router = _make_engine_with_mocked_router()
    pair = _make_pair()

    engine._routing_memory.ban_channel("stale-hop/0", ttl_seconds=300)

    first_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["934667x311x8/1"],
    )
    second_fail = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
        excluded=["222x2x0/1"],
    )

    executor = MagicMock()
    executor.execute.side_effect = [first_fail, second_fail]

    mock_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=7,
        final_hop_fee_ppm=100,
        hops=2,
        route=[
            {
                "id": "03" + "a" * 64,
                "channel": "100x1x0",
                "amount_msat": 1007000,
                "delay": 100,
            },
            {
                "id": "03" + "u" * 64,
                "channel": "200x2x0",
                "amount_msat": 1000000,
                "delay": 40,
            },
        ],
    )

    result = engine._execute_pair(pair, executor)

    assert result.success is False
    call_kwargs = mock_router.price_pair.call_args.kwargs
    assert set(call_kwargs["exclude"]) == {"934667x311x8/1", "stale-hop/0"}


def test_find_candidates_passes_remembered_excludes_to_initial_pricing(monkeypatch):
    """Initial pair pricing should include transient excludes learned earlier."""
    from types import SimpleNamespace

    from modules.rebalance_router_v2 import RouteResult

    engine, mock_router = _make_engine_with_mocked_router()
    engine.router_v2 = mock_router

    pair = _make_pair()
    engine._routing_memory.ban_channel("934667x311x8/1", ttl_seconds=300)

    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=500,
        )
    )

    class FakePlanner:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self, snapshot):
            return SimpleNamespace(selected=[pair], skipped=[])

    mock_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=7,
        final_hop_fee_ppm=100,
        hops=2,
        route=[
            {
                "id": "03" + "a" * 64,
                "channel": "100x1x0",
                "amount_msat": 1007000,
                "delay": 100,
            },
            {
                "id": "03" + "u" * 64,
                "channel": "200x2x0",
                "amount_msat": 1000000,
                "delay": 40,
            },
        ],
    )

    monkeypatch.setattr("modules.rebalance_engine_v2.RebalancePlanner", FakePlanner)

    candidates = engine.find_candidates()

    assert candidates == [pair]
    assert mock_router.price_pair.call_args.kwargs["exclude"] == ["934667x311x8/1"]


def test_run_cycle_records_failed_excludes_for_future_cycles(monkeypatch):
    """A failed execution should populate transient excludes for the next cycle."""
    engine, _ = _make_engine_with_mocked_router()
    pair = _make_pair()
    engine._cycle_router = None

    failed = _make_execution_result(
        success=False,
        error="retriable_failure: WIRE_FEE_INSUFFICIENT",
        excluded=["934667x311x8/1"],
    )

    class FakeExecutor:
        def __init__(self, plugin, database):
            pass

        def execute(self, **kwargs):
            return failed

    engine.find_candidates = MagicMock(return_value=[pair])
    monkeypatch.setattr("modules.rebalance_engine_v2.RebalanceExecutor", FakeExecutor)

    result = engine.run_cycle()

    assert len(result.executions) == 1
    assert engine._routing_memory.current_excludes() == ["934667x311x8/1"]
