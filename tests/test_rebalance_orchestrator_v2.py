"""Tests for the v2 rebalance engine orchestrator."""

from unittest.mock import MagicMock, patch

from modules.rebalance_engine_v2 import RebalanceEngine, CycleResult
from modules.rebalance_execution import ExecutionResult
from modules.rebalance_router_v2 import RouteResult


def _make_engine(channels=None, capex_budgets=None):
    """Create engine with mocked dependencies."""
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02" + "aa" * 32}

    # Default: two channels — one over-local, one over-remote
    if channels is None:
        channels = [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "bb" * 32,
                "short_channel_id": "111x1x0",
                "total_msat": 2_000_000_000,
                "our_amount_msat": 1_800_000_000,  # 90% local
                "updates": {
                    "remote": {"fee_proportional_millionths": 200},
                    "local": {"fee_proportional_millionths": 100},
                },
            },
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "cc" * 32,
                "short_channel_id": "222x2x0",
                "total_msat": 2_000_000_000,
                "our_amount_msat": 200_000_000,  # 10% local
                "updates": {
                    "remote": {"fee_proportional_millionths": 300},
                    # Sats-EV gate input: the refill destination earns our
                    # outbound fee when the restored liquidity routes.
                    "local": {"fee_proportional_millionths": 1_000},
                },
            },
        ]

    plugin.rpc.listpeerchannels.return_value = {"channels": channels}
    plugin.rpc.listchannels.return_value = {
        "channels": [{
            "source": channels[0]["peer_id"],
            "destination": "02" + "dd" * 32,
            "short_channel_id": "333x3x0",
            "fee_per_millionth": 0,
            "base_fee_millisatoshi": 0,
            "delay": 12,
        }],
    }

    config = MagicMock(spec=[])  # empty spec prevents auto-attribute creation
    config.low_liquidity_threshold = 0.35
    config.high_liquidity_threshold = 0.65
    config.rebalance_max_amount = 2_000_000
    # Auto-run rebalance execution requires a non-zero global spend budget;
    # pair-level capex budgets alone are not sufficient.
    config.daily_budget_sats = 500

    database = MagicMock()
    database.record_rebalance.return_value = 1
    database.reserve_budget.return_value = (True, 9999)
    database.mark_budget_spent.return_value = True
    database.release_budget_reservation.return_value = True

    capex = MagicMock()
    if capex_budgets:
        capex.compute_allocations.return_value = capex_budgets
    else:
        # Default: 500 sat budget for each channel
        from modules.capex_budget import CapexAllocations, ChannelCapexBudget
        capex.compute_allocations.return_value = CapexAllocations(
            channel_budgets={
                "111x1x0": ChannelCapexBudget(
                    channel_id="111x1x0", budget_msat=500_000,
                ),
                "222x2x0": ChannelCapexBudget(
                    channel_id="222x2x0", budget_msat=500_000,
                ),
            }
        )

    # Mark channels as active (>5 forwards)
    profitability = MagicMock()
    prof_result = MagicMock()
    prof_result.revenue.total_contribution_msat = 0
    prof_result.total_forward_count = 10
    profitability.get_channel_profitability.return_value = prof_result

    engine = RebalanceEngine(
        plugin=plugin,
        config=config,
        database=database,
        capex_engine=capex,
        profitability=profitability,
    )

    return engine, plugin


def _set_price_result(engine, *, success=True, route_cost_sats=100, error=""):
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = RouteResult(
        success=success,
        route_cost_sats=route_cost_sats,
        route=[],
        error=error,
    )


class TestFindCandidates:
    def test_returns_candidates_for_imbalanced_channels(self):
        engine, plugin = _make_engine()
        _set_price_result(engine, route_cost_sats=1)

        candidates = engine.find_candidates()

        assert len(candidates) == 1
        assert candidates[0].source_channel_id == "111x1x0"
        assert candidates[0].dest_channel_id == "222x2x0"

    def test_returns_empty_for_balanced_channels(self):
        channels = [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "bb" * 32,
                "short_channel_id": "111x1x0",
                "total_msat": 2_000_000_000,
                "our_amount_msat": 1_000_000_000,  # 50% — inside band
                "updates": {"remote": {"fee_proportional_millionths": 200}},
            },
        ]
        engine, plugin = _make_engine(channels=channels)

        candidates = engine.find_candidates()

        assert candidates == []

    def test_emits_audit_logs(self):
        engine, plugin = _make_engine()
        _set_price_result(engine, route_cost_sats=1)

        engine.find_candidates()

        # Check that structured audit logs were emitted
        log_calls = [str(c) for c in plugin.log.call_args_list]
        assert any("REBAL_PICK" in c for c in log_calls)
        assert any("REBAL_CYCLE" in c for c in log_calls)

    def test_skips_route_over_budget(self):
        engine, plugin = _make_engine()
        # Expensive route: 1000 sat fee exceeds 500 sat budget
        _set_price_result(engine, route_cost_sats=1000)

        candidates = engine.find_candidates()

        assert candidates == []
        log_calls = [str(c) for c in plugin.log.call_args_list]
        assert any("route_over_budget" in c for c in log_calls)

    def test_skips_no_route(self):
        engine, plugin = _make_engine()
        _set_price_result(engine, success=False, error="no_route: test")

        candidates = engine.find_candidates()

        assert candidates == []


class TestRunCycle:
    def test_executes_candidates(self):
        engine, plugin = _make_engine()
        _set_price_result(engine, route_cost_sats=1)

        with patch("modules.rebalance_engine_v2.NativeRouteExecutor") as executor_cls:
            executor = executor_cls.return_value
            executor.is_available.return_value = True
            executor.execute.return_value = ExecutionResult(
                success=True,
                amount_sats=500_000,
                fee_sats=10,
                fee_msat=10_000,
                fee_ppm=20,
                route_type="native",
            )

            result = engine.run_cycle()

        assert len(result.candidates) >= 1
        assert len(result.executions) >= 1
        assert result.executions[0].success is True
