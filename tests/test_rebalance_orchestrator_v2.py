"""Tests for the v2 rebalance engine orchestrator."""

from unittest.mock import MagicMock, patch

from modules.rebalance_engine_v2 import RebalanceEngine, CycleResult


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
                "updates": {"remote": {"fee_proportional_millionths": 200}},
            },
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "cc" * 32,
                "short_channel_id": "222x2x0",
                "total_msat": 2_000_000_000,
                "our_amount_msat": 200_000_000,  # 10% local
                "updates": {"remote": {"fee_proportional_millionths": 300}},
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

    database = MagicMock()

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


class TestFindCandidates:
    def test_returns_candidates_for_imbalanced_channels(self):
        engine, plugin = _make_engine()
        # Mock getroute to return a cheap route
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "02" + "dd" * 32, "channel": "333x3x0",
                 "amount_msat": 500_100_000, "delay": 40},
                {"id": "02" + "aa" * 32, "channel": "222x2x0",
                 "amount_msat": 500_000_000, "delay": 10},
            ]
        }

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
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "02" + "dd" * 32, "channel": "333x3x0",
                 "amount_msat": 500_100_000, "delay": 40},
                {"id": "02" + "aa" * 32, "channel": "222x2x0",
                 "amount_msat": 500_000_000, "delay": 10},
            ]
        }

        engine.find_candidates()

        # Check that structured audit logs were emitted
        log_calls = [str(c) for c in plugin.log.call_args_list]
        assert any("REBAL_PICK" in c for c in log_calls)
        assert any("REBAL_CYCLE" in c for c in log_calls)

    def test_skips_route_over_budget(self):
        engine, plugin = _make_engine()
        # Expensive route: 1000 sat fee exceeds 500 sat budget
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "02" + "dd" * 32, "channel": "333x3x0",
                 "amount_msat": 501_000_000_000, "delay": 40},
                {"id": "02" + "aa" * 32, "channel": "222x2x0",
                 "amount_msat": 500_000_000, "delay": 10},
            ]
        }

        candidates = engine.find_candidates()

        assert candidates == []
        log_calls = [str(c) for c in plugin.log.call_args_list]
        assert any("route_over_budget" in c for c in log_calls)

    def test_skips_no_route(self):
        engine, plugin = _make_engine()
        plugin.rpc.getroute.side_effect = Exception("no route found")

        candidates = engine.find_candidates()

        assert candidates == []


class TestRunCycle:
    def test_executes_candidates(self):
        engine, plugin = _make_engine()
        # Route for find_candidates
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "02" + "dd" * 32, "channel": "333x3x0",
                 "amount_msat": 500_100_000, "delay": 40},
                {"id": "02" + "aa" * 32, "channel": "222x2x0",
                 "amount_msat": 500_000_000, "delay": 10},
            ]
        }
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }
        plugin.rpc.waitsendpay.return_value = {
            "amount_sent_msat": 500_100_000,
        }

        result = engine.run_cycle()

        assert len(result.candidates) >= 1
        assert len(result.executions) >= 1
        assert result.executions[0].success is True
