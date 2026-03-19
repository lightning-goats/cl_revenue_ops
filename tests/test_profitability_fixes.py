"""
Tests for profitability analyzer accounting fixes.

Covers 5 fixes:
1. 50/50 revenue attribution (phantom wealth)
2. Splice costs in ChannelCosts
3. 30-day trailing marginal ROI
4. Zombie false-positive guard
5. ROI sorting & volume double-count
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import (
    ChannelCosts, ChannelRevenue, ChannelProfitability,
    ProfitabilityClass, ChannelProfitabilityAnalyzer
)


# ============================================================
# Helpers
# ============================================================

def _make_costs(open_cost=500, rebalance=1000, splice=0, effective=0):
    return ChannelCosts(
        channel_id="111x222x0",
        peer_id="02" + "a" * 64,
        open_cost_sats=open_cost,
        rebalance_cost_sats=rebalance,
        effective_rebalance_cost_sats=effective,
        splice_cost_sats=splice,
    )


def _make_revenue(fees=2000, sourced_fees=0, volume=1_000_000, sourced_volume=0,
                  forwards=100, sourced_forwards=0):
    return ChannelRevenue(
        channel_id="111x222x0",
        fees_earned_sats=fees,
        volume_routed_sats=volume,
        forward_count=forwards,
        sourced_volume_sats=sourced_volume,
        sourced_fee_contribution_sats=sourced_fees,
        sourced_forward_count=sourced_forwards,
    )


def _make_profitability(costs=None, revenue=None, capacity=2_000_000,
                        marginal_profit_30d=0, rebalance_cost_30d=0):
    if costs is None:
        costs = _make_costs()
    if revenue is None:
        revenue = _make_revenue()
    contribution = revenue.total_contribution_sats
    net = contribution - costs.total_cost_sats
    roi = net / costs.total_cost_sats if costs.total_cost_sats > 0 else 0.0
    return ChannelProfitability(
        channel_id="111x222x0",
        peer_id="02" + "a" * 64,
        capacity_sats=capacity,
        costs=costs,
        revenue=revenue,
        net_profit_sats=net,
        roi_percent=roi * 100,
        classification=ProfitabilityClass.PROFITABLE,
        cost_per_sat_routed=0.001,
        fee_per_sat_routed=0.002,
        days_open=30,
        last_routed=None,
        marginal_profit_30d_sats=marginal_profit_30d,
        rebalance_cost_30d_sats=rebalance_cost_30d,
    )


def _make_analyzer():
    """Build a ChannelProfitabilityAnalyzer with mocked dependencies."""
    plugin = MagicMock()
    config = MagicMock()
    config.estimated_open_cost_sats = 2000
    database = MagicMock()
    analyzer = ChannelProfitabilityAnalyzer(plugin, config, database)
    return analyzer


# ============================================================
# Fix 1: 50/50 Revenue Attribution
# ============================================================

class TestFiftyFiftySplit:
    """Verify that total_contribution_sats applies 50/50 split."""

    def test_direct_only(self):
        """Direct-only channel: contribution = fees / 2."""
        rev = _make_revenue(fees=1000, sourced_fees=0)
        assert rev.total_contribution_sats == 500

    def test_sourced_only(self):
        """Source-only channel: contribution = sourced / 2."""
        rev = _make_revenue(fees=0, sourced_fees=800)
        assert rev.total_contribution_sats == 400

    def test_both_directions(self):
        """Channel with fees in both directions gets (a+b)//2."""
        rev = _make_revenue(fees=1000, sourced_fees=600)
        # (1000 + 600) // 2 = 800
        assert rev.total_contribution_sats == 800

    def test_no_phantom_wealth(self):
        """Two channels that share the same 10-sat forward can't claim 20 sats total."""
        # Channel A earns 10 sats as exit, Channel B sources it
        exit_rev = _make_revenue(fees=10, sourced_fees=0)
        source_rev = _make_revenue(fees=0, sourced_fees=10)
        total = exit_rev.total_contribution_sats + source_rev.total_contribution_sats
        # 5 + 5 = 10, not 20
        assert total == 10

    def test_odd_rounding(self):
        """Integer division truncates, not rounds."""
        rev = _make_revenue(fees=7, sourced_fees=0)
        # 7 // 2 = 3
        assert rev.total_contribution_sats == 3


# ============================================================
# Fix 2: Splice Costs
# ============================================================

class TestSpliceCosts:
    """Verify splice_cost_sats is included in total_cost_sats."""

    def test_splice_in_total(self):
        """total_cost_sats = open + splice + rebalance."""
        costs = _make_costs(open_cost=500, rebalance=1000, splice=200)
        assert costs.total_cost_sats == 1700

    def test_no_splice(self):
        """Default splice cost is 0, doesn't affect total."""
        costs = _make_costs(open_cost=500, rebalance=1000)
        assert costs.splice_cost_sats == 0
        assert costs.total_cost_sats == 1500

    def test_splice_fetched_in_get_channel_costs(self):
        """_get_channel_costs() queries splice history and sums fee_sats."""
        analyzer = _make_analyzer()
        analyzer.database.get_channel_rebalance_costs.return_value = 100
        analyzer.database.get_channel_rebalance_success_rate.return_value = None
        analyzer.database.get_channel_splice_history.return_value = [
            {"fee_sats": 50, "amount_sats": 500000},
            {"fee_sats": 30, "amount_sats": 200000},
        ]
        analyzer._get_rebalance_costs_from_bookkeeper = MagicMock(return_value=0)
        analyzer._get_open_cost_from_bookkeeper = MagicMock(return_value=None)
        analyzer.database.get_channel_open_cost.return_value = 500

        costs = analyzer._get_channel_costs("111x222x0", "02" + "a" * 64, "abc123", 2_000_000)
        assert costs.splice_cost_sats == 80
        assert costs.total_cost_sats == 500 + 80 + 100

    def test_splice_none_fee_sats_treated_as_zero(self):
        """Splice records with None fee_sats are treated as 0."""
        analyzer = _make_analyzer()
        analyzer.database.get_channel_rebalance_costs.return_value = 0
        analyzer.database.get_channel_rebalance_success_rate.return_value = None
        analyzer.database.get_channel_splice_history.return_value = [
            {"fee_sats": None},
            {"fee_sats": 20},
        ]
        analyzer._get_rebalance_costs_from_bookkeeper = MagicMock(return_value=0)
        analyzer._get_open_cost_from_bookkeeper = MagicMock(return_value=None)
        analyzer.database.get_channel_open_cost.return_value = 500

        costs = analyzer._get_channel_costs("111x222x0", "02" + "a" * 64, "abc123", 2_000_000)
        assert costs.splice_cost_sats == 20


# ============================================================
# Fix 3: 30-Day Trailing Marginal ROI
# ============================================================

class TestMarginalRoi30d:
    """Verify marginal_roi uses 30-day trailing window."""

    def test_profitable_30d(self):
        """Positive 30d marginal profit yields positive ROI."""
        prof = _make_profitability(
            marginal_profit_30d=500, rebalance_cost_30d=200
        )
        # 500 / 200 = 2.5
        assert abs(prof.marginal_roi - 2.5) < 0.01

    def test_unprofitable_30d(self):
        """Negative 30d marginal profit yields negative ROI."""
        prof = _make_profitability(
            marginal_profit_30d=-300, rebalance_cost_30d=600
        )
        # -300 / 600 = -0.5
        assert abs(prof.marginal_roi - (-0.5)) < 0.01

    def test_zero_cost_earning(self):
        """Zero rebalance cost in 30d + positive profit = 1.0."""
        prof = _make_profitability(
            marginal_profit_30d=100, rebalance_cost_30d=0
        )
        assert prof.marginal_roi == 1.0

    def test_zero_cost_zero_profit(self):
        """Zero rebalance cost + zero profit = 0.0."""
        prof = _make_profitability(
            marginal_profit_30d=0, rebalance_cost_30d=0
        )
        assert prof.marginal_roi == 0.0

    def test_analyze_channel_populates_30d_fields(self):
        """analyze_channel() fetches 30d P&L and passes fields through."""
        analyzer = _make_analyzer()

        channel_info = {
            "peer_id": "02" + "a" * 64,
            "capacity": 2_000_000,
            "funding_txid": "abc123",
            "opener": "local",
            "open_timestamp": int(time.time()) - 86400 * 30,
        }
        analyzer._get_channel_costs = MagicMock(return_value=_make_costs())
        analyzer._get_channel_revenue = MagicMock(return_value=_make_revenue())
        analyzer._get_last_routing_time = MagicMock(return_value=int(time.time()) - 3600)
        analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 0, "last_success_time": 0,
        }

        # 30d P&L: 800 contribution, 300 rebalance cost
        analyzer.database.get_channel_full_pnl.return_value = {
            'total_contribution_sats': 800,
            'rebalance_cost_sats': 300,
        }

        result = analyzer.analyze_channel("111x222x0", channel_info=channel_info)
        assert result is not None
        assert result.rebalance_cost_30d_sats == 300
        assert result.marginal_profit_30d_sats == 500  # 800 - 300


# ============================================================
# Fix 4: Zombie False-Positive Guard
# ============================================================

class TestZombieFalsePositive:
    """Verify active channels with failed diagnostics aren't ZOMBIE."""

    def test_failed_diag_inactive_7d_is_zombie(self):
        """Channel inactive >= 7d with failed diagnostics -> ZOMBIE."""
        analyzer = _make_analyzer()
        analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 3,
            "last_success_time": 0,
        }
        now = int(time.time())
        last_routed = now - 86400 * 10  # 10 days ago
        result = analyzer._classify_channel(
            roi=-0.20, net_profit=-5000,
            last_routed=last_routed, days_open=60,
            channel_id="111x222x0"
        )
        assert result == ProfitabilityClass.ZOMBIE

    def test_failed_diag_active_recently_not_zombie(self):
        """Channel that routed 2 days ago with failed diagnostics -> NOT ZOMBIE."""
        analyzer = _make_analyzer()
        analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 3,
            "last_success_time": 0,
        }
        now = int(time.time())
        last_routed = now - 86400 * 2  # 2 days ago
        result = analyzer._classify_channel(
            roi=-0.20, net_profit=-5000,
            last_routed=last_routed, days_open=60,
            channel_id="111x222x0"
        )
        # Should be UNDERWATER (or STAGNANT_CANDIDATE), NOT ZOMBIE
        assert result != ProfitabilityClass.ZOMBIE

    def test_failed_diag_exactly_7d_is_zombie(self):
        """Channel inactive exactly 7 days with failed diag -> ZOMBIE."""
        analyzer = _make_analyzer()
        analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 2,
            "last_success_time": 0,
        }
        now = int(time.time())
        last_routed = now - 86400 * 7  # exactly 7 days
        result = analyzer._classify_channel(
            roi=-0.20, net_profit=-5000,
            last_routed=last_routed, days_open=60,
            channel_id="111x222x0"
        )
        assert result == ProfitabilityClass.ZOMBIE


# ============================================================
# Fix 5: ROI Sorting & Volume Double-Count
# ============================================================

class TestRoiSortingAndVolume:
    """Verify zero-cost ROI uses ROC and volume uses max()."""

    def test_zero_cost_uses_return_on_capacity(self):
        """Zero-cost channel ROI should reflect contribution/capacity, not flat 1.0."""
        analyzer = _make_analyzer()
        analyzer._get_all_channels = MagicMock(return_value={
            "111x222x0": {
                "peer_id": "02" + "a" * 64,
                "capacity": 2_000_000,
                "funding_txid": "abc123",
                "opener": "remote",
                "open_timestamp": int(time.time()) - 86400 * 30,
            }
        })
        # Remote open -> open_cost = 0; no rebalancing -> total_cost = 0
        analyzer._get_channel_costs = MagicMock(return_value=_make_costs(
            open_cost=0, rebalance=0, splice=0
        ))
        analyzer._get_channel_revenue = MagicMock(return_value=_make_revenue(
            fees=100, sourced_fees=0
        ))
        analyzer._get_last_routing_time = MagicMock(return_value=int(time.time()) - 3600)
        analyzer.database.get_channel_full_pnl.return_value = {
            'total_contribution_sats': 50,
            'rebalance_cost_sats': 0,
        }

        result = analyzer.analyze_channel("111x222x0")
        assert result is not None
        # contribution = (100 + 0) // 2 = 50; capacity = 2_000_000
        # roi = 50 / 2_000_000 = 0.000025
        expected_roi = 50 / 2_000_000
        assert abs(result.roi_percent - expected_roi * 100) < 0.001
        # Crucially, NOT 1.0 * 100 = 100%
        assert result.roi_percent < 1.0

    def test_zero_cost_no_contribution_is_zero(self):
        """Zero-cost channel with no revenue -> roi = 0."""
        analyzer = _make_analyzer()
        analyzer._get_all_channels = MagicMock(return_value={
            "111x222x0": {
                "peer_id": "02" + "a" * 64,
                "capacity": 2_000_000,
                "funding_txid": "abc123",
                "opener": "remote",
                "open_timestamp": int(time.time()) - 86400 * 30,
            }
        })
        analyzer._get_channel_costs = MagicMock(return_value=_make_costs(
            open_cost=0, rebalance=0, splice=0
        ))
        analyzer._get_channel_revenue = MagicMock(return_value=_make_revenue(
            fees=0, sourced_fees=0
        ))
        analyzer._get_last_routing_time = MagicMock(return_value=None)
        analyzer.database.get_channel_full_pnl.return_value = {
            'total_contribution_sats': 0,
            'rebalance_cost_sats': 0,
        }

        result = analyzer.analyze_channel("111x222x0")
        assert result is not None
        assert result.roi_percent == 0.0

    def test_volume_uses_max_not_sum(self):
        """total_volume should be max(exit, sourced), not sum."""
        analyzer = _make_analyzer()
        analyzer._get_all_channels = MagicMock(return_value={
            "111x222x0": {
                "peer_id": "02" + "a" * 64,
                "capacity": 2_000_000,
                "funding_txid": "abc123",
                "opener": "local",
                "open_timestamp": int(time.time()) - 86400 * 30,
            }
        })
        analyzer._get_channel_costs = MagicMock(return_value=_make_costs(
            open_cost=500, rebalance=100
        ))
        rev = _make_revenue(
            fees=1000, sourced_fees=500,
            volume=500_000, sourced_volume=300_000,
        )
        analyzer._get_channel_revenue = MagicMock(return_value=rev)
        analyzer._get_last_routing_time = MagicMock(return_value=int(time.time()) - 3600)
        analyzer.database.get_channel_full_pnl.return_value = {
            'total_contribution_sats': 750,
            'rebalance_cost_sats': 100,
        }

        result = analyzer.analyze_channel("111x222x0")
        assert result is not None
        # max(500_000, 300_000) = 500_000
        # cost_per_sat = total_cost / 500_000
        total_cost = 500 + 100  # no splice
        expected_cost_per_sat = total_cost / 500_000
        assert abs(result.cost_per_sat_routed - expected_cost_per_sat) < 1e-9


# ============================================================
# Fix: Channel open timestamp passed to record_channel_open_cost
# ============================================================

class TestOpenTimestampPassthrough:
    """record_channel_open_cost must store the actual channel open time,
    not time.time(), so windowed budget queries don't count old opens as recent."""

    def test_get_channel_costs_passes_open_timestamp(self):
        """_get_channel_costs passes open_timestamp to record_channel_open_cost."""
        analyzer = _make_analyzer()
        old_ts = int(time.time()) - 90 * 86400  # 90 days ago

        # No cached cost → forces bookkeeper lookup → records result
        analyzer.database.get_channel_open_cost.return_value = None
        analyzer.database.get_channel_rebalance_costs.return_value = 0
        analyzer.database.get_channel_splice_history.return_value = []
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            "success_rate": 1.0, "total": 0, "avg_cost_ppm": 0, "avg_amount_sats": 0
        }

        # Bookkeeper returns a cost
        with patch.object(analyzer, '_get_rebalance_costs_from_bookkeeper', return_value=0), \
             patch.object(analyzer, '_get_open_cost_from_bookkeeper', return_value=150):
            analyzer._get_channel_costs(
                "100x1x0", "peer1", "abc123", 2_000_000,
                opener="local", open_timestamp=old_ts
            )

        # The recorded timestamp must be the actual open time, not now
        call_args = analyzer.database.record_channel_open_cost.call_args
        assert call_args is not None, "record_channel_open_cost was not called"
        # Check the timestamp kwarg or positional arg
        if call_args.kwargs.get("timestamp") is not None:
            recorded_ts = call_args.kwargs["timestamp"]
        else:
            # positional: (channel_id, peer_id, cost, capacity, timestamp)
            recorded_ts = call_args[0][4] if len(call_args[0]) > 4 else None
        assert recorded_ts == old_ts, (
            f"Expected open_timestamp={old_ts}, got {recorded_ts}. "
            "Budget queries will miscount old opens as recent spend."
        )

    def test_remote_channel_self_heal_uses_open_timestamp(self):
        """Self-healing remote channel cost correction preserves open timestamp."""
        analyzer = _make_analyzer()
        old_ts = int(time.time()) - 180 * 86400  # 180 days ago

        # Remote channel with stale nonzero cost triggers self-heal
        analyzer.database.get_channel_open_cost.return_value = 500
        analyzer.database.get_channel_rebalance_costs.return_value = 0
        analyzer.database.get_channel_splice_history.return_value = []
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            "success_rate": 1.0, "total": 0, "avg_cost_ppm": 0, "avg_amount_sats": 0
        }

        with patch.object(analyzer, '_get_rebalance_costs_from_bookkeeper', return_value=0):
            analyzer._get_channel_costs(
                "200x2x0", "peer2", "def456", 3_000_000,
                opener="remote", open_timestamp=old_ts
            )

        call_args = analyzer.database.record_channel_open_cost.call_args
        assert call_args is not None, "Self-heal did not call record_channel_open_cost"
        if call_args.kwargs.get("timestamp") is not None:
            recorded_ts = call_args.kwargs["timestamp"]
        else:
            recorded_ts = call_args[0][4] if len(call_args[0]) > 4 else None
        assert recorded_ts == old_ts, (
            f"Self-heal used timestamp={recorded_ts}, expected {old_ts}"
        )

    def test_sanity_check_correction_uses_open_timestamp(self):
        """_sanity_check_open_cost passes timestamp when self-healing."""
        analyzer = _make_analyzer()
        old_ts = int(time.time()) - 60 * 86400

        # Set up a local channel with an invalid open cost (triggers sanity check)
        analyzer.database.get_channel_open_cost.return_value = 5_000_000  # > capacity, invalid
        analyzer.database.get_channel_rebalance_costs.return_value = 0
        analyzer.database.get_channel_splice_history.return_value = []
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            "success_rate": 1.0, "total": 0, "avg_cost_ppm": 0, "avg_amount_sats": 0
        }

        with patch.object(analyzer, '_get_rebalance_costs_from_bookkeeper', return_value=0), \
             patch.object(analyzer, '_get_open_cost_from_bookkeeper', return_value=300):
            analyzer._get_channel_costs(
                "300x3x0", "peer3", "ghi789", 2_000_000,
                opener="local", open_timestamp=old_ts
            )

        call_args = analyzer.database.record_channel_open_cost.call_args
        assert call_args is not None
        if call_args.kwargs.get("timestamp") is not None:
            recorded_ts = call_args.kwargs["timestamp"]
        else:
            recorded_ts = call_args[0][4] if len(call_args[0]) > 4 else None
        assert recorded_ts == old_ts


# ============================================================
# Fix 6: Guard marginal ROI against negative rebalance costs
# ============================================================

class TestMarginalRoiEdgeCases:
    """Negative rebalance costs must not invert ROI sign."""

    def test_negative_rebalance_cost_positive_profit_returns_one(self):
        """Negative rebalance costs with positive profit returns 1.0."""
        prof = _make_profitability(
            marginal_profit_30d=500, rebalance_cost_30d=-100
        )
        assert prof.marginal_roi == 1.0

    def test_negative_rebalance_cost_zero_profit_returns_zero(self):
        """Negative rebalance costs with zero profit returns 0.0."""
        prof = _make_profitability(
            marginal_profit_30d=0, rebalance_cost_30d=-50
        )
        assert prof.marginal_roi == 0.0

    def test_negative_rebalance_cost_negative_profit_returns_zero(self):
        """Negative rebalance costs with negative profit returns 0.0."""
        prof = _make_profitability(
            marginal_profit_30d=-200, rebalance_cost_30d=-50
        )
        assert prof.marginal_roi == 0.0

    def test_negative_rebalance_cost_never_inverts_sign(self):
        """Ensure marginal_roi is always >= 0 when rebalance cost is negative."""
        for cost in [-1, -10, -100, -1000]:
            for profit in [-500, -1, 0, 1, 500]:
                prof = _make_profitability(
                    marginal_profit_30d=profit, rebalance_cost_30d=cost
                )
                assert prof.marginal_roi >= 0.0, (
                    f"marginal_roi={prof.marginal_roi} for profit={profit}, cost={cost}"
                )


# ============================================================
# Fix 7: Capacity division guards (audit confirmation)
# ============================================================

class TestCapacityDivisionGuards:
    """Verify zero/negative capacity does not cause ZeroDivisionError."""

    def test_zero_capacity_zero_cost_roi(self):
        """Zero capacity with zero cost should not raise ZeroDivisionError."""
        analyzer = _make_analyzer()
        channel_info = {
            "peer_id": "02" + "a" * 64,
            "capacity": 0,
            "funding_txid": "abc123",
            "opener": "remote",
            "open_timestamp": int(time.time()) - 86400 * 30,
        }
        analyzer._get_channel_costs = MagicMock(return_value=_make_costs(
            open_cost=0, rebalance=0, splice=0
        ))
        analyzer._get_channel_revenue = MagicMock(return_value=_make_revenue(
            fees=100, sourced_fees=0
        ))
        analyzer._get_last_routing_time = MagicMock(return_value=int(time.time()) - 3600)
        analyzer.database.get_channel_full_pnl.return_value = {
            'total_contribution_sats': 50,
            'rebalance_cost_sats': 0,
        }

        # Should not raise ZeroDivisionError
        result = analyzer.analyze_channel("111x222x0", channel_info=channel_info)
        assert result is not None

    def test_zero_volume_cost_per_sat(self):
        """Zero volume should yield 0.0 cost_per_sat, not ZeroDivisionError."""
        analyzer = _make_analyzer()
        channel_info = {
            "peer_id": "02" + "a" * 64,
            "capacity": 2_000_000,
            "funding_txid": "abc123",
            "opener": "local",
            "open_timestamp": int(time.time()) - 86400 * 30,
        }
        analyzer._get_channel_costs = MagicMock(return_value=_make_costs(
            open_cost=500, rebalance=100
        ))
        analyzer._get_channel_revenue = MagicMock(return_value=_make_revenue(
            fees=0, sourced_fees=0, volume=0, sourced_volume=0,
            forwards=0, sourced_forwards=0
        ))
        analyzer._get_last_routing_time = MagicMock(return_value=None)
        analyzer.database.get_channel_full_pnl.return_value = {
            'total_contribution_sats': 0,
            'rebalance_cost_sats': 0,
        }
        analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 0, "last_success_time": 0,
        }

        result = analyzer.analyze_channel("111x222x0", channel_info=channel_info)
        assert result is not None
        assert result.cost_per_sat_routed == 0.0
        assert result.fee_per_sat_routed == 0.0
