"""Tests for inbound source valuation fixes."""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import ChannelRevenue


class TestChannelRevenueMsat:
    """ChannelRevenue stores msat internally, exposes sats via properties."""

    def test_fees_earned_msat_stored_directly(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5500,
            volume_routed_msat=1_000_000_000,
            forward_count=10,
        )
        assert rev.fees_earned_msat == 5500

    def test_fees_earned_sats_truncates_correctly(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5500,
            volume_routed_msat=1_000_000_000,
            forward_count=10,
        )
        assert rev.fees_earned_sats == 5

    def test_sub_sat_fee_rounds_up_to_1(self):
        """A channel earning 50 msat should show 1 sat, not 0."""
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=50,
            volume_routed_msat=500_000,
            forward_count=1,
        )
        assert rev.fees_earned_sats == 1

    def test_zero_fee_stays_zero(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
        )
        assert rev.fees_earned_sats == 0

    def test_sourced_fee_sub_sat_rounds_up(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_fee_contribution_msat=200,
            sourced_forward_count=3,
        )
        assert rev.sourced_fee_contribution_sats == 1

    def test_volume_sats_truncates_no_ceiling(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=500,
            forward_count=0,
        )
        assert rev.volume_routed_sats == 0


class TestTotalContributionMax:
    """total_contribution uses max(earned, sourced) for valuation."""

    def test_exit_dominant_channel(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=10_000,
            volume_routed_msat=5_000_000,
            forward_count=50,
            sourced_fee_contribution_msat=2_000,
            sourced_forward_count=5,
        )
        assert rev.total_contribution_msat == 10_000
        assert rev.total_contribution_sats == 10

    def test_inbound_dominant_channel(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=1_000,
            volume_routed_msat=500_000,
            forward_count=2,
            sourced_fee_contribution_msat=8_000,
            sourced_forward_count=40,
        )
        assert rev.total_contribution_msat == 8_000
        assert rev.total_contribution_sats == 8

    def test_pure_inbound_gateway(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_fee_contribution_msat=120_000,
            sourced_forward_count=63,
        )
        assert rev.total_contribution_msat == 120_000
        assert rev.total_contribution_sats == 120

    def test_sub_sat_sourced_fee_valued(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_fee_contribution_msat=500,
            sourced_forward_count=5,
        )
        assert rev.total_contribution_msat == 500
        assert rev.total_contribution_sats == 1

    def test_total_forward_count(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5_000,
            volume_routed_msat=1_000_000,
            forward_count=10,
            sourced_forward_count=50,
        )
        assert rev.total_forward_count == 60


import sqlite3
import time


def _create_test_db():
    """Create an in-memory DB with the forwards schema and test data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE forwards (
            in_channel TEXT,
            out_channel TEXT,
            in_msat INTEGER,
            out_msat INTEGER,
            fee_msat INTEGER,
            timestamp INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE daily_forwarding_stats (
            channel_id TEXT,
            date INTEGER,
            forward_count INTEGER,
            total_fee_msat INTEGER,
            total_out_msat INTEGER,
            total_in_msat INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE daily_forwarding_stats_inbound (
            channel_id TEXT,
            date INTEGER,
            forward_count INTEGER,
            total_fee_msat INTEGER,
            total_in_msat INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE rebalance_costs (
            channel_id TEXT,
            cost_sats INTEGER,
            timestamp INTEGER
        )
    """)
    return conn


class TestDatabaseMsatReturns:
    """Database methods return msat values without truncation."""

    def test_get_all_channels_revenue_totals_returns_msat_keys(self):
        conn = _create_test_db()
        now = int(time.time())
        conn.execute(
            "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
            ("100x1x0", "200x1x0", 50000, 49500, 500, now)
        )
        conn.commit()

        from modules.database import Database
        db = Database.__new__(Database)
        db.plugin = MagicMock()
        db._get_connection = lambda: conn

        totals = db.get_all_channels_revenue_totals()

        exit_data = totals["200x1x0"]
        assert "fees_earned_msat" in exit_data
        assert exit_data["fees_earned_msat"] == 500
        assert "volume_routed_msat" in exit_data
        assert exit_data["volume_routed_msat"] == 49500

        entry_data = totals["100x1x0"]
        assert "sourced_fee_contribution_msat" in entry_data
        assert entry_data["sourced_fee_contribution_msat"] == 500
        assert "sourced_volume_msat" in entry_data
        assert entry_data["sourced_volume_msat"] == 50000

    def test_get_channel_revenue_totals_returns_msat_keys(self):
        conn = _create_test_db()
        now = int(time.time())
        conn.execute(
            "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
            ("100x1x0", "200x1x0", 50000, 49500, 500, now)
        )
        conn.commit()

        from modules.database import Database
        db = Database.__new__(Database)
        db.plugin = MagicMock()
        db._get_connection = lambda: conn

        totals = db.get_channel_revenue_totals("200x1x0")
        assert "fees_earned_msat" in totals
        assert totals["fees_earned_msat"] == 500

    def test_get_channel_pnl_returns_msat(self):
        conn = _create_test_db()
        now = int(time.time())
        conn.execute(
            "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
            ("100x1x0", "200x1x0", 50000, 49500, 500, now)
        )
        conn.commit()

        from modules.database import Database
        db = Database.__new__(Database)
        db.plugin = MagicMock()
        db._get_connection = lambda: conn

        pnl = db.get_channel_pnl("200x1x0", window_days=30)
        assert "revenue_msat" in pnl
        assert pnl["revenue_msat"] == 500

    def test_get_channel_inbound_contribution_returns_msat(self):
        conn = _create_test_db()
        now = int(time.time())
        conn.execute(
            "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
            ("100x1x0", "200x1x0", 50000, 49500, 500, now)
        )
        conn.commit()

        from modules.database import Database
        db = Database.__new__(Database)
        db.plugin = MagicMock()
        db._get_connection = lambda: conn

        inbound = db.get_channel_inbound_contribution("100x1x0", window_days=30)
        assert "sourced_fee_contribution_msat" in inbound
        assert inbound["sourced_fee_contribution_msat"] == 500
        assert "sourced_volume_msat" in inbound
        assert inbound["sourced_volume_msat"] == 50000

    def test_get_channel_full_pnl_returns_msat(self):
        conn = _create_test_db()
        now = int(time.time())
        conn.execute(
            "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
            ("100x1x0", "200x1x0", 50000, 49500, 500, now)
        )
        conn.commit()

        from modules.database import Database
        db = Database.__new__(Database)
        db.plugin = MagicMock()
        db._get_connection = lambda: conn

        pnl = db.get_channel_full_pnl("200x1x0", window_days=30)
        assert "direct_revenue_msat" in pnl
        assert pnl["direct_revenue_msat"] == 500
        assert "total_contribution_msat" in pnl
        assert "net_pnl_msat" in pnl

    def test_get_total_routing_revenue_returns_msat(self):
        conn = _create_test_db()
        now = int(time.time())
        for _ in range(3):
            conn.execute(
                "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
                ("100x1x0", "200x1x0", 50000, 49700, 300, now)
            )
        conn.commit()

        from modules.database import Database
        db = Database.__new__(Database)
        db.plugin = MagicMock()
        db._get_connection = lambda: conn

        since = now - 86400
        total = db.get_total_routing_revenue(since)
        assert total == 900  # 3 * 300 = 900 msat (not 0 from truncation)


class TestRevenueDataConstruction:
    """Revenue data constructors use msat fields from database."""

    def test_get_all_revenue_data_builds_msat_channel_revenue(self):
        from modules.profitability_analyzer import ChannelProfitabilityAnalyzer

        mock_plugin = MagicMock()
        mock_db = MagicMock()
        mock_db.get_all_channels_revenue_totals.return_value = {
            "200x1x0": {
                "fees_earned_msat": 5500,
                "volume_routed_msat": 1_000_000,
                "forward_count": 10,
                "sourced_volume_msat": 0,
                "sourced_fee_contribution_msat": 0,
                "sourced_forward_count": 0,
            }
        }

        analyzer = ChannelProfitabilityAnalyzer.__new__(ChannelProfitabilityAnalyzer)
        analyzer.plugin = mock_plugin
        analyzer.database = mock_db

        result = analyzer._get_all_revenue_data()
        rev = result["200x1x0"]
        assert rev.fees_earned_msat == 5500
        assert rev.fees_earned_sats == 5
        assert rev.volume_routed_msat == 1_000_000

    def test_get_channel_revenue_builds_msat_channel_revenue(self):
        from modules.profitability_analyzer import ChannelProfitabilityAnalyzer

        mock_plugin = MagicMock()
        mock_db = MagicMock()
        mock_db.get_channel_revenue_totals.return_value = {
            "fees_earned_msat": 500,
            "volume_routed_msat": 50_000,
            "forward_count": 1,
            "sourced_volume_msat": 0,
            "sourced_fee_contribution_msat": 0,
            "sourced_forward_count": 0,
        }

        analyzer = ChannelProfitabilityAnalyzer.__new__(ChannelProfitabilityAnalyzer)
        analyzer.plugin = mock_plugin
        analyzer.database = mock_db

        rev = analyzer._get_channel_revenue("200x1x0")
        assert rev.fees_earned_msat == 500
        assert rev.fees_earned_sats == 1  # sub-sat ceiling


class TestClassificationTotalForwardCount:
    """Classification sees total (exit + sourced) forward count."""

    def test_inbound_gateway_not_stagnant(self):
        """Channel with 0 exit forwards but 100 sourced forwards is not STAGNANT."""
        from modules.profitability_analyzer import (
            ChannelProfitabilityAnalyzer as ProfitabilityAnalyzer, ProfitabilityClass, ChannelCosts
        )

        mock_plugin = MagicMock()
        mock_db = MagicMock()

        # Channel with 0 exit but 100 sourced forwards
        mock_db.get_channel_revenue_totals.return_value = {
            "fees_earned_msat": 0,
            "volume_routed_msat": 0,
            "forward_count": 0,
            "sourced_volume_msat": 50_000_000,
            "sourced_fee_contribution_msat": 120_000,
            "sourced_forward_count": 100,
        }
        mock_db.get_channel_full_pnl.return_value = {
            'total_contribution_msat': 120_000,
            'total_contribution_sats': 120,
            'rebalance_cost_sats': 50,
            'net_pnl_msat': 70_000,
            'net_pnl_sats': 70,
            'direct_revenue_msat': 0,
            'direct_revenue_sats': 0,
            'direct_forward_count': 0,
            'sourced_fee_contribution_msat': 120_000,
            'sourced_fee_contribution_sats': 120,
            'sourced_volume_msat': 50_000_000,
            'sourced_volume_sats': 50_000,
            'sourced_forward_count': 100,
            'revenue_sats': 0,
            'forward_count': 0,
        }
        mock_db.get_last_forward_time_any_direction.return_value = int(time.time()) - 3600
        mock_db.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 0, "last_success_time": None
        }

        analyzer = ProfitabilityAnalyzer.__new__(ProfitabilityAnalyzer)
        analyzer.plugin = mock_plugin
        analyzer.database = mock_db
        analyzer.hive_hints = None
        analyzer._profitability_cache = {}
        analyzer._cache_timestamp = 0
        analyzer._cache_ttl = 300
        analyzer._bleeder_cache = None
        analyzer._bleeder_cache_time = 0

        peer_id = "02" + "b" * 64
        analyzer._get_channel_costs = lambda *args, **kwargs: ChannelCosts(
            channel_id="100x1x0", peer_id=peer_id,
            open_cost_sats=500, rebalance_cost_sats=50
        )

        channel_info = {
            "peer_id": peer_id,
            "capacity": 2_000_000,
            "funding_txid": "abc123",
            "opener": "local",
            "open_timestamp": int(time.time()) - 86400 * 30,
        }

        result = analyzer.analyze_channel("100x1x0", channel_info=channel_info)
        assert result is not None
        # Must NOT be STAGNANT or ZOMBIE — it has 100 sourced forwards
        assert result.classification not in (
            ProfitabilityClass.STAGNANT_CANDIDATE,
            ProfitabilityClass.ZOMBIE,
        ), f"Got {result.classification.value} — inbound gateway with 100 sourced forwards should not be stagnant/zombie"

    def test_fleet_member_protection_with_sourced_forwards(self):
        """Hive member with sourced forwards is protected from UNDERWATER."""
        from modules.profitability_analyzer import ChannelProfitabilityAnalyzer as ProfitabilityAnalyzer, ProfitabilityClass

        mock_plugin = MagicMock()
        analyzer = ProfitabilityAnalyzer.__new__(ProfitabilityAnalyzer)
        analyzer.plugin = mock_plugin
        analyzer.database = MagicMock()
        analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 0, "last_success_time": None
        }

        mock_hints = MagicMock()
        mock_hints.is_hive_member.return_value = True
        mock_hints.get_centrality.return_value = 0.001
        mock_hints.get_corridor_role.return_value = None
        analyzer.hive_hints = mock_hints

        peer_id = "02" + "c" * 64
        # Underwater ROI but has sourced forwards (total_forward_count=50 passed as forward_count)
        classification = analyzer._classify_channel(
            roi=-0.5, net_profit=-500,
            last_routed=int(time.time()) - 3600,
            days_open=30,
            channel_id="100x1x0",
            peer_id=peer_id,
            forward_count=50,
        )
        assert classification == ProfitabilityClass.BREAK_EVEN, (
            f"Hive member with 50 forwards should be BREAK_EVEN, got {classification.value}"
        )
