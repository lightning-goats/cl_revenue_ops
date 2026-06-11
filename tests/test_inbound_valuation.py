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

    def test_fees_earned_sats_ceils_correctly(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5500,
            volume_routed_msat=1_000_000_000,
            forward_count=10,
        )
        assert rev.fees_earned_sats == 6  # ceiling: 5500 msat → 6 sats

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
    conn.execute("""
        CREATE TABLE channel_costs (
            channel_id TEXT PRIMARY KEY,
            peer_id TEXT,
            open_cost_sats INTEGER,
            capacity_sats INTEGER,
            opened_at INTEGER
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

    def test_profitability_reads_normalized_scid_aliases(self):
        """Profitability DB reads must see legacy ':' SCID rows and canonical 'x' rows."""
        conn = _create_test_db()
        now = int(time.time())
        conn.execute(
            "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
            ("100:1:0", "200:1:0", 50_000, 49_500, 500, now)
        )
        conn.execute(
            "INSERT INTO forwards VALUES (?, ?, ?, ?, ?, ?)",
            ("100x1x0", "200x1x0", 70_000, 69_300, 700, now)
        )
        conn.execute(
            "INSERT INTO rebalance_costs VALUES (?, ?, ?)",
            ("200:1:0", 1, now)
        )
        conn.execute(
            "INSERT INTO channel_costs VALUES (?, ?, ?, ?, ?)",
            ("200:1:0", "peer", 11, 1_000_000, now)
        )
        conn.commit()

        from modules.database import Database
        db = Database.__new__(Database)
        db.plugin = MagicMock()
        db._get_connection = lambda: conn

        all_totals = db.get_all_channels_revenue_totals()
        assert all_totals["200x1x0"]["fees_earned_msat"] == 1200
        assert all_totals["200x1x0"]["forward_count"] == 2

        single = db.get_channel_revenue_totals("200x1x0")
        assert single["fees_earned_msat"] == 1200
        assert single["forward_count"] == 2

        pnl = db.get_channel_full_pnl("200x1x0", window_days=30)
        assert pnl["channel_id"] == "200x1x0"
        assert pnl["direct_revenue_msat"] == 1200
        assert pnl["rebalance_cost_msat"] == 1000
        assert pnl["net_pnl_msat"] == 200

        assert db.get_last_forward_time_any_direction("200x1x0") == now
        assert db.get_channel_rebalance_costs("200x1x0") == 1
        assert db.get_channel_open_cost("200x1x0") == 11


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
        assert rev.fees_earned_sats == 6  # ceiling: 5500 msat → 6 sats
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


class TestPolicyManagerZombieCheck:
    """Policy manager ZOMBIE check uses total activity."""

    def test_inbound_gateway_not_flagged_as_zombie(self):
        """Channel with 0 exit forwards but 50 sourced forwards is not ZOMBIE."""
        import threading
        from modules.policy_manager import PolicyManager

        mock_plugin = MagicMock()
        mock_db = MagicMock()
        mock_db.get_channel_rebalance_success_rate.return_value = None

        pm = PolicyManager.__new__(PolicyManager)
        pm.plugin = mock_plugin
        pm.database = mock_db
        pm._policies = {}
        pm._cache = {}
        pm._cache_valid = True  # Prevent _load_cache from hitting DB
        pm._cache_lock = threading.Lock()
        pm._change_timestamps = {}
        pm.hive_hints = None

        mock_pa = MagicMock()
        mock_pa.identify_bleeders.return_value = [{
            'channel_id': '100x1x0',
            'peer_id': '02' + 'a' * 64,
            'capacity_sats': 2_000_000,
            'direct_revenue_sats': 0,
            'sourced_fee_contribution_sats': 120,
            'sourced_volume_sats': 50_000,
            'total_contribution_sats': 120,
            'rebalance_cost_sats': 200,
            'net_pnl_sats': -80,
            'direct_forward_count': 0,
            'sourced_forward_count': 50,
            'total_forward_count': 50,
            'loss_per_forward': 2,
            'revenue_sats': 0,
            'forward_count': 0,
        }]

        suggestions = pm.get_policy_suggestions(profitability_analyzer=mock_pa)

        zombie_suggestions = [s for s in suggestions if s.get('action') == 'consider_close']
        assert len(zombie_suggestions) == 0, (
            f"Inbound gateway with 50 sourced forwards should not be flagged as zombie. "
            f"Got: {zombie_suggestions}"
        )


class TestBleederIdentification:
    """Bleeder identification includes sourced metrics."""

    def test_identify_bleeders_includes_sourced_forward_count(self):
        """identify_bleeders output includes sourced_forward_count."""
        try:
            from modules.profitability_analyzer import ChannelProfitabilityAnalyzer as ProfitabilityAnalyzer
        except ImportError:
            from modules.profitability_analyzer import ProfitabilityAnalyzer

        mock_plugin = MagicMock()
        mock_db = MagicMock()

        mock_db.get_channel_full_pnl.return_value = {
            'direct_revenue_msat': 0,
            'direct_revenue_sats': 0,
            'direct_forward_count': 0,
            'sourced_volume_msat': 50_000_000,
            'sourced_volume_sats': 50_000,
            'sourced_fee_contribution_msat': 120_000,
            'sourced_fee_contribution_sats': 120,
            'sourced_forward_count': 50,
            'total_contribution_msat': 120_000,
            'total_contribution_sats': 120,
            'rebalance_cost_sats': 200,
            'net_pnl_msat': -80_000,
            'net_pnl_sats': -80,
            'revenue_sats': 0,
            'forward_count': 0,
        }

        analyzer = ProfitabilityAnalyzer.__new__(ProfitabilityAnalyzer)
        analyzer.plugin = mock_plugin
        analyzer.database = mock_db
        analyzer._get_all_channels = lambda: {
            "100x1x0": {"peer_id": "02" + "a" * 64, "capacity": 2_000_000}
        }

        bleeders = analyzer.identify_bleeders(window_days=30)
        assert len(bleeders) == 1
        b = bleeders[0]
        assert b['channel_id'] == "100x1x0"
        assert b['short_channel_id'] == "100x1x0"
        assert b['sourced_forward_count'] == 50
        assert b['total_forward_count'] == 50


class TestCapacityPlannerClosureProtection:
    """Inbound gateway closure protection enhanced."""

    def test_inbound_gateway_channel_role(self):
        """Verify pure inbound gateway gets INBOUND_GATEWAY role."""
        from modules.profitability_analyzer import (
            ChannelRevenue, ChannelProfitability, ChannelCosts,
            ProfitabilityClass
        )
        try:
            from modules.profitability_analyzer import ChannelRole
        except ImportError:
            pytest.skip("ChannelRole not available")

        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_volume_msat=50_000_000,
            sourced_fee_contribution_msat=120_000,
            sourced_forward_count=100,
        )
        costs = ChannelCosts(
            channel_id="100x1x0",
            peer_id="02" + "a" * 64,
            open_cost_sats=1000,
            rebalance_cost_sats=500,
        )
        prof = ChannelProfitability(
            channel_id="100x1x0",
            peer_id="02" + "a" * 64,
            capacity_sats=2_000_000,
            costs=costs,
            revenue=rev,
            net_profit_sats=-600,
            roi_percent=-40.0,
            classification=ProfitabilityClass.UNDERWATER,
            cost_per_sat_routed=0.0,
            fee_per_sat_routed=0.0,
            days_open=60,
            last_routed=int(time.time()) - 3600,
            marginal_profit_30d_sats=-200,
            rebalance_cost_30d_sats=500,
        )
        assert prof.channel_role == ChannelRole.INBOUND_GATEWAY
        assert rev.sourced_fee_contribution_sats >= 100


class TestRpcOutputSourcedMetrics:
    """RPC output includes sourced metrics in batch view."""

    def test_batch_output_includes_sourced_fields(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5_000,
            volume_routed_msat=1_000_000,
            forward_count=10,
            sourced_volume_msat=50_000_000,
            sourced_fee_contribution_msat=120_000,
            sourced_forward_count=63,
        )

        channel_summary = {
            "channel_id": "100x1x0",
            "fees_earned_sats": rev.fees_earned_sats,
            "volume_routed_sats": rev.volume_routed_sats,
            "forward_count": rev.forward_count,
            "sourced_forward_count": rev.sourced_forward_count,
            "sourced_fee_contribution_sats": rev.sourced_fee_contribution_sats,
            "sourced_volume_sats": rev.sourced_volume_sats,
            "total_contribution_sats": rev.total_contribution_sats,
            "total_forward_count": rev.total_forward_count,
        }

        assert channel_summary["sourced_forward_count"] == 63
        assert channel_summary["sourced_fee_contribution_sats"] == 120
        assert channel_summary["total_contribution_sats"] == 120
        assert channel_summary["total_forward_count"] == 73


class TestFleetPnlMsat:
    """Fleet P&L uses msat from database, converts at reporting boundary."""

    def test_sub_sat_fees_aggregate_before_truncation(self):
        """3 forwards of 300 msat each = 900 msat total should be at least 1 sat."""
        try:
            from modules.profitability_analyzer import ChannelProfitabilityAnalyzer as ProfitabilityAnalyzer
        except ImportError:
            from modules.profitability_analyzer import ProfitabilityAnalyzer

        mock_plugin = MagicMock()
        mock_db = MagicMock()

        mock_db.get_total_routing_revenue.return_value = 900  # 900 msat
        mock_db.get_total_volume_since.return_value = 150_000
        mock_db.get_total_forward_count_since.return_value = 3
        mock_db.get_total_rebalance_fees.return_value = 0
        mock_db.get_closure_costs_since.return_value = 0

        analyzer = ProfitabilityAnalyzer.__new__(ProfitabilityAnalyzer)
        analyzer.plugin = mock_plugin
        analyzer.database = mock_db

        summary = analyzer.get_pnl_summary(window_days=30)
        assert summary['gross_revenue_sats'] >= 1, (
            "Sub-satoshi fees should not be truncated to 0 at fleet level"
        )


class TestSourcedFeeDecay:
    """Audit F2: sourced-fee close protection must decay.

    The DATA side: ChannelProfitability exposes sourced_fee_30d_msat and
    role_30d so the capacity-planner close gate can consume windowed values
    instead of all-time aggregates that protect a channel forever.
    """

    def _make_prof(self, sourced_lifetime_sats, sourced_30d_msat,
                   sourced_fwd_lifetime=200, sourced_fwd_30d=0,
                   window_available=True):
        from modules.profitability_analyzer import (
            ChannelRevenue, ChannelProfitability, ChannelCosts,
            ProfitabilityClass,
        )
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_volume_msat=sourced_lifetime_sats * 1_000_000,
            sourced_fee_contribution_msat=sourced_lifetime_sats * 1000,
            sourced_forward_count=sourced_fwd_lifetime,
        )
        costs = ChannelCosts(
            channel_id="100x1x0", peer_id="02" + "a" * 64,
            open_cost_sats=1000, rebalance_cost_sats=0,
        )
        return ChannelProfitability(
            channel_id="100x1x0",
            peer_id="02" + "a" * 64,
            capacity_sats=2_000_000,
            costs=costs,
            revenue=rev,
            net_profit_sats=0,
            roi_percent=0.0,
            classification=ProfitabilityClass.BREAK_EVEN,
            cost_per_sat_routed=0.0,
            fee_per_sat_routed=0.0,
            days_open=800,
            last_routed=int(time.time()) - 86400 * 700,
            sourced_fee_30d_msat=sourced_30d_msat,
            sourced_forward_count_30d=sourced_fwd_30d,
            window_30d_available=window_available,
        )

    def test_ancient_sourcer_exposes_zero_30d_sourced_fees(self):
        """5000 sats sourced two years ago, zero in 30d -> windowed field is 0."""
        prof = self._make_prof(sourced_lifetime_sats=5000, sourced_30d_msat=0)
        assert prof.revenue.sourced_fee_contribution_msat == 5_000_000
        assert prof.sourced_fee_30d_msat == 0

    def test_ancient_gateway_role_30d_decays_to_dormant(self):
        """Lifetime INBOUND_GATEWAY with no 30d forwards -> role_30d DORMANT."""
        from modules.profitability_analyzer import ChannelRole
        prof = self._make_prof(sourced_lifetime_sats=5000, sourced_30d_msat=0)
        assert prof.channel_role == ChannelRole.INBOUND_GATEWAY
        assert prof.role_30d == ChannelRole.DORMANT

    def test_active_gateway_role_30d_stays_gateway(self):
        """A currently-sourcing gateway keeps INBOUND_GATEWAY in the window."""
        from modules.profitability_analyzer import ChannelRole
        prof = self._make_prof(
            sourced_lifetime_sats=5000, sourced_30d_msat=120_000,
            sourced_fwd_30d=40,
        )
        assert prof.role_30d == ChannelRole.INBOUND_GATEWAY

    def test_role_30d_falls_back_to_lifetime_without_window(self):
        """Objects without windowed data fall back to the lifetime role."""
        from modules.profitability_analyzer import ChannelRole
        prof = self._make_prof(
            sourced_lifetime_sats=5000, sourced_30d_msat=0,
            window_available=False,
        )
        assert prof.role_30d == ChannelRole.INBOUND_GATEWAY

    def test_role_30d_balanced_flow(self):
        """Mixed 30d flow classifies BALANCED in the window."""
        from modules.profitability_analyzer import ChannelRole
        prof = self._make_prof(
            sourced_lifetime_sats=5000, sourced_30d_msat=50_000,
            sourced_fwd_30d=10,
        )
        prof.forward_count_30d = 10
        assert prof.role_30d == ChannelRole.BALANCED

    def test_analyze_channel_exposes_windowed_sourced_fees(self):
        """End-to-end: analyze_channel wires sourced 30d fees from the P&L."""
        from modules.profitability_analyzer import (
            ChannelProfitabilityAnalyzer as ProfitabilityAnalyzer, ChannelCosts,
        )
        mock_plugin = MagicMock()
        mock_db = MagicMock()
        mock_db.get_channel_revenue_totals.return_value = {
            "fees_earned_msat": 0,
            "volume_routed_msat": 0,
            "forward_count": 0,
            "sourced_volume_msat": 5_000_000_000,
            "sourced_fee_contribution_msat": 5_000_000,  # 5000 sats lifetime
            "sourced_forward_count": 200,
        }
        # Dead 30d window
        mock_db.get_channel_full_pnl.return_value = {
            'total_contribution_msat': 0,
            'total_contribution_sats': 0,
            'rebalance_cost_sats': 0,
            'direct_revenue_msat': 0,
            'sourced_fee_contribution_msat': 0,
            'direct_forward_count': 0,
            'sourced_forward_count': 0,
        }
        mock_db.get_last_forward_time_any_direction.return_value = (
            int(time.time()) - 86400 * 700)
        mock_db.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 0, "last_success_time": None,
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
            open_cost_sats=500, rebalance_cost_sats=0,
        )
        channel_info = {
            "peer_id": peer_id,
            "capacity": 2_000_000,
            "funding_txid": "abc123",
            "opener": "local",
            "open_timestamp": int(time.time()) - 86400 * 800,
        }

        result = analyzer.analyze_channel("100x1x0", channel_info=channel_info)
        assert result is not None
        assert result.revenue.sourced_fee_contribution_msat == 5_000_000
        assert result.sourced_fee_30d_msat == 0
        assert result.window_30d_available is True
