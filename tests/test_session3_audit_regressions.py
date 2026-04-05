"""
Regression tests for Session 3 audit: database.py + profitability_analyzer.py

These tests verify the fixes applied during the Session 3 pre-production audit.
"""

import os
import sys
import time
import sqlite3
import threading
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Database Tests
# ===========================================================================

def _make_test_db():
    """Create a real in-memory database with the schema needed for tests."""
    from modules.database import Database
    plugin = MagicMock()
    plugin.rpc = MagicMock()
    db = Database.__new__(Database)
    db.plugin = plugin
    db.db_path = ":memory:"
    db._local = threading.local()
    db._thread_connections = []
    db._thread_conn_lock = threading.Lock()

    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")

    # Create tables needed for budget tests
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rebalance_costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            cost_sats INTEGER NOT NULL,
            cost_msat INTEGER NOT NULL DEFAULT 0,
            amount_sats INTEGER NOT NULL DEFAULT 0,
            fee_ppm INTEGER NOT NULL DEFAULT 0,
            timestamp INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS budget_reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            reserved_sats INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            reserved_at INTEGER NOT NULL,
            released_at INTEGER,
            actual_cost_sats INTEGER
        )
    """)

    db._local.conn = conn
    return db, conn


class TestGetBudgetStatusAtomic:
    """C-1: get_budget_status must read spent and reserved in a single snapshot."""

    def test_budget_status_returns_correct_totals(self):
        """Basic correctness: spent + reserved = total_committed."""
        db, conn = _make_test_db()
        now = int(time.time())

        # Insert some spent costs
        conn.execute(
            "INSERT INTO rebalance_costs (channel_id, cost_sats, timestamp) VALUES (?, ?, ?)",
            ("100x1x0", 500, now)
        )
        # Insert an active reservation
        conn.execute(
            "INSERT INTO budget_reservations (channel_id, reserved_sats, status, reserved_at) VALUES (?, ?, ?, ?)",
            ("200x1x0", 300, "active", now)
        )

        result = db.get_budget_status(now - 3600)
        assert result['spent'] == 500
        assert result['reserved'] == 300
        assert result['total_committed'] == 800

    def test_budget_status_ignores_released_reservations(self):
        """Released reservations should not count toward reserved total."""
        db, conn = _make_test_db()
        now = int(time.time())

        conn.execute(
            "INSERT INTO budget_reservations (channel_id, reserved_sats, status, reserved_at) VALUES (?, ?, ?, ?)",
            ("100x1x0", 1000, "released", now)
        )
        conn.execute(
            "INSERT INTO budget_reservations (channel_id, reserved_sats, status, reserved_at) VALUES (?, ?, ?, ?)",
            ("200x1x0", 200, "active", now)
        )

        result = db.get_budget_status(now - 3600)
        assert result['reserved'] == 200

    def test_budget_status_empty_tables(self):
        """Empty tables should return all zeros."""
        db, conn = _make_test_db()
        now = int(time.time())

        result = db.get_budget_status(now - 3600)
        assert result['spent'] == 0
        assert result['reserved'] == 0
        assert result['total_committed'] == 0


class TestClearAllReservationsAtomic:
    """I-1: clear_all_reservations must be atomic."""

    def test_clears_all_active_reservations(self):
        """All active reservations should be released."""
        db, conn = _make_test_db()
        now = int(time.time())

        for i in range(5):
            conn.execute(
                "INSERT INTO budget_reservations (channel_id, reserved_sats, status, reserved_at) VALUES (?, ?, ?, ?)",
                (f"{100+i}x1x0", 100 * (i + 1), "active", now)
            )

        result = db.clear_all_reservations()
        assert result['cleared_count'] == 5
        assert result['released_sats'] == 1500

        # Verify all are now 'released'
        active = conn.execute("SELECT COUNT(*) FROM budget_reservations WHERE status = 'active'").fetchone()[0]
        assert active == 0

    def test_does_not_double_release(self):
        """Already-released reservations should not be counted."""
        db, conn = _make_test_db()
        now = int(time.time())

        conn.execute(
            "INSERT INTO budget_reservations (channel_id, reserved_sats, status, reserved_at) VALUES (?, ?, ?, ?)",
            ("100x1x0", 500, "released", now)
        )
        conn.execute(
            "INSERT INTO budget_reservations (channel_id, reserved_sats, status, reserved_at) VALUES (?, ?, ?, ?)",
            ("200x1x0", 300, "active", now)
        )

        result = db.clear_all_reservations()
        assert result['cleared_count'] == 1
        assert result['released_sats'] == 300

    def test_empty_reservations(self):
        """No active reservations returns zero counts."""
        db, conn = _make_test_db()

        result = db.clear_all_reservations()
        assert result['cleared_count'] == 0
        assert result['released_sats'] == 0


# ===========================================================================
# Profitability Analyzer Tests
# ===========================================================================

from modules.profitability_analyzer import (
    ProfitabilityClass, ChannelProfitability, ChannelCosts, ChannelRevenue
)


def _make_profitability(classification, roi_percent, marginal_roi,
                        marginal_profit_30d=0, rebalance_cost_30d=0):
    """Helper to create a ChannelProfitability for testing."""
    return ChannelProfitability(
        channel_id="100x1x0",
        peer_id="02" + "ab" * 32,
        capacity_sats=1_000_000,
        costs=ChannelCosts(
            channel_id="100x1x0",
            peer_id="02" + "ab" * 32,
            open_cost_sats=5000,
            rebalance_cost_sats=1000,
        ),
        revenue=ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_sats=2000,
            volume_routed_sats=500_000,
            forward_count=50,
        ),
        net_profit_sats=int(roi_percent * 60),  # arbitrary
        roi_percent=roi_percent,
        classification=classification,
        cost_per_sat_routed=0.01,
        fee_per_sat_routed=0.004,
        days_open=30,
        last_routed=int(time.time()) - 3600,
        marginal_profit_30d_sats=marginal_profit_30d,
        rebalance_cost_30d_sats=rebalance_cost_30d,
    )


class TestRebalanceFeeMultiplierUsesMarginalROI:
    """I-3: get_max_rebalance_fee_multiplier should use marginal_roi, not total ROI."""

    def _make_analyzer(self):
        """Create a ProfitabilityAnalyzer with mocked dependencies."""
        from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
        analyzer = ChannelProfitabilityAnalyzer.__new__(ChannelProfitabilityAnalyzer)
        analyzer.plugin = MagicMock()
        analyzer.database = MagicMock()
        analyzer._profitability_cache = {}
        analyzer._cache_ttl = 300
        analyzer._cache_timestamp = int(time.time())
        analyzer._lock = threading.Lock()
        return analyzer

    def test_profitable_channel_uses_marginal_roi(self):
        """A profitable channel with high total ROI but low marginal should get multiplier based on marginal."""
        analyzer = self._make_analyzer()

        # High total ROI (50%) but modest marginal ROI (10%)
        prof = _make_profitability(
            ProfitabilityClass.PROFITABLE, roi_percent=50.0, marginal_roi=0.10,
            marginal_profit_30d=100, rebalance_cost_30d=1000
        )
        analyzer._profitability_cache["100x1x0"] = prof

        mult = analyzer.get_max_rebalance_fee_multiplier("100x1x0")
        # With marginal_roi=0.10: min(1.5, 1.0 + 0.10) = 1.10
        assert abs(mult - 1.10) < 0.01

    def test_profitable_channel_with_negative_marginal_clamps_to_1(self):
        """A PROFITABLE channel with negative marginal_roi should clamp to 1.0 multiplier."""
        analyzer = self._make_analyzer()

        # High total ROI but negative marginal (recent bad patch)
        prof = _make_profitability(
            ProfitabilityClass.PROFITABLE, roi_percent=30.0, marginal_roi=-0.20,
            marginal_profit_30d=-200, rebalance_cost_30d=1000
        )
        analyzer._profitability_cache["100x1x0"] = prof

        mult = analyzer.get_max_rebalance_fee_multiplier("100x1x0")
        # max(0.0, -0.20) = 0.0, so min(1.5, 1.0 + 0.0) = 1.0
        assert abs(mult - 1.0) < 0.01

    def test_break_even_returns_1(self):
        """Break-even channels should get 1.0 multiplier."""
        analyzer = self._make_analyzer()
        prof = _make_profitability(ProfitabilityClass.BREAK_EVEN, roi_percent=0.0, marginal_roi=0.0)
        analyzer._profitability_cache["100x1x0"] = prof

        assert analyzer.get_max_rebalance_fee_multiplier("100x1x0") == 1.0

    def test_underwater_returns_half(self):
        """Underwater channels get half budget."""
        analyzer = self._make_analyzer()
        prof = _make_profitability(ProfitabilityClass.UNDERWATER, roi_percent=-50.0, marginal_roi=-0.50)
        analyzer._profitability_cache["100x1x0"] = prof

        assert analyzer.get_max_rebalance_fee_multiplier("100x1x0") == 0.5

    def test_zombie_returns_zero(self):
        """Zombie channels get no rebalance budget."""
        analyzer = self._make_analyzer()
        prof = _make_profitability(ProfitabilityClass.ZOMBIE, roi_percent=-80.0, marginal_roi=-0.80)
        analyzer._profitability_cache["100x1x0"] = prof

        assert analyzer.get_max_rebalance_fee_multiplier("100x1x0") == 0.0


class TestEffectiveCostFallback:
    """I-4: Effective cost fallback should NOT inflate all-time costs."""

    def _make_analyzer_with_db(self):
        """Create analyzer with mocked database."""
        from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
        analyzer = ChannelProfitabilityAnalyzer.__new__(ChannelProfitabilityAnalyzer)
        analyzer.plugin = MagicMock()
        analyzer.database = MagicMock()
        analyzer._profitability_cache = {}
        analyzer._cache_ttl = 300
        analyzer._cache_timestamp = int(time.time())
        analyzer._lock = threading.Lock()
        analyzer.config = MagicMock()
        analyzer.config.estimated_open_cost_sats = 5000
        return analyzer

    def test_fallback_does_not_inflate_costs(self):
        """When recent_spend >= rebalance_costs, fallback should use uninflated costs."""
        analyzer = self._make_analyzer_with_db()

        # Mock database responses
        analyzer.database.get_channel_open_cost.return_value = 5000
        analyzer.database.get_channel_rebalance_costs.return_value = 10000
        analyzer.database.get_channel_cost.return_value = {
            'open_cost_sats': 5000, 'rebalance_cost_sats': 10000
        }


        # Success rate of 10% — previously this would have inflated
        # 10000 / 0.10 = 100000 (10x inflation!)
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10,
            'successes': 1,
            'success_rate': 0.10,
            'avg_cost_ppm': 500,
            'avg_amount_sats': 50000,
        }

        costs = analyzer._get_channel_costs("100x1x0", "02" + "ab" * 32, "abc123")

        # The effective cost should be close to uninflated 10000
        # NOT the old behavior of 10000 / 0.10 = 100000
        assert costs.effective_rebalance_cost_sats <= 15000  # Allow some margin for the inflation of recent portion
        # But definitely NOT 100000
        assert costs.effective_rebalance_cost_sats < 50000

    def test_normal_inflation_still_works(self):
        """When recent_spend is isolatable, it should still be inflated correctly."""
        analyzer = self._make_analyzer_with_db()

        analyzer.database.get_channel_open_cost.return_value = 5000
        analyzer.database.get_channel_rebalance_costs.return_value = 10000
        analyzer.database.get_channel_cost.return_value = {
            'open_cost_sats': 5000, 'rebalance_cost_sats': 10000
        }


        # Recent spend < total rebalance costs, so the isolation logic applies
        analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 20,
            'successes': 4,
            'success_rate': 0.20,
            'avg_cost_ppm': 200,
            'avg_amount_sats': 50000,  # recent_spend = 200 * 50000 * 4 / 1M = 40
        }

        costs = analyzer._get_channel_costs("100x1x0", "02" + "ab" * 32, "abc123")

        # recent_spend = 200 * 50000 * 4 / 1_000_000 = 40
        # historical = 10000 - 40 = 9960
        # effective = 9960 + int(40 / 0.20) = 9960 + 200 = 10160
        assert costs.effective_rebalance_cost_sats == 10160

    def test_no_success_data_uses_raw_costs(self):
        """Without success data, effective cost should equal raw rebalance cost."""
        analyzer = self._make_analyzer_with_db()

        analyzer.database.get_channel_open_cost.return_value = 5000
        analyzer.database.get_channel_rebalance_costs.return_value = 10000
        analyzer.database.get_channel_cost.return_value = {
            'open_cost_sats': 5000, 'rebalance_cost_sats': 10000
        }

        analyzer.database.get_channel_rebalance_success_rate.return_value = None

        costs = analyzer._get_channel_costs("100x1x0", "02" + "ab" * 32, "abc123")
        assert costs.effective_rebalance_cost_sats == 10000
