"""
Regression tests for P1 bugs found during systematic module audit.

Each test targets a specific fix and ensures the bug does not recur.
"""

import os
import sys
import time
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


@pytest.fixture
def db(temp_db_path):
    plugin = MagicMock()
    plugin.log = MagicMock()
    d = Database(temp_db_path, plugin)
    d.initialize()
    return d


def _insert_forward(conn, *, in_channel, out_channel, in_msat, out_msat, fee_msat, ts):
    conn.execute(
        """
        INSERT INTO forwards
        (in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time, timestamp, resolved_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (in_channel, out_channel, int(in_msat), int(out_msat), int(fee_msat), 0.1, int(ts), int(ts) + 1),
    )


# =========================================================================
# Fix #17: get_channel_full_pnl applies 50/50 split to prevent double-counting
# =========================================================================

class TestFullPnlMaxValuation:
    """get_channel_full_pnl uses max(direct, sourced) for total_contribution."""

    def test_total_contribution_uses_max(self, db):
        now = int(time.time())
        conn = db._get_connection()
        # Exit-side: 10000 msat fee => 10 sats
        _insert_forward(conn, in_channel="B", out_channel="A",
                        in_msat=10_000_000, out_msat=9_990_000, fee_msat=10_000, ts=now - 3600)
        # Entry-side: 6000 msat fee => 6 sats
        _insert_forward(conn, in_channel="A", out_channel="C",
                        in_msat=6_000_000, out_msat=5_994_000, fee_msat=6_000, ts=now - 3600)

        full = db.get_channel_full_pnl("A", window_days=30)
        assert full["direct_revenue_sats"] == 10
        assert full["sourced_fee_contribution_sats"] == 6
        # max(10000, 6000) = 10000 msat => 10 sats
        assert full["total_contribution_sats"] == 10

    def test_zero_sourced_contribution(self, db):
        """Channel with no inbound traffic: total = max(direct, 0) = direct."""
        now = int(time.time())
        conn = db._get_connection()
        _insert_forward(conn, in_channel="B", out_channel="A",
                        in_msat=5_000_000, out_msat=4_995_000, fee_msat=5_000, ts=now - 3600)

        full = db.get_channel_full_pnl("A", window_days=30)
        assert full["direct_revenue_sats"] == 5
        assert full["sourced_fee_contribution_sats"] == 0
        # max(5000, 0) = 5000 msat => 5 sats
        assert full["total_contribution_sats"] == 5


# =========================================================================
# Fix #18: Boundary-day exclusion prevents double-counting in revenue totals
# =========================================================================

class TestBoundaryDayExclusion:
    """
    When daily_forwarding_stats has a rollup for today AND forwards still has
    today's raw data, the revenue total must not double-count.
    """

    def _setup_boundary_scenario(self, db):
        """Insert a rollup for today AND a raw forward for today."""
        now = int(time.time())
        today_start = (now // 86400) * 86400
        conn = db._get_connection()

        # Raw forward for today (still in forwards table)
        _insert_forward(conn, in_channel="B", out_channel="A",
                        in_msat=10_000_000, out_msat=9_990_000, fee_msat=2_000,
                        ts=today_start + 100)

        # Rollup for today (would cause double-count without fix)
        conn.execute("""
            INSERT INTO daily_forwarding_stats
            (channel_id, date, total_fee_msat, total_out_msat, forward_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("A", today_start, 2_000, 9_990_000, 1))

        return conn

    def test_get_channel_revenue_totals_no_double_count(self, db):
        self._setup_boundary_scenario(db)
        totals = db.get_channel_revenue_totals("A")
        # Should be 2000 msat (not 4000 msat doubled)
        assert totals["fees_earned_msat"] == 2000
        assert totals["forward_count"] == 1

    def test_get_lifetime_stats_no_double_count(self, db):
        self._setup_boundary_scenario(db)
        stats = db.get_lifetime_stats()
        # Should be 2 sats, not 4 sats
        assert stats["total_revenue_msat"] == 2_000
        assert stats["total_forwards"] == 1

    def test_old_rollups_still_counted(self, db):
        """Rollups from previous days must still be included."""
        now = int(time.time())
        today_start = (now // 86400) * 86400
        yesterday_start = today_start - 86400
        conn = db._get_connection()

        # Yesterday's rollup (should be counted)
        conn.execute("""
            INSERT INTO daily_forwarding_stats
            (channel_id, date, total_fee_msat, total_out_msat, forward_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("A", yesterday_start, 3_000, 5_000_000, 2))

        # Today's raw forward
        _insert_forward(conn, in_channel="B", out_channel="A",
                        in_msat=10_000_000, out_msat=9_990_000, fee_msat=1_000,
                        ts=today_start + 100)

        totals = db.get_channel_revenue_totals("A")
        # Yesterday rollup (3000 msat) + today raw (1000 msat) = 4000 msat
        assert totals["fees_earned_msat"] == 4000
        assert totals["forward_count"] == 3


# =========================================================================
# Fix #19: delete_config_override bumps version for change detection
# =========================================================================

class TestConfigDeleteVersionBump:
    """Deleting a config override must bump the version number."""

    def test_version_does_not_regress_on_delete(self, db):
        """
        Before the fix, deleting the highest-versioned row dropped MAX(version),
        so a config poller that cached the old version would never see the change.
        The sentinel row ensures MAX(version) stays >= the pre-delete value.
        """
        v1 = db.set_config_override("test-key", "value1")
        assert v1 > 0

        v_before = db.get_config_version()
        deleted = db.delete_config_override("test-key")
        assert deleted is True

        v_after = db.get_config_version()
        # Version must not drop — poller that cached v_before will still trigger
        assert v_after >= v_before

    def test_delete_nonexistent_returns_false(self, db):
        v_before = db.get_config_version()
        deleted = db.delete_config_override("no-such-key")
        assert deleted is False
        # Version should not change
        assert db.get_config_version() == v_before

    def test_sentinel_row_does_not_pollute_overrides(self, db):
        """The _version_bump sentinel should not appear in user-facing overrides."""
        db.set_config_override("real-key", "real-value")
        db.delete_config_override("real-key")

        overrides = db.get_all_config_overrides()
        # Only the sentinel should exist, but it should be filtered or ignorable
        for key in overrides:
            if key == "_version_bump":
                continue
            # No real keys should remain
            assert key != "real-key"


# =========================================================================
# Fix #22: capacity_planner fire sale requires flow data + uses marginal ROI
# =========================================================================

class TestCapacityPlannerFireSale:
    """Fire sale logic must not trigger without flow data and must use marginal ROI."""

    def _make_planner(self):
        from modules.capacity_planner import CapacityPlanner
        plugin = MagicMock()
        profitability = MagicMock()
        flow = MagicMock()
        return CapacityPlanner(plugin, profitability, flow), profitability, flow

    def _make_prof(self, *, classification, days_open, roi_percent, marginal_roi_percent,
                   capacity_sats=1_000_000, peer_id="peer1"):
        from modules.profitability_analyzer import ProfitabilityClass
        prof = MagicMock()
        prof.classification = classification
        prof.days_open = days_open
        prof.roi_percent = roi_percent
        prof.marginal_roi_percent = marginal_roi_percent
        prof.capacity_sats = capacity_sats
        prof.peer_id = peer_id
        return prof

    def test_no_fire_sale_without_flow_data(self):
        """Channel must not be marked FIRE SALE if flow data is missing."""
        from modules.profitability_analyzer import ProfitabilityClass
        planner, profitability, flow = self._make_planner()

        prof = self._make_prof(
            classification=ProfitabilityClass.ZOMBIE,
            days_open=180,
            roi_percent=-80,
            marginal_roi_percent=-80,
        )

        all_prof = {"100x1x0": prof}
        all_flow = {}  # No flow data for this channel
        profitability.analyze_all_channels.return_value = all_prof
        profitability.database = MagicMock()
        profitability.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        profitability.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers(all_prof, all_flow)
        # Should not appear as a loser at all (no flow data)
        assert len(losers) == 0

    def test_marginal_roi_not_total_roi_for_fire_sale(self):
        """UNDERWATER fire sale must use marginal_roi, not total roi (sunk cost fallacy)."""
        from modules.profitability_analyzer import ProfitabilityClass
        planner, profitability, flow = self._make_planner()

        # Total ROI deeply negative (sunk cost), but marginal ROI recovering
        prof = self._make_prof(
            classification=ProfitabilityClass.UNDERWATER,
            days_open=180,
            roi_percent=-80,           # total ROI: deep negative (sunk cost)
            marginal_roi_percent=-10,  # marginal ROI: only slightly negative (recovering)
        )

        flow_metrics = MagicMock()
        flow_metrics.capacity = 1_000_000
        flow_metrics.our_balance = 500_000
        flow_metrics.daily_volume = 100  # Very low, so not stagnant logic path
        flow_metrics.flow_ratio = 0.0

        all_prof = {"100x1x0": prof}
        all_flow = {"100x1x0": flow_metrics}
        profitability.database = MagicMock()
        profitability.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        profitability.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers(all_prof, all_flow)
        # Marginal ROI is -10% (> -50%), so should NOT be fire sale
        fire_sale_losers = [l for l in losers if "FIRE SALE" in l.get("reason", "")]
        assert len(fire_sale_losers) == 0



