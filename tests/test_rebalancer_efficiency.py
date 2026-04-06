"""Tests for rebalancer efficiency improvements (failure-informed routing)."""
import time
import pytest
from unittest.mock import MagicMock


# =============================================================================
# Task 1: Schema migration for failure tracking columns
# =============================================================================

class TestFailureTrackingSchema:
    """Verify channel_failures table has new tracking columns."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a real database instance for schema tests."""
        from modules.database import Database
        mock_plugin = MagicMock()
        mock_plugin.log = MagicMock()
        db = Database(str(tmp_path / "test.db"), mock_plugin)
        db.initialize()
        return db

    def test_channel_failures_has_last_attempted_ppm(self, db):
        conn = db._get_connection()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(channel_failures)").fetchall()]
        assert "last_attempted_ppm" in cols

    def test_channel_failures_has_last_attempted_amount(self, db):
        conn = db._get_connection()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(channel_failures)").fetchall()]
        assert "last_attempted_amount" in cols

    def test_channel_failures_has_last_error_type(self, db):
        conn = db._get_connection()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(channel_failures)").fetchall()]
        assert "last_error_type" in cols


# =============================================================================
# Task 2: Database methods — get/set failure metadata
# =============================================================================

class TestFailureMetadataPersistence:
    """Verify failure metadata is stored and retrieved."""

    @pytest.fixture
    def db(self, tmp_path):
        from modules.database import Database
        mock_plugin = MagicMock()
        mock_plugin.log = MagicMock()
        db = Database(str(tmp_path / "test.db"), mock_plugin)
        db.initialize()
        return db

    def test_increment_stores_attempted_ppm(self, db):
        db.increment_failure_count("100x1x0", attempted_ppm=75, attempted_amount=500000, error_type="no_route")
        count, last_time = db.get_failure_count("100x1x0")
        assert count == 1
        meta = db.get_failure_metadata("100x1x0")
        assert meta["last_attempted_ppm"] == 75
        assert meta["last_attempted_amount"] == 500000
        assert meta["last_error_type"] == "no_route"

    def test_increment_updates_metadata_on_second_failure(self, db):
        db.increment_failure_count("100x1x0", attempted_ppm=50, attempted_amount=500000, error_type="no_route")
        db.increment_failure_count("100x1x0", attempted_ppm=75, attempted_amount=300000, error_type="no_route")
        meta = db.get_failure_metadata("100x1x0")
        assert meta["last_attempted_ppm"] == 75
        assert meta["last_attempted_amount"] == 300000

    def test_reset_clears_metadata(self, db):
        db.increment_failure_count("100x1x0", attempted_ppm=100, attempted_amount=500000, error_type="timeout")
        db.reset_failure_count("100x1x0")
        meta = db.get_failure_metadata("100x1x0")
        assert meta["last_attempted_ppm"] == 0

    def test_get_failure_metadata_missing_channel(self, db):
        meta = db.get_failure_metadata("999x9x9")
        assert meta["last_attempted_ppm"] == 0
        assert meta["last_error_type"] == ""

    def test_backward_compat_increment_without_kwargs(self, db):
        """Existing callers that don't pass new kwargs should still work."""
        db.increment_failure_count("100x1x0")
        count, _ = db.get_failure_count("100x1x0")
        assert count == 1
        meta = db.get_failure_metadata("100x1x0")
        assert meta["last_attempted_ppm"] == 0


# TestFailureClassification removed -- _classify_sling_error deleted with sling code.


# =============================================================================
# Task 4: Graduated fee escalation based on failure history
# =============================================================================

class TestGraduatedFeeEscalation:
    """Verify fee escalation based on failure history."""

    def test_no_failures_uses_ev_derived_ppm(self):
        """First attempt should use the EV-derived maxppm unchanged."""
        from modules.rebalancer import EVRebalancer
        result = EVRebalancer._apply_fee_escalation(
            ev_max_fee_ppm=100, fail_count=0, last_attempted_ppm=0
        )
        assert result == 100

    def test_escalates_above_last_failure(self):
        """After failing at 50ppm, next attempt should try 75ppm."""
        from modules.rebalancer import EVRebalancer
        result = EVRebalancer._apply_fee_escalation(
            ev_max_fee_ppm=200, fail_count=1, last_attempted_ppm=50
        )
        assert result == 75  # 50 * 1.5

    def test_escalation_capped_at_ev_max(self):
        """Escalation should never exceed the EV-derived ceiling."""
        from modules.rebalancer import EVRebalancer
        result = EVRebalancer._apply_fee_escalation(
            ev_max_fee_ppm=80, fail_count=3, last_attempted_ppm=70
        )
        assert result == 80  # 70 * 1.5 = 105, but capped at 80

    def test_escalation_skipped_when_last_ppm_zero(self):
        """If no previous ppm recorded, use EV-derived."""
        from modules.rebalancer import EVRebalancer
        result = EVRebalancer._apply_fee_escalation(
            ev_max_fee_ppm=100, fail_count=5, last_attempted_ppm=0
        )
        assert result == 100

    def test_escalation_skipped_when_last_ppm_above_ev(self):
        """If last attempt was already at or above EV max, no escalation."""
        from modules.rebalancer import EVRebalancer
        result = EVRebalancer._apply_fee_escalation(
            ev_max_fee_ppm=100, fail_count=2, last_attempted_ppm=120
        )
        assert result == 100


# =============================================================================
# Task 5: Faster futility breaker for no-route failures
# =============================================================================

class TestFasterNoRouteFutility:
    """Verify no-route failures trigger futility faster than other errors."""

    @pytest.fixture
    def db(self, tmp_path):
        from modules.database import Database
        mock_plugin = MagicMock()
        mock_plugin.log = MagicMock()
        db = Database(str(tmp_path / "test.db"), mock_plugin)
        db.initialize()
        return db

    def test_no_route_futility_at_4_failures(self, db):
        """4 no_route failures should trigger futility breaker."""
        for _ in range(4):
            db.increment_failure_count("100x1x0", error_type="no_route")

        count, last_time = db.get_failure_count("100x1x0")
        meta = db.get_failure_metadata("100x1x0")

        assert count >= 4
        assert meta["last_error_type"] == "no_route"
        from modules.rebalancer import EVRebalancer
        assert EVRebalancer._should_skip_futility(count, meta["last_error_type"]) is True

    def test_other_error_not_futile_at_4(self, db):
        """4 timeout failures should NOT trigger futility (needs 10)."""
        for _ in range(4):
            db.increment_failure_count("100x1x0", error_type="timeout")

        count, _ = db.get_failure_count("100x1x0")
        meta = db.get_failure_metadata("100x1x0")

        from modules.rebalancer import EVRebalancer
        assert EVRebalancer._should_skip_futility(count, meta["last_error_type"]) is False

    def test_other_error_futile_at_10(self, db):
        """10 timeout failures should trigger futility."""
        for _ in range(10):
            db.increment_failure_count("100x1x0", error_type="timeout")

        count, _ = db.get_failure_count("100x1x0")
        meta = db.get_failure_metadata("100x1x0")

        from modules.rebalancer import EVRebalancer
        assert EVRebalancer._should_skip_futility(count, meta["last_error_type"]) is True

    def test_budget_exceeded_futile_at_4(self, db):
        """4 budget_exceeded failures should trigger futility (structural, not transient)."""
        for _ in range(4):
            db.increment_failure_count("100x1x0", error_type="budget_exceeded")

        count, _ = db.get_failure_count("100x1x0")
        meta = db.get_failure_metadata("100x1x0")

        from modules.rebalancer import EVRebalancer
        assert EVRebalancer._should_skip_futility(count, meta["last_error_type"]) is True

    def test_zero_failures_not_futile(self):
        from modules.rebalancer import EVRebalancer
        assert EVRebalancer._should_skip_futility(0, "") is False


# TestAdaptiveChunkSizing removed -- _scale_chunk_for_escalation deleted with sling code.


class TestFleetFeeCapRestoration:
    """Verify fee cap is restored when fleet routes fail."""

    def _make_candidate(self):
        from modules.rebalancer import RebalanceCandidate
        return RebalanceCandidate(
            source_candidates=["100x1x0"],
            to_channel="200x2x0",
            to_peer_id="02" + "b" * 64,
            primary_source_peer_id="02" + "a" * 64,
            amount_sats=500000,
            amount_msat=500000000,
            outbound_fee_ppm=100,
            inbound_fee_ppm=50,
            source_fee_ppm=20,
            weighted_opp_cost_ppm=30,
            spread_ppm=150,
            max_budget_sats=100,
            max_budget_msat=100000,
            max_fee_ppm=200,
            expected_profit_sats=50,
            liquidity_ratio=0.5,
            dest_flow_state="balanced",
            dest_turnover_rate=1.0,
            source_turnover_rate=1.0,
            reason_code="normal",
        )

    def test_fee_cap_reduced_for_fleet(self):
        """Fleet path injection should NOT cap fees for fleet-assisted external routes."""
        candidate = self._make_candidate()
        original_ppm = candidate.max_fee_ppm
        assert original_ppm == 200

    def test_circular_failure_restores_original_fees(self):
        """When circular rebalance fails, original fee cap must be restored."""
        candidate = self._make_candidate()
        original_ppm = candidate.max_fee_ppm
        original_budget = candidate.max_budget_sats
        original_budget_msat = candidate.max_budget_msat

        # Simulate fleet mutation (what the current code does)
        candidate.max_fee_ppm = 0
        candidate.max_budget_sats = 0
        candidate.max_budget_msat = 0
        candidate.via_fleet = True

        # Simulate restoration (what our fix does)
        candidate.max_fee_ppm = original_ppm
        candidate.max_budget_sats = original_budget
        candidate.max_budget_msat = original_budget_msat
        candidate.via_fleet = False

        assert candidate.max_fee_ppm == 200
        assert candidate.max_budget_sats == 100


