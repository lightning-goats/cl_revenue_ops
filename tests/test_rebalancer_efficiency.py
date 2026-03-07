"""Tests for rebalancer efficiency improvements (failure-informed routing + hive fixes)."""
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


# =============================================================================
# Task 3: Classify sling failure messages
# =============================================================================

class TestFailureClassification:
    """Verify sling error messages are classified correctly."""

    def test_no_route_classified(self):
        from modules.rebalancer import JobManager
        assert JobManager._classify_sling_error("no route found") == "no_route"

    def test_no_route_variant(self):
        from modules.rebalancer import JobManager
        assert JobManager._classify_sling_error("WIRE_UNKNOWN_NEXT_PEER") == "no_route"

    def test_timeout_classified(self):
        from modules.rebalancer import JobManager
        assert JobManager._classify_sling_error("timeout waiting for response") == "timeout"

    def test_budget_exceeded_classified(self):
        from modules.rebalancer import JobManager
        assert JobManager._classify_sling_error("exceeded fee budget") == "budget_exceeded"

    def test_unknown_error_is_other(self):
        from modules.rebalancer import JobManager
        assert JobManager._classify_sling_error("something weird happened") == "other"

    def test_empty_error_is_other(self):
        from modules.rebalancer import JobManager
        assert JobManager._classify_sling_error("") == "other"


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
