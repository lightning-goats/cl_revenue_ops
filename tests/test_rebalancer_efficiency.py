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
