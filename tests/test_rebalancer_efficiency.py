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
