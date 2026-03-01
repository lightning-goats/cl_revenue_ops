"""
Tests for Database module — specifically get_channel_rebalance_success_rate.

Uses real SQLite (temp files) to verify actual SQL logic.
"""

import time
import os
import sys
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


class TestChannelRebalanceSuccessRate:
    """Real SQLite tests for get_channel_rebalance_success_rate."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_sr.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_success_rate_calculation(self, tmp_path):
        """Insert mix of success/failed, verify rate."""
        db = self._make_db(tmp_path)
        conn = db._get_connection()
        now = int(time.time())
        channel = "111x222x0"

        # Insert 6 successes and 4 failures = 60% success rate
        for i in range(6):
            conn.execute(
                "INSERT INTO rebalance_history "
                "(from_channel, to_channel, amount_sats, max_fee_sats, actual_fee_sats, "
                "expected_profit_sats, status, rebalance_type, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("src", channel, 50000, 100, 10, 50, "success", "normal", now - i * 3600)
            )
        for i in range(4):
            conn.execute(
                "INSERT INTO rebalance_history "
                "(from_channel, to_channel, amount_sats, max_fee_sats, "
                "expected_profit_sats, status, rebalance_type, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("src", channel, 50000, 100, 50, "failed", "normal", now - (6 + i) * 3600)
            )
        conn.commit()

        result = db.get_channel_rebalance_success_rate(channel, 30)

        assert result is not None
        assert result['total'] == 10
        assert result['successes'] == 6
        assert result['failures'] == 4
        assert abs(result['success_rate'] - 0.6) < 0.01

    def test_success_rate_window_filtering(self, tmp_path):
        """Insert old + new records, verify window works."""
        db = self._make_db(tmp_path)
        conn = db._get_connection()
        now = int(time.time())
        channel = "111x222x0"

        # 2 recent successes (within 7 days)
        for i in range(2):
            conn.execute(
                "INSERT INTO rebalance_history "
                "(from_channel, to_channel, amount_sats, max_fee_sats, actual_fee_sats, "
                "expected_profit_sats, status, rebalance_type, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("src", channel, 50000, 100, 10, 50, "success", "normal", now - i * 3600)
            )

        # 3 old failures (40 days ago — outside 30-day window)
        for i in range(3):
            conn.execute(
                "INSERT INTO rebalance_history "
                "(from_channel, to_channel, amount_sats, max_fee_sats, "
                "expected_profit_sats, status, rebalance_type, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("src", channel, 50000, 100, 50, "failed", "normal", now - 40 * 86400 - i * 3600)
            )
        conn.commit()

        # 30-day window should only see the 2 recent successes
        result = db.get_channel_rebalance_success_rate(channel, 30)
        assert result is not None
        assert result['total'] == 2
        assert result['successes'] == 2
        assert result['success_rate'] == 1.0

        # 60-day window should see all 5
        result_wide = db.get_channel_rebalance_success_rate(channel, 60)
        assert result_wide is not None
        assert result_wide['total'] == 5
        assert result_wide['successes'] == 2
        assert abs(result_wide['success_rate'] - 0.4) < 0.01

    def test_success_rate_no_history(self, tmp_path):
        """No records -> returns None."""
        db = self._make_db(tmp_path)
        result = db.get_channel_rebalance_success_rate("999x999x0", 30)
        assert result is None


# =============================================================================
# Audit Round 8 – Turn 3 Regression Tests
# =============================================================================

class TestAuditTurn3ConfigOverrideVersioning:
    """P0-1 regression: set_config_override version monotonicity."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_config.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_sequential_overrides_produce_incrementing_versions(self, tmp_path):
        """Two sequential set_config_override calls produce distinct versions."""
        db = self._make_db(tmp_path)

        v1 = db.set_config_override("key_a", "value_a")
        v2 = db.set_config_override("key_b", "value_b")

        assert v2 > v1, f"Version should strictly increase: v1={v1}, v2={v2}"

    def test_override_same_key_increments_version(self, tmp_path):
        """Updating the same key should still increment version."""
        db = self._make_db(tmp_path)

        v1 = db.set_config_override("key_a", "val1")
        v2 = db.set_config_override("key_a", "val2")

        assert v2 > v1, f"Same-key update should increment: v1={v1}, v2={v2}"
        # Verify the value was actually updated
        assert db.get_config_override("key_a") == "val2"

    def test_version_survives_rollback_scenario(self, tmp_path):
        """After multiple writes, version reflects all changes."""
        db = self._make_db(tmp_path)

        v1 = db.set_config_override("a", "1")
        v2 = db.set_config_override("b", "2")
        v3 = db.set_config_override("c", "3")

        assert v3 == 3
        current = db.get_config_version()
        assert current == 3


class TestAuditTurn3KalmanNaNGuard:
    """P1-1 regression: save_kalman_state rejects NaN/Inf."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_kalman.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_nan_flow_ratio_not_persisted(self, tmp_path):
        """NaN in flow_ratio should prevent persistence."""
        db = self._make_db(tmp_path)

        db.save_kalman_state("100x1x0", {
            "flow_ratio": float('nan'),
            "flow_velocity": 0.1,
        })

        # Should not have been saved
        states = db.get_all_kalman_states()
        assert len(states) == 0
        db.plugin.log.assert_called()

    def test_inf_variance_not_persisted(self, tmp_path):
        """Inf in variance_ratio should prevent persistence."""
        db = self._make_db(tmp_path)

        db.save_kalman_state("100x1x0", {
            "flow_ratio": 0.5,
            "variance_ratio": float('inf'),
        })

        states = db.get_all_kalman_states()
        assert len(states) == 0

    def test_valid_state_persisted(self, tmp_path):
        """Normal finite values should be persisted successfully."""
        db = self._make_db(tmp_path)

        db.save_kalman_state("100x1x0", {
            "flow_ratio": 0.3,
            "flow_velocity": 0.05,
            "variance_ratio": 0.1,
            "variance_velocity": 0.1,
            "covariance": 0.01,
            "last_update": int(time.time()),
            "innovation_variance": 0.01,
            "last_innovation": 0.0,
        })

        states = db.get_all_kalman_states()
        assert len(states) == 1
        assert abs(states[0]["flow_ratio"] - 0.3) < 0.001
