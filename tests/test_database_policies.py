"""Tests for policy CRUD methods in database.py."""

import os
import sys
import time
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


@pytest.fixture
def db(tmp_path):
    """Create a test database with peer_policies table."""
    db_path = str(tmp_path / "test.db")
    # Use a mock plugin for Database construction
    class MockPlugin:
        def log(self, msg, level='info'):
            pass
    database = Database(db_path, MockPlugin())
    database.initialize()
    return database


class TestGetAllPolicies:
    def test_empty(self, db):
        assert db.get_all_policies() == []

    def test_returns_rows(self, db):
        conn = db._get_connection()
        conn.execute("""
            INSERT INTO peer_policies (peer_id, strategy, rebalance_mode, updated_at)
            VALUES ('peer1', 'dynamic', 'enabled', 1000)
        """)
        rows = db.get_all_policies()
        assert len(rows) == 1
        assert rows[0]["peer_id"] == "peer1"

    def test_ordered_by_updated_at_desc(self, db):
        conn = db._get_connection()
        conn.execute("INSERT INTO peer_policies (peer_id, strategy, rebalance_mode, updated_at) VALUES ('a', 'dynamic', 'enabled', 100)")
        conn.execute("INSERT INTO peer_policies (peer_id, strategy, rebalance_mode, updated_at) VALUES ('b', 'dynamic', 'enabled', 200)")
        rows = db.get_all_policies()
        assert rows[0]["peer_id"] == "b"
        assert rows[1]["peer_id"] == "a"


class TestGetPolicy:
    def test_not_found(self, db):
        assert db.get_policy("nonexistent") is None

    def test_found(self, db):
        conn = db._get_connection()
        conn.execute("INSERT INTO peer_policies (peer_id, strategy, rebalance_mode, updated_at) VALUES ('peer1', 'dynamic', 'enabled', 1000)")
        row = db.get_policy("peer1")
        assert row["peer_id"] == "peer1"
        assert row["strategy"] == "dynamic"


class TestUpsertPolicy:
    def test_insert_new(self, db):
        db.upsert_policy("peer1", "dynamic", "enabled", 500, "[]", 1000, 0.8, 1.2, None)
        row = db.get_policy("peer1")
        assert row is not None
        assert row["fee_ppm_target"] == 500

    def test_update_existing(self, db):
        db.upsert_policy("peer1", "dynamic", "enabled", 500, "[]", 1000, 0.8, 1.2, None)
        db.upsert_policy("peer1", "aggressive", "disabled", 999, "[]", 2000, 0.5, 2.0, None)
        row = db.get_policy("peer1")
        assert row["strategy"] == "aggressive"
        assert row["fee_ppm_target"] == 999


class TestDeletePolicy:
    def test_delete_existing(self, db):
        db.upsert_policy("peer1", "dynamic", "enabled", 500, "[]", 1000, 0.8, 1.2, None)
        assert db.delete_policy("peer1") is True
        assert db.get_policy("peer1") is None

    def test_delete_nonexistent(self, db):
        assert db.delete_policy("nonexistent") is False


class TestDeleteExpiredPolicies:
    def test_deletes_expired(self, db):
        now = int(time.time())
        db.upsert_policy("expired", "dynamic", "enabled", 500, "[]", now, 0.8, 1.2, now - 100)
        db.upsert_policy("active", "dynamic", "enabled", 500, "[]", now, 0.8, 1.2, now + 3600)
        expired_ids = db.delete_expired_policies(now)
        assert expired_ids == ["expired"]
        assert db.get_policy("expired") is None
        assert db.get_policy("active") is not None

    def test_no_expired(self, db):
        now = int(time.time())
        db.upsert_policy("active", "dynamic", "enabled", 500, "[]", now, 0.8, 1.2, now + 3600)
        assert db.delete_expired_policies(now) == []

    def test_null_expires_not_deleted(self, db):
        now = int(time.time())
        db.upsert_policy("permanent", "dynamic", "enabled", 500, "[]", now, 0.8, 1.2, None)
        assert db.delete_expired_policies(now) == []
        assert db.get_policy("permanent") is not None


class TestUpsertPoliciesBatch:
    def test_batch_insert(self, db):
        rows = [
            ("peer1", "dynamic", "enabled", 500, "[]", 1000, 0.8, 1.2, None),
            ("peer2", "aggressive", "disabled", 999, "[]", 1000, 0.5, 2.0, None),
        ]
        db.upsert_policies_batch(rows)
        assert db.get_policy("peer1") is not None
        assert db.get_policy("peer2") is not None

    def test_batch_is_atomic(self, db):
        """If any row fails, none should be inserted."""
        rows = [
            ("peer1", "dynamic", "enabled", 500, "[]", 1000, 0.8, 1.2, None),
        ]
        db.upsert_policies_batch(rows)
        assert db.get_policy("peer1") is not None


class TestGetPolicyChangesSince:
    def test_returns_changes_after_timestamp(self, db):
        db.upsert_policy("old", "dynamic", "enabled", 500, "[]", 100, 0.8, 1.2, None)
        db.upsert_policy("new", "dynamic", "enabled", 500, "[]", 200, 0.8, 1.2, None)
        rows = db.get_policy_changes_since(150)
        assert len(rows) == 1
        assert rows[0]["peer_id"] == "new"

    def test_empty_when_none(self, db):
        assert db.get_policy_changes_since(0) == []


class TestGetLastPolicyChangeTimestamp:
    def test_returns_max(self, db):
        db.upsert_policy("a", "dynamic", "enabled", 500, "[]", 100, 0.8, 1.2, None)
        db.upsert_policy("b", "dynamic", "enabled", 500, "[]", 200, 0.8, 1.2, None)
        assert db.get_last_policy_change_timestamp() == 200

    def test_empty_returns_zero(self, db):
        assert db.get_last_policy_change_timestamp() == 0


class TestGetTotalCapexByChannel:
    def test_sums_rebalance_costs_and_spend_events(self, db):
        now = int(time.time())
        conn = db._get_connection()
        conn.execute("INSERT INTO rebalance_costs (channel_id, peer_id, cost_sats, amount_sats, timestamp) VALUES ('ch1', 'p1', 100, 10000, ?)", (now,))
        conn.execute("INSERT INTO spend_events (event_id, category, amount_sats, channel_id, timestamp) VALUES ('ev1', 'open', 200, 'ch1', ?)", (now,))
        result = db.get_total_capex_by_channel(window_days=30)
        assert result["ch1"] == 300

    def test_excludes_old_data(self, db):
        old = int(time.time()) - 90 * 86400
        conn = db._get_connection()
        conn.execute("INSERT INTO rebalance_costs (channel_id, peer_id, cost_sats, amount_sats, timestamp) VALUES ('ch1', 'p1', 100, 10000, ?)", (old,))
        result = db.get_total_capex_by_channel(window_days=30)
        assert result == {}

    def test_ignores_null_channel_id(self, db):
        now = int(time.time())
        conn = db._get_connection()
        conn.execute("INSERT INTO spend_events (event_id, category, amount_sats, channel_id, timestamp) VALUES ('ev1', 'treasury', 500, NULL, ?)", (now,))
        result = db.get_total_capex_by_channel(window_days=30)
        assert result == {}


class TestCleanupOrphanedRebalances:
    def test_marks_orphans_as_failed(self, db):
        old = int(time.time()) - 7200  # 2 hours ago
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO rebalance_history (id, from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, status, timestamp) VALUES (1, 'ch1', 'ch2', 1000, 500, 0, 'pending', ?)",
            (old,)
        )
        orphan_ids = db.cleanup_orphaned_rebalances(timeout_seconds=3600)
        assert orphan_ids == [1]
        row = conn.execute("SELECT status, error_message FROM rebalance_history WHERE id = 1").fetchone()
        assert row["status"] == "failed"
        assert row["error_message"] == "orphaned_on_restart"

    def test_skips_recent_pending(self, db):
        now = int(time.time())
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO rebalance_history (id, from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, status, timestamp) VALUES (1, 'ch1', 'ch2', 1000, 500, 0, 'pending', ?)",
            (now,)
        )
        orphan_ids = db.cleanup_orphaned_rebalances(timeout_seconds=3600)
        assert orphan_ids == []

    def test_skips_completed(self, db):
        old = int(time.time()) - 7200
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO rebalance_history (id, from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, status, timestamp) VALUES (1, 'ch1', 'ch2', 1000, 500, 0, 'completed', ?)",
            (old,)
        )
        orphan_ids = db.cleanup_orphaned_rebalances(timeout_seconds=3600)
        assert orphan_ids == []
