"""
Tests for database maintenance/audit fixes:

1. close_all_connections() actually closes connections created by other
   threads (check_same_thread=False).
2. get_channel_cost_history() optional since_timestamp filter.
3. cleanup_old_data() prunes terminal budget/spend reservations and stale
   pair_rebalance_failures; remove_closed_channel_data() removes pair rows.
4. record_pair_rebalance_failure() increments atomically under concurrency.
5. delete_expired_policies() returns exactly what it deleted (transactional).
6. set_config_override() returns the version written inside its transaction.

Uses real SQLite (temp files) to verify actual SQL logic.
"""

import os
import sys
import sqlite3
import threading
import time
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


def _make_db(tmp_path):
    db_path = os.path.join(str(tmp_path), "test_maintenance.db")
    plugin = MagicMock()
    db = Database(db_path, plugin)
    db.initialize()
    return db


NOW = int(time.time())
OLD = NOW - (120 * 86400)   # well past the 90-day ledger cutoff
FRESH = NOW - 3600          # recent


class TestCloseAllConnectionsCrossThread:
    """Fix 1: shutdown must close connections owned by worker threads."""

    def test_worker_thread_connection_closed_from_main_thread(self, tmp_path):
        db = _make_db(tmp_path)
        worker_conn = {}
        errors = []

        def worker():
            try:
                worker_conn['conn'] = db._get_connection()
                worker_conn['conn'].execute("SELECT 1").fetchone()
            except Exception as e:  # pragma: no cover
                errors.append(e)

        t = threading.Thread(target=worker, name="db-worker")
        t.start()
        t.join()
        assert not errors
        assert 'conn' in worker_conn

        # Shutdown from the main thread must actually close the worker's conn
        db.close_all_connections()

        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            worker_conn['conn'].execute("SELECT 1")

    def test_tracking_list_emptied_after_close_all(self, tmp_path):
        db = _make_db(tmp_path)
        db._get_connection()
        t = threading.Thread(target=db._get_connection)
        t.start()
        t.join()
        db.close_all_connections()
        with db._thread_conn_lock:
            assert db._thread_connections == []


class TestChannelCostHistorySince:
    """Fix 2: optional since_timestamp filter on get_channel_cost_history."""

    def _seed(self, db, channel="111x222x0"):
        conn = db._get_connection()
        for ts in (NOW - 86400 * 30, NOW - 86400 * 10, NOW - 3600):
            conn.execute(
                "INSERT INTO rebalance_costs "
                "(channel_id, peer_id, cost_sats, amount_sats, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (channel, "peer1", 10, 50000, ts),
            )
        return channel

    def test_default_returns_full_history(self, tmp_path):
        db = _make_db(tmp_path)
        channel = self._seed(db)
        rows = db.get_channel_cost_history(channel)
        assert len(rows) == 3
        # Newest first
        assert rows[0]['timestamp'] >= rows[1]['timestamp'] >= rows[2]['timestamp']

    def test_since_timestamp_filters_old_rows(self, tmp_path):
        db = _make_db(tmp_path)
        channel = self._seed(db)
        cutoff = NOW - 86400 * 14
        rows = db.get_channel_cost_history(channel, since_timestamp=cutoff)
        assert len(rows) == 2
        assert all(r['timestamp'] >= cutoff for r in rows)

    def test_since_timestamp_none_is_unfiltered(self, tmp_path):
        db = _make_db(tmp_path)
        channel = self._seed(db)
        assert db.get_channel_cost_history(channel, since_timestamp=None) == \
            db.get_channel_cost_history(channel)


class TestCleanupLedgerTables:
    """Fix 3: cleanup_old_data prunes terminal reservation rows and stale
    pair failures, but never touches active rows, fresh rows, or
    spend_events / rebalance_costs."""

    def _seed_reservations(self, db):
        conn = db._get_connection()
        rows = [
            # (id, status, reserved_at) — budget_reservations
            ("old-spent", "spent", OLD),
            ("old-released", "released", OLD),
            ("old-active", "active", OLD),
            ("fresh-spent", "spent", FRESH),
            ("fresh-active", "active", FRESH),
        ]
        for rid, status, ts in rows:
            conn.execute(
                "INSERT INTO budget_reservations "
                "(reservation_id, reserved_sats, reserved_at, job_channel_id, status) "
                "VALUES (?, 100, ?, 'chan', ?)",
                (rid, ts, status),
            )
            conn.execute(
                "INSERT INTO spend_reservations "
                "(reservation_id, category, reserved_sats, reserved_at, status) "
                "VALUES (?, 'rebalance', 100, ?, ?)",
                (rid, ts, status),
            )

    def test_old_terminal_reservations_pruned(self, tmp_path):
        db = _make_db(tmp_path)
        self._seed_reservations(db)
        db.cleanup_old_data()
        conn = db._get_connection()
        for table in ("budget_reservations", "spend_reservations"):
            remaining = {
                r["reservation_id"]
                for r in conn.execute(f"SELECT reservation_id FROM {table}").fetchall()
            }
            assert remaining == {"old-active", "fresh-spent", "fresh-active"}, table

    def test_stale_pair_failures_pruned(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        seed = [
            # (src, dst, last_failure_at, cooldown_until, expect_kept)
            ("100x1x0", "100x2x0", OLD, OLD, False),       # fully stale
            ("100x3x0", "100x4x0", OLD, NOW + 3600, True),  # cooldown still live
            ("100x5x0", "100x6x0", FRESH, FRESH, True),     # recent failure
        ]
        for src, dst, last_at, cd, _ in seed:
            conn.execute(
                "INSERT INTO pair_rebalance_failures "
                "(source_channel_id, dest_channel_id, failure_kind, "
                "failure_count, last_failure_at, cooldown_until) "
                "VALUES (?, ?, 'other_retriable', 3, ?, ?)",
                (src, dst, last_at, cd),
            )
        db.cleanup_old_data()
        remaining = {
            (r["source_channel_id"], r["dest_channel_id"])
            for r in conn.execute(
                "SELECT source_channel_id, dest_channel_id FROM pair_rebalance_failures"
            ).fetchall()
        }
        assert remaining == {(s, d) for s, d, _, _, keep in seed if keep}

    def test_spend_events_never_pruned(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO spend_events (event_id, category, amount_sats, timestamp) "
            "VALUES ('ancient', 'rebalance', 100, ?)",
            (OLD,),
        )
        db.cleanup_old_data()
        count = conn.execute("SELECT COUNT(*) AS c FROM spend_events").fetchone()["c"]
        assert count == 1

    def test_remove_closed_channel_deletes_pair_failures(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        closed = "200x1x0"
        for src, dst in [(closed, "200x2x0"), ("200x3x0", closed), ("200x4x0", "200x5x0")]:
            conn.execute(
                "INSERT INTO pair_rebalance_failures "
                "(source_channel_id, dest_channel_id, failure_kind, "
                "failure_count, last_failure_at, cooldown_until) "
                "VALUES (?, ?, 'other_retriable', 1, ?, ?)",
                (src, dst, NOW, NOW + 3600),
            )
        deleted = db.remove_closed_channel_data(closed)
        assert deleted.get("pair_rebalance_failures") == 2
        remaining = conn.execute(
            "SELECT source_channel_id, dest_channel_id FROM pair_rebalance_failures"
        ).fetchall()
        assert len(remaining) == 1
        assert (remaining[0]["source_channel_id"], remaining[0]["dest_channel_id"]) == \
            ("200x4x0", "200x5x0")


class TestPairFailureAtomicIncrement:
    """Fix 4: record_pair_rebalance_failure must not lose increments."""

    def test_semantics_preserved(self, tmp_path):
        db = _make_db(tmp_path)
        base = 1000
        first = db.record_pair_rebalance_failure(
            "300x1x0", "300x2x0", "no_route", base, now=5000)
        assert first["failure_count"] == 1
        assert first["cooldown_until"] == 5000 + base * 1
        assert first["failure_kind"] == "no_route"
        assert first["last_failure_at"] == 5000

        second = db.record_pair_rebalance_failure(
            "300x1x0", "300x2x0", "fee_too_high", base, now=6000)
        assert second["failure_count"] == 2
        assert second["cooldown_until"] == 6000 + base * 2
        assert second["failure_kind"] == "fee_too_high"

        # Multiplier caps at 6
        for i in range(3, 10):
            row = db.record_pair_rebalance_failure(
                "300x1x0", "300x2x0", "no_route", base, now=7000)
            assert row["failure_count"] == i
            assert row["cooldown_until"] == 7000 + base * min(i, 6)

    def test_concurrent_increments_not_lost(self, tmp_path):
        db = _make_db(tmp_path)
        per_thread = 25
        n_threads = 4
        barrier = threading.Barrier(n_threads)
        errors = []

        def hammer():
            try:
                barrier.wait()
                for _ in range(per_thread):
                    db.record_pair_rebalance_failure(
                        "400x1x0", "400x2x0", "no_route", 60)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        conn = db._get_connection()
        row = conn.execute(
            "SELECT failure_count FROM pair_rebalance_failures "
            "WHERE source_channel_id = '400x1x0' AND dest_channel_id = '400x2x0'"
        ).fetchone()
        assert row["failure_count"] == per_thread * n_threads


class TestDeleteExpiredPoliciesTransactional:
    """Fix 5: returned peer list must match what was deleted."""

    def test_returns_exactly_deleted_peers(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO peer_policies (peer_id, strategy, updated_at, expires_at) "
            "VALUES ('peer_expired', 'static', ?, ?)", (NOW, NOW - 100))
        conn.execute(
            "INSERT INTO peer_policies (peer_id, strategy, updated_at, expires_at) "
            "VALUES ('peer_live', 'static', ?, ?)", (NOW, NOW + 10000))
        conn.execute(
            "INSERT INTO peer_policies (peer_id, strategy, updated_at) "
            "VALUES ('peer_no_expiry', 'static', ?)", (NOW,))

        deleted = db.delete_expired_policies(NOW)
        assert deleted == ["peer_expired"]
        remaining = {
            r["peer_id"]
            for r in conn.execute("SELECT peer_id FROM peer_policies").fetchall()
        }
        assert remaining == {"peer_live", "peer_no_expiry"}

    def test_no_expired_returns_empty(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.delete_expired_policies(NOW) == []


class TestSetConfigOverrideVersion:
    """Fix 6: returned version is the one computed inside the transaction."""

    def test_returns_version_of_written_row(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        v1 = db.set_config_override("alpha", "1")
        row = conn.execute(
            "SELECT version FROM config_overrides WHERE key = 'alpha'"
        ).fetchone()
        assert v1 == row["version"]

        # Simulate a concurrent writer bumping versions after our commit:
        # the next call must still return ITS OWN row's version.
        conn.execute(
            "INSERT INTO config_overrides (key, value, version, updated_at) "
            "VALUES ('external', 'x', ?, ?)", (v1 + 50, NOW))
        v2 = db.set_config_override("beta", "2")
        row = conn.execute(
            "SELECT version FROM config_overrides WHERE key = 'beta'"
        ).fetchone()
        assert v2 == row["version"]
        assert v2 == v1 + 51
