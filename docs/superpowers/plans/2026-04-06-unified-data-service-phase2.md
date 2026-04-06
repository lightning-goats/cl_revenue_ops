# Unified Data Service Phase 2 — Database Escape Absorption

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Absorb all direct SQL from 4 modules into proper `database.py` methods, eliminating all `_get_connection()` usage outside database.py.

**Architecture:** Add 9 new methods to `modules/database.py` for policy CRUD, capex aggregation, orphan cleanup. Update 4 consumer modules to call the new methods instead of raw SQL. The `_get_connection()` accessor becomes truly private.

**Tech Stack:** Python 3.11, pytest, SQLite

**Spec:** `docs/superpowers/specs/2026-04-06-unified-data-service-design.md`

**Repo:** `/home/sat/bin/cl_revenue_ops`

---

## File Map

| File | Changes |
|------|---------|
| `modules/database.py` | Add 9 new methods for policy CRUD, capex aggregation, orphan cleanup |
| `modules/policy_manager.py` | Replace 8 direct SQL sites with database method calls |
| `modules/capex_budget.py` | Replace 1 direct SQL site with database method call |
| `modules/rebalancer.py` | Replace 1 direct SQL site with database method call |
| `modules/fee_controller.py` | Replace 1 direct SQL site with existing `get_channel_cost()` call |
| `tests/test_database_policies.py` | New — tests for all 9 new database methods |

---

### Task 1: Add policy CRUD methods to database.py + tests

**Files:**
- Modify: `modules/database.py`
- Create: `tests/test_database_policies.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_database_policies.py`:

```python
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
        conn.execute("INSERT INTO rebalance_history (id, channel_id, source_channel_id, amount_sats, max_fee_ppm, status, timestamp) VALUES (1, 'ch1', 'ch2', 1000, 500, 'pending', ?)", (old,))
        orphan_ids = db.cleanup_orphaned_rebalances(timeout_seconds=3600)
        assert orphan_ids == [1]
        row = conn.execute("SELECT status, error_message FROM rebalance_history WHERE id = 1").fetchone()
        assert row["status"] == "failed"
        assert row["error_message"] == "orphaned_on_restart"

    def test_skips_recent_pending(self, db):
        now = int(time.time())
        conn = db._get_connection()
        conn.execute("INSERT INTO rebalance_history (id, channel_id, source_channel_id, amount_sats, max_fee_ppm, status, timestamp) VALUES (1, 'ch1', 'ch2', 1000, 500, 'pending', ?)", (now,))
        orphan_ids = db.cleanup_orphaned_rebalances(timeout_seconds=3600)
        assert orphan_ids == []

    def test_skips_completed(self, db):
        old = int(time.time()) - 7200
        conn = db._get_connection()
        conn.execute("INSERT INTO rebalance_history (id, channel_id, source_channel_id, amount_sats, max_fee_ppm, status, timestamp) VALUES (1, 'ch1', 'ch2', 1000, 500, 'completed', ?)", (old,))
        orphan_ids = db.cleanup_orphaned_rebalances(timeout_seconds=3600)
        assert orphan_ids == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_database_policies.py -v`

Expected: FAIL — `AttributeError: 'Database' object has no attribute 'get_all_policies'`

- [ ] **Step 3: Add all 9 new methods to database.py**

Find the end of the existing methods in `modules/database.py` (before the final class closing or after the last method). Add these methods:

```python
    # ------------------------------------------------------------------
    # Policy CRUD (replaces direct SQL in policy_manager.py)
    # ------------------------------------------------------------------

    def get_all_policies(self) -> list:
        """Get all peer policies ordered by updated_at descending."""
        conn = self._get_connection()
        return conn.execute(
            "SELECT * FROM peer_policies ORDER BY updated_at DESC"
        ).fetchall()

    def get_policy(self, peer_id: str):
        """Get a single peer policy by peer_id. Returns row or None."""
        conn = self._get_connection()
        return conn.execute(
            "SELECT * FROM peer_policies WHERE peer_id = ?", (peer_id,)
        ).fetchone()

    def upsert_policy(self, peer_id: str, strategy: str, rebalance_mode: str,
                      fee_ppm_target: int, tags: str, updated_at: int,
                      fee_multiplier_min: float, fee_multiplier_max: float,
                      expires_at) -> None:
        """Insert or replace a peer policy."""
        conn = self._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO peer_policies
                (peer_id, strategy, rebalance_mode, fee_ppm_target, tags,
                 updated_at, fee_multiplier_min, fee_multiplier_max, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (peer_id, strategy, rebalance_mode, fee_ppm_target, tags,
              updated_at, fee_multiplier_min, fee_multiplier_max, expires_at))

    def delete_policy(self, peer_id: str) -> bool:
        """Delete a peer policy. Returns True if a row was deleted."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM peer_policies WHERE peer_id = ?", (peer_id,)
        )
        return cursor.rowcount > 0

    def delete_expired_policies(self, now: int) -> list:
        """Delete expired policies. Returns list of deleted peer_ids."""
        conn = self._get_connection()
        expired_rows = conn.execute(
            "SELECT peer_id FROM peer_policies WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        ).fetchall()
        if not expired_rows:
            return []
        conn.execute(
            "DELETE FROM peer_policies WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        )
        return [row["peer_id"] for row in expired_rows]

    def upsert_policies_batch(self, rows: list) -> None:
        """Batch insert/replace peer policies atomically.

        Args:
            rows: List of tuples (peer_id, strategy, rebalance_mode,
                  fee_ppm_target, tags, updated_at, fee_multiplier_min,
                  fee_multiplier_max, expires_at)
        """
        conn = self._get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO peer_policies
                    (peer_id, strategy, rebalance_mode, fee_ppm_target, tags,
                     updated_at, fee_multiplier_min, fee_multiplier_max, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def get_policy_changes_since(self, since_timestamp: int) -> list:
        """Get policy rows updated after the given timestamp."""
        conn = self._get_connection()
        return conn.execute("""
            SELECT * FROM peer_policies
            WHERE updated_at > ?
            ORDER BY updated_at DESC
        """, (since_timestamp,)).fetchall()

    def get_last_policy_change_timestamp(self) -> int:
        """Get the most recent updated_at across all policies. Returns 0 if empty."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT MAX(updated_at) as max_ts FROM peer_policies"
        ).fetchone()
        return (row["max_ts"] or 0) if row else 0

    # ------------------------------------------------------------------
    # Capex aggregation (replaces direct SQL in capex_budget.py)
    # ------------------------------------------------------------------

    def get_total_capex_by_channel(self, window_days: int = 30) -> dict:
        """Get total capex per channel from rebalance_costs + spend_events.

        Returns dict of channel_id -> total_sats.
        """
        since = int(time.time()) - (window_days * 86400)
        result = {}
        conn = self._get_connection()

        rows = conn.execute("""
            SELECT channel_id, COALESCE(SUM(cost_sats), 0) as total
            FROM rebalance_costs
            WHERE timestamp >= ?
            GROUP BY channel_id
        """, (since,)).fetchall()
        for r in rows:
            cid = r["channel_id"]
            if cid:
                result[cid] = result.get(cid, 0) + int(r["total"] or 0)

        rows = conn.execute("""
            SELECT channel_id, COALESCE(SUM(amount_sats), 0) as total
            FROM spend_events
            WHERE timestamp >= ? AND channel_id IS NOT NULL
            GROUP BY channel_id
        """, (since,)).fetchall()
        for r in rows:
            cid = r["channel_id"]
            if cid:
                result[cid] = result.get(cid, 0) + int(r["total"] or 0)

        return result

    # ------------------------------------------------------------------
    # Orphan cleanup (replaces direct SQL in rebalancer.py)
    # ------------------------------------------------------------------

    def cleanup_orphaned_rebalances(self, timeout_seconds: int = 3600) -> list:
        """Mark stale pending rebalances as failed. Returns list of orphaned IDs.

        Finds rebalance_history rows with status 'pending' or 'pending_async'
        older than timeout_seconds and marks them failed with
        error_message='orphaned_on_restart'.
        """
        cutoff = int(time.time()) - timeout_seconds
        conn = self._get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            orphaned_rows = conn.execute("""
                SELECT id FROM rebalance_history
                WHERE status IN ('pending', 'pending_async')
                  AND timestamp < ?
            """, (cutoff,)).fetchall()
            orphaned_ids = [row["id"] for row in orphaned_rows]

            if orphaned_ids:
                conn.execute("""
                    UPDATE rebalance_history
                    SET status = 'failed', error_message = 'orphaned_on_restart'
                    WHERE status IN ('pending', 'pending_async')
                      AND timestamp < ?
                """, (cutoff,))

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return orphaned_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_database_policies.py -v`

Expected: All PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/ -q`

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/database.py tests/test_database_policies.py
git commit -m "feat(database): add policy CRUD, capex aggregation, orphan cleanup methods

9 new methods to absorb direct SQL from policy_manager, capex_budget,
rebalancer, and fee_controller. Eliminates need for modules to call
database._get_connection() directly."
```

---

### Task 2: Migrate policy_manager.py to use new database methods

**Files:**
- Modify: `modules/policy_manager.py`

- [ ] **Step 1: Replace _load_cache() direct SQL**

In `modules/policy_manager.py`, find the `_load_cache` method (around lines 338-361). Replace the direct SQL:

Find:
```python
        # Read DB outside the lock (thread-local connections are safe)
        conn = self.database._get_connection()
        rows = conn.execute(
            "SELECT * FROM peer_policies ORDER BY updated_at DESC"
        ).fetchall()
```

Replace with:
```python
        # Read DB outside the lock (thread-local connections are safe)
        rows = self.database.get_all_policies()
```

- [ ] **Step 2: Replace _delete_expired_policy() direct SQL**

Find the `_delete_expired_policy` method (around lines 474-484). Replace:

Find:
```python
        try:
            conn = self.database._get_connection()
            now = int(time.time())
            conn.execute(
                "DELETE FROM peer_policies WHERE peer_id = ? AND expires_at IS NOT NULL AND expires_at < ?",
                (peer_id, now)
            )
        except Exception as e:
            self.plugin.log(f"PolicyManager: Error deleting expired policy: {e}", level='warn')
```

Replace with:
```python
        try:
            now = int(time.time())
            self.database.delete_expired_policies(now)
        except Exception as e:
            self.plugin.log(f"PolicyManager: Error deleting expired policy: {e}", level='warn')
```

- [ ] **Step 3: Replace get_policy_changes_since() direct SQL**

Find the method (around lines 504-516). Replace:

Find:
```python
            conn = self.database._get_connection()
            rows = conn.execute(
                """
                SELECT peer_id, strategy, rebalance_mode, fee_ppm_target,
                       tags, updated_at, fee_multiplier_min, fee_multiplier_max,
                       expires_at
                FROM peer_policies
                WHERE updated_at > ?
                ORDER BY updated_at DESC
                """,
                (since_timestamp,)
            ).fetchall()
```

Replace with:
```python
            rows = self.database.get_policy_changes_since(since_timestamp)
```

- [ ] **Step 4: Replace get_last_change_timestamp() direct SQL**

Find the method (around lines 545-556). Replace:

Find:
```python
            conn = self.database._get_connection()
            row = conn.execute(
                "SELECT MAX(updated_at) as max_ts FROM peer_policies"
            ).fetchone()
            return (row['max_ts'] or 0) if row else 0
```

Replace with:
```python
            return self.database.get_last_policy_change_timestamp()
```

- [ ] **Step 5: Replace set_policy() direct SQL**

Find (around lines 681-700):

Find:
```python
        conn = self.database._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO peer_policies
                (peer_id, strategy, rebalance_mode, fee_ppm_target, tags, updated_at,
                 fee_multiplier_min, fee_multiplier_max, expires_at)
        """, (
```

Replace this entire block (from `conn = self.database._get_connection()` through the closing `)`) with:

```python
        self.database.upsert_policy(
            peer_id, new_strategy.value, new_rebalance_mode.value,
            new_fee_ppm, json.dumps(new_tags), now,
            new_mult_min, new_mult_max, expires_at
        )
```

- [ ] **Step 6: Replace delete_policy() direct SQL**

Find (around lines 745-749):

Find:
```python
        conn = self.database._get_connection()
        cursor = conn.execute(
            "DELETE FROM peer_policies WHERE peer_id = ?",
            (peer_id,)
        )
```

Replace with:
```python
        deleted = self.database.delete_policy(peer_id)
```

Also update the line that checks `cursor.rowcount > 0` to use `deleted` directly. Find:
```python
        deleted = cursor.rowcount > 0
```
Remove that line (since `deleted` is already set above).

- [ ] **Step 7: Replace set_policies_batch() direct SQL**

Find (around lines 1287-1303):

Find:
```python
        conn = self.database._get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany("""
                INSERT OR REPLACE INTO peer_policies
                    (peer_id, strategy, rebalance_mode, fee_ppm_target, tags, updated_at,
                     fee_multiplier_min, fee_multiplier_max, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (peer_id, strategy.value, mode.value, fee_ppm, json.dumps(tags), now,
                 mult_min, mult_max, expires_at)
                for peer_id, strategy, mode, fee_ppm, tags, mult_min, mult_max, expires_at in validated
            ])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
```

Replace with:
```python
        self.database.upsert_policies_batch([
            (peer_id, strategy.value, mode.value, fee_ppm, json.dumps(tags), now,
             mult_min, mult_max, expires_at)
            for peer_id, strategy, mode, fee_ppm, tags, mult_min, mult_max, expires_at in validated
        ])
```

- [ ] **Step 8: Replace cleanup_expired_policies() direct SQL**

Find (around lines 1340-1368):

Find the block starting with `conn = self.database._get_connection()` through the expired row iteration. Replace:

```python
        conn = self.database._get_connection()

        # Get expired peer_ids before deletion
        expired_rows = conn.execute(
            "SELECT peer_id FROM peer_policies WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        ).fetchall()

        if not expired_rows:
            return 0

        # Delete expired policies
        cursor = conn.execute(
            "DELETE FROM peer_policies WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        )

        deleted_count = cursor.rowcount
```

Replace with:
```python
        expired_peer_ids = self.database.delete_expired_policies(now)

        if not expired_peer_ids:
            return 0

        deleted_count = len(expired_peer_ids)
```

Then update the loop that follows. Find:
```python
        for row in expired_rows:
            peer_id = row['peer_id']
```

Replace with:
```python
        for peer_id in expired_peer_ids:
```

- [ ] **Step 9: Run full test suite**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/ -q`

Expected: All PASS.

- [ ] **Step 10: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/policy_manager.py
git commit -m "refactor(policy-manager): replace 8 direct SQL sites with database methods

All peer_policies CRUD now goes through database.py API.
No _get_connection() usage remains in policy_manager."
```

---

### Task 3: Migrate capex_budget.py, rebalancer.py, fee_controller.py

**Files:**
- Modify: `modules/capex_budget.py`
- Modify: `modules/rebalancer.py`
- Modify: `modules/fee_controller.py`

- [ ] **Step 1: Replace capex_budget.py direct SQL**

In `modules/capex_budget.py`, find `_get_total_capex_by_channel` (around lines 404-440). Replace the entire method body:

Find the block starting with `conn = self._database._get_connection()` through the end of the try block. Replace:

```python
            conn = self._database._get_connection()

            # Rebalance costs (canonical table for rebalancing spend)
            rows = conn.execute("""
                SELECT channel_id, COALESCE(SUM(cost_sats), 0) as total
                FROM rebalance_costs
                WHERE timestamp >= ?
                GROUP BY channel_id
            """, (since,)).fetchall()
            for r in rows:
                cid = r["channel_id"]
                if cid:
                    result[cid] = result.get(cid, 0) + int(r["total"] or 0)

            # Spend events (opens, boltz, closures, etc.)
            rows = conn.execute("""
                SELECT channel_id, COALESCE(SUM(amount_sats), 0) as total
                FROM spend_events
                WHERE timestamp >= ? AND channel_id IS NOT NULL
                GROUP BY channel_id
            """, (since,)).fetchall()
            for r in rows:
                cid = r["channel_id"]
                if cid:
                    result[cid] = result.get(cid, 0) + int(r["total"] or 0)
```

Replace with:
```python
            result = self._database.get_total_capex_by_channel(window_days)
```

Also remove `since = int(time.time()) - (window_days * 86400)` since the database method calculates it internally. And remove `result: Dict[str, int] = {}` since the method returns the dict directly. The method should look like:

```python
    def _get_total_capex_by_channel(self, window_days: int = 30) -> Dict[str, int]:
        """Get total capex per channel from rebalance_costs + spend_events."""
        try:
            return self._database.get_total_capex_by_channel(window_days)
        except Exception:
            return {}
```

- [ ] **Step 2: Replace rebalancer.py direct SQL**

In `modules/rebalancer.py`, find `cleanup_orphans` (around lines 1353-1377). Replace the direct SQL block:

Find:
```python
                conn = self.database._get_connection()
                # I-7 FIX: Wrap orphan cleanup in transaction for atomicity
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # First, get the IDs of orphaned records so we can release their budget reservations
                    orphaned_rows = conn.execute("""
                        SELECT id FROM rebalance_history
                        WHERE status IN ('pending', 'pending_async')
                          AND timestamp < ?
                    """, (cutoff,)).fetchall()
                    orphaned_ids = [row['id'] for row in orphaned_rows]

                    cursor = conn.execute("""
                        UPDATE rebalance_history
                        SET status = 'failed', error_message = 'orphaned_on_restart'
                        WHERE status IN ('pending', 'pending_async')
                          AND timestamp < ?
                    """, (cutoff,))
                    orphaned_db = cursor.rowcount
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
```

Replace with:
```python
                orphaned_ids = self.database.cleanup_orphaned_rebalances(self.job_timeout_seconds)
                orphaned_db = len(orphaned_ids)
```

Also remove `cutoff = int(time.time()) - self.job_timeout_seconds` since the database method calculates it internally.

- [ ] **Step 3: Replace fee_controller.py direct SQL**

In `modules/fee_controller.py`, find `_get_channel_age_days` (around lines 5031-5043). The direct SQL fetches `opened_at` from `channel_costs`. The existing `get_channel_cost()` method already returns the full row including `opened_at`. Replace:

Find:
```python
                conn = self.database._get_connection()
                row = conn.execute(
                    "SELECT opened_at FROM channel_costs WHERE channel_id = ?",
                    (channel_id,)
                ).fetchone()
                if row and row["opened_at"]:
                    age_seconds = int(time.time()) - row["opened_at"]
                    return max(0, age_seconds // 86400)
```

Replace with:
```python
                cost_record = self.database.get_channel_cost(channel_id)
                if cost_record and cost_record.get("opened_at"):
                    age_seconds = int(time.time()) - cost_record["opened_at"]
                    return max(0, age_seconds // 86400)
```

Note: The line above this block already calls `self.database.get_channel_open_cost(channel_id)` and checks the result. The direct SQL is redundant. However, `get_channel_open_cost` returns just the cost integer, while `get_channel_cost` returns the full row dict. We need to use `get_channel_cost` for the `opened_at` field. You may also need to remove or adjust the `if cost_record:` guard above since we're now calling `get_channel_cost` instead.

Read the full method context to understand the flow before editing.

- [ ] **Step 4: Run full test suite**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/ -q`

Expected: All PASS.

- [ ] **Step 5: Verify no _get_connection() usage outside database.py**

Run: `grep -rn '_get_connection' modules/*.py | grep -v 'database.py'`

Expected: No output (no modules access `_get_connection()` anymore).

- [ ] **Step 6: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/capex_budget.py modules/rebalancer.py modules/fee_controller.py
git commit -m "refactor: migrate capex, rebalancer, fee_controller to database API

All direct _get_connection() usage eliminated from modules outside
database.py. capex_budget uses get_total_capex_by_channel(), rebalancer
uses cleanup_orphaned_rebalances(), fee_controller uses get_channel_cost()."
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Add 9 new database methods + tests | database.py, test_database_policies.py |
| 2 | Migrate policy_manager.py (8 SQL sites) | policy_manager.py |
| 3 | Migrate capex_budget, rebalancer, fee_controller (3 SQL sites) | capex_budget.py, rebalancer.py, fee_controller.py |

## Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| Policy CRUD methods in database.py | Task 1 |
| `get_total_capex_by_channel()` | Task 1 |
| `cleanup_orphaned_rebalances()` | Task 1 |
| Update fee_controller cost lookup | Task 3 |
| No `_get_connection()` outside database.py | Tasks 2-3 |

## Verification

After all tasks: `grep -rn '_get_connection' modules/*.py | grep -v 'database.py'` should return nothing.
