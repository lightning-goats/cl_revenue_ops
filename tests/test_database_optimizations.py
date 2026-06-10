"""
Tests for database performance optimizations:

1. Index & PRAGMA work:
   - idx_rebalance_costs_time covers timestamp-only SUM(cost_sats) predicates
     (verified with EXPLAIN QUERY PLAN)
   - idx_fee_changes_time covers the hourly cleanup DELETE
   - idx_forwards_channels (strict prefix of idx_forwards_unique) is dropped,
     including on existing databases, without breaking get_top_route_pairs
   - performance PRAGMAs (cache_size, mmap_size, temp_store) applied per-connection

2. New batch methods (parity with their single-row counterparts is the contract):
   - save_kalman_states_batch
   - save_temporal_profiles_batch
   - get_hourly_forward_histograms_all
   - update_fee_strategy_states_batch
   - get_all_fee_strategy_channel_ids
   - get_last_rebalance_times
   - get_all_channels_full_pnl
   - record_forward_and_reputation

Uses real SQLite (temp files) to verify actual SQL logic and query plans.
"""

import math
import os
import sys
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


NOW = int(time.time())


def _make_db(tmp_path, name="test_optimizations.db"):
    db_path = os.path.join(str(tmp_path), name)
    plugin = MagicMock()
    db = Database(db_path, plugin)
    db.initialize()
    return db


def _plan(conn, sql, params=()):
    """Return concatenated EXPLAIN QUERY PLAN detail text for a statement."""
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return " | ".join(str(r["detail"]) for r in rows)


def _index_names(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA index_list({table})").fetchall()}


# =============================================================================
# 1. Indexes & PRAGMAs
# =============================================================================

class TestIndexes:

    def test_rebalance_costs_time_index_used_for_timestamp_sum(self, tmp_path):
        """SELECT SUM(cost_sats) ... WHERE timestamp >= ? must use the new
        covering index instead of a full table scan."""
        db = _make_db(tmp_path)
        conn = db._get_connection()
        assert "idx_rebalance_costs_time" in _index_names(conn, "rebalance_costs")

        plan = _plan(
            conn,
            "SELECT SUM(cost_sats) FROM rebalance_costs WHERE timestamp >= ?",
            (NOW - 86400,),
        )
        assert "idx_rebalance_costs_time" in plan
        # (timestamp, cost_sats) makes the index covering for this query
        assert "COVERING INDEX" in plan.upper()
        assert "SCAN rebalance_costs" not in plan

    def test_fee_changes_time_index_used_for_cleanup_delete(self, tmp_path):
        """The hourly audit-log cleanup DELETE must be index-driven."""
        db = _make_db(tmp_path)
        conn = db._get_connection()
        assert "idx_fee_changes_time" in _index_names(conn, "fee_changes")

        plan = _plan(
            conn,
            "DELETE FROM fee_changes WHERE timestamp < ?",
            (NOW - 90 * 86400,),
        )
        assert "idx_fee_changes_time" in plan

    def test_forwards_channels_index_dropped(self, tmp_path):
        """idx_forwards_channels is a strict prefix of idx_forwards_unique and
        must no longer exist after initialize()."""
        db = _make_db(tmp_path)
        conn = db._get_connection()
        names = _index_names(conn, "forwards")
        assert "idx_forwards_channels" not in names
        assert "idx_forwards_unique" in names  # superset index still present

    def test_forwards_channels_index_dropped_on_existing_db(self, tmp_path):
        """Existing databases that already have idx_forwards_channels converge
        to the new schema on the next initialize()."""
        db = _make_db(tmp_path)
        conn = db._get_connection()
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_forwards_channels "
            "ON forwards(in_channel, out_channel)"
        )
        assert "idx_forwards_channels" in _index_names(conn, "forwards")

        db.initialize()  # re-run schema init path
        assert "idx_forwards_channels" not in _index_names(conn, "forwards")

    def test_get_top_route_pairs_unaffected_by_index_drop(self, tmp_path):
        """get_top_route_pairs never needed idx_forwards_channels; verify the
        plan does not reference it and results are still correct."""
        db = _make_db(tmp_path)
        conn = db._get_connection()

        # Seed a route pair above min_forwards
        for i in range(4):
            db.record_forward("100x1x0", "200x2x0", 1_000_000, 999_000, 1_000,
                              NOW - 3600 - i, NOW - 3600 - i)

        plan = _plan(conn, """
            SELECT in_channel, out_channel,
                   SUM(fee_msat) as total_fee_msat,
                   COUNT(*) as forward_count,
                   AVG(fee_msat) as avg_fee_msat
            FROM forwards
            WHERE timestamp >= ? AND fee_msat > 0
            GROUP BY in_channel, out_channel
            HAVING forward_count >= ?
            ORDER BY total_fee_msat DESC
            LIMIT ?
        """, (NOW - 30 * 86400, 3, 20))
        assert "idx_forwards_channels" not in plan

        pairs = db.get_top_route_pairs(days=30, min_forwards=3, limit=20)
        assert len(pairs) == 1
        assert pairs[0]["in_channel"] == "100x1x0"
        assert pairs[0]["out_channel"] == "200x2x0"
        assert pairs[0]["forward_count"] == 4
        assert pairs[0]["total_fee_msat"] == 4_000


class TestPragmas:

    def test_performance_pragmas_applied(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -16000
        assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 134217728
        # temp_store: 2 == MEMORY
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2


# =============================================================================
# 2. Batch methods — parity with single-row counterparts
# =============================================================================

class TestSaveKalmanStatesBatch:

    def _state(self, seed: float):
        return {
            "flow_ratio": 0.1 * seed,
            "flow_velocity": 0.01 * seed,
            "variance_ratio": 0.2,
            "variance_velocity": 0.3,
            "covariance": 0.05,
            "innovation_variance": 0.02,
            "last_innovation": -0.01 * seed,
            "last_update": NOW - int(seed),
            "observation_count": int(seed),
        }

    def test_batch_matches_single_row_writes(self, tmp_path):
        db = _make_db(tmp_path)

        # Single-row writes into channel A/B...
        db.save_kalman_state("100x1x0", self._state(1))
        db.save_kalman_state("200x2x0", self._state(2))
        single = {cid: db.get_kalman_state(cid) for cid in ("100x1x0", "200x2x0")}

        # ...must equal batch writes of the same payloads into C/D
        rows = [
            {"channel_id": "300x3x0", "state": self._state(1)},
            {"channel_id": "400x4x0", "state": self._state(2)},
        ]
        assert db.save_kalman_states_batch(rows) == 2

        for src, dst in (("100x1x0", "300x3x0"), ("200x2x0", "400x4x0")):
            got = db.get_kalman_state(dst)
            want = dict(single[src])
            want["channel_id"] = dst
            assert got == want

    def test_batch_skips_nan_rows(self, tmp_path):
        db = _make_db(tmp_path)
        rows = [
            {"channel_id": "100x1x0", "state": self._state(1)},
            {"channel_id": "666x6x0",
             "state": {**self._state(2), "flow_ratio": float("nan")}},
            {"channel_id": "777x7x0",
             "state": {**self._state(3), "covariance": float("inf")}},
        ]
        assert db.save_kalman_states_batch(rows) == 3  # contract: len(rows)
        assert db.get_kalman_state("100x1x0") is not None
        assert db.get_kalman_state("666x6x0") is None
        assert db.get_kalman_state("777x7x0") is None

    def test_empty_batch(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.save_kalman_states_batch([]) == 0


class TestSaveTemporalProfilesBatch:

    def test_batch_matches_single_row_writes(self, tmp_path):
        db = _make_db(tmp_path)
        # temporal profiles are stored on channel_states rows; seed them first
        for cid in ("100x1x0", "200x2x0"):
            db.update_channel_state(cid, "02" + "ab" * 32, "balanced",
                                    0.5, 1000, 1000, 1_000_000)

        db.save_temporal_profile("100x1x0", '{"v": 1}')
        single = db.load_temporal_profile("100x1x0")

        rows = [
            {"channel_id": "100x1x0", "profile_json": '{"v": 1}'},
            {"channel_id": "200x2x0", "profile_json": '{"v": 2}'},
        ]
        assert db.save_temporal_profiles_batch(rows) == 2
        assert db.load_temporal_profile("100x1x0") == single == '{"v": 1}'
        assert db.load_temporal_profile("200x2x0") == '{"v": 2}'

    def test_empty_batch(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.save_temporal_profiles_batch([]) == 0


class TestHourlyForwardHistogramsAll:

    def test_parity_with_single_channel_helper(self, tmp_path):
        db = _make_db(tmp_path)

        # Seed forwards across several channels, hours, and days
        channels = ("100x1x0", "200x2x0", "300x3x0")
        ts_base = NOW - 5 * 86400
        i = 0
        for day in range(5):
            for hour in (0, 7, 13, 22):
                in_ch = channels[i % 3]
                out_ch = channels[(i + 1) % 3]
                ts = ts_base + day * 86400 + hour * 3600 + (i % 50)
                db.record_forward(in_ch, out_ch,
                                  1_000_000 + i * 12_345,
                                  995_000 + i * 12_345,
                                  5_000 + i * 7,
                                  ts, ts + 2)
                i += 1

        batch = db.get_hourly_forward_histograms_all(window_days=7)
        assert set(batch.keys()) == set(channels)
        for cid in channels:
            single = db.get_hourly_forward_histogram(cid, window_days=7)
            assert batch[cid] == single, f"histogram mismatch for {cid}"
            assert len(batch[cid]) == 24

    def test_empty_db(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.get_hourly_forward_histograms_all(window_days=7) == {}


class TestUpdateFeeStrategyStatesBatch:

    def _kwargs(self, cid, seed):
        return {
            "channel_id": cid,
            "last_revenue_rate": 1.5 * seed,
            "last_fee_ppm": 100 * seed,
            "trend_direction": -1 if seed % 2 else 1,
            "step_ppm": 25 * seed,
            "consecutive_same_direction": seed,
            "last_broadcast_fee_ppm": 90 * seed,
            "last_state": "source",
            "is_sleeping": seed % 2,
            "sleep_until": NOW + seed,
            "stable_cycles": seed,
            "forward_count_since_update": 3 * seed,
            "last_volume_sats": 1000 * seed,
            "v2_state_json": f'{{"seed": {seed}}}',
            "last_update": NOW - seed,
        }

    def test_batch_matches_single_row_writes(self, tmp_path):
        db = _make_db(tmp_path)

        db.update_fee_strategy_state(**self._kwargs("100x1x0", 1))
        db.update_fee_strategy_state(**self._kwargs("200x2x0", 2))
        single = {cid: db.get_fee_strategy_state(cid)
                  for cid in ("100x1x0", "200x2x0")}

        rows = [self._kwargs("300x3x0", 1), self._kwargs("400x4x0", 2)]
        assert db.update_fee_strategy_states_batch(rows) == 2

        for src, dst in (("100x1x0", "300x3x0"), ("200x2x0", "400x4x0")):
            got = db.get_fee_strategy_state(dst)
            want = dict(single[src])
            want["channel_id"] = dst
            assert got == want

    def test_defaults_match_single_row_defaults(self, tmp_path):
        db = _make_db(tmp_path)
        minimal = {"channel_id": None, "last_revenue_rate": 2.0,
                   "last_fee_ppm": 150, "trend_direction": 1}

        db.update_fee_strategy_state(**{**minimal, "channel_id": "100x1x0"})
        single = db.get_fee_strategy_state("100x1x0")

        assert db.update_fee_strategy_states_batch(
            [{**minimal, "channel_id": "200x2x0"}]) == 1
        got = db.get_fee_strategy_state("200x2x0")

        # last_update defaults to now() in both paths; tolerate clock skew
        assert abs(got.pop("last_update") - single.pop("last_update")) <= 2
        single["channel_id"] = "200x2x0"
        assert got == single

    def test_empty_batch(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.update_fee_strategy_states_batch([]) == 0


class TestGetAllFeeStrategyChannelIds:

    def test_returns_ids_only(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.get_all_fee_strategy_channel_ids() == []

        for cid in ("100x1x0", "200x2x0", "300x3x0"):
            db.update_fee_strategy_state(
                channel_id=cid, last_revenue_rate=1.0,
                last_fee_ppm=100, trend_direction=1,
                v2_state_json='{"big": "blob"}')

        ids = db.get_all_fee_strategy_channel_ids()
        assert sorted(ids) == ["100x1x0", "200x2x0", "300x3x0"]
        assert all(isinstance(i, str) for i in ids)
        # parity with the heavyweight method
        full = {s["channel_id"] for s in db.get_all_fee_strategy_states()}
        assert set(ids) == full


class TestGetLastRebalanceTimes:

    def _seed_rebalance(self, db, to_channel, status, ts):
        conn = db._get_connection()
        conn.execute("""
            INSERT INTO rebalance_history
            (from_channel, to_channel, amount_sats, max_fee_sats,
             expected_profit_sats, status, timestamp)
            VALUES (?, ?, 1000, 10, 5, ?, ?)
        """, ("999x9x0", to_channel, status, ts))

    def test_parity_with_single_channel_method(self, tmp_path):
        db = _make_db(tmp_path)
        self._seed_rebalance(db, "100x1x0", "success", NOW - 5000)
        self._seed_rebalance(db, "100x1x0", "success", NOW - 1000)  # max
        self._seed_rebalance(db, "100x1x0", "failed", NOW - 100)    # ignored
        self._seed_rebalance(db, "200x2x0", "success", NOW - 9000)
        self._seed_rebalance(db, "300x3x0", "failed", NOW - 50)     # no success

        times = db.get_last_rebalance_times()
        assert times == {
            "100x1x0": NOW - 1000,
            "200x2x0": NOW - 9000,
        }
        for cid, ts in times.items():
            assert db.get_last_rebalance_time(cid) == ts
        assert db.get_last_rebalance_time("300x3x0") is None
        assert "300x3x0" not in times

    def test_empty_db(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.get_last_rebalance_times() == {}


class TestGetAllChannelsFullPnl:

    WINDOW = 30

    def _seed(self, db):
        conn = db._get_connection()
        today_start = (NOW // 86400) * 86400

        # Live forwards (within window), mixed roles
        db.record_forward("100x1x0", "200x2x0", 2_000_000, 1_998_500, 1_500,
                          NOW - 3600, NOW - 3590)
        db.record_forward("200x2x0", "100x1x0", 5_000_000, 4_997_000, 3_000,
                          NOW - 7200, NOW - 7100)
        db.record_forward("300x3x0", "100x1x0", 1_000_000, 999_300, 700,
                          NOW - 1800, NOW - 1700)

        # Legacy colon-form SCID row written directly (alias normalization)
        conn.execute("""
            INSERT INTO forwards
            (in_channel, out_channel, in_msat, out_msat, fee_msat,
             resolution_time, timestamp, resolved_time)
            VALUES ('100:1:0', '400:4:0', 750000, 749100, 900, 0, ?, ?)
        """, (NOW - 5400, NOW - 5300))

        # Completed-day rollups inside the window (date < today_start)
        d1 = today_start - 3 * 86400
        d2 = today_start - 5 * 86400
        conn.execute("""
            INSERT INTO daily_forwarding_stats
            (channel_id, date, total_in_msat, total_out_msat, total_fee_msat, forward_count)
            VALUES ('200x2x0', ?, 9000000, 8990000, 10000, 7)
        """, (d1,))
        conn.execute("""
            INSERT INTO daily_forwarding_stats
            (channel_id, date, total_in_msat, total_out_msat, total_fee_msat, forward_count)
            VALUES ('100:1:0', ?, 4000000, 3996000, 4000, 2)
        """, (d2,))
        conn.execute("""
            INSERT INTO daily_forwarding_stats_inbound
            (channel_id, date, total_in_msat, total_fee_msat, forward_count)
            VALUES ('100x1x0', ?, 6000000, 5500, 4)
        """, (d1,))
        # Out-of-window rollup must be excluded by both paths
        old = today_start - (self.WINDOW + 10) * 86400
        conn.execute("""
            INSERT INTO daily_forwarding_stats
            (channel_id, date, total_in_msat, total_out_msat, total_fee_msat, forward_count)
            VALUES ('200x2x0', ?, 77000000, 76000000, 99999, 11)
        """, (old,))

        # Rebalance costs (canonical + colon-form + msat-precision row)
        db.record_rebalance_cost("100x1x0", "02" + "ab" * 32, 120, 100_000)
        conn.execute("""
            INSERT INTO rebalance_costs
            (channel_id, peer_id, cost_sats, cost_msat, amount_sats, timestamp)
            VALUES ('200:2:0', ?, 55, 54321, 50000, ?)
        """, ("02" + "cd" * 32, NOW - 4000))

    def test_parity_with_per_channel_method(self, tmp_path):
        db = _make_db(tmp_path)
        self._seed(db)

        batch = db.get_all_channels_full_pnl(self.WINDOW)

        # Every channel with activity appears once, under its canonical SCID
        assert set(batch.keys()) == {"100x1x0", "200x2x0", "300x3x0", "400x4x0"}

        for cid, batch_pnl in batch.items():
            single_pnl = db.get_channel_full_pnl(cid, self.WINDOW)
            assert batch_pnl == single_pnl, f"full P&L mismatch for {cid}"

    def test_colon_alias_rows_are_merged(self, tmp_path):
        db = _make_db(tmp_path)
        self._seed(db)
        batch = db.get_all_channels_full_pnl(self.WINDOW)

        # '100:1:0' rows (forwards entry-side, daily stats, none split off)
        assert "100:1:0" not in batch
        assert "200:2:0" not in batch
        # 200x2x0 cost merges the colon-form msat-precision row
        assert batch["200x2x0"]["rebalance_cost_msat"] == 54321
        # 100x1x0 direct revenue includes the colon-keyed daily rollup
        # live out fee (3000 from 200->100, 700 from 300->100) + rollup 4000
        assert batch["100x1x0"]["direct_revenue_msat"] == 3000 + 700 + 4000

    def test_empty_db(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.get_all_channels_full_pnl(self.WINDOW) == {}


class TestRecordForwardAndReputation:

    PEER = "02" + "ee" * 32

    def test_combined_write_matches_individual_methods(self, tmp_path):
        db = _make_db(tmp_path)
        fwd = {
            "in_channel": "100:1:0",  # colon form must be normalized
            "out_channel": "200x2x0",
            "in_msat": 1_000_000,
            "out_msat": 999_000,
            "fee_msat": 1_000,
            "received_time": NOW - 60,
            "resolved_time": NOW - 58,
            "resolution_time": 2.0,
        }
        db.record_forward_and_reputation(fwd, self.PEER, True)

        conn = db._get_connection()
        row = conn.execute("SELECT * FROM forwards").fetchone()
        assert row["in_channel"] == "100x1x0"
        assert row["out_channel"] == "200x2x0"
        assert row["in_msat"] == 1_000_000
        assert row["out_msat"] == 999_000
        assert row["fee_msat"] == 1_000
        assert row["timestamp"] == NOW - 60
        assert row["resolved_time"] == NOW - 58

        rep = db.get_peer_reputation(self.PEER)
        assert rep["successes"] == 1
        assert rep["failures"] == 0

    def test_failure_increments_failure_count(self, tmp_path):
        db = _make_db(tmp_path)
        fwd = {
            "in_channel": "100x1x0", "out_channel": "200x2x0",
            "in_msat": 500_000, "out_msat": 499_500, "fee_msat": 500,
            "received_time": NOW - 30, "resolved_time": NOW - 29,
        }
        db.record_forward_and_reputation(fwd, self.PEER, False)
        rep = db.get_peer_reputation(self.PEER)
        assert rep["successes"] == 0
        assert rep["failures"] == 1

    def test_duplicate_forward_is_idempotent_but_reputation_updates(self, tmp_path):
        db = _make_db(tmp_path)
        fwd = {
            "in_channel": "100x1x0", "out_channel": "200x2x0",
            "in_msat": 500_000, "out_msat": 499_500, "fee_msat": 500,
            "received_time": NOW - 30, "resolved_time": NOW - 29,
        }
        db.record_forward_and_reputation(fwd, self.PEER, True)
        db.record_forward_and_reputation(fwd, self.PEER, True)

        conn = db._get_connection()
        count = conn.execute("SELECT COUNT(*) FROM forwards").fetchone()[0]
        assert count == 1  # INSERT OR IGNORE under idx_forwards_unique
        rep = db.get_peer_reputation(self.PEER)
        assert rep["successes"] == 2

    def test_derives_timestamps_like_record_forward(self, tmp_path):
        """resolved_time derived from received_time + resolution_time when absent."""
        db = _make_db(tmp_path)
        fwd = {
            "in_channel": "100x1x0", "out_channel": "200x2x0",
            "in_msat": 500_000, "out_msat": 499_500, "fee_msat": 500,
            "received_time": NOW - 100, "resolution_time": 3.0,
        }
        db.record_forward_and_reputation(fwd, self.PEER, True)
        conn = db._get_connection()
        row = conn.execute("SELECT timestamp, resolved_time FROM forwards").fetchone()
        assert row["timestamp"] == NOW - 100
        assert row["resolved_time"] == NOW - 97

    def test_transaction_rolls_back_on_error(self, tmp_path):
        """If the reputation upsert fails, the forward insert must roll back."""
        db = _make_db(tmp_path)
        conn = db._get_connection()
        # Force the second statement in the transaction to fail
        conn.execute("DROP TABLE peer_reputation")
        fwd = {
            "in_channel": "100x1x0", "out_channel": "200x2x0",
            "in_msat": 500_000, "out_msat": 499_500, "fee_msat": 500,
            "received_time": NOW - 30, "resolved_time": NOW - 29,
        }
        with pytest.raises(Exception):
            db.record_forward_and_reputation(fwd, self.PEER, True)
        count = conn.execute("SELECT COUNT(*) FROM forwards").fetchone()[0]
        assert count == 0


class TestNodeRealizedFeePpm:
    def _db(self, tmp_path):
        from modules.database import Database
        plugin = MagicMock()
        db = Database(str(tmp_path / "ppm.db"), plugin)
        db.initialize()
        return db

    def test_realized_ppm_from_forwards(self, tmp_path):
        db = self._db(tmp_path)
        conn = db._get_connection()
        now = int(time.time())
        # 2 forwards: vary in_msat to avoid unique-index collision
        # Each row: out_msat=500000, fee_msat=500 => 1000 ppm
        for offset in range(2):
            conn.execute(
                "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat,"
                " fee_msat, timestamp) VALUES ('1x1x1','2x2x2',?,500000,500,?)",
                (500500 + offset, now - 3600),
            )
        assert db.get_node_realized_fee_ppm(window_days=7) == 1000

    def test_realized_ppm_zero_when_no_forwards(self, tmp_path):
        db = self._db(tmp_path)
        assert db.get_node_realized_fee_ppm(window_days=7) == 0

    def test_realized_ppm_excludes_old_forwards(self, tmp_path):
        db = self._db(tmp_path)
        conn = db._get_connection()
        old = int(time.time()) - 10 * 86400
        conn.execute(
            "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat,"
            " fee_msat, timestamp) VALUES ('1x1x1','2x2x2',500500,500000,500,?)",
            (old,),
        )
        assert db.get_node_realized_fee_ppm(window_days=7) == 0


class TestCategorySpendSats:
    def _db(self, tmp_path):
        from modules.database import Database
        plugin = MagicMock()
        db = Database(str(tmp_path / "spend.db"), plugin)
        db.initialize()
        return db

    def test_sums_matching_category_and_subcategory(self, tmp_path):
        db = self._db(tmp_path)
        now = int(time.time())
        db.record_spend_event(event_id="e1", amount_sats=100, category="boltz",
                              subcategory="structural", timestamp=now - 100)
        db.record_spend_event(event_id="e2", amount_sats=40, category="boltz",
                              subcategory="structural", timestamp=now - 200)
        db.record_spend_event(event_id="e3", amount_sats=999, category="boltz",
                              subcategory="swap_fee", timestamp=now - 200)
        db.record_spend_event(event_id="e4", amount_sats=999, category="boltz",
                              subcategory="structural", timestamp=now - 2 * 86400)
        got = db.get_category_spend_sats("boltz", subcategory="structural",
                                         since_timestamp=now - 86400)
        assert got == 140

    def test_subcategory_none_sums_whole_category(self, tmp_path):
        db = self._db(tmp_path)
        now = int(time.time())
        db.record_spend_event(event_id="e1", amount_sats=10, category="boltz",
                              subcategory="swap_fee", timestamp=now - 100)
        db.record_spend_event(event_id="e2", amount_sats=20, category="boltz",
                              subcategory="structural", timestamp=now - 100)
        assert db.get_category_spend_sats("boltz",
                                          since_timestamp=now - 86400) == 30
