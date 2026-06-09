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


class TestPeerAndSourceRebalanceSuccessRate:
    """Real SQLite tests for peer/source rebalance success aggregation."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_peer_source_sr.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_peer_and_source_success_rates_use_existing_rebalance_history(self, tmp_path):
        """Peer and source success rates should aggregate existing rebalance_history rows."""
        db = self._make_db(tmp_path)
        conn = db._get_connection()
        now = int(time.time())
        peer_id = "02" + "b" * 64

        # Two destination channels for the same peer
        conn.execute(
            "INSERT INTO channel_states "
            "(channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("111x1x0", peer_id, "balanced", 0.5, 0, 0, 1_000_000, now),
        )
        conn.execute(
            "INSERT INTO channel_states "
            "(channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("111x2x0", peer_id, "balanced", 0.5, 0, 0, 1_000_000, now),
        )

        rows = [
            ("src-good", "111x1x0", 50_000, 100, 10, 50, "success", "normal", now - 3600),
            ("src-good", "111x2x0", 50_000, 100, 10, 50, "success", "normal", now - 7200),
            ("src-good", "111x2x0", 50_000, 100, None, 50, "failed", "normal", now - 10_800),
            ("src-bad", "111x1x0", 50_000, 100, None, 50, "failed", "normal", now - 14_400),
        ]
        for row in rows:
            conn.execute(
                "INSERT INTO rebalance_history "
                "(from_channel, to_channel, amount_sats, max_fee_sats, actual_fee_sats, "
                "expected_profit_sats, status, rebalance_type, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        conn.commit()

        peer_result = db.get_peer_rebalance_success_rate(peer_id, 30)
        assert peer_result is not None
        assert peer_result["total"] == 4
        assert peer_result["successes"] == 2
        assert peer_result["failures"] == 2
        assert abs(peer_result["success_rate"] - 0.5) < 0.01

        source_result = db.get_source_rebalance_success_rate("src-good", 30)
        assert source_result is not None
        assert source_result["total"] == 3
        assert source_result["successes"] == 2
        assert source_result["failures"] == 1
        assert abs(source_result["success_rate"] - (2 / 3)) < 0.01


class TestLastRebalanceTime:
    """Real SQLite tests for reason-aware rebalance cooldown lookups."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_last_rebalance_time.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_unfiltered_last_rebalance_time_preserves_existing_behavior(self, tmp_path):
        db = self._make_db(tmp_path)
        conn = db._get_connection()
        channel = "111x222x0"
        now = int(time.time())

        conn.execute(
            "INSERT INTO rebalance_history "
            "(from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, "
            "status, rebalance_type, reason_code, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("src-a", channel, 50_000, 100, 10, "success", "normal", "ev_positive", now - 120),
        )
        conn.execute(
            "INSERT INTO rebalance_history "
            "(from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, "
            "status, rebalance_type, reason_code, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("src-b", channel, 50_000, 100, 10, "success", "normal", "hive_equalization", now - 30),
        )
        conn.commit()

        assert db.get_last_rebalance_time(channel) == now - 30

    def test_reason_filtered_last_rebalance_time_returns_latest_matching_success(self, tmp_path):
        db = self._make_db(tmp_path)
        conn = db._get_connection()
        channel = "111x222x0"
        now = int(time.time())

        rows = [
            ("src-a", channel, 50_000, 100, 10, "success", "normal", "ev_positive", now - 240),
            ("src-b", channel, 50_000, 100, 10, "success", "normal", "hive_equalization", now - 180),
            ("src-c", channel, 50_000, 100, 10, "failed", "normal", "hive_equalization", now - 60),
            ("src-d", channel, 50_000, 100, 10, "success", "normal", "hive_equalization", now - 45),
        ]
        for row in rows:
            conn.execute(
                "INSERT INTO rebalance_history "
                "(from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, "
                "status, rebalance_type, reason_code, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        conn.commit()

        assert db.get_last_rebalance_time(channel, reason_code="hive_equalization") == now - 45
        assert db.get_last_rebalance_time(channel, reason_code="ev_positive") == now - 240
        assert db.get_last_rebalance_time(channel, reason_code="capex_fallback") is None

    def test_reason_filtered_lookup_keeps_distinct_cooldowns_per_reason(self, tmp_path):
        db = self._make_db(tmp_path)
        conn = db._get_connection()
        channel = "111x222x0"
        now = int(time.time())

        rows = [
            ("src-a", channel, 50_000, 100, 10, "success", "normal", "capex_fallback", now - 300),
            ("src-b", channel, 50_000, 100, 10, "success", "normal", "ev_positive", now - 200),
            ("src-c", channel, 50_000, 100, 10, "success", "normal", "hive_equalization", now - 100),
        ]
        for row in rows:
            conn.execute(
                "INSERT INTO rebalance_history "
                "(from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, "
                "status, rebalance_type, reason_code, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        conn.commit()

        assert db.get_last_rebalance_time(channel) == now - 100
        assert db.get_last_rebalance_time(channel, reason_code="capex_fallback") == now - 300
        assert db.get_last_rebalance_time(channel, reason_code="ev_positive") == now - 200
        assert db.get_last_rebalance_time(channel, reason_code="hive_equalization") == now - 100


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


class TestDeadCapitalStages:
    """Real SQLite tests for dead-capital stage persistence."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_dead_capital.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_dead_capital_stage_round_trip(self, tmp_path):
        """Stages can be inserted and read back by channel id."""
        db = self._make_db(tmp_path)

        db.upsert_dead_capital_stage("100x1x0", "fee_reduction", 123)

        stages = db.get_dead_capital_stages()
        assert stages["100x1x0"]["stage"] == "fee_reduction"
        assert stages["100x1x0"]["entered_at"] == 123

    def test_upsert_replaces_existing_stage(self, tmp_path):
        """Upsert should replace stage and entered_at for an existing row."""
        db = self._make_db(tmp_path)

        db.upsert_dead_capital_stage("100x1x0", "fee_reduction", 123)
        db.upsert_dead_capital_stage("100x1x0", "close", 456)

        stages = db.get_dead_capital_stages()
        assert stages["100x1x0"]["stage"] == "close"
        assert stages["100x1x0"]["entered_at"] == 456

    def test_delete_dead_capital_stage(self, tmp_path):
        """Deleting a stage removes it from the lookup."""
        db = self._make_db(tmp_path)

        db.upsert_dead_capital_stage("100x1x0", "close", 123)
        db.delete_dead_capital_stage("100x1x0")

        assert "100x1x0" not in db.get_dead_capital_stages()

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


class TestUpdateChannelStatesBatch:
    """Real SQLite tests for the batched channel-state upsert."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_batch.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def _row(self, channel_id, **overrides):
        row = dict(
            channel_id=channel_id, peer_id="peer_" + channel_id,
            state="balanced", flow_ratio=0.5, sats_in=100, sats_out=200,
            capacity=1_000_000, confidence=0.9, velocity=0.1,
            flow_multiplier=1.2, ema_decay=0.7, forward_count=3,
            kalman_flow_ratio=0.4, kalman_velocity=0.05,
            kalman_uncertainty=0.2,
        )
        row.update(overrides)
        return row

    def test_empty_batch_is_noop(self, tmp_path):
        db = self._make_db(tmp_path)
        db.update_channel_states_batch([])
        assert db.get_all_channel_states() == []

    def test_batch_insert_matches_single_upsert(self, tmp_path):
        db = self._make_db(tmp_path)
        db.update_channel_states_batch([self._row("1x1x0"), self._row("2x2x0")])

        state = db.get_channel_state("1x1x0")
        assert state is not None
        assert state["peer_id"] == "peer_1x1x0"
        assert state["flow_ratio"] == 0.5
        assert state["kalman_uncertainty"] == 0.2
        assert len(db.get_all_channel_states()) == 2

    def test_batch_updates_existing_rows(self, tmp_path):
        db = self._make_db(tmp_path)
        db.update_channel_state(
            channel_id="1x1x0", peer_id="old_peer", state="source",
            flow_ratio=0.1, sats_in=1, sats_out=2, capacity=500,
        )
        db.update_channel_states_batch([self._row("1x1x0", state="sink")])

        state = db.get_channel_state("1x1x0")
        assert state["state"] == "sink"
        assert state["peer_id"] == "peer_1x1x0"
        assert len(db.get_all_channel_states()) == 1

    def test_batch_defaults_for_optional_fields(self, tmp_path):
        db = self._make_db(tmp_path)
        db.update_channel_states_batch([dict(
            channel_id="3x3x0", peer_id="p3", state="balanced",
            flow_ratio=0.5, sats_in=0, sats_out=0, capacity=100,
        )])
        state = db.get_channel_state("3x3x0")
        assert state["confidence"] == 1.0
        assert state["ema_decay"] == 0.8
        assert state["kalman_uncertainty"] == 0.1


class TestForwardDoubleDipPrevention:
    """TODO #19: hook-path and hydration-path inserts of the same forward
    must deduplicate under idx_forwards_unique regardless of order."""

    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_double_dip.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    FWD = dict(
        in_channel="111x1x0", out_channel="222x2x0",
        in_msat=1_000_000, out_msat=999_000, fee_msat=1_000,
        received_time=1_700_000_100, resolved_time=1_700_000_103,
    )

    def _count(self, db):
        conn = db._get_connection()
        return conn.execute("SELECT COUNT(*) FROM forwards").fetchone()[0]

    def test_hook_then_hydration_no_duplicate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(tmp)
            f = self.FWD
            db.record_forward(
                f["in_channel"], f["out_channel"], f["in_msat"], f["out_msat"],
                f["fee_msat"], f["received_time"], f["resolved_time"], 3,
            )
            inserted = db.bulk_insert_forwards([dict(
                f, resolution_time=3,
            )])
            assert inserted == 0
            assert self._count(db) == 1

    def test_hydration_then_hook_no_duplicate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(tmp)
            f = self.FWD
            assert db.bulk_insert_forwards([dict(f, resolution_time=3)]) == 1
            db.record_forward(
                f["in_channel"], f["out_channel"], f["in_msat"], f["out_msat"],
                f["fee_msat"], f["received_time"], f["resolved_time"], 3,
            )
            assert self._count(db) == 1

    def test_legacy_colon_scid_still_deduplicates(self):
        """Hook path historically normalized ':' to 'x'; hydration uses
        normalize_scid. Same forward in both spellings must not double-dip."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(tmp)
            f = self.FWD
            db.record_forward(
                "111:1:0", "222:2:0", f["in_msat"], f["out_msat"],
                f["fee_msat"], f["received_time"], f["resolved_time"], 3,
            )
            inserted = db.bulk_insert_forwards([dict(f, resolution_time=3)])
            assert inserted == 0
            assert self._count(db) == 1
