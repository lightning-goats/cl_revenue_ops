import os

from unittest.mock import MagicMock

from modules.database import Database


class TestPairRebalanceCooldown:
    def _make_db(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_pair_cooldown.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_record_pair_failure_persists_active_cooldown(self, tmp_path):
        db = self._make_db(tmp_path)

        row = db.record_pair_rebalance_failure(
            "111x1x0",
            "222x1x0",
            "temporary_channel_failure",
            cooldown_seconds=1800,
            now=1_000,
        )

        assert row["failure_kind"] == "temporary_channel_failure"
        assert row["failure_count"] == 1
        assert row["cooldown_until"] == 2_800

        fetched = db.get_pair_rebalance_cooldown("111x1x0", "222x1x0", now=1_500)
        assert fetched is not None
        assert fetched["failure_kind"] == "temporary_channel_failure"
        assert fetched["failure_count"] == 1
        assert fetched["cooldown_until"] == 2_800

    def test_repeated_pair_failures_extend_cooldown(self, tmp_path):
        db = self._make_db(tmp_path)

        first = db.record_pair_rebalance_failure(
            "111x1x0",
            "222x1x0",
            "temporary_channel_failure",
            cooldown_seconds=600,
            now=1_000,
        )
        second = db.record_pair_rebalance_failure(
            "111x1x0",
            "222x1x0",
            "temporary_channel_failure",
            cooldown_seconds=600,
            now=1_100,
        )

        assert first["failure_count"] == 1
        assert second["failure_count"] == 2
        assert second["cooldown_until"] > first["cooldown_until"]

    def test_anchor_state_round_trip_for_drift_override(self, tmp_path):
        """Phase 3.3: the database tracks post-rebalance local ratio so the
        engine can detect drift since the last successful rebalance."""
        db = self._make_db(tmp_path)

        rebalance_id = db.record_rebalance(
            from_channel="111x1x0",
            to_channel="222x2x0",
            amount_sats=500_000,
            max_fee_sats=100,
            expected_profit_sats=50,
            reason_code="manual_test",
        )
        db.update_rebalance_result(
            rebalance_id,
            status="success",
            actual_fee_sats=80,
            post_local_ratio=0.55,
        )

        anchor = db.get_last_post_rebalance_state("222x2x0")
        assert anchor is not None
        assert abs(anchor["post_local_ratio"] - 0.55) < 1e-6
        assert anchor["amount_sats"] == 500_000

    def test_update_rebalance_result_can_replace_planned_with_actual_amount(
        self, tmp_path
    ):
        db = self._make_db(tmp_path)

        rebalance_id = db.record_rebalance(
            from_channel="111x1x0",
            to_channel="222x2x0",
            amount_sats=500_000,
            max_fee_sats=100,
            expected_profit_sats=50,
            reason_code="manual_test",
        )
        db.update_rebalance_result(
            rebalance_id,
            status="success",
            actual_fee_sats=6,
            post_local_ratio=0.40,
            amount_sats=5_000,
        )

        anchor = db.get_last_post_rebalance_state("222x2x0")
        assert anchor is not None
        assert anchor["amount_sats"] == 5_000

    def test_clear_pair_failure_removes_cooldown(self, tmp_path):
        db = self._make_db(tmp_path)

        db.record_pair_rebalance_failure(
            "111x1x0",
            "222x1x0",
            "temporary_channel_failure",
            cooldown_seconds=600,
            now=1_000,
        )

        assert db.clear_pair_rebalance_failure("111x1x0", "222x1x0") is True
        assert db.get_pair_rebalance_cooldown("111x1x0", "222x1x0", now=1_100) is None
