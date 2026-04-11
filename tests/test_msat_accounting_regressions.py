import os
import sys
import time
from unittest.mock import MagicMock

import pytest

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


def _make_db(tmp_path):
    db_path = os.path.join(tmp_path, "msat_accounting.db")
    plugin = MagicMock()
    plugin.log = MagicMock()
    db = Database(db_path, plugin)
    db.initialize()
    return db


def _ensure_column(conn, table: str, ddl: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        conn.commit()
    except Exception:
        pass


def _insert_forward(conn, *, in_channel, out_channel, in_msat, out_msat, fee_msat, ts):
    conn.execute(
        """
        INSERT INTO forwards
        (in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time, timestamp, resolved_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (in_channel, out_channel, int(in_msat), int(out_msat), int(fee_msat), 0.1, int(ts), int(ts) + 1),
    )


class TestRebalanceFeePrecision:
    def test_get_rebalance_history_by_peer_prefers_actual_fee_msat(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        now = int(time.time())
        peer_id = "02" + "b" * 64

        _ensure_column(conn, "rebalance_history", "actual_fee_msat INTEGER")

        conn.execute(
            """
            INSERT INTO channel_states
            (channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("222x333x0", peer_id, "balanced", 0.5, 0, 0, 1_000_000, now),
        )
        conn.execute(
            """
            INSERT INTO rebalance_history
            (from_channel, to_channel, amount_sats, max_fee_sats, actual_fee_sats, actual_fee_msat,
             expected_profit_sats, status, rebalance_type, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("111x222x0", "222x333x0", 50_000, 10, 2, 1501, 0, "success", "normal", now),
        )
        conn.commit()

        rows = db.get_rebalance_history_by_peer(peer_id)

        assert len(rows) == 1
        assert rows[0]["fee_paid_msat"] == 1501

    def test_historical_inbound_fee_ppm_prefers_actual_fee_msat(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        now = int(time.time())
        peer_id = "02" + "c" * 64

        _ensure_column(conn, "rebalance_history", "actual_fee_msat INTEGER")

        conn.execute(
            """
            INSERT INTO channel_states
            (channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("333x444x0", peer_id, "balanced", 0.5, 0, 0, 1_000_000, now),
        )

        rows = [
            ("111x1x0", "333x444x0", 50_000, 10, 2, 1501, 0, "success", "normal", now - 10),
            ("111x2x0", "333x444x0", 50_000, 10, 2, 1499, 0, "success", "normal", now - 20),
            ("111x3x0", "333x444x0", 50_000, 10, 2, 1500, 0, "success", "normal", now - 30),
        ]
        for row in rows:
            conn.execute(
                """
                INSERT INTO rebalance_history
                (from_channel, to_channel, amount_sats, max_fee_sats, actual_fee_sats, actual_fee_msat,
                 expected_profit_sats, status, rebalance_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
        conn.commit()

        result = db.get_historical_inbound_fee_ppm(peer_id, window_days=30, min_samples=3)

        assert result is not None
        assert result["avg_fee_ppm"] == 30
        assert result["median_fee_ppm"] == 30


class TestSignedNetConversions:
    def test_get_channel_full_pnl_rounds_negative_msat_toward_zero(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db._get_connection()
        now = int(time.time())

        _insert_forward(
            conn,
            in_channel="111x111x0",
            out_channel="222x222x0",
            in_msat=1_000_000,
            out_msat=999_001,
            fee_msat=999,
            ts=now - 10,
        )
        conn.execute(
            """
            INSERT INTO rebalance_costs
            (channel_id, peer_id, cost_sats, amount_sats, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("222x222x0", "02" + "d" * 64, 1, 50_000, now - 5),
        )
        conn.commit()

        pnl = db.get_channel_full_pnl("222x222x0", window_days=30)

        assert pnl["net_pnl_msat"] == -1
        assert pnl["net_pnl_sats"] == 0
