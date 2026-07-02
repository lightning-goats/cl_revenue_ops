"""
Tests for Database.get_channel_flow_window — windowed directional forward facts.

Uses real SQLite (temp files) to verify actual SQL logic, following the
pattern established in tests/test_database.py.
"""

import os
import sys
from unittest.mock import MagicMock

# Mock pyln.client before importing modules (matches tests/test_database.py)
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


def _make_db(tmp_path):
    db_path = os.path.join(str(tmp_path), "flow_facts.db")
    plugin = MagicMock()
    db = Database(db_path, plugin)
    db.initialize()
    return db


def _seed(db, out_channel, in_channel, out_msat, in_msat, ts):
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (in_channel, out_channel, in_msat, out_msat, 0, ts),
    )
    conn.commit()


def test_get_channel_flow_window_sums_directional(tmp_path):
    db = _make_db(tmp_path)
    now = 1_000_000
    # Channel A is out_channel (outbound) twice within the window
    _seed(db, out_channel="A", in_channel="B", out_msat=30_000_000, in_msat=30_100_000, ts=now - 10)
    _seed(db, out_channel="A", in_channel="C", out_msat=20_000_000, in_msat=20_050_000, ts=now - 20)
    # Channel A is in_channel (inbound) once within the window
    _seed(db, out_channel="D", in_channel="A", out_msat=15_000_000, in_msat=15_030_000, ts=now - 30)
    # Out-of-window forward on channel A (older than since_timestamp) — must be excluded
    _seed(db, out_channel="A", in_channel="B", out_msat=99_000_000, in_msat=99_000_000, ts=now - 10_000)

    out_sats, in_sats, count = db.get_channel_flow_window("A", since_timestamp=now - 100)

    assert out_sats == 50_000
    assert in_sats == 15_030
    assert count == 3
