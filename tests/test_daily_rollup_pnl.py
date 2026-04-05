import os
import sys
import time
from unittest.mock import MagicMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


@pytest.fixture
def db(temp_db_path):
    plugin = MagicMock()
    plugin.log = MagicMock()
    d = Database(temp_db_path, plugin)
    d.initialize()
    return d


def _insert_forward(conn, *, in_channel, out_channel, in_msat, out_msat, fee_msat, ts):
    # resolved_time is used for dedupe uniqueness; keep it stable and non-zero.
    conn.execute(
        """
        INSERT INTO forwards
        (in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time, timestamp, resolved_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (in_channel, out_channel, int(in_msat), int(out_msat), int(fee_msat), 0.1, int(ts), int(ts) + 1),
    )


def test_windowed_pnl_includes_pruned_daily_rollups(db):
    """
    Regression: profitability windows (e.g. 30d) must still work when forwards are
    aggressively pruned (default keep ~8d) by relying on daily rollups.
    """
    now = int(time.time())
    old_ts = now - (20 * 86400)
    recent_ts = now - (2 * 86400)

    conn = db._get_connection()

    # Exit-side revenue for channel A: two old forwards (will be rolled up) + one recent forward (kept raw)
    _insert_forward(conn, in_channel="B", out_channel="A", in_msat=10_000_000, out_msat=9_990_000, fee_msat=1000, ts=old_ts + 10)
    _insert_forward(conn, in_channel="C", out_channel="A", in_msat=20_000_000, out_msat=19_980_000, fee_msat=2000, ts=old_ts + 20)
    _insert_forward(conn, in_channel="X", out_channel="A", in_msat=30_000_000, out_msat=29_950_000, fee_msat=5000, ts=recent_ts + 30)

    # Entry-side contribution for channel A: one old (rolled up) + one recent (kept raw)
    _insert_forward(conn, in_channel="A", out_channel="Y", in_msat=10_000_000, out_msat=9_996_000, fee_msat=4000, ts=old_ts + 40)
    _insert_forward(conn, in_channel="A", out_channel="Z", in_msat=1_000_000, out_msat=996_000, fee_msat=6000, ts=now - 86400)

    # Prune old forwards into daily rollups.
    db.cleanup_old_data(days_to_keep=8)

    # Channel direct PnL window should include rolled-up + recent
    pnl = db.get_channel_pnl("A", window_days=30)
    assert pnl["revenue_msat"] == 8000  # (1000 + 2000 + 5000) msat
    assert pnl["forward_count"] == 3

    inbound = db.get_channel_inbound_contribution("A", window_days=30)
    assert inbound["sourced_fee_contribution_msat"] == 10000  # (4000 + 6000) msat
    assert inbound["sourced_volume_msat"] == 11_000_000  # (10_000_000 + 1_000_000) msat
    assert inbound["sourced_forward_count"] == 2

    full = db.get_channel_full_pnl("A", window_days=30)
    assert full["direct_revenue_sats"] == 8
    assert full["sourced_fee_contribution_sats"] == 10
    assert full["total_contribution_msat"] == 10000  # max(8000, 10000) msat


def test_total_routing_revenue_includes_daily_rollups(db):
    now = int(time.time())
    old_ts = now - (20 * 86400)

    conn = db._get_connection()
    _insert_forward(conn, in_channel="B", out_channel="A", in_msat=10_000_000, out_msat=9_990_000, fee_msat=1000, ts=old_ts + 10)
    _insert_forward(conn, in_channel="A", out_channel="Y", in_msat=10_000_000, out_msat=9_996_000, fee_msat=4000, ts=old_ts + 20)

    db.cleanup_old_data(days_to_keep=8)

    since = now - (30 * 86400)
    assert db.get_total_routing_revenue(since) == 5000  # (1000 + 4000) msat


def test_last_forward_time_any_direction(db):
    now = int(time.time())
    old_ts = now - (20 * 86400)
    recent_ts = now - (2 * 86400)

    conn = db._get_connection()
    _insert_forward(conn, in_channel="A", out_channel="Y", in_msat=10_000_000, out_msat=9_996_000, fee_msat=4000, ts=old_ts + 20)
    _insert_forward(conn, in_channel="B", out_channel="A", in_msat=10_000_000, out_msat=9_990_000, fee_msat=1000, ts=recent_ts + 10)

    db.cleanup_old_data(days_to_keep=8)

    # Should reflect the most recent timestamp from raw forwards table (recent_ts + 10).
    last = db.get_last_forward_time_any_direction("A")
    assert last is not None
    assert last >= recent_ts

