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


from modules.rebalance_flow_facts import ChannelFlowFacts, compute_channel_flow_facts


class _Cfg:
    rebalance_activity_window_seconds = 3600
    rebalance_utilization_window_days = 7
    rebalance_utilization_floor = 0.05
    rebalance_utilization_ceiling = 1.0
    rebalance_utilization_min_forwards = 5


def test_realized_utilization_clamped_and_ratio(tmp_path):
    db = _make_db(tmp_path)
    now = 2_000_000
    for i in range(6):
        _seed(db, out_channel="A", in_channel="B", out_msat=100_000_000, in_msat=100_000_000, ts=now - 100 * (i + 1))
    facts = compute_channel_flow_facts(db, "A", capacity_sats=1_000_000, now=now, cfg=_Cfg())
    assert abs(facts.realized_utilization - 0.6) < 1e-6   # 600k out / 1M cap
    assert facts.utilization_is_realized is True
    assert facts.forward_count_window == 6


def test_thin_history_falls_back_to_prior(tmp_path):
    db = _make_db(tmp_path)
    now = 2_000_000
    _seed(db, out_channel="A", in_channel="B", out_msat=10_000_000, in_msat=10_000_000, ts=now - 50)
    facts = compute_channel_flow_facts(db, "A", capacity_sats=1_000_000, now=now, cfg=_Cfg())
    assert facts.utilization_is_realized is False   # 1 fwd < min_forwards 5
    assert facts.realized_utilization == 0.5


def test_zero_capacity_is_safe(tmp_path):
    db = _make_db(tmp_path)
    facts = compute_channel_flow_facts(db, "A", capacity_sats=0, now=2_000_000, cfg=_Cfg())
    assert facts.realized_utilization == 0.5
    assert facts.utilization_is_realized is False
    assert facts.out_sats_window == 0 and facts.in_sats_window == 0


def test_util_clamped_to_ceiling(tmp_path):
    db = _make_db(tmp_path)
    now = 2_000_000
    for i in range(6):
        _seed(db, out_channel="A", in_channel="B", out_msat=500_000_000, in_msat=500_000_000, ts=now - 10 * (i + 1))
    facts = compute_channel_flow_facts(db, "A", capacity_sats=1_000_000, now=now, cfg=_Cfg())
    assert facts.realized_utilization == 1.0   # 3M/1M clamped to ceiling


def test_fail_open_on_db_error():
    class _RaisingDb:
        def get_channel_flow_window(self, channel_id, since):
            raise RuntimeError("db exploded")
    facts = compute_channel_flow_facts(_RaisingDb(), "A", capacity_sats=1_000_000, now=2_000_000, cfg=_Cfg())
    assert facts.realized_utilization == 0.5
    assert facts.utilization_is_realized is False
    assert facts.out_sats_window == 0 and facts.in_sats_window == 0
    assert facts.forward_count_window == 0
    assert facts.channel_id == "A"


def test_falsy_config_values_are_respected(tmp_path):
    """min_forwards=0 disables the thin-history gate and floor=0.0 removes the
    floor clamp — legitimate falsy settings must not collapse to defaults."""
    db = _make_db(tmp_path)
    now = 2_000_000
    # a single small outbound forward: 1000 sats out of a 1M-sat channel -> raw 0.001
    _seed(db, out_channel="A", in_channel="B", out_msat=1_000_000, in_msat=1_000_000, ts=now - 50)

    class _FalsyCfg:
        rebalance_activity_window_seconds = 3600
        rebalance_utilization_window_days = 7
        rebalance_utilization_floor = 0.0        # no floor
        rebalance_utilization_ceiling = 1.0
        rebalance_utilization_min_forwards = 0   # gate disabled

    facts = compute_channel_flow_facts(db, "A", capacity_sats=1_000_000, now=now, cfg=_FalsyCfg())
    # gate disabled -> realized even with 1 forward; floor 0.0 -> raw 0.001 not lifted to 0.05
    assert facts.utilization_is_realized is True
    assert abs(facts.realized_utilization - 0.001) < 1e-9
