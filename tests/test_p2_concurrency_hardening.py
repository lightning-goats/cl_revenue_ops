"""Concurrency-hardening regression tests (deep-audit Phase 2 P2-001..P2-007).

Each test pins a NON-BEHAVIORAL safety fix that adds locking / transactions /
batching. They assert the race is closed without changing money decisions.
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

# Mock pyln.client before importing modules (mirrors other DB tests).
_mock_pyln = MagicMock()
_mock_pyln.Plugin = MagicMock
_mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', _mock_pyln)
sys.modules.setdefault('pyln.client', _mock_pyln)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_db(tmp_path, name="p2.db"):
    from modules.database import Database

    db = Database(os.path.join(str(tmp_path), name), MagicMock())
    db.initialize()
    return db


class _BeginCounter:
    """Count BEGIN IMMEDIATE statements via sqlite3 trace callback."""

    def __init__(self, conn):
        self.begins = 0
        conn.set_trace_callback(self._trace)

    def _trace(self, sql):
        if sql and sql.strip().upper().startswith("BEGIN IMMEDIATE"):
            self.begins += 1


# ---------------------------------------------------------------------------
# P2-001 — Kalman filter mutation must be serialized (flow_analysis.py)
# ---------------------------------------------------------------------------
class _ReentrancyDetectingFilter:
    """Stand-in KalmanFlowFilter that flags concurrent predict/update.

    If two threads mutate the same filter at once, one of them observes the
    in-progress flag set on entry and records a violation. With the fix
    (predict/update held under _kalman_lock) the calls serialize and no
    violation is ever recorded.
    """

    def __init__(self):
        self._busy = False
        self.violations = 0
        self._nan_recovery_count = 0

        class _State:
            flow_ratio = 0.0
            flow_velocity = 0.0
            observation_count = 0
            last_update = 0

            def to_dict(self):
                return {
                    "flow_ratio": self.flow_ratio,
                    "flow_velocity": self.flow_velocity,
                    "observation_count": self.observation_count,
                    "last_update": self.last_update,
                    "velocity_unit": "per_hour",
                }

        self.state = _State()

    def _enter(self):
        if self._busy:
            self.violations += 1
        self._busy = True
        time.sleep(0.002)  # widen the interleave window

    def _exit(self):
        self._busy = False

    def predict(self, dt_hours, volatility=1.0):
        self._enter()
        self._exit()

    def update(self, observed_ratio, confidence=1.0):
        self._enter()
        self.state.observation_count += 1
        self._exit()
        return 0.0

    def _has_nan(self):
        return False

    def _reset_state(self):
        pass

    def is_regime_change(self, threshold=2.5):
        return False

    def get_uncertainty(self):
        return 0.1


def _make_flow_analyzer():
    from unittest.mock import MagicMock
    from modules.flow_analysis import FlowAnalyzer

    plugin = MagicMock()
    plugin.log = MagicMock()
    config = MagicMock()
    database = MagicMock()
    database.kalman_purge_needed.return_value = False
    return FlowAnalyzer(plugin, config, database)


def test_p2_001_kalman_mutation_serialized():
    """Two threads applying the Kalman filter to the same channel must not
    interleave predict/update (torn covariance / double-applied observation)."""
    from modules import flow_analysis

    if not flow_analysis.ENABLE_KALMAN_FILTER:
        pytest.skip("Kalman filter disabled")

    analyzer = _make_flow_analyzer()
    fake = _ReentrancyDetectingFilter()
    analyzer._kalman_filters["chanX"] = fake

    def worker():
        for _ in range(15):
            analyzer._apply_kalman_filter(
                channel_id="chanX",
                observed_ratio=0.5,
                confidence=1.0,
                daily_buckets=[],
                has_observation=True,
            )

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert fake.violations == 0, (
        f"predict/update interleaved {fake.violations} times — filter mutation "
        "is not serialized under _kalman_lock"
    )


def test_p2_001_analyze_channel_takes_analysis_lock():
    """The RPC analyze_channel path must respect the same _analysis_lock the
    flow daemon uses so it cannot run a fresh Kalman cycle concurrently with a
    bulk analyze_all_channels cycle."""
    analyzer = _make_flow_analyzer()

    # Hold the analysis lock as the daemon would during a bulk cycle.
    acquired = analyzer._analysis_lock.acquire(blocking=False)
    assert acquired

    started = threading.Event()
    finished = threading.Event()

    def call_rpc():
        started.set()
        # analyze_channel should block (bounded) on the held lock rather than
        # barge in. _get_channel returns None quickly once it gets the lock.
        analyzer._get_channel = lambda cid: None
        analyzer.analyze_channel("chanX")
        finished.set()

    t = threading.Thread(target=call_rpc)
    t.start()
    started.wait(1.0)
    time.sleep(0.1)
    # While we still hold the lock, the RPC must not have completed.
    assert not finished.is_set(), "analyze_channel did not wait on _analysis_lock"

    analyzer._analysis_lock.release()
    finished.wait(3.0)
    assert finished.is_set(), "analyze_channel never completed after lock release"
    t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# P2-002 — long batches must be chunked so the WAL writer is bounded
# ---------------------------------------------------------------------------
def _mk_fwd(i, ts, out_chan="A", fee=10):
    return {
        "in_channel": f"in{i}",
        "out_channel": out_chan,
        "in_msat": 1000,
        "out_msat": 900,
        "fee_msat": fee,
        "received_time": ts,
        "resolved_time": ts + 1,
        "resolution_time": 1,
    }


def test_p2_002_bulk_insert_forwards_chunked(tmp_path):
    db = _make_db(tmp_path)
    db.BULK_WRITE_BATCH_SIZE = 3  # small batch to force chunking
    now = int(time.time())
    fwds = [_mk_fwd(i, now - i) for i in range(7)]

    conn = db._get_connection()
    counter = _BeginCounter(conn)
    inserted = db.bulk_insert_forwards(fwds)
    conn.set_trace_callback(None)

    assert inserted == 7  # net effect preserved
    # ceil(7/3) == 3 transactions => writer released between chunks
    assert counter.begins == 3, f"expected 3 chunked transactions, got {counter.begins}"

    row = conn.execute("SELECT COUNT(*) AS c FROM forwards").fetchone()
    assert row["c"] == 7


def test_p2_002_cleanup_old_data_chunked_same_net_effect(tmp_path):
    db = _make_db(tmp_path)
    now = int(time.time())
    old_ts = now - (20 * 86400)  # older than the 8-day cutoff
    day_ts = (old_ts // 86400) * 86400

    # 7 old forwards, same out_channel + same day so they aggregate into one row.
    fwds = [_mk_fwd(i, old_ts, out_chan="OUT", fee=100) for i in range(7)]
    db.bulk_insert_forwards(fwds)

    db.BULK_WRITE_BATCH_SIZE = 3
    conn = db._get_connection()
    counter = _BeginCounter(conn)
    db.cleanup_old_data(days_to_keep=8)
    conn.set_trace_callback(None)

    # forwards prune ran in chunks (bounded writer hold)
    assert counter.begins >= 3, f"forwards prune not chunked: {counter.begins} tx"

    # net effect: all old forwards gone
    remaining = conn.execute("SELECT COUNT(*) AS c FROM forwards").fetchone()["c"]
    assert remaining == 0

    # net effect: aggregation additive across chunks == single-pass total
    agg = conn.execute(
        "SELECT total_fee_msat, forward_count FROM daily_forwarding_stats "
        "WHERE channel_id = ? AND date = ?",
        ("OUT", day_ts),
    ).fetchone()
    assert agg["forward_count"] == 7
    assert agg["total_fee_msat"] == 7 * 100


def test_p2_002_concurrent_small_write_during_bulk_insert(tmp_path):
    """A concurrent small write must not raise 'database is locked' while a
    large chunked bulk insert runs."""
    db = _make_db(tmp_path)
    db.BULK_WRITE_BATCH_SIZE = 50
    now = int(time.time())
    big = [_mk_fwd(i, now - i) for i in range(600)]

    errors = []

    def small_writer():
        try:
            # Own thread => own connection. Hammer a small write repeatedly.
            for k in range(200):
                db.record_fee_change(
                    channel_id=f"c{k}", peer_id="p", old_fee_ppm=1, new_fee_ppm=2,
                    reason="test",
                )
                time.sleep(0)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    t = threading.Thread(target=small_writer)
    t.start()
    db.bulk_insert_forwards(big)
    t.join()

    assert not errors, f"concurrent small write failed: {errors}"
    assert db._get_connection().execute(
        "SELECT COUNT(*) AS c FROM forwards"
    ).fetchone()["c"] == 600


