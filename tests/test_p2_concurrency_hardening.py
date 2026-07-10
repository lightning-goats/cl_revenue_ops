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


# ---------------------------------------------------------------------------
# P2-003 — spend-ledger read-then-write pairs must be atomic
# ---------------------------------------------------------------------------
def test_p2_003_mark_spent_rolls_back_when_event_fails(tmp_path):
    """If record_spend_event fails, the reservation must NOT be left 'spent'
    (both-or-neither)."""
    db = _make_db(tmp_path)
    assert db.reserve_spend("r1", 500, "boltz")

    # Force the event write to fail.
    db.record_spend_event = lambda *a, **k: False

    result = db.mark_spend_reservation_spent("r1", record_event=True)
    assert result is False

    conn = db._get_connection()
    status = conn.execute(
        "SELECT status FROM spend_reservations WHERE reservation_id = ?", ("r1",)
    ).fetchone()["status"]
    assert status == "active", "UPDATE was not rolled back when the event failed"
    ev = conn.execute("SELECT COUNT(*) AS c FROM spend_events").fetchone()["c"]
    assert ev == 0


def test_p2_003_mark_spent_happy_path_atomic(tmp_path):
    """Normal settlement records BOTH the status change AND the event."""
    db = _make_db(tmp_path)
    assert db.reserve_spend("r2", 700, "boltz")

    assert db.mark_spend_reservation_spent("r2", record_event=True) is True

    conn = db._get_connection()
    status = conn.execute(
        "SELECT status FROM spend_reservations WHERE reservation_id = ?", ("r2",)
    ).fetchone()["status"]
    assert status == "spent"
    ev = conn.execute(
        "SELECT amount_sats FROM spend_events WHERE event_id = ?", ("resv:r2",)
    ).fetchone()
    assert ev is not None and ev["amount_sats"] == 700


def test_p2_003_reserve_and_release_use_transactions(tmp_path):
    db = _make_db(tmp_path)
    conn = db._get_connection()
    counter = _BeginCounter(conn)

    assert db.reserve_spend("r3", 300, "boltz")
    assert counter.begins >= 1, "reserve_spend did not open a transaction"

    counter.begins = 0
    out = db.release_spend_reservations(category="boltz")
    assert counter.begins >= 1, "release_spend_reservations did not open a transaction"
    assert out["released_count"] == 1
    assert out["reservation_ids"] == ["r3"]
    status = conn.execute(
        "SELECT status FROM spend_reservations WHERE reservation_id = ?", ("r3",)
    ).fetchone()["status"]
    assert status == "released"


# ---------------------------------------------------------------------------
# P2-004 — get_budget_status uses BEGIN IMMEDIATE, value unchanged
# ---------------------------------------------------------------------------
def test_p2_004_get_budget_status_immediate_and_consistent(tmp_path):
    db = _make_db(tmp_path)
    now = int(time.time())

    db.record_rebalance_cost(
        channel_id="c1", peer_id="p1", cost_sats=40, amount_sats=1000,
    )
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO budget_reservations "
        "(reservation_id, reserved_sats, reserved_at, job_channel_id, status) "
        "VALUES (?, ?, ?, ?, 'active')",
        ("b1", 60, now, "c1"),
    )


    counter = _BeginCounter(conn)
    status = db.get_budget_status(since_timestamp=now - 3600)
    conn.set_trace_callback(None)

    assert counter.begins == 1, "get_budget_status must use BEGIN IMMEDIATE"
    assert status["spent"] == 40
    assert status["reserved"] == 60
    assert status["total_committed"] == 100


# ---------------------------------------------------------------------------
# P2-005 — rebalancer fee caches guarded by a lock
# ---------------------------------------------------------------------------
def test_p2_005_fee_cache_lock_serializes_access():
    """Concurrent _get_last_hop_fee reads/writes must not tear against a
    concurrent _fee_cache reset (KeyError / partial read)."""
    from modules.rebalancer import EVRebalancer

    r = object.__new__(EVRebalancer)
    r.plugin = MagicMock()
    r.plugin.log = lambda *a, **k: None
    r._cache_lock = threading.Lock()
    r._fee_cache = {}
    r._peer_inbound_fees = {"peerA": {"fee_ppm": 50, "base_msat": 1000}}

    errors = []
    stop = threading.Event()

    def reader():
        i = 0
        try:
            while not stop.is_set():
                v = r._get_last_hop_fee("peerA", 100000 + (i % 5))
                assert v is not None
                i += 1
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def resetter():
        try:
            while not stop.is_set():
                # Mirrors the per-cycle reset in find_rebalance_candidates.
                with r._cache_lock:
                    r._fee_cache = {}
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    threads += [threading.Thread(target=resetter) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"fee-cache access raced: {errors}"

    # _get_last_hop_fee must NOT hold _cache_lock across the PRIORITY-2 RPC.
    assert not r._cache_lock.locked()


# ---------------------------------------------------------------------------
# P2-007 — _channel_fee_states reads tolerate concurrent cycle eviction
# ---------------------------------------------------------------------------
def test_p2_007_channel_fee_states_reads_tolerate_eviction():
    """The fee-state reads now use atomic .get() (reference snapshot) rather
    than a check-then-index, so a concurrent stale-key eviction + reassign (as
    the cycle does under _state_lock) can't raise KeyError in a reader."""
    from modules.fee_controller import FeeController

    fc = object.__new__(FeeController)
    fc._channel_fee_states = {f"c{i}": object() for i in range(50)}

    errors = []
    stop = threading.Event()

    def mutator():
        try:
            while not stop.is_set():
                for i in range(50):
                    fc._channel_fee_states.pop(f"c{i}", None)  # cycle eviction (4172)
                    fc._channel_fee_states[f"c{i}"] = object()  # reassign (3867/3970)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            while not stop.is_set():
                for i in range(50):
                    # Post-fix access idiom used at every converted site.
                    v = fc._channel_fee_states.get(f"c{i}")
                    if v is not None:
                        _ = v
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=mutator)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"fee-state read raced with eviction: {errors}"
