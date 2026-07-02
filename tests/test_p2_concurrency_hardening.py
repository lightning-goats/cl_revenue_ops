"""Concurrency-hardening regression tests (deep-audit Phase 2 P2-001..P2-007).

Each test pins a NON-BEHAVIORAL safety fix that adds locking / transactions /
batching. They assert the race is closed without changing money decisions.
"""
import threading
import time

import pytest


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
