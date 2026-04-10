"""Tests for the pair-level futility tracker in rebalance_engine_v2.

A pair that fails execution N times within the futility window should be
skipped from future cycles (with a `pair_futility` audit skip reason)
until the failure timestamps decay out of the window.
"""

import time
from unittest.mock import MagicMock


def _make_engine(futility_threshold=3, futility_window_sec=1800):
    """Minimal RebalanceEngine for futility-tracker tests."""
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    plugin.rpc.call.side_effect = Exception("askrene unavailable")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}

    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    del config.snapshot

    database = MagicMock()
    engine = RebalanceEngine(plugin=plugin, config=config, database=database)
    # Override thresholds for deterministic tests
    engine._futility_threshold = futility_threshold
    engine._futility_window_sec = futility_window_sec
    return engine


def test_engine_tracks_pair_failures_in_dict():
    engine = _make_engine()
    assert hasattr(engine, "_pair_failures")
    assert isinstance(engine._pair_failures, dict)
    assert engine._pair_failures == {}


def test_record_pair_failure_increments_counter():
    engine = _make_engine()
    engine._record_pair_failure("100x1x0", "200x2x0")
    assert ("100x1x0", "200x2x0") in engine._pair_failures
    assert len(engine._pair_failures[("100x1x0", "200x2x0")]) == 1

    engine._record_pair_failure("100x1x0", "200x2x0")
    assert len(engine._pair_failures[("100x1x0", "200x2x0")]) == 2


def test_pair_not_in_futility_below_threshold():
    engine = _make_engine(futility_threshold=3)
    engine._record_pair_failure("100x1x0", "200x2x0")
    engine._record_pair_failure("100x1x0", "200x2x0")
    assert engine._is_pair_in_futility("100x1x0", "200x2x0") is False


def test_pair_in_futility_at_threshold():
    engine = _make_engine(futility_threshold=3)
    for _ in range(3):
        engine._record_pair_failure("100x1x0", "200x2x0")
    assert engine._is_pair_in_futility("100x1x0", "200x2x0") is True


def test_pair_futility_decays_after_window():
    engine = _make_engine(futility_threshold=3, futility_window_sec=10)
    # Manually inject old failures (11 seconds old — beyond window)
    old_ts = time.time() - 11
    engine._pair_failures[("100x1x0", "200x2x0")] = [old_ts, old_ts, old_ts]
    # Should NOT be in futility because all failures are stale
    assert engine._is_pair_in_futility("100x1x0", "200x2x0") is False
    # And the stale entries should have been pruned
    assert engine._pair_failures.get(("100x1x0", "200x2x0"), []) == []


def test_pair_futility_mixed_old_and_new():
    engine = _make_engine(futility_threshold=3, futility_window_sec=10)
    old_ts = time.time() - 11
    fresh_ts = time.time()
    engine._pair_failures[("100x1x0", "200x2x0")] = [
        old_ts, old_ts, fresh_ts, fresh_ts,
    ]
    # Only 2 fresh failures — below threshold of 3
    assert engine._is_pair_in_futility("100x1x0", "200x2x0") is False
    # Old entries pruned
    assert len(engine._pair_failures[("100x1x0", "200x2x0")]) == 2


def test_record_pair_success_clears_failure_history():
    engine = _make_engine()
    engine._record_pair_failure("100x1x0", "200x2x0")
    engine._record_pair_failure("100x1x0", "200x2x0")
    assert ("100x1x0", "200x2x0") in engine._pair_failures

    engine._record_pair_success("100x1x0", "200x2x0")
    assert ("100x1x0", "200x2x0") not in engine._pair_failures


def test_pair_futility_is_directional():
    """(A, B) and (B, A) are tracked as separate pairs."""
    engine = _make_engine(futility_threshold=2)
    engine._record_pair_failure("100x1x0", "200x2x0")
    engine._record_pair_failure("100x1x0", "200x2x0")

    # Forward direction is in futility
    assert engine._is_pair_in_futility("100x1x0", "200x2x0") is True
    # Reverse direction is NOT in futility — they're different pairs
    assert engine._is_pair_in_futility("200x2x0", "100x1x0") is False


def test_pair_futility_skip_reason_in_valid_set():
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS
    assert "pair_futility" in VALID_SKIP_REASONS
