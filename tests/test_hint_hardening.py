"""Hardening tests for the cl-hive -> cl_revenue_ops trust boundary.

Treat the hint snapshot as untrusted, possibly-adversarial input (a poisoned
CLN datastore entry, a buggy/compromised producer, a hostile fleet member
whose values flow through cl-hive into the hints).
"""

import time
from unittest.mock import MagicMock

import pytest

from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


def _adapter_with(plugin, snapshot, *, ttl_override=0):
    adapter = HiveHintAdapter(plugin, ttl_override=ttl_override)
    plugin.rpc.call.return_value = snapshot
    # Force the RPC path (no datastore entry) for deterministic tests.
    adapter.data_service = MagicMock()
    adapter.data_service.list_datastore.return_value = {"datastore": []}
    adapter.poll()
    return adapter


def _base_snapshot(**overrides):
    snap = {
        "generated_at": int(time.time()),
        "ttl_seconds": 900,
        "hints": {
            "02aabb": {"member": True, "direct_channel_peer": True},
        },
    }
    snap.update(overrides)
    return snap


class TestFutureTimestamp:
    def test_far_future_snapshot_is_not_fresh(self, mock_plugin):
        """A future generated_at must not pin the snapshot as fresh forever."""
        snap = _base_snapshot(generated_at=int(time.time()) + 100_000)
        adapter = _adapter_with(mock_plugin, snap)
        assert adapter.is_fresh() is False

    def test_far_future_snapshot_is_not_usable(self, mock_plugin):
        snap = _base_snapshot(generated_at=int(time.time()) + 100_000)
        adapter = _adapter_with(mock_plugin, snap)
        assert adapter.is_usable() is False

    def test_small_clock_skew_still_fresh(self, mock_plugin):
        """A few seconds of forward clock skew is tolerated."""
        snap = _base_snapshot(generated_at=int(time.time()) + 5)
        adapter = _adapter_with(mock_plugin, snap)
        assert adapter.is_fresh() is True


class TestTtlBounds:
    def test_oversized_ttl_is_capped(self, mock_plugin):
        """A huge ttl_seconds must not make an old snapshot read as fresh."""
        snap = _base_snapshot(
            generated_at=int(time.time()) - 100_000,  # ~27h old
            ttl_seconds=10 ** 12,
        )
        adapter = _adapter_with(mock_plugin, snap)
        assert adapter.is_fresh() is False


class TestClampFloat:
    def test_nan_returns_default(self):
        assert HiveHintAdapter._clamp_float(float("nan"), -1.0, 1.0, default=0.0) == 0.0

    def test_inf_returns_default(self):
        assert HiveHintAdapter._clamp_float(float("inf"), -1.0, 1.0, default=0.0) == 0.0
        assert HiveHintAdapter._clamp_float(float("-inf"), -1.0, 1.0, default=0.0) == 0.0

    def test_finite_value_clamps_normally(self):
        assert HiveHintAdapter._clamp_float(5.0, -1.0, 1.0, default=0.0) == 1.0
        assert HiveHintAdapter._clamp_float(0.3, -1.0, 1.0, default=0.0) == 0.3


class TestOpenCandidatesRace:
    def test_no_crash_if_snapshot_cleared_after_freshness_check(self, mock_plugin):
        """get_open_candidates must read one consistent snapshot under the lock.

        Simulates poll() clearing the snapshot between the freshness check and
        the iteration: is_fresh() reports True while _snapshot is None.
        """
        adapter = _adapter_with(mock_plugin, _base_snapshot())
        adapter._snapshot = None
        adapter.is_fresh = lambda: True  # the racing window
        assert adapter.get_open_candidates() == []


class TestFleetBalanceBounds:
    def test_negative_capacity_rejected(self, mock_plugin):
        snap = _base_snapshot()
        snap["hints"]["02aabb"]["fleet_capacity_sats"] = -5
        snap["hints"]["02aabb"]["fleet_available_sats"] = 10
        adapter = _adapter_with(mock_plugin, snap)
        assert adapter.get_fleet_balance("02aabb") == {}

    def test_available_exceeding_capacity_rejected(self, mock_plugin):
        snap = _base_snapshot()
        snap["hints"]["02aabb"]["fleet_capacity_sats"] = 1_000_000
        snap["hints"]["02aabb"]["fleet_available_sats"] = 5_000_000
        adapter = _adapter_with(mock_plugin, snap)
        assert adapter.get_fleet_balance("02aabb") == {}

    def test_absurd_capacity_rejected(self, mock_plugin):
        snap = _base_snapshot()
        snap["hints"]["02aabb"]["fleet_capacity_sats"] = 2 ** 70
        snap["hints"]["02aabb"]["fleet_available_sats"] = 2 ** 69
        adapter = _adapter_with(mock_plugin, snap)
        assert adapter.get_fleet_balance("02aabb") == {}

    def test_valid_balance_passes(self, mock_plugin):
        snap = _base_snapshot()
        snap["hints"]["02aabb"]["fleet_capacity_sats"] = 5_000_000
        snap["hints"]["02aabb"]["fleet_available_sats"] = 2_000_000
        adapter = _adapter_with(mock_plugin, snap)
        bal = adapter.get_fleet_balance("02aabb")
        assert bal.get("capacity_sats") == 5_000_000
        assert bal.get("available_sats") == 2_000_000
