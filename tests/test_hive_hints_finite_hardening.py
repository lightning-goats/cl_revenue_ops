"""Trust-boundary hardening tests for non-finite (Infinity/NaN) hint input.

The hint producer and the CLN datastore contents are untrusted. A poisoned
snapshot containing JSON Infinity/NaN literals, non-finite numeric hint
fields, an unbounded fleet_fee_prior, or an unbounded priority_score must
neutralize safely instead of crashing poll()/get_status() or pinning a
maximum bias.
"""

import math
import time

import pytest
from unittest.mock import MagicMock

from modules.hive_hints import HiveHintAdapter
from modules.rebalance_coordination_overlay import (
    _priority_score as overlay_priority_score,
)
from modules.rebalance_route_policy import (
    _priority_score as route_priority_score,
)


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


def _poisoned_datastore_adapter(mock_plugin, payload: str) -> HiveHintAdapter:
    adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
    adapter.data_service = MagicMock()
    adapter.data_service.list_datastore.return_value = {
        "datastore": [{"string": payload}]
    }
    mock_plugin.rpc.call.side_effect = Exception("connection refused")
    return adapter


def _rpc_snapshot_adapter(mock_plugin, snapshot: dict) -> HiveHintAdapter:
    adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
    adapter.data_service = MagicMock()
    adapter.data_service.list_datastore.return_value = {"datastore": []}
    mock_plugin.rpc.call.return_value = snapshot
    return adapter


class TestNonFiniteJsonLiteralsRejectedAtParse:
    def test_infinity_generated_at_in_datastore_neutralizes_poll(self, mock_plugin):
        payload = '{"generated_at": Infinity, "ttl_seconds": 900, "hints": {}}'
        adapter = _poisoned_datastore_adapter(mock_plugin, payload)
        adapter.poll()  # must not raise
        assert adapter._snapshot is None
        assert adapter.is_usable() is False
        assert adapter.get_fee_bias("02peer") == 1.0

    def test_nan_ttl_in_datastore_neutralizes_poll(self, mock_plugin):
        payload = (
            '{"generated_at": %d, "ttl_seconds": NaN, '
            '"hints": {"02peer": {"member": true}}}' % int(time.time())
        )
        adapter = _poisoned_datastore_adapter(mock_plugin, payload)
        adapter.poll()  # must not raise
        assert adapter._snapshot is None
        assert adapter.is_hive_member("02peer") is False

    def test_negative_infinity_in_hex_datastore_neutralizes_poll(self, mock_plugin):
        payload = '{"generated_at": -Infinity, "ttl_seconds": 900, "hints": {}}'
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        adapter.data_service.list_datastore.return_value = {
            "datastore": [{"hex": payload.encode().hex()}]
        }
        mock_plugin.rpc.call.side_effect = Exception("connection refused")
        adapter.poll()  # must not raise
        assert adapter._snapshot is None

    def test_get_status_survives_poisoned_datastore(self, mock_plugin):
        payload = '{"generated_at": Infinity, "ttl_seconds": 900, "hints": {}}'
        adapter = _poisoned_datastore_adapter(mock_plugin, payload)
        adapter.poll()
        status = adapter.get_status()  # must not raise
        assert status["snapshot_fresh"] is False
        assert status["snapshot_usable"] is False
        # Diagnostics must stay functional, not degrade to a refresh error.
        assert status.get("refresh_result") != "status_refresh_error"

    def test_repeated_polls_with_poisoned_datastore_stay_safe(self, mock_plugin):
        payload = '{"generated_at": Infinity, "ttl_seconds": 900, "hints": {}}'
        adapter = _poisoned_datastore_adapter(mock_plugin, payload)
        for _ in range(3):
            adapter.poll()
        assert adapter._snapshot is None


class TestNonFiniteSnapshotNumericsRejectedAtValidation:
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_generated_at_via_rpc_neutralizes(self, mock_plugin, bad):
        snapshot = {"generated_at": bad, "ttl_seconds": 900, "hints": {}}
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()  # must not raise
        assert adapter._snapshot is None
        assert adapter.is_usable() is False

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_ttl_seconds_via_rpc_neutralizes(self, mock_plugin, bad):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": bad,
            "hints": {"02peer": {"member": True}},
        }
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()  # must not raise
        assert adapter._snapshot is None
        assert adapter.is_hive_member("02peer") is False

    def test_get_status_survives_nonfinite_rpc_snapshot(self, mock_plugin):
        snapshot = {"generated_at": float("inf"), "ttl_seconds": 900, "hints": {}}
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()
        status = adapter.get_status()  # must not raise
        assert status["snapshot_fresh"] is False
        assert status.get("refresh_result") != "status_refresh_error"

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_nonfinite_nested_metabolic_timestamps_neutralize(self, mock_plugin, bad):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {"02peer": {"member": True, "direct_channel_peer": True}},
            "metabolic_influence": {
                "schema_version": "metabolic-influence/v1",
                "generated_at": bad,
                "ttl_seconds": bad,
                "confidence": "high",
                "coverage": {"24h": "sufficient"},
                "peer_effects": {"02peer": {"fee_bias_delta": 0.05}},
            },
        }
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()  # must not raise
        status = adapter.get_metabolic_status()  # must not raise
        assert status["usable"] is False
        assert adapter.get_metabolic_fee_bias("02peer") == 1.0
        # get_status walks the same nested payload; it must not raise either.
        full_status = adapter.get_status()
        assert full_status["metabolic_influence"]["usable"] is False

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_nonfinite_nested_immune_timestamps_neutralize(self, mock_plugin, bad):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {"02peer": {"member": True, "direct_channel_peer": True}},
            "immune_influence": {
                "schema_version": "immune-influence/v1",
                "generated_at": bad,
                "ttl_seconds": bad,
                "confidence": "high",
                "peer_effects": {"02peer": {"fee_bias_delta": 0.05}},
            },
        }
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()  # must not raise
        status = adapter.get_immune_status()  # must not raise
        assert status["usable"] is False
        assert adapter.get_immune_fee_bias("02peer") == 1.0
        full_status = adapter.get_status()
        assert full_status["immune_influence"]["usable"] is False


class TestNonFiniteConfidenceNeutralizesBias:
    @staticmethod
    def _snapshot_with_confidence(confidence):
        return {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02peer": {
                    "member": True,
                    "corridor_role": "owner",
                    "competition_bias": 1,
                    "peer_quality_score": 1.0,
                    "rebalance_preference": "sink",
                    "traffic_confidence": confidence,
                },
            },
        }

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_confidence_yields_exactly_neutral(self, mock_plugin, bad):
        adapter = _rpc_snapshot_adapter(
            mock_plugin, self._snapshot_with_confidence(bad)
        )
        adapter.poll()
        assert adapter.get_fee_bias("02peer") == 1.0
        assert adapter.get_rebalance_bias("02peer") == 1.0
        assert adapter.get_corridor_utilization_bias("02peer") == 1.0

    def test_nan_competition_bias_does_not_contribute(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02peer": {
                    "member": True,
                    "competition_bias": float("nan"),
                    "traffic_confidence": 1.0,
                },
            },
        }
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()
        assert adapter.get_fee_bias("02peer") == 1.0

    def test_nan_peer_quality_score_does_not_contribute(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02peer": {
                    "member": True,
                    "peer_quality_score": float("nan"),
                    "traffic_confidence": 1.0,
                },
            },
        }
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()
        assert adapter.get_rebalance_bias("02peer") == 1.0

    def test_nonfinite_scalar_hint_lookups_do_not_crash(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02peer": {
                    "member": True,
                    "traffic_confidence": float("nan"),
                    "peer_quality_score": float("nan"),
                    "external_centrality": float("inf"),
                    "reputation_score": float("nan"),
                    "fee_elasticity": float("inf"),
                    "optimal_fee_estimate_ppm": float("inf"),
                    "channel_open_hint": {
                        "open_preference": "open",
                        "topology_confidence": float("nan"),
                    },
                },
            },
        }
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()
        # int(nan)/int(inf) conversions must not raise; non-finite values
        # neutralize to the documented defaults.
        assert adapter.get_peer_quality_score("02peer") == 0.5
        assert adapter.get_centrality("02peer") == 0.0
        assert adapter.get_reputation_score("02peer") == 50
        assert adapter.get_traffic_confidence("02peer") == 0.0
        assert adapter.get_fee_elasticity("02peer") == 0.0
        assert adapter.get_optimal_fee_estimate("02peer") == 0
        assert "topology_confidence" not in adapter.get_channel_open_hint("02peer")


class TestFleetFeePriorBounded:
    @staticmethod
    def _adapter_with_fee(mock_plugin, fee):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02peer": {"member": True, "fleet_fee_median": fee},
            },
        }
        adapter = _rpc_snapshot_adapter(mock_plugin, snapshot)
        adapter.poll()
        return adapter

    def test_sane_fee_passes_through(self, mock_plugin):
        adapter = self._adapter_with_fee(mock_plugin, 250)
        assert adapter.get_fleet_fee_prior("02peer") == 250

    def test_absurd_fee_rejected_not_clamped(self, mock_plugin):
        adapter = self._adapter_with_fee(mock_plugin, 1_000_000)
        assert adapter.get_fleet_fee_prior("02peer") is None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_fee_rejected(self, mock_plugin, bad):
        adapter = self._adapter_with_fee(mock_plugin, bad)
        assert adapter.get_fleet_fee_prior("02peer") is None

    @pytest.mark.parametrize("bad", [0, -5, True])
    def test_nonpositive_or_bool_fee_rejected(self, mock_plugin, bad):
        adapter = self._adapter_with_fee(mock_plugin, bad)
        assert adapter.get_fleet_fee_prior("02peer") is None

    def test_boundary_fee_accepted(self, mock_plugin):
        adapter = self._adapter_with_fee(mock_plugin, 10_000)
        assert adapter.get_fleet_fee_prior("02peer") == 10_000

    def test_just_above_boundary_rejected(self, mock_plugin):
        adapter = self._adapter_with_fee(mock_plugin, 10_001)
        assert adapter.get_fleet_fee_prior("02peer") is None


class TestPriorityScoreClamped:
    @pytest.mark.parametrize(
        "score_fn", [route_priority_score, overlay_priority_score]
    )
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_priority_score_neutralizes(self, score_fn, bad):
        assert score_fn({"priority_score": bad}) == 0.0

    @pytest.mark.parametrize(
        "score_fn", [route_priority_score, overlay_priority_score]
    )
    def test_huge_priority_score_clamped_to_upper_bound(self, score_fn):
        assert score_fn({"priority_score": 1e9}) == 100.0

    @pytest.mark.parametrize(
        "score_fn", [route_priority_score, overlay_priority_score]
    )
    def test_sane_priority_score_passes_through(self, score_fn):
        assert score_fn({"priority_score": 50.0}) == 50.0
        assert score_fn({"priority_score": -5.0}) == 0.0
        assert score_fn({}) == 0.0
        assert score_fn({"priority_score": "junk"}) == 0.0
