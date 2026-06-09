# Tests for cl-mycelium immune_influence/v1 hint consumption.

import copy
import time
from unittest.mock import MagicMock

import pytest

from modules.hive_hints import HiveHintAdapter


DIRECT_PEER = "02direct"
FLEET_PEER = "02fleet"
OUT_PEER = "02out"
DEST_PEER = "02dest"


@pytest.fixture
def mock_plugin():
    plugin = MagicMock()
    plugin.rpc = MagicMock()
    plugin.log = MagicMock()
    return plugin


def _base_snapshot(**overrides):
    now = int(time.time())
    snapshot = {
        "generated_at": now,
        "ttl_seconds": 300,
        "producer": "cl-mycelium",
        "schema_version": "legacy-hints/m2",
        "m2_scope": "channel_and_fleet_peers",
        "hints": {
            DIRECT_PEER: {"direct_channel_peer": True, "member": False},
            FLEET_PEER: {"direct_channel_peer": False, "member": True},
            DEST_PEER: {"direct_channel_peer": True, "member": False},
            OUT_PEER: {"direct_channel_peer": False, "member": False},
        },
        "immune_influence": {
            "schema_version": "immune-influence/v1",
            "generated_at": now,
            "ttl_seconds": 300,
            "source": "immune_advisory",
            "enabled": True,
            "m2_scope": "channel_and_fleet_peers",
            "immune_posture": "guardrail",
            "confidence": "high",
            "global_effects": {
                "growth_allowed": False,
                "rebalance_allowed": "repair_only",
                "exploration_allowed": False,
                "peer_suppression_allowed": False,
                "closure_execution_allowed": False,
            },
            "peer_effects": {
                DIRECT_PEER: {
                    "immune_peer_posture": "hazard",
                    "pathology_class": "toxic",
                    "rebalance_priority_delta": -0.30,
                    "fee_bias_delta": -0.20,
                    "open_confidence_delta": -0.25,
                    "closure_watch_priority_delta": 0.25,
                    "reason_codes": ["toxic_peer_detected"],
                },
                DEST_PEER: {
                    "immune_peer_posture": "rehabilitating",
                    "pathology_class": "rehabilitating",
                    "rebalance_priority_delta": 0.20,
                    "fee_bias_delta": 0.20,
                    "open_confidence_delta": 0.0,
                    "closure_watch_priority_delta": -0.20,
                    "reason_codes": ["rehabilitation_in_progress"],
                },
                OUT_PEER: {
                    "immune_peer_posture": "hazard",
                    "pathology_class": "toxic",
                    "rebalance_priority_delta": -0.30,
                    "fee_bias_delta": -0.20,
                    "open_confidence_delta": -0.25,
                    "closure_watch_priority_delta": 0.25,
                    "reason_codes": ["out_of_scope_fixture"],
                },
            },
            "safety": {
                "executor_required": True,
                "executor_authority": "cl_revenue_ops",
                "direct_execution": False,
                "budget_authority": "cl_revenue_ops",
                "m2_scope_mutated": False,
                "budgets_mutated": False,
                "hints_are_advisory": True,
                "peer_suppression_applied": False,
            },
        },
    }
    snapshot.update(overrides)
    return snapshot


def _adapter(plugin, snapshot, *, allow_all_hints=False):
    adapter = HiveHintAdapter(
        plugin,
        ttl_override=0,
        allow_all_hints_m2_scope=allow_all_hints,
    )
    plugin.rpc.call.return_value = snapshot
    adapter.poll()
    return adapter


def test_no_cl_mycelium_or_missing_immune_influence_is_neutral(mock_plugin):
    snapshot = _base_snapshot()
    snapshot.pop("immune_influence")
    adapter = _adapter(mock_plugin, snapshot)

    status = adapter.get_immune_status()
    assert status["present"] is False
    assert status["usable"] is False
    assert adapter.get_immune_fee_bias(DIRECT_PEER) == 1.0
    assert adapter.get_immune_rebalance_bias(DIRECT_PEER, DEST_PEER) == 1.0
    assert adapter.get_immune_open_bias(DIRECT_PEER) == 1.0
    assert adapter.get_immune_closure_watch_bias(DIRECT_PEER) == 1.0
    constraints = adapter.get_immune_action_constraints()
    assert constraints["additional_permission"] is False
    assert constraints["execution_authority"] == "cl_revenue_ops"
    assert constraints["peer_suppression_allowed"] is False


@pytest.mark.parametrize(
    "mutator, expected_reason",
    [
        (lambda s: s["immune_influence"].update({"generated_at": int(time.time()) - 1_000, "ttl_seconds": 1}), "stale"),
        (lambda s: s.update({"immune_influence": ["bad"]}), "malformed"),
        (lambda s: s["immune_influence"].update({"schema_version": "immune-influence/v9"}), "unsupported_schema"),
        (lambda s: s["immune_influence"].update({"confidence": "low"}), "low_confidence"),
        (lambda s: s["immune_influence"].update({"peer_effects": ["bad"]}), "malformed"),
    ],
)
def test_unusable_immune_influence_neutralizes_without_crash(mock_plugin, mutator, expected_reason):
    snapshot = _base_snapshot()
    mutator(snapshot)
    adapter = _adapter(mock_plugin, snapshot)

    status = adapter.get_immune_status()
    assert status["usable"] is False
    assert expected_reason in str(status.get("reason") or "")
    assert adapter.get_immune_fee_bias(DIRECT_PEER) == 1.0
    assert adapter.get_immune_rebalance_bias(DIRECT_PEER, DEST_PEER) == 1.0
    assert adapter.get_immune_open_bias(DIRECT_PEER) == 1.0


def test_consumer_scope_neutralizes_out_of_scope_peer_effects(mock_plugin):
    adapter = _adapter(mock_plugin, _base_snapshot())

    assert adapter.get_immune_fee_bias(OUT_PEER) == 1.0
    assert adapter.get_immune_open_bias(OUT_PEER) == 1.0
    effect = adapter.get_immune_peer_effect(OUT_PEER)
    assert effect["usable"] is False
    assert effect["reason"] == "out_of_scope"
    status = adapter.get_immune_status()
    assert status["out_of_scope_peer_effect_count"] == 1
    assert status["neutralized_peer_effect_count"] >= 1


def test_all_hints_immune_scope_requires_explicit_operator_enablement(mock_plugin):
    snapshot = _base_snapshot(m2_scope="all_hints")
    snapshot["immune_influence"]["m2_scope"] = "all_hints"

    default_adapter = _adapter(mock_plugin, copy.deepcopy(snapshot))
    assert default_adapter.get_immune_status()["usable"] is False
    assert "all_hints" in str(default_adapter.get_immune_status().get("reason") or "")
    assert default_adapter.get_immune_fee_bias(OUT_PEER) == 1.0

    lab_adapter = _adapter(mock_plugin, copy.deepcopy(snapshot), allow_all_hints=True)
    assert lab_adapter.get_immune_status()["usable"] is True
    assert lab_adapter.get_immune_fee_bias(OUT_PEER) < 1.0


def test_in_scope_immune_biases_are_bounded(mock_plugin):
    adapter = _adapter(mock_plugin, _base_snapshot())

    assert adapter.get_immune_fee_bias(DIRECT_PEER) == pytest.approx(0.95)
    assert adapter.get_immune_fee_bias(DEST_PEER) == pytest.approx(1.05)
    assert adapter.get_immune_rebalance_bias(DIRECT_PEER, DEST_PEER) == pytest.approx(0.85)
    assert adapter.get_immune_open_bias(DIRECT_PEER) == pytest.approx(0.85)
    assert adapter.get_immune_open_bias(DEST_PEER) == pytest.approx(1.0)
    assert adapter.get_immune_closure_watch_bias(DIRECT_PEER) == pytest.approx(1.15)


def test_status_reports_immune_influence_diagnostics(mock_plugin):
    adapter = _adapter(mock_plugin, _base_snapshot())

    status = adapter.get_status(live_refresh=False)
    immune = status["immune_influence"]
    assert immune["present"] is True
    assert immune["schema_version"] == "immune-influence/v1"
    assert immune["fresh"] is True
    assert immune["usable"] is True
    assert immune["m2_scope"] == "channel_and_fleet_peers"
    assert immune["peer_effect_count"] == 3
    assert immune["confidence"] == "high"


def test_peer_effect_missing_delta_fields_degrades_to_neutral(mock_plugin):
    """A peer effect with only classification fields (no deltas) is neutral.

    Mirrors a producer that classifies a peer but emits no numeric deltas —
    e.g. current cl-hive always sends fee_bias_delta=0.0 and future producers
    may omit fields entirely.
    """
    snapshot = _base_snapshot()
    snapshot["immune_influence"]["peer_effects"] = {
        DIRECT_PEER: {
            "immune_peer_posture": "hazard",
            "pathology_class": "toxic",
            "reason_codes": ["toxic_peer_detected"],
        },
    }
    adapter = _adapter(mock_plugin, snapshot)

    assert adapter.get_immune_fee_bias(DIRECT_PEER) == 1.0
    assert adapter.get_immune_rebalance_bias(DIRECT_PEER, DEST_PEER) == 1.0
    assert adapter.get_immune_open_bias(DIRECT_PEER) == 1.0
    assert adapter.get_immune_closure_watch_bias(DIRECT_PEER) == 1.0
