"""Tests for cl-mycelium metabolic_influence/v1 hint consumption."""

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
        "metabolic_influence": {
            "schema_version": "metabolic-influence/v1",
            "generated_at": now,
            "ttl_seconds": 300,
            "source": "metabolic_arbitration",
            "enabled": True,
            "m2_scope": "channel_and_fleet_peers",
            "metabolic_posture": "growth_ready",
            "confidence": "high",
            "coverage": {
                "1h": "sufficient",
                "6h": "sufficient",
                "24h": "sufficient",
                "7d": "sufficient",
                "30d": "partial",
            },
            "global_effects": {
                "growth_allowed": True,
                "rebalance_allowed": "normal",
                "exploration_allowed": True,
                "max_rebalance_burn_sats": 500,
            },
            "peer_effects": {
                DIRECT_PEER: {
                    "metabolic_peer_posture": "source_food",
                    "rebalance_priority_delta": 0.30,
                    "fee_bias_delta": 0.20,
                    "open_confidence_delta": 0.25,
                    "closure_watch_priority_delta": 0.25,
                    "reason_codes": ["positive_net_usable_energy"],
                },
                DEST_PEER: {
                    "metabolic_peer_posture": "repair_target",
                    "rebalance_priority_delta": 0.10,
                    "fee_bias_delta": -0.20,
                    "open_confidence_delta": -0.25,
                    "closure_watch_priority_delta": 0.05,
                    "reason_codes": ["repair_target"],
                },
                OUT_PEER: {
                    "metabolic_peer_posture": "hazard",
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


def test_no_cl_mycelium_or_missing_metabolic_influence_is_neutral(mock_plugin):
    snapshot = _base_snapshot()
    snapshot.pop("metabolic_influence")
    adapter = _adapter(mock_plugin, snapshot)

    status = adapter.get_metabolic_status()
    assert status["present"] is False
    assert status["usable"] is False
    assert adapter.get_metabolic_fee_bias(DIRECT_PEER) == 1.0
    assert adapter.get_metabolic_rebalance_bias(DIRECT_PEER, DEST_PEER) == 1.0
    assert adapter.get_metabolic_open_bias(DIRECT_PEER) == 1.0
    assert adapter.get_metabolic_closure_watch_bias(DIRECT_PEER) == 1.0
    constraints = adapter.get_metabolic_action_constraints()
    assert constraints["additional_permission"] is False
    assert constraints["execution_authority"] == "cl_revenue_ops"


@pytest.mark.parametrize(
    "mutator, expected_reason",
    [
        (lambda s: s["metabolic_influence"].update({"generated_at": int(time.time()) - 1_000, "ttl_seconds": 1}), "stale"),
        (lambda s: s.update({"metabolic_influence": ["bad"]}), "malformed"),
        (lambda s: s["metabolic_influence"].update({"schema_version": "metabolic-influence/v9"}), "unsupported_schema"),
        (lambda s: s["metabolic_influence"].update({"confidence": "low"}), "low_confidence"),
        (lambda s: s["metabolic_influence"].update({"coverage": {"24h": "insufficient", "7d": "sufficient"}}), "insufficient_coverage"),
    ],
)
def test_unusable_metabolic_influence_neutralizes_without_crash(mock_plugin, mutator, expected_reason):
    snapshot = _base_snapshot()
    mutator(snapshot)
    adapter = _adapter(mock_plugin, snapshot)

    status = adapter.get_metabolic_status()
    assert status["usable"] is False
    assert expected_reason in str(status.get("reason") or "")
    assert adapter.get_metabolic_fee_bias(DIRECT_PEER) == 1.0
    assert adapter.get_metabolic_rebalance_bias(DIRECT_PEER, DEST_PEER) == 1.0
    assert adapter.get_metabolic_open_bias(DIRECT_PEER) == 1.0


def test_consumer_scope_neutralizes_out_of_scope_peer_effects(mock_plugin):
    adapter = _adapter(mock_plugin, _base_snapshot())

    assert adapter.get_metabolic_fee_bias(OUT_PEER) == 1.0
    assert adapter.get_metabolic_open_bias(OUT_PEER) == 1.0
    effect = adapter.get_metabolic_peer_effect(OUT_PEER)
    assert effect["usable"] is False
    assert effect["reason"] == "out_of_scope"
    status = adapter.get_metabolic_status()
    assert status["out_of_scope_peer_effect_count"] == 1
    assert status["neutralized_peer_effect_count"] >= 1


def test_all_hints_metabolic_scope_requires_explicit_operator_enablement(mock_plugin):
    snapshot = _base_snapshot(m2_scope="all_hints")
    snapshot["metabolic_influence"]["m2_scope"] = "all_hints"

    default_adapter = _adapter(mock_plugin, copy.deepcopy(snapshot))
    assert default_adapter.get_metabolic_status()["usable"] is False
    assert "all_hints" in str(default_adapter.get_metabolic_status().get("reason") or "")
    assert default_adapter.get_metabolic_fee_bias(OUT_PEER) == 1.0

    lab_adapter = _adapter(mock_plugin, copy.deepcopy(snapshot), allow_all_hints=True)
    assert lab_adapter.get_metabolic_status()["usable"] is True
    assert lab_adapter.get_metabolic_fee_bias(OUT_PEER) < 1.0


def test_in_scope_metabolic_biases_are_bounded(mock_plugin):
    adapter = _adapter(mock_plugin, _base_snapshot())

    assert adapter.get_metabolic_fee_bias(DIRECT_PEER) == pytest.approx(1.05)
    assert adapter.get_metabolic_fee_bias(DEST_PEER) == pytest.approx(0.95)
    assert adapter.get_metabolic_rebalance_bias(DIRECT_PEER, DEST_PEER) == pytest.approx(1.15)
    assert adapter.get_metabolic_open_bias(DIRECT_PEER) == pytest.approx(1.10)
    assert adapter.get_metabolic_open_bias(DEST_PEER) == pytest.approx(0.85)
    assert adapter.get_metabolic_closure_watch_bias(DIRECT_PEER) == pytest.approx(1.15)


def test_status_reports_metabolic_influence_diagnostics(mock_plugin):
    adapter = _adapter(mock_plugin, _base_snapshot())

    status = adapter.get_status(live_refresh=False)
    metabolic = status["metabolic_influence"]
    assert metabolic["present"] is True
    assert metabolic["schema_version"] == "metabolic-influence/v1"
    assert metabolic["fresh"] is True
    assert metabolic["usable"] is True
    assert metabolic["m2_scope"] == "channel_and_fleet_peers"
    assert metabolic["peer_effect_count"] == 3
    assert metabolic["confidence"] == "high"


def test_non_object_metabolic_peer_effects_are_malformed_and_neutral(mock_plugin):
    snapshot = _base_snapshot()
    snapshot["metabolic_influence"]["peer_effects"] = ["bad"]
    adapter = _adapter(mock_plugin, snapshot)

    status = adapter.get_metabolic_status()
    assert status["usable"] is False
    assert status["reason"] == "malformed"
    assert adapter.get_metabolic_fee_bias(DIRECT_PEER) == 1.0
    assert adapter.get_metabolic_rebalance_bias(DIRECT_PEER, DEST_PEER) == 1.0
