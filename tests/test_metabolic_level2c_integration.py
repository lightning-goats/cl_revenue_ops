"""Cross-repo metabolic Level 2c integration tests.

These tests build cl-mycelium metabolic_influence/v1 payloads from the local
cl-hive checkout and feed them to cl_revenue_ops consumers. They intentionally
stay in zero-budget / dry-run style unit surfaces and do not call action RPCs.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.capacity_planner import CapacityPlanner
from modules.config import Config
from modules.fee_controller import FeeController
from modules.hive_hints import HiveHintAdapter
from modules.rebalance_engine_v2 import RebalanceEngine
from modules.rebalance_types_v2 import PairCandidate
from tests.plugin_test_utils import load_plugin_module


CL_MYCELIUM_ROOT = Path("/home/sat/bin/cl-hive")
DIRECT = "02" + "a" * 64
FLEET = "02" + "b" * 64
OUT_OF_SCOPE = "02" + "c" * 64


def _load_mycelium_builder():
    module_path = CL_MYCELIUM_ROOT / "modules" / "organism" / "metabolic_influence.py"
    spec = importlib.util.spec_from_file_location("cl_mycelium_metabolic_influence", module_path)
    assert spec is not None and spec.loader is not None, module_path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_metabolic_influence


def _producer_config(**overrides):
    values = {
        "organism_metabolism_m2_influence_enabled": True,
        "organism_metabolism_hint_scope": "channel_and_fleet_peers",
        "organism_metabolism_min_confidence": 0.50,
        "organism_metabolism_require_coverage": "24h",
        "organism_metabolism_max_peer_effect": 0.15,
        "organism_metabolism_allow_growth_effects": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hints():
    return {
        DIRECT: {
            "direct_channel_peer": True,
            "member": False,
            "traffic_confidence": 1.0,
            "rebalance_preference": "sink",
        },
        FLEET: {
            "direct_channel_peer": False,
            "member": True,
            "traffic_confidence": 1.0,
            "rebalance_preference": "neutral",
        },
        OUT_OF_SCOPE: {
            "direct_channel_peer": False,
            "member": False,
            "traffic_confidence": 1.0,
            "rebalance_preference": "sink",
        },
    }


def _arbitration(posture: str, *, net_msat: int, burn_pressure: str = "high"):
    return {
        "schema_version": "metabolic-arbitration/v1",
        "level": 2,
        "advisory_effect_applied": True,
        "executor_behavior_applied": False,
        "behavior_applied": False,
        "canonical_source": "cl_revenue_ops",
        "inputs": {
            "coverage": {
                "1h": "sufficient",
                "6h": "sufficient",
                "24h": "sufficient",
                "7d": "sufficient",
                "30d": "partial",
            },
            "net_usable_energy_msat": net_msat,
            "energy_intake_msat": max(net_msat, 0),
            "metabolic_burn_msat": 10_000,
            "actual_energy_reserves_msat": 2_000_000,
            "profitability_freshness": "fresh",
            "capex_freshness": "fresh",
        },
        "state": {
            "metabolic_posture": posture,
            "burn_pressure": burn_pressure,
            "reserve_quality": "healthy" if posture == "growth_ready" else "thin",
            "coverage_confidence": "high",
        },
        "advisory_outputs": {
            "exploration_allowed": posture == "growth_ready",
        },
        "reason_codes": ["integration_fixture", posture],
        "safety": {
            "hints_mutated": False,
            "m2_scope_mutated": False,
            "budgets_mutated": False,
            "executor_policy_mutated": False,
            "action_rpcs_called": False,
        },
    }


def _snapshot_from_mycelium(posture="guardrail", *, now=None, config=None):
    now = int(time.time()) if now is None else int(now)
    build_metabolic_influence = _load_mycelium_builder()
    hints = _hints()
    net_msat = 150_000 if posture == "growth_ready" else -150_000
    payload = build_metabolic_influence(
        _arbitration(posture, net_msat=net_msat, burn_pressure="low" if posture == "growth_ready" else "high"),
        hints,
        config=config or _producer_config(),
        now=now,
        ttl_seconds=300,
    )
    assert payload is not None
    return {
        "generated_at": now,
        "ttl_seconds": 300,
        "producer": "cl-mycelium",
        "schema_version": "legacy-hints/m2",
        "m2_scope": payload["m2_scope"],
        "hints": hints,
        "metabolic_influence": payload,
    }


def _adapter(snapshot):
    plugin = MagicMock()
    plugin.rpc = MagicMock()
    plugin.log = MagicMock()
    plugin.rpc.call.return_value = snapshot
    adapter = HiveHintAdapter(plugin, ttl_override=0)
    adapter.poll()
    return adapter, plugin


def _fee_debug(adapter):
    plugin = MagicMock()
    config = MagicMock()
    config.snapshot.return_value = config
    database = MagicMock()
    controller = FeeController(plugin, config, database)
    controller.hive_hints = adapter
    return controller.get_hive_fee_hint_debug(DIRECT), plugin


def _rebalance_debug(adapter):
    plugin = MagicMock()
    plugin.rpc = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "03" + "f" * 64}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {"configs": {"cltv-final": {"value_int": 18}}}
    database = MagicMock()
    cfg = Config(dry_run=True, daily_budget_sats=0, rebalance_router="v3")
    engine = RebalanceEngine(plugin=plugin, config=cfg, database=database)
    engine._hive_hints = adapter
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=DIRECT,
        dest_peer_id=FLEET,
        amount_sats=100_000,
        pair_budget_sats=0,
        score=10.0,
    )
    engine._apply_metabolic_rebalance_bias(pair)
    debug = engine._metabolic_rebalance_debug_for_candidates([pair], [])
    return debug, pair, plugin


def test_negative_mycelium_fixture_consumes_scoped_guardrail_influence_without_execution():
    snapshot = _snapshot_from_mycelium("guardrail")
    # Producer-side scoping excludes OUT_OF_SCOPE. Add one bad peer effect to
    # prove the consumer also neutralizes out-of-scope metabolic influence.
    snapshot["metabolic_influence"]["peer_effects"][OUT_OF_SCOPE] = dict(
        snapshot["metabolic_influence"]["peer_effects"][DIRECT]
    )
    adapter, plugin = _adapter(snapshot)

    status = adapter.get_status(live_refresh=False)["metabolic_influence"]
    assert status["present"] is True
    assert status["fresh"] is True
    assert status["usable"] is True
    assert status["m2_scope"] == "channel_and_fleet_peers"
    assert status["out_of_scope_peer_effect_count"] == 1

    assert adapter.get_metabolic_open_bias(DIRECT) < 1.0
    assert adapter.get_metabolic_rebalance_bias(DIRECT, FLEET) > 1.0
    assert adapter.get_metabolic_open_bias(OUT_OF_SCOPE) == 1.0
    assert adapter.get_metabolic_peer_effect(OUT_OF_SCOPE)["reason"] == "out_of_scope"

    constraints = adapter.get_metabolic_action_constraints()
    assert constraints["additional_permission"] is False
    assert constraints["execution_authority"] == "cl_revenue_ops"
    assert constraints["budget_authority"] == "cl_revenue_ops"

    fee_debug, fee_plugin = _fee_debug(adapter)
    assert fee_debug["metabolic_fee_influence"]["seen"] is True
    assert fee_debug["metabolic_fee_influence"]["usable"] is True
    assert fee_debug["metabolic_fee_influence"]["bias"] == pytest.approx(1.0)

    rebalance_debug, pair, rebalance_plugin = _rebalance_debug(adapter)
    assert rebalance_debug["seen"] is True
    assert rebalance_debug["usable"] is True
    assert rebalance_debug["candidate_bias_applied"] is True
    assert pair.metabolic_rebalance_bias <= 1.15
    assert pair.score == pytest.approx(10.0 * pair.metabolic_rebalance_bias)
    assert rebalance_debug["constraints"]["additional_permission"] is False

    action_names = {"pay", "sendpay", "keysend", "withdraw", "fundchannel", "close", "setchannelfee"}
    for rpc in (plugin.rpc, fee_plugin.rpc, rebalance_plugin.rpc):
        called = {str(call.args[0]) for call in rpc.call.call_args_list if call.args}
        assert called.isdisjoint(action_names)


def test_growth_ready_fixture_adds_bounded_positive_scores_without_budget_authority():
    snapshot = _snapshot_from_mycelium(
        "growth_ready",
        config=_producer_config(organism_metabolism_allow_growth_effects=True),
    )
    adapter, _plugin = _adapter(snapshot)

    assert adapter.get_metabolic_status()["usable"] is True
    assert adapter.get_metabolic_open_bias(DIRECT) == pytest.approx(1.075)
    assert adapter.get_metabolic_rebalance_bias(DIRECT, FLEET) == pytest.approx(1.075)
    assert adapter.get_metabolic_action_constraints()["additional_permission"] is False

    planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock(), config=MagicMock())
    planner.hive_hints = adapter
    assert planner._get_hive_open_score_multiplier(DIRECT) > 1.0
    zero_budget_cfg = SimpleNamespace(daily_budget_sats=0)
    ok, reason = planner._check_unified_budget(1, zero_budget_cfg)
    assert ok is False
    assert "zero limit" in reason.lower()


def test_stale_mycelium_metabolic_influence_is_neutralized():
    snapshot = _snapshot_from_mycelium("guardrail")
    snapshot["metabolic_influence"]["generated_at"] = int(time.time()) - 1_000
    snapshot["metabolic_influence"]["ttl_seconds"] = 1
    adapter, _plugin = _adapter(snapshot)

    status = adapter.get_metabolic_status()
    assert status["usable"] is False
    assert status["reason"] == "stale"
    assert adapter.get_metabolic_open_bias(DIRECT) == 1.0
    assert adapter.get_metabolic_rebalance_bias(DIRECT, FLEET) == 1.0


def test_malformed_mycelium_metabolic_influence_is_neutralized_without_crash():
    snapshot = _snapshot_from_mycelium("guardrail")
    snapshot["metabolic_influence"]["peer_effects"] = ["not", "an", "object"]
    adapter, _plugin = _adapter(snapshot)

    status = adapter.get_metabolic_status()
    assert status["usable"] is False
    assert status["reason"] == "malformed"
    assert adapter.get_metabolic_open_bias(DIRECT) == 1.0
    assert adapter.get_metabolic_rebalance_bias(DIRECT, FLEET) == 1.0


def test_revenue_status_reports_no_spend_or_execution_in_canary_fixture():
    module = load_plugin_module()
    module.database = MagicMock()
    module.database.get_all_channel_states.return_value = []
    module.database.get_recent_fee_changes.return_value = []
    module.database.get_recent_rebalances.return_value = []
    module.config = MagicMock()
    module.config.public_runtime_keys.return_value = ["daily_budget_sats", "dry_run"]
    module.config.public_runtime_dict.return_value = {"daily_budget_sats": 0, "dry_run": True}
    module.fee_controller = MagicMock()
    module.fee_controller.get_last_decision_summary.return_value = {
        "action": "hold",
        "reason": "dry_run_canary",
        "dominant_input": "fee_controller",
        "safety_block": False,
    }
    module.rebalancer = MagicMock()
    module.rebalancer.get_last_decision_summary.return_value = {
        "action": "hold",
        "reason": "zero_budget_canary",
        "dominant_input": "rebalancer",
        "safety_block": True,
        "budget_blocked": True,
    }

    status = module.revenue_status(MagicMock())

    assert status["operator_controls"]["values"]["daily_budget_sats"] == 0
    assert status["operator_controls"]["values"]["dry_run"] is True
    assert status["recent_fee_changes"] == []
    assert status["recent_rebalances"] == []
    assert status["fee_decision"]["action"] == "hold"
    assert status["rebalance_decision"]["action"] == "hold"
    assert status["rebalance_decision"]["budget_blocked"] is True
