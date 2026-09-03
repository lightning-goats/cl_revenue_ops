"""The Grand Prix scorer fails closed and requires crossed cell evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "grand_prix_scorecard.py"
CALIBRATION = ROOT / "tests" / "fixtures" / "competitive_improvement" / "calibration.v1.json"
PROTOCOL = ROOT / "tests" / "fixtures" / "competitive_improvement" / "grand-prix.v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("grand_prix_scorecard", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _topology():
    path = ROOT / "tools" / "grand_prix_manifest.py"
    spec = importlib.util.spec_from_file_location("scorecard_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return module.build_topology(calibration, public_seed=20260901)


def _protocol():
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    value["status"] = "frozen"
    value["traffic"]["sealed_holdout_seed_commitment"] = "sha256:" + "a" * 64
    return value


def _raw_metrics(count, volume, fee):
    return {"settled_count": count, "volume_msat": volume, "fee_msat": fee}


def _delta(module, count, volume, fee):
    result = _raw_metrics(count, volume, fee)
    result.update({
        "fee_msat_per_130m_sat": fee / 130_000_000,
        "fee_ppm_on_forwarded_volume": fee * 1_000_000 / volume if volume else 0.0,
    })
    return result


def _state(module, topology, replica, *, revenue_volume=2_000, clboss_volume=1_000):
    assignment = (
        {"revenue_ops": "identity-a", "clboss": "identity-b"}
        if replica % 2 else
        {"revenue_ops": "identity-b", "clboss": "identity-a"}
    )
    records = []
    identity_totals = {
        identity: {"settled_count": 0, "volume_msat": 0, "fee_msat": 0}
        for identity in ("identity-a", "identity-b")
    }
    after = {
        identity: {"settled_count": 0, "volume_msat": 0, "fee_msat": 0}
        for identity in ("identity-a", "identity-b")
    }
    for item in topology["traffic"]:
        deltas = {
            assignment["revenue_ops"]: _delta(module, 1, revenue_volume, 200),
            assignment["clboss"]: _delta(module, 1, clboss_volume, 50),
        }
        for identity in ("identity-a", "identity-b"):
            for metric in ("settled_count", "volume_msat", "fee_msat"):
                identity_totals[identity][metric] += int(deltas[identity][metric])
                after[identity][metric] = identity_totals[identity][metric]
        records.append({
            "sequence": item["sequence"],
            "class": item["class"],
            "payer": item["payer"],
            "sink": item["sink"],
            "amount_sats": item["amount_sats"],
            "outcome": "settled",
            "payment": {"status": "complete"},
            "contender_delta": deltas,
            "contender_after": copy.deepcopy(after),
        })
    overall = {
        identity: _delta(
            module,
            identity_totals[identity]["settled_count"],
            identity_totals[identity]["volume_msat"],
            identity_totals[identity]["fee_msat"],
        )
        for identity in ("identity-a", "identity-b")
    }
    zero = {identity: _delta(module, 0, 0, 0) for identity in ("identity-a", "identity-b")}
    return {
        "schema": module.RUNNER_SCHEMA,
        "status": "public_traffic_complete",
        "replica": replica,
        "topology_digest": module._digest(topology),
        "assignment": assignment,
        "image_attestation": {
            "image_id": "sha256:" + "b" * 64,
            "labels": {"org.opencontainers.image.experiment.patch_digest": "sha256:" + "c" * 64},
        },
        "controller_readback": {
            "revenue_ops": {
                "daily_budget_sats": 0, "paused": False,
                "market_fee_mode": "yield_aware",
            },
            "clboss": {"auto_close": False, "rebalance_mode": "off"},
            "warmup_seconds": 75,
            "warm_policies": {
                identity: {"channels": 16, "active_channels": 16}
                for identity in ("identity-a", "identity-b")
            },
        },
        "public_traffic": {
            "seed": topology["public_seed"],
            "records": records,
            "per_payment_attribution_complete": True,
            "post_traffic_unattributed_delta": zero,
            "settled_count": len(records),
            "failed_count": 0,
            "settled_volume_sats": sum(row["amount_sats"] for row in records),
            "contender_delta": overall,
        },
    }


def test_six_crossed_replicas_with_positive_nested_ci_win():
    module = _module()
    topology = _topology()
    states = [(f"r{replica}", _state(module, topology, replica)) for replica in range(1, 7)]
    result = module.score_states(
        topology, _protocol(), states, arm="revenue_enhanced", stage="public"
    )
    assert result["verdict"] == "revenue_ops_wins"
    assert result["promotion_eligible"] is False
    assert result["coverage"]["observed"] == {"identity-a": 3, "identity-b": 3}
    assert result["gates"] == {
        "protocol_frozen": True,
        "crossed_replica_coverage": True,
        "safety": True,
        "payment_delivery": True,
        "per_payment_cell_attribution": True,
        "cell_volume_retention": True,
        "nested_bootstrap_positive": True,
    }
    assert result["nested_bootstrap"]["ci95_msat"][0] > 0


def test_old_aggregate_only_state_is_insufficient_not_promoted():
    module = _module()
    topology = _topology()
    states = []
    for replica in range(1, 7):
        state = _state(module, topology, replica)
        state["public_traffic"]["per_payment_attribution_complete"] = False
        for record in state["public_traffic"]["records"]:
            record.pop("contender_delta")
        states.append((f"r{replica}", state))
    result = module.score_states(
        topology, _protocol(), states, arm="revenue_enhanced", stage="public"
    )
    assert result["verdict"] == "insufficient_evidence"
    assert result["gates"]["per_payment_cell_attribution"] is False
    assert result["gates"]["cell_volume_retention"] is False


def test_completed_state_remains_scoreable_after_scoped_lab_cleanup():
    module = _module()
    topology = _topology()
    state = _state(module, topology, 1)
    state["status"] = "stopped"
    state["events"] = [{"event": "lab_stopped", "backend": "docker"}]

    row = module.validate_state(state, topology, source="archived")

    assert row["replica"] == 1


def test_stopped_state_without_verified_cleanup_event_fails_closed():
    module = _module()
    topology = _topology()
    state = _state(module, topology, 1)
    state["status"] = "stopped"
    state["events"] = [{"event": "public_traffic_complete"}]

    with pytest.raises(module.ScorecardError, match="not a completed traffic state"):
        module.validate_state(state, topology, source="ambiguous")


def test_cell_volume_loss_rejects_even_when_total_profit_wins():
    module = _module()
    topology = _topology()
    states = [
        (f"r{replica}", _state(
            module, topology, replica, revenue_volume=900, clboss_volume=1_000
        ))
        for replica in range(1, 7)
    ]
    result = module.score_states(
        topology, _protocol(), states, arm="revenue_enhanced", stage="public"
    )
    assert result["verdict"] == "rejected"
    assert result["minimum_cell_volume_retention_ratio"] == pytest.approx(0.9)


def test_incomplete_matrix_does_not_prematurely_reject_noisy_cell():
    module = _module()
    topology = _topology()
    state = _state(module, topology, 2, revenue_volume=200, clboss_volume=1_000)
    result = module.score_states(
        topology, _protocol(), [("r2", state)],
        arm="revenue_enhanced", stage="public",
    )
    assert result["minimum_cell_volume_retention_ratio"] == pytest.approx(0.2)
    assert result["verdict"] == "insufficient_evidence"


def test_per_payment_totals_must_reconcile():
    module = _module()
    topology = _topology()
    state = _state(module, topology, 1)
    identity = state["assignment"]["revenue_ops"]
    state["public_traffic"]["contender_delta"][identity]["fee_msat"] += 1
    with pytest.raises(module.ScorecardError, match="does not reconcile"):
        module.validate_state(state, topology, source="bad")


def _as_equivalent(state, identifier, comparison_class):
    controls = state["controller_readback"]
    controls.pop("clboss", None)
    model = {"frozen": True}
    controls["competitor"] = {
        "id": identifier,
        "comparison_class": comparison_class,
        "direct_runtime": False,
        "configuration_digest": "sha256:" + "d" * 64,
        "model_digest": _module()._digest(model),
        "claim_scope": f"Clean-room {identifier} model; not a product runtime",
        "rebalance_mode": "off",
        "model": model,
    }
    return state


@pytest.mark.parametrize(
    ("identifier", "comparison_class"),
    [("ln_operator", "algorithm_equivalent"), ("torq", "workflow_equivalent")],
)
def test_equivalent_comparator_claim_scope_is_preserved(identifier, comparison_class):
    module = _module()
    topology = _topology()
    states = [
        (f"r{replica}", _as_equivalent(
            _state(module, topology, replica), identifier, comparison_class
        ))
        for replica in range(1, 7)
    ]
    result = module.score_states(
        topology, _protocol(), states, arm="competitor_equivalent", stage="public"
    )
    assert result["competitor"] == {
        "id": identifier,
        "comparison_class": comparison_class,
        "configuration_digest": "sha256:" + "d" * 64,
        "model_digest": module._digest({"frozen": True}),
        "claim_scope": f"Clean-room {identifier} model; not a product runtime",
        "direct_product_claim": False,
    }
    assert result["gates"]["safety"] is True


def test_mixed_equivalent_comparators_cannot_share_one_score():
    module = _module()
    topology = _topology()
    first = _as_equivalent(
        _state(module, topology, 1), "ln_operator", "algorithm_equivalent"
    )
    second = _as_equivalent(
        _state(module, topology, 2), "torq", "workflow_equivalent"
    )
    with pytest.raises(module.ScorecardError, match="one frozen competitor"):
        module.score_states(
            topology, _protocol(), [("r1", first), ("r2", second)],
            arm="competitor_equivalent", stage="public",
        )


def test_tampered_equivalent_model_fails_safety_gate():
    module = _module()
    topology = _topology()
    state = _as_equivalent(
        _state(module, topology, 1), "torq", "workflow_equivalent"
    )
    state["controller_readback"]["competitor"]["model"]["changed"] = True
    row = module.validate_state(state, topology, source="tampered")
    assert row["safety_ok"] is False
