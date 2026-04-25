import importlib.util
import sys
import time
from pathlib import Path


def load_audit():
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "hive_hints_truth_audit.py"
    spec = importlib.util.spec_from_file_location("hive_hints_truth_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OWN = "02" + "0" * 64
PEER_A = "02" + "a" * 64
PEER_B = "03" + "b" * 64


def evidence(*, open_existing_member: bool = False, role: str = "none"):
    now = int(time.time())
    open_hint = {
        "open_preference": "open",
        "topology_confidence": 0.59,
        "suggested_size_bucket": "medium",
        "reason": "member_connectivity",
    }
    hints = {
        PEER_A: {
            "member": True,
            "corridor_role": role,
            "competition_bias": 0,
            "traffic_confidence": 0.5,
            "optimal_fee_estimate_ppm": 30,
            "rebalance_preference": "neutral",
        },
        PEER_B: {
            "member": True,
            "corridor_role": "none",
            "competition_bias": 0,
            "traffic_confidence": 0.5,
            "optimal_fee_estimate_ppm": 45,
            "rebalance_preference": "neutral",
        },
    }
    if open_existing_member:
        hints[PEER_A]["channel_open_hint"] = open_hint

    return {
        "hive_status": {
            "membership": {"pubkey": OWN},
        },
        "hive_members": {
            "members": [
                {"peer_id": OWN},
                {"peer_id": PEER_A},
                {"peer_id": PEER_B},
            ]
        },
        "hive_export_hints": {
            "generated_at": now,
            "ttl_seconds": 900,
            "peer_count": len(hints),
            "hints": hints,
        },
        "hive_corridor_assignments": {"assignments": []},
        "hive_yield_metrics": {
            "channels": [
                {"peer_id": PEER_A, "flow_direction": "balanced", "volume_routed_msat": 0},
                {"peer_id": PEER_B, "flow_direction": "balanced", "volume_routed_msat": 0},
            ]
        },
        "hive_traffic_intelligence": {"profiles": []},
        "hive_fee_profiles": {
            "profiles": [
                {
                    "peer_id": PEER_A,
                    "optimal_fee_estimate": 30,
                    "confidence": 0.24,
                    "total_hive_volume": 0,
                },
                {
                    "peer_id": PEER_B,
                    "optimal_fee_estimate": 45,
                    "confidence": 0.74,
                    "total_hive_volume": 1_000_000,
                },
            ]
        },
        "hive_member_connectivity": {
            "recommended_connections": (
                [{"member_id": PEER_A, "reason": "fleet_member"}]
                if open_existing_member else
                []
            )
        },
        "listpeerchannels": {
            "channels": (
                [{"peer_id": PEER_A, "state": "CHANNELD_NORMAL"}]
                if open_existing_member else
                []
            )
        },
    }


def test_truth_audit_warns_on_defaulted_low_confidence_data_without_false_hints():
    audit = load_audit()

    analysis = audit.analyze_evidence(evidence())

    assert analysis["verdict"] == "warn"
    assert analysis["issue_counts"]["errors"] == 0
    assert {obs["code"] for obs in analysis["observations"]} >= {
        "traffic_confidence_defaulted",
        "low_fee_profile_confidence",
    }
    assert analysis["usefulness"]["optimal_fee_peers"] == [PEER_A, PEER_B]


def test_truth_audit_fails_open_hint_to_existing_member_channel():
    audit = load_audit()

    analysis = audit.analyze_evidence(evidence(open_existing_member=True))

    assert analysis["verdict"] == "fail"
    codes = {issue["code"] for issue in analysis["issues"]}
    assert "open_hint_existing_member_channel" in codes
    assert "connectivity_report_ignores_direct_channels" in codes
    assert analysis["summary"]["false_open_hint_count"] == 1
    assert analysis["usefulness"]["true_open_hint_peers"] == []


def test_truth_audit_detects_corridor_role_mismatch():
    audit = load_audit()

    analysis = audit.analyze_evidence(evidence(role="owner"))

    assert analysis["verdict"] == "fail"
    assert "corridor_role_mismatch" in {issue["code"] for issue in analysis["issues"]}


def test_truth_audit_markdown_summarizes_verdict_and_issues():
    audit = load_audit()
    analysis = audit.analyze_evidence(evidence(open_existing_member=True))

    report = audit.render_markdown(analysis)

    assert "Verdict: **FAIL**" in report
    assert "open_hint_existing_member_channel" in report
