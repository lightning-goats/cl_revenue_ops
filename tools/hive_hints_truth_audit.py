#!/usr/bin/env python3
"""Audit whether live cl-hive hints are true and useful for cl_revenue_ops.

This is intentionally an evidence audit, not just a schema check.  It compares
exported hints against the producer-side state each hint claims to summarize:
membership, corridor assignments, yield metrics, traffic profiles, fee profiles,
member connectivity, and current direct channels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import competitive_fee_tournament as tournament  # noqa: E402


def _load_hive_hint_adapter():
    path = REPO_ROOT / "modules" / "hive_hints.py"
    spec = importlib.util.spec_from_file_location("hive_hints_standalone", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.HiveHintAdapter


HiveHintAdapter = _load_hive_hint_adapter()


VALID_ROLES = {"owner", "secondary", "contested", "none"}
VALID_REBALANCE_PREFS = {"source", "sink", "neutral"}
VALID_OPEN_PREFS = {"open", "neutral", "avoid"}


class _RpcStub:
    def __init__(self, snapshot: dict[str, Any]):
        self._snapshot = snapshot

    def call(self, method: str, *args, **kwargs) -> dict[str, Any]:
        if method != "hive-export-hints":
            raise RuntimeError(f"unexpected RPC call: {method}")
        return self._snapshot

    def listdatastore(self, *args, **kwargs) -> dict[str, Any]:
        return {"datastore": []}


class _PluginStub:
    def __init__(self, snapshot: dict[str, Any]):
        self.rpc = _RpcStub(snapshot)
        self.logs: list[tuple[str, str]] = []

    def log(self, message: str, level: str = "info") -> None:
        self.logs.append((level, message))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_ok(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("ok", True)) and "error" not in result


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _hints(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = snapshot.get("hints", {})
    if not isinstance(raw, dict):
        return {}
    return {str(peer_id): hint for peer_id, hint in raw.items() if isinstance(hint, dict)}


def _active_direct_peers(peer_channels: dict[str, Any]) -> set[str]:
    peers: set[str] = set()
    for channel in _as_list(peer_channels.get("channels")):
        if not isinstance(channel, dict):
            continue
        if channel.get("state") != "CHANNELD_NORMAL":
            continue
        peer_id = str(channel.get("peer_id") or "")
        if peer_id:
            peers.add(peer_id)
    return peers


def _member_ids(members_result: dict[str, Any]) -> set[str]:
    return {
        str(member.get("peer_id") or "")
        for member in _as_list(members_result.get("members"))
        if isinstance(member, dict) and member.get("peer_id")
    }


def _own_pubkey(status_result: dict[str, Any]) -> str:
    membership = status_result.get("membership", {}) if isinstance(status_result, dict) else {}
    return str(membership.get("pubkey") or "")


def _expected_corridor_roles(assignments_result: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for assignment in _as_list(assignments_result.get("assignments")):
        if not isinstance(assignment, dict):
            continue
        primary = str(assignment.get("primary_member") or "")
        if primary:
            roles[primary] = "contested" if roles.get(primary) == "secondary" else "owner"
        for secondary in _as_list(assignment.get("secondary_members")):
            peer_id = str(secondary or "")
            if not peer_id:
                continue
            roles[peer_id] = "contested" if roles.get(peer_id) == "owner" else "secondary"
    return roles


def _expected_competition_bias(assignments_result: dict[str, Any]) -> dict[str, int]:
    votes: dict[str, list[int]] = {}
    for assignment in _as_list(assignments_result.get("assignments")):
        if not isinstance(assignment, dict):
            continue
        corridor = assignment.get("corridor", {})
        level = str(corridor.get("competition_level") or "none") if isinstance(corridor, dict) else "none"
        if level in {"high", "medium"}:
            vote = -1
        elif level in {"low", "none"}:
            vote = 1
        else:
            vote = 0
        members = [assignment.get("primary_member"), *_as_list(assignment.get("secondary_members"))]
        for member in members:
            peer_id = str(member or "")
            if peer_id:
                votes.setdefault(peer_id, []).append(vote)

    biases: dict[str, int] = {}
    for peer_id, peer_votes in votes.items():
        score = sum(peer_votes)
        if score > 0:
            biases[peer_id] = 1
        elif score < 0:
            biases[peer_id] = -1
        else:
            biases[peer_id] = 0
    return biases


def _expected_rebalance_preferences(yield_result: dict[str, Any]) -> dict[str, str]:
    best: dict[str, tuple[int, str]] = {}
    for channel in _as_list(yield_result.get("channels")):
        if not isinstance(channel, dict):
            continue
        peer_id = str(channel.get("peer_id") or "")
        pref = str(channel.get("flow_direction") or "")
        if not peer_id or pref not in {"sink", "source"}:
            continue
        try:
            volume = int(channel.get("volume_routed_msat") or 0)
        except (TypeError, ValueError):
            volume = 0
        if peer_id not in best or volume > best[peer_id][0]:
            best[peer_id] = (volume, pref)
    return {peer_id: pref for peer_id, (_volume, pref) in best.items()}


def _fee_profiles_by_peer(fee_profiles_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(profile.get("peer_id") or ""): profile
        for profile in _as_list(fee_profiles_result.get("profiles"))
        if isinstance(profile, dict) and profile.get("peer_id")
    }


def _traffic_profiles_by_peer(traffic_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_peer: dict[str, list[dict[str, Any]]] = {}
    for profile in _as_list(traffic_result.get("profiles")):
        if not isinstance(profile, dict):
            continue
        peer_id = str(profile.get("peer_id") or "")
        if peer_id:
            by_peer.setdefault(peer_id, []).append(profile)
    return by_peer


def _adapter_effects(snapshot: dict[str, Any]) -> dict[str, Any]:
    adapter = HiveHintAdapter(_PluginStub(snapshot), ttl_override=0)
    adapter.poll()
    effects: dict[str, Any] = {
        "status": adapter.get_status(),
        "peers": {},
        "open_candidate_count": 0,
    }
    try:
        effects["open_candidate_count"] = len(adapter.get_open_candidates())
    except Exception:
        effects["open_candidate_count"] = 0

    for peer_id in _hints(snapshot):
        effects["peers"][peer_id] = {
            "fee_bias": adapter.get_fee_bias(peer_id),
            "rebalance_bias": adapter.get_rebalance_bias(peer_id),
            "corridor_role": adapter.get_corridor_role(peer_id),
            "traffic_confidence": adapter.get_traffic_confidence(peer_id),
            "optimal_fee_estimate_ppm": adapter.get_optimal_fee_estimate(peer_id),
            "is_hive_member": adapter.is_hive_member(peer_id),
            "channel_open_hint": adapter.get_channel_open_hint(peer_id),
        }
    return effects


def analyze_evidence(evidence: dict[str, Any], *, now: int | None = None) -> dict[str, Any]:
    now = int(now or time.time())
    snapshot = evidence.get("hive_export_hints", {})
    hints = _hints(snapshot) if isinstance(snapshot, dict) else {}
    status = evidence.get("hive_status", {})
    members_result = evidence.get("hive_members", {})
    peer_channels = evidence.get("listpeerchannels", {})
    corridor_result = evidence.get("hive_corridor_assignments", {})
    yield_result = evidence.get("hive_yield_metrics", {})
    traffic_result = evidence.get("hive_traffic_intelligence", {})
    fee_profiles_result = evidence.get("hive_fee_profiles", {})
    connectivity_result = evidence.get("hive_member_connectivity", {})

    issues: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    direct_peers = _active_direct_peers(peer_channels)
    member_ids = _member_ids(members_result)
    own_pubkey = _own_pubkey(status)
    expected_member_hints = member_ids - {own_pubkey} if own_pubkey else member_ids

    if not _is_ok(snapshot) or not hints:
        issues.append({
            "severity": "error",
            "code": "missing_exported_hints",
            "message": "hive-export-hints did not return a usable hints map",
        })

    generated_at = snapshot.get("generated_at") if isinstance(snapshot, dict) else None
    ttl_seconds = snapshot.get("ttl_seconds", 900) if isinstance(snapshot, dict) else 900
    if isinstance(generated_at, (int, float)):
        age_seconds = now - int(generated_at)
        if age_seconds > int(ttl_seconds or 900):
            issues.append({
                "severity": "error",
                "code": "stale_snapshot",
                "message": "hive-export-hints snapshot is older than ttl_seconds",
                "age_seconds": age_seconds,
                "ttl_seconds": ttl_seconds,
            })
    else:
        issues.append({
            "severity": "error",
            "code": "missing_generated_at",
            "message": "hive-export-hints snapshot has no numeric generated_at",
        })

    false_member_hints = sorted(
        peer_id for peer_id, hint in hints.items()
        if bool(hint.get("member")) and peer_id not in expected_member_hints
    )
    missing_member_hints = sorted(expected_member_hints - set(hints))
    if false_member_hints:
        issues.append({
            "severity": "error",
            "code": "false_member_hint",
            "message": "member=true was exported for peers not present in hive-members",
            "peers": false_member_hints,
        })
    if missing_member_hints:
        issues.append({
            "severity": "warning",
            "code": "missing_member_hint",
            "message": "hive members were not represented in exported hints",
            "peers": missing_member_hints,
        })

    expected_roles = _expected_corridor_roles(corridor_result)
    expected_biases = _expected_competition_bias(corridor_result)
    for peer_id, hint in hints.items():
        role = hint.get("corridor_role", "none")
        if role not in VALID_ROLES:
            issues.append({
                "severity": "error",
                "code": "invalid_corridor_role",
                "peer_id": peer_id,
                "observed": role,
            })
        expected_role = expected_roles.get(peer_id, "none")
        if role in VALID_ROLES and role != expected_role:
            issues.append({
                "severity": "error",
                "code": "corridor_role_mismatch",
                "peer_id": peer_id,
                "observed": role,
                "expected": expected_role,
            })

        observed_bias = hint.get("competition_bias", 0)
        if observed_bias not in (-1, 0, 1):
            issues.append({
                "severity": "error",
                "code": "invalid_competition_bias",
                "peer_id": peer_id,
                "observed": observed_bias,
            })
        expected_bias = expected_biases.get(peer_id, 0)
        if observed_bias in (-1, 0, 1) and observed_bias != expected_bias:
            issues.append({
                "severity": "error",
                "code": "competition_bias_mismatch",
                "peer_id": peer_id,
                "observed": observed_bias,
                "expected": expected_bias,
            })

    expected_rebalance = _expected_rebalance_preferences(yield_result)
    for peer_id, hint in hints.items():
        observed_pref = hint.get("rebalance_preference", "neutral")
        if observed_pref not in VALID_REBALANCE_PREFS:
            issues.append({
                "severity": "error",
                "code": "invalid_rebalance_preference",
                "peer_id": peer_id,
                "observed": observed_pref,
            })
        expected_pref = expected_rebalance.get(peer_id, "neutral")
        if observed_pref in VALID_REBALANCE_PREFS and observed_pref != expected_pref:
            issues.append({
                "severity": "error",
                "code": "rebalance_preference_mismatch",
                "peer_id": peer_id,
                "observed": observed_pref,
                "expected": expected_pref,
            })

    traffic_by_peer = _traffic_profiles_by_peer(traffic_result)
    defaulted_traffic = sorted(
        peer_id for peer_id, hint in hints.items()
        if peer_id not in traffic_by_peer and hint.get("traffic_confidence") in (0.2, 0.3, 0.5)
    )
    if defaulted_traffic:
        observations.append({
            "code": "traffic_confidence_defaulted",
            "message": "traffic_confidence is a producer fallback, not measured traffic evidence",
            "peers": defaulted_traffic,
        })

    profiles_by_peer = _fee_profiles_by_peer(fee_profiles_result)
    low_fee_profile_confidence = []
    zero_volume_fee_profiles = []
    for peer_id, hint in hints.items():
        profile = profiles_by_peer.get(peer_id)
        observed_optimal = hint.get("optimal_fee_estimate_ppm")
        if profile:
            expected_optimal = profile.get("optimal_fee_estimate")
            if isinstance(observed_optimal, (int, float)) and int(observed_optimal) != int(expected_optimal or 0):
                issues.append({
                    "severity": "error",
                    "code": "optimal_fee_mismatch",
                    "peer_id": peer_id,
                    "observed": int(observed_optimal),
                    "expected": int(expected_optimal or 0),
                })
            confidence = float(profile.get("confidence") or 0.0)
            if confidence < 0.5:
                low_fee_profile_confidence.append(peer_id)
            if int(profile.get("total_hive_volume") or 0) <= 0:
                zero_volume_fee_profiles.append(peer_id)
    if low_fee_profile_confidence:
        observations.append({
            "code": "low_fee_profile_confidence",
            "message": "optimal_fee_estimate exists but is based on low-confidence fee intelligence",
            "peers": sorted(low_fee_profile_confidence),
        })
    if zero_volume_fee_profiles:
        observations.append({
            "code": "zero_volume_fee_profiles",
            "message": "some optimal_fee_estimate values are based on fee reports with no routed volume",
            "peers": sorted(zero_volume_fee_profiles),
        })

    open_hints = []
    false_open_hints = []
    hive_topology_member_edges = []
    hive_topology_candidate_targets = set()
    for peer_id, hint in hints.items():
        topology = hint.get("fleet_topology", [])
        if isinstance(topology, list):
            for target in topology:
                target_id = str(target or "")
                if not target_id or target_id == peer_id or target_id not in member_ids:
                    continue
                hive_topology_member_edges.append({
                    "source_member": peer_id,
                    "target_member": target_id,
                })
                if target_id != own_pubkey and target_id not in direct_peers:
                    hive_topology_candidate_targets.add(target_id)

        ch_hint = hint.get("channel_open_hint")
        if not isinstance(ch_hint, dict):
            continue
        pref = ch_hint.get("open_preference")
        if pref not in VALID_OPEN_PREFS:
            issues.append({
                "severity": "error",
                "code": "invalid_open_preference",
                "peer_id": peer_id,
                "observed": pref,
            })
            continue
        if pref == "open":
            open_hints.append(peer_id)
            if ch_hint.get("reason") == "member_connectivity" and peer_id in direct_peers:
                false_open_hints.append(peer_id)

    if false_open_hints:
        issues.append({
            "severity": "error",
            "code": "open_hint_existing_member_channel",
            "message": "member_connectivity open hints target peers that already have CHANNELD_NORMAL direct channels",
            "peers": sorted(false_open_hints),
        })

    connectivity_recs = [
        str(rec.get("member_id") or "")
        for rec in _as_list(connectivity_result.get("recommended_connections"))
        if isinstance(rec, dict) and rec.get("member_id")
    ]
    bad_connectivity_recs = sorted(peer_id for peer_id in connectivity_recs if peer_id in direct_peers)
    if bad_connectivity_recs:
        issues.append({
            "severity": "error",
            "code": "connectivity_report_ignores_direct_channels",
            "message": "hive-member-connectivity recommends peers that are already direct CHANNELD_NORMAL peers",
            "peers": bad_connectivity_recs,
        })

    if member_ids and len(member_ids) > 1 and not hive_topology_member_edges:
        observations.append({
            "code": "missing_hive_member_topology_edges",
            "message": "exported fleet_topology does not contain hive-member edges, so second-hop hive mesh discovery has no producer evidence",
        })

    effects = _adapter_effects(snapshot if isinstance(snapshot, dict) else {})
    non_neutral_fee_bias = sorted(
        peer_id for peer_id, effect in effects.get("peers", {}).items()
        if abs(float(effect.get("fee_bias", 1.0)) - 1.0) > 1e-9
    )
    non_neutral_rebalance_bias = sorted(
        peer_id for peer_id, effect in effects.get("peers", {}).items()
        if abs(float(effect.get("rebalance_bias", 1.0)) - 1.0) > 1e-9
    )
    optimal_fee_peers = sorted(
        peer_id for peer_id, effect in effects.get("peers", {}).items()
        if int(effect.get("optimal_fee_estimate_ppm") or 0) > 0
    )

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    verdict = "fail" if error_count else ("warn" if warning_count or observations else "pass")

    return {
        "generated_at": now,
        "verdict": verdict,
        "issue_counts": {
            "errors": error_count,
            "warnings": warning_count,
            "observations": len(observations),
        },
        "summary": {
            "hint_count": len(hints),
            "member_count": len(member_ids),
            "expected_member_hint_count": len(expected_member_hints),
            "active_direct_peer_count": len(direct_peers),
            "corridor_assignment_count": len(_as_list(corridor_result.get("assignments"))),
            "yield_channel_count": len(_as_list(yield_result.get("channels"))),
            "traffic_profile_count": len(_as_list(traffic_result.get("profiles"))),
            "fee_profile_count": len(profiles_by_peer),
            "open_hint_count": len(open_hints),
            "false_open_hint_count": len(false_open_hints),
            "hive_topology_member_edge_count": len(hive_topology_member_edges),
            "hive_topology_candidate_target_count": len(hive_topology_candidate_targets),
        },
        "usefulness": {
            "member_hints": sorted(peer_id for peer_id, hint in hints.items() if hint.get("member") is True),
            "optimal_fee_peers": optimal_fee_peers,
            "non_neutral_fee_bias_peers": non_neutral_fee_bias,
            "non_neutral_rebalance_bias_peers": non_neutral_rebalance_bias,
            "open_hint_peers": sorted(open_hints),
            "true_open_hint_peers": sorted(set(open_hints) - set(false_open_hints)),
            "hive_topology_candidate_targets": sorted(hive_topology_candidate_targets),
            "adapter_open_candidate_count": effects.get("open_candidate_count", 0),
        },
        "issues": issues,
        "observations": observations,
        "adapter_effects": effects,
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    summary = analysis.get("summary", {})
    usefulness = analysis.get("usefulness", {})
    counts = analysis.get("issue_counts", {})
    lines = [
        "# Hive Hints Truth Audit",
        "",
        f"- Verdict: **{analysis.get('verdict', 'unknown').upper()}**",
        f"- Issues: {counts.get('errors', 0)} errors, {counts.get('warnings', 0)} warnings, {counts.get('observations', 0)} observations",
        f"- Hints: {summary.get('hint_count', 0)} for {summary.get('member_count', 0)} hive members",
        f"- Direct active peers: {summary.get('active_direct_peer_count', 0)}",
        f"- Corridor assignments: {summary.get('corridor_assignment_count', 0)}",
        f"- Yield channels: {summary.get('yield_channel_count', 0)}",
        f"- Traffic profiles: {summary.get('traffic_profile_count', 0)}",
        f"- Fee profiles: {summary.get('fee_profile_count', 0)}",
        f"- Open hints: {summary.get('open_hint_count', 0)} total, {summary.get('false_open_hint_count', 0)} false by direct-channel evidence",
        f"- Hive topology member edges: {summary.get('hive_topology_member_edge_count', 0)}",
        "",
        "## Usefulness",
        "",
        f"- Member hints usable by routing/rebalance modules: {len(usefulness.get('member_hints', []))}",
        f"- Optimal-fee estimates usable by fee controller: {len(usefulness.get('optimal_fee_peers', []))}",
        f"- Non-neutral fee-bias peers: {len(usefulness.get('non_neutral_fee_bias_peers', []))}",
        f"- Non-neutral rebalance-bias peers: {len(usefulness.get('non_neutral_rebalance_bias_peers', []))}",
        f"- True channel-open hint peers: {len(usefulness.get('true_open_hint_peers', []))}",
        f"- Second-hop hive topology targets: {len(usefulness.get('hive_topology_candidate_targets', []))}",
        "",
        "## Issues",
        "",
    ]
    if analysis.get("issues"):
        for issue in analysis["issues"]:
            peers = issue.get("peers")
            peer_text = f" peers={', '.join(peers)}" if isinstance(peers, list) and peers else ""
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}`: {issue.get('message', '')}{peer_text}")
    else:
        lines.append("- No truth mismatches found.")

    lines.extend(["", "## Observations", ""])
    if analysis.get("observations"):
        for obs in analysis["observations"]:
            peers = obs.get("peers")
            peer_text = f" peers={', '.join(peers)}" if isinstance(peers, list) and peers else ""
            lines.append(f"- `{obs.get('code')}`: {obs.get('message', '')}{peer_text}")
    else:
        lines.append("- No low-confidence observations.")
    lines.append("")
    return "\n".join(lines)


def collect_evidence(node: str, *, ensure_hive: bool = False) -> dict[str, Any]:
    setup: dict[str, Any] | None = None
    if ensure_hive:
        setup = tournament.ensure_cl_hive(node=node)

    status = tournament.cln(node, "hive-status")
    own = _own_pubkey(status)
    connectivity = (
        tournament.cln(node, "hive-member-connectivity", own)
        if own else
        {"error": "own pubkey unavailable"}
    )

    evidence = {
        "collected_at": int(time.time()),
        "node": node,
        "setup": setup,
        "hive_status": status,
        "hive_members": tournament.cln(node, "hive-members"),
        "hive_export_hints": tournament.cln(node, "hive-export-hints"),
        "hive_correlation_note": "All evidence collected from the same Polar node; timestamps may differ by RPC latency.",
        "hive_corridor_assignments": tournament.cln(node, "hive-corridor-assignments"),
        "hive_yield_metrics": tournament.cln(node, "hive-yield-metrics"),
        "hive_traffic_intelligence": tournament.cln(node, "hive-traffic-intelligence"),
        "hive_fee_profiles": tournament.cln(node, "hive-fee-profiles"),
        "hive_expansion_recommendations": tournament.cln(node, "hive-expansion-recommendations", "limit=20"),
        "hive_network_metrics": tournament.cln(node, "hive-network-metrics"),
        "hive_member_connectivity": connectivity,
        "listpeerchannels": tournament.cln(node, "listpeerchannels"),
        "datastore_hints": tournament.cln(node, "listdatastore", '["hive","hints"]'),
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default=tournament.REVENUE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ensure-hive", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        stamp = time.strftime("%Y%m%dT%H%M%S%z")
        out_dir = REPO_ROOT / "results" / f"hive-hints-truth-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence = collect_evidence(args.node, ensure_hive=args.ensure_hive)
    analysis = analyze_evidence(evidence)

    _write_json(out_dir / "evidence.json", evidence)
    _write_json(out_dir / "analysis.json", analysis)
    (out_dir / "ANALYSIS.md").write_text(render_markdown(analysis), encoding="utf-8")

    print(json.dumps({
        "ok": analysis.get("verdict") != "fail",
        "verdict": analysis.get("verdict"),
        "out_dir": str(out_dir),
        "issue_counts": analysis.get("issue_counts"),
        "summary": analysis.get("summary"),
    }, indent=2, sort_keys=True))
    return 1 if analysis.get("verdict") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
