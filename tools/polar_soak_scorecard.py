#!/usr/bin/env python3
"""Capture and score mixed-client traffic soaks in an existing Polar lab.

Snapshots are read-only Docker RPC queries. Traffic itself remains owned by
``polar_mixed_client_lab.py`` and therefore flows through Polar's MCP bridge.
The score command compares per-family before/after router counters and rejects
one-client, one-direction, one-size, ambiguous, or unsafe-final-state results.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


NETWORK_ID_RE = re.compile(r"^[1-9][0-9]*$")
ROUTERS = {
    "revenue-node": "cln",
    "cln-competitor": "cln",
    "lnd-competitor": "lnd",
}


class ScorecardError(RuntimeError):
    """Snapshot or scorecard evidence is incomplete or malformed."""


def _network_id(value: int | str) -> int:
    rendered = str(value)
    if not NETWORK_ID_RE.fullmatch(rendered):
        raise ScorecardError("network id must be a positive integer")
    return int(rendered)


def _msat(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ScorecardError(f"invalid msat value: {value!r}")
    rendered = str(value).strip()
    if rendered.endswith("msat"):
        rendered = rendered[:-4]
    try:
        parsed = int(rendered)
    except (TypeError, ValueError) as exc:
        raise ScorecardError(f"invalid msat value: {value!r}") from exc
    if parsed < 0:
        raise ScorecardError(f"negative msat value: {value!r}")
    return parsed


def _docker_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ScorecardError(f"non-JSON command output: {command!r}") from exc
    if not isinstance(payload, dict):
        raise ScorecardError(f"non-object command output: {command!r}")
    return payload


def aggregate_cln_forwards(payload: dict[str, Any]) -> dict[str, int]:
    forwards = payload.get("forwards")
    if not isinstance(forwards, list):
        raise ScorecardError("CLN listforwards response lacks a forward list")
    settled = [
        row
        for row in forwards
        if isinstance(row, dict) and row.get("status") == "settled"
    ]
    return {
        "settled_count": len(settled),
        "fee_msat": sum(_msat(row.get("fee_msat")) for row in settled),
        "volume_msat": sum(_msat(row.get("out_msat")) for row in settled),
    }


def aggregate_lnd_forwards(payload: dict[str, Any]) -> dict[str, int]:
    forwards = payload.get("forwarding_events")
    if not isinstance(forwards, list) or any(
        not isinstance(row, dict) for row in forwards
    ):
        raise ScorecardError("LND fwdinghistory response is malformed")
    return {
        "settled_count": len(forwards),
        "fee_msat": sum(_msat(row.get("fee_msat")) for row in forwards),
        "volume_msat": sum(_msat(row.get("amt_out_msat")) for row in forwards),
    }


def capture_snapshot(network_id: int) -> dict[str, Any]:
    network_id = _network_id(network_id)
    router_totals: dict[str, dict[str, int]] = {}
    for role, implementation in ROUTERS.items():
        container = f"polar-n{network_id}-{role}"
        if implementation == "cln":
            payload = _docker_json(
                [
                    "docker", "exec", "-u", "clightning", container,
                    "lightning-cli", "--network=regtest", "listforwards",
                ]
            )
            router_totals[role] = aggregate_cln_forwards(payload)
        else:
            payload = _docker_json(
                [
                    "docker", "exec", "-u", "lnd", container,
                    "lncli", "--network=regtest", "fwdinghistory",
                    "--start_time", "0", "--max_events", "50000",
                    "--skip_peer_alias_lookup",
                ]
            )
            router_totals[role] = aggregate_lnd_forwards(payload)

    revenue = f"polar-n{network_id}-revenue-node"
    cln = [
        "docker", "exec", "-u", "clightning", revenue,
        "lightning-cli", "--network=regtest",
    ]
    module_health = {
        "status": _docker_json(cln + ["revenue-status"]),
        "config": _docker_json(cln + ["revenue-config", "get"]),
        "fee_debug": _docker_json(cln + ["revenue-fee-debug"]),
        "rebalance_debug": _docker_json(cln + ["revenue-rebalance-debug"]),
        "profitability": _docker_json(cln + ["revenue-profitability"]),
        "budget": _docker_json(cln + ["revenue-budget", "ledger"]),
        "econ_reconcile": _docker_json(cln + ["revenue-econ-reconcile"]),
    }
    return {
        "schema": "polar-soak-snapshot-v1",
        "captured_at": int(time.time()),
        "network_id": network_id,
        "routers": router_totals,
        "module_health": module_health,
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScorecardError(f"could not read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScorecardError(f"JSON evidence must be an object: {path}")
    return value


def _router_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for router in ROUTERS:
        old = before.get("routers", {}).get(router)
        new = after.get("routers", {}).get(router)
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise ScorecardError(f"missing router snapshot for {router}")
        values = {
            key: int(new.get(key, 0)) - int(old.get(key, 0))
            for key in ("settled_count", "fee_msat", "volume_msat")
        }
        if any(value < 0 for value in values.values()):
            raise ScorecardError(f"router counters regressed for {router}")
        delta[router] = values
    total = sum(row["settled_count"] for row in delta.values())
    for row in delta.values():
        row["share"] = (
            round(row["settled_count"] / total, 6) if total else 0.0
        )
    return delta


def _traffic_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        traffic = _load_object(path).get("traffic")
        if not isinstance(traffic, list) or any(
            not isinstance(row, dict) for row in traffic
        ):
            raise ScorecardError(f"traffic evidence is malformed: {path}")
        records.extend(traffic)
    return records


def score_phase(
    family: str,
    before: dict[str, Any],
    after: dict[str, Any],
    traffic: list[dict[str, Any]],
    *,
    min_per_direction: int,
    min_distinct_amounts: int,
) -> dict[str, Any]:
    if family not in {"lnd", "cln"}:
        raise ScorecardError(f"unsupported client family: {family}")
    successes = [
        row
        for row in traffic
        if isinstance(row.get("payment"), dict)
        and row["payment"].get("success") is True
    ]
    unknown = [row for row in traffic if row.get("payment_outcome")]
    wrong_family = [
        row for row in traffic if not str(row.get("payer", "")).startswith(f"{family}-")
    ]
    directions = {
        "forward": sum("-payer" in str(row.get("payer", "")) for row in successes),
        "reverse": sum("-sink" in str(row.get("payer", "")) for row in successes),
    }
    amounts = sorted({int(row.get("amount_sats", 0) or 0) for row in successes})
    router_delta = _router_delta(before, after)
    router_forwards = sum(row["settled_count"] for row in router_delta.values())
    checks = {
        "all_payments_settled": len(successes) == len(traffic) and not unknown,
        "single_expected_client_family": not wrong_family and bool(traffic),
        "forward_coverage": directions["forward"] >= min_per_direction,
        "reverse_coverage": directions["reverse"] >= min_per_direction,
        "amount_diversity": len(amounts) >= min_distinct_amounts,
        "router_attribution_complete": router_forwards >= len(successes),
    }
    return {
        "family": family,
        "attempted": len(traffic),
        "settled": len(successes),
        "unknown": len(unknown),
        "directions": directions,
        "amounts_sats": amounts,
        "router_delta": router_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def final_module_checks(snapshot: dict[str, Any]) -> dict[str, bool]:
    health = snapshot.get("module_health", {})
    config = health.get("config", {}).get("config", {})
    reconcile = health.get("econ_reconcile", {})
    budget = health.get("budget", {})
    return {
        "plugin_running": health.get("status", {}).get("status") == "running",
        "paused": config.get("paused") is True,
        "daily_budget_zero": config.get("daily_budget_sats") == 0,
        "zero_active_reservations": int(budget.get("reserved_24h_sats", 0) or 0) == 0,
        "econ_reconcile_clean": reconcile.get("divergences") == [],
        "fee_module_readable": "error" not in health.get("fee_debug", {}),
        "rebalance_module_readable": "error" not in health.get("rebalance_debug", {}),
        "profitability_module_readable": "error" not in health.get("profitability", {}),
    }


def parse_phase(value: str) -> tuple[str, Path, Path, list[Path]]:
    parts = value.split(",")
    if len(parts) < 4:
        raise argparse.ArgumentTypeError(
            "phase must be FAMILY,BEFORE,AFTER,TRAFFIC[,TRAFFIC...]"
        )
    return parts[0], Path(parts[1]), Path(parts[2]), [Path(p) for p in parts[3:]]


def emit(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--network-id", type=int, required=True)
    snapshot_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--phase", action="append", type=parse_phase, required=True)
    score_parser.add_argument("--min-per-direction", type=int, default=5)
    score_parser.add_argument("--min-distinct-amounts", type=int, default=3)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "snapshot":
        payload = capture_snapshot(args.network_id)
        emit(payload, args.output)
        return 0

    phases = []
    final_snapshot = None
    for family, before_path, after_path, traffic_paths in args.phase:
        before = _load_object(before_path)
        after = _load_object(after_path)
        final_snapshot = after
        phases.append(
            score_phase(
                family,
                before,
                after,
                _traffic_records(traffic_paths),
                min_per_direction=max(1, args.min_per_direction),
                min_distinct_amounts=max(1, args.min_distinct_amounts),
            )
        )
    module_checks = final_module_checks(final_snapshot or {})
    families = {phase["family"] for phase in phases if phase["passed"]}
    payload = {
        "schema": "polar-soak-scorecard-v1",
        "generated_at": int(time.time()),
        "phases": phases,
        "module_checks": module_checks,
        "both_client_families_passed": families == {"lnd", "cln"},
    }
    payload["passed"] = (
        payload["both_client_families_passed"]
        and all(phase["passed"] for phase in phases)
        and all(module_checks.values())
    )
    emit(payload, args.output)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
