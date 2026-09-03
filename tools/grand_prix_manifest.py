#!/usr/bin/env python3
"""Validate private calibration data and build the Grand Prix manifest.

The tool is offline and side-effect free except for writing an explicitly
requested output file.  It has no production, Polar, Docker, CLN, or payment
client.  Raw production identifiers and records are forbidden by schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
from typing import Any


CALIBRATION_SCHEMA = "polar-grand-prix-calibration-v1"
TOPOLOGY_SCHEMA = "polar-grand-prix-topology-v1"
MAX_BYTES = 128 * 1024
CAPACITY_BUCKETS = ("lt_2m", "2m_5m", "5m_10m", "10m_20m", "gte_20m")
BALANCE_BUCKETS = ("0_10pct", "10_30pct", "30_70pct", "70_90pct", "90_100pct")
FEE_BUCKETS = ("0_49", "50_99", "100_249", "250_499", "gte_500")
AGE_BUCKETS = ("lt_4320", "4320_12960", "12960_52560", "gte_52560")
SIZE_BUCKETS = ("lt_5k", "5k_20k", "20k_100k", "100k_500k", "gte_500k")
ARRIVAL_BUCKETS = ("lt_1", "1_10", "10_60", "60_300", "gte_300")
DAY_BUCKETS = ("00_06", "06_12", "12_18", "18_24")
ROLE_BUCKETS = ("inbound_dominant", "balanced", "outbound_dominant")
FORBIDDEN_TOKENS = (
    "node_id", "peer_id", "short_channel_id", "channel_id", "payment_hash",
    "invoice", "raw_forward", "exact_balance", "funding_txid",
)


class ManifestError(ValueError):
    """A calibration or topology manifest is unsafe or internally inconsistent."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{label} must be an integer >= {minimum}")
    return value


def _histogram(value: Any, keys: tuple[str, ...], label: str) -> dict[str, int]:
    row = _object(value, label)
    if set(row) != set(keys):
        raise ManifestError(f"{label} has unexpected or missing buckets")
    return {key: _integer(row[key], f"{label}.{key}") for key in keys}


def _assert_private(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in FORBIDDEN_TOKENS):
                raise ManifestError(f"forbidden production field at {path}.{key}")
            _assert_private(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_private(item, f"{path}[{index}]")


def validate_calibration(payload: Any) -> dict[str, Any]:
    root = _object(payload, "calibration")
    if root.get("schema") != CALIBRATION_SCHEMA:
        raise ManifestError("unsupported calibration schema")
    _assert_private(root)
    if root.get("source") != "production_read_only_remote_aggregation":
        raise ManifestError("calibration source must be read-only remote aggregation")
    privacy = _object(root.get("privacy"), "privacy")
    if privacy != {
        "aggregated_on_production_host": True,
        "raw_identifiers_exported": False,
        "raw_payment_records_exported": False,
    }:
        raise ManifestError("privacy proof is incomplete")
    date = root.get("observed_date_utc")
    if not isinstance(date, str) or len(date) != 10:
        raise ManifestError("observed_date_utc must be YYYY-MM-DD")

    channels = _object(root.get("channels"), "channels")
    count = _integer(channels.get("channel_count"), "channel_count", minimum=1)
    active = _integer(channels.get("active_count"), "active_count")
    if active > count:
        raise ManifestError("active_count exceeds channel_count")
    capacity = _histogram(channels.get("capacity_histogram_sats"), CAPACITY_BUCKETS, "capacity_histogram_sats")
    balance = _histogram(channels.get("local_balance_ratio_distribution"), BALANCE_BUCKETS, "local_balance_ratio_distribution")
    fees = _histogram(channels.get("fee_distribution_ppm"), FEE_BUCKETS, "fee_distribution_ppm")
    age = _histogram(channels.get("channel_age_blocks"), AGE_BUCKETS, "channel_age_blocks")
    if any(sum(hist.values()) != count for hist in (capacity, balance, fees)):
        raise ManifestError("channel histograms must reconcile to channel_count")
    if sum(age.values()) > count:
        raise ManifestError("channel age coverage exceeds channel_count")
    degree = _object(channels.get("peer_degree"), "peer_degree")
    observed_peers = _integer(degree.get("observed_peer_count"), "observed_peer_count", minimum=1)
    degree_hist = _histogram(
        degree.get("buckets"), ("lt_10", "10_49", "50_199", "200_499", "gte_500"),
        "peer_degree.buckets",
    )
    if sum(degree_hist.values()) != observed_peers:
        raise ManifestError("peer degree buckets do not reconcile")

    traffic = _object(root.get("traffic_30d"), "traffic_30d")
    if traffic.get("window_days") != 30:
        raise ManifestError("traffic calibration must use a 30-day window")
    settled = _integer(traffic.get("settled_count"), "settled_count", minimum=1)
    _integer(traffic.get("failed_count"), "failed_count")
    sizes = _histogram(traffic.get("forward_size_distribution_sats"), SIZE_BUCKETS, "forward_size_distribution_sats")
    arrivals = _histogram(traffic.get("interarrival_distribution_seconds"), ARRIVAL_BUCKETS, "interarrival_distribution_seconds")
    daytime = _histogram(traffic.get("time_of_day_distribution_utc"), DAY_BUCKETS, "time_of_day_distribution_utc")
    _histogram(traffic.get("forward_direction_distribution"), ROLE_BUCKETS, "forward_direction_distribution")
    if sum(sizes.values()) != settled or sum(daytime.values()) != settled:
        raise ManifestError("traffic histograms must reconcile to settled_count")
    if sum(arrivals.values()) != max(0, settled - 1):
        raise ManifestError("interarrival histogram must contain settled_count - 1 intervals")
    concentration = _object(traffic.get("revenue_concentration_pct"), "revenue_concentration_pct")
    top5 = float(concentration.get("top_5", -1))
    top10 = float(concentration.get("top_10", -1))
    if not (0 <= top5 <= top10 <= 100):
        raise ManifestError("revenue concentration is invalid")
    missing = _object(root.get("unavailable"), "unavailable")
    if set(missing) != {"route_length_distribution", "failure_code_distribution", "rebalance_cost_distribution"}:
        raise ManifestError("unavailable evidence declarations are incomplete")
    if any(not isinstance(reason, str) or not reason for reason in missing.values()):
        raise ManifestError("unavailable evidence requires explicit reasons")
    return {
        "schema": CALIBRATION_SCHEMA,
        "valid": True,
        "observed_date_utc": date,
        "channel_count": count,
        "active_count": active,
        "settled_forwards_30d": settled,
    }


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _node(name: str, implementation: str, role: str) -> dict[str, str]:
    return {"name": name, "implementation": implementation, "role": role}


def build_topology(calibration: dict[str, Any], *, public_seed: int) -> dict[str, Any]:
    validate_calibration(calibration)
    if isinstance(public_seed, bool) or not isinstance(public_seed, int) or public_seed <= 0:
        raise ManifestError("public_seed must be a positive integer")
    nodes = [
        _node("identity-a", "cln", "contender"),
        _node("identity-b", "cln", "contender"),
        _node("cln-payer", "cln", "payer"),
        _node("lnd-payer", "lnd", "payer"),
        _node("cln-sink", "cln", "sink"),
        _node("lnd-sink", "lnd", "sink"),
    ]
    nodes += [_node(f"hub-{index}", "lnd" if index in {5, 6} else "cln", "hub") for index in range(1, 7)]
    nodes += [_node(f"lateral-{index}", "lnd" if index in {5, 6} else "cln", "competitive_lateral") for index in range(1, 7)]
    nodes += [_node(f"specialist-{index}", "lnd" if index == 4 else "cln", "specialist") for index in range(1, 5)]
    nodes += [_node("edge-1", "cln", "edge"), _node("edge-2", "cln", "edge")]

    # Preserve the production-shaped capacity multiset and 130M sats per
    # contender, but put useful burst capacity on the four traffic-facing
    # payer/sink edges. Assigning the four smallest channels there made the
    # public workload test endpoint bottlenecks rather than routing policy.
    capacities = [8_000_000, 10_000_000, 12_000_000, 15_000_000, 1_000_000,
                  1_500_000, 2_000_000, 2_500_000, 3_000_000, 3_500_000,
                  5_000_000, 5_000_000, 7_500_000, 25_000_000, 20_000_000, 9_000_000]
    peers = ["cln-payer", "lnd-payer", "cln-sink", "lnd-sink"]
    peers += [f"hub-{index}" for index in range(1, 5)]
    peers += [f"lateral-{index}" for index in range(1, 5)]
    peers += [f"specialist-{index}" for index in range(1, 5)]
    edges: list[dict[str, Any]] = []
    for contender in ("identity-a", "identity-b"):
        for index, (peer, capacity) in enumerate(zip(peers, capacities, strict=True)):
            payer_side = peer.endswith("payer")
            source, destination = (peer, contender) if payer_side else (contender, peer)
            edges.append({
                "source": source,
                "destination": destination,
                "capacity_sats": capacity,
                "fee_ppm": (40, 75, 150, 300, 600)[index % 5],
                "initial_source_ratio": (0.9, 0.7, 0.5, 0.3, 0.1)[index % 5],
                "contender_lane": contender,
            })
    background_pairs = [
        ("cln-payer", "hub-5"), ("lnd-payer", "hub-6"),
        ("hub-5", "lateral-5"), ("hub-6", "lateral-6"),
        ("lateral-5", "cln-sink"), ("lateral-6", "lnd-sink"),
        ("cln-payer", "specialist-1"), ("lnd-payer", "specialist-2"),
        ("specialist-1", "hub-1"), ("specialist-2", "hub-2"),
        ("hub-1", "cln-sink"), ("hub-2", "lnd-sink"),
        ("hub-1", "hub-3"), ("hub-2", "hub-4"), ("hub-3", "hub-4"),
        ("hub-3", "lateral-1"), ("hub-4", "lateral-2"),
        ("lateral-1", "specialist-3"), ("lateral-2", "specialist-4"),
        ("specialist-3", "edge-1"), ("specialist-4", "edge-2"),
        ("edge-1", "cln-sink"), ("edge-2", "lnd-sink"),
        ("lateral-3", "lateral-5"), ("lateral-4", "lateral-6"),
    ]
    for index, (source, destination) in enumerate(background_pairs):
        edges.append({
            "source": source, "destination": destination,
            "capacity_sats": capacities[index % len(capacities)],
            "fee_ppm": (60, 120, 180, 360)[index % 4],
            "initial_source_ratio": (0.25, 0.5, 0.75)[index % 3],
            "contender_lane": None,
        })

    rng = random.Random(public_seed)
    size_values = (2_000, 10_000, 50_000, 250_000, 750_000)
    size_weights = tuple(calibration["traffic_30d"]["forward_size_distribution_sats"][key] for key in SIZE_BUCKETS)
    classes = (
        ("baseline_retail", 35), ("merchant_directional", 25),
        ("exchange_burst", 20), ("competitive_displacement", 15),
        ("shock_fault", 5),
    )
    traffic = []
    flow_roles = calibration["traffic_30d"]["forward_direction_distribution"]
    forward_weight = flow_roles["outbound_dominant"] + flow_roles["balanced"] / 2
    reverse_weight = flow_roles["inbound_dominant"] + flow_roles["balanced"] / 2
    for sequence in range(240):
        class_name = rng.choices([row[0] for row in classes], [row[1] for row in classes], k=1)[0]
        direction = rng.choices(
            ("forward", "reverse"), (forward_weight, reverse_weight), k=1
        )[0]
        if direction == "forward":
            payer = rng.choice(("cln-payer", "lnd-payer"))
            sink = rng.choice(("cln-sink", "lnd-sink"))
        else:
            payer = rng.choice(("cln-sink", "lnd-sink"))
            sink = rng.choice(("cln-payer", "lnd-payer"))
        traffic.append({
            "sequence": sequence,
            "class": class_name,
            "direction": direction,
            "payer": payer,
            "sink": sink,
            "amount_sats": rng.choices(size_values, size_weights, k=1)[0],
            "interarrival_bucket": rng.choices(
                ARRIVAL_BUCKETS,
                [calibration["traffic_30d"]["interarrival_distribution_seconds"][key] for key in ARRIVAL_BUCKETS],
                k=1,
            )[0],
        })
    result = {
        "schema": TOPOLOGY_SCHEMA,
        "calibration_digest": _canonical_digest(calibration),
        "public_seed": public_seed,
        "holdout": {"seed_present": False, "commitment_required": True},
        "nodes": nodes,
        "channels": edges,
        "traffic": traffic,
        "controller_assignments": [
            {"replica_parity": "odd", "revenue_ops": "identity-a", "clboss": "identity-b"},
            {"replica_parity": "even", "revenue_ops": "identity-b", "clboss": "identity-a"},
        ],
    }
    validate_topology(result)
    return result


def _reachable(adjacency: dict[str, set[str]], source: str, sink: str, excluded: set[str]) -> bool:
    queue = [source]
    seen = set(excluded)
    while queue:
        node = queue.pop()
        if node == sink:
            return True
        if node in seen:
            continue
        seen.add(node)
        queue.extend(adjacency.get(node, set()) - seen)
    return False


def validate_topology(payload: Any) -> dict[str, Any]:
    root = _object(payload, "topology")
    if root.get("schema") != TOPOLOGY_SCHEMA:
        raise ManifestError("unsupported topology schema")
    nodes = root.get("nodes")
    channels = root.get("channels")
    traffic = root.get("traffic")
    if not isinstance(nodes, list) or len(nodes) != 24:
        raise ManifestError("topology must contain exactly 24 lightning nodes")
    names = {row.get("name") for row in nodes if isinstance(row, dict)}
    if len(names) != 24 or None in names:
        raise ManifestError("topology node names must be unique")
    if {row.get("implementation") for row in nodes} != {"cln", "lnd"}:
        raise ManifestError("topology must contain CLN and LND nodes")
    if not isinstance(channels, list) or not channels:
        raise ManifestError("topology channels are missing")
    adjacency = {name: set() for name in names}
    contender_vectors: dict[str, list[tuple[int, int, float]]] = {"identity-a": [], "identity-b": []}
    seen_edges: set[frozenset[str]] = set()
    for index, edge in enumerate(channels):
        row = _object(edge, f"channels[{index}]")
        source, destination = row.get("source"), row.get("destination")
        if source not in names or destination not in names or source == destination:
            raise ManifestError(f"channels[{index}] has invalid endpoints")
        key = frozenset((source, destination))
        if key in seen_edges:
            raise ManifestError("parallel or duplicate topology edge is forbidden")
        seen_edges.add(key)
        adjacency[source].add(destination)
        adjacency[destination].add(source)
        capacity = _integer(row.get("capacity_sats"), f"channels[{index}].capacity_sats", minimum=500_000)
        fee = _integer(row.get("fee_ppm"), f"channels[{index}].fee_ppm")
        ratio = row.get("initial_source_ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < ratio < 1:
            raise ManifestError(f"channels[{index}].initial_source_ratio is invalid")
        lane = row.get("contender_lane")
        if lane is not None:
            if lane not in contender_vectors or lane not in {source, destination}:
                raise ManifestError(f"channels[{index}] has invalid contender lane")
            contender_vectors[lane].append((capacity, fee, float(ratio)))
    if sorted(contender_vectors["identity-a"]) != sorted(contender_vectors["identity-b"]):
        raise ManifestError("contender channel portfolios are not matched")
    for payer in ("cln-payer", "lnd-payer"):
        for sink in ("cln-sink", "lnd-sink"):
            if not _reachable(adjacency, payer, sink, set()):
                raise ManifestError(f"no route for {payer}->{sink}")
            for contender in ("identity-a", "identity-b"):
                if not _reachable(adjacency, payer, sink, {contender}):
                    raise ManifestError(f"no alternative route when {contender} is removed")
    if root.get("holdout") != {"seed_present": False, "commitment_required": True}:
        raise ManifestError("holdout seed must not be present in the public topology")
    if not isinstance(traffic, list) or len(traffic) != 240:
        raise ManifestError("public traffic schedule must contain 240 payments")
    forward_sources = {"cln-payer", "lnd-payer"}
    forward_sinks = {"cln-sink", "lnd-sink"}
    for row in traffic:
        if not isinstance(row, dict):
            raise ManifestError("traffic rows must be objects")
        direction = row.get("direction")
        if direction == "forward":
            valid = row.get("payer") in forward_sources and row.get("sink") in forward_sinks
        elif direction == "reverse":
            valid = row.get("payer") in forward_sinks and row.get("sink") in forward_sources
        else:
            valid = False
        if not valid:
            raise ManifestError("traffic endpoints are invalid")
    assignments = root.get("controller_assignments")
    if assignments != [
        {"replica_parity": "odd", "revenue_ops": "identity-a", "clboss": "identity-b"},
        {"replica_parity": "even", "revenue_ops": "identity-b", "clboss": "identity-a"},
    ]:
        raise ManifestError("controller identity crossover is incomplete")
    return {
        "schema": TOPOLOGY_SCHEMA,
        "valid": True,
        "nodes": len(nodes),
        "channels": len(channels),
        "traffic_payments": len(traffic),
        "matched_contender_channels": len(contender_vectors["identity-a"]),
    }


def _load(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ManifestError("input must be a regular file")
        raw = os.read(descriptor, MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_BYTES:
        raise ManifestError("input exceeds size limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("input is not valid JSON") from exc
    return _object(value, "input")


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-calibration")
    validate.add_argument("calibration", type=Path)
    build = sub.add_parser("build-topology")
    build.add_argument("calibration", type=Path)
    build.add_argument("--public-seed", type=int, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate-topology")
    check.add_argument("topology", type=Path)
    args = parser.parse_args(arguments)
    try:
        if args.command == "validate-calibration":
            result = validate_calibration(_load(args.calibration))
        elif args.command == "build-topology":
            result = build_topology(_load(args.calibration), public_seed=args.public_seed)
            _write(args.output, result)
            result = validate_topology(result)
        else:
            result = validate_topology(_load(args.topology))
    except (OSError, ManifestError, TypeError, ValueError) as exc:
        sys.stderr.write(f"grand prix manifest error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
