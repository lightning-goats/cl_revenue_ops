#!/usr/bin/env python3
"""Run rebalance/capex validation loops in Polar.

The loop can run in standalone mode or with cl-hive enabled. It optionally
drives forwarding traffic through the revenue node to create imbalance, runs
one automatic rebalance cycle, and records whether the rebalancer, hive hints,
and capex engine agree on spend:

- candidate quality: depleted destinations, source inventory, selected pairs
- execution quality: successes, failures, route/cooldown reasons
- accounting quality: budget reservations clear and successful fees hit the
  rebalance cost ledger used by capex budgets
- hive quality: fresh/member hints, hive-selected pairs, and intrahive success
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import competitive_fee_tournament as tournament


REVENUE = tournament.REVENUE
DEFAULT_PLUGIN_PATH = "/tmp/cl_revenue_ops/cl-revenue-ops.py"
DEFAULT_CONTAINER_DIR = "/tmp/cl_revenue_ops"
STALE_HIVE_LAYERS = (
    "hive-fleet",
    "hive-observed-liquidity",
    "hive-traffic",
    "hive-corridors",
    "hive-reputation",
)
DEFAULT_INTRAHIVE_SOURCE_NODE = "sink-a"
DEFAULT_INTRAHIVE_RELAY_NODE = "competitor-a"
DEFAULT_INTRAHIVE_DEST_NODE = "payer-a"
PURE_HIVE_TOPOLOGIES = ("none", "triangle", "square")


@dataclass
class IterationAnalysis:
    iteration: int
    valid: bool
    action: str
    reason: str
    traffic_ok: bool
    payment_timeouts: int
    selected_pairs: int
    executions: int
    successes: int
    failures: int
    fee_sats: int
    budget_spent_delta_sats: int
    budget_reserved_delta_sats: int
    channel_budget_delta_sats: int
    reservation_leak: bool
    accounting_ok: bool
    hive_mode: str
    hive_ok: bool
    hive_hints_fresh: bool
    hive_hints_usable: bool
    hive_hints_count: int
    hive_member_hints_count: int
    hive_selected_pairs: int
    intra_hive_selected_pairs: int
    hive_executions: int
    hive_successes: int
    intra_hive_executions: int
    intra_hive_successes: int
    pure_hive_required: bool
    pure_hive_selected_pairs: int
    pure_hive_executions: int
    pure_hive_successes: int
    pure_hive_route_ok: bool
    hive_disabled_ok: bool
    convergence_perturb_ok: bool
    convergence_pair_selected: bool
    convergence_source_scid: str
    convergence_dest_scid: str
    convergence_source_target_ratio: float | None
    convergence_dest_target_ratio: float | None
    convergence_source_before_ratio: float | None
    convergence_dest_before_ratio: float | None
    convergence_source_after_ratio: float | None
    convergence_dest_after_ratio: float | None
    convergence_source_error: float | None
    convergence_dest_error: float | None
    convergence_max_error: float | None
    convergence_restored_sats: int
    convergence_fee_per_restored_sat: float | None
    convergence_ok: bool


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_msat(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value.rstrip("msat").strip() or "0")
    return int(value)


def _node_id(node: str) -> str:
    return str(tournament.cln(node, "getinfo").get("id") or "")


def _peer_channel(node: str, peer_id: str) -> dict[str, Any]:
    channels = tournament.cln(node, "listpeerchannels")
    for channel in channels.get("channels", []) if isinstance(channels, dict) else []:
        if (
            isinstance(channel, dict)
            and str(channel.get("peer_id") or "") == peer_id
            and channel.get("state") == "CHANNELD_NORMAL"
        ):
            return channel
    return {}


def _channel_local_sats(channel: dict[str, Any]) -> int:
    return _parse_msat(channel.get("to_us_msat", channel.get("our_amount_msat", 0))) // 1000


def _channel_capacity_sats(channel: dict[str, Any]) -> int:
    return _parse_msat(channel.get("total_msat", 0)) // 1000


def _channel_summary(node: str, peer_id: str) -> dict[str, Any]:
    channel = _peer_channel(node, peer_id)
    capacity = _channel_capacity_sats(channel)
    local = _channel_local_sats(channel)
    return {
        "node": node,
        "peer_id": peer_id,
        "short_channel_id": str(channel.get("short_channel_id") or ""),
        "local_sats": local,
        "capacity_sats": capacity,
        "local_ratio": round(local / capacity, 6) if capacity else 0.0,
    }


def _pure_hive_path_nodes(
    topology: str,
    *,
    source_node: str,
    relay_node: str,
    dest_node: str,
) -> list[str]:
    if topology == "triangle":
        return [source_node, relay_node]
    if topology == "square":
        return [source_node, relay_node, dest_node]
    return []


def _amount_to_target_local(channel: dict[str, Any], target_ratio: float) -> int:
    capacity = _channel_capacity_sats(channel)
    if capacity <= 0:
        return 0
    target = int(capacity * float(target_ratio))
    return max(0, target - _channel_local_sats(channel))


def _amount_to_reduce_local(channel: dict[str, Any], target_ratio: float) -> int:
    capacity = _channel_capacity_sats(channel)
    if capacity <= 0:
        return 0
    target = int(capacity * float(target_ratio))
    return max(0, _channel_local_sats(channel) - target)


def _pay_between(
    *,
    payer: str,
    payee: str,
    amount_sats: int,
    label_prefix: str,
) -> dict[str, Any]:
    if amount_sats <= 0:
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_at_or_above_target",
            "payer": payer,
            "payee": payee,
            "amount_sats": amount_sats,
        }
    label = f"{label_prefix}-{payer}-to-{payee}-{int(time.time() * 1000)}"
    invoice = tournament.cln(
        payee,
        "invoice",
        str(int(amount_sats) * 1000),
        label,
        f"{label_prefix} {payer} to {payee}",
    )
    bolt11 = str(invoice.get("bolt11") or "")
    payment_hash = str(invoice.get("payment_hash") or "")
    payment_secret = str(invoice.get("payment_secret") or "")
    payee_id = _node_id(payee)
    connected = (
        tournament.cln(payer, "connect", f"{payee_id}@{payee}:9735")
        if payee_id else
        {"ok": False, "reason": "payee_id_missing"}
    )
    route_result = (
        tournament.cln(
            payer,
            "getroute",
            f"id={payee_id}",
            f"amount_msat={int(amount_sats) * 1000}",
            "riskfactor=1",
            "maxhops=1",
        )
        if bolt11 and payment_hash and payee_id else
        {"ok": False, "reason": "direct_route_inputs_missing"}
    )
    route = route_result.get("route") if isinstance(route_result, dict) else None
    if not (isinstance(route, list) and route) and payee_id:
        direct_channel = _peer_channel(payer, payee_id)
        direct_scid = str(direct_channel.get("short_channel_id") or "")
        if direct_scid:
            route = [
                {
                    "id": payee_id,
                    "channel": direct_scid,
                    "amount_msat": int(amount_sats) * 1000,
                    "delay": 9,
                    "style": "tlv",
                }
            ]
            route_result = {
                "ok": True,
                "route": route,
                "fallback": "manual_single_hop_peer_channel",
                "getroute": route_result,
            }
    if isinstance(route, list) and route and bolt11 and payment_hash:
        sendpay_args = [
            "sendpay",
            f"route={json.dumps(route, separators=(',', ':'))}",
            f"payment_hash={payment_hash}",
            f"amount_msat={int(amount_sats) * 1000}",
            f"bolt11={bolt11}",
        ]
        if payment_secret:
            sendpay_args.append(f"payment_secret={payment_secret}")
        sent = tournament.cln(payer, *sendpay_args, timeout_seconds=45.0)
        paid = (
            tournament.cln(
                payer,
                "waitsendpay",
                payment_hash,
                "45",
                timeout_seconds=50.0,
            )
            if tournament.rpc_result_ok(sent) else
            {"ok": False, "reason": "sendpay_failed", "sendpay": sent}
        )
    else:
        sent = {"ok": False, "reason": "direct_route_missing", "route_result": route_result}
        paid = sent
    return {
        "ok": (
            tournament.rpc_result_ok(invoice)
            and tournament.rpc_result_ok(route_result)
            and tournament.rpc_result_ok(sent)
            and tournament.rpc_result_ok(paid)
        ),
        "payer": payer,
        "payee": payee,
        "amount_sats": amount_sats,
        "label": label,
        "invoice": invoice,
        "connect": connected,
        "route": route_result,
        "sendpay": sent,
        "pay": paid,
    }


def _set_channel_policy(
    *,
    node: str,
    short_channel_id: str,
    base_msat: int,
    ppm: int,
) -> dict[str, Any]:
    if not short_channel_id:
        return {"ok": False, "reason": "short_channel_id_missing", "node": node}
    return tournament.cln(
        node,
        "setchannel",
        short_channel_id,
        str(base_msat),
        str(ppm),
    )


def prepare_intrahive_corridor(
    *,
    source_node: str,
    relay_node: str,
    dest_node: str,
    source_target_ratio: float,
    dest_target_ratio: float,
    corridor_target_ratio: float,
    corridor_base_msat: int,
    corridor_fee_ppm: int,
) -> dict[str, Any]:
    """Create a controlled revenue -> source -> relay -> dest -> revenue cycle."""
    return prepare_pure_hive_corridor(
        path_nodes=[source_node, relay_node, dest_node],
        source_target_ratio=source_target_ratio,
        dest_target_ratio=dest_target_ratio,
        corridor_target_ratio=corridor_target_ratio,
        corridor_base_msat=corridor_base_msat,
        corridor_fee_ppm=corridor_fee_ppm,
    )


def prepare_pure_hive_corridor(
    *,
    path_nodes: list[str],
    source_target_ratio: float,
    dest_target_ratio: float,
    corridor_target_ratio: float,
    corridor_base_msat: int,
    corridor_fee_ppm: int,
) -> dict[str, Any]:
    """Create a controlled circular route through only the requested hive nodes.

    The revenue-node source channel is pushed above the high threshold, the
    destination channel below the low threshold, and each middle corridor edge
    is topped up in the direction needed by circular rebalancing.
    """
    if len(path_nodes) < 2:
        return {
            "ok": False,
            "reason": "pure_hive_path_requires_at_least_two_member_nodes",
            "path_nodes": path_nodes,
        }
    revenue_id = _node_id(REVENUE)
    source_node = path_nodes[0]
    dest_node = path_nodes[-1]
    path_ids = {node: _node_id(node) for node in path_nodes}
    source_id = path_ids[source_node]
    dest_id = path_ids[dest_node]
    ids = {
        "revenue": revenue_id,
        "source": source_id,
        "dest": dest_id,
        "path": path_ids,
    }

    before = {
        "revenue_source": _channel_summary(REVENUE, source_id),
        "revenue_dest": _channel_summary(REVENUE, dest_id),
    }
    for index, (left_node, right_node) in enumerate(zip(path_nodes, path_nodes[1:]), start=1):
        before[f"corridor_{index}_{left_node}_to_{right_node}"] = _channel_summary(
            left_node,
            path_ids[right_node],
        )

    revenue_source = _peer_channel(REVENUE, source_id)
    revenue_dest = _peer_channel(REVENUE, dest_id)

    corridor_channels: dict[str, dict[str, Any]] = {}
    policy_updates: dict[str, dict[str, Any]] = {}
    payments: dict[str, dict[str, Any]] = {}
    missing_channels: list[str] = []
    for index, (left_node, right_node) in enumerate(zip(path_nodes, path_nodes[1:]), start=1):
        right_id = path_ids[right_node]
        channel = _peer_channel(left_node, right_id)
        key = f"corridor_{index}_{left_node}_to_{right_node}"
        corridor_channels[key] = channel
        if not channel:
            missing_channels.append(f"{left_node}->{right_node}")
        policy_updates[key] = _set_channel_policy(
            node=left_node,
            short_channel_id=str(channel.get("short_channel_id") or ""),
            base_msat=corridor_base_msat,
            ppm=corridor_fee_ppm,
        )
        payments[f"{key}_liquidity"] = _pay_between(
            payer=right_node,
            payee=left_node,
            amount_sats=_amount_to_target_local(channel, corridor_target_ratio),
            label_prefix=f"pure-hive-{index}",
        )
    if not revenue_source:
        missing_channels.append(f"{REVENUE}->{source_node}")
    if not revenue_dest:
        missing_channels.append(f"{REVENUE}->{dest_node}")

    payments.update({
        "source_to_revenue_source_balance": _pay_between(
            payer=source_node,
            payee=REVENUE,
            amount_sats=_amount_to_target_local(revenue_source, source_target_ratio),
            label_prefix="intrahive-source-balance",
        ),
        "revenue_to_dest_dest_balance": _pay_between(
            payer=REVENUE,
            payee=dest_node,
            amount_sats=_amount_to_reduce_local(revenue_dest, dest_target_ratio),
            label_prefix="intrahive-dest-balance",
        ),
    })

    after = {
        "revenue_source": _channel_summary(REVENUE, source_id),
        "revenue_dest": _channel_summary(REVENUE, dest_id),
    }
    for index, (left_node, right_node) in enumerate(zip(path_nodes, path_nodes[1:]), start=1):
        after[f"corridor_{index}_{left_node}_to_{right_node}"] = _channel_summary(
            left_node,
            path_ids[right_node],
        )
    return {
        "ok": not missing_channels
        and all(tournament.rpc_result_ok(result) for result in policy_updates.values())
        and all(result.get("ok", False) for result in payments.values()),
        "nodes": {
            "source_node": source_node,
            "dest_node": dest_node,
            "path_nodes": path_nodes,
        },
        "ids": ids,
        "targets": {
            "source_target_ratio": source_target_ratio,
            "dest_target_ratio": dest_target_ratio,
            "corridor_target_ratio": corridor_target_ratio,
            "corridor_base_msat": corridor_base_msat,
            "corridor_fee_ppm": corridor_fee_ppm,
        },
        "missing_channels": missing_channels,
        "before": before,
        "policy_updates": policy_updates,
        "payments": payments,
        "after": after,
    }


def prepare_corridor_for_args(
    *,
    args: argparse.Namespace,
    pure_hive_path_nodes: list[str],
    prepare_pure_hive: bool,
) -> dict[str, Any]:
    if prepare_pure_hive:
        return prepare_pure_hive_corridor(
            path_nodes=pure_hive_path_nodes,
            source_target_ratio=args.intrahive_source_target_ratio,
            dest_target_ratio=args.intrahive_dest_target_ratio,
            corridor_target_ratio=args.intrahive_corridor_target_ratio,
            corridor_base_msat=args.intrahive_corridor_base_msat,
            corridor_fee_ppm=args.intrahive_corridor_fee_ppm,
        )
    if args.prepare_intrahive_corridor:
        return prepare_intrahive_corridor(
            source_node=args.intrahive_source_node,
            relay_node=args.intrahive_relay_node,
            dest_node=args.intrahive_dest_node,
            source_target_ratio=args.intrahive_source_target_ratio,
            dest_target_ratio=args.intrahive_dest_target_ratio,
            corridor_target_ratio=args.intrahive_corridor_target_ratio,
            corridor_base_msat=args.intrahive_corridor_base_msat,
            corridor_fee_ppm=args.intrahive_corridor_fee_ppm,
        )
    return {"ok": True, "skipped": True}


def deploy_revenue_ops(
    *,
    repo_root: Path,
    node: str = REVENUE,
    container_dir: str = DEFAULT_CONTAINER_DIR,
) -> dict[str, Any]:
    modules_src = repo_root / "modules"
    plugin_src = repo_root / "cl-revenue-ops.py"
    if not plugin_src.exists() or not modules_src.is_dir():
        return {
            "ok": False,
            "reason": "expected cl-revenue-ops.py and modules/ in repo root",
            "repo_root": str(repo_root),
        }

    steps = [
        tournament.docker_exec(node, "mkdir", "-p", container_dir, f"{container_dir}/modules"),
        tournament.docker_cp_to_node(plugin_src, node, f"{container_dir}/cl-revenue-ops.py"),
        tournament.docker_cp_to_node(f"{modules_src}/.", node, f"{container_dir}/modules/"),
        tournament.docker_exec(node, "chmod", "+x", f"{container_dir}/cl-revenue-ops.py"),
    ]
    return {
        "ok": all(step.get("ok", False) for step in steps),
        "repo_root": str(repo_root),
        "container_dir": container_dir,
        "steps": steps,
    }


def _datastore_key_arg(key: list[str]) -> str:
    return json.dumps(key, separators=(",", ":"))


def clear_hive_hints_datastore(node: str = REVENUE) -> dict[str, Any]:
    current = tournament.cln(node, "listdatastore", _datastore_key_arg(["hive", "hints"]))
    entries = current.get("datastore") if isinstance(current, dict) else []
    deleted: list[dict[str, Any]] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key") if isinstance(entry.get("key"), list) else ["hive", "hints"]
            args = ["deldatastore", _datastore_key_arg([str(part) for part in key])]
            if entry.get("generation") is not None:
                args.append(str(entry.get("generation")))
            deleted.append(tournament.cln(node, *args))
    after = tournament.cln(node, "listdatastore", _datastore_key_arg(["hive", "hints"]))
    return {
        "ok": tournament.rpc_result_ok(after) and not bool(after.get("datastore")),
        "before": current,
        "deleted": deleted,
        "after": after,
    }


def disable_cl_hive(node: str = REVENUE) -> dict[str, Any]:
    status_before = tournament.cln(node, "hive-status")
    stopped = tournament.cln(node, "plugin", "stop", tournament.DEFAULT_CL_HIVE_PLUGIN_PATH)
    status_after = tournament.cln(node, "hive-status")
    cleared_datastore = clear_hive_hints_datastore(node=node)
    removed_layers = {
        layer: remove_askrene_layer(layer, node=node)
        for layer in STALE_HIVE_LAYERS
    }
    return {
        "ok": not tournament.rpc_result_ok(status_after) and cleared_datastore.get("ok", False),
        "status_before": status_before,
        "stop_result": stopped,
        "status_after": status_after,
        "cleared_datastore": cleared_datastore,
        "removed_askrene_layers": removed_layers,
    }


def enable_cl_hive(
    *,
    host_path: Path,
    plugin_path: str,
    hive_id: str,
    deploy: bool,
    start: bool,
    genesis: bool,
    install_deps: bool,
    node: str = REVENUE,
) -> dict[str, Any]:
    return tournament.ensure_cl_hive(
        host_path=host_path,
        plugin_path=plugin_path,
        deploy=deploy,
        start=start,
        genesis=genesis,
        install_deps=install_deps,
        hive_id=hive_id,
        node=node,
    )


def push_hive_hints_datastore(node: str = REVENUE) -> dict[str, Any]:
    """Force current cl-hive hints into the shared datastore.

    The Polar harness seeds synthetic members by writing cl-hive's SQLite DB
    directly. cl_revenue_ops prefers the datastore snapshot, so tests must
    refresh that snapshot immediately instead of waiting for the background
    cl-hive hint-push loop.
    """
    exported = tournament.cln(node, "hive-export-hints")
    if not tournament.rpc_result_ok(exported):
        return {
            "ok": False,
            "stage": "hive-export-hints",
            "export_hints": exported,
        }
    payload = json.dumps(exported, separators=(",", ":"), sort_keys=True)
    payload_hex = payload.encode("utf-8").hex()
    pushed = tournament.cln(
        node,
        "datastore",
        'key=["hive","hints"]',
        f"hex={payload_hex}",
        "mode=create-or-replace",
    )
    return {
        "ok": tournament.rpc_result_ok(pushed),
        "export_hints": exported,
        "datastore": pushed,
    }


def seed_hive_channel_peers(
    *,
    peer_ids: list[str] | None = None,
    include_channel_peers: bool = True,
    hive_id: str = tournament.DEFAULT_CL_HIVE_ID,
    node: str = REVENUE,
) -> dict[str, Any]:
    channels = tournament.cln(node, "listpeerchannels")
    discovered = sorted({
        str(ch.get("peer_id") or "")
        for ch in (channels.get("channels") or [])
        if isinstance(ch, dict)
        and ch.get("state") == "CHANNELD_NORMAL"
        and ch.get("peer_id")
    })
    selected = sorted(
        set((discovered if include_channel_peers else []) + list(peer_ids or []))
    )
    script = (
        "import json,sqlite3,sys,time;"
        "peers=json.loads(sys.argv[1]);hive_id=sys.argv[2];"
        "db='/home/clightning/.lightning/cl_hive.db';"
        "con=sqlite3.connect(db);now=int(time.time());"
        "before=con.execute('select count(*) from hive_members').fetchone()[0];"
        "meta=json.dumps({'hive_id':hive_id});"
        "[(con.execute('insert or ignore into hive_members "
        "(peer_id,tier,joined_at,last_seen,metadata) values (?,?,?,?,?)',"
        "(p,'member',now,now,meta))) for p in peers];"
        "con.commit();"
        "rows=con.execute('select peer_id,tier from hive_members order by peer_id').fetchall();"
        "after=len(rows);"
        "print(json.dumps({'ok':True,'before_count':before,'after_count':after,"
        "'inserted':after-before,'seeded_peers':peers,"
        "'members':[{'peer_id':r[0],'tier':r[1]} for r in rows]}))"
    )
    seeded = tournament.docker_exec(
        node,
        "python3",
        "-c",
        script,
        json.dumps(selected),
        hive_id,
    )
    pushed = push_hive_hints_datastore(node)
    return {
        "ok": bool(seeded.get("ok", False)) and bool(pushed.get("ok", False)),
        "discovered_channel_peers": discovered,
        "include_channel_peers": include_channel_peers,
        "selected_peers": selected,
        "seeded": seeded,
        "export_hints": pushed.get("export_hints"),
        "push_datastore": pushed,
    }


def refresh_cl_hive_runtime(node: str = REVENUE) -> dict[str, Any]:
    triggered = tournament.cln(node, "hive-trigger-all", timeout_seconds=45.0)
    pushed = push_hive_hints_datastore(node)
    exported = pushed.get("export_hints", {})
    layers = tournament.cln(node, "askrene-listlayers")
    return {
        "ok": (
            (tournament.rpc_result_ok(triggered) or "error" not in triggered)
            and bool(pushed.get("ok", False))
            and tournament.rpc_result_ok(layers)
        ),
        "trigger_all": triggered,
        "export_hints": exported,
        "push_datastore": pushed,
        "askrene_layers": layers,
    }


def clear_rebalance_cooldowns(node: str = REVENUE) -> dict[str, Any]:
    script = (
        "import json,sqlite3;"
        "db='/home/clightning/.lightning/revenue_ops.db';"
        "con=sqlite3.connect(db);"
        "pair_before=con.execute('select count(*) from pair_rebalance_failures').fetchone()[0];"
        "hist_before=con.execute(\"select count(*) from rebalance_history where status='success'\").fetchone()[0];"
        "con.execute('delete from pair_rebalance_failures');"
        "con.execute(\"delete from rebalance_history where status='success'\");"
        "con.commit();"
        "pair_after=con.execute('select count(*) from pair_rebalance_failures').fetchone()[0];"
        "hist_after=con.execute(\"select count(*) from rebalance_history where status='success'\").fetchone()[0];"
        "print(json.dumps({'ok':True,"
        "'pair_before_count':pair_before,'pair_after_count':pair_after,"
        "'pair_deleted':pair_before-pair_after,"
        "'success_history_before_count':hist_before,"
        "'success_history_after_count':hist_after,"
        "'success_history_deleted':hist_before-hist_after}))"
    )
    return tournament.docker_exec(node, "python3", "-c", script)


def set_hive_hints_disabled(node: str = REVENUE) -> dict[str, Any]:
    first = tournament.cln(
        node,
        "setconfig",
        "revenue-ops-hive-hints-enabled",
        "false",
        "true",
    )
    if first.get("ok", False) and "error" not in first:
        return {"ok": True, "method": "positional", "result": first}
    second = tournament.cln(
        node,
        "setconfig",
        "config=revenue-ops-hive-hints-enabled",
        "val=false",
        "transient=true",
    )
    return {
        "ok": second.get("ok", False) and "error" not in second,
        "method": "named",
        "first": first,
        "result": second,
    }


def set_revenue_config_overrides(
    overrides: dict[str, Any],
    *,
    node: str = REVENUE,
) -> dict[str, Any]:
    if not overrides:
        return {"ok": True, "skipped": True, "overrides": {}}

    results: dict[str, Any] = {}
    for option, value in overrides.items():
        value_text = str(value).lower() if isinstance(value, bool) else str(value)
        named = tournament.cln(
            node,
            "setconfig",
            f"config={option}",
            f"val={value_text}",
            "transient=true",
        )
        if named.get("ok", False) and "error" not in named:
            results[option] = {"ok": True, "method": "named", "value": value_text, "result": named}
            continue
        positional = tournament.cln(node, "setconfig", option, value_text, "true")
        results[option] = {
            "ok": positional.get("ok", False) and "error" not in positional,
            "method": "positional",
            "value": value_text,
            "named": named,
            "result": positional,
        }
    return {
        "ok": all(item.get("ok", False) for item in results.values()),
        "overrides": overrides,
        "results": results,
        "requires_restart": False,
    }


def restart_revenue_ops(
    *,
    plugin_path: str = DEFAULT_PLUGIN_PATH,
    wait_seconds: float = 3.0,
    node: str = REVENUE,
) -> dict[str, Any]:
    stopped = tournament.cln(node, "plugin", "stop", plugin_path)
    started = tournament.cln(node, "plugin", "start", plugin_path)
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    status = tournament.cln(node, "revenue-status")
    return {
        "ok": tournament.rpc_result_ok(status),
        "stopped": stopped,
        "started": started,
        "status": status,
        "wait_seconds": wait_seconds,
    }


def _missing_layer_ok(result: dict[str, Any]) -> bool:
    text = " ".join(str(result.get(key) or "") for key in ("stdout", "stderr", "error"))
    return "Unknown layer" in text


def remove_askrene_layer(layer: str, node: str = REVENUE) -> dict[str, Any]:
    result = tournament.cln(node, "askrene-remove-layer", layer)
    if isinstance(result, dict) and not result.get("ok", False) and _missing_layer_ok(result):
        normalized = dict(result)
        normalized["ok"] = True
        normalized["skipped"] = True
        normalized["reason"] = "layer_missing"
        return normalized
    return result


def maybe_drive_traffic(
    *,
    payments: int,
    amount_sat: int,
    competitor_ppm: int,
    payer_time_pref: float,
    payment_timeout_seconds: float,
    stop_on_failure: bool,
    policy_settle_seconds: float,
    reset_mission_control: bool,
    out_dir: Path,
) -> dict[str, Any]:
    if payments <= 0:
        return {"ok": True, "skipped": True, "payments": []}

    context = tournament.collect_static_context()
    policy = tournament.set_lnd_policy(
        tournament.COMPETITOR,
        competitor_ppm,
        cltv_delta=40,
    )
    if policy_settle_seconds > 0:
        time.sleep(policy_settle_seconds)
    mc_reset = (
        tournament.lnd(tournament.PAYER, "resetmc")
        if reset_mission_control else
        {"ok": True, "skipped": True}
    )
    before_revenue = tournament.total_forwards_cln(tournament.REVENUE)
    before_competitor = tournament.total_forwards_lnd(tournament.COMPETITOR)
    started = int(time.time())
    paid: list[dict[str, Any]] = []
    for index in range(payments):
        label = f"rebalance-capex-loop-{started}-{index}"
        paid.append(
            tournament.pay_one(
                amount_sat,
                label,
                payer_time_pref=payer_time_pref,
                timeout_seconds=payment_timeout_seconds,
            )
        )
        if stop_on_failure and paid[-1].get("ok") is not True:
            break
        time.sleep(0.2)
    after_revenue = tournament.total_forwards_cln(tournament.REVENUE)
    after_competitor = tournament.total_forwards_lnd(tournament.COMPETITOR)
    result = {
        "ok": all(item.get("ok", False) for item in paid),
        "context": context,
        "policy": policy,
        "mission_control_reset": mc_reset,
        "payments": paid,
        "payments_attempted": len(paid),
        "payments_succeeded": sum(1 for item in paid if item.get("ok", False)),
        "payment_timeouts": sum(
            1
            for item in paid
            if isinstance(item.get("result"), dict)
            and item["result"].get("timeout") is True
        ),
        "revenue_forwards_delta": max(0, after_revenue - before_revenue),
        "competitor_forwards_delta": max(0, after_competitor - before_competitor),
    }
    write_json(out_dir / f"traffic_{started}.json", result)
    return result


def snapshot(label: str, *, max_candidates: int) -> dict[str, Any]:
    return {
        "label": label,
        "timestamp": int(time.time()),
        "getinfo": tournament.cln(REVENUE, "getinfo"),
        "listpeerchannels": tournament.cln(REVENUE, "listpeerchannels"),
        "listfunds": tournament.cln(REVENUE, "listfunds"),
        "revenue_status": tournament.cln(REVENUE, "revenue-status"),
        "rebalance_debug": tournament.cln(
            REVENUE,
            "revenue-rebalance-debug",
            "summary_only=false",
            f"max_candidates={max_candidates}",
        ),
        "capex_status": tournament.cln(REVENUE, "revenue-capex-status"),
        "total_cost_budget": tournament.cln(REVENUE, "revenue-total-cost-budget"),
    }


def run_rebalance_cycle(max_candidates: int) -> dict[str, Any]:
    return tournament.cln(
        REVENUE,
        "revenue-rebalance-cycle",
        f"max_candidates={max_candidates}",
    )


def _budget(status: dict[str, Any], key: str) -> int:
    budget = status.get("total_cost_budget") or {}
    try:
        return int(budget.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _channel_budget_total(status: dict[str, Any]) -> int:
    capex = status.get("capex_status") or {}
    channels = capex.get("channels") or {}
    total = 0
    if isinstance(channels, dict):
        iterable = channels.values()
    else:
        iterable = channels
    for item in iterable:
        if not isinstance(item, dict):
            continue
        try:
            total += int(item.get("budget_sats", 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _hive_disabled_ok(status: dict[str, Any]) -> bool:
    debug = status.get("rebalance_debug") or {}
    hints = debug.get("hive_hints") or {}
    if not hints:
        return True
    if hints.get("enabled") is False:
        return True
    try:
        return int(hints.get("hints_count", 0) or 0) == 0
    except (TypeError, ValueError):
        return False


def _hive_status(status: dict[str, Any]) -> dict[str, Any]:
    debug = status.get("rebalance_debug") or {}
    hints = debug.get("hive_hints") or {}
    return hints if isinstance(hints, dict) else {}


def _hive_mode_ok(
    status: dict[str, Any],
    *,
    hive_mode: str,
    require_member_hints: bool,
) -> bool:
    hints = _hive_status(status)
    if hive_mode == "disabled":
        return _hive_disabled_ok(status)
    try:
        hints_count = int(hints.get("hints_count", 0) or 0)
    except (TypeError, ValueError):
        hints_count = 0
    try:
        member_count = int(hints.get("member_hints_count", 0) or 0)
    except (TypeError, ValueError):
        member_count = 0
    usable = bool(hints.get("snapshot_usable", hints.get("snapshot_fresh", False)))
    if require_member_hints:
        return usable and hints_count > 0 and member_count > 0
    return usable and hints_count > 0


def _hive_count(hints: dict[str, Any], key: str) -> int:
    try:
        return int(hints.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_value_class(candidate: dict[str, Any], key: str) -> str:
    value = str(candidate.get(key) or "")
    if value:
        return value
    score = candidate.get("score_decomposition") or {}
    inputs = score.get("inputs") or {}
    return str(inputs.get(key) or "")


def _is_hive_candidate(candidate: dict[str, Any]) -> bool:
    route_policy = str(candidate.get("route_policy") or "")
    reason = str(candidate.get("reason_code") or "")
    if route_policy in {"hive_only", "hybrid"}:
        return True
    if reason in {"hive_equalization", "coordinated_rebalance"}:
        return True
    return (
        _candidate_value_class(candidate, "source_value_class") == "hive"
        or _candidate_value_class(candidate, "dest_value_class") == "hive"
    )


def _is_intra_hive_candidate(candidate: dict[str, Any]) -> bool:
    return (
        _candidate_value_class(candidate, "source_value_class") == "hive"
        and _candidate_value_class(candidate, "dest_value_class") == "hive"
    )


def _hive_pair_stats(
    selected: list[Any],
    executions: list[Any],
) -> dict[str, int]:
    candidate_flags: list[tuple[bool, bool]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        candidate_flags.append((_is_hive_candidate(item), _is_intra_hive_candidate(item)))

    hive_executions = 0
    hive_successes = 0
    intra_executions = 0
    intra_successes = 0
    for index, execution in enumerate(executions):
        if not isinstance(execution, dict):
            continue
        is_hive, is_intra = candidate_flags[index] if index < len(candidate_flags) else (False, False)
        route_type = str(execution.get("route_type") or "")
        is_hive = is_hive or route_type == "fleet"
        if is_hive:
            hive_executions += 1
            if execution.get("success") is True:
                hive_successes += 1
        if is_intra:
            intra_executions += 1
            if execution.get("success") is True:
                intra_successes += 1

    return {
        "hive_selected_pairs": sum(1 for is_hive, _ in candidate_flags if is_hive),
        "intra_hive_selected_pairs": sum(1 for _, is_intra in candidate_flags if is_intra),
        "hive_executions": hive_executions,
        "hive_successes": hive_successes,
        "intra_hive_executions": intra_executions,
        "intra_hive_successes": intra_successes,
    }


def _route_in_member_set(
    candidate: dict[str, Any],
    *,
    member_ids: set[str],
    our_node_id: str,
) -> bool:
    route = candidate.get("route_summary") or []
    if not isinstance(route, list) or not member_ids:
        return False
    interior_ids = [
        str(hop.get("id") or "")
        for hop in route
        if isinstance(hop, dict) and str(hop.get("id") or "") != our_node_id
    ]
    return bool(interior_ids) and all(node_id in member_ids for node_id in interior_ids)


def _pure_hive_pair_stats(
    selected: list[Any],
    executions: list[Any],
    *,
    member_ids: set[str],
    our_node_id: str,
) -> dict[str, Any]:
    candidate_flags: list[bool] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        candidate_flags.append(
            _is_intra_hive_candidate(item)
            and _route_in_member_set(item, member_ids=member_ids, our_node_id=our_node_id)
        )

    pure_executions = 0
    pure_successes = 0
    for index, execution in enumerate(executions):
        if not isinstance(execution, dict):
            continue
        is_pure = candidate_flags[index] if index < len(candidate_flags) else False
        if is_pure:
            pure_executions += 1
            if execution.get("success") is True:
                pure_successes += 1

    return {
        "pure_hive_selected_pairs": sum(1 for flag in candidate_flags if flag),
        "pure_hive_executions": pure_executions,
        "pure_hive_successes": pure_successes,
        "pure_hive_route_ok": bool(candidate_flags) and all(candidate_flags),
    }


def _traffic_ok(traffic: dict[str, Any]) -> bool:
    return bool(traffic.get("ok", False)) or bool(traffic.get("skipped", False))


def _payment_timeouts(traffic: dict[str, Any]) -> int:
    try:
        return int(traffic.get("payment_timeouts", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _execution_errors(executions: list[Any]) -> list[str]:
    errors: list[str] = []
    for item in executions:
        if not isinstance(item, dict):
            continue
        error = str(item.get("error") or "").strip()
        if error:
            errors.append(error)
    return errors


def _channel_state_by_scid(snapshot_data: dict[str, Any], scid: str) -> dict[str, Any]:
    if not scid:
        return {}
    channels = (snapshot_data.get("listpeerchannels") or {}).get("channels") or []
    for channel in channels:
        if isinstance(channel, dict) and str(channel.get("short_channel_id") or "") == scid:
            capacity = _channel_capacity_sats(channel)
            local = _channel_local_sats(channel)
            return {
                "short_channel_id": scid,
                "local_sats": local,
                "capacity_sats": capacity,
                "local_ratio": local / capacity if capacity else None,
            }
    return {}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _matching_selected_candidate(
    cycle: dict[str, Any],
    *,
    source_scid: str,
    dest_scid: str,
) -> dict[str, Any]:
    last_cycle = cycle.get("last_cycle") if isinstance(cycle, dict) else {}
    selected = (last_cycle or {}).get("selected_candidates") or []
    for candidate in selected if isinstance(selected, list) else []:
        if not isinstance(candidate, dict):
            continue
        if (
            str(candidate.get("source_channel_id") or "") == source_scid
            and str(candidate.get("dest_channel_id") or "") == dest_scid
        ):
            return candidate
    return {}


def _convergence_metrics(
    *,
    perturbation: dict[str, Any] | None,
    cycle: dict[str, Any],
    after: dict[str, Any],
    fee_sats: int,
    tolerance_ratio: float,
) -> dict[str, Any]:
    base = {
        "convergence_perturb_ok": True,
        "convergence_pair_selected": False,
        "convergence_source_scid": "",
        "convergence_dest_scid": "",
        "convergence_source_target_ratio": None,
        "convergence_dest_target_ratio": None,
        "convergence_source_before_ratio": None,
        "convergence_dest_before_ratio": None,
        "convergence_source_after_ratio": None,
        "convergence_dest_after_ratio": None,
        "convergence_source_error": None,
        "convergence_dest_error": None,
        "convergence_max_error": None,
        "convergence_restored_sats": 0,
        "convergence_fee_per_restored_sat": None,
        "convergence_ok": False,
    }
    if not perturbation or perturbation.get("skipped"):
        return base

    base["convergence_perturb_ok"] = bool(perturbation.get("ok", False))
    prepared = perturbation.get("after") or {}
    source_pre = prepared.get("revenue_source") or {}
    dest_pre = prepared.get("revenue_dest") or {}
    source_scid = str(source_pre.get("short_channel_id") or "")
    dest_scid = str(dest_pre.get("short_channel_id") or "")
    base["convergence_source_scid"] = source_scid
    base["convergence_dest_scid"] = dest_scid
    base["convergence_source_before_ratio"] = _as_float(source_pre.get("local_ratio"))
    base["convergence_dest_before_ratio"] = _as_float(dest_pre.get("local_ratio"))

    candidate = _matching_selected_candidate(
        cycle,
        source_scid=source_scid,
        dest_scid=dest_scid,
    )
    base["convergence_pair_selected"] = bool(candidate)
    inputs = ((candidate.get("score_decomposition") or {}).get("inputs") or {}) if candidate else {}
    source_target = _as_float(inputs.get("target_band_high"), 0.7)
    dest_target = _as_float(inputs.get("target_band_low"), 0.3)
    base["convergence_source_target_ratio"] = source_target
    base["convergence_dest_target_ratio"] = dest_target

    source_after = _channel_state_by_scid(after, source_scid)
    dest_after = _channel_state_by_scid(after, dest_scid)
    source_after_ratio = _as_float(source_after.get("local_ratio"))
    dest_after_ratio = _as_float(dest_after.get("local_ratio"))
    base["convergence_source_after_ratio"] = source_after_ratio
    base["convergence_dest_after_ratio"] = dest_after_ratio

    source_error = (
        abs(source_after_ratio - source_target)
        if source_after_ratio is not None and source_target is not None else
        None
    )
    dest_error = (
        abs(dest_after_ratio - dest_target)
        if dest_after_ratio is not None and dest_target is not None else
        None
    )
    base["convergence_source_error"] = source_error
    base["convergence_dest_error"] = dest_error
    errors = [err for err in (source_error, dest_error) if err is not None]
    base["convergence_max_error"] = max(errors) if errors else None

    try:
        restored = max(
            0,
            int(dest_after.get("local_sats", 0) or 0)
            - int(dest_pre.get("local_sats", 0) or 0),
        )
    except (TypeError, ValueError):
        restored = 0
    base["convergence_restored_sats"] = restored
    if restored > 0:
        base["convergence_fee_per_restored_sat"] = round(float(fee_sats) / restored, 8)
    base["convergence_ok"] = (
        bool(base["convergence_perturb_ok"])
        and bool(base["convergence_pair_selected"])
        and restored > 0
        and base["convergence_max_error"] is not None
        and float(base["convergence_max_error"]) <= tolerance_ratio
    )
    return base


def _primary_skip_reason(skipped: list[Any]) -> str:
    priorities = (
        "below_hold_margin",
        "route_over_budget",
        "pair_cooldown",
        "no_route",
        "native_unavailable",
        "cooldown",
    )
    reasons = [
        str(item.get("reason") or "")
        for item in skipped
        if isinstance(item, dict) and item.get("reason")
    ]
    for priority in priorities:
        if priority in reasons:
            return priority
    return reasons[0] if reasons else ""


def analyze_iteration(
    *,
    iteration: int,
    hive_mode: str,
    require_hive_member_hints: bool,
    require_pure_hive_route: bool,
    pure_hive_member_ids: set[str],
    traffic: dict[str, Any],
    before: dict[str, Any],
    cycle: dict[str, Any],
    after: dict[str, Any],
    perturbation: dict[str, Any] | None,
    convergence_tolerance_ratio: float,
) -> IterationAnalysis:
    last_cycle = cycle.get("last_cycle") if isinstance(cycle, dict) else {}
    if not isinstance(last_cycle, dict):
        last_cycle = {}
    summary = last_cycle.get("summary") or {}
    executions = last_cycle.get("executions") or []
    selected = last_cycle.get("selected_candidates") or []
    errors = _execution_errors(executions if isinstance(executions, list) else [])
    fee_sats = sum(int(item.get("fee_sats", 0) or 0) for item in executions if isinstance(item, dict))
    successes = int(summary.get("execution_success_count", 0) or 0)
    execution_count = int(summary.get("execution_count", len(executions)) or 0)
    failures = max(0, execution_count - successes)
    budget_spent_delta = _budget(after, "actual_spent_sats") - _budget(before, "actual_spent_sats")
    budget_reserved_delta = _budget(after, "reserved_sats") - _budget(before, "reserved_sats")
    channel_budget_delta = _channel_budget_total(after) - _channel_budget_total(before)
    reservation_leak = budget_reserved_delta > 0
    accounting_ok = True
    if successes > 0 and fee_sats > 0:
        accounting_ok = budget_spent_delta >= fee_sats and not reservation_leak
    hive_status = _hive_status(after)
    hive_ok = _hive_mode_ok(
        after,
        hive_mode=hive_mode,
        require_member_hints=require_hive_member_hints,
    )
    hive_pair_stats = _hive_pair_stats(
        selected if isinstance(selected, list) else [],
        executions if isinstance(executions, list) else [],
    )
    our_node_id = str((after.get("getinfo") or {}).get("id") or "")
    pure_hive_stats = _pure_hive_pair_stats(
        selected if isinstance(selected, list) else [],
        executions if isinstance(executions, list) else [],
        member_ids=pure_hive_member_ids,
        our_node_id=our_node_id,
    )
    traffic_ok = _traffic_ok(traffic)
    payment_timeouts = _payment_timeouts(traffic)
    convergence = _convergence_metrics(
        perturbation=perturbation,
        cycle=cycle,
        after=after,
        fee_sats=fee_sats,
        tolerance_ratio=convergence_tolerance_ratio,
    )
    convergence_required = bool(perturbation and not perturbation.get("skipped"))
    pure_hive_ok = (
        not require_pure_hive_route
        or (
            pure_hive_stats["pure_hive_selected_pairs"] > 0
            and pure_hive_stats["pure_hive_executions"] == execution_count
            and pure_hive_stats["pure_hive_successes"] == successes
        )
    )
    convergence_selection_ok = (
        not convergence_required
        or (
            bool(convergence["convergence_perturb_ok"])
            and bool(convergence["convergence_pair_selected"])
        )
    )

    valid = (
        bool(cycle.get("ok", False))
        and "error" not in cycle
        and hive_ok
        and traffic_ok
        and pure_hive_ok
        and convergence_selection_ok
    )
    if not valid:
        action = "refine_test"
        if convergence_required and not convergence["convergence_perturb_ok"]:
            reason = "convergence perturbation failed before the rebalance cycle"
        elif not traffic_ok:
            reason = "traffic generation failed or timed out before the rebalance cycle"
        elif not hive_ok and hive_mode == "enabled":
            reason = "hive mode requested but usable/member hive hints were not active"
        elif not hive_ok:
            reason = "standalone mode requested but hive hints are still active"
        elif convergence_required and not convergence["convergence_pair_selected"]:
            action = "tune_or_change_code"
            reason = "convergence pair was not selected"
        elif not pure_hive_ok:
            action = "change_code_or_topology"
            reason = "pure hive route required but selected/executed route left the active member set"
        else:
            reason = "cycle RPC failed"
    elif successes > 0 and not accounting_ok:
        action = "change_code"
        reason = "successful rebalance spend did not hit unified budget/capex accounting"
    elif (
        convergence_required
        and execution_count > 0
        and successes > 0
        and not convergence["convergence_ok"]
    ):
        action = "tune_or_change_code"
        reason = (
            "rebalance executed but convergence target was missed"
            + (
                f" (max_error={convergence['convergence_max_error']:.6f})"
                if convergence["convergence_max_error"] is not None else
                ""
            )
        )
    elif execution_count > 0 and failures > 0:
        joined_errors = " ".join(errors).lower()
        if "retry_no_route" in joined_errors and "erring_channel" in joined_errors:
            action = "repeat_or_refine"
            reason = (
                "native execution identified the failed channel direction, "
                "but the topology had no alternate route"
            )
        elif any(
            token in joined_errors
            for token in (
                "noroutes",
                "no_routes",
                "nocheaproute",
                "native_sendpay_error",
                "native_route_invalid",
                "native_route_over_budget",
            )
        ):
            action = "change_code_or_executor"
            reason = "priced pairs reached the executor but at least one route failed execution"
        else:
            action = "repeat_or_refine"
            reason = "rebalancer found executable pairs but at least one failed"
    elif int(summary.get("considered_pairs", 0) or 0) == 0:
        action = "refine_test"
        reason = "no imbalanced, funded candidate pairs were available"
    elif int(summary.get("selected_pairs", 0) or 0) == 0:
        skipped = last_cycle.get("skipped") or []
        first_reason = _primary_skip_reason(skipped if isinstance(skipped, list) else [])
        action = "change_code_or_topology" if first_reason == "no_route" else "repeat_or_refine"
        outcome = "none selected"
        if first_reason == "no_route":
            outcome = "none priced"
        elif first_reason in {"below_hold_margin", "route_over_budget"}:
            outcome = "none survived pricing gates"
        reason = (
            f"{summary.get('considered_pairs')} pairs were considered but {outcome}"
            + (f" ({first_reason})" if first_reason else "")
        )
    else:
        action = "repeat"
        reason = "cycle completed with internally consistent accounting"

    return IterationAnalysis(
        iteration=iteration,
        valid=valid,
        action=action,
        reason=reason,
        traffic_ok=traffic_ok,
        payment_timeouts=payment_timeouts,
        selected_pairs=len(selected),
        executions=execution_count,
        successes=successes,
        failures=failures,
        fee_sats=fee_sats,
        budget_spent_delta_sats=budget_spent_delta,
        budget_reserved_delta_sats=budget_reserved_delta,
        channel_budget_delta_sats=channel_budget_delta,
        reservation_leak=reservation_leak,
        accounting_ok=accounting_ok,
        hive_mode=hive_mode,
        hive_ok=hive_ok,
        hive_hints_fresh=bool(hive_status.get("snapshot_fresh", False)),
        hive_hints_usable=bool(hive_status.get("snapshot_usable", False)),
        hive_hints_count=_hive_count(hive_status, "hints_count"),
        hive_member_hints_count=_hive_count(hive_status, "member_hints_count"),
        hive_selected_pairs=hive_pair_stats["hive_selected_pairs"],
        intra_hive_selected_pairs=hive_pair_stats["intra_hive_selected_pairs"],
        hive_executions=hive_pair_stats["hive_executions"],
        hive_successes=hive_pair_stats["hive_successes"],
        intra_hive_executions=hive_pair_stats["intra_hive_executions"],
        intra_hive_successes=hive_pair_stats["intra_hive_successes"],
        pure_hive_required=require_pure_hive_route,
        pure_hive_selected_pairs=pure_hive_stats["pure_hive_selected_pairs"],
        pure_hive_executions=pure_hive_stats["pure_hive_executions"],
        pure_hive_successes=pure_hive_stats["pure_hive_successes"],
        pure_hive_route_ok=bool(pure_hive_stats["pure_hive_route_ok"]),
        hive_disabled_ok=hive_ok,
        **convergence,
    )


def write_analysis(path: Path, analyses: list[IterationAnalysis]) -> None:
    lines = ["# Rebalance/Capex Loop Analysis", ""]
    if not analyses:
        lines.append("No iterations ran.")
    else:
        total_successes = sum(a.successes for a in analyses)
        total_exec = sum(a.executions for a in analyses)
        total_fee = sum(a.fee_sats for a in analyses)
        leaks = sum(1 for a in analyses if a.reservation_leak)
        accounting_failures = sum(1 for a in analyses if not a.accounting_ok)
        total_hive_successes = sum(a.hive_successes for a in analyses)
        total_intra_hive_successes = sum(a.intra_hive_successes for a in analyses)
        total_pure_hive_successes = sum(a.pure_hive_successes for a in analyses)
        convergence_runs = sum(1 for a in analyses if a.convergence_source_scid)
        convergence_ok = sum(1 for a in analyses if a.convergence_ok)
        convergence_errors = [
            a.convergence_max_error
            for a in analyses
            if a.convergence_max_error is not None
        ]
        avg_convergence_error = (
            sum(convergence_errors) / len(convergence_errors)
            if convergence_errors else
            None
        )
        lines.extend([
            f"- Iterations: {len(analyses)}",
            f"- Executions: {total_exec}",
            f"- Successes: {total_successes}",
            f"- Fees paid: {total_fee} sats",
            f"- Reservation leaks: {leaks}",
            f"- Accounting failures: {accounting_failures}",
            f"- Hive successes: {total_hive_successes}",
            f"- Intrahive successes: {total_intra_hive_successes}",
            f"- Pure hive successes: {total_pure_hive_successes}",
            f"- Convergence runs: {convergence_runs}",
            f"- Convergence ok: {convergence_ok}",
            (
                f"- Avg convergence max error: {avg_convergence_error:.6f}"
                if avg_convergence_error is not None else
                "- Avg convergence max error: n/a"
            ),
            "",
            "| iter | mode | action | hive ok | selected | exec | success | fee sats | conv ok | conv max err | restored | fee/restored | reason |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ])
        for a in analyses:
            conv_error = (
                f"{a.convergence_max_error:.6f}"
                if a.convergence_max_error is not None else
                ""
            )
            fee_per_restored = (
                f"{a.convergence_fee_per_restored_sat:.8f}"
                if a.convergence_fee_per_restored_sat is not None else
                ""
            )
            lines.append(
                f"| {a.iteration} | {a.hive_mode} | {a.action} | {a.hive_ok} | "
                f"{a.selected_pairs} | {a.executions} | {a.successes} | "
                f"{a.fee_sats} | {a.convergence_ok} | {conv_error} | "
                f"{a.convergence_restored_sats} | {fee_per_restored} | {a.reason} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--restart-wait-seconds", type=float, default=5.0)
    parser.add_argument("--plugin-path", default=DEFAULT_PLUGIN_PATH)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--restart-plugin", action="store_true")
    parser.add_argument("--hive-mode", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--skip-disable-cl-hive", action="store_true")
    parser.add_argument("--cl-hive-host-path", type=Path, default=tournament.DEFAULT_CL_HIVE_HOST_PATH)
    parser.add_argument("--cl-hive-plugin-path", default=tournament.DEFAULT_CL_HIVE_PLUGIN_PATH)
    parser.add_argument("--cl-hive-id", default=tournament.DEFAULT_CL_HIVE_ID)
    parser.add_argument("--install-cl-hive-deps", action="store_true")
    parser.add_argument("--skip-cl-hive-deploy", action="store_true")
    parser.add_argument("--skip-cl-hive-start", action="store_true")
    parser.add_argument("--skip-cl-hive-genesis", action="store_true")
    parser.add_argument("--hive-seed-channel-peers", action="store_true")
    parser.add_argument("--hive-seed-peer-id", action="append", default=[])
    parser.add_argument("--require-hive-member-hints", action="store_true")
    parser.add_argument("--hive-start-member-plugins", action="store_true")
    parser.add_argument("--hive-member-node", action="append", default=[])
    parser.add_argument("--pure-hive-topology", choices=PURE_HIVE_TOPOLOGIES, default="none")
    parser.add_argument("--require-pure-hive-route", action="store_true")
    parser.add_argument("--prepare-intrahive-corridor", action="store_true")
    parser.add_argument("--intrahive-source-node", default=DEFAULT_INTRAHIVE_SOURCE_NODE)
    parser.add_argument("--intrahive-relay-node", default=DEFAULT_INTRAHIVE_RELAY_NODE)
    parser.add_argument("--intrahive-dest-node", default=DEFAULT_INTRAHIVE_DEST_NODE)
    parser.add_argument("--intrahive-source-target-ratio", type=float, default=0.82)
    parser.add_argument("--intrahive-dest-target-ratio", type=float, default=0.18)
    parser.add_argument("--intrahive-corridor-target-ratio", type=float, default=0.55)
    parser.add_argument("--intrahive-corridor-base-msat", type=int, default=1)
    parser.add_argument("--intrahive-corridor-fee-ppm", type=int, default=1)
    parser.add_argument("--clear-rebalance-cooldowns", action="store_true")
    parser.add_argument("--clear-rebalance-cooldowns-each-iteration", action="store_true")
    parser.add_argument("--convergence-perturb-each-iteration", action="store_true")
    parser.add_argument("--convergence-ok-error-ratio", type=float, default=0.01)
    parser.add_argument("--tune-pair-fee-cap-ppm", type=int, default=None)
    parser.add_argument("--tune-rebalance-hold-margin", type=float, default=None)
    parser.add_argument("--tune-hive-bootstrap-budget-sats", type=int, default=None)
    parser.add_argument("--drive-payments", type=int, default=6)
    parser.add_argument("--payment-amount-sat", type=int, default=20_000)
    parser.add_argument("--payment-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--continue-after-payment-failure", action="store_true")
    parser.add_argument("--competitor-ppm", type=int, default=500)
    parser.add_argument("--payer-time-pref", type=float, default=-1.0)
    parser.add_argument("--policy-settle-seconds", type=float, default=3.0)
    parser.add_argument("--no-reset-mc", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    started = time.strftime("%Y%m%dT%H%M%S%z")
    out_dir = args.out_dir or repo_root / "results" / f"rebalance-capex-loop-{started}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pure_hive_path_nodes = _pure_hive_path_nodes(
        args.pure_hive_topology,
        source_node=args.intrahive_source_node,
        relay_node=args.intrahive_relay_node,
        dest_node=args.intrahive_dest_node,
    )
    prepare_pure_hive = args.pure_hive_topology != "none"
    pure_hive_member_ids: set[str] = set()
    for node in pure_hive_path_nodes:
        node_id = _node_id(node)
        if node_id:
            pure_hive_member_ids.add(node_id)
    tuning_overrides: dict[str, Any] = {}
    if args.tune_pair_fee_cap_ppm is not None:
        tuning_overrides["revenue-ops-pair-fee-cap-ppm"] = args.tune_pair_fee_cap_ppm
    if args.tune_rebalance_hold_margin is not None:
        tuning_overrides["revenue-ops-rebalance-hold-margin"] = args.tune_rebalance_hold_margin
    if args.tune_hive_bootstrap_budget_sats is not None:
        tuning_overrides["revenue-ops-hive-rebalance-bootstrap-budget-sats"] = (
            args.tune_hive_bootstrap_budget_sats
        )

    if args.hive_mode == "enabled":
        member_nodes = sorted(set(
            args.hive_member_node
            + (
                pure_hive_path_nodes
                if prepare_pure_hive else
                [
                    args.intrahive_source_node,
                    args.intrahive_relay_node,
                    args.intrahive_dest_node,
                ]
                if args.prepare_intrahive_corridor else
                []
            )
        ))
        hive_member_setup = (
            {
                node: enable_cl_hive(
                    host_path=args.cl_hive_host_path,
                    plugin_path=args.cl_hive_plugin_path,
                    hive_id=args.cl_hive_id,
                    deploy=not args.skip_cl_hive_deploy,
                    start=not args.skip_cl_hive_start,
                    genesis=not args.skip_cl_hive_genesis,
                    install_deps=args.install_cl_hive_deps,
                    node=node,
                )
                for node in member_nodes
            }
            if args.hive_start_member_plugins else
            {"skipped": True, "reason": "hive member plugin start not requested"}
        )
        hive_setup = enable_cl_hive(
            host_path=args.cl_hive_host_path,
            plugin_path=args.cl_hive_plugin_path,
            hive_id=args.cl_hive_id,
            deploy=not args.skip_cl_hive_deploy,
            start=not args.skip_cl_hive_start,
            genesis=not args.skip_cl_hive_genesis,
            install_deps=args.install_cl_hive_deps,
        )
        seed_peer_ids = list(args.hive_seed_peer_id or [])
        if prepare_pure_hive or args.prepare_intrahive_corridor:
            for node in (
                pure_hive_path_nodes
                if prepare_pure_hive else
                [
                    args.intrahive_source_node,
                    args.intrahive_relay_node,
                    args.intrahive_dest_node,
                ]
            ):
                node_id = _node_id(node)
                if node_id:
                    seed_peer_ids.append(node_id)
        hive_seed = (
            seed_hive_channel_peers(
                peer_ids=seed_peer_ids or None,
                include_channel_peers=args.hive_seed_channel_peers,
                hive_id=args.cl_hive_id,
            )
            if args.hive_seed_channel_peers or seed_peer_ids else
            {"ok": True, "skipped": True}
        )
        hive_disable = {"ok": True, "skipped": True, "reason": "hive_mode_enabled"}
        hive_config = {
            "ok": True,
            "skipped": True,
            "reason": "hive hints are non-dynamic; enabled by plugin default/config",
        }
    else:
        hive_member_setup = {"ok": True, "skipped": True, "reason": "hive_mode_disabled"}
        hive_setup = {"ok": True, "skipped": True, "reason": "hive_mode_disabled"}
        hive_seed = {"ok": True, "skipped": True, "reason": "hive_mode_disabled"}
        hive_disable = (
            disable_cl_hive()
            if not args.skip_disable_cl_hive else
            {"ok": True, "skipped": True}
        )
        hive_config = set_hive_hints_disabled()

    run_initial_corridor_prepare = (
        (prepare_pure_hive or args.prepare_intrahive_corridor)
        and not args.convergence_perturb_each_iteration
    )
    intrahive_setup = (
        prepare_corridor_for_args(
            args=args,
            pure_hive_path_nodes=pure_hive_path_nodes,
            prepare_pure_hive=prepare_pure_hive,
        )
        if run_initial_corridor_prepare else
        {
            "ok": True,
            "skipped": True,
            "reason": (
                "per_iteration_convergence_perturbation"
                if args.convergence_perturb_each_iteration else
                "not_requested"
            ),
        }
    )
    hive_refresh = (
        refresh_cl_hive_runtime()
        if args.hive_mode == "enabled" else
        {"ok": True, "skipped": True, "reason": "hive_mode_disabled"}
    )

    setup: dict[str, Any] = {
        "started": started,
        "repo_root": str(repo_root),
        "plugin_path": args.plugin_path,
        "hive_mode": args.hive_mode,
        "pure_hive_topology": args.pure_hive_topology,
        "pure_hive_path_nodes": pure_hive_path_nodes,
        "pure_hive_member_ids": sorted(pure_hive_member_ids),
        "require_pure_hive_route": args.require_pure_hive_route,
        "require_hive_member_hints": args.require_hive_member_hints,
        "deploy": (
            deploy_revenue_ops(repo_root=repo_root)
            if args.deploy else
            {"ok": True, "skipped": True}
        ),
        "disable_cl_hive": hive_disable,
        "enable_cl_hive": hive_setup,
        "enable_cl_hive_members": hive_member_setup,
        "seed_hive_channel_peers": hive_seed,
        "prepare_intrahive_corridor": intrahive_setup,
        "refresh_cl_hive_runtime": hive_refresh,
        "set_hive_hints_disabled": hive_config,
    }
    if args.restart_plugin:
        setup["restart_revenue_ops"] = restart_revenue_ops(
            plugin_path=args.plugin_path,
            wait_seconds=args.restart_wait_seconds,
        )
    setup["tuning_overrides"] = set_revenue_config_overrides(tuning_overrides)
    setup["clear_rebalance_cooldowns"] = (
        clear_rebalance_cooldowns()
        if args.clear_rebalance_cooldowns else
        {"ok": True, "skipped": True}
    )

    setup_errors: list[str] = []
    if args.hive_mode == "enabled":
        if not setup["enable_cl_hive"].get("ok", False):
            setup_errors.append("revenue cl-hive setup failed")
        if not setup["seed_hive_channel_peers"].get("ok", False):
            setup_errors.append("revenue cl-hive peer seeding failed")
        if not setup["refresh_cl_hive_runtime"].get("ok", False):
            setup_errors.append("revenue cl-hive runtime refresh failed")
        if args.hive_start_member_plugins:
            members = setup["enable_cl_hive_members"]
            for node, result in members.items():
                if not isinstance(result, dict) or not result.get("ok", False):
                    setup_errors.append(f"cl-hive member setup failed for {node}")
    if run_initial_corridor_prepare and not setup["prepare_intrahive_corridor"].get("ok", False):
        setup_errors.append("intrahive corridor preparation failed")
    if tuning_overrides and not setup["tuning_overrides"].get("ok", False):
        setup_errors.append("revenue config tuning override failed")
    if args.restart_plugin and not setup["restart_revenue_ops"].get("ok", False):
        setup_errors.append("revenue-ops restart failed")
    setup["ok"] = not setup_errors
    setup["setup_errors"] = setup_errors
    write_json(out_dir / "setup.json", setup)
    if setup_errors:
        summary = {
            "started": started,
            "out_dir": str(out_dir),
            "hive_mode": args.hive_mode,
            "require_hive_member_hints": args.require_hive_member_hints,
            "analyses": [],
            "next_action": "setup_failed",
            "next_reason": "; ".join(setup_errors),
        }
        write_json(out_dir / "loop.json", summary)
        write_analysis(out_dir / "ANALYSIS.md", [])
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    analyses: list[IterationAnalysis] = []
    for iteration in range(1, args.iterations + 1):
        iter_dir = out_dir / f"iteration_{iteration:03d}"
        perturbation: dict[str, Any] | None = None
        if args.convergence_perturb_each_iteration:
            perturbation = prepare_corridor_for_args(
                args=args,
                pure_hive_path_nodes=pure_hive_path_nodes,
                prepare_pure_hive=prepare_pure_hive,
            )
            write_json(iter_dir / "perturbation.json", perturbation)
        elif iteration == 1 and run_initial_corridor_prepare:
            perturbation = intrahive_setup

        if args.clear_rebalance_cooldowns_each_iteration:
            cooldown_clear = clear_rebalance_cooldowns()
            write_json(iter_dir / "clear_rebalance_cooldowns.json", cooldown_clear)

        traffic = maybe_drive_traffic(
            payments=args.drive_payments,
            amount_sat=args.payment_amount_sat,
            competitor_ppm=args.competitor_ppm,
            payer_time_pref=args.payer_time_pref,
            payment_timeout_seconds=args.payment_timeout_seconds,
            stop_on_failure=not args.continue_after_payment_failure,
            policy_settle_seconds=args.policy_settle_seconds if iteration == 1 else 0.0,
            reset_mission_control=not args.no_reset_mc and iteration == 1,
            out_dir=iter_dir,
        )
        before = snapshot("before", max_candidates=args.max_candidates)
        cycle = run_rebalance_cycle(args.max_candidates)
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)
        after = snapshot("after", max_candidates=args.max_candidates)
        analysis = analyze_iteration(
            iteration=iteration,
            hive_mode=args.hive_mode,
            require_hive_member_hints=args.require_hive_member_hints,
            require_pure_hive_route=args.require_pure_hive_route,
            pure_hive_member_ids=pure_hive_member_ids,
            traffic=traffic,
            before=before,
            cycle=cycle,
            after=after,
            perturbation=perturbation,
            convergence_tolerance_ratio=args.convergence_ok_error_ratio,
        )
        analyses.append(analysis)
        write_json(iter_dir / "traffic.json", traffic)
        write_json(iter_dir / "before.json", before)
        write_json(iter_dir / "cycle.json", cycle)
        write_json(iter_dir / "after.json", after)
        write_json(iter_dir / "analysis.json", asdict(analysis))

    summary = {
        "started": started,
        "out_dir": str(out_dir),
        "hive_mode": args.hive_mode,
        "pure_hive_topology": args.pure_hive_topology,
        "pure_hive_path_nodes": pure_hive_path_nodes,
        "pure_hive_member_ids": sorted(pure_hive_member_ids),
        "require_pure_hive_route": args.require_pure_hive_route,
        "require_hive_member_hints": args.require_hive_member_hints,
        "convergence_perturb_each_iteration": args.convergence_perturb_each_iteration,
        "convergence_ok_error_ratio": args.convergence_ok_error_ratio,
        "tuning_overrides": tuning_overrides,
        "analyses": [asdict(a) for a in analyses],
        "next_action": analyses[-1].action if analyses else "none",
        "next_reason": analyses[-1].reason if analyses else "no iterations ran",
    }
    write_json(out_dir / "loop.json", summary)
    write_analysis(out_dir / "ANALYSIS.md", analyses)
    latest = repo_root / "results" / "rebalance-capex-loop-latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.name)
    except OSError:
        pass
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
