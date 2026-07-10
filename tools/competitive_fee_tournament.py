#!/usr/bin/env python3
"""
WARNING: this tool calls listdatastore against live nodes. listdatastore
FULL-SCANS the datastore table synchronously on lightningd's main loop —
on a node with a bloated datastore this freezes the entire daemon for
minutes (observed 2026-07-10). Do not run against production nodes unless
the datastore is known-small.
Run competitive fee-market phases against the local Polar lab.

This runner assumes the expanded lab topology exists:

    lnd-payer-c -> revenue-node -> lnd-sink-c
    lnd-payer-c -> lnd-competitor-c -> lnd-sink-c

It records both quote-level route choice and actual forwarding counters because
LND mission control can keep using a previously successful route after policy
quotes have already flipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


CLN_DIR = "/home/clightning/.lightning"
LND_DIR = "/home/lnd/.lnd"
NETWORK = "regtest"

PAYER = "lnd-payer-c"
REVENUE = "revenue-node"
COMPETITOR = "lnd-competitor-c"
SINK = "lnd-sink-c"

DEFAULT_CL_HIVE_HOST_PATH = Path(__file__).resolve().parents[2] / "cl-hive"
DEFAULT_CL_HIVE_CONTAINER_DIR = "/tmp/cl_hive"
DEFAULT_CL_HIVE_PLUGIN_PATH = f"{DEFAULT_CL_HIVE_CONTAINER_DIR}/cl-hive.py"
DEFAULT_CL_HIVE_ID = "cl-revenue-ops-test"

_LAST_COMPETITOR_POLICY_UPDATE_MONOTONIC: float | None = None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def run(
    cmd: list[str],
    check: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "ok": False,
            "timeout": True,
            "timeout_seconds": timeout_seconds,
            "cmd": cmd,
            "stdout": _safe_text(exc.stdout),
            "stderr": _safe_text(exc.stderr),
        }
        if check:
            raise RuntimeError(json.dumps(result, indent=2))
        return result
    if proc.returncode != 0:
        result = {
            "ok": False,
            "returncode": proc.returncode,
            "cmd": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if check:
            raise RuntimeError(json.dumps(result, indent=2))
        return result
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        data = {"stdout": proc.stdout}
    if isinstance(data, dict):
        data.setdefault("ok", True)
    return data


def cln(
    node: str,
    *args: str,
    check: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"check": check}
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    return run(
        [
            "docker",
            "exec",
            f"polar-n1-{node}",
            "lightning-cli",
            f"--lightning-dir={CLN_DIR}",
            f"--network={NETWORK}",
            *args,
        ],
        **kwargs,
    )


def lnd(
    node: str,
    *args: str,
    check: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"check": check}
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    return run(
        [
            "docker",
            "exec",
            f"polar-n1-{node}",
            "lncli",
            f"--lnddir={LND_DIR}",
            f"--network={NETWORK}",
            *args,
        ],
        **kwargs,
    )


def docker_exec(
    node: str,
    *args: str,
    check: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"check": check}
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    return run(
        ["docker", "exec", f"polar-n1-{node}", *args],
        **kwargs,
    )


def docker_cp_to_node(
    source: str | Path,
    node: str,
    destination: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    return run(
        ["docker", "cp", str(source), f"polar-n1-{node}:{destination}"],
        **kwargs,
    )


def rpc_result_ok(result: dict[str, Any]) -> bool:
    return bool(result.get("ok", False)) and "error" not in result


def datastore_has_records(result: dict[str, Any]) -> bool:
    return rpc_result_ok(result) and bool(result.get("datastore"))


def _datastore_key_arg(key: list[str]) -> str:
    return json.dumps(key, separators=(",", ":"))


def push_hive_hints_datastore(node: str = REVENUE) -> dict[str, Any]:
    """Force current cl-hive hints into CLN datastore for deterministic tests."""
    exported = cln(node, "hive-export-hints")
    if not rpc_result_ok(exported):
        return {
            "ok": False,
            "stage": "hive-export-hints",
            "export_hints": exported,
        }

    payload = json.dumps(exported, separators=(",", ":"), sort_keys=True)
    pushed = cln(
        node,
        "datastore",
        f"key={_datastore_key_arg(['hive', 'hints'])}",
        f"hex={payload.encode('utf-8').hex()}",
        "mode=create-or-replace",
    )
    return {
        "ok": rpc_result_ok(pushed),
        "export_hints": exported,
        "datastore": pushed,
    }


def clear_hive_hints_datastore(node: str = REVENUE) -> dict[str, Any]:
    current = cln(node, "listdatastore", _datastore_key_arg(["hive", "hints"]))
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
            deleted.append(cln(node, *args))
    after = cln(node, "listdatastore", _datastore_key_arg(["hive", "hints"]))
    return {
        "ok": rpc_result_ok(after) and not bool(after.get("datastore")),
        "before": current,
        "deleted": deleted,
        "after": after,
    }


def disable_cl_hive(
    *,
    node: str = REVENUE,
    plugin_path: str = DEFAULT_CL_HIVE_PLUGIN_PATH,
) -> dict[str, Any]:
    status_before = cln(node, "hive-status")
    stopped = cln(node, "plugin", "stop", plugin_path)
    status_after = cln(node, "hive-status")
    cleared = clear_hive_hints_datastore(node=node)
    return {
        "ok": not rpc_result_ok(status_after) and cleared.get("ok", False),
        "status_before": status_before,
        "stop_result": stopped,
        "status_after": status_after,
        "cleared_datastore": cleared,
    }


def _result_text(result: dict[str, Any]) -> str:
    return " ".join(str(result.get(key) or "") for key in ("error", "stdout", "stderr"))


def deploy_cl_hive(
    *,
    host_path: str | Path,
    plugin_path: str = DEFAULT_CL_HIVE_PLUGIN_PATH,
    node: str = REVENUE,
) -> dict[str, Any]:
    """Copy the companion cl-hive plugin into the Polar revenue node."""
    host = Path(host_path).expanduser().resolve()
    plugin_source = host / "cl-hive.py"
    modules_source = host / "modules"
    container_dir = str(Path(plugin_path).parent)
    if not plugin_source.exists() or not modules_source.is_dir():
        return {
            "ok": False,
            "stage": "locate_cl_hive",
            "host_path": str(host),
            "reason": "expected cl-hive.py and modules/ under host path",
        }

    steps = [
        docker_exec(node, "mkdir", "-p", container_dir, f"{container_dir}/modules"),
        docker_cp_to_node(plugin_source, node, plugin_path),
        docker_cp_to_node(f"{modules_source}/.", node, f"{container_dir}/modules/"),
        docker_exec(node, "chmod", "+x", plugin_path),
    ]
    return {
        "ok": all(step.get("ok", False) for step in steps),
        "host_path": str(host),
        "plugin_path": plugin_path,
        "steps": steps,
    }


def probe_cl_hive_dependencies(node: str = REVENUE) -> dict[str, Any]:
    probe = docker_exec(
        node,
        "sh",
        "-lc",
        "command -v python3 >/dev/null && python3 -c 'import pyln.client'",
    )
    return {
        "ok": probe.get("ok", False),
        "probe": probe,
        "requires": ["python3", "pyln-client"],
    }


def install_cl_hive_dependencies(node: str = REVENUE) -> dict[str, Any]:
    """Install Python runtime dependencies in an ephemeral Polar CLN container."""
    before = probe_cl_hive_dependencies(node)
    if before.get("ok", False):
        return {"ok": True, "skipped": True, "reason": "already_available", "probe": before}

    steps = [
        docker_exec(node, "apt-get", "update"),
        docker_exec(node, "apt-get", "install", "-y", "python3", "python3-pip"),
    ]
    if all(step.get("ok", False) for step in steps):
        steps.append(
            docker_exec(
                node,
                "python3",
                "-m",
                "pip",
                "install",
                "--break-system-packages",
                "pyln-client>=24.0",
            )
        )
    after = probe_cl_hive_dependencies(node)
    return {
        "ok": after.get("ok", False),
        "before": before,
        "steps": steps,
        "after": after,
    }


def start_cl_hive(
    *,
    plugin_path: str = DEFAULT_CL_HIVE_PLUGIN_PATH,
    node: str = REVENUE,
) -> dict[str, Any]:
    """Start cl-hive if it is not already active."""
    before = cln(node, "hive-status")
    if rpc_result_ok(before):
        return {"ok": True, "skipped": True, "reason": "already_active", "status": before}

    started = cln(node, "plugin", "start", plugin_path)
    after = cln(node, "hive-status")
    return {
        "ok": rpc_result_ok(started) or rpc_result_ok(after),
        "plugin_path": plugin_path,
        "status_before": before,
        "start_result": started,
        "status_after": after,
    }


def maybe_run_hive_genesis(
    *,
    hive_id: str = DEFAULT_CL_HIVE_ID,
    node: str = REVENUE,
) -> dict[str, Any]:
    """Ensure the local cl-hive instance can export hints."""
    before = cln(node, "hive-export-hints")
    if rpc_result_ok(before):
        return {"ok": True, "skipped": True, "reason": "already_member", "export_hints": before}

    text = _result_text(before)
    if "Not a Hive member" not in text:
        return {"ok": False, "stage": "export_hints_before_genesis", "export_hints": before}

    genesis = cln(node, "hive-genesis", hive_id)
    after = cln(node, "hive-export-hints")
    genesis_text = _result_text(genesis)
    return {
        "ok": rpc_result_ok(after) or "Genesis already performed" in genesis_text,
        "hive_id": hive_id,
        "export_hints_before": before,
        "genesis": genesis,
        "export_hints_after": after,
    }


def ensure_cl_hive(
    *,
    host_path: str | Path = DEFAULT_CL_HIVE_HOST_PATH,
    plugin_path: str = DEFAULT_CL_HIVE_PLUGIN_PATH,
    deploy: bool = True,
    start: bool = True,
    genesis: bool = True,
    install_deps: bool = False,
    hive_id: str = DEFAULT_CL_HIVE_ID,
    node: str = REVENUE,
) -> dict[str, Any]:
    """Deploy and bootstrap cl-hive for Polar tournament runs."""
    setup: dict[str, Any] = {
        "enabled": True,
        "host_path": str(Path(host_path).expanduser()),
        "plugin_path": plugin_path,
        "hive_id": hive_id,
    }

    status_before = cln(node, "hive-status")
    setup["status_before"] = status_before
    active_before = rpc_result_ok(status_before)

    if deploy and not active_before:
        setup["deploy"] = deploy_cl_hive(host_path=host_path, plugin_path=plugin_path, node=node)
    else:
        setup["deploy"] = {"ok": True, "skipped": True, "reason": "disabled_or_already_active"}

    if not active_before:
        setup["dependencies"] = (
            install_cl_hive_dependencies(node) if install_deps else probe_cl_hive_dependencies(node)
        )
    else:
        setup["dependencies"] = {"ok": True, "skipped": True, "reason": "already_active"}

    if start and not active_before and setup["dependencies"].get("ok", False):
        setup["start"] = start_cl_hive(plugin_path=plugin_path, node=node)
    elif start and not active_before:
        setup["start"] = {"ok": False, "skipped": True, "reason": "dependency_probe_failed"}
    else:
        setup["start"] = {"ok": True, "skipped": True, "reason": "disabled_or_already_active"}

    if genesis:
        setup["genesis"] = maybe_run_hive_genesis(hive_id=hive_id, node=node)
    else:
        setup["genesis"] = {"ok": True, "skipped": True, "reason": "disabled"}

    setup["push_datastore"] = push_hive_hints_datastore(node=node)
    setup["status_after"] = cln(node, "hive-status")
    setup["export_hints_after"] = cln(node, "hive-export-hints")
    setup["ok"] = (
        rpc_result_ok(setup["status_after"])
        and rpc_result_ok(setup["export_hints_after"])
        and bool(setup["push_datastore"].get("ok", False))
    )
    return setup


def collect_hive_snapshot(enabled: bool, node: str = REVENUE) -> dict[str, Any]:
    if not enabled:
        return {"ok": True, "skipped": True, "reason": "cl-hive disabled"}

    status = cln(node, "hive-status")
    export_hints = cln(node, "hive-export-hints")
    datastore = cln(node, "listdatastore", '["hive","hints"]')
    return {
        "ok": rpc_result_ok(status) and (rpc_result_ok(export_hints) or datastore_has_records(datastore)),
        "status": status,
        "export_hints": export_hints,
        "datastore_hints": datastore,
    }


def set_lnd_policy(node: str, ppm: int, base_fee_msat: int = 1, cltv_delta: int = 40) -> dict[str, Any]:
    return lnd(
        node,
        "updatechanpolicy",
        "--base_fee_msat",
        str(base_fee_msat),
        "--fee_rate_ppm",
        str(ppm),
        "--time_lock_delta",
        str(cltv_delta),
    )


def int_field(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def graph_policy_matches(
    policy: dict[str, Any],
    *,
    ppm: int,
    base_fee_msat: int,
    cltv_delta: int,
) -> bool:
    return (
        int_field(policy.get("fee_rate_milli_msat")) == ppm
        and int_field(policy.get("fee_base_msat")) == base_fee_msat
        and int_field(policy.get("time_lock_delta")) == cltv_delta
        and policy.get("disabled") is not True
    )


def node_policy(chan_info: dict[str, Any], node_pubkey: str) -> dict[str, Any]:
    if chan_info.get("node1_pub") == node_pubkey:
        return chan_info.get("node1_policy") or {}
    if chan_info.get("node2_pub") == node_pubkey:
        return chan_info.get("node2_policy") or {}
    return {}


def cln_direction_policy(chan_info: dict[str, Any], source: str, destination: str) -> dict[str, Any]:
    for channel in chan_info.get("channels", []) if isinstance(chan_info, dict) else []:
        if channel.get("source") == source and channel.get("destination") == destination:
            return channel
    return {}


def cln_graph_policy_matches(
    policy: dict[str, Any],
    *,
    ppm: int,
    base_fee_msat: int,
    cltv_delta: int,
) -> bool:
    return (
        int_field(policy.get("fee_per_millionth")) == ppm
        and int_field(policy.get("base_fee_millisatoshi")) == base_fee_msat
        and int_field(policy.get("delay")) == cltv_delta
        and policy.get("active") is True
    )


def competitor_sink_scid(sink_pubkey: str) -> str:
    result = lnd(COMPETITOR, "listchannels")
    for channel in result.get("channels", []) if isinstance(result, dict) else []:
        if channel.get("remote_pubkey") == sink_pubkey:
            return str(channel.get("scid") or "")
    return ""


def competitor_sink_channel_ids(sink_pubkey: str) -> dict[str, str]:
    result = lnd(COMPETITOR, "listchannels")
    for channel in result.get("channels", []) if isinstance(result, dict) else []:
        if channel.get("remote_pubkey") == sink_pubkey:
            return {
                "lnd_channel_id": str(channel.get("scid") or ""),
                "short_channel_id": str(channel.get("scid_str") or ""),
            }
    return {"lnd_channel_id": "", "short_channel_id": ""}


def wait_for_competitor_policy_update_window(
    *,
    context: dict[str, str],
    min_interval_seconds: float,
) -> dict[str, Any]:
    """Avoid scripted LND policy churn that remote LND nodes rate-limit."""
    global _LAST_COMPETITOR_POLICY_UPDATE_MONOTONIC

    if min_interval_seconds <= 0:
        return {"ok": True, "skipped": True, "reason": "policy update interval disabled"}

    channel_ids = competitor_sink_channel_ids(context["sink_pubkey"])
    lnd_channel_id = channel_ids["lnd_channel_id"]
    payer_result: dict[str, Any] = {}
    payer_policy: dict[str, Any] = {}
    last_graph_update = 0
    if lnd_channel_id:
        payer_result = lnd(PAYER, "getchaninfo", lnd_channel_id)
        if payer_result.get("ok", False):
            payer_policy = node_policy(payer_result, context["competitor_pubkey"])
            last_graph_update = int_field(payer_policy.get("last_update"), 0)

    now_wall = time.time()
    wall_wait = (
        max(0.0, min_interval_seconds - max(0.0, now_wall - float(last_graph_update)))
        if last_graph_update > 0 else
        0.0
    )
    monotonic_wait = (
        max(0.0, min_interval_seconds - (time.monotonic() - _LAST_COMPETITOR_POLICY_UPDATE_MONOTONIC))
        if _LAST_COMPETITOR_POLICY_UPDATE_MONOTONIC is not None else
        0.0
    )
    wait_seconds = max(wall_wait, monotonic_wait)
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    return {
        "ok": True,
        "min_interval_seconds": min_interval_seconds,
        "wait_seconds": wait_seconds,
        "wall_wait_seconds": wall_wait,
        "monotonic_wait_seconds": monotonic_wait,
        "lnd_channel_id": lnd_channel_id,
        "short_channel_id": channel_ids["short_channel_id"],
        "last_graph_update": last_graph_update,
        "payer_policy": payer_policy,
        "payer_result_ok": payer_result.get("ok", False) if payer_result else False,
    }


def wait_for_competitor_graph_policy(
    *,
    context: dict[str, str],
    ppm: int,
    base_fee_msat: int = 1,
    cltv_delta: int,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    """Wait until route and controller graph views have the competitor policy."""
    if timeout_seconds <= 0:
        return {"ok": True, "skipped": True, "reason": "policy graph verification disabled"}

    channel_ids = competitor_sink_channel_ids(context["sink_pubkey"])
    lnd_channel_id = channel_ids["lnd_channel_id"]
    short_channel_id = channel_ids["short_channel_id"]
    if not lnd_channel_id or not short_channel_id:
        return {
            "ok": False,
            "stage": "find_competitor_sink_channel",
            "reason": "competitor->sink channel not found",
        }

    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_payer_policy: dict[str, Any] = {}
    last_revenue_policy: dict[str, Any] = {}
    last_payer_result: dict[str, Any] = {}
    last_revenue_result: dict[str, Any] = {}
    while True:
        attempts += 1
        payer_result = lnd(PAYER, "getchaninfo", lnd_channel_id)
        revenue_result = cln(REVENUE, "listchannels", short_channel_id)
        last_payer_result = payer_result
        last_revenue_result = revenue_result
        payer_ok = False
        revenue_ok = False
        if payer_result.get("ok", False):
            last_payer_policy = node_policy(payer_result, context["competitor_pubkey"])
            payer_ok = graph_policy_matches(
                last_payer_policy,
                ppm=ppm,
                base_fee_msat=base_fee_msat,
                cltv_delta=cltv_delta,
            )
        if revenue_result.get("ok", False):
            last_revenue_policy = cln_direction_policy(
                revenue_result,
                context["competitor_pubkey"],
                context["sink_pubkey"],
            )
            revenue_ok = cln_graph_policy_matches(
                last_revenue_policy,
                ppm=ppm,
                base_fee_msat=base_fee_msat,
                cltv_delta=cltv_delta,
            )
        if payer_ok and revenue_ok:
                return {
                    "ok": True,
                    "channel_scid": short_channel_id,
                    "lnd_channel_id": lnd_channel_id,
                    "attempts": attempts,
                    "policy": last_payer_policy,
                    "payer_policy": last_payer_policy,
                    "revenue_policy": last_revenue_policy,
                    "payer_last_update": payer_result.get("last_update"),
                    "revenue_last_update": last_revenue_policy.get("last_update"),
                }

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "ok": False,
                "stage": "wait_for_graph_policy",
                "channel_scid": short_channel_id,
                "lnd_channel_id": lnd_channel_id,
                "attempts": attempts,
                "expected": {
                    "fee_rate_milli_msat": ppm,
                    "fee_base_msat": base_fee_msat,
                    "time_lock_delta": cltv_delta,
                },
                "payer_ok": payer_ok,
                "revenue_ok": revenue_ok,
                "last_policy": last_payer_policy,
                "last_payer_policy": last_payer_policy,
                "last_revenue_policy": last_revenue_policy,
                "last_payer_result": last_payer_result,
                "last_revenue_result": last_revenue_result,
            }
        time.sleep(min(poll_seconds, remaining))


def forwards_count_cln(node: str, since: int) -> int:
    result = cln(node, "listforwards")
    forwards = result.get("forwards", []) if isinstance(result, dict) else []
    return sum(
        1
        for fwd in forwards
        if fwd.get("status") == "settled" and float(fwd.get("received_time", 0)) >= since
    )


def forwards_count_lnd(node: str, since: int) -> int:
    result = lnd(node, "fwdinghistory", "--start_time", str(since))
    events = result.get("forwarding_events", []) if isinstance(result, dict) else []
    return len(events)


def total_forwards_cln(node: str) -> int:
    result = cln(node, "listforwards")
    forwards = result.get("forwards", []) if isinstance(result, dict) else []
    return sum(1 for fwd in forwards if fwd.get("status") == "settled")


def total_forwards_lnd(node: str) -> int:
    result = lnd(node, "fwdinghistory")
    events = result.get("forwarding_events", []) if isinstance(result, dict) else []
    return len(events)


def best_route(
    amount_sat: int,
    sink_pubkey: str,
    revenue_pubkey: str,
    competitor_pubkey: str,
    payer_time_pref: float = 0.0,
) -> dict[str, Any]:
    quote = lnd(
        PAYER,
        "queryroutes",
        sink_pubkey,
        str(amount_sat),
        "--fee_limit",
        "1000",
        "--time_pref",
        str(payer_time_pref),
    )
    routes = quote.get("routes", []) if isinstance(quote, dict) else []
    if not routes:
        return {"route": "none", "quote": quote}

    route = routes[0]
    hops = route.get("hops", [])
    first_hop = hops[0].get("pub_key", "") if hops else ""
    if first_hop == revenue_pubkey:
        route_name = "revenue"
    elif first_hop == competitor_pubkey:
        route_name = "competitor"
    else:
        route_name = "unknown"
    return {
        "route": route_name,
        "first_hop": first_hop,
        "total_fees_msat": route.get("total_fees_msat"),
        "total_amt_msat": route.get("total_amt_msat"),
        "hops": [
            {
                "pub_key": hop.get("pub_key"),
                "chan_id": hop.get("chan_id"),
                "fee_msat": hop.get("fee_msat"),
                "amt_to_forward_msat": hop.get("amt_to_forward_msat"),
            }
            for hop in hops
        ],
    }


def payment_succeeded(result: dict[str, Any]) -> bool:
    if result.get("ok") is False:
        return False

    status = str(result.get("status") or result.get("payment_status") or "").upper()
    if status in {"SUCCEEDED", "SUCCESS"}:
        return True
    if status in {"FAILED", "FAILURE", "IN_FLIGHT"}:
        return False

    stdout = str(result.get("stdout") or "").upper()
    if "PAYMENT STATUS: SUCCEEDED" in stdout:
        return True
    if "PAYMENT STATUS: FAILED" in stdout or "[LNCLI] FAILED" in stdout:
        return False

    if result.get("payment_error"):
        return False

    return result.get("ok") is True and result.get("returncode", 0) == 0


def pay_one(
    amount_sat: int,
    label: str,
    payer_time_pref: float = 0.0,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    invoice = lnd(
        SINK,
        "addinvoice",
        "--amt",
        str(amount_sat),
        "--memo",
        label,
        timeout_seconds=timeout_seconds,
    )
    payment_request = invoice.get("payment_request")
    if not payment_request:
        return {"ok": False, "stage": "invoice", "label": label, "result": invoice}

    paid = lnd(
        PAYER,
        "payinvoice",
        "--force",
        "--time_pref",
        str(payer_time_pref),
        payment_request,
        timeout_seconds=timeout_seconds,
    )
    ok = payment_succeeded(paid)
    return {
        "ok": ok,
        "stage": "pay",
        "label": label,
        "amount_sat": amount_sat,
        "result": paid,
    }


def force_fee_cycle(plugin_path: str, wait_seconds: int) -> dict[str, Any]:
    cycle = cln(REVENUE, "revenue-fee-cycle")
    if cycle.get("ok", False) and "error" not in cycle:
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        return {"method": "revenue-fee-cycle", "result": cycle, "wait_seconds": wait_seconds}

    stopped = cln(REVENUE, "plugin", "stop", plugin_path)
    started = cln(REVENUE, "plugin", "start", plugin_path)
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    return {
        "method": "plugin_restart",
        "cycle_attempt": cycle,
        "stopped": stopped,
        "started": started,
        "wait_seconds": wait_seconds,
    }


def collect_static_context() -> dict[str, str]:
    return {
        "revenue_pubkey": cln(REVENUE, "getinfo").get("id", ""),
        "competitor_pubkey": lnd(COMPETITOR, "getinfo").get("identity_pubkey", ""),
        "sink_pubkey": lnd(SINK, "getinfo").get("identity_pubkey", ""),
    }


def run_phase(
    *,
    name: str,
    rounds: int,
    amount_sat: int,
    competitor_ppm: int,
    reset_mc: bool,
    policy_settle_seconds: float,
    post_payment_settle_seconds: float,
    force_cycle_before: bool,
    force_cycle_after: bool,
    cycle_wait: int,
    plugin_path: str,
    out_dir: Path,
    competitor_cltv_delta: int = 40,
    payer_time_pref: float = 0.0,
    set_competitor_policy: bool = True,
    competitor_controller: str = "scripted",
    policy_verify_timeout_seconds: float = 30.0,
    policy_min_update_interval_seconds: float = 0.0,
    with_cl_hive: bool = False,
) -> dict[str, Any]:
    started = int(time.time())
    context = collect_static_context()
    policy_update_window = (
        wait_for_competitor_policy_update_window(
            context=context,
            min_interval_seconds=policy_min_update_interval_seconds,
        )
        if set_competitor_policy else
        {"ok": True, "skipped": True}
    )
    policy_result = (
        set_lnd_policy(COMPETITOR, competitor_ppm, cltv_delta=competitor_cltv_delta)
        if set_competitor_policy else
        {
            "ok": True,
            "skipped": True,
            "reason": "competitor policy managed outside this runner",
            "competitor_controller": competitor_controller,
        }
    )
    if set_competitor_policy and policy_result.get("ok", False):
        global _LAST_COMPETITOR_POLICY_UPDATE_MONOTONIC
        _LAST_COMPETITOR_POLICY_UPDATE_MONOTONIC = time.monotonic()
    if set_competitor_policy and not policy_result.get("ok", False):
        raise RuntimeError(
            json.dumps(
                {
                    "stage": "policy_update",
                    "policy_update_window": policy_update_window,
                    "result": policy_result,
                },
                indent=2,
            )
        )
    reset_result = lnd(PAYER, "resetmc") if reset_mc else {"ok": True, "skipped": True}
    if policy_settle_seconds > 0:
        time.sleep(policy_settle_seconds)
    policy_graph_result = (
        wait_for_competitor_graph_policy(
            context=context,
            ppm=competitor_ppm,
            base_fee_msat=1,
            cltv_delta=competitor_cltv_delta,
            timeout_seconds=policy_verify_timeout_seconds,
        )
        if set_competitor_policy else
        {"ok": True, "skipped": True}
    )
    if set_competitor_policy and not policy_graph_result.get("ok", False):
        raise RuntimeError(
            json.dumps(
                {
                    "stage": "policy_graph_verify",
                    "policy_update_window": policy_update_window,
                    "policy_result": policy_result,
                    "policy_graph_result": policy_graph_result,
                },
                indent=2,
            )
        )
    before_cycle = (
        force_fee_cycle(plugin_path, cycle_wait) if force_cycle_before else {"ok": True, "skipped": True}
    )
    revenue_forwards_before = total_forwards_cln(REVENUE)
    competitor_forwards_before = total_forwards_lnd(COMPETITOR)

    before = {
        "route": best_route(
            amount_sat,
            context["sink_pubkey"],
            context["revenue_pubkey"],
            context["competitor_pubkey"],
            payer_time_pref,
        ),
        "revenue_forwards_total": revenue_forwards_before,
        "competitor_forwards_total": competitor_forwards_before,
        "revenue_forwards_since_start": 0,
        "competitor_forwards_since_start": 0,
        "revenue_fee_debug": cln(REVENUE, "revenue-fee-debug"),
        "cl_hive": collect_hive_snapshot(with_cl_hive),
        "lnd_mission_control": lnd(PAYER, "querymc"),
    }

    payments: list[dict[str, Any]] = []
    route_quotes: list[dict[str, Any]] = []
    for index in range(rounds):
        label = f"tournament-{name}-{started}-{index}"
        route_quotes.append(
            best_route(
                amount_sat,
                context["sink_pubkey"],
                context["revenue_pubkey"],
                context["competitor_pubkey"],
                payer_time_pref,
            )
        )
        result = pay_one(amount_sat, label, payer_time_pref)
        payments.append(result)
        if not result.get("ok"):
            break
        time.sleep(0.25)
    if post_payment_settle_seconds > 0:
        time.sleep(post_payment_settle_seconds)

    after_cycle = (
        force_fee_cycle(plugin_path, cycle_wait) if force_cycle_after else {"ok": True, "skipped": True}
    )
    revenue_forwards_after = total_forwards_cln(REVENUE)
    competitor_forwards_after = total_forwards_lnd(COMPETITOR)
    after = {
        "route": best_route(
            amount_sat,
            context["sink_pubkey"],
            context["revenue_pubkey"],
            context["competitor_pubkey"],
            payer_time_pref,
        ),
        "revenue_forwards_total": revenue_forwards_after,
        "competitor_forwards_total": competitor_forwards_after,
        "revenue_forwards_since_start": max(0, revenue_forwards_after - revenue_forwards_before),
        "competitor_forwards_since_start": max(0, competitor_forwards_after - competitor_forwards_before),
        "revenue_fee_debug": cln(REVENUE, "revenue-fee-debug"),
        "revenue_status": cln(REVENUE, "revenue-status"),
        "cl_hive": collect_hive_snapshot(with_cl_hive),
        "lnd_mission_control": lnd(PAYER, "querymc"),
    }

    phase = {
        "name": name,
        "started": started,
        "amount_sat": amount_sat,
        "rounds": rounds,
        "competitor_ppm": competitor_ppm,
        "competitor_cltv_delta": competitor_cltv_delta,
        "payer_time_pref": payer_time_pref,
        "competitor_controller": competitor_controller,
        "set_competitor_policy": set_competitor_policy,
        "reset_mc": reset_mc,
        "policy_min_update_interval_seconds": policy_min_update_interval_seconds,
        "policy_settle_seconds": policy_settle_seconds,
        "post_payment_settle_seconds": post_payment_settle_seconds,
        "pubkeys": context,
        "policy_update_window": policy_update_window,
        "policy_result": policy_result,
        "policy_graph_result": policy_graph_result,
        "reset_result": reset_result,
        "before_cycle": before_cycle,
        "before": before,
        "route_quotes": route_quotes,
        "payments": payments,
        "payments_attempted": len(payments),
        "payments_succeeded": sum(1 for payment in payments if payment.get("ok")),
        "payments_failed": [payment for payment in payments if not payment.get("ok")],
        "after_cycle": after_cycle,
        "after": after,
    }
    out_path = out_dir / f"phase_{name}_{started}.json"
    out_path.write_text(json.dumps(phase, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    phase["output_path"] = str(out_path)
    return phase


def parse_phase_spec(spec: str) -> tuple[str, int, bool]:
    """Parse ``name:ppm:resetmc`` phase specs."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("phase must be name:competitor_ppm[:resetmc]")
    name = parts[0]
    try:
        ppm = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("competitor_ppm must be an integer") from exc
    reset_mc = len(parts) >= 3 and parts[2].lower() in {"1", "true", "yes", "resetmc"}
    return name, ppm, reset_mc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--amount-sat", type=int, default=20_000)
    parser.add_argument("--cycle-wait", type=int, default=0)
    parser.add_argument("--plugin-path", default="/tmp/cl_revenue_ops/cl-revenue-ops.py")
    parser.add_argument("--with-cl-hive", action="store_true")
    parser.add_argument("--cl-hive-host-path", type=Path, default=DEFAULT_CL_HIVE_HOST_PATH)
    parser.add_argument("--cl-hive-plugin-path", default=DEFAULT_CL_HIVE_PLUGIN_PATH)
    parser.add_argument("--cl-hive-id", default=DEFAULT_CL_HIVE_ID)
    parser.add_argument("--install-cl-hive-deps", action="store_true")
    parser.add_argument("--skip-cl-hive-deploy", action="store_true")
    parser.add_argument("--skip-cl-hive-start", action="store_true")
    parser.add_argument("--skip-cl-hive-genesis", action="store_true")
    parser.add_argument("--policy-settle-seconds", type=float, default=12.0)
    parser.add_argument("--policy-verify-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--policy-min-update-interval-seconds",
        type=float,
        default=75.0,
        help="Minimum spacing between scripted competitor policy updates to avoid LND gossip rate limits.",
    )
    parser.add_argument("--post-payment-settle-seconds", type=float, default=2.0)
    parser.add_argument("--competitor-cltv-delta", type=int, default=40)
    parser.add_argument("--payer-time-pref", type=float, default=0.0)
    parser.add_argument(
        "--phase",
        action="append",
        type=parse_phase_spec,
        help="Phase spec name:competitor_ppm[:resetmc]. May be repeated.",
    )
    parser.add_argument("--force-cycle-before", action="store_true")
    parser.add_argument("--force-cycle-after", action="store_true")
    parser.add_argument(
        "--skip-policy-updates",
        action="store_true",
        help="Do not set LND competitor policy; use when an external controller manages competitors.",
    )
    parser.add_argument(
        "--competitor-controller",
        default="scripted",
        help="Metadata label for competitor controller, e.g. scripted, clboss, external.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    phases = args.phase or [
        ("baseline_150", 150, False),
        ("boundary_80_resetmc", 80, True),
    ]

    started = int(time.time())
    cl_hive_setup = (
        ensure_cl_hive(
            host_path=args.cl_hive_host_path,
            plugin_path=args.cl_hive_plugin_path,
            deploy=not args.skip_cl_hive_deploy,
            start=not args.skip_cl_hive_start,
            genesis=not args.skip_cl_hive_genesis,
            install_deps=args.install_cl_hive_deps,
            hive_id=args.cl_hive_id,
        )
        if args.with_cl_hive else
        {"enabled": False}
    )
    results = []
    for name, ppm, reset_mc in phases:
        results.append(
            run_phase(
                name=name,
                rounds=args.rounds,
                amount_sat=args.amount_sat,
                competitor_ppm=ppm,
                reset_mc=reset_mc,
                policy_settle_seconds=args.policy_settle_seconds,
                post_payment_settle_seconds=args.post_payment_settle_seconds,
                force_cycle_before=args.force_cycle_before,
                force_cycle_after=args.force_cycle_after,
                cycle_wait=args.cycle_wait,
                plugin_path=args.plugin_path,
                competitor_cltv_delta=args.competitor_cltv_delta,
                payer_time_pref=args.payer_time_pref,
                set_competitor_policy=not args.skip_policy_updates,
                competitor_controller=args.competitor_controller,
                policy_verify_timeout_seconds=args.policy_verify_timeout_seconds,
                policy_min_update_interval_seconds=args.policy_min_update_interval_seconds,
                with_cl_hive=args.with_cl_hive,
                out_dir=args.out_dir,
            )
        )

    aggregate = {
        "started": started,
        "rounds": args.rounds,
        "amount_sat": args.amount_sat,
        "cycle_wait": args.cycle_wait,
        "cl_hive_setup": cl_hive_setup,
        "phases": results,
    }
    aggregate_path = args.out_dir / f"tournament_{started}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(aggregate_path), "phases": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
