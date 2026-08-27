#!/usr/bin/env python3
"""Provision and drive a restartable CLBOSS competition in Polar network 4.

The runner adds two fresh, equal-version CLN containers to the *existing*
mixed-client network.  Polar MCP remains the only traffic generator; Docker
is used for the two external contenders and for exact RPC readback from the
existing nodes.  Mutating commands require ``--apply`` and every completed
mutation is checkpointed before the next one begins.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from polar_mixed_client_lab import (  # noqa: E402
    PolarMcp,
    PolarMcpError,
    select_traffic_lanes,
)


SCHEMA = "polar-clboss-runner-state-v1"
IMAGE = "cl-revenue-ops-polar-clboss:9d9ed85"
NETWORK_ID = 4
DOCKER_NETWORK = "polar-network-4_default"
BACKEND = "polar-n4-backend1"
REVENUE_PLUGIN = "/opt/cl_revenue_ops/cl-revenue-ops-polar-wrapper"
CLBOSS_PLUGIN = "/usr/local/libexec/clboss"
XREBALANCE_PLUGIN = "/usr/local/libexec/xrebalance"
IDENTITIES = ("identity-a", "identity-b")
CHANNEL_CAPACITY_SATS = 1_000_000
# Covers the 1% reserve on a 1M channel plus route fees.  A smaller fee-only
# buffer still leaves the sink unable to spend the newly received balance.
REVERSE_FEE_BUFFER_SATS = 25_000
CONTAINER_RE = re.compile(r"^polar-n[1-9][0-9]*-clboss-r[1-9][0-9]*-identity-[ab]$")
EXPECTED_ORIGINALS = (
    "backend1",
    "revenue-node",
    "cln-competitor",
    "lnd-competitor",
    "cln-payer",
    "lnd-payer",
    "cln-sink",
    "lnd-sink",
)


class RunnerError(RuntimeError):
    """A tournament invariant or external command failed."""


class ReconciliationError(RunnerError):
    """A payment was dispatched but could not be proven settled or failed."""

    def __init__(self, message: str, records: list[dict[str, Any]], operation: dict[str, Any]):
        super().__init__(message)
        self.records = records
        self.operation = operation


def positive_int(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def assignment_for(replica: int) -> dict[str, str]:
    """Cross identities deterministically between replicas."""
    positive_int(replica)
    revenue_identity = IDENTITIES[(replica - 1) % 2]
    clboss_identity = IDENTITIES[replica % 2]
    return {"revenue_ops": revenue_identity, "clboss": clboss_identity}


def container_name(network_id: int, replica: int, identity: str) -> str:
    positive_int(network_id)
    positive_int(replica)
    if identity not in IDENTITIES:
        raise RunnerError(f"unknown contender identity: {identity!r}")
    name = f"polar-n{network_id}-clboss-r{replica}-{identity}"
    if not CONTAINER_RE.fullmatch(name):
        raise RunnerError(f"unsafe contender container name: {name!r}")
    return name


def state_path(results_dir: Path, replica: int) -> Path:
    return results_dir / f"replica-{positive_int(replica)}" / "state.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read runner state {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise RunnerError(f"invalid runner state schema in {path}")
    return payload


def _run(
    command: Sequence[str], *, check: bool = True, timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command), check=check, text=True, capture_output=True, timeout=timeout
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RunnerError(f"command failed ({exc.returncode}): {command!r}: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"command timed out: {command!r}") from exc


def _json(command: Sequence[str], *, timeout: float = 120) -> dict[str, Any]:
    completed = _run(command, timeout=timeout)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"command returned non-JSON: {command!r}") from exc
    if not isinstance(payload, dict):
        raise RunnerError(f"command returned a non-object: {command!r}")
    return payload


def docker_running(name: str) -> bool:
    completed = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name], check=False
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def cln_rpc(container: str, *arguments: object) -> dict[str, Any]:
    if not (container.startswith(f"polar-n{NETWORK_ID}-") or CONTAINER_RE.fullmatch(container)):
        raise RunnerError(f"refusing unsafe CLN target {container!r}")
    user = [] if CONTAINER_RE.fullmatch(container) else ["-u", "clightning"]
    return _json(
        ["docker", "exec", *user, container, "lightning-cli", "--network=regtest"]
        + [str(argument) for argument in arguments]
    )


def lnd_rpc(container: str, *arguments: object) -> dict[str, Any]:
    allowed = {
        f"polar-n{NETWORK_ID}-lnd-payer",
        f"polar-n{NETWORK_ID}-lnd-sink",
        f"polar-n{NETWORK_ID}-lnd-competitor",
    }
    if container not in allowed:
        raise RunnerError(f"refusing unsafe LND target {container!r}")
    return _json(
        ["docker", "exec", "-u", "lnd", container, "lncli", "--network=regtest"]
        + [str(argument) for argument in arguments]
    )


def network_record(bridge: PolarMcp, network_id: int) -> dict[str, Any]:
    rows = bridge.call("list_networks", {}).get("networks")
    if not isinstance(rows, list):
        raise RunnerError("Polar MCP list_networks returned no network list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == network_id]
    if len(matches) != 1:
        raise RunnerError(f"Polar network {network_id} was not found exactly once")
    return matches[0]


def preflight(bridge: PolarMcp, network_id: int, image: str) -> dict[str, Any]:
    health = bridge.health()
    network = network_record(bridge, network_id)
    if str(network.get("status", "")).casefold() != "started":
        raise RunnerError(f"Polar network {network_id} is not started")
    other_running = []
    for row in bridge.call("list_networks", {}).get("networks", []):
        if (
            isinstance(row, dict)
            and row.get("id") != network_id
            and str(row.get("status", "")).casefold() in {"started", "starting"}
        ):
            other_running.append(row.get("id"))
    if other_running:
        raise RunnerError(f"other Polar networks are running: {other_running}")

    runtime: dict[str, str] = {}
    for role in EXPECTED_ORIGINALS:
        name = f"polar-n{network_id}-{role}"
        if not docker_running(name):
            raise RunnerError(f"required original container is not running: {name}")
        runtime[role] = name
    inspected = json.loads(_run(["docker", "image", "inspect", image]).stdout)
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise RunnerError(f"image inspect was malformed for {image}")
    image_labels = (inspected[0].get("Config") or {}).get("Labels") or {}
    if not image_labels.get("org.opencontainers.image.revision.revenue_ops"):
        raise RunnerError("competition image lacks a pinned revenue_ops revision label")
    return {
        "polar_health": health,
        "network_id": network_id,
        "network_name": network.get("name"),
        "polar_metadata_statuses": {
            str(node.get("name")): str(node.get("status"))
            for node in network.get("nodes", {}).get("lightning", [])
            if isinstance(node, dict)
        },
        "runtime_containers": runtime,
        "image": image,
        "image_id": inspected[0].get("Id"),
        "image_labels": image_labels,
    }


def _checkpoint(path: Path, state: dict[str, Any], event: str, **details: Any) -> None:
    state.setdefault("events", []).append(
        {"at": int(time.time()), "event": event, **details}
    )
    state["updated_at"] = int(time.time())
    write_json_atomic(path, state)


def launch_contender(
    *, name: str, identity: str, data_dir: Path, image: str, network_id: int
) -> None:
    if docker_running(name) or _run(["docker", "inspect", name], check=False).returncode == 0:
        raise RunnerError(f"fresh-only setup refused because {name} already exists")
    data_dir.mkdir(parents=True, exist_ok=False)
    # The official entrypoint watches this directory before foregrounding
    # lightningd, so it must exist on a fresh bind mount.
    (data_dir / "regtest").mkdir()
    _run(
        [
            "docker", "run", "--detach", "--name", name,
            "--network", f"polar-network-{network_id}_default",
            "--network-alias", identity,
            "--volume", f"{data_dir.resolve()}:/root/.lightning",
            "--env", "LIGHTNINGD_NETWORK=regtest",
            image,
            f"--alias={identity}",
            f"--addr={identity}:9735",
            "--bitcoin-rpcuser=polaruser",
            "--bitcoin-rpcpassword=polarpass",
            f"--bitcoin-rpcconnect=polar-n{network_id}-backend1",
            "--bitcoin-rpcport=18443",
            "--log-level=debug",
            "--dev-bitcoind-poll=2",
            "--dev-fast-gossip",
            "--developer",
        ]
    )


def wait_cln(container: str, timeout_seconds: float = 90) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last = "not ready"
    while time.monotonic() < deadline:
        try:
            result = cln_rpc(container, "getinfo")
            if result.get("id"):
                return result
            last = f"getinfo lacked id: {result}"
        except RunnerError as exc:
            last = str(exc)
        time.sleep(1)
    raise RunnerError(f"{container} did not become RPC-ready: {last}")


def _mine(bridge: PolarMcp, network_id: int, blocks: int = 6) -> None:
    bridge.call("mine_blocks", {"networkId": network_id, "blocks": blocks})


def onchain_address(payload: dict[str, Any]) -> str:
    """Choose CLN's preferred regtest address across old/new RPC shapes."""
    for key in ("p2tr", "bech32"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith("bcrt1"):
            return value
    raise RunnerError("newaddr returned no regtest segwit address")


def _fund_wallet(container: str, bridge: PolarMcp, network_id: int) -> str:
    addresses = [onchain_address(cln_rpc(container, "newaddr")) for _ in range(2)]
    # Two independent confirmed UTXOs let the contender fund its two outbound
    # channels without relying on unconfirmed transaction change.
    for address in addresses:
        _run(
            [
                "docker", "exec", f"polar-n{network_id}-backend1",
                "bitcoin-cli", "-regtest", "-rpcuser=polaruser", "-rpcpassword=polarpass",
                "sendtoaddress", address, "0.01025",
            ]
        )
    _mine(bridge, network_id, 6)
    return addresses[0]


def wait_wallet_funds(
    container: str, minimum_sats: int = 2_000_000, timeout_seconds: float = 120
) -> None:
    """Wait for a newly funded v26 wallet to catch up to the mined block."""
    deadline = time.monotonic() + timeout_seconds
    last = 0
    while time.monotonic() < deadline:
        outputs = cln_rpc(container, "listfunds").get("outputs", [])
        if isinstance(outputs, list):
            last = sum(
                msat_value(row.get("amount_msat", 0)) // 1000
                for row in outputs
                if isinstance(row, dict) and row.get("status") == "confirmed"
            )
            if last >= minimum_sats:
                return
        time.sleep(1)
    raise RunnerError(
        f"{container} wallet did not expose {minimum_sats} confirmed sats; saw {last}"
    )


def _connect_cln(source: str, peer_id: str, host: str) -> None:
    try:
        cln_rpc(source, "connect", peer_id, host, 9735)
    except RunnerError as exc:
        if "already connected" not in str(exc).casefold():
            raise


def _connect_lnd(source: str, peer_id: str, host: str) -> None:
    try:
        lnd_rpc(source, "connect", f"{peer_id}@{host}:9735", "--perm")
    except RunnerError as exc:
        if "already connected" not in str(exc).casefold():
            raise


def _open_channels(
    bridge: PolarMcp,
    network_id: int,
    contenders: dict[str, dict[str, str]],
    on_open: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    cln_payer = f"polar-n{network_id}-cln-payer"
    lnd_payer = f"polar-n{network_id}-lnd-payer"
    cln_sink = f"polar-n{network_id}-cln-sink"
    lnd_sink = f"polar-n{network_id}-lnd-sink"
    peers = {
        "cln_sink": str(cln_rpc(cln_sink, "getinfo")["id"]),
        "lnd_sink": str(lnd_rpc(lnd_sink, "getinfo")["identity_pubkey"]),
    }
    opened: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> None:
        opened.append(row)
        if on_open is not None:
            on_open(row)
        # CLN will not spend the unconfirmed change from the first 1M channel
        # into the second.  Confirmation here also removes mixed-client
        # funding races before the next mutation is dispatched.
        _mine(bridge, network_id, 6)

    for identity in IDENTITIES:
        contender = contenders[identity]
        node_id = contender["node_id"]
        name = contender["container"]
        _connect_cln(cln_payer, node_id, identity)
        cln_open = cln_rpc(cln_payer, "fundchannel", node_id, CHANNEL_CAPACITY_SATS)
        record({"funder": "cln-payer", "identity": identity, "result": cln_open})

        _connect_lnd(lnd_payer, node_id, identity)
        lnd_open = lnd_rpc(
            lnd_payer, "openchannel", "--node_key", node_id,
            "--local_amt", CHANNEL_CAPACITY_SATS,
        )
        record({"funder": "lnd-payer", "identity": identity, "result": lnd_open})

        _connect_cln(name, peers["cln_sink"], "cln-sink")
        cln_sink_open = cln_rpc(name, "fundchannel", peers["cln_sink"], CHANNEL_CAPACITY_SATS)
        record({"funder": identity, "sink": "cln-sink", "result": cln_sink_open})
        wait_wallet_funds(name, minimum_sats=1_000_000)

        _connect_cln(name, peers["lnd_sink"], "lnd-sink")
        lnd_sink_open = cln_rpc(name, "fundchannel", peers["lnd_sink"], CHANNEL_CAPACITY_SATS)
        record({"funder": identity, "sink": "lnd-sink", "result": lnd_sink_open})
    return opened


def active_channels(container: str) -> list[dict[str, Any]]:
    rows = channel_rows(container)
    return [
        row for row in rows
        if row.get("state") == "CHANNELD_NORMAL"
        and isinstance(row.get("short_channel_id"), str)
    ]


def channel_rows(container: str) -> list[dict[str, Any]]:
    rows = cln_rpc(container, "listpeerchannels").get("channels")
    if not isinstance(rows, list):
        raise RunnerError(f"listpeerchannels malformed for {container}")
    if any(not isinstance(row, dict) for row in rows):
        raise RunnerError(f"listpeerchannels contains malformed rows for {container}")
    return rows


def wait_channels(contenders: dict[str, dict[str, str]], count: int = 4) -> None:
    deadline = time.monotonic() + 120
    last: dict[str, int] = {}
    while time.monotonic() < deadline:
        last = {
            identity: len(active_channels(row["container"]))
            for identity, row in contenders.items()
        }
        if all(value == count for value in last.values()):
            return
        time.sleep(2)
    raise RunnerError(f"contender channels did not become active: {last}")


def set_initial_fees(contenders: dict[str, dict[str, str]], ppm: int = 10) -> None:
    for row in contenders.values():
        result = cln_rpc(row["container"], "setchannel", "all", 1, ppm)
        if len(result.get("channels", [])) != 4:
            raise RunnerError(f"setchannel did not update four channels on {row['container']}")


def capture_background_policies(network_id: int) -> dict[str, Any]:
    captured: dict[str, Any] = {"cln": {}, "lnd": []}
    for role in ("revenue-node", "cln-competitor"):
        container = f"polar-n{network_id}-{role}"
        policies = []
        for row in active_channels(container):
            update = (row.get("updates") or {}).get("local") or {}
            policies.append(
                {
                    "short_channel_id": row["short_channel_id"],
                    "fee_base_msat": int(update["fee_base_msat"]),
                    "fee_ppm": int(update["fee_proportional_millionths"]),
                }
            )
        if not policies:
            raise RunnerError(f"cannot isolate empty background router {role}")
        captured["cln"][role] = policies

    lnd_container = f"polar-n{network_id}-lnd-competitor"
    local_id = str(lnd_rpc(lnd_container, "getinfo")["identity_pubkey"])
    channels = lnd_rpc(lnd_container, "listchannels").get("channels", [])
    if not isinstance(channels, list) or not channels:
        raise RunnerError("cannot isolate empty LND background router")
    for row in channels:
        edge = lnd_rpc(lnd_container, "getchaninfo", "--chan_id", row["scid"])
        if edge.get("node1_pub") == local_id:
            policy = edge.get("node1_policy")
        elif edge.get("node2_pub") == local_id:
            policy = edge.get("node2_policy")
        else:
            raise RunnerError("LND channel graph edge does not contain local node")
        if not isinstance(policy, dict):
            raise RunnerError("LND local channel policy is missing")
        captured["lnd"].append(
            {
                "channel_point": row["channel_point"],
                "fee_base_msat": int(policy["fee_base_msat"]),
                "fee_ppm": int(policy["fee_rate_milli_msat"]),
                "time_lock_delta": int(policy["time_lock_delta"]),
                "min_htlc_msat": int(policy["min_htlc"]),
                "max_htlc_msat": int(policy["max_htlc_msat"]),
            }
        )
    return captured


def _set_lnd_policy(container: str, row: dict[str, Any], *, ppm: int | None = None) -> None:
    lnd_rpc(
        container,
        "updatechanpolicy",
        "--base_fee_msat", row["fee_base_msat"],
        "--fee_rate_ppm", row["fee_ppm"] if ppm is None else ppm,
        "--time_lock_delta", row["time_lock_delta"],
        "--min_htlc_msat", row["min_htlc_msat"],
        "--max_htlc_msat", row["max_htlc_msat"],
        "--chan_point", row["channel_point"],
    )


def apply_background_ppm(state: dict[str, Any], background_ppm: int) -> None:
    if background_ppm <= 0:
        raise RunnerError("background ppm must be positive")
    captured = state.get("background_policies")
    if not isinstance(captured, dict):
        raise RunnerError("background policies have not been captured")
    for role, policies in captured["cln"].items():
        container = f"polar-n{state['network_id']}-{role}"
        for row in policies:
            cln_rpc(
                container, "setchannel", row["short_channel_id"],
                row["fee_base_msat"], background_ppm,
            )
    lnd_container = f"polar-n{state['network_id']}-lnd-competitor"
    for row in captured["lnd"]:
        _set_lnd_policy(lnd_container, row, ppm=background_ppm)


def isolate_background(
    *, replica: int, results_dir: Path, background_ppm: int = 10_000
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {"fee_only_ready", "smoke_complete"}:
        raise RunnerError(f"replica is not ready for isolation: {state.get('status')}")
    if "background_policies" in state:
        raise RunnerError("background policy capture already exists; refusing to overwrite it")
    captured = capture_background_policies(int(state["network_id"]))
    state["background_policies"] = captured
    _checkpoint(path, state, "background_policies_captured")
    apply_background_ppm(state, background_ppm)
    state["status"] = "isolated_fee_only_ready"
    _checkpoint(path, state, "background_isolated", fee_ppm=background_ppm)
    return state


def retune_background(
    *, replica: int, results_dir: Path, background_ppm: int
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {"isolated_fee_only_ready", "smoke_complete"}:
        raise RunnerError(f"replica is not isolated: {state.get('status')}")
    apply_background_ppm(state, background_ppm)
    state["status"] = "isolated_fee_only_ready"
    _checkpoint(path, state, "background_retuned", fee_ppm=background_ppm)
    return state


def restore_background(state: dict[str, Any]) -> bool:
    captured = state.get("background_policies")
    if not isinstance(captured, dict) or state.get("background_restored") is True:
        return False
    for role, policies in captured.get("cln", {}).items():
        container = f"polar-n{state['network_id']}-{role}"
        for row in policies:
            cln_rpc(
                container, "setchannel", row["short_channel_id"],
                row["fee_base_msat"], row["fee_ppm"],
            )
    lnd_container = f"polar-n{state['network_id']}-lnd-competitor"
    for row in captured.get("lnd", []):
        _set_lnd_policy(lnd_container, row)
    state["background_restored"] = True
    return True


def start_revenue(container: str) -> dict[str, Any]:
    cln_rpc(
        container, "-k", "plugin", "subcommand=start", f"plugin={REVENUE_PLUGIN}",
        "revenue-ops-dry-run=false", "revenue-ops-daily-budget-sats=0",
    )
    cln_rpc(container, "revenue-config", "set", "paused", "true")
    status = cln_rpc(container, "revenue-status")
    controls = status.get("operator_controls", {}).get("values", {})
    if controls.get("paused") is not True or controls.get("daily_budget_sats") != 0:
        raise RunnerError("revenue_ops did not enter paused zero-budget state")
    cln_rpc(container, "revenue-config", "set", "paused", "false")
    return cln_rpc(container, "revenue-status")


def start_clboss(container: str, peer_ids: list[str]) -> dict[str, Any]:
    cln_rpc(container, "plugin", "start", XREBALANCE_PLUGIN)
    cln_rpc(
        container, "-k", "plugin", "subcommand=start", f"plugin={CLBOSS_PLUGIN}",
        "clboss-auto-close=false", "clboss-rebalance-mode=off",
    )
    cln_rpc(container, "clboss-ignore-onchain", 96)
    for peer_id in sorted(peer_ids):
        cln_rpc(container, "clboss-unmanage", peer_id, "open,close")
    status = cln_rpc(container, "clboss-status")
    unmanaged = status.get("unmanaged")
    if not isinstance(unmanaged, dict):
        raise RunnerError("CLBOSS status lacks unmanaged safety readback")
    for peer_id in peer_ids:
        tags = str(unmanaged.get(peer_id, ""))
        if "open" not in tags or "close" not in tags:
            raise RunnerError(f"CLBOSS peer safety tags missing for {peer_id}")
    return status


def setup(
    bridge: PolarMcp,
    *,
    network_id: int,
    replica: int,
    image: str,
    results_dir: Path,
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    if path.exists():
        raise RunnerError(f"fresh-only setup refused because state exists: {path}")
    preflight_result = preflight(bridge, network_id, image)
    assignment = assignment_for(replica)
    state: dict[str, Any] = {
        "schema": SCHEMA,
        "network_id": network_id,
        "replica": replica,
        "assignment": assignment,
        "image": image,
        "preflight": preflight_result,
        "events": [],
        "contenders": {},
        "channels": [],
    }
    _checkpoint(path, state, "preflight_complete")
    replica_dir = path.parent
    try:
        for identity in IDENTITIES:
            name = container_name(network_id, replica, identity)
            launch_contender(
                name=name,
                identity=identity,
                data_dir=replica_dir / identity / "lightning",
                image=image,
                network_id=network_id,
            )
            _checkpoint(path, state, "container_launched", identity=identity, container=name)
            info = wait_cln(name)
            state["contenders"][identity] = {
                "container": name,
                "node_id": str(info["id"]),
                "version": str(info.get("version")),
            }
            _checkpoint(path, state, "container_ready", identity=identity)

        for identity in IDENTITIES:
            _fund_wallet(state["contenders"][identity]["container"], bridge, network_id)
            wait_wallet_funds(state["contenders"][identity]["container"])
            _checkpoint(path, state, "wallet_funded", identity=identity)

        def checkpoint_channel(row: dict[str, Any]) -> None:
            state["channels"].append(row)
            _checkpoint(path, state, "channel_open_dispatched", channel=row)

        _open_channels(
            bridge, network_id, state["contenders"], on_open=checkpoint_channel
        )
        _checkpoint(path, state, "channels_open_dispatched", count=len(state["channels"]))
        wait_channels(state["contenders"])
        _checkpoint(path, state, "channels_active")
        set_initial_fees(state["contenders"])
        _checkpoint(path, state, "initial_fees_set", fee_ppm=10)

        revenue_identity = assignment["revenue_ops"]
        revenue_status = start_revenue(state["contenders"][revenue_identity]["container"])
        _checkpoint(path, state, "revenue_ops_started", identity=revenue_identity)
        clboss_identity = assignment["clboss"]
        clboss_container = state["contenders"][clboss_identity]["container"]
        peers = [str(row["peer_id"]) for row in active_channels(clboss_container)]
        clboss_status = start_clboss(clboss_container, peers)
        _checkpoint(path, state, "clboss_started", identity=clboss_identity)
        state["controller_readback"] = {
            "revenue_ops": revenue_status,
            "clboss": clboss_status,
        }
        state["status"] = "fee_only_ready"
        _checkpoint(path, state, "setup_complete")
        return state
    except Exception as exc:
        state["status"] = "setup_failed_cleanup_required"
        _checkpoint(path, state, "setup_failed", error=str(exc))
        raise


@dataclass(frozen=True)
class Totals:
    forward_count: int
    volume_msat: int
    routing_fee_msat: int
    mean_local_liquidity_sats: int
    policy_fingerprint: tuple[tuple[str, int, int], ...]


def msat_value(value: Any) -> int:
    """Normalize CLN's integer, ``Nmsat`` and amount-object encodings."""
    if isinstance(value, dict):
        value = value.get("msat")
    if isinstance(value, bool) or value is None:
        raise RunnerError(f"invalid msat value: {value!r}")
    rendered = str(value).strip()
    if rendered.endswith("msat"):
        rendered = rendered[:-4]
    try:
        parsed = int(rendered)
    except ValueError as exc:
        raise RunnerError(f"invalid msat value: {value!r}") from exc
    if parsed < 0:
        raise RunnerError(f"negative msat value: {value!r}")
    return parsed


def contender_totals(container: str) -> Totals:
    forwards = cln_rpc(container, "listforwards").get("forwards", [])
    settled = [row for row in forwards if isinstance(row, dict) and row.get("status") == "settled"]
    channels = active_channels(container)
    local_sats = []
    policies = []
    for row in channels:
        local_sats.append(msat_value(row.get("to_us_msat", 0)) // 1000)
        update = (row.get("updates") or {}).get("local") or {}
        policies.append(
            (
                str(row["short_channel_id"]),
                int(update.get("fee_base_msat", 0)),
                int(update.get("fee_proportional_millionths", 0)),
            )
        )
    if not local_sats:
        raise RunnerError(f"no active channels on {container}")
    return Totals(
        forward_count=len(settled),
        volume_msat=sum(msat_value(row.get("out_msat", 0)) for row in settled),
        routing_fee_msat=sum(msat_value(row.get("fee_msat", 0)) for row in settled),
        mean_local_liquidity_sats=sum(local_sats) // len(local_sats),
        policy_fingerprint=tuple(sorted(policies)),
    )


def totals_delta(before: Totals, after: Totals) -> dict[str, int]:
    values = {
        "forward_count": after.forward_count - before.forward_count,
        "volume_msat": after.volume_msat - before.volume_msat,
        "routing_fee_msat": after.routing_fee_msat - before.routing_fee_msat,
    }
    if any(value < 0 for value in values.values()):
        raise RunnerError("contender cumulative counters regressed")
    values["mean_local_liquidity_sats"] = (
        before.mean_local_liquidity_sats + after.mean_local_liquidity_sats
    ) // 2
    values["policy_changes"] = int(before.policy_fingerprint != after.policy_fingerprint)
    values["rebalance_cost_msat"] = 0
    return values


def _invoice_hash(invoice: str) -> str:
    payload = cln_rpc(f"polar-n{NETWORK_ID}-cln-payer", "decode", invoice)
    payment_hash = payload.get("payment_hash")
    if not isinstance(payment_hash, str) or not payment_hash:
        raise RunnerError("decodepay returned no payment hash")
    return payment_hash


def invoice_settled(sink: str, payment_hash: str) -> bool:
    container = f"polar-n{NETWORK_ID}-{sink}"
    if sink.startswith("cln-"):
        invoices = cln_rpc(
            container, "-k", "listinvoices", f"payment_hash={payment_hash}"
        ).get("invoices", [])
        return isinstance(invoices, list) and any(
            isinstance(row, dict) and row.get("status") == "paid" for row in invoices
        )
    invoice = lnd_rpc(container, "lookupinvoice", "--rhash", payment_hash)
    return invoice.get("state") == "SETTLED" or invoice.get("settled") is True


def run_reconciled_traffic(
    bridge: PolarMcp,
    *,
    network_id: int,
    rounds: int,
    amount_sats: int,
    pause_seconds: float,
    lanes: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Dispatch each payment once and reconcile Polar's known post-pay UI 500."""
    records: list[dict[str, Any]] = []
    for round_index in range(rounds):
        for payer, sink in lanes:
            invoice_payload = bridge.call(
                "create_invoice",
                {
                    "networkId": network_id,
                    "nodeName": sink,
                    "amount": amount_sats,
                    "memo": f"clboss-competition-{round_index}-{payer}",
                },
            )
            invoice = invoice_payload.get("invoice")
            if not isinstance(invoice, str) or not invoice:
                raise RunnerError("Polar create_invoice returned no invoice")
            payment_hash = _invoice_hash(invoice)
            try:
                payment = bridge.call(
                    "pay_invoice",
                    {"networkId": network_id, "fromNode": payer, "invoice": invoice},
                )
            except PolarMcpError as exc:
                settled = False
                for _ in range(10):
                    try:
                        if invoice_settled(sink, payment_hash):
                            settled = True
                            break
                    except RunnerError:
                        pass
                    time.sleep(0.5)
                if not settled:
                    operation = {
                        "round": round_index,
                        "payer": payer,
                        "sink": sink,
                        "amount_sats": amount_sats,
                        "payment_hash": payment_hash,
                        "outcome": "unknown_do_not_retry",
                        "error": str(exc),
                    }
                    raise ReconciliationError(
                        "payment dispatch could not be reconciled", list(records), operation
                    ) from exc
                payment = {
                    "success": True,
                    "reconciled_after_mcp_error": True,
                    "bridge_error": str(exc),
                }
            records.append(
                {
                    "round": round_index,
                    "payer": payer,
                    "sink": sink,
                    "amount_sats": amount_sats,
                    "payment": payment,
                }
            )
            if pause_seconds:
                time.sleep(pause_seconds)
    return records


def traffic_schedule(
    rounds: int, amount_sats: int
) -> tuple[tuple[str, str, int], ...]:
    """Interleave directions and seed each family reserve exactly once."""
    if rounds <= 0 or amount_sats <= 0:
        raise RunnerError("traffic rounds and amount must be positive")
    schedule = []
    for round_index in range(rounds):
        for family in ("cln", "lnd"):
            forward_amount = amount_sats + (
                REVERSE_FEE_BUFFER_SATS if round_index == 0 else 0
            )
            schedule.append((family, "forward", forward_amount))
            schedule.append((family, "reverse", amount_sats))
    return tuple(schedule)


def run_smoke(
    bridge: PolarMcp,
    *,
    replica: int,
    results_dir: Path,
    rounds: int,
    amount_sats: int,
    pause_seconds: float,
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {
        "fee_only_ready", "smoke_complete", "isolated_fee_only_ready"
    }:
        raise RunnerError(f"replica is not traffic-ready: {state.get('status')}")
    assignment = state["assignment"]
    controller_containers = {
        controller: state["contenders"][identity]["container"]
        for controller, identity in assignment.items()
    }
    before = {name: contender_totals(container) for name, container in controller_containers.items()}
    started = time.time()
    records: list[dict[str, Any]] = []
    try:
        for family, direction, traffic_amount in traffic_schedule(rounds, amount_sats):
            records.extend(
                run_reconciled_traffic(
                    bridge,
                    network_id=NETWORK_ID,
                    rounds=1,
                    amount_sats=traffic_amount,
                    pause_seconds=pause_seconds,
                    lanes=select_traffic_lanes(direction, family),
                )
            )
    except ReconciliationError as exc:
        result = {
            "status": "traffic_outcome_unknown",
            "completed": exc.records,
            "uncertain_operation": exc.operation,
        }
        _checkpoint(path, state, "traffic_unknown_do_not_retry", result=result)
        raise
    after = {name: contender_totals(container) for name, container in controller_containers.items()}
    block = {
        "schema": "polar-clboss-smoke-v1",
        "replica": f"replica-{replica}",
        "league": "fee_only",
        "block": f"smoke-{int(started)}",
        "duration_seconds": max(1.0, time.time() - started),
        "traffic": {
            "attempted": len(records),
            "settled": sum(
                1 for row in records
                if isinstance(row.get("payment"), dict) and row["payment"].get("success") is True
            ),
        },
        "contenders": {
            name: totals_delta(before[name], after[name]) for name in controller_containers
        },
    }
    state["status"] = "smoke_complete"
    _checkpoint(path, state, "smoke_complete", block=block)
    write_json_atomic(path.parent / f"{block['block']}.json", block)
    return block


def status(replica: int, results_dir: Path) -> dict[str, Any]:
    state = read_state(state_path(results_dir, replica))
    live = {}
    for identity, row in state.get("contenders", {}).items():
        container = row.get("container")
        if isinstance(container, str) and docker_running(container):
            live[identity] = {
                "running": True,
                "getinfo": cln_rpc(container, "getinfo"),
                "channels": active_channels(container),
            }
        else:
            live[identity] = {"running": False}
    return {"state": state, "live": live}


def _stop_plugins(state: dict[str, Any]) -> None:
    assignment = state.get("assignment", {})
    contenders = state.get("contenders", {})
    revenue_identity = assignment.get("revenue_ops")
    if revenue_identity in contenders:
        container = contenders[revenue_identity]["container"]
        if docker_running(container):
            try:
                cln_rpc(container, "revenue-config", "set", "paused", "true")
            except RunnerError:
                pass
            _run(
                ["docker", "exec", container, "lightning-cli", "--network=regtest", "-k",
                 "plugin", "subcommand=stop", f"plugin={REVENUE_PLUGIN}"],
                check=False,
            )
    clboss_identity = assignment.get("clboss")
    if clboss_identity in contenders:
        container = contenders[clboss_identity]["container"]
        if docker_running(container):
            try:
                cln_rpc(container, "setconfig", "clboss-rebalance-mode", "off")
            except RunnerError:
                pass


def cleanup(bridge: PolarMcp, *, replica: int, results_dir: Path) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    _stop_plugins(state)
    _checkpoint(path, state, "controllers_stopped_or_paused")
    if restore_background(state):
        _checkpoint(path, state, "background_policies_restored")
    # Confirm any recently dispatched funding transaction before classifying
    # channels.  AWAITING_LOCKIN is still live state and must never be hidden
    # by the active-channel filter used by traffic readiness.
    if any(docker_running(row["container"]) for row in state.get("contenders", {}).values()):
        _mine(bridge, int(state["network_id"]), 6)
        time.sleep(3)
    remaining: dict[str, list[str]] = {}
    for identity, row in state.get("contenders", {}).items():
        container = row["container"]
        if not docker_running(container):
            continue
        channels = channel_rows(container)
        for channel in channels:
            if channel.get("state") in {"ONCHAIN", "CLOSED"}:
                continue
            channel_id = str(
                channel.get("short_channel_id") or channel.get("channel_id") or ""
            )
            if not channel_id:
                remaining.setdefault(identity, []).append("unidentified-live-channel")
                continue
            result = _run(
                ["docker", "exec", container, "lightning-cli", "--network=regtest",
                 "close", channel_id],
                check=False,
                timeout=60,
            )
            if result.returncode != 0:
                remaining.setdefault(identity, []).append(channel_id)
        _checkpoint(path, state, "cooperative_closes_dispatched", identity=identity)
    if any(docker_running(row["container"]) for row in state.get("contenders", {}).values()):
        _mine(bridge, int(state["network_id"]), 6)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        live: dict[str, list[str]] = {}
        for identity, row in state.get("contenders", {}).items():
            container = row["container"]
            if not docker_running(container):
                continue
            for channel in channel_rows(container):
                if channel.get("state") not in {"ONCHAIN", "CLOSED"}:
                    label = str(
                        channel.get("short_channel_id")
                        or channel.get("channel_id")
                        or "unidentified-live-channel"
                    )
                    live.setdefault(identity, []).append(label)
        if not live:
            remaining = {}
            break
        remaining = live
        time.sleep(2)
    if remaining:
        state["status"] = "cleanup_incomplete"
        _checkpoint(path, state, "cleanup_incomplete", remaining=remaining)
        raise RunnerError(f"cleanup refused to remove containers with active channels: {remaining}")
    for row in state.get("contenders", {}).values():
        name = row["container"]
        if _run(["docker", "inspect", name], check=False).returncode == 0:
            _run(["docker", "rm", "--force", name])
    state["status"] = "cleaned"
    _checkpoint(path, state, "cleanup_complete")
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "setup", "isolate", "retune", "smoke", "status", "cleanup"),
    )
    parser.add_argument("--network-id", type=positive_int, default=NETWORK_ID)
    parser.add_argument("--replica", type=positive_int, default=1)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--results-dir", type=Path, default=Path("results/polar-clboss"))
    parser.add_argument("--rounds", type=positive_int, default=2)
    parser.add_argument("--amount-sats", type=positive_int, default=5_000)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument("--background-ppm", type=positive_int, default=10_000)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.network_id != NETWORK_ID:
        raise RunnerError(f"this runner is pinned to the inspected Polar network {NETWORK_ID}")
    bridge = PolarMcp()
    if args.command == "preflight":
        result = preflight(bridge, args.network_id, args.image)
    else:
        if args.command in {"setup", "isolate", "retune", "smoke", "cleanup"} and not args.apply:
            raise RunnerError(f"{args.command} mutates the fake-sat lab; pass --apply")
        if args.command == "setup":
            result = setup(
                bridge, network_id=args.network_id, replica=args.replica,
                image=args.image, results_dir=args.results_dir,
            )
        elif args.command == "isolate":
            result = isolate_background(
                replica=args.replica, results_dir=args.results_dir,
                background_ppm=args.background_ppm,
            )
        elif args.command == "retune":
            result = retune_background(
                replica=args.replica, results_dir=args.results_dir,
                background_ppm=args.background_ppm,
            )
        elif args.command == "smoke":
            if args.pause_seconds < 0:
                raise RunnerError("pause seconds must be nonnegative")
            result = run_smoke(
                bridge, replica=args.replica, results_dir=args.results_dir,
                rounds=args.rounds, amount_sats=args.amount_sats,
                pause_seconds=args.pause_seconds,
            )
        elif args.command == "status":
            result = status(args.replica, args.results_dir)
        else:
            result = cleanup(bridge, replica=args.replica, results_dir=args.results_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
