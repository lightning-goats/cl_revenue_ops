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
FUNDING_UTXO_SATS = 1_100_000
# Covers the 1% reserve on a 1M channel plus route fees.  A smaller fee-only
# buffer still leaves the sink unable to spend the newly received balance.
REVERSE_FEE_BUFFER_SATS = 25_000
TOURNAMENT_CYCLE_SECONDS = 60
ACQUISITION_MIN_PPM = 0
MARKET_PROFILES = {
    # Purpose-built low-fee regime for the bounded acquisition experiment.
    "acquisition": {"fee_base_msat": 1, "fee_ppm": 10},
    # Rounded 2026-08-28 public-graph medians (1ML: 0.437 sat / 150 ppm).
    # Keeping the snapshot explicit makes historical tournament runs auditable.
    "realistic": {"fee_base_msat": 500, "fee_ppm": 150},
}
REALISTIC_TRAFFIC_AMOUNTS_SATS = (5_000, 15_000, 35_000, 100_000)
BASELINE_POLL_SECONDS = 5.0
BASELINE_POLL_ATTEMPTS = 19
NATIVE_CYCLE_POLL_SECONDS = 5.0
NATIVE_CYCLE_POLL_ATTEMPTS = 37
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


def nonnegative_arg(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise RunnerError(f"{label} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunnerError(f"{label} must be a nonnegative integer") from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise RunnerError(f"{label} must be a nonnegative integer")
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


def docker_exists(name: str) -> bool:
    completed = _run(["docker", "inspect", name], check=False)
    if completed.returncode == 0:
        return True
    detail = (completed.stderr or completed.stdout or "").casefold()
    if "no such object" in detail or "no such container" in detail:
        return False
    raise RunnerError(f"cannot inspect Docker container {name!r}: {detail.strip()}")


def docker_running(name: str) -> bool:
    completed = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name], check=False
    )
    if completed.returncode == 0:
        value = completed.stdout.strip()
        if value in {"true", "false"}:
            return value == "true"
        raise RunnerError(f"Docker returned an invalid running state for {name!r}: {value!r}")
    detail = (completed.stderr or completed.stdout or "").casefold()
    if "no such object" in detail or "no such container" in detail:
        return False
    raise RunnerError(f"cannot inspect Docker container {name!r}: {detail.strip()}")


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
    if docker_exists(name):
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
                "sendtoaddress", address, f"{FUNDING_UTXO_SATS / 100_000_000:.8f}",
            ]
        )
    _mine(bridge, network_id, 6)
    return addresses[0]


def wait_wallet_funds(
    container: str,
    minimum_sats: int = FUNDING_UTXO_SATS * 2,
    timeout_seconds: float = 120,
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


def peer_families(network_id: int) -> dict[str, dict[str, str]]:
    """Return the four original peer IDs that define scored traffic lanes."""
    return {
        str(cln_rpc(f"polar-n{network_id}-cln-payer", "getinfo")["id"]): {
            "family": "cln", "side": "payer",
        },
        str(lnd_rpc(f"polar-n{network_id}-lnd-payer", "getinfo")["identity_pubkey"]): {
            "family": "lnd", "side": "payer",
        },
        str(cln_rpc(f"polar-n{network_id}-cln-sink", "getinfo")["id"]): {
            "family": "cln", "side": "sink",
        },
        str(lnd_rpc(f"polar-n{network_id}-lnd-sink", "getinfo")["identity_pubkey"]): {
            "family": "lnd", "side": "sink",
        },
    }


def resolve_lane_map(
    state: dict[str, Any], *, network_id: int = NETWORK_ID
) -> dict[str, dict[str, dict[str, str]]]:
    """Map every contender SCID to client family and payer/sink side.

    The mapping is derived from live peer IDs rather than funding-RPC return
    shapes, which differ between CLN and LND.  Missing or duplicate lanes fail
    closed so family economics can never be silently misattributed.
    """
    peers = peer_families(network_id)
    lane_map: dict[str, dict[str, dict[str, str]]] = {}
    contenders = state.get("contenders")
    if not isinstance(contenders, dict):
        raise RunnerError("runner state lacks contenders for lane mapping")
    for identity, contender in contenders.items():
        container = contender.get("container") if isinstance(contender, dict) else None
        if not isinstance(container, str) or not container:
            raise RunnerError(f"contender {identity} lacks a container")
        mapped: dict[str, dict[str, str]] = {}
        roles: set[tuple[str, str]] = set()
        for row in active_channels(container):
            scid = str(row.get("short_channel_id") or "")
            peer_id = str(row.get("peer_id") or "")
            role = peers.get(peer_id)
            if not scid or role is None:
                raise RunnerError(f"cannot attribute contender lane {identity}:{scid}")
            role_key = (role["family"], role["side"])
            if role_key in roles:
                raise RunnerError(f"duplicate contender lane role {identity}:{role_key}")
            roles.add(role_key)
            mapped[scid] = {**role, "peer_id": peer_id}
        expected = {(family, side) for family in ("cln", "lnd") for side in ("payer", "sink")}
        if roles != expected:
            raise RunnerError(f"incomplete contender lane map for {identity}: {sorted(roles)}")
        lane_map[str(identity)] = mapped
    return lane_map


def set_initial_fees(
    contenders: dict[str, dict[str, str]], *, fee_base_msat: int, fee_ppm: int
) -> None:
    nonnegative_int(fee_base_msat, "initial fee base")
    nonnegative_int(fee_ppm, "initial fee rate")
    for row in contenders.values():
        result = cln_rpc(
            row["container"], "setchannel", "all", fee_base_msat, fee_ppm
        )
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


def enable_full_stack(
    *, replica: int, results_dir: Path, spend_cap_sats: int
) -> dict[str, Any]:
    """Enable equal rebalance authority without reopening topology authority."""
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {"isolated_fee_only_ready", "smoke_complete"}:
        raise RunnerError(f"replica is not eligible for full-stack mode: {state.get('status')}")
    if state.get("background_restored") is True:
        raise RunnerError("full-stack mode requires isolated background routes")

    assignment = state["assignment"]
    contenders = state["contenders"]
    revenue_container = contenders[assignment["revenue_ops"]]["container"]
    clboss_container = contenders[assignment["clboss"]]["container"]

    baseline_non_rebalance_sats = wait_for_setup_spend_baseline(revenue_container)
    revenue_budget_sats = baseline_non_rebalance_sats + positive_int(spend_cap_sats)
    cln_rpc(
        revenue_container,
        "revenue-config", "set", "daily_budget_sats", str(revenue_budget_sats),
    )
    revenue_status = cln_rpc(revenue_container, "revenue-status")
    controls = revenue_status.get("operator_controls", {}).get("values", {})
    if (
        controls.get("paused") is not False
        or controls.get("daily_budget_sats") != revenue_budget_sats
    ):
        raise RunnerError("revenue_ops full-stack budget readback mismatch")

    for key, value in (
        ("clboss-xrebalance-gain", "1"),
        ("clboss-xrebalance-grant", "0"),
        ("clboss-xrebalance-route-cost-floor", "auto"),
    ):
        cln_rpc(clboss_container, "setconfig", key, value)
    cln_rpc(clboss_container, "setconfig", "clboss-rebalance-mode", "xrebalance")
    configs = cln_rpc(clboss_container, "listconfigs").get("configs", {})
    expected = {
        "clboss-rebalance-mode": "xrebalance",
        "clboss-xrebalance-gain": "1",
        "clboss-xrebalance-grant": "0",
        "clboss-xrebalance-route-cost-floor": "auto",
    }
    for key, value in expected.items():
        row = configs.get(key, {}) if isinstance(configs, dict) else {}
        if str(row.get("value_str")) != value:
            raise RunnerError(f"CLBOSS full-stack readback mismatch for {key}")

    state["league"] = "full_stack"
    state["full_stack"] = {
        "spend_cap_sats_per_controller": spend_cap_sats,
        "revenue_baseline_non_rebalance_sats": baseline_non_rebalance_sats,
        "revenue_runtime_budget_sats": revenue_budget_sats,
        "revenue_rebalance_cost_baseline_msat": rebalance_cost_msat(revenue_container),
        "clboss_rebalance_cost_baseline_msat": rebalance_cost_msat(clboss_container),
        "clboss": expected,
    }
    state["status"] = "isolated_full_stack_ready"
    _checkpoint(path, state, "full_stack_enabled", controls=state["full_stack"])
    return state


def acquisition_lanes(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve the revenue contender's two sink-facing experiment lanes."""
    assignment = state.get("assignment", {})
    contenders = state.get("contenders", {})
    identity = assignment.get("revenue_ops")
    contender = contenders.get(identity) if isinstance(identity, str) else None
    container = contender.get("container") if isinstance(contender, dict) else None
    if not isinstance(container, str) or not container:
        raise RunnerError("acquisition treatment lacks a revenue contender")
    active_by_id = {
        str(row.get("channel_id")): row for row in active_channels(container)
    }
    lanes: dict[str, dict[str, Any]] = {}
    for opened in state.get("channels", []):
        if not isinstance(opened, dict) or opened.get("funder") != identity:
            continue
        sink = str(opened.get("sink") or "")
        family = sink.removesuffix("-sink")
        if family not in {"cln", "lnd"}:
            continue
        result = opened.get("result")
        channel_id = result.get("channel_id") if isinstance(result, dict) else None
        row = active_by_id.get(str(channel_id))
        if row is None:
            raise RunnerError(f"cannot resolve active {family} acquisition lane")
        lanes[family] = {
            "family": family,
            "channel_id": str(row["channel_id"]),
            "short_channel_id": str(row["short_channel_id"]),
            "peer_id": str(row["peer_id"]),
            "fee_base_msat": int(
                ((row.get("updates") or {}).get("local") or {}).get("fee_base_msat", 0)
            ),
            "fee_ppm": int(
                ((row.get("updates") or {}).get("local") or {}).get(
                    "fee_proportional_millionths", 0
                )
            ),
        }
    if set(lanes) != {"cln", "lnd"}:
        raise RunnerError(f"acquisition treatment requires CLN and LND lanes: {sorted(lanes)}")
    return lanes


def _policy_write(
    container: str, action: str, peer_id: str, *, fee_ppm: int | None = None
) -> dict[str, Any]:
    arguments = [
        "-k", "revenue-policy", f"action={action}", f"peer_id={peer_id}",
        "internal=true",
    ]
    if action == "set":
        arguments.extend(
            [
                "strategy=static",
                f"fee_ppm={nonnegative_int(fee_ppm, 'acquisition fee')}",
                "expires_in_hours=1",
            ]
        )
    response = cln_rpc(container, *arguments)
    if response.get("error") or response.get("status") == "error":
        raise RunnerError(f"revenue-policy {action} failed: {response}")
    return response


def _revenue_config_set(container: str, key: str, value: int | bool) -> dict[str, Any]:
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    response = cln_rpc(container, "revenue-config", "set", key, rendered)
    if response.get("error") or response.get("status") == "error":
        raise RunnerError(f"revenue-config set {key} failed: {response}")
    return response


def _acquisition_rows(container: str) -> list[dict[str, Any]]:
    rows = cln_rpc(container, "revenue-status").get("acquisition_experiments")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RunnerError("revenue-status returned malformed acquisition experiments")
    return rows


def start_automatic_acquisition(
    *, replica: int, results_dir: Path,
    attempts: int = NATIVE_CYCLE_POLL_ATTEMPTS,
    poll_seconds: float = NATIVE_CYCLE_POLL_SECONDS,
) -> dict[str, Any]:
    """Enable the product's default-off acquisition gate and await native admission."""
    if attempts <= 0 or poll_seconds < 0:
        raise RunnerError("invalid automatic acquisition polling bounds")
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {"isolated_full_stack_ready", "smoke_complete"}:
        raise RunnerError(f"replica is not automatic-acquisition-ready: {state.get('status')}")
    if "automatic_acquisition" in state or "acquisition_treatment" in state:
        raise RunnerError("an acquisition treatment capture already exists")
    identity = state["assignment"]["revenue_ops"]
    container = state["contenders"][identity]["container"]
    lanes = acquisition_lanes(state)
    policies = cln_rpc(container, "revenue-policy", "list").get("policies")
    if not isinstance(policies, list):
        raise RunnerError("revenue-policy list returned malformed policies")
    lane_peers = {lane["peer_id"] for lane in lanes.values()}
    if any(
        isinstance(row, dict) and row.get("peer_id") in lane_peers for row in policies
    ):
        raise RunnerError("automatic acquisition refuses explicit sink-lane policies")
    controls = (
        cln_rpc(container, "revenue-status")
        .get("operator_controls", {})
        .get("values", {})
    )
    original_enabled = controls.get("acquisition_experiment_enabled")
    if not isinstance(original_enabled, bool):
        raise RunnerError("automatic acquisition lacks boolean gate readback")
    original_min = nonnegative_int(
        controls.get("min_fee_ppm_saturated"),
        "revenue saturated-channel minimum fee",
    )
    state["automatic_acquisition"] = {
        "status": "captured",
        "previous_status": state["status"],
        "original_acquisition_experiment_enabled": original_enabled,
        "original_min_fee_ppm_saturated": original_min,
        "lanes_before": lanes,
    }
    _checkpoint(path, state, "automatic_acquisition_captured")
    _revenue_config_set(container, "min_fee_ppm_saturated", 0)
    _revenue_config_set(container, "acquisition_experiment_enabled", True)
    last_rows: list[dict[str, Any]] = []
    for attempt in range(attempts):
        last_rows = _acquisition_rows(container)
        active = [row for row in last_rows if row.get("state") == "active"]
        if len(active) > 1:
            raise RunnerError("automatic acquisition admitted more than one active lane")
        if active:
            episode = active[0]
            matching = [
                lane for lane in acquisition_lanes(state).values()
                if lane["channel_id"] == str(episode.get("channel_id"))
            ]
            if len(matching) != 1:
                raise RunnerError("automatic acquisition selected an unscored lane")
            lane = matching[0]
            target_ppm = nonnegative_int(
                episode.get("target_fee_ppm"), "automatic acquisition target fee"
            )
            if lane["fee_ppm"] == target_ppm:
                state["automatic_acquisition"].update(
                    status="active", episode=episode, lane=lane,
                )
                state["status"] = "automatic_acquisition_ready"
                _checkpoint(path, state, "automatic_acquisition_active")
                return state
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    state["status"] = "automatic_acquisition_waiting"
    _checkpoint(path, state, "automatic_acquisition_timeout", rows=last_rows)
    raise RunnerError("native fee cycles did not admit an acquisition lane before timeout")


def start_retention_treatment(
    *, replica: int, results_dir: Path, fee_ppm: int,
    attempts: int = NATIVE_CYCLE_POLL_ATTEMPTS,
    poll_seconds: float = NATIVE_CYCLE_POLL_SECONDS,
) -> dict[str, Any]:
    """End automatic acquisition and price its proven lane above zero."""
    fee_ppm = nonnegative_int(fee_ppm, "retention fee")
    if fee_ppm <= ACQUISITION_MIN_PPM:
        raise RunnerError("retention fee must be above the acquisition price")
    if attempts <= 0 or poll_seconds < 0:
        raise RunnerError("invalid retention polling bounds")
    path = state_path(results_dir, replica)
    state = read_state(path)
    automatic = state.get("automatic_acquisition")
    if state.get("status") != "smoke_complete" or not isinstance(automatic, dict):
        raise RunnerError("retention requires a completed automatic-acquisition smoke block")
    if automatic.get("status") != "active" or "retention_treatment" in state:
        raise RunnerError("retention requires one active, untreated acquisition episode")
    identity = state["assignment"]["revenue_ops"]
    container = state["contenders"][identity]["container"]
    episode = automatic.get("episode")
    lane = automatic.get("lane")
    if not isinstance(episode, dict) or not isinstance(lane, dict):
        raise RunnerError("retention lacks captured acquisition evidence")
    experiment_id = nonnegative_int(episode.get("id"), "acquisition experiment id")
    baseline_ppm = nonnegative_int(
        episode.get("baseline_fee_ppm"), "acquisition baseline fee"
    )
    _revenue_config_set(container, "acquisition_experiment_enabled", False)
    completed: dict[str, Any] | None = None
    for attempt in range(attempts):
        rows = _acquisition_rows(container)
        completed = next(
            (row for row in rows if row.get("id") == experiment_id and row.get("state") == "completed"),
            None,
        )
        live_lane = next(
            (row for row in acquisition_lanes(state).values() if row["channel_id"] == lane["channel_id"]),
            None,
        )
        if completed is not None and live_lane is not None and live_lane["fee_ppm"] == baseline_ppm:
            break
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    else:
        raise RunnerError("automatic acquisition did not restore its captured baseline")
    _policy_write(container, "set", lane["peer_id"], fee_ppm=fee_ppm)
    for attempt in range(attempts):
        live_lane = next(
            (row for row in acquisition_lanes(state).values() if row["channel_id"] == lane["channel_id"]),
            None,
        )
        if live_lane is not None and live_lane["fee_ppm"] == fee_ppm:
            state["retention_treatment"] = {
                "status": "active",
                "fee_ppm": fee_ppm,
                "baseline_fee_ppm": baseline_ppm,
                "lane": live_lane,
                "completed_acquisition": completed,
            }
            automatic["status"] = "completed"
            state["status"] = "retention_ready"
            _checkpoint(path, state, "retention_treatment_active")
            return state
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    raise RunnerError("native fee cycles did not apply the retention price before timeout")


def apply_acquisition_treatment(
    *, replica: int, results_dir: Path, family: str, fee_ppm: int
) -> dict[str, Any]:
    """Pin one revenue lane to a bounded acquisition fee and its peer as control."""
    if family not in {"cln", "lnd"}:
        raise RunnerError(f"unsupported acquisition family: {family!r}")
    fee_ppm = nonnegative_int(fee_ppm, "acquisition fee")
    if fee_ppm < ACQUISITION_MIN_PPM:
        raise RunnerError(
            f"acquisition fee must respect the {ACQUISITION_MIN_PPM}-ppm absolute rail"
        )
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {"isolated_full_stack_ready", "smoke_complete"}:
        raise RunnerError(f"replica is not acquisition-ready: {state.get('status')}")
    if "acquisition_treatment" in state:
        raise RunnerError("acquisition treatment capture already exists; refusing to overwrite it")

    identity = state["assignment"]["revenue_ops"]
    container = state["contenders"][identity]["container"]
    lanes = acquisition_lanes(state)
    treatment = lanes[family]
    control = lanes["lnd" if family == "cln" else "cln"]
    policies = cln_rpc(container, "revenue-policy", "list").get("policies", [])
    if not isinstance(policies, list):
        raise RunnerError("revenue-policy list returned malformed policies")
    target_peers = {treatment["peer_id"], control["peer_id"]}
    existing = [
        row for row in policies
        if isinstance(row, dict) and row.get("peer_id") in target_peers
    ]
    if existing:
        raise RunnerError("acquisition treatment refuses to overwrite explicit peer policies")
    status = cln_rpc(container, "revenue-status")
    controls = status.get("operator_controls", {}).get("values", {})
    original_min_fee_ppm_saturated = nonnegative_int(
        controls.get("min_fee_ppm_saturated"),
        "revenue saturated-channel minimum fee",
    )
    previous_status = str(state["status"])
    state["acquisition_treatment"] = {
        "status": "captured",
        "family": family,
        "fee_ppm": fee_ppm,
        "control_family": control["family"],
        "treatment_lane": treatment,
        "control_lane": control,
        "original_min_fee_ppm_saturated": original_min_fee_ppm_saturated,
        "temporary_min_fee_ppm_saturated": min(
            original_min_fee_ppm_saturated, fee_ppm, control["fee_ppm"]
        ),
        "previous_status": previous_status,
    }
    _checkpoint(path, state, "acquisition_treatment_captured")

    temporary_min = state["acquisition_treatment"]["temporary_min_fee_ppm_saturated"]
    _revenue_config_set(container, "min_fee_ppm_saturated", temporary_min)
    _policy_write(container, "set", treatment["peer_id"], fee_ppm=fee_ppm)
    _policy_write(container, "set", control["peer_id"], fee_ppm=control["fee_ppm"])
    cln_rpc(container, "revenue-fee-cycle")
    readback = acquisition_lanes(state)
    if readback[family]["fee_ppm"] != fee_ppm:
        raise RunnerError("acquisition treatment fee readback mismatch")
    if readback[control["family"]]["fee_ppm"] != control["fee_ppm"]:
        raise RunnerError("acquisition control fee readback mismatch")
    controls = (
        cln_rpc(container, "revenue-status")
        .get("operator_controls", {})
        .get("values", {})
    )
    if controls.get("min_fee_ppm_saturated") != temporary_min:
        raise RunnerError("acquisition saturated-channel minimum-fee readback mismatch")
    state["acquisition_treatment"]["status"] = "active"
    state["acquisition_treatment"]["readback"] = readback
    state["status"] = "acquisition_ready"
    _checkpoint(path, state, "acquisition_treatment_active")
    return state


def restore_acquisition_treatment(state: dict[str, Any]) -> bool:
    treatment = state.get("acquisition_treatment")
    if (
        not isinstance(treatment, dict)
        or treatment.get("status") not in {"captured", "active"}
    ):
        return False
    identity = state["assignment"]["revenue_ops"]
    container = state["contenders"][identity]["container"]
    for key in ("treatment_lane", "control_lane"):
        lane = treatment.get(key)
        if not isinstance(lane, dict) or not lane.get("peer_id"):
            raise RunnerError("acquisition restoration lacks captured lane state")
        _policy_write(container, "delete", str(lane["peer_id"]))
    original_min = nonnegative_int(
        treatment.get("original_min_fee_ppm_saturated"),
        "captured revenue saturated-channel minimum fee",
    )
    _revenue_config_set(container, "min_fee_ppm_saturated", original_min)
    cln_rpc(container, "revenue-fee-cycle")
    controls = (
        cln_rpc(container, "revenue-status")
        .get("operator_controls", {})
        .get("values", {})
    )
    if controls.get("min_fee_ppm_saturated") != original_min:
        raise RunnerError("acquisition saturated-channel minimum-fee restoration mismatch")
    treatment["status"] = "restored"
    if state.get("status") == "acquisition_ready":
        state["status"] = treatment.get("previous_status", "isolated_full_stack_ready")
    return True


def restore_acquisition(*, replica: int, results_dir: Path) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if not restore_acquisition_treatment(state):
        raise RunnerError("no active acquisition treatment to restore")
    _checkpoint(path, state, "acquisition_treatment_restored")
    return state


def restore_automatic_treatments(state: dict[str, Any]) -> bool:
    """Restore automatic-acquisition controls and remove paid-retention policy."""
    automatic = state.get("automatic_acquisition")
    retention = state.get("retention_treatment")
    if not isinstance(automatic, dict):
        return False
    retention_active = isinstance(retention, dict) and retention.get("status") == "active"
    if automatic.get("status") == "restored" and not retention_active:
        return False
    identity = state["assignment"]["revenue_ops"]
    container = state["contenders"][identity]["container"]
    if retention_active:
        lane = retention.get("lane")
        if not isinstance(lane, dict) or not lane.get("peer_id"):
            raise RunnerError("retention restoration lacks captured lane state")
        _policy_write(container, "delete", str(lane["peer_id"]))
        retention["status"] = "restored"
    # Disable first so an active persisted episode selects its exact captured
    # baseline on the forced cleanup cycle. Scored phases never force cycles.
    _revenue_config_set(container, "acquisition_experiment_enabled", False)
    cln_rpc(container, "revenue-fee-cycle")
    _revenue_config_set(
        container,
        "min_fee_ppm_saturated",
        nonnegative_int(
            automatic.get("original_min_fee_ppm_saturated"),
            "captured saturated-channel minimum fee",
        ),
    )
    original_enabled = automatic.get("original_acquisition_experiment_enabled")
    if not isinstance(original_enabled, bool):
        raise RunnerError("automatic acquisition lacks captured boolean gate")
    _revenue_config_set(container, "acquisition_experiment_enabled", original_enabled)
    automatic["status"] = "restored"
    return True


def restore_automatic(*, replica: int, results_dir: Path) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if not restore_automatic_treatments(state):
        raise RunnerError("no automatic acquisition treatment to restore")
    state["status"] = "isolated_full_stack_ready"
    _checkpoint(path, state, "automatic_acquisition_restored")
    return state


def wait_for_setup_spend_baseline(
    revenue_container: str,
    *,
    attempts: int = BASELINE_POLL_ATTEMPTS,
    poll_seconds: float = BASELINE_POLL_SECONDS,
) -> int:
    """Wait for the contender's two setup opens to reach unified accounting."""
    if attempts <= 0 or poll_seconds < 0:
        raise RunnerError("invalid setup-spend baseline polling bounds")
    last_baseline = 0
    for attempt in range(attempts):
        revenue_debug = cln_rpc(
            revenue_container, "revenue-rebalance-debug", "summary_only=true"
        )
        breakdown = (
            revenue_debug.get("capital_controls", {})
            .get("total_liquidity_breakdown", {})
            .get("actual_spent_by_category", {})
        )
        if not isinstance(breakdown, dict):
            raise RunnerError("revenue_ops lacks unified spend-category readback")
        last_baseline = sum(
            nonnegative_int(value, f"revenue spend category {key}")
            for key, value in breakdown.items()
            if key != "rebalance"
        )
        # Every complete tournament contender funds two outbound channels.
        # A zero baseline means the asynchronous spend ledger has not caught
        # up yet and would silently consume part of the rebalance allowance.
        if last_baseline > 0:
            return last_baseline
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    raise RunnerError(
        "revenue_ops setup spend did not reach unified accounting before full-stack start"
    )


def start_revenue(container: str) -> dict[str, Any]:
    cln_rpc(
        container, "-k", "plugin", "subcommand=start", f"plugin={REVENUE_PLUGIN}",
        "revenue-ops-dry-run=false", "revenue-ops-daily-budget-sats=0",
        f"revenue-ops-flow-interval={TOURNAMENT_CYCLE_SECONDS}",
        f"revenue-ops-fee-interval={TOURNAMENT_CYCLE_SECONDS}",
        f"revenue-ops-rebalance-interval={TOURNAMENT_CYCLE_SECONDS}",
    )
    configs = cln_rpc(container, "listconfigs").get("configs", {})
    for key in (
        "revenue-ops-flow-interval",
        "revenue-ops-fee-interval",
        "revenue-ops-rebalance-interval",
    ):
        row = configs.get(key, {}) if isinstance(configs, dict) else {}
        if str(row.get("value_str")) != str(TOURNAMENT_CYCLE_SECONDS):
            raise RunnerError(f"revenue_ops tournament cadence mismatch for {key}")
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
    market_profile: str = "realistic",
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    if path.exists():
        raise RunnerError(f"fresh-only setup refused because state exists: {path}")
    preflight_result = preflight(bridge, network_id, image)
    market_seed = MARKET_PROFILES.get(market_profile)
    if market_seed is None:
        raise RunnerError(f"unknown market profile: {market_profile}")
    assignment = assignment_for(replica)
    state: dict[str, Any] = {
        "schema": SCHEMA,
        "network_id": network_id,
        "replica": replica,
        "assignment": assignment,
        "image": image,
        "preflight": preflight_result,
        "market_profile": market_profile,
        "market_seed": dict(market_seed),
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
        state["lane_map"] = resolve_lane_map(state, network_id=network_id)
        _checkpoint(path, state, "lane_map_captured", lane_map=state["lane_map"])
        set_initial_fees(state["contenders"], **market_seed)
        _checkpoint(path, state, "initial_fees_set", **market_seed)

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
    rebalance_cost_msat: int = 0
    min_local_balance_ppm: int = 0
    max_local_balance_ppm: int = 0
    worst_channel_imbalance_ppm: int = 0
    channel_metrics: tuple[tuple[str, int, int, int], ...] = ()


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


def rebalance_cost_msat(container: str) -> int:
    """Return fees spent by completed circular payments from this contender."""
    node_id = str(cln_rpc(container, "getinfo").get("id") or "")
    if not node_id:
        raise RunnerError(f"getinfo returned no node id for {container}")
    payments = cln_rpc(container, "listsendpays").get("payments", [])
    if not isinstance(payments, list):
        raise RunnerError(f"listsendpays returned no payment list for {container}")
    cost = 0
    for row in payments:
        if (
            not isinstance(row, dict)
            or row.get("status") != "complete"
            or row.get("destination") != node_id
        ):
            continue
        sent = msat_value(row.get("amount_sent_msat", 0))
        delivered = msat_value(row.get("amount_msat", 0))
        if sent < delivered:
            raise RunnerError(f"circular payment cost regressed on {container}")
        cost += sent - delivered
    return cost


def enforce_clboss_spend_cap(state: dict[str, Any]) -> dict[str, Any] | None:
    """Disable CLBOSS rebalancing once its full-stack budget is exhausted.

    CLBOSS does not expose a native absolute rebalance budget.  The tournament
    therefore polls completed circular-payment costs between traffic entries
    and fails closed by switching its rebalance mode off at the configured cap.
    """
    controls = state.get("full_stack")
    if not isinstance(controls, dict):
        return None
    cap_msat = nonnegative_int(
        controls.get("spend_cap_sats_per_controller"),
        "full-stack spend cap",
    ) * 1000
    baseline_msat = nonnegative_int(
        controls.get("clboss_rebalance_cost_baseline_msat"),
        "CLBOSS rebalance-cost baseline",
    )
    assignment = state.get("assignment")
    contenders = state.get("contenders")
    if not isinstance(assignment, dict) or not isinstance(contenders, dict):
        raise RunnerError("full-stack state is missing contender assignment")
    identity = assignment.get("clboss")
    contender = contenders.get(identity) if isinstance(identity, str) else None
    container = contender.get("container") if isinstance(contender, dict) else None
    if not isinstance(container, str) or not container:
        raise RunnerError("full-stack state is missing the CLBOSS container")

    cumulative_msat = rebalance_cost_msat(container)
    if cumulative_msat < baseline_msat:
        raise RunnerError("CLBOSS cumulative rebalance cost regressed")
    spent_msat = cumulative_msat - baseline_msat
    disabled_now = False
    if spent_msat >= cap_msat and not controls.get("clboss_spend_cap_enforced"):
        cln_rpc(container, "setconfig", "clboss-rebalance-mode", "off")
        controls["clboss_spend_cap_enforced"] = True
        controls["clboss_spend_cap_enforced_at_msat"] = spent_msat
        disabled_now = True
    controls["clboss_rebalance_cost_msat"] = spent_msat
    return {
        "cap_msat": cap_msat,
        "spent_msat": spent_msat,
        "cap_enforced": bool(controls.get("clboss_spend_cap_enforced")),
        "disabled_now": disabled_now,
    }


def contender_totals(container: str) -> Totals:
    forwards = cln_rpc(container, "listforwards").get("forwards", [])
    settled = [row for row in forwards if isinstance(row, dict) and row.get("status") == "settled"]
    channels = active_channels(container)
    local_sats = []
    local_balance_ppm = []
    policies = []
    for row in channels:
        to_us_msat = msat_value(row.get("to_us_msat", 0))
        total_msat = msat_value(row.get("total_msat", 0))
        if total_msat <= 0 or to_us_msat > total_msat:
            raise RunnerError(f"invalid channel balance on {container}")
        local_sats.append(to_us_msat // 1000)
        local_balance_ppm.append(to_us_msat * 1_000_000 // total_msat)
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
    per_channel: dict[str, list[int]] = {}
    for row in settled:
        channel_id = str(row.get("out_channel") or "")
        if not channel_id:
            continue
        metrics = per_channel.setdefault(channel_id, [0, 0, 0])
        metrics[0] += 1
        metrics[1] += msat_value(row.get("out_msat", 0))
        metrics[2] += msat_value(row.get("fee_msat", 0))
    return Totals(
        forward_count=len(settled),
        volume_msat=sum(msat_value(row.get("out_msat", 0)) for row in settled),
        routing_fee_msat=sum(msat_value(row.get("fee_msat", 0)) for row in settled),
        mean_local_liquidity_sats=sum(local_sats) // len(local_sats),
        policy_fingerprint=tuple(sorted(policies)),
        rebalance_cost_msat=rebalance_cost_msat(container),
        min_local_balance_ppm=min(local_balance_ppm),
        max_local_balance_ppm=max(local_balance_ppm),
        worst_channel_imbalance_ppm=max(
            abs((2 * ratio) - 1_000_000) for ratio in local_balance_ppm
        ),
        channel_metrics=tuple(
            (channel_id, metrics[0], metrics[1], metrics[2])
            for channel_id, metrics in sorted(per_channel.items())
        ),
    )


def totals_delta(before: Totals, after: Totals) -> dict[str, Any]:
    values = {
        "forward_count": after.forward_count - before.forward_count,
        "volume_msat": after.volume_msat - before.volume_msat,
        "routing_fee_msat": after.routing_fee_msat - before.routing_fee_msat,
        "rebalance_cost_msat": after.rebalance_cost_msat - before.rebalance_cost_msat,
    }
    if any(value < 0 for value in values.values()):
        raise RunnerError("contender cumulative counters regressed")
    values["mean_local_liquidity_sats"] = (
        before.mean_local_liquidity_sats + after.mean_local_liquidity_sats
    ) // 2
    values["ending_min_local_balance_ppm"] = after.min_local_balance_ppm
    values["ending_max_local_balance_ppm"] = after.max_local_balance_ppm
    values["ending_worst_channel_imbalance_ppm"] = after.worst_channel_imbalance_ppm
    before_channels = {row[0]: row[1:] for row in before.channel_metrics}
    after_channels = {row[0]: row[1:] for row in after.channel_metrics}
    channel_values: dict[str, dict[str, int]] = {}
    for channel_id in sorted(set(before_channels) | set(after_channels)):
        old = before_channels.get(channel_id, (0, 0, 0))
        new = after_channels.get(channel_id, (0, 0, 0))
        delta = tuple(new[index] - old[index] for index in range(3))
        if any(value < 0 for value in delta):
            raise RunnerError("contender per-channel counters regressed")
        if any(delta):
            channel_values[channel_id] = {
                "forward_count": delta[0],
                "volume_msat": delta[1],
                "routing_fee_msat": delta[2],
            }
    values["channels"] = channel_values
    values["policy_changes"] = int(before.policy_fingerprint != after.policy_fingerprint)
    return values


def family_metrics(
    contender_delta: dict[str, Any], lane_map: dict[str, dict[str, str]]
) -> dict[str, dict[str, int]]:
    """Aggregate a contender delta by the outgoing channel's client family."""
    result = {
        family: {"forward_count": 0, "volume_msat": 0, "routing_fee_msat": 0}
        for family in ("cln", "lnd")
    }
    channels = contender_delta.get("channels")
    if not isinstance(channels, dict):
        raise RunnerError("contender delta lacks per-channel metrics")
    for scid, row in channels.items():
        lane = lane_map.get(str(scid))
        if lane is None:
            raise RunnerError(f"cannot attribute nonzero channel metrics for {scid}")
        family = lane.get("family")
        if family not in result or not isinstance(row, dict):
            raise RunnerError(f"malformed family attribution for {scid}")
        for metric in ("forward_count", "volume_msat", "routing_fee_msat"):
            result[family][metric] += nonnegative_int(
                row.get(metric), f"{scid}.{metric}"
            )
    for metric in ("forward_count", "volume_msat", "routing_fee_msat"):
        if sum(result[family][metric] for family in result) != contender_delta[metric]:
            raise RunnerError(f"family {metric} does not reconcile to contender total")
    return result


def block_safety_violations(
    before: dict[str, Totals], after: dict[str, Totals]
) -> tuple[list[str], dict[str, list[str]]]:
    """Detect scored-window topology changes and unsupported counter shapes."""
    overall: list[str] = []
    by_contender: dict[str, list[str]] = {name: [] for name in before}
    for name in before:
        old_scids = {row[0] for row in before[name].policy_fingerprint}
        new_scids = {row[0] for row in after[name].policy_fingerprint}
        if old_scids != new_scids:
            violation = f"{name}:active_channel_set_changed"
            overall.append(violation)
            by_contender[name].append("active_channel_set_changed")
    return overall, by_contender


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
                    "payment_hash": payment_hash,
                    "payment": payment,
                }
            )
            if pause_seconds:
                time.sleep(pause_seconds)
    return records


def traffic_schedule(
    rounds: int,
    amount_sats: int,
    pattern: str = "balanced",
    amount_profile: str = "fixed",
) -> tuple[tuple[str, str, int], ...]:
    """Build balanced competition or one-way liquidity-pressure traffic."""
    if rounds <= 0 or amount_sats <= 0:
        raise RunnerError("traffic rounds and amount must be positive")
    if pattern not in {"balanced", "forward-pressure"}:
        raise RunnerError(f"unknown traffic pattern: {pattern}")
    if amount_profile not in {"fixed", "realistic"}:
        raise RunnerError(f"unknown amount profile: {amount_profile}")
    schedule = []
    for round_index in range(rounds):
        round_amount = (
            REALISTIC_TRAFFIC_AMOUNTS_SATS[
                round_index % len(REALISTIC_TRAFFIC_AMOUNTS_SATS)
            ]
            if amount_profile == "realistic"
            else amount_sats
        )
        for family in ("cln", "lnd"):
            if pattern == "forward-pressure":
                schedule.append((family, "forward", round_amount))
                continue
            forward_amount = round_amount + (
                REVERSE_FEE_BUFFER_SATS if round_index == 0 else 0
            )
            schedule.append((family, "forward", forward_amount))
            schedule.append((family, "reverse", round_amount))
    return tuple(schedule)


def run_smoke(
    bridge: PolarMcp,
    *,
    replica: int,
    results_dir: Path,
    rounds: int,
    amount_sats: int,
    pause_seconds: float,
    traffic_pattern: str = "balanced",
    amount_profile: str = "auto",
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {
        "fee_only_ready", "smoke_complete", "isolated_fee_only_ready",
        "isolated_full_stack_ready", "acquisition_ready",
        "automatic_acquisition_ready", "retention_ready",
    }:
        raise RunnerError(f"replica is not traffic-ready: {state.get('status')}")
    assignment = state["assignment"]
    if amount_profile == "auto":
        effective_amount_profile = (
            "realistic" if state.get("market_profile") == "realistic" else "fixed"
        )
    elif amount_profile in {"fixed", "realistic"}:
        effective_amount_profile = amount_profile
    else:
        raise RunnerError(f"unknown amount profile: {amount_profile}")
    controller_containers = {
        controller: state["contenders"][identity]["container"]
        for controller, identity in assignment.items()
    }
    cache_mode = "warm" if any(
        event.get("event") == "smoke_complete"
        for event in state.get("events", []) if isinstance(event, dict)
    ) else "cold"
    before = {name: contender_totals(container) for name, container in controller_containers.items()}
    started = time.time()
    block_id = f"smoke-{int(started)}"
    progress_path = path.parent / f"{block_id}-progress.json"
    progress: dict[str, Any] = {
        "schema": "polar-clboss-traffic-progress-v1",
        "block": block_id,
        "replica": f"replica-{replica}",
        "status": "running",
        "traffic_pattern": traffic_pattern,
        "market_profile": str(state.get("market_profile") or "legacy_low_fee"),
        "amount_profile": effective_amount_profile,
        "before": {
            name: {
                "forward_count": totals.forward_count,
                "volume_msat": totals.volume_msat,
                "routing_fee_msat": totals.routing_fee_msat,
                "mean_local_liquidity_sats": totals.mean_local_liquidity_sats,
                "min_local_balance_ppm": totals.min_local_balance_ppm,
                "max_local_balance_ppm": totals.max_local_balance_ppm,
                "worst_channel_imbalance_ppm": totals.worst_channel_imbalance_ppm,
                "channel_metrics": totals.channel_metrics,
                "policy_fingerprint": totals.policy_fingerprint,
            }
            for name, totals in before.items()
        },
        "records": [],
    }
    treatment = state.get("acquisition_treatment")
    if isinstance(treatment, dict) and treatment.get("status") == "active":
        progress["acquisition_treatment"] = treatment
    automatic = state.get("automatic_acquisition")
    if isinstance(automatic, dict) and automatic.get("status") == "active":
        progress["automatic_acquisition"] = automatic
    retention = state.get("retention_treatment")
    if isinstance(retention, dict) and retention.get("status") == "active":
        progress["retention_treatment"] = retention
    write_json_atomic(progress_path, progress)
    state["status"] = "smoke_running"
    _checkpoint(
        path,
        state,
        "smoke_started",
        block=block_id,
        progress_file=str(progress_path),
    )
    records: list[dict[str, Any]] = []
    try:
        entries_per_round = 4 if traffic_pattern == "balanced" else 2
        for schedule_index, (family, direction, traffic_amount) in enumerate(traffic_schedule(
            rounds, amount_sats, traffic_pattern, effective_amount_profile
        )):
            completed = run_reconciled_traffic(
                bridge,
                network_id=NETWORK_ID,
                rounds=1,
                amount_sats=traffic_amount,
                pause_seconds=pause_seconds,
                lanes=select_traffic_lanes(direction, family),
            )
            for row in completed:
                row.update(
                    round=schedule_index // entries_per_round,
                    family=family,
                    direction=direction,
                )
            records.extend(completed)
            progress["records"] = records
            spend_monitor = enforce_clboss_spend_cap(state)
            if spend_monitor is not None:
                progress["clboss_spend_monitor"] = spend_monitor
                if spend_monitor["disabled_now"]:
                    _checkpoint(
                        path,
                        state,
                        "clboss_spend_cap_enforced",
                        monitor=spend_monitor,
                    )
            write_json_atomic(progress_path, progress)
    except ReconciliationError as exc:
        for row in exc.records:
            row.update(
                round=schedule_index // entries_per_round,
                family=family,
                direction=direction,
            )
        exc.operation.update(
            round=schedule_index // entries_per_round,
            family=family,
            direction=direction,
        )
        records.extend(exc.records)
        partial_after = {
            name: contender_totals(container)
            for name, container in controller_containers.items()
        }
        partial_contenders = {
            name: totals_delta(before[name], partial_after[name])
            for name in controller_containers
        }
        result = {
            "status": "traffic_outcome_unknown",
            "completed_count": len(records),
            "uncertain_operation": exc.operation,
            "progress_file": str(progress_path),
            "partial_contenders": partial_contenders,
        }
        progress["status"] = "traffic_outcome_unknown"
        progress["records"] = records
        progress["uncertain_operation"] = exc.operation
        progress["partial_contenders"] = partial_contenders
        write_json_atomic(progress_path, progress)
        state["status"] = "traffic_outcome_unknown"
        _checkpoint(path, state, "traffic_unknown_do_not_retry", result=result)
        raise
    after = {name: contender_totals(container) for name, container in controller_containers.items()}
    lane_map = state.get("lane_map")
    if not isinstance(lane_map, dict):
        lane_map = resolve_lane_map(state, network_id=NETWORK_ID)
        state["lane_map"] = lane_map
        _checkpoint(path, state, "lane_map_captured", lane_map=lane_map)
    controller_lane_maps = {}
    for controller, identity in assignment.items():
        mapped = lane_map.get(identity)
        if not isinstance(mapped, dict):
            raise RunnerError(f"lane map missing for {controller}")
        controller_lane_maps[controller] = mapped
    contender_deltas = {
        name: totals_delta(before[name], after[name]) for name in controller_containers
    }
    family_deltas = {
        name: family_metrics(contender_deltas[name], controller_lane_maps[name])
        for name in controller_containers
    }
    safety_violations, contender_violations = block_safety_violations(before, after)
    for name, violations in contender_violations.items():
        contender_deltas[name]["safety_violations"] = violations
    family_rows = {}
    for family in ("cln", "lnd"):
        family_records = [row for row in records if row.get("family") == family]
        family_rows[family] = {
            "attempted": len(family_records),
            "settled": sum(
                1 for row in family_records
                if isinstance(row.get("payment"), dict)
                and row["payment"].get("success") is True
            ),
            "contenders": {
                name: family_deltas[name][family] for name in controller_containers
            },
        }
    settled_count = sum(row["settled"] for row in family_rows.values())
    contender_forward_count = sum(
        delta["forward_count"] for delta in contender_deltas.values()
    )
    fallback_settled = max(0, settled_count - contender_forward_count)
    if contender_forward_count > settled_count:
        safety_violations.append("unattributed_extra_contender_forwards")
    block = {
        "schema": "polar-clboss-smoke-v1",
        "replica": f"replica-{replica}",
        "league": str(state.get("league") or "fee_only"),
        "market_profile": str(state.get("market_profile") or "legacy_low_fee"),
        "block": block_id,
        "duration_seconds": max(1.0, time.time() - started),
        "cache_mode": cache_mode,
        "traffic": {
            "pattern": traffic_pattern,
            "amount_profile": effective_amount_profile,
            "amounts_sats": sorted({int(row["amount_sats"]) for row in records}),
            "attempted": len(records),
            "settled": settled_count,
            "fallback_settled": fallback_settled,
        },
        "families": family_rows,
        "contenders": contender_deltas,
        "safety_violations": safety_violations,
    }
    if isinstance(treatment, dict) and treatment.get("status") == "active":
        block["acquisition_treatment"] = treatment
        block["phase"] = "manual_acquisition"
    elif isinstance(automatic, dict) and automatic.get("status") == "active":
        block["automatic_acquisition"] = automatic
        block["phase"] = "automatic_acquisition"
    elif isinstance(retention, dict) and retention.get("status") == "active":
        block["retention_treatment"] = retention
        block["phase"] = "paid_retention"
    else:
        block["phase"] = "baseline"
    progress["status"] = "complete"
    progress["records"] = records
    write_json_atomic(progress_path, progress)
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
    if restore_automatic_treatments(state):
        _checkpoint(path, state, "automatic_acquisition_restored")
    if restore_acquisition_treatment(state):
        _checkpoint(path, state, "acquisition_treatment_restored")
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
        if docker_exists(name):
            _run(["docker", "rm", "--force", name])
    state["status"] = "cleaned"
    _checkpoint(path, state, "cleanup_complete")
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight", "setup", "isolate", "retune", "full-stack",
            "acquire", "restore-acquire", "auto-acquire", "retain",
            "restore-auto", "smoke", "status", "cleanup",
        ),
    )
    parser.add_argument("--network-id", type=positive_int, default=NETWORK_ID)
    parser.add_argument("--replica", type=positive_int, default=1)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--results-dir", type=Path, default=Path("results/polar-clboss"))
    parser.add_argument("--rounds", type=positive_int, default=2)
    parser.add_argument("--amount-sats", type=positive_int, default=5_000)
    parser.add_argument(
        "--amount-profile",
        choices=("auto", "fixed", "realistic"),
        default="auto",
    )
    parser.add_argument(
        "--market-profile",
        choices=tuple(MARKET_PROFILES),
        default="realistic",
    )
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument(
        "--traffic-pattern",
        choices=("balanced", "forward-pressure"),
        default="balanced",
    )
    parser.add_argument("--background-ppm", type=positive_int, default=10_000)
    parser.add_argument("--spend-cap-sats", type=positive_int, default=1_000)
    parser.add_argument("--acquisition-family", choices=("cln", "lnd"), default="cln")
    parser.add_argument("--acquisition-ppm", type=nonnegative_arg, default=2)
    parser.add_argument("--retention-ppm", type=positive_int, default=1)
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
        if args.command in {
            "setup", "isolate", "retune", "full-stack", "acquire",
            "restore-acquire", "auto-acquire", "retain", "restore-auto",
            "smoke", "cleanup"
        } and not args.apply:
            raise RunnerError(f"{args.command} mutates the fake-sat lab; pass --apply")
        if args.command == "setup":
            result = setup(
                bridge, network_id=args.network_id, replica=args.replica,
                image=args.image, results_dir=args.results_dir,
                market_profile=args.market_profile,
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
        elif args.command == "full-stack":
            result = enable_full_stack(
                replica=args.replica,
                results_dir=args.results_dir,
                spend_cap_sats=args.spend_cap_sats,
            )
        elif args.command == "acquire":
            result = apply_acquisition_treatment(
                replica=args.replica,
                results_dir=args.results_dir,
                family=args.acquisition_family,
                fee_ppm=args.acquisition_ppm,
            )
        elif args.command == "restore-acquire":
            result = restore_acquisition(
                replica=args.replica,
                results_dir=args.results_dir,
            )
        elif args.command == "auto-acquire":
            result = start_automatic_acquisition(
                replica=args.replica,
                results_dir=args.results_dir,
            )
        elif args.command == "retain":
            result = start_retention_treatment(
                replica=args.replica,
                results_dir=args.results_dir,
                fee_ppm=args.retention_ppm,
            )
        elif args.command == "restore-auto":
            result = restore_automatic(
                replica=args.replica,
                results_dir=args.results_dir,
            )
        elif args.command == "smoke":
            if args.pause_seconds < 0:
                raise RunnerError("pause seconds must be nonnegative")
            result = run_smoke(
                bridge, replica=args.replica, results_dir=args.results_dir,
                rounds=args.rounds, amount_sats=args.amount_sats,
                pause_seconds=args.pause_seconds,
                traffic_pattern=args.traffic_pattern,
                amount_profile=args.amount_profile,
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
