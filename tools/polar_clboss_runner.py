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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from polar_mixed_client_lab import (  # noqa: E402
    PolarMcp,
    PolarMcpError,
    select_traffic_lanes,
)


SCHEMA = "polar-clboss-runner-state-v1"
EXPECTED_REVENUE_REVISION = "f23ba830464895fa26b7463f9545c58c1c9d5e70"
IMAGE = f"cl-revenue-ops-polar-clboss:{EXPECTED_REVENUE_REVISION[:7]}"
EXPECTED_CLN_VERSION = "v26.06.7"
EXPECTED_CLN_ARTIFACT_DIGEST = (
    "sha256:53ddf124fe7058b6a2fc059d104976cc54ba5be21dc55b295cd82d01cabeb39c"
)
NETWORK_ID = 4
DOCKER_NETWORK = "polar-network-4_default"
BACKEND = "polar-n4-backend1"
REVENUE_PLUGIN = "/opt/cl_revenue_ops/cl-revenue-ops-polar-wrapper"
CLBOSS_PLUGIN = "/usr/local/libexec/clboss"
XREBALANCE_PLUGIN = "/usr/local/libexec/xrebalance"
IDENTITIES = ("identity-a", "identity-b")
CHANNEL_CAPACITY_SATS = 1_000_000
RETURN_PATH_CAPACITY_SATS = 2_000_000
CONTROLLED_DEPLETION_SATS = 750_000
RETURN_PATH_FUNDING_BUFFER_SATS = 100_000
# Synthetic return liquidity participates in the same public gossip pool that
# Revenue Ops uses for market pricing.  Keep it in the controlled corridor's
# realistic band instead of advertising an artificial 1-ppm outlier.
RETURN_PATH_FEE_BASE_MSAT = 500
RETURN_PATH_FEE_PPM = 120
FUNDING_UTXO_SATS = 1_100_000
# Covers the 1% reserve on a 1M channel plus route fees.  A smaller fee-only
# buffer still leaves the sink unable to spend the newly received balance.
REVERSE_FEE_BUFFER_SATS = 25_000
TOURNAMENT_CYCLE_SECONDS = 15
CLBOSS_REBALANCES_PER_HOUR = 120
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
GOSSIP_POLICY_POLL_SECONDS = 1.0
GOSSIP_POLICY_POLL_ATTEMPTS = 31
NATIVE_CYCLE_POLL_SECONDS = 5.0
NATIVE_CYCLE_POLL_ATTEMPTS = 37
EXPECTED_ACQUISITION_MARKETS = 2
# Polar's UI-facing payment call can time out while the underlying CLN/LND
# payment is still resolving.  Wait long enough to observe an authoritative
# payer-side terminal state, but never dispatch the same invoice again.
PAYMENT_RECONCILIATION_POLL_SECONDS = 1.0
PAYMENT_RECONCILIATION_POLL_ATTEMPTS = 90
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
    revenue_revision = image_labels.get("org.opencontainers.image.revision.revenue_ops")
    if not revenue_revision:
        raise RunnerError("competition image lacks a pinned revenue_ops revision label")
    if image == IMAGE and revenue_revision != EXPECTED_REVENUE_REVISION:
        raise RunnerError(
            "default competition image has unexpected revenue_ops revision: "
            f"{revenue_revision}"
        )
    if image == IMAGE:
        cln_version = image_labels.get("org.opencontainers.image.version.cln")
        if cln_version != EXPECTED_CLN_VERSION:
            raise RunnerError(
                "default competition image lacks verified CLN runtime version: "
                f"{cln_version!r}"
            )
        cln_digest = image_labels.get("org.opencontainers.image.digest.cln")
        if cln_digest != EXPECTED_CLN_ARTIFACT_DIGEST:
            raise RunnerError(
                "default competition image has unexpected CLN artifact digest: "
                f"{cln_digest!r}"
            )
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


def _send_fake_onchain_funds(
    bridge: PolarMcp, network_id: int, address: str, amount_sats: int
) -> None:
    """Fund a lab wallet from the regtest backend and confirm the output."""
    if not isinstance(address, str) or not address.startswith("bcrt1"):
        raise RunnerError("return-path wallet returned no regtest address")
    amount_sats = nonnegative_int(amount_sats, "return-path funding amount")
    if amount_sats == 0:
        raise RunnerError("return-path funding amount must be positive")
    _run(
        [
            "docker", "exec", f"polar-n{network_id}-backend1",
            "bitcoin-cli", "-regtest", "-rpcuser=polaruser", "-rpcpassword=polarpass",
            "sendtoaddress", address, f"{amount_sats / 100_000_000:.8f}",
        ]
    )
    _mine(bridge, network_id, 6)


def ensure_payer_wallet_funds(
    bridge: PolarMcp,
    network_id: int,
    *,
    minimum_sats: int = (2 * CHANNEL_CAPACITY_SATS) + RETURN_PATH_FUNDING_BUFFER_SATS,
) -> dict[str, dict[str, int]]:
    """Top up depleted original payer wallets before contender setup."""
    minimum_sats = nonnegative_int(minimum_sats, "payer wallet minimum")
    if minimum_sats == 0:
        raise RunnerError("payer wallet minimum must be positive")
    cln_payer = f"polar-n{network_id}-cln-payer"
    lnd_payer = f"polar-n{network_id}-lnd-payer"
    outputs = cln_rpc(cln_payer, "listfunds").get("outputs")
    if not isinstance(outputs, list):
        raise RunnerError("CLN payer listfunds is malformed")
    cln_confirmed = sum(
        msat_value(row.get("amount_msat", 0)) // 1000
        for row in outputs
        if isinstance(row, dict) and row.get("status") == "confirmed"
    )
    lnd_balance = lnd_rpc(lnd_payer, "walletbalance")
    lnd_confirmed = nonnegative_int(
        lnd_balance.get("confirmed_balance"), "LND payer confirmed balance"
    )
    result = {
        "cln": {"before_sats": cln_confirmed, "topup_sats": 0},
        "lnd": {"before_sats": lnd_confirmed, "topup_sats": 0},
    }
    if cln_confirmed < minimum_sats:
        topup = minimum_sats - cln_confirmed
        address = onchain_address(cln_rpc(cln_payer, "newaddr"))
        _send_fake_onchain_funds(bridge, network_id, address, topup)
        result["cln"]["topup_sats"] = topup
    if lnd_confirmed < minimum_sats:
        topup = minimum_sats - lnd_confirmed
        address = lnd_rpc(lnd_payer, "newaddress", "p2tr").get("address")
        _send_fake_onchain_funds(bridge, network_id, str(address or ""), topup)
        result["lnd"]["topup_sats"] = topup
    return result


def _live_lnd_channels(container: str) -> list[dict[str, Any]]:
    rows = lnd_rpc(container, "listchannels").get("channels")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RunnerError(f"listchannels malformed for {container}")
    return rows


def _wait_return_paths(
    paths: list[dict[str, Any]], *, timeout_seconds: float = 120
) -> list[dict[str, Any]]:
    """Resolve the exact active SCID/channel point for each synthetic path."""
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, str] = {}
    while time.monotonic() < deadline:
        resolved: list[dict[str, Any]] = []
        for path in paths:
            family = path.get("family")
            source = path.get("source_container")
            peer_id = path.get("peer_id")
            if not all(isinstance(value, str) and value for value in (family, source, peer_id)):
                raise RunnerError("malformed return-path checkpoint")
            if family == "cln":
                matches = [
                    row for row in active_channels(source)
                    if row.get("peer_id") == peer_id
                    and row.get("channel_id") == path.get("channel_id")
                ]
                if len(matches) != 1:
                    last[family] = f"active matches={len(matches)}"
                    break
                resolved.append({**path, "short_channel_id": matches[0]["short_channel_id"]})
            elif family == "lnd":
                matches = [
                    row for row in _live_lnd_channels(source)
                    if row.get("remote_pubkey") == peer_id and row.get("active") is True
                ]
                if len(matches) != 1:
                    last[family] = f"active matches={len(matches)}"
                    break
                point = matches[0].get("channel_point")
                scid = matches[0].get("scid_str")
                if not isinstance(point, str) or not point or not isinstance(scid, str) or not scid:
                    raise RunnerError("active LND return path lacks channel identifiers")
                resolved.append({**path, "channel_point": point, "short_channel_id": scid})
            else:
                raise RunnerError(f"unknown return-path family: {family!r}")
        if len(resolved) == len(paths):
            return resolved
        time.sleep(2)
    raise RunnerError(f"synthetic return paths did not become active: {last}")


def _wait_lnd_local_policy(
    container: str,
    scid: object,
    local_id: str,
    *,
    attempts: int = GOSSIP_POLICY_POLL_ATTEMPTS,
    poll_seconds: float = GOSSIP_POLICY_POLL_SECONDS,
) -> dict[str, Any]:
    """Wait for LND's public graph to expose its exact local direction."""
    last = "policy absent"
    for attempt in range(attempts):
        try:
            edge = lnd_rpc(container, "getchaninfo", "--chan_id", scid)
            if edge.get("node1_pub") == local_id:
                policy = edge.get("node1_policy")
            elif edge.get("node2_pub") == local_id:
                policy = edge.get("node2_policy")
            else:
                policy = None
                last = "graph edge lacks payer identity"
            if isinstance(policy, dict):
                return policy
            last = "local directional policy absent"
        except RunnerError as exc:
            last = str(exc)
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    raise RunnerError(f"LND return-path policy readback failed: {last}")


def controlled_depletion_snapshot(
    state: dict[str, Any], *, family: str = "cln", depleted_side: str = "sink"
) -> dict[str, Any]:
    """Read each controller's selected source/depleted lanes by live SCID."""
    if family not in {"cln", "lnd"}:
        raise RunnerError(f"unknown controlled depletion family: {family!r}")
    if depleted_side not in {"payer", "sink"}:
        raise RunnerError(f"unknown controlled depleted side: {depleted_side!r}")
    assignment = state.get("assignment")
    contenders = state.get("contenders")
    lane_map = state.get("lane_map")
    if not all(isinstance(value, dict) for value in (assignment, contenders, lane_map)):
        raise RunnerError("controlled depletion lacks contender lane metadata")
    result: dict[str, Any] = {}
    for controller, identity in assignment.items():
        contender = contenders.get(identity)
        identity_lanes = lane_map.get(identity)
        if not isinstance(contender, dict) or not isinstance(identity_lanes, dict):
            raise RunnerError(f"controlled depletion lacks lanes for {identity}")
        live = {
            str(row["short_channel_id"]): row
            for row in active_channels(str(contender["container"]))
        }
        roles: dict[str, dict[str, Any]] = {}
        for scid, lane in identity_lanes.items():
            if not isinstance(lane, dict) or lane.get("family") != family:
                continue
            side = lane.get("side")
            row = live.get(str(scid))
            if side not in {"payer", "sink"} or row is None:
                raise RunnerError(f"controlled depletion cannot resolve {identity}:{scid}")
            total_msat = msat_value(row.get("total_msat", 0))
            local_msat = msat_value(row.get("to_us_msat", 0))
            if total_msat <= 0 or local_msat > total_msat:
                raise RunnerError(f"controlled depletion found invalid balance on {scid}")
            roles[str(side)] = {
                "short_channel_id": str(scid),
                "local_balance_ppm": local_msat * 1_000_000 // total_msat,
            }
        if set(roles) != {"payer", "sink"}:
            raise RunnerError(f"controlled depletion lacks both CLN roles for {identity}")
        source_side = "payer" if depleted_side == "sink" else "sink"
        result[str(controller)] = {
            "source": roles[source_side],
            "depleted": roles[depleted_side],
        }
    return result


def _short_channel_id_int(scid: object) -> int:
    """Convert a canonical blockxtxxout SCID to LND's packed integer form."""
    parts = str(scid).split("x")
    if len(parts) != 3:
        raise RunnerError(f"invalid short channel id: {scid!r}")
    try:
        block, transaction, output = (int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"invalid short channel id: {scid!r}") from exc
    if not (0 <= block < 2**24 and 0 <= transaction < 2**24 and 0 <= output < 2**16):
        raise RunnerError(f"short channel id component out of range: {scid!r}")
    return (block << 40) | (transaction << 16) | output


def _prepare_lnd_counterflow_admission(
    state: dict[str, Any],
    before: dict[str, Any],
    *,
    amount_sats: int,
    attempts: int = GOSSIP_POLICY_POLL_ATTEMPTS,
    poll_seconds: float = GOSSIP_POLICY_POLL_SECONDS,
) -> list[dict[str, Any]]:
    """Make a paused LND counterflow routable without overstating liquidity.

    Payer-depletion first moves liquidity into each contender's payer-facing
    channel.  Revenue Ops may still advertise its truthful startup floor until
    the next scheduled admission refresh, but the fixture deliberately holds
    both controllers before that cycle.  Apply the same spendable-backed
    ceiling formula to both contenders and require the LND source's graph to
    observe it before dispatching the earning payments.
    """
    required_msat = nonnegative_int(amount_sats, "counterflow amount") * 1000
    assignment = state.get("assignment")
    contenders = state.get("contenders")
    if not isinstance(assignment, dict) or not isinstance(contenders, dict):
        raise RunnerError("counterflow admission lacks contender metadata")
    prepared: list[dict[str, Any]] = []
    for controller in ("revenue_ops", "clboss"):
        identity = assignment.get(controller)
        contender = contenders.get(identity) if isinstance(identity, str) else None
        lanes = before.get(controller)
        depleted = lanes.get("depleted") if isinstance(lanes, dict) else None
        if not isinstance(contender, dict) or not isinstance(depleted, dict):
            raise RunnerError(f"counterflow admission lacks {controller} lane")
        container = str(contender.get("container") or "")
        source = str(contender.get("node_id") or "")
        scid = str(depleted.get("short_channel_id") or "")
        matches = [
            row for row in active_channels(container)
            if str(row.get("short_channel_id") or "") == scid
        ]
        if len(matches) != 1 or not source:
            raise RunnerError(f"counterflow admission cannot resolve {controller}:{scid}")
        row = matches[0]
        local = (row.get("updates") or {}).get("local")
        if not isinstance(local, dict):
            raise RunnerError(f"counterflow admission lacks local policy for {controller}")
        try:
            fee_base_msat = nonnegative_int(local.get("fee_base_msat"), "fee base")
            fee_ppm = nonnegative_int(
                local.get("fee_proportional_millionths"), "fee ppm"
            )
            htlc_min_msat = nonnegative_int(
                local.get("htlc_minimum_msat"), "HTLC minimum"
            )
            current_htlc_max_msat = nonnegative_int(
                local.get("htlc_maximum_msat"), "HTLC maximum"
            )
            spendable_msat = msat_value(row.get("spendable_msat"))
        except (TypeError, ValueError) as exc:
            raise RunnerError(
                f"counterflow admission has malformed {controller} policy"
            ) from exc
        if spendable_msat < required_msat:
            raise RunnerError(
                f"counterflow admission exceeds {controller} spendable liquidity"
            )
        target_htlc_max_msat = spendable_msat
        result = cln_rpc(
            container, "setchannel", scid, fee_base_msat, fee_ppm,
            htlc_min_msat, target_htlc_max_msat,
        )
        if len(result.get("channels", [])) != 1:
            raise RunnerError(f"counterflow admission did not update {controller}:{scid}")
        prepared.append({
            "controller": controller,
            "short_channel_id": scid,
            "required_msat": required_msat,
            "previous_htlc_max_msat": current_htlc_max_msat,
            "applied_htlc_max_msat": target_htlc_max_msat,
            "spendable_msat": spendable_msat,
            "source": source,
        })

    observer = f"polar-n{int(state['network_id'])}-lnd-sink"
    pending = {row["controller"]: row for row in prepared}
    last = "policy absent"
    for attempt in range(attempts):
        for controller, expected in list(pending.items()):
            try:
                edge = lnd_rpc(
                    observer, "getchaninfo", "--chan_id",
                    _short_channel_id_int(expected["short_channel_id"]),
                )
                if edge.get("node1_pub") == expected["source"]:
                    policy = edge.get("node1_policy")
                elif edge.get("node2_pub") == expected["source"]:
                    policy = edge.get("node2_policy")
                else:
                    policy = None
                observed = (
                    nonnegative_int(policy.get("max_htlc_msat"), "gossip HTLC maximum")
                    if isinstance(policy, dict) else None
                )
                if observed == expected["applied_htlc_max_msat"]:
                    expected["gossip_verified"] = True
                    pending.pop(controller)
                else:
                    last = f"{controller} observed htlcmax={observed!r}"
            except (RunnerError, TypeError, ValueError) as exc:
                last = str(exc)
        if not pending:
            return prepared
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    raise RunnerError(f"LND counterflow admission gossip readback failed: {last}")


def _directed_cln_fixture_payment(
    bridge: PolarMcp,
    *,
    network_id: int,
    target_scid: str,
    amount_sats: int,
    label: str,
    direction: str = "forward",
) -> dict[str, Any]:
    """Pay once through a selected contender first hop for fixture setup."""
    if direction not in {"forward", "reverse"}:
        raise RunnerError(f"unknown controlled CLN direction: {direction!r}")
    source_role = "cln-payer" if direction == "forward" else "cln-sink"
    destination_role = "cln-sink" if direction == "forward" else "cln-payer"
    source = f"polar-n{network_id}-{source_role}"
    source_scids = {
        str(row["short_channel_id"]) for row in active_channels(source)
    }
    if target_scid not in source_scids:
        raise RunnerError(
            f"controlled depletion target is not a {source_role} channel: {target_scid}"
        )
    exclusions = [
        f"{scid}/{channel_direction}"
        for scid in sorted(source_scids - {target_scid})
        for channel_direction in (0, 1)
    ]
    invoice_payload = bridge.call(
        "create_invoice",
        {
            "networkId": network_id,
            "nodeName": destination_role,
            "amount": amount_sats,
            "memo": f"clboss-controlled-depletion-{label}",
        },
    )
    invoice = invoice_payload.get("invoice")
    if not isinstance(invoice, str) or not invoice:
        raise RunnerError("controlled depletion invoice is missing")
    payment = cln_rpc(
        source,
        "-k",
        "pay",
        f"bolt11={invoice}",
        "retry_for=30",
        f"exclude={json.dumps(exclusions)}",
    )
    if payment.get("status") != "complete":
        raise RunnerError(f"controlled depletion payment did not complete: {label}")
    return {
        "controller": label,
        "target_short_channel_id": target_scid,
        "amount_sats": amount_sats,
        "direction": direction,
        "excluded_first_hops": exclusions,
        "payment_hash": payment.get("payment_hash"),
    }


def _directed_lnd_fixture_payment(
    bridge: PolarMcp,
    *,
    network_id: int,
    target_scid: str,
    amount_sats: int,
    label: str,
    direction: str = "forward",
) -> dict[str, Any]:
    """Pay through one exact LND-payer contender channel and exact last hop."""
    if direction not in {"forward", "reverse"}:
        raise RunnerError(f"unknown controlled LND direction: {direction!r}")
    source_role = "lnd-payer" if direction == "forward" else "lnd-sink"
    destination_role = "lnd-sink" if direction == "forward" else "lnd-payer"
    source = f"polar-n{network_id}-{source_role}"
    matches = [
        row for row in _live_lnd_channels(source)
        if row.get("active") is True and row.get("scid_str") == target_scid
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"controlled depletion LND target is not uniquely active: {target_scid}"
        )
    channel = matches[0]
    chan_id = channel.get("scid")
    last_hop = channel.get("remote_pubkey")
    if not isinstance(chan_id, str) or not chan_id.isdigit():
        raise RunnerError("controlled depletion LND target lacks numeric routing id")
    if not isinstance(last_hop, str) or not last_hop:
        raise RunnerError("controlled depletion LND target lacks last-hop identity")
    invoice_payload = bridge.call(
        "create_invoice",
        {
            "networkId": network_id,
            "nodeName": destination_role,
            "amount": amount_sats,
            "memo": f"clboss-controlled-depletion-{label}",
        },
    )
    invoice = invoice_payload.get("invoice")
    if not isinstance(invoice, str) or not invoice:
        raise RunnerError("controlled depletion LND invoice is missing")
    payment = lnd_rpc(
        source,
        "payinvoice",
        "--force",
        "--json",
        "--fee_limit=1000",
        "--timeout=30s",
        f"--outgoing_chan_id={chan_id}",
        f"--last_hop={last_hop}",
        invoice,
    )
    if payment.get("status") != "SUCCEEDED":
        raise RunnerError(f"controlled depletion LND payment did not complete: {label}")
    return {
        "controller": label,
        "target_short_channel_id": target_scid,
        "target_channel_id": chan_id,
        "last_hop": last_hop,
        "amount_sats": amount_sats,
        "direction": direction,
        "multipart_allowed": True,
        "payment_hash": payment.get("payment_hash"),
    }


def prepare_controlled_depletion(
    bridge: PolarMcp,
    *,
    replica: int,
    results_dir: Path,
    amount_sats: int = CONTROLLED_DEPLETION_SATS,
    fixture_fee_ppm: int | None = None,
    family: str = "cln",
    depleted_side: str = "sink",
) -> dict[str, Any]:
    """Create equal source/depleted client-family pairs while controllers are held."""
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") != "isolated_full_stack_ready":
        raise RunnerError("controlled depletion requires a fresh full-stack replica")
    if any(
        isinstance(event, dict) and event.get("event") == "smoke_complete"
        for event in state.get("events", [])
    ):
        raise RunnerError("controlled depletion refuses a previously scored replica")
    if family not in {"cln", "lnd"}:
        raise RunnerError(f"unknown controlled depletion family: {family!r}")
    if depleted_side not in {"payer", "sink"}:
        raise RunnerError(f"unknown controlled depleted side: {depleted_side!r}")
    amount_sats = nonnegative_int(amount_sats, "controlled depletion amount")
    if (
        amount_sats <= CHANNEL_CAPACITY_SATS // 2
        or amount_sats >= CHANNEL_CAPACITY_SATS
    ):
        raise RunnerError(
            "controlled depletion amount must be above half and below channel capacity"
        )

    assignment = state["assignment"]
    contenders = state["contenders"]
    revenue_container = contenders[assignment["revenue_ops"]]["container"]
    clboss_container = contenders[assignment["clboss"]]["container"]
    cln_rpc(revenue_container, "revenue-config", "set", "paused", "true")
    revenue_controls = (
        cln_rpc(revenue_container, "revenue-status")
        .get("operator_controls", {})
        .get("values", {})
    )
    if revenue_controls.get("paused") is not True:
        raise RunnerError("Revenue Ops did not pause for controlled depletion")
    cln_rpc(clboss_container, "setconfig", "clboss-rebalance-mode", "off")
    clboss_mode = (
        cln_rpc(clboss_container, "clboss-status")
        .get("rebalance_mode", {})
        .get("mode")
    )
    if clboss_mode != "off":
        raise RunnerError("CLBOSS did not pause for controlled depletion")

    before = (
        controlled_depletion_snapshot(state, family=family)
        if depleted_side == "sink"
        else controlled_depletion_snapshot(
            state, family=family, depleted_side=depleted_side
        )
    )
    if fixture_fee_ppm is not None:
        fixture_fee_ppm = nonnegative_int(
            fixture_fee_ppm, "controlled depletion fixture fee"
        )
        expected_fixture_policies = []
        for controller in ("revenue_ops", "clboss"):
            identity = assignment[controller]
            container = contenders[identity]["container"]
            source = str(contenders[identity].get("node_id") or "")
            if not source:
                raise RunnerError(f"controlled depletion lacks {controller} node identity")
            scid = before[controller]["depleted"]["short_channel_id"]
            result = cln_rpc(container, "setchannel", scid, 0, fixture_fee_ppm)
            channels = result.get("channels")
            if not isinstance(channels, list) or len(channels) != 1:
                raise RunnerError(
                    f"controlled depletion fixture fee did not update {controller}:{scid}"
                )
            expected_fixture_policies.append({
                "short_channel_id": scid,
                "source": source,
                "fee_ppm": fixture_fee_ppm,
            })
        # The payer and both controllers price the downstream hop from gossip.
        # Verify exact crossed readback rather than relying on a fixed sleep.
        wait_gossip_policies(contenders, expected_fixture_policies)
    payment_fn = (
        _directed_cln_fixture_payment
        if family == "cln"
        else _directed_lnd_fixture_payment
    )
    payments = []
    for controller in ("revenue_ops", "clboss"):
        forward_target = (
            before[controller]["source"]
            if depleted_side == "sink"
            else before[controller]["depleted"]
        )
        payments.append(
            payment_fn(
                bridge,
                network_id=int(state["network_id"]),
                target_scid=forward_target["short_channel_id"],
                amount_sats=amount_sats,
                label=controller,
            )
        )
    counterflow_amount_sats = 0
    counterflow_admission = []
    counterflow_parts_sats: list[int] = []
    if depleted_side == "payer":
        counterflow_amount_sats = 2 * amount_sats - CHANNEL_CAPACITY_SATS
        # The controller's active profile deliberately requires three settled
        # forwards before closing a sub-15-minute observation window.  Split
        # the fixed counterflow total into exactly three payments so the
        # accelerated lab cadence exercises that real evidence threshold
        # without changing liquidity, quoted fees, or either contender's
        # economic opportunity.
        if family == "lnd":
            quotient, remainder = divmod(counterflow_amount_sats, 3)
            counterflow_parts_sats = [
                quotient + (1 if index < remainder else 0)
                for index in range(3)
            ]
        else:
            counterflow_parts_sats = [counterflow_amount_sats]
        if family == "lnd":
            counterflow_admission = _prepare_lnd_counterflow_admission(
                state, before, amount_sats=max(counterflow_parts_sats),
            )
        for controller in ("revenue_ops", "clboss"):
            for part_index, part_sats in enumerate(counterflow_parts_sats, start=1):
                payments.append(
                    payment_fn(
                        bridge,
                        network_id=int(state["network_id"]),
                        target_scid=before[controller]["source"]["short_channel_id"],
                        amount_sats=part_sats,
                        label=f"payer-depletion-{controller}-{part_index}",
                        direction="reverse",
                    )
                )
    after = (
        controlled_depletion_snapshot(state, family=family)
        if depleted_side == "sink"
        else controlled_depletion_snapshot(
            state, family=family, depleted_side=depleted_side
        )
    )
    for controller, lanes in after.items():
        if lanes["source"]["local_balance_ppm"] < 700_000:
            raise RunnerError(f"controlled depletion did not create a source for {controller}")
        if lanes["depleted"]["local_balance_ppm"] > 300_000:
            raise RunnerError(f"controlled depletion did not deplete a sink for {controller}")
    state["controlled_depletion"] = {
        "family": family,
        "depleted_side": depleted_side,
        "amount_sats_per_controller": amount_sats,
        "counterflow_amount_sats_per_controller": counterflow_amount_sats,
        "counterflow_parts_sats_per_controller": counterflow_parts_sats,
        "counterflow_admission": counterflow_admission,
        "controllers_held": True,
        "fixture_fee_ppm": fixture_fee_ppm,
        "before": before,
        "after": after,
        "payments": payments,
    }
    state["status"] = "controlled_depletion_ready"
    _checkpoint(path, state, "controlled_depletion_ready", fixture=state["controlled_depletion"])
    return state


def apply_equal_targeted_pressure(
    bridge: PolarMcp,
    *,
    replica: int,
    results_dir: Path,
    amounts_sats: Sequence[int] = REALISTIC_TRAFFIC_AMOUNTS_SATS,
) -> dict[str, Any]:
    """Apply identical, exact-path LND demand to both held controllers.

    This is a functional replenishment lane, not competitive route-share
    scoring.  It separates controller response from payer path selection after
    a scored block has established live profitability and a fresh cooldown.
    """
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") != "post_rebalance_ready":
        raise RunnerError(
            "equal targeted pressure requires retired return paths after an observation"
        )
    held = state.get("post_rebalance_controllers_held")
    if not isinstance(held, dict) or any(
        held.get(controller) is not True for controller in ("revenue_ops", "clboss")
    ):
        raise RunnerError("equal targeted pressure requires both controllers held")
    if held.get("forced_cycles") is not False:
        raise RunnerError("equal targeted pressure rejects forced-cycle lineage")
    if state.get("targeted_pressure") is not None:
        raise RunnerError("equal targeted pressure already completed for this epoch")

    amounts = tuple(nonnegative_int(value, "targeted pressure amount") for value in amounts_sats)
    if not amounts or any(value <= 0 for value in amounts):
        raise RunnerError("targeted pressure amounts must be positive")
    total_sats = sum(amounts)
    before = controlled_depletion_snapshot(state, family="lnd")
    state["status"] = "targeted_pressure_running"
    _checkpoint(
        path,
        state,
        "equal_targeted_pressure_started",
        amounts_sats=list(amounts),
    )

    payments = []
    for amount_sats in amounts:
        for controller in ("revenue_ops", "clboss"):
            payments.append(
                _directed_lnd_fixture_payment(
                    bridge,
                    network_id=int(state["network_id"]),
                    target_scid=before[controller]["source"]["short_channel_id"],
                    amount_sats=amount_sats,
                    label=f"targeted-{controller}",
                )
            )
    after = controlled_depletion_snapshot(state, family="lnd")
    minimum_drop_ppm = total_sats * 900_000 // CHANNEL_CAPACITY_SATS
    for controller in ("revenue_ops", "clboss"):
        drop_ppm = (
            before[controller]["depleted"]["local_balance_ppm"]
            - after[controller]["depleted"]["local_balance_ppm"]
        )
        if drop_ppm < minimum_drop_ppm:
            raise RunnerError(
                f"equal targeted pressure did not deplete {controller}: {drop_ppm} ppm"
            )
    state["targeted_pressure"] = {
        "family": "lnd",
        "competitive_scoring": False,
        "amounts_sats_per_controller": list(amounts),
        "total_sats_per_controller": total_sats,
        "controllers_held": True,
        "forced_cycles": False,
        "before": before,
        "after": after,
        "payments": payments,
    }
    state["status"] = "post_rebalance_ready"
    _checkpoint(
        path,
        state,
        "equal_targeted_pressure_complete",
        result=state["targeted_pressure"],
    )
    return state


def provision_return_paths(
    bridge: PolarMcp,
    *,
    replica: int,
    results_dir: Path,
    capacity_sats: int = RETURN_PATH_CAPACITY_SATS,
    fee_ppm: int = RETURN_PATH_FEE_PPM,
) -> dict[str, Any]:
    """Open fresh return liquidity toward the controlled depleted lane.

    These tournament-only channels are deliberately absent during scored
    payments. Once opened they provide both controllers with the same fresh
    circular-return direction: payer-to-sink for the historical sink-depleted
    fixture, or sink-to-payer for a payer-depleted fixture. They do not change
    either contender's four-channel topology or offer a scoring-time bypass.
    """
    path = state_path(results_dir, replica)
    state = read_state(path)
    targeted_ready = (
        state.get("status") == "post_rebalance_ready"
        and isinstance(state.get("targeted_pressure"), dict)
    )
    if (
        state.get("status") not in {"smoke_complete", "controlled_depletion_ready"}
        and not targeted_ready
    ) or state.get("league") != "full_stack":
        raise RunnerError(
            "return paths require a completed full-stack traffic block "
            "or controlled depletion"
        )
    existing_paths = state.get("return_paths")
    resume_partial = (
        isinstance(existing_paths, list)
        and len(existing_paths) == 2
        and {row.get("family") for row in existing_paths if isinstance(row, dict)}
        == {"cln", "lnd"}
        and all(
            isinstance(row, dict) and not row.get("short_channel_id")
            for row in existing_paths
        )
        and all(
            any(
                isinstance(event, dict)
                and event.get("event") == "return_path_open_dispatched"
                and event.get("family") == family
                for event in state.get("events", [])
            )
            for family in ("cln", "lnd")
        )
    )
    if existing_paths and not resume_partial:
        prepare_return_path_renewal(path, state)
    if state.get("return_paths") and not resume_partial:
        raise RunnerError("return paths are already checkpointed")
    for key in ("acquisition_treatment", "automatic_acquisition", "retention_treatment"):
        treatment = state.get(key)
        if isinstance(treatment, dict) and treatment.get("status") == "active":
            raise RunnerError(f"return paths require restored treatment: {key}")
    capacity_sats = nonnegative_int(capacity_sats, "return-path capacity")
    if capacity_sats == 0:
        raise RunnerError("return-path capacity must be positive")
    fee_ppm = nonnegative_int(fee_ppm, "return-path fee")
    if fee_ppm == 0:
        raise RunnerError("return-path fee must be positive")
    network_id = int(state["network_id"])
    cln_payer = f"polar-n{network_id}-cln-payer"
    cln_sink = f"polar-n{network_id}-cln-sink"
    lnd_payer = f"polar-n{network_id}-lnd-payer"
    lnd_sink = f"polar-n{network_id}-lnd-sink"
    cln_payer_id = str(cln_rpc(cln_payer, "getinfo").get("id") or "")
    cln_sink_id = str(cln_rpc(cln_sink, "getinfo").get("id") or "")
    lnd_payer_id = str(lnd_rpc(lnd_payer, "getinfo").get("identity_pubkey") or "")
    lnd_sink_id = str(lnd_rpc(lnd_sink, "getinfo").get("identity_pubkey") or "")
    if not cln_payer_id or not cln_sink_id or not lnd_payer_id or not lnd_sink_id:
        raise RunnerError("return-path endpoint identity readback failed")
    controlled = state.get("controlled_depletion")
    depleted_side = (
        str(controlled.get("depleted_side") or "sink")
        if isinstance(controlled, dict)
        else "sink"
    )
    if depleted_side not in {"payer", "sink"}:
        raise RunnerError(f"return path has invalid depleted side: {depleted_side!r}")
    if depleted_side == "payer":
        cln_source, cln_destination = cln_sink, cln_payer
        cln_source_id, cln_destination_id = cln_sink_id, cln_payer_id
        cln_destination_alias = "cln-payer"
        lnd_source, lnd_destination = lnd_sink, lnd_payer
        lnd_source_id, lnd_destination_id = lnd_sink_id, lnd_payer_id
        lnd_destination_alias = "lnd-payer"
        return_direction = "sink_to_payer"
    else:
        cln_source, cln_destination = cln_payer, cln_sink
        cln_source_id, cln_destination_id = cln_payer_id, cln_sink_id
        cln_destination_alias = "cln-sink"
        lnd_source, lnd_destination = lnd_payer, lnd_sink
        lnd_source_id, lnd_destination_id = lnd_payer_id, lnd_sink_id
        lnd_destination_alias = "lnd-sink"
        return_direction = "payer_to_sink"
    if resume_partial:
        state["return_paths"] = _wait_return_paths(existing_paths)
        _checkpoint(path, state, "return_paths_reconciled_after_partial_open")
    else:
        if any(
            row.get("peer_id") == cln_destination_id
            and row.get("state") == "CHANNELD_NORMAL"
            for row in channel_rows(cln_source)
        ):
            raise RunnerError("CLN return source already has an active destination channel")
        if any(
            row.get("remote_pubkey") == lnd_destination_id
            for row in _live_lnd_channels(lnd_source)
        ):
            raise RunnerError("LND return source already has an active destination channel")

        funding_sats = capacity_sats + RETURN_PATH_FUNDING_BUFFER_SATS
        cln_address = onchain_address(cln_rpc(cln_source, "newaddr"))
        lnd_address = lnd_rpc(lnd_source, "newaddress", "p2tr").get("address")
        _send_fake_onchain_funds(bridge, network_id, cln_address, funding_sats)
        _checkpoint(path, state, "return_path_cln_wallet_funded", amount_sats=funding_sats)
        _send_fake_onchain_funds(bridge, network_id, str(lnd_address or ""), funding_sats)
        _checkpoint(path, state, "return_path_lnd_wallet_funded", amount_sats=funding_sats)

        _connect_cln(cln_source, cln_destination_id, cln_destination_alias)
        cln_open = cln_rpc(
            cln_source, "fundchannel", cln_destination_id, capacity_sats
        )
        cln_channel_id = cln_open.get("channel_id")
        if not isinstance(cln_channel_id, str) or not cln_channel_id:
            raise RunnerError("CLN return-path funding returned no channel id")
        state["return_paths"] = [{
            "family": "cln", "source_container": cln_source,
            "sink_container": cln_destination, "peer_id": cln_destination_id,
            "channel_id": cln_channel_id, "capacity_sats": capacity_sats,
        }]
        _checkpoint(path, state, "return_path_open_dispatched", family="cln")
        _mine(bridge, network_id, 6)

        _connect_lnd(lnd_source, lnd_destination_id, lnd_destination_alias)
        lnd_open = lnd_rpc(
            lnd_source, "openchannel", "--node_key", lnd_destination_id,
            "--local_amt", capacity_sats,
        )
        state["return_paths"].append({
            "family": "lnd", "source_container": lnd_source,
            "sink_container": lnd_destination, "peer_id": lnd_destination_id,
            "funding_txid": lnd_open.get("funding_txid"),
            "capacity_sats": capacity_sats,
        })
        _checkpoint(path, state, "return_path_open_dispatched", family="lnd")
        _mine(bridge, network_id, 6)
        state["return_paths"] = _wait_return_paths(state["return_paths"])
    # This public channel is tournament plumbing, but it still participates in
    # Revenue Ops' real market-gossip inputs.  The default prices it in the
    # realistic corridor band. Explicit cheap-route lanes remain positive-fee,
    # crossed, and read back from both contenders before controller release.
    cln_rpc(
        cln_source, "setchannel", state["return_paths"][0]["short_channel_id"],
        RETURN_PATH_FEE_BASE_MSAT, fee_ppm,
    )
    lnd_row = next(
        row for row in _live_lnd_channels(lnd_source)
        if row.get("channel_point") == state["return_paths"][1]["channel_point"]
    )
    local_id = str(lnd_rpc(lnd_source, "getinfo")["identity_pubkey"])
    if local_id != lnd_source_id:
        raise RunnerError("LND return-path source identity changed during setup")
    policy = _wait_lnd_local_policy(lnd_source, lnd_row["scid"], local_id)
    _set_lnd_policy(lnd_source, {
        "channel_point": lnd_row["channel_point"],
        "fee_base_msat": RETURN_PATH_FEE_BASE_MSAT,
        "fee_ppm": fee_ppm,
        "time_lock_delta": int(policy["time_lock_delta"]),
        "min_htlc_msat": int(policy["min_htlc"]),
        "max_htlc_msat": int(policy["max_htlc_msat"]),
    })
    wait_gossip_policies(state["contenders"], [
        {
            "short_channel_id": state["return_paths"][0]["short_channel_id"],
            "source": cln_source_id,
            "fee_ppm": fee_ppm,
        },
        {
            "short_channel_id": state["return_paths"][1]["short_channel_id"],
            "source": local_id,
            "fee_ppm": fee_ppm,
        },
    ])
    state["return_path_fixture"] = {
        "present_during_scored_traffic": False,
        "fee_base_msat": RETURN_PATH_FEE_BASE_MSAT,
        "fee_ppm": fee_ppm,
        "direction": return_direction,
        "depleted_side": depleted_side,
        "purpose": "isolate post-pressure circular-route availability",
    }
    state["status"] = "return_paths_ready"
    _checkpoint(path, state, "return_paths_ready", paths=state["return_paths"])
    return state


def prepare_return_path_renewal(path: Path, state: dict[str, Any]) -> bool:
    """Archive one confirmed-dead fixture pair before a warm controller epoch."""
    paths = state.get("return_paths")
    if not isinstance(paths, list) or not paths:
        return False
    retired = state.get("return_paths_retired")
    renewal_status = state.get("status") == "smoke_complete" or (
        state.get("status") == "post_rebalance_ready"
        and isinstance(state.get("targeted_pressure"), dict)
    )
    if (
        not renewal_status
        or not isinstance(retired, dict)
        or retired.get("confirmed_absent") is not True
    ):
        return False
    remaining = _live_return_paths(state)
    if remaining:
        raise RunnerError(
            f"return-path renewal found retired fixture still live: {remaining}"
        )
    history = state.setdefault("return_path_history", [])
    if not isinstance(history, list):
        raise RunnerError("return-path history is malformed")
    history.append({
        "paths": paths,
        "retired": retired,
        "fixture": state.get("return_path_fixture"),
    })
    state["return_paths"] = []
    state["return_path_closes_dispatched"] = False
    state.pop("return_paths_retired", None)
    state.pop("return_path_fixture", None)
    _checkpoint(
        path,
        state,
        "return_paths_archived_for_warm_epoch",
        completed_epoch=len(history),
    )
    return True


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


def original_peer_hosts(network_id: int) -> dict[str, str]:
    """Map the four fixed mixed-client peer identities to Docker DNS names."""
    return {
        str(cln_rpc(f"polar-n{network_id}-cln-payer", "getinfo")["id"]):
            "cln-payer",
        str(lnd_rpc(f"polar-n{network_id}-lnd-payer", "getinfo")["identity_pubkey"]):
            "lnd-payer",
        str(cln_rpc(f"polar-n{network_id}-cln-sink", "getinfo")["id"]):
            "cln-sink",
        str(lnd_rpc(f"polar-n{network_id}-lnd-sink", "getinfo")["identity_pubkey"]):
            "lnd-sink",
    }


def wait_gossip_policies(
    contenders: dict[str, dict[str, str]],
    expected: Sequence[dict[str, Any]],
    *,
    attempts: int = GOSSIP_POLICY_POLL_ATTEMPTS,
    poll_seconds: float = GOSSIP_POLICY_POLL_SECONDS,
) -> dict[str, dict[str, int]]:
    """Fail closed until both contenders see exact directional fee policies."""
    attempts = nonnegative_int(attempts, "gossip policy poll attempts")
    if attempts == 0:
        raise RunnerError("gossip policy poll attempts must be positive")
    if poll_seconds < 0:
        raise RunnerError("gossip policy poll seconds must be nonnegative")
    if not isinstance(contenders, dict) or len(contenders) != 2:
        raise RunnerError("gossip verification requires exactly two contenders")
    normalized: list[tuple[str, str, int]] = []
    for row in expected:
        if not isinstance(row, dict):
            raise RunnerError("malformed expected gossip policy")
        scid = str(row.get("short_channel_id") or "")
        source = str(row.get("source") or "")
        ppm = nonnegative_int(row.get("fee_ppm"), "expected gossip fee")
        if not scid or not source:
            raise RunnerError("expected gossip policy lacks channel identity")
        normalized.append((scid, source, ppm))
    if not normalized:
        raise RunnerError("gossip verification requires at least one policy")

    last: dict[str, dict[str, int]] = {}
    for attempt in range(attempts):
        last = {}
        complete = True
        for identity, contender in contenders.items():
            container = str(contender.get("container") or "")
            rows = cln_rpc(container, "listchannels").get("channels")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise RunnerError(f"malformed gossip readback from {identity}")
            observed: dict[tuple[str, str], int] = {}
            for row in rows:
                try:
                    observed[(
                        str(row.get("short_channel_id") or ""),
                        str(row.get("source") or ""),
                    )] = int(row.get("fee_per_millionth"))
                except (TypeError, ValueError):
                    continue
            identity_result: dict[str, int] = {}
            for scid, source, ppm in normalized:
                value = observed.get((scid, source))
                if value is None:
                    complete = False
                    continue
                identity_result[f"{scid}:{source}"] = value
                if value != ppm:
                    complete = False
            last[str(identity)] = identity_result
        if complete:
            return last
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    raise RunnerError(f"gossip policies did not converge: {last}")


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
    hosts = original_peer_hosts(network_id)
    for role in ("revenue-node", "cln-competitor"):
        container = f"polar-n{network_id}-{role}"
        source = str(cln_rpc(container, "getinfo").get("id") or "")
        if not source:
            raise RunnerError(f"cannot identify background router {role}")
        policies = []
        for row in active_channels(container):
            update = (row.get("updates") or {}).get("local") or {}
            peer_id = str(row.get("peer_id") or "")
            peer_host = hosts.get(peer_id)
            if peer_host is None:
                raise RunnerError(
                    f"cannot map background peer {role}:{peer_id} to a fixed host"
                )
            policies.append(
                {
                    "short_channel_id": row["short_channel_id"],
                    "source": source,
                    "peer_id": peer_id,
                    "peer_host": peer_host,
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


def apply_background_ppm(
    state: dict[str, Any], background_ppm: int
) -> dict[str, dict[str, int]]:
    if background_ppm <= 0:
        raise RunnerError("background ppm must be positive")
    captured = state.get("background_policies")
    if not isinstance(captured, dict):
        raise RunnerError("background policies have not been captured")
    for role, policies in captured["cln"].items():
        container = f"polar-n{state['network_id']}-{role}"
        for row in policies:
            # A setchannel update on an entirely disconnected CLN node remains
            # local.  Reconnect first so the exact directional policy reaches
            # both contenders' gossip views before a scored controller cycle.
            _connect_cln(container, row["peer_id"], row["peer_host"])
            cln_rpc(
                container, "setchannel", row["short_channel_id"],
                row["fee_base_msat"], background_ppm,
            )
    lnd_container = f"polar-n{state['network_id']}-lnd-competitor"
    for row in captured["lnd"]:
        _set_lnd_policy(lnd_container, row, ppm=background_ppm)
    expected = [
        {
            "short_channel_id": row["short_channel_id"],
            "source": row["source"],
            "fee_ppm": background_ppm,
        }
        for policies in captured["cln"].values()
        for row in policies
    ]
    return wait_gossip_policies(state["contenders"], expected)


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
    gossip = apply_background_ppm(state, background_ppm)
    state["status"] = "isolated_fee_only_ready"
    _checkpoint(
        path, state, "background_isolated", fee_ppm=background_ppm,
        gossip_verified=gossip,
    )
    return state


def prime_forced_paths(
    bridge: PolarMcp,
    *,
    replica: int,
    results_dir: Path,
    amount_sats: int = 5_000,
) -> dict[str, Any]:
    """Prove every contender/client/direction path before scored selection.

    These equal, exact-path readiness payments are deliberately unscored. They
    prevent a sender's first automatic route lookup from mistaking an untried
    contender path for an unavailable one and selecting the expensive fallback
    router. Both directions keep the readiness transfer approximately neutral.
    """
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") != "isolated_fee_only_ready":
        raise RunnerError(
            "forced-path readiness requires fresh isolated fee-only state"
        )
    if "forced_path_readiness" in state or any(
        isinstance(event, dict) and event.get("event") == "smoke_complete"
        for event in state.get("events", [])
    ):
        raise RunnerError(
            "forced-path readiness refuses prior readiness or scored traffic"
        )
    amount_sats = positive_int(amount_sats)
    acquisition_before = wait_for_native_acquisition_markets(
        state, expected_phase="acquisition"
    )
    # Payer-funded contender channels start with zero local balance. A reverse
    # proof needs the 1% channel reserve plus the probe amount to be present
    # first; otherwise the route is correctly unavailable despite healthy
    # gossip. The 1M-sat formal topology has a 10k-sat reserve plus variable
    # anchor/commitment-fee headroom. Seed 5% of capacity so the proof remains
    # valid across the lab's live feerate range.
    forward_amount_sats = max(amount_sats, CHANNEL_CAPACITY_SATS // 20)
    lane_map = state.get("lane_map")
    assignment = state.get("assignment")
    if not isinstance(lane_map, dict) or not isinstance(assignment, dict):
        raise RunnerError("forced-path readiness lacks lane metadata")
    payments: list[dict[str, Any]] = []
    for controller in ("revenue_ops", "clboss"):
        identity = assignment.get(controller)
        lanes = lane_map.get(identity) if isinstance(identity, str) else None
        if not isinstance(lanes, dict):
            raise RunnerError(f"forced-path readiness lacks {controller} lanes")
        by_role = {
            (row.get("family"), row.get("side")): scid
            for scid, row in lanes.items() if isinstance(row, dict)
        }
        expected = {
            (family, side)
            for family in ("cln", "lnd")
            for side in ("payer", "sink")
        }
        if set(by_role) != expected:
            raise RunnerError(
                f"forced-path readiness has incomplete {controller} lane roles"
            )
        for family in ("cln", "lnd"):
            payment_fn = (
                _directed_cln_fixture_payment
                if family == "cln" else _directed_lnd_fixture_payment
            )
            for direction, side in (("forward", "payer"), ("reverse", "sink")):
                payment_amount_sats = (
                    forward_amount_sats if direction == "forward" else amount_sats
                )
                payments.append(payment_fn(
                    bridge,
                    network_id=int(state["network_id"]),
                    target_scid=by_role[(family, side)],
                    amount_sats=payment_amount_sats,
                    label=f"readiness-{controller}-{family}-{direction}",
                    direction=direction,
                ))
    state["forced_path_readiness"] = {
        "forward_amount_sats": forward_amount_sats,
        "reverse_amount_sats": amount_sats,
        "scored": False,
        "payments": payments,
        "acquisition_before": acquisition_before,
        "acquisition_after": wait_for_native_acquisition_markets(
            state, expected_phase="retention"
        ),
    }
    _checkpoint(path, state, "forced_paths_primed", readiness=state["forced_path_readiness"])
    return state


def wait_for_native_acquisition_markets(
    state: dict[str, Any],
    *,
    expected_phase: str,
    attempts: int = NATIVE_CYCLE_POLL_ATTEMPTS,
    poll_seconds: float = NATIVE_CYCLE_POLL_SECONDS,
) -> list[dict[str, Any]]:
    """Await two native, paid-lifecycle acquisition markets read-only.

    Formal readiness itself supplies the first 50k-sat demand observation on
    each sink lane. Waiting for acquisition before those forwards prevents the
    fixture from suppressing cold-start eligibility; waiting for retention
    afterwards prevents free readiness traffic from leaking into scoring.
    No fee/action RPC is used here: the product's native timer and forward wake
    remain the only decision triggers.
    """
    if expected_phase not in {"acquisition", "retention"}:
        raise RunnerError(f"invalid acquisition readiness phase: {expected_phase!r}")
    if attempts <= 0 or poll_seconds < 0:
        raise RunnerError("invalid acquisition readiness polling bounds")
    assignment = state.get("assignment")
    contenders = state.get("contenders")
    lane_map = state.get("lane_map")
    identity = assignment.get("revenue_ops") if isinstance(assignment, dict) else None
    contender = contenders.get(identity) if isinstance(contenders, dict) else None
    container = contender.get("container") if isinstance(contender, dict) else None
    identity_lanes = lane_map.get(identity) if isinstance(lane_map, dict) else None
    if not isinstance(container, str) or not isinstance(identity_lanes, dict):
        raise RunnerError("acquisition readiness lacks Revenue contender lanes")
    expected_by_scid = {
        str(scid): str(row.get("family"))
        for scid, row in identity_lanes.items()
        if isinstance(row, dict)
        and row.get("side") == "sink"
        and row.get("family") in {"cln", "lnd"}
    }
    if set(expected_by_scid.values()) != {"cln", "lnd"}:
        raise RunnerError("acquisition readiness lacks both sink client families")

    last_rows: list[dict[str, Any]] = []
    for attempt in range(attempts):
        last_rows = _acquisition_rows(container)
        active = [row for row in last_rows if row.get("state") == "active"]
        selected: list[dict[str, Any]] = []
        live_by_scid = {
            str(row.get("short_channel_id")): row
            for row in active_channels(container)
        }
        for episode in active:
            scid = str(episode.get("channel_id") or "")
            family = expected_by_scid.get(scid)
            if family is None or episode.get("phase") != expected_phase:
                continue
            live = live_by_scid.get(scid)
            updates = live.get("updates") if isinstance(live, dict) else None
            local = updates.get("local") if isinstance(updates, dict) else None
            if expected_phase == "acquisition":
                ppm_key, base_key = "target_fee_ppm", "target_base_fee_msat"
            else:
                ppm_key, base_key = "retention_fee_ppm", "retention_base_fee_msat"
            try:
                expected_ppm = nonnegative_int(
                    episode.get(ppm_key), f"{family} {expected_phase} ppm"
                )
                expected_base = nonnegative_int(
                    episode.get(base_key), f"{family} {expected_phase} base fee"
                )
                live_ppm = nonnegative_int(
                    local.get("fee_proportional_millionths")
                    if isinstance(local, dict) else None,
                    f"live {family} acquisition ppm",
                )
                live_base = nonnegative_int(
                    local.get("fee_base_msat") if isinstance(local, dict) else None,
                    f"live {family} acquisition base fee",
                )
            except RunnerError:
                continue
            if (live_ppm, live_base) != (expected_ppm, expected_base):
                continue
            selected.append({
                "family": family,
                "channel_id": scid,
                "experiment_id": episode.get("id"),
                "phase": expected_phase,
                "fee_ppm": live_ppm,
                "fee_base_msat": live_base,
            })
        if (
            len(active) == EXPECTED_ACQUISITION_MARKETS
            and len(selected) == EXPECTED_ACQUISITION_MARKETS
            and {row["family"] for row in selected} == {"cln", "lnd"}
        ):
            return sorted(selected, key=lambda row: row["family"])
        if attempt + 1 < attempts:
            time.sleep(poll_seconds)
    raise RunnerError(
        f"native acquisition did not reach {expected_phase} on both client "
        f"markets before timeout: {last_rows}"
    )


def retune_background(
    *, replica: int, results_dir: Path, background_ppm: int
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {"isolated_fee_only_ready", "smoke_complete"}:
        raise RunnerError(f"replica is not isolated: {state.get('status')}")
    gossip = apply_background_ppm(state, background_ppm)
    state["status"] = "isolated_fee_only_ready"
    _checkpoint(
        path, state, "background_retuned", fee_ppm=background_ppm,
        gossip_verified=gossip,
    )
    return state


def restore_background(state: dict[str, Any]) -> bool:
    captured = state.get("background_policies")
    if not isinstance(captured, dict) or state.get("background_restored") is True:
        return False
    for role, policies in captured.get("cln", {}).items():
        container = f"polar-n{state['network_id']}-{role}"
        for row in policies:
            if row.get("peer_id") and row.get("peer_host"):
                _connect_cln(container, row["peer_id"], row["peer_host"])
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
    """Enable native rebalancing without reopening topology authority.

    ``spend_cap_sats`` is retained as a CLI/API compatibility name but applies
    only to Revenue Ops' native budget. CLBOSS remains uncapped, matching its
    ordinary behavior; its actual circular-payment cost is still measured.
    """
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
        ("clboss-xrebalance-per-hour", str(CLBOSS_REBALANCES_PER_HOUR)),
    ):
        cln_rpc(clboss_container, "setconfig", key, value)
    cln_rpc(clboss_container, "setconfig", "clboss-rebalance-mode", "xrebalance")
    configs = cln_rpc(clboss_container, "listconfigs").get("configs", {})
    expected = {
        "clboss-rebalance-mode": "xrebalance",
        "clboss-xrebalance-gain": "1",
        "clboss-xrebalance-grant": "0",
        "clboss-xrebalance-route-cost-floor": "auto",
        "clboss-xrebalance-per-hour": str(CLBOSS_REBALANCES_PER_HOUR),
    }
    for key, value in expected.items():
        row = configs.get(key, {}) if isinstance(configs, dict) else {}
        if str(row.get("value_str")) != value:
            raise RunnerError(f"CLBOSS full-stack readback mismatch for {key}")

    state["league"] = "full_stack"
    state["full_stack"] = {
        "revenue_rebalance_allowance_sats": spend_cap_sats,
        "clboss_spend_policy": "native_unbounded",
        "revenue_baseline_non_rebalance_sats": baseline_non_rebalance_sats,
        "revenue_runtime_budget_sats": revenue_budget_sats,
        "revenue_rebalance_cost_baseline_msat": rebalance_cost_msat(revenue_container),
        "clboss_rebalance_cost_baseline_msat": rebalance_cost_msat(clboss_container),
        "clboss": expected,
        "cadence": {
            "revenue_seconds": TOURNAMENT_CYCLE_SECONDS,
            "clboss_rebalances_per_hour": CLBOSS_REBALANCES_PER_HOUR,
        },
    }
    state["status"] = "isolated_full_stack_ready"
    _checkpoint(path, state, "full_stack_enabled", controls=state["full_stack"])
    return state


def accelerate_controllers(*, replica: int, results_dir: Path) -> dict[str, Any]:
    """Restart Revenue Ops at the lab cadence and retune CLBOSS dynamically."""
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {
        "isolated_full_stack_ready", "smoke_complete", "return_paths_ready",
    }:
        raise RunnerError(f"replica is not cadence-ready: {state.get('status')}")
    full_stack = state.get("full_stack")
    if not isinstance(full_stack, dict):
        raise RunnerError("cadence acceleration requires full-stack controls")
    assignment = state["assignment"]
    contenders = state["contenders"]
    revenue_container = contenders[assignment["revenue_ops"]]["container"]
    clboss_container = contenders[assignment["clboss"]]["container"]
    budget_sats = nonnegative_int(
        full_stack.get("revenue_runtime_budget_sats"), "Revenue runtime budget"
    )
    if "revenue_rebalance_allowance_sats" not in full_stack:
        full_stack["revenue_rebalance_allowance_sats"] = nonnegative_int(
            full_stack.get("spend_cap_sats_per_controller"),
            "legacy Revenue rebalance allowance",
        )
    full_stack.pop("spend_cap_sats_per_controller", None)
    stopped = _run(
        [
            "docker", "exec", revenue_container, "lightning-cli", "--network=regtest",
            "-k", "plugin", "subcommand=stop", f"plugin={REVENUE_PLUGIN}",
        ],
        check=False,
    )
    if stopped.returncode != 0:
        detail = (stopped.stderr or stopped.stdout or "").strip()
        raise RunnerError(f"Revenue cadence restart could not stop plugin: {detail}")
    cln_rpc(
        revenue_container, "-k", "plugin", "subcommand=start", f"plugin={REVENUE_PLUGIN}",
        "revenue-ops-dry-run=false",
        f"revenue-ops-daily-budget-sats={budget_sats}",
        f"revenue-ops-flow-interval={TOURNAMENT_CYCLE_SECONDS}",
        f"revenue-ops-fee-interval={TOURNAMENT_CYCLE_SECONDS}",
        f"revenue-ops-rebalance-interval={TOURNAMENT_CYCLE_SECONDS}",
    )
    configs = cln_rpc(revenue_container, "listconfigs").get("configs", {})
    for key in (
        "revenue-ops-flow-interval", "revenue-ops-fee-interval",
        "revenue-ops-rebalance-interval",
    ):
        row = configs.get(key, {}) if isinstance(configs, dict) else {}
        if str(row.get("value_str")) != str(TOURNAMENT_CYCLE_SECONDS):
            raise RunnerError(f"Revenue cadence restart mismatch for {key}")
    # ``paused`` is a durable runtime control.  A cadence-only plugin restart
    # must explicitly release a previously held controller before claiming
    # acceleration succeeded.
    cln_rpc(revenue_container, "revenue-config", "set", "paused", "false")
    status = cln_rpc(revenue_container, "revenue-status")
    controls = status.get("operator_controls", {}).get("values", {})
    if controls.get("paused") is not False or controls.get("daily_budget_sats") != budget_sats:
        raise RunnerError("Revenue cadence restart changed operator controls")
    cln_rpc(
        clboss_container, "setconfig", "clboss-xrebalance-per-hour",
        str(CLBOSS_REBALANCES_PER_HOUR),
    )
    cln_rpc(clboss_container, "setconfig", "clboss-rebalance-mode", "xrebalance")
    clboss_configs = cln_rpc(clboss_container, "listconfigs").get("configs", {})
    cadence_row = (
        clboss_configs.get("clboss-xrebalance-per-hour", {})
        if isinstance(clboss_configs, dict) else {}
    )
    if str(cadence_row.get("value_str")) != str(CLBOSS_REBALANCES_PER_HOUR):
        raise RunnerError("CLBOSS accelerated cadence readback mismatch")
    full_stack["clboss_spend_policy"] = "native_unbounded"
    full_stack["cadence"] = {
        "revenue_seconds": TOURNAMENT_CYCLE_SECONDS,
        "clboss_rebalances_per_hour": CLBOSS_REBALANCES_PER_HOUR,
    }
    _checkpoint(path, state, "controllers_accelerated", cadence=full_stack["cadence"])
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


def _lane_matches_channel_identifier(lane: dict[str, Any], identifier: object) -> bool:
    """Match Revenue Ops rows whether they persist a funding id or live SCID."""
    value = str(identifier or "")
    if not value:
        return False
    return value in {
        str(lane.get("channel_id") or ""),
        str(lane.get("short_channel_id") or ""),
    }


def start_automatic_acquisition(
    *, replica: int, results_dir: Path,
    attempts: int = NATIVE_CYCLE_POLL_ATTEMPTS,
    poll_seconds: float = NATIVE_CYCLE_POLL_SECONDS,
) -> dict[str, Any]:
    """Capture the product's acquisition gate and await native admission."""
    if attempts <= 0 or poll_seconds < 0:
        raise RunnerError("invalid automatic acquisition polling bounds")
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") not in {"isolated_full_stack_ready", "smoke_complete"}:
        raise RunnerError(f"replica is not automatic-acquisition-ready: {state.get('status')}")
    prior_automatic = state.get("automatic_acquisition")
    if isinstance(prior_automatic, dict) and prior_automatic.get("status") == "restored":
        history = state.setdefault("automatic_acquisition_history", [])
        if not isinstance(history, list):
            raise RunnerError("automatic acquisition history is malformed")
        history.append(prior_automatic)
        del state["automatic_acquisition"]
    elif "automatic_acquisition" in state or "acquisition_treatment" in state:
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
                if _lane_matches_channel_identifier(
                    lane, episode.get("channel_id")
                )
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


def refresh_automatic_acquisition_phase(state: dict[str, Any]) -> str | None:
    """Read back a native acquisition/retention phase before scoring traffic."""
    automatic = state.get("automatic_acquisition")
    if not isinstance(automatic, dict) or automatic.get("status") != "active":
        return None
    episode = automatic.get("episode")
    if not isinstance(episode, dict):
        raise RunnerError("automatic acquisition lacks captured episode evidence")
    experiment_id = nonnegative_int(
        episode.get("id"), "automatic acquisition experiment id"
    )
    identity = state["assignment"]["revenue_ops"]
    container = state["contenders"][identity]["container"]
    acquisition_rows = _acquisition_rows(container)
    matching_rows = [
        row for row in acquisition_rows
        if row.get("id") == experiment_id and row.get("state") == "active"
    ]
    lanes: dict[str, dict[str, Any]] | None = None
    if len(matching_rows) == 1:
        live_episode = matching_rows[0]
    elif not matching_rows:
        completed_rows = [
            row for row in acquisition_rows
            if row.get("id") == experiment_id and row.get("state") == "completed"
        ]
        active_rows = [
            row for row in acquisition_rows if row.get("state") == "active"
        ]
        if len(completed_rows) != 1 or len(active_rows) != 1:
            raise RunnerError(
                "automatic acquisition is no longer exactly one active episode"
            )
        lanes = acquisition_lanes(state)
        completed = completed_rows[0]
        live_episode = active_rows[0]
        next_id = nonnegative_int(
            live_episode.get("id"), "automatic rollover experiment id"
        )
        if next_id == experiment_id:
            raise RunnerError("automatic acquisition rollover reused an episode id")
        previous_lane = automatic.get("lane")
        if not isinstance(previous_lane, dict):
            raise RunnerError("automatic acquisition rollover lacks captured lane")
        restored_lanes = [
            lane for lane in lanes.values()
            if _lane_matches_channel_identifier(
                lane, completed.get("channel_id")
            )
        ]
        next_lanes = [
            lane for lane in lanes.values()
            if _lane_matches_channel_identifier(
                lane, live_episode.get("channel_id")
            )
        ]
        if len(restored_lanes) != 1 or len(next_lanes) != 1:
            raise RunnerError("automatic acquisition rollover selected an unscored lane")
        restored_lane, next_lane = restored_lanes[0], next_lanes[0]
        if restored_lane["channel_id"] == next_lane["channel_id"]:
            raise RunnerError("automatic acquisition rollover ignored channel cooldown")
        restored_ppm = nonnegative_int(
            completed.get("restored_fee_ppm"),
            "automatic rollover restored fee",
        )
        restored_base_msat = nonnegative_int(
            completed.get("restored_base_fee_msat"),
            "automatic rollover restored base fee",
        )
        baseline_ppm = nonnegative_int(
            completed.get("baseline_fee_ppm"),
            "automatic rollover baseline fee",
        )
        baseline_base_msat = nonnegative_int(
            completed.get("baseline_base_fee_msat"),
            "automatic rollover baseline base fee",
        )
        if (
            restored_ppm != baseline_ppm
            or restored_base_msat != baseline_base_msat
        ):
            raise RunnerError(
                "automatic acquisition rollover did not restore its captured lane"
            )
        rollovers = automatic.setdefault("rollovers", [])
        if not isinstance(rollovers, list):
            raise RunnerError("automatic acquisition rollover history is malformed")
        rollovers.append({
            "completed_episode": completed,
            "restored_lane": restored_lane,
            "restoration_evidence": {
                "baseline_fee_ppm": baseline_ppm,
                "baseline_base_fee_msat": baseline_base_msat,
                "restored_fee_ppm": restored_ppm,
                "restored_base_fee_msat": restored_base_msat,
                "current_fee_ppm": restored_lane["fee_ppm"],
                "current_base_fee_msat": restored_lane["fee_base_msat"],
            },
            "next_episode_id": next_id,
        })
        automatic.update(episode=live_episode, lane=next_lane)
    else:
        raise RunnerError("automatic acquisition has duplicate active episode rows")
    phase = str(live_episode.get("phase") or "acquisition")
    if phase not in {"acquisition", "retention"}:
        raise RunnerError(f"automatic acquisition returned invalid phase {phase!r}")
    if lanes is None:
        lanes = acquisition_lanes(state)
    matching_lanes = [
        lane for lane in lanes.values()
        if _lane_matches_channel_identifier(lane, live_episode.get("channel_id"))
    ]
    if len(matching_lanes) != 1:
        raise RunnerError("automatic acquisition phase selected an unscored lane")
    lane = matching_lanes[0]
    target_key = "retention_fee_ppm" if phase == "retention" else "target_fee_ppm"
    base_target_key = (
        "retention_base_fee_msat"
        if phase == "retention"
        else "target_base_fee_msat"
    )
    target_ppm = nonnegative_int(
        live_episode.get(target_key), f"automatic {phase} target fee"
    )
    target_base_msat = nonnegative_int(
        live_episode.get(base_target_key, 0),
        f"automatic {phase} target base fee",
    )
    if phase == "retention" and (target_ppm != 0 or target_base_msat <= 0):
        raise RunnerError(
            "automatic paid retention did not expose a positive base-fee undercut"
        )
    if lane["fee_ppm"] != target_ppm:
        raise RunnerError(
            f"automatic {phase} fee readback mismatch: "
            f"live={lane['fee_ppm']} target={target_ppm}"
        )
    if lane["fee_base_msat"] != target_base_msat:
        raise RunnerError(
            f"automatic {phase} base-fee readback mismatch: "
            f"live={lane['fee_base_msat']} target={target_base_msat}"
        )
    automatic.update(episode=live_episode, lane=lane, phase=phase)
    return phase


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

        state["payer_wallet_topups"] = ensure_payer_wallet_funds(bridge, network_id)
        _checkpoint(
            path, state, "payer_wallets_ready", balances=state["payer_wallet_topups"]
        )

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
    rebalance_delivered_msat: int = 0
    rebalance_completed_count: int = 0
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


def rebalance_totals(container: str) -> dict[str, int]:
    """Return completed circular-payment count, volume, and routing cost."""
    node_id = str(cln_rpc(container, "getinfo").get("id") or "")
    payments = cln_rpc(container, "listsendpays").get("payments", [])
    if not node_id or not isinstance(payments, list):
        raise RunnerError(f"malformed circular-payment readback for {container}")
    completed = [
        row for row in payments
        if isinstance(row, dict)
        and row.get("status") == "complete"
        and row.get("destination") == node_id
    ]
    delivered = sum(msat_value(row.get("amount_msat", 0)) for row in completed)
    sent = sum(msat_value(row.get("amount_sent_msat", 0)) for row in completed)
    if sent < delivered:
        raise RunnerError(f"circular payment cost regressed on {container}")
    return {
        "completed_count": len(completed),
        "delivered_msat": delivered,
        "cost_msat": sent - delivered,
    }


def rebalance_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = ("completed_count", "delivered_msat", "cost_msat")
    delta = {
        key: nonnegative_int(after.get(key), f"after {key}")
        - nonnegative_int(before.get(key), f"before {key}")
        for key in keys
    }
    if any(value < 0 for value in delta.values()):
        raise RunnerError("circular-payment counters regressed")
    return delta


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
    native_rebalances = rebalance_totals(container)
    return Totals(
        forward_count=len(settled),
        volume_msat=sum(msat_value(row.get("out_msat", 0)) for row in settled),
        routing_fee_msat=sum(msat_value(row.get("fee_msat", 0)) for row in settled),
        mean_local_liquidity_sats=sum(local_sats) // len(local_sats),
        policy_fingerprint=tuple(sorted(policies)),
        rebalance_cost_msat=native_rebalances["cost_msat"],
        rebalance_delivered_msat=native_rebalances["delivered_msat"],
        rebalance_completed_count=native_rebalances["completed_count"],
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
        "rebalance_delivered_msat": (
            after.rebalance_delivered_msat - before.rebalance_delivered_msat
        ),
        "rebalance_completed_count": (
            after.rebalance_completed_count - before.rebalance_completed_count
        ),
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


def payer_payment_outcome(payer: str, payment_hash: str) -> str:
    """Return a payer's authoritative aggregate outcome, or ``unknown``.

    Part-level CLN ``listsendpays`` rows are not terminal evidence: the pay
    plugin can create another part after all currently visible parts fail.
    ``listpays`` provides the aggregate state.  LND exposes the corresponding
    aggregate state through ``listpayments --include_incomplete``.
    """
    if not isinstance(payment_hash, str) or not payment_hash:
        return "unknown"
    container = f"polar-n{NETWORK_ID}-{payer}"
    try:
        if payer.startswith("cln-"):
            payload = cln_rpc(
                container, "-k", "listpays", f"payment_hash={payment_hash}"
            )
            rows = payload.get("pays")
            if not isinstance(rows, list):
                return "unknown"
            matches = [
                row for row in rows
                if isinstance(row, dict)
                and row.get("payment_hash") == payment_hash
            ]
            statuses = {
                str(row.get("status") or "").casefold() for row in matches
            }
            if "complete" in statuses:
                return "settled"
            if "pending" in statuses:
                return "pending"
            if matches and statuses == {"failed"}:
                return "failed"
            return "unknown"
        if payer.startswith("lnd-"):
            payload = lnd_rpc(container, "listpayments", "--include_incomplete")
            rows = payload.get("payments")
            if not isinstance(rows, list):
                return "unknown"
            matches = [
                row for row in rows
                if isinstance(row, dict)
                and str(row.get("payment_hash") or "").casefold()
                == payment_hash.casefold()
            ]
            statuses = {
                str(row.get("status") or "").casefold() for row in matches
            }
            if "succeeded" in statuses:
                return "settled"
            if "in_flight" in statuses:
                return "pending"
            if matches and statuses == {"failed"}:
                return "failed"
            return "unknown"
    except RunnerError:
        return "unknown"
    return "unknown"


def payment_outcome(payer: str, sink: str, payment_hash: str) -> str:
    """Reconcile destination and payer evidence without raising on bad RPC data."""
    try:
        if invoice_settled(sink, payment_hash):
            return "settled"
    except RunnerError:
        pass
    return payer_payment_outcome(payer, payment_hash)


def has_terminal_payment_failure(records: list[dict[str, Any]]) -> bool:
    """Stop a traffic block after a proven failure; never retry depleted paths."""
    return any(
        not isinstance(row, dict)
        or not isinstance(row.get("payment"), dict)
        or row["payment"].get("success") is not True
        for row in records
    )


def run_reconciled_traffic(
    bridge: PolarMcp,
    *,
    network_id: int,
    rounds: int,
    amount_sats: int,
    pause_seconds: float,
    lanes: tuple[tuple[str, str], ...],
    reconciliation_attempts: int = PAYMENT_RECONCILIATION_POLL_ATTEMPTS,
    reconciliation_poll_seconds: float = PAYMENT_RECONCILIATION_POLL_SECONDS,
) -> list[dict[str, Any]]:
    """Dispatch each payment once and reconcile Polar's known post-pay UI 500."""
    if reconciliation_attempts <= 0 or reconciliation_poll_seconds < 0:
        raise RunnerError("invalid payment reconciliation polling bounds")
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
                outcome = "unknown"
                for attempt in range(reconciliation_attempts):
                    outcome = payment_outcome(payer, sink, payment_hash)
                    if outcome in {"settled", "failed"}:
                        break
                    if attempt + 1 < reconciliation_attempts:
                        time.sleep(reconciliation_poll_seconds)
                if outcome not in {"settled", "failed"}:
                    operation = {
                        "round": round_index,
                        "payer": payer,
                        "sink": sink,
                        "amount_sats": amount_sats,
                        "payment_hash": payment_hash,
                        "outcome": "unknown_do_not_retry",
                        "last_observed_outcome": outcome,
                        "error": str(exc),
                    }
                    raise ReconciliationError(
                        "payment dispatch could not be reconciled", list(records), operation
                    ) from exc
                if outcome == "settled":
                    payment = {
                        "success": True,
                        "reconciled_after_mcp_error": True,
                        "bridge_error": str(exc),
                    }
                else:
                    payment = {
                        "success": False,
                        "reconciled_terminal_failure": True,
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
    traffic_family: str = "both",
) -> tuple[tuple[str, str, int], ...]:
    """Build balanced competition or one-way liquidity-pressure traffic."""
    if rounds <= 0 or amount_sats <= 0:
        raise RunnerError("traffic rounds and amount must be positive")
    if pattern not in {"balanced", "forward-pressure", "reverse-pressure"}:
        raise RunnerError(f"unknown traffic pattern: {pattern}")
    if amount_profile not in {"fixed", "realistic"}:
        raise RunnerError(f"unknown amount profile: {amount_profile}")
    if traffic_family not in {"both", "cln", "lnd"}:
        raise RunnerError(f"unknown traffic family: {traffic_family}")
    families = ("cln", "lnd") if traffic_family == "both" else (traffic_family,)
    schedule = []
    for round_index in range(rounds):
        round_amount = (
            REALISTIC_TRAFFIC_AMOUNTS_SATS[
                round_index % len(REALISTIC_TRAFFIC_AMOUNTS_SATS)
            ]
            if amount_profile == "realistic"
            else amount_sats
        )
        for family in families:
            if pattern in {"forward-pressure", "reverse-pressure"}:
                direction = (
                    "forward" if pattern == "forward-pressure" else "reverse"
                )
                schedule.append((family, direction, round_amount))
                continue
            forward_amount = round_amount + (
                REVERSE_FEE_BUFFER_SATS if round_index == 0 else 0
            )
            schedule.append((family, "forward", forward_amount))
            schedule.append((family, "reverse", round_amount))
    return tuple(schedule)


def _completed_smoke_exists_for_league(replica_dir: Path, league: str) -> bool:
    """Return whether this replica already has a valid block in ``league``.

    Route-cache state is league-local: switching from fee-only to full-stack
    starts a new cold observation even though the contender wallets remain the
    same. Existing artifacts are authoritative and malformed artifacts fail
    closed instead of silently turning a warm block into a cold one.
    """
    if league not in {"fee_only", "full_stack"}:
        raise RunnerError(f"unknown smoke league: {league!r}")
    for artifact in sorted(replica_dir.glob("smoke-*.json")):
        if artifact.name.endswith("-progress.json"):
            continue
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunnerError(f"cannot read prior smoke artifact {artifact}: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "polar-clboss-smoke-v1"
        ):
            raise RunnerError(f"unexpected prior smoke schema in {artifact}")
        if payload.get("league") == league:
            return True
    return False


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
    traffic_family: str = "both",
) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    entry_status = state.get("status")
    if state.get("status") not in {
        "fee_only_ready", "smoke_complete", "isolated_fee_only_ready",
        "isolated_full_stack_ready", "acquisition_ready",
        "automatic_acquisition_ready", "retention_ready",
        "post_rebalance_ready",
    }:
        raise RunnerError(f"replica is not traffic-ready: {state.get('status')}")
    if entry_status == "post_rebalance_ready":
        observation = state.get("last_rebalance_observation")
        retired = state.get("return_paths_retired")
        held = state.get("post_rebalance_controllers_held")
        if not isinstance(observation, str) or not observation:
            raise RunnerError("post-rebalance traffic lacks observation lineage")
        if not isinstance(retired, dict) or retired.get("confirmed_absent") is not True:
            raise RunnerError("post-rebalance traffic requires retired return paths")
        if not isinstance(held, dict) or any(
            held.get(controller) is not True for controller in ("revenue_ops", "clboss")
        ):
            raise RunnerError("post-rebalance traffic requires both controllers held")
        if held.get("forced_cycles") is not False:
            raise RunnerError("post-rebalance traffic has invalid controller-hold lineage")
    refresh_automatic_acquisition_phase(state)
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
    current_league = str(state.get("league") or "fee_only")
    cache_mode = (
        "warm"
        if _completed_smoke_exists_for_league(path.parent, current_league)
        else "cold"
    )
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
        "traffic_family": traffic_family,
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
        family_count = 2 if traffic_family == "both" else 1
        entries_per_round = family_count * (
            2 if traffic_pattern == "balanced" else 1
        )
        for schedule_index, (family, direction, traffic_amount) in enumerate(traffic_schedule(
            rounds,
            amount_sats,
            traffic_pattern,
            effective_amount_profile,
            traffic_family,
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
            write_json_atomic(progress_path, progress)
            if has_terminal_payment_failure(completed):
                break
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
        settled_records = [
            row for row in family_records
            if isinstance(row.get("payment"), dict)
            and row["payment"].get("success") is True
        ]
        settled_volume_msat = sum(
            nonnegative_int(row.get("amount_sats"), "settled traffic amount") * 1000
            for row in settled_records
        )
        contender_volume_msat = sum(
            family_deltas[name][family]["volume_msat"]
            for name in controller_containers
        )
        family_rows[family] = {
            "attempted": len(family_records),
            "settled": len(settled_records),
            "settled_volume_msat": settled_volume_msat,
            "contender_volume_msat": contender_volume_msat,
            "contenders": {
                name: family_deltas[name][family] for name in controller_containers
            },
        }
    settled_count = sum(row["settled"] for row in family_rows.values())
    settled_volume_msat = sum(
        row["settled_volume_msat"] for row in family_rows.values()
    )
    contender_forward_count = sum(
        delta["forward_count"] for delta in contender_deltas.values()
    )
    contender_volume_msat = sum(
        row["contender_volume_msat"] for row in family_rows.values()
    )
    fallback_volume_msat = sum(
        max(0, row["settled_volume_msat"] - row["contender_volume_msat"])
        for row in family_rows.values()
    )
    raw_extra_volume_msat = sum(
        max(0, row["contender_volume_msat"] - row["settled_volume_msat"])
        for row in family_rows.values()
    )
    native_rebalance_volume_msat = sum(
        delta["rebalance_delivered_msat"] for delta in contender_deltas.values()
    )
    native_rebalance_completed_count = sum(
        delta["rebalance_completed_count"] for delta in contender_deltas.values()
    )
    native_rebalance_cost_msat = sum(
        delta["rebalance_cost_msat"] for delta in contender_deltas.values()
    )
    native_rebalance_sent_msat = (
        native_rebalance_volume_msat + native_rebalance_cost_msat
    )
    # A completed self-payment can use a non-contender return path (zero extra
    # volume) or traverse the opposing contender once. At an intermediate hop,
    # out_msat is bounded by the payment's delivered and sent amounts because
    # it may still include downstream fees. A value below delivered or above
    # sent could hide fallback traffic or unrelated forwards and fails closed.
    attributable_native_rebalance = (
        raw_extra_volume_msat == 0
        or (
            native_rebalance_volume_msat > 0
            and native_rebalance_volume_msat
            <= raw_extra_volume_msat
            <= native_rebalance_sent_msat
        )
    )
    attributed_native_rebalance_volume_msat = (
        raw_extra_volume_msat if attributable_native_rebalance else 0
    )
    extra_volume_msat = (
        0 if attributable_native_rebalance else raw_extra_volume_msat
    )
    # A single Lightning payment may settle through multiple HTLCs. Count
    # equality therefore rejects valid MPP traffic; exact per-family outgoing
    # volume remains invariant under splitting and still catches fallback or
    # unrelated contender traffic. The legacy count-shaped field remains a
    # conservative gate (zero or at least one), not a claimed exact count.
    fallback_settled = 1 if fallback_volume_msat else 0
    if settled_count != len(records):
        safety_violations.append("unsettled_payments")
    if fallback_settled:
        safety_violations.append("fallback_settled")
    if extra_volume_msat:
        safety_violations.append("unattributed_extra_contender_volume")
    if native_rebalance_volume_msat and not attributable_native_rebalance:
        safety_violations.append("native_rebalance_volume_mismatch")
    block = {
        "schema": "polar-clboss-smoke-v1",
        "replica": f"replica-{replica}",
        "league": current_league,
        "market_profile": str(state.get("market_profile") or "legacy_low_fee"),
        "block": block_id,
        "duration_seconds": max(1.0, time.time() - started),
        "cache_mode": cache_mode,
        "traffic": {
            "pattern": traffic_pattern,
            "family_scope": traffic_family,
            "amount_profile": effective_amount_profile,
            "amounts_sats": sorted({int(row["amount_sats"]) for row in records}),
            "attempted": len(records),
            "settled": settled_count,
            "fallback_settled": fallback_settled,
            "settled_volume_msat": settled_volume_msat,
            "contender_volume_msat": contender_volume_msat,
            "fallback_volume_msat": fallback_volume_msat,
            "raw_extra_volume_msat": raw_extra_volume_msat,
            "native_rebalance_volume_msat": native_rebalance_volume_msat,
            "native_rebalance_sent_msat": native_rebalance_sent_msat,
            "native_rebalance_completed_count": native_rebalance_completed_count,
            "attributed_native_rebalance_volume_msat": (
                attributed_native_rebalance_volume_msat
            ),
            "extra_volume_msat": extra_volume_msat,
            "contender_forward_count": contender_forward_count,
            "multipart_forward_splits": max(
                0,
                contender_forward_count
                - settled_count
                - native_rebalance_completed_count,
            ),
            "attribution_method": "exact_family_volume_plus_native_rebalance",
        },
        "families": family_rows,
        "contenders": contender_deltas,
        "safety_violations": safety_violations,
    }
    if entry_status == "post_rebalance_ready":
        observation = state.get("last_rebalance_observation")
        retired = state.get("return_paths_retired")
        block["phase"] = "post_rebalance_demand"
        block["post_rebalance"] = {
            "observation_block": observation,
            "return_paths_retired": retired,
            "controllers_held": state.get("post_rebalance_controllers_held"),
            "controlled_start": (
                state.get("controlled_depletion", {}).get("after")
                if isinstance(state.get("controlled_depletion"), dict)
                else None
            ),
        }
    elif isinstance(treatment, dict) and treatment.get("status") == "active":
        block["acquisition_treatment"] = treatment
        block["phase"] = "manual_acquisition"
    elif isinstance(automatic, dict) and automatic.get("status") == "active":
        block["automatic_acquisition"] = automatic
        block["phase"] = (
            "automatic_retention"
            if automatic.get("phase") == "retention"
            else "automatic_acquisition"
        )
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


def return_path_snapshot(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = state.get("return_paths")
    if not isinstance(paths, list) or len(paths) != 2:
        raise RunnerError("runner state lacks exactly two return paths")
    snapshot: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not isinstance(path, dict):
            raise RunnerError("malformed return-path state")
        family = path.get("family")
        source = path.get("source_container")
        if not isinstance(family, str) or not isinstance(source, str):
            raise RunnerError("malformed return-path identity")
        if family == "cln":
            matches = [
                row for row in channel_rows(source)
                if row.get("channel_id") == path.get("channel_id")
            ]
            if len(matches) != 1:
                raise RunnerError("CLN return-path snapshot is not unique")
            row = matches[0]
            snapshot[family] = {
                "active": row.get("state") == "CHANNELD_NORMAL",
                "short_channel_id": row.get("short_channel_id"),
                "local_balance_sats": msat_value(
                    row.get("to_us_msat", row.get("our_amount_msat", 0))
                ) // 1000,
                "capacity_sats": msat_value(
                    row.get("total_msat", row.get("amount_msat", 0))
                ) // 1000,
            }
        elif family == "lnd":
            matches = [
                row for row in _live_lnd_channels(source)
                if row.get("channel_point") == path.get("channel_point")
            ]
            if len(matches) != 1:
                raise RunnerError("LND return-path snapshot is not unique")
            row = matches[0]
            snapshot[family] = {
                "active": row.get("active") is True,
                "short_channel_id": row.get("scid_str"),
                "local_balance_sats": nonnegative_int(
                    row.get("local_balance"), "LND return local balance"
                ),
                "capacity_sats": nonnegative_int(
                    row.get("capacity"), "LND return capacity"
                ),
            }
        else:
            raise RunnerError(f"unknown return-path family: {family!r}")
    if set(snapshot) != {"cln", "lnd"}:
        raise RunnerError("return-path snapshot lacks both client families")
    return snapshot


def resume_controlled_depletion_controllers(
    path: Path, state: dict[str, Any]
) -> dict[str, Any] | None:
    """Release a prepared fixture without forcing either rebalance cycle."""
    fixture = state.get("controlled_depletion")
    if not isinstance(fixture, dict) or fixture.get("controllers_held") is not True:
        return None
    assignment = state["assignment"]
    contenders = state["contenders"]
    revenue_container = contenders[assignment["revenue_ops"]]["container"]
    clboss_container = contenders[assignment["clboss"]]["container"]
    cln_rpc(revenue_container, "revenue-config", "set", "paused", "false")
    controls = (
        cln_rpc(revenue_container, "revenue-status")
        .get("operator_controls", {})
        .get("values", {})
    )
    if controls.get("paused") is not False:
        raise RunnerError("Revenue Ops did not resume for controlled observation")
    cln_rpc(
        clboss_container,
        "setconfig",
        "clboss-xrebalance-per-hour",
        str(CLBOSS_REBALANCES_PER_HOUR),
    )
    cln_rpc(clboss_container, "setconfig", "clboss-rebalance-mode", "xrebalance")
    status = cln_rpc(clboss_container, "clboss-status")
    if status.get("rebalance_mode", {}).get("mode") != "xrebalance":
        raise RunnerError("CLBOSS did not resume for controlled observation")
    fixture["controllers_held"] = False
    fixture["controllers_resumed_at"] = int(time.time())
    activation = {
        "revenue_ops": "native_unpaused",
        "clboss": "native_xrebalance_unbounded",
        "forced_cycles": False,
        "at": fixture["controllers_resumed_at"],
    }
    _checkpoint(path, state, "controlled_depletion_controllers_resumed", activation=activation)
    return activation


def resume_post_demand_controllers(
    path: Path, state: dict[str, Any]
) -> dict[str, Any] | None:
    """Release a scored-demand hold exactly once for the next native epoch."""
    held = state.get("post_rebalance_controllers_held")
    if not isinstance(held, dict) or any(
        held.get(controller) is not True for controller in ("revenue_ops", "clboss")
    ):
        return None
    if held.get("forced_cycles") is not False:
        raise RunnerError("warm observation hold has invalid forced-cycle lineage")
    held_at = nonnegative_int(held.get("at"), "warm observation hold time")
    demand_events = [
        event
        for event in state.get("events", [])
        if isinstance(event, dict) and event.get("event") in {
            "smoke_complete", "equal_targeted_pressure_complete"
        }
    ]
    demand_times = [
        nonnegative_int(event.get("at"), "warm observation demand time")
        for event in demand_events
    ]
    if not demand_times or max(demand_times) < held_at:
        raise RunnerError("warm observation hold lacks a completed demand block")
    latest_demand = max(demand_events, key=lambda event: int(event.get("at") or 0))
    activation_source = (
        "equal_targeted_pressure"
        if latest_demand.get("event") == "equal_targeted_pressure_complete"
        else "post_rebalance_demand"
    )
    assignment = state["assignment"]
    contenders = state["contenders"]
    revenue_container = contenders[assignment["revenue_ops"]]["container"]
    clboss_container = contenders[assignment["clboss"]]["container"]
    cln_rpc(revenue_container, "revenue-config", "set", "paused", "false")
    controls = (
        cln_rpc(revenue_container, "revenue-status")
        .get("operator_controls", {})
        .get("values", {})
    )
    if controls.get("paused") is not False:
        raise RunnerError("Revenue Ops did not resume for warm observation")
    cln_rpc(
        clboss_container,
        "setconfig",
        "clboss-xrebalance-per-hour",
        str(CLBOSS_REBALANCES_PER_HOUR),
    )
    cln_rpc(clboss_container, "setconfig", "clboss-rebalance-mode", "xrebalance")
    status = cln_rpc(clboss_container, "clboss-status")
    if status.get("rebalance_mode", {}).get("mode") != "xrebalance":
        raise RunnerError("CLBOSS did not resume for warm observation")
    resumed_at = int(time.time())
    held.update({
        "revenue_ops": False,
        "clboss": False,
        "resumed_at": resumed_at,
    })
    activation = {
        "revenue_ops": "native_unpaused",
        "clboss": "native_xrebalance_unbounded",
        "forced_cycles": False,
        "source": activation_source,
        "at": resumed_at,
    }
    _checkpoint(
        path,
        state,
        "post_demand_controllers_resumed_for_warm_epoch",
        activation=activation,
    )
    return activation


def observe_rebalances(
    *, replica: int, results_dir: Path, observe_seconds: float
) -> dict[str, Any]:
    """Measure autonomous controllers after synthetic return liquidity appears."""
    if observe_seconds < 0:
        raise RunnerError("observation seconds must be nonnegative")
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") != "return_paths_ready":
        raise RunnerError(f"replica is not ready for rebalance observation: {state.get('status')}")
    assignment = state.get("assignment")
    contenders = state.get("contenders")
    if not isinstance(assignment, dict) or not isinstance(contenders, dict):
        raise RunnerError("runner state lacks controller assignment")
    containers = {
        controller: contenders[identity]["container"]
        for controller, identity in assignment.items()
    }
    before_totals = {
        controller: contender_totals(container)
        for controller, container in containers.items()
    }
    before_rebalances = {
        controller: rebalance_totals(container)
        for controller, container in containers.items()
    }
    return_before = return_path_snapshot(state)
    activation = resume_controlled_depletion_controllers(path, state)
    if activation is None:
        activation = resume_post_demand_controllers(path, state)
    diagnostics_before = {
        "revenue_ops": cln_rpc(
            containers["revenue_ops"], "revenue-rebalance-debug", "summary_only=true"
        ),
        "clboss": cln_rpc(containers["clboss"], "clboss-status"),
    }
    started = time.time()
    monotonic_started = time.monotonic()
    deadline = monotonic_started + observe_seconds
    while time.monotonic() < deadline:
        time.sleep(min(NATIVE_CYCLE_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    after_totals = {
        controller: contender_totals(container)
        for controller, container in containers.items()
    }
    after_rebalances = {
        controller: rebalance_totals(container)
        for controller, container in containers.items()
    }
    return_after = return_path_snapshot(state)
    diagnostics_after = {
        "revenue_ops": cln_rpc(
            containers["revenue_ops"], "revenue-rebalance-debug", "summary_only=true"
        ),
        "clboss": cln_rpc(containers["clboss"], "clboss-status"),
    }
    full_stack = state.get("full_stack")
    if not isinstance(full_stack, dict):
        raise RunnerError("rebalance observation lacks full-stack controls")
    allowance_sats = nonnegative_int(
        full_stack.get("revenue_rebalance_allowance_sats"),
        "Revenue rebalance allowance",
    )
    if allowance_sats == 0:
        raise RunnerError("Revenue rebalance allowance must be positive")
    controller_rows = {}
    safety_violations = []
    for controller in containers:
        movement = rebalance_delta(before_rebalances[controller], after_rebalances[controller])
        if (
            controller == "revenue_ops"
            and movement["cost_msat"] > allowance_sats * 1000
        ):
            safety_violations.append("revenue_ops:native_budget_exceeded")
        controller_rows[controller] = {
            "circular_payments": movement,
            "balance_before": asdict(before_totals[controller]),
            "balance_after": asdict(after_totals[controller]),
            "worst_imbalance_improvement_ppm": (
                before_totals[controller].worst_channel_imbalance_ppm
                - after_totals[controller].worst_channel_imbalance_ppm
            ),
        }
    result = {
        "schema": "polar-clboss-rebalance-observation-v1",
        "replica": f"replica-{replica}",
        # Wall time can step under NTP while a compressed tournament epoch is
        # running. Keep the wall timestamp for block identity, but measure the
        # interval on the monotonic clock used by the scheduler itself.
        "duration_seconds": max(0.0, time.monotonic() - monotonic_started),
        "fixture": state.get("return_path_fixture"),
        "controlled_depletion": state.get("controlled_depletion"),
        "targeted_pressure": state.get("targeted_pressure"),
        "controller_activation": activation,
        "return_paths": {"before": return_before, "after": return_after},
        "controllers": controller_rows,
        "diagnostics": {"before": diagnostics_before, "after": diagnostics_after},
        "clboss_spend_policy": "native_unbounded",
        "safety_violations": safety_violations,
    }
    state["status"] = "rebalance_observed"
    block_id = f"rebalance-{int(started)}"
    state["last_rebalance_observation"] = block_id
    _checkpoint(path, state, "rebalance_observed", block=block_id, result=result)
    write_json_atomic(path.parent / f"{block_id}.json", result)
    return result


def retire_return_paths(
    bridge: PolarMcp,
    *,
    replica: int,
    results_dir: Path,
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    """Remove the direct fixture bypass before post-rebalance scored demand."""
    if timeout_seconds < 0:
        raise RunnerError("return-path retirement timeout must be nonnegative")
    path = state_path(results_dir, replica)
    state = read_state(path)
    if state.get("status") != "rebalance_observed":
        raise RunnerError(
            f"return paths can be retired only after observation: {state.get('status')}"
        )
    observation = state.get("last_rebalance_observation")
    if not isinstance(observation, str) or not observation:
        raise RunnerError("return-path retirement lacks rebalance observation lineage")
    assignment = state.get("assignment")
    contenders = state.get("contenders")
    if not isinstance(assignment, dict) or not isinstance(contenders, dict):
        raise RunnerError("post-rebalance transition lacks controller assignment")
    try:
        revenue_container = contenders[assignment["revenue_ops"]]["container"]
        clboss_container = contenders[assignment["clboss"]]["container"]
    except (KeyError, TypeError) as exc:
        raise RunnerError("post-rebalance transition has malformed contenders") from exc
    # Freeze the completed native observation before removing fixture paths.
    # A delayed circular payment must not overlap scored customer demand,
    # where its forwards would be indistinguishable from routed traffic.
    cln_rpc(revenue_container, "revenue-config", "set", "paused", "true")
    revenue_controls = (
        cln_rpc(revenue_container, "revenue-status")
        .get("operator_controls", {})
        .get("values", {})
    )
    if revenue_controls.get("paused") is not True:
        raise RunnerError("Revenue Ops did not hold after rebalance observation")
    cln_rpc(clboss_container, "setconfig", "clboss-rebalance-mode", "off")
    clboss_mode = (
        cln_rpc(clboss_container, "clboss-status")
        .get("rebalance_mode", {})
        .get("mode")
    )
    if clboss_mode != "off":
        raise RunnerError("CLBOSS did not hold after rebalance observation")
    state["post_rebalance_controllers_held"] = {
        "revenue_ops": True,
        "clboss": True,
        "forced_cycles": False,
        "at": int(time.time()),
    }
    _checkpoint(
        path,
        state,
        "post_rebalance_controllers_held_for_scoring",
        result=state["post_rebalance_controllers_held"],
    )
    if _close_return_paths(state):
        _checkpoint(path, state, "return_path_closes_dispatched_for_scoring")
    if state.get("return_path_closes_dispatched") is not True:
        raise RunnerError("return-path close dispatch was not checkpointed")
    _mine(bridge, int(state["network_id"]), 6)
    deadline = time.monotonic() + timeout_seconds
    remaining = _live_return_paths(state)
    while remaining and time.monotonic() < deadline:
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        remaining = _live_return_paths(state)
    if remaining:
        raise RunnerError(
            f"direct return paths remain live; post-rebalance scoring blocked: {remaining}"
        )
    state["return_paths_retired"] = {
        "confirmed_absent": True,
        "at": int(time.time()),
        "observation_block": observation,
        "purpose": "prevent direct fixture bypass during post-rebalance demand",
    }
    state["status"] = "post_rebalance_ready"
    _checkpoint(
        path,
        state,
        "return_paths_retired_for_scoring",
        result=state["return_paths_retired"],
    )
    return state


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


def _close_return_paths(state: dict[str, Any]) -> bool:
    """Dispatch cooperative closes for checkpointed synthetic lab channels."""
    paths = state.get("return_paths")
    if not isinstance(paths, list) or state.get("return_path_closes_dispatched") is True:
        return False
    for path in paths:
        if not isinstance(path, dict):
            raise RunnerError("malformed return path during cleanup")
        family = path.get("family")
        source = path.get("source_container")
        if not isinstance(source, str):
            raise RunnerError("return path lacks cleanup source")
        if family == "cln":
            channel_id = path.get("short_channel_id") or path.get("channel_id")
            if not isinstance(channel_id, str) or not channel_id:
                raise RunnerError("CLN return path lacks cleanup channel id")
            matches = [
                row for row in channel_rows(source)
                if row.get("channel_id") == path.get("channel_id")
            ]
            if len(matches) != 1:
                raise RunnerError("CLN return path lacks unique cleanup state")
            if matches[0].get("state") not in {
                "CLOSINGD_COMPLETE", "ONCHAIN", "CLOSED",
            }:
                result = _run(
                    [
                        "docker", "exec", "-u", "clightning", source,
                        "lightning-cli", "--network=regtest", "close", channel_id,
                    ],
                    check=False,
                    timeout=60,
                )
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or "").strip()
                    raise RunnerError(f"CLN return-path cooperative close failed: {detail}")
        elif family == "lnd":
            channel_point = path.get("channel_point")
            if not isinstance(channel_point, str) or not channel_point:
                # A funding RPC can be checkpointed before the active channel
                # is resolved. Recover its exact point from the live peer.
                matches = [
                    row for row in _live_lnd_channels(source)
                    if row.get("remote_pubkey") == path.get("peer_id")
                ]
                if len(matches) != 1 or not isinstance(matches[0].get("channel_point"), str):
                    raise RunnerError("LND return path lacks unique cleanup channel point")
                channel_point = matches[0]["channel_point"]
            result = _run(
                [
                    "docker", "exec", "-u", "lnd", source,
                    "lncli", "--network=regtest", "closechannel",
                    "--chan_point", channel_point,
                ],
                check=False,
                timeout=60,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise RunnerError(f"LND return-path cooperative close failed: {detail}")
        else:
            raise RunnerError(f"unknown return-path family during cleanup: {family!r}")
    state["return_path_closes_dispatched"] = True
    return True


def _live_return_paths(state: dict[str, Any]) -> dict[str, list[str]]:
    live: dict[str, list[str]] = {}
    for path in state.get("return_paths", []):
        if not isinstance(path, dict):
            live.setdefault("malformed", []).append("unknown")
            continue
        family = str(path.get("family") or "unknown")
        source = path.get("source_container")
        if not isinstance(source, str):
            live.setdefault(family, []).append("missing-source")
            continue
        if family == "cln":
            matches = [
                row for row in channel_rows(source)
                if row.get("channel_id") == path.get("channel_id")
                and row.get("state") not in {"ONCHAIN", "CLOSED"}
            ]
            live[family] = [
                str(row.get("short_channel_id") or row.get("channel_id") or "unknown")
                for row in matches
            ]
        elif family == "lnd":
            live[family] = [
                str(row.get("channel_point") or "unknown")
                for row in _live_lnd_channels(source)
                if row.get("channel_point") == path.get("channel_point")
            ]
    return {family: rows for family, rows in live.items() if rows}


def cleanup(bridge: PolarMcp, *, replica: int, results_dir: Path) -> dict[str, Any]:
    path = state_path(results_dir, replica)
    state = read_state(path)
    if restore_automatic_treatments(state):
        _checkpoint(path, state, "automatic_acquisition_restored")
    if restore_acquisition_treatment(state):
        _checkpoint(path, state, "acquisition_treatment_restored")
    _stop_plugins(state)
    _checkpoint(path, state, "controllers_stopped_or_paused")
    if _close_return_paths(state):
        _checkpoint(path, state, "return_path_closes_dispatched")
    if restore_background(state):
        _checkpoint(path, state, "background_policies_restored")
    # Confirm any recently dispatched funding transaction before classifying
    # channels.  AWAITING_LOCKIN is still live state and must never be hidden
    # by the active-channel filter used by traffic readiness.
    if any(docker_running(row["container"]) for row in state.get("contenders", {}).values()):
        _mine(bridge, int(state["network_id"]), 6)
        time.sleep(3)
    if state.get("return_path_closes_dispatched") is True:
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
    return_path_deadline = time.monotonic() + 60
    return_paths_remaining = _live_return_paths(state)
    while return_paths_remaining and time.monotonic() < return_path_deadline:
        time.sleep(2)
        return_paths_remaining = _live_return_paths(state)
    if return_paths_remaining:
        state["status"] = "cleanup_incomplete"
        _checkpoint(
            path, state, "cleanup_incomplete",
            return_paths_remaining=return_paths_remaining,
        )
        raise RunnerError(
            f"cleanup found active synthetic return paths: {return_paths_remaining}"
        )
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
            "restore-auto", "accelerate", "smoke", "return-path", "observe-rebalance",
            "retire-return-paths", "deplete", "target-pressure", "prime-paths",
            "status", "cleanup",
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
        choices=("balanced", "forward-pressure", "reverse-pressure"),
        default="balanced",
    )
    parser.add_argument(
        "--traffic-family",
        choices=("both", "cln", "lnd"),
        default="both",
    )
    parser.add_argument("--background-ppm", type=positive_int, default=10_000)
    parser.add_argument(
        "--revenue-rebalance-budget-sats", "--spend-cap-sats",
        dest="spend_cap_sats", type=positive_int, default=1_000,
        help="Revenue Ops-only native rebalance allowance; CLBOSS remains uncapped",
    )
    parser.add_argument(
        "--return-capacity-sats", type=positive_int,
        default=RETURN_PATH_CAPACITY_SATS,
    )
    parser.add_argument(
        "--return-fee-ppm", type=positive_int, default=RETURN_PATH_FEE_PPM,
        help=(
            "synthetic return-path proportional fee; the 120-ppm default is "
            "the realistic corridor fixture, while explicit cheap-route lanes "
            "must keep the same crossed policy readback invariant"
        ),
    )
    parser.add_argument(
        "--depletion-sats", type=positive_int, default=CONTROLLED_DEPLETION_SATS,
    )
    parser.add_argument(
        "--depletion-family", choices=("cln", "lnd"), default="cln",
        help="client-family lanes to place in the equal controlled liquidity state",
    )
    parser.add_argument(
        "--depletion-side", choices=("payer", "sink"), default="sink",
        help=(
            "contender lane role to leave below 30%% local; payer-side depletion "
            "also adds equal reverse earning traffic before controller release"
        ),
    )
    parser.add_argument(
        "--fixture-fee-ppm", type=nonnegative_arg,
        help=(
            "optional equal outgoing CLN fee on both controlled destinations; "
            "use only to test a specific evidence band"
        ),
    )
    parser.add_argument("--observe-seconds", type=float, default=360.0)
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
            "restore-acquire", "auto-acquire", "retain", "restore-auto", "accelerate",
            "smoke", "return-path", "observe-rebalance", "retire-return-paths",
            "deplete", "target-pressure", "prime-paths", "cleanup"
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
        elif args.command == "prime-paths":
            result = prime_forced_paths(
                bridge,
                replica=args.replica,
                results_dir=args.results_dir,
                amount_sats=args.amount_sats,
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
        elif args.command == "accelerate":
            result = accelerate_controllers(
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
                traffic_family=args.traffic_family,
            )
        elif args.command == "return-path":
            result = provision_return_paths(
                bridge,
                replica=args.replica,
                results_dir=args.results_dir,
                capacity_sats=args.return_capacity_sats,
                fee_ppm=args.return_fee_ppm,
            )
        elif args.command == "deplete":
            result = prepare_controlled_depletion(
                bridge,
                replica=args.replica,
                results_dir=args.results_dir,
                amount_sats=args.depletion_sats,
                fixture_fee_ppm=args.fixture_fee_ppm,
                family=args.depletion_family,
                depleted_side=args.depletion_side,
            )
        elif args.command == "target-pressure":
            result = apply_equal_targeted_pressure(
                bridge,
                replica=args.replica,
                results_dir=args.results_dir,
            )
        elif args.command == "observe-rebalance":
            result = observe_rebalances(
                replica=args.replica,
                results_dir=args.results_dir,
                observe_seconds=args.observe_seconds,
            )
        elif args.command == "retire-return-paths":
            result = retire_return_paths(
                bridge,
                replica=args.replica,
                results_dir=args.results_dir,
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
