#!/usr/bin/env python3
"""Run the production-shaped Grand Prix directly through Docker.

The manifest is the source of truth. Docker owns every ordinary node and all
background channels; the two exact-image contender nodes are attached in a
separate, checkpointed phase. Mutations require ``--apply`` and are designed
to be safely resumed after a process or RPC interruption.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, IO, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grand_prix_manifest import validate_topology  # noqa: E402
from docker_grand_prix_lab import DockerGrandPrixLab, DockerLabError  # noqa: E402
from equivalent_competitor_controller import (  # noqa: E402
    EquivalentControllerError,
    load_models,
    policy_intents,
    validate_models,
)


# Retain the v1 identifier so completed Polar-era replicas remain scoreable.
SCHEMA = "polar-grand-prix-runner-state-v1"
DEFAULT_NAME = "revenue-ops-grand-prix"
EXPECTED_REVENUE_REVISION = "2987608d525075ec974ace6e17ee93986c1d0ba5"
DEFAULT_IMAGE = f"cl-revenue-ops-grand-prix-base:{EXPECTED_REVENUE_REVISION[:7]}"
EXPECTED_CLN_VERSION = "v26.06.7"
EXPECTED_CLN_DIGEST = "sha256:53ddf124fe7058b6a2fc059d104976cc54ba5be21dc55b295cd82d01cabeb39c"
CONTENDER_WALLET_SATS = 140_000_000
PAYER_WALLET_SATS = 50_000_000
REVENUE_PLUGIN = "/opt/cl_revenue_ops/cl-revenue-ops-lab-wrapper"
LEGACY_REVENUE_PLUGIN = "/opt/cl_revenue_ops/cl-revenue-ops-polar-wrapper"
CLBOSS_PLUGIN = "/usr/local/libexec/clboss"
XREBALANCE_PLUGIN = "/usr/local/libexec/xrebalance"
CONTROLLER_CYCLE_SECONDS = 15
CONTROLLER_WARMUP_SECONDS = 75
EQUIVALENT_CONTROLLER_CONFIG = (
    Path(__file__).resolve().parent / "grand-prix" / "equivalent-controllers.v1.json"
)
COMPETITOR_CONTROLLERS = frozenset({"clboss", "ln_operator", "torq"})
MUTATING_COMMANDS = {
    "create-base", "start-base", "wire-background", "launch-contenders",
    "wire-contenders", "shape-liquidity", "seed-fees", "start-controllers",
    "top-up-payers", "run-public", "advance-timeout", "reconcile-public",
    "stop-lab",
}
CONTAINER_RE = re.compile(
    r"^revenue-gp-n[1-9][0-9]*-grand-prix-r[1-9][0-9]*-identity-[ab]$"
)
ACTIVE_BACKEND = "docker"
LAB_ERRORS = (DockerLabError,)


class RunnerError(RuntimeError):
    """A Grand Prix runtime invariant was not satisfied."""


class LabBackend(Protocol):
    """Minimal orchestration surface consumed by the Grand Prix runner."""

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def _base_container_name(network_id: int, node_name: str) -> str:
    return f"revenue-gp-n{network_id}-{node_name}"


def _docker_network_name(network_id: int) -> str:
    return f"revenue-gp-n{network_id}"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"expected a JSON object in {path}")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _acquire_mutation_lock(state_path: Path) -> IO[str]:
    """Prevent overlapping mutation processes for one durable runner state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RunnerError(
            f"another mutating runner process holds {lock_path}; do not overlap retries"
        ) from exc
    return handle


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


def _revenue_plugin_path(container: str) -> str:
    """Resolve the generic wrapper while preserving frozen-image replayability."""
    if not CONTAINER_RE.fullmatch(container):
        raise RunnerError(f"refusing unsafe contender container name {container!r}")
    for path in (REVENUE_PLUGIN, LEGACY_REVENUE_PLUGIN):
        result = _run(
            ["docker", "exec", container, "test", "-x", path],
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            return path
    raise RunnerError(
        "Revenue Ops contender contains neither the current nor legacy executable wrapper"
    )


def _json_command(command: Sequence[str], *, timeout: float = 120) -> dict[str, Any]:
    completed = _run(command, timeout=timeout)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"command returned non-JSON: {command!r}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"command returned a non-object: {command!r}")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RunnerError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunnerError(f"{label} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise RunnerError(f"{label} must be a positive integer")
    return parsed


def contender_container(network_id: int, replica: int, identity: str) -> str:
    network_id = _positive(network_id, "network id")
    replica = _positive(replica, "replica")
    if identity not in {"identity-a", "identity-b"}:
        raise RunnerError(f"unknown contender identity: {identity!r}")
    name = f"revenue-gp-n{network_id}-grand-prix-r{replica}-{identity}"
    if not CONTAINER_RE.fullmatch(name):
        raise RunnerError(f"unsafe contender container name: {name!r}")
    return name


def assignment_for(replica: int) -> dict[str, str]:
    replica = _positive(replica, "replica")
    if replica % 2:
        return {"revenue_ops": "identity-a", "clboss": "identity-b"}
    return {"revenue_ops": "identity-b", "clboss": "identity-a"}


def _docker_exists(name: str) -> bool:
    completed = _run(["docker", "inspect", name], check=False)
    if completed.returncode == 0:
        return True
    detail = (completed.stderr or completed.stdout or "").casefold()
    if "no such object" in detail or "no such container" in detail:
        return False
    raise RunnerError(f"cannot inspect Docker container {name!r}: {detail.strip()}")


def _cln_rpc(container: str, *arguments: Any, base_managed: bool = False) -> dict[str, Any]:
    if not base_managed and not CONTAINER_RE.fullmatch(container):
        raise RunnerError(f"refusing unsafe contender RPC target {container!r}")
    user = ["-u", "clightning"] if base_managed else []
    return _json_command(
        [
            "docker", "exec", *user, container, "lightning-cli",
            "--network=regtest", "--notifications=none",
        ]
        + [str(value) for value in arguments]
    )


def _lnd_rpc(
    container: str, *arguments: Any, timeout: float = 120
) -> dict[str, Any]:
    if not re.fullmatch(r"revenue-gp-n[1-9][0-9]*-lnd-payer", container):
        raise RunnerError(f"refusing unsafe LND RPC target {container!r}")
    return _json_command(
        ["docker", "exec", "-u", "lnd", container, "lncli", "--network=regtest"]
        + [str(value) for value in arguments],
        timeout=timeout,
    )


def _wait_lnd_read_rpc(
    container: str, command: str, *, deadline_seconds: float = 120
) -> dict[str, Any]:
    """Retry a bounded, read-only LND readiness probe."""
    if command not in {"getinfo", "walletbalance"}:
        raise RunnerError(f"refusing non-read-only LND readiness RPC {command!r}")
    deadline = time.monotonic() + deadline_seconds
    last = "not probed"
    while time.monotonic() < deadline:
        try:
            return _lnd_rpc(container, command, timeout=10)
        except RunnerError as exc:
            last = str(exc)
            time.sleep(1)
    raise RunnerError(f"LND {command} did not become ready: {last}")


def _image_attestation(image: str) -> dict[str, Any]:
    completed = _run(["docker", "image", "inspect", image])
    try:
        inspected = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"Docker returned malformed image metadata for {image}") from exc
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise RunnerError(f"Docker returned ambiguous image metadata for {image}")
    labels = ((inspected[0].get("Config") or {}).get("Labels") or {})
    expected = {
        "org.opencontainers.image.revision.revenue_ops": EXPECTED_REVENUE_REVISION,
        "org.opencontainers.image.version.cln": EXPECTED_CLN_VERSION,
        "org.opencontainers.image.digest.cln": EXPECTED_CLN_DIGEST,
    }
    mismatches = {key: {"expected": value, "actual": labels.get(key)}
                  for key, value in expected.items() if labels.get(key) != value}
    if image == DEFAULT_IMAGE and mismatches:
        raise RunnerError(f"default contender image attestation failed: {mismatches}")
    if not labels.get("org.opencontainers.image.revision.revenue_ops"):
        raise RunnerError("contender image has no Revenue Ops revision label")
    experiment_digest = labels.get(
        "org.opencontainers.image.experiment.patch_digest"
    )
    if (
        image != DEFAULT_IMAGE
        and labels.get("org.opencontainers.image.revision.revenue_ops")
        == EXPECTED_REVENUE_REVISION
        and not re.fullmatch(r"sha256:[0-9a-f]{64}", str(experiment_digest or ""))
    ):
        raise RunnerError(
            "custom image on the baseline revision has no valid experiment patch digest"
        )
    return {"image": image, "image_id": inspected[0].get("Id"), "labels": labels}


def base_nodes(topology: dict[str, Any]) -> list[dict[str, str]]:
    """Return the 22 Docker-managed nodes in deterministic manifest order."""
    validate_topology(topology)
    rows = [
        {"name": str(row["name"]), "implementation": str(row["implementation"])}
        for row in topology["nodes"]
        if row.get("role") != "contender"
    ]
    if len(rows) != 22:
        raise RunnerError("topology must leave exactly two external contenders")
    counts = {implementation: sum(row["implementation"] == implementation for row in rows)
              for implementation in ("cln", "lnd")}
    if counts != {"cln": 15, "lnd": 7}:
        raise RunnerError(f"unexpected base implementation mix: {counts}")
    return rows


def background_channels(topology: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only channels that do not touch either external contender."""
    validate_topology(topology)
    contender_names = {"identity-a", "identity-b"}
    rows = [
        row for row in topology["channels"]
        if not ({str(row["source"]), str(row["destination"])} & contender_names)
    ]
    if len(rows) != 25 or any(row.get("contender_lane") is not None for row in rows):
        raise RunnerError("topology must contain exactly 25 ordinary background channels")
    return rows


def runtime_plan(topology: dict[str, Any], name: str = DEFAULT_NAME) -> dict[str, Any]:
    nodes = base_nodes(topology)
    channels = background_channels(topology)
    return {
        "schema": SCHEMA,
        "backend": ACTIVE_BACKEND,
        "network_name": name,
        "topology_digest": _digest(topology),
        "base_nodes": len(nodes),
        "docker_nodes": len(nodes),
        "external_contenders": 2,
        "implementation_counts": {
            "c-lightning": sum(row["implementation"] == "cln" for row in nodes),
            "LND": sum(row["implementation"] == "lnd" for row in nodes),
        },
        "background_channels": len(channels),
        "contender_channels_deferred": len(topology["channels"]) - len(channels),
        "mutations_required": False,
    }


def _networks(bridge: LabBackend) -> list[dict[str, Any]]:
    rows = bridge.call("list_networks", {}).get("networks")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RunnerError("lab backend returned a malformed network list")
    return rows


def _network_by_name(bridge: LabBackend, name: str) -> dict[str, Any] | None:
    matches = [row for row in _networks(bridge) if row.get("name") == name]
    if len(matches) > 1:
        raise RunnerError(f'lab backend has multiple networks named "{name}"')
    return matches[0] if matches else None


def _generated_renames(network: dict[str, Any], expected: list[dict[str, str]]) -> list[tuple[str, str]]:
    lightning = (network.get("nodes") or {}).get("lightning")
    if not isinstance(lightning, list):
        raise RunnerError("created Docker lab has no lightning node list")
    implementation_map = {"cln": "c-lightning", "lnd": "LND"}
    renames: list[tuple[str, str]] = []
    for manifest_implementation, backend_implementation in implementation_map.items():
        actual = [str(row.get("name") or "") for row in lightning
                  if isinstance(row, dict) and row.get("implementation") == backend_implementation]
        wanted = [row["name"] for row in expected
                  if row["implementation"] == manifest_implementation]
        if len(actual) != len(wanted) or any(not value for value in actual):
            raise RunnerError(
                f"Docker created {len(actual)} {backend_implementation} nodes; expected {len(wanted)}"
            )
        renames.extend(zip(actual, wanted, strict=True))
    return renames


def create_base(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    name: str,
    state_path: Path,
) -> dict[str, Any]:
    """Create and rename a stopped base network; refuse ambiguous recovery."""
    expected = base_nodes(topology)
    if state_path.exists():
        raise RunnerError(f"fresh create refused because state exists: {state_path}")
    existing = _network_by_name(bridge, name)
    if existing is not None:
        raise RunnerError(
            f'network "{name}" already exists without local state; inspect it before recovery'
        )
    running = [row for row in _networks(bridge)
               if str(row.get("status", "")).casefold() in {"started", "starting"}]
    if running:
        raise RunnerError("another Docker Grand Prix lab is running; stop it first")
    result = bridge.call(
        "create_network",
        {
            "name": name,
            "description": "24-node production-shaped cl_revenue_ops Grand Prix base",
            "nodes": [
                {"implementation": "bitcoind", "count": 1},
                {"implementation": "c-lightning", "count": 15},
                {"implementation": "LND", "count": 7},
            ],
        },
    )
    network = result.get("network")
    if not isinstance(network, dict) or not isinstance(network.get("id"), int):
        raise RunnerError(f"create_network returned no network id: {result!r}")
    network_id = int(network["id"])
    renamed: list[dict[str, str]] = []
    for old_name, new_name in _generated_renames(network, expected):
        if old_name != new_name:
            bridge.call(
                "rename_node",
                {"networkId": network_id, "oldName": old_name, "newName": new_name},
            )
        renamed.append({"old": old_name, "new": new_name})
    state = {
        "schema": SCHEMA,
        "backend": ACTIVE_BACKEND,
        "network_id": network_id,
        "network_name": name,
        "topology_digest": _digest(topology),
        "status": "base_created",
        "renamed_nodes": renamed,
        "background_channels": [],
        "events": [{"event": "base_created", "at": int(time.time())}],
    }
    _write_json_atomic(state_path, state)
    return state


def _read_state(path: Path, topology: dict[str, Any]) -> dict[str, Any]:
    state = _load_json(path)
    if state.get("schema") != SCHEMA:
        raise RunnerError("unsupported runner state schema")
    if state.get("topology_digest") != _digest(topology):
        raise RunnerError("runner state does not match the topology manifest")
    if not isinstance(state.get("network_id"), int):
        raise RunnerError("runner state has no network id")
    recorded_backend = state.get("backend", ACTIVE_BACKEND)
    if recorded_backend != ACTIVE_BACKEND:
        raise RunnerError(
            f"runner state backend is {recorded_backend!r}, not {ACTIVE_BACKEND!r}"
        )
    return state


def _checkpoint(path: Path, state: dict[str, Any], event: str, **details: Any) -> None:
    state.setdefault("events", []).append({"event": event, "at": int(time.time()), **details})
    _write_json_atomic(path, state)


def _is_timeout_error(exc: BaseException) -> bool:
    rendered = str(exc).casefold()
    return "timed out" in rendered or "timeout" in rendered


def _list_channels_with_timeout_retries(
    bridge: LabBackend,
    *,
    network_id: int,
    node_name: str,
    attempts: int = 10,
) -> Any:
    """Retry only the read-only channel probe after transient RPC timeouts."""
    for attempt in range(attempts):
        try:
            return bridge.call(
                "list_channels",
                {"networkId": network_id, "nodeName": node_name},
            ).get("channels")
        except LAB_ERRORS as exc:
            if not _is_timeout_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


def start_base(bridge: LabBackend, topology: dict[str, Any], *, state_path: Path) -> dict[str, Any]:
    state = _read_state(state_path, topology)
    network_id = state["network_id"]
    network = _network_by_name(bridge, str(state["network_name"]))
    if network is None or network.get("id") != network_id:
        raise RunnerError("recorded Docker network is missing or has a different id")
    if str(network.get("status", "")).casefold() != "started":
        try:
            bridge.call("start_network", {"networkId": network_id})
        except LAB_ERRORS as exc:
            if "timed out" not in str(exc).casefold():
                raise
    expected = [row["name"] for row in base_nodes(topology)]
    deadline = time.monotonic() + 300
    last_error = "not probed"
    while time.monotonic() < deadline:
        try:
            for node_name in expected:
                bridge.call("get_node_info", {"networkId": network_id, "nodeName": node_name})
            state["status"] = "base_started"
            _checkpoint(state_path, state, "base_started")
            return state
        except LAB_ERRORS as exc:
            last_error = str(exc)
            time.sleep(3)
    raise RunnerError(f"base nodes did not become RPC-ready: {last_error}")


def _matching_channels(
    channels: Any, destination_pubkey: str, capacity_sats: int
) -> list[dict[str, Any]]:
    if not isinstance(channels, list):
        return []
    return [
        row for row in channels
        if (
        isinstance(row, dict)
        and row.get("pubkey") == destination_pubkey
        and str(row.get("capacity")) == str(capacity_sats)
        and row.get("status") in {"Open", "Opening"}
        )
    ]


def _has_exactly_one_channel(
    channels: Any, destination_pubkey: str, capacity_sats: int, *, edge: str
) -> bool:
    matches = _matching_channels(channels, destination_pubkey, capacity_sats)
    if len(matches) > 1:
        raise RunnerError(
            f"topology mismatch: {edge} has {len(matches)} channels with capacity "
            f"{capacity_sats}; expected at most one"
        )
    return len(matches) == 1


def _native_base_pubkey(
    network_id: int, node_name: str, topology: dict[str, Any]
) -> str:
    implementations = {
        str(row["name"]): str(row["implementation"])
        for row in base_nodes(topology)
    }
    implementation = implementations.get(node_name)
    if implementation == "cln":
        payload = _cln_rpc(
            _base_container_name(network_id, node_name), "getinfo", base_managed=True
        )
        value = payload.get("id")
    elif implementation == "lnd":
        container = _base_container_name(network_id, node_name)
        payload = _wait_lnd_read_rpc(container, "getinfo")
        value = payload.get("identity_pubkey")
    else:
        raise RunnerError(f"unknown base node implementation for {node_name!r}")
    if not isinstance(value, str) or not value:
        raise RunnerError(f"native getinfo returned no pubkey for {node_name}")
    return value


def _native_base_channels(
    network_id: int, node_name: str, topology: dict[str, Any]
) -> list[dict[str, Any]]:
    implementations = {
        str(row["name"]): str(row["implementation"])
        for row in base_nodes(topology)
    }
    implementation = implementations.get(node_name)
    if implementation == "cln":
        payload = _cln_rpc(
            _base_container_name(network_id, node_name),
            "listpeerchannels",
            base_managed=True,
        )
        rows = payload.get("channels")
        if not isinstance(rows, list):
            raise RunnerError(f"native CLN channel list is malformed for {node_name}")
        return [
            {
                "pubkey": row.get("peer_id"),
                "capacity": str(_msat(row.get("total_msat")) // 1000),
                "status": (
                    "Open" if row.get("state") == "CHANNELD_NORMAL" else "Opening"
                ),
            }
            for row in rows
            if isinstance(row, dict) and row.get("total_msat") is not None
        ]
    if implementation == "lnd":
        payload = _lnd_node_rpc(network_id, node_name, topology, "listchannels")
        rows = payload.get("channels")
        if not isinstance(rows, list):
            raise RunnerError(f"native LND channel list is malformed for {node_name}")
        return [
            {
                "pubkey": row.get("remote_pubkey"),
                "capacity": str(row.get("capacity")),
                "status": "Open" if row.get("active") is True else "Opening",
            }
            for row in rows
            if isinstance(row, dict) and row.get("capacity") is not None
        ]
    raise RunnerError(f"unknown base node implementation for {node_name!r}")


def _native_mine_blocks(network_id: int, blocks: int = 6) -> None:
    network_id = _positive(network_id, "network id")
    blocks = _positive(blocks, "block count")
    command = [
        "docker", "exec", _base_container_name(network_id, "backend1"), "bitcoin-cli",
        "-regtest", "-rpcuser=labuser", "-rpcpassword=labpass",
    ]
    address = _run(command + ["getnewaddress"]).stdout.strip()
    if not address.startswith("bcrt1"):
        raise RunnerError("Docker bitcoind returned no regtest mining address")
    _run(command + ["generatetoaddress", str(blocks), address])


def wire_background(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
    native_io: bool = False,
) -> dict[str, Any]:
    """Idempotently open the 25 non-contender edges with per-edge checkpoints."""
    state = _read_state(state_path, topology)
    if state.get("status") not in {"base_started", "background_wiring", "background_ready"}:
        raise RunnerError("base network must be started before background wiring")
    network_id = state["network_id"]
    state["status"] = "background_wiring"
    _checkpoint(state_path, state, "background_wiring_started")
    completed = {str(row.get("edge")) for row in state.get("background_channels", [])
                 if isinstance(row, dict)}
    for row in background_channels(topology):
        source = str(row["source"])
        destination = str(row["destination"])
        capacity = int(row["capacity_sats"])
        edge = "--".join(sorted((source, destination)))
        if edge in completed:
            continue
        # Start with liquidity on the endpoint closest to the requested ratio;
        # exact ratios are established later by reconciled shaping payments.
        if float(row["initial_source_ratio"]) >= 0.5:
            funder, peer = source, destination
        else:
            funder, peer = destination, source
        if native_io:
            peer_id = _native_base_pubkey(network_id, peer, topology)
        else:
            info = bridge.call(
                "get_node_info", {"networkId": network_id, "nodeName": peer}
            ).get("info")
            peer_id = info.get("pubkey") if isinstance(info, dict) else None
        if not isinstance(peer_id, str) or not peer_id:
            raise RunnerError(f"could not resolve pubkey for {peer}")
        channels = (
            _native_base_channels(network_id, funder, topology)
            if native_io
            else _list_channels_with_timeout_retries(
                bridge,
                network_id=network_id,
                node_name=funder,
            )
        )
        reconciled = _has_exactly_one_channel(
            channels, peer_id, capacity, edge=edge
        )
        if not reconciled:
            try:
                bridge.call(
                    "open_channel",
                    {
                        "networkId": network_id,
                        "fromNode": funder,
                        "toNode": peer,
                        "sats": capacity,
                        "isPrivate": False,
                        "autoFund": True,
                    },
                )
            except LAB_ERRORS as exc:
                if native_io:
                    # The backend may time out while the
                    # underlying mixed-client funding workflow continues for
                    # another minute. Poll the native node state long enough
                    # to reconcile that single in-flight mutation safely.
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        after = _native_base_channels(
                            network_id, funder, topology
                        )
                        if _has_exactly_one_channel(
                            after, peer_id, capacity, edge=edge
                        ):
                            reconciled = True
                            break
                        time.sleep(1)
                    if not reconciled:
                        raise exc
                else:
                    time.sleep(2)
                    after = _list_channels_with_timeout_retries(
                        bridge,
                        network_id=network_id,
                        node_name=funder,
                    )
                    if not _has_exactly_one_channel(
                        after, peer_id, capacity, edge=edge
                    ):
                        raise exc
                    reconciled = True
        record = {
            "edge": edge,
            "funder": funder,
            "peer": peer,
            "capacity_sats": capacity,
            "reconciled_after_error": reconciled,
        }
        state.setdefault("background_channels", []).append(record)
        completed.add(edge)
        _checkpoint(state_path, state, "background_channel_opened", channel=record)
        try:
            if native_io:
                _native_mine_blocks(network_id)
            else:
                bridge.call("mine_blocks", {"networkId": network_id, "blocks": 6})
        except LAB_ERRORS as exc:
            # A backend timeout can arrive after the mining mutation was
            # dispatched.  The channel itself is already reconciled and
            # checkpointed, so replaying the edge is neither necessary nor
            # desirable.  Later successful mining calls confirm any earlier
            # opening; active-channel checks still fail closed downstream.
            if not _is_timeout_error(exc):
                raise
            _checkpoint(
                state_path,
                state,
                "background_mine_timeout_uncertain",
                edge=edge,
            )
    state["status"] = "background_ready"
    _checkpoint(state_path, state, "background_ready", count=len(completed))
    return state


def _mine(bridge: LabBackend, network_id: int, blocks: int = 6) -> None:
    # Native regtest mining avoids orchestration timeouts after the underlying
    # mutation has already succeeded. Keep ``bridge`` in the signature so the
    # checkpointed caller API remains stable.
    if ACTIVE_BACKEND == "docker":
        bridge.call("mine_blocks", {"networkId": network_id, "blocks": blocks})
    else:
        _native_mine_blocks(network_id, blocks)


def _regtest_address(payload: dict[str, Any]) -> str:
    for key in ("p2tr", "bech32", "address"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith("bcrt1"):
            return value
    raise RunnerError("wallet returned no regtest segwit address")


def _send_onchain(
    bridge: LabBackend, network_id: int, address: str, amount_sats: int
) -> None:
    if not address.startswith("bcrt1"):
        raise RunnerError("refusing non-regtest funding address")
    amount_sats = _positive(amount_sats, "funding amount")
    _run([
        "docker", "exec", _base_container_name(network_id, "backend1"),
        "bitcoin-cli", "-regtest", "-rpcuser=labuser", "-rpcpassword=labpass",
        "sendtoaddress", address, f"{amount_sats / 100_000_000:.8f}",
    ])
    _mine(bridge, network_id)


def _wait_cln(container: str, *, base_managed: bool = False) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    last = "not ready"
    while time.monotonic() < deadline:
        try:
            info = _cln_rpc(container, "getinfo", base_managed=base_managed)
            if info.get("id"):
                return info
            last = "getinfo returned no id"
        except RunnerError as exc:
            last = str(exc)
        time.sleep(1)
    raise RunnerError(f"{container} did not become RPC-ready: {last}")


def _msat(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("msat")
    if isinstance(value, bool) or value is None:
        raise RunnerError(f"invalid msat value: {value!r}")
    rendered = str(value).strip().removesuffix("msat")
    try:
        parsed = int(rendered)
    except ValueError as exc:
        raise RunnerError(f"invalid msat value: {value!r}") from exc
    if parsed < 0:
        raise RunnerError(f"invalid msat value: {value!r}")
    return parsed


def _wait_cln_wallet(
    container: str, minimum_sats: int, *, base_managed: bool = False
) -> None:
    deadline = time.monotonic() + 120
    observed = 0
    while time.monotonic() < deadline:
        rows = _cln_rpc(container, "listfunds", base_managed=base_managed).get("outputs")
        if isinstance(rows, list):
            observed = sum(
                _msat(row.get("amount_msat")) // 1000
                for row in rows
                if isinstance(row, dict)
                and row.get("status") == "confirmed"
                and row.get("reserved") is not True
            )
            if observed >= minimum_sats:
                return
        time.sleep(1)
    raise RunnerError(f"{container} wallet has {observed} confirmed sats; need {minimum_sats}")


def _cln_wallet_output_totals(container: str) -> tuple[int, int]:
    rows = _cln_rpc(container, "listfunds").get("outputs")
    if not isinstance(rows, list):
        raise RunnerError(f"{container} returned a malformed wallet output list")
    confirmed = 0
    pending = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("reserved") is True:
            continue
        amount = _msat(row.get("amount_msat")) // 1000
        if row.get("status") == "confirmed":
            confirmed += amount
        elif row.get("status") in {"unconfirmed", "immature"}:
            pending += amount
    return confirmed, pending


def launch_contenders(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
    replica: int,
    image: str,
) -> dict[str, Any]:
    """Launch and fund two fresh, exactly attested CLN contender containers."""
    state = _read_state(state_path, topology)
    if state.get("status") not in {"background_ready", "contenders_launching"}:
        raise RunnerError("background graph must be ready before contenders launch")
    network_id = int(state["network_id"])
    attestation = _image_attestation(image)
    replica = _positive(replica, "replica")
    prior_contenders = state.get("contenders", {})
    if not isinstance(prior_contenders, dict):
        raise RunnerError("checkpointed contenders are malformed")
    contenders: dict[str, dict[str, Any]] = dict(prior_contenders)
    state.update({
        "status": "contenders_launching",
        "replica": replica,
        "assignment": assignment_for(replica),
        "image_attestation": attestation,
        "contenders": contenders,
    })
    _checkpoint(state_path, state, "contender_launch_resumed")
    data_root = state_path.parent / f"replica-{replica}"
    for identity in ("identity-a", "identity-b"):
        container = contender_container(network_id, replica, identity)
        checkpoint = contenders.get(identity)
        if isinstance(checkpoint, dict):
            if checkpoint.get("container") != container or not _docker_exists(container):
                raise RunnerError(
                    f"checkpointed contender {identity} does not match its container"
                )
            info = _wait_cln(container)
            if (
                checkpoint.get("node_id") != info.get("id")
                or checkpoint.get("version") != info.get("version")
            ):
                raise RunnerError(f"checkpointed contender {identity} identity drifted")
            continue
        if _docker_exists(container):
            raise RunnerError(f"fresh-only launch refused because {container} exists")
        data_dir = data_root / identity / "lightning"
        data_dir.mkdir(parents=True, exist_ok=False)
        (data_dir / "regtest").mkdir()
        _run([
            "docker", "run", "--detach", "--name", container,
            "--network", _docker_network_name(network_id),
            "--network-alias", identity,
            "--volume", f"{data_dir.resolve()}:/root/.lightning",
            "--env", "LIGHTNINGD_NETWORK=regtest",
            image,
            f"--alias={identity}", f"--addr={identity}:9735",
            "--bitcoin-rpcuser=labuser", "--bitcoin-rpcpassword=labpass",
            f"--bitcoin-rpcconnect={_base_container_name(network_id, 'backend1')}",
            "--bitcoin-rpcport=18443", "--log-level=debug",
            "--dev-bitcoind-poll=2", "--dev-fast-gossip", "--developer",
        ])
        info = _wait_cln(container)
        contenders[identity] = {
            "container": container,
            "node_id": str(info["id"]),
            "version": str(info.get("version")),
        }
        state["contenders"] = contenders
        _checkpoint(state_path, state, "contender_started", identity=identity, container=container)

    for identity in ("identity-a", "identity-b"):
        container = contenders[identity]["container"]
        confirmed, pending = _cln_wallet_output_totals(container)
        if confirmed < CONTENDER_WALLET_SATS:
            missing = CONTENDER_WALLET_SATS - confirmed - pending
            if missing > 0:
                address = _regtest_address(_cln_rpc(container, "newaddr"))
                _send_onchain(bridge, network_id, address, missing)
            else:
                _mine(bridge, network_id)
        _wait_cln_wallet(container, CONTENDER_WALLET_SATS)
        already_recorded = any(
            row.get("event") == "contender_funded" and row.get("identity") == identity
            for row in state.get("events", []) if isinstance(row, dict)
        )
        if not already_recorded:
            _checkpoint(state_path, state, "contender_funded", identity=identity)

    cln_payer = _base_container_name(network_id, "cln-payer")
    cln_address = _regtest_address(
        _cln_rpc(cln_payer, "newaddr", base_managed=True)
    )
    _send_onchain(bridge, network_id, cln_address, PAYER_WALLET_SATS)
    _wait_cln_wallet(cln_payer, PAYER_WALLET_SATS, base_managed=True)
    lnd_payer = _base_container_name(network_id, "lnd-payer")
    lnd_address = _regtest_address(_lnd_rpc(lnd_payer, "newaddress", "p2wkh"))
    _send_onchain(bridge, network_id, lnd_address, PAYER_WALLET_SATS)
    deadline = time.monotonic() + 120
    lnd_confirmed = 0
    while time.monotonic() < deadline:
        balance = _wait_lnd_read_rpc(lnd_payer, "walletbalance")
        try:
            lnd_confirmed = int(balance.get("confirmed_balance", 0))
        except (TypeError, ValueError):
            lnd_confirmed = 0
        if lnd_confirmed >= PAYER_WALLET_SATS:
            break
        time.sleep(1)
    if lnd_confirmed < PAYER_WALLET_SATS:
        raise RunnerError(f"LND payer has only {lnd_confirmed} confirmed sats")
    _checkpoint(state_path, state, "payer_wallets_funded")
    state.update({"status": "contenders_ready", "contender_channels": []})
    _checkpoint(state_path, state, "contenders_ready", assignment=state["assignment"])
    return state


def contender_channels(topology: dict[str, Any]) -> list[dict[str, Any]]:
    validate_topology(topology)
    rows = [row for row in topology["channels"] if row.get("contender_lane") is not None]
    if len(rows) != 32:
        raise RunnerError("topology must contain exactly 32 contender channels")
    return rows


def top_up_payers(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
    amount_sats: int,
) -> dict[str, Any]:
    """Add checkpointed non-scored on-chain reserve to both traffic payers."""
    state = _read_state(state_path, topology)
    if state.get("status") not in {"contenders_ready", "contenders_wiring"}:
        raise RunnerError("payer top-up is allowed only during contender wiring")
    amount_sats = _positive(amount_sats, "payer top-up")
    if amount_sats > 100_000_000:
        raise RunnerError("payer top-up must not exceed 100000000 sats")
    network_id = int(state["network_id"])

    cln_payer = _base_container_name(network_id, "cln-payer")
    cln_address = _regtest_address(
        _cln_rpc(cln_payer, "newaddr", base_managed=True)
    )
    _send_onchain(bridge, network_id, cln_address, amount_sats)
    _wait_cln_wallet(cln_payer, amount_sats, base_managed=True)

    lnd_payer = _base_container_name(network_id, "lnd-payer")
    lnd_address = _regtest_address(_lnd_rpc(lnd_payer, "newaddress", "p2wkh"))
    _send_onchain(bridge, network_id, lnd_address, amount_sats)
    deadline = time.monotonic() + 120
    confirmed = 0
    while time.monotonic() < deadline:
        try:
            confirmed = int(
                _wait_lnd_read_rpc(lnd_payer, "walletbalance").get(
                    "confirmed_balance", 0
                )
            )
        except (TypeError, ValueError):
            confirmed = 0
        if confirmed >= amount_sats:
            break
        time.sleep(1)
    if confirmed < amount_sats:
        raise RunnerError(f"LND payer has only {confirmed} confirmed sats after top-up")
    _checkpoint(
        state_path, state, "payer_wallets_topped_up", amount_sats=amount_sats
    )
    return state


def _connect_cln(container: str, peer_id: str, host: str, *, base_managed: bool = False) -> None:
    try:
        _cln_rpc(container, "connect", peer_id, host, 9735, base_managed=base_managed)
    except RunnerError as exc:
        if "already connected" not in str(exc).casefold():
            raise


def wire_contenders(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
) -> dict[str, Any]:
    """Open the two matched 16-channel contender portfolios and confirm each edge."""
    state = _read_state(state_path, topology)
    if state.get("status") not in {"contenders_ready", "contenders_wiring"}:
        raise RunnerError("fresh contenders must be ready before channel wiring")
    network_id = int(state["network_id"])
    contenders = state.get("contenders")
    if not isinstance(contenders, dict) or set(contenders) != {"identity-a", "identity-b"}:
        raise RunnerError("runner state has no matched contender pair")
    state["status"] = "contenders_wiring"
    _checkpoint(state_path, state, "contender_wiring_started")
    completed = {str(row.get("edge")) for row in state.get("contender_channels", [])
                 if isinstance(row, dict)}
    for row in contender_channels(topology):
        source, destination = str(row["source"]), str(row["destination"])
        capacity = int(row["capacity_sats"])
        push_sats = round(capacity * (1.0 - float(row["initial_source_ratio"])))
        identity = str(row["contender_lane"])
        edge = "--".join(sorted((source, destination)))
        if edge in completed:
            continue
        contender = contenders[identity]
        contender_container_name = str(contender["container"])
        contender_id = str(contender["node_id"])
        peer = destination if source == identity else source
        peer_id = _native_base_pubkey(network_id, peer, topology)

        if source == identity:
            _connect_cln(contender_container_name, peer_id, peer)
            _wait_cln_wallet(contender_container_name, capacity + 2_000)
            result = _cln_rpc(
                contender_container_name, "-k", "fundchannel", f"id={peer_id}",
                f"amount={capacity}", f"push_msat={push_sats * 1000}msat",
            )
            funder = identity
        elif peer == "cln-payer":
            payer = _base_container_name(network_id, "cln-payer")
            _connect_cln(payer, contender_id, identity, base_managed=True)
            result = _cln_rpc(
                payer, "-k", "fundchannel", f"id={contender_id}",
                f"amount={capacity}", f"push_msat={push_sats * 1000}msat",
                base_managed=True,
            )
            funder = peer
        elif peer == "lnd-payer":
            payer = _base_container_name(network_id, "lnd-payer")
            try:
                _lnd_rpc(payer, "connect", f"{contender_id}@{identity}:9735", "--perm")
            except RunnerError as exc:
                if "already connected" not in str(exc).casefold():
                    raise
            result = _lnd_rpc(
                payer, "openchannel", "--node_key", contender_id, "--local_amt", capacity,
                "--push_amt", push_sats,
            )
            funder = peer
        else:
            raise RunnerError(f"unexpected inbound contender channel {source}->{destination}")
        record = {
            "edge": edge,
            "identity": identity,
            "peer": peer,
            "funder": funder,
            "capacity_sats": capacity,
            "push_sats": push_sats,
            "fee_ppm": int(row["fee_ppm"]),
            "open_result": result,
        }
        state.setdefault("contender_channels", []).append(record)
        completed.add(edge)
        _checkpoint(state_path, state, "contender_channel_opened", channel=record)
        _mine(bridge, network_id)
        # CLN can lag the regtest backend briefly after confirmation. The next
        # fundchannel must not race its wallet rescan and incorrectly report no
        # available UTXOs.
        if funder == identity:
            _wait_cln_wallet(contender_container_name, 1)

    for identity in ("identity-a", "identity-b"):
        container = str(contenders[identity]["container"])
        deadline = time.monotonic() + 180
        active = 0
        while time.monotonic() < deadline:
            rows = _cln_rpc(container, "listpeerchannels").get("channels")
            if isinstance(rows, list):
                active = sum(
                    isinstance(channel, dict) and channel.get("state") == "CHANNELD_NORMAL"
                    for channel in rows
                )
            if active >= 16:
                break
            time.sleep(2)
        if active < 16:
            raise RunnerError(f"{identity} has only {active}/16 active channels")
    state["status"] = "contender_channels_ready"
    _checkpoint(state_path, state, "contender_channels_ready", count=len(completed))
    return state


def _invoice_on_contender(container: str, amount_sats: int, label: str) -> str:
    result = _cln_rpc(
        container, "invoice", amount_sats * 1000, label, "Grand Prix liquidity shaping"
    )
    invoice = result.get("bolt11")
    if not isinstance(invoice, str) or not invoice:
        raise RunnerError("contender invoice RPC returned no bolt11")
    return invoice


def _contender_invoice_paid(container: str, label: str) -> bool:
    rows = _cln_rpc(container, "listinvoices", label).get("invoices")
    return isinstance(rows, list) and len(rows) == 1 and rows[0].get("status") == "paid"


def _compact_payment_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"result_type": type(value).__name__}
    return {
        key: value[key] for key in ("status", "amount_msat", "amount_sent_msat") if key in value
    }


def _contender_balance_ratio(container: str, peer_id: str) -> float:
    rows = _cln_rpc(container, "listpeerchannels", peer_id).get("channels")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RunnerError(f"expected one contender channel to peer {peer_id}")
    row = rows[0]
    total = _msat(row.get("total_msat", row.get("amount_msat")))
    local = _msat(row.get("to_us_msat", row.get("our_amount_msat")))
    if total <= 0:
        raise RunnerError("contender channel has zero capacity")
    return local / total


def shape_liquidity(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
) -> dict[str, Any]:
    """Move each contender edge to its manifest ratio using direct neighbor payments."""
    state = _read_state(state_path, topology)
    if state.get("status") not in {
        "contender_channels_ready", "liquidity_shaping", "liquidity_ready"
    }:
        raise RunnerError("contender channels must be ready before liquidity shaping")
    network_id = int(state["network_id"])
    contenders = state.get("contenders")
    if not isinstance(contenders, dict):
        raise RunnerError("runner state has no contenders")
    state["status"] = "liquidity_shaping"
    state.setdefault("liquidity_shapes", [])
    _checkpoint(state_path, state, "liquidity_shaping_started")
    completed = {str(row.get("edge")) for row in state["liquidity_shapes"]
                 if isinstance(row, dict) and row.get("status") == "complete"}
    manifest_by_edge = {
        "--".join(sorted((str(row["source"]), str(row["destination"])))): row
        for row in contender_channels(topology)
    }
    for index, (edge, row) in enumerate(manifest_by_edge.items()):
        if edge in completed:
            continue
        source, destination = str(row["source"]), str(row["destination"])
        capacity = int(row["capacity_sats"])
        ratio = float(row["initial_source_ratio"])
        amount_sats = round(capacity * (1.0 - ratio))
        identity = str(row["contender_lane"])
        container = str(contenders[identity]["container"])
        label = f"grand-prix-r{state['replica']}-shape-{index}"
        peer = destination if source == identity else source
        peer_id = _native_base_pubkey(network_id, peer, topology)
        desired_contender_ratio = ratio if source == identity else 1.0 - ratio
        before_ratio = _contender_balance_ratio(container, peer_id)
        if abs(before_ratio - desired_contender_ratio) <= 0.015:
            record = {
                "edge": edge,
                "identity": identity,
                "amount_sats": amount_sats,
                "target_source_ratio": ratio,
                "observed_contender_ratio": round(before_ratio, 6),
                "reconciled_existing_balance": True,
                "payment": {"status": "complete"},
                "status": "complete",
            }
            state["liquidity_shapes"].append(record)
            completed.add(edge)
            _checkpoint(state_path, state, "liquidity_shape_reconciled", shape=record)
            continue
        if destination == identity:
            invoice = _invoice_on_contender(container, amount_sats, label)
            try:
                payment = _pay_native(
                    network_id, topology, source, invoice, amount_sats
                )
                reconciled = False
            except RunnerError as exc:
                if not _contender_invoice_paid(container, label):
                    state["status"] = "liquidity_shape_unknown"
                    _checkpoint(
                        state_path, state, "liquidity_shape_unknown", edge=edge, error=str(exc)
                    )
                    raise RunnerError(
                        f"payment result for {edge} is unknown; do not retry without reconciliation"
                    ) from exc
                payment = {"status": "complete"}
                reconciled = True
        elif source == identity:
            invoice, token = _create_native_invoice(
                network_id, topology, destination, amount_sats, label
            )
            try:
                payment = _cln_rpc(container, "pay", invoice)
                reconciled = False
            except RunnerError as exc:
                if _native_invoice_state(
                    network_id, topology, destination, token
                ) not in {"paid", "settled"}:
                    state["status"] = "liquidity_shape_unknown"
                    _checkpoint(
                        state_path, state, "liquidity_shape_unknown",
                        edge=edge, error=str(exc),
                    )
                    raise RunnerError(
                        f"payment result for {edge} is unknown; "
                        "do not retry without reconciliation"
                    ) from exc
                payment = {"status": "complete"}
                reconciled = True
        else:
            raise RunnerError(f"shape edge {edge} does not contain its contender")
        actual_contender_ratio = _contender_balance_ratio(container, peer_id)
        if abs(actual_contender_ratio - desired_contender_ratio) > 0.015:
            raise RunnerError(
                f"liquidity ratio mismatch on {edge}: wanted {desired_contender_ratio:.3f}, "
                f"observed {actual_contender_ratio:.3f}"
            )
        record = {
            "edge": edge,
            "identity": identity,
            "amount_sats": amount_sats,
            "target_source_ratio": ratio,
            "observed_contender_ratio": round(actual_contender_ratio, 6),
            "reconciled_after_mcp_error": reconciled,
            "payment": _compact_payment_result(payment),
            "status": "complete",
        }
        state["liquidity_shapes"].append(record)
        completed.add(edge)
        _checkpoint(state_path, state, "liquidity_shape_complete", shape=record)
    state["status"] = "liquidity_ready"
    _checkpoint(state_path, state, "liquidity_ready", count=len(completed))
    return state


def fee_policy_plan(topology: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand each undirected manifest fee into two local outgoing policies."""
    validate_topology(topology)
    implementations = {str(row["name"]): str(row["implementation"])
                       for row in topology["nodes"]}
    result = []
    for edge in topology["channels"]:
        source, destination = str(edge["source"]), str(edge["destination"])
        for node, peer in ((source, destination), (destination, source)):
            result.append({
                "node": node,
                "peer": peer,
                "implementation": implementations[node],
                "fee_base_msat": 1_000,
                "fee_ppm": int(edge["fee_ppm"]),
            })
    if len(result) != 114:
        raise RunnerError(f"expected 114 directed fee policies, got {len(result)}")
    return result


def _lnd_node_rpc(network_id: int, node_name: str, topology: dict[str, Any], *arguments: Any) -> dict[str, Any]:
    allowed = {
        str(row["name"]) for row in base_nodes(topology)
        if row["implementation"] == "lnd"
    }
    if node_name not in allowed:
        raise RunnerError(f"refusing unsafe LND node target {node_name!r}")
    container = _base_container_name(network_id, node_name)
    return _json_command(
        ["docker", "exec", "-u", "lnd", container, "lncli", "--network=regtest"]
        + [str(value) for value in arguments]
    )


def seed_fees(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
) -> dict[str, Any]:
    """Apply production-shaped, symmetric starting fees to all 57 channels."""
    state = _read_state(state_path, topology)
    if state.get("status") not in {"liquidity_ready", "fee_seeding", "fees_ready"}:
        raise RunnerError("liquidity must be ready before initial fee seeding")
    network_id = int(state["network_id"])
    contenders = state.get("contenders")
    if not isinstance(contenders, dict):
        raise RunnerError("runner state has no contenders")
    state["status"] = "fee_seeding"
    state.setdefault("fee_policies", [])
    _checkpoint(state_path, state, "fee_seeding_started")
    completed = {f"{row.get('node')}-->{row.get('peer')}" for row in state["fee_policies"]
                 if isinstance(row, dict) and row.get("status") == "complete"}
    for policy in fee_policy_plan(topology):
        node, peer = str(policy["node"]), str(policy["peer"])
        key = f"{node}-->{peer}"
        if key in completed:
            continue
        peer_id = (
            contenders[peer]["node_id"]
            if peer in contenders
            else _native_base_pubkey(network_id, peer, topology)
        )
        if not isinstance(peer_id, str) or not peer_id:
            raise RunnerError(f"could not resolve fee peer {peer}")
        if node in contenders:
            _cln_rpc(
                str(contenders[node]["container"]), "setchannel", peer_id,
                int(policy["fee_base_msat"]), int(policy["fee_ppm"]),
            )
        elif policy["implementation"] == "cln":
            if node not in {row["name"] for row in base_nodes(topology)}:
                raise RunnerError(f"refusing unknown CLN policy node {node}")
            _cln_rpc(
                _base_container_name(network_id, node), "setchannel", peer_id,
                int(policy["fee_base_msat"]), int(policy["fee_ppm"]),
                base_managed=True,
            )
        else:
            channels = _lnd_node_rpc(network_id, node, topology, "listchannels").get("channels")
            matches = [row for row in channels or []
                       if isinstance(row, dict) and row.get("remote_pubkey") == peer_id]
            if len(matches) != 1 or not isinstance(matches[0].get("channel_point"), str):
                raise RunnerError(f"expected one LND channel from {node} to {peer}")
            _lnd_node_rpc(
                network_id, node, topology, "updatechanpolicy",
                "--base_fee_msat", int(policy["fee_base_msat"]),
                "--fee_rate", f"{int(policy['fee_ppm']) / 1_000_000:.6f}",
                "--time_lock_delta", 40,
                "--chan_point", matches[0]["channel_point"],
            )
        record = {**policy, "status": "complete"}
        state["fee_policies"].append(record)
        completed.add(key)
        _checkpoint(state_path, state, "fee_policy_seeded", policy=record)
    state["status"] = "fees_ready"
    _checkpoint(state_path, state, "fees_ready", count=len(completed))
    return state


def _clboss_status(container: str) -> dict[str, Any]:
    completed = _run([
        "docker", "exec", container, "lightning-cli", "--network=regtest",
        "--notifications=none", "clboss-status",
    ])
    normalized = re.sub(
        r"([:\[,]\s*)[-+]?nan(?=\s*[,}\]])", r"\1null", completed.stdout,
        flags=re.IGNORECASE,
    )
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise RunnerError("CLBOSS status returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise RunnerError("CLBOSS status returned a non-object")
    return value


def _wait_clboss(container: str) -> dict[str, Any]:
    last = "not ready"
    for _attempt in range(60):
        try:
            status = _clboss_status(container)
            if isinstance(status.get("unmanaged"), dict):
                return status
            last = "unmanaged safety map missing"
        except RunnerError as exc:
            last = str(exc)
        time.sleep(1)
    raise RunnerError(f"CLBOSS did not become ready: {last}")


def _apply_equivalent_competitor_policy(
    container: str, model: dict[str, Any]
) -> dict[str, Any]:
    """Apply one pure model response and retain only anonymous evidence."""
    channels = _cln_rpc(container, "listpeerchannels").get("channels")
    intents = policy_intents(model, channels)
    targets: list[int] = []
    for intent in intents:
        _cln_rpc(
            container, "setchannel", intent["peer_id"],
            intent["fee_base_msat"], intent["fee_ppm"],
        )
        targets.append(int(intent["fee_ppm"]))
    return {
        "eligible_channels": len(channels) if isinstance(channels, list) else 0,
        "changed_channels": len(intents),
        "target_fee_ppm": {
            "min": min(targets) if targets else None,
            "max": max(targets) if targets else None,
        },
    }


def _equivalent_model_context(
    competitor_controller: str, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        catalog = load_models(config_path)
        validation = validate_models(catalog)
    except (OSError, EquivalentControllerError) as exc:
        raise RunnerError(f"invalid equivalent-controller configuration: {exc}") from exc
    model = catalog["models"].get(competitor_controller)
    if not isinstance(model, dict):
        raise RunnerError(f"equivalent-controller model {competitor_controller!r} is absent")
    return model, validation


def start_controllers(
    topology: dict[str, Any],
    *,
    state_path: Path,
    revenue_market_mode: str = "undercut",
    revenue_max_fee_ppm: int = 2000,
    revenue_dynamic_htlcmax: bool = True,
    competitor_controller: str = "clboss",
    equivalent_controller_config: Path = EQUIVALENT_CONTROLLER_CONFIG,
) -> dict[str, Any]:
    """Start a fee-only crossed league with explicit comparator claim scope."""
    allowed_modes = {
        "undercut", "match", "premium", "competition_aware", "yield_aware"
    }
    if revenue_market_mode not in allowed_modes:
        raise RunnerError(f"unsupported Revenue Ops market mode {revenue_market_mode!r}")
    if (
        isinstance(revenue_max_fee_ppm, bool)
        or not isinstance(revenue_max_fee_ppm, int)
        or not 1 <= revenue_max_fee_ppm <= 100_000
    ):
        raise RunnerError("Revenue Ops max fee must be an integer from 1 to 100000 ppm")
    if not isinstance(revenue_dynamic_htlcmax, bool):
        raise RunnerError("Revenue Ops dynamic htlcmax arm must be boolean")
    if competitor_controller not in COMPETITOR_CONTROLLERS:
        raise RunnerError(f"unsupported competitor controller {competitor_controller!r}")
    model = None
    model_validation = None
    if competitor_controller != "clboss":
        model, model_validation = _equivalent_model_context(
            competitor_controller, equivalent_controller_config
        )
    state = _read_state(state_path, topology)
    if state.get("status") != "fees_ready":
        raise RunnerError("initial fee policies must be ready before controllers start")
    contenders = state.get("contenders")
    assignment = state.get("assignment")
    if not isinstance(contenders, dict) or not isinstance(assignment, dict):
        raise RunnerError("runner state has no crossed contender assignment")

    competitor_identity = str(assignment["clboss"])
    competitor_container = str(contenders[competitor_identity]["container"])
    competitor_readback: dict[str, Any]
    if competitor_controller == "clboss":
        _cln_rpc(competitor_container, "plugin", "start", XREBALANCE_PLUGIN)
        _cln_rpc(
            competitor_container, "-k", "plugin", "subcommand=start",
            f"plugin={CLBOSS_PLUGIN}", "clboss-auto-close=false",
            "clboss-rebalance-mode=off",
        )
        _wait_clboss(competitor_container)
        _cln_rpc(competitor_container, "clboss-ignore-onchain", 96)
        peer_rows = _cln_rpc(competitor_container, "listpeerchannels").get("channels")
        peer_ids = sorted({str(row.get("peer_id")) for row in peer_rows or []
                           if isinstance(row, dict) and row.get("peer_id")})
        if len(peer_ids) != 16:
            raise RunnerError(f"CLBOSS contender has {len(peer_ids)}/16 peers")
        for peer_id in peer_ids:
            _cln_rpc(competitor_container, "clboss-unmanage", peer_id, "open,close")
        clboss_status = _wait_clboss(competitor_container)
        unmanaged = clboss_status.get("unmanaged")
        if not isinstance(unmanaged, dict) or any(
            "open" not in str(unmanaged.get(peer_id, ""))
            or "close" not in str(unmanaged.get(peer_id, ""))
            for peer_id in peer_ids
        ):
            raise RunnerError("CLBOSS open/close safety tags failed readback")
        competitor_readback = {
            "id": "clboss", "comparison_class": "direct_runtime",
            "direct_runtime": True, "identity": competitor_identity,
            "peer_safety_count": len(peer_ids), "auto_close": False,
            "rebalance_mode": "off",
        }
    else:
        assert model is not None and model_validation is not None
        initial_response = _apply_equivalent_competitor_policy(
            competitor_container, model
        )
        competitor_readback = {
            "id": competitor_controller,
            "comparison_class": model["comparison_class"],
            "direct_runtime": False,
            "identity": competitor_identity,
            "configuration_digest": model_validation["catalog_digest"],
            "model_digest": _digest(model),
            "source_revision": model["source_revision"],
            "claim_scope": model["claim_scope"],
            "model": model,
            "trigger": model["trigger"],
            "rebalance_mode": "off",
            "initial_response": initial_response,
            "refresh_count": 1,
        }
    state["competitor_controller"] = competitor_controller
    _checkpoint(
        state_path, state, "competitor_started",
        identity=competitor_identity, controller=competitor_controller,
    )

    revenue_identity = str(assignment["revenue_ops"])
    revenue_container = str(contenders[revenue_identity]["container"])
    revenue_plugin = _revenue_plugin_path(revenue_container)
    _cln_rpc(
        revenue_container, "-k", "plugin", "subcommand=start", f"plugin={revenue_plugin}",
        "revenue-ops-dry-run=false", "revenue-ops-daily-budget-sats=0",
        f"revenue-ops-market-fee-mode={revenue_market_mode}",
        f"revenue-ops-max-fee-ppm={revenue_max_fee_ppm}",
        "revenue-ops-enable-dynamic-htlcmax="
        f"{'true' if revenue_dynamic_htlcmax else 'false'}",
        f"revenue-ops-flow-interval={CONTROLLER_CYCLE_SECONDS}",
        f"revenue-ops-fee-interval={CONTROLLER_CYCLE_SECONDS}",
        f"revenue-ops-rebalance-interval={CONTROLLER_CYCLE_SECONDS}",
    )
    _cln_rpc(revenue_container, "revenue-config", "set", "paused", "true")
    paused = _cln_rpc(revenue_container, "revenue-status")
    paused_values = ((paused.get("operator_controls") or {}).get("values") or {})
    if paused_values.get("paused") is not True or paused_values.get("daily_budget_sats") != 0:
        raise RunnerError("Revenue Ops failed paused zero-budget safety readback")
    max_fee_readback = _cln_rpc(
        revenue_container, "revenue-config", "get", "max_fee_ppm"
    )
    market_mode_readback = _cln_rpc(
        revenue_container, "revenue-config", "get", "market_fee_mode"
    )
    dynamic_htlcmax_readback = _cln_rpc(
        revenue_container, "revenue-config", "get", "enable_dynamic_htlcmax"
    )
    if max_fee_readback.get("value") != revenue_max_fee_ppm:
        raise RunnerError("Revenue Ops max-fee arm failed readback")
    if market_mode_readback.get("value") != revenue_market_mode:
        raise RunnerError("Revenue Ops market-mode arm failed readback")
    if dynamic_htlcmax_readback.get("value") is not revenue_dynamic_htlcmax:
        raise RunnerError("Revenue Ops dynamic-htlcmax arm failed readback")
    _cln_rpc(revenue_container, "revenue-config", "set", "paused", "false")
    revenue_status = _cln_rpc(revenue_container, "revenue-status")
    revenue_values = ((revenue_status.get("operator_controls") or {}).get("values") or {})
    if revenue_values.get("paused") is not False or revenue_values.get("daily_budget_sats") != 0:
        raise RunnerError("Revenue Ops failed active zero-budget safety readback")
    _checkpoint(state_path, state, "revenue_ops_started", identity=revenue_identity)

    # Give every controller a complete cold-start observation window before
    # traffic so startup order cannot decide route share. This also covers
    # CLBOSS's approximately one-minute initial publication cadence.
    time.sleep(CONTROLLER_WARMUP_SECONDS)
    warm_policies = {
        identity: _channel_policy_snapshot(str(contenders[identity]["container"]))
        for identity in ("identity-a", "identity-b")
    }
    if any(row["active_channels"] != 16 for row in warm_policies.values()):
        raise RunnerError("controller warm-up lost an active contender channel")

    state["controller_readback"] = {
        "order": [competitor_controller, "revenue_ops"],
        "competitor": competitor_readback,
        "revenue_ops": {
            "identity": revenue_identity,
            "paused": False,
            "daily_budget_sats": 0,
            "cycle_seconds": CONTROLLER_CYCLE_SECONDS,
            "market_fee_mode": revenue_market_mode,
            "max_fee_ppm": revenue_max_fee_ppm,
            "dynamic_htlcmax": revenue_dynamic_htlcmax,
        },
        "warmup_seconds": CONTROLLER_WARMUP_SECONDS,
        "warm_policies": warm_policies,
    }
    if competitor_controller == "clboss":
        # Frozen pre-expansion states and scorers used this exact key.
        state["controller_readback"]["clboss"] = competitor_readback
    state["status"] = "controllers_ready"
    _checkpoint(
        state_path, state, "controllers_ready", league="fee_only",
        competitor=competitor_controller,
    )
    return state


def _refresh_equivalent_competitor(state: dict[str, Any]) -> dict[str, Any] | None:
    """Re-evaluate an admitted event-driven model from its frozen state copy."""
    controls = state.get("controller_readback")
    contenders = state.get("contenders")
    assignment = state.get("assignment")
    if not all(isinstance(value, dict) for value in (controls, contenders, assignment)):
        raise RunnerError("runner state lacks controller refresh context")
    competitor = controls.get("competitor")
    if not isinstance(competitor, dict) or competitor.get("id") != "torq":
        return None
    model = competitor.get("model")
    if not isinstance(model, dict):
        raise RunnerError("equivalent competitor lacks its frozen model")
    identity = str(assignment["clboss"])
    container = str(contenders[identity]["container"])
    response = _apply_equivalent_competitor_policy(container, model)
    competitor["refresh_count"] = int(competitor.get("refresh_count", 0)) + 1
    competitor["last_response"] = response
    return response


def _equivalent_refresh_due(state: dict[str, Any], record: dict[str, Any]) -> bool:
    """Return whether a Torq-style managed channel actually changed enough."""
    try:
        controls = state["controller_readback"]
        competitor = controls["competitor"]
        if competitor.get("id") != "torq" or record.get("outcome") != "settled":
            return False
        threshold_msat = int(
            competitor["trigger"]["minimum_balance_change_sats"]
        ) * 1000
        competitor_identity = state["assignment"]["clboss"]
        changed_msat = int(
            record["contender_delta"][competitor_identity]["volume_msat"]
        )
        return threshold_msat >= 0 and changed_msat >= threshold_msat
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _forward_totals(container: str) -> dict[str, int]:
    rows = _cln_rpc(container, "listforwards").get("forwards")
    if not isinstance(rows, list):
        raise RunnerError("listforwards returned no forward list")
    settled = [row for row in rows
               if isinstance(row, dict) and row.get("status") == "settled"]
    fee_msat = 0
    volume_msat = 0
    for row in settled:
        incoming = _msat(row.get("in_msat", row.get("received_msat", 0)))
        outgoing = _msat(row.get("out_msat", 0))
        fee_msat += _msat(row["fee_msat"]) if row.get("fee_msat") is not None else max(
            0, incoming - outgoing
        )
        volume_msat += outgoing
    return {"settled_count": len(settled), "volume_msat": volume_msat, "fee_msat": fee_msat}


def _channel_policy_snapshot(container: str) -> dict[str, Any]:
    rows = _cln_rpc(container, "listpeerchannels").get("channels")
    if not isinstance(rows, list):
        raise RunnerError("listpeerchannels returned no channel list")
    valid = [
        row for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("fee_proportional_millionths"), int)
        and row.get("total_msat") is not None
        and row.get("to_us_msat") is not None
    ]
    if not valid:
        raise RunnerError("listpeerchannels returned no usable policy rows")
    fees = sorted(int(row["fee_proportional_millionths"]) for row in valid)
    total_msat = sum(_msat(row["total_msat"]) for row in valid)
    local_msat = sum(_msat(row["to_us_msat"]) for row in valid)
    return {
        "channels": len(valid),
        "active_channels": sum(bool(row.get("peer_connected")) for row in valid),
        "fee_ppm": {
            "min": fees[0],
            "median": float(statistics.median(fees)),
            "mean": sum(fees) / len(fees),
            "max": fees[-1],
        },
        "local_balance_ratio": local_msat / total_msat if total_msat else 0.0,
    }


def inspect_live_contenders(topology: dict[str, Any], *, state_path: Path) -> dict[str, Any]:
    """Return anonymous aggregate live policy state without mutating the lab."""
    state = _read_state(state_path, topology)
    contenders = state.get("contenders")
    if not isinstance(contenders, dict):
        raise RunnerError("runner state has no contenders")
    return {
        "status": state.get("status"),
        "assignment": state.get("assignment"),
        "contenders": {
            identity: _channel_policy_snapshot(str(contenders[identity]["container"]))
            for identity in ("identity-a", "identity-b")
        },
    }


def _contender_metric_snapshot(state: dict[str, Any]) -> dict[str, dict[str, int]]:
    contenders = state.get("contenders")
    if not isinstance(contenders, dict):
        raise RunnerError("runner state has no contenders")
    return {
        identity: _forward_totals(str(contenders[identity]["container"]))
        for identity in ("identity-a", "identity-b")
    }


def _metric_delta(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for identity in ("identity-a", "identity-b"):
        row = {key: after[identity][key] - before[identity][key]
               for key in ("settled_count", "volume_msat", "fee_msat")}
        row["fee_ppm_on_forwarded_volume"] = (
            row["fee_msat"] * 1_000_000 / row["volume_msat"]
            if row["volume_msat"] else 0.0
        )
        row["fee_msat_per_130m_sat"] = row["fee_msat"] / 130_000_000
        result[identity] = row
    return result


def _create_native_invoice(
    network_id: int, topology: dict[str, Any], sink: str, amount_sats: int, label: str
) -> tuple[str, dict[str, str]]:
    implementations = {
        str(row["name"]): str(row["implementation"])
        for row in base_nodes(topology)
    }
    if implementations.get(sink) == "cln":
        result = _cln_rpc(
            _base_container_name(network_id, sink), "invoice", amount_sats * 1000,
            label, "Grand Prix public traffic", base_managed=True,
        )
        invoice = result.get("bolt11")
        token = {"family": "cln", "label": label}
    elif implementations.get(sink) == "lnd":
        result = _lnd_node_rpc(
            network_id, sink, topology, "addinvoice", "--amt", amount_sats, "--memo", label
        )
        invoice = result.get("payment_request")
        token = {"family": "lnd", "r_hash": str(result.get("r_hash") or "")}
    else:
        raise RunnerError(f"unsupported traffic sink {sink}")
    if not isinstance(invoice, str) or not invoice:
        raise RunnerError(f"{sink} returned no invoice")
    if token["family"] == "lnd" and not token["r_hash"]:
        raise RunnerError("LND sink returned no invoice hash")
    return invoice, token


def _native_invoice_state(
    network_id: int, topology: dict[str, Any], sink: str, token: dict[str, str]
) -> str:
    if token["family"] == "cln":
        rows = _cln_rpc(
            _base_container_name(network_id, sink), "listinvoices", token["label"],
            base_managed=True,
        ).get("invoices")
        if isinstance(rows, list) and len(rows) == 1:
            return str(rows[0].get("status", "unknown")).casefold()
    else:
        result = _lnd_node_rpc(
            network_id, sink, topology, "lookupinvoice", token["r_hash"]
        )
        return str(result.get("state", "unknown")).casefold()
    return "unknown"


def _native_payer_has_onchain_channel(
    network_id: int, topology: dict[str, Any], payer: str
) -> bool:
    """Return whether timeout resolution has destroyed graph comparability."""
    implementations = {
        str(row["name"]): str(row["implementation"])
        for row in base_nodes(topology)
    }
    if implementations.get(payer) == "cln":
        rows = _cln_rpc(
            _base_container_name(network_id, payer), "listpeerchannels",
            base_managed=True,
        ).get("channels")
        return any(
            isinstance(row, dict)
            and str(row.get("state", "")).casefold() == "onchain"
            for row in rows or []
        )
    if implementations.get(payer) == "lnd":
        pending = _lnd_node_rpc(
            network_id, payer, topology, "pendingchannels"
        )
        return any(
            isinstance(pending.get(key), list) and bool(pending[key])
            for key in (
                "pending_force_closing_channels",
                "waiting_close_channels",
                "pending_closing_channels",
            )
        )
    raise RunnerError(f"unsupported traffic payer {payer}")


def _pay_native(
    network_id: int,
    topology: dict[str, Any],
    payer: str,
    invoice: str,
    amount_sats: int,
) -> dict[str, Any]:
    amount_sats = _positive(amount_sats, "payment amount")
    implementations = {
        str(row["name"]): str(row["implementation"])
        for row in base_nodes(topology)
    }
    if implementations.get(payer) == "cln":
        max_fee_msat = max(10_000, amount_sats * 100)
        return _cln_rpc(
            _base_container_name(network_id, payer), "-k", "xpay", f"invstring={invoice}",
            "retry_for=5", f"maxfee={max_fee_msat}msat", base_managed=True,
        )
    if implementations.get(payer) == "lnd":
        return _lnd_node_rpc(
            network_id, payer, topology, "payinvoice", "--force", "--timeout", "5s",
            "--fee_limit_percent", "10", invoice
        )
    raise RunnerError(f"unsupported traffic payer {payer}")


def run_public_traffic(
    topology: dict[str, Any],
    *,
    state_path: Path,
    limit: int,
) -> dict[str, Any]:
    """Run or resume the public deterministic traffic seed using native pathfinding."""
    state = _read_state(state_path, topology)
    if state.get("status") not in {
        "controllers_ready", "public_traffic_running", "public_traffic_partial",
        "public_traffic_complete",
    }:
        raise RunnerError("fee-only controllers must be ready before public traffic")
    limit = _positive(limit, "traffic limit")
    traffic = topology["traffic"]
    target = min(limit, len(traffic))
    network_id = int(state["network_id"])
    run = state.setdefault("public_traffic", {
        "seed": int(topology["public_seed"]),
        "before": _contender_metric_snapshot(state),
        "records": [],
    })
    if run.get("seed") != int(topology["public_seed"]) or not isinstance(run.get("records"), list):
        raise RunnerError("public traffic checkpoint is malformed")
    for prior in run["records"]:
        if isinstance(prior, dict) and "error" in prior:
            prior["error_code"] = "native_payment_failed"
            prior.pop("error", None)
    completed = {int(row["sequence"]) for row in run["records"]
                 if isinstance(row, dict) and isinstance(row.get("sequence"), int)}
    if run["records"]:
        last_after = run["records"][-1].get("contender_after")
        if not isinstance(last_after, dict):
            # Older checkpoints remain resumable, but cannot be promoted by
            # the cell-attribution scorer because their early payments lack
            # per-payment contender deltas.
            run["per_payment_attribution_complete"] = False
            metric_cursor = _contender_metric_snapshot(state)
        else:
            metric_cursor = last_after
    else:
        metric_cursor = run["before"]
        run["per_payment_attribution_complete"] = True
    state["status"] = "public_traffic_running"
    _checkpoint(state_path, state, "public_traffic_started", target=target)
    initial_refresh = _refresh_equivalent_competitor(state)
    if initial_refresh is not None:
        _checkpoint(
            state_path, state, "equivalent_competitor_refreshed",
            changed_channels=initial_refresh["changed_channels"],
        )
    pauses = {"lt_1": 0.01, "1_10": 0.03, "10_60": 0.05,
              "60_300": 0.08, "gte_300": 0.12}
    for item in traffic[:target]:
        sequence = int(item["sequence"])
        if sequence in completed:
            continue
        payer, sink = str(item["payer"]), str(item["sink"])
        amount_sats = int(item["amount_sats"])
        label = f"grand-prix-public-r{state['replica']}-{sequence}"
        invoice, token = _create_native_invoice(
            network_id, topology, sink, amount_sats, label
        )
        try:
            payment = _pay_native(
                network_id, topology, payer, invoice, amount_sats
            )
            outcome = "settled"
            error = None
        except RunnerError as exc:
            time.sleep(0.25)
            invoice_state = _native_invoice_state(network_id, topology, sink, token)
            if invoice_state in {"paid", "settled"}:
                outcome = "settled"
                payment = {"status": "complete"}
                error = None
            elif "timed out" in str(exc).casefold() and invoice_state in {"open", "accepted"}:
                state["status"] = "public_traffic_unknown"
                _checkpoint(
                    state_path, state, "public_payment_unknown", sequence=sequence,
                    payer=payer, sink=sink,
                )
                raise RunnerError(
                    f"public payment {sequence} is unresolved; do not retry"
                ) from exc
            else:
                outcome = "failed"
                payment = {}
                error = "native_payment_failed"
        record = {
            "sequence": sequence,
            "class": str(item["class"]),
            "payer": payer,
            "sink": sink,
            "amount_sats": amount_sats,
            "outcome": outcome,
            "payment": _compact_payment_result(payment),
        }
        if error:
            record["error_code"] = error
        contender_after = _contender_metric_snapshot(state)
        record["contender_delta"] = _metric_delta(metric_cursor, contender_after)
        record["contender_after"] = contender_after
        metric_cursor = contender_after
        run["records"].append(record)
        completed.add(sequence)
        _checkpoint(state_path, state, "public_payment_complete", record=record)
        if _equivalent_refresh_due(state, record):
            response = _refresh_equivalent_competitor(state)
            assert response is not None
            _checkpoint(
                state_path, state, "equivalent_competitor_refreshed",
                sequence=sequence, changed_channels=response["changed_channels"],
            )
        time.sleep(pauses[str(item["interarrival_bucket"])])
    run["after"] = _contender_metric_snapshot(state)
    run["contender_delta"] = _metric_delta(run["before"], run["after"])
    run["post_traffic_unattributed_delta"] = _metric_delta(metric_cursor, run["after"])
    run["settled_count"] = sum(row["outcome"] == "settled" for row in run["records"])
    run["failed_count"] = sum(row["outcome"] == "failed" for row in run["records"])
    run["settled_volume_sats"] = sum(
        row["amount_sats"] for row in run["records"] if row["outcome"] == "settled"
    )
    state["status"] = (
        "public_traffic_complete" if len(completed) == len(traffic) else "public_traffic_partial"
    )
    _checkpoint(
        state_path, state, state["status"], completed=len(completed), total=len(traffic)
    )
    return state


def reconcile_public_unknown(
    topology: dict[str, Any], *, state_path: Path
) -> dict[str, Any]:
    """Resolve one stopped public payment from sink and payer terminal readback."""
    state = _read_state(state_path, topology)
    if state.get("status") != "public_traffic_unknown":
        raise RunnerError("there is no unknown public payment to reconcile")
    events = state.get("events")
    if not isinstance(events, list):
        raise RunnerError("unknown-payment checkpoint is malformed")
    unknown_indexes = [
        index for index, row in enumerate(events)
        if isinstance(row, dict) and row.get("event") == "public_payment_unknown"
    ]
    if not unknown_indexes:
        raise RunnerError("unknown-payment checkpoint is malformed")
    unknown_index = unknown_indexes[-1]
    if any(
        not isinstance(row, dict) or row.get("event") != "public_timeout_blocks_mined"
        for row in events[unknown_index + 1:]
    ):
        raise RunnerError("unknown-payment checkpoint has unexpected later events")
    event = events[unknown_index]
    sequence = int(event["sequence"])
    item = topology["traffic"][sequence]
    payer, sink = str(item["payer"]), str(item["sink"])
    network_id = int(state["network_id"])
    label = f"grand-prix-public-r{state['replica']}-{sequence}"
    if sink in {"cln-sink", "cln-payer"}:
        rows = _cln_rpc(
            _base_container_name(network_id, sink), "listinvoices", label, base_managed=True
        ).get("invoices")
        matches = rows if isinstance(rows, list) else []
        invoice_state = str(matches[0].get("status", "unknown")).casefold() if len(matches) == 1 else "unknown"
    else:
        rows = _lnd_node_rpc(network_id, sink, topology, "listinvoices").get("invoices")
        matches = [row for row in rows or []
                   if isinstance(row, dict) and row.get("memo") == label]
        invoice_state = str(matches[0].get("state", "unknown")).casefold() if len(matches) == 1 else "unknown"
    if payer in {"cln-payer", "cln-sink"}:
        payments = _cln_rpc(
            _base_container_name(network_id, payer), "listsendpays", base_managed=True
        ).get("payments")
        inflight = any(isinstance(row, dict) and row.get("status") == "pending"
                       for row in payments or [])
    else:
        payments = _lnd_node_rpc(network_id, payer, topology, "listpayments").get("payments")
        inflight = any(isinstance(row, dict) and str(row.get("status", "")).casefold() == "in_flight"
                       for row in payments or [])
    if invoice_state in {"paid", "settled"}:
        outcome = "settled"
    elif invoice_state in {"open", "unpaid", "canceled", "cancelled"} and not inflight:
        outcome = "failed"
    elif inflight and _native_payer_has_onchain_channel(
        network_id, topology, payer
    ):
        # Once timeout handling closes a channel, later traffic no longer sees
        # the frozen graph. Preserve the ambiguous checkpoint, but make the
        # whole replica explicitly non-scoreable instead of inviting further
        # block advances or a misleading resume.
        state["status"] = "public_traffic_invalid"
        _checkpoint(
            state_path,
            state,
            "public_reconciliation_invalidated",
            sequence=sequence,
            reason="payer_channel_onchain",
        )
        return state
    else:
        raise RunnerError(
            f"payment {sequence} remains unresolved (invoice={invoice_state}, inflight={inflight})"
        )
    record = {
        "sequence": sequence,
        "class": str(item["class"]),
        "payer": payer,
        "sink": sink,
        "amount_sats": int(item["amount_sats"]),
        "outcome": outcome,
        "payment": {"status": "complete"} if outcome == "settled" else {},
        "reconciled_after_timeout": True,
    }
    if outcome == "failed":
        record["error_code"] = "native_payment_timeout_terminal"
    run = state["public_traffic"]
    if run.get("records"):
        metric_before = run["records"][-1].get("contender_after")
    else:
        metric_before = run.get("before")
    if isinstance(metric_before, dict):
        contender_after = _contender_metric_snapshot(state)
        record["contender_delta"] = _metric_delta(metric_before, contender_after)
        record["contender_after"] = contender_after
    else:
        run["per_payment_attribution_complete"] = False
    run["records"].append(record)
    state["status"] = "public_traffic_partial"
    _checkpoint(state_path, state, "public_payment_reconciled", record=record)
    return state


def advance_public_timeout(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
    blocks: int,
) -> dict[str, Any]:
    """Advance regtest only to resolve a checkpointed ambiguous HTLC attempt."""
    state = _read_state(state_path, topology)
    if state.get("status") != "public_traffic_unknown":
        raise RunnerError("timeout blocks are allowed only for an unknown public payment")
    blocks = _positive(blocks, "timeout blocks")
    if blocks > 2016:
        raise RunnerError("timeout blocks must not exceed 2016")
    bridge.call("mine_blocks", {"networkId": int(state["network_id"]), "blocks": blocks})
    _checkpoint(state_path, state, "public_timeout_blocks_mined", blocks=blocks)
    return state


def stop_lab(
    bridge: LabBackend,
    topology: dict[str, Any],
    *,
    state_path: Path,
) -> dict[str, Any]:
    """Stop one completed lab and remove its exactly named contenders.

    Score/state artifacts remain on disk. Docker-only labs additionally remove
    their labeled network and data volumes inside the backend implementation.
    """
    state = _read_state(state_path, topology)
    network_id = int(state["network_id"])
    replica = state.get("replica")
    if isinstance(replica, int) and replica > 0:
        for identity in ("identity-a", "identity-b"):
            container = contender_container(network_id, replica, identity)
            if _docker_exists(container):
                _run(["docker", "rm", "-f", container])
    bridge.call("stop_network", {"networkId": network_id})
    state["status"] = "stopped"
    _checkpoint(state_path, state, "lab_stopped", backend=ACTIVE_BACKEND)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=(
            "plan", "create-base", "start-base", "wire-background",
            "launch-contenders", "wire-contenders", "shape-liquidity", "seed-fees",
            "start-controllers", "run-public", "advance-timeout", "reconcile-public", "status",
            "inspect-live", "top-up-payers", "stop-lab",
        )
    )
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument(
        "--state", type=Path, default=Path("results/grand-prix/runner-state.json")
    )
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--replica", type=int, default=1)
    parser.add_argument(
        "--image",
        help="explicit contender image tag (required by launch-contenders)",
    )
    parser.add_argument("--traffic-limit", type=int, default=240)
    parser.add_argument("--timeout-blocks", type=int, default=144)
    parser.add_argument("--payer-top-up-sats", type=int, default=40_000_000)
    parser.add_argument(
        "--revenue-market-mode",
        choices=(
            "undercut", "match", "premium", "competition_aware", "yield_aware"
        ),
        default="undercut",
    )
    parser.add_argument("--revenue-max-fee-ppm", type=int, default=2000)
    parser.add_argument(
        "--competitor-controller", choices=sorted(COMPETITOR_CONTROLLERS),
        default="clboss",
    )
    parser.add_argument(
        "--equivalent-controller-config", type=Path,
        default=EQUIVALENT_CONTROLLER_CONFIG,
    )
    parser.add_argument(
        "--revenue-dynamic-htlcmax", choices=("on", "off"), default="on"
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    topology = _load_json(args.topology)
    validate_topology(topology)
    if args.command in MUTATING_COMMANDS and not args.apply:
        raise RunnerError(f"{args.command} mutates the fake-sat lab; pass --apply")
    lock = _acquire_mutation_lock(args.state) if args.command in MUTATING_COMMANDS else None
    try:
        if args.command == "plan":
            result = runtime_plan(topology, args.name)
        elif args.command == "status":
            result = _read_state(args.state, topology)
        elif args.command == "inspect-live":
            result = inspect_live_contenders(topology, state_path=args.state)
        else:
            bridge = DockerGrandPrixLab(args.state)
            if args.command == "create-base":
                result = create_base(bridge, topology, name=args.name, state_path=args.state)
            elif args.command == "start-base":
                result = start_base(bridge, topology, state_path=args.state)
            elif args.command == "wire-background":
                result = wire_background(
                    bridge,
                    topology,
                    state_path=args.state,
                    native_io=False,
                )
            elif args.command == "launch-contenders":
                if not args.image:
                    raise RunnerError(
                        "launch-contenders requires an explicit --image; implicit "
                        "defaults are unsafe for attested experiments"
                    )
                result = launch_contenders(
                    bridge, topology, state_path=args.state, replica=args.replica,
                    image=args.image
                )
            elif args.command == "wire-contenders":
                result = wire_contenders(bridge, topology, state_path=args.state)
            elif args.command == "shape-liquidity":
                result = shape_liquidity(bridge, topology, state_path=args.state)
            elif args.command == "seed-fees":
                result = seed_fees(bridge, topology, state_path=args.state)
            elif args.command == "top-up-payers":
                result = top_up_payers(
                    bridge,
                    topology,
                    state_path=args.state,
                    amount_sats=args.payer_top_up_sats,
                )
            elif args.command == "start-controllers":
                result = start_controllers(
                    topology,
                    state_path=args.state,
                    revenue_market_mode=args.revenue_market_mode,
                    revenue_max_fee_ppm=args.revenue_max_fee_ppm,
                    revenue_dynamic_htlcmax=(args.revenue_dynamic_htlcmax == "on"),
                    competitor_controller=args.competitor_controller,
                    equivalent_controller_config=args.equivalent_controller_config,
                )
            elif args.command == "run-public":
                result = run_public_traffic(
                    topology, state_path=args.state, limit=args.traffic_limit
                )
            elif args.command == "advance-timeout":
                result = advance_public_timeout(
                    bridge, topology, state_path=args.state, blocks=args.timeout_blocks
                )
            elif args.command == "stop-lab":
                result = stop_lab(bridge, topology, state_path=args.state)
            else:
                result = reconcile_public_unknown(topology, state_path=args.state)
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunnerError, DockerLabError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
