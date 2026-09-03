#!/usr/bin/env python3
"""Direct-Docker backend for the production-shaped Grand Prix laboratory.

The class implements the small orchestration surface used by the tournament
runner. Every created Docker object carries a run-scoped label, and cleanup
resolves resources from that label before removing them.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


BITCOIND_IMAGE = (
    "polarlightning/bitcoind@"
    "sha256:6b15e7efb79995a18441806f509e40316428a901f1cdc5c54cd25b03ac513cb9"
)
CLN_IMAGE = (
    "polarlightning/clightning@"
    "sha256:a9cf89b0e1afacca961dcc8d3cc7a94a0dbc87854d0714dcd86849ba9dc388fb"
)
LND_IMAGE = (
    "polarlightning/lnd@"
    "sha256:ad708a2dacccd6ae104e78577f6a724095b80bac76ddf363f4bf8d22fbe0979f"
)
RESOURCE_LABEL = "io.lightning-goats.revenue-ops-grand-prix"
SAFE_RESOURCE_RE = re.compile(r"^revenue-gp-n[1-9][0-9]*(?:-[a-z0-9-]+)?$")


class DockerLabError(RuntimeError):
    """A Docker-only lab operation failed or violated a scope invariant."""


def _run(
    command: Sequence[str], *, check: bool = True, timeout: float = 180
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=check,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise DockerLabError(
            f"command failed ({exc.returncode}): {list(command)!r}: {detail}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerLabError(f"command timed out: {list(command)!r}") from exc


def _json_command(command: Sequence[str], *, timeout: float = 180) -> dict[str, Any]:
    completed = _run(command, timeout=timeout)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DockerLabError(f"command returned non-JSON: {list(command)!r}") from exc
    if not isinstance(value, dict):
        raise DockerLabError(f"command returned a non-object: {list(command)!r}")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise DockerLabError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DockerLabError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise DockerLabError(f"{label} must be a positive integer")
    return parsed


def _msat(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("msat")
    rendered = str(value or "").removesuffix("msat")
    try:
        parsed = int(rendered)
    except ValueError as exc:
        raise DockerLabError(f"invalid msat value: {value!r}") from exc
    if parsed < 0:
        raise DockerLabError(f"invalid msat value: {value!r}")
    return parsed


class DockerGrandPrixLab:
    """Run one state-file-scoped regtest laboratory directly through Docker."""

    def __init__(self, runner_state_path: Path):
        self.runner_state_path = Path(runner_state_path)
        self.metadata_path = self.runner_state_path.with_suffix(
            self.runner_state_path.suffix + ".docker-lab.json"
        )

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "list_networks": self._list_networks,
            "create_network": self._create_network,
            "rename_node": self._rename_node,
            "start_network": self._start_network,
            "get_node_info": self._get_node_info,
            "list_channels": self._list_channels,
            "open_channel": self._open_channel,
            "mine_blocks": self._mine_blocks,
            "stop_network": self._stop_network,
        }
        handler = handlers.get(tool)
        if handler is None:
            raise DockerLabError(f"unsupported Docker lab operation: {tool!r}")
        return handler(arguments)

    def container_name(self, network_id: int, node_name: str) -> str:
        name = f"revenue-gp-n{_positive(network_id, 'network id')}-{node_name}"
        if not SAFE_RESOURCE_RE.fullmatch(name):
            raise DockerLabError(f"unsafe Docker lab container name: {name!r}")
        return name

    def network_name(self, network_id: int) -> str:
        name = f"revenue-gp-n{_positive(network_id, 'network id')}"
        if not SAFE_RESOURCE_RE.fullmatch(name):
            raise DockerLabError(f"unsafe Docker lab network name: {name!r}")
        return name

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DockerLabError(f"cannot read Docker lab metadata: {exc}") from exc
        if not isinstance(value, dict):
            raise DockerLabError("Docker lab metadata is not an object")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.metadata_path)

    def _network_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        nodes = metadata.get("nodes") or []
        network_id = metadata.get("id")
        expected = ["backend1"] + [
            str(row.get("name")) for row in nodes if isinstance(row, dict)
        ]
        actually_started = (
            isinstance(network_id, int)
            and bool(expected)
            and all(
                self._container_running(self.container_name(network_id, node))
                for node in expected
            )
        )
        status = "Started" if actually_started else "Stopped"
        return {
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "description": metadata.get("description"),
            "status": status,
            "nodes": {
                "bitcoin": [
                    {
                        "name": "backend1",
                        "implementation": "bitcoind",
                        "status": status,
                    }
                ],
                "lightning": [
                    {
                        "name": row["name"],
                        "implementation": row["implementation"],
                        "status": status,
                    }
                    for row in nodes
                    if isinstance(row, dict)
                ],
            },
        }

    def _list_networks(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {"networks": []}
        return {"networks": [self._network_payload(self._load())]}

    def _create_network(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.metadata_path.exists():
            raise DockerLabError("Docker lab metadata already exists for this state path")
        requested = arguments.get("nodes")
        if not isinstance(requested, list):
            raise DockerLabError("create_network nodes must be a list")
        counts = {
            str(row.get("implementation")): _positive(row.get("count"), "node count")
            for row in requested
            if isinstance(row, dict)
        }
        if counts != {"bitcoind": 1, "c-lightning": 15, "LND": 7}:
            raise DockerLabError(f"unexpected Docker lab node counts: {counts}")
        network_id = int(time.time())
        while _run(
            ["docker", "network", "inspect", self.network_name(network_id)],
            check=False,
        ).returncode == 0:
            network_id += 1
        nodes = [
            {"name": f"cln{i}", "implementation": "c-lightning"}
            for i in range(1, 16)
        ] + [
            {"name": f"lnd{i}", "implementation": "LND"}
            for i in range(1, 8)
        ]
        metadata = {
            "schema": "docker-grand-prix-lab-v1",
            "id": network_id,
            "name": str(arguments.get("name") or "revenue-ops-grand-prix"),
            "description": str(arguments.get("description") or ""),
            "status": "Stopped",
            "nodes": nodes,
        }
        self._write(metadata)
        return {"network": self._network_payload(metadata)}

    def _require_id(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], int]:
        metadata = self._load()
        network_id = _positive(arguments.get("networkId"), "network id")
        if metadata.get("id") != network_id:
            raise DockerLabError("Docker lab network id does not match metadata")
        return metadata, network_id

    def _rename_node(self, arguments: dict[str, Any]) -> dict[str, Any]:
        metadata, _network_id = self._require_id(arguments)
        if metadata.get("status") != "Stopped":
            raise DockerLabError("Docker nodes can only be renamed before start")
        old = str(arguments.get("oldName") or "")
        new = str(arguments.get("newName") or "")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", new):
            raise DockerLabError(f"unsafe Docker node name: {new!r}")
        matches = [row for row in metadata["nodes"] if row.get("name") == old]
        if len(matches) != 1 or any(row.get("name") == new for row in metadata["nodes"]):
            raise DockerLabError(f"ambiguous Docker node rename: {old!r} -> {new!r}")
        matches[0]["name"] = new
        self._write(metadata)
        return {"success": True}

    def _labels(self, network_id: int) -> list[str]:
        return [
            "--label", f"{RESOURCE_LABEL}=true",
            "--label", f"{RESOURCE_LABEL}.network-id={network_id}",
        ]

    def _create_volume(self, network_id: int, node_name: str) -> str:
        name = self.container_name(network_id, node_name) + "-data"
        if not SAFE_RESOURCE_RE.fullmatch(name):
            raise DockerLabError(f"unsafe Docker volume name: {name!r}")
        result = _run(["docker", "volume", "inspect", name], check=False)
        if result.returncode != 0:
            _run(["docker", "volume", "create", *self._labels(network_id), name])
        return name

    def _start_network(self, arguments: dict[str, Any]) -> dict[str, Any]:
        metadata, network_id = self._require_id(arguments)
        network = self.network_name(network_id)
        if _run(["docker", "network", "inspect", network], check=False).returncode != 0:
            _run(["docker", "network", "create", *self._labels(network_id), network])
        self._start_bitcoind(network_id)
        self._wait_bitcoind(network_id)
        self._ensure_bitcoin_wallet(network_id)
        if self._block_height(network_id) < 101:
            self._mine(network_id, 101 - self._block_height(network_id))
        for row in metadata["nodes"]:
            self._start_lightning_node(network_id, row)
        metadata["status"] = "Started"
        self._write(metadata)
        return {"success": True, "network": self._network_payload(metadata)}

    def _start_bitcoind(self, network_id: int) -> None:
        container = self.container_name(network_id, "backend1")
        if _run(["docker", "inspect", container], check=False).returncode == 0:
            if not self._container_running(container):
                _run(["docker", "start", container])
            return
        volume = self._create_volume(network_id, "backend1")
        _run([
            "docker", "run", "--detach", "--name", container,
            "--network", self.network_name(network_id), "--network-alias", "backend1",
            *self._labels(network_id), "--volume", f"{volume}:/home/bitcoin/.bitcoin",
            BITCOIND_IMAGE,
            "bitcoind", "-server=1", "-regtest=1", "-rpcuser=labuser",
            "-rpcpassword=labpass", "-debug=1", "-txindex=1", "-dnsseed=0",
            "-rpcbind=0.0.0.0", "-rpcallowip=0.0.0.0/0", "-rpcport=18443",
            "-listen=1", "-listenonion=0", "-fallbackfee=0.0002",
            "-zmqpubrawblock=tcp://0.0.0.0:28334",
            "-zmqpubrawtx=tcp://0.0.0.0:28335",
        ])

    def _start_lightning_node(self, network_id: int, row: dict[str, Any]) -> None:
        node = str(row["name"])
        implementation = str(row["implementation"])
        container = self.container_name(network_id, node)
        if _run(["docker", "inspect", container], check=False).returncode == 0:
            if not self._container_running(container):
                _run(["docker", "start", container])
            return
        volume = self._create_volume(network_id, node)
        common = [
            "docker", "run", "--detach", "--name", container,
            "--network", self.network_name(network_id), "--network-alias", node,
            *self._labels(network_id), "--env", "USERID=1000", "--env", "GROUPID=1000",
        ]
        backend = self.container_name(network_id, "backend1")
        if implementation == "c-lightning":
            _run(common + [
                "--volume", f"{volume}:/home/clightning/.lightning", CLN_IMAGE,
                "lightningd", f"--alias={node}", f"--addr={node}:9735",
                "--network=regtest",
                "--bitcoin-rpcuser=labuser", "--bitcoin-rpcpassword=labpass",
                f"--bitcoin-rpcconnect={backend}", "--bitcoin-rpcport=18443",
                "--log-level=debug", "--dev-bitcoind-poll=2", "--dev-fast-gossip",
                "--log-file=-", "--developer",
            ])
        elif implementation == "LND":
            _run(common + [
                "--volume", f"{volume}:/home/lnd/.lnd", LND_IMAGE,
                "lnd", "--noseedbackup", "--debuglevel=debug", f"--alias={node}",
                f"--externalip={node}", f"--tlsextradomain={node}",
                "--listen=0.0.0.0:9735", "--rpclisten=0.0.0.0:10009",
                "--restlisten=0.0.0.0:8080", "--bitcoin.active", "--bitcoin.regtest",
                "--bitcoin.node=bitcoind", f"--bitcoind.rpchost={backend}",
                "--bitcoind.rpcuser=labuser", "--bitcoind.rpcpass=labpass",
                f"--bitcoind.zmqpubrawblock=tcp://{backend}:28334",
                f"--bitcoind.zmqpubrawtx=tcp://{backend}:28335",
                # CLN's accelerated regtest gossip can emit historical batches
                # while channels are mined in rapid succession. Keep LND
                # passive for historical graph sync in this isolated lab; live
                # peer/channel announcements are still processed normally.
                "--numgraphsyncpeers=0",
                "--accept-keysend", "--accept-amp",
            ])
        else:
            raise DockerLabError(f"unknown implementation: {implementation!r}")

    @staticmethod
    def _container_running(container: str) -> bool:
        result = _run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container],
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _bitcoin(self, network_id: int, *arguments: str) -> subprocess.CompletedProcess[str]:
        return _run([
            "docker", "exec", self.container_name(network_id, "backend1"),
            "bitcoin-cli", "-regtest", "-rpcuser=labuser", "-rpcpassword=labpass",
            *arguments,
        ])

    def _wait_bitcoind(self, network_id: int) -> None:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = _run([
                "docker", "exec", self.container_name(network_id, "backend1"),
                "bitcoin-cli", "-regtest", "-rpcuser=labuser",
                "-rpcpassword=labpass", "getblockchaininfo",
            ], check=False)
            if result.returncode == 0:
                return
            time.sleep(1)
        raise DockerLabError("Docker bitcoind did not become ready")

    def _block_height(self, network_id: int) -> int:
        return int(self._bitcoin(network_id, "getblockcount").stdout.strip())

    def _ensure_bitcoin_wallet(self, network_id: int) -> None:
        try:
            wallets = json.loads(self._bitcoin(network_id, "listwallets").stdout)
        except json.JSONDecodeError as exc:
            raise DockerLabError("bitcoind returned malformed wallet inventory") from exc
        if not isinstance(wallets, list):
            raise DockerLabError("bitcoind wallet inventory is not a list")
        if not wallets:
            self._bitcoin(network_id, "createwallet", "grand-prix")

    def _mine(self, network_id: int, blocks: int) -> None:
        address = self._bitcoin(network_id, "getnewaddress").stdout.strip()
        if not address.startswith("bcrt1"):
            raise DockerLabError("bitcoind returned no regtest mining address")
        self._bitcoin(network_id, "generatetoaddress", str(blocks), address)

    def _node(self, metadata: dict[str, Any], name: str) -> dict[str, Any]:
        matches = [row for row in metadata["nodes"] if row.get("name") == name]
        if len(matches) != 1:
            raise DockerLabError(f"unknown or ambiguous Docker node: {name!r}")
        return matches[0]

    def _cln(self, network_id: int, node: str, *arguments: str) -> dict[str, Any]:
        return _json_command([
            "docker", "exec", "-u", "clightning", self.container_name(network_id, node),
            "lightning-cli", "--network=regtest", "--notifications=none", *arguments,
        ])

    def _lnd(self, network_id: int, node: str, *arguments: str) -> dict[str, Any]:
        return _json_command([
            "docker", "exec", "-u", "lnd", self.container_name(network_id, node),
            "lncli", "--network=regtest", *arguments,
        ])

    def _wait_node(self, network_id: int, row: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + 180
        last = "not ready"
        while time.monotonic() < deadline:
            try:
                if row["implementation"] == "c-lightning":
                    result = self._cln(network_id, row["name"], "getinfo")
                    if result.get("id"):
                        return result
                else:
                    result = self._lnd(network_id, row["name"], "getinfo")
                    if result.get("identity_pubkey"):
                        return result
                last = "RPC returned no node id"
            except DockerLabError as exc:
                last = str(exc)
            time.sleep(1)
        raise DockerLabError(f"{row['name']} did not become ready: {last}")

    def _get_node_info(self, arguments: dict[str, Any]) -> dict[str, Any]:
        metadata, network_id = self._require_id(arguments)
        row = self._node(metadata, str(arguments.get("nodeName") or ""))
        info = self._wait_node(network_id, row)
        pubkey = info.get("id") if row["implementation"] == "c-lightning" else info.get("identity_pubkey")
        return {"info": {"pubkey": pubkey}, "node": row}

    def _list_channels(self, arguments: dict[str, Any]) -> dict[str, Any]:
        metadata, network_id = self._require_id(arguments)
        row = self._node(metadata, str(arguments.get("nodeName") or ""))
        if row["implementation"] == "c-lightning":
            channels = self._cln(network_id, row["name"], "listpeerchannels").get("channels")
            normalized = [
                {
                    "pubkey": channel.get("peer_id"),
                    "capacity": str(_msat(channel.get("total_msat")) // 1000),
                    "status": "Open" if channel.get("state") == "CHANNELD_NORMAL" else "Opening",
                }
                for channel in channels or [] if isinstance(channel, dict)
            ]
        else:
            channels = self._lnd(network_id, row["name"], "listchannels").get("channels")
            normalized = [
                {
                    "pubkey": channel.get("remote_pubkey"),
                    "capacity": str(channel.get("capacity")),
                    "status": "Open" if channel.get("active") is True else "Opening",
                }
                for channel in channels or [] if isinstance(channel, dict)
            ]
        return {"channels": normalized}

    def _fund_wallet(
        self, network_id: int, row: dict[str, Any], required_sats: int
    ) -> None:
        balance = self._confirmed_wallet_balance(network_id, row)
        if balance >= required_sats:
            return
        if row["implementation"] == "c-lightning":
            address_payload = self._cln(network_id, row["name"], "newaddr")
            address = address_payload.get("bech32") or address_payload.get("p2tr")
        else:
            address = self._lnd(network_id, row["name"], "newaddress", "p2wkh").get("address")
        if not isinstance(address, str) or not address.startswith("bcrt1"):
            raise DockerLabError(f"{row['name']} returned no regtest funding address")
        amount = required_sats - balance + 500_000
        self._bitcoin(network_id, "sendtoaddress", address, f"{amount / 100_000_000:.8f}")
        self._mine(network_id, 6)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            balance = self._confirmed_wallet_balance(network_id, row)
            if balance >= required_sats:
                return
            time.sleep(1)
        raise DockerLabError(
            f"{row['name']} wallet has {balance} confirmed sats; need {required_sats}"
        )

    def _confirmed_wallet_balance(
        self, network_id: int, row: dict[str, Any]
    ) -> int:
        if row["implementation"] == "c-lightning":
            outputs = self._cln(
                network_id, row["name"], "listfunds"
            ).get("outputs") or []
            if not isinstance(outputs, list):
                raise DockerLabError(f"{row['name']} returned malformed wallet outputs")
            return sum(
                _msat(item.get("amount_msat")) // 1000
                for item in outputs
                if isinstance(item, dict) and item.get("status") == "confirmed"
            )
        payload = self._lnd(network_id, row["name"], "walletbalance")
        try:
            return int(payload.get("confirmed_balance") or 0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DockerLabError(
                f"{row['name']} returned malformed confirmed balance"
            ) from exc

    def _connect(
        self,
        network_id: int,
        source: dict[str, Any],
        destination: dict[str, Any],
        destination_pubkey: str,
    ) -> None:
        if source["implementation"] == "c-lightning":
            command = [
                "docker", "exec", "-u", "clightning",
                self.container_name(network_id, source["name"]), "lightning-cli",
                "--network=regtest", "--notifications=none", "connect",
                destination_pubkey, destination["name"], "9735",
            ]
        else:
            command = [
                "docker", "exec", "-u", "lnd",
                self.container_name(network_id, source["name"]), "lncli",
                "--network=regtest", "connect",
                f"{destination_pubkey}@{destination['name']}:9735",
            ]
        deadline = time.monotonic() + 90
        last = "not attempted"
        while time.monotonic() < deadline:
            result = _run(command, check=False)
            detail = (result.stderr or result.stdout or "").casefold()
            last = detail.strip() or f"exit {result.returncode}"
            if result.returncode == 0 or "already connected" in detail:
                if self._peer_connected(
                    network_id, source, destination_pubkey
                ):
                    return
            time.sleep(1)
        raise DockerLabError(f"peer did not become online: {last}")

    def _peer_connected(
        self, network_id: int, source: dict[str, Any], peer_id: str
    ) -> bool:
        try:
            if source["implementation"] == "c-lightning":
                rows = self._cln(
                    network_id, source["name"], "listpeers", peer_id
                ).get("peers")
                return bool(
                    isinstance(rows, list)
                    and len(rows) == 1
                    and isinstance(rows[0], dict)
                    and rows[0].get("connected") is True
                )
            rows = self._lnd(
                network_id, source["name"], "listpeers"
            ).get("peers")
            return bool(
                isinstance(rows, list)
                and any(
                    isinstance(row, dict) and row.get("pub_key") == peer_id
                    for row in rows
                )
            )
        except DockerLabError:
            return False

    def _open_channel(self, arguments: dict[str, Any]) -> dict[str, Any]:
        metadata, network_id = self._require_id(arguments)
        source = self._node(metadata, str(arguments.get("fromNode") or ""))
        destination = self._node(metadata, str(arguments.get("toNode") or ""))
        capacity = _positive(arguments.get("sats"), "channel capacity")
        destination_info = self._wait_node(network_id, destination)
        destination_pubkey = (
            destination_info.get("id")
            if destination["implementation"] == "c-lightning"
            else destination_info.get("identity_pubkey")
        )
        if not isinstance(destination_pubkey, str) or not destination_pubkey:
            raise DockerLabError("destination returned no public key")
        self._fund_wallet(network_id, source, capacity + 100_000)
        if source["implementation"] == "c-lightning":
            self._connect(network_id, source, destination, destination_pubkey)
            result = self._cln(
                network_id, source["name"], "-k", "fundchannel",
                f"id={destination_pubkey}", f"amount={capacity}",
                "announce=true", "minconf=1",
            )
        else:
            connect_argument: list[str] = [
                f"--connect={destination['name']}:9735"
            ]
            if destination["implementation"] == "c-lightning":
                source_info = self._wait_node(network_id, source)
                source_pubkey = source_info.get("identity_pubkey")
                if not isinstance(source_pubkey, str) or not source_pubkey:
                    raise DockerLabError("LND source returned no public key")
                self._connect(
                    network_id, destination, source, source_pubkey
                )
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if self._peer_connected(
                        network_id, source, destination_pubkey
                    ):
                        break
                    time.sleep(0.1)
                else:
                    raise DockerLabError(
                        "reverse-initiated LND peer did not become online"
                    )
                connect_argument = []
            result = self._lnd(
                network_id, source["name"], "openchannel",
                f"--node_key={destination_pubkey}", f"--local_amt={capacity}",
                *connect_argument,
                "--private=false", "--min_confs=1",
            )
        self._mine(network_id, 6)
        return {"success": True, "channel": result}

    def _mine_blocks(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _metadata, network_id = self._require_id(arguments)
        blocks = _positive(arguments.get("blocks"), "block count")
        self._mine(network_id, blocks)
        return {"success": True, "blocks": blocks}

    def _labeled_names(self, kind: str, network_id: int) -> list[str]:
        command = ["docker", kind, "ls"]
        if kind == "container":
            command.extend(["--all"])
        command.extend([
            "--filter", f"label={RESOURCE_LABEL}.network-id={network_id}",
            "--format", "{{.Name}}" if kind == "volume" else "{{.Names}}",
        ])
        if kind == "network":
            command[-1] = "{{.Name}}"
        rows = [row.strip() for row in _run(command).stdout.splitlines() if row.strip()]
        for row in rows:
            if not SAFE_RESOURCE_RE.fullmatch(row):
                raise DockerLabError(f"refusing cleanup of unscoped {kind}: {row!r}")
        return rows

    def _stop_network(self, arguments: dict[str, Any]) -> dict[str, Any]:
        metadata, network_id = self._require_id(arguments)
        containers = self._labeled_names("container", network_id)
        if containers:
            _run(["docker", "rm", "-f", *containers])
        networks = self._labeled_names("network", network_id)
        if networks:
            _run(["docker", "network", "rm", *networks])
        volumes = self._labeled_names("volume", network_id)
        if volumes:
            _run(["docker", "volume", "rm", *volumes])
        metadata["status"] = "Stopped"
        metadata["removed_resources"] = {
            "containers": containers,
            "networks": networks,
            "volumes": volumes,
        }
        self._write(metadata)
        return {"success": True, "network": self._network_payload(metadata)}
