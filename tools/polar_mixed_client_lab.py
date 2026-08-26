#!/usr/bin/env python3
"""Provision and drive the isolated mixed-client Polar revenue-ops lab.

The script talks only to Polar's documented localhost MCP bridge.  It creates
the topology and sends deterministic invoices/payments; it deliberately does
not start the revenue plugin, alter fee policy, or perform a rebalance.  Those
are separately approval-gated experiment steps documented in
docs/optimization/plans/2026-08-26-polar-mixed-client-lab.md.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BRIDGE_URL = "http://127.0.0.1:37373"
DEFAULT_LAB_NAME = "revenue-ops-mixed-client"
DEFAULT_REVERSE_FEE_BUFFER_SATS = 25_000


@dataclass(frozen=True)
class ChannelSpec:
    """A funded public channel in the parallel-route laboratory."""

    source: str
    destination: str
    capacity_sats: int = 2_000_000


# The funder direction deliberately leaves local liquidity on revenue-node's
# payer channels and remote liquidity on its sink channels.  The resulting
# circle through a competitor is usable for a real rebalance experiment.
CHANNELS = (
    ChannelSpec("lnd-payer", "revenue-node"),
    ChannelSpec("lnd-payer", "cln-competitor"),
    ChannelSpec("lnd-payer", "lnd-competitor"),
    ChannelSpec("cln-payer", "revenue-node"),
    ChannelSpec("cln-payer", "cln-competitor"),
    ChannelSpec("cln-payer", "lnd-competitor"),
    ChannelSpec("revenue-node", "lnd-sink"),
    ChannelSpec("cln-competitor", "lnd-sink"),
    ChannelSpec("lnd-competitor", "lnd-sink"),
    ChannelSpec("revenue-node", "cln-sink"),
    ChannelSpec("cln-competitor", "cln-sink"),
    ChannelSpec("lnd-competitor", "cln-sink"),
)

ROLE_NAMES = {
    "c-lightning": ("revenue-node", "cln-competitor", "cln-payer", "cln-sink"),
    "LND": ("lnd-competitor", "lnd-payer", "lnd-sink"),
}

FORWARD_TRAFFIC_LANES = (
    ("lnd-payer", "lnd-sink"),
    ("cln-payer", "cln-sink"),
)

REVERSE_TRAFFIC_LANES = tuple((sink, payer) for payer, sink in FORWARD_TRAFFIC_LANES)

# Backward-compatible name for callers that imported the original lane set.
TRAFFIC_LANES = FORWARD_TRAFFIC_LANES


class PolarMcpError(RuntimeError):
    """A local Polar MCP bridge call did not succeed."""


class PolarTrafficError(PolarMcpError):
    """A payment RPC failed after dispatch, so settlement is unknowable."""

    def __init__(
        self,
        message: str,
        *,
        completed_records: list[dict[str, Any]],
        uncertain_operation: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.completed_records = completed_records
        self.uncertain_operation = uncertain_operation


class PolarMcp:
    def __init__(self, base_url: str = DEFAULT_BRIDGE_URL, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/mcp/execute", {"tool": tool, "arguments": arguments})

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except OSError:
                detail = ""
            raise PolarMcpError(
                f"Polar MCP {method} {path} failed: HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PolarMcpError(f"Polar MCP {method} {path} failed: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            raise PolarMcpError(f"Polar MCP {method} {path} returned: {payload}")
        return payload


def role_renames(network: dict[str, Any]) -> list[tuple[str, str]]:
    """Map Polar's generated names to stable experiment roles."""
    lightning = network.get("nodes", {}).get("lightning", [])
    if not isinstance(lightning, list):
        raise PolarMcpError("created network does not contain a lightning-node list")
    by_implementation: dict[str, list[str]] = {key: [] for key in ROLE_NAMES}
    for node in lightning:
        if isinstance(node, dict) and node.get("implementation") in by_implementation:
            by_implementation[str(node["implementation"])].append(str(node.get("name") or ""))

    renames: list[tuple[str, str]] = []
    for implementation, expected_names in ROLE_NAMES.items():
        actual_names = by_implementation[implementation]
        if len(actual_names) != len(expected_names) or any(not name for name in actual_names):
            raise PolarMcpError(
                f"expected {len(expected_names)} {implementation} nodes, got {actual_names!r}"
            )
        renames.extend(zip(actual_names, expected_names, strict=True))
    return renames


def create_lab(bridge: PolarMcp, name: str, description: str) -> int:
    """Create the stopped lab and assign deterministic node names."""
    existing = bridge.call("list_networks", {}).get("networks", [])
    if not isinstance(existing, list):
        raise PolarMcpError("list_networks did not return a network list")
    if any(isinstance(network, dict) and network.get("name") == name for network in existing):
        raise PolarMcpError(f'network "{name}" already exists; choose a new --name')
    running = [
        network
        for network in existing
        if isinstance(network, dict)
        and str(network.get("status", "")).casefold() in {"started", "starting"}
    ]
    if running:
        labels = ", ".join(
            f'{network.get("name", "unnamed")} (id={network.get("id", "unknown")})'
            for network in running
        )
        raise PolarMcpError(
            "cannot create a fresh lab while another Polar network is running: "
            f"{labels}; stop it first or reuse it with --network-id"
        )
    result = bridge.call(
        "create_network",
        {
            "name": name,
            "description": description,
            "nodes": [
                {"implementation": "bitcoind", "count": 1},
                {"implementation": "c-lightning", "count": 4},
                {"implementation": "LND", "count": 3},
            ],
        },
    )
    network = result.get("network")
    if not isinstance(network, dict) or not isinstance(network.get("id"), int):
        raise PolarMcpError(f"create_network did not return a network id: {result!r}")
    network_id = network["id"]
    for old_name, new_name in role_renames(network):
        if old_name != new_name:
            bridge.call("rename_node", {"networkId": network_id, "oldName": old_name, "newName": new_name})
    return network_id


def wait_for_lightning_nodes(bridge: PolarMcp, network_id: int, timeout_seconds: float = 180.0) -> None:
    """Wait for every target node's own RPC endpoint, not just Docker startup."""
    deadline = time.monotonic() + timeout_seconds
    expected = [name for names in ROLE_NAMES.values() for name in names]
    last_error = "nodes have not responded yet"
    while time.monotonic() < deadline:
        try:
            for node_name in expected:
                bridge.call("get_node_info", {"networkId": network_id, "nodeName": node_name})
            return
        except PolarMcpError as exc:
            last_error = str(exc)
            time.sleep(3)
    raise PolarMcpError(f"Polar nodes were not RPC-ready within {timeout_seconds:.0f}s: {last_error}")


def has_required_channel(
    channels: list[dict[str, Any]], destination_pubkey: str, capacity_sats: int
) -> bool:
    """Whether a usable or pending channel already satisfies a lab edge."""
    return any(
        channel.get("pubkey") == destination_pubkey
        and str(channel.get("capacity")) == str(capacity_sats)
        and channel.get("status") in {"Open", "Opening"}
        for channel in channels
        if isinstance(channel, dict)
    )


def start_and_wire_lab(bridge: PolarMcp, network_id: int) -> list[dict[str, Any]]:
    """Start an already-created lab, wire its public parallel routes, and confirm them."""
    # Polar's bridge limits tool responses to 30 seconds. Image pulls can take
    # longer even when Docker has successfully started the network, so the RPC
    # readiness check below is authoritative rather than the initial response.
    try:
        bridge.call("start_network", {"networkId": network_id})
    except PolarMcpError as exc:
        # Image pulls can exceed the bridge response timeout; do not conceal
        # configuration or API errors that the readiness probe cannot explain.
        if "timed out" not in str(exc).lower():
            raise
    return wire_lab(bridge, network_id)


def wire_lab(bridge: PolarMcp, network_id: int) -> list[dict[str, Any]]:
    """Wire a running lab created in an earlier MCP session."""
    wait_for_lightning_nodes(bridge, network_id)
    opened = []
    for channel in CHANNELS:
        destination = bridge.call(
            "get_node_info", {"networkId": network_id, "nodeName": channel.destination}
        )
        destination_info = destination.get("info", {})
        destination_pubkey = destination_info.get("pubkey") if isinstance(destination_info, dict) else None
        source_channels = bridge.call(
            "list_channels", {"networkId": network_id, "nodeName": channel.source}
        ).get("channels", [])
        if not isinstance(destination_pubkey, str) or not isinstance(source_channels, list):
            raise PolarMcpError(f"could not inspect existing edge {channel.source}->{channel.destination}")
        if has_required_channel(source_channels, destination_pubkey, channel.capacity_sats):
            opened.append(
                {
                    "skipped": True,
                    "reason": "required channel already exists",
                    "fromNode": channel.source,
                    "toNode": channel.destination,
                }
            )
            continue
        try:
            opened.append(
                bridge.call(
                    "open_channel",
                    {
                        "networkId": network_id,
                        "fromNode": channel.source,
                        "toNode": channel.destination,
                        "sats": channel.capacity_sats,
                        "isPrivate": False,
                        "autoFund": True,
                    },
                )
            )
        except PolarMcpError as exc:
            # Some mixed-client opens complete after Polar's REST bridge drops
            # its socket. Probe the source once before declaring the funding
            # operation unknown; this makes retries safe without hiding a
            # genuinely failed open.
            time.sleep(2)
            after_error = bridge.call(
                "list_channels", {"networkId": network_id, "nodeName": channel.source}
            ).get("channels", [])
            if not isinstance(after_error, list) or not has_required_channel(
                after_error, destination_pubkey, channel.capacity_sats
            ):
                raise exc
            opened.append(
                {
                    "reconciled_after_mcp_error": True,
                    "reason": str(exc),
                    "fromNode": channel.source,
                    "toNode": channel.destination,
                }
            )
        # Confirm each funding transaction before the next edge. Polar's
        # heterogeneous nodes otherwise race their own wallet/peer restarts.
        bridge.call("mine_blocks", {"networkId": network_id, "blocks": 6})
    return opened


def run_deterministic_traffic(
    bridge: PolarMcp,
    network_id: int,
    rounds: int,
    amount_sats: int,
    pause_seconds: float,
    lanes: tuple[tuple[str, str], ...] = TRAFFIC_LANES,
) -> list[dict[str, Any]]:
    """Send deterministic traffic over one or more competing route lanes."""
    if rounds <= 0 or amount_sats <= 0:
        raise ValueError("rounds and amount_sats must be positive")
    if not lanes:
        raise ValueError("at least one traffic lane is required")
    records: list[dict[str, Any]] = []
    for round_index in range(rounds):
        for payer, sink in lanes:
            invoice = bridge.call(
                "create_invoice",
                {
                    "networkId": network_id,
                    "nodeName": sink,
                    "amount": amount_sats,
                    "memo": f"revenue-ops-lab-{round_index}-{payer}",
                },
            )
            try:
                payment = bridge.call(
                    "pay_invoice",
                    {
                        "networkId": network_id,
                        "fromNode": payer,
                        "invoice": invoice["invoice"],
                    },
                )
            except PolarMcpError as exc:
                # Polar can dispatch the client payment successfully and then
                # fail while serializing the bridge response (observed with a
                # missing UI activeId). Never retry an ambiguous money-path
                # operation. Preserve a credential-free checkpoint instead.
                uncertain = {
                    "round": round_index,
                    "payer": payer,
                    "sink": sink,
                    "amount_sats": amount_sats,
                    "payment_outcome": "unknown_do_not_retry",
                    "invoice_created": True,
                    "error": str(exc),
                }
                raise PolarTrafficError(
                    "Polar payment result is unknown; do not retry without "
                    "reconciling client and router histories",
                    completed_records=list(records),
                    uncertain_operation=uncertain,
                ) from exc
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


def select_traffic_lanes(direction: str, lane: str) -> tuple[tuple[str, str], ...]:
    """Resolve a CLI direction/lane selector into deterministic endpoint pairs."""
    lane_sets = {
        "forward": FORWARD_TRAFFIC_LANES,
        "reverse": REVERSE_TRAFFIC_LANES,
    }
    directions = ("forward", "reverse") if direction == "both" else (direction,)
    selected = tuple(pair for name in directions for pair in lane_sets[name])
    if lane == "all":
        return selected
    marker = f"{lane}-"
    return tuple(pair for pair in selected if pair[0].startswith(marker))


def traffic_batches(
    direction: str,
    lane: str,
    amount_sats: int,
    reverse_fee_buffer_sats: int,
) -> tuple[tuple[tuple[tuple[str, str], ...], int], ...]:
    """Build executable traffic batches, seeding return liquidity when needed."""
    if amount_sats <= 0:
        raise ValueError("amount_sats must be positive")
    if reverse_fee_buffer_sats < 0:
        raise ValueError("reverse_fee_buffer_sats must be nonnegative")
    if direction != "both":
        return ((select_traffic_lanes(direction, lane), amount_sats),)
    # A sink cannot return the exact amount it just received because its
    # outgoing payment must cover routing fees while leaving the channel
    # reserve intact. Seed an explicit surplus on the forward pass so a fresh
    # topology can run both directions.
    return (
        (select_traffic_lanes("forward", lane), amount_sats + reverse_fee_buffer_sats),
        (select_traffic_lanes("reverse", lane), amount_sats),
    )


def emit_result(result: dict[str, Any], output: Path | None) -> None:
    """Write and print one deterministic experiment result."""
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_LAB_NAME)
    parser.add_argument("--bridge-url", default=DEFAULT_BRIDGE_URL)
    parser.add_argument("--apply", action="store_true", help="create and mutate a local Polar test network")
    parser.add_argument(
        "--network-id",
        type=int,
        help="wire and drive an already-started Polar lab instead of creating a new one",
    )
    parser.add_argument("--traffic-rounds", type=int, default=0)
    parser.add_argument("--amount-sats", type=int, default=20_000)
    parser.add_argument(
        "--reverse-fee-buffer-sats",
        type=int,
        default=DEFAULT_REVERSE_FEE_BUFFER_SATS,
        help="extra forward liquidity for fees/reserve when --traffic-direction=both",
    )
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument(
        "--traffic-direction",
        choices=("forward", "reverse", "both"),
        default="forward",
        help="payment direction relative to the payer-to-sink baseline",
    )
    parser.add_argument(
        "--traffic-lane",
        choices=("all", "lnd", "cln"),
        default="all",
        help="drive both mixed-client lanes or only one endpoint family",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = {
        "name": args.name,
        "node_counts": {"bitcoind": 1, "c-lightning": 4, "LND": 3},
        "roles": ROLE_NAMES,
        "channels": [channel.__dict__ for channel in CHANNELS],
        "traffic_lanes": {
            "forward": FORWARD_TRAFFIC_LANES,
            "reverse": REVERSE_TRAFFIC_LANES,
        },
        "polar_simulation_note": (
            "Polar v4.0.0 exposes simulation activity rules in its UI, not its MCP tool list. "
            "Use the UI for long-running sim-ln background rules; this tool drives deterministic MCP traffic. "
            "Eclair 0.13.1 is excluded from this host's lab after a reproducible native crash during funding."
        ),
    }
    if args.apply:
        bridge = PolarMcp(args.bridge_url)
        bridge.health()
        network_id = args.network_id or create_lab(
            bridge, args.name, "Mixed LND/CLN fee and rebalance experiment lab for cl-revenue-ops."
        )
        plan["network_id"] = network_id
        plan["opened_channels"] = (
            wire_lab(bridge, network_id) if args.network_id else start_and_wire_lab(bridge, network_id)
        )
        if args.traffic_rounds:
            plan["traffic"] = []
            for selected_lanes, amount_sats in traffic_batches(
                args.traffic_direction,
                args.traffic_lane,
                args.amount_sats,
                args.reverse_fee_buffer_sats,
            ):
                try:
                    batch_records = run_deterministic_traffic(
                        bridge,
                        network_id,
                        args.traffic_rounds,
                        amount_sats,
                        args.pause_seconds,
                        selected_lanes,
                    )
                except PolarTrafficError as exc:
                    plan["traffic"].extend(exc.completed_records)
                    plan["traffic"].append(exc.uncertain_operation)
                    plan["status"] = "traffic_outcome_unknown"
                    plan["required_operator_action"] = (
                        "Reconcile payer payment and router forwarding histories; "
                        "do not retry the uncertain payment."
                    )
                    emit_result(plan, args.output)
                    return 2
                plan["traffic"].extend(batch_records)
    emit_result(plan, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
