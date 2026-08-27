#!/usr/bin/env python3
"""Run a bounded, restartable fee sweep and active Polar endurance campaign.

This tool is intentionally confined to an existing mixed-client Polar lab. It
drives invoices through Polar's localhost MCP bridge and uses docker exec only
for the named revenue-node plugin and its local channel policies. Every action
phase is checkpointed and a finally block restores the original policies,
pause, zero daily budget, and dry-run mode.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from polar_mixed_client_lab import (
    PolarMcp,
    PolarTrafficError,
    run_deterministic_traffic,
    select_traffic_lanes,
    wait_for_traffic_ready,
)
from polar_soak_scorecard import capture_snapshot


class CampaignError(RuntimeError):
    """The campaign could not preserve a required safety invariant."""


def positive_network_id(value: int | str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("network id must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise argparse.ArgumentTypeError("network id must be a positive integer")
    return parsed


def csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive")
    return parsed


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignError(f"non-JSON command response: {command!r}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"non-object command response: {command!r}")
    if payload.get("error") or payload.get("code"):
        raise CampaignError(f"command rejected: {command!r}: {payload}")
    return payload


@dataclass(frozen=True)
class ChannelPolicy:
    channel_id: str
    fee_base_msat: int
    fee_ppm: int


class RevenueNode:
    """Strictly scoped command adapter for one Polar revenue container."""

    def __init__(self, network_id: int):
        self.network_id = positive_network_id(network_id)
        self.container = f"polar-n{self.network_id}-revenue-node"
        self.prefix = [
            "docker", "exec", "-u", "clightning", self.container,
            "lightning-cli", "--network=regtest",
        ]
        self.plugin_path = "/opt/cl_revenue_ops/cl-revenue-ops-polar-wrapper"

    def rpc(self, *arguments: object) -> dict[str, Any]:
        return _run_json(self.prefix + [str(argument) for argument in arguments])

    def policies(self) -> tuple[ChannelPolicy, ...]:
        rows = self.rpc("listpeerchannels").get("channels")
        if not isinstance(rows, list) or not rows:
            raise CampaignError("revenue node has no peer channels")
        policies: list[ChannelPolicy] = []
        for row in rows:
            if not isinstance(row, dict):
                raise CampaignError("malformed listpeerchannels row")
            channel_id = row.get("short_channel_id")
            update = (row.get("updates") or {}).get("local")
            if not isinstance(channel_id, str) or not isinstance(update, dict):
                raise CampaignError("channel lacks an active local policy")
            policies.append(
                ChannelPolicy(
                    channel_id=channel_id,
                    fee_base_msat=int(update["fee_base_msat"]),
                    fee_ppm=int(update["fee_proportional_millionths"]),
                )
            )
        return tuple(sorted(policies, key=lambda item: item.channel_id))

    def set_policies(self, policies: Iterable[ChannelPolicy]) -> None:
        for policy in policies:
            response = self.rpc(
                "setchannel", policy.channel_id,
                policy.fee_base_msat, policy.fee_ppm,
            )
            channels = response.get("channels")
            if not isinstance(channels, list) or not channels:
                raise CampaignError(f"setchannel did not confirm {policy.channel_id}")

    def set_fee_ppm(self, fee_ppm: int, baseline: Iterable[ChannelPolicy]) -> None:
        if fee_ppm < 0:
            raise CampaignError("fee ppm must be nonnegative")
        self.set_policies(
            ChannelPolicy(policy.channel_id, policy.fee_base_msat, fee_ppm)
            for policy in baseline
        )

    def set_config(self, key: str, value: object) -> dict[str, Any]:
        response = self.rpc("revenue-config", "set", key, str(value).lower())
        if response.get("status") not in {"success", "ok"}:
            raise CampaignError(f"revenue-config set {key} failed: {response}")
        return response

    def dry_run(self) -> bool:
        response = self.rpc("listconfigs", "revenue-ops-dry-run")
        value = response.get("configs", {}).get("revenue-ops-dry-run", {})
        if "value_bool" in value:
            return value["value_bool"] is True
        rendered = str(value.get("value_str", "")).casefold()
        if rendered not in {"true", "false"}:
            raise CampaignError(f"dry-run readback is malformed: {response}")
        return rendered == "true"

    def set_dry_run(self, enabled: bool) -> dict[str, Any]:
        """Restart only the dynamic plugin because dry-run is startup-only."""
        try:
            if self.dry_run() is enabled:
                return {"status": "unchanged", "dry_run": enabled}
        except CampaignError:
            # A previous interrupted restart may have left the plugin absent;
            # the start below is the recovery path.
            pass
        try:
            _run_json(
                self.prefix + [
                    "-k", "plugin", "subcommand=stop",
                    f"plugin={self.plugin_path}",
                ]
            )
        except CampaignError as exc:
            if "not found" not in str(exc).casefold() and "not running" not in str(exc).casefold():
                raise
        response = _run_json(
            self.prefix + [
                "-k", "plugin", "subcommand=start",
                f"plugin={self.plugin_path}",
                f"revenue-ops-dry-run={str(enabled).lower()}",
                "revenue-ops-daily-budget-sats=0",
            ]
        )
        deadline = time.monotonic() + 30
        last_error = "plugin did not answer"
        while time.monotonic() < deadline:
            try:
                status = self.rpc("revenue-status")
                if status.get("status") == "running" and self.dry_run() is enabled:
                    return response
                last_error = f"unexpected status: {status}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise CampaignError(f"plugin restart readback failed: {last_error}")

    def cycle(self, subsystem: str) -> dict[str, Any]:
        return self.rpc("revenue-cycle", subsystem)


def router_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for router in ("revenue-node", "cln-competitor", "lnd-competitor"):
        old = before.get("routers", {}).get(router)
        new = after.get("routers", {}).get(router)
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise CampaignError(f"missing router totals for {router}")
        delta = {
            key: int(new.get(key, 0)) - int(old.get(key, 0))
            for key in ("settled_count", "fee_msat", "volume_msat")
        }
        if any(value < 0 for value in delta.values()):
            raise CampaignError(f"router counters regressed for {router}")
        result[router] = delta
    return result


def phase_summary(
    family: str,
    fee_ppm: int,
    records: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    successes = [
        row for row in records
        if isinstance(row.get("payment"), dict) and row["payment"].get("success") is True
    ]
    deltas = router_delta(before, after)
    forwards = sum(row["settled_count"] for row in deltas.values())
    revenue_forwards = deltas["revenue-node"]["settled_count"]
    return {
        "family": family,
        "fee_ppm": fee_ppm,
        "attempted": len(records),
        "settled": len(successes),
        "router_delta": deltas,
        "revenue_route_share": round(revenue_forwards / forwards, 6) if forwards else 0.0,
        "revenue_fee_msat": deltas["revenue-node"]["fee_msat"],
        "passed": len(successes) == len(records) and forwards >= len(successes),
    }


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def send_cell(
    bridge: PolarMcp,
    network_id: int,
    family: str,
    amounts: tuple[int, ...],
    rounds: int,
    pause_seconds: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for amount in amounts:
        for direction in ("forward", "reverse"):
            records.extend(
                run_deterministic_traffic(
                    bridge,
                    network_id,
                    rounds,
                    amount,
                    pause_seconds,
                    select_traffic_lanes(direction, family),
                    invoice_retries=2,
                )
            )
    return records


def safe_final_state(snapshot: dict[str, Any], *, dry_run: bool) -> bool:
    health = snapshot.get("module_health", {})
    config = health.get("config", {}).get("config", {})
    return (
        dry_run is True
        and config.get("paused") is True
        and config.get("daily_budget_sats") == 0
        and health.get("budget", {}).get("reserved_24h_sats") == 0
        and health.get("econ_reconcile", {}).get("divergences") == []
    )


def enable_live_controller(node: RevenueNode, live_budget_sats: int) -> None:
    """Enable actions without letting the startup-only restart erase the budget."""
    node.set_dry_run(False)
    node.set_config("daily_budget_sats", live_budget_sats)
    node.set_config("paused", False)


def enable_live_controller_if_needed(
    node: RevenueNode,
    live_budget_sats: int,
    completed_epochs: int,
    invocation_limit: int,
) -> bool:
    """Enter live mode only when this invocation will execute an epoch."""
    if completed_epochs >= invocation_limit:
        return False
    enable_live_controller(node, live_budget_sats)
    return True


def endurance_epoch_limit(completed: int, target: int, max_new: int) -> int:
    """Return the exclusive epoch limit for this resumable invocation."""
    if min(completed, target, max_new) < 0:
        raise CampaignError("endurance epoch counts must be nonnegative")
    return target if max_new == 0 else min(target, completed + max_new)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    if not args.resume and args.output.exists():
        raise CampaignError("output already exists; use --resume or choose a new path")
    bridge = PolarMcp(args.bridge_url)
    bridge.health()
    if args.mine_preflight_block:
        bridge.call("mine_blocks", {"networkId": args.network_id, "blocks": 1})
    wait_for_traffic_ready(bridge, args.network_id)
    node = RevenueNode(args.network_id)
    current_policies = node.policies()
    if args.resume:
        try:
            result = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(f"cannot resume checkpoint: {exc}") from exc
        if (
            not isinstance(result, dict)
            or result.get("schema") != "polar-endurance-campaign-v1"
            or result.get("network_id") != args.network_id
        ):
            raise CampaignError("checkpoint does not match this campaign/network")
        baseline_policies = tuple(
            ChannelPolicy(**row) for row in result.get("baseline_policies", [])
        )
        if not baseline_policies:
            raise CampaignError("checkpoint has no baseline policies")
        result["resume_count"] = int(result.get("resume_count", 0)) + 1
        result["cleanup"] = {"attempted": False, "errors": []}
        result.pop("final_snapshot", None)
        result.pop("finished_at", None)
        result.pop("ambiguous_payment", None)
        result.pop("completed_before_ambiguity", None)
    else:
        baseline_policies = current_policies
        result = {
            "schema": "polar-endurance-campaign-v1",
            "network_id": args.network_id,
            "started_at": int(time.time()),
            "baseline_policies": [policy.__dict__ for policy in baseline_policies],
            "fee_phases": [],
            "endurance_epochs": [],
            "cleanup": {"attempted": False, "errors": []},
        }
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        # Fee sweep is observational at the plugin layer: keep the plugin
        # paused/dry-run and alter only this disposable node's advertised fee.
        node.set_config("paused", True)
        node.set_config("daily_budget_sats", 0)
        node.set_dry_run(True)
        phase_cursor = len(result["fee_phases"])
        phase_index = 0
        for fee_ppm in args.fee_ppms:
            if stop_requested:
                break
            node.set_fee_ppm(fee_ppm, baseline_policies)
            time.sleep(args.gossip_settle_seconds)
            for family in ("lnd", "cln"):
                if phase_index < phase_cursor:
                    phase_index += 1
                    continue
                before = capture_snapshot(args.network_id)
                try:
                    records = send_cell(
                        bridge, args.network_id, family, args.amounts,
                        args.rounds_per_cell, args.payment_pause_seconds,
                    )
                except PolarTrafficError as exc:
                    result["ambiguous_payment"] = exc.uncertain_operation
                    result["completed_before_ambiguity"] = exc.completed_records
                    write_checkpoint(args.output, result)
                    raise CampaignError(str(exc)) from exc
                after = capture_snapshot(args.network_id)
                result["fee_phases"].append(
                    phase_summary(family, fee_ppm, records, before, after)
                )
                phase_index += 1
                write_checkpoint(args.output, result)

        # Active controller endurance: execution is allowed only inside this
        # bounded window. Rebalance spend is capped by the temporary budget.
        if args.endurance_seconds > 0 and not stop_requested:
            node.set_policies(baseline_policies)
            target_epochs = max(1, math.ceil(args.endurance_seconds / args.epoch_seconds))
            epoch = len(result["endurance_epochs"])
            invocation_limit = endurance_epoch_limit(
                epoch, target_epochs, args.max_new_endurance_epochs,
            )
            result["target_endurance_epochs"] = target_epochs
            enable_live_controller_if_needed(
                node,
                args.live_budget_sats,
                epoch,
                invocation_limit,
            )
            while epoch < invocation_limit and not stop_requested:
                epoch_started = time.monotonic()
                before = capture_snapshot(args.network_id)
                records: list[dict[str, Any]] = []
                try:
                    for family in ("lnd", "cln"):
                        amount = args.amounts[(epoch + (0 if family == "lnd" else 1)) % len(args.amounts)]
                        for direction in ("forward", "reverse"):
                            records.extend(
                                run_deterministic_traffic(
                                    bridge,
                                    args.network_id,
                                    1,
                                    amount,
                                    args.payment_pause_seconds,
                                    select_traffic_lanes(direction, family),
                                    invoice_retries=2,
                                )
                            )
                except PolarTrafficError as exc:
                    result["ambiguous_payment"] = exc.uncertain_operation
                    result["completed_before_ambiguity"] = exc.completed_records
                    write_checkpoint(args.output, result)
                    raise CampaignError(str(exc)) from exc
                fee_cycle = node.cycle("fees")
                rebalance_cycle = node.cycle("rebalance")
                after = capture_snapshot(args.network_id)
                result["endurance_epochs"].append({
                    "epoch": epoch,
                    "elapsed_seconds": round(time.monotonic() - epoch_started, 3),
                    "payments": len(records),
                    "all_payments_settled": all(
                        isinstance(row.get("payment"), dict)
                        and row["payment"].get("success") is True
                        for row in records
                    ),
                    "router_delta": router_delta(before, after),
                    "fee_cycle": fee_cycle,
                    "rebalance_cycle": rebalance_cycle,
                })
                write_checkpoint(args.output, result)
                epoch += 1
                if epoch >= invocation_limit or stop_requested:
                    break
                remaining = args.epoch_seconds - (time.monotonic() - epoch_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        result["cleanup"]["attempted"] = True
        cleanup_steps = (
            ("pause", lambda: node.set_config("paused", True)),
            ("budget_zero", lambda: node.set_config("daily_budget_sats", 0)),
            ("dry_run", lambda: node.set_dry_run(True)),
            ("policies", lambda: node.set_policies(baseline_policies)),
        )
        for label, operation in cleanup_steps:
            try:
                operation()
            except Exception as exc:  # cleanup must attempt every rail
                result["cleanup"]["errors"].append({"step": label, "error": str(exc)})
        try:
            result["final_dry_run"] = node.dry_run()
            result["final_snapshot"] = capture_snapshot(args.network_id)
            result["cleanup"]["safe_final_state"] = safe_final_state(
                result["final_snapshot"],
                dry_run=result["final_dry_run"],
            )
        except Exception as exc:
            result["cleanup"]["errors"].append({"step": "final_snapshot", "error": str(exc)})
            result["cleanup"]["safe_final_state"] = False
        result["finished_at"] = int(time.time())
        target_fee_phases = len(args.fee_ppms) * 2
        target_endurance_epochs = (
            max(1, math.ceil(args.endurance_seconds / args.epoch_seconds))
            if args.endurance_seconds > 0 else 0
        )
        result["campaign_complete"] = (
            len(result["fee_phases"]) >= target_fee_phases
            and len(result["endurance_epochs"]) >= target_endurance_epochs
        )
        write_checkpoint(args.output, result)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if result["cleanup"]["errors"] or not result["cleanup"].get("safe_final_state"):
        raise CampaignError("campaign cleanup did not prove every safety rail")
    if any(not phase["passed"] for phase in result["fee_phases"]):
        raise CampaignError("one or more fee sweep phases failed")
    if any(not epoch["all_payments_settled"] for epoch in result["endurance_epochs"]):
        raise CampaignError("one or more endurance payments failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-id", required=True, type=positive_network_id)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:37373")
    parser.add_argument("--fee-ppms", type=csv_positive_ints, default=(10, 25, 5, 50, 10))
    parser.add_argument("--amounts", type=csv_positive_ints, default=(5_000, 15_000, 35_000))
    parser.add_argument("--rounds-per-cell", type=int, default=5)
    parser.add_argument("--payment-pause-seconds", type=float, default=0.1)
    parser.add_argument("--gossip-settle-seconds", type=float, default=3.0)
    parser.add_argument("--endurance-seconds", type=int, default=0)
    parser.add_argument("--epoch-seconds", type=float, default=300.0)
    parser.add_argument(
        "--max-new-endurance-epochs", type=int, default=0,
        help="stop safely after this many new epochs (0 runs all remaining epochs)",
    )
    parser.add_argument("--live-budget-sats", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mine-preflight-block", action="store_true",
        help="mine one fake regtest block through Polar MCP before route readiness",
    )
    args = parser.parse_args()
    if args.rounds_per_cell <= 0:
        parser.error("--rounds-per-cell must be positive")
    if args.payment_pause_seconds < 0 or args.gossip_settle_seconds < 0:
        parser.error("pause values must be nonnegative")
    if args.endurance_seconds < 0 or args.epoch_seconds <= 0:
        parser.error("duration values are invalid")
    if args.max_new_endurance_epochs < 0:
        parser.error("--max-new-endurance-epochs must be nonnegative")
    if args.live_budget_sats < 0:
        parser.error("--live-budget-sats must be nonnegative")
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
