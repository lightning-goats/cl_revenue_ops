#!/usr/bin/env python3
"""Automate iterative fee-tournament runs and analysis.

The loop is intentionally conservative: it can run bounded Polar tournaments,
restore payer liquidity between iterations, analyze results, and classify the
next action. It does not silently edit production fee-controller code. Instead
it emits a decision that the agent/operator can act on:

- repeat_test: valid data, collect more samples
- refine_tests: harness/lab artifact invalidated the run
- consider_algorithm_change: valid data points at controller tuning

This keeps code changes traceable while still making the test/analyze/refine
cycle repeatable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import analyze_fee_tournament as analyzer
import competitive_fee_tournament as tournament
import long_fee_tournament


CHANNEL_ALIASES = ("revenue-node", "lnd-competitor-c")


@dataclass(frozen=True)
class PayerProfile:
    name: str
    payer_time_pref: float
    competitor_cltv_delta: int
    notes: str


PROFILES: dict[str, PayerProfile] = {
    "balanced": PayerProfile(
        name="balanced",
        payer_time_pref=0.0,
        competitor_cltv_delta=40,
        notes="Default LND route scoring; mixes fee, time, and probability preferences.",
    ),
    "fee_sensitive": PayerProfile(
        name="fee_sensitive",
        payer_time_pref=-1.0,
        competitor_cltv_delta=18,
        notes="Fee-sensitive LND route scoring with the minimum supported LND CLTV delta.",
    ),
}


@dataclass
class LoopDecision:
    action: str
    reason: str
    valid_for_fee_performance: bool
    code_change_warranted: bool
    test_refinement_warranted: bool
    next_step: str


def parse_int_list(value: str) -> list[int]:
    return long_fee_tournament.parse_int_list(value)


def _channels_by_alias(channels: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for channel in channels.get("channels", []) if isinstance(channels, dict) else []:
        alias = str(channel.get("peer_alias") or "")
        if alias in CHANNEL_ALIASES:
            result[alias] = channel
    return result


def channel_local_balances(channels: dict[str, Any]) -> dict[str, int]:
    balances: dict[str, int] = {}
    for alias, channel in _channels_by_alias(channels).items():
        try:
            balances[alias] = int(channel.get("local_balance") or 0)
        except (TypeError, ValueError):
            balances[alias] = 0
    return balances


def estimate_plan_payment_volume(phases: list[long_fee_tournament.PlannedPhase], rounds: int) -> int:
    return sum(phase.amount_sat * rounds for phase in phases)


def classify_summary(
    summary: dict[str, Any],
    *,
    min_success_rate: float = 1.0,
    min_forward_attribution_rate: float = 0.95,
    max_quote_forward_divergence: int = 0,
) -> LoopDecision:
    totals = summary.get("totals", {})
    success_rate = float(totals.get("payment_success_rate") or 0.0)
    attribution = float(totals.get("forward_attribution_rate") or 0.0)
    divergence = int(totals.get("quote_forward_divergent_bursts") or 0)
    successes = int(totals.get("payments_succeeded") or 0)
    failures = int(totals.get("payments_failed") or 0)
    revenue_forwards = int(totals.get("revenue_forwards") or 0)
    competitor_forwards = int(totals.get("competitor_forwards") or 0)
    boundary = summary.get("inferred_market_boundary_ppm")
    latest_fee = _latest_revenue_fee(summary)

    if successes == 0:
        return LoopDecision(
            action="refine_tests",
            reason="no successful payments were observed",
            valid_for_fee_performance=False,
            code_change_warranted=False,
            test_refinement_warranted=True,
            next_step="fix route/liquidity preconditions before testing fee logic",
        )

    if success_rate < min_success_rate or failures:
        return LoopDecision(
            action="refine_tests",
            reason=f"payment success rate {success_rate:.3f} below required {min_success_rate:.3f}",
            valid_for_fee_performance=False,
            code_change_warranted=False,
            test_refinement_warranted=True,
            next_step="inspect failed payment records and reduce payment volume or restore liquidity",
        )

    if attribution < min_forward_attribution_rate:
        return LoopDecision(
            action="refine_tests",
            reason=(
                f"forward attribution rate {attribution:.3f} below required "
                f"{min_forward_attribution_rate:.3f}"
            ),
            valid_for_fee_performance=False,
            code_change_warranted=False,
            test_refinement_warranted=True,
            next_step="restore liquidity or fix forward-counter collection before scoring fees",
        )

    if divergence > max_quote_forward_divergence:
        return LoopDecision(
            action="refine_tests",
            reason=f"{divergence} quote/forward divergent bursts observed",
            valid_for_fee_performance=True,
            code_change_warranted=False,
            test_refinement_warranted=True,
            next_step="split clean-payer and sticky-payer profiles, then repeat with explicit mission-control handling",
        )

    if (
        boundary is not None
        and latest_fee is not None
        and latest_fee > float(boundary)
        and competitor_forwards > revenue_forwards
    ):
        return LoopDecision(
            action="consider_algorithm_change",
            reason=(
                f"valid run shows fee {latest_fee:.2f} ppm above inferred boundary "
                f"{float(boundary):.2f} ppm while competitor captured more flow"
            ),
            valid_for_fee_performance=True,
            code_change_warranted=True,
            test_refinement_warranted=False,
            next_step="tighten boundary downshift or margin strategy, then rerun the same profile",
        )

    return LoopDecision(
        action="repeat_test",
        reason="run is valid, but does not justify production algorithm changes yet",
        valid_for_fee_performance=True,
        code_change_warranted=False,
        test_refinement_warranted=False,
        next_step="repeat with more samples or a different payer profile",
    )


def _latest_revenue_fee(summary: dict[str, Any]) -> float | None:
    for burst in reversed(summary.get("bursts", [])):
        fee = burst.get("revenue_fee_ppm")
        if fee is not None:
            try:
                return float(fee)
            except (TypeError, ValueError):
                return None
    return None


def collect_preflight(*, with_cl_hive: bool = False) -> dict[str, Any]:
    payer_channels = tournament.lnd(tournament.PAYER, "listchannels")
    sink_channels = tournament.lnd(tournament.SINK, "listchannels")
    fee_debug = tournament.cln(tournament.REVENUE, "revenue-fee-debug")
    return {
        "payer_channels": payer_channels,
        "sink_channels": sink_channels,
        "payer_local_balances": channel_local_balances(payer_channels),
        "sink_local_balances": channel_local_balances(sink_channels),
        "revenue_fee_debug": fee_debug,
        "cl_hive": tournament.collect_hive_snapshot(with_cl_hive),
    }


def collect_loop_preflight(args: Any) -> dict[str, Any]:
    if not getattr(args, "execute", False):
        return {}

    cl_hive_setup: dict[str, Any] | None = None
    if getattr(args, "with_cl_hive", False):
        cl_hive_setup = tournament.ensure_cl_hive(
            host_path=getattr(args, "cl_hive_host_path", tournament.DEFAULT_CL_HIVE_HOST_PATH),
            plugin_path=getattr(args, "cl_hive_plugin_path", tournament.DEFAULT_CL_HIVE_PLUGIN_PATH),
            deploy=not bool(getattr(args, "skip_cl_hive_deploy", False)),
            start=not bool(getattr(args, "skip_cl_hive_start", False)),
            genesis=not bool(getattr(args, "skip_cl_hive_genesis", False)),
            install_deps=bool(getattr(args, "install_cl_hive_deps", False)),
            hive_id=getattr(args, "cl_hive_id", tournament.DEFAULT_CL_HIVE_ID),
        )
    elif not bool(getattr(args, "skip_disable_cl_hive", False)):
        cl_hive_setup = {
            "enabled": False,
            "disabled": tournament.disable_cl_hive(
                plugin_path=getattr(args, "cl_hive_plugin_path", tournament.DEFAULT_CL_HIVE_PLUGIN_PATH),
            ),
        }

    preflight = collect_preflight(with_cl_hive=bool(getattr(args, "with_cl_hive", False)))
    if cl_hive_setup is not None:
        preflight["cl_hive_setup"] = cl_hive_setup
    return preflight


def restore_payer_liquidity(
    *,
    target_local_sats: int,
    reserve_sats: int,
    max_restore_sats: int,
) -> list[dict[str, Any]]:
    payer_channels = _channels_by_alias(tournament.lnd(tournament.PAYER, "listchannels"))
    sink_channels = _channels_by_alias(tournament.lnd(tournament.SINK, "listchannels"))
    actions: list[dict[str, Any]] = []

    for alias in CHANNEL_ALIASES:
        payer_channel = payer_channels.get(alias)
        sink_channel = sink_channels.get(alias)
        if not payer_channel or not sink_channel:
            actions.append({"alias": alias, "ok": False, "reason": "missing channel"})
            continue

        payer_local = int(payer_channel.get("local_balance") or 0)
        sink_local = int(sink_channel.get("local_balance") or 0)
        wanted = max(0, target_local_sats - payer_local)
        available = max(0, sink_local - reserve_sats)
        amount = min(wanted, available, max_restore_sats)
        if amount <= 0:
            actions.append(
                {
                    "alias": alias,
                    "ok": True,
                    "skipped": True,
                    "payer_local_sats": payer_local,
                    "sink_local_sats": sink_local,
                }
            )
            continue

        invoice = tournament.lnd(
            tournament.PAYER,
            "addinvoice",
            "--amt",
            str(amount),
            "--memo",
            f"loop-restore-{alias}-{int(time.time())}",
        )
        payment_request = invoice.get("payment_request")
        if not payment_request:
            actions.append({"alias": alias, "ok": False, "stage": "invoice", "result": invoice})
            continue

        paid = tournament.lnd(
            tournament.SINK,
            "payinvoice",
            "--force",
            "--outgoing_chan_id",
            str(sink_channel.get("scid")),
            "--fee_limit",
            "1000",
            payment_request,
        )
        actions.append(
            {
                "alias": alias,
                "ok": tournament.payment_succeeded(paid),
                "amount_sat": amount,
                "payer_local_sats_before": payer_local,
                "sink_local_sats_before": sink_local,
                "outgoing_chan_id": sink_channel.get("scid"),
                "payment_result": paid,
            }
        )
    return actions


def build_args(
    *,
    out_dir: Path,
    profile: PayerProfile,
    amounts: list[int],
    fixed_ppms: list[int],
    scenarios: list[str],
    cycles: int,
    rounds_per_phase: int,
    include_sticky: bool,
    seed: int,
    cycle_wait: int,
    plugin_path: str,
    policy_settle_seconds: float,
    policy_verify_timeout_seconds: float,
    post_payment_settle_seconds: float,
    channel_id: str,
    with_cl_hive: bool = False,
    cl_hive_host_path: str | Path = tournament.DEFAULT_CL_HIVE_HOST_PATH,
    cl_hive_plugin_path: str = tournament.DEFAULT_CL_HIVE_PLUGIN_PATH,
    cl_hive_id: str = tournament.DEFAULT_CL_HIVE_ID,
    install_cl_hive_deps: bool = False,
    skip_cl_hive_deploy: bool = False,
    skip_cl_hive_start: bool = False,
    skip_cl_hive_genesis: bool = False,
    skip_disable_cl_hive: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        out_dir=out_dir,
        scenario=scenarios,
        amounts=amounts,
        fixed_ppms=fixed_ppms,
        cycles=cycles,
        rounds_per_phase=rounds_per_phase,
        include_sticky=include_sticky,
        seed=seed,
        execute=True,
        cycle_wait=cycle_wait,
        plugin_path=plugin_path,
        policy_settle_seconds=policy_settle_seconds,
        policy_verify_timeout_seconds=policy_verify_timeout_seconds,
        post_payment_settle_seconds=post_payment_settle_seconds,
        competitor_cltv_delta=profile.competitor_cltv_delta,
        payer_time_pref=profile.payer_time_pref,
        channel_id=channel_id,
        with_cl_hive=with_cl_hive,
        cl_hive_host_path=cl_hive_host_path,
        cl_hive_plugin_path=cl_hive_plugin_path,
        cl_hive_id=cl_hive_id,
        install_cl_hive_deps=install_cl_hive_deps,
        skip_cl_hive_deploy=skip_cl_hive_deploy,
        skip_cl_hive_start=skip_cl_hive_start,
        skip_cl_hive_genesis=skip_cl_hive_genesis,
        skip_disable_cl_hive=skip_disable_cl_hive,
        competitor_controller="scripted",
        clboss_nodes=[],
        adaptive_undercut_ppm=5,
        adaptive_min_ppm=1,
        adaptive_max_ppm=1000,
        adaptive_fallback_ppm=150,
    )


def run_iteration(
    *,
    iteration_dir: Path,
    profile: PayerProfile,
    amounts: list[int],
    fixed_ppms: list[int],
    scenarios: list[str],
    cycles: int,
    rounds_per_phase: int,
    include_sticky: bool,
    seed: int,
    cycle_wait: int,
    plugin_path: str,
    policy_settle_seconds: float,
    policy_verify_timeout_seconds: float,
    post_payment_settle_seconds: float,
    channel_id: str,
    execute: bool,
    with_cl_hive: bool = False,
    cl_hive_host_path: str | Path = tournament.DEFAULT_CL_HIVE_HOST_PATH,
    cl_hive_plugin_path: str = tournament.DEFAULT_CL_HIVE_PLUGIN_PATH,
    cl_hive_id: str = tournament.DEFAULT_CL_HIVE_ID,
    install_cl_hive_deps: bool = False,
    skip_cl_hive_deploy: bool = False,
    skip_cl_hive_start: bool = False,
    skip_cl_hive_genesis: bool = False,
    skip_disable_cl_hive: bool = False,
) -> dict[str, Any]:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    args = build_args(
        out_dir=iteration_dir,
        profile=profile,
        amounts=amounts,
        fixed_ppms=fixed_ppms,
        scenarios=scenarios,
        cycles=cycles,
        rounds_per_phase=rounds_per_phase,
        include_sticky=include_sticky,
        seed=seed,
        cycle_wait=cycle_wait,
        plugin_path=plugin_path,
        policy_settle_seconds=policy_settle_seconds,
        policy_verify_timeout_seconds=policy_verify_timeout_seconds,
        post_payment_settle_seconds=post_payment_settle_seconds,
        channel_id=channel_id,
        with_cl_hive=with_cl_hive,
        cl_hive_host_path=cl_hive_host_path,
        cl_hive_plugin_path=cl_hive_plugin_path,
        cl_hive_id=cl_hive_id,
        install_cl_hive_deps=install_cl_hive_deps,
        skip_cl_hive_deploy=skip_cl_hive_deploy,
        skip_cl_hive_start=skip_cl_hive_start,
        skip_cl_hive_genesis=skip_cl_hive_genesis,
        skip_disable_cl_hive=skip_disable_cl_hive,
    )
    phases = long_fee_tournament.build_plan(
        scenarios=scenarios,
        amounts=amounts,
        cycles=cycles,
        fixed_ppms=fixed_ppms,
        include_sticky=include_sticky,
        competitor_controller="scripted",
        seed=seed,
    )
    plan = {
        "profile": asdict(profile),
        "settings": {
            "amounts": amounts,
            "fixed_ppms": fixed_ppms,
            "cycles": cycles,
            "rounds_per_phase": rounds_per_phase,
            "include_sticky": include_sticky,
            "with_cl_hive": with_cl_hive,
            "cl_hive_plugin_path": cl_hive_plugin_path,
            "install_cl_hive_deps": install_cl_hive_deps,
            "skip_disable_cl_hive": skip_disable_cl_hive,
            "policy_settle_seconds": policy_settle_seconds,
            "policy_verify_timeout_seconds": policy_verify_timeout_seconds,
            "estimated_payment_volume_sats": estimate_plan_payment_volume(phases, rounds_per_phase),
        },
        "phases": [asdict(phase) for phase in phases],
    }
    plan_path = iteration_dir / "loop_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not execute:
        return {
            "profile": profile.name,
            "plan_path": str(plan_path),
            "phases": len(phases),
            "executed": False,
        }

    try:
        result = long_fee_tournament.run_plan(args, phases)
    except RuntimeError as exc:
        error_text = str(exc)
        try:
            error_payload: Any = json.loads(error_text)
        except json.JSONDecodeError:
            error_payload = {"error": error_text}
        error_path = iteration_dir / "loop_error.json"
        error_path.write_text(json.dumps(error_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        decision = LoopDecision(
            action="refine_tests",
            code_change_warranted=False,
            test_refinement_warranted=True,
            reason="phase runner failed before producing valid fee-performance evidence",
            next_step="refine the tournament configuration or lab topology before interpreting fees",
            valid_for_fee_performance=False,
        )
        decision_path = iteration_dir / "decision.json"
        decision_path.write_text(json.dumps(asdict(decision), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "profile": profile.name,
            "plan_path": str(plan_path),
            "error_path": str(error_path),
            "decision_path": str(decision_path),
            "phases": len(phases),
            "executed": True,
            "decision": asdict(decision),
            "totals": {},
            "inferred_market_boundary_ppm": None,
        }

    result_path = iteration_dir / "loop_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = analyzer.load_records([result_path])
    metrics = [analyzer.extract_metrics(path, record, channel_id) for path, record in records]
    summary = analyzer.summarize(metrics)
    analysis_path = iteration_dir / "analysis.json"
    analysis_md_path = iteration_dir / "ANALYSIS.md"
    analysis_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    analysis_md_path.write_text(analyzer.markdown_report(summary), encoding="utf-8")
    decision = classify_summary(summary)
    decision_path = iteration_dir / "decision.json"
    decision_path.write_text(json.dumps(asdict(decision), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "profile": profile.name,
        "plan_path": str(plan_path),
        "result_path": str(result_path),
        "analysis_path": str(analysis_path),
        "analysis_markdown_path": str(analysis_md_path),
        "decision_path": str(decision_path),
        "phases": len(phases),
        "executed": True,
        "decision": asdict(decision),
        "totals": summary.get("totals", {}),
        "inferred_market_boundary_ppm": summary.get("inferred_market_boundary_ppm"),
    }


def render_loop_report(loop: dict[str, Any]) -> str:
    lines = [
        "# Fee Tournament Loop",
        "",
        f"- `started`: {loop['started']}",
        f"- `execute`: {loop['execute']}",
        f"- `profiles`: {', '.join(loop['profiles'])}",
        f"- `iterations`: {loop['iterations_requested']}",
        "",
        "## Results",
        "",
    ]
    for item in loop["results"]:
        decision = item.get("decision", {})
        totals = item.get("totals", {})
        lines.append(
            "- "
            f"`{item['profile']}` iteration={item['iteration']}, "
            f"action={decision.get('action', 'planned')}, "
            f"success_rate={totals.get('payment_success_rate')}, "
            f"forward_attribution={totals.get('forward_attribution_rate')}, "
            f"revenue_share={totals.get('revenue_share')}, "
            f"boundary={item.get('inferred_market_boundary_ppm')}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--profile",
        action="append",
        choices=["balanced", "fee_sensitive", "both"],
        default=None,
        help="Profile to run. May be repeated. Default: balanced.",
    )
    parser.add_argument("--amounts", type=parse_int_list, default=[1_000, 20_000])
    parser.add_argument("--fixed-ppms", type=parse_int_list, default=[60, 80, 100, 150])
    parser.add_argument(
        "--scenario",
        action="append",
        choices=["fixed_market", "step_shock", "adaptive_competitor", "clboss_external", "external"],
        default=None,
        help="Scenario to include. May be repeated. Default: fixed_market + step_shock + adaptive_competitor.",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--rounds-per-phase", type=int, default=1)
    parser.add_argument("--include-sticky", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cycle-wait", type=int, default=0)
    parser.add_argument("--plugin-path", default="/tmp/cl_revenue_ops/cl-revenue-ops.py")
    parser.add_argument("--with-cl-hive", action="store_true")
    parser.add_argument("--cl-hive-host-path", type=Path, default=tournament.DEFAULT_CL_HIVE_HOST_PATH)
    parser.add_argument("--cl-hive-plugin-path", default=tournament.DEFAULT_CL_HIVE_PLUGIN_PATH)
    parser.add_argument("--cl-hive-id", default=tournament.DEFAULT_CL_HIVE_ID)
    parser.add_argument("--install-cl-hive-deps", action="store_true")
    parser.add_argument("--skip-cl-hive-deploy", action="store_true")
    parser.add_argument("--skip-cl-hive-start", action="store_true")
    parser.add_argument("--skip-cl-hive-genesis", action="store_true")
    parser.add_argument("--skip-disable-cl-hive", action="store_true")
    parser.add_argument("--policy-settle-seconds", type=float, default=12.0)
    parser.add_argument("--policy-verify-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--post-payment-settle-seconds", type=float, default=1.0)
    parser.add_argument("--channel-id", default="277x1x0")
    parser.add_argument("--restore-liquidity", action="store_true")
    parser.add_argument("--target-payer-local-sats", type=int, default=450_000)
    parser.add_argument("--liquidity-reserve-sats", type=int, default=20_000)
    parser.add_argument("--max-restore-sats", type=int, default=450_000)
    args = parser.parse_args()

    selected = args.profile or ["balanced"]
    profile_names: list[str] = []
    for item in selected:
        if item == "both":
            profile_names.extend(["balanced", "fee_sensitive"])
        else:
            profile_names.append(item)
    scenarios = args.scenario or ["fixed_market", "step_shock", "adaptive_competitor"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = int(time.time())
    results: list[dict[str, Any]] = []
    preflight = collect_loop_preflight(args)

    for iteration in range(1, args.iterations + 1):
        for profile_name in profile_names:
            profile = PROFILES[profile_name]
            iteration_dir = args.out_dir / f"iteration_{iteration:03d}_{profile.name}"
            restore_actions = []
            if args.execute and args.restore_liquidity:
                restore_actions = restore_payer_liquidity(
                    target_local_sats=args.target_payer_local_sats,
                    reserve_sats=args.liquidity_reserve_sats,
                    max_restore_sats=args.max_restore_sats,
                )
            item = run_iteration(
                iteration_dir=iteration_dir,
                profile=profile,
                amounts=args.amounts,
                fixed_ppms=args.fixed_ppms,
                scenarios=scenarios,
                cycles=args.cycles,
                rounds_per_phase=args.rounds_per_phase,
                include_sticky=args.include_sticky,
                seed=args.seed + iteration - 1,
                cycle_wait=args.cycle_wait,
                plugin_path=args.plugin_path,
                policy_settle_seconds=args.policy_settle_seconds,
                policy_verify_timeout_seconds=args.policy_verify_timeout_seconds,
                post_payment_settle_seconds=args.post_payment_settle_seconds,
                channel_id=args.channel_id,
                execute=args.execute,
                with_cl_hive=args.with_cl_hive,
                cl_hive_host_path=args.cl_hive_host_path,
                cl_hive_plugin_path=args.cl_hive_plugin_path,
                cl_hive_id=args.cl_hive_id,
                install_cl_hive_deps=args.install_cl_hive_deps,
                skip_cl_hive_deploy=args.skip_cl_hive_deploy,
                skip_cl_hive_start=args.skip_cl_hive_start,
                skip_cl_hive_genesis=args.skip_cl_hive_genesis,
                skip_disable_cl_hive=args.skip_disable_cl_hive,
            )
            item["iteration"] = iteration
            item["restore_actions"] = restore_actions
            results.append(item)

            decision = item.get("decision", {})
            if decision.get("action") in {"refine_tests", "consider_algorithm_change"}:
                break
        if results and results[-1].get("decision", {}).get("action") in {
            "refine_tests",
            "consider_algorithm_change",
        }:
            break

    loop = {
        "schema": "fee_tournament_loop_v1",
        "started": started,
        "execute": args.execute,
        "iterations_requested": args.iterations,
        "profiles": profile_names,
        "settings": {
            "amounts": args.amounts,
            "fixed_ppms": args.fixed_ppms,
            "scenarios": scenarios,
            "cycles": args.cycles,
            "rounds_per_phase": args.rounds_per_phase,
            "cycle_wait": args.cycle_wait,
            "include_sticky": args.include_sticky,
            "with_cl_hive": args.with_cl_hive,
            "cl_hive_plugin_path": args.cl_hive_plugin_path,
            "install_cl_hive_deps": args.install_cl_hive_deps,
            "skip_disable_cl_hive": args.skip_disable_cl_hive,
            "restore_liquidity": args.restore_liquidity,
            "target_payer_local_sats": args.target_payer_local_sats,
            "policy_settle_seconds": args.policy_settle_seconds,
            "policy_verify_timeout_seconds": args.policy_verify_timeout_seconds,
        },
        "preflight": preflight,
        "results": results,
    }
    loop_path = args.out_dir / "loop.json"
    report_path = args.out_dir / "REPORT.md"
    loop_path.write_text(json.dumps(loop, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_loop_report(loop), encoding="utf-8")
    print(json.dumps({"loop_path": str(loop_path), "report_path": str(report_path), "results": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
