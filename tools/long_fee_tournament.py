#!/usr/bin/env python3
"""Plan or run longer competitive fee tournaments against the Polar lab.

The existing ``competitive_fee_tournament.py`` is intentionally small and
phase-oriented. This orchestrator builds repeatable long-run matrices on top of
that runner:

- fixed market fee sweeps
- step-shock competitor changes
- sticky-payer versus clean-payer phases
- adaptive scripted undercut competitors
- optional external/CLBOSS-managed competitor metadata

By default this tool writes a plan only. Pass ``--execute`` to run it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import competitive_fee_tournament as tournament


DEFAULT_AMOUNTS = [1_000, 5_000, 20_000, 100_000, 250_000, 500_000]
FIXED_MARKET_PPMS = [40, 60, 80, 100, 150, 250]
STEP_SHOCK_PPMS = [150, 80, 80, 150]


@dataclass(frozen=True)
class PlannedPhase:
    scenario: str
    name: str
    cycle: int
    amount_sat: int
    competitor_ppm: int | None
    reset_mc: bool
    force_cycle_before: bool
    force_cycle_after: bool
    competitor_controller: str = "scripted"
    notes: str = ""


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    if not result:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return result


def current_revenue_fee_ppm(channel_id: str) -> int | None:
    debug = tournament.cln(tournament.REVENUE, "revenue-fee-debug")
    channels = debug.get("channels", []) if isinstance(debug, dict) else []
    for channel in channels:
        if channel.get("channel_id") == channel_id:
            try:
                return int(channel.get("last_broadcast_fee_ppm"))
            except (TypeError, ValueError):
                return None
    return None


def collect_clboss_status(nodes: list[str]) -> dict[str, Any]:
    """Collect CLBOSS status if available; fail open when plugin is absent."""
    statuses: dict[str, Any] = {}
    for node in nodes:
        statuses[node] = tournament.cln(node, "clboss-status")
    return statuses


def fixed_market_phases(
    *,
    amounts: list[int],
    cycles: int,
    ppms: list[int],
    include_sticky: bool,
) -> list[PlannedPhase]:
    phases: list[PlannedPhase] = []
    for cycle in range(cycles):
        for ppm in ppms:
            for amount in amounts:
                phases.append(
                    PlannedPhase(
                        scenario="fixed_market",
                        name=f"fixed_{ppm}_clean_c{cycle}_a{amount}",
                        cycle=cycle,
                        amount_sat=amount,
                        competitor_ppm=ppm,
                        reset_mc=True,
                        force_cycle_before=False,
                        force_cycle_after=True,
                        notes="clean payer state; scripted fixed competitor fee",
                    )
                )
                if include_sticky:
                    phases.append(
                        PlannedPhase(
                            scenario="fixed_market",
                            name=f"fixed_{ppm}_sticky_c{cycle}_a{amount}",
                            cycle=cycle,
                            amount_sat=amount,
                            competitor_ppm=ppm,
                            reset_mc=False,
                            force_cycle_before=False,
                            force_cycle_after=False,
                            notes="sticky payer state; measures mission-control memory",
                        )
                    )
    return phases


def step_shock_phases(
    *,
    amounts: list[int],
    cycles: int,
    include_sticky: bool,
) -> list[PlannedPhase]:
    phases: list[PlannedPhase] = []
    for cycle in range(cycles):
        for step, ppm in enumerate(STEP_SHOCK_PPMS):
            for amount in amounts:
                clean = PlannedPhase(
                    scenario="step_shock",
                    name=f"shock_s{step}_{ppm}_clean_c{cycle}_a{amount}",
                    cycle=cycle,
                    amount_sat=amount,
                    competitor_ppm=ppm,
                    reset_mc=True,
                    force_cycle_before=False,
                    force_cycle_after=True,
                    notes="competitor fee step shock with mission control reset",
                )
                phases.append(clean)
                if include_sticky and step in {1, 2}:
                    phases.append(
                        PlannedPhase(
                            scenario="step_shock",
                            name=f"shock_s{step}_{ppm}_sticky_c{cycle}_a{amount}",
                            cycle=cycle,
                            amount_sat=amount,
                            competitor_ppm=ppm,
                            reset_mc=False,
                            force_cycle_before=False,
                            force_cycle_after=False,
                            notes="same shock without reset; isolates payer memory",
                        )
                    )
    return phases


def adaptive_competitor_phases(*, amounts: list[int], cycles: int) -> list[PlannedPhase]:
    phases: list[PlannedPhase] = []
    for cycle in range(cycles):
        for amount in amounts:
            phases.append(
                PlannedPhase(
                    scenario="adaptive_competitor",
                    name=f"adaptive_clean_c{cycle}_a{amount}",
                    cycle=cycle,
                    amount_sat=amount,
                    competitor_ppm=None,
                    reset_mc=True,
                    force_cycle_before=False,
                    force_cycle_after=True,
                    notes="competitor ppm resolved at runtime as revenue_fee - undercut",
                )
            )
    return phases


def external_controller_phases(
    *,
    amounts: list[int],
    cycles: int,
    competitor_controller: str,
) -> list[PlannedPhase]:
    phases: list[PlannedPhase] = []
    for cycle in range(cycles):
        for amount in amounts:
            phases.append(
                PlannedPhase(
                    scenario=f"{competitor_controller}_external",
                    name=f"{competitor_controller}_external_c{cycle}_a{amount}",
                    cycle=cycle,
                    amount_sat=amount,
                    competitor_ppm=None,
                    reset_mc=True,
                    force_cycle_before=False,
                    force_cycle_after=True,
                    competitor_controller=competitor_controller,
                    notes="competitor policy is managed outside this runner",
                )
            )
    return phases


def build_plan(
    *,
    scenarios: list[str],
    amounts: list[int],
    cycles: int,
    fixed_ppms: list[int],
    include_sticky: bool,
    competitor_controller: str,
    seed: int,
) -> list[PlannedPhase]:
    phases: list[PlannedPhase] = []
    for scenario in scenarios:
        if scenario == "fixed_market":
            phases.extend(
                fixed_market_phases(
                    amounts=amounts,
                    cycles=cycles,
                    ppms=fixed_ppms,
                    include_sticky=include_sticky,
                )
            )
        elif scenario == "step_shock":
            phases.extend(
                step_shock_phases(
                    amounts=amounts,
                    cycles=cycles,
                    include_sticky=include_sticky,
                )
            )
        elif scenario == "adaptive_competitor":
            phases.extend(adaptive_competitor_phases(amounts=amounts, cycles=cycles))
        elif scenario == "clboss_external":
            phases.extend(
                external_controller_phases(
                    amounts=amounts,
                    cycles=cycles,
                    competitor_controller="clboss",
                )
            )
        elif scenario == "external":
            phases.extend(
                external_controller_phases(
                    amounts=amounts,
                    cycles=cycles,
                    competitor_controller=competitor_controller,
                )
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    rng = random.Random(seed)
    rng.shuffle(phases)
    return phases


def resolve_competitor_ppm(
    phase: PlannedPhase,
    *,
    channel_id: str,
    adaptive_undercut_ppm: int,
    adaptive_min_ppm: int,
    adaptive_max_ppm: int,
    fallback_ppm: int,
) -> int | None:
    if phase.competitor_ppm is not None:
        return phase.competitor_ppm
    if phase.scenario != "adaptive_competitor":
        return None

    revenue_fee = current_revenue_fee_ppm(channel_id)
    if revenue_fee is None:
        revenue_fee = fallback_ppm
    return max(adaptive_min_ppm, min(adaptive_max_ppm, revenue_fee - adaptive_undercut_ppm))


def run_plan(args: argparse.Namespace, phases: list[PlannedPhase]) -> dict[str, Any]:
    started = int(time.time())
    executed_phases: list[dict[str, Any]] = []
    with_cl_hive = bool(getattr(args, "with_cl_hive", False))
    cl_hive_setup = (
        tournament.ensure_cl_hive(
            host_path=getattr(args, "cl_hive_host_path", tournament.DEFAULT_CL_HIVE_HOST_PATH),
            plugin_path=getattr(args, "cl_hive_plugin_path", tournament.DEFAULT_CL_HIVE_PLUGIN_PATH),
            deploy=not bool(getattr(args, "skip_cl_hive_deploy", False)),
            start=not bool(getattr(args, "skip_cl_hive_start", False)),
            genesis=not bool(getattr(args, "skip_cl_hive_genesis", False)),
            install_deps=bool(getattr(args, "install_cl_hive_deps", False)),
            hive_id=getattr(args, "cl_hive_id", tournament.DEFAULT_CL_HIVE_ID),
        )
        if with_cl_hive else
        (
            {
                "enabled": False,
                "disabled": tournament.disable_cl_hive(
                    plugin_path=getattr(args, "cl_hive_plugin_path", tournament.DEFAULT_CL_HIVE_PLUGIN_PATH),
                ),
            }
            if not bool(getattr(args, "skip_disable_cl_hive", False)) else
            {"enabled": False, "disabled": {"ok": True, "skipped": True}}
        )
    )
    clboss_before = (
        collect_clboss_status(args.clboss_nodes)
        if args.competitor_controller == "clboss" and args.clboss_nodes else
        {}
    )

    for index, phase in enumerate(phases, start=1):
        competitor_ppm = resolve_competitor_ppm(
            phase,
            channel_id=args.channel_id,
            adaptive_undercut_ppm=args.adaptive_undercut_ppm,
            adaptive_min_ppm=args.adaptive_min_ppm,
            adaptive_max_ppm=args.adaptive_max_ppm,
            fallback_ppm=args.adaptive_fallback_ppm,
        )
        set_policy = args.competitor_controller == "scripted" and competitor_ppm is not None
        result = tournament.run_phase(
            name=phase.name,
            rounds=args.rounds_per_phase,
            amount_sat=phase.amount_sat,
            competitor_ppm=competitor_ppm or args.adaptive_fallback_ppm,
            reset_mc=phase.reset_mc,
            policy_settle_seconds=args.policy_settle_seconds,
            post_payment_settle_seconds=args.post_payment_settle_seconds,
            force_cycle_before=phase.force_cycle_before,
            force_cycle_after=phase.force_cycle_after,
            cycle_wait=args.cycle_wait,
            plugin_path=args.plugin_path,
            out_dir=args.out_dir,
            competitor_cltv_delta=args.competitor_cltv_delta,
            payer_time_pref=args.payer_time_pref,
            set_competitor_policy=set_policy,
            competitor_controller=phase.competitor_controller,
            policy_verify_timeout_seconds=args.policy_verify_timeout_seconds,
            with_cl_hive=with_cl_hive,
        )
        result["long_tournament_phase"] = asdict(phase)
        result["long_tournament_index"] = index
        result["resolved_competitor_ppm"] = competitor_ppm
        executed_phases.append(result)

    clboss_after = (
        collect_clboss_status(args.clboss_nodes)
        if args.competitor_controller == "clboss" and args.clboss_nodes else
        {}
    )
    return {
        "schema": "long_fee_tournament_v1",
        "started": started,
        "settings": {
            "rounds_per_phase": args.rounds_per_phase,
            "cycle_wait": args.cycle_wait,
            "policy_settle_seconds": args.policy_settle_seconds,
            "policy_verify_timeout_seconds": args.policy_verify_timeout_seconds,
            "post_payment_settle_seconds": args.post_payment_settle_seconds,
            "channel_id": args.channel_id,
            "competitor_controller": args.competitor_controller,
            "competitor_cltv_delta": args.competitor_cltv_delta,
            "payer_time_pref": args.payer_time_pref,
            "seed": args.seed,
            "with_cl_hive": with_cl_hive,
            "cl_hive_plugin_path": getattr(args, "cl_hive_plugin_path", tournament.DEFAULT_CL_HIVE_PLUGIN_PATH),
            "install_cl_hive_deps": bool(getattr(args, "install_cl_hive_deps", False)),
        },
        "cl_hive_setup": cl_hive_setup,
        "clboss_before": clboss_before,
        "clboss_after": clboss_after,
        "phases": executed_phases,
    }


def render_plan_markdown(phases: list[PlannedPhase], args: argparse.Namespace) -> str:
    planned_controllers = sorted({phase.competitor_controller for phase in phases})
    lines = [
        "# Long Fee Tournament Plan",
        "",
        f"- `execute`: {args.execute}",
        f"- `scenarios`: {', '.join(args.scenario)}",
        f"- `cycles`: {args.cycles}",
        f"- `rounds_per_phase`: {args.rounds_per_phase}",
        f"- `amounts`: {', '.join(str(a) for a in args.amounts)}",
        f"- `with_cl_hive`: {getattr(args, 'with_cl_hive', False)}",
        f"- `requested_competitor_controller`: {args.competitor_controller}",
        f"- `planned_competitor_controllers`: {', '.join(planned_controllers)}",
        f"- `total_phases`: {len(phases)}",
        "",
        "## Phases",
        "",
    ]
    for phase in phases:
        lines.append(
            "- "
            f"`{phase.name}` scenario={phase.scenario}, cycle={phase.cycle}, "
            f"amount={phase.amount_sat}, competitor_ppm={phase.competitor_ppm}, "
            f"reset_mc={phase.reset_mc}, force_cycle_after={phase.force_cycle_after}, "
            f"controller={phase.competitor_controller}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=["fixed_market", "step_shock", "adaptive_competitor", "clboss_external", "external"],
        default=None,
        help="Scenario to include. May be repeated. Default: fixed_market + step_shock + adaptive_competitor.",
    )
    parser.add_argument("--amounts", type=parse_int_list, default=DEFAULT_AMOUNTS)
    parser.add_argument("--fixed-ppms", type=parse_int_list, default=FIXED_MARKET_PPMS)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--rounds-per-phase", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--include-sticky", action="store_true")
    parser.add_argument("--execute", action="store_true")
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
    parser.add_argument("--post-payment-settle-seconds", type=float, default=2.0)
    parser.add_argument("--competitor-cltv-delta", type=int, default=40)
    parser.add_argument("--payer-time-pref", type=float, default=0.0)
    parser.add_argument("--channel-id", default="277x1x0")
    parser.add_argument(
        "--competitor-controller",
        choices=["scripted", "clboss", "external"],
        default="scripted",
        help="scripted sets LND competitor policies; clboss/external only observe externally managed policies.",
    )
    parser.add_argument("--clboss-nodes", type=lambda s: [p for p in s.split(",") if p], default=[])
    parser.add_argument("--adaptive-undercut-ppm", type=int, default=5)
    parser.add_argument("--adaptive-min-ppm", type=int, default=1)
    parser.add_argument("--adaptive-max-ppm", type=int, default=1000)
    parser.add_argument("--adaptive-fallback-ppm", type=int, default=150)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.scenario is None:
        args.scenario = ["fixed_market", "step_shock", "adaptive_competitor"]

    phases = build_plan(
        scenarios=args.scenario,
        amounts=args.amounts,
        cycles=args.cycles,
        fixed_ppms=args.fixed_ppms,
        include_sticky=args.include_sticky,
        competitor_controller=args.competitor_controller,
        seed=args.seed,
    )

    started = int(time.time())
    plan = {
        "schema": "long_fee_tournament_plan_v1",
        "started": started,
        "settings": {
            "scenarios": args.scenario,
            "amounts": args.amounts,
            "fixed_ppms": args.fixed_ppms,
            "cycles": args.cycles,
            "rounds_per_phase": args.rounds_per_phase,
            "include_sticky": args.include_sticky,
            "requested_competitor_controller": args.competitor_controller,
            "planned_competitor_controllers": sorted(
                {phase.competitor_controller for phase in phases}
            ),
            "competitor_cltv_delta": args.competitor_cltv_delta,
            "payer_time_pref": args.payer_time_pref,
            "policy_verify_timeout_seconds": args.policy_verify_timeout_seconds,
            "with_cl_hive": args.with_cl_hive,
            "cl_hive_plugin_path": args.cl_hive_plugin_path,
            "install_cl_hive_deps": args.install_cl_hive_deps,
            "seed": args.seed,
        },
        "phases": [asdict(phase) for phase in phases],
    }
    plan_path = args.out_dir / f"long_tournament_plan_{started}.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.out_dir / f"long_tournament_plan_{started}.md"
    markdown_path.write_text(render_plan_markdown(phases, args), encoding="utf-8")

    output: dict[str, Any] = {
        "plan_path": str(plan_path),
        "markdown_path": str(markdown_path),
        "phases": len(phases),
        "execute": args.execute,
    }
    if args.execute:
        result = run_plan(args, phases)
        result_path = args.out_dir / f"long_tournament_{started}.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output["result_path"] = str(result_path)

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
