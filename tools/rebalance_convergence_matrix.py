#!/usr/bin/env python3
"""Run a repeatable rebalance convergence tuning matrix in Polar."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_HOLD_MARGINS = ["default", "0.20", "0.25", "0.30", "0.35"]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def margin_label(value: str) -> str:
    if value == "default":
        return "default"
    return "hold_" + value.replace(".", "p").replace("-", "neg")


def parse_margin(value: str) -> float | None:
    return None if value == "default" else float(value)


def selected_candidate_scores(run_dir: Path) -> list[float]:
    scores: list[float] = []
    for cycle_path in sorted(run_dir.glob("iteration_*/cycle.json")):
        try:
            cycle = json.loads(cycle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        selected = ((cycle.get("last_cycle") or {}).get("selected_candidates") or [])
        if not isinstance(selected, list):
            continue
        for candidate in selected:
            if not isinstance(candidate, dict):
                continue
            score = (candidate.get("score_decomposition") or {}).get("final_score")
            if score is None:
                score = candidate.get("score")
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue
    return scores


def build_run_args(
    *,
    loop_script: Path,
    out_dir: Path,
    topology: str,
    hold_margin: str,
    iterations: int,
    deploy: bool,
    restart_plugin: bool,
    install_cl_hive_deps: bool,
    restart_wait_seconds: float,
) -> list[str]:
    args = [
        sys.executable,
        str(loop_script),
        "--hive-mode",
        "enabled",
        "--hive-start-member-plugins",
        "--pure-hive-topology",
        topology,
        "--require-pure-hive-route",
        "--require-hive-member-hints",
        "--convergence-perturb-each-iteration",
        "--clear-rebalance-cooldowns",
        "--clear-rebalance-cooldowns-each-iteration",
        "--iterations",
        str(iterations),
        "--drive-payments",
        "0",
        "--settle-seconds",
        "3",
        "--restart-wait-seconds",
        str(restart_wait_seconds),
        "--intrahive-source-target-ratio",
        "0.98",
        "--intrahive-dest-target-ratio",
        "0.02",
        "--intrahive-corridor-target-ratio",
        "0.55",
        "--out-dir",
        str(out_dir),
    ]
    if deploy:
        args.append("--deploy")
    if restart_plugin:
        args.append("--restart-plugin")
    if install_cl_hive_deps:
        args.append("--install-cl-hive-deps")
    args.extend([
        "--tune-rebalance-hold-margin",
        "0.0" if hold_margin == "default" else hold_margin,
    ])
    return args


def summarize_loop(
    loop_json: dict[str, Any],
    *,
    topology: str,
    hold_margin: str,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    analyses = loop_json.get("analyses") or []
    convergence_runs = sum(1 for item in analyses if item.get("convergence_source_scid"))
    convergence_ok = sum(1 for item in analyses if item.get("convergence_ok") is True)
    selected = sum(int(item.get("selected_pairs", 0) or 0) for item in analyses)
    executions = sum(int(item.get("executions", 0) or 0) for item in analyses)
    successes = sum(int(item.get("successes", 0) or 0) for item in analyses)
    fees = sum(int(item.get("fee_sats", 0) or 0) for item in analyses)
    restored = sum(int(item.get("convergence_restored_sats", 0) or 0) for item in analyses)
    errors = [
        float(item["convergence_max_error"])
        for item in analyses
        if item.get("convergence_max_error") is not None
    ]
    scores = selected_candidate_scores(run_dir) if run_dir is not None else []
    min_score = round(min(scores), 6) if scores else None
    margin_value = 0.0 if hold_margin == "default" else parse_margin(hold_margin)
    pass_all = bool(analyses) and convergence_runs == len(analyses) and convergence_ok == len(analyses)
    return {
        "topology": topology,
        "hold_margin": hold_margin,
        "hold_margin_value": margin_value,
        "iterations": len(analyses),
        "valid_iterations": sum(1 for item in analyses if item.get("valid") is True),
        "convergence_runs": convergence_runs,
        "convergence_ok": convergence_ok,
        "selected_pairs": selected,
        "executions": executions,
        "successes": successes,
        "fees_sats": fees,
        "restored_sats": restored,
        "fee_per_restored_sat": round(fees / restored, 8) if restored else None,
        "avg_convergence_max_error": round(sum(errors) / len(errors), 8) if errors else None,
        "min_selected_final_score": min_score,
        "min_score_headroom": (
            round(min_score - margin_value, 6)
            if min_score is not None and margin_value is not None else
            None
        ),
        "pass": pass_all,
        "next_action": loop_json.get("next_action"),
        "next_reason": loop_json.get("next_reason"),
    }


def recommend_hold_margin(rows: list[dict[str, Any]]) -> float | None:
    numeric = [
        row
        for row in rows
        if row.get("hold_margin_value") is not None and row.get("topology") == "square"
    ]
    passing = sorted(float(row["hold_margin_value"]) for row in numeric if row.get("pass"))
    failing = sorted(float(row["hold_margin_value"]) for row in numeric if not row.get("pass"))
    if not passing:
        return None
    score_floor_values = [
        float(row["min_selected_final_score"])
        for row in numeric
        if row.get("pass") and row.get("min_selected_final_score") is not None
    ]
    score_guard_cap = min(score_floor_values) - 0.05 if score_floor_values else None
    if failing:
        guarded_cap = max(0.0, failing[0] - 0.05)
        if score_guard_cap is not None:
            guarded_cap = min(guarded_cap, max(0.0, score_guard_cap))
        guarded = [value for value in passing if value <= guarded_cap + 1e-9]
        if guarded:
            return round(max(guarded), 4)
    if score_guard_cap is not None:
        guarded = [value for value in passing if value <= score_guard_cap + 1e-9]
        if guarded:
            return round(max(guarded), 4)
    return round(max(passing), 4)


def write_analysis(path: Path, rows: list[dict[str, Any]], recommendation: float | None) -> None:
    lines = [
        "# Rebalance Convergence Tuning Matrix",
        "",
        f"- Runs: {len(rows)}",
        f"- Passing runs: {sum(1 for row in rows if row.get('pass'))}",
        (
            f"- Recommended hold margin: {recommendation:.2f}"
            if recommendation is not None else
            "- Recommended hold margin: n/a"
        ),
        "",
        "| topology | hold margin | pass | iterations | conv ok | selected | success | fees | restored | fee/restored | min score | headroom | next action | reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['topology']} | {row['hold_margin']} | {row['pass']} | "
            f"{row['iterations']} | {row['convergence_ok']} | {row['selected_pairs']} | "
            f"{row['successes']} | {row['fees_sats']} | {row['restored_sats']} | "
            f"{row['fee_per_restored_sat'] if row['fee_per_restored_sat'] is not None else ''} | "
            f"{row['min_selected_final_score'] if row['min_selected_final_score'] is not None else ''} | "
            f"{row['min_score_headroom'] if row['min_score_headroom'] is not None else ''} | "
            f"{row.get('next_action') or ''} | {row.get('next_reason') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matrix(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    loop_script = repo_root / "tools" / "rebalance_capex_loop.py"
    started = time.strftime("%Y%m%dT%H%M%S%z")
    out_dir = args.out_dir or repo_root / "results" / f"rebalance-convergence-matrix-{started}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for topology in args.topology:
        for hold_margin in args.hold_margin:
            run_dir = out_dir / f"{topology}_{margin_label(hold_margin)}"
            command = build_run_args(
                loop_script=loop_script,
                out_dir=run_dir,
                topology=topology,
                hold_margin=hold_margin,
                iterations=args.iterations,
                deploy=args.deploy,
                restart_plugin=args.restart_plugin,
                install_cl_hive_deps=args.install_cl_hive_deps,
                restart_wait_seconds=args.restart_wait_seconds,
            )
            proc = subprocess.run(command, text=True, capture_output=True, check=False)
            loop_path = run_dir / "loop.json"
            loop_json = json.loads(loop_path.read_text(encoding="utf-8")) if loop_path.exists() else {}
            summary = summarize_loop(
                loop_json,
                topology=topology,
                hold_margin=hold_margin,
                run_dir=run_dir,
            )
            summary["returncode"] = proc.returncode
            summary["out_dir"] = str(run_dir)
            rows.append(summary)
            runs.append({
                "topology": topology,
                "hold_margin": hold_margin,
                "command": command,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "summary": summary,
            })
            write_json(run_dir / "matrix_run.json", runs[-1])
            if proc.returncode != 0 and not args.keep_going:
                break

    recommendation = recommend_hold_margin(rows)
    matrix = {
        "started": started,
        "out_dir": str(out_dir),
        "iterations": args.iterations,
        "topologies": args.topology,
        "hold_margins": args.hold_margin,
        "recommended_hold_margin": recommendation,
        "rows": rows,
        "runs": runs,
    }
    write_json(out_dir / "matrix.json", matrix)
    write_analysis(out_dir / "ANALYSIS.md", rows, recommendation)
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--topology", action="append", choices=("triangle", "square"), default=[])
    parser.add_argument("--hold-margin", action="append", default=[])
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--restart-plugin", action="store_true")
    parser.add_argument("--install-cl-hive-deps", action="store_true")
    parser.add_argument("--restart-wait-seconds", type=float, default=5.0)
    parser.add_argument("--keep-going", action="store_true", default=True)
    args = parser.parse_args()
    if not args.topology:
        args.topology = ["square"]
    if not args.hold_margin:
        args.hold_margin = list(DEFAULT_HOLD_MARGINS)
    for value in args.hold_margin:
        if value != "default":
            parse_margin(value)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
