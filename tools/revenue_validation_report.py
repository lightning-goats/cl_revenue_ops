from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import revenue_validation_common as common
from tools import revenue_validation_watch as watch


def _parse_run_date(run_date: str | date) -> date:
    if isinstance(run_date, date):
        return run_date
    return date.fromisoformat(str(run_date))


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    with candidate.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    rows = []
    for line in candidate.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_rule(findings: Iterable[Mapping[str, Any]], rule: str) -> Mapping[str, Any] | None:
    for finding in findings:
        if finding.get("rule") == rule:
            return finding
    return None


def _format_int(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header_row, divider, body])


def _latest_trend_rows(results_root: str | Path, node_names: Iterable[str], run_date: str | date) -> dict[str, dict[str, Any]]:
    target_date = str(run_date)
    latest: dict[str, dict[str, Any]] = {}
    for node_name in node_names:
        rows = _read_jsonl(Path(results_root) / "trends" / f"{node_name}.jsonl")
        filtered = [row for row in rows if str(row.get("date")) <= target_date]
        if filtered:
            latest[node_name] = filtered[-1]
    return latest


def _watch_day_node(results_root: str | Path, run_date: str | date, node_name: str) -> dict[str, Any]:
    payload = _read_json(Path(results_root) / "watch" / f"{run_date}.json")
    nodes = payload.get("nodes")
    if not isinstance(nodes, Mapping):
        return {}
    node_payload = nodes.get(node_name)
    return node_payload if isinstance(node_payload, Mapping) else {}


def _date_strings(start_date: date, end_date: date) -> list[str]:
    current = start_date
    out = []
    while current <= end_date:
        out.append(current.isoformat())
        current += timedelta(days=1)
    return out


def _read_log_lines(results_root: str | Path, day: str, node_name: str) -> list[str]:
    base = Path(results_root) / day / node_name
    for filename in ("rollback-watch.log", "debug-log-extract.log"):
        path = base / filename
        if path.exists():
            return path.read_text(encoding="utf-8").splitlines()
    return []


def _activation_summary(results_root: str | Path, node_name: str, start_date: date, end_date: date) -> dict[str, int]:
    counts = {
        "intra_fleet_policy": 0,
        "non_default_blend": 0,
        "competition_aware_preserve": 0,
        "competition_aware_undercut": 0,
        "variance_gated_undercut": 0,
        "dynamic_close_cost": 0,
        "coordination_reserved_slots": 0,
    }
    for day in _date_strings(start_date, end_date):
        for line in _read_log_lines(results_root, day, node_name):
            lowered = line.lower()
            if (
                "hive member: zero-fee fleet policy" in lowered
                or "hive member: 1-ppm fleet policy" in lowered
            ):
                counts["intra_fleet_policy"] += 1
            if "blend:" in lowered and "blend:0.15" not in lowered and "blend:0.20" not in lowered:
                counts["non_default_blend"] += 1
            if "competition_aware preserve" in lowered:
                counts["competition_aware_preserve"] += 1
            if "competition_aware undercut" in lowered:
                counts["competition_aware_undercut"] += 1
            if "undercut explore" in lowered:
                counts["variance_gated_undercut"] += 1
            if "estimated_closure_cost" in lowered or "_estimate_close_cost" in lowered:
                counts["dynamic_close_cost"] += 1
            if "coordination_reserved_slots" in lowered or "rebalance_coordination" in lowered:
                counts["coordination_reserved_slots"] += 1
    return counts


def _watch_history(results_root: str | Path, node_name: str, start_date: date, end_date: date) -> list[dict[str, Any]]:
    history = []
    for day in _date_strings(start_date, end_date):
        node_payload = _watch_day_node(results_root, day, node_name)
        if node_payload:
            history.append({"date": day, **node_payload})
    return history


def _highest_history_severity(history: Iterable[Mapping[str, Any]]) -> str:
    highest = "green"
    for item in history:
        severity = str(item.get("highest_severity", "green"))
        if watch.SEVERITY_RANK.get(severity, 0) > watch.SEVERITY_RANK[highest]:
            highest = severity
    return highest


def _count_history_days(history: Iterable[Mapping[str, Any]], severity: str) -> int:
    return sum(1 for item in history if item.get("highest_severity") == severity)


def _latest_day_json(results_root: str | Path, run_date: str | date, node_name: str, filename: str) -> dict[str, Any]:
    return _read_json(Path(results_root) / str(run_date) / node_name / filename)


def _hive_fee_status(peerchannels: Mapping[str, Any], hive_members: Mapping[str, Any]) -> tuple[str, str]:
    hive_peer_ids = set()
    for member in hive_members.get("members") or []:
        if isinstance(member, Mapping):
            peer_id = member.get("peer_id") or member.get("id") or member.get("node_id")
            if isinstance(peer_id, str):
                hive_peer_ids.add(peer_id)
        elif isinstance(member, str):
            hive_peer_ids.add(member)

    if not hive_peer_ids:
        return "inconclusive", "no hive members in saved snapshot"

    mismatches = []
    matched = 0
    for channel in peerchannels.get("channels") or []:
        peer_id = channel.get("peer_id")
        if not isinstance(peer_id, str) or peer_id not in hive_peer_ids:
            continue
        matched += 1
        if channel.get("fee_proportional_millionths") != 1:
            mismatches.append(channel.get("short_channel_id") or peer_id)
    if matched == 0:
        return "inconclusive", "no hive channels found in saved snapshot"
    if mismatches:
        return "refuted", f"hive channels not at 1 ppm: {', '.join(map(str, mismatches))}"
    return "confirmed", "all observed hive channels are at 1 ppm"


def _window_fee_sats(listforwards: Mapping[str, Any], start_dt: datetime, end_dt: datetime) -> int:
    total = 0
    for forward in listforwards.get("forwards") or []:
        if str(forward.get("status", "")).lower() != "settled":
            continue
        received_time = watch._unix_from_event(forward, "received_time", "resolved_time")
        if received_time is None:
            continue
        received_dt = datetime.fromtimestamp(received_time, tz=timezone.utc)
        if start_dt <= received_dt < end_dt:
            total += watch._fee_sats(forward)
    return total


def _window_rebalance_cost_sats(listpays: Mapping[str, Any], start_dt: datetime, end_dt: datetime) -> int:
    total = 0
    for pay in listpays.get("pays") or []:
        if not watch._is_rebalance_pay(pay):
            continue
        created_at = watch._unix_from_event(pay, "created_at", "created_time", "completed_at")
        if created_at is None:
            continue
        created_dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
        if not (start_dt <= created_dt < end_dt):
            continue
        fee = watch._coerce_int(pay.get("fee_msat"))
        if fee is None:
            sent = watch._coerce_int(pay.get("amount_sent_msat"))
            amt = watch._coerce_int(pay.get("amount_msat"))
            fee = 0 if sent is None or amt is None else max(sent - amt, 0)
        total += fee or 0
    return total


def _due_for_checkpoint(latest_rows: Mapping[str, Mapping[str, Any]], checkpoint_days: int) -> bool:
    return any(int(row.get("days_since_t0", -1)) >= checkpoint_days for row in latest_rows.values())


def _report_exists(reports_root: str | Path, checkpoint_label: str) -> bool:
    return any(Path(reports_root).glob(f"*-production-{checkpoint_label}-findings.md"))


def _t14_recommendation(history: list[dict[str, Any]]) -> str:
    highest = _highest_history_severity(history)
    if highest == "red":
        return "pause"
    if highest == "yellow":
        return "continue with investigation"
    return "continue"


def _t28_decision(latest_rows: Mapping[str, Mapping[str, Any]], histories: Mapping[str, list[dict[str, Any]]]) -> str:
    if any(_highest_history_severity(history) == "red" for history in histories.values()):
        return "rollback"
    if latest_rows and all((row.get("net_profit_sats_30d") or 0) >= 0 for row in latest_rows.values()):
        silent_paths = 0
        for history in histories.values():
            if _highest_history_severity(history) != "green":
                silent_paths += 1
        return "ship with notes" if silent_paths else "ship"
    return "investigate"


def _render_t14_report(
    run_date: str | date,
    results_root: str | Path,
    config: Mapping[str, Any],
    latest_rows: Mapping[str, Mapping[str, Any]],
) -> str:
    run_day = _parse_run_date(run_date)
    headers = [
        "Node",
        "Days Since T0",
        "Gross Revenue 30d",
        "Net Profit 30d",
        "Forwards 30d",
        "Volume 30d",
        "Rebalance 7d",
        "Severity",
    ]
    rows = []
    activation_lines = []
    yellow_lines = []
    recommendation = "continue"

    for node_name, node_cfg in config["nodes"].items():
        latest = latest_rows.get(node_name, {})
        t0_date = _parse_timestamp(node_cfg["t0"]).date()
        window_end = min(run_day, t0_date + timedelta(days=14))
        history = _watch_history(results_root, node_name, t0_date, window_end)
        latest_watch = _watch_day_node(results_root, run_date, node_name)
        rebalance = _find_rule(latest_watch.get("findings", []), "rebalance_success_rate") or {}
        rows.append(
            [
                node_name,
                _format_int(latest.get("days_since_t0")),
                _format_int(latest.get("gross_revenue_sats_30d")),
                _format_int(latest.get("net_profit_sats_30d")),
                _format_int(latest.get("forward_count_30d")),
                _format_int(latest.get("volume_sats_30d")),
                f"{_format_int(rebalance.get('attempt_count'))} attempts / {_format_int(rebalance.get('success_rate_pct'))}%",
                str(latest_watch.get("highest_severity", "green")),
            ]
        )
        activations = _activation_summary(results_root, node_name, t0_date, window_end)
        activation_lines.append(
            f"- `{node_name}`: 1-PPM lines {_format_int(activations['intra_fleet_policy'])}, "
            f"non-default blend {_format_int(activations['non_default_blend'])}, "
            f"competition-aware preserve {_format_int(activations['competition_aware_preserve'])}, "
            f"competition-aware undercut {_format_int(activations['competition_aware_undercut'])}, "
            f"dynamic close-cost {_format_int(activations['dynamic_close_cost'])}, "
            f"coordination/reserved slots {_format_int(activations['coordination_reserved_slots'])}."
        )
        yellow_rules = []
        for day in history:
            for finding in day.get("findings", []):
                if finding.get("severity") == "yellow":
                    yellow_rules.append(finding.get("rule"))
        unique_yellow_rules = sorted(set(str(rule) for rule in yellow_rules))
        yellow_lines.append(
            f"- `{node_name}`: highest severity `{_highest_history_severity(history)}`, "
            f"red days {_count_history_days(history, 'red')}, yellow days {_count_history_days(history, 'yellow')}, "
            f"yellow rules {', '.join(unique_yellow_rules) if unique_yellow_rules else 'none'}."
        )
        node_recommendation = _t14_recommendation(history)
        if node_recommendation == "pause":
            recommendation = "pause"
        elif node_recommendation != "continue" and recommendation == "continue":
            recommendation = node_recommendation

    deploy_lines = [
        f"- `{node_name}`: T0 `{node_cfg['t0']}`, checkpoint state `{watch.checkpoint_state(node_cfg['t0'], run_date)}`."
        for node_name, node_cfg in config["nodes"].items()
    ]

    return "\n".join(
        [
            "# Production Validation Checkpoint — T+14",
            "",
            f"Generated from saved evidence on `{run_date}`.",
            "",
            "## Deploy timestamp confirmation",
            *deploy_lines,
            "",
            "## Red-flag status",
            f"- Provisional recommendation: `{recommendation}`.",
            *yellow_lines,
            "",
            "## Per-node behavior summary table",
            _render_table(headers, rows),
            "",
            "## New-path activation summary",
            *activation_lines,
            "",
            "## Yellow flags observed + recommendation",
            f"- Draft recommendation: `{recommendation}`.",
            "- Red flags should still trigger operator review before any action.",
            "",
            "## Inconclusive items",
            "- This draft uses saved daily extracts and the latest rolling dashboard snapshot. It does not assert causal revenue improvement yet.",
            "- The fleet is only two nodes, so silent code paths can still be expected even when deployment is healthy.",
            "",
            "## Specific data/commands the operator should pull next",
            "- Keep the daily collector running through T+28.",
            "- Review any non-green watch findings before changing production settings.",
            "- If revenue movement appears material, compare the final T+28 draft against saved raw `listforwards` and `listpays` snapshots.",
            "",
        ]
    )


def _render_t28_report(
    run_date: str | date,
    results_root: str | Path,
    config: Mapping[str, Any],
    latest_rows: Mapping[str, Mapping[str, Any]],
) -> str:
    run_day = _parse_run_date(run_date)
    comparison_headers = [
        "Node",
        "Gross Revenue 30d",
        "Net Profit 30d",
        "Opex 30d",
        "Pre 28d Fees",
        "Post 28d Fees",
        "Post 28d Rebalance Cost",
        "Severity",
    ]
    comparison_rows = []
    per_pr_lines = []
    yellow_lines = []
    histories: dict[str, list[dict[str, Any]]] = {}

    for node_name, node_cfg in config["nodes"].items():
        latest = latest_rows.get(node_name, {})
        t0_dt = _parse_timestamp(node_cfg["t0"])
        checkpoint_end = min(
            datetime.combine(run_day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1),
            t0_dt + timedelta(days=28),
        )
        pre_start = t0_dt - timedelta(days=28)
        pre_end = t0_dt
        post_start = t0_dt

        listforwards = _latest_day_json(results_root, run_date, node_name, "listforwards.json")
        listpays = _latest_day_json(results_root, run_date, node_name, "listpays.json")
        pre_fees = _window_fee_sats(listforwards, pre_start, pre_end)
        post_fees = _window_fee_sats(listforwards, post_start, checkpoint_end)
        post_rebalance_cost = _window_rebalance_cost_sats(listpays, post_start, checkpoint_end)

        history = _watch_history(results_root, node_name, t0_dt.date(), checkpoint_end.date())
        histories[node_name] = history
        latest_watch = _watch_day_node(results_root, run_date, node_name)
        comparison_rows.append(
            [
                node_name,
                _format_int(latest.get("gross_revenue_sats_30d")),
                _format_int(latest.get("net_profit_sats_30d")),
                _format_int(latest.get("opex_sats_30d")),
                _format_int(pre_fees),
                _format_int(post_fees),
                _format_int(post_rebalance_cost),
                str(latest_watch.get("highest_severity", "green")),
            ]
        )

        activations = _activation_summary(results_root, node_name, t0_dt.date(), checkpoint_end.date())
        peerchannels = _latest_day_json(results_root, run_date, node_name, "listpeerchannels.json")
        hive_members = _latest_day_json(results_root, run_date, node_name, "hive-members.json")
        hive_status, hive_note = _hive_fee_status(peerchannels, hive_members)
        compaware_status = "confirmed" if activations["competition_aware_preserve"] or activations["competition_aware_undercut"] else "inconclusive"
        variance_status = "confirmed" if activations["variance_gated_undercut"] else "inconclusive"
        close_cost_status = "confirmed" if activations["dynamic_close_cost"] else "inconclusive"
        coordination_status = (
            "confirmed"
            if activations["coordination_reserved_slots"]
            else "inconclusive (expected on 2-node fleet)"
        )
        per_pr_lines.extend(
            [
                f"- `{node_name}` #87 intra-fleet ppm=1: `{hive_status}`. {hive_note}.",
                f"- `{node_name}` #87 competition_aware: `{compaware_status}`. Preserve lines {_format_int(activations['competition_aware_preserve'])}, undercut lines {_format_int(activations['competition_aware_undercut'])}.",
                f"- `{node_name}` #87 variance-gated undercut: `{variance_status}`. `undercut explore` lines {_format_int(activations['variance_gated_undercut'])}.",
                f"- `{node_name}` #88 dynamic close cost: `{close_cost_status}`. Estimator lines {_format_int(activations['dynamic_close_cost'])}.",
                f"- `{node_name}` #89 reserved slots: `{coordination_status}`. Coordination lines {_format_int(activations['coordination_reserved_slots'])}.",
            ]
        )

        yellow_rules = []
        for day in history:
            for finding in day.get("findings", []):
                if finding.get("severity") == "yellow":
                    yellow_rules.append(finding.get("rule"))
        yellow_lines.append(
            f"- `{node_name}`: highest severity `{_highest_history_severity(history)}`, "
            f"yellow rules {', '.join(sorted(set(yellow_rules))) if yellow_rules else 'none'}."
        )

    decision = _t28_decision(latest_rows, histories)

    return "\n".join(
        [
            "# Production Validation Checkpoint — T+28",
            "",
            f"Generated from saved evidence on `{run_date}`.",
            "",
            "## Executive summary",
            f"- Provisional decision: `{decision}`.",
            "- Confidence remains limited by two-node fleet size and the lack of a clean pre-deploy profitability snapshot.",
            "",
            "## Revenue / profit comparison, honest confidence bounds",
            _render_table(comparison_headers, comparison_rows),
            "",
            "## Per-PR hypothesis status",
            *per_pr_lines,
            "",
            "## Amortized capex treatment",
            "- This draft does not yet automate opener-aware amortized capex. Use the saved `revenue-profitability` and channel-history evidence for manual confirmation before a final ship/rollback call.",
            "",
            "## Yellow-flag events observed during the window",
            *yellow_lines,
            "",
            "## Recommendations for next iteration",
            f"- Current draft outcome: `{decision}`.",
            "- If the final manual review confirms harm greater than the noise floor, revert in reverse merge order `#89 -> #88 -> #87`.",
            "- If economics are flat and behavior remains healthy, keep the automation in place for quarterly comparison.",
            "",
            "## Inconclusive items",
            "- Competition-aware and coordination-slot features may stay quiet on a 2-node fleet even when code is correct.",
            "- Capex treatment and opener-aware channel framing still need manual review against saved raw evidence before a final economic conclusion.",
            "",
        ]
    )


def generate_checkpoint_reports(config: Mapping[str, Any], run_date: str | date) -> list[Path]:
    results_root = Path(config["paths"]["results_root"])
    reports_root = Path(config["paths"]["reports_root"])
    reports_root.mkdir(parents=True, exist_ok=True)

    latest_rows = _latest_trend_rows(results_root, config["nodes"].keys(), run_date)
    generated: list[Path] = []

    if _due_for_checkpoint(latest_rows, 14) and not _report_exists(reports_root, "t14"):
        t14_path = reports_root / f"{run_date}-production-t14-findings.md"
        t14_path.write_text(
            _render_t14_report(run_date, results_root, config, latest_rows),
            encoding="utf-8",
        )
        generated.append(t14_path)

    if _due_for_checkpoint(latest_rows, 28) and not _report_exists(reports_root, "t28"):
        t28_path = reports_root / f"{run_date}-production-t28-findings.md"
        t28_path.write_text(
            _render_t28_report(run_date, results_root, config, latest_rows),
            encoding="utf-8",
        )
        generated.append(t28_path)

    return generated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate checkpoint reports from saved revenue validation evidence.")
    parser.add_argument(
        "--config",
        default="config/revenue_validation.yaml",
        help="Path to revenue validation config.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Run date in YYYY-MM-DD format. Defaults to local today.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = common.load_config(args.config)
    generate_checkpoint_reports(config, args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
