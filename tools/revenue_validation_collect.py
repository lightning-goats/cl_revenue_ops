from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import revenue_validation_common as common

JSON_COMMANDS: list[tuple[str, str]] = [
    ("revenue-dashboard 30", "revenue-dashboard-30.json"),
    ("revenue-report summary", "revenue-report-summary.json"),
    ("revenue-profitability", "revenue-profitability.json"),
    ("revenue-status", "revenue-status.json"),
    ("revenue-config get", "revenue-config.json"),
    ("listforwards", "listforwards.json"),
    ("listpays", "listpays.json"),
    ("listpeerchannels", "listpeerchannels.json"),
    ("hive-members", "hive-members.json"),
    ("feerates perkb", "feerates.json"),
]

INT_RE = re.compile(r"-?\d+")


@dataclass(slots=True)
class CommandResult:
    ok: bool
    stdout_json: Any | None
    stderr: str
    returncode: int


def _parse_json(text: str) -> Any:
    payload = text.strip()
    if not payload:
        return {}
    return json.loads(payload)


def _run_remote_command(node_cfg: Mapping[str, Any], remote_cmd: str) -> common.RunResult:
    command = common.build_node_command(node_cfg, remote_cmd)
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    return common.RunResult(
        command=command,
        ok=completed.returncode == 0,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def run_json_rpc(node_cfg: Mapping[str, Any], command: str) -> CommandResult:
    remote_cmd = f"{node_cfg['lightning_cli_prefix']} {command}"
    result = _run_remote_command(node_cfg, remote_cmd)
    if not result.ok:
        return CommandResult(
            ok=False,
            stdout_json=None,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    try:
        payload = _parse_json(result.stdout)
    except json.JSONDecodeError as exc:
        return CommandResult(
            ok=False,
            stdout_json=None,
            stderr=f"invalid json: {exc}",
            returncode=result.returncode,
        )
    return CommandResult(
        ok=True,
        stdout_json=payload,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def run_text_command(node_cfg: Mapping[str, Any], remote_cmd: str) -> common.RunResult:
    return _run_remote_command(node_cfg, remote_cmd)


def _parse_t0(timestamp: str) -> datetime:
    normalized = timestamp.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _days_since_t0(run_date: str | date, t0: str) -> int:
    if isinstance(run_date, date):
        run_day = run_date
    else:
        run_day = date.fromisoformat(str(run_date))
    return (run_day - _parse_t0(t0).date()).days


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, dict):
        if "sats" in value:
            return _coerce_int(value["sats"])
        if "sat" in value:
            return _coerce_int(value["sat"])
        if "msat" in value:
            msat_value = _coerce_int(value["msat"])
            return None if msat_value is None else msat_value // 1000
        if "value" in value:
            return _coerce_int(value["value"])
        return None
    if isinstance(value, str):
        compact = value.replace(",", "")
        match = INT_RE.search(compact)
        if not match:
            return None
        number = int(match.group(0))
        if "msat" in compact:
            return number // 1000
        return number
    return None


def _nested_value(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _extract_first(payload: Mapping[str, Any], candidates: list[tuple[str, ...]]) -> int | None:
    for path in candidates:
        value = _nested_value(payload, *path)
        coerced = _coerce_int(value)
        if coerced is not None:
            return coerced
    return None


def build_trend_record(
    node_name: str,
    node_cfg: Mapping[str, Any],
    run_date: str | date,
    dashboard: Mapping[str, Any],
    revenue_status: Mapping[str, Any],
) -> dict[str, Any]:
    controls = _nested_value(revenue_status, "operator_controls", "values") or {}
    return {
        "date": str(run_date),
        "node": node_name,
        "t0": node_cfg["t0"],
        "days_since_t0": _days_since_t0(run_date, node_cfg["t0"]),
        "gross_revenue_sats_30d": _extract_first(
            dashboard,
            [
                ("financial_health", "gross_revenue_sats"),
                ("financial_health", "gross_revenue"),
                ("period", "gross_revenue_sats"),
                ("period", "gross_revenue"),
                ("gross_revenue_sats",),
                ("gross_revenue",),
            ],
        ),
        "net_profit_sats_30d": _extract_first(
            dashboard,
            [
                ("financial_health", "net_profit_sats"),
                ("financial_health", "net_profit"),
                ("period", "net_profit_sats"),
                ("period", "net_profit"),
                ("net_profit_sats",),
                ("net_profit",),
            ],
        ),
        "opex_sats_30d": _extract_first(
            dashboard,
            [
                ("financial_health", "opex_sats"),
                ("financial_health", "opex"),
                ("period", "opex_sats"),
                ("period", "opex"),
                ("opex_sats",),
                ("opex",),
            ],
        ),
        "forward_count_30d": _extract_first(
            dashboard,
            [
                ("period", "forward_count"),
                ("period", "forwards"),
                ("forward_count",),
                ("forwards",),
            ],
        ),
        "volume_sats_30d": _extract_first(
            dashboard,
            [
                ("period", "volume_sats"),
                ("period", "routed_volume_sats"),
                ("volume_sats",),
                ("routed_volume_sats",),
            ],
        ),
        "planner_enabled": controls.get("planner_enabled"),
        "planner_execute_closes": controls.get("planner_execute_closes"),
        "planner_max_opens_per_cycle": controls.get("planner_max_opens_per_cycle"),
        "planner_max_closes_per_cycle": controls.get("planner_max_closes_per_cycle"),
    }


def append_trend_record(results_root: str | Path, node_name: str, trend_record: Mapping[str, Any]) -> None:
    path = common.trends_file(results_root, node_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trend_record, sort_keys=True) + "\n")


def collect_node_day(
    node_name: str,
    node_cfg: Mapping[str, Any],
    day_dir: str | Path,
    run_date: str | date,
) -> dict[str, Any]:
    out_dir = Path(day_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots: dict[str, Any] = {}
    errors: dict[str, dict[str, Any]] = {}

    for command, filename in JSON_COMMANDS:
        result = run_json_rpc(node_cfg, command)
        if not result.ok:
            errors[filename] = {
                "command": command,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
            continue
        snapshots[filename] = result.stdout_json
        common.write_json(out_dir / filename, result.stdout_json)

    log_command = node_cfg.get("log_extract_command")
    if log_command:
        log_result = run_text_command(node_cfg, str(log_command))
        if log_result.ok:
            (out_dir / "rollback-watch.log").write_text(log_result.stdout, encoding="utf-8")
        else:
            errors["rollback-watch.log"] = {
                "command": str(log_command),
                "stderr": log_result.stderr,
                "returncode": log_result.returncode,
            }

    trend_record = build_trend_record(
        node_name=node_name,
        node_cfg=node_cfg,
        run_date=run_date,
        dashboard=snapshots.get("revenue-dashboard-30.json", {}),
        revenue_status=snapshots.get("revenue-status.json", {}),
    )

    return {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "trend_record": trend_record,
    }


def collect_all_nodes(config: Mapping[str, Any], run_date: str | date) -> dict[str, Any]:
    results_root = Path(config["paths"]["results_root"])
    manifest = {
        "date": str(run_date),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": {},
    }

    for node_name, node_cfg in config["nodes"].items():
        day_dir = common.node_day_dir(config, run_date, node_name)
        day_dir.mkdir(parents=True, exist_ok=True)
        result = collect_node_day(node_name, node_cfg, day_dir, run_date)
        manifest["nodes"][node_name] = {
            "status": result["status"],
            "errors": result.get("errors", {}),
        }
        trend_record = result.get("trend_record")
        if trend_record:
            append_trend_record(results_root, node_name, trend_record)

    common.write_json(results_root / "manifests" / f"{run_date}.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect daily revenue validation evidence.")
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
    manifest = collect_all_nodes(config, args.date)
    return 0 if all(node["status"] == "ok" for node in manifest["nodes"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
