from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import revenue_validation_common as common

REQUIRED_FOR_COMPLETENESS = "required_for_completeness"
REQUIRED_FOR_ECONOMIC_METRICS = "required_for_economic_metrics"
OPTIONAL_DIAGNOSTIC = "optional_diagnostic"


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    command: str
    filename: str
    role: str
    required_paths: tuple[tuple[str, ...], ...]
    required_types: tuple[tuple[tuple[str, ...], type], ...] = ()


JSON_COMMANDS: tuple[CollectionSpec, ...] = (
    CollectionSpec(
        "revenue-dashboard 30", "revenue-dashboard-30.json",
        REQUIRED_FOR_ECONOMIC_METRICS,
        (("financial_health", "net_profit_sats"),
         ("period", "gross_revenue_sats"), ("period", "opex_sats"),
         ("period", "forward_count"), ("period", "volume_sats")),
        ((("financial_health", "net_profit_sats"), int),
         (("period", "gross_revenue_sats"), int),
         (("period", "opex_sats"), int),
         (("period", "forward_count"), int),
         (("period", "volume_sats"), int)),
    ),
    CollectionSpec(
        "revenue-report summary", "revenue-report-summary.json",
        OPTIONAL_DIAGNOSTIC, (("type",), ("policies",)),
    ),
    CollectionSpec(
        "revenue-profitability", "revenue-profitability.json",
        REQUIRED_FOR_ECONOMIC_METRICS, (("summary",), ("channels_by_class",)),
        ((("summary",), Mapping), (("channels_by_class",), Mapping)),
    ),
    CollectionSpec(
        "revenue-status", "revenue-status.json", REQUIRED_FOR_COMPLETENESS,
        (("status",), ("operator_controls", "values", "paused"),
         ("operator_controls", "values", "daily_budget_sats"),
         ("operator_controls", "values", "weekly_budget_sats"),
         ("operator_controls", "values", "risk_profile"),
         ("operator_controls", "values", "authority_level")),
        ((("status",), str),
         (("operator_controls", "values", "paused"), bool),
         (("operator_controls", "values", "daily_budget_sats"), int),
         (("operator_controls", "values", "weekly_budget_sats"), int),
         (("operator_controls", "values", "risk_profile"), str),
         (("operator_controls", "values", "authority_level"), str)),
    ),
    CollectionSpec(
        "revenue-config get", "revenue-config.json", REQUIRED_FOR_COMPLETENESS,
        (("config", "daily_budget_sats"), ("config", "weekly_budget_sats"),
         ("config", "authority_level"), ("version",)),
        ((("config", "daily_budget_sats"), int),
         (("config", "weekly_budget_sats"), int),
         (("config", "authority_level"), str), (("version",), int)),
    ),
    CollectionSpec(
        "listforwards", "listforwards.json", REQUIRED_FOR_ECONOMIC_METRICS,
        (("forwards",),), ((("forwards",), list),),
    ),
    CollectionSpec(
        "listpays", "listpays.json", REQUIRED_FOR_ECONOMIC_METRICS,
        (("pays",),), ((("pays",), list),),
    ),
    CollectionSpec(
        "listpeerchannels", "listpeerchannels.json",
        REQUIRED_FOR_ECONOMIC_METRICS, (("channels",),),
        ((("channels",), list),),
    ),
    CollectionSpec(
        "-k revenue-budget section=total_cost window_hours=24",
        "revenue-budget.json", REQUIRED_FOR_COMPLETENESS,
        (("coverage_status",), ("actual_spent_by_category",)),
        ((("coverage_status",), str), (("actual_spent_by_category",), Mapping)),
    ),
    CollectionSpec(
        "feerates perkb", "feerates.json", OPTIONAL_DIAGNOSTIC, (("perkb",),),
    ),
)


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


def _parse_run_date(run_date: str | date) -> date:
    if isinstance(run_date, date):
        return run_date
    try:
        return date.fromisoformat(str(run_date))
    except (TypeError, ValueError) as exc:
        raise ValueError("run_date must be an ISO UTC calendar date (YYYY-MM-DD)") from exc


def _reconciliation_command(run_day: date) -> str:
    start = datetime.combine(run_day, datetime.min.time(), tzinfo=timezone.utc)
    since = int(start.timestamp())
    until = int((start + timedelta(days=1)).timestamp())
    return (
        "-k revenue-econ-reconcile "
        f"history_since={since} history_until={until} history_limit=24"
    )


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
        run_day = _parse_run_date(run_date)
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
    evaluation = common.evaluation_identity(node_cfg)
    return {
        "evaluation_id": evaluation["id"],
        "evaluation_version": evaluation["version"],
        "evaluation_state": evaluation["state"],
        "formal_window_active": evaluation["formal_window_active"],
        "date": str(run_date),
        "node": node_name,
        "t0": evaluation["t0"],
        "days_since_t0": _days_since_t0(run_date, evaluation["t0"]),
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
        "paused": controls.get("paused"),
        "daily_budget_sats": controls.get("daily_budget_sats"),
        "weekly_budget_sats": controls.get("weekly_budget_sats"),
        "risk_profile": controls.get("risk_profile"),
    }


def append_trend_record(results_root: str | Path, node_name: str, trend_record: Mapping[str, Any]) -> None:
    path = common.trends_file(results_root, node_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trend_record, sort_keys=True) + "\n")


def _payload_error(
    payload: Any,
    required_paths: tuple[tuple[str, ...], ...],
    required_types: tuple[tuple[tuple[str, ...], type], ...] = (),
) -> str | None:
    if not isinstance(payload, Mapping):
        return "expected object payload"
    missing = []
    for path in required_paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                missing.append(".".join(path))
                break
            current = current[key]
    if missing:
        return "missing required keys: " + ", ".join(missing)
    for path, expected_type in required_types:
        current = payload
        for key in path:
            current = current[key]
        if (
            not isinstance(current, expected_type)
            or (expected_type is int and isinstance(current, bool))
        ):
            return (
                "invalid type for " + ".".join(path)
                + ": expected "
                + getattr(expected_type, "__name__", str(expected_type))
            )
    return None


def _list_record_error(filename: str, payload: Mapping[str, Any]) -> str | None:
    list_key = {
        "listforwards.json": "forwards",
        "listpays.json": "pays",
        "listpeerchannels.json": "channels",
    }.get(filename)
    if list_key is None:
        return None
    records = payload[list_key]
    for index, record in enumerate(records):
        prefix = f"{list_key}[{index}]"
        if not isinstance(record, Mapping):
            return f"{prefix}: expected object"
        if filename in {"listforwards.json", "listpays.json"}:
            if not isinstance(record.get("status"), str):
                return f"{prefix}.status: expected string"
        if filename == "listforwards.json" and record.get("status", "").lower() == "settled":
            if _coerce_int(record.get("fee_msat")) is None:
                return f"{prefix}.fee_msat: expected amount"
            forward_time = next(
                (record.get(key) for key in ("received_time", "resolved_time")
                 if record.get(key) is not None),
                None,
            )
            if _coerce_int(forward_time) is None:
                return f"{prefix}: missing or invalid forward timestamp"
        if filename == "listpays.json":
            values = (record.get(key) for key in ("label", "description", "bolt11"))
            is_rebalance = any(
                isinstance(value, str) and "rebalance" in value.lower()
                for value in values
            )
            if is_rebalance:
                pay_time = next(
                    (record.get(key) for key in
                     ("created_at", "created_time", "completed_at")
                     if record.get(key) is not None),
                    None,
                )
                if _coerce_int(pay_time) is None:
                    return f"{prefix}: missing or invalid rebalance timestamp"
                fee = _coerce_int(record.get("fee_msat"))
                sent = _coerce_int(record.get("amount_sent_msat"))
                amount = _coerce_int(record.get("amount_msat"))
                if fee is None and (sent is None or amount is None):
                    return f"{prefix}: missing rebalance fee evidence"
        if filename == "listpeerchannels.json":
            if not isinstance(record.get("peer_id"), str):
                return f"{prefix}.peer_id: expected string"
            fee_ppm = record.get("fee_proportional_millionths")
            if not isinstance(fee_ppm, int) or isinstance(fee_ppm, bool):
                return f"{prefix}.fee_proportional_millionths: expected int"
    return None


def _manifest_status(errors: Mapping[str, Mapping[str, Any]]) -> str:
    if not errors:
        return "complete"
    if all(error.get("role") == OPTIONAL_DIAGNOSTIC for error in errors.values()):
        return "collection_warning"
    return "incomplete"


def collect_node_day(
    node_name: str,
    node_cfg: Mapping[str, Any],
    day_dir: str | Path,
    run_date: str | date,
) -> dict[str, Any]:
    run_day = _parse_run_date(run_date)
    evaluation = common.evaluation_identity(node_cfg)
    out_dir = Path(day_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots: dict[str, Any] = {}
    errors: dict[str, dict[str, Any]] = {}

    for spec in JSON_COMMANDS:
        result = run_json_rpc(node_cfg, spec.command)
        payload_error = (
            None if not result.ok
            else _payload_error(
                result.stdout_json, spec.required_paths, spec.required_types
            )
        )
        if not payload_error and result.ok:
            payload_error = _list_record_error(spec.filename, result.stdout_json)
        if not result.ok or payload_error:
            errors[spec.filename] = {
                "command": spec.command,
                "role": spec.role,
                "stderr": result.stderr if not result.ok else payload_error,
                "returncode": result.returncode,
            }
            continue
        snapshots[spec.filename] = result.stdout_json
        common.write_json(out_dir / spec.filename, result.stdout_json)

    reconciliation_command = _reconciliation_command(run_day)
    reconciliation_result = run_json_rpc(node_cfg, reconciliation_command)
    reconciliation_payload_error = (
        None if not reconciliation_result.ok
        else _payload_error(
            reconciliation_result.stdout_json,
            (("history", "runs"),
             ("history", "summary", "expected_runs"),
             ("history", "summary", "complete"),
             ("history", "summary", "all_clean"),
             ("history", "summary", "fee_intent_complete")),
            (
                (("history", "runs"), list),
                (("history", "summary", "expected_runs"), int),
                (("history", "summary", "complete"), bool),
                (("history", "summary", "all_clean"), bool),
                (("history", "summary", "fee_intent_complete"), bool),
            ),
        )
    )
    if not reconciliation_result.ok or reconciliation_payload_error:
        errors["revenue-econ-reconcile.json"] = {
            "command": reconciliation_command,
            "role": REQUIRED_FOR_COMPLETENESS,
            "stderr": (
                reconciliation_result.stderr
                if not reconciliation_result.ok
                else reconciliation_payload_error
            ),
            "returncode": reconciliation_result.returncode,
        }
    else:
        snapshots["revenue-econ-reconcile.json"] = reconciliation_result.stdout_json
        common.write_json(
            out_dir / "revenue-econ-reconcile.json",
            reconciliation_result.stdout_json,
        )

    log_command = node_cfg.get("log_extract_command")
    if log_command:
        log_result = run_text_command(node_cfg, str(log_command))
        if log_result.ok:
            (out_dir / "rollback-watch.log").write_text(log_result.stdout, encoding="utf-8")
        else:
            errors["rollback-watch.log"] = {
                "command": str(log_command),
                "role": REQUIRED_FOR_COMPLETENESS,
                "stderr": log_result.stderr,
                "returncode": log_result.returncode,
            }

    else:
        errors["rollback-watch.log"] = {
            "command": None,
            "role": REQUIRED_FOR_COMPLETENESS,
            "stderr": "log_extract_command is not configured",
            "returncode": None,
        }

    trend_record = None
    if (
        "revenue-dashboard-30.json" in snapshots
        and "revenue-status.json" in snapshots
    ):
        trend_record = build_trend_record(
            node_name=node_name,
            node_cfg=node_cfg,
            run_date=run_day,
            dashboard=snapshots["revenue-dashboard-30.json"],
            revenue_status=snapshots["revenue-status.json"],
        )

    return {
        "status": _manifest_status(errors),
        "evaluation": evaluation,
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
        try:
            day_dir = common.node_day_dir(config, run_date, node_name)
            day_dir.mkdir(parents=True, exist_ok=True)
            result = collect_node_day(node_name, node_cfg, day_dir, run_date)
        except Exception as exc:
            result = {
                "status": "collection_failure",
                "errors": {
                    "collector": {
                        "command": None,
                        "role": REQUIRED_FOR_COMPLETENESS,
                        "stderr": f"{type(exc).__name__}: {exc}",
                        "returncode": None,
                    }
                },
            }
        trend_record = result.get("trend_record")
        if trend_record:
            try:
                append_trend_record(results_root, node_name, trend_record)
            except Exception as exc:
                result = {
                    "status": "collection_failure",
                    "errors": {
                        "collector": {
                            "command": None,
                            "role": REQUIRED_FOR_COMPLETENESS,
                            "stderr": (
                                "trend persistence failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "returncode": None,
                        }
                    },
                }
        manifest["nodes"][node_name] = {
            "status": result["status"],
            "evaluation": result.get("evaluation"),
            "errors": result.get("errors", {}),
        }

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
    return 0 if all(
        node["status"] in {"complete", "collection_warning"}
        for node in manifest["nodes"].values()
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
