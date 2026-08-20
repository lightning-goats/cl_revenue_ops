from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import revenue_validation_collect as collect
from tools import revenue_validation_common as common

HOUR_SECONDS = 3600
DAY_SECONDS = 86400


def reconciliation_command(*, since: int, until: int, hours: int) -> str:
    return (
        "-k revenue-econ-reconcile apply=false "
        f"history_since={since} history_until={until} history_limit={hours}"
    )


def forward_history_command(*, since: int, until: int) -> str:
    return (
        "-k revenue-forward-history "
        f"history_since={since} history_until={until} limit=5000"
    )


def _clean_run(run: Any, expected_slot: int) -> bool:
    if not isinstance(run, Mapping):
        return False
    return (
        run.get("slot_started_at") == expected_slot
        and run.get("result") == "clean"
        and run.get("ledger_projection_status") == "aligned"
        and run.get("fee_intent_completeness") == "ok"
        and run.get("unexplained_divergence_count") == 0
        and isinstance(run.get("completed_at"), int)
        and not isinstance(run.get("completed_at"), bool)
        and run.get("error") is None
    )


def trailing_clean_hours(payload: Any, *, until: int, hours: int) -> int:
    if not isinstance(payload, Mapping):
        return 0
    history = payload.get("history")
    if not isinstance(history, Mapping):
        return 0
    runs = history.get("runs")
    if not isinstance(runs, list):
        return 0
    by_slot: dict[int, Any] = {}
    duplicates: set[int] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        slot = run.get("slot_started_at")
        if isinstance(slot, bool) or not isinstance(slot, int):
            continue
        if slot in by_slot:
            duplicates.add(slot)
        by_slot[slot] = run
    count = 0
    for expected in range(
        until - HOUR_SECONDS,
        until - hours * HOUR_SECONDS - 1,
        -HOUR_SECONDS,
    ):
        if expected in duplicates or not _clean_run(by_slot.get(expected), expected):
            break
        count += 1
    return count


def archive_window_complete(payload: Any, *, until_day: int, days: int) -> bool:
    if not isinstance(payload, Mapping):
        return False
    since = until_day - days * DAY_SECONDS
    if (
        payload.get("history_since") != since
        or payload.get("history_until") != until_day
        or payload.get("complete") is not True
        or payload.get("truncated") is not False
    ):
        return False
    coverage = payload.get("coverage")
    if not isinstance(coverage, list) or len(coverage) != days:
        return False
    rows: dict[int, Mapping[str, Any]] = {}
    for row in coverage:
        if not isinstance(row, Mapping):
            return False
        day = row.get("date_utc")
        if isinstance(day, bool) or not isinstance(day, int) or day in rows:
            return False
        rows[day] = row
    for day in range(since, until_day, DAY_SECONDS):
        row = rows.get(day)
        if row is None or (
            row.get("created_sync_complete") is not True
            or row.get("updated_sync_complete") is not True
            or row.get("aggregate_complete") is not True
            or row.get("reconciliation_status") != "complete"
            or row.get("reasons") != []
        ):
            return False
    return True


def _reconciliation_payload_shaped(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and isinstance(payload.get("history"), Mapping)
        and isinstance(payload["history"].get("runs"), list)
    )


def _archive_payload_shaped(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    required_types = {
        "history_since": int,
        "history_until": int,
        "coverage": list,
        "complete": bool,
        "truncated": bool,
    }
    for key, expected_type in required_types.items():
        value = payload.get(key)
        if not isinstance(value, expected_type):
            return False
        if expected_type is int and isinstance(value, bool):
            return False
    return True


def _utc_epoch(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("not_before must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("not_before must be timezone-aware")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("not_before must resolve to UTC")
    epoch = int(parsed.timestamp())
    if epoch < 0 or epoch % HOUR_SECONDS:
        raise ValueError("not_before must be a nonnegative UTC-hour boundary")
    return epoch


def monitor_node(
    node_name: str,
    node_cfg: Mapping[str, Any],
    *,
    observed_at: datetime,
    hours: int,
    not_before: int | None = None,
) -> dict[str, Any]:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    if hours < 1 or hours > 10_000:
        raise ValueError("hours must be between 1 and 10000")
    if (
        not_before is not None
        and (
            isinstance(not_before, bool)
            or not isinstance(not_before, int)
            or not_before < 0
            or not_before % HOUR_SECONDS
        )
    ):
        raise ValueError("not_before must be a nonnegative UTC-hour boundary")

    observed = observed_at.astimezone(timezone.utc)
    observed_epoch = int(observed.timestamp())
    until_hour = observed_epoch - observed_epoch % HOUR_SECONDS
    requested_since = until_hour - hours * HOUR_SECONDS
    since_hour = min(
        until_hour, max(requested_since, not_before or requested_since)
    )
    observable_hours = max(0, (until_hour - since_hour) // HOUR_SECONDS)
    until_day = observed_epoch - observed_epoch % DAY_SECONDS
    archive_days = max(1, (hours + 23) // 24)
    since_day = until_day - archive_days * DAY_SECONDS

    reconcile = None
    if observable_hours:
        reconcile = collect.run_json_rpc(
            node_cfg,
            reconciliation_command(
                since=since_hour,
                until=until_hour,
                hours=hours,
            ),
        )
    archive = collect.run_json_rpc(
        node_cfg,
        forward_history_command(since=since_day, until=until_day),
    )

    errors: list[str] = []
    if reconcile is not None and not reconcile.ok:
        errors.append(
            "reconciliation query failed: "
            + (reconcile.stderr.strip() or f"exit {reconcile.returncode}")
        )
    if not archive.ok:
        errors.append(
            "forward history query failed: "
            + (archive.stderr.strip() or f"exit {archive.returncode}")
        )
    reconcile_shaped = (
        reconcile is not None
        and reconcile.ok
        and _reconciliation_payload_shaped(reconcile.stdout_json)
    )
    archive_shaped = archive.ok and _archive_payload_shaped(archive.stdout_json)
    if reconcile is not None and reconcile.ok and not reconcile_shaped:
        errors.append("malformed reconciliation evidence")
    if archive.ok and not archive_shaped:
        errors.append("malformed forward archive evidence")

    clean_hours = (
        trailing_clean_hours(
            reconcile.stdout_json,
            until=until_hour,
            hours=observable_hours,
        )
        if reconcile_shaped
        else 0
    )
    archive_complete = (
        archive_window_complete(
            archive.stdout_json,
            until_day=until_day,
            days=archive_days,
        )
        if archive_shaped
        else False
    )
    ready = not errors and clean_hours >= hours and archive_complete
    return {
        "node": node_name,
        "observed_at": observed.isoformat().replace("+00:00", "Z"),
        "status": "error" if errors else ("ready" if ready else "pending"),
        "hours_required": hours,
        "gate_not_before": not_before,
        "reconciliation_since": since_hour,
        "reconciliation_until": until_hour,
        "consecutive_clean_hours": clean_hours,
        "archive_since": since_day,
        "archive_until": until_day,
        "archive_days_required": archive_days,
        "archive_complete": archive_complete,
        "reconciliation_and_archive_ready": ready,
        "formal_window_activation_authorized": False,
        "requires_daily_completeness_review": True,
        "errors": errors,
    }


def append_observation(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe the durable 72-hour measurement preflight read-only."
    )
    parser.add_argument(
        "--config",
        default="config/revenue_validation.yaml",
        help="Path to revenue validation config.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="Consecutive reconciliation hours required (default: 72).",
    )
    parser.add_argument(
        "--not-before",
        help="Earliest UTC-hour boundary eligible for the preflight.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = common.load_config(args.config)
    observed_at = datetime.now(timezone.utc)
    not_before = _utc_epoch(args.not_before) if args.not_before else None
    results_root = Path(config["paths"]["results_root"])
    results = []
    for node_name, node_cfg in config.get("nodes", {}).items():
        try:
            result = monitor_node(
                node_name,
                node_cfg,
                observed_at=observed_at,
                hours=args.hours,
                not_before=not_before,
            )
        except Exception as exc:
            result = {
                "node": node_name,
                "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
                "status": "error",
                "formal_window_activation_authorized": False,
                "errors": [str(exc) or exc.__class__.__name__],
            }
        append_observation(
            results_root / "preflight" / f"{node_name}.jsonl",
            result,
        )
        results.append(result)
    print(json.dumps({"observations": results}, indent=2, sort_keys=True))
    return 1 if any(item["status"] == "error" for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
