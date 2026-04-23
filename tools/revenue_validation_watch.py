from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import revenue_validation_common as common

SEVERITY_RANK = {
    "green": 0,
    "yellow": 1,
    "red": 2,
}

SCID_RE = re.compile(r"\b\d+x\d+x\d+\b")


def _parse_run_date(run_date: str | date) -> date:
    if isinstance(run_date, date):
        return run_date
    return date.fromisoformat(str(run_date))


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _severity_max(findings: Iterable[Mapping[str, Any]]) -> str:
    highest = "green"
    for finding in findings:
        severity = str(finding.get("severity", "green"))
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK[highest]:
            highest = severity
    return highest


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    with candidate.open(encoding="utf-8") as f:
        return json.load(f)


def _read_lines(path: str | Path) -> list[str]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    return candidate.read_text(encoding="utf-8").splitlines()


def _node_day_dir(config: Mapping[str, Any], run_date: str | date, node_name: str) -> Path:
    return common.node_day_dir(config, run_date, node_name)


def _manifest_path(results_root: str | Path, run_date: str | date) -> Path:
    return Path(results_root) / "manifests" / f"{run_date}.json"


def _watch_path(results_root: str | Path, run_date: str | date) -> Path:
    return Path(results_root) / "watch" / f"{run_date}.json"


def _hive_peer_ids(hive_members: Mapping[str, Any]) -> set[str]:
    members = hive_members.get("members") or []
    peer_ids: set[str] = set()
    for member in members:
        if isinstance(member, str):
            peer_ids.add(member)
            continue
        if isinstance(member, Mapping):
            peer_id = member.get("peer_id") or member.get("id") or member.get("node_id")
            if isinstance(peer_id, str):
                peer_ids.add(peer_id)
    return peer_ids


def _config_value(payload: Mapping[str, Any], key: str) -> Any:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return None
    value = config.get(key)
    if isinstance(value, Mapping):
        if "value" in value:
            return value["value"]
        if "current" in value:
            return value["current"]
    return value


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "")
        if cleaned.endswith("msat"):
            digits = re.sub(r"[^0-9-]", "", cleaned[:-4])
            return int(digits) // 1000 if digits else None
        digits = re.sub(r"[^0-9-]", "", cleaned)
        return int(digits) if digits else None
    if isinstance(value, Mapping):
        for field in ("value", "sats", "sat", "msat"):
            if field in value:
                parsed = _coerce_int(value[field])
                if parsed is not None:
                    if field == "msat":
                        return parsed // 1000
                    return parsed
    return None


def _channel_ref(channel: Mapping[str, Any]) -> str:
    for key in ("short_channel_id", "channel_id", "peer_id"):
        value = channel.get(key)
        if isinstance(value, str):
            return value
    return "unknown-channel"


def _unix_from_event(item: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = item.get(key)
        parsed = _coerce_int(value)
        if parsed is not None:
            return parsed
    return None


def _window_bounds(run_date: str | date, days: int) -> tuple[int, int]:
    end_dt = datetime.combine(_parse_run_date(run_date), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
    start_dt = end_dt - timedelta(days=days)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def _daily_log_lines(lines: list[str], run_date: str | date) -> list[str]:
    prefix = f"{_parse_run_date(run_date).isoformat()}T"
    filtered = [line for line in lines if line.startswith(prefix)]
    return filtered or lines


def check_plugin_restart_count(lines: list[str], limit: int) -> dict[str, Any]:
    restart_lines = [
        line
        for line in lines
        if "initializing cl-revenue-ops plugin" in line.lower()
    ]
    severity = "red" if len(restart_lines) > limit else "green"
    return {
        "rule": "plugin_restart_count",
        "severity": severity,
        "count": len(restart_lines),
        "threshold": limit,
        "message": f"{len(restart_lines)} plugin restart lines in extracted log window",
    }


def check_zero_ppm_non_hive(peerchannels: Mapping[str, Any], hive_members: Mapping[str, Any]) -> dict[str, Any]:
    hive_peer_ids = _hive_peer_ids(hive_members)
    offenders = []
    for channel in peerchannels.get("channels") or []:
        fee_ppm = _coerce_int(channel.get("fee_proportional_millionths"))
        peer_id = channel.get("peer_id")
        if fee_ppm == 0 and isinstance(peer_id, str) and peer_id not in hive_peer_ids:
            offenders.append(_channel_ref(channel))
    severity = "red" if offenders else "green"
    return {
        "rule": "zero_ppm_non_hive",
        "severity": severity,
        "count": len(offenders),
        "channels": offenders,
        "message": f"{len(offenders)} non-hive channels at 0 ppm",
    }


def check_ceiling_pricing(
    peerchannels: Mapping[str, Any],
    hive_members: Mapping[str, Any],
    revenue_config: Mapping[str, Any],
) -> dict[str, Any]:
    max_fee_ppm = _coerce_int(_config_value(revenue_config, "max_fee_ppm"))
    if max_fee_ppm is None:
        return {
            "rule": "ceiling_pricing",
            "severity": "green",
            "count": 0,
            "channels": [],
            "message": "max_fee_ppm unavailable",
        }

    hive_peer_ids = _hive_peer_ids(hive_members)
    offenders = []
    for channel in peerchannels.get("channels") or []:
        peer_id = channel.get("peer_id")
        fee_ppm = _coerce_int(channel.get("fee_proportional_millionths"))
        if isinstance(peer_id, str) and peer_id not in hive_peer_ids and fee_ppm == max_fee_ppm:
            offenders.append(_channel_ref(channel))
    severity = "red" if offenders else "green"
    return {
        "rule": "ceiling_pricing",
        "severity": severity,
        "count": len(offenders),
        "channels": offenders,
        "threshold": max_fee_ppm,
        "message": f"{len(offenders)} non-hive channels at max_fee_ppm",
    }


def _is_rebalance_pay(pay: Mapping[str, Any]) -> bool:
    for key in ("label", "description", "bolt11"):
        value = pay.get(key)
        if isinstance(value, str) and "rebalance" in value.lower():
            return True
    return False


def check_rebalance_success_rate(
    listpays: Mapping[str, Any],
    run_date: str | date,
    floor_pct: int,
) -> dict[str, Any]:
    start_ts, end_ts = _window_bounds(run_date, 7)
    attempts = []
    for pay in listpays.get("pays") or []:
        if not _is_rebalance_pay(pay):
            continue
        created_at = _unix_from_event(pay, "created_at", "created_time", "completed_at")
        if created_at is None or created_at < start_ts or created_at >= end_ts:
            continue
        attempts.append(pay)

    success_statuses = {"complete", "completed", "success", "succeeded"}
    success_count = sum(1 for pay in attempts if str(pay.get("status", "")).lower() in success_statuses)
    attempt_count = len(attempts)
    success_rate = 100.0 if attempt_count == 0 else round((success_count / attempt_count) * 100, 1)
    severity = "red" if attempt_count and success_rate < floor_pct else "green"
    return {
        "rule": "rebalance_success_rate",
        "severity": severity,
        "attempt_count": attempt_count,
        "success_count": success_count,
        "success_rate_pct": success_rate,
        "threshold_pct": floor_pct,
        "message": f"rebalance success rate {success_rate}% over last 7 days",
    }


def _is_settled_forward(forward: Mapping[str, Any]) -> bool:
    return str(forward.get("status", "")).lower() == "settled"


def _fee_sats(forward: Mapping[str, Any]) -> int:
    fee = forward.get("fee_msat")
    parsed = _coerce_int(fee)
    return 0 if parsed is None else parsed


def check_revenue_drop(
    listforwards: Mapping[str, Any],
    t0: str,
    run_date: str | date,
    drop_pct: int,
) -> dict[str, Any]:
    run_end = datetime.combine(_parse_run_date(run_date), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
    t0_dt = _parse_timestamp(t0)
    pre_start = t0_dt - timedelta(days=14)
    pre_end = t0_dt
    post_window_complete = run_end >= t0_dt + timedelta(days=14)
    post_start = max(t0_dt, run_end - timedelta(days=14))

    pre_fees = 0
    post_fees = 0
    for forward in listforwards.get("forwards") or []:
        if not _is_settled_forward(forward):
            continue
        received_at = _unix_from_event(forward, "received_time", "resolved_time")
        if received_at is None:
            continue
        received_dt = datetime.fromtimestamp(received_at, tz=timezone.utc)
        fee_sats = _fee_sats(forward)
        if pre_start <= received_dt < pre_end:
            pre_fees += fee_sats
        if post_start <= received_dt < run_end:
            post_fees += fee_sats

    post_days = max((run_end - post_start).days, 1)
    pre_avg = pre_fees / 14 if pre_fees else 0
    post_avg = post_fees / post_days if post_fees else 0
    drop_observed = post_window_complete and pre_avg > 0 and post_avg < pre_avg * (1 - drop_pct / 100)
    severity = "red" if drop_observed else "green"
    return {
        "rule": "revenue_drop",
        "severity": severity,
        "pre_14d_fee_sats": pre_fees,
        "post_window_fee_sats": post_fees,
        "post_window_days": post_days,
        "window_complete": post_window_complete,
        "pre_avg_sats_per_day": round(pre_avg, 2),
        "post_avg_sats_per_day": round(post_avg, 2),
        "threshold_pct": drop_pct,
        "message": (
            "routing revenue comparison vs pre-deploy trailing 14-day average"
            if post_window_complete
            else "post-deploy revenue window incomplete; deferring drop check"
        ),
    }


def check_traceback_volume(lines: list[str], threshold: int = 10) -> dict[str, Any]:
    error_lines = [
        line
        for line in lines
        if any(token in line for token in ("Traceback", "TypeError", "AttributeError", "KeyError"))
    ]
    severity = "yellow" if len(error_lines) > threshold else "green"
    return {
        "rule": "traceback_volume",
        "severity": severity,
        "count": len(error_lines),
        "threshold": threshold,
        "message": f"{len(error_lines)} traceback/error lines in extracted log window",
    }


def check_rebalance_floor_volume(lines: list[str], threshold: int = 100) -> dict[str, Any]:
    count = sum(1 for line in lines if "REBALANCE_FLOOR" in line)
    severity = "yellow" if count > threshold else "green"
    return {
        "rule": "rebalance_floor_volume",
        "severity": severity,
        "count": count,
        "threshold": threshold,
        "message": f"{count} REBALANCE_FLOOR lines in extracted log window",
    }


def check_competition_aware_oscillation(lines: list[str]) -> dict[str, Any]:
    preserve_scids: set[str] = set()
    undercut_scids: set[str] = set()
    for line in lines:
        lowered = line.lower()
        matches = set(SCID_RE.findall(line))
        if "competition_aware preserve" in lowered:
            preserve_scids.update(matches)
        if "competition_aware undercut" in lowered:
            undercut_scids.update(matches)
    oscillating = sorted(preserve_scids & undercut_scids)
    severity = "yellow" if oscillating else "green"
    return {
        "rule": "competition_aware_oscillation",
        "severity": severity,
        "count": len(oscillating),
        "channels": oscillating,
        "message": f"{len(oscillating)} channels showed preserve/undercut oscillation",
    }


def checkpoint_state(t0: str, run_date: str | date) -> str:
    days_since_t0 = (_parse_run_date(run_date) - _parse_timestamp(t0).date()).days
    if days_since_t0 < 14:
        return "pre_t14"
    if days_since_t0 == 14:
        return "ready_t14"
    if days_since_t0 < 28:
        return "between_t14_t28"
    if days_since_t0 == 28:
        return "ready_t28"
    return "post_t28"


def evaluate_node_day(
    node_name: str,
    node_cfg: Mapping[str, Any],
    day_dir: str | Path,
    manifest_status: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    run_date: str | date,
) -> dict[str, Any]:
    if manifest_status.get("status") != "ok":
        findings = [
            {
                "rule": "collection_failure",
                "severity": "red",
                "message": f"daily collection failed: {manifest_status.get('errors', {})}",
            }
        ]
        return {
            "status": "collection_failed",
            "checkpoint_state": checkpoint_state(node_cfg["t0"], run_date),
            "highest_severity": "red",
            "findings": findings,
        }

    node_path = Path(day_dir)
    peerchannels = _read_json(node_path / "listpeerchannels.json")
    hive_members = _read_json(node_path / "hive-members.json")
    listpays = _read_json(node_path / "listpays.json")
    listforwards = _read_json(node_path / "listforwards.json")
    revenue_config = _read_json(node_path / "revenue-config.json")
    log_lines = _read_lines(node_path / "rollback-watch.log")
    if not log_lines:
        log_lines = _read_lines(node_path / "debug-log-extract.log")
    daily_log_lines = _daily_log_lines(log_lines, run_date)

    restart_finding = check_plugin_restart_count(daily_log_lines, int(thresholds["plugin_restart_limit_24h"]))
    if _parse_run_date(run_date) == _parse_timestamp(node_cfg["t0"]).date():
        restart_finding["severity"] = "green"
        restart_finding["message"] = "deploy-day restart check suppressed on T0"

    findings = [
        restart_finding,
        check_zero_ppm_non_hive(peerchannels, hive_members),
        check_ceiling_pricing(peerchannels, hive_members, revenue_config),
        check_rebalance_success_rate(listpays, run_date, int(thresholds["rebalance_success_floor_pct"])),
        check_revenue_drop(listforwards, node_cfg["t0"], run_date, int(thresholds["revenue_drop_pct"])),
        check_traceback_volume(daily_log_lines),
        check_rebalance_floor_volume(daily_log_lines),
        check_competition_aware_oscillation(daily_log_lines),
    ]

    return {
        "status": "ok",
        "checkpoint_state": checkpoint_state(node_cfg["t0"], run_date),
        "highest_severity": _severity_max(findings),
        "findings": findings,
    }


def evaluate_all_nodes(config: Mapping[str, Any], run_date: str | date) -> dict[str, Any]:
    results_root = config["paths"]["results_root"]
    manifest = _read_json(_manifest_path(results_root, run_date))
    thresholds = config["thresholds"]["rollback"]

    findings: dict[str, Any] = {
        "date": str(run_date),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "green",
        "nodes": {},
    }

    for node_name, node_cfg in config["nodes"].items():
        node_manifest = (manifest.get("nodes") or {}).get(node_name, {"status": "missing"})
        node_result = evaluate_node_day(
            node_name=node_name,
            node_cfg=node_cfg,
            day_dir=_node_day_dir(config, run_date, node_name),
            manifest_status=node_manifest,
            thresholds=thresholds,
            run_date=run_date,
        )
        findings["nodes"][node_name] = node_result
        if SEVERITY_RANK[node_result["highest_severity"]] > SEVERITY_RANK[findings["status"]]:
            findings["status"] = node_result["highest_severity"]

    common.write_json(_watch_path(results_root, run_date), findings)
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate daily revenue validation watch findings.")
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
    findings = evaluate_all_nodes(config, args.date)
    return 1 if findings["status"] == "red" else 0


if __name__ == "__main__":
    sys.exit(main())
