from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class RunResult:
    command: list[str]
    ok: bool
    stdout: str
    stderr: str
    returncode: int


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _utc_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must resolve to UTC")
    return value


def evaluation_identity(node_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = node_cfg.get("evaluation")
    if raw is None:
        t0 = _utc_timestamp(node_cfg.get("t0"), "t0")
        return {
            "id": None,
            "version": None,
            "state": "legacy_unversioned",
            "formal_window_active": False,
            "t0": t0,
        }
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation must be an object")
    evaluation_id = raw.get("id")
    version = raw.get("version")
    state = raw.get("state")
    formal_window_active = raw.get("formal_window_active")
    if not isinstance(evaluation_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", evaluation_id
    ):
        raise ValueError(
            "evaluation.id must use the filename-safe form "
            "[A-Za-z0-9][A-Za-z0-9._-]*"
        )
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("evaluation.version must be a positive integer")
    if state not in {"preflight", "active", "closed"}:
        raise ValueError("evaluation.state must be preflight, active, or closed")
    if not isinstance(formal_window_active, bool):
        raise ValueError("evaluation.formal_window_active must be boolean")
    if formal_window_active and state != "active":
        raise ValueError("only an active evaluation may have a formal window")
    return {
        "id": evaluation_id,
        "version": version,
        "state": state,
        "formal_window_active": formal_window_active,
        "t0": _utc_timestamp(raw.get("t0"), "evaluation.t0"),
    }


def evaluation_t0(node_cfg: Mapping[str, Any]) -> str:
    return str(evaluation_identity(node_cfg)["t0"])


def build_node_command(node_cfg: Mapping[str, Any], remote_cmd: str) -> list[str]:
    transport = list(node_cfg["transport"])
    return [*transport, remote_cmd]


def _day_string(run_date: str | date) -> str:
    if isinstance(run_date, date):
        return run_date.isoformat()
    return str(run_date)


def dated_results_dir(results_root: str | Path, run_date: str | date, node_name: str) -> Path:
    return Path(results_root) / _day_string(run_date) / node_name


def node_day_dir(config: Mapping[str, Any], run_date: str | date, node_name: str) -> Path:
    return dated_results_dir(config["paths"]["results_root"], run_date, node_name)


def trends_file(results_root: str | Path, node_name: str) -> Path:
    return Path(results_root) / "trends" / f"{node_name}.jsonl"


def write_json_file(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json(path: str | Path, payload: Any) -> None:
    write_json_file(path, payload)
