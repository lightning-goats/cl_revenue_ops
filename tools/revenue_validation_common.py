from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
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
