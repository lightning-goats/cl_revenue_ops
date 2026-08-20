"""Bounded, observational capture of terminal rebalance-cycle results.

This module deliberately has no plugin or RPC dependency.  Capture failures
are local evidence failures: callers receive no exception and retain their
existing rebalance result.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from modules.rebalance_cycle_replay_wire import (
    MAX_ENVELOPE_BYTES,
    SCHEMA_VERSION,
    seal_envelope,
)


RETENTION_MAX_FILES = 32
RETENTION_MAX_BYTES = 256 * 1024 * 1024
WRITER_QUEUE_SIZE = 2
MAX_SNAPSHOT_CHANNELS = 1024
MAX_GENERATED_PAIRS = 4096
MAX_FINAL_PAIRS = 64
MAX_OUTCOMES = 64
MAX_SKIPS = 128
_MAX_TEXT = 512
_DRAIN_SENTINEL = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_text(value: Any) -> str:
    try:
        text = str(value)
    except Exception:
        return "unprintable"
    return text[:_MAX_TEXT]


def _safe_value(value: Any, depth: int = 0) -> Any:
    """Return a bounded primitive-only observation; never inspect ``__dict__``."""
    if depth >= 4:
        return "truncated"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth + 1) for item in value[:32]]
    if isinstance(value, dict):
        return {
            _bounded_text(key): _safe_value(item, depth + 1)
            for key, item in list(value.items())[:32]
        }
    return {"unsupported_type": type(value).__name__}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _required(value: Any, name: str, label: str) -> Any:
    if isinstance(value, dict):
        if name not in value:
            raise ValueError(f"{label}.{name} is required")
        return value[name]
    if not hasattr(value, name):
        raise ValueError(f"{label}.{name} is required")
    return getattr(value, name)


def _strict_string(value: Any, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{label} must be a {qualifier}string")
    return value[:_MAX_TEXT]


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _strict_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {kind}")
    return value


def _strict_number(value: Any, label: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return value


def _bounded_sequence(value: Any, label: str, maximum: int) -> list | tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a bounded sequence")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds capture bound")
    return value


def _pair_identity(pair: Any) -> tuple[str, str]:
    source = _strict_string(
        _required(pair, "source_channel_id", "pair"),
        "pair.source_channel_id", nonempty=True,
    )
    destination = _strict_string(
        _required(pair, "dest_channel_id", "pair"),
        "pair.dest_channel_id", nonempty=True,
    )
    return source, destination


def _project_route(route: Any) -> list[dict]:
    if route is None:
        return []
    route = _bounded_sequence(route, "pair.route", 20)
    projected = []
    for index, hop in enumerate(route):
        if not isinstance(hop, dict):
            raise ValueError("pair.route hop must be an object")
        projected.append({
            "index": index,
            "channel": _strict_string(hop.get("channel", ""), "route.channel"),
            "direction": _safe_value(hop.get("direction")),
            "id": _strict_string(hop.get("id", ""), "route.id"),
            "amount_msat": _safe_value(hop.get("amount_msat")),
            "delay": _safe_value(hop.get("delay")),
        })
    return projected


def _effective_budget(pair: Any, decomposition: dict) -> Optional[int]:
    if isinstance(pair, dict):
        explicit_present = "effective_budget_sats" in pair
    else:
        explicit_present = hasattr(pair, "effective_budget_sats")
    if explicit_present:
        return _strict_int(
            _field(pair, "effective_budget_sats"),
            "pair.effective_budget_sats",
        )
    inputs = decomposition.get("inputs")
    if isinstance(inputs, dict) and "effective_budget_sats" in inputs:
        return _strict_int(
            inputs["effective_budget_sats"],
            "pair.score_decomposition.inputs.effective_budget_sats",
        )
    return None


def _project_pair(pair: Any, bootstrap: dict) -> dict:
    source, destination = _pair_identity(pair)
    decomposition = _required(pair, "score_decomposition", "pair")
    if not isinstance(decomposition, dict):
        raise ValueError("pair.score_decomposition must be an object")
    rejection = _required(pair, "planner_rejection_reason", "pair")
    if not isinstance(rejection, str):
        raise ValueError("pair.planner_rejection_reason must be a string")
    planner_selected = _strict_bool(
        _required(pair, "planner_selected", "pair"),
        "pair.planner_selected",
    )
    route_cost = _field(pair, "route_cost_sats", None)
    if route_cost is not None:
        route_cost = _strict_int(route_cost, "pair.route_cost_sats")
    return {
        "source_channel_id": source,
        "dest_channel_id": destination,
        "planned_amount_sats": _strict_int(
            _required(pair, "amount_sats", "pair"), "pair.amount_sats", positive=True,
        ),
        "pair_budget_sats": _strict_int(
            _required(pair, "pair_budget_sats", "pair"), "pair.pair_budget_sats",
        ),
        "source_excess_sats": _strict_int(
            _required(pair, "source_excess_sats", "pair"), "pair.source_excess_sats", positive=True,
        ),
        "dest_need_sats": _strict_int(
            _required(pair, "dest_need_sats", "pair"), "pair.dest_need_sats", positive=True,
        ),
        "max_chunk_sats": _strict_int(
            _required(pair, "max_chunk_sats", "pair"), "pair.max_chunk_sats", positive=True,
        ),
        "cheap_rank": _strict_int(
            _required(pair, "cheap_rank", "pair"), "pair.cheap_rank", positive=True,
        ),
        "cheap_score": _strict_number(
            _required(pair, "score", "pair"), "pair.score",
        ),
        "planner_selected": planner_selected,
        "planner_rejection_reason": rejection or None,
        "bootstrap_score_decomposition": _safe_value(bootstrap),
        "score_decomposition": _safe_value(decomposition),
        "route_cost_sats": route_cost,
        "effective_budget_sats": _effective_budget(pair, decomposition),
        "rejection_reason": _strict_string(
            _required(pair, "rejection_reason", "pair"), "pair.rejection_reason",
        ),
        "route_summary": _project_route(_required(pair, "route", "pair")),
    }


def _project_identity(pair: Any) -> dict:
    source, destination = _pair_identity(pair)
    return {"source_channel_id": source, "dest_channel_id": destination}


_CHANNEL_FIELDS = (
    "channel_id", "peer_id", "capacity_sats", "local_ratio",
    "actual_inbound_fee_ppm", "value_class", "is_valuable",
    "remaining_budget_sats", "cooldown_active", "source_eligible",
    "dest_eligible", "source_reason", "dest_reason", "dest_urgency",
    "source_drain_score", "budget_source", "local_out_fee_ppm",
    "historical_direct_fee_ppm", "historical_sourced_fee_ppm", "is_active",
    "realized_utilization", "utilization_is_realized", "activity_out_sats",
    "activity_in_sats", "target_band_low", "target_band_high",
)


def _project_snapshot(snapshot: Any) -> dict:
    if snapshot is None:
        raise ValueError("snapshot is required")
    channels = _bounded_sequence(
        _required(snapshot, "channels", "snapshot"),
        "snapshot.channels", MAX_SNAPSHOT_CHANNELS,
    )
    projected_channels = []
    identities = set()
    for index, channel in enumerate(channels):
        label = f"snapshot channel {index}"
        values = {name: _required(channel, name, label) for name in _CHANNEL_FIELDS}
        channel_id = _strict_string(values["channel_id"], f"{label}.channel_id", nonempty=True)
        if channel_id in identities:
            raise ValueError("duplicate snapshot channel identity")
        identities.add(channel_id)
        projected_channels.append({
            "channel_id": channel_id,
            "peer_id": _strict_string(values["peer_id"], f"{label}.peer_id", nonempty=True),
            "capacity_sats": _strict_int(values["capacity_sats"], f"{label}.capacity_sats"),
            "local_ratio": _strict_number(values["local_ratio"], f"{label}.local_ratio"),
            "actual_inbound_fee_ppm": _strict_int(values["actual_inbound_fee_ppm"], f"{label}.actual_inbound_fee_ppm"),
            "value_class": _strict_string(values["value_class"], f"{label}.value_class", nonempty=True),
            "is_valuable": _strict_bool(values["is_valuable"], f"{label}.is_valuable"),
            "remaining_budget_sats": _strict_int(values["remaining_budget_sats"], f"{label}.remaining_budget_sats"),
            "cooldown_active": _strict_bool(values["cooldown_active"], f"{label}.cooldown_active"),
            "source_eligible": _strict_bool(values["source_eligible"], f"{label}.source_eligible"),
            "dest_eligible": _strict_bool(values["dest_eligible"], f"{label}.dest_eligible"),
            "source_reason": _strict_string(values["source_reason"], f"{label}.source_reason"),
            "dest_reason": _strict_string(values["dest_reason"], f"{label}.dest_reason"),
            "dest_urgency": _strict_number(values["dest_urgency"], f"{label}.dest_urgency"),
            "source_drain_score": _strict_number(values["source_drain_score"], f"{label}.source_drain_score"),
            "budget_source": _strict_string(values["budget_source"], f"{label}.budget_source", nonempty=True),
            "local_out_fee_ppm": _strict_int(values["local_out_fee_ppm"], f"{label}.local_out_fee_ppm"),
            "historical_direct_fee_ppm": _strict_number(values["historical_direct_fee_ppm"], f"{label}.historical_direct_fee_ppm"),
            "historical_sourced_fee_ppm": _strict_number(values["historical_sourced_fee_ppm"], f"{label}.historical_sourced_fee_ppm"),
            "is_active": _strict_bool(values["is_active"], f"{label}.is_active"),
            "realized_utilization": _strict_number(values["realized_utilization"], f"{label}.realized_utilization"),
            "utilization_is_realized": _strict_bool(values["utilization_is_realized"], f"{label}.utilization_is_realized"),
            "activity_out_sats": _strict_int(values["activity_out_sats"], f"{label}.activity_out_sats"),
            "activity_in_sats": _strict_int(values["activity_in_sats"], f"{label}.activity_in_sats"),
            "target_band_low": _strict_number(values["target_band_low"], f"{label}.target_band_low"),
            "target_band_high": _strict_number(values["target_band_high"], f"{label}.target_band_high"),
        })
    return {"normalized_snapshot": {
        "channels": projected_channels,
        "total_capacity_sats": _strict_int(
            _required(snapshot, "total_capacity_sats", "snapshot"),
            "snapshot.total_capacity_sats",
        ),
        "total_remaining_budget_sats": _strict_int(
            _required(snapshot, "total_remaining_budget_sats", "snapshot"),
            "snapshot.total_remaining_budget_sats",
        ),
        "valuable_channel_count": _strict_int(
            _required(snapshot, "valuable_channel_count", "snapshot"),
            "snapshot.valuable_channel_count",
        ),
    }}


_SAFE_FAILURE_FIELDS = frozenset({
    "failure_class", "failure_kind", "failure_code", "erring_channel",
    "erring_direction", "erring_node", "wire_message",
    "retry_excluded_channels", "previous_failure", "partial_fill",
})


def _project_failure_data(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        key: _safe_value(value[key])
        for key in _SAFE_FAILURE_FIELDS
        if key in value
    }


def _project_execution_result(value: Any) -> Optional[dict]:
    if not isinstance(value, dict) and not hasattr(value, "success"):
        return None
    excluded = _field(value, "excluded_channels", [])
    if not isinstance(excluded, (list, tuple)):
        excluded = []
    return {
        "success": bool(_field(value, "success", False)),
        "amount_sats": max(0, int(_field(value, "amount_sats", 0) or 0)),
        "fee_sats": max(0, int(_field(value, "fee_sats", 0) or 0)),
        "fee_msat": max(0, int(_field(value, "fee_msat", 0) or 0)),
        "fee_ppm": max(0, int(_field(value, "fee_ppm", 0) or 0)),
        "attempts": max(0, int(_field(value, "attempts", 0) or 0)),
        "hops": max(0, int(_field(value, "hops", 0) or 0)),
        "parts": max(0, int(_field(value, "parts", 0) or 0)),
        "route_type": _bounded_text(_field(value, "route_type", "")),
        "error": _bounded_text(_field(value, "error", "")),
        "excluded_channels": [_bounded_text(item) for item in excluded[:32]],
        "failure_data": _project_failure_data(_field(value, "failure_data", {})),
        "payment_pending": bool(_field(value, "payment_pending", False)),
    }


def _project_outcomes(outcomes: Any, final_identities: set[tuple[str, str]]) -> list[dict]:
    outcomes = _bounded_sequence(outcomes, "pair_outcomes", MAX_OUTCOMES)
    projected = []
    seen = set()
    for outcome in outcomes:
        identity = _pair_identity(outcome)
        if identity not in final_identities:
            raise ValueError("execution outcome is absent from final selection")
        if identity in seen:
            raise ValueError("duplicate execution outcome")
        seen.add(identity)
        status = _strict_string(_field(outcome, "status", "returned"), "outcome.status", nonempty=True)
        row = {"source_channel_id": identity[0], "dest_channel_id": identity[1], "status": status}
        result = _project_execution_result(_field(outcome, "result", None))
        if result is not None:
            row["result"] = result
        projected.append(row)
    return projected


def _configuration(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    expected = {
        "config_version", "target_band_low", "target_band_high",
        "max_chunk_sats", "max_pairs", "pair_fee_cap_ppm",
    }
    if set(value) != expected:
        raise ValueError("configuration fields are incomplete or unknown")
    return {
        "config_version": _strict_int(value["config_version"], "configuration.config_version", positive=True),
        "target_band_low": _strict_number(value["target_band_low"], "configuration.target_band_low"),
        "target_band_high": _strict_number(value["target_band_high"], "configuration.target_band_high"),
        "max_chunk_sats": _strict_int(value["max_chunk_sats"], "configuration.max_chunk_sats", positive=True),
        "max_pairs": _strict_int(value["max_pairs"], "configuration.max_pairs", positive=True),
        "pair_fee_cap_ppm": _strict_int(value["pair_fee_cap_ppm"], "configuration.pair_fee_cap_ppm"),
    }


def _preflight_cycle_result(result: Any) -> None:
    snapshot = _required(result, "snapshot", "cycle result")
    if snapshot is None:
        raise ValueError("snapshot is required")
    _bounded_sequence(_required(snapshot, "channels", "snapshot"), "snapshot.channels", MAX_SNAPSHOT_CHANNELS)
    _bounded_sequence(_required(result, "considered_candidates", "cycle result"), "considered_candidates", MAX_GENERATED_PAIRS)
    _bounded_sequence(_required(result, "candidates", "cycle result"), "candidates", MAX_FINAL_PAIRS)
    _bounded_sequence(_required(result, "pair_outcomes", "cycle result"), "pair_outcomes", MAX_OUTCOMES)
    _bounded_sequence(_required(result, "audit_records", "cycle result"), "audit_records", MAX_SKIPS)
    _bounded_sequence(_required(result, "planner_bootstrap_evidence", "cycle result"), "planner_bootstrap_evidence", MAX_GENERATED_PAIRS)


@dataclass(frozen=True)
class RebalanceCycleCaptureReference:
    capture_run_id: str
    capture_seq: int
    cycle_id: str
    configuration: dict
    producer: dict


def project_cycle_result(reference: Any, result: Any, terminal_stage: str = "completed") -> dict:
    """Project one explicit, bounded, strict terminal observation."""
    allowed_stages = {
        "completed", "planning_only", "no_router", "missing_snapshot",
        "failed", "lock_contended",
    }
    if terminal_stage not in allowed_stages:
        raise ValueError("terminal_stage is invalid")
    _preflight_cycle_result(result)

    bootstrap_rows = _field(result, "planner_bootstrap_evidence", [])
    bootstrap_by_identity = {}
    for row in bootstrap_rows:
        identity = _pair_identity(row)
        if identity in bootstrap_by_identity:
            raise ValueError("duplicate planner bootstrap evidence")
        decomposition = _required(row, "score_decomposition", "planner bootstrap evidence")
        if not isinstance(decomposition, dict):
            raise ValueError("planner bootstrap score_decomposition must be an object")
        bootstrap_by_identity[identity] = decomposition

    generated = []
    seen = set()
    ranks = set()
    for pair in _field(result, "considered_candidates", []):
        identity = _pair_identity(pair)
        if identity not in bootstrap_by_identity:
            raise ValueError("missing planner bootstrap evidence")
        projected = _project_pair(pair, bootstrap_by_identity[identity])
        if identity in seen:
            raise ValueError("duplicate generated pair")
        if projected["cheap_rank"] in ranks:
            raise ValueError("duplicate generated pair rank")
        seen.add(identity)
        ranks.add(projected["cheap_rank"])
        generated.append(projected)
    if set(bootstrap_by_identity) != seen:
        raise ValueError("orphan planner bootstrap evidence")
    if ranks != set(range(1, len(generated) + 1)):
        raise ValueError("generated pair ranks must be contiguous beginning at 1")

    planner_selected = [
        _project_identity(pair) for pair in generated if pair["planner_selected"]
    ]
    generated_ids = {
        (row["source_channel_id"], row["dest_channel_id"]) for row in generated
    }
    planner_ids = {
        (row["source_channel_id"], row["dest_channel_id"])
        for row in planner_selected
    }
    final_selected = []
    final_seen = set()
    for pair in _field(result, "candidates", []):
        identity = _pair_identity(pair)
        if identity not in generated_ids or identity not in planner_ids or identity in final_seen:
            raise ValueError("invalid final selected pair relation")
        final_seen.add(identity)
        final_selected.append(_project_pair(pair, bootstrap_by_identity[identity]))
    final_ids = {
        (row["source_channel_id"], row["dest_channel_id"])
        for row in final_selected
    }

    skipped = []
    for skip in _field(result, "audit_records", []):
        skipped.append({
            "channel_id": _strict_string(_required(skip, "channel_id", "skip"), "skip.channel_id"),
            "reason": _strict_string(_required(skip, "reason", "skip"), "skip.reason", nonempty=True),
            "detail": _bounded_text(_field(skip, "detail", "") or ""),
        })

    producer_value = _field(reference, "producer", None)
    if not isinstance(producer_value, dict):
        raise ValueError("producer must be an object")
    producer = {}
    for field in (
        "started_at", "completed_at", "python_commit",
        "algorithm_version", "trigger",
    ):
        if field not in producer_value:
            raise ValueError(f"producer.{field} is required")
        producer[field] = _strict_string(
            producer_value[field], f"producer.{field}", nonempty=True,
        )

    outcomes = _project_outcomes(_field(result, "pair_outcomes", []), final_ids)
    return {
        "schema_name": "rebalance_cycle_replay",
        "schema_version": SCHEMA_VERSION,
        "capture_run_id": _strict_string(_required(reference, "capture_run_id", "reference"), "capture_run_id", nonempty=True),
        "capture_seq": _strict_int(_required(reference, "capture_seq", "reference"), "capture_seq", positive=True),
        "cycle_id": _strict_string(_required(reference, "cycle_id", "reference"), "cycle_id", nonempty=True),
        "terminal_stage": terminal_stage,
        "producer": producer,
        "configuration": _configuration(_required(reference, "configuration", "reference")),
        "pre_state": _project_snapshot(_field(result, "snapshot", None)),
        "funnel": {
            "generated_pairs": generated,
            "planner_selected_pairs": planner_selected,
            "final_selected_pairs": final_selected,
            "skipped": skipped,
        },
        "execution": {"pair_outcomes": outcomes},
        "completeness": {
            "generated_pair_count": len(generated),
            "retained_generated_pair_count": len(generated),
            "planner_selected_pair_count": len(planner_selected),
            "final_selected_pair_count": len(final_selected),
            "execution_outcome_count": len(outcomes),
            "candidate_universe_truncated": False,
            "eligible": terminal_stage == "completed",
        },
    }


class RebalanceCycleCaptureManager:
    """One daemon writer with non-blocking, bounded observational intake."""

    def __init__(self, database_path: Any, logger: Any):
        database = Path(os.path.expanduser(os.path.expandvars(os.fspath(database_path))))
        self.output_dir = database.parent / "revenue_ops_rebalance_replay"
        self._logger = logger
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._queue: queue.Queue = queue.Queue(maxsize=WRITER_QUEUE_SIZE)
        self._slots = threading.BoundedSemaphore(WRITER_QUEUE_SIZE)
        self.python_commit = self._discover_python_commit()
        self._enabled = False
        self._writer: Optional[threading.Thread] = None
        self._run_id: Optional[str] = None
        self._next_seq = 0
        self._active = set()
        self._inflight = 0
        self._manifest: Optional[dict] = None

    @staticmethod
    def _discover_python_commit() -> str:
        root = Path(__file__).resolve().parents[1]
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            return completed.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def set_enabled(self, enabled: bool, timeout_seconds: float = 5.0) -> bool:
        try:
            return self._enable() if enabled else self._disable(timeout_seconds)
        except Exception:
            return False

    def begin_cycle(self, configuration: Any, producer: Any) -> Optional[RebalanceCycleCaptureReference]:
        try:
            with self._condition:
                if not self._enabled or self._run_id is None or self._manifest is None:
                    return None
                configuration_value = configuration() if callable(configuration) else configuration
                configuration_copy = _configuration(copy.deepcopy(configuration_value))
                producer_value = producer() if callable(producer) else producer
                producer_copy = copy.deepcopy(producer_value)
                if not isinstance(producer_copy, dict):
                    return None
                producer_copy.setdefault("started_at", _utc_now())
                producer_copy.setdefault("python_commit", self.python_commit)
                producer_copy.setdefault("algorithm_version", "rebalance-v2-phase1a")
                trigger = producer_copy.get("trigger")
                if not isinstance(trigger, str) or not trigger:
                    return None
                allowed = {"started_at", "python_commit", "algorithm_version", "trigger"}
                if set(producer_copy) != allowed:
                    return None
                self._next_seq += 1
                sequence = self._next_seq
                cycle_id = f"{self._run_id}:{sequence:08d}"
                reference = RebalanceCycleCaptureReference(
                    self._run_id, sequence, cycle_id,
                    configuration_copy, producer_copy,
                )
                self._active.add(sequence)
                self._manifest["attempted"] += 1
                self._manifest["attempts"].append({
                    "capture_seq": sequence,
                    "cycle_id": cycle_id,
                    "status": "active",
                    "terminal_stage": None,
                    "eligible": False,
                })
                self._prune_attempts_locked()
                return reference
        except Exception:
            return None

    def finish_cycle(self, reference: Any, result: Any, terminal_stage: str = "completed") -> None:
        slot_acquired = False
        try:
            with self._condition:
                sequence = _field(reference, "capture_seq", None)
                if not isinstance(sequence, int) or sequence not in self._active:
                    return
                self._active.remove(sequence)
                attempt = self._attempt_locked(sequence)
                if attempt is None:
                    return
                attempt["terminal_stage"] = terminal_stage
                if not self._slots.acquire(blocking=False):
                    attempt.update(
                        status="dropped", error_category="queue_full",
                        eligible=False,
                    )
                    self._manifest["dropped"] += 1
                    self._condition.notify_all()
                    return
                slot_acquired = True

            completed_producer = dict(_field(reference, "producer", {}) or {})
            completed_producer["completed_at"] = _utc_now()
            completed_reference = replace(reference, producer=completed_producer)
            _preflight_cycle_result(result)
            body = project_cycle_result(completed_reference, result, terminal_stage)
            envelope = seal_envelope(body)
            serialized = json.dumps(
                envelope, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            if len(serialized) > MAX_ENVELOPE_BYTES:
                raise ValueError("complete sealed envelope exceeds 32 MiB")

            with self._condition:
                self._queue.put_nowait((completed_reference, serialized, attempt))
                attempt["status"] = "queued"
                attempt["eligible"] = bool(body["completeness"]["eligible"])
                slot_acquired = False
                self._condition.notify_all()
        except Exception as exc:
            if slot_acquired:
                self._slots.release()
            try:
                with self._condition:
                    attempt = locals().get("attempt")
                    if isinstance(attempt, dict):
                        self._fail_locked(attempt, exc)
                    self._condition.notify_all()
            except Exception:
                pass


    # Short alias used by callers that treat the capture reference as a lease.
    finish = finish_cycle

    def read_manifest(self) -> dict:
        with self._lock:
            self._prune_attempts_locked()
            return copy.deepcopy(self._manifest) if self._manifest else {}

    def _enable(self) -> bool:
        with self._condition:
            if self._enabled:
                return True
            if self._writer is not None and self._writer.is_alive():
                return False
            self._ensure_output_dir()
            self._run_id = uuid.uuid4().hex
            self._next_seq = 0
            self._active.clear()
            self._queue = queue.Queue(maxsize=WRITER_QUEUE_SIZE)
            self._slots = threading.BoundedSemaphore(WRITER_QUEUE_SIZE)
            self._manifest = {
                "schema_name": "rebalance_cycle_capture_manifest", "schema_version": SCHEMA_VERSION,
                "capture_run_id": self._run_id, "state": "active", "attempted": 0,
                "completed": 0, "failed": 0, "dropped": 0, "writer_health": "healthy",
                "last_error_category": None, "attempts": [], "updated_at": _utc_now(),
            }
            self._enabled = True
            self._writer = threading.Thread(target=self._writer_main, name="rebalance-cycle-capture-writer", daemon=True)
            self._writer.start()
            self._publish_manifest_locked()
            return True

    def _disable(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        with self._condition:
            self._enabled = False
            if self._manifest is None:
                return True
            self._manifest["state"] = "draining"
            self._publish_manifest_locked()
            while self._active or self._inflight or not self._queue.empty():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # A rejected transition must restore intake and config truth.
                    self._enabled = True
                    self._manifest["state"] = "active"
                    self._publish_manifest_locked()
                    return False
                self._condition.wait(remaining)
            try:
                self._queue.put_nowait(_DRAIN_SENTINEL)
            except queue.Full:
                self._enabled = True
                self._manifest["state"] = "active"
                self._publish_manifest_locked()
                return False
        writer = self._writer
        if writer is not None:
            writer.join(max(0.0, deadline - time.monotonic()))
        with self._condition:
            if self._manifest is not None:
                self._manifest["state"] = "closed"
                self._publish_manifest_locked()
            if writer is None or not writer.is_alive():
                self._writer = None
            # Once the sentinel is accepted, intake is disabled coherently even
            # if the daemon needs a little longer to return from its loop.
            return True

    def _writer_main(self) -> None:
        while True:
            item = self._queue.get()
            if item is _DRAIN_SENTINEL:
                self._queue.task_done()
                return
            reference, serialized, attempt = item
            with self._condition:
                self._inflight += 1
                attempt["status"] = "writing"
            try:
                self._publish_envelope(reference, serialized)
                with self._condition:
                    attempt["status"] = "completed"
                    self._manifest["completed"] += 1
            except Exception as exc:
                with self._condition:
                    self._fail_locked(attempt, exc)
            finally:
                with self._condition:
                    self._inflight -= 1
                    self._publish_manifest_locked()
                    self._condition.notify_all()
                self._slots.release()
                self._queue.task_done()

    def _publish_envelope(self, reference: Any, serialized: bytes) -> None:
        filename = f"{reference.capture_run_id}-{reference.capture_seq:08d}-{reference.cycle_id}.json"
        self._atomic_write(self.output_dir / filename, serialized)
        self._rotate_capture_files()

    def _prune_attempts_locked(self) -> None:
        if self._manifest is None:
            return
        attempts = [item for item in self._manifest.get("attempts", []) if isinstance(item, dict)]
        live = [item for item in attempts if item.get("status") in {"active", "queued", "writing"}]
        terminal = [item for item in attempts if item.get("status") not in {"active", "queued", "writing"}]
        terminal.sort(key=lambda item: int(item.get("capture_seq", -1)))
        self._manifest["attempts"] = terminal[-max(0, RETENTION_MAX_FILES - len(live)):] + live
        self._manifest["attempts"].sort(key=lambda item: int(item.get("capture_seq", -1)))

    def _attempt_locked(self, sequence: int) -> Optional[dict]:
        for attempt in self._manifest.get("attempts", []) if self._manifest else []:
            if attempt.get("capture_seq") == sequence:
                return attempt
        return None

    def _fail_locked(self, attempt: dict, exc: Exception) -> None:
        attempt.update(status="failed", error_category=type(exc).__name__, error=_bounded_text(exc))
        if self._manifest is not None:
            self._manifest["failed"] += 1
            self._manifest["writer_health"] = "degraded"
            self._manifest["last_error_category"] = type(exc).__name__

    def _manifest_path(self) -> Path:
        return self.output_dir / f"manifest-{self._run_id}.v{SCHEMA_VERSION}.json"

    def _publish_manifest_locked(self) -> None:
        if self._manifest is None or self._run_id is None:
            return
        self._prune_attempts_locked()
        self._manifest["updated_at"] = _utc_now()
        try:
            self._atomic_write(self._manifest_path(), json.dumps(self._manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            self._rotate_capture_files()
        except Exception as exc:
            self._manifest["writer_health"] = "degraded"
            self._manifest["last_error_category"] = type(exc).__name__

    def _ensure_output_dir(self) -> None:
        if self.output_dir.is_symlink():
            raise OSError("capture output directory is a symlink")
        created = not self.output_dir.exists()
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.output_dir.is_symlink() or not self.output_dir.is_dir():
            raise OSError("capture output path is unsafe")
        os.chmod(self.output_dir, 0o700)
        if created:
            self._fsync_directory(self.output_dir.parent)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        self._ensure_output_dir()
        if path.is_symlink():
            raise OSError("capture destination is a symlink")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_directory(self.output_dir)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _rotate_capture_files(self) -> None:
        candidates = []
        envelope_name = re.compile(r"[0-9a-f]{32}-[0-9]{8}-[0-9a-f]{32}:[0-9]{8}\.json\Z")
        manifest_name = re.compile(r"manifest-[0-9a-f]{32}\.v0\.json\Z")
        for path in self.output_dir.glob("*.json"):
            if path.is_symlink() or not (envelope_name.fullmatch(path.name) or manifest_name.fullmatch(path.name)):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            candidates.append((stat.st_mtime_ns, path.name, stat.st_size, path))
        candidates.sort(reverse=True)
        retained_files = retained_bytes = 0
        removed = False
        for _mtime, _name, size, path in candidates:
            if retained_files >= RETENTION_MAX_FILES or retained_bytes + size > RETENTION_MAX_BYTES:
                path.unlink()
                removed = True
            else:
                retained_files += 1
                retained_bytes += size
        if removed:
            self._fsync_directory(self.output_dir)
