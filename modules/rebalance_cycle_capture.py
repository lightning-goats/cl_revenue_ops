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
import threading
import time
import uuid
from dataclasses import dataclass
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


def _positive(value: Any, default: int = 1) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _nonnegative(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _pair_identity(pair: Any) -> Optional[tuple[str, str]]:
    source = _field(pair, "source_channel_id", "")
    destination = _field(pair, "dest_channel_id", "")
    if not isinstance(source, str) or not source or not isinstance(destination, str) or not destination:
        return None
    return source, destination


def _project_pair(pair: Any, fallback_rank: int) -> Optional[dict]:
    identity = _pair_identity(pair)
    if identity is None:
        return None
    source, destination = identity
    score = _field(pair, "score", 0.0)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = 0.0
    decomposition = _field(pair, "score_decomposition", {})
    if not isinstance(decomposition, dict):
        decomposition = {}
    rejection = _field(pair, "planner_rejection_reason", "")
    if not isinstance(rejection, str):
        rejection = ""
    return {
        "source_channel_id": source,
        "dest_channel_id": destination,
        "planned_amount_sats": _positive(_field(pair, "amount_sats", 0)),
        "pair_budget_sats": _nonnegative(_field(pair, "pair_budget_sats", 0)),
        "source_excess_sats": _positive(_field(pair, "source_excess_sats", 0)),
        "dest_need_sats": _positive(_field(pair, "dest_need_sats", 0)),
        "max_chunk_sats": _positive(_field(pair, "max_chunk_sats", 0)),
        "cheap_rank": _positive(_field(pair, "cheap_rank", fallback_rank), fallback_rank),
        "cheap_score": score,
        "planner_selected": bool(_field(pair, "planner_selected", False)),
        "planner_rejection_reason": rejection or None,
        "bootstrap_score_decomposition": _safe_value(decomposition),
        "score_decomposition": _safe_value(decomposition),
        "route_cost_sats": _nonnegative(_field(pair, "route_cost_sats", 0)),
        "effective_budget_sats": _nonnegative(decomposition.get("effective_budget_sats", _field(pair, "pair_budget_sats", 0))),
        "rejection_reason": _bounded_text(_field(pair, "rejection_reason", "")),
        "route_summary": [
            {key: _safe_value(hop.get(key)) for key in ("channel", "id", "amount_msat", "delay") if key in hop}
            for hop in (_field(pair, "route", []) or [])[:20] if isinstance(hop, dict)
        ],
    }


def _project_identity(pair: Any) -> Optional[dict]:
    identity = _pair_identity(pair)
    if identity is None:
        return None
    return {"source_channel_id": identity[0], "dest_channel_id": identity[1]}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _project_snapshot(snapshot: Any) -> dict:
    channels = _field(snapshot, "channels", []) if snapshot is not None else []
    if not isinstance(channels, (list, tuple)):
        channels = []
    if len(channels) > 1024:
        raise ValueError("normalized snapshot exceeds capture bound")
    projected_channels = []
    identities = set()
    for channel in list(channels)[:1024]:
        channel_id = _field(channel, "channel_id", "")
        if not isinstance(channel_id, str) or not channel_id or channel_id in identities:
            continue
        identities.add(channel_id)
        peer_id = _field(channel, "peer_id", "")
        value_class = _field(channel, "value_class", "neutral")
        budget_source = _field(channel, "budget_source", "none")
        projected_channels.append({
            "channel_id": channel_id, "peer_id": peer_id if isinstance(peer_id, str) else "",
            "capacity_sats": _nonnegative(_field(channel, "capacity_sats", 0)),
            "local_ratio": _number(_field(channel, "local_ratio", 0.0)),
            "actual_inbound_fee_ppm": _nonnegative(_field(channel, "actual_inbound_fee_ppm", 0)),
            "value_class": value_class if isinstance(value_class, str) else "neutral",
            "is_valuable": bool(_field(channel, "is_valuable", False)),
            "remaining_budget_sats": _nonnegative(_field(channel, "remaining_budget_sats", 0)),
            "cooldown_active": bool(_field(channel, "cooldown_active", False)),
            "source_eligible": bool(_field(channel, "source_eligible", False)),
            "dest_eligible": bool(_field(channel, "dest_eligible", False)),
            "source_reason": _bounded_text(_field(channel, "source_reason", "")),
            "dest_reason": _bounded_text(_field(channel, "dest_reason", "")),
            "dest_urgency": _number(_field(channel, "dest_urgency", 0.0)),
            "source_drain_score": _number(_field(channel, "source_drain_score", 0.0)),
            "budget_source": budget_source if isinstance(budget_source, str) else "none",
            "local_out_fee_ppm": _nonnegative(_field(channel, "local_out_fee_ppm", 0)),
            "historical_direct_fee_ppm": _number(_field(channel, "historical_direct_fee_ppm", 0.0)),
            "historical_sourced_fee_ppm": _number(_field(channel, "historical_sourced_fee_ppm", 0.0)),
            "is_active": bool(_field(channel, "is_active", False)),
            "realized_utilization": _number(_field(channel, "realized_utilization", 0.0)),
            "utilization_is_realized": bool(_field(channel, "utilization_is_realized", False)),
            "activity_out_sats": _nonnegative(_field(channel, "activity_out_sats", 0)),
            "activity_in_sats": _nonnegative(_field(channel, "activity_in_sats", 0)),
            "target_band_low": _number(_field(channel, "target_band_low", 0.35)),
            "target_band_high": _number(_field(channel, "target_band_high", 0.65)),
        })
    return {
        "normalized_snapshot": {
            "channels": projected_channels,
            "total_capacity_sats": _nonnegative(_field(snapshot, "total_capacity_sats", 0)),
            "total_remaining_budget_sats": _nonnegative(_field(snapshot, "total_remaining_budget_sats", 0)),
            "valuable_channel_count": _nonnegative(_field(snapshot, "valuable_channel_count", 0)),
        }
    }


def _project_execution_result(value: Any) -> Optional[dict]:
    if not isinstance(value, dict) and not hasattr(value, "success"):
        return None
    route_type = _field(value, "route_type", "")
    error = _field(value, "error", "")
    return {
        "success": bool(_field(value, "success", False)),
        "amount_sats": _nonnegative(_field(value, "amount_sats", 0)),
        "fee_sats": _nonnegative(_field(value, "fee_sats", 0)),
        "fee_msat": _nonnegative(_field(value, "fee_msat", 0)),
        "fee_ppm": _nonnegative(_field(value, "fee_ppm", 0)),
        "attempts": _nonnegative(_field(value, "attempts", 0)),
        "hops": _nonnegative(_field(value, "hops", 0)),
        "parts": _nonnegative(_field(value, "parts", 0)),
        "route_type": route_type[:_MAX_TEXT] if isinstance(route_type, str) else "",
        "error": error[:_MAX_TEXT] if isinstance(error, str) else "",
        "payment_pending": bool(_field(value, "payment_pending", False)),
    }


def _project_outcomes(outcomes: Any, final_identities: set[tuple[str, str]]) -> list[dict]:
    if not isinstance(outcomes, list):
        return []
    projected = []
    for outcome in outcomes[:64]:
        identity = _pair_identity(outcome)
        if identity is None or identity not in final_identities:
            continue
        status = _field(outcome, "status", "returned")
        row = {
            "source_channel_id": identity[0],
            "dest_channel_id": identity[1],
            "status": status[:_MAX_TEXT] if isinstance(status, str) else "malformed",
        }
        result = _project_execution_result(_field(outcome, "result", None))
        if result is not None:
            row["result"] = result
        projected.append(row)
    return projected


def _configuration(value: Any) -> dict:
    value = value if isinstance(value, dict) else {}
    return {
        "config_version": _positive(value.get("config_version", 1)),
        "target_band_low": value.get("target_band_low", 0.35),
        "target_band_high": value.get("target_band_high", 0.65),
        "max_chunk_sats": _positive(value.get("max_chunk_sats", 2_000_000)),
        "max_pairs": _positive(value.get("max_pairs", 1)),
        "pair_fee_cap_ppm": _nonnegative(value.get("pair_fee_cap_ppm", 0)),
    }


@dataclass(frozen=True)
class RebalanceCycleCaptureReference:
    capture_run_id: str
    capture_seq: int
    cycle_id: str
    configuration: dict
    producer: dict


def project_cycle_result(reference: Any, result: Any, terminal_stage: str = "completed") -> dict:
    """Pure explicit-field projection of a terminal cycle result.

    Bad observational fields become neutral omissions.  Projection never calls
    a router, reserve, executor, plugin, or database surface.
    """
    generated = []
    seen = set()
    for ordinal, pair in enumerate(_field(result, "considered_candidates", []) or [], 1):
        projected = _project_pair(pair, ordinal)
        if projected is None:
            raise ValueError("malformed generated pair")
        identity = (projected["source_channel_id"], projected["dest_channel_id"])
        if identity in seen:
            raise ValueError("duplicate generated pair")
        seen.add(identity)
        generated.append(projected)

    planner_selected = [_project_identity(pair) for pair in generated if pair["planner_selected"]]
    generated_ids = {(row["source_channel_id"], row["dest_channel_id"]) for row in generated}
    planner_ids = {(row["source_channel_id"], row["dest_channel_id"]) for row in planner_selected}
    final_selected = []
    final_seen = set()
    for pair in _field(result, "candidates", []) or []:
        row = _project_pair(pair, 1)
        if row is None:
            raise ValueError("malformed final selected pair")
        identity = (row["source_channel_id"], row["dest_channel_id"])
        if identity not in generated_ids or identity not in planner_ids or identity in final_seen:
            raise ValueError("invalid final selected pair relation")
        final_seen.add(identity)
        final_selected.append(row)
    final_ids = {(row["source_channel_id"], row["dest_channel_id"]) for row in final_selected}
    skipped = []
    for skip in list(_field(result, "audit_records", []) or [])[:128]:
        channel_id = _field(skip, "channel_id", "")
        reason = _field(skip, "reason", "")
        if isinstance(channel_id, str) and isinstance(reason, str):
            skipped.append({
                "channel_id": channel_id[:_MAX_TEXT], "reason": reason[:_MAX_TEXT],
                "detail": _bounded_text(_field(skip, "detail", "")),
            })

    producer = dict(_field(reference, "producer", {}) or {})
    producer = {
        "started_at": _bounded_text(producer.get("started_at") or _utc_now()),
        "completed_at": _bounded_text(producer.get("completed_at") or _utc_now()),
        "python_commit": _bounded_text(producer.get("python_commit") or "unknown"),
        "algorithm_version": _bounded_text(producer.get("algorithm_version") or "rebalance-v2-phase1a"),
        "trigger": _bounded_text(producer.get("trigger") or "unknown"),
    }
    outcomes = _project_outcomes(_field(result, "pair_outcomes", []), final_ids)
    return {
        "schema_name": "rebalance_cycle_replay",
        "schema_version": SCHEMA_VERSION,
        "capture_run_id": _bounded_text(_field(reference, "capture_run_id", "")),
        "capture_seq": _positive(_field(reference, "capture_seq", 1)),
        "cycle_id": _bounded_text(_field(reference, "cycle_id", "")),
        "producer": producer,
        "configuration": _configuration(_field(reference, "configuration", {})),
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
        self._enabled = False
        self._writer: Optional[threading.Thread] = None
        self._run_id: Optional[str] = None
        self._next_seq = 0
        self._active = set()
        self._inflight = 0
        self._manifest: Optional[dict] = None

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
                configuration = _configuration(copy.deepcopy(configuration() if callable(configuration) else configuration))
                producer = copy.deepcopy(producer() if callable(producer) else producer)
                if not isinstance(producer, dict):
                    return None
                self._next_seq += 1
                sequence = self._next_seq
                cycle_id = f"{self._run_id}:{sequence:08d}"
                reference = RebalanceCycleCaptureReference(self._run_id, sequence, cycle_id, configuration, producer)
                self._active.add(sequence)
                self._manifest["attempted"] += 1
                self._manifest["attempts"].append({"capture_seq": sequence, "cycle_id": cycle_id, "status": "active"})
                # Durability is writer-owned; this hot path only allocates an identity.
                return reference
        except Exception:
            return None

    def finish_cycle(self, reference: Any, result: Any, terminal_stage: str = "completed") -> None:
        try:
            with self._condition:
                sequence = _field(reference, "capture_seq", None)
                if not isinstance(sequence, int) or sequence not in self._active:
                    return
                self._active.remove(sequence)
                attempt = self._attempt_locked(sequence)
                if attempt is None:
                    return
                # Keep the engine path bounded: only a non-blocking handoff.
                try:
                    self._queue.put_nowait((reference, result, terminal_stage, attempt))
                    attempt["status"] = "queued"
                except queue.Full:
                    attempt.update(status="dropped", error_category="queue_full")
                    self._manifest["dropped"] += 1
                except Exception as exc:
                    self._fail_locked(attempt, exc)
                self._condition.notify_all()
        except Exception:
            return

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
                    return False
                self._condition.wait(remaining)
            try:
                self._queue.put_nowait(_DRAIN_SENTINEL)
            except queue.Full:
                return False
        writer = self._writer
        if writer is not None:
            writer.join(max(0.0, deadline - time.monotonic()))
            if writer.is_alive():
                return False
        with self._condition:
            if self._manifest is not None:
                self._manifest["state"] = "closed"
                self._publish_manifest_locked()
            return True

    def _writer_main(self) -> None:
        while True:
            item = self._queue.get()
            if item is _DRAIN_SENTINEL:
                self._queue.task_done()
                return
            reference, result, terminal_stage, attempt = item
            with self._condition:
                self._inflight += 1
                attempt["status"] = "writing"
            try:
                payload = project_cycle_result(reference, result, terminal_stage)
                self._publish_envelope(reference, payload)
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
                self._queue.task_done()

    def _publish_envelope(self, reference: Any, body: dict) -> None:
        envelope = seal_envelope(body)
        serialized = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(serialized) > MAX_ENVELOPE_BYTES:
            raise ValueError("complete sealed envelope exceeds 32 MiB")
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
