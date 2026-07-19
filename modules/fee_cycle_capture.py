import copy
import json
import os
import queue
import random
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from modules.fee_cycle_replay_wire import (
    MAX_ENVELOPE_BYTES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    seal_envelope,
)


RETENTION_MAX_FILES = 32
RETENTION_MAX_BYTES = 256 * 1024 * 1024


@dataclass
class FeeCycleCaptureSession:
    capture_run_id: str
    capture_seq: int
    cycle_id: str
    producer: dict
    configuration: dict
    pre_state: dict = field(default_factory=dict)
    observations: dict = field(
        default_factory=lambda: {
            "evidence": [],
            "clock": [],
            "entropy": [],
            "governor": [],
            "execution": [],
        }
    )
    expected: dict = field(default_factory=dict)
    invalid_reason: Optional[str] = None

    def record_pre_state(self, value: dict) -> None:
        self.pre_state = copy.deepcopy(value)

    def record_observation(self, family: str, entry: dict) -> None:
        if family not in self.observations:
            self.mark_invalid(f"unknown observation family: {family}")
            return
        self.observations[family].append(copy.deepcopy(entry))

    def record_expected(self, value: dict) -> None:
        self.expected = copy.deepcopy(value)

    def mark_invalid(self, reason: str) -> None:
        if self.invalid_reason is None:
            self.invalid_reason = str(reason)

    def to_body(self) -> dict:
        outcomes = self.expected.get("ordered_outcomes", [])
        channels = self.pre_state.get("ordered_channels", [])
        complete = self.invalid_reason is None and len(outcomes) == len(channels)
        return {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "capture_run_id": self.capture_run_id,
            "capture_seq": self.capture_seq,
            "cycle_id": self.cycle_id,
            "producer": copy.deepcopy(self.producer),
            "started_at": self.producer["started_at"],
            "configuration": copy.deepcopy(self.configuration),
            "pre_state": copy.deepcopy(self.pre_state),
            "observations": copy.deepcopy(self.observations),
            "expected": copy.deepcopy(self.expected),
            "completeness": {
                "evaluated_channels": len(channels),
                "terminal_outcomes": len(outcomes),
                "evidence_entries": len(self.observations["evidence"]),
                "clock_entries": len(self.observations["clock"]),
                "entropy_entries": len(self.observations["entropy"]),
                "complete": complete,
            },
        }


_CURRENT_CAPTURE: ContextVar[Optional[FeeCycleCaptureSession]] = ContextVar(
    "fee_cycle_capture", default=None
)


@contextmanager
def bind_capture(
    session: FeeCycleCaptureSession,
) -> Iterator[FeeCycleCaptureSession]:
    token = _CURRENT_CAPTURE.set(session)
    try:
        yield session
    finally:
        _CURRENT_CAPTURE.reset(token)


def current_capture() -> Optional[FeeCycleCaptureSession]:
    return _CURRENT_CAPTURE.get()


def _observation_ordinal(session: Any, family: str) -> int:
    try:
        entries = session.observations[family]
        return len(entries) if isinstance(entries, list) else 0
    except Exception:
        return 0


def _record_observation_no_throw(session: Any, family: str, entry: Any) -> None:
    if session is None:
        return
    try:
        resolved_entry = (
            entry(_observation_ordinal(session, family))
            if callable(entry)
            else entry
        )
        session.record_observation(family, capture_value(resolved_entry))
    except Exception as exc:
        try:
            session.mark_invalid(
                f"capture recorder failure: {type(exc).__name__}"
            )
        except Exception:
            pass


def capture_value(value: Any) -> Any:
    """Return a detached, replay-wire-safe representation of a live value."""
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, Enum):
        return capture_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: capture_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): capture_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [capture_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((capture_value(item) for item in value), key=repr)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return capture_value(to_dict())
        except Exception:
            pass
    return {"unsupported_type": type(value).__name__}


def record_capture_pre_state(session: Any, value: dict) -> None:
    if session is None:
        return
    try:
        session.record_pre_state(capture_value(value))
    except Exception as exc:
        try:
            session.mark_invalid(f"capture recorder failure: {type(exc).__name__}")
        except Exception:
            pass


def record_capture_expected(session: Any, value: dict) -> None:
    if session is None:
        return
    try:
        session.record_expected(capture_value(value))
    except Exception as exc:
        try:
            session.mark_invalid(f"capture recorder failure: {type(exc).__name__}")
        except Exception:
            pass


def mark_capture_invalid(session: Any, reason: str) -> None:
    if session is None:
        return
    try:
        session.mark_invalid(reason)
    except Exception:
        pass


def record_capture_observation(family: str, entry: Any) -> None:
    session = current_capture()
    if session is None:
        return
    _record_observation_no_throw(session, family, entry)


def record_effective_evidence_result(
    op: Any,
    args: Any,
    result: Any,
    fallback_error: Optional[Exception] = None,
) -> Any:
    """Record one already-computed effective fallback result without rerunning I/O."""
    session = current_capture()
    if session is None:
        return result
    entry = {
        "ordinal": _observation_ordinal(session, "evidence"),
        "op": op,
        "args": args,
        "result": result,
    }
    if fallback_error is not None:
        entry["fallback_error"] = {
            "category": type(fallback_error).__name__,
            "message": _stable_error_message(fallback_error),
        }
    _record_observation_no_throw(session, "evidence", entry)
    return result


def decision_now(label: str) -> int:
    value = int(time.time())
    session = current_capture()
    _record_observation_no_throw(
        session,
        "clock",
        {
            "ordinal": _observation_ordinal(session, "clock"),
            "label": label,
            "value": value,
        },
    )
    return value


def decision_random(label: str) -> float:
    value = random.random()
    session = current_capture()
    _record_observation_no_throw(
        session,
        "entropy",
        {
            "ordinal": _observation_ordinal(session, "entropy"),
            "op": "random",
            "label": label,
            "args": [],
            "result": value,
        },
    )
    return value


def decision_gauss(label: str, mu: float, sigma: float) -> float:
    value = random.gauss(mu, sigma)
    session = current_capture()
    _record_observation_no_throw(
        session,
        "entropy",
        {
            "ordinal": _observation_ordinal(session, "entropy"),
            "op": "gauss",
            "label": label,
            "args": [mu, sigma],
            "result": value,
        },
    )
    return value


def _stable_error_message(exc: Exception) -> str:
    try:
        return str(exc)
    except Exception:
        return "unprintable exception"


def record_effective_evidence(op: Any, args: Any, fn: Any) -> Any:
    session = current_capture()
    if session is None:
        return fn()
    ordinal = _observation_ordinal(session, "evidence")
    try:
        result = fn()
    except Exception as exc:
        _record_observation_no_throw(
            session,
            "evidence",
            {
                "ordinal": ordinal,
                "op": op,
                "args": args,
                "error": {
                    "category": type(exc).__name__,
                    "message": _stable_error_message(exc),
                },
            },
        )
        raise
    _record_observation_no_throw(
        session,
        "evidence",
        {
            "ordinal": ordinal,
            "op": op,
            "args": args,
            "result": result,
        },
    )
    return result


class FeeCycleCaptureManager:
    def __init__(self, database_path: Any, logger: Any):
        expanded = os.path.expandvars(os.path.expanduser(os.fspath(database_path)))
        database = Path(expanded)
        self.output_dir = database.parent / "revenue_ops_fee_replay"
        self._logger = logger
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._writer_thread: Optional[threading.Thread] = None
        self._enabled = False
        self._capture_run_id: Optional[str] = None
        self._next_seq = 0
        self._manifest: Optional[Dict[str, Any]] = None
        self._manifest_path: Optional[Path] = None
        self._active_sessions: Dict[int, str] = {}
        self._queued_count = 0
        self._inflight_seq: Optional[int] = None
        self._retained_sequences = set()
        self._revision = 0
        self._published_revision = 0
        self._publish_blocked_revision = -1
        self._close_retry_revision = -1
        self._durable_closed = False
        self.python_commit = self._discover_python_commit()

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
            revision = completed.stdout.strip()
            return revision or "unknown"
        except Exception:
            return "unknown"

    def set_enabled(self, enabled: bool, timeout_seconds: float = 5.0) -> bool:
        try:
            if enabled:
                return self._enable()
            return self._disable(timeout_seconds)
        except Exception:
            return False

    def begin_cycle(
        self, configuration: Any, producer: dict
    ) -> Optional[FeeCycleCaptureSession]:
        try:
            with self._condition:
                if (
                    not self._enabled
                    or self._manifest is None
                    or self._manifest["state"] != "active"
                    or self._capture_run_id is None
                    or self._active_sessions
                ):
                    return None
                configuration_value = configuration() if callable(configuration) else configuration
                configuration_copy = copy.deepcopy(configuration_value)
                producer_copy = copy.deepcopy(producer)
                if not isinstance(configuration_copy, dict):
                    raise TypeError("capture configuration must be a dict")
                if not isinstance(producer_copy, dict):
                    raise TypeError("capture producer must be a dict")
                producer_copy.setdefault("started_at", _utc_now())
                producer_copy.setdefault("python_commit", self.python_commit)
                capture_seq = self._next_seq + 1
                cycle_id = f"{self._capture_run_id}:{capture_seq:08d}"
                session = FeeCycleCaptureSession(
                    capture_run_id=self._capture_run_id,
                    capture_seq=capture_seq,
                    cycle_id=cycle_id,
                    producer=producer_copy,
                    configuration=configuration_copy,
                )
                self._next_seq = capture_seq
                self._active_sessions[capture_seq] = cycle_id
                self._manifest["attempted"] += 1
                self._manifest["last_attempted_seq"] = capture_seq
                self._manifest["attempts"].append(
                    {
                        "capture_seq": capture_seq,
                        "cycle_id": cycle_id,
                        "status": "active",
                        "eligible": False,
                    }
                )
                self._prune_attempts_locked()
                self._mark_dirty_locked()
                return session
        except Exception:
            return None

    def finish_cycle(self, session: FeeCycleCaptureSession) -> None:
        try:
            self._finish_cycle(session)
        except Exception:
            return

    def read_manifest(self) -> dict:
        with self._condition:
            return copy.deepcopy(self._manifest) if self._manifest else {}

    def _enable(self) -> bool:
        with self._condition:
            if self._enabled:
                return True
            if self._manifest is not None and self._manifest["state"] == "draining":
                return False
            run_id = uuid.uuid4().hex
            now = _utc_now()
            work_queue: queue.Queue = queue.Queue(maxsize=2)
            self._queue = work_queue
            self._capture_run_id = run_id
            self._next_seq = 0
            self._active_sessions = {}
            self._queued_count = 0
            self._inflight_seq = None
            self._retained_sequences = set()
            self._manifest_path = (
                self.output_dir / f"manifest-{run_id}.v{SCHEMA_VERSION}.json"
            )
            self._manifest = {
                "schema_name": "fee_cycle_capture_manifest",
                "schema_version": SCHEMA_VERSION,
                "capture_run_id": run_id,
                "state": "active",
                "queue_drained": False,
                "started_at": now,
                "updated_at": now,
                "attempted": 0,
                "completed": 0,
                "failed": 0,
                "dropped": 0,
                "last_attempted_seq": None,
                "last_completed_seq": None,
                "retained_sequence_range": {"first": None, "last": None},
                "writer_health": "healthy",
                "last_error_category": None,
                "attempts": [],
            }
            self._enabled = True
            self._revision = 0
            self._published_revision = 0
            self._publish_blocked_revision = -1
            self._close_retry_revision = -1
            self._durable_closed = False
            self._mark_dirty_locked()
            try:
                self._ensure_writer_locked(run_id, work_queue)
            except Exception:
                self._enabled = False
                self._manifest = None
                self._manifest_path = None
                self._capture_run_id = None
                return False
            return True

    def _disable(self, timeout_seconds: float) -> bool:
        timeout = max(0.0, float(timeout_seconds))
        deadline = time.monotonic() + timeout
        with self._condition:
            self._enabled = False
            if self._manifest is None:
                return True
            if self._manifest["state"] == "closed" and self._durable_closed:
                return True
            if self._manifest["state"] != "draining":
                self._manifest["state"] = "draining"
                self._manifest["queue_drained"] = False
            self._mark_dirty_locked()
            while not self._durable_closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _finish_cycle(self, session: FeeCycleCaptureSession) -> None:
        if not isinstance(session, FeeCycleCaptureSession):
            return
        with self._condition:
            if (
                self._manifest is None
                or self._manifest["state"] not in {"active", "draining"}
                or session.capture_run_id != self._capture_run_id
                or self._active_sessions.get(session.capture_seq) != session.cycle_id
            ):
                return
            attempt = self._attempt_for_seq_locked(session.capture_seq)
            if attempt is None:
                return
            try:
                queued_session = copy.deepcopy(session)
            except Exception as exc:
                self._active_sessions.pop(session.capture_seq, None)
                attempt["status"] = "failed"
                attempt["eligible"] = False
                attempt["error_category"] = _error_category(exc)
                attempt["error"] = _error_text(exc)
                self._manifest["failed"] += 1
                self._record_health_error_locked(exc)
                self._prune_attempts_locked()
                self._mark_dirty_locked()
                return

            self._active_sessions.pop(session.capture_seq, None)
            attempt["status"] = "queued"
            attempt["eligible"] = False
            try:
                self._queue.put_nowait((queued_session, attempt))
                self._queued_count += 1
            except queue.Full:
                attempt["status"] = "dropped"
                attempt["eligible"] = False
                attempt["error_category"] = "queue_full"
                attempt["error"] = "capture queue full"
                self._manifest["dropped"] += 1
            except Exception as exc:
                attempt["status"] = "failed"
                attempt["eligible"] = False
                attempt["error_category"] = _error_category(exc)
                attempt["error"] = _error_text(exc)
                self._manifest["failed"] += 1
                self._record_health_error_locked(exc)
            self._prune_attempts_locked()
            self._mark_dirty_locked()

    def _writer_main(self, run_id: str, work_queue: queue.Queue) -> None:
        while True:
            with self._condition:
                if self._capture_run_id != run_id or self._manifest is None:
                    return
                try:
                    session, attempt = work_queue.get_nowait()
                except queue.Empty:
                    session = None
                    attempt = None

                if session is not None and attempt is not None:
                    self._queued_count -= 1
                    self._inflight_seq = session.capture_seq
                    attempt["status"] = "writing"
                    attempt["eligible"] = False
                    self._mark_dirty_locked()
                    snapshot = copy.deepcopy(self._manifest)
                    revision = self._revision
                    action = "write"
                elif self._can_close_locked() and (
                    self._revision > self._close_retry_revision
                ):
                    snapshot = None
                    revision = self._revision
                    action = "close"
                elif self._revision > max(
                    self._published_revision, self._publish_blocked_revision
                ):
                    snapshot = copy.deepcopy(self._manifest)
                    revision = self._revision
                    action = "publish"
                else:
                    self._condition.wait()
                    continue

            if action == "write":
                self._publish_manifest_snapshot(run_id, snapshot, revision)
                self._write_queued_session(run_id, work_queue, session, attempt)
            elif action == "publish":
                self._publish_manifest_snapshot(run_id, snapshot, revision)
            elif self._close_run_durably(run_id):
                return

    def _publish_manifest_snapshot(
        self, run_id: str, snapshot: dict, revision: int
    ) -> bool:
        with self._condition:
            path = self._manifest_path
        if path is None:
            return False
        try:
            self._ensure_output_dir()
            self._atomic_write(path, _json_bytes(snapshot))
        except Exception as exc:
            with self._condition:
                if self._capture_run_id == run_id and self._manifest is not None:
                    self._record_health_error_locked(exc)
                    self._mark_dirty_locked()
                    self._publish_blocked_revision = self._revision
            self._log_failure("capture manifest publication failed", exc)
            return False
        with self._condition:
            if self._capture_run_id == run_id:
                self._published_revision = max(self._published_revision, revision)
                self._publish_blocked_revision = -1
                self._condition.notify_all()
        return True

    def _write_queued_session(
        self,
        run_id: str,
        work_queue: queue.Queue,
        session: FeeCycleCaptureSession,
        attempt: dict,
    ) -> None:
        try:
            publication = self._publish_envelope(session)
        except Exception as exc:
            with self._condition:
                if self._capture_run_id == run_id and self._manifest is not None:
                    attempt["status"] = "failed"
                    attempt["eligible"] = False
                    attempt["error_category"] = _error_category(exc)
                    attempt["error"] = _error_text(exc)
                    self._manifest["failed"] += 1
                    self._record_health_error_locked(exc)
                    self._inflight_seq = None
                    work_queue.task_done()
                    self._prune_attempts_locked()
                    self._mark_dirty_locked()
            self._log_failure("capture writer failed", exc)
            return

        retained_index = None
        rotation_error = None
        try:
            retained_index = self._rotate_capture_files()
            self._reconcile_closed_manifests(retained_index, run_id)
        except Exception as exc:
            rotation_error = exc
            self._log_failure("capture retention failed", exc)

        with self._condition:
            if self._capture_run_id != run_id or self._manifest is None:
                work_queue.task_done()
                return
            attempt["status"] = "completed"
            attempt["eligible"] = publication["eligible"]
            attempt["filename"] = publication["filename"]
            attempt["bytes"] = publication["bytes"]
            self._manifest["completed"] += 1
            self._manifest["last_completed_seq"] = session.capture_seq
            if retained_index is None:
                self._retained_sequences.add(session.capture_seq)
            else:
                self._set_retained_sequences_locked(
                    retained_index.get(run_id, set())
                )
            if rotation_error is not None:
                attempt["rotation_error"] = _error_text(rotation_error)
                self._record_health_error_locked(rotation_error)
            self._inflight_seq = None
            work_queue.task_done()
            self._prune_attempts_locked()
            self._mark_dirty_locked()

    def _close_run_durably(self, run_id: str) -> bool:
        try:
            self._ensure_output_dir()
            retained_index = self._current_capture_index()
            self._reconcile_closed_manifests(retained_index, run_id)
            with self._condition:
                if not self._can_close_locked() or self._capture_run_id != run_id:
                    return False
                self._set_retained_sequences_locked(
                    retained_index.get(run_id, set())
                )
                closed_snapshot = copy.deepcopy(self._manifest)
                closed_snapshot["state"] = "closed"
                closed_snapshot["queue_drained"] = True
                closed_snapshot["updated_at"] = _utc_now()
                path = self._manifest_path
                revision = self._revision
            if path is None:
                return False
            self._atomic_write(path, _json_bytes(closed_snapshot))
        except Exception as exc:
            with self._condition:
                if self._capture_run_id == run_id and self._manifest is not None:
                    self._record_health_error_locked(exc)
                    self._mark_dirty_locked()
                    self._close_retry_revision = self._revision
            self._log_failure("capture close publication failed", exc)
            return False

        with self._condition:
            if self._capture_run_id != run_id or not self._can_close_locked():
                return False
            self._manifest["state"] = "closed"
            self._manifest["queue_drained"] = True
            self._manifest["updated_at"] = closed_snapshot["updated_at"]
            self._durable_closed = True
            self._published_revision = max(self._published_revision, revision)
            if self._writer_thread is threading.current_thread():
                self._writer_thread = None
            self._condition.notify_all()
        return True

    def _publish_envelope(self, session: FeeCycleCaptureSession) -> dict:
        body = session.to_body()
        sealed = seal_envelope(body)
        serialized = _json_bytes(sealed)
        if len(serialized) > MAX_ENVELOPE_BYTES:
            raise ValueError(
                "complete sealed envelope exceeds "
                f"{MAX_ENVELOPE_BYTES} byte capture limit"
            )
        filename = (
            f"{session.capture_run_id}-{session.capture_seq:08d}-"
            f"{session.cycle_id}.json"
        )
        self._atomic_write(self.output_dir / filename, serialized)
        return {
            "filename": filename,
            "bytes": len(serialized),
            "eligible": bool(body["completeness"]["complete"]),
        }

    def _rotate_capture_files(self) -> Dict[str, set]:
        candidates = []
        for path in self.output_dir.glob("*.json"):
            if self._capture_identity(path) is None:
                continue
            stat = path.stat()
            candidates.append((stat.st_mtime_ns, path.name, stat.st_size, path))
        candidates.sort(reverse=True)
        retained_count = 0
        retained_bytes = 0
        removed = False
        for _mtime, _name, size, path in candidates:
            if (
                retained_count >= RETENTION_MAX_FILES
                or retained_bytes + size > RETENTION_MAX_BYTES
            ):
                path.unlink()
                removed = True
                continue
            retained_count += 1
            retained_bytes += size
        if removed:
            self._fsync_directory()
        return self._capture_index(
            path for _mtime, _name, _size, path in candidates if path.exists()
        )

    def _current_capture_index(self) -> Dict[str, set]:
        return self._capture_index(self.output_dir.glob("*.json"))

    def _capture_index(self, paths: Iterator[Path]) -> Dict[str, set]:
        retained: Dict[str, set] = {}
        for path in paths:
            identity = self._capture_identity(path)
            if identity is None:
                continue
            run_id, sequence = identity
            retained.setdefault(run_id, set()).add(sequence)
        return retained

    @staticmethod
    def _capture_identity(path: Path) -> Optional[tuple]:
        name = path.name
        if name.startswith("manifest-") or not name.endswith(".json"):
            return None
        try:
            run_id, sequence_text, cycle_id = name[:-5].split("-", 2)
        except ValueError:
            return None
        if (
            len(run_id) != 32
            or any(character not in "0123456789abcdef" for character in run_id)
            or len(sequence_text) != 8
            or any(character not in "0123456789" for character in sequence_text)
            or int(sequence_text) < 1
            or cycle_id != f"{run_id}:{sequence_text}"
        ):
            return None
        return run_id, int(sequence_text)

    def _reconcile_closed_manifests(
        self, retained_index: Dict[str, set], current_run_id: str
    ) -> None:
        current_path = self._manifest_path
        removed = False
        for path in self.output_dir.glob(
            f"manifest-*.v{SCHEMA_VERSION}.json"
        ):
            if path == current_path:
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except Exception:
                continue
            if not isinstance(manifest, dict) or manifest.get("state") != "closed":
                continue
            run_id = manifest.get("capture_run_id")
            retained_sequences = retained_index.get(str(run_id), set())
            if not retained_sequences:
                path.unlink()
                removed = True
                continue
            updated = copy.deepcopy(manifest)
            updated["attempts"] = self._bounded_attempt_records(
                updated.get("attempts", []), retained_sequences
            )
            updated["retained_sequence_range"] = self._sequence_range(
                retained_sequences
            )
            if updated != manifest:
                updated["updated_at"] = _utc_now()
                self._atomic_write(path, _json_bytes(updated))
        if removed:
            self._fsync_directory()

    def _ensure_output_dir(self) -> None:
        if self.output_dir.is_symlink():
            raise OSError(f"capture output directory is a symlink: {self.output_dir}")
        created = not self.output_dir.exists()
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.output_dir.is_symlink():
            raise OSError(f"capture output directory is a symlink: {self.output_dir}")
        if not self.output_dir.is_dir():
            raise OSError(f"capture output path is not a directory: {self.output_dir}")
        os.chmod(self.output_dir, 0o700)
        if created:
            self._fsync_directory(self.output_dir.parent)

    def _ensure_writer_locked(self, run_id: str, work_queue: queue.Queue) -> None:
        if self._writer_thread is not None:
            raise RuntimeError("capture writer already active")
        self._writer_thread = threading.Thread(
            target=self._writer_main,
            args=(run_id, work_queue),
            name="fee-cycle-capture-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _mark_dirty_locked(self) -> None:
        if self._manifest is not None:
            self._manifest["updated_at"] = _utc_now()
        self._revision += 1
        self._condition.notify_all()

    def _can_close_locked(self) -> bool:
        return bool(
            self._manifest is not None
            and self._manifest["state"] == "draining"
            and not self._active_sessions
            and self._queued_count == 0
            and self._inflight_seq is None
        )

    def _attempt_for_seq_locked(self, capture_seq: int) -> Optional[dict]:
        if self._manifest is None:
            return None
        for attempt in self._manifest["attempts"]:
            if attempt.get("capture_seq") == capture_seq:
                return attempt
        return None

    def _record_health_error_locked(self, exc: Exception) -> None:
        if self._manifest is None:
            return
        self._manifest["writer_health"] = "degraded"
        self._manifest["last_error_category"] = _error_category(exc)

    def _set_retained_sequences_locked(self, sequences: set) -> None:
        self._retained_sequences = set(sequences)
        if self._manifest is not None:
            self._manifest["retained_sequence_range"] = self._sequence_range(
                self._retained_sequences
            )
        self._prune_attempts_locked()

    def _prune_attempts_locked(self) -> None:
        if self._manifest is None:
            return
        self._manifest["attempts"] = self._bounded_attempt_records(
            self._manifest["attempts"], self._retained_sequences
        )

    @staticmethod
    def _bounded_attempt_records(attempts: Any, retained_sequences: set) -> list:
        if not isinstance(attempts, list):
            return []
        live = []
        retained = []
        failures = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            status = attempt.get("status")
            sequence = attempt.get("capture_seq")
            if status in {"active", "queued", "writing"}:
                live.append(attempt)
            elif status == "completed" and sequence in retained_sequences:
                retained.append(attempt)
            elif status in {"failed", "dropped"}:
                failures.append(attempt)
        failures = sorted(
            failures, key=lambda item: int(item.get("capture_seq", -1))
        )[-RETENTION_MAX_FILES:]
        return sorted(
            retained + failures + live,
            key=lambda item: int(item.get("capture_seq", -1)),
        )

    @staticmethod
    def _sequence_range(sequences: set) -> dict:
        if not sequences:
            return {"first": None, "last": None}
        return {"first": min(sequences), "last": max(sequences)}

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        descriptor = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o600)
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._fsync_directory()
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _fsync_directory(self, directory: Optional[Path] = None) -> None:
        target = self.output_dir if directory is None else Path(directory)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(target, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _log_failure(self, message: str, exc: Exception) -> None:
        try:
            self._logger(f"{message}: {_error_text(exc)}", level="warn")
        except Exception:
            pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _error_category(exc: Exception) -> str:
    return type(exc).__name__
