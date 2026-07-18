import copy
import json
import os
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


class FeeCycleCaptureManager:
    def __init__(self, database_path: Any, logger: Any):
        expanded = os.path.expandvars(os.path.expanduser(os.fspath(database_path)))
        database = Path(expanded)
        self.output_dir = database.with_name(f"{database.stem}_fee_replay")
        self._logger = logger
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._writer_thread: Optional[threading.Thread] = None
        self._enabled = False
        self._capture_run_id: Optional[str] = None
        self._next_seq = 0
        self._manifest: Optional[Dict[str, Any]] = None
        self._manifest_path: Optional[Path] = None
        self._submitted = set()

    def set_enabled(self, enabled: bool, timeout_seconds: float = 5.0) -> bool:
        try:
            if enabled:
                return self._enable()
            return self._disable(timeout_seconds)
        except Exception as exc:
            self._log_failure("capture enable state change failed", exc)
            return False

    def begin_cycle(
        self, configuration: dict, producer: dict
    ) -> Optional[FeeCycleCaptureSession]:
        try:
            with self._lock:
                if (
                    not self._enabled
                    or self._manifest is None
                    or self._manifest["state"] != "open"
                    or self._capture_run_id is None
                ):
                    return None
                configuration_copy = copy.deepcopy(configuration)
                producer_copy = copy.deepcopy(producer)
                if not isinstance(configuration_copy, dict):
                    raise TypeError("capture configuration must be a dict")
                if not isinstance(producer_copy, dict):
                    raise TypeError("capture producer must be a dict")
                producer_copy.setdefault("started_at", _utc_now())
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
                return session
        except Exception as exc:
            self._log_failure("capture session creation failed", exc)
            return None

    def finish_cycle(self, session: FeeCycleCaptureSession) -> None:
        try:
            self._finish_cycle(session)
        except Exception as exc:
            self._log_failure("capture submission failed", exc)

    def read_manifest(self) -> dict:
        with self._lock:
            path = self._manifest_path
            fallback = copy.deepcopy(self._manifest) if self._manifest else {}
        if path is None:
            return fallback
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else fallback
        except Exception:
            return fallback

    def _enable(self) -> bool:
        with self._lock:
            if self._enabled:
                return True
            if self._manifest is not None and self._manifest["state"] == "draining":
                return False
            self._ensure_output_dir()
            run_id = uuid.uuid4().hex
            now = _utc_now()
            self._capture_run_id = run_id
            self._next_seq = 0
            self._submitted = set()
            self._manifest_path = (
                self.output_dir / f"manifest-{run_id}.v{SCHEMA_VERSION}.json"
            )
            self._manifest = {
                "schema_name": "fee_cycle_capture_manifest",
                "schema_version": SCHEMA_VERSION,
                "capture_run_id": run_id,
                "state": "open",
                "queue_drained": False,
                "started_at": now,
                "updated_at": now,
                "completed": 0,
                "failed": 0,
                "dropped": 0,
                "attempts": [],
            }
            self._enabled = True
            self._ensure_writer_locked()
            if not self._publish_manifest_locked():
                self._enabled = False
                self._manifest = None
                self._manifest_path = None
                self._capture_run_id = None
                return False
            return True

    def _disable(self, timeout_seconds: float) -> bool:
        timeout = max(0.0, float(timeout_seconds))
        with self._lock:
            self._enabled = False
            if self._manifest is None:
                return True
            if self._manifest["state"] == "closed":
                return True
            if self._manifest["state"] != "draining":
                self._manifest["state"] = "draining"
                self._manifest["queue_drained"] = False
                self._touch_manifest_locked()
                self._publish_manifest_locked()
            if self._queue.unfinished_tasks == 0:
                self._close_run_locked()
                return True

        if not self._wait_for_queue(timeout):
            return False

        with self._lock:
            self._close_run_locked()
            return self._manifest is not None and self._manifest["state"] == "closed"

    def _finish_cycle(self, session: FeeCycleCaptureSession) -> None:
        with self._lock:
            if (
                not isinstance(session, FeeCycleCaptureSession)
                or self._manifest is None
                or not self._enabled
                or self._manifest["state"] != "open"
                or session.capture_run_id != self._capture_run_id
                or session.capture_seq in self._submitted
            ):
                return
            self._submitted.add(session.capture_seq)
            attempt = {
                "capture_seq": session.capture_seq,
                "cycle_id": session.cycle_id,
                "status": "queued",
                "eligible": True,
            }
            self._manifest["attempts"].append(attempt)
            self._touch_manifest_locked()
            self._publish_manifest_locked()
            try:
                queued_session = copy.deepcopy(session)
                self._queue.put_nowait((queued_session, attempt))
            except queue.Full:
                attempt["status"] = "dropped"
                attempt["eligible"] = False
                attempt["error"] = "capture queue full"
                self._manifest["dropped"] += 1
                self._touch_manifest_locked()
                self._publish_manifest_locked()
            except Exception as exc:
                attempt["status"] = "failed"
                attempt["eligible"] = False
                attempt["error"] = _error_text(exc)
                self._manifest["failed"] += 1
                self._touch_manifest_locked()
                self._publish_manifest_locked()

    def _writer_main(self) -> None:
        while True:
            session, attempt = self._queue.get()
            try:
                with self._lock:
                    attempt["status"] = "writing"
                    self._touch_manifest_locked()
                    self._publish_manifest_locked()
                try:
                    publication = self._publish_envelope(session)
                except Exception as exc:
                    with self._lock:
                        attempt["status"] = "failed"
                        attempt["eligible"] = False
                        attempt["error"] = _error_text(exc)
                        if self._manifest is not None:
                            self._manifest["failed"] += 1
                            self._touch_manifest_locked()
                            self._publish_manifest_locked()
                    self._log_failure("capture writer failed", exc)
                else:
                    rotation_error = None
                    try:
                        self._rotate_capture_files()
                    except Exception as exc:
                        rotation_error = _error_text(exc)
                        self._log_failure("capture retention failed", exc)
                    with self._lock:
                        attempt["status"] = "completed"
                        attempt["eligible"] = publication["eligible"]
                        attempt["filename"] = publication["filename"]
                        attempt["bytes"] = publication["bytes"]
                        if rotation_error is not None:
                            attempt["rotation_error"] = rotation_error
                        if self._manifest is not None:
                            self._manifest["completed"] += 1
                            self._touch_manifest_locked()
                            self._publish_manifest_locked()
            except Exception as exc:
                self._log_failure("capture writer bookkeeping failed", exc)
            finally:
                self._queue.task_done()
                with self._lock:
                    if (
                        self._manifest is not None
                        and self._manifest["state"] == "draining"
                        and self._queue.unfinished_tasks == 0
                    ):
                        self._close_run_locked()

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
            f"capture-{session.capture_run_id}-{session.capture_seq:08d}."
            f"v{SCHEMA_VERSION}.json"
        )
        self._atomic_write(self.output_dir / filename, serialized)
        return {
            "filename": filename,
            "bytes": len(serialized),
            "eligible": bool(body["completeness"]["complete"]),
        }

    def _rotate_capture_files(self) -> None:
        candidates = []
        for path in self.output_dir.glob(
            f"capture-*.v{SCHEMA_VERSION}.json"
        ):
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

    def _ensure_output_dir(self) -> None:
        if self.output_dir.is_symlink():
            raise OSError(f"capture output directory is a symlink: {self.output_dir}")
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.output_dir.is_dir():
            raise OSError(f"capture output path is not a directory: {self.output_dir}")
        os.chmod(self.output_dir, 0o700)

    def _ensure_writer_locked(self) -> None:
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        self._writer_thread = threading.Thread(
            target=self._writer_main,
            name="fee-cycle-capture-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _publish_manifest_locked(self) -> bool:
        if self._manifest is None or self._manifest_path is None:
            return False
        try:
            self._atomic_write(self._manifest_path, _json_bytes(self._manifest))
            return True
        except Exception as exc:
            self._log_failure("capture manifest publication failed", exc)
            return False

    def _touch_manifest_locked(self) -> None:
        if self._manifest is not None:
            self._manifest["updated_at"] = _utc_now()

    def _close_run_locked(self) -> None:
        if self._manifest is None or self._manifest["state"] == "closed":
            return
        if self._queue.unfinished_tasks != 0:
            return
        self._manifest["state"] = "closed"
        self._manifest["queue_drained"] = True
        self._touch_manifest_locked()
        self._publish_manifest_locked()

    def _wait_for_queue(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
            return True

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

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.output_dir, flags)
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
