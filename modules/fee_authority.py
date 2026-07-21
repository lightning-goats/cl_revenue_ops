"""Thread-safe Python fee-authority state and execution ownership.

The gate publishes immutable status snapshots and owns a reentrant execution
lease.  Disabling authority first closes admission, drains accepted work, and
only then publishes the disabled generation.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass(frozen=True)
class FeeAuthorityStatus:
    enabled: bool
    generation: int
    transitioned_at: int
    reason: str


class FeeAuthorityTransitionError(RuntimeError):
    """Raised when an authority transition cannot complete safely."""


class FeeAuthorityGate:
    def __init__(
        self,
        enabled: bool = True,
        now_fn: Callable[[], float] = time.time,
        *,
        drain_timeout_seconds: float = 30.0,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ):
        self._now_fn = now_fn
        self._monotonic_fn = monotonic_fn
        self._drain_timeout_seconds = float(drain_timeout_seconds)
        if self._drain_timeout_seconds < 0:
            raise ValueError("drain_timeout_seconds must be non-negative")

        self._condition = threading.Condition(threading.RLock())
        self._transition_lock = threading.Lock()
        self._lease_local = threading.local()
        self._active_outer_leases = 0
        self._accepting = bool(enabled)
        self._status = FeeAuthorityStatus(
            enabled=bool(enabled),
            generation=0,
            transitioned_at=int(self._now_fn()),
            reason="initial",
        )

    def snapshot(self) -> FeeAuthorityStatus:
        with self._condition:
            return self._status

    def set_enabled(
        self,
        enabled: bool,
        reason: str,
        *,
        timeout_seconds: float | None = None,
    ) -> FeeAuthorityStatus:
        enabled = bool(enabled)
        timeout = (
            self._drain_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if timeout < 0:
            raise ValueError("timeout_seconds must be non-negative")

        if not enabled and getattr(self._lease_local, "depth", 0) > 0:
            raise FeeAuthorityTransitionError(
                "cannot disable fee authority from inside an active execution lease"
            )

        with self._transition_lock:
            with self._condition:
                if enabled == self._status.enabled:
                    return self._status

                if enabled:
                    self._status = self._transition_status(True, reason)
                    self._accepting = True
                    self._condition.notify_all()
                    return self._status

                # Close admission before waiting. Existing lease owners may
                # continue through nested controller boundaries on this thread.
                self._accepting = False
                deadline = self._monotonic_fn() + timeout
                while self._active_outer_leases:
                    remaining = deadline - self._monotonic_fn()
                    if remaining <= 0:
                        self._accepting = True
                        self._condition.notify_all()
                        raise FeeAuthorityTransitionError(
                            "timed out draining active fee-authority execution leases"
                        )
                    self._condition.wait(timeout=remaining)

                self._status = self._transition_status(False, reason)
                return self._status

    @contextmanager
    def execution_lease(
        self,
        operation: str,
    ) -> Iterator[dict[str, object] | None]:
        acquired = False
        denial: dict[str, object] | None = None

        with self._condition:
            depth = getattr(self._lease_local, "depth", 0)
            if depth > 0:
                self._lease_local.depth = depth + 1
                acquired = True
            elif self._accepting and self._status.enabled:
                self._active_outer_leases += 1
                self._lease_local.depth = 1
                acquired = True
            else:
                denial = self._blocked_reason(operation, self._status)

        try:
            yield denial
        finally:
            if acquired:
                with self._condition:
                    depth = getattr(self._lease_local, "depth", 0)
                    if depth > 1:
                        self._lease_local.depth = depth - 1
                    elif depth == 1:
                        del self._lease_local.depth
                        self._active_outer_leases -= 1
                        self._condition.notify_all()
                    else:
                        raise RuntimeError("fee-authority execution lease underflow")

    def deny_reason(self, operation: str) -> dict[str, object] | None:
        with self._condition:
            if self._accepting and self._status.enabled:
                return None
            return self._blocked_reason(operation, self._status)

    def _transition_status(
        self,
        enabled: bool,
        reason: str,
    ) -> FeeAuthorityStatus:
        return FeeAuthorityStatus(
            enabled=enabled,
            generation=self._status.generation + 1,
            transitioned_at=max(
                self._status.transitioned_at,
                int(self._now_fn()),
            ),
            reason=reason,
        )

    @staticmethod
    def _blocked_reason(
        operation: str,
        status: FeeAuthorityStatus,
    ) -> dict[str, object]:
        return {
            "status": "blocked",
            "reason": "fee_authority_disabled",
            "operation": operation,
            "generation": status.generation,
            "transitioned_at": status.transitioned_at,
        }
