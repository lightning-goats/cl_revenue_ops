"""Thread-safe Python fee-authority state.

This standalone model is intentionally independent of plugin wiring and CLN
RPCs. Consumers receive immutable snapshots so a transition cannot be observed
partially.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FeeAuthorityStatus:
    enabled: bool
    generation: int
    transitioned_at: int
    reason: str


class FeeAuthorityGate:
    def __init__(
        self,
        enabled: bool = True,
        now_fn: Callable[[], float] = time.time,
    ):
        self._now_fn = now_fn
        self._lock = threading.Lock()
        self._status = FeeAuthorityStatus(
            enabled=enabled,
            generation=0,
            transitioned_at=int(self._now_fn()),
            reason="initial",
        )

    def snapshot(self) -> FeeAuthorityStatus:
        with self._lock:
            return self._status

    def set_enabled(self, enabled: bool, reason: str) -> FeeAuthorityStatus:
        with self._lock:
            if enabled == self._status.enabled:
                return self._status
            self._status = FeeAuthorityStatus(
                enabled=enabled,
                generation=self._status.generation + 1,
                transitioned_at=int(self._now_fn()),
                reason=reason,
            )
            return self._status

    def deny_reason(self, operation: str) -> dict[str, object] | None:
        status = self.snapshot()
        if status.enabled:
            return None
        return {
            "status": "blocked",
            "reason": "fee_authority_disabled",
            "operation": operation,
            "generation": status.generation,
            "transitioned_at": status.transitioned_at,
        }
