"""Deterministic cycle context (refactor Phase 1, Workstream J3).

Policies must not read the wall clock or unseeded randomness. A
CycleContext is constructed once per economic cycle by the orchestrator
(Workstream H) and injected into every policy: cycle_time is the only
clock, and rng()/derive_seed() the only randomness, so the same context
reproduces byte-identical decisions.

Additive Phase 1 foundation — not yet consumed by live decision paths.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .econ_types import U63_MAX, EconArithmeticError, UnixTime


@dataclass(frozen=True)
class CycleContext:
    cycle_id: str
    cycle_time: UnixTime
    seed: int
    snapshot_id: str

    def __post_init__(self):
        if not isinstance(self.cycle_id, str) or not self.cycle_id:
            raise EconArithmeticError(
                f"CycleContext.cycle_id invalid: {self.cycle_id!r}")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise EconArithmeticError(
                f"CycleContext.snapshot_id invalid: {self.snapshot_id!r}")
        if not isinstance(self.cycle_time, UnixTime):
            raise EconArithmeticError(
                f"CycleContext.cycle_time must be UnixTime: "
                f"{self.cycle_time!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) \
                or not (0 <= self.seed <= U63_MAX):
            raise EconArithmeticError(
                f"CycleContext.seed out of [0, 2^63-1]: {self.seed!r}")

    def rng(self) -> random.Random:
        """A FRESH deterministic generator each call — repeatable."""
        return random.Random(self.seed)

    def derive_seed(self, component: str) -> int:
        """Independent-but-deterministic seed for a named component."""
        if not isinstance(component, str) or not component:
            raise EconArithmeticError(
                f"derive_seed component invalid: {component!r}")
        digest = hashlib.sha256(
            f"{self.seed}:{component}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & U63_MAX
