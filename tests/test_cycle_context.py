"""Phase 1: deterministic cycle context (wire-contract-spec.md J3)."""
import dataclasses

import pytest

from modules.cycle_context import CycleContext
from modules.econ_types import EconArithmeticError, UnixTime


def _ctx(seed=42):
    return CycleContext(
        cycle_id="cycle-000001",
        cycle_time=UnixTime(1_752_400_000),
        seed=seed,
        snapshot_id="snap-000001",
    )


def test_rng_is_repeatable_across_calls():
    ctx = _ctx()
    first = [ctx.rng().random() for _ in range(3)]
    assert first[0] == first[1] == first[2]


def test_rng_seed_42_first_draw_pinned():
    # Characterization pin: random.Random(42).random() is stable across
    # CPython versions (Mersenne Twister is part of the language spec).
    assert _ctx(seed=42).rng().random() == pytest.approx(
        0.6394267984578837, abs=0, rel=0)


def test_different_seeds_differ():
    assert _ctx(seed=1).rng().random() != _ctx(seed=2).rng().random()


def test_derive_seed_deterministic_and_component_scoped():
    ctx = _ctx()
    a1 = ctx.derive_seed("fee-policy")
    a2 = ctx.derive_seed("fee-policy")
    b = ctx.derive_seed("rebalance-policy")
    assert a1 == a2
    assert a1 != b
    assert 0 <= a1 <= 2**63 - 1


def test_validation_and_frozen():
    with pytest.raises(EconArithmeticError):
        CycleContext(cycle_id="", cycle_time=UnixTime(1), seed=1,
                     snapshot_id="s")
    with pytest.raises(EconArithmeticError):
        CycleContext(cycle_id="c", cycle_time=UnixTime(1), seed=-1,
                     snapshot_id="s")
    with pytest.raises(dataclasses.FrozenInstanceError):
        _ctx().seed = 7
