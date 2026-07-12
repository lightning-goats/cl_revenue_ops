"""Phase 2 (Workstream C): intent arbiter in shadow mode — pure,
deterministic conflict resolution over typed intents."""
import random

import pytest

from modules.econ_arbiter import arbitrate
from modules.econ_intents import Explanation, make_intent
from modules.econ_types import Micro, Msat, SignedMsat, UnixTime

NOW = 1_752_400_000


def _intent(intent_type="SET_FEE", target="123x456x0", priority=50,
            benefit=0, confidence=500_000, capital=0, amount=None,
            snapshot_id="snap-1", expires=600):
    return make_intent(
        intent_type=intent_type,
        snapshot_id=snapshot_id,
        created_at=UnixTime(NOW),
        expires_at=UnixTime(NOW + expires),
        target=target,
        amount_msat=None if amount is None else Msat(amount),
        expected_benefit_msat=SignedMsat(benefit),
        max_cost_msat=Msat(0),
        capital_committed_msat=Msat(capital),
        confidence_micro=Micro(confidence),
        reason_codes=(),
        explanation=Explanation("test", (("t", target),)),
        preconditions=(),
        priority=priority,
        budget_bucket="fees",
        origin_policy="test",
        reversible=True,
    )


def test_no_conflicts_passes_through_ordered():
    a = _intent(target="100x1x0", priority=80)
    b = _intent(target="200x1x0", priority=20)
    result = arbitrate([b, a], now=NOW)
    assert [e.target for e in result.ordered] == ["100x1x0", "200x1x0"]
    assert result.rejected == ()


def test_duplicate_idempotency_keys_superseded():
    a = _intent(target="100x1x0")
    b = _intent(target="100x1x0")  # identical inputs -> identical key
    assert a.idempotency_key == b.idempotency_key
    result = arbitrate([a, b], now=NOW)
    assert len(result.ordered) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0][1] == "INTENT_SUPERSEDED"


def test_contradictory_fee_changes_same_channel():
    low = _intent(target="100x1x0", amount=100_000, priority=40)
    high = _intent(target="100x1x0", amount=250_000, priority=60)
    result = arbitrate([low, high], now=NOW)
    kept = [e for e in result.ordered if e.intent_type == "SET_FEE"]
    assert len(kept) == 1
    assert kept[0].priority == 60  # tie-break: higher priority wins
    assert result.rejected[0][1] == "INTENT_SUPERSEDED"
    assert result.superseded[result.rejected[0][0].intent_id.value] == \
        kept[0].intent_id.value


def test_close_beats_rebalance_on_same_channel():
    close = _intent(intent_type="CLOSE_CHANNEL", target="100x1x0")
    reb = _intent(intent_type="REBALANCE", target="100x1x0",
                  amount=500_000_000, capital=500_000_000)
    result = arbitrate([reb, close], now=NOW)
    assert [e.intent_type for e in result.ordered] == ["CLOSE_CHANNEL"]
    assert result.rejected[0][0].intent_type == "REBALANCE"
    assert result.rejected[0][1] == "CONFLICT_CLOSE_REBALANCE"


def test_stale_intent_rejected():
    fresh = _intent(target="100x1x0")
    stale = _intent(target="200x1x0", expires=60)
    result = arbitrate([fresh, stale], now=NOW + 120)
    assert [e.target for e in result.ordered] == ["100x1x0"]
    assert result.rejected[0][1] == "INTENT_STALE"


def test_tie_break_ladder():
    # Same priority: higher expected benefit wins.
    a = _intent(target="100x1x0", benefit=5_000)
    b = _intent(target="200x1x0", benefit=1_000)
    result = arbitrate([b, a], now=NOW)
    assert [e.target for e in result.ordered] == ["100x1x0", "200x1x0"]
    # Same priority+benefit: higher confidence wins.
    c = _intent(target="300x1x0", confidence=900_000)
    d = _intent(target="400x1x0", confidence=100_000)
    result = arbitrate([d, c], now=NOW)
    assert [e.target for e in result.ordered] == ["300x1x0", "400x1x0"]
    # All equal: stable target identifier decides.
    e1 = _intent(target="500x1x0")
    e2 = _intent(target="600x1x0")
    result = arbitrate([e2, e1], now=NOW)
    assert [e.target for e in result.ordered] == ["500x1x0", "600x1x0"]


def test_input_order_never_matters():
    intents = [
        _intent(target="100x1x0", priority=10),
        _intent(target="200x1x0", priority=90, benefit=1_000),
        _intent(intent_type="CLOSE_CHANNEL", target="300x1x0"),
        _intent(intent_type="REBALANCE", target="300x1x0",
                amount=1_000_000, capital=1_000_000),
        _intent(target="400x1x0", expires=30),
        _intent(intent_type="SET_HTLC_MAX", target="200x1x0",
                amount=500_000_000),
    ]
    baseline = arbitrate(list(intents), now=NOW + 60)
    for seed in range(5):
        shuffled = list(intents)
        random.Random(seed).shuffle(shuffled)
        result = arbitrate(shuffled, now=NOW + 60)
        assert [e.intent_id for e in result.ordered] == \
            [e.intent_id for e in baseline.ordered]
        assert [(r[0].intent_id, r[1]) for r in result.rejected] == \
            [(r[0].intent_id, r[1]) for r in baseline.rejected]


def test_close_precedence_orders_before_fees():
    fee = _intent(target="100x1x0", priority=100)
    close = _intent(intent_type="CLOSE_CHANNEL", target="200x1x0",
                    priority=10)
    result = arbitrate([fee, close], now=NOW)
    # capital preservation precedence outranks revenue protection even
    # at lower requested priority (spec precedence order).
    assert [e.intent_type for e in result.ordered] == [
        "CLOSE_CHANNEL", "SET_FEE"]
