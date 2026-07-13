"""PR 10 (gap-closure Phase G): the two missing arbiter conflict rules,
behind econ_conflict_rules_extended (default OFF = current behavior).

- CONFLICT_DUPLICATE_OPEN: a second OPEN_CHANNEL to the same peer is
  rejected (covers the spec's open-vs-LN+ rule — both paths emit
  OPEN_CHANNEL intents targeting the peer; the LN+ obligation carries
  higher priority and wins in a batch, and first-registered wins live).
- CONFLICT_REBALANCE_SWAP: circular rebalance vs structural swap on one
  channel. Batch: the structural SWAP_OUT outranks (capital-structure
  intent beats liquidity maintenance — same precedence logic as close);
  live registry: EITHER active intent blocks the other (no overlapping
  opposite work on one channel).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.econ_arbiter import ActiveIntentRegistry, arbitrate
from modules.econ_intents import Explanation, make_intent
from modules.econ_types import Micro, Msat, SignedMsat, UnixTime

NOW = 1_752_400_000


def _env(intent_type="REBALANCE", target="111x222x0", amount=400_000,
         priority=50, bucket="rebalance", policy="test"):
    return make_intent(
        intent_type=intent_type, snapshot_id="snap-1",
        created_at=UnixTime(NOW), expires_at=UnixTime(NOW + 600),
        target=target,
        amount_msat=Msat(amount * 1000) if amount else None,
        expected_benefit_msat=SignedMsat(0),
        max_cost_msat=Msat(3_000_000), capital_committed_msat=Msat(0),
        confidence_micro=Micro(0), reason_codes=(),
        explanation=Explanation("t", (("x", 1),)), preconditions=(),
        priority=priority, budget_bucket=bucket, origin_policy=policy,
        reversible=False)


class TestBatchDuplicateOpen:
    def test_second_open_to_same_peer_rejected(self):
        lnplus = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                      amount=2_000_000, priority=80,
                      bucket="channel_open", policy="lnplus")
        planner = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                       amount=1_000_000, priority=50,
                       bucket="channel_open", policy="planner")
        result = arbitrate([planner, lnplus], now=NOW,
                           extended_rules=True)
        assert [e.origin_policy for e in result.ordered] == ["lnplus"]
        assert result.rejected[0][1] == "CONFLICT_DUPLICATE_OPEN"

    def test_different_peers_both_survive(self):
        a = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                 bucket="channel_open")
        b = _env("OPEN_CHANNEL", target="03" + "c" * 64,
                 bucket="channel_open")
        result = arbitrate([a, b], now=NOW, extended_rules=True)
        assert len(result.ordered) == 2

    def test_flag_off_preserves_current_behavior(self):
        a = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                 amount=2_000_000, bucket="channel_open")
        b = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                 amount=1_000_000, bucket="channel_open")
        result = arbitrate([a, b], now=NOW)
        assert len(result.ordered) == 2  # today: both pass


class TestBatchRebalanceVsSwap:
    def test_swap_out_blocks_rebalance_same_target(self):
        swap = _env("SWAP_OUT", target="111x222x0", amount=250_000,
                    bucket="rebalance", policy="boltz")
        reb = _env("REBALANCE", target="111x222x0")
        result = arbitrate([reb, swap], now=NOW, extended_rules=True)
        assert [e.intent_type for e in result.ordered] == ["SWAP_OUT"]
        assert result.rejected[0][1] == "CONFLICT_REBALANCE_SWAP"

    def test_different_targets_no_conflict(self):
        swap = _env("SWAP_OUT", target="111x222x0", amount=250_000)
        reb = _env("REBALANCE", target="900x1x0")
        result = arbitrate([reb, swap], now=NOW, extended_rules=True)
        assert len(result.ordered) == 2

    def test_flag_off_preserves_current_behavior(self):
        swap = _env("SWAP_OUT", target="111x222x0", amount=250_000)
        reb = _env("REBALANCE", target="111x222x0")
        result = arbitrate([reb, swap], now=NOW)
        assert len(result.ordered) == 2


class TestLiveRegistryExtended:
    def _registry(self, extended=True):
        return ActiveIntentRegistry(
            extended_rules_provider=(lambda: True) if extended
            else None)

    def test_duplicate_open_blocked_live(self):
        registry = self._registry()
        first = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                     amount=2_000_000, bucket="channel_open")
        second = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                      amount=1_000_000, bucket="channel_open")
        assert registry.check_and_register(first, NOW) is None
        assert registry.check_and_register(second, NOW) == \
            "CONFLICT_DUPLICATE_OPEN"

    def test_rebalance_vs_swap_blocks_both_directions(self):
        registry = self._registry()
        swap = _env("SWAP_OUT", target="111x222x0", amount=250_000)
        reb = _env("REBALANCE", target="111x222x0")
        assert registry.check_and_register(swap, NOW) is None
        assert registry.check_and_register(reb, NOW) == \
            "CONFLICT_REBALANCE_SWAP"
        registry2 = self._registry()
        assert registry2.check_and_register(reb, NOW) is None
        assert registry2.check_and_register(swap, NOW) == \
            "CONFLICT_REBALANCE_SWAP"

    def test_release_clears_the_conflict(self):
        registry = self._registry()
        swap = _env("SWAP_OUT", target="111x222x0", amount=250_000)
        reb = _env("REBALANCE", target="111x222x0")
        registry.check_and_register(swap, NOW)
        registry.release(swap.idempotency_key)
        assert registry.check_and_register(reb, NOW) is None

    def test_no_provider_means_legacy_rules_only(self):
        registry = ActiveIntentRegistry()
        first = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                     amount=2_000_000, bucket="channel_open")
        second = _env("OPEN_CHANNEL", target="02" + "b" * 64,
                      amount=1_000_000, bucket="channel_open")
        assert registry.check_and_register(first, NOW) is None
        assert registry.check_and_register(second, NOW) is None

    def test_provider_error_fails_to_legacy(self):
        registry = ActiveIntentRegistry(
            extended_rules_provider=MagicMock(side_effect=RuntimeError()))
        swap = _env("SWAP_OUT", target="111x222x0", amount=250_000)
        reb = _env("REBALANCE", target="111x222x0")
        assert registry.check_and_register(swap, NOW) is None
        assert registry.check_and_register(reb, NOW) is None

    def test_legacy_rules_still_enforced_with_extension(self):
        registry = self._registry()
        close = _env("CLOSE_CHANNEL", target="111x222x0", amount=0,
                     bucket="channel_open")
        reb = _env("REBALANCE", target="111x222x0")
        assert registry.check_and_register(close, NOW) is None
        assert registry.check_and_register(reb, NOW) == \
            "CONFLICT_CLOSE_REBALANCE"


def test_reason_codes_catalogued():
    from modules.reason_codes import CATALOG
    assert "CONFLICT_DUPLICATE_OPEN" in CATALOG
    assert "CONFLICT_REBALANCE_SWAP" in CATALOG
