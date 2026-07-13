"""PR 3e (gap-closure Phase B): fee-controller canonical-snapshot
adoption via per-cycle observation freezing.

The six mutable-source reads the audit flagged (market prior, neighbor
median/percentile TTL caches, inbound gossip, chain feerates, channel
state) freeze into a per-cycle memo activated around
_adjust_all_fees_inner: within one fee cycle every observation is
computed at most once and is immutable — the policy cannot observe a
mid-cycle TTL refresh or gossip change. Outside a cycle (manual sets,
RPC debug paths) the memo is inactive and behavior is byte-identical
legacy. DTS+PID controller state is deliberately NOT part of this memo
(Phase C contract keeps controller_state a distinct input).

Fee intents keep their timestamped identity labels (same rationale as
LN+: identity semantics); the canonical-snapshot linkage is recorded as
ledger evidence.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.econ_ledger import EconLedger
from modules.fee_controller import FeeController

SCID = "111x222x0"
PEER_A = "02" + "a" * 64
PEER_B = "03" + "b" * 64
NOW = 1_752_400_000


def _fc():
    fc = FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_governor_fees_enabled=True, paused=False,
        authority_level="capital")
    fc.config = cfg
    return fc


class TestFrozenObservationMechanics:
    def test_inactive_memo_is_pure_passthrough(self):
        fc = _fc()
        assert fc._cycle_observations is None
        compute = MagicMock(side_effect=[1, 2])
        assert fc._frozen_observation(("k",), compute) == 1
        assert fc._frozen_observation(("k",), compute) == 2  # no freeze
        assert compute.call_count == 2

    def test_active_memo_computes_once(self):
        fc = _fc()
        fc._cycle_observations = {}
        compute = MagicMock(side_effect=[1, 2])
        assert fc._frozen_observation(("k",), compute) == 1
        assert fc._frozen_observation(("k",), compute) == 1  # frozen
        assert compute.call_count == 1

    def test_distinct_keys_distinct_values(self):
        fc = _fc()
        fc._cycle_observations = {}
        assert fc._frozen_observation(("a",), lambda: 1) == 1
        assert fc._frozen_observation(("b",), lambda: 2) == 2


class TestChainCostsFrozen:
    def test_frozen_within_cycle(self):
        fc = _fc()
        fc.data_service = MagicMock()
        fc.data_service.get_feerates.return_value = {
            "perkb": {"opening": 5000, "estimates": []}}
        fc._cycle_observations = {}
        first = fc._get_dynamic_chain_costs()
        fc.data_service.get_feerates.return_value = {
            "perkb": {"opening": 9999, "estimates": []}}
        second = fc._get_dynamic_chain_costs()
        assert second == first  # mid-cycle change invisible
        assert fc.data_service.get_feerates.call_count == 1

    def test_legacy_when_no_cycle(self):
        fc = _fc()
        fc.data_service = MagicMock()
        fc.data_service.get_feerates.return_value = {
            "perkb": {"opening": 5000, "estimates": []}}
        fc._get_dynamic_chain_costs()
        fc._get_dynamic_chain_costs()
        assert fc.data_service.get_feerates.call_count == 2


class TestNeighborStatsFrozen:
    def test_ttl_refresh_invisible_within_cycle(self):
        """Phase B required test: cache mutation AFTER the cycle's
        first observation must not change what the policy sees."""
        fc = _fc()
        fc._cycle_observations = {}
        fc._neighbor_fee_cache[f"neighbor_fee_{PEER_A}"] = {
            "value": 120, "ts": __import__("time").time()}
        assert fc._get_neighbor_fee_median(PEER_A) == 120
        # TTL cache mutates mid-cycle (as a background refresh would)
        fc._neighbor_fee_cache[f"neighbor_fee_{PEER_A}"] = {
            "value": 999, "ts": __import__("time").time()}
        assert fc._get_neighbor_fee_median(PEER_A) == 120  # frozen

    def test_per_peer_keys_are_distinct(self):
        import time as _t
        fc = _fc()
        fc._cycle_observations = {}
        fc._neighbor_fee_cache[f"neighbor_fee_{PEER_A}"] = {
            "value": 100, "ts": _t.time()}
        fc._neighbor_fee_cache[f"neighbor_fee_{PEER_B}"] = {
            "value": 200, "ts": _t.time()}
        assert fc._get_neighbor_fee_median(PEER_A) == 100
        assert fc._get_neighbor_fee_median(PEER_B) == 200


def test_cycle_activates_and_always_clears_memo(monkeypatch):
    """adjust_all_fees activates the memo around the inner cycle and
    clears it even when the cycle raises."""
    fc = _fc()
    seen = {}

    def fake_inner(**kwargs):
        seen["memo_active"] = fc._cycle_observations is not None
        raise RuntimeError("boom")

    monkeypatch.setattr(fc, "_adjust_all_fees_inner", fake_inner)
    fc.profitability = None
    with pytest.raises(RuntimeError):
        fc.adjust_all_fees()
    assert seen["memo_active"] is True
    assert fc._cycle_observations is None


class TestGovernedFeeEvidence:
    def test_label_unchanged_and_snapshot_in_ledger_evidence(self, tmp_path):
        fc = _fc()
        ledger = EconLedger(str(tmp_path / "ledger.db"))
        stub = MagicMock()
        stub.ledger_for_reconciliation.return_value = ledger
        stub.arbitration_registry.return_value = None
        stub.snapshot_ref.return_value = {"snapshot_id": "snap-55",
                                          "observed_at": NOW}
        fc.econ_shadow = stub
        ok, _ = fc._governed_authorize_fee_broadcast(
            channel_id=SCID, fee_ppm=250, old_fee_ppm=100,
            reason="dts", reason_code="dts_pid_sample")
        assert ok is True
        proposed = [e for e in ledger.events()
                    if e["event_type"] == "intent_proposed"]
        assert proposed
        assert proposed[0]["cycle_id"].startswith("fee-broadcast-")
        assert proposed[0]["details"]["canonical_snapshot_id"] == "snap-55"

    def test_no_hub_no_evidence_field(self, tmp_path):
        fc = _fc()
        ledger = EconLedger(str(tmp_path / "ledger.db"))
        stub = MagicMock()
        stub.ledger_for_reconciliation.return_value = ledger
        stub.arbitration_registry.return_value = None
        stub.snapshot_ref.side_effect = RuntimeError("down")
        fc.econ_shadow = stub
        ok, _ = fc._governed_authorize_fee_broadcast(
            channel_id=SCID, fee_ppm=250, old_fee_ppm=100,
            reason="dts", reason_code=None)
        assert ok is True
        proposed = [e for e in ledger.events()
                    if e["event_type"] == "intent_proposed"]
        assert "canonical_snapshot_id" not in proposed[0]["details"]
