"""PR 3d (gap-closure Phase B): LN+ canonical-snapshot adoption.

Event-driven policy, so "cycle" = one evaluation pass:
- our node id becomes a process-constant cache (it never changes);
- the existing-channel gate consults ONE peer-channel capture frozen at
  pass entry (same any-state semantics), with the per-swap live read
  kept as the fail-open fallback;
- the swap-scoped intent snapshot_id label is DELIBERATELY unchanged —
  it provides cross-attempt idempotency for contractual obligations
  (idempotency key hashes snapshot_id, not created_at); the canonical
  snapshot ref is recorded as ledger evidence instead.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.econ_ledger import EconLedger
from modules.lnplus_swaps import SwapEvaluator, SwapLifecycle

PK_A = "02" + "a" * 64
PK_B = "03" + "b" * 64
PEER = "02" + "b" * 64
NOW = 1_752_400_000


def _swap(swap_id="sw1"):
    return {
        "id": swap_id, "status": "pending",
        "capacity_sats": 5_000_000, "duration_months": 3,
        "participant_max_count": 3,
        "participant_applied_count": 2,
        "participant_waiting_for_count": 1,
        "clearnet_connection_allowed": True,
        "tor_connection_allowed": True,
        "platform": "any",
        "participants": [
            {"participant_identifier": "A", "pubkey": PK_A,
             "positive_ratings_count": 20, "negative_ratings_count": 0,
             "lnplus_rank_number": 9,
             "address_1": "1.2.3.4:9735", "capacity_sats": 100_000_000,
             "channels_count": 40},
            {"participant_identifier": "B", "pubkey": PK_B,
             "positive_ratings_count": 12, "negative_ratings_count": 1,
             "lnplus_rank_number": 8,
             "address_1": "5.6.7.8:9735", "capacity_sats": 80_000_000,
             "channels_count": 25},
        ],
    }


def _evaluator(swaps, channels=None):
    cfg = Config()
    cfg.lnplus_swaps_enabled = True
    plugin, rpc = MagicMock(), MagicMock()
    rpc.feerates.return_value = {"perkw": {"opening": 2500}}
    rpc.getinfo.return_value = {"id": "02" + "e" * 64}
    rpc.listfunds.return_value = {"outputs": [
        {"amount_msat": 100_000_000_000, "status": "confirmed",
         "reserved": False}]}
    rpc.listpeerchannels.return_value = {"channels": channels or []}
    db = MagicMock()
    db.lnplus_get_peer.return_value = None
    client = MagicMock()
    client.get_applicable_swaps.return_value = swaps
    planner = MagicMock()
    planner._calculate_open_ev.return_value = 1000.0
    planner._estimate_open_cost.return_value = 2000
    planner._capex_engine.get_fleet_exploration_budget.return_value = \
        1_000_000_000
    lifecycle = MagicMock()
    lifecycle.breaker_tripped.return_value = None
    lifecycle.has_inflight.return_value = False
    lifecycle.reconcile_ok.return_value = True
    ev = SwapEvaluator(plugin, rpc, db, cfg, client, planner, lifecycle)
    return ev, cfg, rpc


class TestOurIdProcessCache:
    def test_getinfo_called_once_across_passes(self):
        ev, cfg, rpc = _evaluator([_swap("sw1"), _swap("sw2")])
        ev.run_cycle(cfg, 0.0)
        ev.run_cycle(cfg, 0.0)
        assert rpc.getinfo.call_count == 1

    def test_getinfo_failure_retries_next_use(self):
        ev, cfg, rpc = _evaluator([_swap("sw1")])
        rpc.getinfo.side_effect = [RuntimeError("down"),
                                   {"id": "02" + "e" * 64}]
        ev.run_cycle(cfg, 0.0)   # failure — not cached
        ev.run_cycle(cfg, 0.0)   # succeeds and caches
        ev.run_cycle(cfg, 0.0)   # cached
        assert rpc.getinfo.call_count == 2


class TestPassFrozenChannelSet:
    def test_one_capture_per_pass(self):
        ev, cfg, rpc = _evaluator([_swap("sw1"), _swap("sw2")])
        ev.run_cycle(cfg, 0.0)
        # exactly one no-arg capture; no per-swap peer queries
        assert rpc.listpeerchannels.call_count == 1
        assert rpc.listpeerchannels.call_args.args == ()

    def test_existing_channel_still_rejects_any_state(self):
        # Swap sw1 infers outbound peer = participant after our slot;
        # give that peer a PENDING (OPENINGD) channel — the gate must
        # still reject (any-state semantics preserved from live read).
        swap = _swap("sw1")
        ev, cfg, rpc = _evaluator([swap])
        outbound = ev._infer_assignment(swap).get("outbound_peer")
        assert outbound
        ev2, cfg2, rpc2 = _evaluator(
            [swap], channels=[{"peer_id": outbound, "state": "OPENINGD"}])
        result = ev2.run_cycle(cfg2, 0.0)
        assert any("existing channel" in r.get("reason", "")
                   for r in result["rejections"])

    def test_capture_failure_falls_back_to_per_swap_read(self):
        swap = _swap("sw1")
        ev, cfg, rpc = _evaluator([swap])
        outbound = ev._infer_assignment(swap).get("outbound_peer")

        def lpc(*args, **kwargs):
            if not args and not kwargs:
                raise RuntimeError("capture down")
            return {"channels": [{"peer_id": outbound,
                                  "state": "CHANNELD_NORMAL"}]}

        rpc.listpeerchannels.side_effect = lpc
        result = ev.run_cycle(cfg, 0.0)
        assert any("existing channel" in r.get("reason", "")
                   for r in result["rejections"])

    def test_feerate_read_stays_once_per_pass(self):
        ev, cfg, rpc = _evaluator([_swap("sw1"), _swap("sw2")])
        ev.run_cycle(cfg, 0.0)
        assert rpc.feerates.call_count == 1


class TestGovernedReserveEvidence:
    def _lifecycle(self, tmp_path, shadow=None):
        db = MagicMock()
        db.reserve_spend.return_value = True
        db.release_spend_reservation.return_value = True
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(
            econ_governor_lnplus_enabled=True, paused=False)
        lifecycle = SwapLifecycle(
            MagicMock(), MagicMock(), db, cfg, MagicMock(), MagicMock())
        ledger = EconLedger(str(tmp_path / "ledger.db"))
        stub = MagicMock()
        stub.ledger_for_reconciliation.return_value = ledger
        stub.arbitration_registry.return_value = None
        stub.snapshot_ref.return_value = shadow
        lifecycle.econ_shadow = stub
        return lifecycle, ledger

    def _reserve(self, lifecycle):
        return lifecycle._governed_reserve_spend(
            reservation_id="lnplus-open-42-1752300000", amount_sats=214,
            metadata={"swap_id": 42, "peer_id": PEER},
            effective_budget_sats=1000, since_timestamp=NOW,
            swap_id=42, peer_id=PEER, capacity_sats=2_000_000)

    def test_label_unchanged_and_snapshot_in_ledger_evidence(self, tmp_path):
        lifecycle, ledger = self._lifecycle(
            tmp_path, shadow={"snapshot_id": "snap-55", "observed_at": NOW})
        assert self._reserve(lifecycle) is True
        proposed = [e for e in ledger.events()
                    if e["event_type"] == "intent_proposed"]
        assert proposed
        # Idempotency-bearing label untouched (obligation retry identity)
        assert proposed[0]["cycle_id"] == "lnplus-swap-42"
        # Canonical linkage lands in evidence
        assert proposed[0]["details"]["canonical_snapshot_id"] == "snap-55"

    def test_no_hub_no_evidence_field(self, tmp_path):
        lifecycle, ledger = self._lifecycle(tmp_path, shadow=None)
        assert self._reserve(lifecycle) is True
        proposed = [e for e in ledger.events()
                    if e["event_type"] == "intent_proposed"]
        assert proposed[0]["cycle_id"] == "lnplus-swap-42"
        assert "canonical_snapshot_id" not in proposed[0]["details"]
