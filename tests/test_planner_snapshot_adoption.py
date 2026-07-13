"""PR 3b (gap-closure Phase B): planner canonical-snapshot adoption.

- Bleeder classification freezes at cycle entry (one per-cycle
  projection, no mid-decision recomputation).
- Arbitration context and governed planner intents carry real snapshot
  ids from the shadow hub, with the synthetic labels as fail-open
  fallbacks (exact pre-adoption behavior).
- The dead never-called peer helpers are gone.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.capacity_planner import CapacityPlanner

NOW = 1_752_400_000


def _shadow_stub(snapshot_id="snap-42", observed_at=NOW):
    shadow = MagicMock()
    shadow.snapshot_ref.return_value = {
        "snapshot_id": snapshot_id, "observed_at": observed_at}
    shadow.ledger_for_reconciliation.return_value = None
    shadow.arbitration_registry.return_value = None
    return shadow


def _planner(enabled=True, shadow=None):
    planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock())
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_cycle_planner_enabled=enabled, paused=False,
        authority_level="capital")
    planner.config = cfg
    planner.econ_shadow = shadow
    return planner


def _loser(scid, roi=-90.0, reason="dead"):
    return {"scid": scid, "peer_id": "02" + "a" * 64,
            "marginal_roi": roi, "reason": reason}


def test_dead_peer_helpers_removed():
    """_has_direct_peer_channel/_is_peer_connected had no callers —
    dead live-RPC read paths removed by the audit (3b)."""
    assert not hasattr(CapacityPlanner, "_has_direct_peer_channel")
    assert not hasattr(CapacityPlanner, "_is_peer_connected")


class TestBleedersFrozenPerCycle:
    def test_unprimed_computes_once_and_stashes(self):
        planner = _planner()
        planner.profitability.identify_bleeders_v2.return_value = []
        planner._identify_losers({}, {})
        planner._identify_losers({}, {})
        assert planner.profitability.identify_bleeders_v2.call_count == 1

    def test_primed_projection_is_used_verbatim(self):
        planner = _planner()
        planner._cycle_bleeders = {}
        planner._identify_losers({}, {})
        planner.profitability.identify_bleeders_v2.assert_not_called()

    def test_init_cycle_cache_clears_projection_and_ref(self):
        planner = _planner()
        planner._cycle_bleeders = {"x": object()}
        planner._cycle_snapshot_ref = {"snapshot_id": "stale"}
        planner._init_cycle_cache()
        assert planner._cycle_bleeders is None
        assert planner._cycle_snapshot_ref is None


class TestArbitrationAdoption:
    def _captured_snapshot_ids(self, planner, monkeypatch):
        import modules.econ_intents as econ_intents
        seen = []
        real = econ_intents.make_intent

        def spy(**kwargs):
            seen.append(kwargs["snapshot_id"])
            return real(**kwargs)

        monkeypatch.setattr(econ_intents, "make_intent", spy)
        planner._arbitrate_close_list(
            [_loser("100x1x0"), _loser("200x1x0")],
            {"skipped_reasons": []})
        return seen

    def test_close_intents_use_hub_snapshot_id(self, monkeypatch):
        planner = _planner(shadow=_shadow_stub("snap-42"))
        ids = self._captured_snapshot_ids(planner, monkeypatch)
        assert ids and set(ids) == {"snap-42"}

    def test_fallback_to_synthetic(self, monkeypatch):
        planner = _planner(shadow=None)
        ids = self._captured_snapshot_ids(planner, monkeypatch)
        assert ids and all(i.startswith("planner-arb-") for i in ids)

    def test_arbitration_stashes_cycle_ref(self):
        planner = _planner(shadow=_shadow_stub("snap-42"))
        planner._arbitrate_close_list([_loser("100x1x0")],
                                      {"skipped_reasons": []})
        assert planner._cycle_snapshot_ref["snapshot_id"] == "snap-42"

    def test_dedup_unchanged(self):
        planner = _planner(shadow=_shadow_stub("snap-5"))
        summary = {"skipped_reasons": []}
        survivors = planner._arbitrate_close_list(
            [_loser("100x1x0"), _loser("100x1x0")], summary)
        assert len(survivors) == 1
        assert summary["skipped_reasons"]


class TestGovernedReserveAdoption:
    def _run_governed(self, planner, monkeypatch):
        import modules.econ_intents as econ_intents
        seen = {}
        real = econ_intents.make_intent

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(econ_intents, "make_intent", spy)
        db = MagicMock()
        db.reserve_spend.return_value = True
        ok = planner._governed_reserve_spend(
            db, reservation_id="open-1", amount_sats=100_000,
            category="channel_open", subcategory=None, metadata=None,
            effective_budget_sats=1_000_000, since_timestamp=NOW - 86400,
            intent_type="OPEN_CHANNEL", target="02" + "b" * 64,
            committed_sats=100_000)
        assert ok is True
        return seen

    def test_intent_uses_stashed_cycle_ref(self, monkeypatch):
        import time
        planner = _planner()
        # observed_at must be fresh: the stash carries a 600s age bound
        # (_governed_reserve_spend evaluates against real wall-clock).
        planner._cycle_snapshot_ref = {"snapshot_id": "snap-77",
                                       "observed_at": int(time.time())}
        seen = self._run_governed(planner, monkeypatch)
        assert seen["snapshot_id"] == "snap-77"

    def test_stale_stash_ignored(self, monkeypatch):
        planner = _planner(shadow=None)
        planner._cycle_snapshot_ref = {"snapshot_id": "snap-old",
                                       "observed_at": NOW}  # ancient
        seen = self._run_governed(planner, monkeypatch)
        assert seen["snapshot_id"].startswith("planner-cycle-")

    def test_intent_asks_hub_when_no_stash(self, monkeypatch):
        planner = _planner(shadow=_shadow_stub("snap-88"))
        seen = self._run_governed(planner, monkeypatch)
        assert seen["snapshot_id"] == "snap-88"

    def test_intent_falls_back_to_synthetic(self, monkeypatch):
        planner = _planner(shadow=None)
        seen = self._run_governed(planner, monkeypatch)
        assert seen["snapshot_id"].startswith("planner-cycle-")
