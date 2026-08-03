"""Wave 2 (2026-08-01): registry lifecycle completion.

Six verified defects around the half-adopted live-arbitration registry:

1. governed callers never release their registry entry on a terminal
   outcome, so a failed execution blocks a legitimate retry with
   INTENT_SUPERSEDED for the full 600s envelope TTL;
2. a post-registration refusal (BUDGET_EXHAUSTED) leaks the just-armed
   entry;
3. the REBALANCE conflict identity omits the source leg, colliding two
   distinct pairs that share dest+amount;
4. first-registered OPEN_CHANNEL wins over a later higher-precedence
   contract obligation open (spec precedence inversion);
5. unlocked lazy registry init can hand two authorizations different
   registries;
6. batch arbitration fails open for the WHOLE cycle when one candidate
   is malformed.

Design decision (recorded): registry-entry release on terminal outcome
only; budget settlement stays on the legacy paths; payment_pending /
unknown-outcome / held-reservation states are NOT terminal.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.econ_arbiter import ActiveIntentRegistry, arbitrate
from modules.econ_intents import Explanation, make_intent
from modules.econ_types import Micro, Msat, SignedMsat, UnixTime
from modules.governor_facade import GovernorFacade

NOW = 1_752_400_000


def _intent(intent_type="REBALANCE", target="111x222x0", snapshot_id="s1",
            source=None, priority=50, max_cost_msat=0, amount_msat=None,
            reason_codes=()):
    components = [("x", 1)]
    if source is not None:
        components = [("source", source), ("dest", target)]
    return make_intent(
        intent_type=intent_type, snapshot_id=snapshot_id,
        created_at=UnixTime(NOW), expires_at=UnixTime(NOW + 600),
        target=target, amount_msat=amount_msat,
        expected_benefit_msat=SignedMsat(0),
        max_cost_msat=Msat(max_cost_msat),
        capital_committed_msat=Msat(0), confidence_micro=Micro(0),
        reason_codes=reason_codes,
        explanation=Explanation("t", tuple(components)),
        preconditions=(), priority=priority, budget_bucket="rebalance",
        origin_policy="test", reversible=False)


def _facade(registry, reserve=None, release=None):
    return GovernorFacade(
        reserve_spend=reserve or MagicMock(return_value=True),
        release_spend=release or MagicMock(return_value=True),
        is_paused=lambda: False,
        registry=registry)


# ---------------------------------------------------------------------------
# Defect 1 — facade completion API frees the registry slot
# ---------------------------------------------------------------------------
class TestFacadeComplete:
    def test_complete_frees_slot_for_retry(self):
        registry = ActiveIntentRegistry()
        facade = _facade(registry)
        env = _intent()
        decision = facade.authorize(env, NOW)
        assert decision.authorized
        # Without completion, the same identity is blocked.
        assert facade.authorize(env, NOW).reason_code == "INTENT_SUPERSEDED"
        facade.complete(decision.token.arbitration_key)
        assert facade.authorize(env, NOW).authorized is True

    def test_complete_without_registry_is_noop(self):
        facade = _facade(registry=None)
        env = _intent()
        decision = facade.authorize(env, NOW)
        facade.complete(decision.token.arbitration_key)  # must not raise

    def test_complete_survives_registry_error(self):
        registry = MagicMock()
        registry.check_and_register.return_value = None
        registry.release.side_effect = RuntimeError("boom")
        facade = _facade(registry)
        decision = facade.authorize(_intent(), NOW)
        facade.complete(decision.token.arbitration_key)  # guarded

    def test_complete_does_not_touch_budget(self):
        """Registry-only by design: budget settlement stays on the
        legacy paths (double-release risk)."""
        release = MagicMock(return_value=True)
        registry = ActiveIntentRegistry()
        facade = _facade(registry, release=release)
        decision = facade.authorize(_intent(max_cost_msat=5_000), NOW)
        facade.complete(decision.token.arbitration_key)
        release.assert_not_called()


# ---------------------------------------------------------------------------
# Defect 2 — refusal after registration releases the just-armed entry
# ---------------------------------------------------------------------------
class TestRefusalReleasesSlot:
    def test_budget_refusal_then_retry_unblocked(self):
        registry = ActiveIntentRegistry()
        reserve = MagicMock(side_effect=[False, True])
        facade = _facade(registry, reserve=reserve)
        env = _intent(max_cost_msat=5_000)
        first = facade.authorize(env, NOW)
        assert first.reason_code == "BUDGET_EXHAUSTED"
        # Budget freed: the same intent identity must not be misblocked
        # as INTENT_SUPERSEDED by the leaked entry.
        second = facade.authorize(env, NOW)
        assert second.authorized is True, second.reason_code

    def test_reserve_exception_releases_entry(self):
        registry = ActiveIntentRegistry()
        reserve = MagicMock(side_effect=[RuntimeError("db gone"), True])
        facade = _facade(registry, reserve=reserve)
        env = _intent(max_cost_msat=5_000)
        with pytest.raises(RuntimeError):
            facade.authorize(env, NOW)
        assert facade.authorize(env, NOW).authorized is True


# ---------------------------------------------------------------------------
# Defect 3 — REBALANCE conflict identity carries the source leg
# ---------------------------------------------------------------------------
class TestSourceLegIdentity:
    def _pair_envs(self):
        a = _intent(source="100x1x0", amount_msat=Msat(400_000_000))
        b = _intent(source="300x1x0", amount_msat=Msat(400_000_000))
        # Wire contract pins the key hash (five-field subset): the keys
        # COLLIDE by construction — the conflict identity must not.
        assert a.idempotency_key == b.idempotency_key
        return a, b

    def test_registry_admits_distinct_sources(self):
        a, b = self._pair_envs()
        registry = ActiveIntentRegistry()
        assert registry.check_and_register(a, NOW) is None
        assert registry.check_and_register(b, NOW) is None
        # A true duplicate (same source) is still superseded.
        assert registry.check_and_register(a, NOW) == "INTENT_SUPERSEDED"

    def test_batch_admits_distinct_sources(self):
        a, b = self._pair_envs()
        result = arbitrate([a, b], now=NOW)
        assert len(result.ordered) == 2
        assert result.rejected == ()

    def test_batch_still_supersedes_true_duplicates(self):
        a, _ = self._pair_envs()
        dup = _intent(source="100x1x0", amount_msat=Msat(400_000_000))
        result = arbitrate([a, dup], now=NOW)
        assert len(result.ordered) == 1
        assert result.rejected[0][1] == "INTENT_SUPERSEDED"

    def test_facade_release_frees_only_its_own_source_slot(self):
        a, b = self._pair_envs()
        registry = ActiveIntentRegistry()
        facade = _facade(registry)
        da = facade.authorize(a, NOW)
        db = facade.authorize(b, NOW)
        assert da.authorized and db.authorized
        facade.complete(da.token.arbitration_key)
        # a's slot is free again; b's stays armed.
        assert facade.authorize(a, NOW).authorized is True
        assert facade.authorize(b, NOW).reason_code == "INTENT_SUPERSEDED"


# ---------------------------------------------------------------------------
# Defect 4 — higher-precedence OPEN_CHANNEL preempts the armed one
# ---------------------------------------------------------------------------
class TestOpenPreemption:
    PEER = "02" + "c" * 64

    def _registry(self):
        return ActiveIntentRegistry(extended_rules_provider=lambda: True)

    def _planner_open(self):
        return _intent(intent_type="OPEN_CHANNEL", target=self.PEER,
                       snapshot_id="planner-cycle-1", priority=50)

    def _higher_priority_open(self):
        return _intent(intent_type="OPEN_CHANNEL", target=self.PEER,
                       snapshot_id="operator-contract-42", priority=80,
                       reason_codes=("CONTRACT_OBLIGATION",))

    def test_obligation_preempts_planner_open(self):
        registry = self._registry()
        assert registry.check_and_register(self._planner_open(), NOW) is None
        # The contract obligation outranks per the batch J3 ladder — it must
        # preempt, not be rejected.
        assert registry.check_and_register(self._higher_priority_open(), NOW) is None
        # The higher-priority entry now holds the slot.
        assert registry.check_and_register(self._planner_open(), NOW) == \
            "CONFLICT_DUPLICATE_OPEN"
        assert registry.active_count(NOW) == 1

    def test_lower_priority_incoming_still_blocked(self):
        registry = self._registry()
        assert registry.check_and_register(self._higher_priority_open(), NOW) is None
        assert registry.check_and_register(self._planner_open(), NOW) == \
            "CONFLICT_DUPLICATE_OPEN"

    def test_mirrors_batch_ordering(self):
        """The live preemption must agree with batch arbitration's
        winner for the same pair of intents."""
        batch = arbitrate([self._planner_open(), self._higher_priority_open()],
                          now=NOW, extended_rules=True)
        assert [e.priority for e in batch.ordered] == [80]


# ---------------------------------------------------------------------------
# Defect 5 — lazy registry init is race-free
# ---------------------------------------------------------------------------
class TestShadowRegistryInitLocked:
    def test_concurrent_first_access_yields_one_registry(self, tmp_path,
                                                         monkeypatch):
        from modules import econ_arbiter
        from modules.econ_shadow import EconShadow

        class SlowRegistry(econ_arbiter.ActiveIntentRegistry):
            def __init__(self, *a, **kw):
                import time as _t
                _t.sleep(0.05)  # widen the check-then-create window
                super().__init__(*a, **kw)

        monkeypatch.setattr(econ_arbiter, "ActiveIntentRegistry",
                            SlowRegistry)
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(
            econ_shadow_enabled=True, econ_arbiter_enabled=True)
        cfg.db_path = str(tmp_path / "revenue_ops.db")
        shadow = EconShadow(MagicMock(), cfg,
                            ledger_path=str(tmp_path / "l.db"))

        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(shadow.arbitration_registry())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 8
        assert all(r is results[0] for r in results), \
            "two authorizations got different registries"


# ---------------------------------------------------------------------------
# Defect 1 (caller side) — engine terminal completion unblocks retry
# ---------------------------------------------------------------------------
class TestEngineTerminalCompletion:
    def _engine(self, registry):
        from modules.config import Config
        from modules.rebalance_engine_v2 import RebalanceEngine
        engine = RebalanceEngine(plugin=MagicMock(),
                                 config=MagicMock(spec=Config),
                                 database=MagicMock())
        engine.database.reserve_budget.return_value = (True, 900)
        engine._executor_mode = lambda: "native"
        shadow = MagicMock()
        shadow.ledger_for_reconciliation.return_value = None
        shadow.arbitration_registry.return_value = registry
        shadow.snapshot_ref.return_value = None
        engine.econ_shadow = shadow
        return engine

    def _pair(self):
        return SimpleNamespace(
            source_channel_id="100x1x0", dest_channel_id="200x1x0",
            amount_sats=500_000, pair_budget_sats=100, score=0.298)

    def _cfg(self):
        return SimpleNamespace(paused=False, weekly_budget_sats=35_000,
                               authority_level="capital",
                               econ_ev_populated=False)

    def _reserve(self, engine, reservation_id):
        return engine._governed_reserve_execution_budget(
            self._pair(), reservation_id=reservation_id,
            max_fee_sats=100, cfg=self._cfg(), budget_limit=1000,
            since_ts=NOW - 86400, now=NOW)

    def test_failure_completion_unblocks_retry(self):
        from modules.rebalance_execution import ExecutionResult
        registry = ActiveIntentRegistry()
        engine = self._engine(registry)
        ok, res = self._reserve(engine, "res-1")
        assert ok is True and res is None
        # Same intent identity blocked while in flight.
        ok2, res2 = self._reserve(engine, "res-2")
        assert ok2 is False and "INTENT_SUPERSEDED" in res2.error
        # Terminal failure -> the slot frees and the retry is admitted.
        engine._finish_execution_budget(
            reservation_id="res-1", reserved_budget=True,
            result=ExecutionResult(success=False, amount_sats=500_000,
                                   error="no_route", route_type="native"))
        ok3, res3 = self._reserve(engine, "res-3")
        assert ok3 is True, getattr(res3, "error", None)

    def test_success_completion_frees_slot(self):
        from modules.rebalance_execution import ExecutionResult
        registry = ActiveIntentRegistry()
        engine = self._engine(registry)
        ok, _ = self._reserve(engine, "res-1")
        assert ok is True
        engine._finish_execution_budget(
            reservation_id="res-1", reserved_budget=True,
            result=ExecutionResult(success=True, amount_sats=500_000,
                                   fee_sats=7, route_type="native"))
        assert registry.active_count(NOW) == 0

    def test_pending_payment_hold_keeps_slot(self):
        """payment_pending with a sweepable hash is NOT terminal: the
        registry entry keeps blocking a concurrent duplicate; expiry
        handles the tail."""
        from modules.rebalance_execution import ExecutionResult
        registry = ActiveIntentRegistry()
        engine = self._engine(registry)
        ok, _ = self._reserve(engine, "res-1")
        assert ok is True
        engine._finish_execution_budget(
            reservation_id="res-1", reserved_budget=True,
            result=ExecutionResult(
                success=False, amount_sats=500_000, route_type="native",
                payment_pending=True,
                failure_data={"payment_hash": "ab" * 32}))
        ok2, res2 = self._reserve(engine, "res-2")
        assert ok2 is False and "INTENT_SUPERSEDED" in res2.error
        # Budget hold untouched (legacy behavior).
        engine.database.release_budget_reservation.assert_not_called()
        engine.database.mark_budget_spent.assert_not_called()

    def test_pending_without_hash_is_terminal(self):
        from modules.rebalance_execution import ExecutionResult
        registry = ActiveIntentRegistry()
        engine = self._engine(registry)
        ok, _ = self._reserve(engine, "res-1")
        assert ok is True
        engine._finish_execution_budget(
            reservation_id="res-1", reserved_budget=True,
            result=ExecutionResult(
                success=False, amount_sats=500_000, route_type="native",
                payment_pending=True, failure_data={}))
        assert registry.active_count(NOW) == 0


# ---------------------------------------------------------------------------
# Defect 1 (caller side) — contract / planner / boltz terminal hooks
# ---------------------------------------------------------------------------
PEER = "02" + "b" * 64


class TestBoltzTerminalCompletion:
    def _manager(self, registry):
        from modules.boltz_manager import BoltzCliConfig, BoltzCliManager
        cfg = MagicMock(spec=BoltzCliConfig)
        cfg.enforce_budget = True
        manager = BoltzCliManager(MagicMock(), MagicMock(), cfg)
        capex = MagicMock()
        capex.reserve_boltz_swap_budget.return_value = True
        capex.release_boltz_swap_reservation.return_value = True
        manager._capex_engine = capex
        manager._get_global_budget_limit = lambda: {"budget_sats": 1000}
        manager.econ_governor_enabled_provider = lambda: True
        shadow = MagicMock()
        shadow.ledger_for_reconciliation.return_value = None
        shadow.arbitration_registry.return_value = registry
        shadow.snapshot_ref.return_value = None
        manager.econ_shadow = shadow
        return manager

    def test_release_frees_slot_for_retry(self):
        registry = ActiveIntentRegistry()
        manager = self._manager(registry)
        rid = manager._open_swap_budget_reservation(
            214, "111x222x0", intent_type="SWAP_OUT")
        assert isinstance(rid, str)
        assert registry.active_count(NOW) == 1
        # Definite failure: finalize with no created swap -> release.
        manager._finalize_swap_budget_reservation(
            rid, None, 214, "111x222x0")
        assert registry.active_count(NOW) == 0
        rid2 = manager._open_swap_budget_reservation(
            214, "111x222x0", intent_type="SWAP_OUT")
        assert isinstance(rid2, str)


class TestPlannerTerminalCompletion:
    def _planner(self, registry):
        from modules.capacity_planner import CapacityPlanner
        planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock())
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(
            econ_governor_planner_enabled=True, paused=False,
            authority_level="capital")
        planner.config = cfg
        shadow = MagicMock()
        shadow.ledger_for_reconciliation.return_value = None
        shadow.arbitration_registry.return_value = registry
        shadow.snapshot_ref.return_value = None
        planner.econ_shadow = shadow
        return planner

    def _reserve(self, planner, db, reservation_id):
        return planner._governed_reserve_spend(
            db, reservation_id=reservation_id, amount_sats=300,
            category="channel_open", subcategory="automated",
            metadata={}, effective_budget_sats=1000, since_timestamp=NOW,
            intent_type="OPEN_CHANNEL", target=PEER,
            committed_sats=1_000_000)

    def test_settle_frees_slot(self):
        registry = ActiveIntentRegistry()
        planner = self._planner(registry)
        db = MagicMock()
        db.reserve_spend.return_value = True
        db.mark_spend_reservation_spent.return_value = True
        assert self._reserve(planner, db, "rid-1") is True
        assert registry.active_count(NOW) == 1
        planner._settle_capex_reservation(db, "rid-1", 300, what="test")
        assert registry.active_count(NOW) == 0

    def test_completion_helper_frees_slot_on_failure_release(self):
        registry = ActiveIntentRegistry()
        planner = self._planner(registry)
        db = MagicMock()
        db.reserve_spend.return_value = True
        assert self._reserve(planner, db, "rid-1") is True
        assert self._reserve(planner, db, "rid-2") is False  # in flight
        planner._complete_governed_intent("rid-1")
        assert self._reserve(planner, db, "rid-3") is True


class TestFeeBroadcastCompletion:
    def test_completion_stashed_and_frees_slot(self):
        from modules.config import Config
        from modules.fee_authority import FeeAuthorityGate
        from modules.fee_controller import FeeController
        registry = ActiveIntentRegistry()
        cfg = MagicMock(spec=Config)
        cfg.snapshot.return_value = SimpleNamespace(
            econ_governor_fees_enabled=True, paused=False,
            authority_level="capital")
        fc = FeeController(MagicMock(), cfg, MagicMock(),
                           fee_authority_gate=FeeAuthorityGate())
        fc.config = cfg
        shadow = MagicMock()
        shadow.ledger_for_reconciliation.return_value = None
        shadow.arbitration_registry.return_value = registry
        shadow.snapshot_ref.return_value = None
        fc.econ_shadow = shadow
        ok, reason = fc._governed_authorize_fee_broadcast(
            channel_id="123x456x0", fee_ppm=250, old_fee_ppm=100,
            reason="dts", reason_code="dts_pid_sample")
        assert ok is True and reason == ""
        assert registry.active_count(NOW) == 1
        completion = fc._governed_intent_completions.pop("123x456x0")
        facade, key = completion
        facade.complete(key)
        assert registry.active_count(NOW) == 0

    def test_broadcast_completion_is_wired(self):
        """Structural pin: the terminal completion runs at the broadcast
        site in _set_channel_fee_inner (success AND failure)."""
        import pathlib
        source = (pathlib.Path(__file__).resolve().parent.parent
                  / "modules" / "fee_controller.py").read_text()
        gate = source.find("gov_ok, gov_reason = "
                           "self._governed_authorize_fee_broadcast(")
        assert gate > 0
        completion = source.find("_governed_intent_completions.pop", gate)
        rpc = source.find("rpc_result = self.data_service.set_channel(",
                          gate)
        assert 0 < completion < rpc, \
            "fee broadcast completion must be popped before the RPC and " \
            "invoked on its terminal outcome"


# ---------------------------------------------------------------------------
# Defect 6 — per-candidate batch isolation
# ---------------------------------------------------------------------------
class TestEngineBatchIsolation:
    def _engine(self):
        from modules.config import Config
        from modules.rebalance_engine_v2 import RebalanceEngine
        engine = RebalanceEngine(plugin=MagicMock(),
                                 config=MagicMock(spec=Config),
                                 database=MagicMock())
        engine._config_snapshot = lambda: SimpleNamespace(
            econ_cycle_rebalance_enabled=True)
        return engine

    def _pair(self, src="100x1x0", dst="200x1x0", amount=500_000):
        return SimpleNamespace(source_channel_id=src, dest_channel_id=dst,
                               amount_sats=amount, pair_budget_sats=100,
                               score=0.298)

    def test_single_malformed_candidate_drops_alone(self):
        from modules.rebalance_engine_v2 import CycleResult
        engine = self._engine()
        good1 = self._pair(dst="100x9x0")
        bad = self._pair(dst="500x9x0", amount=-5)  # Msat(-5000) raises
        good2 = self._pair(dst="900x9x0")
        result = CycleResult()
        ordered = engine._arbitrate_execution_list(
            [good1, bad, good2], result)
        assert [p.dest_channel_id for p in ordered] == \
            ["100x9x0", "900x9x0"]

    def test_distinct_source_pairs_both_admitted(self):
        from modules.rebalance_engine_v2 import CycleResult
        engine = self._engine()
        a = self._pair(src="100x1x0", dst="200x1x0")
        b = self._pair(src="300x1x0", dst="200x1x0")
        ordered = engine._arbitrate_execution_list([a, b], CycleResult())
        assert len(ordered) == 2
        assert {p.source_channel_id for p in ordered} == \
            {"100x1x0", "300x1x0"}

    def test_infra_failure_still_fails_open(self, monkeypatch):
        from modules.rebalance_engine_v2 import CycleResult
        engine = self._engine()
        pairs = [self._pair()]
        monkeypatch.setattr("modules.econ_arbiter.arbitrate",
                            MagicMock(side_effect=RuntimeError("boom")))
        assert engine._arbitrate_execution_list(pairs, CycleResult()) \
            is pairs


class TestPlannerBatchIsolation:
    def _planner(self):
        from modules.capacity_planner import CapacityPlanner
        planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock())
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(
            econ_cycle_planner_enabled=True)
        planner.config = cfg
        planner.econ_shadow = None
        return planner

    def _loser(self, scid, roi=-90.0):
        return {"scid": scid, "peer_id": "02" + "a" * 64,
                "marginal_roi": roi, "reason": "dead"}

    def test_single_malformed_candidate_drops_alone(self):
        planner = self._planner()
        good = self._loser("100x1x0")
        bad = self._loser("500x1x0", roi="not-a-number")  # float() raises
        summary = {"skipped_reasons": []}
        survivors = planner._arbitrate_close_list([good, bad], summary)
        assert [l["scid"] for l in survivors] == ["100x1x0"]
        assert any("500x1x0" in r for r in summary["skipped_reasons"])
