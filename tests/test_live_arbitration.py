"""Phase 3F: live conflict arbitration at the governor boundary."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.econ_arbiter import ActiveIntentRegistry
from modules.econ_intents import Explanation, make_intent
from modules.econ_types import Micro, Msat, SignedMsat, UnixTime
from modules.governor_facade import GovernorFacade

NOW = 1_752_400_000


def _intent(intent_type="REBALANCE", target="111x222x0", snapshot_id="s1",
            expires=600):
    return make_intent(
        intent_type=intent_type, snapshot_id=snapshot_id,
        created_at=UnixTime(NOW), expires_at=UnixTime(NOW + expires),
        target=target, amount_msat=None,
        expected_benefit_msat=SignedMsat(0), max_cost_msat=Msat(0),
        capital_committed_msat=Msat(0), confidence_micro=Micro(0),
        reason_codes=(), explanation=Explanation("t", (("x", 1),)),
        preconditions=(), priority=50, budget_bucket="rebalance",
        origin_policy="test", reversible=False)


class TestRegistry:
    def test_duplicate_key_superseded(self):
        registry = ActiveIntentRegistry()
        env = _intent()
        assert registry.check_and_register(env, NOW) is None
        assert registry.check_and_register(env, NOW) == "INTENT_SUPERSEDED"

    def test_close_blocks_rebalance_on_same_target(self):
        registry = ActiveIntentRegistry()
        close = _intent(intent_type="CLOSE_CHANNEL", target="111x222x0",
                        snapshot_id="close-cycle")
        assert registry.check_and_register(close, NOW) is None
        reb = _intent(intent_type="REBALANCE", target="111x222x0")
        assert registry.check_and_register(reb, NOW) == \
            "CONFLICT_CLOSE_REBALANCE"
        # Different target is unaffected.
        other = _intent(intent_type="REBALANCE", target="999x1x0")
        assert registry.check_and_register(other, NOW) is None

    def test_expiry_prunes(self):
        registry = ActiveIntentRegistry()
        close = _intent(intent_type="CLOSE_CHANNEL", expires=60)
        registry.check_and_register(close, NOW)
        reb = _intent(intent_type="REBALANCE")
        # After the close intent expires, the rebalance is allowed.
        assert registry.check_and_register(reb, NOW + 120) is None
        assert registry.active_count(NOW + 120) == 1

    def test_release_frees_the_slot(self):
        registry = ActiveIntentRegistry()
        env = _intent()
        registry.check_and_register(env, NOW)
        registry.release(env.idempotency_key)
        assert registry.check_and_register(env, NOW) is None


class TestFacadeIntegration:
    def _facade(self, registry, ledger=None):
        return GovernorFacade(
            reserve_spend=MagicMock(return_value=True),
            release_spend=MagicMock(return_value=True),
            is_paused=lambda: False,
            ledger=ledger, registry=registry)

    def test_conflict_rejected_and_ledgered(self, tmp_path):
        from modules.econ_ledger import EconLedger
        ledger = EconLedger(str(tmp_path / "l.db"))
        registry = ActiveIntentRegistry()
        facade = self._facade(registry, ledger)

        close = _intent(intent_type="CLOSE_CHANNEL", target="111x222x0",
                        snapshot_id="planner-cycle-1")
        assert facade.authorize(close, NOW).authorized is True

        reb = _intent(intent_type="REBALANCE", target="111x222x0")
        decision = facade.authorize(reb, NOW)
        assert decision.authorized is False
        assert decision.reason_code == "CONFLICT_CLOSE_REBALANCE"
        rejected = [e for e in ledger.events()
                    if e["event_type"] == "intent_rejected"]
        assert rejected and rejected[0]["details"]["arbitration"] is True

    def test_release_unregisters(self):
        registry = ActiveIntentRegistry()
        facade = self._facade(registry)
        env = _intent()
        decision = facade.authorize(env, NOW)
        assert decision.authorized
        facade.release(decision.token, NOW)
        # Same intent can be re-authorized after release.
        assert facade.authorize(env, NOW).authorized is True

    def test_no_registry_means_no_arbitration(self):
        facade = self._facade(registry=None)
        env = _intent()
        assert facade.authorize(env, NOW).authorized is True
        assert facade.authorize(env, NOW).authorized is True  # dup fine


class TestReservationKeyCoherence:
    """Live finding 2026-07-13 (first governed rebalance in production):
    the facade ledgered budget_reserved under the ENVELOPE key while the
    delegate reserved under the caller's reservation_id — every governed
    spend left a phantom ledger reservation for the sweep to correct.
    Pin: with reservation_id passed, the ledger trail keys consistently
    and replays clean with ZERO reconciliation needed."""

    def test_governed_spend_replays_clean(self, tmp_path):
        from modules.econ_ledger import EconLedger
        from modules.econ_reconcile import reconcile
        ledger = EconLedger(str(tmp_path / "l.db"))
        facade = GovernorFacade(
            reserve_spend=MagicMock(return_value=True),
            release_spend=MagicMock(return_value=True),
            is_paused=lambda: False, ledger=ledger)
        env = _intent()
        env = make_intent(
            intent_type="REBALANCE", snapshot_id="s1",
            created_at=UnixTime(NOW), expires_at=UnixTime(NOW + 600),
            target="111x222x0", amount_msat=None,
            expected_benefit_msat=SignedMsat(0),
            max_cost_msat=Msat(400_000),  # nonzero -> reservation
            capital_committed_msat=Msat(0), confidence_micro=Micro(0),
            reason_codes=(), explanation=Explanation("t", (("x", 1),)),
            preconditions=(), priority=50, budget_bucket="rebalance",
            origin_policy="test", reversible=False)
        decision = facade.authorize(env, NOW, reservation_id="504")
        assert decision.authorized
        assert decision.token.reservation_id == "504"
        assert decision.token.arbitration_key == env.idempotency_key
        facade.release(decision.token, NOW + 5)

        events = ledger.events()
        reserved_keys = {e["idempotency_key"] for e in events
                         if e["event_type"] == "budget_reserved"}
        released_keys = {e["idempotency_key"] for e in events
                         if e["event_type"] == "reservation_released"}
        assert reserved_keys == {"504"} == released_keys
        state = ledger.replay()
        assert state.reserved_msat == {}  # no phantom
        report = reconcile(ledger, {}, now=NOW + 10)
        assert report.divergences == ()  # nothing for the sweep to fix

    def test_registry_slot_freed_despite_key_split(self):
        registry = ActiveIntentRegistry()
        facade = GovernorFacade(
            reserve_spend=MagicMock(return_value=True),
            release_spend=MagicMock(return_value=True),
            is_paused=lambda: False, registry=registry)
        env = _intent()
        decision = facade.authorize(env, NOW, reservation_id="504")
        assert decision.authorized
        facade.release(decision.token, NOW)
        # The arbitration slot (keyed by envelope key) must be free.
        assert facade.authorize(env, NOW,
                                reservation_id="505").authorized is True


class TestShadowAccessor:
    def _shadow(self, tmp_path, arbiter=True):
        from modules.econ_shadow import EconShadow
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(
            econ_shadow_enabled=True, econ_arbiter_enabled=arbiter)
        cfg.db_path = str(tmp_path / "revenue_ops.db")
        return EconShadow(MagicMock(), cfg,
                          ledger_path=str(tmp_path / "l.db"))

    def test_flag_gates_registry(self, tmp_path):
        assert self._shadow(tmp_path, arbiter=False) \
            .arbitration_registry() is None
        shadow = self._shadow(tmp_path, arbiter=True)
        registry = shadow.arbitration_registry()
        assert registry is not None
        # Singleton: every governed path shares the same state.
        assert shadow.arbitration_registry() is registry

    def test_cross_path_conflict_end_to_end(self, tmp_path):
        """A planner CLOSE_CHANNEL authorization blocks a subsequent
        engine REBALANCE authorization on the same channel — through
        two separately-constructed facades sharing the shadow registry."""
        shadow = self._shadow(tmp_path, arbiter=True)
        registry = shadow.arbitration_registry()

        planner_facade = GovernorFacade(
            reserve_spend=MagicMock(return_value=True),
            release_spend=MagicMock(return_value=True),
            is_paused=lambda: False, registry=registry)
        engine_facade = GovernorFacade(
            reserve_spend=MagicMock(return_value=True),
            release_spend=MagicMock(return_value=True),
            is_paused=lambda: False,
            registry=shadow.arbitration_registry())

        close = _intent(intent_type="CLOSE_CHANNEL", target="111x222x0",
                        snapshot_id="planner-cycle-9")
        assert planner_facade.authorize(close, NOW).authorized is True
        reb = _intent(intent_type="REBALANCE", target="111x222x0")
        decision = engine_facade.authorize(reb, NOW)
        assert decision.authorized is False
        assert decision.reason_code == "CONFLICT_CLOSE_REBALANCE"
