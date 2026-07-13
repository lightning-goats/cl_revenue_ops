"""PR 3c (gap-closure Phase B): Boltz canonical-snapshot adoption —
snapshot_id threading only (the audit found the Boltz decision path
already clean). Real snapshot ids come from the shadow hub; synthetic
labels remain as the fail-open fallback."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager
from tests.plugin_test_utils import load_plugin_module

NOW = 1_752_400_000


def _shadow_stub(snapshot_id="snap-42"):
    shadow = MagicMock()
    shadow.snapshot_ref.return_value = {"snapshot_id": snapshot_id,
                                        "observed_at": NOW}
    shadow.ledger_for_reconciliation.return_value = None
    shadow.arbitration_registry.return_value = None
    return shadow


def _manager(shadow=None):
    cfg = MagicMock(spec=BoltzCliConfig)
    cfg.enforce_budget = True
    manager = BoltzCliManager(MagicMock(), MagicMock(), cfg)
    capex = MagicMock()
    capex.reserve_boltz_swap_budget.return_value = True
    capex.release_boltz_swap_reservation.return_value = True
    manager._capex_engine = capex
    manager._get_global_budget_limit = lambda: {"budget_sats": 1000}
    manager.econ_governor_enabled_provider = lambda: True
    manager.econ_shadow = shadow
    return manager


def _captured(monkeypatch):
    import modules.econ_intents as econ_intents
    seen = {}
    real = econ_intents.make_intent

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(econ_intents, "make_intent", spy)
    return seen


class TestManagerReservation:
    def test_intent_uses_hub_snapshot_id(self, monkeypatch):
        seen = _captured(monkeypatch)
        manager = _manager(shadow=_shadow_stub("snap-63"))
        result = manager._open_swap_budget_reservation(
            214, "111x222x0", structural=False, intent_type="SWAP_OUT")
        assert result  # reservation granted
        assert seen["snapshot_id"] == "snap-63"

    def test_fallback_to_synthetic(self, monkeypatch):
        seen = _captured(monkeypatch)
        manager = _manager(shadow=None)
        assert manager._open_swap_budget_reservation(
            214, "111x222x0", structural=False, intent_type="SWAP_OUT")
        assert seen["snapshot_id"].startswith("boltz-swap-")

    def test_hub_error_fails_open(self, monkeypatch):
        seen = _captured(monkeypatch)
        shadow = _shadow_stub()
        shadow.snapshot_ref.side_effect = RuntimeError("boom")
        manager = _manager(shadow=shadow)
        assert manager._open_swap_budget_reservation(
            214, "111x222x0", structural=False, intent_type="SWAP_OUT")
        assert seen["snapshot_id"].startswith("boltz-swap-")


def _rec(channel="100x1x0", direction="loop_out", fee=100):
    return {"channel_id": channel, "direction": direction,
            "amount_sats": 250_000,
            "economics": {"passes_profit_guard": True,
                          "estimated_swap_fee_sats": fee}}


def _mod(shadow=None):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.config = MagicMock()
    mod.config.snapshot.return_value = SimpleNamespace(
        econ_cycle_boltz_enabled=True)
    mod.econ_shadow = shadow
    return mod


class TestBatchArbitration:
    def test_envelopes_use_hub_snapshot_id(self, monkeypatch):
        seen_ids = []
        import modules.econ_intents as econ_intents
        real = econ_intents.make_intent

        def spy(**kwargs):
            seen_ids.append(kwargs["snapshot_id"])
            return real(**kwargs)

        monkeypatch.setattr(econ_intents, "make_intent", spy)
        mod = _mod(shadow=_shadow_stub("snap-9"))
        survivors = mod._arbitrate_boltz_recommendations(
            [_rec(), _rec("200x1x0")])
        assert len(survivors) == 2
        assert set(seen_ids) == {"snap-9"}

    def test_fallback_to_synthetic(self, monkeypatch):
        seen_ids = []
        import modules.econ_intents as econ_intents
        real = econ_intents.make_intent

        def spy(**kwargs):
            seen_ids.append(kwargs["snapshot_id"])
            return real(**kwargs)

        monkeypatch.setattr(econ_intents, "make_intent", spy)
        mod = _mod(shadow=None)
        mod._arbitrate_boltz_recommendations([_rec()])
        assert seen_ids and all(
            i.startswith("boltz-arb-") for i in seen_ids)

    def test_dedup_unchanged(self):
        mod = _mod(shadow=_shadow_stub("snap-9"))
        survivors = mod._arbitrate_boltz_recommendations(
            [_rec("100x1x0"), _rec("100x1x0")])
        assert len(survivors) == 1
