"""Phase 2C/2D: shadow-governor hook on the rebalance engine's real
reservation decision, and (D) the flag-gated governed entry."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.rebalance_engine_v2 import RebalanceEngine


def _pair():
    return SimpleNamespace(
        source_channel_id="100x1x0", dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64, dest_peer_id="02" + "b" * 64,
        amount_sats=500_000, score=0.298)


def _engine(reserve_result=(True, 900), max_fee_sats=100,
            paused=False, governed=False):
    plugin = MagicMock()
    cfg = MagicMock(spec=Config)
    database = MagicMock()
    database.reserve_budget.return_value = reserve_result
    engine = RebalanceEngine(plugin=plugin, config=cfg, database=database)
    snap = SimpleNamespace(
        total_cost_budget_window_hours=24,
        weekly_budget_sats=35_000,
        allow_zero_cost_auto_rebalance_when_budget_zero=False,
        paused=paused,
        econ_governor_rebalance_enabled=governed,
    )
    engine._config_snapshot = lambda: snap
    engine._get_global_budget_limit = lambda cfg: 1000
    engine._pair_max_fee_sats = lambda pair: max_fee_sats
    engine._executor_mode = lambda: "native"
    return engine


class TestShadowHook:
    def test_hook_fires_on_reserved(self):
        engine = _engine(reserve_result=(True, 900))
        shadow = MagicMock()
        engine.econ_shadow = shadow
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is True and result is None
        kwargs = shadow.shadow_authorize_rebalance.call_args.kwargs
        assert kwargs["legacy_reserved"] is True
        assert kwargs["remaining_sats"] == 900
        assert kwargs["max_fee_sats"] == 100
        assert kwargs["budget_limit_sats"] == 1000

    def test_hook_fires_on_budget_block(self):
        engine = _engine(reserve_result=(False, 20))
        shadow = MagicMock()
        engine.econ_shadow = shadow
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is False
        assert "local_budget_block" in result.error
        kwargs = shadow.shadow_authorize_rebalance.call_args.kwargs
        assert kwargs["legacy_reserved"] is False
        assert kwargs["remaining_sats"] == 20

    def test_raising_shadow_never_affects_decision(self):
        engine = _engine(reserve_result=(True, 900))
        shadow = MagicMock()
        shadow.shadow_authorize_rebalance.side_effect = RuntimeError("boom")
        engine.econ_shadow = shadow
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is True and result is None

    def test_absent_shadow_harmless(self):
        engine = _engine(reserve_result=(True, 900))
        assert engine.econ_shadow is None
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is True and result is None


class TestGovernedEntry:
    """Phase 2D: flag-gated authorization boundary. Default OFF ==
    byte-identical legacy behavior (TestShadowHook above runs with the
    flag unset/false)."""

    def test_governed_success_reserves_under_original_id(self):
        engine = _engine(reserve_result=(True, 900), governed=True)
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-orig")
        assert ok is True and result is None
        kwargs = engine.database.reserve_budget.call_args.kwargs
        assert kwargs["reservation_id"] == "res-orig"  # finish path intact

    def test_governed_paused_blocks_without_reserving(self):
        engine = _engine(reserve_result=(True, 900), governed=True,
                         paused=True)
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is False
        assert "governor_block: PAUSED" in result.error
        engine.database.reserve_budget.assert_not_called()

    def test_governed_budget_block_keeps_legacy_error_shape(self):
        engine = _engine(reserve_result=(False, 20), governed=True)
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is False
        assert result.error.startswith("local_budget_block")

    def test_governed_path_fails_closed_on_internal_error(self):
        engine = _engine(reserve_result=(True, 900), governed=True)
        engine.database.reserve_budget.side_effect = RuntimeError("db gone")
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is False
        assert "governor_block" in result.error \
            or "local_budget_block" in result.error

    def test_legacy_mode_ignores_governor_flag_absent(self):
        # MagicMock snapshots (spec-less attrs) must NOT accidentally
        # enable the governed path: the flag check is strict.
        engine = _engine(reserve_result=(True, 900))
        engine._config_snapshot = lambda: MagicMock()  # truthy attrs
        ok, result = engine._reserve_execution_budget(
            _pair(), reservation_id="res-1")
        assert ok is True and result is None
        # legacy path used: reserve_budget called directly
        engine.database.reserve_budget.assert_called_once()
