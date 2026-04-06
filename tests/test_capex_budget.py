"""Tests for unified capex budget engine."""

import os
import sys
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import Config, ConfigSnapshot


class TestCapexBudgetConfig:
    """Capex budget config fields exist with correct defaults."""

    def test_reinvestment_rate(self):
        assert Config().capex_reinvestment_rate == 0.50

    def test_bootstrap_bps(self):
        assert Config().capex_bootstrap_bps == 10

    def test_bootstrap_max_sats(self):
        assert Config().capex_bootstrap_max_sats == 200

    def test_grace_days(self):
        assert Config().capex_grace_days == 14

    def test_exploration_rate(self):
        assert Config().capex_exploration_rate == 0.10

    def test_tactical_rate(self):
        assert Config().capex_tactical_rate == 0.15

    def test_global_envelope(self):
        assert Config().capex_global_envelope_sats == 0

    def test_snapshot_includes_all_fields(self):
        snap = Config().snapshot()
        assert snap.capex_reinvestment_rate == 0.50
        assert snap.capex_bootstrap_bps == 10
        assert snap.capex_bootstrap_max_sats == 200
        assert snap.capex_grace_days == 14
        assert snap.capex_exploration_rate == 0.10
        assert snap.capex_tactical_rate == 0.15
        assert snap.capex_global_envelope_sats == 0


from modules.capex_budget import (
    CapexBudgetEngine,
    ChannelCapexBudget,
    CapexAllocations,
)


def _make_engine(
    channel_profitabilities=None,
    bleeders=None,
    spend_by_channel=None,
    rebalance_cost_by_channel=None,
    hive_hints=None,
    reserve_deficit=0,
    config_overrides=None,
):
    """Create a CapexBudgetEngine with mocked dependencies."""
    mock_profitability = MagicMock()
    mock_profitability.analyze_all_channels.return_value = channel_profitabilities or {}
    mock_profitability.get_bleeder_status.return_value = None

    mock_db = MagicMock()
    mock_db.get_confirmed_onchain_sats.return_value = 1_000_000 - reserve_deficit

    cfg = Config()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    engine = CapexBudgetEngine(
        profitability_analyzer=mock_profitability,
        database=mock_db,
        config=cfg,
        hive_hints=hive_hints,
    )
    # Patch the capex lookup to return test data
    capex_data = {}
    if rebalance_cost_by_channel:
        capex_data.update(rebalance_cost_by_channel)
    if spend_by_channel:
        for k, v in spend_by_channel.items():
            capex_data[k] = capex_data.get(k, 0) + v
    engine._get_total_capex_by_channel = lambda window_days=30: capex_data
    return engine


class TestEngineConstruction:
    """Engine can be constructed and has correct interface."""

    def test_construction_without_hive(self):
        engine = _make_engine()
        assert engine is not None

    def test_construction_with_hive(self):
        mock_hive = MagicMock()
        engine = _make_engine(hive_hints=mock_hive)
        assert engine is not None

    def test_compute_allocations_returns_dataclass(self):
        engine = _make_engine()
        alloc = engine.compute_allocations()
        assert isinstance(alloc, CapexAllocations)
        assert isinstance(alloc.channel_budgets, dict)
        assert isinstance(alloc.fleet_exploration_budget_sats, int)
        assert isinstance(alloc.tactical_budget_sats, int)
        assert alloc.priority_class in ("defensive", "preservation", "operational", "growth")
