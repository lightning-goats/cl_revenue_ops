"""Tests for capex-aware rebalancer (engine-driven)."""

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
from modules.rebalancer import EVRebalancer as Rebalancer, RebalanceReasonCode
from modules.capex_budget import CapexBudgetEngine, ChannelCapexBudget, CapexAllocations


def _make_rebalancer(capex_engine=None):
    """Create a minimal Rebalancer with optional capex engine."""
    mock_plugin = MagicMock()
    mock_plugin.rpc = MagicMock()
    mock_db = MagicMock()
    mock_config = MagicMock()
    mock_config.rebalance_max_amount = 5_000_000
    mock_config.rebalance_min_amount = 50_000
    mock_config.snapshot.return_value = mock_config

    rebalancer = Rebalancer.__new__(Rebalancer)
    rebalancer.plugin = mock_plugin
    rebalancer.database = mock_db
    rebalancer.config = mock_config
    rebalancer._profitability_analyzer = MagicMock()
    rebalancer._capex_engine = capex_engine
    rebalancer._hive_router = None
    return rebalancer


class TestEngineInjection:
    """Capex engine can be injected into the rebalancer."""

    def test_set_capex_engine(self):
        r = _make_rebalancer()
        mock_engine = MagicMock()
        r.set_capex_engine(mock_engine)
        assert r._capex_engine is mock_engine

    def test_default_no_engine(self):
        r = _make_rebalancer()
        assert r._capex_engine is None


class TestCapexFallbackWithEngine:
    """_capex_fallback_pass uses engine budgets."""

    def _make_engine_with_budget(self, channel_id, budget_sats, tier, tier_ppm):
        """Create a mock engine that returns a specific budget for one channel."""
        mock_engine = MagicMock(spec=CapexBudgetEngine)
        budget = ChannelCapexBudget(
            channel_id=channel_id,
            budget_msat=budget_sats * 1000,
            tier=tier,
            tier_ppm=tier_ppm,
            priority_class="preservation" if tier in ("proven", "active") else "growth",
        )
        mock_engine.get_channel_budget.return_value = budget
        return mock_engine

    def test_proven_channel_gets_candidate(self):
        """Proven channel with budget gets a capex rebalance candidate."""
        engine = self._make_engine_with_budget("100x1x0", 300, "proven", 2000)
        r = _make_rebalancer(capex_engine=engine)

        # Mock source selection
        source_info = {
            "peer_id": "02" + "b" * 64,
            "capacity": 5_000_000,
            "fee_ppm": 100,
            "spendable": 3_000_000,
        }
        r._select_source_candidates = MagicMock(return_value=[
            ("200x1x0", source_info, 50.0, 10)
        ])

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]
        source_channels = [("200x1x0", source_info, 0.80)]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=source_channels,
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 1
        c = candidates[0]
        assert c.reason_code == "capex_fallback"
        assert c.max_budget_sats > 0
        assert c.max_budget_sats <= 300
        assert c.max_fee_ppm <= 2000
        engine.get_channel_budget.assert_called_with("100x1x0")

    def test_blocked_channel_skipped(self):
        """Blocked channel gets no candidate."""
        engine = self._make_engine_with_budget("100x1x0", 0, "blocked", 0)
        r = _make_rebalancer(capex_engine=engine)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[],
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_zero_budget_channel_skipped(self):
        """Channel with exhausted budget gets no candidate."""
        engine = self._make_engine_with_budget("100x1x0", 0, "proven", 2000)
        r = _make_rebalancer(capex_engine=engine)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[],
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_no_engine_returns_empty(self):
        """Without engine, fallback returns empty list."""
        r = _make_rebalancer(capex_engine=None)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[("200x1x0", {}, 0.80)],
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_active_channels_excluded(self):
        """Channels with active rebalance jobs are skipped."""
        engine = self._make_engine_with_budget("100x1x0", 300, "proven", 2000)
        r = _make_rebalancer(capex_engine=engine)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[("200x1x0", {}, 0.80)],
            active_channels={"100x1x0"},  # Already being rebalanced
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_respects_available_slots(self):
        """Stops after available_slots candidates found."""
        mock_engine = MagicMock(spec=CapexBudgetEngine)

        def _budget_for(ch_id):
            return ChannelCapexBudget(
                channel_id=ch_id,
                budget_msat=500_000,
                tier="proven",
                tier_ppm=2000,
                priority_class="preservation",
            )

        mock_engine.get_channel_budget.side_effect = _budget_for
        r = _make_rebalancer(capex_engine=mock_engine)

        source_info = {
            "peer_id": "02" + "b" * 64,
            "capacity": 5_000_000,
            "fee_ppm": 100,
            "spendable": 3_000_000,
        }
        r._select_source_candidates = MagicMock(return_value=[
            ("900x1x0", source_info, 50.0, 10)
        ])

        depleted = [
            (f"{i}00x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10)
            for i in range(1, 6)  # 5 depleted channels
        ]
        source_channels = [("900x1x0", source_info, 0.80)]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=source_channels,
            active_channels=set(),
            available_slots=2,  # Only room for 2
            cfg=r.config,
        )

        assert len(candidates) == 2

    def test_bootstrap_channel_gets_candidate(self):
        """Bootstrap channel with small budget still gets candidate."""
        engine = self._make_engine_with_budget("100x1x0", 200, "bootstrap", 250)
        r = _make_rebalancer(capex_engine=engine)

        source_info = {
            "peer_id": "02" + "b" * 64,
            "capacity": 5_000_000,
            "fee_ppm": 100,
            "spendable": 3_000_000,
        }
        r._select_source_candidates = MagicMock(return_value=[
            ("200x1x0", source_info, 50.0, 10)
        ])

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]
        source_channels = [("200x1x0", source_info, 0.80)]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=source_channels,
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 1
        c = candidates[0]
        assert c.max_fee_ppm <= 250  # Bootstrap ceiling


class TestDefaultBudgetPreserved:
    """Basic config fields still exist."""

    def test_default_daily_budget_preserved(self):
        cfg = Config()
        assert cfg.daily_budget_sats >= 0
