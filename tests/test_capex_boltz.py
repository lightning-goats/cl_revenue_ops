"""Tests for capex-engine integration with Boltz manager."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capex_budget import CapexBudgetEngine, ChannelCapexBudget


class TestBoltzEngineInjection:
    """Capex engine can be injected into the Boltz manager."""

    def _make_boltz(self, capex_engine=None):
        from modules.boltz_manager import BoltzCliManager, BoltzCliConfig
        mock_plugin = MagicMock()
        mock_rpc = MagicMock()
        cfg = BoltzCliConfig(
            enabled=True,
            cli_path="/usr/bin/boltzcli",
            datadir="/tmp/test_boltz",
            daily_budget_sats=3000,
            enforce_budget=True,
        )
        mgr = BoltzCliManager(mock_plugin, mock_rpc, cfg)
        if capex_engine:
            mgr.set_capex_engine(capex_engine)
        return mgr

    def test_set_capex_engine(self):
        mgr = self._make_boltz()
        mock_engine = MagicMock()
        mgr.set_capex_engine(mock_engine)
        assert mgr._capex_engine is mock_engine

    def test_default_no_engine(self):
        mgr = self._make_boltz()
        assert mgr._capex_engine is None


class TestTacticalBudgetGate:
    """Pure treasury swaps gated by tactical budget."""

    def _make_boltz_with_engine(self, tactical_budget=0):
        from modules.boltz_manager import BoltzCliManager, BoltzCliConfig
        mock_plugin = MagicMock()
        mock_rpc = MagicMock()
        cfg = BoltzCliConfig(
            enabled=True,
            cli_path="/usr/bin/boltzcli",
            datadir="/tmp/test_boltz",
            daily_budget_sats=50000,
            enforce_budget=True,
        )
        mgr = BoltzCliManager(mock_plugin, mock_rpc, cfg)

        mock_engine = MagicMock(spec=CapexBudgetEngine)
        mock_engine.get_tactical_budget.return_value = tactical_budget
        mgr.set_capex_engine(mock_engine)
        return mgr

    def test_tactical_budget_zero_blocks_treasury_swap(self):
        """When tactical budget is 0, pure treasury swaps are blocked."""
        mgr = self._make_boltz_with_engine(tactical_budget=0)

        result = mgr.check_tactical_budget(estimated_fee_sats=500, channel_id=None)
        assert result["allowed"] is False
        assert "tactical" in result["reason"].lower()

    def test_tactical_budget_sufficient_allows_swap(self):
        """When tactical budget >= fee, treasury swap is allowed."""
        mgr = self._make_boltz_with_engine(tactical_budget=1000)

        result = mgr.check_tactical_budget(estimated_fee_sats=500, channel_id=None)
        assert result["allowed"] is True

    def test_channel_targeted_bypasses_tactical_gate(self):
        """Channel-targeted swaps are not gated by tactical budget."""
        mgr = self._make_boltz_with_engine(tactical_budget=0)

        result = mgr.check_tactical_budget(estimated_fee_sats=500, channel_id="100x1x0")
        assert result["allowed"] is True

    def test_no_engine_bypasses_gate(self):
        """Without engine, tactical gate is not applied."""
        from modules.boltz_manager import BoltzCliManager, BoltzCliConfig
        cfg = BoltzCliConfig(
            enabled=True,
            cli_path="/usr/bin/boltzcli",
            datadir="/tmp/test_boltz",
            daily_budget_sats=3000,
            enforce_budget=True,
        )
        mgr = BoltzCliManager(MagicMock(), MagicMock(), cfg)

        result = mgr.check_tactical_budget(estimated_fee_sats=500, channel_id=None)
        assert result["allowed"] is True


class TestCostAttribution:
    """Swap costs attributed via engine.attribute_boltz_cost()."""

    def _make_boltz_with_engine(self):
        from modules.boltz_manager import BoltzCliManager, BoltzCliConfig
        mock_plugin = MagicMock()
        mock_rpc = MagicMock()
        cfg = BoltzCliConfig(
            enabled=True,
            cli_path="/usr/bin/boltzcli",
            datadir="/tmp/test_boltz",
            daily_budget_sats=50000,
            enforce_budget=True,
        )
        mgr = BoltzCliManager(mock_plugin, mock_rpc, cfg)

        mock_engine = MagicMock(spec=CapexBudgetEngine)
        mock_engine.attribute_boltz_cost.return_value = {"channel": 100, "tactical": 100}
        mock_engine.get_tactical_budget.return_value = 10000
        mgr.set_capex_engine(mock_engine)
        return mgr

    def test_pure_treasury_all_tactical(self):
        """Pure treasury swap attributed 100% to tactical."""
        mgr = self._make_boltz_with_engine()
        mgr._capex_engine.attribute_boltz_cost.return_value = {"channel": 0, "tactical": 200}

        result = mgr.compute_cost_attribution(cost_sats=200, channel_id=None)

        assert result["channel"] == 0
        assert result["tactical"] == 200
        mgr._capex_engine.attribute_boltz_cost.assert_called_with(200, channel_id=None)

    def test_channel_targeted_50_50(self):
        """Channel-targeted swap gets 50/50 split."""
        mgr = self._make_boltz_with_engine()
        mgr._capex_engine.attribute_boltz_cost.return_value = {"channel": 100, "tactical": 100}

        result = mgr.compute_cost_attribution(cost_sats=200, channel_id="100x1x0")

        assert result["channel"] == 100
        assert result["tactical"] == 100
        mgr._capex_engine.attribute_boltz_cost.assert_called_with(200, channel_id="100x1x0")

    def test_no_engine_returns_all_tactical(self):
        """Without engine, all cost attributed to tactical (safe default)."""
        from modules.boltz_manager import BoltzCliManager, BoltzCliConfig
        cfg = BoltzCliConfig(
            enabled=True,
            cli_path="/usr/bin/boltzcli",
            datadir="/tmp/test_boltz",
            daily_budget_sats=3000,
            enforce_budget=True,
        )
        mgr = BoltzCliManager(MagicMock(), MagicMock(), cfg)

        result = mgr.compute_cost_attribution(cost_sats=200, channel_id=None)

        assert result["channel"] == 0
        assert result["tactical"] == 200
