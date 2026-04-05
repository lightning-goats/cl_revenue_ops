"""Tests for rebalancer rpc_cache migration — listfunds and getinfo."""

import pytest
from unittest.mock import MagicMock

from modules.rebalancer import EVRebalancer, JobManager


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_config():
    c = MagicMock()
    c.min_wallet_reserve = 500_000
    c.snapshot.return_value = c
    return c


@pytest.fixture
def mock_database():
    db = MagicMock()
    db.get_rebalance_budget_used_24h.return_value = 0
    return db


@pytest.fixture
def mock_rpc_cache():
    cache = MagicMock()
    cache.listfunds.return_value = {
        "outputs": [{"amount_msat": 1_000_000_000, "status": "confirmed"}],
        "channels": [
            {"short_channel_id": "800000x1x0", "our_amount_msat": 500_000_000,
             "state": "CHANNELD_NORMAL"},
        ]
    }
    cache.getinfo.return_value = {"blockheight": 900000}
    return cache


class TestCapitalControlsUsesCache:
    """_check_capital_controls uses rpc_cache.listfunds when available."""

    def test_uses_rpc_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = mock_rpc_cache

        result = ev._check_capital_controls(mock_config)
        mock_rpc_cache.listfunds.assert_called()
        mock_plugin.rpc.listfunds.assert_not_called()

    def test_falls_back_without_cache(self, mock_plugin, mock_config, mock_database):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = None

        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 1_000_000_000, "status": "confirmed"}],
            "channels": [
                {"short_channel_id": "800000x1x0", "our_amount_msat": 500_000_000,
                 "state": "CHANNELD_NORMAL"},
            ]
        }

        result = ev._check_capital_controls(mock_config)
        mock_plugin.rpc.listfunds.assert_called()


class TestChannelAgeDaysUsesCache:
    """_get_channel_age_days uses rpc_cache.getinfo when available."""

    def test_uses_rpc_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = mock_rpc_cache

        age = ev._get_channel_age_days("800000x1x0")
        mock_rpc_cache.getinfo.assert_called()
        mock_plugin.rpc.getinfo.assert_not_called()
        assert age > 0

    def test_falls_back_without_cache(self, mock_plugin, mock_config, mock_database):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.rpc_cache = None
        mock_plugin.rpc.getinfo.return_value = {"blockheight": 900000}

        age = ev._get_channel_age_days("800000x1x0")
        mock_plugin.rpc.getinfo.assert_called()


class TestJobManagerUsesCache:
    """JobManager listfunds calls use rpc_cache when available."""

    def test_get_channel_local_balance_uses_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        jm = JobManager(mock_plugin, mock_config, mock_database)
        jm.rpc_cache = mock_rpc_cache

        balance = jm._get_channel_local_balance("800000x1x0")
        mock_rpc_cache.listfunds.assert_called()
        mock_plugin.rpc.listfunds.assert_not_called()
        assert balance == 500_000

    def test_get_local_balances_map_uses_cache(self, mock_plugin, mock_config, mock_database, mock_rpc_cache):
        jm = JobManager(mock_plugin, mock_config, mock_database)
        jm.rpc_cache = mock_rpc_cache

        balances = jm._get_local_balances_map()
        mock_rpc_cache.listfunds.assert_called()
        mock_plugin.rpc.listfunds.assert_not_called()
        assert "800000x1x0" in balances

    def test_falls_back_without_cache(self, mock_plugin, mock_config, mock_database):
        jm = JobManager(mock_plugin, mock_config, mock_database)
        jm.rpc_cache = None
        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [],
            "channels": [
                {"short_channel_id": "800000x1x0", "our_amount_msat": 500_000_000,
                 "state": "CHANNELD_NORMAL"},
            ]
        }

        balance = jm._get_channel_local_balance("800000x1x0")
        mock_plugin.rpc.listfunds.assert_called()
        assert balance == 500_000
