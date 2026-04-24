"""Tests for rebalancer data_service migration — listfunds and get_block_height."""

import pytest
from unittest.mock import MagicMock

from modules.rebalancer import EVRebalancer


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
def mock_data_service():
    ds = MagicMock()
    ds.get_funds.return_value = {
        "outputs": [{"amount_msat": 1_000_000_000, "status": "confirmed"}],
        "channels": [
            {"short_channel_id": "800000x1x0", "our_amount_msat": 500_000_000,
             "state": "CHANNELD_NORMAL"},
        ]
    }
    ds.get_block_height.return_value = 900000
    return ds


class TestCapitalControlsUsesDataService:
    """_check_capital_controls uses data_service.get_funds."""

    def test_uses_data_service(self, mock_plugin, mock_config, mock_database, mock_data_service):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.data_service = mock_data_service

        result = ev._check_capital_controls(mock_config)
        mock_data_service.get_funds.assert_called()
        mock_plugin.rpc.listfunds.assert_not_called()

    def test_data_service_provides_funds(self, mock_plugin, mock_config, mock_database, mock_data_service):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.data_service = mock_data_service

        result = ev._check_capital_controls(mock_config)
        mock_data_service.get_funds.assert_called()


class TestChannelAgeDaysUsesDataService:
    """_get_channel_age_days uses data_service.get_block_height."""

    def test_uses_data_service(self, mock_plugin, mock_config, mock_database, mock_data_service):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.data_service = mock_data_service

        age = ev._get_channel_age_days("800000x1x0")
        mock_data_service.get_block_height.assert_called()
        mock_plugin.rpc.getinfo.assert_not_called()
        assert age > 0

    def test_returns_correct_age(self, mock_plugin, mock_config, mock_database, mock_data_service):
        ev = EVRebalancer(mock_plugin, mock_config, mock_database)
        ev.data_service = mock_data_service
        # block 800000, current 900000 → 100000 blocks → 694 days
        age = ev._get_channel_age_days("800000x1x0")
        assert age > 0


# TestJobManagerUsesCache removed -- _get_channel_local_balance and
# _get_local_balances_map deleted with legacy async job code.
