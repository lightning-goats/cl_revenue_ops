"""Tests for fee intelligence improvements."""
import time
import pytest
from unittest.mock import MagicMock

from modules.fee_controller import FeeController, GaussianThompsonState


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p

@pytest.fixture
def mock_config():
    c = MagicMock()
    c.min_fee_ppm = 10
    c.max_fee_ppm = 5000
    c.thompson_prior_std_fee = 100
    return c

@pytest.fixture
def mock_database():
    return MagicMock()


class TestNetworkInformedPriors:
    def test_prior_from_peer_fees(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"fee_per_millionth": 100},
                {"fee_per_millionth": 200},
                {"fee_per_millionth": 300},
                {"fee_per_millionth": 500},
                {"fee_per_millionth": 800},
            ]
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        result = fc._get_network_fee_prior("02peer", "123x1x0")
        assert result is not None
        assert result["mean"] == 300  # median
        assert result["std"] >= 50

    def test_prior_none_when_no_channels(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.listchannels.return_value = {"channels": []}
        fc = FeeController(mock_plugin, mock_config, mock_database)
        result = fc._get_network_fee_prior("02peer", "123x1x0")
        assert result is None

    def test_prior_none_on_rpc_failure(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.listchannels.side_effect = Exception("RPC error")
        fc = FeeController(mock_plugin, mock_config, mock_database)
        result = fc._get_network_fee_prior("02peer", "123x1x0")
        assert result is None

    def test_prior_filters_extreme_fees(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"fee_per_millionth": 0},      # Too low
                {"fee_per_millionth": 200},
                {"fee_per_millionth": 50000},   # Too high
            ]
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        result = fc._get_network_fee_prior("02peer", "123x1x0")
        assert result is not None
        assert result["mean"] == 200  # Only sane value
