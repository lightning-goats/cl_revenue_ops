"""Tests for fee intelligence improvements."""
import time
import pytest
from unittest.mock import MagicMock

from modules.fee_controller import FeeController, GaussianThompsonState, ChannelFeeState


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


class TestRebalanceCostFloor:
    def test_cost_ppm_from_rebalance(self, mock_plugin, mock_config, mock_database):
        mock_database.get_last_rebalance_cost.return_value = {
            "cost_sats": 500, "amount_sats": 1_000_000
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        cost = fc._get_channel_rebalance_cost_ppm("123x1x0")
        assert cost == 500  # 500 sats / 1M sats * 1M = 500 PPM

    def test_cost_ppm_zero_when_no_history(self, mock_plugin, mock_config, mock_database):
        mock_database.get_last_rebalance_cost.return_value = None
        fc = FeeController(mock_plugin, mock_config, mock_database)
        cost = fc._get_channel_rebalance_cost_ppm("123x1x0")
        assert cost == 0

    def test_cost_ppm_zero_on_error(self, mock_plugin, mock_config, mock_database):
        mock_database.get_last_rebalance_cost.side_effect = Exception("DB error")
        fc = FeeController(mock_plugin, mock_config, mock_database)
        cost = fc._get_channel_rebalance_cost_ppm("123x1x0")
        assert cost == 0

    def test_cost_ppm_handles_zero_amount(self, mock_plugin, mock_config, mock_database):
        mock_database.get_last_rebalance_cost.return_value = {
            "cost_sats": 500, "amount_sats": 0
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        cost = fc._get_channel_rebalance_cost_ppm("123x1x0")
        assert cost == 0

    def test_cost_ppm_handles_none_cost(self, mock_plugin, mock_config, mock_database):
        mock_database.get_last_rebalance_cost.return_value = {
            "cost_sats": None, "amount_sats": 1_000_000
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        cost = fc._get_channel_rebalance_cost_ppm("123x1x0")
        assert cost == 0

    def test_cost_ppm_no_database(self, mock_plugin, mock_config):
        fc = FeeController(mock_plugin, mock_config, None)
        cost = fc._get_channel_rebalance_cost_ppm("123x1x0")
        assert cost == 0


class TestFailedForwardObservation:
    def test_failed_forward_adjusts_posterior(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = GaussianThompsonState()
        state.posterior_mean = 500.0
        state.posterior_std = 50.0
        cfs = ChannelFeeState(thompson=state)
        fc._channel_fee_states["123x1x0"] = cfs

        original_mean = state.posterior_mean
        fc.record_failed_forward("123x1x0", 500)

        # Should pull mean down slightly (toward 80% of 500 = 400)
        assert state.posterior_mean < original_mean
        assert state.posterior_mean > 400  # But not all the way to 400

    def test_failed_forward_no_crash_missing_state(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        # No state for this channel — should not crash
        fc.record_failed_forward("999x1x0", 500)

    def test_failed_forward_no_crash_zero_fee(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.record_failed_forward("123x1x0", 0)  # Should not crash

    def test_failed_forward_no_crash_empty_channel_id(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.record_failed_forward("", 500)  # Should not crash

    def test_failed_forward_preserves_min_std(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = GaussianThompsonState()
        state.posterior_mean = 500.0
        state.posterior_std = float(GaussianThompsonState.MIN_STD)
        cfs = ChannelFeeState(thompson=state)
        fc._channel_fee_states["123x1x0"] = cfs

        fc.record_failed_forward("123x1x0", 500)

        # Std should never drop below MIN_STD
        assert state.posterior_std >= GaussianThompsonState.MIN_STD
