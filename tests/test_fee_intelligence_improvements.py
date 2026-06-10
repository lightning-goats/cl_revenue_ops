"""Tests for fee intelligence improvements."""
import time
import pytest
from unittest.mock import MagicMock

from modules.fee_controller import FeeController, GaussianThompsonState, ChannelFeeState
from modules.hive_hints import HiveHintAdapter


def _make_data_service(mock_plugin):
    """Build a data_service MagicMock that delegates to mock_plugin.rpc."""
    ds = MagicMock()
    ds.get_peer_channels.side_effect = lambda peer_id=None, **kw: (
        mock_plugin.rpc.listpeerchannels(peer_id) if peer_id is not None
        else mock_plugin.rpc.listpeerchannels()
    )
    ds.get_channels.side_effect = lambda **kw: mock_plugin.rpc.listchannels(**kw)
    ds.get_node_id.side_effect = lambda: mock_plugin.rpc.getinfo().get("id", "")
    ds.set_channel.side_effect = lambda **kw: mock_plugin.rpc.setchannel(**kw)
    ds.get_feerates.side_effect = lambda **kw: mock_plugin.rpc.feerates(**kw)
    ds.get_askrene_layers.side_effect = lambda: mock_plugin.rpc.call("askrene-listlayers", {})
    return ds


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
    # Keep the old threshold so legacy tests that count "few channels"
    # still classify 1-2 competitors as insufficient.
    c.neighbor_median_min_competitors = 3
    c.snapshot = MagicMock(return_value=c)
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
        fc.data_service = _make_data_service(mock_plugin)
        result = fc._get_network_fee_prior("02peer", "123x1x0")
        assert result is not None
        assert result["mean"] == 300  # median
        assert result["std"] >= 50

    def test_prior_none_when_no_channels(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.listchannels.return_value = {"channels": []}
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        result = fc._get_network_fee_prior("02peer", "123x1x0")
        assert result is None

    def test_prior_none_on_rpc_failure(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.listchannels.side_effect = Exception("RPC error")
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
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
        fc.data_service = _make_data_service(mock_plugin)
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
    """Audit DTS-4: the failed-forward nudge is only valid for failures
    that implicate OUR fee on the OUT channel (WIRE_FEE_INSUFFICIENT
    family). Liquidity/downstream failures — and failures with no usable
    failcode at all — must produce NO nudge: a misdirected systematic
    signal is worse than none. Callers are responsible for passing the
    OUTGOING channel id (BOLT 7: our advertised fee applies to HTLCs we
    forward OUT through a channel)."""

    WIRE_FEE_INSUFFICIENT = 0x1000 | 12  # 4108
    WIRE_TEMPORARY_CHANNEL_FAILURE = 0x1000 | 7  # 4103 (liquidity)

    def _seed_channel(self, fc, channel_id="123x1x0"):
        state = GaussianThompsonState()
        state.posterior_mean = 500.0
        state.posterior_std = 50.0
        fc._channel_fee_states[channel_id] = ChannelFeeState(thompson=state)
        return state

    def test_fee_failcode_adjusts_posterior(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = self._seed_channel(fc)

        original_mean = state.posterior_mean
        fc.record_failed_forward(
            "123x1x0", 500,
            failcode=self.WIRE_FEE_INSUFFICIENT,
            failreason="WIRE_FEE_INSUFFICIENT",
        )

        # Should pull mean down slightly (toward 80% of 500 = 400)
        assert state.posterior_mean < original_mean
        assert state.posterior_mean > 400  # But not all the way to 400

    def test_fee_failreason_string_alone_adjusts_posterior(
        self, mock_plugin, mock_config, mock_database
    ):
        """Some payloads carry only the symbolic failreason."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = self._seed_channel(fc)

        original_mean = state.posterior_mean
        fc.record_failed_forward("123x1x0", 500, failreason="WIRE_FEE_INSUFFICIENT")

        assert state.posterior_mean < original_mean

    def test_liquidity_failcode_produces_no_nudge(
        self, mock_plugin, mock_config, mock_database
    ):
        """A liquidity failure says nothing about our fee — the sender
        already chose our edge."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = self._seed_channel(fc)

        original_mean = state.posterior_mean
        fc.record_failed_forward(
            "123x1x0", 500,
            failcode=self.WIRE_TEMPORARY_CHANNEL_FAILURE,
            failreason="WIRE_TEMPORARY_CHANNEL_FAILURE",
        )

        assert state.posterior_mean == original_mean
        assert state.posterior_bias == []

    def test_missing_failcode_produces_no_nudge(
        self, mock_plugin, mock_config, mock_database
    ):
        """No usable failure reason at all (CLN's plain 'failed' status):
        drop the nudge entirely."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = self._seed_channel(fc)

        original_mean = state.posterior_mean
        fc.record_failed_forward("123x1x0", 500)

        assert state.posterior_mean == original_mean
        assert state.posterior_bias == []

    def test_failed_forward_no_crash_missing_state(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        # No state for this channel — should not crash
        fc.record_failed_forward("999x1x0", 500, failcode=self.WIRE_FEE_INSUFFICIENT)

    def test_failed_forward_no_crash_zero_fee(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        # Should not crash
        fc.record_failed_forward("123x1x0", 0, failcode=self.WIRE_FEE_INSUFFICIENT)

    def test_failed_forward_no_crash_empty_channel_id(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        # Should not crash
        fc.record_failed_forward("", 500, failcode=self.WIRE_FEE_INSUFFICIENT)

    def test_failed_forward_no_crash_garbage_failcode(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = self._seed_channel(fc)
        original_mean = state.posterior_mean
        fc.record_failed_forward("123x1x0", 500, failcode="not-a-number")
        assert state.posterior_mean == original_mean

    def test_failed_forward_preserves_min_std(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        state = GaussianThompsonState()
        state.posterior_mean = 500.0
        state.posterior_std = float(GaussianThompsonState.MIN_STD)
        cfs = ChannelFeeState(thompson=state)
        fc._channel_fee_states["123x1x0"] = cfs

        fc.record_failed_forward(
            "123x1x0", 500, failcode=self.WIRE_FEE_INSUFFICIENT
        )

        # Std should never drop below MIN_STD
        assert state.posterior_std >= GaussianThompsonState.MIN_STD


class TestConfidenceScaledBlend:
    """Phase A.2 semantics: blend ratio is a function of posterior_std,
    not of the sparse_data_conservative flag. The flag remained in the
    signature for call-site compatibility but no longer gates the
    confidence boost."""

    def test_high_confidence_increases_blend(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        high_var_ratio = fc._get_target_blend_ratio(False, True, posterior_std=250.0)
        confident_ratio = fc._get_target_blend_ratio(False, False, posterior_std=20.0)
        assert confident_ratio > high_var_ratio

    def test_blend_capped_at_sixty_percent(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        ratio = fc._get_target_blend_ratio(False, False, posterior_std=1.0)
        assert ratio <= 0.60

    def test_tight_posterior_under_sparse_flag_gets_boost(self, mock_plugin, mock_config, mock_database):
        """Prior design stuck sparse channels at 0.20 even with tight posterior.
        Phase A.2: posterior_std drives the ratio; sparse flag no longer caps."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        ratio = fc._get_target_blend_ratio(False, True, posterior_std=10.0)
        assert ratio == 0.60  # Tight posterior -> fast convergence regardless of flag


class TestNeighborFeeAwareness:
    def test_neighbor_median_computed(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node"}
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "02node1", "fee_per_millionth": 100, "active": True},
                {"source": "02node2", "fee_per_millionth": 150, "active": True},
                {"source": "02node3", "fee_per_millionth": 200, "active": True},
                {"source": "02node4", "fee_per_millionth": 300, "active": True},
                {"source": "02node5", "fee_per_millionth": 500, "active": True},
            ]
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        median = fc._get_neighbor_fee_median("02peer123")
        assert median == 200

    def test_neighbor_none_with_few_channels(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node"}
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "02node1", "fee_per_millionth": 100, "active": True},
            ]
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        assert fc._get_neighbor_fee_median("02peer") is None

    def test_neighbor_excludes_our_node(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node"}
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "02our_node", "fee_per_millionth": 999, "active": True},
                {"source": "02node1", "fee_per_millionth": 100, "active": True},
                {"source": "02node2", "fee_per_millionth": 200, "active": True},
                {"source": "02node3", "fee_per_millionth": 300, "active": True},
            ]
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        median = fc._get_neighbor_fee_median("02peer")
        assert median == 200  # Our 999 excluded

    def test_neighbor_cached(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node"}
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": f"02node{i}", "fee_per_millionth": 100 + i * 50, "active": True}
            for i in range(5)
        ]}
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        fc._get_neighbor_fee_median("02peer")
        fc._get_neighbor_fee_median("02peer")
        # listchannels called only once due to cache
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_neighbor_excludes_inactive_channels(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node"}
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "02node1", "fee_per_millionth": 100, "active": True},
                {"source": "02node2", "fee_per_millionth": 200, "active": False},
                {"source": "02node3", "fee_per_millionth": 300, "active": True},
                {"source": "02node4", "fee_per_millionth": 400, "active": True},
            ]
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        median = fc._get_neighbor_fee_median("02peer")
        assert median == 300  # Inactive node2 excluded; [100, 300, 400] -> 300

    def test_neighbor_excludes_extreme_fees(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node"}
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "02node1", "fee_per_millionth": 0, "active": True},
                {"source": "02node2", "fee_per_millionth": 100, "active": True},
                {"source": "02node3", "fee_per_millionth": 200, "active": True},
                {"source": "02node4", "fee_per_millionth": 300, "active": True},
                {"source": "02node5", "fee_per_millionth": 50000, "active": True},
            ]
        }
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        median = fc._get_neighbor_fee_median("02peer")
        assert median == 200  # 0 and 50000 filtered out; [100, 200, 300] -> 200

    def test_neighbor_none_on_rpc_error(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.side_effect = Exception("RPC error")
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        assert fc._get_neighbor_fee_median("02peer") is None

    def test_neighbor_cache_eviction(self, mock_plugin, mock_config, mock_database):
        """Stale entries older than 1 hour are evicted when cache exceeds 500."""
        mock_plugin.rpc.getinfo.return_value = {"id": "02our_node"}
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": f"02node{i}", "fee_per_millionth": 100 + i * 50, "active": True}
            for i in range(5)
        ]}
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        # Fill cache with 501 stale entries
        old_ts = time.time() - 7200  # 2 hours old
        for i in range(501):
            fc._neighbor_fee_cache[f"neighbor_fee_stale_{i}"] = {"value": 100, "ts": old_ts}
        # Trigger eviction by calling the method
        fc._get_neighbor_fee_median("02peer_new")
        # Stale entries should have been evicted
        stale_remaining = sum(1 for k in fc._neighbor_fee_cache if k.startswith("neighbor_fee_stale_"))
        assert stale_remaining == 0


class TestFleetFeePriors:
    def test_fleet_prior_used_for_initial_fee(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fleet_fee_prior.return_value = 350
        fc.hive_hints = adapter
        # Fleet prior should be available
        fee = adapter.get_fleet_fee_prior("02peer")
        assert fee == 350

    def test_fleet_prior_none_when_no_data(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {"02peer": {"member": True}}
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        assert adapter.get_fleet_fee_prior("02peer") is None

    def test_fleet_prior_from_hint(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {"02peer": {"member": True, "fleet_fee_median": 250}}
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        assert adapter.get_fleet_fee_prior("02peer") == 250
