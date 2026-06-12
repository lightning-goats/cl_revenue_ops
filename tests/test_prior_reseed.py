"""One-time Thompson prior re-seed for the fleet_fee_median-skew era.

Channels opened while get_fleet_fee_prior returned a skewed fleet fee
median had their PERSISTENT Thompson priors seeded from that skew
(set_initial_fee, audit F5). Active channels self-correct through
observations, but quiet channels (< MIN_OBSERVATIONS) keep sampling
around the bad prior forever.

These tests pin the bounded repair contract
(_maybe_reseed_skewed_prior):
- a quiet channel whose prior_mean_fee diverges > 15% from the CURRENT
  best prior source (fleet median > gossip > default) is re-seeded once,
  with one durable nudge recorded;
- an active channel (>= MIN_OBSERVATIONS) is never touched;
- the check resolves at most once per channel (reseeded_at marker), so a
  second cycle is a no-op even if the prior sources move;
- reseeded_at round-trips through persistence.
"""

import time

import pytest
from unittest.mock import MagicMock

from modules.fee_controller import (
    ChannelFeeState,
    FeeController,
    GaussianThompsonState,
)

CHANNEL = "123x456x0"
PEER = "02" + "a" * 64


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
    c.snapshot = MagicMock(return_value=c)
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


def _make_fc(mock_plugin, mock_config, mock_database, fleet_fee=500,
             gossip_channels=None):
    fc = FeeController(mock_plugin, mock_config, mock_database)
    fc.data_service = MagicMock()
    fc.data_service.get_channels.return_value = {
        "channels": gossip_channels or []
    }
    hints = MagicMock()
    hints.get_fleet_fee_prior.return_value = fleet_fee
    fc.hive_hints = hints
    # Isolate persistence; the marker round-trip is tested separately.
    fc._save_channel_fee_state = MagicMock()
    return fc


def _quiet_state(prior_mean=1500, n_obs=0):
    state = ChannelFeeState()
    state.thompson.prior_mean_fee = prior_mean
    state.thompson.prior_std_fee = 300
    now = int(time.time())
    state.thompson.observations = [
        (prior_mean, 1.0, 1.0, now, "normal") for _ in range(n_obs)
    ]
    return state


class TestMaybeReseedSkewedPrior:
    def test_quiet_skewed_channel_reseeds_once(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = _make_fc(mock_plugin, mock_config, mock_database, fleet_fee=500)
        state = _quiet_state(prior_mean=1500, n_obs=2)

        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is True

        ts = state.thompson
        assert ts.prior_mean_fee == 500
        assert ts.prior_std_fee == FeeController.FLEET_PRIOR_STD_PPM
        assert ts.reseeded_at > 0
        # Exactly one durable nudge toward the new prior.
        assert len(ts.posterior_bias) == 1
        assert ts.posterior_bias[0][0] == 500.0
        fc._save_channel_fee_state.assert_called_once_with(CHANNEL, state)

    def test_active_channel_untouched(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = _make_fc(mock_plugin, mock_config, mock_database, fleet_fee=500)
        state = _quiet_state(
            prior_mean=1500, n_obs=GaussianThompsonState.MIN_OBSERVATIONS
        )

        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is False

        ts = state.thompson
        assert ts.prior_mean_fee == 1500
        assert ts.prior_std_fee == 300
        assert ts.reseeded_at == 0
        assert ts.posterior_bias == []
        fc._save_channel_fee_state.assert_not_called()

    def test_second_cycle_is_noop(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = _make_fc(mock_plugin, mock_config, mock_database, fleet_fee=500)
        state = _quiet_state(prior_mean=1500, n_obs=1)

        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is True
        # Fleet prior moves; the channel must NOT chase it.
        fc.hive_hints.get_fleet_fee_prior.return_value = 900

        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is False

        ts = state.thompson
        assert ts.prior_mean_fee == 500
        assert len(ts.posterior_bias) == 1
        assert fc._save_channel_fee_state.call_count == 1

    def test_within_tolerance_resolves_without_reseed(
        self, mock_plugin, mock_config, mock_database
    ):
        # 540 vs 500 = 8% divergence (< 15%): no correction needed, but the
        # check is marked resolved so it never becomes per-cycle work.
        fc = _make_fc(mock_plugin, mock_config, mock_database, fleet_fee=500)
        state = _quiet_state(prior_mean=540, n_obs=1)

        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is False

        ts = state.thompson
        assert ts.prior_mean_fee == 540
        assert ts.posterior_bias == []
        assert ts.reseeded_at > 0

        # Even if the fleet prior later moves far away, resolved stays resolved.
        fc.hive_hints.get_fleet_fee_prior.return_value = 2000
        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is False
        assert ts.prior_mean_fee == 540

    def test_no_prior_source_retries_later(
        self, mock_plugin, mock_config, mock_database
    ):
        # Neither fleet nor gossip evidence yet: do NOT burn the one-shot
        # marker — the check may retry once a source becomes available.
        fc = _make_fc(
            mock_plugin, mock_config, mock_database,
            fleet_fee=None, gossip_channels=[],
        )
        state = _quiet_state(prior_mean=1500, n_obs=1)

        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is False
        assert state.thompson.reseeded_at == 0
        assert state.thompson.prior_mean_fee == 1500

        # Fleet prior becomes available later -> re-seed proceeds.
        fc.hive_hints.get_fleet_fee_prior.return_value = 500
        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is True
        assert state.thompson.prior_mean_fee == 500

    def test_no_gossip_rpc_inside_cycle_check(
        self, mock_plugin, mock_config, mock_database
    ):
        # The per-cycle check must be RPC-free: the gossip fallback is an
        # uncached listchannels call, so without a fleet prior the check
        # must NOT fall through to it (the skew era seeded priors FROM the
        # fleet median, so the affected population has fleet priors).
        fc = _make_fc(
            mock_plugin, mock_config, mock_database,
            fleet_fee=None,
            gossip_channels=[
                {"fee_per_millionth": 300, "satoshis": 5_000_000},
            ],
        )
        state = _quiet_state(prior_mean=1500, n_obs=1)

        assert fc._maybe_reseed_skewed_prior(CHANNEL, PEER, state) is False
        assert state.thompson.prior_mean_fee == 1500
        assert state.thompson.reseeded_at == 0
        fc.data_service.get_channels.assert_not_called()

    def test_set_initial_fee_selection_keeps_gossip_fallback(
        self, mock_plugin, mock_config, mock_database
    ):
        # Out-of-cycle callers (set_initial_fee) keep the full chain:
        # fleet > gossip > default.
        fc = _make_fc(
            mock_plugin, mock_config, mock_database,
            fleet_fee=None,
            gossip_channels=[
                {"fee_per_millionth": 300, "satoshis": 5_000_000},
                {"fee_per_millionth": 300, "satoshis": 5_000_000},
                {"fee_per_millionth": 320, "satoshis": 1_000_000},
            ],
        )

        prior = fc._select_best_fee_prior(PEER, CHANNEL)

        assert prior is not None
        assert prior["source"] == "network"
        assert prior["mean"] == 300

    def test_selection_prefers_fleet_prior(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = _make_fc(
            mock_plugin, mock_config, mock_database,
            fleet_fee=500,
            gossip_channels=[
                {"fee_per_millionth": 300, "satoshis": 5_000_000},
            ],
        )

        prior = fc._select_best_fee_prior(PEER, CHANNEL)

        assert prior == {
            "mean": 500,
            "std": FeeController.FLEET_PRIOR_STD_PPM,
            "source": "fleet",
        }


class TestReseededAtPersistence:
    def test_roundtrip(self):
        ts = GaussianThompsonState()
        ts.reseeded_at = 12345

        restored = GaussianThompsonState.from_dict(ts.to_dict())

        assert restored.reseeded_at == 12345

    def test_legacy_dict_defaults_to_zero(self):
        d = GaussianThompsonState().to_dict()
        d.pop("reseeded_at", None)

        assert GaussianThompsonState.from_dict(d).reseeded_at == 0

    def test_channel_fee_state_v2_roundtrip(self):
        state = ChannelFeeState()
        state.thompson.reseeded_at = 67890

        restored = ChannelFeeState.from_v2_dict(state.to_v2_dict())

        assert restored.thompson.reseeded_at == 67890


class TestCycleHook:
    """The re-seed check runs inside the regular DTS fee cycle path."""

    def _make_cycle_fc(self, mock_plugin, mock_config, mock_database):
        fc = _make_fc(mock_plugin, mock_config, mock_database, fleet_fee=500)
        # Restore real persistence (writes go to the MagicMock database).
        del fc._save_channel_fee_state

        cfg = mock_config.snapshot()
        cfg.fee_interval = 1800
        cfg.flow_interval = 3600
        cfg.htlc_congestion_threshold = 0.8
        cfg.inbound_fee_estimate_ppm = 200
        cfg.routing_intelligence_enabled = False

        mock_database.get_channel_probe.return_value = None
        mock_database.get_last_rebalance_cost.return_value = None
        mock_database.get_volume_since.return_value = 1000
        mock_database.get_forward_count_since.return_value = 5
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.5,
            "kalman_velocity": 0.0,
        }
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 10.0,
            "last_fee_ppm": 100,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": int(time.time()) - 7200,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 5,
            "last_volume_sats": 1000,
            "v2_state_json": None,
        }
        mock_database.get_last_forward_time.return_value = int(time.time()) - 3600
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_peer_latency_stats.return_value = {
            'avg': 0.0, 'std': 0.0, 'count': 0
        }
        fc.data_service.get_feerates.return_value = {
            "perkw": {"opening": 1000}
        }
        return fc, cfg

    def test_adjust_channel_fee_reseeds_quiet_skewed_channel(
        self, mock_plugin, mock_config, mock_database
    ):
        fc, cfg = self._make_cycle_fc(mock_plugin, mock_config, mock_database)
        # Pre-populate the state cache with a quiet, skewed-era channel.
        fc._channel_fee_states[CHANNEL] = _quiet_state(prior_mean=1500, n_obs=1)

        channel_info = {
            "fee_proportional_millionths": 100,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        state = {"state": "balanced", "forward_count": 5}

        fc._adjust_channel_fee(CHANNEL, PEER, state, channel_info, cfg=cfg)

        ts = fc._channel_fee_states[CHANNEL].thompson
        assert ts.reseeded_at > 0
        assert ts.prior_mean_fee == 500
