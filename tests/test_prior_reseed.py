"""Prior-source selection + reseeded_at persistence.

The fleet_fee_median skew-era repair (_maybe_reseed_skewed_prior) was
removed with the fleet prior itself (Phase 5 de-hive): with no fleet
source, the in-cycle check (allow_rpc=False) could never resolve, so it
was a provable per-cycle no-op. What remains:
- _select_best_fee_prior's network-gossip chain (used by set_initial_fee);
- the thompson.reseeded_at marker, kept only for state-blob
  compatibility with already-persisted channel states.
"""

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


def _make_fc(mock_plugin, mock_config, mock_database, gossip_channels=None):
    fc = FeeController(mock_plugin, mock_config, mock_database)
    fc.data_service = MagicMock()
    fc.data_service.get_channels.return_value = {
        "channels": gossip_channels or []
    }
    return fc


class TestSelectBestFeePrior:
    def test_gossip_prior_selected(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = _make_fc(
            mock_plugin, mock_config, mock_database,
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

    def test_no_source_returns_none(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = _make_fc(mock_plugin, mock_config, mock_database)
        assert fc._select_best_fee_prior(PEER, CHANNEL) is None

    def test_allow_rpc_false_skips_gossip(
        self, mock_plugin, mock_config, mock_database
    ):
        # The gossip fallback is an uncached listchannels RPC; in-cycle
        # callers must not take it.
        fc = _make_fc(
            mock_plugin, mock_config, mock_database,
            gossip_channels=[
                {"fee_per_millionth": 300, "satoshis": 5_000_000},
            ],
        )
        assert fc._select_best_fee_prior(PEER, CHANNEL, allow_rpc=False) is None
        fc.data_service.get_channels.assert_not_called()


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
