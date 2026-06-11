"""Tests for Step 4 competition_aware market_fee_mode.

Covers the mode enum validation and the neighbor percentile/default-fee
helpers. (_get_neighbor_fee_min was removed as zero-caller dead code; the
competition_aware preserve trigger uses _get_neighbor_fee_percentile.)
The full pipeline integration is exercised by the Polar lab run.
"""

import time
import pytest
from unittest.mock import MagicMock

from modules.fee_controller import FeeController
from modules.config import STRING_ENUM_VALID_VALUES


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
    c.fee_ppm_intra_fleet = 1
    c.neighbor_median_min_competitors = 2  # Lab-default; prod may set 3.
    c.snapshot = MagicMock(return_value=c)
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


class TestMarketFeeModeEnum:
    def test_competition_aware_is_valid_value(self):
        assert 'competition_aware' in STRING_ENUM_VALID_VALUES['market_fee_mode']

    def test_legacy_modes_still_valid(self):
        valid = STRING_ENUM_VALID_VALUES['market_fee_mode']
        assert 'undercut' in valid
        assert 'match' in valid
        assert 'premium' in valid


class TestClnDefaultFeeFilter:
    """Phase B.1 (2026-04-23): dormant CLN-default competitors must not
    drag down the neighbor median / min. They represent nodes that never
    touched fee config — not real competitors for pricing decisions."""

    def test_default_tuple_flagged_pre_24(self):
        ch = {"base_fee_millisatoshi": 1000, "fee_per_millionth": 10}
        assert FeeController._is_cln_default_fee(ch) is True

    def test_default_tuple_flagged_post_24(self):
        ch = {"fee_base_msat": 1000, "fee_per_millionth": 10}
        assert FeeController._is_cln_default_fee(ch) is True

    def test_custom_ppm_not_default(self):
        ch = {"fee_base_msat": 1000, "fee_per_millionth": 200}
        assert FeeController._is_cln_default_fee(ch) is False

    def test_custom_base_not_default(self):
        ch = {"fee_base_msat": 0, "fee_per_millionth": 10}
        assert FeeController._is_cln_default_fee(ch) is False

    def test_missing_base_keeps_channel_conservative(self):
        ch = {"fee_per_millionth": 10}
        assert FeeController._is_cln_default_fee(ch) is False


class TestNeighborFeePercentile:
    """Phase D.1 (2026-04-23): competition_aware now preserves DTS when
    we're below the p25 of competitor fees (instead of strict-cheapest).
    """

    def _make_channels(self, fees: list[int]):
        return [
            {"source": f"02c{i:02x}", "active": True,
             "fee_per_millionth": fee, "fee_base_msat": 0}
            for i, fee in enumerate(fees)
        ]

    def _setup_fc(self, mock_plugin, mock_config, mock_database, channels: list):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc._our_node_id = "02ourid"
        fc.data_service = MagicMock()
        fc.data_service.get_channels = MagicMock(return_value={"channels": channels})
        fc._neighbor_fee_cache = {}
        return fc

    def test_p25_of_sorted_fees(self, mock_plugin, mock_config, mock_database):
        # Fees [50, 100, 200, 400, 800]; p25 is the 2nd value (index 1) = 100.
        channels = self._make_channels([50, 100, 200, 400, 800])
        fc = self._setup_fc(mock_plugin, mock_config, mock_database, channels)
        assert fc._get_neighbor_fee_percentile("02peer", 0.25) == 100

    def test_p50_is_median(self, mock_plugin, mock_config, mock_database):
        channels = self._make_channels([50, 100, 200, 400, 800])
        fc = self._setup_fc(mock_plugin, mock_config, mock_database, channels)
        assert fc._get_neighbor_fee_percentile("02peer", 0.50) == 200

    def test_default_fee_competitors_filtered(self, mock_plugin, mock_config, mock_database):
        # Default-fee channel (10 ppm, base=1000) excluded; remaining fees [50, 100, 200, 400].
        channels = [
            {"source": "02a", "active": True, "fee_per_millionth": 10, "fee_base_msat": 1000},
            {"source": "02b", "active": True, "fee_per_millionth": 50, "fee_base_msat": 0},
            {"source": "02c", "active": True, "fee_per_millionth": 100, "fee_base_msat": 0},
            {"source": "02d", "active": True, "fee_per_millionth": 200, "fee_base_msat": 0},
            {"source": "02e", "active": True, "fee_per_millionth": 400, "fee_base_msat": 0},
        ]
        fc = self._setup_fc(mock_plugin, mock_config, mock_database, channels)
        # After filter, fees are [50, 100, 200, 400]. p25 is index 1 = 100.
        assert fc._get_neighbor_fee_percentile("02peer", 0.25) == 100

    def test_below_min_competitors_returns_none(self, mock_plugin, mock_config, mock_database):
        channels = self._make_channels([50])  # only 1 competitor
        fc = self._setup_fc(mock_plugin, mock_config, mock_database, channels)
        assert fc._get_neighbor_fee_percentile("02peer", 0.25) is None

    def test_cache_reuses_value(self, mock_plugin, mock_config, mock_database):
        channels = self._make_channels([50, 100, 200])
        fc = self._setup_fc(mock_plugin, mock_config, mock_database, channels)
        assert fc._get_neighbor_fee_percentile("02peer", 0.25) == 50
        fc.data_service.get_channels = MagicMock(return_value={"channels": []})
        assert fc._get_neighbor_fee_percentile("02peer", 0.25) == 50


class TestUndercutExplorationThreshold:
    """Phase B.3 (2026-04-23): variance-gated undercut.

    Constant UNDERCUT_EXPLORATION_STD_THRESHOLD gates whether undercut
    (and the undercut fallback inside competition_aware) actually clamps
    DTS down. High-variance (exploring) posteriors are preserved so DTS
    sees a meaningful fee range before observations lock in a low guess.

    Branch logic itself runs deep in adjust_fees and requires a full
    FeeController boot. Exhaustive coverage lives in the lab run; this
    sanity test pins the constant and documents intent.
    """

    def test_constant_defined(self):
        assert FeeController.UNDERCUT_EXPLORATION_STD_THRESHOLD == 100.0

    def test_constant_above_sparse_boundary(self):
        """Threshold matches the 'very uncertain' boundary at line 4123."""
        assert FeeController.UNDERCUT_EXPLORATION_STD_THRESHOLD >= 100.0
