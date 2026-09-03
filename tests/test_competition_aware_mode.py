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


from modules.fee_authority import FeeAuthorityGate

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

    def test_yield_aware_is_valid_value(self):
        assert 'yield_aware' in STRING_ENUM_VALID_VALUES['market_fee_mode']


class TestYieldAwareDemandTarget:
    def test_yield_market_multiplier_is_explicitly_bounded(self):
        assert FeeController.YIELD_MARKET_MIN_MULTIPLIER == 0.88
        assert FeeController.YIELD_MARKET_MAX_MULTIPLIER == 1.0
        assert FeeController.YIELD_DEMAND_MAX_PREMIUM == 0.02

    def _target(self, **overrides):
        values = {
            "current_fee_ppm": 200,
            "neighbor_median_ppm": 300,
            "outbound_ratio": 0.5,
            "forwards_since_update": 2,
            "market_power_weight": 0.10,
            "max_fee_ppm": 50_000,
        }
        values.update(overrides)
        return FeeController._yield_aware_demand_target(**values)

    def test_paid_demand_targets_above_current_but_below_market(self):
        assert 200 < self._target() < 300

    def test_scarce_inventory_accelerates_yield_target(self):
        assert self._target(outbound_ratio=0.1) > self._target(outbound_ratio=0.5)

    def test_more_paid_demand_increases_but_ceiling_bounds_target(self):
        low = self._target(forwards_since_update=1)
        high = self._target(forwards_since_update=100)
        assert high > low
        assert self._target(
            current_fee_ppm=40_000,
            neighbor_median_ppm=80_000,
            forwards_since_update=100,
        ) == 50_000

    @pytest.mark.parametrize(
        "overrides",
        [
            {"forwards_since_update": "malformed"},
            {"outbound_ratio": float("nan")},
            {"outbound_ratio": True},
            {"max_fee_ppm": 0},
        ],
    )
    def test_absent_malformed_or_saturated_evidence_is_neutral(self, overrides):
        assert self._target(**overrides) is None

    def test_missing_neighbor_median_still_uses_paid_local_demand(self):
        assert self._target(neighbor_median_ppm=None) > 200

    def test_market_quote_initializes_before_paid_demand(self):
        assert 200 < self._target(forwards_since_update=0) < 300

    def test_absent_market_and_demand_is_neutral(self):
        assert self._target(
            neighbor_median_ppm=None, forwards_since_update=0
        ) is None

    def test_saturated_inventory_undercuts_more_aggressively(self):
        assert self._target(outbound_ratio=0.90) < self._target(outbound_ratio=0.5)


class TestYieldAwareBalanceWake:
    def _controller(self, mock_plugin, mock_config, mock_database):
        mock_config.market_fee_mode = "yield_aware"
        return FeeController(
            mock_plugin,
            mock_config,
            mock_database,
            fee_authority_gate=FeeAuthorityGate(enabled=True),
        )

    def test_material_volume_wakes_with_per_channel_cooldown(
        self, mock_plugin, mock_config, mock_database, monkeypatch
    ):
        now = [100]
        monkeypatch.setattr(
            "modules.fee_controller.decision_now", lambda _label: now[0]
        )
        fc = self._controller(mock_plugin, mock_config, mock_database)

        assert not fc.should_wake_yield_inventory_cycle("1x1x0", 25_000_000)
        assert fc.should_wake_yield_inventory_cycle("1x1x0", 25_000_000)
        assert not fc.should_wake_yield_inventory_cycle("1x1x0", 50_000_000)
        now[0] += fc.YIELD_BALANCE_WAKE_MIN_INTERVAL_SECONDS
        assert fc.should_wake_yield_inventory_cycle("1x1x0", 1)
        # A different channel has independent volume and cooldown state.
        assert fc.should_wake_yield_inventory_cycle("2x2x0", 50_000_000)
        assert fc._claim_yield_inventory_wake_channels() == {
            "1x1x0", "2x2x0",
        }
        assert fc._claim_yield_inventory_wake_channels() == set()

    @pytest.mark.parametrize(
        "channel_id,out_msat",
        [
            (None, 50_000_000),
            ("1x1x0", "malformed"),
            ("1x1x0", True),
            ("1x1x0", 0),
            ("1x1x0", float("nan")),
        ],
    )
    def test_absent_or_malformed_evidence_is_neutral(
        self, mock_plugin, mock_config, mock_database, channel_id, out_msat
    ):
        fc = self._controller(mock_plugin, mock_config, mock_database)
        assert not fc.should_wake_yield_inventory_cycle(channel_id, out_msat)

    def test_non_yield_mode_is_neutral_and_performs_no_io(
        self, mock_plugin, mock_config, mock_database
    ):
        mock_config.market_fee_mode = "undercut"
        fc = FeeController(
            mock_plugin,
            mock_config,
            mock_database,
            fee_authority_gate=FeeAuthorityGate(enabled=True),
        )

        assert not fc.should_wake_yield_inventory_cycle(
            "1x1x0", 50_000_000
        )
        mock_plugin.rpc.assert_not_called()
        mock_database.assert_not_called()

    def test_snapshot_failure_is_neutral(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = self._controller(mock_plugin, mock_config, mock_database)
        mock_config.snapshot.side_effect = RuntimeError("config unavailable")
        assert not fc.should_wake_yield_inventory_cycle(
            "1x1x0", 50_000_000
        )

    def test_clock_failure_is_neutral(
        self, mock_plugin, mock_config, mock_database, monkeypatch
    ):
        fc = self._controller(mock_plugin, mock_config, mock_database)
        monkeypatch.setattr(
            "modules.fee_controller.decision_now",
            MagicMock(side_effect=RuntimeError("clock unavailable")),
        )
        assert not fc.should_wake_yield_inventory_cycle(
            "1x1x0", 50_000_000
        )


class TestYieldScarceMarketAnchor:
    def test_healthy_inventory_keeps_broad_yield_anchor(self):
        assert FeeController._yield_scarce_market_anchor(
            yield_anchor_ppm=10_000,
            frontier_ppm=400,
            substitute_ppm=2_500,
            outbound_ratio=0.50,
        ) == 10_000

    def test_scarce_inventory_bridges_to_frontier(self):
        assert FeeController._yield_scarce_market_anchor(
            yield_anchor_ppm=10_000,
            frontier_ppm=400,
            substitute_ppm=None,
            outbound_ratio=0.20,
        ) == 2_000

    def test_comparable_substitute_is_an_undercut_ceiling(self):
        assert FeeController._yield_scarce_market_anchor(
            yield_anchor_ppm=10_000,
            frontier_ppm=2_500,
            substitute_ppm=3_000,
            outbound_ratio=0.20,
        ) == 2_970

    @pytest.mark.parametrize(
        "overrides",
        [
            {"yield_anchor_ppm": None},
            {"frontier_ppm": "bad"},
            {"frontier_ppm": 20_000},
            {"outbound_ratio": float("nan")},
            {"outbound_ratio": True},
        ],
    )
    def test_absent_or_malformed_required_evidence_is_neutral(self, overrides):
        values = {
            "yield_anchor_ppm": 10_000,
            "frontier_ppm": 400,
            "substitute_ppm": 2_500,
            "outbound_ratio": 0.20,
        }
        values.update(overrides)
        assert FeeController._yield_scarce_market_anchor(**values) is None


class TestYieldScarceInventoryFloor:
    def _floor(self, **overrides):
        values = {
            "inventory_floor_ppm": 14_000,
            "policy_floor_ppm": 60,
            "yield_market_anchor_ppm": 1_800,
            "yield_broad_market_anchor_ppm": 9_000,
            "frontier_ppm": 360,
            "outbound_ratio": 0.15,
        }
        values.update(overrides)
        return FeeController._yield_scarce_inventory_floor(**values)

    def test_valid_scarce_frontier_constrains_depleted_inventory_floor(self):
        assert self._floor() == 1_800

    def test_policy_floor_remains_hard(self):
        assert self._floor(
            policy_floor_ppm=2_000,
            yield_market_anchor_ppm=1_800,
        ) == 2_000

    def test_healthy_inventory_preserves_inventory_floor(self):
        assert self._floor(outbound_ratio=0.50) == 14_000

    def test_survival_reserve_preserves_inventory_floor(self):
        assert self._floor(outbound_ratio=0.049) == 14_000

    @pytest.mark.parametrize("outbound_ratio", [0.05, 0.097, 0.10])
    def test_reserve_aware_admission_allows_market_relief(self, outbound_ratio):
        assert self._floor(outbound_ratio=outbound_ratio) == 1_800

    def test_broad_anchor_without_frontier_discount_preserves_floor(self):
        assert self._floor(yield_market_anchor_ppm=9_000) == 14_000

    @pytest.mark.parametrize(
        "overrides",
        [
            {"yield_market_anchor_ppm": None},
            {"yield_broad_market_anchor_ppm": "bad"},
            {"frontier_ppm": None},
            {"frontier_ppm": 10_000},
            {"outbound_ratio": float("nan")},
            {"outbound_ratio": True},
        ],
    )
    def test_absent_or_malformed_evidence_preserves_floor(self, overrides):
        assert self._floor(**overrides) == 14_000


class TestYieldMarketAnchor:
    def _controller(self, mock_plugin, mock_config, mock_database, channels):
        mock_config.fee_interval = 15
        mock_config.max_fee_ppm = 50_000
        controller = FeeController(
            mock_plugin, mock_config, mock_database,
            fee_authority_gate=FeeAuthorityGate(),
        )
        controller._our_node_id = "02ourid"
        controller.data_service = MagicMock()
        controller.data_service.get_channels.return_value = {"channels": channels}
        controller._neighbor_fee_cache = {}
        return controller

    def test_capacity_weighted_p75_retains_high_feasible_quotes(
        self, mock_plugin, mock_config, mock_database
    ):
        channels = [
            {"source": "02a", "active": True, "fee_per_millionth": 100,
             "fee_base_msat": 0, "satoshis": 1_000_000},
            {"source": "02b", "active": True, "fee_per_millionth": 1_000,
             "fee_base_msat": 0, "satoshis": 1_000_000},
            {"source": "02c", "active": True, "fee_per_millionth": 20_000,
             "fee_base_msat": 0, "satoshis": 10_000_000},
            {"source": "02d", "active": True, "fee_per_millionth": 90_000,
             "fee_base_msat": 0, "satoshis": 100_000_000},
        ]
        controller = self._controller(
            mock_plugin, mock_config, mock_database, channels
        )
        assert controller._get_yield_market_anchor_live("02peer", mock_config) == 20_000

    def test_missing_or_malformed_market_is_neutral(
        self, mock_plugin, mock_config, mock_database
    ):
        controller = self._controller(
            mock_plugin, mock_config, mock_database,
            [{"source": "02a", "active": True, "fee_per_millionth": "bad"}],
        )
        assert controller._get_yield_market_anchor_live("02peer", mock_config) is None

    def test_frontier_uses_capacity_weighted_p25(
        self, mock_plugin, mock_config, mock_database
    ):
        channels = [
            {"source": "02a", "active": True, "fee_per_millionth": 120,
             "fee_base_msat": 0, "satoshis": 1_500_000},
            {"source": "02b", "active": True, "fee_per_millionth": 180,
             "fee_base_msat": 0, "satoshis": 2_000_000},
            {"source": "02c", "active": True, "fee_per_millionth": 360,
             "fee_base_msat": 0, "satoshis": 5_000_000},
            {"source": "02d", "active": True, "fee_per_millionth": 11_000,
             "fee_base_msat": 0, "satoshis": 15_000_000},
        ]
        controller = self._controller(
            mock_plugin, mock_config, mock_database, channels
        )
        assert controller._get_yield_market_frontier_live(
            "02peer", mock_config
        ) == 360

    def test_substitute_quote_uses_only_comparable_capacity(
        self, mock_plugin, mock_config, mock_database
    ):
        channels = [
            {"source": "02small", "active": True, "fee_per_millionth": 100,
             "fee_base_msat": 0, "satoshis": 2_000_000},
            {"source": "02peer1", "active": True, "fee_per_millionth": 2_400,
             "fee_base_msat": 0, "satoshis": 15_000_000},
            {"source": "02peer2", "active": True, "fee_per_millionth": 3_000,
             "fee_base_msat": 0, "satoshis": 14_000_000},
        ]
        controller = self._controller(
            mock_plugin, mock_config, mock_database, channels
        )
        assert controller._get_yield_substitute_quote_live(
            "02peer", 15_000_000, mock_config
        ) == 2_400

    def test_substitute_quote_requires_broader_market_corroboration(
        self, mock_plugin, mock_config, mock_database
    ):
        controller = self._controller(
            mock_plugin,
            mock_config,
            mock_database,
            [{"source": "02only", "active": True, "fee_per_millionth": 10,
              "fee_base_msat": 0, "satoshis": 15_000_000}],
        )
        assert controller._get_yield_substitute_quote_live(
            "02peer", 15_000_000, mock_config
        ) is None


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
        fc = FeeController(mock_plugin, mock_config, mock_database, fee_authority_gate=FeeAuthorityGate())
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

    def test_force_refresh_bypasses_derived_and_gossip_caches(
        self, mock_plugin, mock_config, mock_database
    ):
        old_channels = self._make_channels([50, 100, 200])
        fresh_channels = self._make_channels([120, 120, 120])
        fc = self._setup_fc(
            mock_plugin, mock_config, mock_database, old_channels
        )
        fc._cycle_observations = {}

        assert fc._get_neighbor_fee_percentile("02peer", 0.25) == 50
        fc.data_service.get_channels.return_value = {"channels": fresh_channels}

        assert fc._get_neighbor_fee_percentile(
            "02peer", 0.25, force_refresh=True
        ) == 120
        assert fc.data_service.get_channels.call_count == 2


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
