"""Tests for capex-aware rebalancer (engine-driven)."""

import os
import sys
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import Config, ConfigSnapshot
from modules.rebalancer import EVRebalancer as Rebalancer, RebalanceReasonCode
from modules.capex_budget import CapexBudgetEngine, ChannelCapexBudget, CapexAllocations


def _make_rebalancer(capex_engine=None):
    """Create a minimal Rebalancer with optional capex engine."""
    mock_plugin = MagicMock()
    mock_plugin.rpc = MagicMock()
    mock_db = MagicMock()
    mock_config = MagicMock()
    mock_config.rebalance_max_amount = 5_000_000
    mock_config.rebalance_min_amount = 50_000
    mock_config.snapshot.return_value = mock_config

    rebalancer = Rebalancer.__new__(Rebalancer)
    rebalancer.plugin = mock_plugin
    rebalancer.database = mock_db
    rebalancer.config = mock_config
    rebalancer._profitability_analyzer = MagicMock()
    rebalancer._capex_engine = capex_engine
    rebalancer._hive_router = None
    return rebalancer


class TestEngineInjection:
    """Capex engine can be injected into the rebalancer."""

    def test_set_capex_engine(self):
        r = _make_rebalancer()
        mock_engine = MagicMock()
        r.set_capex_engine(mock_engine)
        assert r._capex_engine is mock_engine

    def test_default_no_engine(self):
        r = _make_rebalancer()
        assert r._capex_engine is None


class TestCapexFallbackWithEngine:
    """_capex_fallback_pass uses engine budgets."""

    def _make_engine_with_budget(self, channel_id, budget_sats, tier, tier_ppm):
        """Create a mock engine that returns a specific budget for one channel."""
        mock_engine = MagicMock(spec=CapexBudgetEngine)
        budget = ChannelCapexBudget(
            channel_id=channel_id,
            budget_msat=budget_sats * 1000,
            tier=tier,
            tier_ppm=tier_ppm,
            priority_class="preservation" if tier in ("proven", "active") else "growth",
        )
        mock_engine.get_channel_budget.return_value = budget
        return mock_engine

    def test_proven_channel_gets_candidate(self):
        """Proven channel with budget gets a capex rebalance candidate."""
        engine = self._make_engine_with_budget("100x1x0", 300, "proven", 2000)
        r = _make_rebalancer(capex_engine=engine)

        # Mock source selection
        source_info = {
            "peer_id": "02" + "b" * 64,
            "capacity": 5_000_000,
            "fee_ppm": 100,
            "spendable": 3_000_000,
        }
        r._select_source_candidates = MagicMock(return_value=[
            ("200x1x0", source_info, 50.0, 10)
        ])

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]
        source_channels = [("200x1x0", source_info, 0.80)]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=source_channels,
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 1
        c = candidates[0]
        assert c.reason_code == "capex_fallback"
        assert c.max_budget_sats > 0
        assert c.max_budget_sats <= 300
        assert c.max_fee_ppm <= 2000
        engine.get_channel_budget.assert_called_with("100x1x0")

    def test_blocked_channel_skipped(self):
        """Blocked channel gets no candidate."""
        engine = self._make_engine_with_budget("100x1x0", 0, "blocked", 0)
        r = _make_rebalancer(capex_engine=engine)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[],
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_zero_budget_channel_skipped(self):
        """Channel with exhausted budget gets no candidate."""
        engine = self._make_engine_with_budget("100x1x0", 0, "proven", 2000)
        r = _make_rebalancer(capex_engine=engine)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[],
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_no_engine_returns_empty(self):
        """Without engine, fallback returns empty list."""
        r = _make_rebalancer(capex_engine=None)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[("200x1x0", {}, 0.80)],
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_active_channels_excluded(self):
        """Channels with active rebalance jobs are skipped."""
        engine = self._make_engine_with_budget("100x1x0", 300, "proven", 2000)
        r = _make_rebalancer(capex_engine=engine)

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=[("200x1x0", {}, 0.80)],
            active_channels={"100x1x0"},  # Already being rebalanced
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 0

    def test_respects_available_slots(self):
        """Stops after available_slots candidates found."""
        mock_engine = MagicMock(spec=CapexBudgetEngine)

        def _budget_for(ch_id):
            return ChannelCapexBudget(
                channel_id=ch_id,
                budget_msat=500_000,
                tier="proven",
                tier_ppm=2000,
                priority_class="preservation",
            )

        mock_engine.get_channel_budget.side_effect = _budget_for
        r = _make_rebalancer(capex_engine=mock_engine)

        source_info = {
            "peer_id": "02" + "b" * 64,
            "capacity": 5_000_000,
            "fee_ppm": 100,
            "spendable": 3_000_000,
        }
        r._select_source_candidates = MagicMock(return_value=[
            ("900x1x0", source_info, 50.0, 10)
        ])

        depleted = [
            (f"{i}00x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10)
            for i in range(1, 6)  # 5 depleted channels
        ]
        source_channels = [("900x1x0", source_info, 0.80)]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=source_channels,
            active_channels=set(),
            available_slots=2,  # Only room for 2
            cfg=r.config,
        )

        assert len(candidates) == 2

    def test_bootstrap_channel_gets_candidate(self):
        """Bootstrap channel with small budget still gets candidate."""
        engine = self._make_engine_with_budget("100x1x0", 200, "bootstrap", 250)
        r = _make_rebalancer(capex_engine=engine)

        source_info = {
            "peer_id": "02" + "b" * 64,
            "capacity": 5_000_000,
            "fee_ppm": 100,
            "spendable": 3_000_000,
        }
        r._select_source_candidates = MagicMock(return_value=[
            ("200x1x0", source_info, 50.0, 10)
        ])

        depleted = [
            ("100x1x0", {
                "peer_id": "02" + "a" * 64,
                "capacity": 5_000_000,
                "spendable": 500_000,
                "fee_ppm": 100,
            }, 0.10),
        ]
        source_channels = [("200x1x0", source_info, 0.80)]

        candidates = r._capex_fallback_pass(
            depleted_channels=depleted,
            source_channels=source_channels,
            active_channels=set(),
            available_slots=5,
            cfg=r.config,
        )

        assert len(candidates) == 1
        c = candidates[0]
        assert c.max_fee_ppm <= 250  # Bootstrap ceiling


class TestCapexMainPath:
    """CapEx budget integration in the main find_rebalance_candidates loop."""

    def test_capex_budget_enables_negative_spread_rebalance(self, mock_plugin, mock_database):
        """Channel with capex budget rebalances even when EV analysis finds nothing."""
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode

        cfg = Config(
            dry_run=True,
            low_liquidity_threshold=0.2,
            high_liquidity_threshold=0.8,
            rebalance_min_amount=50_000,
            rebalance_max_amount=5_000_000,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Inject mock data_service (needed for _report_hive_liquidity_state)
        mock_ds = MagicMock()
        mock_ds.datastore_push.return_value = True
        r.data_service = mock_ds

        # Set up CapEx engine with ACTIVE tier budget
        dest_scid = "100x1x0"
        source_scid = "200x1x0"
        dest_peer = "02" + "a" * 64
        source_peer = "02" + "b" * 64

        mock_engine = MagicMock(spec=CapexBudgetEngine)
        budget = ChannelCapexBudget(
            channel_id=dest_scid,
            budget_msat=500_000,  # 500 sats
            tier="active",
            tier_ppm=1500,
            priority_class="preservation",
        )
        mock_engine.get_channel_budget.return_value = budget
        mock_engine.compute_allocations.return_value = None
        r.set_capex_engine(mock_engine)

        # Mock infrastructure
        mock_database.cleanup_stale_reservations.return_value = 0
        mock_database.list_hot_channel_protection_override_peers.return_value = []
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_failure_metadata.return_value = {"last_error_type": "other"}
        mock_database.get_last_rebalance_time.return_value = 0
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_top_route_pairs.return_value = []
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.01)

        # Depleted dest (10% local, 25 ppm) + overfull source (99% local)
        r._get_channels_with_balances = MagicMock(return_value={
            dest_scid: {
                "peer_id": dest_peer,
                "capacity": 5_000_000,
                "spendable_sats": 500_000,  # 10% local
                "fee_ppm": 25,
            },
            source_scid: {
                "peer_id": source_peer,
                "capacity": 5_000_000,
                "spendable_sats": 4_950_000,  # 99% local
                "fee_ppm": 100,
            },
        })

        # EV analysis returns None (negative spread — no profitable candidate)
        r._analyze_rebalance_ev = MagicMock(return_value=None)

        # Source selection returns a source when called with max_cost_ppm
        source_info = {
            "peer_id": source_peer,
            "capacity": 5_000_000,
            "fee_ppm": 100,
            "spendable_sats": 4_950_000,
        }
        r._select_source_candidates = MagicMock(return_value=[
            (source_scid, source_info, 50.0, 10)
        ])

        result = r.find_rebalance_candidates()

        # Assert: at least 1 candidate with reason_code CAPEX_FALLBACK
        assert len(result) >= 1
        capex_candidates = [c for c in result if c.reason_code == RebalanceReasonCode.CAPEX_FALLBACK.value]
        assert len(capex_candidates) >= 1
        c = capex_candidates[0]
        assert c.to_channel == dest_scid
        assert c.max_fee_ppm <= 1500  # tier_ppm ceiling
        assert c.max_budget_sats <= 500  # budget ceiling

        # Verify _select_source_candidates was called with max_cost_ppm
        select_calls = r._select_source_candidates.call_args_list
        capex_call = [
            call for call in select_calls
            if call.kwargs.get("max_cost_ppm", 0) > 0
            or (len(call.args) > 6 and call.args[6] > 0)
        ]
        assert len(capex_call) >= 1

    def test_ev_positive_takes_priority_over_capex(self, mock_plugin, mock_database):
        """When EV analysis returns a candidate, capex fallback is not triggered."""
        from modules.rebalancer import EVRebalancer, RebalanceCandidate, RebalanceReasonCode

        cfg = Config(
            dry_run=True,
            low_liquidity_threshold=0.2,
            high_liquidity_threshold=0.8,
            rebalance_min_amount=50_000,
            rebalance_max_amount=5_000_000,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_ds = MagicMock()
        mock_ds.datastore_push.return_value = True
        r.data_service = mock_ds

        dest_scid = "100x1x0"
        source_scid = "200x1x0"
        dest_peer = "02" + "a" * 64
        source_peer = "02" + "b" * 64

        # Set up CapEx engine (should NOT be called for source selection)
        mock_engine = MagicMock(spec=CapexBudgetEngine)
        mock_engine.compute_allocations.return_value = None
        r.set_capex_engine(mock_engine)

        mock_database.cleanup_stale_reservations.return_value = 0
        mock_database.list_hot_channel_protection_override_peers.return_value = []
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_failure_metadata.return_value = {"last_error_type": "other"}
        mock_database.get_last_rebalance_time.return_value = 0
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_top_route_pairs.return_value = []
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.01)

        r._get_channels_with_balances = MagicMock(return_value={
            dest_scid: {
                "peer_id": dest_peer,
                "capacity": 5_000_000,
                "spendable_sats": 500_000,
                "fee_ppm": 500,
            },
            source_scid: {
                "peer_id": source_peer,
                "capacity": 5_000_000,
                "spendable_sats": 4_950_000,
                "fee_ppm": 100,
            },
        })

        # EV analysis returns a profitable candidate
        ev_candidate = RebalanceCandidate(
            source_candidates=[source_scid],
            to_channel=dest_scid,
            primary_source_peer_id=source_peer,
            to_peer_id=dest_peer,
            amount_sats=100_000,
            amount_msat=100_000_000,
            outbound_fee_ppm=500,
            inbound_fee_ppm=0,
            source_fee_ppm=100,
            weighted_opp_cost_ppm=50,
            spread_ppm=350,
            max_budget_sats=10,
            max_budget_msat=10_000,
            max_fee_ppm=2000,
            expected_profit_sats=5,
            liquidity_ratio=0.10,
            dest_flow_state="balanced",
            dest_turnover_rate=0.01,
            source_turnover_rate=0.01,
            reason_code=RebalanceReasonCode.EV_POSITIVE.value,
        )
        r._analyze_rebalance_ev = MagicMock(return_value=ev_candidate)

        result = r.find_rebalance_candidates()

        assert len(result) >= 1
        assert result[0].reason_code == RebalanceReasonCode.EV_POSITIVE.value
        # CapEx engine should NOT have been queried for budget
        mock_engine.get_channel_budget.assert_not_called()

    def test_capex_blocked_tier_skipped(self, mock_plugin, mock_database):
        """Channel with blocked CapEx tier gets no capex fallback."""
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode

        cfg = Config(
            dry_run=True,
            low_liquidity_threshold=0.2,
            high_liquidity_threshold=0.8,
            rebalance_min_amount=50_000,
            rebalance_max_amount=5_000_000,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_ds = MagicMock()
        mock_ds.datastore_push.return_value = True
        r.data_service = mock_ds

        dest_scid = "100x1x0"
        source_scid = "200x1x0"
        dest_peer = "02" + "a" * 64
        source_peer = "02" + "b" * 64

        # Blocked tier budget
        mock_engine = MagicMock(spec=CapexBudgetEngine)
        blocked_budget = ChannelCapexBudget(
            channel_id=dest_scid,
            budget_msat=0,
            tier="blocked",
            tier_ppm=0,
            priority_class="growth",
        )
        mock_engine.get_channel_budget.return_value = blocked_budget
        mock_engine.compute_allocations.return_value = None
        r.set_capex_engine(mock_engine)

        mock_database.cleanup_stale_reservations.return_value = 0
        mock_database.list_hot_channel_protection_override_peers.return_value = []
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_failure_metadata.return_value = {"last_error_type": "other"}
        mock_database.get_last_rebalance_time.return_value = 0
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_top_route_pairs.return_value = []
        r._check_capital_controls = MagicMock(return_value=True)
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.01)

        r._get_channels_with_balances = MagicMock(return_value={
            dest_scid: {
                "peer_id": dest_peer,
                "capacity": 5_000_000,
                "spendable_sats": 500_000,
                "fee_ppm": 25,
            },
            source_scid: {
                "peer_id": source_peer,
                "capacity": 5_000_000,
                "spendable_sats": 4_950_000,
                "fee_ppm": 100,
            },
        })

        r._analyze_rebalance_ev = MagicMock(return_value=None)

        result = r.find_rebalance_candidates()

        # No candidates because EV is None and capex tier is blocked
        capex_candidates = [c for c in result if c.reason_code == RebalanceReasonCode.CAPEX_FALLBACK.value]
        assert len(capex_candidates) == 0


class TestDefaultBudgetPreserved:
    """Basic config fields still exist."""

    def test_default_daily_budget_preserved(self):
        cfg = Config()
        assert cfg.daily_budget_sats >= 0
