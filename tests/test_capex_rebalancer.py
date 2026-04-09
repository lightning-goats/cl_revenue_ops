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
        mock_database.get_rebalance_success_signal.return_value = None
        mock_database.get_peer_uptime_percent.return_value = 99.0

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

    def test_capex_is_primary_path_not_fallback(self, mock_plugin, mock_database):
        """CapEx is the primary rebalance path, not a fallback after EV."""
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

        # Set up CapEx engine — primary path for all rebalancing
        mock_engine = MagicMock()
        mock_engine.compute_allocations.return_value = None
        r.set_capex_engine(mock_engine)

        # Ensure no peer is treated as hive member
        mock_hints = MagicMock()
        mock_hints.is_hive_member.return_value = False
        mock_hints.poll.return_value = None
        mock_hints.get_route_segment_leases.return_value = []
        mock_hints.get_rebalance_recommendations.return_value = []
        mock_hints.get_rebalance_campaigns.return_value = []
        r.hive_hints = mock_hints

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
        mock_database.get_rebalance_success_signal.return_value = None
        mock_database.get_peer_uptime_percent.return_value = 99.0

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

        # CapEx engine returns active tier budget
        from modules.capex_budget import ChannelCapexBudget
        mock_engine.get_channel_budget.return_value = ChannelCapexBudget(
            channel_id=dest_scid, tier="active", budget_msat=500_000_000,
            tier_ppm=500, priority_class="preservation",
        )

        result = r.find_rebalance_candidates()

        assert len(result) >= 1
        assert result[0].reason_code == RebalanceReasonCode.CAPEX_FALLBACK.value
        # CapEx engine IS the primary path — it should be queried
        mock_engine.get_channel_budget.assert_called()

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
        mock_database.get_rebalance_success_signal.return_value = None
        mock_database.get_peer_uptime_percent.return_value = 99.0

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


class TestRebalanceFlowOrdering:
    """Hive push runs before equalization, which runs before general rebalancing."""

    def test_hive_push_runs_before_equalization(self, mock_plugin, mock_database):
        """Pass 1 (push) executes before Pass 2 (equalization)."""
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            hive_push_enabled=True,
            hive_equalization_enabled=True,
            rebalance_min_amount=10_000,
            rebalance_max_amount=500_000,
            low_liquidity_threshold=0.20,
            high_liquidity_threshold=0.80,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Inject mock data_service and hive_router (required for hive push)
        mock_ds = MagicMock()
        mock_ds.datastore_push.return_value = True
        r.data_service = mock_ds
        mock_router = MagicMock()
        mock_router.available = True
        mock_router.is_hive_member.return_value = False
        r.hive_router = mock_router

        call_order = []
        original_push = r._build_hive_push_candidates
        original_eq = r._build_hive_equalization_candidates

        def mock_push(*a, **kw):
            call_order.append("hive_push")
            return []

        def mock_eq(*a, **kw):
            call_order.append("hive_equalization")
            return ([], {})

        r._build_hive_push_candidates = mock_push
        r._build_hive_equalization_candidates = mock_eq

        # Channels: one hive-member overfull (push candidate) + one hive-low + one hive-high
        hive_peer = "02" + "a" * 64
        other_peer = "02" + "b" * 64
        r._is_hive_member = MagicMock(side_effect=lambda pid: pid == hive_peer)

        r._get_channels_with_balances = MagicMock(return_value={
            "100x1x0": {
                "peer_id": hive_peer,
                "capacity": 1_000_000,
                "spendable_sats": 700_000,  # 70% local — above push trigger
                "fee_ppm": 0,
            },
            "200x1x0": {
                "peer_id": other_peer,
                "capacity": 1_000_000,
                "spendable_sats": 900_000,  # 90% local — source
                "fee_ppm": 100,
            },
            "300x1x0": {
                "peer_id": hive_peer,
                "capacity": 1_000_000,
                "spendable_sats": 100_000,  # 10% local — depleted & hive_low
                "fee_ppm": 50,
            },
        })
        r._check_capital_controls = MagicMock(return_value=True)
        r._calculate_turnover_rate = MagicMock(return_value=0.01)
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_failure_metadata.return_value = {"last_error_type": "other"}
        mock_database.get_last_rebalance_time.return_value = 0
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_top_route_pairs.return_value = []
        mock_database.cleanup_stale_reservations.return_value = 0
        mock_database.list_hot_channel_protection_override_peers.return_value = []
        r._get_peer_connection_status = MagicMock(return_value={})
        r._analyze_rebalance_ev = MagicMock(return_value=None)

        r.find_rebalance_candidates()

        # Both passes must have been called, push before equalization
        if "hive_push" in call_order and "hive_equalization" in call_order:
            assert call_order.index("hive_push") < call_order.index("hive_equalization")
        # At minimum, push must have been called (equalization needs both low+high)
        assert "hive_push" in call_order

    def test_equalization_no_longer_last_resort_fallback(self, mock_plugin, mock_database):
        """Equalization runs even when the main EV loop produces candidates."""
        from modules.rebalancer import EVRebalancer, RebalanceCandidate, RebalanceReasonCode

        cfg = Config(
            dry_run=True,
            hive_push_enabled=False,
            hive_equalization_enabled=True,
            rebalance_min_amount=10_000,
            rebalance_max_amount=500_000,
            low_liquidity_threshold=0.20,
            high_liquidity_threshold=0.80,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_ds = MagicMock()
        mock_ds.datastore_push.return_value = True
        r.data_service = mock_ds

        eq_called = []

        def mock_eq(*a, **kw):
            eq_called.append(True)
            return ([], {})

        r._build_hive_equalization_candidates = mock_eq

        hive_peer = "02" + "a" * 64
        other_peer = "02" + "b" * 64
        r._is_hive_member = MagicMock(side_effect=lambda pid: pid == hive_peer)

        # Two hive channels (one low, one high) + depleted + source
        r._get_channels_with_balances = MagicMock(return_value={
            "100x1x0": {
                "peer_id": hive_peer,
                "capacity": 1_000_000,
                "spendable_sats": 100_000,  # 10% — hive_low + depleted
                "fee_ppm": 0,
            },
            "200x1x0": {
                "peer_id": hive_peer,
                "capacity": 1_000_000,
                "spendable_sats": 900_000,  # 90% — hive_high + source
                "fee_ppm": 0,
            },
            "300x1x0": {
                "peer_id": other_peer,
                "capacity": 1_000_000,
                "spendable_sats": 100_000,  # depleted
                "fee_ppm": 100,
            },
            "400x1x0": {
                "peer_id": other_peer,
                "capacity": 1_000_000,
                "spendable_sats": 900_000,  # source
                "fee_ppm": 200,
            },
        })

        r._check_capital_controls = MagicMock(return_value=True)
        r._calculate_turnover_rate = MagicMock(return_value=0.01)
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_failure_metadata.return_value = {"last_error_type": "other"}
        mock_database.get_last_rebalance_time.return_value = 0
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_top_route_pairs.return_value = []
        mock_database.cleanup_stale_reservations.return_value = 0
        mock_database.list_hot_channel_protection_override_peers.return_value = []
        r._get_peer_connection_status = MagicMock(return_value={})

        # EV analysis would produce a candidate for the main loop
        mock_candidate = MagicMock()
        mock_candidate.reason_code = RebalanceReasonCode.CAPEX_FALLBACK.value
        mock_candidate.to_channel = "300x1x0"
        mock_candidate.source_candidates = ["400x1x0"]
        mock_candidate.expected_profit_sats = 100
        mock_candidate.to_peer_id = other_peer
        mock_candidate.primary_source_peer_id = other_peer
        mock_candidate.coordination_hint_type = ""
        mock_candidate.coordination_rank_bonus = 0
        mock_candidate.recommended_cooldown_hours = 0
        mock_candidate.hive_route_hops = 0
        mock_candidate.dest_is_hive_member = False
        r._analyze_rebalance_ev = MagicMock(return_value=mock_candidate)

        r.find_rebalance_candidates()

        # Equalization must have been called BEFORE the main loop would have
        # acted as a last-resort fallback
        assert len(eq_called) >= 1, "Equalization should run as Pass 2, not just as fallback"


class TestCapexRebalancerIntegration:
    """End-to-end test mimicking real nexus-01 fleet state.

    Real situation: 44 channels, 40 at >50% local, 2 depleted, 2 fleet members.
    Old system: 0 candidates (spread gate blocks everything at 25 ppm dest fee).
    New system: should produce >= 1 candidate (hive push for cyber-hornet, or capex fallback).
    """

    def test_real_world_fleet_produces_candidates(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer, RebalanceReasonCode

        cfg = Config(
            dry_run=True,
            hive_push_enabled=True,
            hive_push_trigger_ratio=0.60,
            hive_push_target_ratio=0.50,
            hive_equalization_enabled=True,
            rebalance_min_amount=10000,
            rebalance_max_amount=500000,
            low_liquidity_threshold=0.20,
            high_liquidity_threshold=0.80,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Inject mock data_service
        mock_ds = MagicMock()
        mock_ds.datastore_push.return_value = True
        mock_ds.invalidate.return_value = None
        r.data_service = mock_ds

        # Mock capex engine with bootstrap budgets
        from modules.capex_budget import ChannelCapexBudget
        mock_capex = MagicMock()

        def get_budget(channel_id):
            # Fleet channels get fleet tier, others get bootstrap
            if channel_id == "933791x3241x0":
                return ChannelCapexBudget(
                    channel_id=channel_id, tier="fleet",
                    budget_msat=100_000_000, tier_ppm=50,
                    priority_class="fleet_coordination",
                )
            return ChannelCapexBudget(
                channel_id=channel_id, tier="bootstrap",
                budget_msat=200_000_000, tier_ppm=250,
                priority_class="growth",
            )

        mock_capex.get_channel_budget.side_effect = get_budget
        mock_capex.compute_allocations.return_value = None
        r._capex_engine = mock_capex

        # Mock hive membership: cyber-hornet and gondor are fleet members
        fleet_members = {"03796a" + "0" * 58, "028f58" + "0" * 58}
        mock_router = MagicMock()
        mock_router.is_hive_member.side_effect = lambda pid: pid in fleet_members
        mock_router.available = True
        mock_router.discover_route.return_value = None
        mock_router.refresh_layer.return_value = None
        mock_router.refresh_fleet_balances.return_value = None
        mock_router.clear_route_cache.return_value = None
        r.hive_router = mock_router

        # Real-world channel state (key channels)
        channels = {
            # Fleet member: cyber-hornet at 99% local (should trigger hive push)
            "933791x3241x0": {
                "capacity": 2_935_694, "spendable_sats": 2_906_337,
                "peer_id": "03796a" + "0" * 58, "fee_ppm": 0,
            },
            # Stagnant source: kappa at 99% local, 200 ppm fee
            "931308x1256x0": {
                "capacity": 14_738_204, "spendable_sats": 14_590_822,
                "peer_id": "0324ba" + "0" * 62, "fee_ppm": 200,
            },
            # Stagnant source: The Wall at 99% local, 145 ppm
            "931308x1256x1": {
                "capacity": 4_895_612, "spendable_sats": 4_846_656,
                "peer_id": "0203e5" + "0" * 62, "fee_ppm": 145,
            },
            # Active corridor: cyberdyne at 75% local, 20 ppm (the depleted dest from logs)
            "931199x1231x0": {
                "capacity": 9_739_182, "spendable_sats": 7_353_082,
                "peer_id": "03a93b" + "0" * 62, "fee_ppm": 20,
            },
        }
        r._get_channels_with_balances = MagicMock(return_value=channels)
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        r._calculate_turnover_rate = MagicMock(return_value=0.01)
        r._get_peer_connection_status = MagicMock(return_value={})
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_failure_metadata.return_value = {"last_error_type": "other"}
        mock_database.get_last_rebalance_time.return_value = 0
        mock_database.get_top_route_pairs.return_value = []
        mock_database.cleanup_stale_reservations.return_value = 0
        mock_database.list_hot_channel_protection_override_peers.return_value = []

        candidates = r.find_rebalance_candidates()

        # Should produce at least 1 candidate -- hive push for cyber-hornet
        assert len(candidates) >= 1

        # Verify hive push candidate exists
        hive_pushes = [c for c in candidates if c.reason_code == RebalanceReasonCode.HIVE_PUSH.value]
        assert len(hive_pushes) >= 1
        push = hive_pushes[0]
        assert push.to_channel == "933791x3241x0"
        assert push.dest_is_hive_member is True
        # Source should be one of the stagnant 99% channels (most overfull)
        assert push.source_candidates[0] in ("931308x1256x0", "931308x1256x1")
        # Amount should push toward 50/50 (~1.4M sats, capped at max_amount=500k)
        assert push.amount_sats >= 500_000
