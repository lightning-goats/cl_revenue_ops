"""Audit wave2: transient RPC failures must fail CONSERVATIVE, not permissive.

Core defect cluster: a transient CLN RPC failure produced empty results that
downstream code treated as confident data, and every default was permissive —
one hiccup could loosen capital controls (bootstrap wallet raids, vanished
hard-bleeder blocks, skipped portfolio governors, 45 ppm/day EV fantasies).

Fix inventory covered here:
  F1: profitability_analyzer — RPC failure never overwrites a good cache and
      never masquerades as "node has no channels".
  F2: capex_budget — bootstrap wallet-exploration requires a REAL zero-revenue
      snapshot, never a data outage.
  F3: capex_budget — unreadable wallet is not an empty wallet (no
      reserve-deficit budget when listfunds fails).
  F4: capacity_planner — portfolio governor and peer-exposure cap fail closed.
  F5: capacity_planner — conservative EV bootstrap prior; no EV opens when
      profitability data is unavailable.
  F6: capacity_planner — execute_cycle single-flight + collision-proof
      reservation ids.
  F7: protection_service — confidence 0.0 is LOW confidence; LN+ contract
      protection expires.
  F8: flow_analysis — flow results keyed by normalized scid.
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import Config
from modules.profitability_analyzer import (
    ChannelProfitabilityAnalyzer,
    BleederClassification,
)
from modules.capex_budget import CapexBudgetEngine, MSAT_PER_SAT
from modules.capacity_planner import CapacityPlanner


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

PEER = "02" + "a" * 64
SCID = "111x222x0"


def _make_analyzer():
    """ChannelProfitabilityAnalyzer with mocked dependencies."""
    plugin = MagicMock()
    config = MagicMock()
    config.estimated_open_cost_sats = 2000
    database = MagicMock()
    analyzer = ChannelProfitabilityAnalyzer(plugin, config, database)
    return analyzer


def _bleeder(channel_id=SCID, classification="hard"):
    return BleederClassification(
        channel_id=channel_id,
        peer_id=PEER,
        classification=classification,
        reason="test",
        rebalance_cost_30d=5000,
        revenue_30d=100,
        net_profit_30d=-4900,
        net_profit_7d=-1000,
        recommended_action="disable_rebalance",
    )


def _make_capex_prof(
    contribution_msat=0,
    fees_earned_msat=0,
    capacity_sats=5_000_000,
    classification="break_even",
):
    prof = MagicMock()
    prof.channel_id = SCID
    prof.peer_id = PEER
    prof.revenue.total_contribution_msat = contribution_msat
    prof.revenue.fees_earned_msat = fees_earned_msat
    prof.revenue.total_forward_count = 100
    prof.revenue.sourced_fee_contribution_msat = 0
    prof.days_open = 60
    prof.capacity_sats = capacity_sats
    prof.classification.value = classification
    prof.marginal_roi = 0.0
    prof.marginal_roi_reliable = True
    prof.contribution_30d_msat = contribution_msat
    prof.fees_earned_30d_msat = fees_earned_msat
    prof.sourced_fee_30d_msat = 0
    prof.window_30d_available = True
    prof.channel_role.value = "balanced"
    return prof


def _make_capex_engine(
    channel_profitabilities=None,
    prof_raises=False,
    prof_available=True,
    confirmed_onchain_sats=5_000_000,
    listfunds_raises=False,
    config_overrides=None,
):
    mock_profitability = MagicMock()
    if prof_raises:
        mock_profitability.analyze_all_channels.side_effect = RuntimeError("rpc down")
    else:
        mock_profitability.analyze_all_channels.return_value = (
            channel_profitabilities or {}
        )
    mock_profitability.data_available.return_value = prof_available
    mock_profitability.get_bleeder_status.return_value = None

    mock_db = MagicMock()
    mock_db.get_confirmed_onchain_sats.return_value = confirmed_onchain_sats
    if listfunds_raises:
        mock_db.plugin.rpc.listfunds.side_effect = RuntimeError("listfunds down")
    mock_db.get_channel_rebalance_success_rate.return_value = None
    mock_db.get_spend_ledger_summary.return_value = {
        "spent_by_category": {},
        "reserved_by_category": {},
    }

    cfg = Config()
    for k, v in (config_overrides or {}).items():
        setattr(cfg, k, v)

    engine = CapexBudgetEngine(
        profitability_analyzer=mock_profitability,
        database=mock_db,
        config=cfg,
        capital_efficiency=None,
    )
    engine._get_total_capex_by_channel = lambda window_days=30: {}
    return engine


def _make_cycle_cfg(**over):
    cfg = MagicMock()
    cfg.planner_enabled = True
    cfg.planner_max_opens_per_cycle = 2
    cfg.planner_max_closes_per_cycle = 2
    cfg.planner_dry_run = False
    cfg.planner_execute_closes = False
    cfg.planner_max_defibrillations_per_cycle = 1
    cfg.planner_max_fee_rate_sat_vb = 50.0
    cfg.min_wallet_reserve = 500000
    cfg.planner_min_channel_sats = 500000
    cfg.planner_max_channel_sats = 10_000_000
    cfg.planner_min_annual_roi_pct = 1.0
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _make_cycle_planner():
    plugin = MagicMock()
    plugin.rpc.feerates.return_value = {"perkb": {"opening": 10000}}
    plugin.rpc.listfunds.return_value = {
        "outputs": [{"amount_msat": 50_000_000_000, "status": "confirmed"}],
        "channels": [],
    }
    plugin.rpc.listnodes.return_value = {"nodes": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}

    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    prof_analyzer.analyze_all_channels.return_value = {}
    flow_analyzer.analyze_all_channels.return_value = {}
    prof_analyzer.data_available.return_value = True
    prof_analyzer.database.get_planner_actions.return_value = []
    prof_analyzer.database.get_recent_planner_actions.return_value = []
    prof_analyzer.database.get_planner_candidates.return_value = []
    prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
    prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
    prof_analyzer.database.get_fee_strategy_state.return_value = None
    prof_analyzer.database.get_peer_reputation.return_value = None
    prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
        'count': 0, 'marginal_roi_proxy': 0,
    }
    prof_analyzer.database.record_planner_action.return_value = 1
    prof_analyzer.identify_bleeders_v2.return_value = []

    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner, plugin, prof_analyzer, flow_analyzer


# ---------------------------------------------------------------------------
# FIX 1: profitability_analyzer conservative failure semantics
# ---------------------------------------------------------------------------

class TestProfitabilityRpcFailure:

    def test_rpc_failure_serves_stale_snapshot(self):
        """A failed pass must NOT overwrite a previously-good cache."""
        analyzer = _make_analyzer()
        prior = {SCID: MagicMock()}
        analyzer._profitability_cache = prior
        analyzer._cache_timestamp = int(time.time()) - 2000  # stale
        analyzer._snapshot_available = True

        analyzer.plugin.rpc.listpeerchannels.side_effect = RuntimeError("rpc down")

        result = analyzer.analyze_all_channels(force=True)

        assert result is prior
        assert analyzer._profitability_cache is prior
        assert analyzer.data_available() is True

    def test_rpc_failure_without_prior_cache_marks_unavailable(self):
        analyzer = _make_analyzer()
        analyzer.plugin.rpc.listpeerchannels.side_effect = RuntimeError("rpc down")

        result = analyzer.analyze_all_channels()

        assert result == {}
        assert analyzer.data_available() is False
        # Timestamp restored so the next caller retries instead of trusting
        # the failed pass for the full 900s TTL.
        assert analyzer._cache_timestamp == 0

    def test_empty_channel_list_is_a_real_snapshot(self, monkeypatch):
        """A node with genuinely zero channels IS available data."""
        analyzer = _make_analyzer()
        analyzer.plugin.rpc.listpeerchannels.return_value = {"channels": []}
        monkeypatch.setattr(
            "modules.profitability_analyzer.BookkeeperCache", MagicMock()
        )
        analyzer._get_all_revenue_data = lambda: {}
        analyzer._get_all_full_pnl_batch = lambda days: {}
        analyzer._get_all_fee_states = lambda: {}
        analyzer._push_profitability_summary = lambda results: None

        result = analyzer.analyze_all_channels()

        assert result == {}
        assert analyzer.data_available() is True

    def test_should_rebalance_unavailable_is_false(self):
        analyzer = _make_analyzer()
        analyzer.plugin.rpc.listpeerchannels.side_effect = RuntimeError("rpc down")

        ok, reason = analyzer.should_rebalance(SCID)

        assert ok is False
        assert reason == "profitability_data_unavailable"

    def test_should_rebalance_missing_channel_with_real_snapshot_stays_true(self):
        analyzer = _make_analyzer()
        analyzer._profitability_cache = {}
        analyzer._cache_timestamp = int(time.time())
        analyzer._snapshot_available = True

        ok, reason = analyzer.should_rebalance(SCID)

        assert ok is True
        assert reason == "no_profitability_data"

    def test_prune_closed_channels_survives_rpc_failure(self):
        """RPC failure must not wipe the whole profitability cache."""
        analyzer = _make_analyzer()
        analyzer._profitability_cache = {SCID: MagicMock(), "333x1x0": MagicMock()}
        analyzer.plugin.rpc.listpeerchannels.side_effect = RuntimeError("rpc down")

        pruned = analyzer.prune_closed_channels()

        assert pruned == 0
        assert len(analyzer._profitability_cache) == 2


class TestBleederStatusUnavailable:

    def test_identify_bleeders_v2_returns_none_on_rpc_failure(self):
        analyzer = _make_analyzer()
        analyzer.plugin.rpc.listpeerchannels.side_effect = RuntimeError("rpc down")

        assert analyzer.identify_bleeders_v2() is None

    def test_unavailable_pass_is_not_cached_as_no_bleeders(self):
        analyzer = _make_analyzer()
        analyzer.identify_bleeders_v2 = MagicMock(return_value=None)

        assert analyzer.get_bleeder_status(SCID) is None
        # Must not have cached "no bleeders" for 300s
        assert analyzer._bleeder_cache is None

        # Recovery on the very next call (no 300s block)
        hard = _bleeder()
        analyzer.identify_bleeders_v2 = MagicMock(return_value=[hard])
        assert analyzer.get_bleeder_status(SCID) is hard

    def test_stale_bleeder_map_served_when_pass_fails(self):
        analyzer = _make_analyzer()
        hard = _bleeder()
        analyzer._bleeder_cache = {SCID: hard}
        analyzer._bleeder_cache_time = time.time() - 999  # stale
        analyzer.identify_bleeders_v2 = MagicMock(return_value=None)

        assert analyzer.get_bleeder_status(SCID) is hard


# ---------------------------------------------------------------------------
# FIX 2: capex bootstrap requires affirmative zero-revenue evidence
# ---------------------------------------------------------------------------

class TestCapexBootstrapGating:

    def test_no_exploration_budget_when_profitability_raises(self):
        engine = _make_capex_engine(prof_raises=True)
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_msat == 0

    def test_no_exploration_budget_when_snapshot_unavailable(self):
        engine = _make_capex_engine(
            channel_profitabilities={}, prof_available=False
        )
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_msat == 0

    def test_real_empty_snapshot_still_gets_bootstrap_budget(self):
        """A genuine zero-revenue node keeps the wallet-excess bootstrap."""
        engine = _make_capex_engine(
            channel_profitabilities={},
            prof_available=True,
            confirmed_onchain_sats=5_000_000,
        )
        alloc = engine.compute_allocations()
        # Wallet excess (5M - 1M reserve) funds bootstrap exploration; the
        # global daily/weekly envelope may clamp it, but it must be nonzero.
        assert alloc.fleet_exploration_budget_msat > 0
        assert alloc.profitability_unavailable is False


# ---------------------------------------------------------------------------
# FIX 3: unreadable wallet is not an empty wallet
# ---------------------------------------------------------------------------

class TestCapexWalletUnreadable:

    def test_confirmed_onchain_none_when_listfunds_fails(self):
        engine = _make_capex_engine(listfunds_raises=True)
        assert engine._get_confirmed_onchain_sats() is None

    def test_confirmed_onchain_reads_direct_listfunds_dict(self):
        engine = _make_capex_engine()
        engine._database.plugin.rpc.listfunds.return_value = {
            "outputs": [
                {"status": "confirmed", "amount_msat": 2_000_000_000},
                {"status": "unconfirmed", "amount_msat": 9_000_000_000},
            ]
        }
        assert engine._get_confirmed_onchain_sats() == 2_000_000

    def test_no_reserve_deficit_budget_when_wallet_unreadable(self):
        prof = _make_capex_prof(
            contribution_msat=10_000 * MSAT_PER_SAT,
            fees_earned_msat=10_000 * MSAT_PER_SAT,
        )
        engine = _make_capex_engine(
            channel_profitabilities={SCID: prof},
            listfunds_raises=True,
            # DB helper double-swallows the same failure to 0 — the engine
            # must not trust that 0.
            confirmed_onchain_sats=0,
            config_overrides={"min_wallet_reserve": 1_000_000},
        )
        alloc = engine.compute_allocations()
        # Unknown wallet state cannot create an operational priority flip.
        assert alloc.priority_class != "operational"

    def test_readable_deficit_still_sets_operational_priority(self):
        prof = _make_capex_prof(
            contribution_msat=10_000 * MSAT_PER_SAT,
            fees_earned_msat=10_000 * MSAT_PER_SAT,
        )
        engine = _make_capex_engine(
            channel_profitabilities={SCID: prof},
            confirmed_onchain_sats=0,
            config_overrides={"min_wallet_reserve": 1_000_000},
        )
        alloc = engine.compute_allocations()
        assert alloc.priority_class == "operational"


# ---------------------------------------------------------------------------
# FIX 4: portfolio governor and exposure cap fail closed
# ---------------------------------------------------------------------------

class TestPortfolioGovernorFailClosed:

    def test_peer_exposure_cap_blocks_when_fetch_fails(self):
        planner, plugin, prof, flow = _make_cycle_planner()
        planner._cycle_peer_channels = None
        plugin.rpc.listpeerchannels.side_effect = RuntimeError("rpc down")
        cfg = _make_cycle_cfg()

        reason = planner._peer_exposure_cap_reason(PEER, cfg)

        assert reason is not None
        assert "unavailable" in reason or "unknown" in reason

    def test_unknown_portfolio_state_blocks_opens(self):
        planner, plugin, prof, flow = _make_cycle_planner()
        plugin.rpc.listpeerchannels.side_effect = RuntimeError("rpc down")
        planner._discover_peers = MagicMock(return_value=[{
            "peer_id": "03" + "b" * 64, "score": 1.0,
            "source": "winner", "reason": "test",
        }])
        planner._execute_open = MagicMock(return_value={"status": "completed"})
        cfg = _make_cycle_cfg()

        result = planner.execute_cycle(cfg)

        assert result["portfolio_state"] == "unknown"
        assert result["opens"] == []
        planner._execute_open.assert_not_called()
        assert any("unknown" in r.lower() for r in result["skipped_reasons"])


# ---------------------------------------------------------------------------
# FIX 5: conservative EV bootstrap prior; no EV opens without data
# ---------------------------------------------------------------------------

class TestOpenEvConservative:

    def _ev_planner(self):
        planner, plugin, prof, flow = _make_cycle_planner()
        prof.database.get_peer_closed_channel_profit_summary.return_value = {
            'daily_net_est_sats': 0,
        }
        prof.database.get_historical_inbound_fee_ppm.return_value = None
        planner._estimate_open_cost = lambda: 1000
        planner._estimate_close_cost = lambda: 1000
        return planner, prof

    def test_bootstrap_prior_is_low_single_digit_ppm(self):
        from modules.capacity_planner import BOOTSTRAP_FORECAST_DAILY_PPM
        assert 0 < BOOTSTRAP_FORECAST_DAILY_PPM <= 5.0

    def test_bootstrap_ev_no_longer_approves_everything(self):
        """With no routing history a 1M-sat open must not look profitable."""
        planner, prof = self._ev_planner()
        planner._observed_daily_ppm_cache = (None,)  # no anchor
        cfg = _make_cycle_cfg()

        ev = planner._calculate_open_ev(PEER, 1_000_000, cfg)

        # Legacy 45 ppm/day fallback made this ~+360 sats; conservative
        # bootstrap prior must reject it.
        assert ev < 0

    def test_ev_zero_when_profitability_unavailable(self):
        planner, prof = self._ev_planner()
        prof.data_available.return_value = False
        prof.database.get_peer_closed_channel_profit_summary.return_value = {
            'daily_net_est_sats': 50,
        }
        cfg = _make_cycle_cfg()

        ev = planner._calculate_open_ev(PEER, 1_000_000, cfg)

        assert ev <= 0


# ---------------------------------------------------------------------------
# FIX 6: single-flight execute_cycle + collision-proof reservation ids
# ---------------------------------------------------------------------------

class TestPlannerSingleFlight:

    def test_concurrent_cycle_is_rejected(self):
        planner, plugin, prof, flow = _make_cycle_planner()
        cfg = _make_cycle_cfg()
        planner._cycle_peer_channels = ["sentinel"]

        assert planner._cycle_lock.acquire(blocking=False)
        try:
            result = planner.execute_cycle(cfg)
        finally:
            planner._cycle_lock.release()

        assert result.get("skipped") is True
        assert result.get("reason") == "cycle_already_running"
        # Shared per-cycle state must not have been reset by the loser.
        assert planner._cycle_peer_channels == ["sentinel"]

    def test_lock_released_after_cycle(self):
        planner, plugin, prof, flow = _make_cycle_planner()
        cfg = _make_cycle_cfg()
        planner.execute_cycle(cfg)
        assert planner._cycle_lock.acquire(blocking=False)
        planner._cycle_lock.release()

    def test_reservation_ids_unique_within_same_second(self):
        planner, plugin, prof, flow = _make_cycle_planner()
        planner._check_open_allowed = lambda peer_id: (True, "ok")
        prof.database.reserve_spend.return_value = False  # abort after reserve
        cfg = _make_cycle_cfg(planner_dry_run=False)

        planner._execute_open(PEER, 1_000_000, cfg, reason="t1")
        planner._execute_open(PEER, 1_000_000, cfg, reason="t2")

        ids = [
            call.kwargs["reservation_id"]
            for call in prof.database.reserve_spend.call_args_list
        ]
        assert len(ids) == 2
        assert ids[0] != ids[1]


# ---------------------------------------------------------------------------
# FIX 7: protection_service permissive defaults
# ---------------------------------------------------------------------------

class TestProtectionService:

    def test_zero_confidence_is_low_confidence(self):
        from types import SimpleNamespace
        from modules.protection_service import close_protection_reason
        prof = SimpleNamespace(
            role_30d=None, channel_role=None, marginal_roi_percent=-90.0,
            window_30d_available=True, sourced_fee_30d_msat=0, days_open=10,
            revenue=SimpleNamespace(sourced_fee_contribution_sats=0),
        )
        flow = SimpleNamespace(confidence=0.0, forward_count=5)

        reason = close_protection_reason(SCID, prof, flow, set())

        assert reason == "KALMAN_LOW_CONFIDENCE"


# ---------------------------------------------------------------------------
# FIX 8: flow results keyed by normalized scid
# ---------------------------------------------------------------------------

class TestFlowScidNormalization:

    def test_colon_scid_normalized_in_analyze_all(self):
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        config.source_threshold = 0.5
        config.sink_threshold = -0.5
        config.flow_window_days = 7
        config.htlc_congestion_threshold = 0.8
        database = MagicMock()

        fa = FlowAnalyzer(plugin, config, database)
        fa._get_channels = lambda: [{
            "short_channel_id": "111:222:0",
            "peer_id": PEER,
            "spendable_msat": 1_000_000_000,
            "receivable_msat": 1_000_000_000,
            "total_msat": 2_000_000_000,
        }]
        fa._get_daily_flow_from_db = lambda: {}
        fa._apply_kalman_reclassification = lambda **kwargs: None
        fa._flush_kalman_saves = lambda rows: None
        fa._update_temporal_profiles_bulk = lambda ids: None
        database.get_continuous_net_flow_all.return_value = {}
        database.get_all_channel_states.return_value = []
        plugin.rpc.listpeerchannels.return_value = {"channels": []}

        results = fa._analyze_all_channels_impl()

        assert "111x222x0" in results
        assert "111:222:0" not in results
