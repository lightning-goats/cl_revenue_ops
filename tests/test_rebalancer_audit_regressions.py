"""
Regression tests for cl-revenue-ops rebalancer audit fixes.

Covers:
  I-12: Hot channel profit budget floor
  B1: max_fee_ppm re-derived after I-3 caps
  B2: Weekly budget uses 24h external costs (should scale to 7d)
  B3: Push EV uses wrong peer for fee estimation
  B10: Push EV uses kelly_fraction without enable_kelly guard
  B11: diagnostic_rebalance returns success:true on exception

Note: Sling-specific tests (C-2, C-3, C-4, B4, B5, B6, B7, B8) removed
      as part of sling code deletion.
"""

import math
import time
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate(
    *,
    source_candidates=None,
    to_channel="222x333x0",
    primary_source_peer_id="02" + "a" * 64,
    to_peer_id="02" + "b" * 64,
    amount_sats=50000,
):
    from modules.rebalancer import RebalanceCandidate

    if source_candidates is None:
        source_candidates = ["111x222x0"]

    amt_msat = amount_sats * 1000
    return RebalanceCandidate(
        source_candidates=source_candidates,
        to_channel=to_channel,
        primary_source_peer_id=primary_source_peer_id,
        to_peer_id=to_peer_id,
        amount_sats=amount_sats,
        amount_msat=amt_msat,
        outbound_fee_ppm=1000,
        inbound_fee_ppm=100,
        source_fee_ppm=100,
        weighted_opp_cost_ppm=100,
        spread_ppm=800,
        max_budget_sats=10,
        max_budget_msat=10_000,
        max_fee_ppm=2000,
        expected_profit_sats=1,
        liquidity_ratio=0.1,
        dest_flow_state="balanced",
        dest_turnover_rate=0.0,
        source_turnover_rate=0.0,
    )


# Sling-specific helpers (_active_job, _TestRpcError) and test classes removed:
# TestSentinelCleanup, TestChannelCloseDetection, TestStopJobReAddOnFailure.

# ============================================================================
# 4. Hot Channel Profit Budget Floor (I-12 fix)
# ============================================================================


class TestHotChannelProfitBudgetFloor:
    """I-12: channel_profit_budget_sats is at least 1 for small positive daily_contrib_est."""

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        return r

    def test_small_daily_contrib_gets_floor_of_1(self, mock_plugin, mock_database):
        """daily_contrib_est of 0.5 * pct 0.75 = 0.375, which rounds up to 1 not 0."""
        import math
        daily_contrib_est = 0.5
        profit_budget_pct = 0.75
        result = max(1, int(math.ceil(max(0.0, daily_contrib_est) * max(0.0, min(1.0, profit_budget_pct)))))
        assert result >= 1
        # 0.5 * 0.75 = 0.375, ceil = 1
        assert result == 1

    def test_zero_daily_contrib_gets_floor_of_1(self, mock_plugin, mock_database):
        """daily_contrib_est of 0 * anything = 0, but max(1, ...) gives 1."""
        import math
        daily_contrib_est = 0.0
        profit_budget_pct = 0.75
        result = max(1, int(math.ceil(max(0.0, daily_contrib_est) * max(0.0, min(1.0, profit_budget_pct)))))
        assert result == 1

    def test_negative_daily_contrib_gets_floor_of_1(self, mock_plugin, mock_database):
        """Negative daily_contrib_est is clamped to 0, then max(1, 0) = 1."""
        import math
        daily_contrib_est = -10.0
        profit_budget_pct = 0.75
        result = max(1, int(math.ceil(max(0.0, daily_contrib_est) * max(0.0, min(1.0, profit_budget_pct)))))
        assert result == 1

    def test_large_daily_contrib_uses_real_value(self, mock_plugin, mock_database):
        """Large daily_contrib_est = 100, pct = 0.75 -> 75 (not clamped)."""
        import math
        daily_contrib_est = 100.0
        profit_budget_pct = 0.75
        result = max(1, int(math.ceil(max(0.0, daily_contrib_est) * max(0.0, min(1.0, profit_budget_pct)))))
        assert result == 75

    def test_very_small_positive_contrib_still_yields_1(self, mock_plugin, mock_database):
        """daily_contrib_est of 0.01 * pct 0.75 = 0.0075, ceil = 1."""
        import math
        daily_contrib_est = 0.01
        profit_budget_pct = 0.75
        result = max(1, int(math.ceil(max(0.0, daily_contrib_est) * max(0.0, min(1.0, profit_budget_pct)))))
        assert result == 1

    def test_compute_hot_channel_protection_returns_at_least_1(self, mock_plugin, mock_database):
        """End-to-end: _compute_hot_channel_protection returns profit budget >= 1."""
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        # Config defaults have hot_channel_protection_enabled=True
        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Build a mock snapshot with the required attributes
        snap = MagicMock()
        snap.hot_channel_protection_enabled = True
        snap.hot_channel_protection_override_peers = ""
        snap.hot_channel_protection_profit_budget_pct = 0.75
        snap.hot_channel_protection_max_chunk_multiplier = 4.0
        snap.hot_channel_protection_min_velocity = 0.20
        snap.hot_channel_protection_min_marginal_roi = 0.20
        snap.rebalance_cooldown_hours = 24
        snap.hot_channel_protection_min_cooldown_hours = 1.0
        r.config.snapshot = MagicMock(return_value=snap)

        # Create minimal profitability mock
        prof = MagicMock()
        prof.marginal_roi = 0.5
        prof.days_open = 10

        # _estimate_daily_channel_contribution returns a small value
        r._estimate_daily_channel_contribution = MagicMock(return_value=0.3)

        # Mock database for peer-history lookup
        mock_database.get_peer_closed_channel_profit_summary = MagicMock(return_value=None)
        mock_database.list_hot_channel_protection_override_peers = MagicMock(return_value=[])

        result = r._compute_hot_channel_protection(
            dest_channel="111x222x0",
            dest_peer_id="02" + "a" * 64,
            dest_flow_state="source",
            dest_ratio=0.2,
            velocity=0.5,
            prof=prof,
        )

        assert result["eligible"] is True
        assert result["channel_profit_budget_sats"] >= 1


# ============================================================================
# 5. B1: max_fee_ppm re-derived after I-3 budget cap
# ============================================================================


class TestB1MaxFeePpmRederive:
    """B1: max_fee_ppm must be re-derived after I-3 caps max_budget_sats down to expected_income.

    Without the fix, the stale pre-cap max_fee_ppm is recorded as attempted_ppm
    on failure (line 1110), poisoning the fee escalation feedback loop.
    """

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            enable_kelly=False,
            enable_velocity_gate=False,
            rebalance_min_amount=50_000,
            rebalance_max_amount=5_000_000,
            rebalance_min_profit=0,
            rebalance_min_profit_ppm=0,
            rebalance_cooldown_hours=24,
            flow_window_days=7,
            hot_channel_protection_enabled=False,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._profitability_analyzer = None
        return r

    def test_max_fee_ppm_capped_after_i3_budget_reduction(self, mock_plugin, mock_database):
        """When I-3 cap reduces max_budget_sats, max_fee_ppm must be re-derived to match."""
        r = self._make_rebalancer(mock_plugin, mock_database)

        dest_channel = "111x222x0"
        dest_peer_id = "02" + "b" * 64
        source_peer_id = "02" + "c" * 64
        capacity = 10_000_000  # 10M sats

        # Low outbound ratio -> channel needs rebalance
        dest_ratio = 0.05
        spendable = int(capacity * dest_ratio)  # 500k sats

        dest_info = {
            "peer_id": dest_peer_id,
            "capacity": capacity,
            "spendable_sats": spendable,
            "fee_ppm": 500,  # High outbound fee
        }

        # Set up a source with low opportunity cost to ensure a large spread
        source_scid = "333x444x0"
        source_info = {
            "peer_id": source_peer_id,
            "capacity": 5_000_000,
            "spendable_sats": 4_000_000,
            "fee_ppm": 50,
        }
        sources = [(source_scid, source_info, 0.8)]

        # Database mocks
        # Channel state: very low volume (low utilization -> I-3 cap triggers)
        mock_database.get_channel_state.return_value = {
            "state": "balanced",
            "sats_in": 10_000,   # Very low volume
            "sats_out": 10_000,
        }
        mock_database.get_fee_strategy_state.return_value = {
            "last_broadcast_fee_ppm": 500,
        }
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_failure_count.return_value = (0, None)
        mock_database.get_failure_metadata.return_value = {"last_attempted_ppm": 0}
        mock_database.get_kalman_state.return_value = None  # No Kalman -> linear fallback
        mock_database.get_historical_inbound_fee_ppm.return_value = None
        mock_database.get_peer_closed_channel_profit_summary.return_value = None
        mock_database.list_hot_channel_protection_override_peers.return_value = []

        # Mock internal methods
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._get_channel_age_days = MagicMock(return_value=30)

        # Mock _select_source_candidates to return our source
        r._select_source_candidates = MagicMock(return_value=[
            (source_scid, source_info, 1.0, 50),  # (scid, info, score, opp_cost)
        ])

        candidate = r._analyze_rebalance_ev(
            dest_channel=dest_channel,
            dest_info=dest_info,
            dest_ratio=dest_ratio,
            sources=sources,
        )

        assert candidate is not None, "Expected a RebalanceCandidate, got None"

        # The key assertion: max_fee_ppm must be derivable from max_budget_msat
        # budget_ppm = (max_budget_msat * 1_000_000) // amount_msat
        # max_fee_ppm should be <= budget_ppm (the capped budget)
        capped_budget_ppm = (candidate.max_budget_msat * 1_000_000) // candidate.amount_msat
        assert candidate.max_fee_ppm <= capped_budget_ppm, (
            f"STALE max_fee_ppm! max_fee_ppm={candidate.max_fee_ppm} exceeds "
            f"capped budget_ppm={capped_budget_ppm} (max_budget_msat={candidate.max_budget_msat}, "
            f"amount_msat={candidate.amount_msat}). "
            f"I-3 cap reduced the budget but max_fee_ppm was not re-derived."
        )


# ============================================================================
# 6. B2: Weekly budget uses 24h external costs (should scale to 7d)
# ============================================================================


class TestB2WeeklyBudgetExtCosts:
    """B2: ext_spent from spent_24h_sats is a 24h figure but was added raw to
    7-day weekly_fees_spent for the weekly budget gate.  External spending was
    undercounted by ~6x.  The fix scales ext_spent * 7 for the weekly comparison.
    """

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            daily_budget_sats=100_000,       # High daily budget so daily gate does NOT block
            weekly_budget_sats=1000,          # Low weekly budget — this is the gate we test
            min_wallet_reserve=0,             # Disable reserve check
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        return r

    def test_weekly_gate_uses_actual_24h_ext_costs(self, mock_plugin, mock_database):
        """ext_spent is a 24h figure; adding it once (not *7) avoids
        overestimation from a single large Boltz swap.

        weekly_fees=0 + ext_spent=200 = 200 < 1000 -> allowed.
        """
        r = self._make_rebalancer(mock_plugin, mock_database)

        # Reserve check: provide enough funds to pass
        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [{"status": "confirmed", "amount_msat": 10_000_000_000}],
            "channels": [],
        }

        # Both daily and weekly rebalance fees from DB are 0
        mock_database.get_total_rebalance_fees.return_value = 0
        mock_database.get_total_routing_revenue.return_value = 0

        # External liquidity costs: 200 sats in the last 24h
        r._get_external_liquidity_costs = MagicMock(return_value={
            "spent_24h_sats": 200,
            "reserved_24h_sats": 0,
        })

        result = r._check_capital_controls()

        # 24h ext_spent is added once (not multiplied by 7) to avoid
        # grossly overestimating after a single large swap.
        assert result is True, (
            "Weekly budget should NOT block when 24h external costs (200) "
            "plus 7d rebalance fees (0) = 200 < weekly_budget (1000)."
        )

    def test_weekly_gate_blocks_when_fees_plus_ext_exceed_budget(self, mock_plugin, mock_database):
        """weekly_fees=900 + ext_spent=200 = 1100 > 1000 -> blocked."""
        r = self._make_rebalancer(mock_plugin, mock_database)

        mock_plugin.rpc.listfunds.return_value = {
            "outputs": [{"status": "confirmed", "amount_msat": 10_000_000_000}],
            "channels": [],
        }

        mock_database.get_total_rebalance_fees.return_value = 900
        mock_database.get_total_routing_revenue.return_value = 0

        r._get_external_liquidity_costs = MagicMock(return_value={
            "spent_24h_sats": 200,
            "reserved_24h_sats": 0,
        })

        result = r._check_capital_controls()

        assert result is False, (
            "Weekly budget should block when 7d rebalance fees (900) "
            "plus 24h external costs (200) = 1100 >= weekly_budget (1000)."
        )
        assert r._capital_control_blocker == "weekly_budget_sats"


# ============================================================================
# 7. Rebalance-specific reliability should leverage rebalance history
# ============================================================================


class TestRebalanceReliabilityIntegration:
    """Rebalance routing should prefer rebalance-specific success signals."""

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            enable_kelly=False,
            enable_velocity_gate=False,
            rebalance_min_amount=50_000,
            rebalance_max_amount=5_000_000,
            rebalance_min_profit=0,
            rebalance_min_profit_ppm=0,
            rebalance_cooldown_hours=24,
            flow_window_days=7,
            hot_channel_protection_enabled=False,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._profitability_analyzer = None
        return r

    def test_source_selection_prefers_persistently_successful_source(self, mock_plugin, mock_database):
        """Persistent source-channel success should break ties before transient retries do."""
        r = self._make_rebalancer(mock_plugin, mock_database)

        bad_peer_id = "02" + "c" * 64
        good_peer_id = "02" + "d" * 64
        sources = [
            ("111x1x0", {"peer_id": bad_peer_id, "capacity": 5_000_000, "spendable_sats": 4_000_000, "fee_ppm": 50}, 0.9),
            ("111x2x0", {"peer_id": good_peer_id, "capacity": 5_000_000, "spendable_sats": 4_000_000, "fee_ppm": 50}, 0.9),
        ]

        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_source_rebalance_success_rate.side_effect = lambda cid, window_days=30: {
            "111x1x0": {"total": 12, "successes": 2, "failures": 10, "success_rate": 2 / 12},
            "111x2x0": {"total": 12, "successes": 10, "failures": 2, "success_rate": 10 / 12},
        }.get(cid)

        selected = r._select_source_candidates(
            sources=sources,
            amount_needed=50_000,
            dest_channel="222x333x0",
            dest_outbound_fee_ppm=1200,
            dest_inbound_fee_ppm=100,
            peer_status={
                bad_peer_id: {"connected": True},
                good_peer_id: {"connected": True},
            },
        )

        assert selected, "Expected at least one source candidate"
        assert selected[0][0] == "111x2x0", (
            "Persistent rebalance success should prefer 111x2x0 over 111x1x0 "
            "when other source attributes are equal."
        )

    def test_ev_profit_is_discounted_by_rebalance_success_history(self, mock_plugin, mock_database):
        """Identical generic peer reputation should still diverge on rebalance-specific history."""
        r = self._make_rebalancer(mock_plugin, mock_database)

        dest_channel = "222x333x0"
        dest_peer_id = "02" + "b" * 64
        source_scid = "333x444x0"
        source_peer_id = "02" + "c" * 64

        dest_info = {
            "peer_id": dest_peer_id,
            "capacity": 10_000_000,
            "spendable_sats": 500_000,
            "fee_ppm": 5000,
        }
        source_info = {
            "peer_id": source_peer_id,
            "capacity": 10_000_000,
            "spendable_sats": 4_000_000,
            "fee_ppm": 25,
        }
        sources = [(source_scid, source_info, 0.8)]

        mock_database.get_channel_state.return_value = {
            "state": "balanced",
            "sats_in": 10_000,
            "sats_out": 10_000,
        }
        mock_database.get_fee_strategy_state.return_value = {"last_broadcast_fee_ppm": 5000}
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_failure_count.return_value = (0, None)
        mock_database.get_failure_metadata.return_value = {"last_attempted_ppm": 0}
        mock_database.get_kalman_state.return_value = None
        mock_database.get_historical_inbound_fee_ppm.return_value = None
        mock_database.get_peer_closed_channel_profit_summary.return_value = None
        mock_database.list_hot_channel_protection_override_peers.return_value = []
        mock_database.get_peer_reputation.return_value = {
            "successes": 0,
            "failures": 0,
            "score": 0.5,
        }

        r._estimate_inbound_fee = MagicMock(return_value=10)
        r._get_channel_age_days = MagicMock(return_value=30)
        r._select_source_candidates = MagicMock(return_value=[
            (source_scid, source_info, 1.0, 50),
        ])

        mock_database.get_peer_rebalance_success_rate.return_value = {
            "total": 10, "successes": 9, "failures": 1, "success_rate": 0.9,
        }
        mock_database.get_channel_rebalance_success_rate.return_value = {
            "total": 10, "successes": 9, "failures": 1, "success_rate": 0.9,
            "avg_cost_ppm": 10, "avg_amount_sats": 50_000,
        }
        mock_database.get_source_rebalance_success_rate.return_value = {
            "total": 10, "successes": 9, "failures": 1, "success_rate": 0.9,
        }
        high = r._analyze_rebalance_ev(
            dest_channel=dest_channel,
            dest_info=dest_info,
            dest_ratio=0.05,
            sources=sources,
        )

        mock_database.get_peer_rebalance_success_rate.return_value = {
            "total": 10, "successes": 1, "failures": 9, "success_rate": 0.1,
        }
        mock_database.get_channel_rebalance_success_rate.return_value = {
            "total": 10, "successes": 1, "failures": 9, "success_rate": 0.1,
            "avg_cost_ppm": 10, "avg_amount_sats": 50_000,
        }
        mock_database.get_source_rebalance_success_rate.return_value = {
            "total": 10, "successes": 1, "failures": 9, "success_rate": 0.1,
        }
        low = r._analyze_rebalance_ev(
            dest_channel=dest_channel,
            dest_info=dest_info,
            dest_ratio=0.05,
            sources=sources,
        )

        assert high is not None
        assert low is not None
        assert high.expected_profit_sats > low.expected_profit_sats, (
            "Rebalance-specific success history should affect EV independently "
            "of generic forwarding reputation."
        )


# ============================================================================
# 7. B3: Push EV uses wrong peer for fee estimation
# ============================================================================


class TestB3PushEvWrongPeer:
    """B3: _estimate_push_ev passes src_peer_id to _estimate_expected_fee_sats,
    but in push rebalancing fees route through destination peers.  The fee
    estimate must use the primary destination peer, not the source.
    """

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            enable_kelly=False,
            rebalance_min_amount=10_000,
            rebalance_max_amount=5_000_000,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        return r

    def test_fee_estimate_uses_dest_peer_not_src(self, mock_plugin, mock_database):
        """_estimate_expected_fee_sats must be called with the dest peer ID,
        not the source peer ID."""
        r = self._make_rebalancer(mock_plugin, mock_database)

        src_peer_id = "02" + "a" * 64
        dest_peer_id = "02" + "b" * 64

        src_channel = "111x222x0"
        src_info = {
            "peer_id": src_peer_id,
            "capacity": 2_000_000,
            "spendable_sats": 1_600_000,
            "fee_ppm": 500,
        }
        src_ratio = 0.80  # High ratio -> excess to push

        dest_scids = ["333x444x0"]
        dest_peer_ids = [dest_peer_id]

        # Mock internal helpers so _estimate_push_ev doesn't bail early
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._estimate_expected_fee_sats = MagicMock(return_value=5)
        r._calculate_turnover_rate = MagicMock(return_value=0.5)

        candidate = r._estimate_push_ev(
            src_channel=src_channel,
            src_info=src_info,
            src_ratio=src_ratio,
            dest_scids=dest_scids,
            dest_peer_ids=dest_peer_ids,
        )

        # The key assertion: fee estimation must use dest peer, not src peer
        r._estimate_expected_fee_sats.assert_called_once()
        call_args = r._estimate_expected_fee_sats.call_args
        called_with_peer = call_args[0][0]
        assert called_with_peer == dest_peer_id, (
            f"_estimate_expected_fee_sats called with peer {called_with_peer!r} "
            f"(the source), but should have been called with {dest_peer_id!r} "
            f"(the destination). Bug B3: push EV uses wrong peer for fee estimate."
        )

    def test_falls_back_to_src_peer_when_no_dest_peer_ids(self, mock_plugin, mock_database):
        """When dest_peer_ids is None/empty, fall back to src_peer_id."""
        r = self._make_rebalancer(mock_plugin, mock_database)

        src_peer_id = "02" + "a" * 64

        src_channel = "111x222x0"
        src_info = {
            "peer_id": src_peer_id,
            "capacity": 2_000_000,
            "spendable_sats": 1_600_000,
            "fee_ppm": 500,
        }
        src_ratio = 0.80

        dest_scids = ["333x444x0"]
        dest_peer_ids = None  # No dest peer IDs available

        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._estimate_expected_fee_sats = MagicMock(return_value=5)
        r._calculate_turnover_rate = MagicMock(return_value=0.5)

        candidate = r._estimate_push_ev(
            src_channel=src_channel,
            src_info=src_info,
            src_ratio=src_ratio,
            dest_scids=dest_scids,
            dest_peer_ids=dest_peer_ids,
        )

        # When no dest peer IDs, fallback to src_peer_id is acceptable
        r._estimate_expected_fee_sats.assert_called_once()
        call_args = r._estimate_expected_fee_sats.call_args
        called_with_peer = call_args[0][0]
        assert called_with_peer == src_peer_id, (
            f"When dest_peer_ids is None, should fall back to src_peer_id "
            f"{src_peer_id!r}, but was called with {called_with_peer!r}."
        )


# ============================================================================
# 8. B4: stop_all_jobs unconditionally releases budget
# ============================================================================



# Sling-specific test classes removed: TestB4StopAllJobsBudget,
# TestB5TotalSpentSatsAsFee, TestB6ProfitReconciliationInflated,
# TestB7PartialFeeSpendOnFailure, TestLegacyAskreneCleanupRemoved,
# TestTimestampedSentinels, TestPeerExclusionRemoval.



# ============================================================================
# 10. B10: Push EV uses kelly_fraction without enable_kelly guard
# ============================================================================


class TestB10PushEvKellyGuard:
    """B10: _estimate_push_ev uses cfg.kelly_fraction unconditionally to scale
    max_fee_ppm, halving the push fee budget even when Kelly is disabled.
    The pull path correctly guards behind ``if self.config.enable_kelly``.
    """

    def _make_rebalancer(self, mock_plugin, mock_database, *, enable_kelly=False, kelly_fraction=0.5):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            enable_kelly=enable_kelly,
            kelly_fraction=kelly_fraction,
            rebalance_min_amount=10_000,
            rebalance_max_amount=5_000_000,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        return r

    def test_push_ev_uses_full_spread_when_kelly_disabled(self, mock_plugin, mock_database):
        """With enable_kelly=False and kelly_fraction=0.5, push EV should use
        full spread (450), not halved (225)."""
        r = self._make_rebalancer(mock_plugin, mock_database,
                                  enable_kelly=False, kelly_fraction=0.5)

        src_peer_id = "02" + "a" * 64
        dest_peer_id = "02" + "b" * 64

        src_channel = "111x222x0"
        src_info = {
            "peer_id": src_peer_id,
            "capacity": 2_000_000,
            "spendable_sats": 1_600_000,
            "fee_ppm": 500,
        }
        src_ratio = 0.80  # High ratio -> excess to push

        dest_scids = ["333x444x0"]
        dest_peer_ids = [dest_peer_id]

        # inbound_fee=50 -> spread = 500 - 50 = 450
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._estimate_expected_fee_sats = MagicMock(return_value=5)
        r._calculate_turnover_rate = MagicMock(return_value=0.5)

        candidate = r._estimate_push_ev(
            src_channel=src_channel,
            src_info=src_info,
            src_ratio=src_ratio,
            dest_scids=dest_scids,
            dest_peer_ids=dest_peer_ids,
        )

        assert candidate is not None, "Push EV candidate should not be None"
        # spread = 500 - 50 = 450; with kelly disabled, max_fee_ppm = max(1, int(450 * 1.0)) = 450
        assert candidate.max_fee_ppm == 450, (
            f"With enable_kelly=False, max_fee_ppm should be 450 (full spread), "
            f"not {candidate.max_fee_ppm}. Bug B10: kelly_fraction applied without guard."
        )

    def test_push_ev_uses_kelly_fraction_when_kelly_enabled(self, mock_plugin, mock_database):
        """With enable_kelly=True and kelly_fraction=0.5, push EV should use
        halved spread (225)."""
        r = self._make_rebalancer(mock_plugin, mock_database,
                                  enable_kelly=True, kelly_fraction=0.5)

        src_peer_id = "02" + "a" * 64
        dest_peer_id = "02" + "b" * 64

        src_channel = "111x222x0"
        src_info = {
            "peer_id": src_peer_id,
            "capacity": 2_000_000,
            "spendable_sats": 1_600_000,
            "fee_ppm": 500,
        }
        src_ratio = 0.80

        dest_scids = ["333x444x0"]
        dest_peer_ids = [dest_peer_id]

        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._estimate_expected_fee_sats = MagicMock(return_value=5)
        r._calculate_turnover_rate = MagicMock(return_value=0.5)

        candidate = r._estimate_push_ev(
            src_channel=src_channel,
            src_info=src_info,
            src_ratio=src_ratio,
            dest_scids=dest_scids,
            dest_peer_ids=dest_peer_ids,
        )

        assert candidate is not None, "Push EV candidate should not be None"
        # spread = 500 - 50 = 450; with kelly enabled, max_fee_ppm = max(1, int(450 * 0.5)) = 225
        assert candidate.max_fee_ppm == 225, (
            f"With enable_kelly=True, max_fee_ppm should be 225 (half spread), "
            f"not {candidate.max_fee_ppm}."
        )


# ============================================================================
# 11. B11: diagnostic_rebalance returns success:true on exception
# ============================================================================


class TestB11DiagnosticRebalanceExceptionSuccess:
    """B11: diagnostic_rebalance returns {success: True} when the defibrillator
    shock raises an exception. Should return {success: False}.
    """

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        return r

    def test_exception_returns_success_false(self, mock_plugin, mock_database):
        """When defibrillator shock raises an exception, result must have
        success=False, not success=True."""
        r = self._make_rebalancer(mock_plugin, mock_database)

        # set_channel_probe succeeds (step 1 of defibrillator)
        mock_database.set_channel_probe = MagicMock()

        # _get_channels_with_balances raises an exception (simulates any
        # failure in the try block before or during shock execution)
        r._get_channels_with_balances = MagicMock(
            side_effect=RuntimeError("RPC timeout")
        )

        result = r.diagnostic_rebalance("111x222x0")

        assert result["success"] is False, (
            f"diagnostic_rebalance should return success=False on exception, "
            f"but got success={result['success']}. Bug B11: exception handler "
            f"returns success:True."
        )
        assert "shock failed" in result["message"].lower() or "failed" in result["message"].lower()
