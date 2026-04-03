"""
Regression tests for cl-revenue-ops rebalancer audit fixes.

Covers:
  C-2: Sentinel cleanup in monitor_jobs
  C-3: Stop job re-add on sling failure
  C-4: Channel close detection in monitor_jobs
  I-12: Hot channel profit budget floor
  B2: Weekly budget uses 24h external costs (should scale to 7d)
  B3: Push EV uses wrong peer for fee estimation
  B4: stop_all_jobs unconditionally releases budget for already-handled jobs
  B10: Push EV uses kelly_fraction without enable_kelly guard
  B11: diagnostic_rebalance returns success:true on exception
"""

import math
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock

from pyln.client import RpcError


# In test context, conftest patches pyln.client so RpcError == Exception.
# This is problematic because `except RpcError` then catches *everything*,
# making it impossible to test the C-3 path where a non-RpcError (e.g.
# TimeoutError) reaches the outer except.  Define a narrow subclass and
# monkey-patch it into the module under test so the inner `except RpcError`
# no longer swallows TimeoutError.
class _TestRpcError(Exception):
    """A narrow RpcError substitute so except-RpcError only catches itself."""
    pass


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


def _active_job(scid="222x333x0", amount_sats=50000, rebalance_id=1,
                initial_local_sats=100_000, direction="pull", **kw):
    from modules.rebalancer import ActiveJob, JobStatus

    cand = kw.pop("candidate", None) or _candidate(to_channel=scid, amount_sats=amount_sats)
    return ActiveJob(
        scid=scid,
        scid_normalized=scid.replace(":", "x"),
        source_candidates=cand.source_candidates,
        start_time=kw.pop("start_time", int(time.time())),
        candidate=cand,
        rebalance_id=rebalance_id,
        target_amount_sats=amount_sats,
        initial_local_sats=initial_local_sats,
        max_fee_ppm=2000,
        status=JobStatus.RUNNING,
        direction=direction,
    )


# ============================================================================
# 1. Sentinel Cleanup (C-2 fix)
# ============================================================================


class TestSentinelCleanup:
    """C-2: Stale None sentinels in _active_jobs are cleaned up by monitor_jobs."""

    def test_stale_none_sentinels_cleaned_up(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Insert a None sentinel (simulates stuck start_job)
        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = None
            jm._active_jobs["333x444x0"] = None

        # Also insert a real job to verify it is NOT cleaned
        real_job = _active_job(scid="555x666x0")
        with jm._jobs_lock:
            jm._active_jobs["555x666x0"] = real_job

        # Mock listfunds to return the real channel
        mock_plugin.rpc.listfunds.return_value = {
            "channels": [
                {
                    "short_channel_id": "555x666x0",
                    "our_amount_msat": 100_000_000,
                    "state": "CHANNELD_NORMAL",
                }
            ]
        }
        # Mock sling-stats
        mock_plugin.rpc.call.return_value = {"stats": {}}

        # Suppress _handle_job_failure / stop_job side-effects for the real job
        jm._handle_job_failure = MagicMock()
        jm._handle_job_timeout = MagicMock()
        jm._handle_job_success = MagicMock()

        summary = jm.monitor_jobs()

        # Sentinels should be gone
        with jm._jobs_lock:
            assert "111x222x0" not in jm._active_jobs
            assert "333x444x0" not in jm._active_jobs
            # Real job should still be tracked (running)
            assert "555x666x0" in jm._active_jobs

    def test_sentinel_cleanup_with_no_real_jobs(self, mock_plugin, mock_database):
        """When only sentinels exist, monitor_jobs returns early after cleanup."""
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = None

        summary = jm.monitor_jobs()

        with jm._jobs_lock:
            assert "111x222x0" not in jm._active_jobs
        # With no real jobs, checked count should be 0
        assert summary["checked"] == 0


# ============================================================================
# 2. Channel Close Detection (C-4 fix)
# ============================================================================


class TestChannelCloseDetection:
    """C-4: Jobs are terminated when channels disappear from _get_local_balances_map."""

    def test_job_terminated_when_channel_missing_from_balances(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job = _active_job(scid="222x333x0", initial_local_sats=100_000)
        with jm._jobs_lock:
            jm._active_jobs["222x333x0"] = job

        # Channel is NOT in listfunds (simulating channel closure)
        mock_plugin.rpc.listfunds.return_value = {"channels": []}
        # sling-stats returns nothing for this channel
        mock_plugin.rpc.call.return_value = {"stats": {}}

        jm._handle_job_failure = MagicMock()
        jm.stop_job = MagicMock(return_value=True)

        summary = jm.monitor_jobs()

        # Job should be marked as failed
        assert summary["failed"] == 1
        jm._handle_job_failure.assert_called_once_with(job, {})

    def test_get_local_balances_map_excludes_non_normal_channels(self, mock_plugin, mock_database):
        """_get_local_balances_map only includes CHANNELD_NORMAL channels."""
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        mock_plugin.rpc.listfunds.return_value = {
            "channels": [
                {
                    "short_channel_id": "111x222x0",
                    "our_amount_msat": 500_000_000,
                    "state": "CHANNELD_NORMAL",
                },
                {
                    "short_channel_id": "222x333x0",
                    "our_amount_msat": 300_000_000,
                    "state": "CHANNELD_AWAITING_LOCKIN",
                },
                {
                    "short_channel_id": "333x444x0",
                    "our_amount_msat": 200_000_000,
                    "state": "ONCHAIN",
                },
                {
                    "short_channel_id": "444x555x0",
                    "our_amount_msat": 100_000_000,
                    # No state field (old CLN / test mock) -> should be included
                },
            ]
        }

        balances = jm._get_local_balances_map()
        assert "111x222x0" in balances
        assert balances["111x222x0"] == 500_000  # 500M msat / 1000
        assert "222x333x0" not in balances  # AWAITING_LOCKIN excluded
        assert "333x444x0" not in balances  # ONCHAIN excluded
        assert "444x555x0" in balances  # No state -> included
        assert balances["444x555x0"] == 100_000


# ============================================================================
# 3. Stop Job Re-add on Failure (C-3 fix)
# ============================================================================


class TestStopJobReAddOnFailure:
    """C-3: stop_job re-adds job to tracking when sling is unreachable."""

    def test_stop_job_readds_when_sling_unreachable(self, mock_plugin, mock_database):
        """When sling-stop throws a non-RpcError, the job is re-added to _active_jobs."""
        import modules.rebalancer as _mod
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)

        # Patch RpcError in the module so the inner `except RpcError` only
        # catches the narrow subclass, not TimeoutError.
        original_rpc_error = _mod.RpcError
        _mod.RpcError = _TestRpcError
        try:
            jm = JobManager(mock_plugin, cfg, mock_database)

            job = _active_job(scid="222x333x0")
            with jm._jobs_lock:
                jm._active_jobs["222x333x0"] = job

            # sling-stop raises a timeout (non-RpcError)
            def call_side_effect(method, params=None):
                if method == "sling-stop":
                    raise TimeoutError("Connection timed out")
                return {}

            mock_plugin.rpc.call.side_effect = call_side_effect

            result = jm.stop_job("222x333x0", reason="test")
            assert result is False

            # Job should be re-added to _active_jobs since sling was unreachable
            with jm._jobs_lock:
                assert "222x333x0" in jm._active_jobs
                assert jm._active_jobs["222x333x0"] is job
        finally:
            _mod.RpcError = original_rpc_error

    def test_stop_job_does_not_readd_on_success(self, mock_plugin, mock_database):
        """When sling-stop succeeds, the job is NOT re-added."""
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job = _active_job(scid="222x333x0")
        with jm._jobs_lock:
            jm._active_jobs["222x333x0"] = job

        # sling-stop and sling-deletejob both succeed
        mock_plugin.rpc.call.return_value = {}

        result = jm.stop_job("222x333x0", reason="test")
        assert result is True

        # Job should NOT be in _active_jobs
        with jm._jobs_lock:
            assert "222x333x0" not in jm._active_jobs

    def test_stop_job_does_not_readd_on_rpc_error(self, mock_plugin, mock_database):
        """When sling-stop raises RpcError (sling responded), job is NOT re-added."""
        import modules.rebalancer as _mod
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)

        # Patch RpcError in the module so only our narrow subclass is caught.
        original_rpc_error = _mod.RpcError
        _mod.RpcError = _TestRpcError
        try:
            jm = JobManager(mock_plugin, cfg, mock_database)

            job = _active_job(scid="222x333x0")
            with jm._jobs_lock:
                jm._active_jobs["222x333x0"] = job

            # sling-stop raises the narrow RpcError (sling did respond, job may already be stopped)
            def call_side_effect(method, params=None):
                if method == "sling-stop":
                    raise _TestRpcError("job not found")
                return {}

            mock_plugin.rpc.call.side_effect = call_side_effect

            result = jm.stop_job("222x333x0", reason="test")
            assert result is True

            # Job should NOT be re-added (RpcError means sling responded)
            with jm._jobs_lock:
                assert "222x333x0" not in jm._active_jobs
        finally:
            _mod.RpcError = original_rpc_error


# ============================================================================
# 4. Hot Channel Profit Budget Floor (I-12 fix)
# ============================================================================


class TestHotChannelProfitBudgetFloor:
    """I-12: channel_profit_budget_sats is at least 1 for small positive daily_contrib_est."""

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True, enable_proportional_budget=False)
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
        cfg = Config(dry_run=True, enable_proportional_budget=False)
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
            enable_proportional_budget=False,
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
            enable_proportional_budget=False,
            daily_budget_sats=100_000,       # High daily budget so daily gate does NOT block
            weekly_budget_sats=1000,          # Low weekly budget — this is the gate we test
            min_wallet_reserve=0,             # Disable reserve check
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        return r

    def test_weekly_gate_blocks_when_ext_costs_scaled_to_7d(self, mock_plugin, mock_database):
        """spent_24h_sats=200 -> 200*7=1400 > weekly_budget=1000 -> blocked.

        Without the fix: 0 + 200 = 200 < 1000 -> wrongly passes.
        With the fix:    0 + 1400 = 1400 > 1000 -> correctly blocked.
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

        # With the fix (ext_spent * 7 = 1400 > 1000), this should be False
        assert result is False, (
            "Weekly budget gate should block when 24h external costs scaled to "
            "7d (200 * 7 = 1400) exceed weekly_budget_sats (1000). "
            "Bug B2: ext_spent was not scaled to 7d for the weekly comparison."
        )
        assert r._capital_control_blocker == "weekly_budget_sats"


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
            enable_proportional_budget=False,
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


class TestB4StopAllJobsBudget:
    """B4: stop_all_jobs must NOT release budget for jobs already handled by
    monitor_jobs (SUCCESS/FAILED/TIMEOUT).  Only PENDING/RUNNING jobs should
    have their reservations released during shutdown.
    """

    def test_only_releases_budget_for_pending_or_running_jobs(self, mock_plugin, mock_database):
        """Create two jobs: one SUCCESS (already handled), one RUNNING.
        stop_all_jobs should only call release_budget_reservation for the RUNNING job.
        """
        from modules.config import Config
        from modules.rebalancer import JobManager, JobStatus

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Job 1: SUCCESS — budget already handled by monitor_jobs
        job_success = _active_job(scid="111x222x0", rebalance_id=100)
        job_success.status = JobStatus.SUCCESS

        # Job 2: RUNNING — budget NOT yet handled, should be released
        job_running = _active_job(scid="333x444x0", rebalance_id=200)
        job_running.status = JobStatus.RUNNING

        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = job_success
            jm._active_jobs["333x444x0"] = job_running

        # stop_job succeeds for both
        mock_plugin.rpc.call.return_value = {}

        jm.stop_all_jobs(reason="shutdown")

        # Collect all calls to release_budget_reservation
        release_calls = mock_database.release_budget_reservation.call_args_list
        released_ids = [c[0][0] for c in release_calls]

        # RUNNING job's reservation should be released
        assert str(job_running.rebalance_id) in released_ids, (
            f"Expected release for RUNNING job (rebalance_id={job_running.rebalance_id}), "
            f"but release_budget_reservation was called with: {released_ids}"
        )

        # SUCCESS job's reservation should NOT be released (already handled)
        assert str(job_success.rebalance_id) not in released_ids, (
            f"SUCCESS job (rebalance_id={job_success.rebalance_id}) should NOT have "
            f"its budget released — it was already handled by monitor_jobs. "
            f"But release_budget_reservation was called with: {released_ids}"
        )

    def test_does_not_release_budget_for_failed_jobs(self, mock_plugin, mock_database):
        """FAILED jobs already had budget handled by monitor_jobs."""
        from modules.config import Config
        from modules.rebalancer import JobManager, JobStatus

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job_failed = _active_job(scid="111x222x0", rebalance_id=300)
        job_failed.status = JobStatus.FAILED

        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = job_failed

        mock_plugin.rpc.call.return_value = {}

        jm.stop_all_jobs(reason="shutdown")

        release_calls = mock_database.release_budget_reservation.call_args_list
        released_ids = [c[0][0] for c in release_calls]

        assert str(job_failed.rebalance_id) not in released_ids, (
            f"FAILED job should NOT have its budget released — already handled. "
            f"But release_budget_reservation was called with: {released_ids}"
        )

    def test_does_not_release_budget_for_timeout_jobs(self, mock_plugin, mock_database):
        """TIMEOUT jobs already had budget handled by _handle_job_timeout."""
        from modules.config import Config
        from modules.rebalancer import JobManager, JobStatus

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job_timeout = _active_job(scid="111x222x0", rebalance_id=400)
        job_timeout.status = JobStatus.TIMEOUT

        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = job_timeout

        mock_plugin.rpc.call.return_value = {}

        jm.stop_all_jobs(reason="shutdown")

        release_calls = mock_database.release_budget_reservation.call_args_list
        released_ids = [c[0][0] for c in release_calls]

        assert str(job_timeout.rebalance_id) not in released_ids, (
            f"TIMEOUT job should NOT have its budget released — already handled. "
            f"But release_budget_reservation was called with: {released_ids}"
        )

    def test_releases_budget_for_pending_jobs(self, mock_plugin, mock_database):
        """PENDING jobs have not been processed yet — budget should be released."""
        from modules.config import Config
        from modules.rebalancer import JobManager, JobStatus

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job_pending = _active_job(scid="111x222x0", rebalance_id=500)
        job_pending.status = JobStatus.PENDING

        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = job_pending

        mock_plugin.rpc.call.return_value = {}

        jm.stop_all_jobs(reason="shutdown")

        release_calls = mock_database.release_budget_reservation.call_args_list
        released_ids = [c[0][0] for c in release_calls]

        assert str(job_pending.rebalance_id) in released_ids, (
            f"PENDING job (rebalance_id={job_pending.rebalance_id}) should have "
            f"its budget released. But release_budget_reservation was called with: {released_ids}"
        )


# ============================================================================
# 9. B5: _handle_job_success total_spent_sats used as fee
# ============================================================================


class TestB5TotalSpentSatsAsFee:
    """B5: Third fallback in _handle_job_success treats total_spent_sats
    (principal + fee) as just fee.  This massively overcounts the fee.
    The fix derives fee as total_spent - amount_transferred.
    """

    def _make_job_manager(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)
        return jm

    def test_total_spent_derives_fee_not_raw(self, mock_plugin, mock_database):
        """When only total_spent_sats is available (50010 for 50000 amount),
        fee should be ~10, not 50010."""
        jm = self._make_job_manager(mock_plugin, mock_database)

        job = _active_job(scid="222x333x0", amount_sats=50000, rebalance_id=42)
        amount_transferred = 50000

        # Stats: no fee_total_sats, no fee_total_msat, only total_spent_sats
        stats = {
            "successes_in_time_window": {
                "total_spent_sats": 50010,  # principal (50000) + fee (10)
            }
        }

        # Stub stop_job to avoid sling RPC
        jm.stop_job = MagicMock(return_value=True)

        jm._handle_job_success(job, amount_transferred, stats)

        # Fee should be 10, not 50010
        call_args = mock_database.update_rebalance_result.call_args
        actual_fee = call_args[0][2]  # positional arg: fee_sats
        assert actual_fee == 10, (
            f"Expected fee_sats=10 (total_spent 50010 - amount 50000), "
            f"got fee_sats={actual_fee}. Bug B5: total_spent_sats treated as fee."
        )

    def test_total_spent_less_than_amount_skips_fallback(self, mock_plugin, mock_database):
        """When total_spent_sats <= amount_transferred, the fallback should not
        set fee_sats (would produce negative or zero fee)."""
        jm = self._make_job_manager(mock_plugin, mock_database)

        job = _active_job(scid="222x333x0", amount_sats=50000, rebalance_id=43)
        amount_transferred = 50000

        # total_spent_sats == amount (no fee component detectable)
        stats = {
            "successes_in_time_window": {
                "total_spent_sats": 50000,
            }
        }

        jm.stop_job = MagicMock(return_value=True)
        jm._handle_job_success(job, amount_transferred, stats)

        # fee_sats should NOT be 50000 (the old bug); it should be the
        # conservative estimate from the fallback estimator
        call_args = mock_database.update_rebalance_result.call_args
        actual_fee = call_args[0][2]
        assert actual_fee != 50000, (
            f"fee_sats should not be {actual_fee} (== amount_transferred). "
            f"total_spent == amount means no fee detectable via this path."
        )


# ============================================================================
# 10. B6: Profit reconciliation inflated for legacy candidates
# ============================================================================


class TestB6ProfitReconciliationInflated:
    """B6: When expected_fee_sats==0 (legacy candidates), the profit
    reconciliation falls back to max_budget_sats as assumed fee, inflating
    actual_profit.  Fix: use fee_sats as fallback so delta is zero.
    """

    def _make_job_manager(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        return JobManager(mock_plugin, cfg, mock_database)

    def test_legacy_candidate_profit_not_inflated(self, mock_plugin, mock_database):
        """With expected_fee_sats=0, max_budget_sats=10, fee_sats=3,
        expected_profit=5:
          Without fix: assumed_fee=10 -> actual_profit=5+(10-3)=12 (INFLATED)
          With fix:    assumed_fee=3  -> actual_profit=5+(3-3)=5  (correct)
        """
        from modules.rebalancer import RebalanceCandidate
        jm = self._make_job_manager(mock_plugin, mock_database)

        cand = _candidate(amount_sats=50000)
        # Override the fields relevant to B6
        cand = RebalanceCandidate(
            source_candidates=cand.source_candidates,
            to_channel=cand.to_channel,
            primary_source_peer_id=cand.primary_source_peer_id,
            to_peer_id=cand.to_peer_id,
            amount_sats=cand.amount_sats,
            amount_msat=cand.amount_msat,
            outbound_fee_ppm=cand.outbound_fee_ppm,
            inbound_fee_ppm=cand.inbound_fee_ppm,
            source_fee_ppm=cand.source_fee_ppm,
            weighted_opp_cost_ppm=cand.weighted_opp_cost_ppm,
            spread_ppm=cand.spread_ppm,
            max_budget_sats=10,
            max_budget_msat=10_000,
            max_fee_ppm=2000,
            expected_profit_sats=5,
            expected_fee_sats=0,  # Legacy: no expected_fee_sats
            liquidity_ratio=0.1,
            dest_flow_state="balanced",
            dest_turnover_rate=0.0,
            source_turnover_rate=0.0,
        )

        job = _active_job(scid="222x333x0", amount_sats=50000, rebalance_id=44,
                          candidate=cand)

        # Stats that produce fee_sats=3 (via fee_total_sats)
        stats = {"fee_total_sats": 3}

        jm.stop_job = MagicMock(return_value=True)
        jm._handle_job_success(job, 50000, stats)

        call_args = mock_database.update_rebalance_result.call_args
        actual_profit = call_args[0][3]  # positional arg: actual_profit
        assert actual_profit == 5, (
            f"Expected actual_profit=5 (no inflation), got {actual_profit}. "
            f"Bug B6: max_budget_sats (10) used as assumed_fee when "
            f"expected_fee_sats==0, inflating profit to {5 + (10 - 3)}=12."
        )


# ============================================================================
# 11. B7: _handle_job_failure ignores partial fee spend
# ============================================================================


class TestB7PartialFeeSpendOnFailure:
    """B7: _handle_job_failure unconditionally calls release_budget_reservation
    without checking for partial fees from failed payment attempts.
    Fix: check for partial fees and call mark_budget_spent if > 0.
    """

    def _make_job_manager(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        return JobManager(mock_plugin, cfg, mock_database)

    def test_failure_with_partial_fee_calls_mark_budget_spent(self, mock_plugin, mock_database):
        """Failure with fee_total_msat: '5000msat' -> should call
        mark_budget_spent('42', 5), NOT release_budget_reservation."""
        jm = self._make_job_manager(mock_plugin, mock_database)

        job = _active_job(scid="222x333x0", amount_sats=50000, rebalance_id=42)

        stats = {
            "last_error": "WIRE_TEMPORARY_CHANNEL_FAILURE",
            "fee_total_msat": "5000msat",
        }

        jm.stop_job = MagicMock(return_value=True)
        jm._handle_job_failure(job, stats)

        # Should call mark_budget_spent, NOT release_budget_reservation
        mock_database.mark_budget_spent.assert_called_once_with("42", 5)
        mock_database.release_budget_reservation.assert_not_called()

    def test_failure_with_no_fees_calls_release(self, mock_plugin, mock_database):
        """Failure with no fee info -> should call release_budget_reservation."""
        jm = self._make_job_manager(mock_plugin, mock_database)

        job = _active_job(scid="222x333x0", amount_sats=50000, rebalance_id=43)

        stats = {
            "last_error": "WIRE_TEMPORARY_CHANNEL_FAILURE",
        }

        jm.stop_job = MagicMock(return_value=True)
        jm._handle_job_failure(job, stats)

        # No partial fee -> should release the reservation
        mock_database.release_budget_reservation.assert_called_once_with("43")
        mock_database.mark_budget_spent.assert_not_called()

    def test_failure_with_fee_total_sats_calls_mark_budget_spent(self, mock_plugin, mock_database):
        """Failure with fee_total_sats (no msat) -> should call mark_budget_spent."""
        jm = self._make_job_manager(mock_plugin, mock_database)

        job = _active_job(scid="222x333x0", amount_sats=50000, rebalance_id=44)

        stats = {
            "last_error": "WIRE_TEMPORARY_CHANNEL_FAILURE",
            "fee_total_sats": 8,
        }

        jm.stop_job = MagicMock(return_value=True)
        jm._handle_job_failure(job, stats)

        mock_database.mark_budget_spent.assert_called_once_with("44", 8)
        mock_database.release_budget_reservation.assert_not_called()


# ============================================================================
# B8: Sentinel cleanup deletes all sentinels (no age check)
# ============================================================================


class TestTimestampedSentinels:
    """B8: Sentinels should use timestamps, fresh ones should survive monitor_jobs."""

    def test_fresh_sentinel_not_cleaned(self, mock_plugin, mock_database):
        """A sentinel placed moments ago (fresh timestamp) must survive monitor_jobs."""
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Insert a *fresh* timestamp sentinel (just now)
        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = time.time()

        # Mock sling-stats (no jobs to report)
        mock_plugin.rpc.call.return_value = {"stats": {}}
        mock_plugin.rpc.listfunds.return_value = {"channels": []}

        jm._handle_job_failure = MagicMock()
        jm._handle_job_timeout = MagicMock()
        jm._handle_job_success = MagicMock()

        jm.monitor_jobs()

        # Fresh sentinel must NOT be cleaned up
        with jm._jobs_lock:
            assert "111x222x0" in jm._active_jobs, \
                "Fresh sentinel was incorrectly cleaned up by monitor_jobs"

    def test_stale_sentinel_cleaned(self, mock_plugin, mock_database):
        """A sentinel older than sentinel_timeout (300s) must be cleaned up."""
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Insert a *stale* timestamp sentinel (10 minutes old)
        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = time.time() - 600

        # Mock sling-stats (no jobs to report)
        mock_plugin.rpc.call.return_value = {"stats": {}}
        mock_plugin.rpc.listfunds.return_value = {"channels": []}

        jm._handle_job_failure = MagicMock()
        jm._handle_job_timeout = MagicMock()
        jm._handle_job_success = MagicMock()

        jm.monitor_jobs()

        # Stale sentinel MUST be cleaned up
        with jm._jobs_lock:
            assert "111x222x0" not in jm._active_jobs, \
                "Stale sentinel was not cleaned up by monitor_jobs"


# ============================================================================
# B9: sync_peer_exclusions only adds, never removes
# ============================================================================


class TestPeerExclusionRemoval:
    """B9: sync_peer_exclusions must remove exclusions for re-enabled peers."""

    def test_stale_exclusion_removed_for_reenabled_peer(self, mock_plugin, mock_database):
        """When a peer is no longer disabled, its sling exclusion should be removed."""
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Two peers currently excluded in sling
        peer_still_disabled = "02" + "a" * 64
        peer_reenabled = "02" + "b" * 64

        # Mock sling-except-peer list to return both peers
        def rpc_side_effect(method, args=None):
            if method == "sling-except-peer":
                if args and args[0] == "list":
                    return [peer_still_disabled, peer_reenabled]
                return {}
            return {"stats": {}}

        mock_plugin.rpc.call.side_effect = rpc_side_effect

        # Policy manager only has one peer disabled
        mock_policy_manager = MagicMock()
        mock_policy = MagicMock()
        mock_policy.peer_id = peer_still_disabled

        from modules.policy_manager import RebalanceMode
        mock_policy.rebalance_mode = RebalanceMode.DISABLED

        mock_policy_manager.get_all_policies.return_value = [mock_policy]

        jm.sync_peer_exclusions(mock_policy_manager)

        # Assert that "remove" was called for the re-enabled peer
        remove_calls = [
            c for c in mock_plugin.rpc.call.call_args_list
            if c[0] == ("sling-except-peer",) and c[1].get("args", c[0][1] if len(c[0]) > 1 else None) is not None
        ]
        # Find the remove call specifically
        found_remove = False
        for c in mock_plugin.rpc.call.call_args_list:
            args_positional = c[0]
            if len(args_positional) >= 2 and args_positional[0] == "sling-except-peer":
                if isinstance(args_positional[1], list) and len(args_positional[1]) >= 2:
                    if args_positional[1][0] == "remove" and args_positional[1][1] == peer_reenabled:
                        found_remove = True

        assert found_remove, (
            f"Expected sling-except-peer remove call for {peer_reenabled[:16]}..., "
            f"but got calls: {mock_plugin.rpc.call.call_args_list}"
        )


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
            enable_proportional_budget=False,
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
            enable_proportional_budget=False,
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
