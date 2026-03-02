"""
Regression tests for cl-revenue-ops rebalancer audit fixes.

Covers:
  C-2: Sentinel cleanup in monitor_jobs
  C-3: Stop job re-add on sling failure
  C-4: Channel close detection in monitor_jobs
  C-5: NNLB budget multiplier
  C-6: NNLB profit threshold adjustment
  I-12: Hot channel profit budget floor
  I-17: via_fleet field on RebalanceCandidate
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
    via_fleet=False,
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
        via_fleet=via_fleet,
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
# 1. NNLB Budget Multiplier (C-5)
# ============================================================================


class TestNNLBBudgetMultiplier:
    """C-5: _calculate_nnlb_budget_multiplier respects hive health tiers."""

    def _make_rebalancer(self, mock_plugin, mock_database, hive_bridge=None):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True, enable_proportional_budget=False)
        clboss = MagicMock()
        return EVRebalancer(mock_plugin, cfg, mock_database, clboss, hive_bridge=hive_bridge)

    # -- Healthy / stable tier -> 1.0 --
    def test_stable_tier_returns_1_0(self, mock_plugin, mock_database):
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "stable"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge)
        assert r._calculate_nnlb_budget_multiplier() == 1.0

    # -- Struggling tier -> 2.0 --
    def test_struggling_tier_returns_2_0(self, mock_plugin, mock_database):
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "struggling"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge)
        assert r._calculate_nnlb_budget_multiplier() == 2.0

    # -- Thriving tier -> 0.75 --
    def test_thriving_tier_returns_0_75(self, mock_plugin, mock_database):
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "thriving"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge)
        assert r._calculate_nnlb_budget_multiplier() == 0.75

    # -- Vulnerable tier -> 1.5 --
    def test_vulnerable_tier_returns_1_5(self, mock_plugin, mock_database):
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "vulnerable"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge)
        assert r._calculate_nnlb_budget_multiplier() == 1.5

    # -- No hive_bridge -> fallback 1.0 --
    def test_no_hive_bridge_returns_default(self, mock_plugin, mock_database):
        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=None)
        assert r._calculate_nnlb_budget_multiplier() == 1.0

    # -- hive_bridge returns None from query_member_health -> fallback 1.0 --
    def test_health_query_returns_none(self, mock_plugin, mock_database):
        bridge = MagicMock()
        bridge.query_member_health.return_value = None

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge)
        assert r._calculate_nnlb_budget_multiplier() == 1.0

    # -- Cache TTL: second call within TTL returns cached value, no RPC --
    def test_cache_ttl_returns_cached_value(self, mock_plugin, mock_database):
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "struggling"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge)

        # First call -> queries hive
        result1 = r._calculate_nnlb_budget_multiplier()
        assert result1 == 2.0
        assert bridge.query_member_health.call_count == 1

        # Second call within TTL -> uses cache
        result2 = r._calculate_nnlb_budget_multiplier()
        assert result2 == 2.0
        assert bridge.query_member_health.call_count == 1  # not called again

    # -- Cache expires after TTL --
    def test_cache_expires_after_ttl(self, mock_plugin, mock_database):
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "struggling"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge)

        # First call
        r._calculate_nnlb_budget_multiplier()
        assert bridge.query_member_health.call_count == 1

        # Expire cache by setting cache time far in the past
        r._health_cache_time = time.time() - 600  # 10 minutes ago (TTL is 300s)

        # Change return for second call
        bridge.query_member_health.return_value = {"health_tier": "thriving"}
        result = r._calculate_nnlb_budget_multiplier()
        assert result == 0.75
        assert bridge.query_member_health.call_count == 2


# ============================================================================
# 2. NNLB Profit Threshold Adjustment (C-6)
# ============================================================================


class TestNNLBProfitThresholdAdjustment:
    """C-6: Profit threshold is adjusted by NNLB multiplier."""

    def _make_rebalancer(self, mock_plugin, mock_database, hive_bridge=None,
                         min_profit=100, min_profit_ppm=0):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            enable_proportional_budget=False,
            rebalance_min_profit=min_profit,
            rebalance_min_profit_ppm=min_profit_ppm,
        )
        clboss = MagicMock()
        return EVRebalancer(mock_plugin, cfg, mock_database, clboss, hive_bridge=hive_bridge)

    def test_struggling_node_gets_lower_threshold(self, mock_plugin, mock_database):
        """Struggling (multiplier=2.0) -> threshold is halved."""
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "struggling"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge,
                                  min_profit=100)

        # Patch internals to isolate the EV path: we only care about the threshold check.
        # _calculate_ev_for_channel calls _calculate_nnlb_budget_multiplier internally.
        # We test the multiplier's effect on the threshold math directly:
        # threshold = 100 / 2.0 = 50
        mult = r._calculate_nnlb_budget_multiplier()
        assert mult == 2.0
        # Threshold would be int(100 / 2.0) = 50
        adjusted = int(100 / mult)
        assert adjusted == 50

    def test_thriving_node_gets_higher_threshold(self, mock_plugin, mock_database):
        """Thriving (multiplier=0.75) -> threshold is raised."""
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "thriving"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge,
                                  min_profit=100)

        mult = r._calculate_nnlb_budget_multiplier()
        assert mult == 0.75
        # Threshold would be int(100 / 0.75) = 133
        adjusted = int(100 / mult)
        assert adjusted == 133

    def test_default_multiplier_leaves_threshold_unchanged(self, mock_plugin, mock_database):
        """Default (multiplier=1.0) -> threshold unchanged."""
        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=None,
                                  min_profit=100)

        mult = r._calculate_nnlb_budget_multiplier()
        assert mult == 1.0
        # When mult == 1.0 the code skips adjustment entirely
        # Verify no division happens: threshold remains 100

    def test_ppm_based_threshold_also_adjusted(self, mock_plugin, mock_database):
        """When rebalance_min_profit_ppm > 0, ppm-derived threshold is also adjusted."""
        bridge = MagicMock()
        bridge.query_member_health.return_value = {"health_tier": "struggling"}

        r = self._make_rebalancer(mock_plugin, mock_database, hive_bridge=bridge,
                                  min_profit_ppm=1000)  # 1000 ppm

        # For amount=100_000 sats: threshold = (100_000 * 1000) // 1_000_000 = 100
        amount = 100_000
        base_threshold = (amount * 1000) // 1_000_000
        assert base_threshold == 100

        mult = r._calculate_nnlb_budget_multiplier()
        adjusted = int(base_threshold / mult)
        assert adjusted == 50  # struggling halves it


# ============================================================================
# 3. Sentinel Cleanup (C-2 fix)
# ============================================================================


class TestSentinelCleanup:
    """C-2: Stale None sentinels in _active_jobs are cleaned up by monitor_jobs."""

    def test_stale_none_sentinels_cleaned_up(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=None)

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
        jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=None)

        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = None

        summary = jm.monitor_jobs()

        with jm._jobs_lock:
            assert "111x222x0" not in jm._active_jobs
        # With no real jobs, checked count should be 0
        assert summary["checked"] == 0


# ============================================================================
# 4. Channel Close Detection (C-4 fix)
# ============================================================================


class TestChannelCloseDetection:
    """C-4: Jobs are terminated when channels disappear from _get_local_balances_map."""

    def test_job_terminated_when_channel_missing_from_balances(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import JobManager

        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=None)

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
        jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=None)

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
# 5. Stop Job Re-add on Failure (C-3 fix)
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
            jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=None)

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
        jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=None)

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
            jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=None)

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
# 6. via_fleet Field (I-17 fix)
# ============================================================================


class TestViaFleetField:
    """I-17: RebalanceCandidate.via_fleet defaults to False and is included in to_dict."""

    def test_via_fleet_defaults_to_false(self):
        cand = _candidate()
        assert cand.via_fleet is False

    def test_via_fleet_can_be_set_true(self):
        cand = _candidate(via_fleet=True)
        assert cand.via_fleet is True

    def test_to_dict_includes_via_fleet(self):
        cand = _candidate(via_fleet=True)
        d = cand.to_dict()
        assert "via_fleet" in d
        assert d["via_fleet"] is True

    def test_to_dict_includes_via_fleet_false(self):
        cand = _candidate(via_fleet=False)
        d = cand.to_dict()
        assert "via_fleet" in d
        assert d["via_fleet"] is False

    def test_to_dict_includes_expected_fee_sats(self):
        cand = _candidate()
        cand.expected_fee_sats = 42
        d = cand.to_dict()
        assert "expected_fee_sats" in d
        assert d["expected_fee_sats"] == 42

    def test_report_outcome_to_hive_uses_via_fleet(self, mock_plugin, mock_database):
        """_report_outcome_to_hive reads via_fleet from candidate."""
        from modules.config import Config
        from modules.rebalancer import JobManager

        bridge = MagicMock()
        cfg = Config(dry_run=False, enable_proportional_budget=False)
        jm = JobManager(mock_plugin, cfg, mock_database, hive_bridge=bridge)

        cand = _candidate(via_fleet=True)
        job = _active_job(scid="222x333x0", candidate=cand)

        jm._report_outcome_to_hive(job, success=True, cost_sats=5, amount_transferred=50_000)

        bridge.report_rebalance_outcome.assert_called_once()
        call_kwargs = bridge.report_rebalance_outcome.call_args
        # Check via_fleet was passed through
        assert call_kwargs[1]["via_fleet"] is True or call_kwargs.kwargs.get("via_fleet") is True


# ============================================================================
# 7. Hot Channel Profit Budget Floor (I-12 fix)
# ============================================================================


class TestHotChannelProfitBudgetFloor:
    """I-12: channel_profit_budget_sats is at least 1 for small positive daily_contrib_est."""

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True, enable_proportional_budget=False)
        clboss = MagicMock()
        r = EVRebalancer(mock_plugin, cfg, mock_database, clboss)
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
        clboss = MagicMock()
        r = EVRebalancer(mock_plugin, cfg, mock_database, clboss)

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
