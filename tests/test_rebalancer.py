"""Tests for rebalancer traffic-aware conflict check (Task 5)."""

from unittest.mock import MagicMock, call


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


def _make_safe_hive_bridge(**conflict_return):
    """Create a MagicMock hive_bridge with safe defaults for execute_rebalance.

    The execute_rebalance method is ~430 lines with many hive_bridge calls.
    This helper sets return values that let execution proceed through all
    phases without hitting MagicMock format-string issues.
    """
    bridge = MagicMock()
    bridge.check_rebalance_conflict.return_value = conflict_return or {"conflict": False}
    bridge.check_circular_flow_risk.return_value = {"risk": False}
    bridge.query_fleet_rebalance_path.return_value = None  # Skip fleet path phase
    return bridge


class TestTrafficAwareConflictCheck:
    """Verify execute_rebalance passes direction/amount to check_rebalance_conflict."""

    def _make_rebalancer(self, mock_plugin, mock_database, dry_run=True):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=dry_run, enable_proportional_budget=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.job_manager.start_job = MagicMock(return_value={"success": True})
        mock_database.record_rebalance = MagicMock(return_value=123)
        mock_database.update_rebalance_result = MagicMock()
        return r

    def test_direction_and_amount_passed_to_conflict_check(
        self, mock_plugin, mock_database
    ):
        """check_rebalance_conflict receives direction='outbound' and amount_sats."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        r.hive_bridge = _make_safe_hive_bridge(conflict=False)

        cand = _candidate(amount_sats=75000)
        r.execute_rebalance(cand)

        r.hive_bridge.check_rebalance_conflict.assert_called_once_with(
            peer_id=cand.to_peer_id,
            direction="outbound",
            amount_sats=75000,
        )

    def test_peak_hour_logged_but_rebalance_not_blocked(
        self, mock_plugin, mock_database
    ):
        """Peak hour info is logged at 'info' level but rebalance proceeds."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        r.hive_bridge = _make_safe_hive_bridge(
            conflict=False,
            peer_in_peak_hours=True,
            suggested_window_utc=["03:00", "07:00"],
        )

        cand = _candidate()
        result = r.execute_rebalance(cand)

        # Rebalance should proceed (dry_run returns success)
        assert result["success"] is True

        # Verify TRAFFIC_INTEL was logged
        log_calls = [
            c for c in mock_plugin.log.call_args_list
            if "TRAFFIC_INTEL" in str(c)
        ]
        assert len(log_calls) >= 1
        logged_msg = str(log_calls[0])
        assert "peer in peak hours" in logged_msg
        assert "suggested window" in logged_msg

    def test_peak_hour_without_suggested_window(
        self, mock_plugin, mock_database
    ):
        """Peak hour logged even when no suggested_window_utc is provided."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        r.hive_bridge = _make_safe_hive_bridge(
            conflict=False,
            peer_in_peak_hours=True,
        )

        cand = _candidate()
        result = r.execute_rebalance(cand)

        assert result["success"] is True

        log_calls = [
            c for c in mock_plugin.log.call_args_list
            if "TRAFFIC_INTEL" in str(c)
        ]
        assert len(log_calls) >= 1
        logged_msg = str(log_calls[0])
        assert "peer in peak hours" in logged_msg
        assert "suggested window" not in logged_msg

    def test_no_peak_hour_no_traffic_intel_log(
        self, mock_plugin, mock_database
    ):
        """When peer_in_peak_hours is false, no TRAFFIC_INTEL line is logged."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        r.hive_bridge = _make_safe_hive_bridge(
            conflict=False,
            peer_in_peak_hours=False,
        )

        cand = _candidate()
        r.execute_rebalance(cand)

        log_calls = [
            c for c in mock_plugin.log.call_args_list
            if "TRAFFIC_INTEL" in str(c)
        ]
        assert len(log_calls) == 0

    def test_conflict_with_peak_hours_skips_fleet_path_and_logs_both(
        self, mock_plugin, mock_database
    ):
        """When conflict=True AND peer_in_peak_hours=True, both are logged."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        r.hive_bridge = _make_safe_hive_bridge(
            conflict=True,
            reason="Fleet member active on same peer",
            peer_in_peak_hours=True,
            suggested_window_utc=["04:00", "08:00"],
        )

        cand = _candidate()
        result = r.execute_rebalance(cand)

        # Rebalance still proceeds (conflict only skips fleet path, doesn't block)
        assert result["success"] is True

        # Both FLEET_CONFLICT and TRAFFIC_INTEL should be logged
        fleet_logs = [
            c for c in mock_plugin.log.call_args_list
            if "FLEET_CONFLICT" in str(c)
        ]
        traffic_logs = [
            c for c in mock_plugin.log.call_args_list
            if "TRAFFIC_INTEL" in str(c)
        ]
        assert len(fleet_logs) >= 1
        assert len(traffic_logs) >= 1
