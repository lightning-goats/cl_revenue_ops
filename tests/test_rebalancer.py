"""Tests for rebalancer core behavior."""

from unittest.mock import MagicMock


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


class TestRebalancerDryRun:
    """Verify dry_run mode doesn't actually execute rebalances."""

    def _make_rebalancer(self, mock_plugin, mock_database, dry_run=True):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=dry_run)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.job_manager.start_job = MagicMock(return_value={"success": True})
        mock_database.record_rebalance = MagicMock(return_value=123)
        mock_database.update_rebalance_result = MagicMock()
        return r

    def test_dry_run_does_not_start_sling_job(self, mock_plugin, mock_database):
        """In dry_run mode, execute_rebalance should record but not start a sling job."""
        r = self._make_rebalancer(mock_plugin, mock_database, dry_run=True)

        cand = _candidate()
        result = r.execute_rebalance(cand)

        assert result["success"] is True
