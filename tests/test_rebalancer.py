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
        mock_database.record_rebalance = MagicMock(return_value=123)
        mock_database.update_rebalance_result = MagicMock()
        return r

    def test_dry_run_does_not_execute_rebalance(self, mock_plugin, mock_database):
        """In dry_run mode, execute_rebalance should record but not execute."""
        r = self._make_rebalancer(mock_plugin, mock_database, dry_run=True)

        cand = _candidate()
        result = r.execute_rebalance(cand)

        assert result["success"] is True


class TestFutilityKeyPeerScoped:
    """DEF-063: the destination futility/failure-count breaker must be keyed
    on the peer pubkey (stable across splices), not the SCID (which a splice
    mints anew, resetting the history and evading the breaker)."""

    PEER = "02" + "d" * 64

    def test_futility_key_uses_peer_id(self):
        from modules.rebalancer import EVRebalancer
        cand = _candidate(to_channel="222x333x0", to_peer_id=self.PEER)
        assert EVRebalancer._futility_key(cand) == self.PEER

    def test_futility_key_falls_back_to_scid_without_peer(self):
        from modules.rebalancer import EVRebalancer
        cand = _candidate(to_channel="222x333x0", to_peer_id="")
        assert EVRebalancer._futility_key(cand) == "222x333x0"

    def test_failure_history_survives_splice(self, tmp_path):
        """A pair fails repeatedly under SCID A, then splices to SCID B (same
        peer). The failure count and the futility breaker must persist."""
        import os
        from unittest.mock import MagicMock
        from modules.database import Database
        from modules.rebalancer import EVRebalancer

        db = Database(os.path.join(tmp_path, "def063.db"), MagicMock())
        db.initialize()

        before = _candidate(to_channel="AAAx1x0", to_peer_id=self.PEER)
        after_splice = _candidate(to_channel="BBBx9x0", to_peer_id=self.PEER)

        # Four no_route failures accrue while the channel is SCID A.
        for _ in range(4):
            db.increment_failure_count(
                EVRebalancer._futility_key(before),
                attempted_ppm=1000, attempted_amount=50000, error_type="no_route",
            )

        # The splice mints SCID B. Keyed on the SCID this would read 0; keyed
        # on the peer it must still see the accrued history.
        count, _ = db.get_failure_count(EVRebalancer._futility_key(after_splice))
        assert count == 4
        assert EVRebalancer._should_skip_futility(count, "no_route") is True

        # Sanity: the raw new SCID has no independent record.
        scid_count, _ = db.get_failure_count("BBBx9x0")
        assert scid_count == 0


class TestInboundFeeBlendNoDoubleCount:
    """DEF-067-S4: the medium-confidence inbound-fee blend must not add the
    inbound_fee_estimate_ppm buffer on top of last_hop — the historical
    median already carries the multi-hop cost, so the buffer double-counts
    it in the blend."""

    PEER = "03" + "e" * 64

    def _rebalancer(self, mock_plugin, mock_database, buffer_ppm):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        cfg = Config(inbound_fee_estimate_ppm=buffer_ppm)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r.hive_hints = None  # bypass the hive-member fast path
        return r

    def test_medium_confidence_blend_excludes_buffer(self, mock_plugin, mock_database):
        from unittest.mock import MagicMock
        r = self._rebalancer(mock_plugin, mock_database, buffer_ppm=50)
        mock_database.get_historical_inbound_fee_ppm = MagicMock(return_value={
            "confidence": "medium",
            "median_fee_ppm": 200,
            "avg_fee_ppm": 200,
            "sample_count": 6,
        })
        r._get_last_hop_fee = MagicMock(return_value=150)

        estimate = r._estimate_inbound_fee(self.PEER)

        # Correct blend: 0.7*200 + 0.3*150 = 185 (raw last_hop, no buffer).
        assert estimate == 185
        # Double-counted blend would have been 0.7*200 + 0.3*(150+50) = 200.
        assert estimate != 200
