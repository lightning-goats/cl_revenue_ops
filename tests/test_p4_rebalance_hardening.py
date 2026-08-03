"""Phase-4 rebalance hardening regression tests.

Covers:
- P4-007: engine _finish_execution_budget must not strand an 'active' budget
  reservation when a payment is pending but carries NO payment_hash (the
  history row is marked 'failed' and the reconcile sweep — which only scans
  'pending_settlement' — will never release it). Release-on-failure.
- P4-008: an orphaned execution worker (still in _execute_pair after the
  cycle timeout released the cycle lock) must not let the next cycle
  re-select the same destination -> double-pay. In-flight-destination guard.
- P4-009: rebalancer.execute_rebalance must mirror the engine's
  hold-on-pending: do NOT release the budget reservation while a payment is
  pending settlement (sweepable), but DO release when terminal/non-sweepable.
- P4-011: the PRIORITY-2 gossip base-fee ppm-equivalent must be capped at
  1_000_000 like the PRIORITY-1 peer path.
- P4-012: with cfg.max_fee_ppm == 0 the historical-fee cap must zero the
  historical EV benefit term rather than leaving it uncapped.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Engine builder (mirrors tests/test_rebalance_engine_guards.py::_make_engine)
# ---------------------------------------------------------------------------

def _make_engine(plugin, database, *, dry_run=True):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=dry_run, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    database.record_rebalance.return_value = 1
    database.reserve_budget.return_value = (True, 9999)
    database.mark_budget_spent.return_value = True
    database.release_budget_reservation.return_value = True
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def _exec_result(**kwargs):
    from modules.rebalance_executor_v2 import ExecutionResult

    return ExecutionResult(**kwargs)


# =========================================================================
# P4-007: no-payment_hash pending edge must release, not strand
# =========================================================================

class TestP4007PendingWithoutHashReleases:
    def test_pending_without_payment_hash_releases_reservation(
        self, mock_plugin, mock_database
    ):
        engine = _make_engine(mock_plugin, mock_database)
        # payment_pending but NO payment_hash in failure_data.
        result = _exec_result(
            success=False,
            payment_pending=True,
            failure_data={},
            error="payment_pending_timeout",
        )

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )

        # The row is marked 'failed' (not 'pending_settlement'), so reconcile
        # never sweeps it: the reservation MUST be released here or it strands
        # 'active' forever.
        mock_database.release_budget_reservation.assert_called_once_with("777")
        mock_database.mark_budget_spent.assert_not_called()

    def test_pending_with_payment_hash_holds_reservation(
        self, mock_plugin, mock_database
    ):
        engine = _make_engine(mock_plugin, mock_database)
        # payment_pending WITH a payment_hash -> sweepable -> keep held.
        result = _exec_result(
            success=False,
            payment_pending=True,
            failure_data={"payment_hash": "ab" * 32},
            error="payment_pending_timeout",
        )

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )

        # Sweepable pending: neither released nor spent — the reconcile sweep
        # owns the terminal transition.
        mock_database.release_budget_reservation.assert_not_called()
        mock_database.mark_budget_spent.assert_not_called()

    def test_terminal_failure_still_releases(self, mock_plugin, mock_database):
        engine = _make_engine(mock_plugin, mock_database)
        result = _exec_result(success=False, error="no_route")

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )
        mock_database.release_budget_reservation.assert_called_once_with("777")

    def test_success_marks_spent(self, mock_plugin, mock_database):
        engine = _make_engine(mock_plugin, mock_database)
        result = _exec_result(success=True, fee_sats=3)

        engine._finish_execution_budget(
            reservation_id="777",
            reserved_budget=True,
            result=result,
        )
        mock_database.mark_budget_spent.assert_called_once_with("777", 3)
        mock_database.release_budget_reservation.assert_not_called()


# =========================================================================
# P4-008: in-flight-destination guard blocks double-pay after cycle timeout
# =========================================================================

def _market_pair(dest="200x9x0"):
    from modules.rebalance_route_policy import RoutePolicy

    return SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id=dest,
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        dest_out_fee_ppm=1_000,  # sats-EV gate: refill earns expected value
        score=1.0,
        route_decision=SimpleNamespace(
            policy=RoutePolicy.MARKET_ONLY,
            allow_market_fallback=True,
            reason="ev_positive",
        ),
        route_cost_sats=None,
        route=None,
    )


def _engine_for_find_candidates(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    route_result = SimpleNamespace(
        success=True,
        route_cost_sats=1,
        route=[],
        probability_ppm=0,
        error="",
    )
    engine.router_v3.price_pair.return_value = route_result
    return engine


class TestP4008InflightDestGuard:
    def test_find_candidates_selects_dest_when_not_inflight(
        self, mock_plugin, mock_database
    ):
        engine = _engine_for_find_candidates(mock_plugin, mock_database)
        pair = _market_pair("200x9x0")

        with patch(
            "modules.rebalance_engine_v2.RebalancePlanner"
        ) as planner_cls:
            planner_cls.return_value.plan.return_value = SimpleNamespace(
                selected=[pair], skipped=[]
            )
            selected = engine.find_candidates()

        assert [p.dest_channel_id for p in selected] == ["200x9x0"]

    def test_find_candidates_excludes_inflight_dest(
        self, mock_plugin, mock_database
    ):
        engine = _engine_for_find_candidates(mock_plugin, mock_database)
        pair = _market_pair("200x9x0")

        # Simulate an orphaned worker from a prior cycle still paying this dest.
        engine._register_inflight_dest("200x9x0")

        with patch(
            "modules.rebalance_engine_v2.RebalancePlanner"
        ) as planner_cls:
            planner_cls.return_value.plan.return_value = SimpleNamespace(
                selected=[pair], skipped=[]
            )
            selected = engine.find_candidates()

        # The in-flight dest must not be re-selected -> no second payment.
        assert selected == []
        # It must never be priced/paid while in-flight.
        engine.router_v3.price_pair.assert_not_called()

    def test_execute_pair_registers_dest_during_execution_and_clears_after(
        self, mock_plugin, mock_database
    ):
        from modules.rebalance_types_v2 import PairCandidate

        engine = _make_engine(mock_plugin, mock_database)
        pair = PairCandidate(
            source_channel_id="100x1x0",
            dest_channel_id="200x9x0",
            source_peer_id="03" + "b" * 64,
            dest_peer_id="03" + "c" * 64,
            amount_sats=50_000,
            pair_budget_sats=10_000,
            route=[],
        )

        seen = {}

        class _FakeExecutor:
            def execute(self, **kwargs):
                seen["during"] = engine._inflight_dest_snapshot()
                return _exec_result(success=True, fee_sats=1)

        engine._execute_pair(
            pair,
            _FakeExecutor(),
            reserve_budget=False,
            account_costs=False,
            rebalance_id=5,
        )

        # Registered while the payment is in flight ...
        assert "200x9x0" in seen["during"]
        # ... and cleared once the worker finishes (so a later cycle can retry).
        assert "200x9x0" not in engine._inflight_dest_snapshot()

    def test_execute_pair_unregisters_dest_even_on_executor_exception(
        self, mock_plugin, mock_database
    ):
        from modules.rebalance_types_v2 import PairCandidate

        engine = _make_engine(mock_plugin, mock_database)
        pair = PairCandidate(
            source_channel_id="100x1x0",
            dest_channel_id="200x9x0",
            source_peer_id="03" + "b" * 64,
            dest_peer_id="03" + "c" * 64,
            amount_sats=50_000,
            pair_budget_sats=10_000,
            route=[],
        )

        class _BoomExecutor:
            def execute(self, **kwargs):
                raise RuntimeError("boom")

        engine._execute_pair(
            pair,
            _BoomExecutor(),
            reserve_budget=False,
            account_costs=False,
            rebalance_id=5,
        )
        assert "200x9x0" not in engine._inflight_dest_snapshot()


# =========================================================================
# P4-009: execute_rebalance mirrors the engine's hold-on-pending
# =========================================================================

def _rebalancer_with_engine(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=False)
    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r.data_service = MagicMock()
    r.data_service.invalidate = MagicMock()
    r.data_service.datastore_push.return_value = True
    r._check_capital_controls = MagicMock(return_value=True)
    r._get_peer_connection_status = MagicMock(return_value={})
    r._calculate_turnover_rate = MagicMock(return_value=0.05)
    r.rebalance_engine_v2 = MagicMock()

    mock_database.record_rebalance = MagicMock(return_value=901)
    mock_database.update_rebalance_result = MagicMock()
    mock_database.reserve_budget = MagicMock(return_value=(True, 9999))
    mock_database.release_budget_reservation = MagicMock(return_value=True)
    mock_database.increment_failure_count = MagicMock()
    return r


def _rebalance_candidate(amount_sats=50_000):
    from modules.rebalancer import RebalanceCandidate

    return RebalanceCandidate(
        source_candidates=["111x1x0"],
        to_channel="222x2x0",
        primary_source_peer_id="02" + "a" * 64,
        to_peer_id="02" + "b" * 64,
        amount_sats=amount_sats,
        amount_msat=amount_sats * 1000,
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


class TestP4009ExecuteRebalanceHoldOnPending:
    def test_pending_with_payment_hash_holds_reservation(
        self, mock_plugin, mock_database
    ):
        r = _rebalancer_with_engine(mock_plugin, mock_database)
        r.rebalance_engine_v2.execute_candidate.return_value = _exec_result(
            success=False,
            payment_pending=True,
            failure_data={"payment_hash": "cd" * 32},
            error="payment_pending_timeout",
            route_type="native",
            attempts=1,
        )

        res = r.execute_rebalance(_rebalance_candidate(), enforce_budget=True)

        assert res["success"] is False
        # Sweepable pending: the engine parked the shared row as
        # 'pending_settlement' and reconcile_pending_settlements owns the
        # release. execute_rebalance must NOT release here (double-spend risk).
        mock_database.release_budget_reservation.assert_not_called()

    def test_pending_without_payment_hash_releases_reservation(
        self, mock_plugin, mock_database
    ):
        r = _rebalancer_with_engine(mock_plugin, mock_database)
        r.rebalance_engine_v2.execute_candidate.return_value = _exec_result(
            success=False,
            payment_pending=True,
            failure_data={},
            error="payment_pending_timeout",
            route_type="native",
            attempts=1,
        )

        res = r.execute_rebalance(_rebalance_candidate(), enforce_budget=True)

        assert res["success"] is False
        # Non-sweepable pending (no payment_hash): the engine marked the row
        # 'failed'; reconcile never touches it, so the reservation must be
        # released here or it strands 'active'.
        mock_database.release_budget_reservation.assert_called_once_with("901")

    def test_terminal_failure_releases_reservation(
        self, mock_plugin, mock_database
    ):
        r = _rebalancer_with_engine(mock_plugin, mock_database)
        r.rebalance_engine_v2.execute_candidate.return_value = _exec_result(
            success=False,
            payment_pending=False,
            error="no_route",
            route_type="native",
            attempts=1,
        )

        res = r.execute_rebalance(_rebalance_candidate(), enforce_budget=True)

        assert res["success"] is False
        mock_database.release_budget_reservation.assert_called_once_with("901")


# =========================================================================
# P4-011: gossip base-fee ppm-equivalent must be capped at 1_000_000
# =========================================================================

class TestP4011GossipBaseFeeCap:
    def _rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=False)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._get_our_node_id = MagicMock(return_value="02" + "f" * 64)
        r.data_service = MagicMock()
        return r

    def test_huge_gossip_base_fee_is_capped(self, mock_plugin, mock_database):
        r = self._rebalancer(mock_plugin, mock_database)
        # PRIORITY-2 gossip path (peer not in _peer_inbound_fees). A garbage
        # base fee must not inflate the inbound-fee estimate past 100% ppm.
        r.data_service.get_channels.return_value = {
            "channels": [
                {
                    "destination": "02" + "f" * 64,
                    "fee_per_millionth": 100,
                    "base_fee_millisatoshi": 10 ** 18,
                }
            ]
        }

        result = r._get_last_hop_fee("02" + "a" * 64, amount_msat=100_000_000)

        # ppm(100) + capped base ppm-equivalent (1_000_000), never the raw
        # ~1e16 the uncapped math would produce.
        assert result == 100 + 1_000_000

    def test_small_gossip_base_fee_not_capped(self, mock_plugin, mock_database):
        r = self._rebalancer(mock_plugin, mock_database)
        r.data_service.get_channels.return_value = {
            "channels": [
                {
                    "destination": "02" + "f" * 64,
                    "fee_per_millionth": 100,
                    "base_fee_millisatoshi": 1_000,  # 1 sat base
                }
            ]
        }

        result = r._get_last_hop_fee("02" + "a" * 64, amount_msat=100_000_000)

        # base_ppm = 1000*1e6//1e8 = 10; well under the cap.
        assert result == 110


# =========================================================================
# P4-012: max_fee_ppm == 0 must zero the historical-fee EV benefit
# =========================================================================

class TestP4012HistoricalFeeCapZeroDisablesBenefit:
    def test_helper_zero_cap_returns_zero(self):
        from modules.rebalance_engine_v2 import _bounded_historical_fee_ppm

        # A configured max_fee_ppm of 0 means "no fee headroom": historical
        # ratios must contribute NO EV benefit, not be left uncapped.
        assert _bounded_historical_fee_ppm(9_999.0, 0.0) == 0.0
        assert _bounded_historical_fee_ppm(1_000_000.0, 0.0) == 0.0

    def test_helper_positive_cap_still_bounds(self):
        from modules.rebalance_engine_v2 import _bounded_historical_fee_ppm

        assert _bounded_historical_fee_ppm(9_999.0, 500.0) == 500.0
        assert _bounded_historical_fee_ppm(200.0, 500.0) == 200.0

    def test_rate_helper_zero_cap_returns_zero(self):
        from modules.rebalance_engine_v2 import _historical_fee_rate_ppm

        # fee/volume ratio would be huge, but cap 0 zeroes it.
        assert _historical_fee_rate_ppm(1_000_000_000, 1_000, 0.0) == 0.0

    def test_score_decomposition_zero_cap_gives_no_historical_benefit(
        self, mock_plugin, mock_database
    ):
        from modules.rebalance_types_v2 import PairCandidate

        engine = _make_engine(mock_plugin, mock_database)
        # Config with max_fee_ppm == 0 (no P1-009 change needed: swap it in
        # directly as the live config so the helper sees cap 0).
        engine.config = SimpleNamespace(
            max_fee_ppm=0,
            high_liquidity_threshold=0.65,
            low_liquidity_threshold=0.35,
        )
        pair = PairCandidate(
            source_channel_id="100x1x0",
            dest_channel_id="200x9x0",
            source_peer_id="03" + "b" * 64,
            dest_peer_id="03" + "c" * 64,
            amount_sats=1_000_000,
            pair_budget_sats=10_000,
            dest_out_fee_ppm=0,  # only a freak historical ratio is available
            source_out_fee_ppm=0,
            dest_historical_direct_fee_ppm=9_999.0,
            source_historical_sourced_fee_ppm=9_999.0,
        )

        decomp = engine._build_score_decomposition(pair, route_cost_sats=1)

        # With cap 0 the historical ppm is zeroed, so it contributes no EV
        # benefit and cannot clear the hold gate for a losing rebalance.
        assert decomp["expected_future_value_sats"] == 0.0
        assert decomp["destination_refill_value_sats"] == 0.0
        assert decomp["source_drain_value_sats"] == 0.0
        assert decomp["inputs"]["dest_historical_direct_fee_ppm"] == 0.0
        assert decomp["inputs"]["source_historical_sourced_fee_ppm"] == 0.0


# =========================================================================
# Mutation guards for the sats-EV gate in _build_score_decomposition
# (beats_do_nothing = not rejection_reason and
#                     (expected_fee_sats <= 0 or final_score_sats >= 0.0))
# =========================================================================

def _negative_score_zero_cost_pair():
    """A zero-cost pair whose sats-EV final score is strictly negative.

    dest earns nothing (dest_out_fee_ppm=0) while the source forgoes a large
    outbound fee, so final_score_sats < 0. Because the route costs nothing
    (route_cost_sats=0 -> expected_fee_sats=0), the do-nothing gate's
    zero-cost bypass must still let it beat do-nothing.
    """
    from modules.rebalance_types_v2 import PairCandidate

    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x9x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=1_000_000,
        pair_budget_sats=10_000,
        dest_out_fee_ppm=0,
        source_out_fee_ppm=5_000,
    )


class TestSatsEVGateMutationGuards:
    def test_zero_cost_route_beats_do_nothing_even_when_score_negative(
        self, mock_plugin, mock_database
    ):
        # Kills the `expected_fee_sats <= 0` -> `< 0` mutation: a zero-cost
        # route (expected_fee_sats == 0) must bypass the score gate.
        engine = _make_engine(mock_plugin, mock_database)
        pair = _negative_score_zero_cost_pair()

        decomp = engine._build_score_decomposition(pair, route_cost_sats=0)

        assert decomp["expected_fee_sats"] == 0
        assert decomp["final_score_sats"] < 0.0
        assert decomp["beats_do_nothing"] is True

    def test_rejected_pair_never_beats_do_nothing(
        self, mock_plugin, mock_database
    ):
        # Kills dropping the `not rejection_reason` guard: even a zero-cost
        # pair (which would otherwise bypass the score gate) must NOT beat
        # do-nothing once it carries a rejection reason.
        engine = _make_engine(mock_plugin, mock_database)
        pair = _negative_score_zero_cost_pair()

        decomp = engine._build_score_decomposition(
            pair, route_cost_sats=0, rejection_reason="no_route"
        )

        assert decomp["expected_fee_sats"] == 0
        assert decomp["beats_do_nothing"] is False


# =========================================================================
# Mutation guard for diagnostic_rebalance honesty: a still-pending shock
# delivered no confirmed liquidity and must read 'pending', not 'completed'.
# =========================================================================
