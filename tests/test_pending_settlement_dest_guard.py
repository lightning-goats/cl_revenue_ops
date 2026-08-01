"""Pending-settlement destination guard regression tests.

A destination whose rebalance payment timed out in waitsendpay is parked as
status='pending_settlement' with its budget reservation held, but the P4-008
in-flight guard (_inflight_dests) is cleared when the worker thread exits.
Nothing then protected the destination: the next cycle could pay a SECOND
invoice into the same dest while the first HTLC was still in flight (HTLCs
can pend for hours). If the stuck HTLC later settled, the dest was
double-filled and both fees were paid — violating the engine's own P8-001
"never pay again on top" invariant.

The guard extends P4-008 across worker lifetimes: any dest with a live
pending_settlement row is excluded from find_candidates and rejected by
execute_candidate until reconcile_pending_settlements resolves the row.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Engine builder (mirrors tests/test_p4_rebalance_hardening.py::_make_engine)
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


def _market_pair(dest="200x9x0"):
    from modules.rebalance_route_policy import RoutePolicy

    return SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id=dest,
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        dest_out_fee_ppm=1_000,
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


def _candidate(dest="200x9x0"):
    return SimpleNamespace(
        from_channel="100x1x0",
        to_channel=dest,
        from_peer_id="03" + "b" * 64,
        to_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=100,
        reason_code="manual",
    )


# =========================================================================
# find_candidates: pending_settlement dest must be excluded
# =========================================================================

class TestFindCandidatesPendingSettlementGuard:
    def test_selects_dest_when_no_pending_rows(
        self, mock_plugin, mock_database
    ):
        engine = _engine_for_find_candidates(mock_plugin, mock_database)
        mock_database.get_pending_settlement_dest_channels.return_value = []
        pair = _market_pair("200x9x0")

        with patch(
            "modules.rebalance_engine_v2.RebalancePlanner"
        ) as planner_cls:
            planner_cls.return_value.plan.return_value = SimpleNamespace(
                selected=[pair], skipped=[]
            )
            selected = engine.find_candidates()

        assert [p.dest_channel_id for p in selected] == ["200x9x0"]

    def test_excludes_pending_settlement_dest(
        self, mock_plugin, mock_database
    ):
        engine = _engine_for_find_candidates(mock_plugin, mock_database)
        # A prior cycle's payment to this dest timed out and is parked as
        # pending_settlement; its worker thread (and _inflight_dests entry)
        # is long gone.
        mock_database.get_pending_settlement_dest_channels.return_value = [
            "200x9x0"
        ]
        pair = _market_pair("200x9x0")

        with patch(
            "modules.rebalance_engine_v2.RebalancePlanner"
        ) as planner_cls:
            plan = SimpleNamespace(selected=[pair], skipped=[])
            planner_cls.return_value.plan.return_value = plan
            selected = engine.find_candidates()

        # The dest must not be re-selected while the HTLC can still settle.
        assert selected == []
        engine.router_v3.price_pair.assert_not_called()
        # And the exclusion must surface in skip reporting like dest_inflight.
        assert [s.reason for s in plan.skipped] == ["dest_pending_settlement"]
        assert plan.skipped[0].channel_id == "200x9x0"

    def test_other_dests_still_selected(self, mock_plugin, mock_database):
        engine = _engine_for_find_candidates(mock_plugin, mock_database)
        mock_database.get_pending_settlement_dest_channels.return_value = [
            "200x9x0"
        ]
        blocked = _market_pair("200x9x0")
        ok = _market_pair("300x5x0")

        with patch(
            "modules.rebalance_engine_v2.RebalancePlanner"
        ) as planner_cls:
            planner_cls.return_value.plan.return_value = SimpleNamespace(
                selected=[blocked, ok], skipped=[]
            )
            selected = engine.find_candidates()

        assert [p.dest_channel_id for p in selected] == ["300x5x0"]


# =========================================================================
# execute_candidate (manual/RPC path): pending_settlement dest rejected
# =========================================================================

class TestExecuteCandidatePendingSettlementGuard:
    def test_rejects_pending_settlement_dest(
        self, mock_plugin, mock_database
    ):
        engine = _make_engine(mock_plugin, mock_database)
        mock_database.get_pending_settlement_dest_channels.return_value = [
            "200x9x0"
        ]

        result = engine.execute_candidate(_candidate("200x9x0"))

        assert result.success is False
        assert result.error == "dest_pending_settlement"

    def test_allows_dest_once_row_resolved(self, mock_plugin, mock_database):
        engine = _make_engine(mock_plugin, mock_database)
        # Reconcile resolved the row (success/failed): the dest is live again
        # and execution proceeds past the guard (to fail later on routing —
        # anything but the guard's own error).
        mock_database.get_pending_settlement_dest_channels.return_value = []

        result = engine.execute_candidate(_candidate("200x9x0"))

        assert result.error != "dest_pending_settlement"


# =========================================================================
# Database: get_pending_settlement_dest_channels
# =========================================================================

@pytest.fixture
def real_database(tmp_path):
    from modules.database import Database

    plugin = MagicMock()
    plugin.log = MagicMock()
    db = Database(str(tmp_path / "test.db"), plugin)
    db.initialize()
    return db


def _park_pending(db, dest, payment_hash="hash-abc"):
    rebalance_id = db.record_rebalance(
        from_channel="100x1x0",
        to_channel=dest,
        amount_sats=100_000,
        max_fee_sats=100,
        expected_profit_sats=0,
        status="pending",
    )
    db.update_rebalance_result(
        rebalance_id,
        "pending_settlement",
        error_message="payment_pending_timeout",
        payment_hash=payment_hash,
    )
    return rebalance_id


class TestDatabasePendingSettlementDestChannels:
    def test_lists_pending_dest_and_clears_on_resolution(self, real_database):
        rebalance_id = _park_pending(real_database, "200x1x0")

        assert real_database.get_pending_settlement_dest_channels() == [
            "200x1x0"
        ]

        # Reconcile resolves the row: the dest becomes eligible again.
        real_database.update_rebalance_result(
            rebalance_id, "success", actual_fee_msat=5000
        )
        assert real_database.get_pending_settlement_dest_channels() == []

    def test_deduplicates_and_ignores_unsweepable_rows(self, real_database):
        _park_pending(real_database, "200x1x0", payment_hash="hash-1")
        _park_pending(real_database, "200x1x0", payment_hash="hash-2")
        # A pending_settlement row WITHOUT a payment_hash can never be
        # resolved by the reconcile sweep; it must not block its dest
        # forever. (The engine never parks such rows — defensive only.)
        real_database.update_rebalance_result(
            real_database.record_rebalance(
                from_channel="100x1x0",
                to_channel="300x1x0",
                amount_sats=100_000,
                max_fee_sats=100,
                expected_profit_sats=0,
                status="pending",
            ),
            "pending_settlement",
        )

        assert real_database.get_pending_settlement_dest_channels() == [
            "200x1x0"
        ]


# =========================================================================
# Audit vocabulary: the new skip reason must be bucketable
# =========================================================================

def test_dest_pending_settlement_is_a_valid_skip_reason():
    from modules.rebalance_audit_v2 import VALID_SKIP_REASONS

    assert "dest_pending_settlement" in VALID_SKIP_REASONS
