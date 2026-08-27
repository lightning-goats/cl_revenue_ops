"""Audit 2026-08-01 wave2 FIX 5: small verified items.

(a) planner cheap_return_term must score the DEST channel's inbound fee —
    the return leg the circular route actually traverses (what
    _get_final_hop_policy prices) — not the source peer's edge toward us,
    which the route never crosses.
(b) segment observation buckets must never over-claim: sub-50k failures
    were recorded as "cannot pass 50k" evidence.
(d) when the 'pending' history insert failed and the payment ends
    payment_pending, the engine must recover a sweepable
    pending_settlement row (or escalate loudly) instead of leaving a
    reservation hold that reconcile can never resolve.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rebalance_execution import ExecutionResult  # noqa: E402


# ---------------------------------------------------------------------------
# 5(a): cheap-return term scores the dest inbound fee
# ---------------------------------------------------------------------------


def _ch(**kwargs):
    from tests.test_rebalance_planner_v2 import _ch as make

    return make(**kwargs)


def _snap(*channels):
    from tests.test_rebalance_planner_v2 import _snap as make

    return make(*channels)


def test_cheap_return_scores_dest_inbound_fee():
    from modules.rebalance_planner_v2 import RebalancePlanner

    planner = RebalancePlanner()
    # Source has an expensive inbound edge (never traversed by the circular
    # route); dest's inbound edge (the actual return leg) is free.
    src = _ch(channel_id="100x1x0", peer_id="02" + "aa" * 32,
              local_ratio=0.90, actual_inbound_fee_ppm=5_000)
    cheap_dest = _ch(channel_id="200x1x0", peer_id="02" + "bb" * 32,
                     local_ratio=0.10, actual_inbound_fee_ppm=0)
    pairs = planner._generate_pairs([src], [cheap_dest])
    assert len(pairs) == 1
    cheap_score = pairs[0].score

    # Same shapes, but now the DEST inbound edge is expensive.
    pricey_dest = _ch(channel_id="200x1x0", peer_id="02" + "bb" * 32,
                      local_ratio=0.10, actual_inbound_fee_ppm=5_000)
    pairs = planner._generate_pairs([src], [pricey_dest])
    assert len(pairs) == 1
    pricey_score = pairs[0].score

    # A free return leg must score strictly higher than a 5000-ppm one;
    # term scale unchanged: (5000 - min(5000, ppm)) / 50_000 = 0.1 delta.
    assert cheap_score > pricey_score
    assert abs((cheap_score - pricey_score) - 0.1) < 1e-9


def test_cheap_return_ignores_source_inbound_fee():
    from modules.rebalance_planner_v2 import RebalancePlanner

    planner = RebalancePlanner()
    dest = _ch(channel_id="200x1x0", peer_id="02" + "bb" * 32,
               local_ratio=0.10, actual_inbound_fee_ppm=1_000)
    src_free = _ch(channel_id="100x1x0", peer_id="02" + "aa" * 32,
                   local_ratio=0.90, actual_inbound_fee_ppm=0)
    src_pricey = _ch(channel_id="100x1x0", peer_id="02" + "aa" * 32,
                     local_ratio=0.90, actual_inbound_fee_ppm=5_000)

    score_free = planner._generate_pairs([src_free], [dest])[0].score
    score_pricey = planner._generate_pairs([src_pricey], [dest])[0].score

    # The source's inbound edge is not on the route: no score impact.
    assert score_free == score_pricey


# ---------------------------------------------------------------------------
# 5(b): amount buckets never over-claim
# ---------------------------------------------------------------------------


def test_bucket_never_exceeds_amount():
    from modules.segment_observations import SegmentObservationStore

    for amount in (1_000, 2_500, 5_000, 25_000, 49_999, 50_000,
                   75_000, 100_000, 999_999, 10_000_000, 99_000_000):
        bucket = SegmentObservationStore.bucket_amount_sats(amount)
        assert 0 < bucket <= amount, (
            f"bucket {bucket} over-claims for amount {amount}"
        )


def test_bucket_small_amounts_use_sub_50k_floor():
    from modules.segment_observations import SegmentObservationStore

    # Partial-fill retries go down to ~1-5k sats; their failures must not
    # be recorded as "cannot pass 50k" evidence.
    assert SegmentObservationStore.bucket_amount_sats(5_000) < 50_000
    assert SegmentObservationStore.bucket_amount_sats(1_000) >= 1_000 or \
        SegmentObservationStore.bucket_amount_sats(1_000) == 0


def test_bucket_below_floor_returns_zero_and_skips_recording():
    from modules.segment_observations import SegmentObservationStore

    floor = SegmentObservationStore.BUCKETS[0]
    below = floor - 1
    assert SegmentObservationStore.bucket_amount_sats(below) == 0

    store = SegmentObservationStore()
    entry = store.record_failure(
        short_channel_id="100x1x0",
        direction=0,
        amount_sats=below,
        failure_class="liquidity",
        confidence=0.9,
    )
    assert entry is None


def test_bucket_exact_and_between_values():
    from modules.segment_observations import SegmentObservationStore

    b = SegmentObservationStore.bucket_amount_sats
    assert b(50_000) == 50_000
    assert b(99_999) == 50_000
    assert b(100_000) == 100_000
    assert b(0) == 0
    assert b(-5) == 0
    assert b("junk") == 0


# ---------------------------------------------------------------------------
# 5(d): failed pending insert + payment_pending result
# ---------------------------------------------------------------------------


OUR_ID = "03" + "u" * 64
SRC_PEER = "02" + "b" * 64
DST_PEER = "02" + "c" * 64


def _make_engine(plugin, database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    database.reserve_budget.return_value = (True, 9_999)
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def _pair():
    from modules.rebalance_types_v2 import PairCandidate

    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=50_000,
        pair_budget_sats=100,
        reason_code="ev_positive",
        route=None,
    )


def _pending_result():
    return ExecutionResult(
        success=False,
        payment_pending=True,
        amount_sats=50_000,
        error="payment_pending_timeout: RPC timeout for method: waitsendpay",
        failure_data={"payment_hash": "hash-xyz"},
    )


def test_recovers_pending_row_after_failed_insert(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    # First insert fails (DB hiccup) -> synthetic reservation id; the
    # recovery insert at result time succeeds.
    mock_database.record_rebalance.side_effect = [None, 55]
    executor = MagicMock()
    executor.execute.return_value = _pending_result()

    engine._execute_pair(_pair(), executor, reserve_budget=True,
                         account_costs=True)

    assert mock_database.record_rebalance.call_count == 2
    calls = [c for c in mock_database.update_rebalance_result.call_args_list
             if c.args[:2] == (55, "pending_settlement")]
    assert len(calls) == 1
    assert calls[0].kwargs.get("payment_hash") == "hash-xyz"
    recovery_call = mock_database.record_rebalance.call_args_list[1]
    assert recovery_call.kwargs["reservation_id"].startswith("v2-")
    # Operator signal: error-level log carrying the payment hash.
    error_logs = [m for (m, lvl) in
                  [(c.args[0], c.kwargs.get("level", c.args[1] if len(c.args) > 1 else "info"))
                   for c in mock_plugin.log.call_args_list]
                  if lvl == "error"]
    assert any("hash-xyz" in m for m in error_logs)


def test_recovery_escalates_when_insert_fails_again(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.side_effect = RuntimeError("db locked")
    executor = MagicMock()
    executor.execute.return_value = _pending_result()

    engine._execute_pair(_pair(), executor, reserve_budget=True,
                         account_costs=True)

    error_logs = [c.args[0] for c in mock_plugin.log.call_args_list
                  if (c.kwargs.get("level") or
                      (c.args[1] if len(c.args) > 1 else "info")) == "error"]
    assert any("hash-xyz" in m for m in error_logs)


def test_no_recovery_for_terminal_results(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.return_value = None
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=False, error="no_route")

    engine._execute_pair(_pair(), executor, reserve_budget=True,
                         account_costs=True)

    # Only the original (failed) insert attempt: terminal failures need no
    # sweepable row.
    assert mock_database.record_rebalance.call_count == 1
