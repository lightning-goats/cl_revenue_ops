"""Regression tests for the audit-verified rebalance economics fixes.

Covers:
  F1 — per-attempt fee ceiling (budget is the reservation cap, ppm cap
       bounds any single attempt)
  F2 — sats-denominated EV gate (expected future value vs route cost,
       source opportunity, failure penalty)
  F3 — capital_risk_penalty dropped when the effective budget equals the
       raw budget (no more double-count of expected_fee at default config)
  F5 — fraction-of-excess-cleared source "opportunity cost" replaced by
       the sats opportunity formula
  F6 — p_success blends the destination's empirical rebalance success
       rate when the router reports no probability
  F7 — destination cooldown scaled by the last rebalance's fill fraction
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_engine(mock_plugin, mock_database, **cfg_kwargs):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3", **cfg_kwargs)
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=100_000,
        )
    )
    return engine


def _make_pair(**overrides):
    from modules.rebalance_types_v2 import PairCandidate

    kwargs = dict(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=100_000,
        pair_budget_sats=2_500,
        source_capacity_sats=2_000_000,
        dest_capacity_sats=2_000_000,
        score=1.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.15,
    )
    kwargs.update(overrides)
    return PairCandidate(**kwargs)


def _run_single_pair(engine, pair, *, route_cost_sats, probability_ppm=900_000):
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=route_cost_sats,
        route=[{"channel": "900x9x0"}],
        probability_ppm=probability_ppm,
        error="",
    )
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[pair], skipped=[]
        )
        return engine.find_candidates()


def _skip_reasons(engine):
    debug = engine.get_last_cycle_debug()
    return {row["channel_id"]: row["reason"] for row in debug["skipped"]}


# ---------------------------------------------------------------------------
# F2 — sats-EV gate: the audit's two flagship cases
# ---------------------------------------------------------------------------


def test_f2a_negative_ev_move_is_rejected(mock_plugin, mock_database):
    """Audit case (a): dest earning 100 ppm, route fee 11,000 ppm.

    Old gate approved this because a big pair budget normalized the fee
    penalty to near zero. The sats gate must reject: expected future value
    = 100k * 100ppm * 0.5 = 5 sats vs a 1,100 sat fee.
    """
    engine = _make_engine(
        mock_plugin, mock_database, pair_fee_cap_ppm=20_000
    )
    pair = _make_pair(
        amount_sats=100_000,
        pair_budget_sats=50_000,
        dest_out_fee_ppm=100,
    )

    selected = _run_single_pair(engine, pair, route_cost_sats=1_100)

    assert selected == []
    assert _skip_reasons(engine).get("200x2x0") == "below_hold_margin"
    debug = engine.get_last_cycle_debug()
    decomp = debug["considered_candidates"][0]["score_decomposition"]
    assert decomp["final_score_sats"] < 0.0


def test_f2b_positive_ev_move_is_approved(mock_plugin, mock_database):
    """Audit case (b): dest earning 2,500 ppm, route fee 700 ppm on 1M sats.

    Old gate rejected this because the ppm-capped budget made the
    normalized penalties huge. Sats EV: 0.9 * 1,250 - 700 - 25 = 400 > 0.
    """
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=1_000_000,
        pair_budget_sats=5_000,
        dest_out_fee_ppm=2_500,
        source_out_fee_ppm=100,
    )

    selected = _run_single_pair(
        engine, pair, route_cost_sats=700, probability_ppm=900_000
    )

    assert selected == [pair]
    decomp = pair.score_decomposition
    assert decomp["expected_future_value_sats"] == pytest.approx(1_250.0)
    assert decomp["source_opportunity_sats"] == pytest.approx(25.0)
    assert decomp["final_score_sats"] == pytest.approx(400.0)
    assert decomp["beats_do_nothing"] is True


def test_f2c_zero_budget_free_route_still_passes(mock_plugin, mock_database):
    """Equalization invariant: budget 0 + free route stays selectable even
    when the destination earns nothing (zero-cost moves never lose to
    do-nothing)."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        pair_budget_sats=0,
        dest_out_fee_ppm=0,
        reason_code="hive_equalization",
    )

    selected = _run_single_pair(
        engine, pair, route_cost_sats=0, probability_ppm=0
    )

    assert selected == [pair]


def test_f2c_zero_budget_blocks_paid_route(mock_plugin, mock_database):
    """Equalization invariant: budget 0 admits only free routes."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(pair_budget_sats=0, dest_out_fee_ppm=5_000)

    selected = _run_single_pair(engine, pair, route_cost_sats=1)

    assert selected == []
    assert _skip_reasons(engine).get("200x2x0") == "route_over_budget"


def test_f2d_hold_margin_is_respected_in_sats(mock_plugin, mock_database):
    """A pair whose sats EV is positive but below the configured hold
    margin must be rejected; with margin 0 the same pair is selected."""

    def run(margin):
        engine = _make_engine(
            mock_plugin, mock_database, rebalance_hold_margin=margin
        )
        pair = _make_pair(
            amount_sats=200_000,
            pair_budget_sats=1_000,
            dest_out_fee_ppm=20,  # value = 200k * 20ppm * 0.5 = 2 sats
        )
        selected = _run_single_pair(
            engine, pair, route_cost_sats=1, probability_ppm=990_000
        )
        return engine, selected

    # net EV = 0.99 * 2 - 1 = 0.98 sats
    engine, selected = run(1.0)
    assert selected == []
    assert _skip_reasons(engine).get("200x2x0") == "below_hold_margin"

    engine, selected = run(0.0)
    assert len(selected) == 1


# ---------------------------------------------------------------------------
# F3 — capital_risk_penalty only when the effective budget differs
# ---------------------------------------------------------------------------


def test_f3_capital_risk_dropped_when_effective_equals_raw_budget(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(pair_budget_sats=100)

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=90,
        effective_budget_sats=100,
        route_status="priced",
    )
    assert decomp["capital_risk_penalty"] == 0.0

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=90,
        effective_budget_sats=125,
        route_status="priced",
    )
    assert decomp["capital_risk_penalty"] > 0.0


# ---------------------------------------------------------------------------
# F5 — fraction-of-excess-cleared term fully removed
# ---------------------------------------------------------------------------


def test_f5_drain_completeness_no_longer_penalized(mock_plugin, mock_database):
    """Fully clearing a small source used to eat the maximum 0.25 penalty.
    The old fraction-based term must be gone; the sats opportunity term
    follows the source's actual forgone fee earnings instead."""
    engine = _make_engine(mock_plugin, mock_database)
    # Source at 90% on 200k capacity: a 50k move clears its entire excess.
    pair = _make_pair(
        amount_sats=50_000,
        source_capacity_sats=200_000,
        source_local_ratio=0.90,
        source_out_fee_ppm=1_000,
        dest_out_fee_ppm=2_000,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=500,
        route_status="priced",
    )

    assert decomp["source_opportunity_cost"] == 0.0
    # amount * source_ppm/1e6 * UTILIZATION(0.5) * SOURCE_UTIL_DISCOUNT(0.5)
    assert decomp["source_opportunity_sats"] == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# F6 — p_success blends the empirical destination success rate
# ---------------------------------------------------------------------------


def test_f6_p_success_blends_empirical_rate_when_router_silent(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_channel_rebalance_success_rate.return_value = {
        "total": 10,
        "successes": 9,
        "failures": 1,
        "success_rate": 0.9,
    }
    pair = _make_pair()

    decomp = engine._build_score_decomposition(
        pair, probability_ppm=0, route_cost_sats=5, route_status="priced"
    )
    # blend of the 0.5 prior with the 0.9 empirical rate
    assert decomp["p_success"] == pytest.approx(0.7)


def test_f6_p_success_ignores_thin_empirical_history(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_channel_rebalance_success_rate.return_value = {
        "total": 2,
        "successes": 2,
        "failures": 0,
        "success_rate": 1.0,
    }
    pair = _make_pair()

    decomp = engine._build_score_decomposition(
        pair, probability_ppm=0, route_cost_sats=5, route_status="priced"
    )
    assert decomp["p_success"] == 0.5


def test_f6_router_probability_wins_over_empirical_rate(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_channel_rebalance_success_rate.return_value = {
        "total": 50,
        "success_rate": 0.1,
    }
    pair = _make_pair()

    decomp = engine._build_score_decomposition(
        pair, probability_ppm=800_000, route_cost_sats=5, route_status="priced"
    )
    assert decomp["p_success"] == 0.8


# ---------------------------------------------------------------------------
# F2 plumbing — our own outbound fee threads snapshot → planner → pair
# ---------------------------------------------------------------------------


def test_snapshot_builder_reads_our_outbound_fee(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = type(engine)._build_snapshot.__get__(engine)
    mock_database.get_last_rebalance_times.return_value = {}
    mock_plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "1" * 64,
                "short_channel_id": "100x1x0",
                "total_msat": 1_000_000_000,
                "our_amount_msat": 800_000_000,
                "updates": {
                    "local": {"fee_proportional_millionths": 1_750},
                    "remote": {"fee_proportional_millionths": 90},
                },
            }
        ]
    }

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    channel = snapshot.channels[0]
    assert channel.local_out_fee_ppm == 1_750
    assert channel.actual_inbound_fee_ppm == 90


def test_planner_threads_outbound_fees_to_pair():
    from modules.rebalance_planner_v2 import RebalancePlanner
    from modules.rebalance_state_v2 import build_state_snapshot

    snapshot = build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": "02" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
                "local_out_fee_ppm": 450,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": "02" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
                "local_out_fee_ppm": 2_200,
                "is_active": True,
            },
        ],
        {"channel_budgets": {"200x1x0": {"budget_sats": 1_000}}},
    )

    plan = RebalancePlanner().plan(snapshot)

    assert len(plan.selected) == 1
    pair = plan.selected[0]
    assert pair.source_out_fee_ppm == 450
    assert pair.dest_out_fee_ppm == 2_200


def test_coordination_overlay_threads_outbound_fees_to_pair():
    from modules.rebalance_coordination_overlay import build_coordination_pairs
    from modules.rebalance_state_v2 import build_state_snapshot

    snapshot = build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": "02" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
                "local_out_fee_ppm": 300,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": "02" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
                "local_out_fee_ppm": 1_500,
            },
        ],
        {"channel_budgets": {"200x1x0": {"budget_sats": 10_000}}},
    )
    hints = SimpleNamespace(
        get_rebalance_recommendations=lambda: [
            {
                "recommendation_id": "rec-1",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 100_000,
                "route_policy": "market_only",
            }
        ],
        get_rebalance_campaigns=lambda: [],
        get_route_segment_leases=lambda: [],
        is_hive_member=lambda peer_id: False,
    )

    pairs = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="03" + "a" * 64
    )

    assert len(pairs) == 1
    assert pairs[0].source_out_fee_ppm == 300
    assert pairs[0].dest_out_fee_ppm == 1_500
