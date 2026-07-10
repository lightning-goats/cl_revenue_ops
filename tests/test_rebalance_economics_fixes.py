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
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=5000)
    pair = _make_pair(
        amount_sats=1_000_000,
        pair_budget_sats=5_000,
        dest_out_fee_ppm=2_500,
        source_out_fee_ppm=100,
        # E-4.3: "earning 2,500 ppm" means VALIDATED realized history —
        # an unvalidated advertised ask is now discounted by the EV gate.
        dest_fee_history_validated=True,
        dest_historical_direct_fee_ppm=2_500,
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
            # E-4.3: validated history keeps the 2-sat value exact.
            dest_fee_history_validated=True,
            dest_historical_direct_fee_ppm=20,
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


def test_f2e_break_even_paid_route_is_approved(mock_plugin, mock_database):
    """Default margin 0 should execute paid routes whose sats EV is exactly 0.

    This is the more-rebalancing/no-negative-profit boundary: route cost equals
    probability-adjusted expected future value, so the route is break-even.
    """
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=1_000_000,
        pair_budget_sats=1_000,
        dest_out_fee_ppm=200,  # expected value = 100 sats; 0.99 * 100 = 99
        source_out_fee_ppm=0,
        # E-4.3: validated history keeps the break-even value exact.
        dest_fee_history_validated=True,
        dest_historical_direct_fee_ppm=200,
    )

    selected = _run_single_pair(
        engine, pair, route_cost_sats=99, probability_ppm=990_000
    )

    assert selected == [pair]
    decomp = pair.score_decomposition
    assert decomp["final_score_sats"] == pytest.approx(0.0)
    assert decomp["beats_do_nothing"] is True


def test_f2f_historical_destination_and_source_value_feed_sats_ev(
    mock_plugin, mock_database
):
    """Historical role profitability should improve the local EV estimate.

    The current destination fee is zero, so the old model sees no future value.
    Historical direct destination earnings plus source sourced-fee contribution
    make the move positive without relying on hive hints or raising budgets.
    """
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=1_000_000,
        pair_budget_sats=1_000,
        dest_out_fee_ppm=0,
        source_out_fee_ppm=0,
    )
    pair.dest_historical_direct_fee_ppm = 200.0
    pair.source_historical_sourced_fee_ppm = 100.0

    selected = _run_single_pair(
        engine, pair, route_cost_sats=120, probability_ppm=990_000
    )

    assert selected == [pair]
    decomp = pair.score_decomposition
    assert decomp["destination_refill_value_sats"] == pytest.approx(100.0)
    assert decomp["source_drain_value_sats"] == pytest.approx(50.0)
    assert decomp["expected_future_value_sats"] == pytest.approx(150.0)
    assert decomp["final_score_sats"] == pytest.approx(28.5)


# ---------------------------------------------------------------------------
# F1 — per-attempt fee ceiling: the 30-day budget is the reservation cap,
# the ppm cap bounds any single attempt
# ---------------------------------------------------------------------------


def test_f1_budget_does_not_double_as_fee_envelope(mock_plugin, mock_database):
    """Audit case: a 2,500-sat capex budget must not approve a 25,000-ppm
    fee on a 100k move. With pair_fee_cap_ppm=1000 (default) the
    per-attempt ceiling on 100k is 100 sats, so a 150-sat route is
    rejected even though the budget would cover it."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=100_000,
        pair_budget_sats=2_500,
        dest_out_fee_ppm=5_000,
    )

    selected = _run_single_pair(engine, pair, route_cost_sats=150)

    assert selected == []
    assert _skip_reasons(engine).get("200x2x0") == "route_over_budget"


def test_f1_execution_honors_ppm_ceiling_not_budget(mock_plugin, mock_database):
    """An accepted route's execution fee ceiling is the per-attempt cap
    (min(budget, ceil(amount * ppm / 1e6))), not the full pair budget."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=100_000,
        pair_budget_sats=2_500,
        dest_out_fee_ppm=5_000,
    )

    selected = _run_single_pair(engine, pair, route_cost_sats=50)

    assert selected == [pair]
    assert int(pair.effective_budget_sats) == 100
    assert engine._pair_max_fee_sats(pair) == 100
    assert engine._execution_kwargs(pair)["max_fee_sats"] == 100


def test_f1_zero_ppm_cap_keeps_budget_envelope(mock_plugin, mock_database):
    """pair_fee_cap_ppm == 0 disables the ceiling (legacy behavior: the
    budget is the per-attempt envelope)."""
    engine = _make_engine(
        mock_plugin, mock_database, pair_fee_cap_ppm=0, max_fee_ppm=5000
    )
    pair = _make_pair(
        amount_sats=100_000,
        pair_budget_sats=2_500,
        dest_out_fee_ppm=4_000,
        # E-4.3: keep this pair EV-positive via validated history so the
        # test still exercises the budget-envelope path it documents.
        dest_fee_history_validated=True,
        dest_historical_direct_fee_ppm=4_000,
    )

    selected = _run_single_pair(engine, pair, route_cost_sats=150)

    assert selected == [pair]
    assert engine._pair_max_fee_sats(pair) == 2_500


def test_f1_small_budget_remains_binding_below_ppm_cap(
    mock_plugin, mock_database
):
    """The ceiling never exceeds the pair budget: a 30-sat budget with a
    1000-ppm cap on 100k (100 sats) still only authorizes 30 sats."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=100_000,
        pair_budget_sats=30,
        dest_out_fee_ppm=5_000,
    )

    selected = _run_single_pair(engine, pair, route_cost_sats=50)

    assert selected == []
    assert _skip_reasons(engine).get("200x2x0") == "route_over_budget"


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
# F4 — overlay/equalization scores use the planner's coefficients
# (0.30 x urgency + 0.20 x drain), not coefficient 1.0 (~2x planner scale)
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# F7 — destination cooldown scales with the last rebalance's fill fraction
# ---------------------------------------------------------------------------


def _f7_engine(mock_plugin, mock_database, *, local_sats, anchor, last_secs_ago):
    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = type(engine)._build_snapshot.__get__(engine)
    scid = "100x1x0"
    mock_database.get_last_rebalance_times.return_value = {
        scid: int(time.time()) - last_secs_ago
    }
    mock_database.get_last_post_rebalance_state.return_value = anchor
    mock_plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "1" * 64,
                "short_channel_id": scid,
                "total_msat": 1_000_000_000,  # 1M sats capacity
                "our_amount_msat": local_sats * 1000,
                "updates": {
                    "local": {"fee_proportional_millionths": 1_000},
                    "remote": {"fee_proportional_millionths": 100},
                },
            }
        ]
    }
    return engine


def test_f7_small_partial_fill_releases_cooldown_quickly(
    mock_plugin, mock_database
):
    """A 5k fill against a ~300k remaining band gap is a ~1.6% fill: the
    effective cooldown shrinks to minutes, so 2 hours later the dest is
    eligible again instead of blocked for 24h."""
    engine = _f7_engine(
        mock_plugin,
        mock_database,
        local_sats=50_000,  # 5% of 1M capacity; band low 0.35 → gap 300k
        anchor={
            "timestamp": int(time.time()) - 2 * 3600,
            "post_local_ratio": 0.055,
            "amount_sats": 5_000,
        },
        last_secs_ago=2 * 3600,
    )

    snapshot = engine._build_snapshot()

    assert snapshot.channels[0].cooldown_active is False


def test_f7_full_fill_keeps_full_cooldown(mock_plugin, mock_database):
    """A refill that closed the band gap keeps the full 24h cooldown."""
    engine = _f7_engine(
        mock_plugin,
        mock_database,
        local_sats=400_000,  # 40% local: at/above band low → no remaining gap
        anchor={
            "timestamp": int(time.time()) - 2 * 3600,
            "post_local_ratio": 0.40,
            "amount_sats": 400_000,
        },
        last_secs_ago=2 * 3600,
    )

    snapshot = engine._build_snapshot()

    assert snapshot.channels[0].cooldown_active is True


def test_f7_majority_fill_keeps_full_cooldown(mock_plugin, mock_database):
    """A >=50% fill fraction keeps the full base cooldown (min(1, 2f))."""
    engine = _f7_engine(
        mock_plugin,
        mock_database,
        local_sats=150_000,  # gap 200k; 300k fill → fraction 0.6 → full
        anchor={
            "timestamp": int(time.time()) - 2 * 3600,
            "post_local_ratio": 0.15,
            "amount_sats": 300_000,
        },
        last_secs_ago=2 * 3600,
    )

    snapshot = engine._build_snapshot()

    assert snapshot.channels[0].cooldown_active is True


def test_f7_no_anchor_keeps_full_cooldown(mock_plugin, mock_database):
    """Without a post-rebalance anchor the conservative full cooldown
    applies (no schema change, fail-safe)."""
    engine = _f7_engine(
        mock_plugin,
        mock_database,
        local_sats=50_000,
        anchor=None,
        last_secs_ago=2 * 3600,
    )

    snapshot = engine._build_snapshot()

    assert snapshot.channels[0].cooldown_active is True


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


def test_snapshot_builder_falls_back_to_top_level_fee(mock_plugin, mock_database):
    """New channels often have no gossip updates yet (updates.local absent):
    the snapshot builder must fall back to the top-level
    fee_proportional_millionths — the same fallback
    fee_controller._get_channels_info uses — otherwise new-channel pairs are
    silently EV-zeroed (local_out_fee_ppm 0 kills the sats-EV gate)."""
    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = type(engine)._build_snapshot.__get__(engine)
    mock_database.get_last_rebalance_times.return_value = {}
    mock_plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {   # No updates at all (brand new channel)
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "1" * 64,
                "short_channel_id": "100x1x0",
                "total_msat": 1_000_000_000,
                "our_amount_msat": 800_000_000,
                "fee_proportional_millionths": 1_234,
            },
            {   # updates present but local missing (remote gossip only)
                "state": "CHANNELD_NORMAL",
                "peer_id": "02" + "2" * 64,
                "short_channel_id": "200x1x0",
                "total_msat": 1_000_000_000,
                "our_amount_msat": 800_000_000,
                "fee_proportional_millionths": 555,
                "updates": {
                    "remote": {"fee_proportional_millionths": 90},
                },
            },
        ]
    }

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    by_id = {c.channel_id: c for c in snapshot.channels}
    assert by_id["100x1x0"].local_out_fee_ppm == 1_234
    assert by_id["200x1x0"].local_out_fee_ppm == 555
    assert by_id["200x1x0"].actual_inbound_fee_ppm == 90


def test_snapshot_builder_prefers_updates_local_over_top_level(
    mock_plugin, mock_database
):
    """When updates.local is present it wins over a stale top-level field."""
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
                "fee_proportional_millionths": 999,
                "updates": {
                    "local": {"fee_proportional_millionths": 1_750},
                },
            }
        ]
    }

    snapshot = engine._build_snapshot()

    assert snapshot.channels[0].local_out_fee_ppm == 1_750


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




# ---------------------------------------------------------------------------
# E-4.3 (2026-07 econ audit) — optimistic EV: the advertised (unvalidated)
# destination fee must not justify spend on its own
# ---------------------------------------------------------------------------


def test_e43_unvalidated_advertised_fee_is_discounted(mock_plugin, mock_database):
    """No validated forward history: the 2,000 ppm ask is optimism and the
    value anchor uses only half of it. Under the old max(current, realized)
    a never-cleared ask fully justified the route cost."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=1_000_000,
        pair_budget_sats=5_000,
        dest_out_fee_ppm=2_000,
        source_out_fee_ppm=0,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=700,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    assert decomp["inputs"]["dest_fee_history_validated"] is False
    assert decomp["inputs"]["dest_value_fee_ppm"] == pytest.approx(1_000.0)
    # 0.9 * (1M * 1000ppm * 0.5) - 700 = -250: the hopeful ask no longer
    # buys a 700-sat route.
    assert decomp["final_score_sats"] < 0.0


def test_e43_validated_realized_rate_overrides_advertised_ask(mock_plugin, mock_database):
    """>5 observed forwards: the realized 100 ppm is the truth even when the
    advertised policy asks 2,000 ppm."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=1_000_000,
        pair_budget_sats=5_000,
        dest_out_fee_ppm=2_000,
        source_out_fee_ppm=0,
        dest_fee_history_validated=True,
        dest_historical_direct_fee_ppm=100.0,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    assert decomp["inputs"]["dest_fee_history_validated"] is True
    assert decomp["inputs"]["dest_value_fee_ppm"] == pytest.approx(100.0)


def test_e43_thin_realized_evidence_still_counts(mock_plugin, mock_database):
    """Unvalidated + a thin realized rate above the discounted ask: the
    realized evidence wins the max()."""
    engine = _make_engine(mock_plugin, mock_database)
    pair = _make_pair(
        amount_sats=1_000_000,
        pair_budget_sats=5_000,
        dest_out_fee_ppm=1_000,
        source_out_fee_ppm=0,
        dest_fee_history_validated=False,
        dest_historical_direct_fee_ppm=800.0,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    # max(1000 * 0.5, 800) = 800
    assert decomp["inputs"]["dest_value_fee_ppm"] == pytest.approx(800.0)


def test_e43_is_active_threads_from_snapshot_to_pair():
    """ChannelState.is_active (>5 lifetime forwards) must reach the
    PairCandidate as dest_fee_history_validated."""
    import dataclasses
    from modules.rebalance_planner_v2 import RebalancePlanner
    from modules.rebalance_state_v2 import build_state_snapshot

    snap = build_state_snapshot(
        channels=[
            {
                "channel_id": "src", "peer_id": "02" + "aa" * 32,
                "capacity_sats": 2_000_000, "local_sats": 1_700_000,
                "local_out_fee_ppm": 500, "is_active": True,
            },
            {
                "channel_id": "dst", "peer_id": "02" + "bb" * 32,
                "capacity_sats": 2_000_000, "local_sats": 100_000,
                "local_out_fee_ppm": 800, "is_active": True,
            },
        ],
        capex_allocations={"channel_budgets": {"src": 1_000, "dst": 1_000}},
    )
    by_id = {ch.channel_id: ch for ch in snap.channels}
    assert by_id["src"].is_active is True
    assert by_id["dst"].is_active is True

    planner = RebalancePlanner(
        target_band_low=0.35, target_band_high=0.65, max_chunk_sats=10_000_000,
    )
    result = planner.plan(snap)
    assert len(result.selected) == 1
    pair = result.selected[0]
    assert pair.dest_channel_id == "dst"
    assert pair.dest_fee_history_validated is True
