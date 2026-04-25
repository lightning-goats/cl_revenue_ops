"""Oracle-style impact tests for cl-hive hints consumed by cl_revenue_ops.

These tests do not try to prove cl-hive's internal assignment algorithm.
They prove that once cl-hive exports a deterministic hint snapshot, each
cl_revenue_ops consumer moves in the expected direction and stale/malformed
hints fail neutral.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.capex_budget import CapexBudgetEngine
from modules.capacity_planner import CapacityPlanner
from modules.fee_controller import FeeController
from modules.hive_hints import HiveHintAdapter
from modules.rebalance_coordination_overlay import build_coordination_pairs
from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_route_policy import RoutePolicy, decide_route_policy
from modules.rebalance_state_v2 import build_state_snapshot


OWNER = "02" + "1" * 64
SECONDARY = "02" + "2" * 64
CONTESTED = "02" + "3" * 64
OPEN_TARGET = "02" + "4" * 64
AVOID_TARGET = "02" + "5" * 64
CLOSURE_TARGET = "02" + "6" * 64


def _snapshot(*, generated_at: int | None = None) -> dict:
    now = int(time.time()) if generated_at is None else generated_at
    return {
        "generated_at": now,
        "ttl_seconds": 900,
        "route_segment_leases": [
            {
                "lease_id": "lease-owner",
                "route_segments": ["100x1x0->200x1x0"],
                "owner_member_id": OWNER,
            }
        ],
        "rebalance_recommendations": [
            {
                "recommendation_id": "rec-hive-only",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "route_policy": "hive_only",
                "allow_market_fallback": False,
                "priority_score": 90.0,
            }
        ],
        "segment_scores": [
            {
                "short_channel_id": "100x1x0",
                "direction": 0,
                "amount_bucket_sats": 100_000,
                "success_score": 0.9,
                "failure_score": 0.0,
                "net_utility": 0.9,
                "confidence": 0.8,
                "observer_count": 3,
                "last_observed_at": now,
            }
        ],
        "hints": {
            OWNER: {
                "member": True,
                "corridor_role": "owner",
                "competition_bias": 1,
                "traffic_confidence": 0.9,
                "rebalance_preference": "sink",
                "peer_quality_score": 0.9,
                "external_centrality": 0.06,
                "fee_elasticity": 0.4,
                "optimal_fee_estimate_ppm": 81,
                "fleet_fee_median": 75,
                "reputation_score": 85,
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 0.8,
                    "suggested_size_bucket": "medium",
                    "reason": "member_connectivity",
                },
            },
            SECONDARY: {
                "member": True,
                "corridor_role": "secondary",
                "competition_bias": -1,
                "traffic_confidence": 0.9,
                "rebalance_preference": "source",
                "peer_quality_score": 0.2,
                "fleet_fee_median": 120,
                "reputation_score": 45,
            },
            CONTESTED: {
                "member": True,
                "corridor_role": "contested",
                "competition_bias": 0,
                "traffic_confidence": 0.9,
                "rebalance_preference": "neutral",
            },
            OPEN_TARGET: {
                "member": False,
                "corridor_role": "none",
                "competition_bias": 0,
                "traffic_confidence": 0.4,
                "reputation_score": 80,
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 0.75,
                    "suggested_size_bucket": "large",
                    "reason": "underserved_corridor",
                },
            },
            AVOID_TARGET: {
                "member": False,
                "corridor_role": "none",
                "competition_bias": 0,
                "traffic_confidence": 0.4,
                "channel_open_hint": {
                    "open_preference": "avoid",
                    "topology_confidence": 0.9,
                    "suggested_size_bucket": "small",
                    "reason": "reduce_overlap",
                },
            },
            CLOSURE_TARGET: {
                "member": False,
                "corridor_role": "none",
                "competition_bias": 0,
                "traffic_confidence": 0.3,
                "closure_recommended": True,
                "closure_reason": "poor_fleet_success_rate",
            },
        },
    }


def _adapter(snapshot: dict | None = None) -> HiveHintAdapter:
    plugin = MagicMock()
    plugin.rpc.call.return_value = snapshot or _snapshot()
    adapter = HiveHintAdapter(plugin, ttl_override=0)
    adapter.poll()
    return adapter


def test_hive_hint_oracle_normalizes_corridor_fee_rebalance_and_open_signals():
    adapter = _adapter()

    assert adapter.get_corridor_role(OWNER) == "owner"
    assert adapter.get_corridor_role(SECONDARY) == "secondary"
    assert adapter.get_corridor_role(CONTESTED) == "contested"

    assert adapter.get_fee_bias(OWNER) > 1.0
    assert adapter.get_fee_bias(SECONDARY) < 1.0
    assert adapter.get_fee_bias(CONTESTED) == pytest.approx(1.0)

    assert adapter.get_rebalance_bias(OWNER) > 1.0
    assert adapter.get_rebalance_bias(SECONDARY) < 1.0
    assert adapter.get_corridor_utilization_bias(OWNER) > 1.0
    assert adapter.get_corridor_utilization_bias(SECONDARY) < 1.0

    assert adapter.get_optimal_fee_estimate(OWNER) == 81
    assert adapter.get_fleet_fee_prior(OWNER) == 75
    assert adapter.is_closure_recommended(CLOSURE_TARGET) is True
    assert adapter.get_segment_score("100:1:0", 0, amount_sats=120_000)["net_utility"] == 0.9

    open_peer_ids = {peer_id for peer_id, _hint in adapter.get_open_candidates()}
    assert OWNER in open_peer_ids
    assert OPEN_TARGET in open_peer_ids
    assert AVOID_TARGET not in open_peer_ids


def test_fee_controller_uses_hive_as_bounded_bias_and_missing_gossip_boundary_fallback():
    cfg = SimpleNamespace(
        vegas_decay_rate=0.1,
        fee_market_boundary_enabled=True,
    )
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02ours"}
    controller = FeeController(plugin, cfg, MagicMock())
    controller.hive_hints = _adapter()

    assert controller._get_hive_fee_bias(OWNER) > 1.0
    assert controller._get_hive_fee_bias(SECONDARY) < 1.0
    assert controller._get_hive_exploration_multiplier(OWNER) > 1.0

    boundary = controller._get_hive_market_boundary_fee(OWNER, cfg)
    assert boundary["boundary_ppm"] == 81
    assert boundary["source"] == "hive_optimal_fee_estimate"
    assert boundary["traffic_confidence"] == pytest.approx(0.9)


def test_rebalancer_policy_and_coordination_follow_hive_recommendation_oracle():
    adapter = _adapter()
    snapshot = build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": OWNER,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
                "is_hive_member": adapter.is_hive_member(OWNER),
                "actual_inbound_fee_ppm": 1,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": SECONDARY,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
                "is_hive_member": adapter.is_hive_member(SECONDARY),
            },
        ],
        {"channel_budgets": {"200x1x0": {"budget_sats": 100}}},
    )

    planned = RebalancePlanner(
        max_pairs=5,
        max_chunk_sats=120_000,
        pair_fee_cap_ppm=1000,
    ).plan(snapshot)
    assert planned.selected
    pair = planned.selected[0]
    assert pair.source_channel_id == "100x1x0"
    assert pair.dest_channel_id == "200x1x0"
    assert pair.source_value_class == "hive"
    assert pair.dest_value_class == "hive"

    decision = decide_route_policy(pair, hive_hints=adapter)
    assert decision.policy is RoutePolicy.HIVE_ONLY
    assert decision.allow_market_fallback is False
    assert decision.hint_id == "rec-hive-only"

    coordination_pairs = build_coordination_pairs(
        snapshot,
        hive_hints=adapter,
        our_node_id=OWNER,
    )
    assert len(coordination_pairs) == 1
    assert coordination_pairs[0].coordination_hint_id == "rec-hive-only"
    assert coordination_pairs[0].route_decision.policy is RoutePolicy.HIVE_ONLY


def _planner(adapter: HiveHintAdapter) -> CapacityPlanner:
    plugin = MagicMock()
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listnodes.return_value = {"nodes": []}

    profitability = MagicMock()
    profitability.database.get_peer_reputation.return_value = None
    profitability.database.get_peer_closed_channel_profit_summary.return_value = None
    profitability.database.get_peer_uptime_percent.return_value = None
    profitability.database.get_historical_inbound_fee_ppm.return_value = None

    planner = CapacityPlanner(plugin, profitability, MagicMock())
    planner.hive_hints = adapter
    return planner


def test_planner_scores_hive_open_hints_and_avoid_hints_in_opposite_directions():
    planner = _planner(_adapter())

    owner_score = planner._score_candidate(OWNER, 1.0)
    open_score = planner._score_candidate(OPEN_TARGET, 1.0)
    avoid_score = planner._score_candidate(AVOID_TARGET, 1.0)

    assert owner_score > 1.0
    assert open_score > 1.0
    assert avoid_score < 1.0

    discovered = {peer_id for peer_id, _hint in planner.hive_hints.get_open_candidates()}
    assert {OWNER, OPEN_TARGET}.issubset(discovered)
    assert AVOID_TARGET not in discovered


def test_capex_owner_and_member_hints_increase_budget_without_changing_baseline():
    adapter = _adapter()
    cfg = SimpleNamespace(
        capex_grace_days=14,
        capex_reinvestment_rate=0.5,
        capex_bootstrap_bps=50,
        capex_bootstrap_max_sats=1_000,
    )
    database = MagicMock()
    database.get_channel_rebalance_success_rate.return_value = None
    engine = CapexBudgetEngine(MagicMock(), database, cfg, hive_hints=adapter)
    prof = SimpleNamespace(
        classification="active",
        revenue=SimpleNamespace(
            total_contribution_msat=1_000_000,
            total_forward_count=10,
        ),
        days_open=30,
        marginal_roi=0.0,
        capacity_sats=1_000_000,
    )

    owner_budget = engine._compute_channel_budget(
        "owner-scid",
        SimpleNamespace(**{**prof.__dict__, "peer_id": OWNER}),
        0,
        "none",
        cfg,
    )
    secondary_budget = engine._compute_channel_budget(
        "secondary-scid",
        SimpleNamespace(**{**prof.__dict__, "peer_id": SECONDARY}),
        0,
        "none",
        cfg,
    )
    baseline_budget = engine._compute_channel_budget(
        "baseline-scid",
        SimpleNamespace(**{**prof.__dict__, "peer_id": OPEN_TARGET}),
        0,
        "none",
        cfg,
    )

    assert owner_budget.hive_multiplier == 2.0
    assert secondary_budget.hive_multiplier == 1.5
    assert baseline_budget.hive_multiplier == 1.0
    assert owner_budget.budget_sats > secondary_budget.budget_sats > baseline_budget.budget_sats


def test_stale_or_malformed_hints_fail_neutral():
    stale = _adapter(_snapshot(generated_at=int(time.time()) - 10_000))
    assert stale.is_usable() is False
    assert stale.get_fee_bias(OWNER) == 1.0
    assert stale.get_rebalance_recommendations() == []
    assert stale.get_open_candidates() == []

    malformed_snapshot = _snapshot()
    malformed_snapshot["hints"][OWNER]["corridor_role"] = "owner<script>"
    malformed = _adapter(malformed_snapshot)
    assert malformed.get_corridor_role(OWNER) == "none"
    assert malformed.get_corridor_utilization_bias(OWNER) == 1.0
