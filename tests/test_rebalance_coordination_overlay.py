from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority
from modules.rebalance_state_v2 import build_state_snapshot
from modules.rebalance_types_v2 import PairCandidate, PlanResult


def _make_snapshot():
    return build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": "02" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
                "is_hive_member": True,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": "02" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
                "is_hive_member": True,
            },
        ],
        {
            "channel_budgets": {
                "100x1x0": {"budget_sats": 25},
                "200x1x0": {"budget_sats": 40},
            }
        },
    )


class FakeHiveHints:
    def __init__(self, recommendations=None, campaigns=None, leases=None):
        self._recommendations = list(recommendations or [])
        self._campaigns = list(campaigns or [])
        self._leases = list(leases or [])

    def is_fresh(self):
        return True

    def get_rebalance_recommendations(self):
        return list(self._recommendations)

    def get_rebalance_campaigns(self):
        return list(self._campaigns)

    def get_route_segment_leases(self):
        return list(self._leases)

    def get_segment_score(self, short_channel_id, direction, amount_sats=None):
        return {}


def _coordinated_pair():
    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "1" * 64,
        dest_peer_id="02" + "2" * 64,
        amount_sats=120_000,
        pair_budget_sats=40,
        score=0.1,
        reason_code="coordinated_rebalance",
        coordination_hint_type="recommendation",
        coordination_hint_id="rec-1",
        coordination_rank_bonus=90.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HIVE_ONLY,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
            allow_market_fallback=False,
            hint_id="rec-1",
            hint_type="recommendation",
            priority_score=90.0,
        ),
    )


def test_overlay_builds_pair_from_recommendation_source_and_sink_scids():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-1",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "route_policy": "hive_only",
                "priority_score": 90.0,
                "allow_market_fallback": False,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert len(candidates) == 1
    assert candidates[0].coordination_hint_id == "rec-1"
    assert candidates[0].reason_code == "coordinated_rebalance"
    assert candidates[0].route_decision.policy is RoutePolicy.HIVE_ONLY


def test_overlay_pair_budget_uses_sink_budget_not_source_budget():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": "02" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
                "is_hive_member": True,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": "02" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
                "is_hive_member": True,
            },
        ],
        {
            "channel_budgets": {
                "100x1x0": {"budget_sats": 10_000},
                "200x1x0": {"budget_sats": 40},
            }
        },
    )
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-budget",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "route_policy": "hive_only",
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert len(candidates) == 1
    assert candidates[0].pair_budget_sats == 40


def test_engine_preserves_coordinated_candidate_even_when_local_pair_score_is_lower():
    from modules.rebalance_coordination_overlay import merge_coordination_pairs

    local_pair = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="02" + "3" * 64,
        dest_peer_id="02" + "4" * 64,
        amount_sats=200_000,
        pair_budget_sats=50,
        score=10.0,
    )
    coordinated_pair = _coordinated_pair()
    plan = PlanResult(selected=[local_pair], skipped=[])

    selected = merge_coordination_pairs(plan, [coordinated_pair], max_pairs=1)

    assert selected == [coordinated_pair]


def test_overlay_skips_candidate_when_foreign_lease_overlaps_route_segments():
    from modules.rebalance_coordination_overlay import suppress_leased_pairs

    leases = [
        {
            "lease_id": "lease-1",
            "owner_member_id": "02other",
            "route_segments": ["100x1x0>200x1x0"],
        }
    ]

    result = suppress_leased_pairs(
        [_coordinated_pair()],
        leases=leases,
        our_node_id="02ours",
    )

    assert result.selected == []
    assert result.skipped[0].reason == "lease_conflict"
    assert "lease-1" in (result.skipped[0].detail or "")


def test_overlay_keeps_pair_when_overlapping_lease_is_ours():
    from modules.rebalance_coordination_overlay import suppress_leased_pairs

    leases = [
        {
            "lease_id": "lease-1",
            "owner_member_id": "02ours",
            "route_segments": ["100x1x0>200x1x0"],
        }
    ]

    result = suppress_leased_pairs(
        [_coordinated_pair()],
        leases=leases,
        our_node_id="02ours",
    )

    assert len(result.selected) == 1


def test_overlay_score_is_normalized_to_planner_units():
    """Overlay scores are band-normalized ratios in planner units, not raw
    sats. _make_snapshot: drain (0.9-0.65)/0.35 + urgency (0.35-0.1)/0.35
    = ~1.4286 base, priority 90 -> x(1 + 0.90 * 0.15)."""
    import pytest

    from modules.rebalance_coordination_overlay import build_coordination_pairs
    from modules.rebalance_state_v2 import _drain_score, _refill_urgency

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-1",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "route_policy": "hive_only",
                "priority_score": 90.0,
                "allow_market_fallback": False,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    expected_base = _drain_score(0.9, 0.65) + _refill_urgency(0.1, 0.35)
    assert len(candidates) == 1
    assert candidates[0].score == pytest.approx(expected_base * 1.135, rel=1e-4)
    # Comparable with planner units (roughly [0.1, 1.5], never sats-scale).
    assert candidates[0].score < 2.3


def test_overlay_score_priority_multiplier_is_bounded():
    """An absurd producer priority_score clamps to 100 -> max x1.15."""
    import pytest

    from modules.rebalance_coordination_overlay import build_coordination_pairs
    from modules.rebalance_state_v2 import _drain_score, _refill_urgency

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-1",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "priority_score": 1_000_000.0,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    expected_base = _drain_score(0.9, 0.65) + _refill_urgency(0.1, 0.35)
    assert len(candidates) == 1
    assert candidates[0].score == pytest.approx(expected_base * 1.15, rel=1e-4)


def test_overlay_does_not_apply_segment_bias_itself():
    """Segment utility bias is applied once, by the engine, to ALL selected
    pairs. The overlay must not pre-multiply it (double application)."""
    import pytest

    from modules.rebalance_coordination_overlay import build_coordination_pairs

    class SegmentAwareHiveHints(FakeHiveHints):
        def get_segment_score(self, short_channel_id, direction, amount_sats=None):
            if short_channel_id == "100x1x0" and direction == 0:
                return {"net_utility": 0.8, "confidence": 0.8}
            if short_channel_id == "200x1x0" and direction == 1:
                return {"net_utility": 0.8, "confidence": 0.8}
            return {}

    recommendations = [
        {
            "recommendation_id": "rec-1",
            "source_scid": "100x1x0",
            "sink_scid": "200x1x0",
            "amount_sats": 120_000,
            "route_policy": "hive_only",
            "priority_score": 90.0,
            "allow_market_fallback": False,
        }
    ]
    our_node_id = "01" + "0" * 64

    biased = build_coordination_pairs(
        _make_snapshot(),
        hive_hints=SegmentAwareHiveHints(recommendations=recommendations),
        our_node_id=our_node_id,
    )
    unbiased = build_coordination_pairs(
        _make_snapshot(),
        hive_hints=FakeHiveHints(recommendations=recommendations),
        our_node_id=our_node_id,
    )

    assert len(biased) == 1 and len(unbiased) == 1
    assert biased[0].score == pytest.approx(unbiased[0].score)


def test_low_merit_coordination_score_comparable_with_planner_scores():
    """A mildly imbalanced coordination pair scores below a high-merit
    planner pair (planner scores live in roughly [0.1, 1.5])."""
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": "02" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 700_000,  # barely over band: excess_ratio 0.05
                "is_hive_member": True,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": "02" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 300_000,  # barely under band: need_ratio 0.05
                "is_hive_member": True,
            },
        ],
        {"channel_budgets": {"200x1x0": {"budget_sats": 40}}},
    )
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-low",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 40_000,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert len(candidates) == 1
    high_merit_planner_score = 1.2
    assert 0.0 < candidates[0].score < high_merit_planner_score


# -----------------------------------------------------------------------------
# Endpoint binding: hints must resolve to local channels explicitly.
# A foreign-fleet hint (unknown SCID, no peer ids) must never wildcard-bind
# to unrelated local channels.
# -----------------------------------------------------------------------------

def test_overlay_skips_foreign_scid_hint_with_empty_peers():
    from modules.rebalance_coordination_overlay import build_coordination_overlay

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-foreign",
                "source_scid": "999x9x9",
                "sink_scid": "888x8x8",
                "amount_sats": 120_000,
                "route_policy": "hive_only",
                "priority_score": 90.0,
            }
        ]
    )

    result = build_coordination_overlay(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert result.selected == []
    assert len(result.skipped) == 1
    assert result.skipped[0].reason == "coordination_unresolvable_endpoint"
    assert "rec-foreign" in (result.skipped[0].detail or "")


def test_overlay_skips_hint_with_no_endpoints_at_all():
    """A recommendation with neither SCIDs nor peer ids must skip, not
    fabricate a pair from the best local source/sink wildcard pool."""
    from modules.rebalance_coordination_overlay import build_coordination_overlay

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-empty",
                "amount_sats": 120_000,
                "route_policy": "hive_only",
            }
        ]
    )

    result = build_coordination_overlay(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert result.selected == []
    assert result.skipped[0].reason == "coordination_unresolvable_endpoint"


def test_overlay_skips_campaign_without_endpoints_or_active_chunk():
    from modules.rebalance_coordination_overlay import build_coordination_overlay

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        campaigns=[
            {
                "campaign_id": "camp-1",
                "status": "active",
                "amount_sats": 100_000,
            }
        ]
    )

    result = build_coordination_overlay(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert result.selected == []
    assert result.skipped[0].reason == "coordination_unresolvable_endpoint"


def test_overlay_resolves_endpoint_by_peer_when_scid_missing():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-peer",
                "source_peer_id": "02" + "1" * 64,
                "destination_peer_id": "02" + "2" * 64,
                "amount_sats": 120_000,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert len(candidates) == 1
    assert candidates[0].source_channel_id == "100x1x0"
    assert candidates[0].dest_channel_id == "200x1x0"


def test_overlay_does_not_wildcard_when_scid_unknown_but_peer_given():
    """An SCID that names a channel we do not have means the hint targets a
    different node; do not silently rebind it to another local channel."""
    from modules.rebalance_coordination_overlay import build_coordination_overlay

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-mismatch",
                "source_scid": "999x9x9",
                "source_peer_id": "02" + "1" * 64,
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
            }
        ]
    )

    result = build_coordination_overlay(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert result.selected == []
    assert result.skipped[0].reason == "coordination_unresolvable_endpoint"


# -----------------------------------------------------------------------------
# Executor gating: hints designating another member as executor must not run
# here unless we appear in the fallback list.
# -----------------------------------------------------------------------------

def test_overlay_skips_hint_designating_another_executor():
    from modules.rebalance_coordination_overlay import build_coordination_overlay

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-exec",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "primary_executor_member_id": "02someoneelse",
                "fallback_executor_member_ids": ["02notus"],
            }
        ]
    )

    result = build_coordination_overlay(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert result.selected == []
    assert result.skipped[0].reason == "not_designated_executor"


def test_overlay_allows_fallback_executor():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-fallback",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "primary_executor_member_id": "02someoneelse",
                "fallback_executor_member_ids": ["02ours"],
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert len(candidates) == 1


def test_overlay_no_executor_gating_when_fields_missing():
    """Back-compat: hints without executor fields are open to any member."""
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-open",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "primary_executor_member_id": "",
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert len(candidates) == 1


# -----------------------------------------------------------------------------
# Phase B.f (2026-04-23): reserved coordination slots bypass max_pairs cap.
# -----------------------------------------------------------------------------

def _plain_pair(src: str, dst: str, score: float = 10.0) -> PairCandidate:
    """Build a non-coordinated planner pair for cap-boundary tests."""
    return PairCandidate(
        source_channel_id=src,
        dest_channel_id=dst,
        source_peer_id="02" + (src * 64)[:64],
        dest_peer_id="02" + (dst * 64)[:64],
        amount_sats=100_000,
        pair_budget_sats=50,
        score=score,
    )


def _coordination_pair(idx: int) -> PairCandidate:
    """Build a coordination pair with unique src/dst so dedup doesn't fire."""
    return PairCandidate(
        source_channel_id=f"9{idx}0x1x0",
        dest_channel_id=f"8{idx}0x1x0",
        source_peer_id="02" + f"c{idx}" + "0" * 60,
        dest_peer_id="02" + f"d{idx}" + "0" * 60,
        amount_sats=100_000,
        pair_budget_sats=50,
        score=0.1,
        reason_code="coordinated_rebalance",
        coordination_hint_type="recommendation",
        coordination_hint_id=f"rec-{idx}",
        coordination_rank_bonus=90.0,
    )


def test_reserved_slots_zero_preserves_strict_cap():
    """Default (0 reserved) keeps the pre-Phase-B.f semantics: planner pairs
    can occupy the full max_pairs slots; coordination pairs displace planner
    pairs rather than bypassing the cap."""
    from modules.rebalance_coordination_overlay import merge_coordination_pairs

    plan = PlanResult(
        selected=[_plain_pair("p1", "q1"), _plain_pair("p2", "q2")],
        skipped=[],
    )
    coord = [_coordination_pair(1), _coordination_pair(2)]

    selected = merge_coordination_pairs(
        plan, coord, max_pairs=2, coordination_reserved_slots=0
    )
    # 2 coordination pairs fill the 2-slot cap; both planner pairs displaced.
    assert len(selected) == 2
    assert all(p.coordination_hint_id for p in selected)


def test_reserved_slots_let_coordination_bypass_cap():
    """Phase B.f: with reserved=2 and max_pairs=2, two coordination pairs
    bypass the cap; planner pairs can still fill the base cap."""
    from modules.rebalance_coordination_overlay import merge_coordination_pairs

    plan = PlanResult(
        selected=[_plain_pair("p1", "q1"), _plain_pair("p2", "q2")],
        skipped=[],
    )
    coord = [_coordination_pair(1), _coordination_pair(2)]

    selected = merge_coordination_pairs(
        plan, coord, max_pairs=2, coordination_reserved_slots=2
    )
    # Coordination uses 2 reserved slots (total 2); plan pairs fill 2 base
    # slots -> 4 selected total (2 coord + 2 plan, no collisions).
    assert len(selected) == 4
    coord_ids = {p.coordination_hint_id for p in selected if p.coordination_hint_id}
    assert coord_ids == {"rec-1", "rec-2"}


def test_plan_pairs_still_respect_max_pairs_when_reserved_unused():
    """If the overlay has no coordination pairs, the reserved slots stay
    idle — planner pairs cannot expand into them."""
    from modules.rebalance_coordination_overlay import merge_coordination_pairs

    plan = PlanResult(
        selected=[
            _plain_pair("p1", "q1"),
            _plain_pair("p2", "q2"),
            _plain_pair("p3", "q3"),
        ],
        skipped=[],
    )

    selected = merge_coordination_pairs(
        plan, [], max_pairs=2, coordination_reserved_slots=2
    )
    # No coordination pairs; plan respects max_pairs=2 only.
    assert len(selected) == 2


def test_coordination_cannot_exceed_reserved_plus_max():
    """10 coordination pairs with reserved=2 and max_pairs=3 admit at most 5."""
    from modules.rebalance_coordination_overlay import merge_coordination_pairs

    plan = PlanResult(selected=[], skipped=[])
    coord = [_coordination_pair(i) for i in range(10)]

    selected = merge_coordination_pairs(
        plan, coord, max_pairs=3, coordination_reserved_slots=2
    )
    assert len(selected) == 5


def test_negative_reserved_clamped_to_zero():
    """Defensive: a misconfigured negative value doesn't inflate the cap."""
    from modules.rebalance_coordination_overlay import merge_coordination_pairs

    plan = PlanResult(selected=[_plain_pair("p1", "q1")], skipped=[])
    selected = merge_coordination_pairs(
        plan, [], max_pairs=2, coordination_reserved_slots=-5
    )
    # Negative clamps to 0; no change vs the default.
    assert len(selected) == 1


# -----------------------------------------------------------------------------
# Budget parity: coordination pairs get the same pair_fee_cap_ppm floor the
# planner gives its pairs.
# -----------------------------------------------------------------------------

def test_overlay_pair_budget_gets_fee_cap_floor_like_planner_pairs():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-floor",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot,
        hive_hints=hints,
        our_node_id="02ours",
        pair_fee_cap_ppm=1_000,
    )

    assert len(candidates) == 1
    # max(sink budget 40, ceil(120_000 * 1000 / 1e6) = 120) = 120
    assert candidates[0].pair_budget_sats == 120


def test_overlay_pair_budget_unchanged_when_fee_cap_disabled():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = _make_snapshot()
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-nofloor",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot, hive_hints=hints, our_node_id="02ours"
    )

    assert len(candidates) == 1
    assert candidates[0].pair_budget_sats == 40


def test_overlay_keeps_larger_sink_budget_over_fee_cap_floor():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    snapshot = build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": "02" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": "02" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
            },
        ],
        {"channel_budgets": {"200x1x0": {"budget_sats": 5_000}}},
    )
    hints = FakeHiveHints(
        recommendations=[
            {
                "recommendation_id": "rec-bigbudget",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
            }
        ]
    )

    candidates = build_coordination_pairs(
        snapshot,
        hive_hints=hints,
        our_node_id="02ours",
        pair_fee_cap_ppm=1_000,
    )

    assert len(candidates) == 1
    assert candidates[0].pair_budget_sats == 5_000
