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


def test_overlay_applies_segment_score_bias_to_coordinated_pair():
    from modules.rebalance_coordination_overlay import build_coordination_pairs

    class SegmentAwareHiveHints(FakeHiveHints):
        def get_segment_score(self, short_channel_id, direction, amount_sats=None):
            if short_channel_id == "100x1x0" and direction == 0:
                return {"net_utility": 0.8, "confidence": 0.8}
            if short_channel_id == "200x1x0" and direction == 1:
                return {"net_utility": 0.8, "confidence": 0.8}
            return {}

    snapshot = _make_snapshot()
    hints = SegmentAwareHiveHints(
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
        snapshot,
        hive_hints=hints,
        our_node_id="01" + "0" * 64,
    )

    assert len(candidates) == 1
    assert candidates[0].score > 250_000
