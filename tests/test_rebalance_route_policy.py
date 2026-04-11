"""Red tests for planned hive route-policy support."""

import pytest


def _import_route_policy():
    try:
        from modules.rebalance_route_policy import (  # type: ignore
            RouteDecision,
            RoutePolicy,
            RoutePriority,
            decide_route_policy,
        )
    except Exception as e:  # pragma: no cover - red test bootstrap
        pytest.fail(f"route policy support missing: {e}")
    return RouteDecision, RoutePolicy, RoutePriority, decide_route_policy


class FakeHiveHints:
    def __init__(self, members=None, recommendations=None, campaigns=None):
        self._members = set(members or set())
        self._recommendations = list(recommendations or [])
        self._campaigns = list(campaigns or [])

    def is_hive_member(self, peer_id: str) -> bool:
        return peer_id in self._members

    def get_rebalance_recommendations(self):
        return list(self._recommendations)

    def get_rebalance_campaigns(self):
        return list(self._campaigns)


def test_decide_route_policy_requires_hive_for_hive_equalization():
    from modules.rebalance_types_v2 import PairCandidate

    _, RoutePolicy, _, decide_route_policy = _import_route_policy()

    pair = PairCandidate(
        source_channel_id="1x1x1",
        dest_channel_id="2x2x2",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        pair_budget_sats=10,
    )
    hints = FakeHiveHints(members={pair.source_peer_id, pair.dest_peer_id})

    decision = decide_route_policy(
        pair,
        reason_code="hive_equalization",
        hive_hints=hints,
    )

    assert decision.policy is RoutePolicy.HIVE_ONLY
    assert decision.allow_market_fallback is False


def test_decide_route_policy_marks_hybrid_when_hints_prefer_fleet():
    from modules.rebalance_types_v2 import PairCandidate

    _, RoutePolicy, _, decide_route_policy = _import_route_policy()

    pair = PairCandidate(
        source_channel_id="3x3x3",
        dest_channel_id="4x4x4",
        source_peer_id="02" + "c" * 64,
        dest_peer_id="02" + "d" * 64,
        amount_sats=50_000,
        pair_budget_sats=25,
    )
    hints = FakeHiveHints(
        members={pair.source_peer_id},
        recommendations=[{
            "source_peer_id": pair.source_peer_id,
            "destination_peer_id": pair.dest_peer_id,
            "route_policy": "hybrid",
        }],
    )

    decision = decide_route_policy(
        pair,
        reason_code="ev_positive",
        hive_hints=hints,
    )

    assert decision.policy is RoutePolicy.HYBRID


def test_decide_route_policy_defaults_to_hive_only_when_both_endpoints_are_hive():
    from modules.rebalance_types_v2 import PairCandidate

    _, RoutePolicy, RoutePriority, decide_route_policy = _import_route_policy()

    pair = PairCandidate(
        source_channel_id="7x7x7",
        dest_channel_id="8x8x8",
        source_peer_id="02" + "7" * 64,
        dest_peer_id="02" + "8" * 64,
        amount_sats=25_000,
        pair_budget_sats=10,
    )
    hints = FakeHiveHints(members={pair.source_peer_id, pair.dest_peer_id})

    decision = decide_route_policy(
        pair,
        reason_code="ev_positive",
        hive_hints=hints,
    )

    assert decision.policy is RoutePolicy.HIVE_ONLY
    assert decision.priority is RoutePriority.HIVE_EQUALIZATION
    assert decision.allow_market_fallback is False


def test_pair_candidate_can_store_route_decision():
    from modules.rebalance_types_v2 import PairCandidate

    RouteDecision, RoutePolicy, RoutePriority, _ = _import_route_policy()

    decision = RouteDecision(
        policy=RoutePolicy.HIVE_ONLY,
        priority=RoutePriority.HIVE_EQUALIZATION,
        reason="hive_equalization",
        allow_market_fallback=False,
    )

    pair = PairCandidate(
        source_channel_id="5x5x5",
        dest_channel_id="6x6x6",
        source_peer_id="02" + "e" * 64,
        dest_peer_id="02" + "f" * 64,
        amount_sats=75_000,
        pair_budget_sats=30,
        route_decision=decision,
    )

    assert pair.route_decision is decision
    assert pair.route_decision.policy is RoutePolicy.HIVE_ONLY


def test_decide_route_policy_matches_scid_based_recommendations():
    from modules.rebalance_types_v2 import PairCandidate

    _, RoutePolicy, RoutePriority, decide_route_policy = _import_route_policy()

    pair = PairCandidate(
        source_channel_id="11x11x11",
        dest_channel_id="12x12x12",
        source_peer_id="02" + "1" * 64,
        dest_peer_id="02" + "2" * 64,
        amount_sats=42_000,
        pair_budget_sats=30,
    )
    hints = FakeHiveHints(
        recommendations=[{
            "recommendation_id": "rec-1",
            "source_scid": pair.source_channel_id,
            "sink_scid": pair.dest_channel_id,
            "amount_sats": pair.amount_sats,
            "route_policy": "hybrid",
            "priority_score": 77.0,
        }],
    )

    decision = decide_route_policy(
        pair,
        reason_code="coordinated_rebalance",
        hive_hints=hints,
    )

    assert decision.policy is RoutePolicy.HYBRID
    assert decision.priority is RoutePriority.COORDINATED
    assert decision.hint_id == "rec-1"
    assert decision.priority_score == 77.0


def test_decide_route_policy_matches_active_campaign_chunk_metadata():
    from modules.rebalance_types_v2 import PairCandidate

    _, RoutePolicy, RoutePriority, decide_route_policy = _import_route_policy()

    pair = PairCandidate(
        source_channel_id="13x13x13",
        dest_channel_id="14x14x14",
        source_peer_id="02" + "3" * 64,
        dest_peer_id="02" + "4" * 64,
        amount_sats=55_000,
        pair_budget_sats=25,
    )
    hints = FakeHiveHints(
        members={pair.source_peer_id, pair.dest_peer_id},
        campaigns=[{
            "campaign_id": "camp-1",
            "status": "active",
            "active_chunk_recommendation": {
                "source_scid": pair.source_channel_id,
                "sink_scid": pair.dest_channel_id,
                "amount_sats": pair.amount_sats,
                "route_policy": "hive_only",
                "allow_market_fallback": False,
            },
        }],
    )

    decision = decide_route_policy(
        pair,
        reason_code="coordinated_rebalance",
        hive_hints=hints,
    )

    assert decision.policy is RoutePolicy.HIVE_ONLY
    assert decision.priority is RoutePriority.COORDINATED
    assert decision.hint_id == "camp-1"
    assert decision.allow_market_fallback is False
