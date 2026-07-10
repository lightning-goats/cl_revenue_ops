"""Tests for standalone (market-only) route-policy classification."""

import pytest


def _import_route_policy():
    try:
        from modules.rebalance_route_policy import (  # type: ignore
            RouteDecision,
            RoutePolicy,
            RoutePriority,
            decide_route_policy,
        )
    except Exception as e:  # pragma: no cover - bootstrap guard
        pytest.fail(f"route policy support missing: {e}")
    return RouteDecision, RoutePolicy, RoutePriority, decide_route_policy


def test_decide_route_policy_is_always_market_only():
    from modules.rebalance_types_v2 import PairCandidate

    _, RoutePolicy, RoutePriority, decide_route_policy = _import_route_policy()

    pair = PairCandidate(
        source_channel_id="1x1x1",
        dest_channel_id="2x2x2",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        pair_budget_sats=10,
    )

    decision = decide_route_policy(pair, reason_code="ev_positive")

    assert decision.policy is RoutePolicy.MARKET_ONLY
    assert decision.priority is RoutePriority.EV_POSITIVE
    assert decision.allow_market_fallback is True
    assert decision.reason == "ev_positive"


def test_decide_route_policy_defaults_reason_when_blank():
    from modules.rebalance_types_v2 import PairCandidate

    _, RoutePolicy, RoutePriority, decide_route_policy = _import_route_policy()

    pair = PairCandidate(
        source_channel_id="3x3x3",
        dest_channel_id="4x4x4",
        source_peer_id="02" + "c" * 64,
        dest_peer_id="02" + "d" * 64,
        amount_sats=50_000,
        pair_budget_sats=25,
    )

    decision = decide_route_policy(pair)

    assert decision.policy is RoutePolicy.MARKET_ONLY
    assert decision.priority is RoutePriority.EV_POSITIVE
    assert decision.reason == "ev_positive"


def test_pair_candidate_can_store_route_decision():
    from modules.rebalance_types_v2 import PairCandidate

    RouteDecision, RoutePolicy, RoutePriority, _ = _import_route_policy()

    decision = RouteDecision(
        policy=RoutePolicy.MARKET_ONLY,
        priority=RoutePriority.EV_POSITIVE,
        reason="ev_positive",
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
    assert pair.route_decision.policy is RoutePolicy.MARKET_ONLY
