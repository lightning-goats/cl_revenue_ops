"""Golden: LN+ SwapEvaluator qualification gates (fill state, terms,
participant quality). Pure functions of (swap dict, cfg) plus mocked
rpc/db/planner lookups — no HTTP, no live node."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.lnplus_swaps import SwapEvaluator
from tests.golden.util import golden_check

OUR_ID = "02" + "f" * 64


def _cfg(**over):
    base = dict(
        planner_min_channel_sats=1_000_000,
        planner_max_channel_sats=10_000_000,
        lnplus_max_duration_months=6,
        lnplus_max_participants=5,
        lnplus_min_participants=3,
        lnplus_min_peer_positive_ratings=5,
        lnplus_min_peer_rank=3,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _evaluator(policy_manager=None):
    rpc = MagicMock()
    rpc.getinfo.return_value = {"id": OUR_ID}
    db = MagicMock()
    db.lnplus_get_peer.return_value = None
    planner = MagicMock()
    planner._score_candidate.return_value = 1.0
    return SwapEvaluator(
        MagicMock(), rpc, db, MagicMock(), MagicMock(), planner,
        MagicMock(), policy_manager=policy_manager,
    )


def _swap(**over):
    base = dict(
        status="pending",
        participant_waiting_for_count=1,
        capacity_sats=2_000_000,
        duration_months=3,
        participant_max_count=4,
        platform="any",
    )
    base.update(over)
    return base


FILTER_SCENARIOS = {
    "qualifying_swap_passes": {},
    "not_pending": {"status": "completed"},
    "not_last_slot": {"participant_waiting_for_count": 2},
    "below_min_capacity": {"capacity_sats": 500_000},
    "above_max_capacity": {"capacity_sats": 50_000_000},
    "duration_too_long": {"duration_months": 12},
    "too_many_participants": {"participant_max_count": 9},
    "dual_swap_rejected": {"participant_max_count": 2},
    "lnd_platform_rejected": {"platform": "lnd"},
}


@pytest.mark.parametrize("name", sorted(FILTER_SCENARIOS))
def test_golden_filter_swap(name):
    ev = _evaluator()
    result = ev._filter_swap(_swap(**FILTER_SCENARIOS[name]), _cfg())
    golden_check(f"lnplus/filter_{name}", {
        "overrides": FILTER_SCENARIOS[name],
        "rejection": result,
    })


def _participant(**over):
    base = dict(
        pubkey="02" + "b" * 64,
        cancelled=False, banned=False,
        address_1="1.2.3.4:9735", address_2=None,
        positive_ratings_count=20, negative_ratings_count=0,
        lnplus_rank_number=5,
    )
    base.update(over)
    return base


PARTICIPANT_SCENARIOS = {
    "good_ring_passes": [_participant()],
    "own_node_in_ring": [_participant(pubkey=OUR_ID)],
    "no_address_rejected": [_participant(address_1=None)],
    "low_ratings_rejected": [_participant(positive_ratings_count=1)],
    "negative_ratio_rejected": [_participant(positive_ratings_count=20,
                                             negative_ratings_count=5)],
    "rank_below_floor": [_participant(lnplus_rank_number=1)],
    "cancelled_peer_skipped": [
        _participant(cancelled=True, positive_ratings_count=0),
        _participant(pubkey="02" + "c" * 64),
    ],
}


@pytest.mark.parametrize("name", sorted(PARTICIPANT_SCENARIOS))
def test_golden_check_participants(name):
    ev = _evaluator()
    swap = _swap(participants=PARTICIPANT_SCENARIOS[name])
    result = ev._check_participants(swap, _cfg())
    golden_check(f"lnplus/participants_{name}", {"rejection": result})


def test_operator_ban_vetoes_and_fails_closed():
    """Non-golden anchor (pins the 2026-07-12 ban-gate behavior)."""
    banned_pm = MagicMock()
    banned_pm.is_peer_banned.return_value = True
    ev = _evaluator(policy_manager=banned_pm)
    res = ev._check_participants(_swap(participants=[_participant()]), _cfg())
    assert res is not None and "operator-banned" in res

    broken_pm = MagicMock()
    broken_pm.is_peer_banned.side_effect = RuntimeError("db gone")
    ev = _evaluator(policy_manager=broken_pm)
    res = ev._check_participants(_swap(participants=[_participant()]), _cfg())
    assert res is not None and "fail closed" in res


def test_filter_gate_prefixes_anchor():
    """Non-golden anchor: rejection strings carry their gate prefix."""
    ev = _evaluator()
    assert ev._filter_swap(_swap(status="completed"), _cfg()).startswith(
        "fill_state:")
    assert ev._filter_swap(_swap(capacity_sats=1), _cfg()).startswith(
        "terms:")
    assert ev._filter_swap(_swap(), _cfg()) is None
