"""Golden: RebalancePlanner.plan — source/dest pairing, amount sizing,
scoring, skip reasons. Pure snapshot-in / PlanResult-out."""
import pytest

from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_state_v2 import ChannelState, StateSnapshot
from tests.golden.util import golden_check


def _ch(cid, local_ratio, **over):
    base = dict(
        channel_id=cid,
        peer_id="02" + cid[0] * 64,
        capacity_sats=2_000_000,
        local_ratio=local_ratio,
        actual_inbound_fee_ppm=100,
        value_class="active",
        is_valuable=True,
        remaining_budget_sats=5_000,
        cooldown_active=False,
        source_eligible=True,
        dest_eligible=True,
        local_out_fee_ppm=250,
        is_active=True,
    )
    base.update(over)
    return ChannelState(**base)


def _snapshot(channels):
    return StateSnapshot(
        channels=tuple(channels),
        total_capacity_sats=sum(c.capacity_sats for c in channels),
        total_remaining_budget_sats=sum(
            c.remaining_budget_sats for c in channels),
        valuable_channel_count=sum(1 for c in channels if c.is_valuable),
    )


SCENARIOS = {
    "single_obvious_pair": [
        _ch("aaa", 0.90), _ch("bbb", 0.10),
    ],
    "no_over_remote_no_pairs": [
        _ch("aaa", 0.90), _ch("bbb", 0.50),
    ],
    "source_ineligible_skipped": [
        _ch("aaa", 0.90, source_eligible=False,
            source_reason="cooldown_active"),
        _ch("bbb", 0.10),
    ],
    "profitable_dest_preferred": [
        _ch("aaa", 0.92),
        _ch("bbb", 0.08, value_class="profitable"),
        _ch("ccc", 0.08, value_class="neutral", is_valuable=False),
    ],
    "custom_band_channel": [
        _ch("aaa", 0.70, target_band_high=0.60),  # over-local per own band
        _ch("bbb", 0.10),
    ],
    "amount_bounded_by_chunk": [
        _ch("aaa", 1.00, capacity_sats=50_000_000,
            remaining_budget_sats=100_000),
        _ch("bbb", 0.00, capacity_sats=50_000_000,
            remaining_budget_sats=100_000),
    ],
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_plan(name):
    planner = RebalancePlanner(
        target_band_low=0.35, target_band_high=0.65,
        max_chunk_sats=2_000_000, max_pairs=10, pair_fee_cap_ppm=0,
    )
    result = planner.plan(_snapshot(SCENARIOS[name]))
    golden_check(f"rebalance/plan_{name}", result)


def test_plan_hand_computed_anchor():
    """Non-golden anchor: one over-local + one over-remote channel with
    identical capacity must yield exactly one selected pair, source=aaa
    dest=bbb, amount > 0 and <= max_chunk_sats."""
    planner = RebalancePlanner()
    result = planner.plan(_snapshot([_ch("aaa", 0.90), _ch("bbb", 0.10)]))
    assert len(result.selected) == 1
    pair = result.selected[0]
    assert pair.source_channel_id == "aaa"
    assert pair.dest_channel_id == "bbb"
    assert 0 < pair.amount_sats <= planner.max_chunk_sats
