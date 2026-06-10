"""Planner drain-demand signal: the over-local residual the circular
rebalancer cannot place, published for the Boltz structural consumer."""

from modules.rebalance_state_v2 import ChannelState, StateSnapshot
from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_types_v2 import DrainDemand, DrainDemandEntry


def _state(channel_id, local_ratio, capacity=1_000_000, value_class="active",
           budget=1000, **kwargs):
    fields = dict(
        channel_id=channel_id,
        peer_id="02" + channel_id[0] * 65,
        capacity_sats=capacity,
        local_ratio=local_ratio,
        actual_inbound_fee_ppm=0,
        value_class=value_class,
        is_valuable=True,
        remaining_budget_sats=budget,
        cooldown_active=False,
        source_drain_score=max(0.0, local_ratio - 0.65),
    )
    fields.update(kwargs)
    return ChannelState(**fields)


def test_drain_demand_dataclass_shape():
    entry = DrainDemandEntry(channel_id="1x1x1", peer_id="02aa",
                             excess_sats=300_000, drain_score=0.9,
                             value_class="active")
    demand = DrainDemand(entries=[entry], total_excess_sats=300_000,
                         over_local_count=1, paired_count=0)
    assert demand.entries[0].excess_sats == 300_000
