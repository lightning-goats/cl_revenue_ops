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


def _plan(channels, max_pairs=5):
    snapshot = StateSnapshot(channels=tuple(channels))
    planner = RebalancePlanner(max_pairs=max_pairs)
    return planner.plan(snapshot)


def test_unpaired_over_local_channels_become_drain_demand():
    # 3 over-local, 1 over-remote: one pairs, two are residual demand.
    result = _plan([
        _state("1x1x1", 0.97),
        _state("2x2x2", 0.95),
        _state("3x3x3", 0.92),
        _state("9x9x9", 0.10),
    ])
    assert len(result.selected) == 1
    demand = result.drain_demand
    assert demand is not None
    assert demand.over_local_count == 3
    assert demand.paired_count == 1
    residual_ids = {e.channel_id for e in demand.entries}
    paired_source = result.selected[0].source_channel_id
    assert paired_source not in residual_ids
    assert len(residual_ids) == 2
    assert demand.total_excess_sats == sum(e.excess_sats for e in demand.entries)
    # ordered by drain score, worst first
    scores = [e.drain_score for e in demand.entries]
    assert scores == sorted(scores, reverse=True)


def test_fully_source_heavy_node_publishes_all_as_demand():
    result = _plan([_state(f"{i}x1x1", 0.97) for i in range(1, 5)])
    assert result.selected == []
    assert result.drain_demand.over_local_count == 4
    assert result.drain_demand.paired_count == 0
    assert len(result.drain_demand.entries) == 4


def test_balanced_node_has_empty_drain_demand():
    result = _plan([_state("1x1x1", 0.50), _state("2x2x2", 0.55)])
    assert result.drain_demand is not None
    assert result.drain_demand.entries == []
    assert result.drain_demand.total_excess_sats == 0


def test_excess_sats_measured_against_band_high():
    # 0.97 local on 1_000_000 cap with band high 0.65 => 320_000 excess
    result = _plan([_state("1x1x1", 0.97)])
    entry = result.drain_demand.entries[0]
    assert entry.excess_sats == 320_000
