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


# ---------------------------------------------------------------------------
# Task 7: engine accessor + debug surface
# ---------------------------------------------------------------------------

from modules.config import Config
from modules.rebalance_engine_v2 import RebalanceEngine


def _make_engine(plugin, database):
    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    database.record_rebalance.return_value = 1
    database.reserve_budget.return_value = (True, 9999)
    database.mark_budget_spent.return_value = True
    database.release_budget_reservation.return_value = True
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def test_engine_exposes_drain_demand(mock_plugin, mock_database):
    from modules.rebalance_engine_v2 import CycleResult
    from modules.rebalance_types_v2 import PlanResult

    engine = _make_engine(mock_plugin, mock_database)
    demand = DrainDemand(entries=[], total_excess_sats=123,
                         over_local_count=4, paired_count=0)
    cycle = CycleResult()
    cycle.plan = PlanResult(drain_demand=demand)
    engine._cache_cycle_result(cycle)

    got = engine.get_drain_demand()
    assert got is demand


def test_engine_drain_demand_none_before_first_cycle(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    assert engine.get_drain_demand() is None


# ---------------------------------------------------------------------------
# Task 7: debug-surface test via load_plugin_module
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock
from tests.plugin_test_utils import load_plugin_module


def _load_debug_module(rebalancer_override=None):
    """Load the plugin module with minimal stubs needed for revenue-rebalance-debug."""
    mod = load_plugin_module()
    mod.config = Config(paused=True)
    mod.hive_hints = None
    mod.hive_router = None
    mod.safe_plugin = SimpleNamespace(rpc=MagicMock())
    mod.data_service = MagicMock()
    mod.data_service.get_funds.return_value = {"outputs": [], "channels": []}
    mod.database = MagicMock()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_daily_rebalance_spend.return_value = {
        "total_spent_sats": 0,
        "total_reserved_sats": 0,
        "stale_reservations": 0,
        "job_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
    }
    mod.database.list_hot_channel_protection_override_peers.return_value = []
    mod.rebalancer = rebalancer_override or SimpleNamespace(
        _get_channels_with_balances=lambda: {},
        job_manager=SimpleNamespace(active_channels=set()),
        get_last_decision_summary=lambda: {
            "action": "hold",
            "reason": "not_run",
            "dominant_input": "startup",
            "safety_block": False,
            "budget_blocked": False,
        },
        rebalance_engine_v2=None,
    )
    mod._total_cost_budget_status = MagicMock(
        return_value={
            "effective_budget_sats": 0,
            "remaining_sats": 0,
            "actual_spent_sats": 0,
            "reserved_sats": 0,
            "actual_spent_by_category": {},
            "reserved_by_category": {},
        }
    )
    mod._boltz_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 0, "reserved_24h_sats": 0}
    )
    return mod


def test_debug_surface_exposes_drain_demand(mock_plugin, mock_database):
    """revenue-rebalance-debug includes drain_demand when engine reports it."""
    entry1 = DrainDemandEntry(
        channel_id="1x1x1", peer_id="02aa",
        excess_sats=300_000, drain_score=0.9, value_class="active",
    )
    entry2 = DrainDemandEntry(
        channel_id="2x2x2", peer_id="02bb",
        excess_sats=150_000, drain_score=0.7, value_class="neutral",
    )
    demand = DrainDemand(
        entries=[entry1, entry2],
        total_excess_sats=450_000,
        over_local_count=2,
        paired_count=0,
    )

    mock_engine = MagicMock()
    mock_engine.get_drain_demand.return_value = demand
    mock_engine.get_last_cycle_debug.return_value = {}

    rebalancer = SimpleNamespace(
        _get_channels_with_balances=lambda: {},
        job_manager=SimpleNamespace(active_channels=set()),
        get_last_decision_summary=lambda: {
            "action": "hold",
            "reason": "not_run",
            "dominant_input": "startup",
            "safety_block": False,
            "budget_blocked": False,
        },
        rebalance_engine_v2=mock_engine,
    )

    mod = _load_debug_module(rebalancer_override=rebalancer)
    result = mod.revenue_rebalance_debug(mod.plugin)

    assert "drain_demand" in result
    dd = result["drain_demand"]
    assert dd is not None
    assert dd["total_excess_sats"] == 450_000
    assert len(dd["top_entries"]) == 2
    assert dd["top_entries"][0]["channel_id"] == "1x1x1"
    assert dd["top_entries"][1]["channel_id"] == "2x2x2"


def test_debug_surface_drain_demand_none_when_engine_absent():
    """drain_demand is None in result when no engine is present."""
    mod = _load_debug_module()
    result = mod.revenue_rebalance_debug(mod.plugin)
    assert "drain_demand" in result
    assert result["drain_demand"] is None
