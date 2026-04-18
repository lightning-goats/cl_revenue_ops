"""Tests for the rebalance engine delegation and integration."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cycle_result(candidates=None, executions=None):
    from modules.rebalance_engine_v2 import CycleResult
    cr = CycleResult()
    cr.candidates = candidates or []
    cr.executions = executions or []
    return cr


def _make_engine(plugin, database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def test_rebalancer_calls_engine_run_cycle(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result()

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.run_cycle.assert_called_once()


def test_rebalancer_without_engine_suppresses(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = None

    result = r.find_rebalance_candidates()

    assert result == []
    summary = r.get_last_decision_summary()
    assert summary["action"] == "suppressed"
    assert "unavailable" in summary["reason"]


def test_rebalancer_reports_no_candidates(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result()

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "hold"
    assert summary["safety_block"] is False


def test_rebalancer_reports_successful_execution(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_executor_v2 import ExecutionResult

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result(
        candidates=["c1"],
        executions=[ExecutionResult(success=True)],
    )

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "rebalance"
    assert "succeeded" in summary["reason"]


def test_rebalancer_reports_failed_execution(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_executor_v2 import ExecutionResult

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result(
        candidates=["c1"],
        executions=[ExecutionResult(success=False, error="no_route")],
    )

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "rebalance"
    assert "0/1" in summary["reason"]


def test_active_engine_honors_hive_only_pairs(mock_plugin, mock_database):
    """Strict hive-only candidates must be priced through the hive router."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_route_policy import RoutePolicy

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._hive_router = MagicMock(name="hive_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    hive_only_candidate = SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=1.0,
        route_decision=SimpleNamespace(
            policy=RoutePolicy.HIVE_ONLY,
            allow_market_fallback=False,
            reason="hive_equalization",
        ),
        route_cost_sats=None,
        route=None,
    )

    route_result = SimpleNamespace(
        success=True,
        route_cost_sats=1,
        route=[],
        probability_ppm=0,
        error="",
    )
    engine._hive_router.price_pair.return_value = route_result

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(
            selected=[hive_only_candidate],
            skipped=[],
        )

        selected = engine.find_candidates()

    assert selected == [hive_only_candidate]
    engine._hive_router.price_pair.assert_called_once()
    engine.router_v3.price_pair.assert_not_called()


def test_engine_debug_exposes_score_decomposition_for_selected_pairs(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    candidate = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10,
        source_capacity_sats=100_000,
        dest_capacity_sats=100_000,
        score=2.0,
        source_local_ratio=0.80,
        dest_local_ratio=0.20,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=2,
        route=[{"channel": "300x3x0"}],
        probability_ppm=750_000,
        error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(
            selected=[candidate],
            skipped=[],
        )

        selected = engine.find_candidates()

    assert selected == [candidate]
    debug = engine.get_last_cycle_debug()
    assert debug["summary"]["considered_pairs"] == 1
    assert debug["summary"]["selected_pairs"] == 1
    decomposition = debug["selected_candidates"][0]["score_decomposition"]
    assert decomposition["p_success"] == 0.75
    assert decomposition["beats_do_nothing"] is True
    assert decomposition["rejection_reason"] == ""
    assert decomposition["inputs"]["expected_fee_sats"] == 2


def test_engine_debug_keeps_route_over_budget_candidate_in_considered_pairs(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    candidate = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10,
        source_capacity_sats=100_000,
        dest_capacity_sats=100_000,
        score=1.0,
        source_local_ratio=0.80,
        dest_local_ratio=0.20,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=12,
        route=[{"channel": "300x3x0"}],
        probability_ppm=600_000,
        error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(
            selected=[candidate],
            skipped=[],
        )

        selected = engine.find_candidates()

    assert selected == []
    debug = engine.get_last_cycle_debug()
    assert debug["summary"]["considered_pairs"] == 1
    assert debug["summary"]["selected_pairs"] == 0
    decomposition = debug["considered_candidates"][0]["score_decomposition"]
    assert decomposition["rejection_reason"] == "route_over_budget"
    assert decomposition["beats_do_nothing"] is False


def test_polar_s2_pairless_failure_names_source_rejected_neutral(
    mock_plugin, mock_database
):
    """Phase 1.4 regression: Polar S2 (fleet-r4) ended with one profitable
    depleted channel and two extreme over-local channels but considered_pairs=0.
    revenue-rebalance-debug.last_cycle must explain that the over-local sources
    were discarded as neutral, not silently report no_rebalance_candidates."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    engine = _make_engine(mock_plugin, mock_database)
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()

    allocations = CapexAllocations(
        channel_budgets={
            "159x1x0": ChannelCapexBudget(channel_id="159x1x0", budget_msat=1_000_000),
        }
    )
    snapshot = build_state_snapshot(
        [
            # depleted profitable -> dest-eligible but no eligible sources to pair with
            ChannelInput(
                channel_id="159x1x0",
                peer_id="02" + "1" * 64,
                capacity_sats=1_000_000,
                local_sats=100_000,
                is_profitable=True,
                is_active=True,
            ),
            # extreme over-local neutral -> rejected as not_valuable
            ChannelInput(
                channel_id="243x1x0",
                peer_id="02" + "2" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
            ),
            ChannelInput(
                channel_id="255x1x0",
                peer_id="02" + "3" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
            ),
        ],
        allocations,
    )
    engine._build_snapshot = MagicMock(return_value=snapshot)

    # Phase 2 turns this from a pairless hold into a valid pair: the over-local
    # neutral sources are no longer rejected as not_valuable. A pair forms with
    # the funded depleted destination.
    engine.router_v3 = MagicMock(name="market_router")
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=False, route_cost_sats=None, route=None, probability_ppm=0,
        error="route_pricer_stubbed",
    )

    selected = engine.find_candidates()
    debug = engine.get_last_cycle_debug()

    assert debug["summary"]["considered_pairs"] >= 1
    diagnostics = debug["hold_diagnostics"]
    assert diagnostics["source_rejected_neutral"] == 0
    considered_sources = {
        c["source_channel_id"] for c in debug["considered_candidates"]
    }
    considered_dests = {
        c["dest_channel_id"] for c in debug["considered_candidates"]
    }
    assert considered_sources & {"243x1x0", "255x1x0"}
    assert "159x1x0" in considered_dests


def test_polar_s9_pairless_failure_names_dest_cooldown_and_neutral_sources(
    mock_plugin, mock_database
):
    """Phase 1.4 regression: Polar S9 (fleet-r2) had one channel at 6.6% local
    blocked by cooldown and two channels at 100% local rejected as neutral.
    revenue-rebalance-debug.last_cycle must surface both blockers."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    engine = _make_engine(mock_plugin, mock_database)
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()

    allocations = CapexAllocations(
        channel_budgets={
            "123x1x0": ChannelCapexBudget(channel_id="123x1x0", budget_msat=1_000_000),
        }
    )
    snapshot = build_state_snapshot(
        [
            # depleted profitable in cooldown -> dest blocked
            ChannelInput(
                channel_id="123x1x0",
                peer_id="02" + "9" * 64,
                capacity_sats=1_000_000,
                local_sats=66_000,
                is_profitable=True,
                cooldown_active=True,
            ),
            # 100%-local neutral channels -> rejected as not_valuable
            ChannelInput(
                channel_id="200x2x0",
                peer_id="02" + "8" * 64,
                capacity_sats=1_000_000,
                local_sats=1_000_000,
            ),
            ChannelInput(
                channel_id="201x2x0",
                peer_id="02" + "7" * 64,
                capacity_sats=1_000_000,
                local_sats=1_000_000,
            ),
        ],
        allocations,
    )
    engine._build_snapshot = MagicMock(return_value=snapshot)

    # Phase 3: the depleted destination at 6.6% local triggers the emergency
    # drift override and becomes refill-eligible despite cooldown. Combined
    # with Phase 2 source eligibility, a pair forms with one of the neutral
    # 100%-local sources. The router is stubbed below so no actual route is
    # priced -- we only need to confirm a candidate emerged from the planner.
    engine.router_v3 = MagicMock(name="market_router")
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=False, route_cost_sats=None, route=None, probability_ppm=0,
        error="route_pricer_stubbed",
    )

    engine.find_candidates()
    debug = engine.get_last_cycle_debug()

    assert debug["summary"]["considered_pairs"] >= 1
    diagnostics = debug["hold_diagnostics"]
    assert diagnostics["dest_blocked_by_cooldown"] == 0  # override fired
    assert diagnostics["source_rejected_neutral"] == 0
    considered_dests = {
        c["dest_channel_id"] for c in debug["considered_candidates"]
    }
    assert "123x1x0" in considered_dests


def test_get_last_cycle_debug_emits_pairless_hold_diagnostics(
    mock_plugin, mock_database
):
    """Phase 1.2: when the planner produces zero pairs, the operator surface
    must explain why -- depleted destinations stuck in cooldown, depleted
    destinations with no budget, over-local sources rejected as neutral, and
    in-band channels that simply aren't candidates."""
    from modules.rebalance_engine_v2 import CycleResult
    from modules.rebalance_state_v2 import ChannelState, StateSnapshot

    engine = _make_engine(mock_plugin, mock_database)

    def _ch(channel_id, *, local_ratio, source_eligible, dest_eligible,
            source_reason="", dest_reason="", value_class="active"):
        return ChannelState(
            channel_id=channel_id,
            peer_id="02" + channel_id[0] * 64,
            capacity_sats=1_000_000,
            local_ratio=local_ratio,
            actual_inbound_fee_ppm=0,
            value_class=value_class,
            is_valuable=value_class != "neutral",
            remaining_budget_sats=1000 if dest_reason != "no_budget" else 0,
            cooldown_active=(source_reason == "cooldown" or dest_reason == "cooldown"),
            source_eligible=source_eligible,
            dest_eligible=dest_eligible,
            source_reason=source_reason,
            dest_reason=dest_reason,
            dest_urgency=max(0.0, 0.35 - local_ratio) / 0.35,
            source_drain_score=max(0.0, local_ratio - 0.65) / 0.35,
        )

    snapshot = StateSnapshot(
        channels=(
            # depleted profitable channel sitting in cooldown -> dest_blocked_by_cooldown
            _ch("100x1x0", local_ratio=0.07, source_eligible=False,
                dest_eligible=False, dest_reason="cooldown",
                source_reason="cooldown", value_class="profitable"),
            # depleted candidate without funding -> dest_not_funded
            _ch("101x1x0", local_ratio=0.10, source_eligible=False,
                dest_eligible=False, dest_reason="no_budget",
                source_reason="no_budget", value_class="active"),
            # over-local neutral channel rejected as not_valuable -> source_rejected_neutral
            _ch("200x2x0", local_ratio=0.95, source_eligible=False,
                dest_eligible=False, source_reason="not_valuable",
                dest_reason="not_valuable", value_class="neutral"),
            # over-local but parked in cooldown -> source_protected
            _ch("201x2x0", local_ratio=0.92, source_eligible=False,
                dest_eligible=False, source_reason="cooldown",
                dest_reason="cooldown", value_class="profitable"),
            # in-band channel -> source_inside_band
            _ch("300x3x0", local_ratio=0.50, source_eligible=True,
                dest_eligible=True, value_class="profitable"),
        ),
        total_capacity_sats=5_000_000,
        total_remaining_budget_sats=4_000,
        valuable_channel_count=4,
    )

    engine._cache_cycle_result(CycleResult(snapshot=snapshot))

    debug = engine.get_last_cycle_debug()

    assert debug["summary"]["considered_pairs"] == 0
    diagnostics = debug["hold_diagnostics"]
    assert diagnostics["dest_blocked_by_cooldown"] == 1
    assert diagnostics["dest_not_funded"] == 1
    assert diagnostics["source_rejected_neutral"] == 1
    assert diagnostics["source_protected"] == 1
    assert diagnostics["source_inside_band"] == 1


def test_first_transient_failure_uses_short_cooldown(mock_plugin, mock_database):
    """Iter2: temporary_channel_failure base cooldown is 300s (5 min) on a
    first failure. Sling classifies these as retriable and the prior 1800s
    base was too aggressive -- it kept good pairs out of contention for
    half an hour after a single transient route failure."""
    engine = _make_engine(mock_plugin, mock_database)

    assert engine._pair_failure_cooldowns["temporary_channel_failure"] == 300
    # Persistent failures still get the long quarantine.
    assert engine._pair_failure_cooldowns["permanent_failure"] >= 21600
    # Retriable buckets stay below the temporary-then-escalation threshold.
    assert engine._pair_failure_cooldowns["other_retriable"] <= 1800


def test_polar_s7_oscillation_does_not_unblock_cooldown(
    mock_plugin, mock_database
):
    """Phase 3.4 regression: a small drift below the post-rebalance anchor
    must NOT trigger the cooldown override. The S7 capital-burn trap depends
    on the rebalancer staying conservative under tiny oscillation."""
    import time as _time
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    engine = _make_engine(mock_plugin, mock_database)
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()

    # Simulate the engine's drift-override path with stubbed database calls.
    mock_database.get_last_rebalance_time.return_value = int(_time.time()) - 600
    mock_database.get_last_post_rebalance_state.return_value = {
        "timestamp": int(_time.time()) - 600,
        "post_local_ratio": 0.50,
        "amount_sats": 200_000,
    }

    allocations = CapexAllocations(
        channel_budgets={
            "777x7x0": ChannelCapexBudget(channel_id="777x7x0", budget_msat=1_000_000),
        }
    )
    snapshot = build_state_snapshot(
        [
            ChannelInput(
                channel_id="777x7x0",
                peer_id="02" + "7" * 64,
                capacity_sats=1_000_000,
                local_sats=450_000,  # only 5% drift from the 50% anchor
                is_profitable=True,
                cooldown_active=True,
                cooldown_override=False,
            ),
            # Add an over-local source so a pair would form if dest unblocked
            ChannelInput(
                channel_id="800x8x0",
                peer_id="02" + "8" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
            ),
        ],
        allocations,
        target_emergency_low=0.10,
    )
    engine._build_snapshot = MagicMock(return_value=snapshot)

    selected = engine.find_candidates()
    debug = engine.get_last_cycle_debug()

    assert selected == []
    assert debug["summary"]["considered_pairs"] == 0
    diagnostics = debug["hold_diagnostics"]
    # Channel sits between low and high band -> inside_band, not depleted.
    assert diagnostics["source_inside_band"] == 1


def test_top_level_hold_reason_maps_specific_blocker_from_engine(
    mock_plugin, mock_database
):
    """Phase 1 (deferred completion): when the engine returns no candidates,
    the rebalancer's last_decision.reason must surface the most specific
    blocker (e.g. dest_blocked_by_cooldown, source_rejected_neutral,
    below_hold_margin) instead of the coarse no_rebalance_candidates."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import CycleResult
    from modules.rebalance_state_v2 import ChannelState, StateSnapshot
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0
    rebalancer = EVRebalancer(mock_plugin, cfg, mock_database)
    rebalancer._check_capital_controls = MagicMock(return_value=True)

    snapshot = StateSnapshot(
        channels=(
            ChannelState(
                channel_id="100x1x0",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_ratio=0.07,
                actual_inbound_fee_ppm=0,
                value_class="profitable",
                is_valuable=True,
                remaining_budget_sats=1000,
                cooldown_active=True,
                source_eligible=False,
                dest_eligible=False,
                source_reason="cooldown",
                dest_reason="cooldown",
            ),
        ),
        valuable_channel_count=1,
    )
    cycle_result = CycleResult(snapshot=snapshot)
    engine = MagicMock()
    engine.run_cycle.return_value = cycle_result
    engine.get_last_cycle_debug.return_value = {
        "summary": {"considered_pairs": 0, "selected_pairs": 0},
        "hold_diagnostics": {
            "dest_blocked_by_cooldown": 1,
            "dest_not_funded": 0,
            "source_rejected_neutral": 0,
            "source_protected": 0,
            "source_inside_band": 0,
        },
        "considered_candidates": [],
        "selected_candidates": [],
        "skipped": [],
        "executions": [],
    }
    rebalancer.rebalance_engine_v2 = engine

    rebalancer.find_rebalance_candidates()

    summary = rebalancer.get_last_decision_summary()
    assert summary["action"] == "hold"
    assert summary["reason"] == "dest_blocked_by_cooldown"


def test_polar_s9_strong_refill_beats_hold_margin(mock_plugin, mock_database):
    """Phase 4.4 ordering: a Polar S9-shaped strong refill candidate (severely
    depleted dest, high-drain neutral source, cheap route) clears a modest
    hold margin and survives the gate."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    cfg = Config(dry_run=True, rebalance_router="v3", rebalance_hold_margin=0.05)
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    strong_pair = PairCandidate(
        source_channel_id="200x2x0",
        dest_channel_id="123x1x0",
        source_peer_id="03" + "8" * 64,
        dest_peer_id="03" + "9" * 64,
        amount_sats=400_000,
        pair_budget_sats=1000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,                # strong additive role-aware score
        source_local_ratio=1.0,   # 100% local source
        dest_local_ratio=0.066,   # 6.6% local dest -- emergency depleted
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True, route_cost_sats=5, route=[{"channel": "300x3x0"}],
        probability_ppm=850_000, error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[strong_pair], skipped=[]
        )
        selected = engine.find_candidates()

    assert len(selected) == 1
    assert selected[0].dest_channel_id == "123x1x0"


def test_polar_s7_oscillation_loses_to_hold_margin(mock_plugin, mock_database):
    """Phase 4.4 ordering: an S7-style tiny oscillation pair has a weak score
    and must lose to the hold margin -- the engine prefers do_nothing over
    capital-destructive churn."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    cfg = Config(dry_run=True, rebalance_router="v3", rebalance_hold_margin=0.05)
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    weak_pair = PairCandidate(
        source_channel_id="500x5x0",
        dest_channel_id="600x6x0",
        source_peer_id="03" + "5" * 64,
        dest_peer_id="03" + "6" * 64,
        amount_sats=10_000,
        pair_budget_sats=100,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=0.05,               # weak planner score -- tiny oscillation
        source_local_ratio=0.70,
        dest_local_ratio=0.30,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True, route_cost_sats=20, route=[{"channel": "700x7x0"}],
        probability_ppm=600_000, error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[weak_pair], skipped=[]
        )
        selected = engine.find_candidates()

    assert selected == []
    debug = engine.get_last_cycle_debug()
    skipped_reasons = {row["channel_id"]: row["reason"] for row in debug["skipped"]}
    assert skipped_reasons.get("600x6x0") == "below_hold_margin"


def test_pair_below_hold_margin_is_rejected_with_explicit_reason(
    mock_plugin, mock_database
):
    """Phase 4.3: do_nothing becomes a hard gate. A priced pair whose
    final_score does not clear the configured hold margin must be rejected
    with rejection_reason='below_hold_margin', not silently picked."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    cfg = Config(dry_run=True, rebalance_router="v3", rebalance_hold_margin=0.50)
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    weak_pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=100,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=0.10,  # weak planner score -> won't clear 0.50 hold margin
        source_local_ratio=0.70,
        dest_local_ratio=0.30,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True, route_cost_sats=2, route=[{"channel": "300x3x0"}],
        probability_ppm=600_000, error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[weak_pair], skipped=[]
        )
        selected = engine.find_candidates()

    assert selected == []  # gate keeps the weak pair out
    debug = engine.get_last_cycle_debug()
    skipped_reasons = {row["channel_id"]: row["reason"] for row in debug["skipped"]}
    assert skipped_reasons.get("200x2x0") == "below_hold_margin"


def test_engine_layer_preserves_route_success_fee_and_penalty_terms(
    mock_plugin, mock_database
):
    """Phase 4.2: route success probability and pair-cost penalties enter
    the final score in the engine, not the planner. Verify the engine's
    final_score = p_success*future_value - expected_fee - source_opp_cost
    - failure_penalty - capital_risk_penalty contract holds."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=100,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=750_000,
        route_cost_sats=10,
        effective_budget_sats=100,
        route_status="priced",
    )

    assert decomp["p_success"] == 0.75
    assert decomp["expected_fee"] > 0.0
    expected = round(
        decomp["p_success"] * decomp["expected_future_value"]
        - decomp["expected_fee"]
        - decomp["source_opportunity_cost"]
        - decomp["failure_penalty"]
        - decomp["capital_risk_penalty"],
        6,
    )
    assert decomp["final_score"] == expected


def test_engine_build_snapshot_uses_membership_router_for_hive_value_class(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    membership_router = MagicMock()
    membership_router.is_hive_member.return_value = True
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "03" + "b" * 64,
                "short_channel_id": "100x1x0",
                "total_msat": "2000000msat",
                "our_amount_msat": "1000000msat",
                "updates": {
                    "remote": {"fee_proportional_millionths": 123}
                },
            }
        ]
    }
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(
        mock_plugin,
        cfg,
        mock_database,
        hive_router=membership_router,
    )

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert snapshot.channels[0].value_class == "hive"
    assert snapshot.channels[0].is_valuable is True
    membership_router.is_hive_member.assert_called_once()


def test_engine_falls_back_to_hive_equalization_when_hive_channels_have_no_budget(
    mock_plugin, mock_database
):
    from modules.rebalance_state_v2 import build_state_snapshot

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._hive_router = MagicMock(name="hive_router")
    engine._hive_router.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=0,
        route=[{"channel": "fleet"}],
        probability_ppm=0,
        error="",
    )
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=build_state_snapshot(
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
            {},
        )
    )

    selected = engine.find_candidates()

    assert len(selected) == 1
    assert selected[0].reason_code == "hive_equalization"
    assert selected[0].pair_budget_sats == 0
    engine._hive_router.price_pair.assert_called_once()
    engine.router_v3.price_pair.assert_not_called()
    assert not any(
        call.kwargs.get("reason") == "no_budget"
        or (len(call.args) > 1 and call.args[1] == "no_budget")
        for call in engine._audit.log_skip.call_args_list
    )


def test_hive_equalization_prefers_direct_same_peer_pair_when_scores_tie(
    mock_plugin, mock_database
):
    from modules.rebalance_state_v2 import build_state_snapshot

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._hive_router = MagicMock(name="hive_router")
    engine._hive_router.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=0,
        route=[{"channel": "fleet"}],
        probability_ppm=0,
        error="",
    )
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=build_state_snapshot(
            [
                {
                    "channel_id": "100x1x0",
                    "peer_id": "02" + "1" * 64,
                    "capacity_sats": 1_000_000,
                    "local_sats": 900_000,
                    "is_hive_member": True,
                },
                {
                    "channel_id": "100x2x0",
                    "peer_id": "02" + "2" * 64,
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
            {},
        )
    )

    selected = engine.find_candidates()

    assert len(selected) == 1
    assert selected[0].reason_code == "hive_equalization"
    assert selected[0].source_channel_id == "100x2x0"
    assert selected[0].dest_channel_id == "200x1x0"
    engine._hive_router.price_pair.assert_called_once()
    engine.router_v3.price_pair.assert_not_called()


def test_engine_execute_candidate_uses_router_and_executor(mock_plugin, mock_database):
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=3,
        route=[
            {
                "channel": "100x1x0",
                "id": "02" + "b" * 64,
                "amount_msat": 50_003_000,
                "delay": 30,
            }
        ],
        probability_ppm=0,
        error="",
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_msat=3000, fee_sats=3)
    )

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )

    result = engine.execute_candidate(candidate)

    assert result.success is True
    engine.router_v3.price_pair.assert_called_once()
    engine._execute_pair.assert_called_once()


def test_engine_execute_candidate_fails_closed_when_router_unavailable(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = None
    engine._execute_pair = MagicMock()

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )

    result = engine.execute_candidate(candidate)

    assert result.success is False
    assert result.error == "router_unavailable"
    engine._execute_pair.assert_not_called()


def test_engine_execute_candidate_fails_closed_when_route_pricing_raises(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.side_effect = RuntimeError("askrene offline")
    engine._execute_pair = MagicMock()

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )

    result = engine.execute_candidate(candidate)

    assert result.success is False
    assert result.error.startswith("route_pricing_failed:")
    engine._execute_pair.assert_not_called()


def test_engine_execute_candidate_continues_when_local_route_pricing_fails(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=False,
        route_cost_sats=0,
        route=[],
        probability_ppm=0,
        error="no_route",
    )

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_msat=1_000, fee_sats=1)
    )

    result = engine.execute_candidate(candidate)

    assert result.success is True
    assert result.route_type == "sling"
    engine.router_v3.price_pair.assert_called_once()
    assert engine.router_v3.price_pair.call_args.kwargs["exclude"] is None
    engine._execute_pair.assert_called_once()


def test_engine_execute_candidate_exports_failure_snapshot(mock_plugin, mock_database):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.sling_segment_observations import SlingSegmentObservationStore

    engine = _make_engine(mock_plugin, mock_database)
    engine._data_service = MagicMock()
    engine._segment_observation_store = SlingSegmentObservationStore()
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=3,
        route=[{"channel": "100x1x0", "id": "02" + "b" * 64}],
        probability_ppm=0,
        error="",
    )

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x2x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=420_000,
        max_budget_sats=100,
        reason_code="manual",
    )

    class FakeExecutor:
        def __init__(self, plugin, database, observation_store=None):
            self._store = observation_store

        def execute(self, **kwargs):
            self._store.record_failure(
                short_channel_id="200x2x0",
                direction=1,
                amount_sats=kwargs["amount_sats"],
                failure_class="liquidity",
                confidence=0.8,
                source_channel_id=kwargs["source_channel_id"],
                dest_channel_id=kwargs["dest_channel_id"],
                route_policy="network",
                router_kind="v2",
                correlation_id="corr-1",
            )
            return ExecutionResult(
                success=False,
                amount_sats=kwargs["amount_sats"],
                error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
            )

    with patch("modules.rebalance_engine_v2.RebalanceExecutor", FakeExecutor):
        result = engine.execute_candidate(candidate)

    assert result.success is False
    engine._data_service.datastore_push.assert_called_once()
    key, snapshot = engine._data_service.datastore_push.call_args.args
    assert key == ["revenue", "segment-observations"]
    assert snapshot["segment_observations"][0]["short_channel_id"] == "200x2x0"


def test_engine_applies_segment_score_bias_to_pair_score(mock_plugin, mock_database):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._our_id = "01" + "0" * 64

    class SegmentAwareHints:
        def get_segment_score(self, short_channel_id, direction, amount_sats=None):
            if short_channel_id == "100x1x0" and direction == 0:
                return {"net_utility": 0.8, "confidence": 0.8}
            if short_channel_id == "200x1x0" and direction == 1:
                return {"net_utility": 0.8, "confidence": 0.8}
            return {}

    engine._hive_hints = SegmentAwareHints()
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=250_000,
        pair_budget_sats=100,
        score=100.0,
    )

    engine._apply_segment_score_bias(pair)

    assert pair.score > 100.0


def test_engine_merges_coordination_pairs_before_pair_cap(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalance_route_policy import (
        RouteDecision,
        RoutePolicy,
        RoutePriority,
    )
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=2,
            total_remaining_budget_sats=20_000,
        )
    )

    local_pair = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="03" + "3" * 64,
        dest_peer_id="03" + "4" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=10.0,
    )
    coordinated_pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=0.1,
        reason_code="coordinated_rebalance",
        coordination_hint_type="recommendation",
        coordination_hint_id="rec-1",
        coordination_rank_bonus=90.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
            allow_market_fallback=True,
            hint_id="rec-1",
            hint_type="recommendation",
            priority_score=90.0,
        ),
    )

    route_result = SimpleNamespace(
        success=True,
        route_cost_sats=1,
        route=[],
        probability_ppm=0,
        error="",
    )
    engine.router_v3.price_pair.return_value = route_result

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls, patch(
        "modules.rebalance_engine_v2.build_coordination_overlay"
    ) as overlay_builder:
        planner = planner_cls.return_value
        planner.max_pairs = 1
        planner.plan.return_value = PlanResult(selected=[local_pair], skipped=[])
        overlay_builder.return_value = PlanResult(
            selected=[coordinated_pair],
            skipped=[],
        )

        selected = engine.find_candidates()

    assert selected == [coordinated_pair]
    engine.router_v3.price_pair.assert_called_once()


def test_engine_skips_pairs_with_persisted_cooldown_before_pricing(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine

    mock_database.get_pair_rebalance_cooldown.return_value = {
        "source_channel_id": "100x1x0",
        "dest_channel_id": "200x1x0",
        "failure_kind": "temporary_channel_failure",
        "failure_count": 2,
        "cooldown_until": int(time.time()) + 1800,
    }

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine.router_v3 = MagicMock()

    pair = SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=1.0,
        route_decision=None,
        reason_code="ev_positive",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(selected=[pair], skipped=[])

        selected = engine.find_candidates()

    assert selected == []
    engine.router_v3.price_pair.assert_not_called()
    assert any(
        call.kwargs.get("reason") == "pair_cooldown" or call.args[1] == "pair_cooldown"
        for call in engine._audit.log_skip.call_args_list
    )


def test_engine_runs_planner_selected_pair_without_local_route_snapshot(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine.find_candidates = MagicMock(
        return_value=[
            PairCandidate(
                source_channel_id="100x1x0",
                dest_channel_id="200x1x0",
                source_peer_id="02" + "b" * 64,
                dest_peer_id="02" + "c" * 64,
                amount_sats=50_000,
                pair_budget_sats=10_000,
                route=None,
            )
        ]
    )

    with patch("modules.rebalance_engine_v2.RebalanceExecutor") as executor_cls:
        executor = executor_cls.return_value
        executor.is_available.return_value = True
        executor.execute.return_value = ExecutionResult(
            success=True,
            fee_msat=1_000,
            fee_sats=1,
        )

        result = engine.run_cycle()

    assert len(result.executions) == 1
    executor.execute.assert_called_once()
    assert executor.execute.call_args.kwargs["route"] == []


def test_engine_signals_local_route_pricing_failure_without_blocking_selection(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        route=None,
    )

    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=False,
        route_cost_sats=0,
        route=[],
        probability_ppm=0,
        error="no_route",
    )
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(selected=[pair], skipped=[])

        selected = engine.find_candidates()

    assert selected == [pair]
    assert pair.route is None
    assert pair.route_cost_sats == pair.pair_budget_sats
    engine.router_v3.price_pair.assert_called_once()
    assert engine.router_v3.price_pair.call_args.kwargs["exclude"] is None
    engine._audit.log_pick.assert_not_called()
    engine._audit.log_skip.assert_called_once()
    assert engine._audit.log_skip.call_args.kwargs["reason"] == "no_route"
    assert "no_route" in engine._audit.log_skip.call_args.kwargs["detail"]


def test_engine_run_cycle_skips_when_sling_unavailable(mock_plugin, mock_database):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate
    from modules.rebalance_executor_v2 import RebalanceExecutor

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        route=[],
    )
    engine.find_candidates = MagicMock(return_value=[pair])

    with patch.object(RebalanceExecutor, "is_available", return_value=False):
        result = engine.run_cycle()

    assert result.executions == []
    engine._audit.log_skip.assert_called()


def test_engine_records_persistent_pair_failure_after_failed_execution(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(
        return_value=[
            SimpleNamespace(
                source_channel_id="100x1x0",
                dest_channel_id="200x1x0",
                amount_sats=50_000,
                pair_budget_sats=10_000,
            )
        ]
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(
            success=False,
            error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
            excluded_channels=["300x1x0/0"],
        )
    )

    engine.run_cycle()

    mock_database.record_pair_rebalance_failure.assert_called_once()
    args, kwargs = mock_database.record_pair_rebalance_failure.call_args
    assert args[:3] == ("100x1x0", "200x1x0", "temporary_channel_failure")
    assert kwargs["cooldown_seconds"] > 0


def test_engine_failed_sling_execution_returns_original_failure_without_retry(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        route=None,
    )
    executor = MagicMock()
    failure = ExecutionResult(
        success=False,
        error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
        excluded_channels=["300x1x0/0"],
    )
    executor.execute.return_value = failure

    result = engine._execute_pair(pair, executor)

    assert result is failure
    executor.execute.assert_called_once()


# Stage 2D Defect 3 regression tests: the v2 engine used to leave
# rebalance_history empty on auto cycles, making revenue-status show
# rebalance_decision.action='rebalance' alongside recent_rebalances=[].


def test_engine_execute_pair_records_pending_then_success_in_rebalance_history(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.return_value = 77

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        reason_code="ev_positive",
        route=None,
    )
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True,
        amount_sats=50_000,
        fee_sats=3,
        fee_msat=3_000,
    )

    engine._execute_pair(pair, executor)

    mock_database.record_rebalance.assert_called_once()
    rkwargs = mock_database.record_rebalance.call_args.kwargs
    assert rkwargs["from_channel"] == "100x1x0"
    assert rkwargs["to_channel"] == "200x1x0"
    assert rkwargs["amount_sats"] == 50_000
    assert rkwargs["max_fee_sats"] == 10_000
    assert rkwargs["status"] == "pending"
    assert rkwargs["reason_code"] == "ev_positive"

    mock_database.update_rebalance_result.assert_called_once()
    uargs, ukwargs = mock_database.update_rebalance_result.call_args
    assert uargs[0] == 77
    assert uargs[1] == "success"
    assert ukwargs.get("actual_fee_sats") == 3
    assert ukwargs.get("actual_fee_msat") == 3_000


def test_engine_execute_pair_records_failed_result_on_executor_failure(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.return_value = 42

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=10_000,
        pair_budget_sats=2_000,
        reason_code="ev_positive",
        route=None,
    )
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=False,
        error="sling_preflight_error: no sling",
    )

    engine._execute_pair(pair, executor)

    mock_database.record_rebalance.assert_called_once()
    mock_database.update_rebalance_result.assert_called_once()
    uargs, ukwargs = mock_database.update_rebalance_result.call_args
    assert uargs[0] == 42
    assert uargs[1] == "failed"
    assert "sling_preflight_error" in (ukwargs.get("error_message") or "")


def test_engine_execute_pair_survives_db_record_failure(
    mock_plugin, mock_database
):
    """Bookkeeping must never prevent execution. If record_rebalance raises,
    the engine still runs the executor and returns the result."""
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.side_effect = RuntimeError("db unavailable")

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        reason_code="ev_positive",
        route=None,
    )
    executor = MagicMock()
    success = ExecutionResult(success=True, amount_sats=50_000, fee_sats=5)
    executor.execute.return_value = success

    result = engine._execute_pair(pair, executor)

    assert result is success
    executor.execute.assert_called_once()
    # update_rebalance_result is guarded: when there is no rebalance_id it must
    # not be called, otherwise the engine would crash on a DB outage.
    mock_database.update_rebalance_result.assert_not_called()


def test_engine_clears_persisted_pair_failure_after_success(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(
        return_value=[
            SimpleNamespace(
                source_channel_id="100x1x0",
                dest_channel_id="200x1x0",
                amount_sats=50_000,
                pair_budget_sats=10_000,
            )
        ]
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_msat=1000, fee_sats=1)
    )

    engine.run_cycle()

    mock_database.clear_pair_rebalance_failure.assert_called_once_with(
        "100x1x0", "200x1x0"
    )
