"""Tests for the rebalance engine delegation and integration."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _audited_skip_reasons(audit_mock):
    """Skip reasons reaching a mocked audit via log_skip OR bulk log_skips."""
    reasons = []
    for call in audit_mock.log_skip.call_args_list:
        reasons.append(
            call.kwargs.get("reason")
            or (call.args[1] if len(call.args) > 1 else None)
        )
    for call in audit_mock.log_skips.call_args_list:
        skips = call.args[0] if call.args else call.kwargs.get("skips", [])
        reasons.extend(getattr(skip, "reason", None) for skip in skips)
    return reasons


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
    database.record_rebalance.return_value = 1
    database.reserve_budget.return_value = (True, 9999)
    database.mark_budget_spent.return_value = True
    database.release_budget_reservation.return_value = True
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def test_rebalancer_dry_run_calls_engine_find_candidates_without_execution(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.find_candidates.return_value = []
    r.rebalance_engine_v2.get_last_cycle_debug.return_value = {}

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.find_candidates.assert_called_once()
    r.rebalance_engine_v2.run_cycle.assert_not_called()


def test_rebalancer_live_mode_calls_engine_run_cycle(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=False)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result()

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.run_cycle.assert_called_once()


def test_engine_run_cycle_respects_max_concurrent_jobs(
    mock_plugin, mock_database
):
    from concurrent.futures import Future

    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.max_concurrent_jobs = 2
    pairs = [
        PairCandidate(
            source_channel_id=f"10{idx}x1x0",
            dest_channel_id=f"20{idx}x1x0",
            source_peer_id="02" + str(idx) * 64,
            dest_peer_id="03" + str(idx) * 64,
            amount_sats=50_000,
            pair_budget_sats=100,
        )
        for idx in range(3)
    ]
    engine.find_candidates = MagicMock(return_value=pairs)
    executor = MagicMock()
    executor.is_available.return_value = True
    engine._make_executor = MagicMock(return_value=executor)
    futures = []
    for _idx in range(2):
        future = Future()
        future.set_result(ExecutionResult(success=True, amount_sats=50_000))
        futures.append(future)
    engine._pool = MagicMock()
    engine._pool.submit.side_effect = futures

    result = engine.run_cycle()

    assert engine._pool.submit.call_count == 2
    assert len(result.candidates) == 2
    assert len(result.executions) == 2
    assert result.audit_records[-1].reason == "max_pairs_reached"
    assert "max_concurrent_jobs=2" in result.audit_records[-1].detail


def test_engine_find_candidates_passes_max_concurrent_jobs_to_planner(
    mock_plugin, mock_database
):
    from modules.rebalance_state_v2 import ChannelState

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.max_concurrent_jobs = 2
    engine.config.hive_equalization_enabled = False
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=(
                ChannelState(
                    channel_id="100x1x0",
                    peer_id="02" + "1" * 64,
                    capacity_sats=1_000_000,
                    local_ratio=0.5,
                    actual_inbound_fee_ppm=0,
                    value_class="neutral",
                    is_valuable=False,
                    remaining_budget_sats=0,
                    cooldown_active=False,
                ),
            ),
            valuable_channel_count=0,
            total_remaining_budget_sats=0,
        )
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[],
            skipped=[],
        )
        engine.find_candidates()

    assert planner_cls.call_args.kwargs["max_pairs"] == 2


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
    r.rebalance_engine_v2.find_candidates.return_value = []
    r.rebalance_engine_v2.get_last_cycle_debug.return_value = {}

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "hold"
    assert summary["safety_block"] is False


def test_rebalancer_reports_successful_execution(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_executor_v2 import ExecutionResult

    cfg = Config(dry_run=False)
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

    cfg = Config(dry_run=False)
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
        dest_out_fee_ppm=1_000,  # sats-EV gate: refill earns 25 sats expected
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
        dest_out_fee_ppm=1_000,  # sats-EV: 0.75 * 25 sats - 2 sats fee > 0
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


def test_no_routes_classified_as_transient_failure(mock_plugin, mock_database):
    """NoRoutes should map to the short-cooldown transient bucket, not the
    catch-all other_retriable bucket."""
    from modules.rebalance_engine_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)

    for err in (
        "retriable_failure: NoRoutes",
        "retriable_failure: no_routes",
        "no route to destination",
    ):
        result = ExecutionResult(error=err)
        assert engine._classify_failure_kind(result) == "temporary_channel_failure"


def test_no_route_pairs_are_not_selected_without_priced_route(
    mock_plugin, mock_database
):
    """Native execution requires an explicit priced route, so router no-route
    pairs must be skipped before execution."""
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
            channels=[object()], valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=500,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=1.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=False, route_cost_sats=None, route=None,
        probability_ppm=0, error="no_route_found",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[pair], skipped=[]
        )
        selected = engine.find_candidates()

    assert selected == []
    debug = engine.get_last_cycle_debug()
    skipped_reasons = {row["channel_id"]: row["reason"] for row in debug["skipped"]}
    assert skipped_reasons.get("200x2x0") == "no_route"


def test_first_transient_failure_uses_short_cooldown(mock_plugin, mock_database):
    """Iter2: temporary_channel_failure base cooldown is 300s (5 min) on a
    first failure so good pairs do not disappear for half an hour after a
    single transient route failure."""
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
    engine.find_candidates.return_value = []
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
        dest_out_fee_ppm=500,     # refill earns 100 sats expected vs 5 sat fee
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

    # pair_fee_cap_ppm raised so the per-attempt ceiling (F1) does not
    # reject the route before the hold-margin gate gets to decide.
    cfg = Config(
        dry_run=True,
        rebalance_router="v3",
        rebalance_hold_margin=0.05,
        pair_fee_cap_ppm=5_000,
    )
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


def test_negative_ev_pair_is_rejected_at_default_hold_margin(
    mock_plugin, mock_database
):
    """Default hold margin is zero, but zero still means do not execute
    pairs whose priced final_score loses to do_nothing."""
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

    weak_pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=100,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=0.10,
        source_local_ratio=0.70,
        dest_local_ratio=0.30,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True, route_cost_sats=50, route=[{"channel": "300x3x0"}],
        probability_ppm=990_000, error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[weak_pair], skipped=[]
        )
        selected = engine.find_candidates()

    assert selected == []
    debug = engine.get_last_cycle_debug()
    skipped_reasons = {row["channel_id"]: row["reason"] for row in debug["skipped"]}
    assert skipped_reasons.get("200x2x0") == "below_hold_margin"
    assert debug["considered_candidates"][0]["score_decomposition"]["final_score"] < 0.0


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
    assert (
        snapshot.channels[0].remaining_budget_sats
        == cfg.hive_rebalance_bootstrap_budget_sats
    )
    assert snapshot.channels[0].budget_source == "hive_bootstrap"
    membership_router.is_hive_member.assert_called_once()


def test_engine_build_snapshot_carries_hive_rebalance_bias(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    peer_id = "03" + "b" * 64
    cfg = Config(dry_run=True, rebalance_router="v3")
    hive_hints = MagicMock()
    hive_hints.get_rebalance_bias.return_value = 1.05
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": peer_id,
                "short_channel_id": "100x1x0",
                "total_msat": "2000000msat",
                "our_amount_msat": "1000000msat",
                "updates": {"remote": {"fee_proportional_millionths": 123}},
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
        hive_hints=hive_hints,
    )

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert snapshot.channels[0].rebalance_bias == pytest.approx(1.05)
    hive_hints.get_rebalance_bias.assert_called_once_with(peer_id)


def test_engine_build_snapshot_toggle_off_carries_flat_bands_for_all_channels(
    mock_plugin, mock_database
):
    """FIX 3: with rebalance_size_tiered_targets=False, the engine must not
    build a target_bands map at all -- every channel's per-channel band
    falls back to the flat cfg thresholds, identical to pre-feature
    behavior, even when channel capacities differ wildly."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(
        dry_run=True,
        rebalance_router="v3",
        rebalance_size_tiered_targets=False,
    )
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
                "updates": {"remote": {"fee_proportional_millionths": 123}},
            },
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "03" + "c" * 64,
                "short_channel_id": "200x2x0",
                # 20x the capacity of the first channel -- would get a
                # size-tiered wider band if the toggle were honored.
                "total_msat": "40000000msat",
                "our_amount_msat": "20000000msat",
                "updates": {"remote": {"fee_proportional_millionths": 123}},
            },
        ]
    }
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert len(snapshot.channels) == 2
    flat_low = cfg.low_liquidity_threshold
    flat_high = cfg.high_liquidity_threshold
    for channel in snapshot.channels:
        assert channel.target_band_low == flat_low
        assert channel.target_band_high == flat_high


def test_engine_build_snapshot_toggle_on_gives_big_channel_wider_band_than_small(
    mock_plugin, mock_database
):
    """FIX 3: with the toggle ON and a mixed-size channel set, a big
    channel's ChannelState must carry a wider per-channel band than a
    small channel's."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(
        dry_run=True,
        rebalance_router="v3",
        rebalance_size_tiered_targets=True,
    )
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
                "updates": {"remote": {"fee_proportional_millionths": 123}},
            },
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": "03" + "c" * 64,
                "short_channel_id": "200x2x0",
                "total_msat": "40000000msat",
                "our_amount_msat": "20000000msat",
                "updates": {"remote": {"fee_proportional_millionths": 123}},
            },
        ]
    }
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    by_id = {c.channel_id: c for c in snapshot.channels}
    small = by_id["100x1x0"]
    big = by_id["200x2x0"]
    small_half_width = small.target_band_high - small.target_band_low
    big_half_width = big.target_band_high - big.target_band_low
    assert big_half_width > small_half_width
    assert small.target_band_low == cfg.low_liquidity_threshold
    assert small.target_band_high == cfg.high_liquidity_threshold


def test_v2_planner_uses_hive_rebalance_bias_for_pair_roles():
    from modules.rebalance_planner_v2 import RebalancePlanner
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    disfavored_source = "100x1x0"
    favored_source = "101x1x0"
    dest = "200x1x0"
    snapshot = build_state_snapshot(
        [
            ChannelInput(
                channel_id=disfavored_source,
                peer_id="03" + "1" * 64,
                capacity_sats=1_000_000,
                local_sats=900_000,
                rebalance_bias=1.05,
            ),
            ChannelInput(
                channel_id=favored_source,
                peer_id="03" + "2" * 64,
                capacity_sats=1_000_000,
                local_sats=900_000,
                rebalance_bias=0.95,
            ),
            ChannelInput(
                channel_id=dest,
                peer_id="03" + "3" * 64,
                capacity_sats=1_000_000,
                local_sats=100_000,
                rebalance_bias=1.05,
            ),
        ],
        {"channel_budgets": {dest: {"budget_sats": 1_000}}},
    )

    selected = RebalancePlanner(max_pairs=1).plan(snapshot).selected

    assert len(selected) == 1
    pair = selected[0]
    assert pair.source_channel_id == favored_source
    assert pair.dest_channel_id == dest
    assert pair.hive_source_rebalance_bias == pytest.approx(0.95)
    assert pair.hive_dest_rebalance_bias == pytest.approx(1.05)
    assert pair.hive_hint_score_multiplier == pytest.approx(1.10)
    inputs = pair.score_decomposition["inputs"]
    assert inputs["hive_source_rebalance_bias"] == pytest.approx(0.95)
    assert inputs["hive_dest_rebalance_bias"] == pytest.approx(1.05)
    assert inputs["hive_hint_score_multiplier"] == pytest.approx(1.10)
    assert pair.score > inputs["pre_hint_pair_score"]


def test_v2_planner_carries_historical_profitability_role_metrics():
    from modules.rebalance_planner_v2 import RebalancePlanner
    from modules.rebalance_state_v2 import build_state_snapshot

    source = "100x1x0"
    dest = "200x1x0"
    snapshot = build_state_snapshot(
        [
            {
                "channel_id": source,
                "peer_id": "03" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
                "historical_direct_fee_ppm": 50.0,
                "historical_sourced_fee_ppm": 100.0,
            },
            {
                "channel_id": dest,
                "peer_id": "03" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
                "is_profitable": True,
                "historical_direct_fee_ppm": 200.0,
                "historical_sourced_fee_ppm": 25.0,
            },
        ],
        {"channel_budgets": {dest: {"budget_sats": 1_000}}},
    )

    selected = RebalancePlanner(max_pairs=1).plan(snapshot).selected

    assert len(selected) == 1
    pair = selected[0]
    assert getattr(pair, "source_historical_direct_fee_ppm", 0.0) == pytest.approx(50.0)
    assert getattr(pair, "source_historical_sourced_fee_ppm", 0.0) == pytest.approx(100.0)
    assert getattr(pair, "dest_historical_direct_fee_ppm", 0.0) == pytest.approx(200.0)
    assert getattr(pair, "dest_historical_sourced_fee_ppm", 0.0) == pytest.approx(25.0)


def test_c1_planner_accepts_known_good_pair_and_explains_no_budget_destination():
    from modules.rebalance_planner_v2 import RebalancePlanner
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    source = "100x1x0"
    known_good_dest = "200x1x0"
    no_budget_dest = "201x1x0"
    snapshot = build_state_snapshot(
        [
            ChannelInput(
                channel_id=source,
                peer_id="03" + "1" * 64,
                capacity_sats=1_000_000,
                local_sats=900_000,
            ),
            ChannelInput(
                channel_id=known_good_dest,
                peer_id="03" + "2" * 64,
                capacity_sats=1_000_000,
                local_sats=100_000,
                is_hive_member=True,
                rebalance_bias=1.05,
            ),
            ChannelInput(
                channel_id=no_budget_dest,
                peer_id="03" + "3" * 64,
                capacity_sats=1_000_000,
                local_sats=100_000,
                is_active=True,
            ),
        ],
        {"channel_budgets": {known_good_dest: {"budget_sats": 500}}},
    )

    by_channel = {channel.channel_id: channel for channel in snapshot.channels}
    assert by_channel[source].source_eligible is True
    assert by_channel[known_good_dest].dest_eligible is True
    assert by_channel[known_good_dest].remaining_budget_sats == 500
    assert by_channel[no_budget_dest].dest_eligible is False
    assert by_channel[no_budget_dest].dest_reason == "no_budget"
    assert snapshot.valuable_channel_count == 2
    assert snapshot.total_remaining_budget_sats == 500

    plan = RebalancePlanner(max_pairs=2).plan(snapshot)

    assert len(plan.selected) == 1
    pair = plan.selected[0]
    assert pair.source_channel_id == source
    assert pair.dest_channel_id == known_good_dest
    assert pair.amount_sats == 250_000
    assert pair.pair_budget_sats == 500
    assert pair.dest_value_class == "hive"
    assert pair.score_decomposition["beats_do_nothing"] is True
    skipped = {skip.channel_id: skip.reason for skip in plan.skipped}
    assert skipped[no_budget_dest] == "no_budget"


def test_c1_engine_selects_known_good_pair_with_positive_margin(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=1_000,
        )
    )
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="03" + "1" * 64,
        dest_peer_id="03" + "2" * 64,
        amount_sats=250_000,
        pair_budget_sats=1_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        source_value_class="neutral",
        dest_value_class="hive",
        source_budget_source="none",
        dest_budget_source="capex",
        dest_out_fee_ppm=200,  # sats-EV: 0.9 * 25 sats - 6 sats fee > 0
        score=2.0,
        source_local_ratio=0.90,
        dest_local_ratio=0.10,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=6,
        route=[{"channel": "300x1x0"}],
        probability_ppm=900_000,
        error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[pair],
            skipped=[],
        )
        selected = engine.find_candidates()

    assert selected == [pair]
    debug = engine.get_last_cycle_debug()
    assert debug["summary"]["selected_pairs"] == 1
    decomposition = debug["selected_candidates"][0]["score_decomposition"]
    assert decomposition["stage"] == "priced"
    assert decomposition["beats_do_nothing"] is True
    assert decomposition["final_score"] > 0.0
    assert decomposition["inputs"]["expected_fee_sats"] == 6
    assert decomposition["inputs"]["pair_budget_sats"] == 1_000


def test_c1_engine_rejects_route_bait_even_when_route_prices(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    # Lift the per-attempt ppm ceiling (F1) above the bait route's cost so
    # this test keeps exercising the hold-margin gate, not the ceiling.
    engine.config.pair_fee_cap_ppm = 10_000
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=100,
        )
    )
    bait_pair = PairCandidate(
        source_channel_id="101x1x0",
        dest_channel_id="201x1x0",
        source_peer_id="03" + "3" * 64,
        dest_peer_id="03" + "4" * 64,
        amount_sats=20_000,
        pair_budget_sats=100,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        source_value_class="neutral",
        dest_value_class="active",
        source_budget_source="none",
        dest_budget_source="capex",
        score=0.20,
        source_local_ratio=0.68,
        dest_local_ratio=0.30,
    )
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=90,
        route=[{"channel": "301x1x0"}],
        probability_ppm=990_000,
        error="",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = SimpleNamespace(
            selected=[bait_pair],
            skipped=[],
        )
        selected = engine.find_candidates()

    assert selected == []
    debug = engine.get_last_cycle_debug()
    assert debug["summary"]["considered_pairs"] == 1
    assert debug["summary"]["selected_pairs"] == 0
    decomposition = debug["considered_candidates"][0]["score_decomposition"]
    assert decomposition["stage"] == "below_hold_margin"
    assert decomposition["rejection_reason"] == "below_hold_margin"
    assert decomposition["beats_do_nothing"] is False
    # Audit F3: no capital-risk double count at default config (the
    # effective budget equals the raw budget, so the term is dropped).
    assert decomposition["capital_risk_penalty"] == 0.0
    assert decomposition["final_score_sats"] < 0.0
    assert decomposition["inputs"]["expected_fee_sats"] == 90
    skipped = {row["channel_id"]: row["reason"] for row in debug["skipped"]}
    assert skipped["201x1x0"] == "below_hold_margin"


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
    assert result.route_type == "native"
    engine.router_v3.price_pair.assert_called_once()
    assert engine.router_v3.price_pair.call_args.kwargs["exclude"] is None
    engine._execute_pair.assert_called_once()


def test_engine_execute_candidate_exports_failure_snapshot(mock_plugin, mock_database):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.segment_observations import SegmentObservationStore

    engine = _make_engine(mock_plugin, mock_database)
    engine._data_service = MagicMock()
    engine._segment_observation_store = SegmentObservationStore()
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
        def __init__(self, plugin, database, observation_store=None, our_id=None):
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

    with patch("modules.rebalance_engine_v2.NativeRouteExecutor", FakeExecutor):
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
        dest_out_fee_ppm=1_000,
        score=10.0,
    )
    coordinated_pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        dest_out_fee_ppm=1_000,
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

    assert selected == [coordinated_pair, local_pair]
    assert engine.router_v3.price_pair.call_count == 2


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
    assert "pair_cooldown" in _audited_skip_reasons(engine._audit)


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

    with patch("modules.rebalance_engine_v2.NativeRouteExecutor") as executor_cls:
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


def test_engine_native_executor_consumes_priced_route(mock_plugin, mock_database):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_executor = "native"
    priced_route = [
        {
            "id": "02" + "b" * 64,
            "channel": "100x1x0",
            "amount_msat": 50_010_000,
            "delay": 40,
        },
        {
            "id": "03" + "u" * 64,
            "channel": "200x1x0",
            "amount_msat": 50_000_000,
            "delay": 18,
        },
    ]
    engine.find_candidates = MagicMock(
        return_value=[
            PairCandidate(
                source_channel_id="100x1x0",
                dest_channel_id="200x1x0",
                source_peer_id="02" + "b" * 64,
                dest_peer_id="02" + "c" * 64,
                amount_sats=50_000,
                pair_budget_sats=10,
                route=priced_route,
            )
        ]
    )

    with patch("modules.rebalance_engine_v2.NativeRouteExecutor") as executor_cls:
        executor = executor_cls.return_value
        executor.is_available.return_value = True
        executor.execute.return_value = ExecutionResult(
            success=True,
            fee_msat=10_000,
            fee_sats=10,
            route_type="native",
        )

        result = engine.run_cycle()

    assert len(result.executions) == 1
    assert result.executions[0].route_type == "native"
    executor.execute.assert_called_once()
    assert executor.execute.call_args.kwargs["route"] == priced_route


def test_engine_native_executor_retries_with_failed_segment_excluded(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_executor = "native"
    engine._cycle_router = MagicMock()
    engine._hive_router = None
    original_route = [
        {
            "id": "02" + "b" * 64,
            "channel": "100x1x0",
            "amount_msat": 50_010_000,
            "delay": 40,
        },
        {
            "id": "03" + "u" * 64,
            "channel": "200x1x0",
            "amount_msat": 50_000_000,
            "delay": 18,
        },
    ]
    retry_route = [
        {
            "id": "02" + "b" * 64,
            "channel": "100x1x0",
            "amount_msat": 50_011_000,
            "delay": 40,
        },
        {
            "id": "02" + "d" * 64,
            "channel": "175x1x0",
            "direction": 0,
            "amount_msat": 50_010_000,
            "delay": 30,
        },
        {
            "id": "03" + "u" * 64,
            "channel": "200x1x0",
            "amount_msat": 50_000_000,
            "delay": 18,
        },
    ]
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=20,
        route=original_route,
    )
    engine._route_pair = MagicMock(
        return_value=(
            SimpleNamespace(
                success=True,
                route_cost_sats=11,
                route=retry_route,
                probability_ppm=0,
                error="",
            ),
            "market",
        )
    )
    executor = MagicMock()
    first = ExecutionResult(
        success=False,
        amount_sats=50_000,
        route_type="native",
        attempts=1,
        error="native_sendpay_error: failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
        excluded_channels=["150x1x0/1"],
    )
    second = ExecutionResult(
        success=True,
        amount_sats=50_000,
        route_type="native",
        attempts=1,
        fee_sats=11,
        fee_msat=11_000,
    )
    executor.execute.side_effect = [first, second]

    result = engine._execute_pair(pair, executor)

    assert result is second
    assert result.success is True
    assert result.attempts == 2
    assert result.failure_data["previous_failure"].startswith("native_sendpay_error")
    assert result.failure_data["retry_excluded_channels"] == ["150x1x0/1"]
    engine._route_pair.assert_called_once()
    assert engine._route_pair.call_args.kwargs["exclude"] == ["150x1x0/1"]
    assert executor.execute.call_count == 2
    assert executor.execute.call_args.kwargs["route"] == retry_route


def test_engine_native_executor_retries_smaller_partial_amount_after_liquidity_failure(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_executor = "native"
    engine._cycle_router = MagicMock()
    engine._hive_router = None
    mock_database.record_rebalance.return_value = 123

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=100_000,
        pair_budget_sats=50,
        dest_capacity_sats=1_000_000,
        dest_local_ratio=0.10,
        route=[{"id": "02" + "b" * 64, "channel": "100x1x0", "amount_msat": 100_010_000}],
    )
    partial_route = [
        {
            "id": "02" + "b" * 64,
            "channel": "100x1x0",
            "amount_msat": 50_006_000,
            "delay": 40,
        },
        {
            "id": "03" + "u" * 64,
            "channel": "200x1x0",
            "amount_msat": 50_000_000,
            "delay": 18,
        },
    ]
    engine._route_pair = MagicMock(
        side_effect=[
            (
                SimpleNamespace(success=False, error="no_route", route_cost_sats=0),
                "market",
            ),
            (
                SimpleNamespace(
                    success=True,
                    route_cost_sats=6,
                    route=partial_route,
                    probability_ppm=0,
                    error="",
                ),
                "market",
            ),
        ]
    )
    executor = MagicMock()
    first = ExecutionResult(
        success=False,
        amount_sats=100_000,
        route_type="native",
        attempts=1,
        error="native_sendpay_error: failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
        excluded_channels=["150x1x0/1"],
        failure_data={"failure_class": "liquidity"},
    )
    partial_success = ExecutionResult(
        success=True,
        amount_sats=50_000,
        route_type="native",
        attempts=1,
        fee_sats=6,
        fee_msat=6_000,
    )
    executor.execute.side_effect = [first, partial_success]

    result = engine._execute_pair(pair, executor)

    assert result.success is True
    assert result.amount_sats == 50_000
    assert result.attempts == 3
    assert result.failure_data["partial_fill"]["planned_amount_sats"] == 100_000
    assert result.failure_data["partial_fill"]["executed_amount_sats"] == 50_000
    assert engine._route_pair.call_args_list[1].kwargs["exclude"] is None
    assert executor.execute.call_count == 2
    assert executor.execute.call_args.kwargs["amount_sats"] == 50_000
    assert executor.execute.call_args.kwargs["route"] == partial_route

    uargs, ukwargs = mock_database.update_rebalance_result.call_args
    assert uargs[0] == 123
    assert uargs[1] == "success"
    assert ukwargs["amount_sats"] == 50_000
    assert ukwargs["post_local_ratio"] == pytest.approx(0.15)


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

    # Native execution requires a priced explicit route, so no-route pairs are
    # skipped before reaching execution. The skip is surfaced through the
    # unified plan.skipped audit loop.
    assert selected == []
    engine.router_v3.price_pair.assert_called_once()
    assert engine.router_v3.price_pair.call_args.kwargs["exclude"] is None
    engine._audit.log_pick.assert_not_called()
    reasons = _audited_skip_reasons(engine._audit)
    assert "no_route" in reasons, f"expected a no_route skip; got {reasons}"


def test_engine_run_cycle_skips_when_native_executor_unavailable(mock_plugin, mock_database):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate
    from modules.rebalance_native_executor_v2 import NativeRouteExecutor

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

    with patch.object(NativeRouteExecutor, "is_available", return_value=False):
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

    with patch("modules.rebalance_engine_v2.NativeRouteExecutor.is_available", return_value=True):
        engine.run_cycle()

    mock_database.record_pair_rebalance_failure.assert_called_once()
    args, kwargs = mock_database.record_pair_rebalance_failure.call_args
    assert args[:3] == ("100x1x0", "200x1x0", "temporary_channel_failure")
    assert kwargs["cooldown_seconds"] > 0


def test_engine_does_not_cooldown_cancelled_execution_after_cycle_timeout(
    mock_plugin, mock_database
):
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    class FakeFuture:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True
            return True

    pair = SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        amount_sats=50_000,
        pair_budget_sats=10_000,
    )
    future = FakeFuture()
    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(return_value=[pair])
    engine._pool = MagicMock()
    engine._pool.submit.return_value = future

    with patch("modules.rebalance_engine_v2.NativeRouteExecutor") as executor_cls, patch(
        "modules.rebalance_engine_v2.as_completed",
        side_effect=FuturesTimeoutError,
    ):
        executor_cls.return_value.is_available.return_value = True
        result = engine.run_cycle()

    assert future.cancelled is True
    assert len(result.executions) == 1
    assert result.executions[0].error == "executor_timeout_cancelled"
    mock_database.record_pair_rebalance_failure.assert_not_called()


def test_engine_does_not_cooldown_still_running_execution_after_cycle_timeout(
    mock_plugin, mock_database
):
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    class FakeFuture:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True
            return False

    pair = SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        amount_sats=50_000,
        pair_budget_sats=10_000,
    )
    future = FakeFuture()
    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(return_value=[pair])
    engine._pool = MagicMock()
    engine._pool.submit.return_value = future

    with patch("modules.rebalance_engine_v2.NativeRouteExecutor") as executor_cls, patch(
        "modules.rebalance_engine_v2.as_completed",
        side_effect=FuturesTimeoutError,
    ):
        executor_cls.return_value.is_available.return_value = True
        result = engine.run_cycle()

    assert future.cancelled is True
    assert result.executions == []
    mock_database.record_pair_rebalance_failure.assert_not_called()


def test_engine_failed_non_native_retry_execution_returns_original_failure_without_retry(
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


def test_engine_auto_execute_pair_blocks_zero_budget_even_for_zero_cost_route(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.daily_budget_sats = 0
    engine.config.allow_zero_cost_auto_rebalance_when_budget_zero = False
    mock_database.record_rebalance.return_value = 66

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=0,
        reason_code="ev_positive",
        route=[],
    )
    executor = MagicMock()

    result = engine._execute_pair(pair, executor, reserve_budget=True, account_costs=True)

    assert result.success is False
    assert result.error == "zero_budget_blocks_auto_rebalance"
    executor.execute.assert_not_called()
    mock_database.reserve_budget.assert_not_called()
    mock_database.update_rebalance_result.assert_called_once()


def test_engine_auto_execute_pair_blocks_zero_budget_for_positive_cost_route(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.daily_budget_sats = 0
    engine.config.allow_zero_cost_auto_rebalance_when_budget_zero = False
    mock_database.record_rebalance.return_value = 67

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=100,
        reason_code="ev_positive",
        route=[],
    )
    executor = MagicMock()

    result = engine._execute_pair(pair, executor, reserve_budget=True, account_costs=True)

    assert result.success is False
    assert result.error == "zero_budget_blocks_auto_rebalance"
    executor.execute.assert_not_called()
    mock_database.reserve_budget.assert_not_called()


def test_engine_auto_execute_pair_allows_zero_cost_route_when_budget_positive(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.daily_budget_sats = 5000
    engine.config.allow_zero_cost_auto_rebalance_when_budget_zero = False
    mock_database.record_rebalance.return_value = 68

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=0,
        reason_code="ev_positive",
        route=[],
    )
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(success=True, amount_sats=50_000, fee_sats=0, fee_msat=0)

    result = engine._execute_pair(pair, executor, reserve_budget=True, account_costs=True)

    assert result.success is True
    executor.execute.assert_called_once()
    mock_database.reserve_budget.assert_not_called()


def test_engine_auto_execute_pair_allows_zero_cost_route_with_explicit_zero_budget_override(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.daily_budget_sats = 0
    engine.config.allow_zero_cost_auto_rebalance_when_budget_zero = True
    mock_database.record_rebalance.return_value = 69

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=0,
        reason_code="operator_allows_zero_cost",
        route=[],
    )
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(success=True, amount_sats=50_000, fee_sats=0, fee_msat=0)

    result = engine._execute_pair(pair, executor, reserve_budget=True, account_costs=True)

    assert result.success is True
    executor.execute.assert_called_once()
    mock_database.reserve_budget.assert_not_called()


def test_engine_auto_execute_pair_accounts_costs_and_budget_reservation(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.return_value = 77
    mock_database.reserve_budget.return_value = (True, 9_900)

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
        fee_msat=2_500,
    )

    engine._execute_pair(
        pair,
        executor,
        reserve_budget=True,
        account_costs=True,
    )

    mock_database.reserve_budget.assert_called_once()
    rkwargs = mock_database.reserve_budget.call_args.kwargs
    assert rkwargs["reservation_id"] == "77"
    assert rkwargs["amount_sats"] == 10_000
    assert rkwargs["channel_id"] == "200x1x0"

    mock_database.record_rebalance_cost.assert_called_once()
    ckwargs = mock_database.record_rebalance_cost.call_args.kwargs
    assert ckwargs["channel_id"] == "200x1x0"
    assert ckwargs["peer_id"] == "02" + "c" * 64
    assert ckwargs["cost_sats"] == 3
    assert ckwargs["cost_msat"] == 2_500
    assert ckwargs["amount_sats"] == 50_000
    mock_database.mark_budget_spent.assert_called_once_with("77", 3)
    mock_database.release_budget_reservation.assert_not_called()


def test_engine_auto_execute_pair_releases_budget_on_failure(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.return_value = 88
    mock_database.reserve_budget.return_value = (True, 9_900)

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
        success=False,
        error="retriable_failure: NoRoutes",
    )

    result = engine._execute_pair(
        pair,
        executor,
        reserve_budget=True,
        account_costs=True,
    )

    assert result.success is False
    mock_database.record_rebalance_cost.assert_not_called()
    mock_database.mark_budget_spent.assert_not_called()
    mock_database.release_budget_reservation.assert_called_once_with("88")


def test_engine_auto_execute_pair_blocks_when_budget_reservation_fails(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    mock_database.record_rebalance.return_value = 99
    mock_database.reserve_budget.return_value = (False, 5)

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

    result = engine._execute_pair(
        pair,
        executor,
        reserve_budget=True,
        account_costs=True,
    )

    assert result.success is False
    assert "local_budget_block" in result.error
    executor.execute.assert_not_called()
    mock_database.update_rebalance_result.assert_called_once()
    uargs, ukwargs = mock_database.update_rebalance_result.call_args
    assert uargs[0] == 99
    assert uargs[1] == "failed"
    assert "local_budget_block" in (ukwargs.get("error_message") or "")


def test_engine_explicit_execute_candidate_leaves_budget_to_caller(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalancer import RebalanceCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._active_router = MagicMock(return_value=MagicMock())
    engine._route_pair = MagicMock(
        return_value=(
            SimpleNamespace(success=True, route_cost_sats=1, route=[]),
            "market",
        )
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_sats=1, fee_msat=1000)
    )

    candidate = RebalanceCandidate(
        source_candidates=["100x1x0"],
        to_channel="200x1x0",
        primary_source_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        amount_msat=50_000_000,
        outbound_fee_ppm=0,
        inbound_fee_ppm=0,
        source_fee_ppm=0,
        weighted_opp_cost_ppm=0,
        spread_ppm=0,
        max_budget_sats=10_000,
        max_budget_msat=10_000_000,
        max_fee_ppm=200_000,
        expected_profit_sats=0,
        liquidity_ratio=0.5,
        dest_flow_state="manual",
        dest_turnover_rate=0.0,
        source_turnover_rate=0.0,
    )

    engine.execute_candidate(candidate)

    assert engine._execute_pair.call_args.kwargs["reserve_budget"] is False
    assert engine._execute_pair.call_args.kwargs["account_costs"] is False


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
        error="native_sendpay_error: temporary_channel_failure",
    )

    engine._execute_pair(pair, executor)

    mock_database.record_rebalance.assert_called_once()
    mock_database.update_rebalance_result.assert_called_once()
    uargs, ukwargs = mock_database.update_rebalance_result.call_args
    assert uargs[0] == 42
    assert uargs[1] == "failed"
    assert "native_sendpay_error" in (ukwargs.get("error_message") or "")


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

    with patch("modules.rebalance_engine_v2.NativeRouteExecutor.is_available", return_value=True):
        engine.run_cycle()

    mock_database.clear_pair_rebalance_failure.assert_called_once_with(
        "100x1x0", "200x1x0"
    )


def test_metabolic_rebalance_bias_is_bounded_score_modifier(mock_plugin, mock_database):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    hive_hints = MagicMock()
    hive_hints.get_metabolic_rebalance_bias.return_value = 1.50
    hive_hints.get_metabolic_peer_effect.return_value = {
        "usable": True,
        "bias_capped": True,
        "reason_codes": ["positive_net_usable_energy"],
    }
    engine._hive_hints = hive_hints
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02source",
        dest_peer_id="02dest",
        amount_sats=100_000,
        pair_budget_sats=100,
        score=10.0,
    )

    engine._apply_metabolic_rebalance_bias(pair)

    assert pair.score == pytest.approx(11.5)
    assert pair.metabolic_rebalance_bias == pytest.approx(1.15)
    assert pair.metabolic_rebalance_influence["bias_capped"] is True
    engine._update_pair_score_decomposition(pair, route_status="planned")
    assert pair.score_decomposition["inputs"]["metabolic_rebalance_bias"] == pytest.approx(1.15)


def test_immune_rebalance_bias_is_bounded_score_modifier(mock_plugin, mock_database):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    hive_hints = MagicMock()
    hive_hints.get_immune_rebalance_bias.return_value = 0.50
    hive_hints.get_immune_peer_effect.return_value = {
        "usable": True,
        "bias_capped": True,
        "reason_codes": ["toxic_peer_detected"],
    }
    engine._hive_hints = hive_hints
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02source",
        dest_peer_id="02dest",
        amount_sats=100_000,
        pair_budget_sats=100,
        score=10.0,
    )

    engine._apply_immune_rebalance_bias(pair)

    assert pair.score == pytest.approx(8.5)
    assert pair.immune_rebalance_bias == pytest.approx(0.85)
    assert pair.immune_rebalance_influence["bias_capped"] is True
    engine._update_pair_score_decomposition(pair, route_status="planned")
    assert pair.score_decomposition["inputs"]["immune_rebalance_bias"] == pytest.approx(0.85)


# -----------------------------------------------------------------------------
# Score-unit normalization: coordination/equalization scores live in planner
# units so the hold-margin EV gate and final ordering are meaningful.
# -----------------------------------------------------------------------------


class _OverlayHints:
    """Minimal hive-hints double for overlay-driven engine tests."""

    def __init__(self, recommendations=None, campaigns=None, leases=None):
        self._recommendations = list(recommendations or [])
        self._campaigns = list(campaigns or [])
        self._leases = list(leases or [])

    def is_fresh(self):
        return True

    def is_hive_member(self, peer_id):
        return False

    def get_rebalance_recommendations(self):
        return list(self._recommendations)

    def get_rebalance_campaigns(self):
        return list(self._campaigns)

    def get_route_segment_leases(self):
        return list(self._leases)

    def get_segment_score(self, short_channel_id, direction, amount_sats=None):
        return {}


def _overlay_snapshot():
    from modules.rebalance_state_v2 import build_state_snapshot

    return build_state_snapshot(
        [
            {
                "channel_id": "100x1x0",
                "peer_id": "02" + "1" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 900_000,
                "local_out_fee_ppm": 50,
            },
            {
                "channel_id": "200x1x0",
                "peer_id": "02" + "2" * 64,
                "capacity_sats": 1_000_000,
                "local_sats": 100_000,
                "local_out_fee_ppm": 500,
            },
        ],
        {"channel_budgets": {"200x1x0": {"budget_sats": 10_000}}},
    )


def test_hive_equalization_score_is_normalized_to_planner_units(
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

    from modules.rebalance_state_v2 import _drain_score, _refill_urgency

    assert len(selected) == 1
    assert selected[0].reason_code == "hive_equalization"
    # Audit F4: planner coefficients — 0.30 x urgency + 0.20 x drain.
    expected = (
        0.30 * _refill_urgency(0.1, 0.35) + 0.20 * _drain_score(0.9, 0.65)
    )
    assert selected[0].score == pytest.approx(expected, rel=1e-4)
    assert selected[0].score < 2.0


def test_coordination_pair_with_negative_ev_is_rejected_by_hold_margin(
    mock_plugin, mock_database
):
    """Pre-normalization, sats-scale coordination scores (e.g. 250k) made
    p*score dwarf every penalty so the hold-margin gate could never reject
    a coordination pair. Normalized scores make the gate real."""
    engine = _make_engine(mock_plugin, mock_database)
    # Lift the per-attempt ppm ceiling (F1) above the route cost so the
    # sats-EV hold-margin gate is what rejects this pair.
    engine.config.pair_fee_cap_ppm = 100_000
    engine.router_v3 = MagicMock(name="market_router")
    # Route prices at nearly the entire pair budget: the sats EV is deeply
    # negative, so the hold-margin gate must reject.
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=9_999,
        route=[{"channel": "100x1x0"}],
        probability_ppm=0,
        error="",
    )
    engine._audit = MagicMock()
    engine._hive_hints = _OverlayHints(
        recommendations=[
            {
                "recommendation_id": "rec-negative-ev",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "route_policy": "market_only",
                "priority_score": 90.0,
            }
        ]
    )
    engine._build_snapshot = MagicMock(return_value=_overlay_snapshot())

    selected = engine.find_candidates()

    assert selected == []
    skip_reasons = {
        call.kwargs.get("reason")
        for call in engine._audit.log_skip.call_args_list
    }
    assert "below_hold_margin" in skip_reasons


def test_coordination_pair_with_positive_ev_survives_hold_margin(
    mock_plugin, mock_database
):
    """Sanity inverse: a cheap route keeps the coordination pair selected."""
    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=5,
        route=[{"channel": "100x1x0"}],
        probability_ppm=0,
        error="",
    )
    engine._audit = MagicMock()
    engine._hive_hints = _OverlayHints(
        recommendations=[
            {
                "recommendation_id": "rec-positive-ev",
                "source_scid": "100x1x0",
                "sink_scid": "200x1x0",
                "amount_sats": 120_000,
                "route_policy": "market_only",
                "priority_score": 90.0,
            }
        ]
    )
    engine._build_snapshot = MagicMock(return_value=_overlay_snapshot())

    selected = engine.find_candidates()

    assert len(selected) >= 1
    assert any(p.coordination_hint_id == "rec-positive-ev" for p in selected)


def test_low_merit_coordination_final_score_below_high_merit_planner_pair(
    mock_plugin, mock_database
):
    """With shared units, final_score comparisons across pair origins are
    meaningful: a low-merit coordination pair no longer trivially outranks
    a high-merit planner pair."""
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)

    coordination_pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "1" * 64,
        dest_peer_id="02" + "2" * 64,
        amount_sats=40_000,
        pair_budget_sats=1_000,
        score=0.1,  # normalized low-merit coordination score
        reason_code="coordinated_rebalance",
        coordination_hint_id="rec-low",
    )
    planner_pair = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="02" + "3" * 64,
        dest_peer_id="02" + "4" * 64,
        amount_sats=40_000,
        pair_budget_sats=1_000,
        score=1.2,  # high-merit planner score
    )

    coord_decomp = engine._build_score_decomposition(
        coordination_pair, route_cost_sats=10, route_status="priced"
    )
    plan_decomp = engine._build_score_decomposition(
        planner_pair, route_cost_sats=10, route_status="priced"
    )

    assert plan_decomp["final_score"] > coord_decomp["final_score"]


# -----------------------------------------------------------------------------
# Fleet leases bind every selection path (planner + equalization), not just
# the coordination overlay.
# -----------------------------------------------------------------------------


def test_fleet_lease_suppresses_matching_planner_pair(mock_plugin, mock_database):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.config import Config
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._hive_hints = _OverlayHints(
        leases=[
            {
                "lease_id": "lease-planner",
                "owner_member_id": "02other",
                "route_segments": ["300x1x0>400x1x0"],
            }
        ]
    )
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=2,
            total_remaining_budget_sats=20_000,
        )
    )

    planner_pair = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="03" + "3" * 64,
        dest_peer_id="03" + "4" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=1.0,
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.max_pairs = 5
        planner.plan.return_value = PlanResult(selected=[planner_pair], skipped=[])

        selected = engine.find_candidates()

    assert selected == []
    plan = engine._last_cycle_result.plan
    lease_skips = [s for s in plan.skipped if s.reason == "fleet_lease_held"]
    assert len(lease_skips) == 1
    assert "lease-planner" in (lease_skips[0].detail or "")
    engine.router_v3.price_pair.assert_not_called()


def test_fleet_lease_owned_by_us_does_not_suppress_planner_pair(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.config import Config
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    cfg = Config(dry_run=True, rebalance_router="v3")
    our_id = "03" + "u" * 64
    mock_plugin.rpc.getinfo.return_value = {"id": our_id}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=1,
        route=[{"channel": "300x1x0"}],
        probability_ppm=0,
        error="",
    )
    engine._audit = MagicMock()
    engine._hive_hints = _OverlayHints(
        leases=[
            {
                "lease_id": "lease-ours",
                "owner_member_id": our_id,
                "route_segments": ["300x1x0>400x1x0"],
            }
        ]
    )
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=2,
            total_remaining_budget_sats=20_000,
        )
    )

    planner_pair = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="03" + "3" * 64,
        dest_peer_id="03" + "4" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        dest_out_fee_ppm=1_000,
        score=1.0,
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.max_pairs = 5
        planner.plan.return_value = PlanResult(selected=[planner_pair], skipped=[])

        selected = engine.find_candidates()

    assert len(selected) == 1


# -----------------------------------------------------------------------------
# Drain demand must be net of overlay/equalization claims so Boltz cannot
# double-drain a source the merged selection already drains.
# -----------------------------------------------------------------------------


def test_equalization_claimed_source_removed_from_drain_demand(
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
    drain = engine.get_drain_demand()
    assert drain is not None
    assert all(e.channel_id != "100x1x0" for e in drain.entries)
    assert drain.total_excess_sats == sum(e.excess_sats for e in drain.entries)


def test_overlay_claimed_source_removed_from_drain_demand(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.config import Config
    from modules.rebalance_route_policy import (
        RouteDecision,
        RoutePolicy,
        RoutePriority,
    )
    from modules.rebalance_types_v2 import (
        DrainDemand,
        DrainDemandEntry,
        PairCandidate,
        PlanResult,
    )

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=1,
        route=[{"channel": "100x1x0"}],
        probability_ppm=0,
        error="",
    )
    engine._audit = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=2,
            total_remaining_budget_sats=20_000,
        )
    )

    coordination_pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        dest_out_fee_ppm=1_000,
        score=1.0,
        reason_code="coordinated_rebalance",
        coordination_hint_type="recommendation",
        coordination_hint_id="rec-1",
        route_decision=RouteDecision(
            policy=RoutePolicy.MARKET_ONLY,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
        ),
    )
    drain_demand = DrainDemand(
        entries=[
            DrainDemandEntry(
                channel_id="100x1x0",
                peer_id="03" + "b" * 64,
                excess_sats=250_000,
                drain_score=0.7,
            ),
            DrainDemandEntry(
                channel_id="500x1x0",
                peer_id="03" + "e" * 64,
                excess_sats=100_000,
                drain_score=0.4,
            ),
        ],
        total_excess_sats=350_000,
        over_local_count=2,
        paired_count=0,
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls, patch(
        "modules.rebalance_engine_v2.build_coordination_overlay"
    ) as overlay_builder:
        planner = planner_cls.return_value
        planner.max_pairs = 5
        planner.plan.return_value = PlanResult(
            selected=[], skipped=[], drain_demand=drain_demand
        )
        overlay_builder.return_value = PlanResult(
            selected=[coordination_pair], skipped=[]
        )

        selected = engine.find_candidates()

    assert len(selected) == 1
    drain = engine.get_drain_demand()
    assert drain is not None
    assert [e.channel_id for e in drain.entries] == ["500x1x0"]
    assert drain.total_excess_sats == 100_000


# -----------------------------------------------------------------------------
# Market pricing keeps observed-liquidity failure evidence while still
# excluding hive bias layers.
# -----------------------------------------------------------------------------


def test_market_price_pair_includes_observed_liquidity_without_hive_layers(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    router = MagicMock()
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "1" * 64,
        dest_peer_id="02" + "2" * 64,
        amount_sats=50_000,
        pair_budget_sats=100,
    )

    RebalanceEngine._market_price_pair(
        router, pair, None, market_only_layers=True
    )

    kwargs = router.price_pair.call_args.kwargs
    # No hive bias layers...
    assert kwargs["layer_names_override"] == []
    # ...but observed liquidity stays on: it is OUR failure evidence
    # (avoid segments we just failed on), not a hive preference.
    assert kwargs["include_observed_liquidity"] is True


def test_engine_threads_pair_fee_cap_into_coordination_overlay(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PlanResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.pair_fee_cap_ppm = 1_000
    engine.config.hive_equalization_enabled = False
    engine._audit = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=0,
            total_remaining_budget_sats=0,
        )
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls, patch(
        "modules.rebalance_engine_v2.build_coordination_overlay"
    ) as overlay_builder:
        planner = planner_cls.return_value
        planner.max_pairs = 5
        planner.plan.return_value = PlanResult(selected=[], skipped=[])
        overlay_builder.return_value = PlanResult(selected=[], skipped=[])

        engine.find_candidates()

    assert overlay_builder.call_args.kwargs["pair_fee_cap_ppm"] == 1_000


def test_build_flow_facts_map_computes_realized_utilization_from_db(
    mock_plugin, mock_database
):
    """_build_flow_facts_map is the seam Task 5 adds between the engine's
    per-cycle channel list and compute_channel_flow_facts. It must turn a
    DB-reported flow window into a ChannelFlowFacts with realized
    utilization, keyed by channel_id, without touching the rest of the
    snapshot build."""
    mock_database.get_channel_flow_window.return_value = (600_000, 0, 6)
    engine = _make_engine(mock_plugin, mock_database)

    channels = [{"channel_id": "A", "capacity_sats": 1_000_000}]
    now_ts = int(time.time())

    flow_facts = engine._build_flow_facts_map(channels, engine.config, now_ts)

    assert flow_facts["A"].realized_utilization == pytest.approx(0.6)
    assert flow_facts["A"].utilization_is_realized is True


def test_build_flow_facts_map_returns_empty_when_no_database(mock_plugin):
    """Fail-open: with no database handle at all, the map-build must not
    raise, and build_state_snapshot's neutral defaults take over."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    engine = RebalanceEngine(plugin=mock_plugin, config=cfg, database=None)

    flow_facts = engine._build_flow_facts_map(
        [{"channel_id": "A", "capacity_sats": 1_000_000}],
        cfg,
        int(time.time()),
    )

    assert flow_facts == {}


def test_realized_utilization_used_for_hot_channel(mock_plugin, mock_database):
    """Feature #2: when the destination carries MEASURED (realized)
    utilization, the EV gate's refill term uses that measured value instead
    of the flat EXPECTED_UTILIZATION prior, and the decomposition reports
    the source as 'realized'."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_500
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=5_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
        dest_out_fee_ppm=dest_out_fee_ppm,
        dest_realized_utilization=0.8,
        dest_utilization_is_realized=True,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    expected_refill = amount_sats * dest_out_fee_ppm / 1_000_000.0 * 0.8
    assert decomp["destination_refill_value_sats"] == pytest.approx(expected_refill)
    assert decomp["expected_utilization"] == pytest.approx(0.8)
    assert decomp["utilization_source"] == "realized"


def test_thin_history_uses_prior(mock_plugin, mock_database):
    """Feature #2 fallback: when the destination has NO realized data
    (thin history / no facts), the EV gate's refill term must fall back to
    the flat EXPECTED_UTILIZATION prior (0.5) -- identical to pre-feature
    behavior -- and the decomposition reports the source as 'prior'."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_500
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=5_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
        dest_out_fee_ppm=dest_out_fee_ppm,
        dest_realized_utilization=0.8,
        dest_utilization_is_realized=False,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    expected_refill = amount_sats * dest_out_fee_ppm / 1_000_000.0 * 0.5
    assert decomp["destination_refill_value_sats"] == pytest.approx(expected_refill)
    assert decomp["expected_utilization"] == pytest.approx(0.5)
    assert decomp["utilization_source"] == "prior"


def test_realized_utilization_used_for_source_side(mock_plugin, mock_database):
    """Feature #2, source side: when the SOURCE carries MEASURED (realized)
    utilization, the EV gate's drain/opportunity terms use that measured
    value instead of the flat EXPECTED_UTILIZATION prior, and the
    decomposition reports the source-side provenance as 'realized' via the
    (new) symmetric 'source_utilization_source' key. Mirrors
    test_realized_utilization_used_for_hot_channel but exercises source_u."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)

    amount_sats = 1_000_000
    # Values kept below the default max_fee_ppm=2000 historical-fee cap
    # (_bounded_historical_fee_ppm) so they aren't clipped before use.
    source_out_fee_ppm = 1_200
    source_historical_sourced_fee_ppm = 1_800

    def build_pair(realized_utilization, is_realized):
        return PairCandidate(
            source_channel_id="100x1x0",
            dest_channel_id="200x2x0",
            source_peer_id="03" + "b" * 64,
            dest_peer_id="03" + "c" * 64,
            amount_sats=amount_sats,
            pair_budget_sats=5_000,
            source_capacity_sats=1_000_000,
            dest_capacity_sats=1_000_000,
            score=2.0,
            source_local_ratio=0.85,
            dest_local_ratio=0.10,
            source_out_fee_ppm=source_out_fee_ppm,
            source_historical_sourced_fee_ppm=source_historical_sourced_fee_ppm,
            source_realized_utilization=realized_utilization,
            source_utilization_is_realized=is_realized,
        )

    # --- realized branch: source_u should be 0.8, not the 0.5 prior. ---
    realized_pair = build_pair(0.8, True)
    realized_decomp = engine._build_score_decomposition(
        realized_pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    expected_drain = (
        amount_sats * source_historical_sourced_fee_ppm / 1_000_000.0 * 0.8
    )
    expected_opportunity = (
        amount_sats * source_out_fee_ppm / 1_000_000.0 * 0.8 * 0.5
    )
    assert realized_decomp["source_utilization"] == pytest.approx(0.8)
    assert realized_decomp["source_drain_value_sats"] == pytest.approx(expected_drain)
    assert realized_decomp["source_opportunity_sats"] == pytest.approx(
        expected_opportunity
    )
    assert realized_decomp["source_utilization_source"] == "realized"

    # --- prior branch: source_utilization_is_realized False must fall back
    # to the flat 0.5 EXPECTED_UTILIZATION prior, identical to pre-feature
    # behavior, even though the same 0.8 measured value is present. ---
    prior_pair = build_pair(0.8, False)
    prior_decomp = engine._build_score_decomposition(
        prior_pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    expected_drain_prior = (
        amount_sats * source_historical_sourced_fee_ppm / 1_000_000.0 * 0.5
    )
    expected_opportunity_prior = (
        amount_sats * source_out_fee_ppm / 1_000_000.0 * 0.5 * 0.5
    )
    assert prior_decomp["source_utilization"] == pytest.approx(0.5)
    assert prior_decomp["source_drain_value_sats"] == pytest.approx(
        expected_drain_prior
    )
    assert prior_decomp["source_opportunity_sats"] == pytest.approx(
        expected_opportunity_prior
    )
    assert prior_decomp["source_utilization_source"] == "prior"


def test_activity_penalty_lowers_score_for_helpful_flow(mock_plugin, mock_database):
    """Feature #1: when live forwarding is already moving the pair's
    channels the 'helpful' way (source draining outbound / dest refilling
    inbound), the sats-EV gate applies a soft, capped penalty proportional
    to that helpful flow -- so a pair a router is already fixing scores
    lower than an otherwise-identical pair with no such activity."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_activity_penalty_coeff = 0.5
    engine.config.rebalance_activity_penalty_cap_frac = 0.5

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_000
    source_activity_out_sats = 100_000

    def build_pair(source_activity_out_sats):
        return PairCandidate(
            source_channel_id="100x1x0",
            dest_channel_id="200x2x0",
            source_peer_id="03" + "b" * 64,
            dest_peer_id="03" + "c" * 64,
            amount_sats=amount_sats,
            pair_budget_sats=5_000,
            source_capacity_sats=1_000_000,
            dest_capacity_sats=1_000_000,
            score=2.0,
            source_local_ratio=0.85,
            dest_local_ratio=0.10,
            dest_out_fee_ppm=dest_out_fee_ppm,
            source_activity_out_sats=source_activity_out_sats,
        )

    base_pair = build_pair(0)
    activity_pair = build_pair(source_activity_out_sats)

    base_decomp = engine._build_score_decomposition(
        base_pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )
    activity_decomp = engine._build_score_decomposition(
        activity_pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    dest_value_fee_ppm = base_decomp["inputs"]["dest_value_fee_ppm"]
    expected_future_value_sats = base_decomp["expected_future_value_sats"]
    raw_penalty = 0.5 * source_activity_out_sats * dest_value_fee_ppm / 1_000_000.0
    cap = 0.5 * max(0.0, expected_future_value_sats)
    expected_penalty = round(min(raw_penalty, cap), 6)

    assert base_decomp["activity_penalty_sats"] == 0.0
    assert activity_decomp["activity_penalty_sats"] == pytest.approx(expected_penalty)
    assert expected_penalty > 0
    assert activity_decomp["final_score_sats"] < base_decomp["final_score_sats"]
    assert activity_decomp["final_score_sats"] == pytest.approx(
        base_decomp["final_score_sats"] - expected_penalty
    )


def test_activity_penalty_is_capped(mock_plugin, mock_database):
    """Feature #1 cap: a pair with a huge helpful live-flow figure must
    have its penalty capped at cap_frac * expected_future_value_sats, so a
    strongly-EV pair still beats do-nothing instead of being zeroed out."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_activity_penalty_coeff = 0.5
    engine.config.rebalance_activity_penalty_cap_frac = 0.5

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_000

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=5_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
        dest_out_fee_ppm=dest_out_fee_ppm,
        source_activity_out_sats=10**12,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    expected_future_value_sats = decomp["expected_future_value_sats"]
    expected_cap = round(0.5 * max(0.0, expected_future_value_sats), 6)

    assert decomp["activity_penalty_sats"] == pytest.approx(expected_cap)
    assert decomp["final_score_sats"] >= 0
    assert decomp["beats_do_nothing"] is True


def test_no_activity_no_penalty(mock_plugin, mock_database):
    """KEY INVARIANT: a pair with no activity facts (activity sats = 0, the
    default) must get a zero activity penalty and a final_score_sats
    IDENTICAL to the pre-feature value -- existing engine tests with no
    activity must stay green."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_activity_penalty_coeff = 0.5
    engine.config.rebalance_activity_penalty_cap_frac = 0.5

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_000

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=5_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
        dest_out_fee_ppm=dest_out_fee_ppm,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    p_success = decomp["p_success"]
    expected_future_value_sats = decomp["expected_future_value_sats"]
    expected_fee_sats = decomp["expected_fee_sats"]
    source_opportunity_sats = decomp["source_opportunity_sats"]
    failure_penalty_sats = decomp["failure_penalty_sats"]
    pre_feature_final_score_sats = round(
        p_success * expected_future_value_sats
        - expected_fee_sats
        - source_opportunity_sats
        - failure_penalty_sats,
        6,
    )

    assert decomp["activity_penalty_sats"] == 0.0
    assert decomp["final_score_sats"] == pytest.approx(pre_feature_final_score_sats)


def test_activity_penalty_includes_dest_side_contribution(mock_plugin, mock_database):
    """Task 8 coverage gap: helpful_flow_sats = source_activity_out_sats +
    dest_activity_in_sats. Existing tests only ever set
    source_activity_out_sats, so a bug that dropped the dest_activity_in_sats
    addend from the sum would still pass them. Set only
    dest_activity_in_sats > 0 (source_activity_out_sats == 0, the default)
    and prove it alone produces a nonzero, correctly-computed penalty."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_activity_penalty_coeff = 0.5
    engine.config.rebalance_activity_penalty_cap_frac = 0.5

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_000
    dest_activity_in_sats = 100_000

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=5_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
        dest_out_fee_ppm=dest_out_fee_ppm,
        dest_activity_in_sats=dest_activity_in_sats,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    dest_value_fee_ppm = decomp["inputs"]["dest_value_fee_ppm"]
    expected_future_value_sats = decomp["expected_future_value_sats"]
    raw_penalty = 0.5 * dest_activity_in_sats * dest_value_fee_ppm / 1_000_000.0
    cap = 0.5 * max(0.0, expected_future_value_sats)
    expected_penalty = round(min(raw_penalty, cap), 6)

    assert expected_penalty > 0
    assert decomp["activity_penalty_sats"] == pytest.approx(expected_penalty)


def test_all_three_features_compose_on_one_pair(mock_plugin, mock_database):
    """Composition guard (final review): the branch ships live default-ON
    and every existing test above exercises realized utilization (#2),
    the activity penalty (#1), or size-tiered bands (#3) in ISOLATION.
    Nothing proves the three coexist correctly on a single channel/pair --
    e.g. that #2 feeding a bigger expected_future_value_sats doesn't
    silently change how #1's cap or #3's sizing behave, or that the terms
    don't double-count.

    This pair models a single big channel pair where BOTH sides carry
    realized (measured) utilization AND live "helpful" forwarding
    activity, with a nonzero dest fee so the value term is real. (The
    size-tiered per-channel band, #3, lives on ChannelState and is
    threaded through the planner -- not a PairCandidate field consumed by
    _build_score_decomposition -- so its coexistence with #2/#1 on one
    big channel is proven at the planner-threading layer by the
    companion test
    test_all_three_features_thread_onto_one_big_channel_pair in
    tests/test_rebalance_planner_v2.py, which reuses this same big-channel
    wide-band scenario.)
    """
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_activity_penalty_coeff = 0.5
    engine.config.rebalance_activity_penalty_cap_frac = 0.5

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_000
    # Kept below the default max_fee_ppm=2000 historical-fee cap
    # (_bounded_historical_fee_ppm) so they aren't clipped before use.
    source_out_fee_ppm = 1_200
    source_historical_sourced_fee_ppm = 1_800
    # Moderate helpful-flow figures: nonzero on BOTH legs, but small enough
    # relative to expected_future_value_sats that the penalty is real
    # without swallowing the pair's EV (still beats do-nothing).
    source_activity_out_sats = 50_000
    dest_activity_in_sats = 50_000

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=5_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
        source_out_fee_ppm=source_out_fee_ppm,
        dest_out_fee_ppm=dest_out_fee_ppm,
        source_historical_sourced_fee_ppm=source_historical_sourced_fee_ppm,
        # Feature #2 on BOTH legs, so both sides' value/opportunity terms
        # use the measured 0.8 instead of the flat 0.5 prior.
        source_realized_utilization=0.8,
        source_utilization_is_realized=True,
        dest_realized_utilization=0.8,
        dest_utilization_is_realized=True,
        # Feature #1 on BOTH legs.
        source_activity_out_sats=source_activity_out_sats,
        dest_activity_in_sats=dest_activity_in_sats,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    # --- #2 is active: realized utilization feeds expected_utilization/
    # utilization_source AND the destination refill value term. ---
    assert decomp["expected_utilization"] == pytest.approx(0.8)
    assert decomp["utilization_source"] == "realized"
    expected_refill = amount_sats * dest_out_fee_ppm / 1_000_000.0 * 0.8
    assert decomp["destination_refill_value_sats"] == pytest.approx(expected_refill)

    # --- #1 is active: a nonzero, capped activity penalty. ---
    dest_value_fee_ppm = decomp["inputs"]["dest_value_fee_ppm"]
    expected_future_value_sats = decomp["expected_future_value_sats"]
    helpful_flow_sats = source_activity_out_sats + dest_activity_in_sats
    raw_penalty = 0.5 * helpful_flow_sats * dest_value_fee_ppm / 1_000_000.0
    activity_cap = 0.5 * max(0.0, expected_future_value_sats)
    expected_penalty = round(min(raw_penalty, activity_cap), 6)
    assert expected_penalty > 0
    assert expected_penalty <= activity_cap + 1e-9
    assert decomp["activity_penalty_sats"] == pytest.approx(expected_penalty)

    # --- No double-count / no missing term: final_score_sats must equal
    # the full composed formula, RECOMPUTED from the decision's own
    # exposed terms (not re-derived independently), proving #2's bigger
    # value term and #1's penalty (proportional to that same value term)
    # combine additively with nothing dropped or counted twice. ---
    recomposed_final_score_sats = round(
        decomp["p_success"] * decomp["expected_future_value_sats"]
        - decomp["expected_fee_sats"]
        - decomp["source_opportunity_sats"]
        - decomp["failure_penalty_sats"]
        - decomp["activity_penalty_sats"],
        6,
    )
    assert decomp["final_score_sats"] == pytest.approx(recomposed_final_score_sats)

    # --- Sanity: composing all three features still yields a coherent,
    # positive-EV decision -- not nonsensically rejected or accepted. ---
    assert decomp["final_score_sats"] > 0
    assert decomp["beats_do_nothing"] is True


def test_activity_penalty_sums_source_and_dest_activity(mock_plugin, mock_database):
    """Task 8 coverage gap: when BOTH source_activity_out_sats and
    dest_activity_in_sats are set, the penalty must be based on their SUM
    (helpful_flow_sats = source_activity_out_sats + dest_activity_in_sats),
    not either term alone."""
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.rebalance_activity_penalty_coeff = 0.5
    engine.config.rebalance_activity_penalty_cap_frac = 0.5

    amount_sats = 1_000_000
    dest_out_fee_ppm = 2_000
    source_activity_out_sats = 60_000
    dest_activity_in_sats = 40_000

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=amount_sats,
        pair_budget_sats=5_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
        dest_out_fee_ppm=dest_out_fee_ppm,
        source_activity_out_sats=source_activity_out_sats,
        dest_activity_in_sats=dest_activity_in_sats,
    )

    decomp = engine._build_score_decomposition(
        pair,
        probability_ppm=900_000,
        route_cost_sats=10,
        effective_budget_sats=5_000,
        route_status="priced",
    )

    dest_value_fee_ppm = decomp["inputs"]["dest_value_fee_ppm"]
    expected_future_value_sats = decomp["expected_future_value_sats"]
    helpful_flow_sats = source_activity_out_sats + dest_activity_in_sats
    raw_penalty = 0.5 * helpful_flow_sats * dest_value_fee_ppm / 1_000_000.0
    cap = 0.5 * max(0.0, expected_future_value_sats)
    expected_penalty = round(min(raw_penalty, cap), 6)

    assert expected_penalty > 0
    assert decomp["activity_penalty_sats"] == pytest.approx(expected_penalty)
