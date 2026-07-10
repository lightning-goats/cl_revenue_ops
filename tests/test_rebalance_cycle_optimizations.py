"""Tests for the rebalance-cycle RPC/DB reduction optimizations.

Covers:
1. compute_allocations runs exactly once per cycle (engine snapshot only;
   the duplicate rebalancer-level call was removed).
2. Pair futility is filtered BEFORE route pricing in find_candidates.
3. Router v3 fixed RPC overhead cuts:
   (a) dest cltv reused from the final-hop policy lookup,
   (b) cycle-scoped askrene-listlayers cache with unknown-layer invalidation,
   (c) executor accepts an injected node id and skips getinfo.
4. HYBRID short-circuit: a free hive route skips the market quote.
5. _build_snapshot uses the batched last-rebalance-time query when available.
6. Reconcile sweep builds the SCID->peer map once, via the data service.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules.rebalance_router_v2 import RouteResult
from modules.rebalance_route_policy import (
    RouteDecision,
    RoutePolicy,
    RoutePriority,
)

OUR_ID = "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3"
SRC_PEER = "03" + "a" * 64
DST_PEER = "03" + "b" * 64


def _make_engine(plugin, database, **kwargs):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": OUR_ID}
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
    return RebalanceEngine(plugin=plugin, config=cfg, database=database, **kwargs)


# ---------------------------------------------------------------------------
# 1. compute_allocations exactly once per cycle
# ---------------------------------------------------------------------------


def test_live_cycle_computes_capex_allocations_exactly_once(
    mock_plugin, mock_database
):
    """One rebalance cycle must trigger exactly one full capex/flow pass.

    The duplicate compute_allocations call in
    EVRebalancer.find_rebalance_candidates (result discarded) was removed;
    the engine's _build_snapshot owns the single per-cycle computation.
    """
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=False)
    mock_database.cleanup_stale_reservations.return_value = 0
    mock_database.get_pending_settlement_rebalances.return_value = []
    mock_database.get_last_rebalance_times.return_value = {}

    capex = MagicMock()
    capex.compute_allocations.return_value = None

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.set_capex_engine(capex)
    r.rebalance_engine_v2 = _make_engine(
        mock_plugin, mock_database, capex_engine=capex
    )
    r.rebalance_engine_v2._audit = MagicMock()

    r.find_rebalance_candidates()

    assert capex.compute_allocations.call_count == 1


def test_dry_run_cycle_computes_capex_allocations_exactly_once(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0
    mock_database.get_last_rebalance_times.return_value = {}

    capex = MagicMock()
    capex.compute_allocations.return_value = None

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.set_capex_engine(capex)
    r.rebalance_engine_v2 = _make_engine(
        mock_plugin, mock_database, capex_engine=capex
    )
    r.rebalance_engine_v2._audit = MagicMock()

    r.find_rebalance_candidates()

    assert capex.compute_allocations.call_count == 1


# ---------------------------------------------------------------------------
# 2. Futility filter runs before pricing
# ---------------------------------------------------------------------------


def test_pair_in_futility_never_reaches_router(mock_plugin, mock_database):
    """A pair at the futility threshold is skipped pre-pricing: the router
    must never be asked to price it."""
    mock_database.get_pair_rebalance_cooldown.return_value = None

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

    now = time.time()
    engine._pair_failures[("100x1x0", "200x1x0")] = [now - 30, now - 20, now - 10]

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
    audited_reasons = [
        call.kwargs.get("reason")
        for call in engine._audit.log_skip.call_args_list
    ] + [
        getattr(skip, "reason", None)
        for call in engine._audit.log_skips.call_args_list
        for skip in (call.args[0] if call.args else call.kwargs.get("skips", []))
    ]
    assert "pair_futility" in audited_reasons


def test_pair_below_futility_threshold_is_still_priced(
    mock_plugin, mock_database
):
    mock_database.get_pair_rebalance_cooldown.return_value = None

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
    engine.router_v3.price_pair.return_value = RouteResult(
        success=False, error="no_route"
    )

    now = time.time()
    engine._pair_failures[("100x1x0", "200x1x0")] = [now - 30, now - 20]

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

        engine.find_candidates()

    engine.router_v3.price_pair.assert_called_once()


# ---------------------------------------------------------------------------
# 3a. v3 dest cltv reused from final-hop policy
# ---------------------------------------------------------------------------


def _make_v3_plugin_with_peer_policy(cltv_delta=40):
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    plugin.rpc.listpeerchannels.return_value = {
        "channels": [
            {
                "peer_id": DST_PEER,
                "short_channel_id": "200x2x0",
                "updates": {
                    "remote": {
                        "fee_proportional_millionths": 0,
                        "fee_base_msat": 0,
                        "cltv_expiry_delta": cltv_delta,
                    }
                },
            }
        ]
    }
    plugin.rpc.listchannels.return_value = {
        "channels": [
            {
                "short_channel_id": "111x1x1",
                "source": SRC_PEER,
                "destination": DST_PEER,
                "fee_per_millionth": 0,
                "base_fee_millisatoshi": 0,
                "delay": 0,
            }
        ]
    }
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            {
                "probability_ppm": 990000,
                "amount_msat": 100000,
                "final_cltv": 40,
                "path": [
                    {
                        "short_channel_id_dir": "111x1x1/0",
                        "next_node_id": DST_PEER,
                        "amount_msat": 100000,
                        "delay": 40,
                    },
                ],
            }
        ],
    }
    return plugin


def _make_v3_router(plugin):
    from modules.rebalance_router_v3 import RebalanceRouterV3

    return RebalanceRouterV3(
        plugin=plugin,
        our_node_id=OUR_ID,
        layer_names=["hive-fleet"],
        log=lambda m, l: None,
    )


def test_v3_price_pair_reuses_final_hop_policy_cltv(monkeypatch):
    """When the final-hop policy carries cltv_delta, the second identical
    listpeerchannels lookup (_get_dest_channel_cltv) must be skipped."""
    plugin = _make_v3_plugin_with_peer_policy(cltv_delta=40)
    router = _make_v3_router(plugin)
    cltv_spy = MagicMock(return_value=40)
    monkeypatch.setattr(router._v2_helpers, "_get_dest_channel_cltv", cltv_spy)

    result = router.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )

    assert result.success is True, result.error
    cltv_spy.assert_not_called()
    # final_cltv passed to getroutes = dest cltv (40) + invoice cltv (18)
    assert plugin.rpc.getroutes.call_args.kwargs["final_cltv"] == 58


def test_v3_price_pair_falls_back_when_policy_cltv_missing(monkeypatch):
    """A zero/absent cltv_delta in the policy still triggers the dedicated
    cltv lookup, preserving the safe default path."""
    plugin = _make_v3_plugin_with_peer_policy(cltv_delta=0)
    router = _make_v3_router(plugin)
    cltv_spy = MagicMock(return_value=40)
    monkeypatch.setattr(router._v2_helpers, "_get_dest_channel_cltv", cltv_spy)

    result = router.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )

    assert result.success is True, result.error
    cltv_spy.assert_called_once()
    assert plugin.rpc.getroutes.call_args.kwargs["final_cltv"] == 58


# ---------------------------------------------------------------------------
# 3b. Cycle-scoped askrene-listlayers cache
# ---------------------------------------------------------------------------


def _count_listlayers_calls(plugin) -> int:
    return sum(
        1
        for call in plugin.rpc.call.call_args_list
        if call.args and call.args[0] == "askrene-listlayers"
    )


def test_v3_router_caches_listlayers_within_cycle():
    plugin = _make_v3_plugin_with_peer_policy()
    router = _make_v3_router(plugin)
    baseline = _count_listlayers_calls(plugin)

    router.begin_cycle()
    try:
        for _ in range(3):
            result = router.price_pair(
                source_channel_id="100x1x0",
                dest_channel_id="200x2x0",
                source_peer_id=SRC_PEER,
                dest_peer_id=DST_PEER,
                amount_sats=100,
            )
            assert result.success is True, result.error
    finally:
        router.end_cycle()

    assert _count_listlayers_calls(plugin) == baseline + 1


def test_v3_router_reprobes_listlayers_outside_cycle():
    """Without an active cycle, every price_pair re-probes (legacy behavior)."""
    plugin = _make_v3_plugin_with_peer_policy()
    router = _make_v3_router(plugin)
    baseline = _count_listlayers_calls(plugin)

    for _ in range(2):
        result = router.price_pair(
            source_channel_id="100x1x0",
            dest_channel_id="200x2x0",
            source_peer_id=SRC_PEER,
            dest_peer_id=DST_PEER,
            amount_sats=100,
        )
        assert result.success is True, result.error

    assert _count_listlayers_calls(plugin) == baseline + 2


def test_v3_router_invalidates_layer_cache_on_unknown_layer_error():
    plugin = _make_v3_plugin_with_peer_policy()
    router = _make_v3_router(plugin)
    good_routes = plugin.rpc.getroutes.return_value
    plugin.rpc.getroutes.side_effect = [
        Exception("Unknown layer hive-fleet"),
        good_routes,
    ]
    baseline = _count_listlayers_calls(plugin)

    router.begin_cycle()
    try:
        failed = router.price_pair(
            source_channel_id="100x1x0",
            dest_channel_id="200x2x0",
            source_peer_id=SRC_PEER,
            dest_peer_id=DST_PEER,
            amount_sats=100,
        )
        assert failed.success is False
        assert "unknown_layer" in failed.error

        # Cache was invalidated: the next call must re-probe and succeed.
        retried = router.price_pair(
            source_channel_id="100x1x0",
            dest_channel_id="200x2x0",
            source_peer_id=SRC_PEER,
            dest_peer_id=DST_PEER,
            amount_sats=100,
        )
        assert retried.success is True, retried.error
    finally:
        router.end_cycle()

    assert _count_listlayers_calls(plugin) == baseline + 2


def test_engine_brackets_v3_router_cycle_around_pricing(
    mock_plugin, mock_database
):
    mock_database.get_pair_rebalance_cooldown.return_value = None

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
    engine.router_v3.price_pair.return_value = RouteResult(
        success=False, error="no_route"
    )
    engine._cycle_router = None  # find_candidates re-captures the router

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

        engine.find_candidates()

    engine.router_v3.begin_cycle.assert_called_once()
    engine.router_v3.end_cycle.assert_called_once()


# ---------------------------------------------------------------------------
# 3c. Executor node id injection
# ---------------------------------------------------------------------------


def test_native_executor_injected_our_id_skips_getinfo():
    from modules.rebalance_native_executor_v2 import NativeRouteExecutor

    plugin = MagicMock()
    executor = NativeRouteExecutor(plugin, our_id=OUR_ID)

    assert executor.is_available() is True
    assert executor._get_our_id() == OUR_ID
    plugin.rpc.call.assert_not_called()


def test_native_executor_without_our_id_keeps_getinfo_path():
    from modules.rebalance_native_executor_v2 import NativeRouteExecutor

    plugin = MagicMock()
    plugin.rpc.call.return_value = {"id": OUR_ID}
    executor = NativeRouteExecutor(plugin)

    assert executor.is_available() is True
    plugin.rpc.call.assert_called_once_with("getinfo", {})


def test_engine_make_executor_injects_cached_node_id(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine._our_id = OUR_ID

    executor = engine._make_executor()

    assert executor._our_id == OUR_ID


# ---------------------------------------------------------------------------
# 5. Batched last-rebalance-time lookup in _build_snapshot
# ---------------------------------------------------------------------------


class _SnapshotDb:
    """Plain object (not MagicMock) so attribute presence is explicit."""

    def __init__(self, with_batch=True, last_ts=None):
        self.batch_calls = 0
        self.point_calls = 0
        self._last_ts = last_ts
        if not with_batch:
            # Hide the batch method entirely.
            self.get_last_rebalance_times = None

    def get_last_rebalance_times(self):
        self.batch_calls += 1
        return {"100x1x0": self._last_ts} if self._last_ts else {}

    def get_last_rebalance_time(self, channel_id, reason_code=None):
        self.point_calls += 1
        return self._last_ts

    def get_last_post_rebalance_state(self, channel_id):
        return None


def _snapshot_channels():
    return {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": SRC_PEER,
                "short_channel_id": "100x1x0",
                "total_msat": 1_000_000_000,
                "our_amount_msat": 900_000_000,
                "updates": {"remote": {"fee_proportional_millionths": 10}},
            }
        ]
    }


def test_build_snapshot_uses_batch_rebalance_times_when_available(
    mock_plugin,
):
    db = _SnapshotDb(with_batch=True, last_ts=int(time.time()) - 60)
    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": []}
    mock_plugin.rpc.listpeerchannels.return_value = _snapshot_channels()

    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    engine = RebalanceEngine(
        plugin=mock_plugin, config=Config(dry_run=True), database=db
    )
    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert db.batch_calls == 1
    assert db.point_calls == 0


def test_build_snapshot_falls_back_to_point_queries_without_batch_method(
    mock_plugin,
):
    db = _SnapshotDb(with_batch=False, last_ts=None)
    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": []}
    mock_plugin.rpc.listpeerchannels.return_value = _snapshot_channels()

    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    engine = RebalanceEngine(
        plugin=mock_plugin, config=Config(dry_run=True), database=db
    )
    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert db.point_calls == 1


def test_build_snapshot_batch_cooldown_matches_point_query_behavior(
    mock_plugin,
):
    """A recent batch timestamp must flag the channel as in cooldown."""
    recent = int(time.time()) - 60
    db = _SnapshotDb(with_batch=True, last_ts=recent)
    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": []}
    mock_plugin.rpc.listpeerchannels.return_value = _snapshot_channels()

    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    engine = RebalanceEngine(
        plugin=mock_plugin, config=Config(dry_run=True), database=db
    )

    captured = {}
    import modules.rebalance_engine_v2 as engine_mod

    real_builder = engine_mod.build_state_snapshot

    def capture(normalized, *args, **kwargs):
        captured["normalized"] = normalized
        return real_builder(normalized, *args, **kwargs)

    with patch.object(engine_mod, "build_state_snapshot", side_effect=capture):
        engine._build_snapshot()

    assert captured["normalized"][0]["cooldown_active"] is True


# ---------------------------------------------------------------------------
# 6. Reconcile sweep peer map built once
# ---------------------------------------------------------------------------


def _settled_row(rebalance_id, payment_hash):
    return {
        "id": rebalance_id,
        "payment_hash": payment_hash,
        "from_channel": "100x1x0",
        "to_channel": "200x1x0",
        "amount_sats": 1_000,
    }


def test_reconcile_sweep_builds_peer_map_once_for_all_rows(
    mock_plugin, mock_database
):
    data_service = MagicMock()
    data_service.get_node_id.return_value = OUR_ID
    data_service.get_askrene_layers.return_value = {"layers": []}
    data_service.get_peer_channels.return_value = {
        "channels": [
            {"short_channel_id": "200x1x0", "peer_id": DST_PEER},
        ]
    }

    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    engine = RebalanceEngine(
        plugin=mock_plugin,
        config=Config(dry_run=True),
        database=mock_database,
        data_service=data_service,
    )
    data_service.get_peer_channels.reset_mock()
    mock_plugin.rpc.listpeerchannels.reset_mock()

    mock_database.get_pending_settlement_rebalances.return_value = [
        _settled_row(1, "aa" * 32),
        _settled_row(2, "bb" * 32),
        _settled_row(3, "cc" * 32),
    ]
    mock_plugin.rpc.call.return_value = {
        "payments": [
            {
                "status": "complete",
                "amount_msat": 1_000_000,
                "amount_sent_msat": 1_002_000,
            }
        ]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 3
    # One listpeerchannels-equivalent fetch for the whole sweep, via the
    # data service — not one per settled row, not via raw plugin RPC.
    assert data_service.get_peer_channels.call_count == 1
    mock_plugin.rpc.listpeerchannels.assert_not_called()
    # Cost attribution still resolved the peer id from the shared map.
    for call in mock_database.record_rebalance_cost.call_args_list:
        assert call.kwargs["peer_id"] == DST_PEER
