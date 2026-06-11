"""Regression tests for the per-cycle RPC/DB scaler fixes.

Covers:
1. The post-rebalance anchor row is fetched once per cooled channel in
   _build_snapshot and shared by the fill-fraction cooldown helper and the
   drift override (was 2 point queries per cooled channel).
2. v3 router reuses a cycle-scoped exclude layer per unique exclude set
   (was create + N updates + remove per price_pair).
3. Final-hop policy lookups are served from the broadcast listpeerchannels
   cache instead of an uncached per-peer RPC.
4. Hive router caches askrene-listlayers for the cycle.
5. The empirical dest success-rate DB aggregate is memoized per cycle.
6. A pending-settlement result without a payment_hash is recorded as a
   terminal failure (never an unsweepable parked row).
7. Skip-log emission is aggregated for non-actionable reasons.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules.rebalance_router_v2 import RouteResult

OUR_ID = "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3"
SRC_PEER = "03" + "a" * 64
DST_PEER = "03" + "b" * 64


# ---------------------------------------------------------------------------
# 1. Anchor row fetched once per cooled channel
# ---------------------------------------------------------------------------


class _AnchorCountingDb:
    """Counts get_last_post_rebalance_state point queries."""

    def __init__(self, last_ts, anchor):
        self._last_ts = last_ts
        self._anchor = anchor
        self.anchor_calls = 0

    def get_last_rebalance_times(self):
        return {"100x1x0": self._last_ts} if self._last_ts else {}

    def get_last_rebalance_time(self, channel_id, reason_code=None):
        return self._last_ts

    def get_last_post_rebalance_state(self, channel_id):
        self.anchor_calls += 1
        return self._anchor


def _cooled_channels():
    return {
        "channels": [
            {
                "state": "CHANNELD_NORMAL",
                "peer_id": SRC_PEER,
                "short_channel_id": "100x1x0",
                "total_msat": 1_000_000_000,
                "our_amount_msat": 200_000_000,
                "updates": {"remote": {"fee_proportional_millionths": 10}},
            }
        ]
    }


def _snapshot_engine(mock_plugin, db):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": []}
    mock_plugin.rpc.listpeerchannels.return_value = _cooled_channels()
    return RebalanceEngine(
        plugin=mock_plugin, config=Config(dry_run=True), database=db
    )


def test_build_snapshot_fetches_anchor_once_per_cooled_channel(mock_plugin):
    """The cooldown fill-fraction helper AND the drift override must share
    one anchor fetch — not issue one point query each."""
    anchor = {"post_local_ratio": 0.6, "amount_sats": 50_000}
    db = _AnchorCountingDb(last_ts=int(time.time()) - 60, anchor=anchor)
    engine = _snapshot_engine(mock_plugin, db)

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert db.anchor_calls == 1


def test_build_snapshot_anchor_drives_both_cooldown_and_drift(mock_plugin):
    """The shared anchor still feeds both consumers: a large drift below the
    anchored post-rebalance ratio sets the override flag."""
    # Channel local ratio is 0.2; anchor says it was 0.6 after the last
    # rebalance -> drift of 0.4 >= default 0.30 threshold.
    anchor = {"post_local_ratio": 0.6, "amount_sats": 50_000}
    db = _AnchorCountingDb(last_ts=int(time.time()) - 60, anchor=anchor)
    engine = _snapshot_engine(mock_plugin, db)

    captured = {}
    import modules.rebalance_engine_v2 as engine_mod

    real_builder = engine_mod.build_state_snapshot

    def capture(normalized, *args, **kwargs):
        captured["normalized"] = normalized
        return real_builder(normalized, *args, **kwargs)

    with patch.object(engine_mod, "build_state_snapshot", side_effect=capture):
        engine._build_snapshot()

    entry = captured["normalized"][0]
    assert entry["cooldown_active"] is True
    assert entry["cooldown_override"] is True
    assert db.anchor_calls == 1


def test_build_snapshot_no_anchor_query_when_not_cooled(mock_plugin):
    """Channels outside the base cooldown window never pay the point query."""
    db = _AnchorCountingDb(last_ts=None, anchor=None)
    engine = _snapshot_engine(mock_plugin, db)

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert db.anchor_calls == 0


# ---------------------------------------------------------------------------
# 2. v3 cycle-scoped exclude-layer cache
# ---------------------------------------------------------------------------


def _make_v3_plugin():
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
                        "cltv_expiry_delta": 40,
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


def _count_rpc(plugin, method) -> int:
    return sum(
        1
        for call in plugin.rpc.call.call_args_list
        if call.args and call.args[0] == method
    )


def _v3_price(router):
    return router.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )


def test_v3_router_reuses_exclude_layer_within_cycle():
    """Two pricings with identical exclude sets in one cycle build the
    throwaway layer once (create + updates), not once per call."""
    plugin = _make_v3_plugin()
    router = _make_v3_router(plugin)

    router.begin_cycle()
    try:
        for _ in range(2):
            result = _v3_price(router)
            assert result.success is True, result.error
        # Layer alive for the whole cycle: no remove yet.
        assert _count_rpc(plugin, "askrene-create-layer") == 1
        assert _count_rpc(plugin, "askrene-remove-layer") == 0
    finally:
        router.end_cycle()

    assert _count_rpc(plugin, "askrene-remove-layer") == 1


def test_v3_router_distinct_exclude_sets_get_distinct_layers():
    plugin = _make_v3_plugin()
    router = _make_v3_router(plugin)

    router.begin_cycle()
    try:
        assert _v3_price(router).success is True
        result = router.price_pair(
            source_channel_id="100x1x0",
            dest_channel_id="200x2x0",
            source_peer_id=SRC_PEER,
            dest_peer_id=DST_PEER,
            amount_sats=100,
            exclude=["333x3x0/1"],
        )
        assert result.success is True, result.error
        assert _count_rpc(plugin, "askrene-create-layer") == 2
    finally:
        router.end_cycle()

    assert _count_rpc(plugin, "askrene-remove-layer") == 2


def test_v3_router_exclude_layer_per_call_outside_cycle():
    """No active cycle (e.g. worker-thread retries): create/remove per call."""
    plugin = _make_v3_plugin()
    router = _make_v3_router(plugin)

    for _ in range(2):
        result = _v3_price(router)
        assert result.success is True, result.error

    assert _count_rpc(plugin, "askrene-create-layer") == 2
    assert _count_rpc(plugin, "askrene-remove-layer") == 2


# ---------------------------------------------------------------------------
# 5. Empirical dest success rate memoized per cycle
# ---------------------------------------------------------------------------


def _success_rate_engine(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": []}
    return RebalanceEngine(
        plugin=mock_plugin, config=Config(dry_run=True), database=mock_database
    )


def test_dest_success_rate_queried_once_per_channel_per_cycle(
    mock_plugin, mock_database
):
    mock_database.get_channel_rebalance_success_rate.return_value = {
        "total": 5,
        "success_rate": 0.8,
    }
    engine = _success_rate_engine(mock_plugin, mock_database)

    for _ in range(5):
        assert engine._empirical_dest_success_rate("200x1x0") == 0.8

    assert mock_database.get_channel_rebalance_success_rate.call_count == 1


def test_dest_success_rate_memoizes_none_results(mock_plugin, mock_database):
    mock_database.get_channel_rebalance_success_rate.return_value = {
        "total": 1,
        "success_rate": 1.0,
    }
    engine = _success_rate_engine(mock_plugin, mock_database)

    assert engine._empirical_dest_success_rate("200x1x0") is None
    assert engine._empirical_dest_success_rate("200x1x0") is None

    assert mock_database.get_channel_rebalance_success_rate.call_count == 1


def test_dest_success_rate_memo_cleared_at_cycle_start(
    mock_plugin, mock_database
):
    mock_database.get_channel_rebalance_success_rate.return_value = {
        "total": 5,
        "success_rate": 0.8,
    }
    engine = _success_rate_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(return_value=None)  # empty cycle

    engine._empirical_dest_success_rate("200x1x0")
    engine.find_candidates()  # new cycle clears the memo
    engine._empirical_dest_success_rate("200x1x0")

    assert mock_database.get_channel_rebalance_success_rate.call_count == 2


# ---------------------------------------------------------------------------
# 4. Hive router cycle-scoped askrene-listlayers cache
# ---------------------------------------------------------------------------


MID_PEER = "02" + "d" * 64


def _make_hive_data_service():
    ds = MagicMock()
    ds.get_askrene_layers.return_value = {
        "layers": [{"layer": "hive-fleet"}, {"layer": "revenue-local"}]
    }
    ds.get_peer_channels.side_effect = lambda peer_id=None: (
        {
            "channels": [
                {
                    "short_channel_id": "100x1x0",
                    "peer_id": SRC_PEER,
                    "state": "CHANNELD_NORMAL",
                },
                {
                    "short_channel_id": "200x1x0",
                    "peer_id": DST_PEER,
                    "state": "CHANNELD_NORMAL",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        }
                    },
                },
            ]
        }
        if peer_id is None
        else {"channels": []}
    )
    ds.get_configs.return_value = {"configs": {"cltv-final": {"value_int": 18}}}
    ds.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            {
                "probability_ppm": 990000,
                "amount_msat": 100000000,
                "path": [
                    {
                        "short_channel_id_dir": "100x1x0/0",
                        "next_node_id": MID_PEER,
                        "amount_msat": 100000000,
                        "delay": 24,
                    },
                    {
                        "short_channel_id_dir": "300x1x0/0",
                        "next_node_id": DST_PEER,
                        "amount_msat": 100000000,
                        "delay": 18,
                    },
                ],
            }
        ],
    }
    return ds


def _make_hive_router(ds):
    from modules.rebalance_hive_router import RebalanceHiveRouter

    class _Hints:
        def is_hive_member(self, peer_id):
            return True

    return RebalanceHiveRouter(
        plugin=MagicMock(),
        our_node_id=OUR_ID,
        hive_hints=_Hints(),
        data_service=ds,
        log=lambda m, l: None,
    )


def _hive_pair():
    from modules.rebalance_types_v2 import PairCandidate

    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100_000,
        pair_budget_sats=100,
    )


def _hive_decision():
    from modules.rebalance_route_policy import (
        RouteDecision,
        RoutePolicy,
        RoutePriority,
    )

    return RouteDecision(
        policy=RoutePolicy.HIVE_ONLY,
        priority=RoutePriority.COORDINATED,
        reason="coordinated_rebalance",
        allow_market_fallback=False,
    )


def test_hive_router_caches_listlayers_within_cycle():
    """5 pricings in one cycle probe askrene-listlayers exactly once."""
    ds = _make_hive_data_service()
    router = _make_hive_router(ds)

    router.begin_cycle()
    try:
        for _ in range(5):
            result = router.price_pair(_hive_pair(), _hive_decision())
            assert result.success is True, result.error
    finally:
        router.end_cycle()

    assert ds.get_askrene_layers.call_count == 1


def test_hive_router_reprobes_listlayers_outside_cycle():
    ds = _make_hive_data_service()
    router = _make_hive_router(ds)

    for _ in range(2):
        result = router.price_pair(_hive_pair(), _hive_decision())
        assert result.success is True, result.error

    assert ds.get_askrene_layers.call_count == 2


def test_hive_router_unknown_layer_retry_reprobes_despite_cycle_cache():
    """The unknown-layer retry path must invalidate the cycle cache so its
    refresh sees the live layer set, not the cached stale one."""
    ds = _make_hive_data_service()
    good = ds.get_routes.return_value
    ds.get_routes.side_effect = [
        good,
        Exception("Unknown layer: hive-fleet"),
        good,
    ]
    ds.get_askrene_layers.side_effect = [
        {"layers": [{"layer": "hive-fleet"}, {"layer": "revenue-local"}]},
        {"layers": [{"layer": "revenue-local"}]},
    ]
    router = _make_hive_router(ds)

    router.begin_cycle()
    try:
        first = router.price_pair(_hive_pair(), _hive_decision())
        assert first.success is True, first.error
        second = router.price_pair(_hive_pair(), _hive_decision())
        assert second.success is True, second.error
    finally:
        router.end_cycle()

    assert ds.get_askrene_layers.call_count == 2
    # The retry's getroutes call used the refreshed layer set.
    retry_layers = ds.get_routes.call_args.kwargs["layers"]
    assert "hive-fleet" not in retry_layers
    assert "revenue-local" in retry_layers


# ---------------------------------------------------------------------------
# 3. Final-hop policy served from the broadcast listpeerchannels cache
# ---------------------------------------------------------------------------


class _RecordingDataService:
    """Minimal data service double recording get_peer_channels arguments."""

    def __init__(self, channels):
        self._channels = list(channels)
        self.peer_channel_calls = []

    def get_peer_channels(self, peer_id=None):
        self.peer_channel_calls.append(peer_id)
        if peer_id:
            return {
                "channels": [
                    ch for ch in self._channels if ch.get("peer_id") == peer_id
                ]
            }
        return {"channels": list(self._channels)}

    def get_channels(self, source=None, destination=None, short_channel_id=None):
        return {"channels": []}

    def get_configs(self):
        return {"configs": {"cltv-final": {"value_int": 18}}}


def _dest_policy_channel():
    return {
        "peer_id": DST_PEER,
        "short_channel_id": "200x2x0",
        "updates": {
            "remote": {
                "fee_proportional_millionths": 250,
                "fee_base_msat": 1000,
                "cltv_expiry_delta": 40,
            }
        },
    }


def test_v2_final_hop_policy_uses_broadcast_cache():
    """When the (cached) broadcast dump contains the peer, no per-peer
    listpeerchannels RPC is issued."""
    from modules.rebalance_router_v2 import RebalanceRouter

    ds = _RecordingDataService([_dest_policy_channel()])
    router = RebalanceRouter(MagicMock(), OUR_ID, data_service=ds)

    policy = router._get_final_hop_policy(DST_PEER, "200x2x0")

    assert policy == {"fee_ppm": 250, "fee_base_msat": 1000, "cltv_delta": 40}
    assert ds.peer_channel_calls == [None]


def test_v2_final_hop_policy_falls_back_to_per_peer_rpc_when_absent():
    """A peer missing from the broadcast (shouldn't happen for our own
    channels) still triggers the per-peer lookup."""
    from modules.rebalance_router_v2 import RebalanceRouter

    other = dict(_dest_policy_channel(), peer_id=SRC_PEER)
    ds = _RecordingDataService([other])
    router = RebalanceRouter(MagicMock(), OUR_ID, data_service=ds)

    router._get_final_hop_policy(DST_PEER, "200x2x0")

    assert ds.peer_channel_calls == [None, DST_PEER]


def test_hive_return_hop_policy_uses_broadcast_cache():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    ds = _RecordingDataService([_dest_policy_channel()])
    router = RebalanceHiveRouter(
        plugin=MagicMock(),
        our_node_id=OUR_ID,
        hive_hints=None,
        data_service=ds,
    )
    pair = SimpleNamespace(dest_channel_id="200x2x0", dest_peer_id=DST_PEER)

    amount_msat = 1_000_000
    required_amount_msat, required_cltv = router._return_hop_policy(
        pair, amount_msat
    )

    # fee = 1000 base + 1_000_000 * 250ppm / 1e6 = 1250 msat
    assert required_amount_msat == amount_msat + 1250
    assert required_cltv == 18 + 40
    assert None in ds.peer_channel_calls
    assert DST_PEER not in ds.peer_channel_calls


def test_hive_return_hop_policy_falls_back_to_per_peer_rpc_when_absent():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    ds = _RecordingDataService([dict(_dest_policy_channel(), peer_id=SRC_PEER)])
    router = RebalanceHiveRouter(
        plugin=MagicMock(),
        our_node_id=OUR_ID,
        hive_hints=None,
        data_service=ds,
    )
    pair = SimpleNamespace(dest_channel_id="200x2x0", dest_peer_id=DST_PEER)

    router._return_hop_policy(pair, 1_000_000)

    assert DST_PEER in ds.peer_channel_calls


def test_v3_pricing_issues_zero_per_peer_listpeerchannels():
    """End to end: a v3 price_pair backed by a broadcast containing the dest
    peer performs no per-peer listpeerchannels calls."""
    from modules.rebalance_router_v3 import RebalanceRouterV3

    plugin = _make_v3_plugin()

    class _V3DataService(_RecordingDataService):
        def get_channels(
            self, source=None, destination=None, short_channel_id=None
        ):
            return plugin.rpc.listchannels.return_value

        def get_askrene_layers(self):
            return {"layers": [{"layer": "hive-fleet"}]}

        def get_routes(self, **kwargs):
            return plugin.rpc.getroutes(**kwargs)

        def askrene_create_layer(self, layer):
            pass

        def askrene_update_channel(self, layer, scid_dir, enabled):
            pass

        def askrene_remove_layer(self, layer):
            pass

    ds = _V3DataService([_dest_policy_channel()])
    router = RebalanceRouterV3(
        plugin=plugin,
        our_node_id=OUR_ID,
        layer_names=["hive-fleet"],
        log=lambda m, l: None,
        data_service=ds,
    )

    result = _v3_price(router)

    assert result.success is True, result.error
    assert all(peer is None for peer in ds.peer_channel_calls)
    plugin.rpc.listpeerchannels.assert_not_called()


def test_v3_router_unknown_layer_error_drops_cached_exclude_layers():
    plugin = _make_v3_plugin()
    router = _make_v3_router(plugin)
    good_routes = plugin.rpc.getroutes.return_value
    plugin.rpc.getroutes.side_effect = [
        good_routes,
        Exception("Unknown layer hive-fleet"),
        good_routes,
    ]

    router.begin_cycle()
    try:
        assert _v3_price(router).success is True
        failed = _v3_price(router)
        assert failed.success is False
        assert "unknown_layer" in failed.error
        # The cached exclude layer was torn down with the cache...
        assert _count_rpc(plugin, "askrene-remove-layer") == 1
        # ...and the next pricing rebuilds a fresh one and succeeds.
        retried = _v3_price(router)
        assert retried.success is True, retried.error
        assert _count_rpc(plugin, "askrene-create-layer") == 2
    finally:
        router.end_cycle()
