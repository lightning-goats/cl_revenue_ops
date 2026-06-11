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
