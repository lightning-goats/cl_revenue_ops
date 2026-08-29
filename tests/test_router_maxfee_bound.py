"""Audit 2026-08-01 wave2 FIX 4: constrain askrene's maxfee to the engine's
acceptance bound.

``getroutes`` was called with ``maxfee_msat == route_amount_msat`` (100% of
the amount as fee), so askrene's MCF could return an expensive
high-reliability route the engine's budget gate then rejected
(``route_over_budget``) even when a cheaper within-budget route existed —
and large-amount calls solved the unconstrained problem (6-16s per call).

The engine now threads the pair's largest gate-acceptable fee down to
``price_pair(maxfee_sats=...)``; the router subtracts the final-hop fee
(embedded in the route amount, not counted by getroutes' maxfee) and caps
the middle-path search. The bound is never TIGHTER than the gate: any route
the gate could accept still fits under it, so the gate stays authoritative.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


OUR_ID = "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3"
SRC_PEER = "03" + "a" * 64
DST_PEER = "03" + "b" * 64


def _make_plugin(final_fee_ppm=0, final_base_msat=0):
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    plugin.rpc.listchannels.return_value = {
        "channels": [{
            "source": DST_PEER,
            "destination": OUR_ID,
            "fee_per_millionth": final_fee_ppm,
            "base_fee_millisatoshi": final_base_msat,
            "delay": 40,
        }]
    }
    plugin.rpc.getroutes.return_value = {"routes": []}
    return plugin


def _make_router(plugin):
    from modules.rebalance_router_v3 import RebalanceRouterV3

    return RebalanceRouterV3(
        plugin=plugin,
        our_node_id=OUR_ID,
        layer_names=["hive-fleet"],
        log=lambda m, l: None,
    )


def _price(router, amount_sats=100, **kwargs):
    return router.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=amount_sats,
        **kwargs,
    )


def test_maxfee_bound_reaches_getroutes():
    plugin = _make_plugin(final_fee_ppm=0)
    router = _make_router(plugin)

    _price(router, amount_sats=100, maxfee_sats=10)

    kwargs = plugin.rpc.getroutes.call_args.kwargs
    # Final-hop fee is 0, so the whole 10-sat bound funds the middle path.
    assert kwargs["maxfee_msat"] == 10_000


def test_maxfee_bound_subtracts_final_hop_fee():
    # 50_000 ppm on 100 sats -> final-hop fee 5 sats, already embedded in
    # the route amount. getroutes' maxfee covers only middle-path fees.
    plugin = _make_plugin(final_fee_ppm=50_000)
    router = _make_router(plugin)

    _price(router, amount_sats=100, maxfee_sats=10)

    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["amount_msat"] == 105_000
    assert kwargs["maxfee_msat"] == 5_000


def test_maxfee_bound_subtracts_exact_sub_sat_final_hop_fee():
    plugin = _make_plugin(final_fee_ppm=1, final_base_msat=1_000)
    router = _make_router(plugin)

    _price(router, amount_sats=155_000, maxfee_sats=4)

    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["amount_msat"] == 155_001_155
    assert kwargs["maxfee_msat"] == 2_845


def test_maxfee_bound_never_negative():
    plugin = _make_plugin(final_fee_ppm=100_000)  # final fee 10 sats
    router = _make_router(plugin)

    _price(router, amount_sats=100, maxfee_sats=4)

    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["maxfee_msat"] == 0


def test_maxfee_bound_capped_at_route_amount():
    plugin = _make_plugin(final_fee_ppm=0)
    router = _make_router(plugin)

    _price(router, amount_sats=100, maxfee_sats=10_000_000)

    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["maxfee_msat"] == kwargs["amount_msat"]


def test_no_bound_keeps_legacy_unconstrained_maxfee():
    plugin = _make_plugin(final_fee_ppm=0)
    router = _make_router(plugin)

    _price(router, amount_sats=100)

    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["maxfee_msat"] == kwargs["amount_msat"]


# ---------------------------------------------------------------------------
# Engine plumbing
# ---------------------------------------------------------------------------


def _make_engine():
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    cfg = Config(dry_run=True, rebalance_router="v3")
    return RebalanceEngine(plugin=plugin, config=cfg, database=MagicMock())


def _pair(budget=100, amount=50_000):
    from modules.rebalance_types_v2 import PairCandidate

    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=amount,
        pair_budget_sats=budget,
        reason_code="ev_positive",
        route=None,
    )


def test_engine_threads_acceptance_bound_to_router():
    engine = _make_engine()
    router = MagicMock()
    # Default pair_fee_cap_ppm=1000 caps an attempt on 50k sats at 50 sats;
    # a 30-sat budget is below that ceiling, so the bound is the budget.
    pair = _pair(budget=30)

    engine._route_pair(pair=pair, router=router, exclude=None)

    kwargs = router.price_pair.call_args.kwargs
    assert kwargs["maxfee_sats"] == 30


def test_engine_bound_includes_max_probability_relaxation():
    engine = _make_engine()
    engine.config.capex_probability_budget_bonus = 0.25
    router = MagicMock()
    pair = _pair(budget=30)

    engine._route_pair(pair=pair, router=router, exclude=None)

    kwargs = router.price_pair.call_args.kwargs
    # Bound uses the MAXIMUM relaxation (probability unknown pre-pricing):
    # 30 * (1 + 0.25) = 37 — never tighter than what the gate can accept.
    assert kwargs["maxfee_sats"] == 37


def test_engine_bound_honors_per_attempt_ppm_ceiling():
    engine = _make_engine()
    engine.config.pair_fee_cap_ppm = 1_000  # 1000 ppm of 50k = 50 sats
    router = MagicMock()
    pair = _pair(budget=100, amount=50_000)

    engine._route_pair(pair=pair, router=router, exclude=None)

    kwargs = router.price_pair.call_args.kwargs
    assert kwargs["maxfee_sats"] == 50


def test_market_price_pair_falls_back_for_routers_without_maxfee():
    """Older router/price_pair signatures (or test doubles) without the
    maxfee_sats kwarg must keep working: retry without the bound."""
    from modules.rebalance_engine_v2 import RebalanceEngine

    calls = []

    class LegacyRouter:
        def price_pair(self, **kwargs):
            if "maxfee_sats" in kwargs:
                raise TypeError(
                    "price_pair() got an unexpected keyword argument "
                    "'maxfee_sats'"
                )
            calls.append(kwargs)
            return "priced"

    result = RebalanceEngine._market_price_pair(
        LegacyRouter(), _pair(), None, maxfee_sats=42,
    )

    assert result == "priced"
    assert len(calls) == 1
    assert "maxfee_sats" not in calls[0]
