"""Tests for RebalanceRouterV2 — route discovery and pricing via CLN RPCs."""

from unittest.mock import MagicMock

import pytest

from modules.rebalance_router_v2 import RebalanceRouterV2, V2RouteResult


OUR_ID = "02" + "aa" * 32
DEST_PEER = "03" + "bb" * 32
SOURCE_SCID = "100x1x0"
DEST_SCID = "200x1x0"
AMOUNT_SATS = 50_000


def _make_plugin(
    peer_channels=None,
    list_channels=None,
    getroute=None,
    getroute_error=None,
    listpeerchannels_error=None,
):
    """Build a mock plugin with configurable RPC responses."""
    plugin = MagicMock()

    if listpeerchannels_error:
        plugin.rpc.listpeerchannels.side_effect = listpeerchannels_error
    else:
        plugin.rpc.listpeerchannels.return_value = peer_channels or {"channels": []}

    plugin.rpc.listchannels.return_value = list_channels or {"channels": []}

    if getroute_error:
        plugin.rpc.getroute.side_effect = getroute_error
    else:
        plugin.rpc.getroute.return_value = getroute or {"route": []}

    return plugin


class TestFinalHopFeeFromListpeerchannels:
    """test_router_gets_actual_final_hop_fee_from_listpeerchannels"""

    def test_router_gets_actual_final_hop_fee_from_listpeerchannels(self):
        """The router extracts fee_proportional_millionths from updates.remote."""
        peer_channels = {
            "channels": [
                {
                    "short_channel_id": DEST_SCID,
                    "peer_id": DEST_PEER,
                    "updates": {
                        "remote": {
                            "fee_proportional_millionths": 275,
                            "cltv_expiry_delta": 40,
                        }
                    },
                }
            ]
        }
        # getroute returns a 2-hop route with fee baked in
        # route_amount_sats = 50_000 + ceil(50_000 * 275 / 1_000_000) = 50_000 + 14 = 50_014
        route_amount_sats = 50_014
        getroute_resp = {
            "route": [
                {
                    "id": "03" + "cc" * 32,
                    "channel": "300x1x0",
                    "amount_msat": route_amount_sats * 1000 + 100_000,
                    "delay": 40,
                },
                {
                    "id": OUR_ID,
                    "channel": DEST_SCID,
                    "amount_msat": route_amount_sats * 1000,
                    "delay": 10,
                },
            ]
        }
        plugin = _make_plugin(peer_channels=peer_channels, getroute=getroute_resp)
        router = RebalanceRouterV2(plugin, OUR_ID)

        result = router.price_pair(SOURCE_SCID, DEST_SCID, DEST_PEER, AMOUNT_SATS)

        assert result.success is True
        assert result.final_hop_fee_ppm == 275
        # listchannels should NOT have been called since listpeerchannels succeeded
        plugin.rpc.listchannels.assert_not_called()


class TestFallbackToListchannels:
    """test_router_falls_back_to_listchannels_when_no_peer_updates"""

    def test_router_falls_back_to_listchannels_when_no_peer_updates(self):
        """When listpeerchannels has no updates.remote, fall back to listchannels."""
        # listpeerchannels returns channel without updates.remote
        peer_channels = {
            "channels": [
                {
                    "short_channel_id": DEST_SCID,
                    "peer_id": DEST_PEER,
                    # no "updates" key at all
                }
            ]
        }
        list_channels = {
            "channels": [
                {
                    "source": DEST_PEER,
                    "destination": OUR_ID,
                    "short_channel_id": DEST_SCID,
                    "fee_per_millionth": 500,
                    "base_fee_millisatoshi": 1000,
                }
            ]
        }
        # route_amount_sats = 50_000 + ceil(50_000 * 500 / 1_000_000) = 50_000 + 25 = 50_025
        route_amount_sats = 50_025
        getroute_resp = {
            "route": [
                {
                    "id": OUR_ID,
                    "channel": DEST_SCID,
                    "amount_msat": route_amount_sats * 1000,
                    "delay": 10,
                }
            ]
        }
        plugin = _make_plugin(
            peer_channels=peer_channels,
            list_channels=list_channels,
            getroute=getroute_resp,
        )
        router = RebalanceRouterV2(plugin, OUR_ID)

        result = router.price_pair(SOURCE_SCID, DEST_SCID, DEST_PEER, AMOUNT_SATS)

        assert result.success is True
        assert result.final_hop_fee_ppm == 500
        # listchannels WAS called as fallback
        plugin.rpc.listchannels.assert_called_once_with(source=DEST_PEER)


class TestTotalRouteCost:
    """test_router_computes_total_route_cost"""

    def test_router_computes_total_route_cost(self):
        """Total cost = intermediate hop fees + final-hop fee."""
        fee_ppm = 200
        # final_hop_fee = ceil(50_000 * 200 / 1_000_000) = 10
        final_hop_fee_sats = 10
        route_amount_sats = AMOUNT_SATS + final_hop_fee_sats  # 50_010

        # The route has 2 hops.  First hop carries 100 sats extra above
        # delivery, so intermediate fee = ceil(100_000 / 1000) = 100 sats.
        intermediate_fee_msat = 100_000

        peer_channels = {
            "channels": [
                {
                    "short_channel_id": DEST_SCID,
                    "peer_id": DEST_PEER,
                    "updates": {
                        "remote": {
                            "fee_proportional_millionths": fee_ppm,
                            "cltv_expiry_delta": 40,
                        }
                    },
                }
            ]
        }
        getroute_resp = {
            "route": [
                {
                    "id": "03" + "cc" * 32,
                    "channel": "300x1x0",
                    "amount_msat": route_amount_sats * 1000 + intermediate_fee_msat,
                    "delay": 40,
                },
                {
                    "id": OUR_ID,
                    "channel": DEST_SCID,
                    "amount_msat": route_amount_sats * 1000,
                    "delay": 10,
                },
            ]
        }
        plugin = _make_plugin(peer_channels=peer_channels, getroute=getroute_resp)
        router = RebalanceRouterV2(plugin, OUR_ID)

        result = router.price_pair(SOURCE_SCID, DEST_SCID, DEST_PEER, AMOUNT_SATS)

        assert result.success is True
        # intermediate = 100 sats, final hop = 10 sats
        assert result.route_cost_sats == 100 + final_hop_fee_sats
        assert result.hops == 2
        assert len(result.route) == 2


class TestExcludePassthrough:
    """test_router_passes_exclude_to_getroute"""

    def test_router_passes_exclude_to_getroute(self):
        """Exclude list is forwarded to getroute for retry diversification."""
        peer_channels = {
            "channels": [
                {
                    "short_channel_id": DEST_SCID,
                    "peer_id": DEST_PEER,
                    "updates": {
                        "remote": {
                            "fee_proportional_millionths": 100,
                            "cltv_expiry_delta": 40,
                        }
                    },
                }
            ]
        }
        getroute_resp = {
            "route": [
                {
                    "id": OUR_ID,
                    "channel": DEST_SCID,
                    "amount_msat": 50_005_000,
                    "delay": 10,
                }
            ]
        }
        plugin = _make_plugin(peer_channels=peer_channels, getroute=getroute_resp)
        router = RebalanceRouterV2(plugin, OUR_ID)

        exclude_list = ["400x1x0/0", "03" + "dd" * 32]
        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, DEST_PEER, AMOUNT_SATS, exclude=exclude_list
        )

        assert result.success is True
        call_kwargs = plugin.rpc.getroute.call_args.kwargs
        assert call_kwargs["exclude"] == exclude_list
        assert call_kwargs["riskfactor"] == 10
        assert call_kwargs["fromid"] == DEST_PEER


class TestFailureOnNoRoute:
    """test_router_returns_failure_on_no_route"""

    def test_router_returns_failure_on_no_route(self):
        """When getroute raises an exception, result is failure with error."""
        peer_channels = {
            "channels": [
                {
                    "short_channel_id": DEST_SCID,
                    "peer_id": DEST_PEER,
                    "updates": {
                        "remote": {
                            "fee_proportional_millionths": 100,
                            "cltv_expiry_delta": 40,
                        }
                    },
                }
            ]
        }
        plugin = _make_plugin(
            peer_channels=peer_channels,
            getroute_error=Exception("Could not find a route"),
        )
        router = RebalanceRouterV2(plugin, OUR_ID)

        result = router.price_pair(SOURCE_SCID, DEST_SCID, DEST_PEER, AMOUNT_SATS)

        assert result.success is False
        assert "getroute failed" in result.error
        assert result.route_cost_sats == 0
        assert result.hops == 0
        assert result.route == []

    def test_router_returns_failure_when_fee_unknown(self):
        """When neither listpeerchannels nor listchannels has fee data, fail."""
        # Both RPCs return empty/no-match results
        plugin = _make_plugin()
        router = RebalanceRouterV2(plugin, OUR_ID)

        result = router.price_pair(SOURCE_SCID, DEST_SCID, DEST_PEER, AMOUNT_SATS)

        assert result.success is False
        assert "cannot determine final-hop fee" in result.error
