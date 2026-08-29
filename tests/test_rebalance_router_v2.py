"""Tests for RebalanceRouter — route discovery and pricing via CLN RPCs."""

from unittest.mock import MagicMock, call

import pytest

from modules.rebalance_router_v2 import RebalanceRouter, RouteResult


OUR_ID = "02" + "aa" * 32
SOURCE_PEER = "03" + "dd" * 32
DEST_PEER = "03" + "bb" * 32
SOURCE_SCID = "100x1x0"
DEST_SCID = "200x1x0"
AMOUNT_SATS = 50_000


def _make_plugin(
    peer_channels_by_id=None,
    list_channels=None,
    getroute=None,
    getroute_error=None,
    listpeerchannels_error=None,
    listconfigs=None,
):
    """Build a mock plugin with configurable RPC responses.

    peer_channels_by_id: dict mapping peer_id -> listpeerchannels response.
    If None, returns empty channels for any id.
    """
    plugin = MagicMock()

    def _listpeerchannels(peer_id=None, short_channel_id=None):
        """Matches pyln.client.LightningRpc.listpeerchannels signature.

        Regression guard: if router code passes ``id=`` instead of
        ``peer_id=`` (as the real pyln-client expects), this mock raises
        TypeError just like the real RPC layer would on the live node.
        """
        if listpeerchannels_error:
            raise listpeerchannels_error
        if peer_channels_by_id and peer_id in peer_channels_by_id:
            return peer_channels_by_id[peer_id]
        return {"channels": []}

    plugin.rpc.listpeerchannels.side_effect = _listpeerchannels
    plugin.rpc.listchannels.return_value = list_channels or {"channels": []}
    plugin.rpc.listconfigs.return_value = listconfigs or {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    if getroute_error:
        plugin.rpc.getroute.side_effect = getroute_error
    else:
        plugin.rpc.getroute.return_value = getroute or {"route": []}

    return plugin


def _dest_peer_channels(fee_ppm=275, cltv=40, fee_base_msat=0):
    """Standard dest peer channel response."""
    return {
        "channels": [{
            "short_channel_id": DEST_SCID,
            "peer_id": DEST_PEER,
            "updates": {
                "remote": {
                    "fee_base_msat": fee_base_msat,
                    "fee_proportional_millionths": fee_ppm,
                    "cltv_expiry_delta": cltv,
                },
            },
        }],
    }


def _source_peer_channels(fee_ppm=0, cltv=18):
    """Standard source peer channel response (our local policy)."""
    return {
        "channels": [{
            "short_channel_id": SOURCE_SCID,
            "peer_id": SOURCE_PEER,
            "updates": {
                "local": {
                    "fee_proportional_millionths": fee_ppm,
                    "fee_base_msat": 0,
                    "cltv_expiry_delta": cltv,
                },
            },
        }],
    }


def _middle_edge_channels(
    next_node_id,
    fee_ppm=0,
    fee_base_msat=0,
    delay=12,
    channel_id="300x1x0",
    extra_channels=None,
):
    """listchannels payload containing the source peer's first forwarded edge."""
    channels = [{
        "source": SOURCE_PEER,
        "destination": next_node_id,
        "short_channel_id": channel_id,
        "base_fee_millisatoshi": fee_base_msat,
        "fee_per_millionth": fee_ppm,
        "delay": delay,
    }]
    if extra_channels:
        channels.extend(extra_channels)
    return {"channels": channels}


class TestFinalHopFeeFromListpeerchannels:
    def test_router_gets_actual_final_hop_fee_from_listpeerchannels(self):
        """Uses listpeerchannels(peer_id=dest_peer_id) for fee lookup."""
        middle_route = [{
            "id": DEST_PEER,
            "channel": "300x1x0",
            "amount_msat": 50_014_000,
            "delay": 40,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=275),
                SOURCE_PEER: _source_peer_channels(),
            },
            list_channels=_middle_edge_channels(DEST_PEER),
            getroute={"route": middle_route},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        assert result.final_hop_fee_ppm == 275
        # listpeerchannels called with peer_id= filter (not unfiltered).
        # Regression guard: pyln.client.LightningRpc.listpeerchannels
        # accepts peer_id=, not id=. See the _make_plugin helper for the
        # enforced signature.
        calls = plugin.rpc.listpeerchannels.call_args_list
        dest_call = [c for c in calls if c.kwargs.get("peer_id") == DEST_PEER]
        assert len(dest_call) >= 1
        plugin.rpc.listchannels.assert_called_with(short_channel_id="300x1x0")

    def test_router_includes_final_hop_base_fee_in_middle_amount(self):
        """Low-amount routes must budget the remote base fee, not only ppm."""
        middle_route = [{
            "id": DEST_PEER,
            "channel": "300x1x0",
            "amount_msat": 50_001_000,
            "delay": 40,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=0, fee_base_msat=1000),
                SOURCE_PEER: _source_peer_channels(),
            },
            list_channels=_middle_edge_channels(DEST_PEER),
            getroute={"route": middle_route},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        assert plugin.rpc.getroute.call_args.kwargs["amount_msat"] == 50_001_000
        assert result.route_cost_sats == 1

    def test_router_preserves_sub_sat_final_hop_fee_precision(self):
        """Do not turn a 1,155-msat forwarding fee into 2,000 msat."""
        amount_sats = 155_000
        exact_route_amount_msat = 155_001_155
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=1, fee_base_msat=1_000),
                SOURCE_PEER: _source_peer_channels(),
            },
            list_channels=_middle_edge_channels(DEST_PEER),
            getroute={"route": [{
                "id": DEST_PEER,
                "channel": "300x1x0",
                "amount_msat": exact_route_amount_msat,
                "delay": 40,
            }]},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, amount_sats
        )

        assert result.success is True
        assert plugin.rpc.getroute.call_args.kwargs["amount_msat"] == exact_route_amount_msat
        assert result.route[0]["amount_msat"] == exact_route_amount_msat
        assert result.route_cost_sats == 2

    def test_router_does_not_call_listpeerchannels_with_legacy_id_kwarg(self):
        """Regression test: earlier versions passed id= which raised
        TypeError on pyln.client.LightningRpc.listpeerchannels. Live nexus-01
        surfaced the bug silently because the listchannels fallback caught it."""
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=275),
                SOURCE_PEER: _source_peer_channels(),
            },
            list_channels=_middle_edge_channels(DEST_PEER),
            getroute={"route": [{
                "id": DEST_PEER,
                "channel": "300x1x0",
                "amount_msat": 50_014_000,
                "delay": 40,
            }]},
        )
        router = RebalanceRouter(plugin, OUR_ID)
        router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        # Every listpeerchannels call must use the real pyln-client keyword
        # (peer_id=), never the buggy id=.
        for c in plugin.rpc.listpeerchannels.call_args_list:
            assert "id" not in c.kwargs, (
                f"listpeerchannels called with legacy id= kwarg: {c.kwargs}"
            )


class TestFallbackToListchannels:
    def test_router_falls_back_to_listchannels_when_no_peer_updates(self):
        bare_peer = {"channels": [{"short_channel_id": DEST_SCID, "peer_id": DEST_PEER}]}
        list_channels = {
            "channels": _middle_edge_channels(
                DEST_PEER,
                extra_channels=[{
                    "source": DEST_PEER,
                    "destination": OUR_ID,
                    "short_channel_id": DEST_SCID,
                    "fee_per_millionth": 500,
                    "delay": 40,
                }],
            )["channels"],
        }
        middle_route = [{
            "id": DEST_PEER,
            "channel": "300x1x0",
            "amount_msat": 50_025_000,
            "delay": 40,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: bare_peer,
                SOURCE_PEER: _source_peer_channels(),
            },
            list_channels=list_channels,
            getroute={"route": middle_route},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        assert result.final_hop_fee_ppm == 500
        assert call(source=DEST_PEER) in plugin.rpc.listchannels.call_args_list
        assert call(short_channel_id="300x1x0") in plugin.rpc.listchannels.call_args_list


class TestFullRoute:
    def test_route_includes_first_and_final_hops(self):
        """Full route: first_hop(source) + middle + final_hop(dest→us)."""
        middle_route = [{
            "id": "03" + "cc" * 32,
            "channel": "300x1x0",
            "amount_msat": 50_010_000,
            "delay": 20,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=200, cltv=40),
                SOURCE_PEER: _source_peer_channels(cltv=18),
            },
            list_channels=_middle_edge_channels(middle_route[0]["id"]),
            getroute={"route": middle_route},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        # Route structure: first_hop + middle + final_hop
        assert len(result.route) == 3
        assert result.route[0]["id"] == SOURCE_PEER
        assert result.route[0]["channel"] == SOURCE_SCID
        assert result.route[-1]["id"] == OUR_ID
        assert result.route[-1]["channel"] == DEST_SCID
        assert result.route[-1]["amount_msat"] == AMOUNT_SATS * 1000

    def test_route_uses_invoice_cltv_and_explicit_directions(self):
        """Synthetic hops must match the sendpay contract used by the executor."""
        middle_route = [{
            "id": DEST_PEER,
            "channel": "300x1x0",
            "direction": 0,
            "style": "tlv",
            "amount_msat": 50_010_000,
            "delay": 58,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=200, cltv=40),
                SOURCE_PEER: _source_peer_channels(cltv=18),
            },
            list_channels=_middle_edge_channels(DEST_PEER),
            getroute={"route": middle_route},
            listconfigs={"configs": {"cltv-final": {"value_int": 18}}},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        plugin.rpc.getroute.assert_called_once()
        kwargs = plugin.rpc.getroute.call_args.kwargs
        assert kwargs["cltv"] == 58
        assert result.route[0]["direction"] == 0
        assert result.route[0]["delay"] == 70
        assert result.route[0]["style"] == "tlv"
        assert result.route[-1]["direction"] == 1
        assert result.route[-1]["delay"] == 18
        assert result.route[-1]["style"] == "tlv"

    def test_first_hop_amount_includes_fee_for_first_middle_edge(self):
        """The prepended first hop must fund the source peer's outgoing fee."""
        middle_route = [{
            "id": "03" + "cc" * 32,
            "channel": "300x1x0",
            "direction": 0,
            "style": "tlv",
            "amount_msat": 50_015_750,
            "delay": 58,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=275, cltv=40),
                SOURCE_PEER: _source_peer_channels(cltv=18),
            },
            list_channels={
                "channels": [{
                    "source": SOURCE_PEER,
                    "destination": middle_route[0]["id"],
                    "short_channel_id": "300x1x0",
                    "direction": 0,
                    "base_fee_millisatoshi": 5000,
                    "fee_per_millionth": 100,
                    "delay": 12,
                }],
            },
            getroute={"route": middle_route},
            listconfigs={"configs": {"cltv-final": {"value_int": 18}}},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        # Middle amount is repriced from the final-hop fee budget before the
        # source peer's forwarding fee is added.
        assert result.route[0]["amount_msat"] == 50_023_752
        assert result.route_cost_sats == 24
        plugin.rpc.listchannels.assert_called_with(short_channel_id="300x1x0")

    def test_first_hop_delay_includes_cltv_delta_for_first_middle_edge(self):
        """The prepended first hop must give the source peer its forwarding delta."""
        next_hop = "03" + "cc" * 32
        middle_route = [{
            "id": next_hop,
            "channel": "300x1x0",
            "direction": 0,
            "style": "tlv",
            "amount_msat": 50_015_750,
            "delay": 58,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=275, cltv=40),
                SOURCE_PEER: _source_peer_channels(cltv=18),
            },
            list_channels={
                "channels": [{
                    "source": SOURCE_PEER,
                    "destination": next_hop,
                    "short_channel_id": "300x1x0",
                    "direction": 0,
                    "base_fee_millisatoshi": 0,
                    "fee_per_millionth": 0,
                    "delay": 12,
                }],
            },
            getroute={"route": middle_route},
            listconfigs={"configs": {"cltv-final": {"value_int": 18}}},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        assert result.route[0]["delay"] == 70


class TestExcludePassthrough:
    def test_router_passes_exclude_to_getroute(self):
        middle_route = [{
            "id": DEST_PEER,
            "channel": "300x1x0",
            "amount_msat": 50_005_000,
            "delay": 10,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=100),
                SOURCE_PEER: _source_peer_channels(),
            },
            list_channels=_middle_edge_channels(DEST_PEER),
            getroute={"route": middle_route},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        exclude_list = ["400x1x0/0", "03" + "dd" * 32]
        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS,
            exclude=exclude_list,
        )

        assert result.success is True
        call_kwargs = plugin.rpc.getroute.call_args.kwargs
        assert call_kwargs["exclude"] == exclude_list
        assert call_kwargs["riskfactor"] == 10
        assert call_kwargs["fromid"] == SOURCE_PEER


class TestFailureOnNoRoute:
    def test_router_returns_failure_on_no_route(self):
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=100),
                SOURCE_PEER: _source_peer_channels(),
            },
            getroute_error=Exception("Could not find a route"),
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is False
        assert "getroute failed" in result.error

    def test_router_returns_failure_when_fee_unknown(self):
        plugin = _make_plugin()
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is False
        assert "cannot determine final-hop fee" in result.error


class TestZeroFeePeer:
    def test_zero_ppm_peer_costs_zero(self):
        """0-PPM peers should cost 0, not 1 sat minimum."""
        middle_route = [{
            "id": DEST_PEER,
            "channel": "300x1x0",
            "amount_msat": AMOUNT_SATS * 1000,
            "delay": 10,
        }]
        plugin = _make_plugin(
            peer_channels_by_id={
                DEST_PEER: _dest_peer_channels(fee_ppm=0),
                SOURCE_PEER: _source_peer_channels(),
            },
            list_channels=_middle_edge_channels(DEST_PEER, delay=0),
            getroute={"route": middle_route},
        )
        router = RebalanceRouter(plugin, OUR_ID)

        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        assert result.final_hop_fee_ppm == 0
        assert result.route_cost_sats == 0


class TestDataServiceRouting:
    def test_router_prefers_data_service_for_reads_and_route_lookup(self):
        middle_route = [{
            "id": DEST_PEER,
            "channel": "300x1x0",
            "amount_msat": 50_014_000,
            "delay": 40,
        }]
        plugin = _make_plugin()
        data_service = MagicMock()

        def _get_peer_channels(peer_id=None):
            if peer_id == DEST_PEER:
                return _dest_peer_channels(fee_ppm=275)
            if peer_id == SOURCE_PEER:
                return _source_peer_channels()
            return {"channels": []}

        data_service.get_peer_channels.side_effect = _get_peer_channels
        data_service.get_channels.side_effect = (
            lambda source=None, destination=None, short_channel_id=None: (
                _middle_edge_channels(DEST_PEER)
                if short_channel_id == "300x1x0"
                else {"channels": []}
            )
        )
        data_service.get_configs.return_value = {
            "configs": {"cltv-final": {"value_int": 18}}
        }
        data_service.get_route.return_value = {"route": middle_route}

        router = RebalanceRouter(plugin, OUR_ID, data_service=data_service)
        result = router.price_pair(
            SOURCE_SCID, DEST_SCID, SOURCE_PEER, DEST_PEER, AMOUNT_SATS
        )

        assert result.success is True
        data_service.get_peer_channels.assert_any_call(DEST_PEER)
        data_service.get_configs.assert_called_once_with()
        data_service.get_route.assert_called_once()
        kwargs = data_service.get_route.call_args.kwargs
        assert kwargs["node_id"] == DEST_PEER
        assert kwargs["fromid"] == SOURCE_PEER
        plugin.rpc.listpeerchannels.assert_not_called()
        plugin.rpc.listchannels.assert_not_called()
        plugin.rpc.listconfigs.assert_not_called()
        plugin.rpc.getroute.assert_not_called()
