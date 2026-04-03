"""Tests for HiveRouter module."""

import time
from unittest.mock import MagicMock, patch, call
import pytest

from modules.hive_router import HiveRouter, HiveRoute


class MockHiveHints:
    def __init__(self, members=None):
        self._members = set(members or [])

    def is_hive_member(self, peer_id):
        return peer_id in self._members


class TestHiveRouterInit:
    def test_defaults(self):
        router = HiveRouter(plugin=MagicMock(), hive_hints=MockHiveHints())
        assert router.available is False
        assert router.get_hive_members() == set()
        assert router.is_hive_member("abc") is False

    def test_no_hints(self):
        router = HiveRouter(plugin=MagicMock(), hive_hints=None)
        assert router.refresh_layer() is False
        assert router.available is False

    def test_no_plugin(self):
        router = HiveRouter(plugin=None, hive_hints=MockHiveHints())
        assert router.refresh_layer() is False


class TestHiveRouterRefresh:
    def test_refresh_creates_layer_and_updates_channels(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {"state": "CHANNELD_NORMAL", "peer_id": "fleet_a", "short_channel_id": "100x1x0"},
                {"state": "CHANNELD_NORMAL", "peer_id": "external", "short_channel_id": "200x1x0"},
                {"state": "CHANNELD_NORMAL", "peer_id": "fleet_b", "short_channel_id": "300x1x0"},
            ]
        }
        plugin.rpc.call.return_value = {}

        hints = MockHiveHints(members=["fleet_a", "fleet_b"])
        router = HiveRouter(plugin, hints)
        result = router.refresh_layer()

        assert result is True
        assert router.available is True
        assert router.get_hive_members() == {"fleet_a", "fleet_b"}
        assert router.is_hive_member("fleet_a") is True
        assert router.is_hive_member("external") is False

    def test_refresh_fails_gracefully_when_askrene_unavailable(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        plugin.rpc.call.side_effect = Exception("Unknown method askrene-create-layer")

        router = HiveRouter(plugin, MockHiveHints(["fleet_a"]))
        result = router.refresh_layer()

        assert result is False
        assert router.available is False


class TestHiveRouterDiscover:
    def test_discover_returns_route(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.call.return_value = {
            "probability_ppm": 850000,
            "routes": [{
                "amount_msat": 500000,
                "path": [
                    {"short_channel_id_dir": "100x1x0/1", "amount_msat": 501000},
                    {"short_channel_id_dir": "200x1x0/0", "amount_msat": 500000},
                ]
            }]
        }

        router = HiveRouter(plugin, MockHiveHints())
        router.available = True
        router._our_id = "our_id"

        route = router.discover_route("dest_peer", 500)
        assert route is not None
        assert route.source_scid == "100x1x0"
        assert route.hops == 2
        assert route.fee_ppm == 2000  # (501000-500000)*1e6/500000 = 2000
        assert route.probability_ppm == 850000

    def test_discover_returns_none_when_unavailable(self):
        router = HiveRouter(MagicMock(), MockHiveHints())
        router.available = False
        assert router.discover_route("dest", 1000) is None

    def test_discover_returns_none_on_no_routes(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {"routes": []}

        router = HiveRouter(plugin, MockHiveHints())
        router.available = True
        router._our_id = "our_id"

        assert router.discover_route("dest", 1000) is None


class TestHiveRouterTopologyScore:
    def test_fleet_peer_loop_out_high_ratio(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("fleet_a", "out", liquidity_ratio=0.9)
        assert score > 1.3  # High ratio = very beneficial

    def test_fleet_peer_loop_out_balanced(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("fleet_a", "out", liquidity_ratio=0.5)
        assert 1.15 < score < 1.25

    def test_fleet_peer_loop_in_low_ratio(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("fleet_a", "in", liquidity_ratio=0.2)
        assert score > 1.15

    def test_non_fleet_peer_neutral(self):
        router = HiveRouter(MagicMock(), MockHiveHints(["fleet_a"]))
        router._member_ids = {"fleet_a"}
        score = router.score_channel_for_hive("random_peer", "out", liquidity_ratio=0.9)
        assert score == 1.0

    def test_no_members_neutral(self):
        router = HiveRouter(MagicMock(), MockHiveHints())
        router._member_ids = set()
        assert router.score_channel_for_hive("any", "out") == 1.0


class TestHiveRouterLayerDetection:
    def test_detects_cl_hive_managed_layer(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {
            "layers": [
                {"layer": "hive-fleet", "persistent": False},
                {"layer": "hive-reputation", "persistent": False},
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {"state": "CHANNELD_NORMAL", "peer_id": "fleet_a", "short_channel_id": "100x1x0"},
            ]
        }

        hints = MockHiveHints(members=["fleet_a"])
        router = HiveRouter(plugin, hints)
        result = router.refresh_layer()

        assert result is True
        assert router.available is True
        assert router.is_hive_member("fleet_a") is True

    def test_falls_back_to_standalone_when_no_cl_hive(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {"state": "CHANNELD_NORMAL", "peer_id": "fleet_a", "short_channel_id": "100x1x0"},
            ]
        }

        def side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            return {}

        plugin.rpc.call.side_effect = side_effect

        hints = MockHiveHints(members=["fleet_a"])
        router = HiveRouter(plugin, hints)
        result = router.refresh_layer()

        assert result is True
        assert router.available is True
