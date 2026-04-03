"""Tests for RebalanceExecutor."""

import time
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass, field
from typing import List
import pytest

from modules.rebalance_executor import (
    RebalanceExecutor, RebalanceJob, RebalanceResult, JobState
)


@dataclass
class MockCandidate:
    source_candidates: List[str] = field(default_factory=lambda: ["100x1x0"])
    to_channel: str = "200x1x0"
    to_peer_id: str = "dest_peer_abc"
    amount_sats: int = 500000
    amount_msat: int = 500000000
    max_budget_sats: int = 100
    max_budget_msat: int = 100000
    max_fee_ppm: int = 200
    hive_route_hops: int = 0
    direction: str = "pull"


class MockHiveRouter:
    def __init__(self, is_member=False, max_through=0):
        self.available = True
        self._is_member = is_member
        self._max = max_through

    def is_hive_member(self, pid):
        return self._is_member

    def max_rebalance_through_member(self, pid):
        return self._max


class TestRebalanceExecutorInit:
    def test_defaults(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        assert executor.active_count == 0
        assert executor.get_active_jobs() == []

    def test_with_hive_router(self):
        executor = RebalanceExecutor(
            MagicMock(), MagicMock(), MagicMock(),
            hive_router=MockHiveRouter()
        )
        assert executor.hive_router is not None


class TestLayerSelection:
    def test_fleet_layers_include_hive(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {
            "layers": [
                {"layer": "hive-fleet"},
                {"layer": "hive-reputation"},
                {"layer": "revenue-local"},
                {"layer": "unrelated"},
            ]
        }
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        layers = executor._get_layers("fleet")
        assert "auto.localchans" in layers
        assert "auto.sourcefree" not in layers  # NEVER for circular rebalancing
        assert "hive-fleet" in layers
        assert "hive-reputation" in layers
        assert "revenue-local" in layers
        assert "unrelated" not in layers

    def test_network_layers_exclude_hive(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {
            "layers": [
                {"layer": "hive-fleet"},
                {"layer": "revenue-local"},
            ]
        }
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        layers = executor._get_layers("network")
        assert "hive-fleet" not in layers
        assert "revenue-local" in layers

    def test_layers_fallback_on_error(self):
        plugin = MagicMock()
        plugin.rpc.call.side_effect = Exception("askrene unavailable")
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        layers = executor._get_layers("fleet")
        assert layers == ["auto.localchans"]


class TestRouteConversion:
    def test_converts_getroutes_to_sendpay(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        path = [
            {"short_channel_id_dir": "100x1x0/1", "next_node_id": "node_a",
             "amount_msat": 501000, "delay": 42},
            {"short_channel_id_dir": "200x1x0/0", "next_node_id": "node_b",
             "amount_msat": 500000, "delay": 24},
        ]
        route = executor._getroutes_to_sendpay(path, "300x1x0", "our_id", 500000)

        assert len(route) == 3  # 2 hops + final circular hop
        assert route[0] == {"channel": "100x1x0", "id": "node_a",
                            "amount_msat": 501000, "delay": 42}
        assert route[1] == {"channel": "200x1x0", "id": "node_b",
                            "amount_msat": 500000, "delay": 24}
        assert route[2] == {"channel": "300x1x0", "id": "our_id",
                            "amount_msat": 500000, "delay": 18}


class TestRouteTypeSelection:
    def test_fleet_when_hive_route_hops(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.call.return_value = {"routes": []}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc", "payment_secret": "def"
        }
        plugin.rpc.listpeerchannels.return_value = {"channels": []}

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=2)
        result = executor.execute(candidate)

        # Should attempt fleet routing (getroutes called with fleet layers)
        assert result.route_type == "fleet"

    def test_network_when_no_hive_route(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.call.return_value = {"routes": []}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc", "payment_secret": "def"
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=0)
        result = executor.execute(candidate)
        assert result.route_type == "network"


class TestExecuteSuccess:
    def test_successful_single_part(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123"
        }

        def rpc_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            if method == "getroutes":
                return {
                    "probability_ppm": 950000,
                    "routes": [{
                        "amount_msat": 500000000,
                        "path": [
                            {"short_channel_id_dir": "100x1x0/1",
                             "next_node_id": "peer_a",
                             "amount_msat": 500050000,
                             "delay": 42},
                        ]
                    }]
                }
            if method == "sendpay":
                return {}
            if method == "waitsendpay":
                return {"status": "complete", "amount_sent_msat": 500050000}
            if method == "askrene-inform-channel":
                return {}
            return {}

        plugin.rpc.call.side_effect = rpc_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=0)
        result = executor.execute(candidate)

        assert result.success is True
        assert result.fee_msat == 50000
        assert result.fee_ppm == 100  # 50000 * 1e6 / 500000000
        assert result.attempts == 1
        assert result.route_type == "network"


class TestExecuteFailure:
    def test_no_routes(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash", "payment_secret": "secret"
        }

        def rpc_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            if method == "getroutes":
                return {"routes": []}
            return {}

        plugin.rpc.call.side_effect = rpc_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate())

        assert result.success is False
        assert result.error == "no_routes"

    def test_cleans_up_invoice_on_failure(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash", "payment_secret": "secret"
        }
        plugin.rpc.call.return_value = {"routes": [], "layers": []}

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        executor.execute(MockCandidate())

        # delinvoice should be called for cleanup
        plugin.rpc.delinvoice.assert_called_once()


class TestCancelAndShutdown:
    def test_cancel_nonexistent(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        assert executor.cancel("999x1x0") is False

    def test_cancel_all_empty(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        assert executor.cancel_all() == 0

    def test_shutdown(self):
        executor = RebalanceExecutor(MagicMock(), MagicMock(), MagicMock())
        executor.shutdown()
        # Should not raise


class TestInformChannel:
    def test_informs_on_success(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {}
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())

        path = [
            {"short_channel_id_dir": "100x1x0/1", "next_node_id": "a",
             "amount_msat": 500000, "delay": 24},
        ]
        executor._inform_result(path, 500000, succeeded=True)

        inform_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "askrene-inform-channel"
        ]
        assert len(inform_calls) == 1
        assert inform_calls[0][0][1]["inform"] == "succeeded"

    def test_informs_on_failure(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {}
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())

        path = [{"short_channel_id_dir": "100x1x0/1", "next_node_id": "a",
                 "amount_msat": 500000, "delay": 24}]
        executor._inform_result(path, 500000, succeeded=False)

        inform_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "askrene-inform-channel"
        ]
        assert len(inform_calls) == 1
        assert inform_calls[0][0][1]["inform"] == "failed"
