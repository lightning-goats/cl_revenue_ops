"""Tests for RebalanceExecutor."""

import time
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass, field
from typing import List
import pytest

from modules.rebalance_executor import (
    RebalanceExecutor, RebalanceJob, RebalanceResult, JobState
)


class FakeRpcError(Exception):
    def __init__(self, command="waitsendpay", error=None):
        self.command = command
        self.error = error or {}
        super().__init__(f"{command}: {self.error}")


@dataclass
class MockCandidate:
    source_candidates: List[str] = field(default_factory=lambda: ["100x1x0"])
    primary_source_peer_id: str = "source_peer_abc"
    to_channel: str = "200x1x0"
    to_peer_id: str = "dest_peer_abc"
    amount_sats: int = 500000
    amount_msat: int = 500000000
    max_budget_sats: int = 100
    max_budget_msat: int = 100000
    max_fee_ppm: int = 200
    hive_route_hops: int = 0
    direction: str = "pull"
    reason_code: str = "ev_positive"
    dest_is_hive_member: bool = False


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
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc", "payment_secret": "def", "bolt11": "lnbc..."
        }
        plugin.rpc.listpeerchannels.return_value = {"channels": []}

        def call_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            return {}
            return {}

        plugin.rpc.call.side_effect = call_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=2)
        result = executor.execute(candidate)

        assert result.route_type == "fleet"
        assert result.success is False  # xpay fails for self-payment

    def test_network_when_no_hive_route(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc", "payment_secret": "def", "bolt11": "lnbc..."
        }

        def call_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            return {}
            return {}

        plugin.rpc.call.side_effect = call_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=0)
        result = executor.execute(candidate)
        assert result.route_type == "network"


class TestExecuteSuccess:
    def test_successful_single_part(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "our_id", "channel": "300x1x0", "amount_msat": 500000000, "delay": 18}
            ]
        }
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete", "amount_sent_msat": 500050000
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=0)
        result = executor.execute(candidate)

        assert result.success is True
        assert result.fee_msat == 50000
        assert result.fee_ppm == 100  # 50000 * 1e6 / 500000000
        assert result.attempts == 1
        assert result.route_type == "network"

    def test_getroute_uses_named_riskfactor_and_cltv(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "dest_peer_abc", "channel": "300x1x0", "direction": 1, "amount_msat": 500000000, "delay": 24}
            ]
        }
        plugin.rpc.listchannels.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete", "amount_sent_msat": 500000000
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate(hive_route_hops=0))

        assert result.success is True
        assert plugin.rpc.getroute.call_args.kwargs["riskfactor"] == 0
        assert plugin.rpc.getroute.call_args.kwargs["cltv"] == 24
        assert plugin.rpc.getroute.call_args.kwargs["fromid"] == "source_peer_abc"

    def test_first_hop_uses_first_return_hop_policy(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {
                    "id": "dest_peer_abc",
                    "channel": "300x1x0",
                    "direction": 1,
                    "amount_msat": 500_000_000,
                    "delay": 24,
                },
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "100x1x0",
                    "updates": {
                        "local": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 1,
                            "cltv_expiry_delta": 6,
                        },
                    },
                },
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 1,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 34,
                    "delay": 34,
                }
            ]
        }
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete", "amount_sent_msat": 500_018_000
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=0)

        result = executor.execute(candidate)

        assert result.success is True
        plugin.rpc.getroute.assert_called_once_with(
            "dest_peer_abc",
            500_000_000,
            riskfactor=0,
            cltv=24,
            fromid="source_peer_abc",
            maxhops=6,
            fuzzpercent=0,
        )
        sendpay_route = plugin.rpc.sendpay.call_args.kwargs["route"]
        assert sendpay_route[0]["channel"] == "100x1x0"
        assert sendpay_route[0]["amount_msat"] == 500_018_000
        assert sendpay_route[0]["delay"] == 58
        assert sendpay_route[-1]["channel"] == "200x1x0"
        assert sendpay_route[-1]["amount_msat"] == 500_000_000

    def test_fleet_execution_uses_getroutes_with_return_hop_fees(self):
        """Fleet routes use getroutes (askrene layers) with proper return hop fee accounting."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }

        def listpeerchannels_side_effect(peer_id=None):
            if peer_id == "dest_peer_abc":
                return {
                    "channels": [
                        {
                            "short_channel_id": "200x1x0",
                            "updates": {
                                "remote": {
                                    "fee_base_msat": 1000,
                                    "fee_proportional_millionths": 100,
                                    "cltv_expiry_delta": 12,
                                },
                            },
                        },
                    ]
                }
            return {"channels": []}

        plugin.rpc.listpeerchannels.side_effect = listpeerchannels_side_effect
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete", "amount_sent_msat": 500_051_000
        }

        def call_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": [{"layer": "hive-fleet"}, {"layer": "revenue-local"}]}
            if method == "getroutes":
                # Verify getroutes receives return hop fee-adjusted amount and CLTV
                assert params["amount_msat"] == 500_051_000  # 500M + 1000 base + (500M * 100 / 1M)
                assert params["final_cltv"] == 30  # 18 + 12 cltv_delta
                assert params["maxparts"] == 1
                assert "auto.no_mpp_support" in params["layers"]
                return {
                    "routes": [{
                        "amount_msat": 500_051_000,
                        "path": [
                            {
                                "short_channel_id_dir": "100x1x0/1",
                                "next_node_id": "fleet_mid",
                                "amount_msat": 500_051_000,
                                "delay": 42,
                            },
                            {
                                "short_channel_id_dir": "300x1x0/0",
                                "next_node_id": "dest_peer_abc",
                                "amount_msat": 500_051_000,
                                "delay": 30,
                            },
                        ],
                    }],
                    "probability_ppm": 999999,
                }
            if method == "askrene-inform-channel":
                return {}
            return {}

        plugin.rpc.call.side_effect = call_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        candidate = MockCandidate(hive_route_hops=2)

        result = executor.execute(candidate)

        assert result.success is True
        assert result.route_type == "fleet"
        # getroute should NOT be called for fleet routes
        plugin.rpc.getroute.assert_not_called()
        # getroutes should be called via plugin.rpc.call
        getroutes_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "getroutes"
        ]
        assert len(getroutes_calls) == 1
        sendpay_route = plugin.rpc.sendpay.call_args.kwargs["route"]
        # Final hop: dest_peer -> us via dest_channel
        assert sendpay_route[-1]["id"] == "our_id"
        assert sendpay_route[-1]["channel"] == "200x1x0"
        assert sendpay_route[-1]["amount_msat"] == 500_000_000
        assert sendpay_route[-1]["delay"] == 18


class TestExecuteFailure:
    def test_rejects_malformed_route_before_sendpay(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "100x1x0",
                    "updates": {"local": {"cltv_expiry_delta": 6}},
                },
                {
                    "short_channel_id": "200x1x0",
                    "updates": {"remote": {"cltv_expiry_delta": 6}},
                },
            ]
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 0,
                    "base_fee_millisatoshi": 0,
                    "fee_per_millionth": 0,
                    "delay": 18,
                }
            ]
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {
                    "id": "dest_peer_abc",
                    "channel": "300x1x0",
                    "direction": 0,
                    "amount_msat": 400_000_000,
                    "delay": 24,
                },
            ]
        }
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete",
            "amount_sent_msat": 400_000_000,
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate(hive_route_hops=0))

        assert result.success is False
        assert result.error == "sendpay_error: increasing_route_amount"
        plugin.rpc.sendpay.assert_not_called()

    def test_no_routes(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash", "payment_secret": "secret"
        }

        def rpc_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            return {}
            return {}

        plugin.rpc.call.side_effect = rpc_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate())

        assert result.success is False

    def test_cleans_up_invoice_on_failure(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash", "payment_secret": "secret", "bolt11": "lnbc..."
        }

        def call_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": []}
            return {}
            return {}

        plugin.rpc.call.side_effect = call_side_effect

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        executor.execute(MockCandidate())

        # delinvoice should be called for cleanup
        plugin.rpc.delinvoice.assert_called_once()


class TestHiveEqualizationRouteValidation:
    def test_equalization_rejects_non_hive_intermediate(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }

        def call_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": [{"layer": "hive-fleet"}, {"layer": "revenue-local"}]}
            if method == "getroutes":
                return {
                    "routes": [{
                        "amount_msat": 500_000_000,
                        "path": [
                            {
                                "short_channel_id_dir": "100x1x0/1",
                                "next_node_id": "non_hive_mid",
                                "amount_msat": 500_000_000,
                                "delay": 42,
                            },
                            {
                                "short_channel_id_dir": "300x1x0/0",
                                "next_node_id": "dest_peer_abc",
                                "amount_msat": 500_000_000,
                                "delay": 30,
                            },
                        ],
                    }],
                    "probability_ppm": 999999,
                }
            return {}

        plugin.rpc.call.side_effect = call_side_effect
        hive_router = MagicMock()
        hive_router.is_hive_member.side_effect = lambda pid: pid == "dest_peer_abc"
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock(), hive_router=hive_router)

        result = executor.execute(
            MockCandidate(
                hive_route_hops=2,
                reason_code="hive_equalization",
                dest_is_hive_member=True,
                max_budget_sats=0,
                max_budget_msat=0,
            )
        )

        assert result.success is False
        assert "non_pure_hive_route" in result.error
        plugin.rpc.getroute.assert_not_called()
        plugin.rpc.sendpay.assert_not_called()

    def test_equalization_accepts_all_hive_intermediates(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete", "amount_sent_msat": 500_000_000
        }

        def call_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": [{"layer": "hive-fleet"}, {"layer": "revenue-local"}]}
            if method == "getroutes":
                return {
                    "routes": [{
                        "amount_msat": 500_000_000,
                        "path": [
                            {
                                "short_channel_id_dir": "100x1x0/1",
                                "next_node_id": "hive_mid",
                                "amount_msat": 500_000_000,
                                "delay": 42,
                            },
                            {
                                "short_channel_id_dir": "300x1x0/0",
                                "next_node_id": "dest_peer_abc",
                                "amount_msat": 500_000_000,
                                "delay": 30,
                            },
                        ],
                    }],
                    "probability_ppm": 999999,
                }
            if method == "askrene-inform-channel":
                return {}
            return {}

        plugin.rpc.call.side_effect = call_side_effect
        hive_router = MagicMock()
        hive_router.is_hive_member.side_effect = (
            lambda pid: pid in {"hive_mid", "dest_peer_abc"}
        )
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock(), hive_router=hive_router)

        result = executor.execute(
            MockCandidate(
                hive_route_hops=2,
                reason_code="hive_equalization",
                dest_is_hive_member=True,
                max_budget_sats=0,
                max_budget_msat=0,
            )
        )

        assert result.success is True
        assert result.route_type == "fleet"
        plugin.rpc.getroute.assert_not_called()

    def test_retries_on_route_failure_with_exclude(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 1,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                },
                {
                    "short_channel_id": "301x1x0",
                    "direction": 0,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                }
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.getroute.side_effect = [
            {
                "route": [
                    {
                        "id": "dest_peer_abc",
                        "channel": "300x1x0",
                        "direction": 1,
                        "amount_msat": 500_000_000,
                        "delay": 24,
                    },
                ]
            },
            {
                "route": [
                    {
                        "id": "dest_peer_abc",
                        "channel": "301x1x0",
                        "direction": 0,
                        "amount_msat": 500_000_000,
                        "delay": 24,
                    },
                ]
            },
        ]
        plugin.rpc.waitsendpay.side_effect = [
            FakeRpcError(error={
                "code": 204,
                "message": "failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
                "data": {
                    "erring_index": 2,
                    "erring_channel": "940851x30x0",
                    "erring_direction": 0,
                    "erring_node": "0217890e3aad8d35bc054f43acc00084b25229ecff0ab68debd82883ad65ee8266",
                    "failcode": 4103,
                    "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                },
            }),
            {
                "status": "complete",
                "amount_sent_msat": 500_006_000,
            },
        ]

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate())

        assert result.success is True
        assert result.attempts == 2
        assert plugin.rpc.getroute.call_count == 2
        assert plugin.rpc.getroute.call_args_list[1].kwargs["exclude"] == ["940851x30x0/0"]
        plugin.rpc.delpay.assert_called_once_with("hash123", "failed")

        inform_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "askrene-inform-channel"
        ]
        assert not any(c[0][1]["inform"] == "failed" for c in inform_calls)
        assert any(c[0][1]["inform"] == "constrained" for c in inform_calls)
        assert any(c[0][1]["inform"] == "succeeded" for c in inform_calls)

    def test_fleet_retries_on_fee_insufficient_with_exclude(self):
        """Fleet attempt 1 uses getroutes, retry falls back to getroute + excludes."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "301x1x0",
                    "direction": 0,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                },
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }

        def call_side_effect(method, params=None):
            if method == "askrene-listlayers":
                return {"layers": [{"layer": "hive-fleet"}]}
            if method == "getroutes":
                return {
                    "routes": [{
                        "amount_msat": 500_000_000,
                        "path": [
                            {
                                "short_channel_id_dir": "300x1x0/1",
                                "next_node_id": "dest_peer_abc",
                                "amount_msat": 500_000_000,
                                "delay": 24,
                            },
                        ],
                    }],
                }
            if method == "askrene-inform-channel":
                return {}
            return {}

        plugin.rpc.call.side_effect = call_side_effect

        # Retry (attempt 2) falls back to getroute + excludes
        plugin.rpc.getroute.return_value = {
            "route": [
                {
                    "id": "dest_peer_abc",
                    "channel": "301x1x0",
                    "direction": 0,
                    "amount_msat": 500_000_000,
                    "delay": 24,
                },
            ]
        }
        plugin.rpc.waitsendpay.side_effect = [
            FakeRpcError(error={
                "code": 204,
                "message": "failed: WIRE_FEE_INSUFFICIENT",
                "data": {
                    "erring_index": 1,
                    "erring_channel": "300x1x0",
                    "erring_direction": 1,
                    "erring_node": "dest_peer_abc",
                    "failcode": 4108,
                    "failcodename": "WIRE_FEE_INSUFFICIENT",
                },
            }),
            {
                "status": "complete",
                "amount_sent_msat": 500_006_000,
            },
        ]

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate(hive_route_hops=2))

        assert result.success is True
        assert result.route_type == "fleet"
        assert result.attempts == 2
        # Attempt 1 used getroutes (fleet), attempt 2 used getroute (network fallback)
        assert plugin.rpc.getroute.call_count == 1
        assert plugin.rpc.getroute.call_args.kwargs["exclude"] == ["300x1x0/1"]

    def test_runtime_memory_excludes_banned_channel(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 1,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                }
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {
                    "id": "dest_peer_abc",
                    "channel": "300x1x0",
                    "direction": 1,
                    "amount_msat": 500_000_000,
                    "delay": 24,
                },
            ]
        }
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete",
            "amount_sent_msat": 500_006_000,
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        executor._routing_memory.ban_channel("940851x30x0/0", ttl_seconds=300)

        result = executor.execute(MockCandidate(hive_route_hops=0))

        assert result.success is True
        assert plugin.rpc.getroute.call_args.kwargs["exclude"] == ["940851x30x0/0"]

    def test_constraint_excludes_oversized_route_before_sendpay(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 1,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                },
                {
                    "short_channel_id": "301x1x0",
                    "direction": 0,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                },
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.getroute.side_effect = [
            {
                "route": [
                    {
                        "id": "dest_peer_abc",
                        "channel": "300x1x0",
                        "direction": 1,
                        "amount_msat": 500_000_000,
                        "delay": 24,
                    },
                ]
            },
            {
                "route": [
                    {
                        "id": "dest_peer_abc",
                        "channel": "301x1x0",
                        "direction": 0,
                        "amount_msat": 500_000_000,
                        "delay": 24,
                    },
                ]
            },
        ]
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete",
            "amount_sent_msat": 500_006_000,
        }

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        executor._routing_memory.constrain_channel("300x1x0/1", 400_000_000, ttl_seconds=300)

        result = executor.execute(MockCandidate(hive_route_hops=0))

        assert result.success is True
        assert plugin.rpc.getroute.call_count == 2
        assert plugin.rpc.getroute.call_args_list[1].kwargs["exclude"] == ["300x1x0/1"]
        plugin.rpc.sendpay.assert_called_once()

    def test_temp_channel_failure_learns_ban(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 1,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                },
                {
                    "short_channel_id": "301x1x0",
                    "direction": 0,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                }
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.getroute.side_effect = [
            {
                "route": [
                    {
                        "id": "dest_peer_abc",
                        "channel": "300x1x0",
                        "direction": 1,
                        "amount_msat": 500_000_000,
                        "delay": 24,
                    },
                ]
            },
            {
                "route": [
                    {
                        "id": "dest_peer_abc",
                        "channel": "301x1x0",
                        "direction": 0,
                        "amount_msat": 500_000_000,
                        "delay": 24,
                    },
                ]
            },
        ]
        plugin.rpc.waitsendpay.side_effect = [
            FakeRpcError(error={
                "code": 204,
                "message": "failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
                "data": {
                    "erring_index": 2,
                    "erring_channel": "940851x30x0",
                    "erring_direction": 0,
                    "failcode": 4103,
                    "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                },
            }),
            {
                "status": "complete",
                "amount_sent_msat": 500_006_000,
            },
        ]

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate(hive_route_hops=0))

        assert result.success is True
        assert "940851x30x0/0" in executor._routing_memory.current_excludes()

    def test_fee_insufficient_learns_ban(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 1,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                }
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {
                    "id": "dest_peer_abc",
                    "channel": "300x1x0",
                    "direction": 1,
                    "amount_msat": 500_000_000,
                    "delay": 24,
                },
            ]
        }
        plugin.rpc.waitsendpay.side_effect = FakeRpcError(error={
            "code": 204,
            "message": "failed: WIRE_FEE_INSUFFICIENT",
            "data": {
                "erring_index": 2,
                "erring_channel": "941153x2443x0",
                "erring_direction": 1,
                "failcode": 4108,
                "failcodename": "WIRE_FEE_INSUFFICIENT",
            },
        })

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        executor.MAX_ATTEMPTS = 1
        result = executor.execute(MockCandidate(hive_route_hops=0))

        assert result.success is False
        assert "941153x2443x0/1" in executor._routing_memory.current_excludes()

    def test_immediate_sendpay_failure_bans_first_hop(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "300x1x0",
                    "direction": 1,
                    "base_fee_millisatoshi": 1000,
                    "fee_per_millionth": 10,
                    "delay": 18,
                }
            ]
        }
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {
                    "id": "dest_peer_abc",
                    "channel": "300x1x0",
                    "direction": 1,
                    "amount_msat": 500_000_000,
                    "delay": 24,
                },
            ]
        }
        plugin.rpc.sendpay.side_effect = FakeRpcError(
            command="sendpay",
            error={"code": 205, "message": "failed: PAY_TRY_OTHER_ROUTE"},
        )

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())
        result = executor.execute(MockCandidate(hive_route_hops=0))

        assert result.success is False
        assert "100x1x0/0" in executor._routing_memory.current_excludes()

    def test_execute_reserves_and_unreserves_capacity(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_id"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "our_id", "channel": "300x1x0", "direction": 0, "amount_msat": 500000000, "delay": 18}
            ]
        }
        plugin.rpc.listchannels.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete", "amount_sent_msat": 500000000
        }
        hive_router = MagicMock()

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock(), hive_router=hive_router)
        candidate = MockCandidate(hive_route_hops=0)

        result = executor.execute(candidate)

        assert result.success is True
        hive_router.reserve_path.assert_called_once()
        hive_router.unreserve_path.assert_called_once()
        reserved_path = hive_router.reserve_path.call_args.args[0]
        unreserved_path = hive_router.unreserve_path.call_args.args[0]
        assert reserved_path == unreserved_path
        assert reserved_path == [
            {"short_channel_id_dir": "100x1x0/0", "amount_msat": 500000000},
            {"short_channel_id_dir": "300x1x0/0", "amount_msat": 500000000},
            {"short_channel_id_dir": "200x1x0/0", "amount_msat": 500000000},
        ]

    def test_unreserves_attempt_path_before_success_inform(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "bbbb"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "cccc", "channel": "300x1x0", "direction": 0, "amount_msat": 500000000, "delay": 24}
            ]
        }
        plugin.rpc.listchannels.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.waitsendpay.return_value = {
            "status": "complete", "amount_sent_msat": 500000000
        }
        events = []

        def call_side_effect(method, params=None):
            if method == "askrene-inform-channel":
                events.append(f"inform:{params['inform']}")
            return {}

        plugin.rpc.call.side_effect = call_side_effect
        hive_router = MagicMock()
        hive_router.reserve_path.side_effect = lambda path: events.append("reserve") or True
        hive_router.unreserve_path.side_effect = lambda path: events.append("unreserve") or True

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock(), hive_router=hive_router)
        candidate = MockCandidate(
            hive_route_hops=0,
            primary_source_peer_id="aaaa",
            to_peer_id="cccc",
        )

        result = executor.execute(candidate)

        assert result.success is True
        assert events[0] == "reserve"
        assert events[1] == "unreserve"
        assert all(event.startswith("inform:") for event in events[2:])

    def test_unreserves_attempt_path_before_failure_inform(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "bbbb"}
        plugin.rpc.invoice.return_value = {
            "payment_hash": "hash123", "payment_secret": "secret123",
            "bolt11": "lnbc5u1..."
        }
        plugin.rpc.getroute.return_value = {
            "route": [
                {"id": "cccc", "channel": "300x1x0", "direction": 0, "amount_msat": 500000000, "delay": 24}
            ]
        }
        plugin.rpc.listchannels.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "short_channel_id": "200x1x0",
                    "updates": {
                        "remote": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        },
                    },
                }
            ]
        }
        plugin.rpc.waitsendpay.side_effect = FakeRpcError(error={
            "code": 204,
            "message": "failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
            "data": {
                "erring_index": 2,
                "erring_channel": "300x1x0",
                "erring_direction": 0,
                "erring_node": "cccc",
                "failcode": 4103,
                "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
            },
        })
        events = []

        def call_side_effect(method, params=None):
            if method == "askrene-inform-channel":
                events.append(f"inform:{params['inform']}")
            return {}

        plugin.rpc.call.side_effect = call_side_effect
        hive_router = MagicMock()
        hive_router.reserve_path.side_effect = lambda path: events.append("reserve") or True
        hive_router.unreserve_path.side_effect = lambda path: events.append("unreserve") or True

        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock(), hive_router=hive_router)
        executor.MAX_ATTEMPTS = 1
        candidate = MockCandidate(
            hive_route_hops=0,
            primary_source_peer_id="aaaa",
            to_peer_id="cccc",
        )

        result = executor.execute(candidate)

        assert result.success is False
        assert events[0] == "reserve"
        assert events[1] == "unreserve"
        assert all(event.startswith("inform:") for event in events[2:])
        assert "inform:unconstrained" in events
        assert "inform:constrained" in events


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
             "amount_msat": 501000, "delay": 24},
        ]
        executor._inform_result(path, 500000, succeeded=True)

        inform_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "askrene-inform-channel"
        ]
        assert len(inform_calls) == 1
        assert inform_calls[0][0][1]["inform"] == "succeeded"
        assert inform_calls[0][0][1]["amount_msat"] == 501000

    def test_informs_on_failure_with_valid_askrene_semantics(self):
        plugin = MagicMock()
        plugin.rpc.call.return_value = {}
        executor = RebalanceExecutor(plugin, MagicMock(), MagicMock())

        path = [
            {"short_channel_id_dir": "100x1x0/0", "amount_msat": 501000},
            {"short_channel_id_dir": "200x1x0/1", "amount_msat": 500000},
            {"short_channel_id_dir": "300x1x0/0", "amount_msat": 500000},
        ]
        failure = {
            "code": 204,
            "erring_channel": "200x1x0",
            "erring_direction": 1,
            "erring_index": 2,
        }
        executor._inform_failure(path, failure)

        inform_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c[0][0] == "askrene-inform-channel"
        ]
        assert [c[0][1]["inform"] for c in inform_calls] == [
            "unconstrained",
            "constrained",
        ]
        assert inform_calls[0][0][1]["short_channel_id_dir"] == "100x1x0/0"
        assert inform_calls[0][0][1]["amount_msat"] == 501000
        assert inform_calls[1][0][1]["short_channel_id_dir"] == "200x1x0/1"
        assert inform_calls[1][0][1]["amount_msat"] == 500000
