"""Tests for the v2 rebalance executor."""

from unittest.mock import MagicMock, patch

import pytest

from modules.rebalance_executor_v2 import RebalanceExecutorV2, V2ExecutionResult


def _make_executor():
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02" + "aa" * 32}
    executor = RebalanceExecutorV2(plugin=plugin)
    return executor, plugin


def _make_route(amount_msat=50_000_000, fee_msat=5_000):
    """Two-hop circular route: us → peer → us."""
    return [
        {
            "id": "02" + "bb" * 32,
            "channel": "111x1x0",
            "amount_msat": amount_msat + fee_msat,
            "delay": 40,
        },
        {
            "id": "02" + "aa" * 32,
            "channel": "222x2x0",
            "amount_msat": amount_msat,
            "delay": 10,
        },
    ]


class TestExecutorSuccess:
    def test_successful_rebalance(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }
        plugin.rpc.waitsendpay.return_value = {
            "amount_sent_msat": 50_005_000,
        }

        result = executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        assert result.success is True
        assert result.fee_sats == 5
        assert result.attempts == 1
        plugin.rpc.sendpay.assert_called_once()
        plugin.rpc.waitsendpay.assert_called_once()

    def test_cleanup_deletes_paid_invoice(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }
        plugin.rpc.waitsendpay.return_value = {
            "amount_sent_msat": 50_005_000,
        }

        executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        plugin.rpc.delinvoice.assert_called_once()
        # delpay should NOT be called on success
        plugin.rpc.delpay.assert_not_called()


class TestExecutorBudgetCheck:
    def test_rejects_route_over_budget(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        result = executor.execute(
            route=_make_route(fee_msat=20_000),  # 20 sat fee
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,  # budget only 10 sats
        )

        assert result.success is False
        assert "route_over_budget" in result.error
        plugin.rpc.sendpay.assert_not_called()

    def test_accepts_route_within_budget(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }
        plugin.rpc.waitsendpay.return_value = {
            "amount_sent_msat": 50_005_000,
        }

        result = executor.execute(
            route=_make_route(fee_msat=5_000),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        assert result.success is True


class TestExecutorFailures:
    def test_returns_error_on_no_node_id(self):
        plugin = MagicMock()
        plugin.rpc.getinfo.side_effect = Exception("connection refused")
        executor = RebalanceExecutorV2(plugin=plugin)

        result = executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        assert result.success is False
        assert result.error == "no_node_id"

    def test_returns_error_on_invoice_failure(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.side_effect = Exception("invoice creation failed")

        result = executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        assert result.success is False
        assert "invoice_error" in result.error

    def test_returns_error_on_sendpay_failure(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }
        plugin.rpc.sendpay.side_effect = Exception("sendpay failed")

        result = executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        assert result.success is False
        assert "sendpay_error" in result.error

    def test_permanent_failure_does_not_retry(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        class RPCError(Exception):
            def __init__(self):
                self.error = {
                    "data": {
                        "erring_channel": "333x3x0",
                        "failcodename": "WIRE_PERMANENT_CHANNEL_FAILURE",
                    }
                }

        plugin.rpc.waitsendpay.side_effect = RPCError()

        result = executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        assert result.success is False
        assert result.attempts == 1
        assert "permanent_failure" in result.error
        assert "333x3x0" in result.excluded_channels

    def test_retriable_failure_records_exclude(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        class RPCError(Exception):
            def __init__(self):
                self.error = {
                    "data": {
                        "erring_channel": "444x4x0",
                        "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                    }
                }

        plugin.rpc.waitsendpay.side_effect = RPCError()

        result = executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        assert result.success is False
        assert "444x4x0" in result.excluded_channels

    def test_cleanup_delpays_on_failure(self):
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }
        plugin.rpc.sendpay.side_effect = Exception("failed")

        executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        plugin.rpc.delpay.assert_called_once_with(
            payment_hash="abc123",
            status="failed",
        )
        plugin.rpc.delinvoice.assert_called_once()
