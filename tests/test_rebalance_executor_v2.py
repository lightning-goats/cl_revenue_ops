"""Tests for the v2 rebalance executor."""

from unittest.mock import MagicMock, patch

import pytest

from modules.rebalance_executor_v2 import RebalanceExecutor, ExecutionResult


def _make_executor():
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02" + "aa" * 32}
    executor = RebalanceExecutor(plugin=plugin)
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
        executor = RebalanceExecutor(plugin=plugin)

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
                        "erring_direction": 1,
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
        # Must be in directional form ``scid/dir`` so getroute and askrene
        # accept it in their exclude parameters.
        assert result.excluded_channels == ["444x4x0/1"]

    def test_exclude_captures_directional_form_when_direction_present(self):
        """Regression guard: live nexus-01 2026-04-10 18:54Z showed getroute
        rejecting bare SCIDs in the exclude list with
        'exclude: should be short_channel_id_dir or node_id'. The executor
        must combine erring_channel + erring_direction into CLN's canonical
        ``scid/dir`` format."""
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        class RPCError(Exception):
            def __init__(self):
                self.error = {
                    "data": {
                        "erring_channel": "934667x311x8",
                        "erring_direction": 0,
                        "failcodename": "WIRE_FEE_INSUFFICIENT",
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

        assert result.excluded_channels == ["934667x311x8/0"]

    def test_exclude_falls_back_to_bare_scid_when_direction_missing(self):
        """Some older CLN versions or wrapped failures may omit
        erring_direction. The executor should still produce a non-empty
        excluded_channels entry rather than dropping the hint entirely."""
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        class RPCError(Exception):
            def __init__(self):
                self.error = {
                    "data": {
                        "erring_channel": "555x5x5",
                        # No erring_direction
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

        assert result.excluded_channels == ["555x5x5"]

    def test_diagnostic_log_fires_on_bare_exception(self):
        """When waitsendpay raises a plain Exception with no .error attr,
        the executor must emit EXECUTOR_FAIL_DIAG so operators can still
        diagnose what happened. Regression guard for the Phase B Task 3
        observation on nexus-01 2026-04-10 where 'Failed:  erring_channel=None'
        masked the actual failure mode."""
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        plugin.rpc.waitsendpay.side_effect = Exception("socket timeout")

        executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        log_calls = [c.args[0] for c in plugin.log.call_args_list]
        diag = [l for l in log_calls if "EXECUTOR_FAIL_DIAG" in l]
        assert diag, f"expected EXECUTOR_FAIL_DIAG log, got: {log_calls}"
        assert "has_error_attr=no" in diag[0]
        assert "socket timeout" in diag[0]

    def test_diagnostic_log_fires_on_rpc_error_with_empty_data(self):
        """When waitsendpay raises an RpcError whose .error dict has no
        'data' field (or an empty dict), EXECUTOR_FAIL_DIAG must show the
        actual keys present so operators can see what CLN returned."""
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        class RPCError(Exception):
            def __init__(self):
                super().__init__("payment failed")
                self.error = {
                    "code": 203,
                    "message": "Ran out of routes to try",
                    # No 'data' field on purpose
                }

        plugin.rpc.waitsendpay.side_effect = RPCError()

        executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        log_calls = [c.args[0] for c in plugin.log.call_args_list]
        diag = [l for l in log_calls if "EXECUTOR_FAIL_DIAG" in l]
        assert diag, f"expected EXECUTOR_FAIL_DIAG log, got: {log_calls}"
        assert "err_code=203" in diag[0]
        assert "Ran out of routes to try" in diag[0]
        assert "data_keys=None" in diag[0]

    def test_diagnostic_log_shows_data_keys_when_present(self):
        """Happy-path diagnostic: when data is a proper dict, its keys
        appear in the log line."""
        executor, plugin = _make_executor()
        plugin.rpc.invoice.return_value = {
            "payment_hash": "abc123",
            "bolt11": "lnbc...",
        }

        class RPCError(Exception):
            def __init__(self):
                self.error = {
                    "code": 203,
                    "message": "Failed",
                    "data": {
                        "erring_channel": "555x5x0",
                        "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                        "erring_index": 1,
                    },
                }

        plugin.rpc.waitsendpay.side_effect = RPCError()

        executor.execute(
            route=_make_route(),
            amount_sats=50_000,
            source_channel_id="111x1x0",
            dest_channel_id="222x2x0",
            max_fee_sats=10,
        )

        log_calls = [c.args[0] for c in plugin.log.call_args_list]
        diag = [l for l in log_calls if "EXECUTOR_FAIL_DIAG" in l]
        assert diag, f"expected EXECUTOR_FAIL_DIAG log, got: {log_calls}"
        line = diag[0]
        assert "erring_channel" in line
        assert "failcodename" in line
        assert "erring_index" in line

    def test_extract_error_data_only_accepts_dict_data_field(self):
        """Regression guard: _extract_error_data used to return err.get('data', {})
        without type-checking, so a string data field would crash the
        downstream .get() call. Now returns {} for any non-dict data."""
        executor, _ = _make_executor()

        class RPCErrStringData(Exception):
            def __init__(self):
                self.error = {"data": "a string, not a dict"}

        result = executor._extract_error_data(RPCErrStringData())
        assert result == {}

        class RPCErrNoData(Exception):
            def __init__(self):
                self.error = {"code": 123}

        result = executor._extract_error_data(RPCErrNoData())
        assert result == {}

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
