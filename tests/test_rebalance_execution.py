"""Tests for shared rebalance execution contracts."""

from modules.rebalance_execution import ExecutionResult, stable_failure_reason


def test_execution_result_defaults_to_native_route_type():
    result = ExecutionResult()

    assert result.route_type == "native"
    assert result.excluded_channels == []
    assert result.failure_data == {}


def test_route_over_budget_maps_to_route_segment_exhausted():
    assert (
        stable_failure_reason("native_route_over_budget: route_over_budget: 10 > 5")
        == "route_segment_exhausted"
    )


def test_native_route_invalid_maps_to_local_policy_block():
    assert stable_failure_reason("native_route_invalid: missing_route") == "local_policy_block"


def test_timeout_maps_to_executor_timeout():
    assert stable_failure_reason("native_waitsendpay_timeout") == "executor_timeout"


def test_temporary_channel_failure_maps_to_shared_conflict_changed():
    assert (
        stable_failure_reason("native_sendpay_error: WIRE_TEMPORARY_CHANNEL_FAILURE")
        == "shared_conflict_changed"
    )


def test_incorrect_cltv_expiry_maps_to_shared_conflict_changed():
    assert (
        stable_failure_reason("native_sendpay_error: WIRE_INCORRECT_CLTV_EXPIRY")
        == "shared_conflict_changed"
    )
