"""Audit 2026-08-01 wave2 FIX 4: revenue-spend-settle must reject a negative
actual_spent_sats at the RPC boundary. A negative spend event silently
inflates the remaining daily budget for every autonomous spender."""

from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


def _mod():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.database = MagicMock()
    mod.database.mark_spend_reservation_spent.return_value = True
    return mod


def test_negative_actual_spent_sats_rejected():
    mod = _mod()

    result = mod.revenue_spend_settle(
        mod.plugin, reservation_id="r1", actual_spent_sats=-500
    )

    assert "error" in result
    mod.database.mark_spend_reservation_spent.assert_not_called()


def test_non_numeric_actual_spent_sats_clean_error():
    mod = _mod()

    result = mod.revenue_spend_settle(
        mod.plugin, reservation_id="r1", actual_spent_sats="oops"
    )

    assert "error" in result
    mod.database.mark_spend_reservation_spent.assert_not_called()


def test_zero_actual_spent_sats_accepted():
    mod = _mod()

    result = mod.revenue_spend_settle(
        mod.plugin, reservation_id="r1", actual_spent_sats=0
    )

    assert result.get("status") == "success"
    kwargs = mod.database.mark_spend_reservation_spent.call_args.kwargs
    assert kwargs["actual_spent_sats"] == 0


def test_positive_actual_spent_sats_accepted():
    mod = _mod()

    result = mod.revenue_spend_settle(
        mod.plugin, reservation_id="r1", actual_spent_sats=750
    )

    assert result.get("status") == "success"
    kwargs = mod.database.mark_spend_reservation_spent.call_args.kwargs
    assert kwargs["actual_spent_sats"] == 750


def test_none_actual_spent_sats_passes_through():
    """None means 'settle at the reserved estimate' — must stay allowed."""
    mod = _mod()

    result = mod.revenue_spend_settle(mod.plugin, reservation_id="r1")

    assert result.get("status") == "success"
    kwargs = mod.database.mark_spend_reservation_spent.call_args.kwargs
    assert kwargs["actual_spent_sats"] is None
