"""Audit 2026-08-01 wave2 FIX 2 + FIX 3: treasury-cycle handling of boltz
status:"error" swaps and non-numeric cooldown_hours.

FIX 2: bm.loop_out() can return status "error" (an exit-0 payload boltzcli
reports as failed — BoltzManager._is_error_swap). The treasury executor must
mirror the balance cycle: restore the pre-claimed C1 cooldown slot, record an
execution_error skip, never consume budget/deficit — and an all-errored (or
all-rejected) treasury run must not suppress the F1 balance-cycle fallback
in _run_boltz_auto_cycle_once.

FIX 3 (P1-012 class): revenue-boltz-expansion-treasury-cycle must coerce a
non-numeric cooldown_hours instead of raising a raw exception out of the RPC
method, mirroring the balance-cycle twin.
"""

from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "b" * 64


def _plan(est_fee=100):
    return {
        "status": "ok",
        "generated_at": 1,
        "pending_swap_count": 0,
        "budget": {"remaining_24h_sats_estimate": 100_000},
        "treasury": {"deficit_sats": 900_000, "preferred_currency": "BTC"},
        "recommendations": [
            {
                "channel_id": "100x1x0",
                "peer_id": PEER,
                "direction": "loop_out",
                "amount_sats": 200_000,
                "quote": {"receiveAmount": 195_000},
                "economics": {
                    "passes_profit_guard": True,
                    "estimated_swap_fee_sats": est_fee,
                    "structural": False,
                },
            }
        ],
    }


def _treasury_module(loop_out_result):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    from modules.config import Config
    mod.config = Config()
    bm = MagicMock()
    bm.loop_out.return_value = loop_out_result
    mod._require_boltz_manager = MagicMock(return_value=bm)
    return mod, bm


# ---------------------------------------------------------------------------
# FIX 2: treasury executor treats "error" like "rejected"
# ---------------------------------------------------------------------------


def test_errored_swap_restores_cooldown_slot():
    """An errored swap moved no funds: the C1 pre-claimed cooldown slot must
    be restored so the channel is not blocked for cooldown_hours."""
    mod, bm = _treasury_module({"status": "error", "error": "swap failed"})

    result = mod._execute_boltz_expansion_treasury_cycle(
        dry_run=False, precomputed_plan=_plan()
    )

    assert "error" not in result
    # Pre-claim restored to the original (absent -> 0) timestamp.
    assert int(mod._boltz_balance_last_action.get("100x1x0", 0) or 0) == 0


def test_errored_swap_recorded_as_execution_error_skip():
    mod, bm = _treasury_module({"status": "error", "error": "swap failed"})

    result = mod._execute_boltz_expansion_treasury_cycle(
        dry_run=False, precomputed_plan=_plan()
    )

    reasons = [s.get("reason") for s in result["skipped"]]
    assert "execution_error" in reasons


def test_errored_swap_consumes_no_budget_or_deficit():
    mod, bm = _treasury_module({"status": "error", "error": "swap failed"})

    result = mod._execute_boltz_expansion_treasury_cycle(
        dry_run=False, precomputed_plan=_plan(est_fee=500)
    )

    assert result["remaining_budget_sats_estimate_after_cycle"] == 100_000
    assert result["remaining_treasury_deficit_sats_estimate_after_cycle"] == 900_000


def test_rejected_swap_still_restores_cooldown_slot():
    """Guard: the pre-existing rejected handling must survive the fix."""
    mod, bm = _treasury_module({"status": "rejected", "reason": "budget"})

    result = mod._execute_boltz_expansion_treasury_cycle(
        dry_run=False, precomputed_plan=_plan()
    )

    assert int(mod._boltz_balance_last_action.get("100x1x0", 0) or 0) == 0
    reasons = [s.get("reason") for s in result["skipped"]]
    assert "execution_rejected" in reasons


def test_accepted_swap_consumes_budget_and_holds_cooldown():
    """Guard: accepted swaps keep the pre-claimed cooldown slot and consume
    budget + deficit as before."""
    mod, bm = _treasury_module({"status": "accepted", "swap_id": "s1"})

    result = mod._execute_boltz_expansion_treasury_cycle(
        dry_run=False, precomputed_plan=_plan(est_fee=500)
    )

    assert result["executed"][0]["status"] == "accepted"
    assert result["remaining_budget_sats_estimate_after_cycle"] == 99_500
    assert result["remaining_treasury_deficit_sats_estimate_after_cycle"] == 705_000
    assert int(mod._boltz_balance_last_action.get("100x1x0", 0) or 0) > 0


# ---------------------------------------------------------------------------
# FIX 2 (F1 leg): an all-errored/rejected treasury run must not suppress the
# balance-cycle fallback for the whole interval
# ---------------------------------------------------------------------------


def _auto_cycle_module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.boltz_manager = MagicMock(enabled=True)
    mod.config = MagicMock()
    mod.config.snapshot.return_value = MagicMock(
        boltz_auto_cycle_enabled=True,
        boltz_auto_cycle_max_actions=1,
        expansion_treasury_enabled=True,
        expansion_treasury_onchain_target_sats=5_000_000,
        expansion_treasury_min_deficit_sats=250_000,
        expansion_treasury_preferred_currency="BTC",
        expansion_treasury_max_actions=1,
        expansion_treasury_min_source_local_pct=80.0,
        expansion_treasury_exclude_protected=True,
        paused=False,
    )
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    mod._build_boltz_expansion_treasury_plan = MagicMock(return_value={
        "status": "ok",
        "recommendations": [
            {"channel_id": "100x1x0",
             "economics": {"passes_profit_guard": True, "structural": False}},
        ],
        "treasury": {"deficit_sats": 900_000, "min_deficit_sats": 250_000},
    })
    balance_plan = {
        "pending_swap_count": 0,
        "budget": {"remaining_24h_sats_estimate": 100_000},
        "recommendations": [{"channel_id": "200x1x0"}],
    }
    mod._build_boltz_balance_plan = MagicMock(return_value=balance_plan)
    mod._execute_boltz_balance_cycle = MagicMock(return_value={
        "status": "executed",
        "executed_count": 1,
    })
    return mod


def test_all_errored_treasury_run_falls_through_to_balance():
    mod = _auto_cycle_module()
    mod._execute_boltz_expansion_treasury_cycle = MagicMock(return_value={
        "status": "executed",
        "executed_count": 1,  # the errored swap is still listed in executed
        "executed": [{"status": "error", "channel_id": "100x1x0"}],
        "skipped_count": 1,
        "skipped": [{"channel_id": "100x1x0", "reason": "execution_error"}],
    })

    result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

    mod._execute_boltz_balance_cycle.assert_called_once()
    assert result["selection_reason"] == "treasury_executed_zero_fallback_to_balance"


def test_all_rejected_treasury_run_falls_through_to_balance():
    mod = _auto_cycle_module()
    mod._execute_boltz_expansion_treasury_cycle = MagicMock(return_value={
        "status": "executed",
        "executed_count": 1,
        "executed": [{"status": "rejected", "channel_id": "100x1x0"}],
        "skipped_count": 1,
        "skipped": [{"channel_id": "100x1x0", "reason": "execution_rejected"}],
    })

    result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

    mod._execute_boltz_balance_cycle.assert_called_once()
    assert result["selection_reason"] == "treasury_executed_zero_fallback_to_balance"


def test_accepted_treasury_swap_still_suppresses_fallback():
    mod = _auto_cycle_module()
    mod._execute_boltz_expansion_treasury_cycle = MagicMock(return_value={
        "status": "executed",
        "executed_count": 1,
        "executed": [{"status": "accepted", "channel_id": "100x1x0"}],
        "skipped_count": 0,
    })

    mod._run_boltz_auto_cycle_once(trigger="scheduler")

    mod._execute_boltz_balance_cycle.assert_not_called()


def test_unknown_outcome_treasury_swap_still_suppresses_fallback():
    """A swap with an unknown outcome may be in flight — conservatively do
    NOT start another swap in the same interval."""
    mod = _auto_cycle_module()
    mod._execute_boltz_expansion_treasury_cycle = MagicMock(return_value={
        "status": "executed",
        "executed_count": 1,
        "executed": [{"status": "unknown", "channel_id": "100x1x0"}],
        "skipped_count": 0,
    })

    mod._run_boltz_auto_cycle_once(trigger="scheduler")

    mod._execute_boltz_balance_cycle.assert_not_called()


def test_legacy_executed_count_zero_still_falls_through():
    """Back-compat: a result without an 'executed' list (older shape) falls
    back to executed_count for the F1 decision."""
    mod = _auto_cycle_module()
    mod._execute_boltz_expansion_treasury_cycle = MagicMock(return_value={
        "status": "executed",
        "executed_count": 0,
        "skipped_count": 1,
    })

    result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

    mod._execute_boltz_balance_cycle.assert_called_once()
    assert result["selection_reason"] == "treasury_executed_zero_fallback_to_balance"


# ---------------------------------------------------------------------------
# FIX 3: non-numeric cooldown_hours must not raise out of the RPC method
# ---------------------------------------------------------------------------


def test_non_numeric_cooldown_hours_does_not_raise():
    mod, bm = _treasury_module({"status": "accepted", "swap_id": "s1"})
    mod._build_boltz_expansion_treasury_plan = MagicMock(return_value=_plan())

    result = mod.revenue_boltz_expansion_treasury_cycle(
        mod.plugin, dry_run=True, cooldown_hours="four"
    )

    assert isinstance(result, dict)
    assert "error" not in result
    assert result["status"] == "dry_run"
    assert result["executed"][0]["status"] == "would_execute"


def test_none_cooldown_hours_does_not_raise():
    mod, bm = _treasury_module({"status": "accepted", "swap_id": "s1"})
    mod._build_boltz_expansion_treasury_plan = MagicMock(return_value=_plan())

    result = mod.revenue_boltz_expansion_treasury_cycle(
        mod.plugin, dry_run=True, cooldown_hours=None
    )

    assert isinstance(result, dict)
    assert "error" not in result
    assert result["status"] == "dry_run"
