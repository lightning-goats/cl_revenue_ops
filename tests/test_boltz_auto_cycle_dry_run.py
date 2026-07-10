"""DD4 / P1-018: revenue-boltz-auto-cycle-run-now exposes a dry_run option that
defaults to the SAFE value (preview). force still bypasses the enabled toggle;
a live cycle while disabled is force=true + dry_run=false. The scheduler path
(_run_boltz_auto_cycle_once default) stays live so the daemon keeps executing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "c" * 64


def _executable_plan():
    """A plan whose single recommendation passes the profit guard, so it would
    execute a swap unless dry_run suppresses it."""
    return {
        "generated_at": 1,
        "pending_swap_count": 0,
        "budget": {"remaining_24h_sats_estimate": 1_000_000},
        "recommendations": [
            {
                "channel_id": "100x1x0",
                "peer_id": PEER,
                "direction": "loop_out",
                "amount_sats": 200_000,
                "economics": {
                    "passes_profit_guard": True,
                    "estimated_swap_fee_sats": 100,
                },
            }
        ],
        "total_candidates": 1,
        "skipped_count": 0,
        "skipped_examples": [],
    }


def _make_module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.boltz_manager = MagicMock(enabled=True)
    mod.capacity_planner = None
    mod.rebalancer = None
    mod.config = MagicMock()
    mod.config.snapshot.return_value = SimpleNamespace(
        boltz_auto_cycle_enabled=True,
        boltz_auto_cycle_max_actions=1,
        expansion_treasury_enabled=False,
        expansion_treasury_max_actions=1,
        boltz_structural_budget_sats_per_day=0,
    )
    mod._build_boltz_balance_plan = MagicMock(return_value=_executable_plan())
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    return mod


def test_dry_run_previews_without_executing():
    mod = _make_module()
    bm = MagicMock()
    mod._require_boltz_manager = MagicMock(return_value=bm)

    result = mod._run_boltz_auto_cycle_once(trigger="manual", dry_run=True)

    assert result["status"] == "dry_run"
    assert result["executed"][0]["status"] == "would_execute"
    # No swap subprocess was invoked.
    bm.loop_out.assert_not_called()
    bm.loop_in.assert_not_called()


def test_live_run_executes_swap():
    mod = _make_module()
    bm = MagicMock()
    bm.loop_out.return_value = {"status": "created", "swap_id": "s1"}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    mod._select_boltz_currency = MagicMock(return_value="LBTC")

    result = mod._run_boltz_auto_cycle_once(trigger="manual", dry_run=False)

    assert result["status"] == "executed"
    bm.loop_out.assert_called_once()


def test_rpc_defaults_to_live_run():
    # DD4 operator ruling 2026-07-02: dry_run defaults False; force=true alone
    # runs live (prior "run one now" semantics). Preview is opt-in (dry_run=true).
    mod = load_plugin_module()
    mod._run_boltz_auto_cycle_once = MagicMock(return_value={"status": "executed"})
    mod._boltz_auto_cycle_mark_state = MagicMock()

    mod.revenue_boltz_auto_cycle_run_now(mod.plugin)

    _, kwargs = mod._run_boltz_auto_cycle_once.call_args
    assert kwargs["dry_run"] is False
    assert kwargs["force"] is False


def test_rpc_dry_run_opt_in_previews():
    mod = load_plugin_module()
    mod._run_boltz_auto_cycle_once = MagicMock(return_value={"status": "dry_run"})
    mod._boltz_auto_cycle_mark_state = MagicMock()

    mod.revenue_boltz_auto_cycle_run_now(mod.plugin, dry_run=True)

    _, kwargs = mod._run_boltz_auto_cycle_once.call_args
    assert kwargs["dry_run"] is True


def test_rpc_force_live_when_explicitly_requested():
    mod = load_plugin_module()
    mod._run_boltz_auto_cycle_once = MagicMock(return_value={"status": "executed"})
    mod._boltz_auto_cycle_mark_state = MagicMock()

    mod.revenue_boltz_auto_cycle_run_now(mod.plugin, force=True, dry_run=False)

    _, kwargs = mod._run_boltz_auto_cycle_once.call_args
    assert kwargs["force"] is True
    assert kwargs["dry_run"] is False


def test_scheduler_default_is_live():
    """The daemon calls _run_boltz_auto_cycle_once(trigger='scheduler') with no
    dry_run, so it must default to live execution."""
    mod = _make_module()
    bm = MagicMock()
    bm.loop_out.return_value = {"status": "created", "swap_id": "s2"}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    mod._select_boltz_currency = MagicMock(return_value="LBTC")

    result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

    assert result["status"] == "executed"
    bm.loop_out.assert_called_once()
