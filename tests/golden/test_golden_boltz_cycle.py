"""Golden: Boltz auto-cycle mode selection and dry-run plan structure.

Setup mirrors tests/test_boltz_auto_cycle_dry_run.py (module-level entry
point `_run_boltz_auto_cycle_once`). Dry-run only — no execution paths
are reachable (AGENTS.md action-RPC rule). Volatile fields are stripped
before goldening.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module
from tests.golden.util import golden_check

PEER = "02" + "c" * 64
VOLATILE_KEYS = {"timestamp", "ts", "duration_ms", "elapsed", "now",
                 "checked_at", "updated_at", "generated_at", "started_at",
                 "finished_at"}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _executable_plan():
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


def _make_module(auto_cycle_enabled=True, balance_plan=None):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.boltz_manager = MagicMock(enabled=True)
    mod.capacity_planner = None
    mod.rebalancer = None
    mod.config = MagicMock()
    mod.config.snapshot.return_value = SimpleNamespace(
        boltz_auto_cycle_enabled=auto_cycle_enabled,
        boltz_auto_cycle_max_actions=1,
        expansion_treasury_enabled=False,
        expansion_treasury_max_actions=1,
        boltz_structural_budget_sats_per_day=0,
    )
    mod._build_boltz_balance_plan = MagicMock(
        return_value=balance_plan if balance_plan is not None
        else _executable_plan())
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    bm = MagicMock()
    mod._require_boltz_manager = MagicMock(return_value=bm)
    return mod, bm


def test_golden_dry_run_executable_balance_plan():
    mod, bm = _make_module()
    result = mod._run_boltz_auto_cycle_once(trigger="golden", dry_run=True)
    golden_check("boltz/cycle_dry_run_executable_balance_plan",
                 _strip_volatile(result))
    bm.loop_out.assert_not_called()
    bm.loop_in.assert_not_called()


def test_golden_dry_run_disabled_cycle():
    mod, bm = _make_module(auto_cycle_enabled=False)
    result = mod._run_boltz_auto_cycle_once(
        trigger="golden", force=False, dry_run=True)
    golden_check("boltz/cycle_disabled", _strip_volatile(result))
    bm.loop_out.assert_not_called()


def test_golden_dry_run_idle_no_recommendations():
    idle_plan = dict(_executable_plan(), recommendations=[],
                     total_candidates=0)
    mod, bm = _make_module(balance_plan=idle_plan)
    result = mod._run_boltz_auto_cycle_once(trigger="golden", dry_run=True)
    golden_check("boltz/cycle_idle_no_recommendations",
                 _strip_volatile(result))
    bm.loop_out.assert_not_called()


MODE_CASES = {
    "treasury_wins_when_executable": dict(
        treasury_plan={
            "status": "ok",
            "treasury": {"deficit_sats": 400_000},
            "recommendations": [
                {"economics": {"passes_profit_guard": True}},
            ],
        },
        balance_plan=_executable_plan(),
    ),
    "guard_failed_treasury_yields_to_balance": dict(
        treasury_plan={
            "status": "ok",
            "treasury": {"deficit_sats": 400_000},
            "recommendations": [
                {"economics": {"passes_profit_guard": False}},
            ],
        },
        balance_plan=_executable_plan(),
    ),
    "structural_credit_rec_not_treasury_executable": dict(
        treasury_plan={
            "status": "ok",
            "treasury": {"deficit_sats": 400_000},
            "recommendations": [
                {"economics": {"passes_profit_guard": True,
                               "structural": True}},
            ],
        },
        balance_plan=_executable_plan(),
    ),
    "idle_when_nothing_eligible": dict(
        treasury_plan={"status": "ok", "treasury": {},
                       "recommendations": []},
        balance_plan={"recommendations": []},
    ),
}


def test_golden_mode_selection():
    mod = load_plugin_module()
    for name, kwargs in sorted(MODE_CASES.items()):
        mode = mod._select_boltz_auto_cycle_mode(**kwargs)
        golden_check(f"boltz/mode_{name}", mode)


def test_mode_selector_hand_computed_anchor():
    """Non-golden anchor: executable treasury rec beats balance recs."""
    mod = load_plugin_module()
    mode = mod._select_boltz_auto_cycle_mode(
        **MODE_CASES["treasury_wins_when_executable"])
    assert mode["mode"] == "treasury"
    mode = mod._select_boltz_auto_cycle_mode(
        **MODE_CASES["idle_when_nothing_eligible"])
    assert mode["mode"] == "idle"
