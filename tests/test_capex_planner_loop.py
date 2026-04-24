import importlib.util
import sys
from pathlib import Path


def load_loop():
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "capex_planner_loop.py"
    spec = importlib.util.spec_from_file_location("capex_planner_loop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_disabled_loop_keeps_baseline_capex_checks():
    loop = load_loop()

    results = loop.run_loop(min_annual_roi_pct=1.0, hive_mode="disabled")

    assert len(results) == 6
    assert all(item.mode == "disabled" for item in results)
    assert all(item.passed for item in results)
    assert {item.name for item in results} == {
        "low_yield_large_open_rejected",
        "legacy_zero_hurdle_accepts_absolute_profit",
        "new_peer_fallback_clears_default_hurdle",
        "high_onchain_fee_rejected",
        "cycle_skips_low_yield_candidate",
        "cycle_opens_hurdle_clearing_candidate",
    }


def test_ab_loop_exercises_hive_discovery_ev_and_capex_budget():
    loop = load_loop()

    results = loop.run_loop(min_annual_roi_pct=1.0, hive_mode="ab")
    by_name = {item.name: item for item in results}

    assert all(item.passed for item in results)
    assert by_name["hive_open_hint_adds_candidate"].opens == 1
    assert by_name["hive_score_bias_prioritizes_stronger_peer"].action == "boost"
    assert by_name["hive_rebalance_bias_clears_marginal_roi_hurdle"].ev_sats > 0
    assert "disabled_ev=" in by_name["hive_rebalance_bias_clears_marginal_roi_hurdle"].reason
    assert by_name["hive_bias_changes_cycle_decision"].opens == 1
    assert by_name["hive_member_gets_fleet_capex_budget"].action == "fleet"
