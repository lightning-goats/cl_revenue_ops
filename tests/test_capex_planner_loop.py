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

    results = loop.run_loop(min_annual_roi_pct=1.0)

    assert len(results) == 6
    assert all(item.mode == "disabled" for item in results)
    assert all(item.passed for item in results)
    assert {item.name for item in results} == {
        "low_yield_large_open_rejected",
        "legacy_zero_hurdle_accepts_absolute_profit",
        "new_peer_bootstrap_respects_default_hurdle",
        "high_onchain_fee_rejected",
        "cycle_skips_low_yield_candidate",
        "cycle_opens_hurdle_clearing_candidate",
    }


