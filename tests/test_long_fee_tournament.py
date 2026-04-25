import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_long_tournament():
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "long_fee_tournament.py"
    spec = importlib.util.spec_from_file_location("long_fee_tournament", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_int_list():
    tool = load_long_tournament()

    assert tool.parse_int_list("1000, 20000,500000") == [1000, 20000, 500000]


def test_fixed_market_plan_includes_clean_and_sticky_phases():
    tool = load_long_tournament()

    phases = tool.build_plan(
        scenarios=["fixed_market"],
        amounts=[1000],
        cycles=1,
        fixed_ppms=[80, 150],
        include_sticky=True,
        competitor_controller="scripted",
        seed=1,
    )

    assert len(phases) == 4
    assert {phase.competitor_ppm for phase in phases} == {80, 150}
    assert sum(1 for phase in phases if phase.reset_mc) == 2
    assert sum(1 for phase in phases if not phase.reset_mc) == 2
    assert all(phase.scenario == "fixed_market" for phase in phases)


def test_external_competitor_plan_does_not_script_policy():
    tool = load_long_tournament()

    phases = tool.build_plan(
        scenarios=["clboss_external"],
        amounts=[1000, 20000],
        cycles=2,
        fixed_ppms=[80],
        include_sticky=False,
        competitor_controller="clboss",
        seed=2,
    )

    assert len(phases) == 4
    assert all(phase.competitor_controller == "clboss" for phase in phases)
    assert all(phase.competitor_ppm is None for phase in phases)
    assert all(phase.reset_mc for phase in phases)


def test_clboss_scenario_implies_clboss_controller():
    tool = load_long_tournament()

    phases = tool.build_plan(
        scenarios=["clboss_external"],
        amounts=[20_000],
        cycles=1,
        fixed_ppms=[80],
        include_sticky=False,
        competitor_controller="scripted",
        seed=3,
    )

    assert len(phases) == 1
    assert phases[0].scenario == "clboss_external"
    assert phases[0].competitor_controller == "clboss"


def test_resolve_adaptive_competitor_ppm_undercuts_revenue_fee(monkeypatch):
    tool = load_long_tournament()
    phase = tool.PlannedPhase(
        scenario="adaptive_competitor",
        name="adaptive",
        cycle=0,
        amount_sat=20_000,
        competitor_ppm=None,
        reset_mc=True,
        force_cycle_before=False,
        force_cycle_after=True,
    )

    monkeypatch.setattr(tool, "current_revenue_fee_ppm", lambda channel_id: 112)

    assert (
        tool.resolve_competitor_ppm(
            phase,
            channel_id="277x1x0",
            adaptive_undercut_ppm=5,
            adaptive_min_ppm=1,
            adaptive_max_ppm=1000,
            fallback_ppm=150,
        )
        == 107
    )


def test_resolve_adaptive_competitor_ppm_uses_bounds_and_fallback(monkeypatch):
    tool = load_long_tournament()
    phase = tool.PlannedPhase(
        scenario="adaptive_competitor",
        name="adaptive",
        cycle=0,
        amount_sat=20_000,
        competitor_ppm=None,
        reset_mc=True,
        force_cycle_before=False,
        force_cycle_after=True,
    )

    monkeypatch.setattr(tool, "current_revenue_fee_ppm", lambda channel_id: None)

    assert (
        tool.resolve_competitor_ppm(
            phase,
            channel_id="277x1x0",
            adaptive_undercut_ppm=500,
            adaptive_min_ppm=10,
            adaptive_max_ppm=1000,
            fallback_ppm=150,
        )
        == 10
    )


def test_run_plan_bootstraps_cl_hive_when_enabled(tmp_path, monkeypatch):
    tool = load_long_tournament()
    phase = tool.PlannedPhase(
        scenario="fixed_market",
        name="fixed",
        cycle=0,
        amount_sat=20_000,
        competitor_ppm=80,
        reset_mc=True,
        force_cycle_before=False,
        force_cycle_after=True,
    )
    ensure_calls = []
    run_phase_calls = []

    def fake_ensure_cl_hive(**kwargs):
        ensure_calls.append(kwargs)
        return {"ok": True, "enabled": True}

    def fake_run_phase(**kwargs):
        run_phase_calls.append(kwargs)
        return {"name": kwargs["name"]}

    monkeypatch.setattr(tool.tournament, "ensure_cl_hive", fake_ensure_cl_hive)
    monkeypatch.setattr(tool.tournament, "run_phase", fake_run_phase)

    args = SimpleNamespace(
        out_dir=tmp_path,
        rounds_per_phase=1,
        cycle_wait=0,
        policy_settle_seconds=0,
        policy_verify_timeout_seconds=1,
        post_payment_settle_seconds=0,
        channel_id="277x1x0",
        competitor_controller="scripted",
        competitor_cltv_delta=40,
        payer_time_pref=0,
        seed=1,
        clboss_nodes=[],
        adaptive_undercut_ppm=5,
        adaptive_min_ppm=1,
        adaptive_max_ppm=1000,
        adaptive_fallback_ppm=150,
        plugin_path="/tmp/cl_revenue_ops/cl-revenue-ops.py",
        with_cl_hive=True,
        cl_hive_host_path=Path("/host/cl-hive"),
        cl_hive_plugin_path="/tmp/cl_hive/cl-hive.py",
        cl_hive_id="test-hive",
        skip_cl_hive_deploy=False,
        skip_cl_hive_start=False,
        skip_cl_hive_genesis=False,
    )

    result = tool.run_plan(args, [phase])

    assert result["cl_hive_setup"] == {"ok": True, "enabled": True}
    assert ensure_calls[0]["hive_id"] == "test-hive"
    assert run_phase_calls[0]["with_cl_hive"] is True


def test_run_plan_disables_cl_hive_when_not_enabled(tmp_path, monkeypatch):
    tool = load_long_tournament()
    phase = tool.PlannedPhase(
        scenario="fixed_market",
        name="fixed",
        cycle=0,
        amount_sat=20_000,
        competitor_ppm=80,
        reset_mc=True,
        force_cycle_before=False,
        force_cycle_after=True,
    )
    disable_calls = []
    run_phase_calls = []

    def fake_disable_cl_hive(**kwargs):
        disable_calls.append(kwargs)
        return {"ok": True}

    def fake_run_phase(**kwargs):
        run_phase_calls.append(kwargs)
        return {"name": kwargs["name"]}

    monkeypatch.setattr(tool.tournament, "disable_cl_hive", fake_disable_cl_hive)
    monkeypatch.setattr(tool.tournament, "run_phase", fake_run_phase)

    args = SimpleNamespace(
        out_dir=tmp_path,
        rounds_per_phase=1,
        cycle_wait=0,
        policy_settle_seconds=0,
        policy_verify_timeout_seconds=1,
        post_payment_settle_seconds=0,
        channel_id="277x1x0",
        competitor_controller="scripted",
        competitor_cltv_delta=40,
        payer_time_pref=0,
        seed=1,
        clboss_nodes=[],
        adaptive_undercut_ppm=5,
        adaptive_min_ppm=1,
        adaptive_max_ppm=1000,
        adaptive_fallback_ppm=150,
        plugin_path="/tmp/cl_revenue_ops/cl-revenue-ops.py",
        with_cl_hive=False,
        cl_hive_plugin_path="/tmp/cl_hive/cl-hive.py",
        skip_disable_cl_hive=False,
    )

    result = tool.run_plan(args, [phase])

    assert result["cl_hive_setup"] == {"enabled": False, "disabled": {"ok": True}}
    assert disable_calls == [{"plugin_path": "/tmp/cl_hive/cl-hive.py"}]
    assert run_phase_calls[0]["with_cl_hive"] is False
