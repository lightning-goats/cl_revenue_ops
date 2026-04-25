import importlib.util
import json
import sys
from pathlib import Path


def load_loop():
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "fee_tournament_loop.py"
    spec = importlib.util.spec_from_file_location("fee_tournament_loop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summary(**overrides):
    data = {
        "bursts": [
            {"revenue_fee_ppm": 75.0},
        ],
        "totals": {
            "payments_succeeded": 10,
            "payments_failed": 0,
            "payment_success_rate": 1.0,
            "forward_attribution_rate": 1.0,
            "quote_forward_divergent_bursts": 0,
            "revenue_forwards": 8,
            "competitor_forwards": 2,
        },
        "inferred_market_boundary_ppm": 70.0,
    }
    for key, value in overrides.items():
        if key == "totals":
            data["totals"].update(value)
        else:
            data[key] = value
    return data


def test_channel_local_balances_extracts_competing_channels():
    loop = load_loop()
    balances = loop.channel_local_balances(
        {
            "channels": [
                {"peer_alias": "revenue-node", "local_balance": "123"},
                {"peer_alias": "lnd-competitor-c", "local_balance": "456"},
                {"peer_alias": "other", "local_balance": "999"},
            ]
        }
    )

    assert balances == {"revenue-node": 123, "lnd-competitor-c": 456}


def test_classify_summary_refines_when_forward_attribution_is_low():
    loop = load_loop()

    decision = loop.classify_summary(summary(totals={"forward_attribution_rate": 0.5}))

    assert decision.action == "refine_tests"
    assert decision.valid_for_fee_performance is False
    assert decision.test_refinement_warranted is True
    assert decision.code_change_warranted is False


def test_classify_summary_flags_algorithm_candidate_when_competitor_wins_below_boundary():
    loop = load_loop()

    decision = loop.classify_summary(
        summary(
            totals={
                "revenue_forwards": 1,
                "competitor_forwards": 9,
            }
        )
    )

    assert decision.action == "consider_algorithm_change"
    assert decision.valid_for_fee_performance is True
    assert decision.code_change_warranted is True


def test_classify_summary_repeats_when_valid_without_algorithm_signal():
    loop = load_loop()

    decision = loop.classify_summary(summary())

    assert decision.action == "repeat_test"
    assert decision.valid_for_fee_performance is True
    assert decision.code_change_warranted is False


def test_estimate_plan_payment_volume():
    loop = load_loop()
    phase = loop.long_fee_tournament.PlannedPhase(
        scenario="fixed_market",
        name="phase",
        cycle=0,
        amount_sat=20_000,
        competitor_ppm=80,
        reset_mc=True,
        force_cycle_before=False,
        force_cycle_after=True,
    )

    assert loop.estimate_plan_payment_volume([phase], rounds=3) == 60_000


def test_build_args_threads_cl_hive_options(tmp_path):
    loop = load_loop()

    args = loop.build_args(
        out_dir=tmp_path,
        profile=loop.PROFILES["balanced"],
        amounts=[1_000],
        fixed_ppms=[80],
        scenarios=["fixed_market"],
        cycles=1,
        rounds_per_phase=1,
        include_sticky=False,
        seed=1,
        cycle_wait=0,
        plugin_path="/tmp/cl_revenue_ops/cl-revenue-ops.py",
        policy_settle_seconds=0,
        policy_verify_timeout_seconds=1,
        post_payment_settle_seconds=0,
        channel_id="277x1x0",
        with_cl_hive=True,
        cl_hive_host_path=Path("/host/cl-hive"),
        cl_hive_plugin_path="/tmp/cl_hive/cl-hive.py",
        cl_hive_id="test-hive",
        install_cl_hive_deps=True,
        skip_cl_hive_deploy=True,
        skip_cl_hive_start=False,
        skip_cl_hive_genesis=True,
    )

    assert args.with_cl_hive is True
    assert args.cl_hive_host_path == Path("/host/cl-hive")
    assert args.cl_hive_plugin_path == "/tmp/cl_hive/cl-hive.py"
    assert args.cl_hive_id == "test-hive"
    assert args.install_cl_hive_deps is True
    assert args.skip_cl_hive_deploy is True
    assert args.skip_cl_hive_genesis is True
    assert args.skip_disable_cl_hive is False


def test_collect_loop_preflight_prepares_enabled_hive(monkeypatch):
    loop = load_loop()
    ensure_calls = []

    def fake_ensure_cl_hive(**kwargs):
        ensure_calls.append(kwargs)
        return {"ok": True, "enabled": True}

    monkeypatch.setattr(loop.tournament, "ensure_cl_hive", fake_ensure_cl_hive)
    monkeypatch.setattr(loop, "collect_preflight", lambda with_cl_hive: {"cl_hive": {"ok": True}})

    args = loop.build_args(
        out_dir=Path("/tmp/out"),
        profile=loop.PROFILES["balanced"],
        amounts=[1_000],
        fixed_ppms=[80],
        scenarios=["fixed_market"],
        cycles=1,
        rounds_per_phase=1,
        include_sticky=False,
        seed=1,
        cycle_wait=0,
        plugin_path="/tmp/cl_revenue_ops/cl-revenue-ops.py",
        policy_settle_seconds=0,
        policy_verify_timeout_seconds=1,
        post_payment_settle_seconds=0,
        channel_id="277x1x0",
        with_cl_hive=True,
        cl_hive_host_path=Path("/host/cl-hive"),
        cl_hive_plugin_path="/tmp/cl_hive/cl-hive.py",
        cl_hive_id="test-hive",
    )

    preflight = loop.collect_loop_preflight(args)

    assert preflight["cl_hive_setup"] == {"ok": True, "enabled": True}
    assert ensure_calls[0]["hive_id"] == "test-hive"


def test_collect_loop_preflight_disables_hive_for_no_hive_runs(monkeypatch):
    loop = load_loop()
    disable_calls = []

    def fake_disable_cl_hive(**kwargs):
        disable_calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(loop.tournament, "disable_cl_hive", fake_disable_cl_hive)
    monkeypatch.setattr(loop, "collect_preflight", lambda with_cl_hive: {"cl_hive": {"skipped": True}})

    args = loop.build_args(
        out_dir=Path("/tmp/out"),
        profile=loop.PROFILES["balanced"],
        amounts=[1_000],
        fixed_ppms=[80],
        scenarios=["fixed_market"],
        cycles=1,
        rounds_per_phase=1,
        include_sticky=False,
        seed=1,
        cycle_wait=0,
        plugin_path="/tmp/cl_revenue_ops/cl-revenue-ops.py",
        policy_settle_seconds=0,
        policy_verify_timeout_seconds=1,
        post_payment_settle_seconds=0,
        channel_id="277x1x0",
        with_cl_hive=False,
        cl_hive_plugin_path="/tmp/cl_hive/cl-hive.py",
    )

    preflight = loop.collect_loop_preflight(args)

    assert preflight["cl_hive_setup"] == {"enabled": False, "disabled": {"ok": True}}
    assert disable_calls == [{"plugin_path": "/tmp/cl_hive/cl-hive.py"}]


def test_run_iteration_records_phase_runner_error(tmp_path, monkeypatch):
    loop = load_loop()

    def fail_run_plan(args, phases):
        raise RuntimeError(json.dumps({"stage": "policy_graph_verify"}))

    monkeypatch.setattr(loop.long_fee_tournament, "run_plan", fail_run_plan)

    result = loop.run_iteration(
        iteration_dir=tmp_path / "iteration",
        profile=loop.PROFILES["fee_sensitive"],
        amounts=[1_000],
        fixed_ppms=[60],
        scenarios=["fixed_market"],
        cycles=1,
        rounds_per_phase=1,
        include_sticky=False,
        seed=1,
        cycle_wait=0,
        plugin_path="/tmp/plugin.py",
        policy_settle_seconds=0,
        policy_verify_timeout_seconds=1,
        post_payment_settle_seconds=0,
        channel_id="277x1x0",
        execute=True,
    )

    assert result["decision"]["action"] == "refine_tests"
    assert Path(result["error_path"]).exists()
