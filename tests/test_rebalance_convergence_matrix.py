import importlib.util
import sys
from pathlib import Path


def load_matrix():
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "rebalance_convergence_matrix.py"
    spec = importlib.util.spec_from_file_location("rebalance_convergence_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_run_args_resets_default_hold_margin(tmp_path):
    matrix = load_matrix()

    args = matrix.build_run_args(
        loop_script=tmp_path / "loop.py",
        out_dir=tmp_path / "out",
        topology="square",
        hold_margin="default",
        iterations=3,
        deploy=True,
        restart_plugin=True,
        install_cl_hive_deps=True,
        restart_wait_seconds=5.0,
    )

    assert "--pure-hive-topology" in args
    assert "square" in args
    flag_index = args.index("--tune-rebalance-hold-margin")
    assert args[flag_index + 1] == "0.0"
    assert "--deploy" in args
    assert "--restart-plugin" in args
    assert "--install-cl-hive-deps" in args


def test_build_run_args_threads_numeric_hold_margin(tmp_path):
    matrix = load_matrix()

    args = matrix.build_run_args(
        loop_script=tmp_path / "loop.py",
        out_dir=tmp_path / "out",
        topology="square",
        hold_margin="0.25",
        iterations=3,
        deploy=False,
        restart_plugin=False,
        install_cl_hive_deps=False,
        restart_wait_seconds=2.0,
    )

    flag_index = args.index("--tune-rebalance-hold-margin")
    assert args[flag_index + 1] == "0.25"
    assert "--deploy" not in args
    assert "--restart-plugin" not in args


def test_summarize_loop_reports_convergence_pass(tmp_path):
    matrix = load_matrix()
    cycle_dir = tmp_path / "iteration_001"
    cycle_dir.mkdir()
    (cycle_dir / "cycle.json").write_text(
        """
        {
          "last_cycle": {
            "selected_candidates": [
              {"score_decomposition": {"final_score": 0.326119}}
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    summary = matrix.summarize_loop(
        {
            "next_action": "repeat",
            "next_reason": "ok",
            "analyses": [
                {
                    "valid": True,
                    "convergence_source_scid": "1x1x0",
                    "convergence_ok": True,
                    "selected_pairs": 1,
                    "executions": 1,
                    "successes": 1,
                    "fee_sats": 4,
                    "convergence_restored_sats": 280_000,
                    "convergence_max_error": 0.000003,
                },
                {
                    "valid": True,
                    "convergence_source_scid": "1x1x0",
                    "convergence_ok": True,
                    "selected_pairs": 1,
                    "executions": 1,
                    "successes": 1,
                    "fee_sats": 4,
                    "convergence_restored_sats": 280_000,
                    "convergence_max_error": 0.000004,
                },
            ],
        },
        topology="square",
        hold_margin="0.25",
        run_dir=tmp_path,
    )

    assert summary["pass"] is True
    assert summary["convergence_ok"] == 2
    assert summary["fee_per_restored_sat"] == 0.00001429
    assert summary["min_selected_final_score"] == 0.326119
    assert summary["min_score_headroom"] == 0.076119


def test_recommend_hold_margin_uses_guard_band_below_first_failure():
    matrix = load_matrix()

    recommendation = matrix.recommend_hold_margin([
        {"topology": "square", "hold_margin_value": 0.20, "pass": True},
        {"topology": "square", "hold_margin_value": 0.25, "pass": True},
        {"topology": "square", "hold_margin_value": 0.30, "pass": True},
        {"topology": "square", "hold_margin_value": 0.35, "pass": False},
    ])

    assert recommendation == 0.30


def test_recommend_hold_margin_uses_score_headroom_guard():
    matrix = load_matrix()

    recommendation = matrix.recommend_hold_margin([
        {
            "topology": "square",
            "hold_margin_value": 0.20,
            "pass": True,
            "min_selected_final_score": 0.326119,
        },
        {
            "topology": "square",
            "hold_margin_value": 0.25,
            "pass": True,
            "min_selected_final_score": 0.326119,
        },
        {
            "topology": "square",
            "hold_margin_value": 0.30,
            "pass": True,
            "min_selected_final_score": 0.326119,
        },
        {"topology": "square", "hold_margin_value": 0.35, "pass": False},
    ])

    assert recommendation == 0.25
