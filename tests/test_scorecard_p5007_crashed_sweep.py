"""P5-007: a crashed sweep must report ERROR, never a PASS/KNOWN verdict.

Several sweeps exit 1 to mean "violations found", so exit code alone cannot
tell a real finding from an uncaught exception (also exit 1) that leaves
partial output. A sweep counts as clean only when it exits 0/1, emits its
end-of-run marker, and prints no traceback.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCORECARD = os.path.join(_REPO, "tools", "audit", "scorecard.py")


def _load_scorecard():
    spec = importlib.util.spec_from_file_location("scorecard_p5007", _SCORECARD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_scorecard()


def test_clean_completion_recognised():
    ok, reason = sc.sweep_completed_cleanly(
        "sweep_fee_stack", 1, "SWEEP RESULTS\n  FC-I1a: 3", "")
    assert ok is True
    assert reason == ""


def test_traceback_in_output_is_a_crash_even_with_marker():
    out = "SWEEP RESULTS\npartial\nTraceback (most recent call last):\n ...\nBoom"
    ok, reason = sc.sweep_completed_cleanly("sweep_fee_stack", 1, out, "")
    assert ok is False
    assert "crash" in reason.lower()


def test_traceback_in_stderr_is_a_crash():
    err = "Traceback (most recent call last):\nRuntimeError: boom"
    ok, reason = sc.sweep_completed_cleanly("sweep_fee_stack", 1, "SWEEP RESULTS", err)
    assert ok is False


def test_missing_marker_is_a_crash():
    ok, reason = sc.sweep_completed_cleanly(
        "sweep_fee_stack", 1, "some partial output with no marker", "")
    assert ok is False
    assert "marker" in reason.lower()


def test_timeout_is_a_crash():
    ok, reason = sc.sweep_completed_cleanly("sweep_fee_stack", -1, "", "timeout")
    assert ok is False


def _fake_run_sweep_factory(overrides):
    def _fake(name, args, env):
        if name in overrides:
            return (name, *overrides[name])
        return name, 0, sc.SWEEP_RAN_MARKERS[name], ""
    return _fake


def _row(card, module):
    for r in card["modules"]:
        if r["module"] == module:
            return r
    raise AssertionError(f"module {module!r} absent from scorecard")


def test_crashed_sweep_reports_error_not_health(monkeypatch, tmp_path):
    crash_out = "starting sweep\npartial\nTraceback (most recent call last):\n RuntimeError"
    crash_err = "Traceback (most recent call last):\nRuntimeError: boom"
    monkeypatch.setattr(
        sc, "run_sweep",
        _fake_run_sweep_factory({"sweep_fee_stack": (1, crash_out, crash_err)}),
    )
    card = sc.build_scorecard(tmp_path, None)
    for module in sc.SWEEP_MODULES["sweep_fee_stack"]:
        row = _row(card, module)
        assert row["status"] == "ERROR", row
    assert "ERROR" in {r["status"] for r in card["modules"]}
