"""P5-001: scorecard.py must not launder the capital_efficiency verdict.

ML-* checks read cl-hive's hive-organism-status.json and are cl-hive-owned;
attributing them to capital_efficiency (whose own code has zero corpus checks)
would let it report PASS/KNOWN while none of its logic ran. They must route to
a distinct cl-hive pseudo-module, leaving capital_efficiency INCONCLUSIVE.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCORECARD = os.path.join(_REPO, "tools", "audit", "scorecard.py")


def _load_scorecard():
    spec = importlib.util.spec_from_file_location("scorecard_p5001", _SCORECARD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so dataclasses can resolve the module
    spec.loader.exec_module(mod)
    return mod


sc = _load_scorecard()


def _check(sweep, check_id, name=None, checked=1, violations=0):
    return sc.Check(sweep, check_id, name or check_id, checked, violations)


def test_ml_checks_route_to_metabolism_ledger_not_capital_efficiency():
    c = _check("sweep_data_budget", "ML-COVER", "ML-COVER anomaly")
    module = sc.module_for("sweep_data_budget", c)
    assert module == sc.METABOLISM_LEDGER_MODULE
    assert module != "capital_efficiency"


def test_no_prefix_routes_to_capital_efficiency():
    routed = [m for (_sw, _pfx, m) in sc.PREFIX_MODULES]
    assert "capital_efficiency" not in routed
    assert sc.METABOLISM_LEDGER_MODULE in routed


def test_metabolism_ledger_pseudo_module_registered():
    assert sc.METABOLISM_LEDGER_MODULE in sc.MODULE_ORDER
    assert sc.METABOLISM_LEDGER_MODULE in sc.SWEEP_MODULES["sweep_data_budget"]
    assert "capital_efficiency" in sc.NO_SWEEP_MODULES


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


def test_capital_efficiency_inconclusive_when_only_ml_checks_fire(monkeypatch, tmp_path):
    data_budget_out = (
        "corpus root: x\n"
        "snapshots swept: 10\n\n"
        "FAIL ML-COVER coverage status==unknown (anomaly)  pass=0 fail=5\n"
        "OK ML-BURN-IDENT burn identical (anomaly)  pass=3 fail=0\n"
    )
    monkeypatch.setattr(
        sc, "run_sweep",
        _fake_run_sweep_factory({"sweep_data_budget": (1, data_budget_out, "")}),
    )
    card = sc.build_scorecard(tmp_path, None)

    cap = _row(card, "capital_efficiency")
    assert cap["status"] == "INCONCLUSIVE", cap
    assert cap["violations_known"] == 0
    assert cap["checks_total"] == 0

    meta = _row(card, sc.METABOLISM_LEDGER_MODULE)
    assert meta["status"] == "KNOWN", meta
    assert meta["violations_known"] == 5
