import copy
import hashlib
import inspect
import json
import random
import sqlite3
import sys
from unittest.mock import Mock

import pytest

from modules import fee_controller as controller
from tools import settled_reward_model_probe as p
from tools import settled_reward_transition_audit as audit

NOW = 1788718966
SCID = "1x2x0"
MODEL_DIGEST = "ccce4f1568c0e712e490162e2a22f8e0b646b5767e0521aa615ebecb1cbf7cc8"


@pytest.fixture
def classes():
    entropy = p.Entropy(NOW)
    classes = p.isolated_classes(controller, entropy)
    yield classes, entropy
    sys.modules.pop(classes.__name__, None)


def state_payload(classes):
    state = classes.ChannelFeeState()
    state.last_context_key = "normal:normal:P"
    state.last_time_bucket = "normal"
    state.pid.integral_error = .75
    state.dynamic_htlcmin_baseline_msat = 1000
    for fee, rate in [(200, 40), (400, 100), (600, 250), (800, 180)] * 4:
        state.thompson.update_posterior(fee, rate, 1)
        state.thompson.update_contextual(state.last_context_key, fee, rate, "normal")
    return {"fee_state": json.loads(p._json(state.to_v2_dict()))}


def row():
    return {"last_update": NOW-3600, "last_revenue_rate": 214.0, "last_fee_ppm": 856,
            "is_sleeping": 1, "sleep_until": NOW+3600, "stable_cycles": 4,
            "last_broadcast_fee_ppm": 856, "last_volume_sats": 250000}


def kwargs():
    return dict(volume_msat=250000000, earned_msat=193750, since=NOW-3600,
                until=NOW, fee_ppm=856, floor=100, ceiling=1200, seeds=32)


def test_probed_class_ast_is_identical_to_pinned_incumbent_and_candidate():
    # Full-source SHA differs, but these actual model/serialization definitions
    # are identical across 294e649, d16f223 and current main.
    assert p.model_digest(inspect.getsource(controller)) == MODEL_DIGEST


def test_source_pin_does_not_depend_on_python_ast_dump_format(monkeypatch):
    monkeypatch.setattr(p.ast, "dump", Mock(side_effect=AssertionError("version dependent")))
    assert p.model_digest(inspect.getsource(controller)) == MODEL_DIGEST
    changed = inspect.getsource(controller).replace('ZERO_PROBE_FLAG = "zero_probe"', 'ZERO_PROBE_FLAG = "changed"')
    assert p.model_digest(changed) != MODEL_DIGEST


def test_isolated_annotations_preserve_production_future_semantics(classes):
    copied, _ = classes
    assert copied.ChannelFeeState.__annotations__ == controller.ChannelFeeState.__annotations__


def test_isolation_preserves_module_clock_and_process_entropy(classes):
    copied, entropy = classes
    original_clock = controller.decision_now
    state_before = random.getstate()
    payload = state_payload(copied)
    before = copy.deepcopy(payload)
    result = p.compare_window(controller, copied, entropy, row(), payload, **kwargs())
    assert result["windows"] == result["positive_windows"] == result["reward_changed"] == 1
    assert result["paired_proposals"] == 32
    assert 0 <= result["max_proposal_delta_ppm"] <= 1100
    assert payload == before
    assert controller.decision_now is original_clock
    assert random.getstate() == state_before


def test_equal_reward_has_identical_models_and_proposals(classes):
    copied, entropy = classes
    values = kwargs() | {"earned_msat": 214000}
    result = p.compare_window(controller, copied, entropy, row(), state_payload(copied), **values)
    assert result["reward_changed"] == result["changed_proposals"] == result["positive_reference_changed"] == 0


def test_restart_and_source_rollback_retain_pid_and_scalar_state(classes):
    copied, _ = classes
    payload = state_payload(copied)
    state = copied.ChannelFeeState.from_v2_dict(payload["fee_state"], row())
    restored = p._roundtrip(state, copied.ChannelFeeState, row())
    assert restored.pid.integral_error == .75
    assert restored.is_sleeping and restored.sleep_until == NOW+3600
    assert restored.stable_cycles == 4 and restored.last_fee_ppm == 856
    assert restored.dynamic_htlcmin_baseline_msat == 1000
    actual_incumbent = controller.ChannelFeeState.from_v2_dict(restored.to_v2_dict(), row())
    assert p._json(actual_incumbent.to_v2_dict()) == p._json(restored.to_v2_dict())


@pytest.mark.parametrize("bad", [None, True, -1, "193750", 1.2, float("nan")])
def test_unknown_reward_never_invokes_model_or_fabricates_zero(classes, bad):
    copied, entropy = classes
    values = kwargs() | {"earned_msat": bad}
    assert p.compare_window(controller, copied, entropy, row(), None, **values) == {"unknown": 1}


@pytest.mark.parametrize("replacement", [{"seeds": 0}, {"seeds": 33}, {"ceiling": 1201},
                                         {"floor": -1}, {"floor": 1201}, {"since": NOW},
                                         {"fee_ppm": True}, {"until": None}])
def test_bad_windows_and_resource_requests_refuse(classes, replacement):
    copied, entropy = classes
    with pytest.raises(audit.AuditError):
        p.compare_window(controller, copied, entropy, row(), {}, **(kwargs() | replacement))


def test_zero_quote_does_not_teach_global_or_contextual_price_response(classes):
    copied, entropy = classes
    result = p.compare_window(controller, copied, entropy, row(), state_payload(copied), **(kwargs() | {"fee_ppm": 0}))
    assert result["reward_changed"] == 1  # Actual base-fee earnings may still exist.
    assert result["changed_proposals"] == result["positive_reference_changed"] == result["zero_streak_changed"] == 0


def test_empty_window_remains_genuine_zero_not_unknown(classes):
    copied, entropy = classes
    result = p.compare_window(controller, copied, entropy, row(), state_payload(copied),
                              **(kwargs() | {"volume_msat": 0, "earned_msat": 0}))
    assert result["unknown"] == result["positive_windows"] == result["changed_proposals"] == 0


def test_bootstrap_uses_changed_real_duration(classes):
    copied, entropy = classes
    result = p.compare_window(controller, copied, entropy, row() | {"last_update": 0}, state_payload(copied),
                              **(kwargs() | {"since": NOW-1800, "earned_msat": 214000}))
    assert result["reward_changed"] == 1


def test_semantic_entropy_does_not_shift_on_other_branch_calls():
    e = p.Entropy(NOW)
    first = e.gauss("polynomial.a", 0, 1)
    e.reset(0)
    e.gauss("prior", 0, 1)
    assert e.gauss("polynomial.a", 0, 1) == first
    e.reset(0)
    assert e.gauss("polynomial.a", 10, 2) == 10 + first*2


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "source.db"
    state = controller.ChannelFeeState()
    state.last_context_key = "normal:normal:P"
    payload = {"fee_state": state.to_v2_dict(), "cycle_state": row()}
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE fee_strategy_state(channel_id TEXT PRIMARY KEY,last_update INTEGER,
                last_revenue_rate REAL,last_fee_ppm INTEGER,is_sleeping INTEGER,sleep_until INTEGER,
                stable_cycles INTEGER,last_broadcast_fee_ppm INTEGER,last_volume_sats INTEGER,v2_state_json TEXT);
            CREATE TABLE forwards(in_msat INTEGER,out_msat INTEGER,fee_msat INTEGER,timestamp INTEGER,out_channel TEXT);
            CREATE INDEX outgoing_window ON forwards(out_channel,timestamp);
        """)
        r = row()
        conn.execute("INSERT INTO fee_strategy_state VALUES (?,?,?,?,?,?,?,?,?,?)", (SCID, *r.values(), p._json(payload)))
        conn.execute("INSERT INTO forwards VALUES (?,?,?,?,?)", (250193750,250000000,193750,NOW-1000,SCID))
    return path


def probe_args(db):
    rpc = Mock(return_value={"channels": [{"state": "CHANNELD_NORMAL", "short_channel_id": SCID,
                                          "fee_proportional_millionths": 856}]})
    return dict(database=db, rpc=rpc, controller=controller,
                expected_controller_sha=hashlib.sha256(inspect.getsource(controller).encode()).hexdigest(),
                expected_model_digest=MODEL_DIGEST, floor=100, ceiling=1200, now=NOW, seeds=2)


def test_readonly_probe_preserves_file_and_exports_aggregates_only(db):
    before = db.read_bytes()
    args = probe_args(db)
    report = p.probe(**args)
    assert report["windows"] == report["positive_windows"] == 1
    assert report["paired_proposals"] == 2
    assert report["production_admission_eligible"] is False
    assert db.read_bytes() == before
    assert SCID not in p._json(report) and str(db) not in p._json(report)
    assert args["rpc"].call_args_list == [(("listpeerchannels", {}),), (("listpeerchannels", {}),)]
    assert "_revenue_reward_model_probe" not in sys.modules


@pytest.mark.parametrize("field", ["expected_controller_sha", "expected_model_digest"])
def test_changed_source_refuses_before_rpc(db, field):
    args = probe_args(db) | {field: "changed"}
    with pytest.raises(audit.AuditError, match="pin"):
        p.probe(**args)
    args["rpc"].assert_not_called()


@pytest.mark.parametrize("sql", ["UPDATE forwards SET fee_msat=-1", "UPDATE forwards SET timestamp=1788718967",
                                 "UPDATE fee_strategy_state SET last_update=1788718966"])
def test_malformed_rows_unknown_and_namespace_cleaned(db, sql):
    with sqlite3.connect(db) as conn:
        conn.execute(sql)
    report = p.probe(**probe_args(db))
    assert report["unknown"] == 1
    assert report.get("windows", 0) == 0
    assert "_revenue_reward_model_probe" not in sys.modules


def test_malformed_nested_state_refuses_without_private_trace(db):
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE fee_strategy_state SET v2_state_json='[]'")
    with pytest.raises(audit.AuditError) as exc:
        p.probe(**probe_args(db))
    assert SCID not in str(exc.value) and str(db) not in str(exc.value)
    assert "_revenue_reward_model_probe" not in sys.modules


def test_expired_budget_and_quote_drift_refuse(db, monkeypatch):
    args = probe_args(db)
    first = args["rpc"].return_value
    args["rpc"].side_effect = [first, {"channels": []}]
    with pytest.raises(audit.AuditError, match="quotes changed"):
        p.probe(**args)
    monkeypatch.setattr(p, "MAX_SECONDS", 0)
    with pytest.raises(audit.AuditError):
        p.probe(**probe_args(db))
