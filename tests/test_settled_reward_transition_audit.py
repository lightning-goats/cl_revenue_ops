"""Production diagnostic uses fixtures only; no live CLN or DB action."""

import json
import sqlite3
from unittest.mock import Mock

import pytest

from tools import settled_reward_transition_audit as a

NOW = 10000
SCID = "1x2x0"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE fee_strategy_state(channel_id TEXT PRIMARY KEY,last_update INTEGER,
                last_revenue_rate REAL,is_sleeping INTEGER,sleep_until INTEGER,v2_state_json TEXT);
            CREATE TABLE forwards(in_msat INTEGER,out_msat INTEGER,fee_msat INTEGER,timestamp INTEGER,out_channel TEXT);
            CREATE INDEX outgoing_window ON forwards(out_channel,timestamp);
        """)
        model = {"thompson_state": {"observations": [[775, 193.75, 1, 6000, "normal"]],
                                  "contextual_posteriors": {"normal": [775, .1, 1, 6000]}}}
        conn.execute("INSERT INTO fee_strategy_state VALUES (?,?,?,?,?,?)", (SCID, 6400, 214, 0, 0, json.dumps(model)))
        conn.execute("INSERT INTO forwards VALUES (?,?,?,?,?)", (250193750, 250000000, 193750, 8000, SCID))
    return path


def rpc(ppm=856):
    return Mock(return_value={"channels": [{"state": "CHANNELD_NORMAL", "short_channel_id": SCID,
                                          "fee_proportional_millionths": ppm}]})


def change(db, sql, values=()):
    with sqlite3.connect(db) as conn:
        conn.execute(sql, values)


def test_actual_earnings_not_current_price_and_no_source_mutation(db):
    before = db.read_bytes()
    transport = rpc()
    report = a.audit(db, transport, now=NOW)
    assert report["earned_msat"] == 193750
    assert report["forward_count"] == report["shortfall_forwards"] == 1
    assert report["reward_changed_windows"] == 1
    assert report["normal_stability_predicate_changes"] == 1
    assert report["max_reward_delta_over_max_one_proxy_sats_per_hour"] == pytest.approx((214-193.75)/214)
    assert report["stored_positive_observations"] == report["stored_observations"] == 1
    assert report["models_without_reward_source_marker"] == 1
    assert report["production_admission_eligible"] is False
    assert db.read_bytes() == before
    assert transport.call_args_list == [(("listpeerchannels", {}),), (("listpeerchannels", {}),)]
    rendered = json.dumps(report)
    assert SCID not in rendered and str(db) not in rendered and "thompson_state" not in report


def test_candidate_rate_agrees_with_real_reader(db):
    from modules.fee_controller import FeeController
    controller = object.__new__(FeeController)
    controller.database = Mock()
    controller.database.get_forward_revenue_msat.return_value = 193750
    controller.plugin = Mock()
    report = a.audit(db, rpc(), now=NOW)
    rate = controller._get_settled_revenue_rate(SCID, 6400, NOW)
    assert report["max_reward_delta_over_max_one_proxy_sats_per_hour"] == pytest.approx(abs(rate-214)/214)
    assert controller.plugin.mock_calls == []


def test_thresholds_match_controller():
    from modules.fee_controller import FeeController
    assert (FeeController.VOLATILITY_THRESHOLD, FeeController.STABILITY_THRESHOLD,
            FeeController.WAKE_UP_THRESHOLD) == (.5, .01, .2)
    assert a._branches(150, 100) == (False, False, True)
    assert a._branches(100, 100) == (False, True, False)
    assert a._branches(1, 0) == (True, False, True)


def test_bootstrap_half_hour_and_sleep_denominators(db):
    change(db, "UPDATE fee_strategy_state SET last_update=0,last_revenue_rate=214,is_sleeping=1,sleep_until=11000")
    change(db, "UPDATE forwards SET timestamp=9000,fee_msat=214000,in_msat=250214000")
    report = a.audit(db, rpc(), now=NOW, fee_interval=1800)
    assert report["bootstrap_windows"] == 1
    assert report["max_reward_delta_over_max_one_proxy_sats_per_hour"] == 1
    assert report["sleep_wake_predicate_changes"] == 1
    assert report["normal_stability_predicate_changes"] == 0  # Last-update guard.


def test_subsixminute_sleep_clamp_is_not_used_for_normal_rate(db):
    change(db, "UPDATE fee_strategy_state SET last_update=9940,last_revenue_rate=2140,is_sleeping=1,sleep_until=11000")
    change(db, "UPDATE forwards SET timestamp=9999,fee_msat=214000,in_msat=250214000")
    report = a.audit(db, rpc(), now=NOW)
    assert report["reward_changed_windows"] == 0
    assert report["sleep_wake_predicate_changes"] == 1


def test_empty_window_is_not_an_upgrade_proof(db):
    change(db, "DELETE FROM forwards")
    report = a.audit(db, rpc(), now=NOW)
    assert report["zero_volume_windows"] == 1
    assert report["reward_changed_windows"] == report["earned_msat"] == 0
    assert report["production_admission_eligible"] is False


def test_missing_strategy_and_inactive_rows_are_explicit(db):
    transport = rpc()
    transport.return_value["channels"][0]["short_channel_id"] = "2x3x0"
    report = a.audit(db, transport, now=NOW)
    assert report["active_without_strategy"] == report["inactive_strategy_rows"] == 1
    assert report["positive_volume_windows"] == 0


@pytest.mark.parametrize("column,value", [("fee_msat", -1), ("out_msat", "bad"), ("in_msat", None),
                                         ("fee_msat", 193751), ("timestamp", NOW+1), ("timestamp", 9000.5)])
def test_bad_forward_window_is_unknown_not_zero(db, column, value):
    change(db, f"UPDATE forwards SET {column}=?", (value,))
    report = a.audit(db, rpc(), now=NOW)
    assert report["unknown_windows"] == 1
    assert report["zero_volume_windows"] == report["earned_msat"] == 0


@pytest.mark.parametrize("column,value", [("last_update", NOW), ("last_update", NOW+1),
                                         ("last_revenue_rate", -1), ("is_sleeping", 2)])
def test_invalid_strategy_is_unknown(db, column, value):
    change(db, f"UPDATE fee_strategy_state SET {column}=?", (value,))
    assert a.audit(db, rpc(), now=NOW)["unknown_windows"] == 1


@pytest.mark.parametrize("raw", ["{", "null", "[]", '{"thompson_state":null}',
                               '{"thompson_state":{"observations":[[775,NaN,1,6000]]}}',
                               '{"thompson_state":{"observations":[null]}}'])
def test_bad_models_refuse_without_disclosing_source(db, raw):
    change(db, "UPDATE fee_strategy_state SET v2_state_json=?", (raw,))
    with pytest.raises(a.AuditError) as exc:
        a.audit(db, rpc(), now=NOW)
    assert SCID not in str(exc.value) and str(db) not in str(exc.value)


def test_quote_drift_refuses_whole_report(db):
    transport = Mock(side_effect=[rpc(856).return_value, rpc(857).return_value])
    with pytest.raises(a.AuditError, match="changed"):
        a.audit(db, transport, now=NOW)


@pytest.mark.parametrize("reply", [None, {}, {"channels": [None]}, {"channels": [{}]},
                                  {"channels": [{"state": "CHANNELD_NORMAL"}]}])
def test_bad_channel_surface(db, reply):
    # An entry without an active state is not included in the audit.
    if reply == {"channels": [{}]}:
        assert a.audit(db, Mock(return_value=reply), now=NOW)["active_channels"] == 0
    else:
        with pytest.raises(a.AuditError):
            a.audit(db, Mock(return_value=reply), now=NOW)


def test_resource_budgets_and_missing_database_refuse(db, monkeypatch):
    monkeypatch.setattr(a, "MAX_STATE_BYTES", 2)
    with pytest.raises(a.AuditError, match="oversized"):
        a.audit(db, rpc(), now=NOW)
    with pytest.raises(a.AuditError):
        a.audit(db.parent / "missing.db", rpc(), now=NOW)
    assert not (db.parent / "missing.db").exists()


def test_transport_cannot_call_action_or_different_params():
    transport = a.ChannelReader("must-not-open")
    for method, params in [("revenue-fee-cycle", {}), ("setchannel", {}), ("listpeerchannels", {"id": SCID})]:
        with pytest.raises(a.AuditError, match="outside"):
            transport(method, params)


def test_bounds_direction_and_integer_rounding(db):
    change(db, "UPDATE forwards SET timestamp=6400")
    assert a.audit(db, rpc(), now=NOW)["forward_count"] == 0
    change(db, "UPDATE forwards SET timestamp=10000,out_msat=1001,fee_msat=0,in_msat=1001")
    report = a.audit(db, rpc(775), now=NOW)
    assert report["forward_count"] == 1 and report["shortfall_forwards"] == 0
    assert report["reward_changed_windows"] == 1  # Subsat rounding isn't silently dropped.


def test_sqlite_overflow_is_unavailable(db):
    change(db, "INSERT INTO forwards VALUES (?,?,?,?,?)", (2**63-1, 2**63-1, 0, 9000, SCID))
    with pytest.raises(a.AuditError, match="unavailable"):
        a.audit(db, rpc(), now=NOW)


def test_nested_model_wins_over_legacy_mirror_like_runtime(db):
    from modules.fee_controller import FeeController
    nested = {"thompson_state": {"observations": [[775, 25, 1, 6000, "normal"], [775, 0, 1, 6200, "normal"]],
                                "contextual_posteriors": {"low": [775, .1, 2, 6200]}, "positive_rate_ref": 25}}
    payload = {"fee_state": nested, "thompson_state": {"observations": []}}
    assert FeeController._extract_fee_state_payload({}, payload)["thompson_state"] == nested["thompson_state"]
    change(db, "UPDATE fee_strategy_state SET v2_state_json=?", (json.dumps(payload),))
    report = a.audit(db, rpc(), now=NOW)
    assert report["stored_observations"] == 2
    assert report["stored_positive_observations"] == 1
    assert report["stored_contexts"] == 1
    assert report["positive_reference_models"] == 1


def test_missing_model_is_explicit_not_an_empty_learning_claim(db):
    change(db, "UPDATE fee_strategy_state SET v2_state_json='{}'")
    assert a.audit(db, rpc(), now=NOW)["models_missing_thompson_state"] == 1


def test_nested_cycle_mismatch_is_unknown(db):
    change(db, "UPDATE fee_strategy_state SET v2_state_json=?", (json.dumps({"cycle_state": {"last_update": 9000}}),))
    report = a.audit(db, rpc(), now=NOW)
    assert report["unknown_windows"] == 1
    assert report["positive_volume_windows"] == 0


def test_nested_model_malformed_cannot_hide_behind_empty_flat_default(db):
    change(db, "UPDATE fee_strategy_state SET v2_state_json=?", (json.dumps({"fee_state": {"thompson_state": None}}),))
    with pytest.raises(a.AuditError):
        a.audit(db, rpc(), now=NOW)


def test_rpc_failure_and_deadline_refuse(db, monkeypatch):
    with pytest.raises(a.AuditError, match="unavailable"):
        a.audit(db, Mock(side_effect=RuntimeError("PRIVATE DETAILS")), now=NOW)
    monkeypatch.setattr(a, "MAX_SECONDS", 0)
    with pytest.raises(a.AuditError):
        a.audit(db, rpc(), now=NOW)


@pytest.mark.parametrize("reply", [b'', b'{"id":"wrong","result":{}}\n\n', b'{"id":"reward-audit","error":{}}\n\n',
                                   b'not-json\n\n', b'x' * (4*1024*1024+1)])
def test_transport_refuses_bad_frames(monkeypatch, reply):
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    connection.recv.return_value = reply
    monkeypatch.setattr(a.socket, "socket", Mock(return_value=connection))
    with pytest.raises(a.AuditError):
        a.ChannelReader("private-socket")("listpeerchannels", {})
    request = json.loads(connection.sendall.call_args.args[0])
    assert request["method"] == "listpeerchannels" and request["params"] == {}


def test_transport_reads_split_reply(monkeypatch):
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    connection.recv.side_effect = [b'{"id":"reward-audit",', b'"result":{"channels":[]}}\n\n']
    monkeypatch.setattr(a.socket, "socket", Mock(return_value=connection))
    assert a.ChannelReader("private-socket")("listpeerchannels", {}) == {"channels": []}
