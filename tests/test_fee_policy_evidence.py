import copy
import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from modules.fee_policy_evidence import capture_fee_request, complete_fee_execution


def test_request_context_is_a_snapshot_not_a_reference():
    info = {"fee_proportional_millionths": 775, "fee_base_msat": 10,
            "spendable_msat": "500000000msat", "capacity": 1_000_000}
    params = {"feeppm": 856, "feebase": 20, "htlcmax": "400000000msat"}
    snapshot = capture_fee_request(info, params)
    info["spendable_msat"] = 1
    params["feeppm"] = 1200
    assert snapshot["prior_policy"] == {"fee_ppm": 775, "base_fee_msat": 10}
    assert snapshot["prior_context"]["spendable_msat"] == 500_000_000
    assert snapshot["requested_policy"]["fee_ppm"] == 856
    assert snapshot["prior_context"]["capacity_sats"] == 1_000_000


def test_absent_context_and_malformed_params_remain_unknown():
    request = capture_fee_request(None, [])
    assert all(value is None for value in request["prior_context"].values())
    assert all(value is None for value in request["requested_policy"].values())


def test_reported_policy_is_not_requested_policy():
    request = capture_fee_request({}, {"feeppm": 856, "feebase": 20})
    before = copy.deepcopy(request)
    response = {"channels": [{"short_channel_id": "1x1x0",
        "fee_proportional_millionths": 850, "fee_base_msat": "19msat"}]}
    result = complete_fee_execution(request, response, "1x1x0", 100)
    assert result["reported_policy"]["fee_ppm"] == 850
    assert result["reported_policy"]["base_fee_msat"] == 19
    assert result["requested_policy"]["fee_ppm"] == 856
    assert result["attribution_status"] == "pending"
    assert result["time_resolution_seconds"] == 1
    assert request == before


@pytest.mark.parametrize("response", [None, {}, {"channels": None},
    {"channels": [{"short_channel_id": "other", "fee_proportional_millionths": 100}]},
    {"channels": [{"short_channel_id": "1x1x0"}, {"short_channel_id": "1x1x0"}]},
])
def test_missing_or_ambiguous_readback_remains_unknown(response):
    result = complete_fee_execution(capture_fee_request({}, {}), response, "1x1x0", 100)
    assert result["reported_policy"] is None


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("nan"), float("inf"),
                                  "bad", "9" * 5000, {}, None])
def test_malformed_values_are_unknown_and_json_is_finite(bad):
    request = capture_fee_request({"fee_base_msat": bad}, {"feeppm": bad})
    assert request["prior_policy"]["base_fee_msat"] is None
    assert request["requested_policy"]["fee_ppm"] is None
    json.dumps(request, allow_nan=False)


def make_controller(mock_plugin, mock_database, **config):
    from modules.config import Config
    from modules.fee_authority import FeeAuthorityGate
    from modules.fee_controller import FeeController
    cfg = Config(min_fee_ppm=10, max_fee_ppm=1200, base_fee_msat=20,
                 econ_governor_fees_enabled=False, **config)
    controller = FeeController(mock_plugin, cfg, mock_database,
                               fee_authority_gate=FeeAuthorityGate())
    controller.data_service = MagicMock()
    controller.data_service.set_channel.return_value = {
        "channels": [{"short_channel_id": "1x1x0",
                      "fee_proportional_millionths": 850, "fee_base_msat": 19}]
    }
    return controller


def info():
    return {"peer_id": "02" + "a" * 64, "fee_proportional_millionths": 775,
            "fee_base_msat": 10, "spendable_msat": 500_000_000}


def test_successful_action_records_pre_action_and_reported_policy(mock_plugin, mock_database):
    fc = make_controller(mock_plugin, mock_database, dry_run=False)
    channel = info()
    response = fc.data_service.set_channel.return_value
    def apply(**kwargs):
        channel["spendable_msat"] = 1
        return response
    fc.data_service.set_channel.side_effect = apply
    result = fc.set_channel_fee("1x1x0", 856, channel_info=channel)
    assert result["success"] is True
    evidence = mock_database.record_fee_change.call_args.kwargs["execution_evidence"]
    assert evidence["prior_context"]["spendable_msat"] == 500_000_000
    assert evidence["prior_policy"] == {"fee_ppm": 775, "base_fee_msat": 10}
    assert evidence["requested_policy"]["fee_ppm"] == 856
    assert evidence["reported_policy"]["fee_ppm"] == 850
    assert evidence["reported_policy"]["base_fee_msat"] == 19
    assert evidence["attribution_status"] == "pending"


@pytest.mark.parametrize("condition", ["dry_run", "denied", "rpc_failure"])
def test_no_applied_exposure_for_unexecuted_or_failed_action(
    mock_plugin, mock_database, condition
):
    fc = make_controller(mock_plugin, mock_database, dry_run=condition == "dry_run")
    if condition == "denied":
        fc.fee_authority_gate.set_enabled(False, reason="test")
    if condition == "rpc_failure":
        fc.data_service.set_channel.side_effect = RuntimeError("not applied")
    fc.set_channel_fee("1x1x0", 856, channel_info=info())
    mock_database.record_fee_change.assert_not_called()
    if condition != "rpc_failure":
        fc.data_service.set_channel.assert_not_called()


def test_capture_failure_cannot_block_authorized_action(mock_plugin, mock_database, monkeypatch):
    fc = make_controller(mock_plugin, mock_database, dry_run=False)
    def broken(*args):
        raise ValueError("unavailable evidence")
    monkeypatch.setattr("modules.fee_controller.capture_fee_request", broken)
    result = fc.set_channel_fee("1x1x0", 856, channel_info=info())
    assert result["success"] is True
    assert "fee_evidence_unavailable" in result["warnings"]
    assert mock_database.record_fee_change.call_args.kwargs["execution_evidence"] is None


def test_completion_failure_keeps_legacy_audit_record(mock_plugin, mock_database, monkeypatch):
    fc = make_controller(mock_plugin, mock_database, dry_run=False)
    def broken(*args):
        raise ValueError("unavailable readback evidence")
    monkeypatch.setattr("modules.fee_controller.complete_fee_execution", broken)
    result = fc.set_channel_fee("1x1x0", 856, channel_info=info())
    assert result["success"] is True
    assert "fee_evidence_unavailable" in result["warnings"]
    assert mock_database.record_fee_change.call_args.kwargs["execution_evidence"] is None


def test_failed_record_does_not_turn_applied_fee_into_failed_action(mock_plugin, mock_database):
    fc = make_controller(mock_plugin, mock_database, dry_run=False)
    mock_database.record_fee_change.side_effect = sqlite3.OperationalError("disk failure")
    result = fc.set_channel_fee("1x1x0", 856, channel_info=info())
    assert result["success"] is True
    assert "record_fee_change_failed" in result["warnings"]


def test_existing_database_migrates_without_inventing_historical_context(tmp_path):
    from modules.database import Database
    db = Database(str(tmp_path / "legacy.db"), MagicMock())
    conn = db._get_connection()
    conn.execute("""CREATE TABLE fee_changes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT NOT NULL,
        peer_id TEXT NOT NULL, old_fee_ppm INTEGER NOT NULL,
        new_fee_ppm INTEGER NOT NULL, reason TEXT, manual INTEGER DEFAULT 0,
        timestamp INTEGER NOT NULL)""")
    conn.execute("INSERT INTO fee_changes VALUES (1, '1x1x0', 'peer', 100, 200, 'old', 0, 10)")
    try:
        db.initialize()
        db.initialize()
        assert conn.execute("SELECT execution_evidence FROM fee_changes WHERE id=1").fetchone()[0] is None
        evidence = complete_fee_execution(capture_fee_request(info(), {}), {}, "1x1x0", 100)
        db.record_fee_change("1x1x0", "peer", 775, 856, "new", execution_evidence=evidence)
        raw = conn.execute("SELECT execution_evidence FROM fee_changes ORDER BY id DESC LIMIT 1").fetchone()[0]
        assert json.loads(raw) == evidence
        with pytest.raises(ValueError):
            db.record_fee_change("1x1x0", "peer", 1, 2, "bad",
                                 execution_evidence={"fee": float("nan")})
        for invalid in ([], {"padding": "x" * 16384}):
            with pytest.raises(ValueError):
                db.record_fee_change("1x1x0", "peer", 1, 2, "bad",
                                     execution_evidence=invalid)
        assert conn.execute("SELECT COUNT(*) FROM fee_changes").fetchone()[0] == 2
    finally:
        db.close()
