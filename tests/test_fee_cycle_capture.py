import json
import threading
import time
from pathlib import Path

import modules.fee_cycle_capture as capture
from modules.fee_cycle_capture import (
    FeeCycleCaptureManager,
    FeeCycleCaptureSession,
    bind_capture,
    current_capture,
)
from modules.fee_cycle_replay_wire import canonical_body_bytes, seal_envelope


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _complete_session(manager, *, padding="", producer=None):
    session = manager.begin_cycle(
        {"version": 1, "padding": padding},
        producer or {"python_commit": "abc"},
    )
    assert session is not None
    session.record_pre_state({"global": {}, "ordered_channels": []})
    session.record_expected(
        {
            "ordered_outcomes": [],
            "post_global_state": {},
            "post_channel_state": [],
        }
    )
    return session


def _capture_files(directory: Path):
    return sorted(
        path
        for path in directory.glob("*.json")
        if not path.name.startswith("manifest-")
    )


def test_disabled_manager_opens_no_session(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )

    assert manager.begin_cycle({"version": 1}, {"python_commit": "abc"}) is None
    assert not (tmp_path / "revenue_ops_fee_replay").exists()


def test_session_records_copies_and_context_binding():
    session = FeeCycleCaptureSession(
        capture_run_id="run-a",
        capture_seq=1,
        cycle_id="run-a:1",
        producer={"started_at": "2026-07-18T00:00:00+00:00"},
        configuration={"version": 1},
    )
    pre_state = {"global": {}, "ordered_channels": [{"channel_id": "1x1x1"}]}
    observation = {"channel_id": "1x1x1", "value": 1.5}
    expected = {
        "ordered_outcomes": [{"channel_id": "1x1x1"}],
        "post_global_state": {},
        "post_channel_state": [],
    }

    session.record_pre_state(pre_state)
    session.record_observation("evidence", observation)
    session.record_expected(expected)
    pre_state["ordered_channels"].clear()
    observation["value"] = 9.0
    expected["ordered_outcomes"].clear()

    assert current_capture() is None
    with bind_capture(session):
        assert current_capture() is session
    assert current_capture() is None

    body = session.to_body()
    assert body["pre_state"]["ordered_channels"] == [{"channel_id": "1x1x1"}]
    assert body["observations"]["evidence"][0]["value"] == 1.5
    assert body["expected"]["ordered_outcomes"] == [{"channel_id": "1x1x1"}]
    assert body["completeness"] == {
        "evaluated_channels": 1,
        "terminal_outcomes": 1,
        "evidence_entries": 1,
        "clock_entries": 0,
        "entropy_entries": 0,
        "complete": True,
    }

    session.record_observation("not-a-family", {})
    assert session.invalid_reason == "unknown observation family: not-a-family"
    assert session.to_body()["completeness"]["complete"] is False


def test_enabled_manager_publishes_private_atomic_cycle(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    session = manager.begin_cycle({"version": 1}, {"python_commit": "abc"})
    assert session is not None
    session.record_pre_state({"global": {}, "ordered_channels": []})
    session.record_expected(
        {
            "ordered_outcomes": [],
            "post_global_state": {},
            "post_channel_state": [],
        }
    )
    manager.finish_cycle(session)
    assert manager.set_enabled(False, timeout_seconds=5.0)

    manifest = manager.read_manifest()
    assert manifest["state"] == "closed"
    assert manifest["queue_drained"] is True
    assert manifest["completed"] == 1
    output_dir = tmp_path / "revenue_ops_fee_replay"
    files = list(output_dir.glob("*.json"))
    captures = [path for path in files if not path.name.startswith("manifest-")]
    manifests = [path for path in files if path.name.startswith("manifest-")]
    assert len(captures) == 1
    assert len(manifests) == 1
    assert output_dir.stat().st_mode & 0o777 == 0o700
    assert captures[0].stat().st_mode & 0o777 == 0o600
    assert manifests[0].stat().st_mode & 0o777 == 0o600
    assert not list(output_dir.glob("*.tmp"))
    assert not list(output_dir.glob(".*.tmp"))


def test_queue_capacity_drops_third_pending_submission(tmp_path, monkeypatch):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    entered = threading.Event()
    release = threading.Event()
    original_publish = manager._publish_envelope

    def blocked_publish(session):
        entered.set()
        assert release.wait(5.0)
        return original_publish(session)

    monkeypatch.setattr(manager, "_publish_envelope", blocked_publish)
    manager.set_enabled(True)

    manager.finish_cycle(_complete_session(manager, padding="active"))
    assert entered.wait(5.0)
    manager.finish_cycle(_complete_session(manager, padding="pending-1"))
    manager.finish_cycle(_complete_session(manager, padding="pending-2"))
    manager.finish_cycle(_complete_session(manager, padding="pending-3"))

    manifest = manager.read_manifest()
    assert manifest["dropped"] == 1
    assert manifest["attempts"][-1]["status"] == "dropped"
    assert manifest["attempts"][-1]["eligible"] is False

    release.set()
    assert manager.set_enabled(False, timeout_seconds=5.0)
    assert manager.read_manifest()["completed"] == 3


def test_writer_exception_fails_open_and_records_failed_attempt(tmp_path, monkeypatch):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )

    def explode(_session):
        raise OSError("disk unavailable")

    monkeypatch.setattr(manager, "_publish_envelope", explode)
    manager.set_enabled(True)
    manager.finish_cycle(_complete_session(manager))
    assert manager.set_enabled(False, timeout_seconds=5.0)

    manifest = manager.read_manifest()
    assert manifest["state"] == "closed"
    assert manifest["failed"] == 1
    assert manifest["completed"] == 0
    assert manifest["attempts"][0]["status"] == "failed"
    assert manifest["attempts"][0]["eligible"] is False
    assert "disk unavailable" in manifest["attempts"][0]["error"]
    assert _capture_files(tmp_path / "revenue_ops_fee_replay") == []


def test_rotation_keeps_newest_32_successful_cycles(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    for seq in range(1, 34):
        manager.finish_cycle(_complete_session(manager, padding=str(seq)))
        _wait_until(lambda: manager.read_manifest()["completed"] == seq)
    assert manager.set_enabled(False, timeout_seconds=10.0)

    files = _capture_files(tmp_path / "revenue_ops_fee_replay")
    assert len(files) == 32
    assert not any("-00000001." in path.name for path in files)
    assert any("-00000033." in path.name for path in files)


def test_rotation_enforces_total_byte_cap_after_successful_publication(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    manager.finish_cycle(_complete_session(manager, padding="x" * 256))
    _wait_until(lambda: manager.read_manifest()["completed"] == 1)

    output_dir = tmp_path / "revenue_ops_fee_replay"
    first_size = _capture_files(output_dir)[0].stat().st_size
    monkeypatch.setattr(capture, "RETENTION_MAX_BYTES", first_size * 2 + 128)

    manager.finish_cycle(_complete_session(manager, padding="y" * 256))
    manager.finish_cycle(_complete_session(manager, padding="z" * 256))
    assert manager.set_enabled(False, timeout_seconds=5.0)

    files = _capture_files(output_dir)
    assert len(files) == 2
    assert sum(path.stat().st_size for path in files) <= capture.RETENTION_MAX_BYTES
    assert any("-00000003." in path.name for path in files)


def test_complete_serialized_envelope_over_cap_is_failed_and_ineligible(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    session = _complete_session(manager)
    body = session.to_body()
    sealed = seal_envelope(body)
    serialized = json.dumps(
        sealed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    body_size = len(canonical_body_bytes(body))
    assert body_size < len(serialized)
    monkeypatch.setattr(capture, "MAX_ENVELOPE_BYTES", body_size)

    manager.finish_cycle(session)
    assert manager.set_enabled(False, timeout_seconds=5.0)

    manifest = manager.read_manifest()
    assert manifest["failed"] == 1
    assert manifest["completed"] == 0
    assert manifest["attempts"][0]["status"] == "failed"
    assert manifest["attempts"][0]["eligible"] is False
    assert "complete sealed envelope" in manifest["attempts"][0]["error"]
    assert _capture_files(tmp_path / "revenue_ops_fee_replay") == []


def test_disable_timeout_leaves_manifest_draining_until_writer_finishes(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    entered = threading.Event()
    release = threading.Event()
    original_publish = manager._publish_envelope

    def blocked_publish(session):
        entered.set()
        assert release.wait(5.0)
        return original_publish(session)

    monkeypatch.setattr(manager, "_publish_envelope", blocked_publish)
    manager.set_enabled(True)
    manager.finish_cycle(_complete_session(manager))
    assert entered.wait(5.0)

    assert manager.set_enabled(False, timeout_seconds=0.01) is False
    manifest = manager.read_manifest()
    assert manifest["state"] == "draining"
    assert manifest["queue_drained"] is False

    release.set()
    assert manager.set_enabled(False, timeout_seconds=5.0)
    assert manager.read_manifest()["state"] == "closed"


def test_restart_uses_unique_run_ids_manifests_and_capture_names(tmp_path):
    database_path = tmp_path / "revenue_ops.db"
    run_ids = []
    for _ in range(2):
        manager = FeeCycleCaptureManager(
            database_path, lambda *_args, **_kw: None
        )
        manager.set_enabled(True)
        session = _complete_session(manager)
        run_ids.append(session.capture_run_id)
        manager.finish_cycle(session)
        assert manager.set_enabled(False, timeout_seconds=5.0)

    output_dir = tmp_path / "revenue_ops_fee_replay"
    manifests = list(output_dir.glob("manifest-*.v0.json"))
    captures = _capture_files(output_dir)
    assert len(set(run_ids)) == 2
    assert len(manifests) == 2
    assert len(captures) == 2
    assert len({path.name for path in manifests + captures}) == 4
