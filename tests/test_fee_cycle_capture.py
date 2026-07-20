import json
import os
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


def test_authority_side_input_errors_do_not_call_logger(tmp_path):
    logged = []
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *args, **kwargs: logged.append(
            (args, kwargs)
        )
    )
    manager.set_enabled(True)

    assert manager.set_enabled(False, timeout_seconds="not-a-timeout") is False
    assert logged == []

    assert manager.set_enabled(False, timeout_seconds=5.0)


def test_session_records_copies_and_context_binding():
    session = FeeCycleCaptureSession(
        capture_run_id="run-a",
        capture_seq=1,
        cycle_id="run-a:1",
        producer={"started_at": "2026-07-18T00:00:00+00:00"},
        configuration={"version": 1},
    )
    identity = {"channel_id": "1x1x1", "peer_id": "peer-a"}
    pre_state = {"global": {}, "ordered_channels": [dict(identity)]}
    observation = {"channel_id": "1x1x1", "value": 1.5}
    expected = {
        "ordered_outcomes": [{**identity, "skip": {"reason": "test"}}],
        "ordered_decision_traces": [
            {
                **identity,
                "terminal_kind": "skip",
                "terminal_reason": "test",
                "decision_source": "test",
                "current_fee_ppm": 100,
                "target_fee_ppm": None,
                "applied_fee_ppm": 100,
                "algorithm_values": None,
                "governor": [],
                "execution": [],
            }
        ],
        "post_global": {},
        "post_channel_state": [dict(identity)],
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
    assert body["pre_state"]["ordered_channels"] == [identity]
    assert body["observations"]["evidence"][0]["value"] == 1.5
    assert body["expected"]["ordered_outcomes"] == [
        {**identity, "skip": {"reason": "test"}}
    ]
    assert body["completeness"] == {
        "evaluated_channels": 1,
        "terminal_outcomes": 1,
        "decision_trace_entries": 1,
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
    assert captures[0].name == (
        f"{session.capture_run_id}-{session.capture_seq:08d}-{session.cycle_id}.json"
    )
    assert manifests[0].name == (
        f"manifest-{session.capture_run_id}.v{capture.SCHEMA_VERSION}.json"
    )
    assert output_dir.stat().st_mode & 0o777 == 0o700
    assert captures[0].stat().st_mode & 0o777 == 0o600
    assert manifests[0].stat().st_mode & 0o777 == 0o600
    assert not list(output_dir.glob("*.tmp"))
    assert not list(output_dir.glob(".*.tmp"))


def test_first_output_creation_fsyncs_parent_and_atomic_publication_fsyncs_output(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    fsynced_directories = []
    original_fsync_directory = manager._fsync_directory

    def observe_fsync_directory(directory=None):
        target = manager.output_dir if directory is None else Path(directory)
        fsynced_directories.append(target)
        if directory is None:
            return original_fsync_directory()
        return original_fsync_directory(directory)

    monkeypatch.setattr(manager, "_fsync_directory", observe_fsync_directory)
    manager.set_enabled(True)
    output_dir = tmp_path / "revenue_ops_fee_replay"
    try:
        _wait_until(lambda: bool(list(output_dir.glob("manifest-*.json"))))
        assert tmp_path in fsynced_directories
        assert output_dir in fsynced_directories
    finally:
        manager.set_enabled(False, timeout_seconds=5.0)


def test_manifest_exposes_complete_active_run_state(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    output_dir = tmp_path / "revenue_ops_fee_replay"
    _wait_until(lambda: bool(list(output_dir.glob("manifest-*.json"))))

    manifest = manager.read_manifest()
    assert manifest["state"] == "active"
    assert manifest["attempted"] == 0
    assert manifest["completed"] == 0
    assert manifest["failed"] == 0
    assert manifest["dropped"] == 0
    assert manifest["last_attempted_seq"] is None
    assert manifest["last_completed_seq"] is None
    assert manifest["retained_sequence_range"] == {"first": None, "last": None}
    assert manifest["writer_health"] == "healthy"
    assert manifest["last_error_category"] is None
    assert manifest["queue_drained"] is False

    assert manager.set_enabled(False, timeout_seconds=5.0)


def test_finish_cycle_never_waits_for_manifest_storage(tmp_path, monkeypatch):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    output_dir = tmp_path / "revenue_ops_fee_replay"
    _wait_until(lambda: bool(list(output_dir.glob("manifest-*.json"))))
    session = _complete_session(manager)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_atomic_write = manager._atomic_write

    def blocked_manifest_write(destination, payload):
        if destination.name.startswith("manifest-"):
            entered.set()
            assert release.wait(5.0)
        return original_atomic_write(destination, payload)

    monkeypatch.setattr(manager, "_atomic_write", blocked_manifest_write)
    authority = threading.Thread(
        target=lambda: (manager.finish_cycle(session), finished.set())
    )
    authority.start()
    try:
        assert entered.wait(5.0)
        assert finished.wait(0.1), "finish_cycle blocked on durable manifest I/O"
    finally:
        release.set()
        authority.join(5.0)
        manager.set_enabled(False, timeout_seconds=5.0)


def test_enable_only_notifies_writer_when_manifest_storage_is_blocked(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_atomic_write = manager._atomic_write

    def blocked_manifest_write(destination, payload):
        if destination.name.startswith("manifest-"):
            entered.set()
            assert release.wait(5.0)
        return original_atomic_write(destination, payload)

    monkeypatch.setattr(manager, "_atomic_write", blocked_manifest_write)
    authority = threading.Thread(
        target=lambda: (manager.set_enabled(True), finished.set())
    )
    authority.start()
    try:
        assert finished.wait(0.1), "enable blocked on durable manifest I/O"
        assert entered.wait(5.0)
    finally:
        release.set()
        authority.join(5.0)
        manager.set_enabled(False, timeout_seconds=5.0)


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


def test_queue_drop_never_waits_for_manifest_storage(tmp_path, monkeypatch):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    envelope_entered = threading.Event()
    envelope_release = threading.Event()
    manifest_release = threading.Event()
    finished = threading.Event()
    original_publish = manager._publish_envelope
    original_atomic_write = manager._atomic_write

    def blocked_envelope(session):
        envelope_entered.set()
        assert envelope_release.wait(5.0)
        return original_publish(session)

    def blocked_manifest_write(destination, payload):
        if destination.name.startswith("manifest-"):
            assert manifest_release.wait(5.0)
        return original_atomic_write(destination, payload)

    monkeypatch.setattr(manager, "_publish_envelope", blocked_envelope)
    manager.set_enabled(True)
    manager.finish_cycle(_complete_session(manager, padding="active"))
    assert envelope_entered.wait(5.0)
    manager.finish_cycle(_complete_session(manager, padding="pending-1"))
    manager.finish_cycle(_complete_session(manager, padding="pending-2"))
    monkeypatch.setattr(manager, "_atomic_write", blocked_manifest_write)

    authority = threading.Thread(
        target=lambda: (
            manager.finish_cycle(_complete_session(manager, padding="dropped")),
            finished.set(),
        )
    )
    authority.start()
    try:
        assert finished.wait(0.1), "queue drop blocked on durable manifest I/O"
    finally:
        manifest_release.set()
        authority.join(5.0)
        envelope_release.set()
        manager.set_enabled(False, timeout_seconds=5.0)


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


def test_writer_rejects_invalid_session_without_publishing_capture(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    session = _complete_session(manager)
    session.mark_invalid("capture recorder failure: TypeError")

    manager.finish_cycle(session)
    assert manager.set_enabled(False, timeout_seconds=5.0)

    manifest = manager.read_manifest()
    assert manifest["attempted"] == 1
    assert manifest["completed"] == 0
    assert manifest["failed"] == 1
    assert manifest["last_completed_seq"] is None
    assert manifest["attempts"] == [
        {
            "capture_seq": 1,
            "cycle_id": session.cycle_id,
            "status": "failed",
            "eligible": False,
            "error_category": "invalid_session",
            "error": "capture session is invalid",
        }
    ]
    assert _capture_files(tmp_path / "revenue_ops_fee_replay") == []


def test_writer_rejects_outcome_count_mismatch_without_publishing_capture(
    tmp_path,
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    session = manager.begin_cycle({"version": 1}, {"python_commit": "abc"})
    assert session is not None
    session.record_pre_state(
        {
            "global": {},
            "ordered_channels": [
                {"channel_id": "1x1x1", "peer_id": "02" + "a" * 64}
            ],
        }
    )
    session.record_expected(
        {
            "ordered_outcomes": [],
            "ordered_decision_traces": [],
            "post_global_state": {},
            "post_channel_state": [],
        }
    )

    manager.finish_cycle(session)
    assert manager.set_enabled(False, timeout_seconds=5.0)

    manifest = manager.read_manifest()
    assert manifest["attempted"] == 1
    assert manifest["completed"] == 0
    assert manifest["failed"] == 1
    assert manifest["last_completed_seq"] is None
    assert manifest["attempts"][0]["status"] == "failed"
    assert manifest["attempts"][0]["eligible"] is False
    assert manifest["attempts"][0]["error_category"] == "count_mismatch"
    assert manifest["attempts"][0]["error"] == (
        "evaluated channel, terminal outcome, decision trace, and "
        "post-state counts differ"
    )
    assert _capture_files(tmp_path / "revenue_ops_fee_replay") == []


def test_pending_attempt_is_ineligible_until_envelope_is_durable(
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

    try:
        attempt = manager.read_manifest()["attempts"][0]
        assert attempt["status"] in {"queued", "writing"}
        assert attempt["eligible"] is False
    finally:
        release.set()
        manager.set_enabled(False, timeout_seconds=5.0)


def test_disable_waits_for_begun_session_and_accepts_its_late_finish(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    session = _complete_session(manager)

    assert manager.set_enabled(False, timeout_seconds=0.01) is False
    manifest = manager.read_manifest()
    assert manifest["state"] == "draining"
    assert manifest["queue_drained"] is False

    manager.finish_cycle(session)
    assert manager.set_enabled(False, timeout_seconds=5.0)
    manifest = manager.read_manifest()
    assert manifest["state"] == "closed"
    assert manifest["queue_drained"] is True
    assert manifest["completed"] == 1


def test_begin_cycle_allows_only_one_unfinished_session_and_late_finish(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    first = _complete_session(manager)
    manifest_before_second = manager.read_manifest()

    second = manager.begin_cycle(
        {"version": 1, "padding": "duplicate-active"},
        {"python_commit": "abc"},
    )

    assert second is None
    manifest_after_second = manager.read_manifest()
    assert manifest_after_second["attempted"] == manifest_before_second["attempted"] == 1
    assert manifest_after_second["last_attempted_seq"] == 1
    assert len(manifest_after_second["attempts"]) == 1

    assert manager.set_enabled(False, timeout_seconds=0.01) is False
    manager.finish_cycle(first)
    assert manager.set_enabled(False, timeout_seconds=5.0)
    assert manager.read_manifest()["completed"] == 1


def test_disable_acknowledges_close_only_after_closed_manifest_is_durable(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    output_dir = tmp_path / "revenue_ops_fee_replay"
    _wait_until(lambda: bool(list(output_dir.glob("manifest-*.json"))))
    original_atomic_write = manager._atomic_write

    def reject_closed_manifest(destination, payload):
        if destination.name.startswith("manifest-"):
            manifest = json.loads(payload)
            if manifest["state"] == "closed":
                raise OSError("closed manifest unavailable")
        return original_atomic_write(destination, payload)

    monkeypatch.setattr(manager, "_atomic_write", reject_closed_manifest)
    assert manager.set_enabled(False, timeout_seconds=0.05) is False
    assert manager.read_manifest()["state"] != "closed"

    monkeypatch.setattr(manager, "_atomic_write", original_atomic_write)
    assert manager.set_enabled(False, timeout_seconds=5.0)
    assert manager.read_manifest()["state"] == "closed"


def test_output_directory_is_fixed_sibling_of_database_parent(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "custom-name.sqlite3", lambda *_args, **_kw: None
    )

    assert manager.output_dir == tmp_path / "revenue_ops_fee_replay"
    manager.set_enabled(True)
    manager.finish_cycle(_complete_session(manager))
    assert manager.set_enabled(False, timeout_seconds=5.0)
    assert (tmp_path / "revenue_ops_fee_replay").is_dir()
    assert not (tmp_path / "custom-name_fee_replay").exists()


def test_closed_worker_terminates_and_reenable_starts_fresh_worker(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    first_worker = manager._writer_thread
    assert first_worker is not None

    assert manager.set_enabled(False, timeout_seconds=5.0)
    first_worker.join(1.0)
    assert not first_worker.is_alive()

    assert manager.set_enabled(True)
    second_worker = manager._writer_thread
    assert second_worker is not None
    assert second_worker is not first_worker
    assert second_worker.is_alive()

    assert manager.set_enabled(False, timeout_seconds=5.0)
    second_worker.join(1.0)
    assert not second_worker.is_alive()


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
    assert not any("-00000001-" in path.name for path in files)
    assert any("-00000033-" in path.name for path in files)
    manifest = manager.read_manifest()
    assert manifest["completed"] == 33
    assert len(manifest["attempts"]) == 32
    assert [attempt["capture_seq"] for attempt in manifest["attempts"]] == list(
        range(2, 34)
    )


def test_retention_ignores_json_outside_manager_envelope_identity(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    session = _complete_session(manager)
    manager.finish_cycle(session)
    assert manager.set_enabled(False, timeout_seconds=5.0)

    output_dir = tmp_path / "revenue_ops_fee_replay"
    valid_capture = _capture_files(output_dir)[0]
    manifest = (
        output_dir
        / f"manifest-{session.capture_run_id}.v{capture.SCHEMA_VERSION}.json"
    )
    ordinary_json = output_dir / "operator-notes.json"
    invalid_contract_shape = output_dir / "notes-00000001-backup.json"
    ordinary_json.write_text("{}", encoding="utf-8")
    invalid_contract_shape.write_text("{}", encoding="utf-8")
    os.utime(invalid_contract_shape, ns=(1, 1))
    os.utime(valid_capture, ns=(2, 2))
    monkeypatch.setattr(capture, "RETENTION_MAX_FILES", 1)

    assert manager._capture_identity(valid_capture) == (
        session.capture_run_id,
        session.capture_seq,
    )
    manager._rotate_capture_files()

    assert valid_capture.exists()
    assert manifest.exists()
    assert ordinary_json.exists()
    assert invalid_contract_shape.exists()


def test_close_does_not_rotate_or_evict_retained_envelope_at_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(capture, "RETENTION_MAX_FILES", 1)
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    manager.finish_cycle(_complete_session(manager))
    _wait_until(lambda: manager.read_manifest()["completed"] == 1)

    retained = _capture_files(tmp_path / "revenue_ops_fee_replay")
    assert len(retained) == capture.RETENTION_MAX_FILES
    retained_path = retained[0]

    def destructive_close_rotation():
        retained_path.unlink()
        return {}

    monkeypatch.setattr(manager, "_rotate_capture_files", destructive_close_rotation)
    assert manager.set_enabled(False, timeout_seconds=5.0)
    assert retained_path.exists()


def test_attempt_rotation_preserves_aggregate_failure_truth(tmp_path, monkeypatch):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    original_publish = manager._publish_envelope

    def fail_first(session):
        if session.capture_seq == 1:
            raise OSError("first capture failed")
        return original_publish(session)

    monkeypatch.setattr(manager, "_publish_envelope", fail_first)
    manager.set_enabled(True)
    manager.finish_cycle(_complete_session(manager, padding="failed"))
    _wait_until(lambda: manager.read_manifest()["failed"] == 1)

    for seq in range(2, 34):
        manager.finish_cycle(_complete_session(manager, padding=str(seq)))
        _wait_until(lambda: manager.read_manifest()["completed"] == seq - 1)
    assert manager.set_enabled(False, timeout_seconds=10.0)

    manifest = manager.read_manifest()
    assert manifest["attempted"] == 33
    assert manifest["completed"] == 32
    assert manifest["failed"] == 1
    assert manifest["dropped"] == 0
    assert manifest["last_attempted_seq"] == 33
    assert manifest["last_completed_seq"] == 33
    assert manifest["retained_sequence_range"] == {"first": 2, "last": 33}
    assert [attempt["capture_seq"] for attempt in manifest["attempts"]] == list(
        range(1, 34)
    )


def test_failed_attempt_after_32_successes_keeps_all_retained_metadata(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    original_publish = manager._publish_envelope
    manager.set_enabled(True)
    for seq in range(1, 33):
        manager.finish_cycle(_complete_session(manager, padding=str(seq)))
        _wait_until(lambda: manager.read_manifest()["completed"] == seq)

    def fail_last(_session):
        raise OSError("last capture failed")

    monkeypatch.setattr(manager, "_publish_envelope", fail_last)
    manager.finish_cycle(_complete_session(manager, padding="failed"))
    _wait_until(lambda: manager.read_manifest()["failed"] == 1)
    assert manager.set_enabled(False, timeout_seconds=10.0)

    manifest = manager.read_manifest()
    assert manifest["attempted"] == 33
    assert manifest["completed"] == 32
    assert manifest["failed"] == 1
    assert manifest["retained_sequence_range"] == {"first": 1, "last": 32}
    assert [attempt["capture_seq"] for attempt in manifest["attempts"]] == list(
        range(1, 34)
    )
    assert manifest["attempts"][-1]["status"] == "failed"
    assert manifest["attempts"][-1]["eligible"] is False

    monkeypatch.setattr(manager, "_publish_envelope", original_publish)


def test_terminal_failure_detail_is_bounded_beside_retained_completions(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    manager.set_enabled(True)
    for seq in range(1, 33):
        manager.finish_cycle(_complete_session(manager, padding=str(seq)))
        _wait_until(lambda: manager.read_manifest()["completed"] == seq)

    def explode(_session):
        raise OSError("persistent writer failure")

    monkeypatch.setattr(manager, "_publish_envelope", explode)
    for failed_count in range(1, 34):
        manager.finish_cycle(_complete_session(manager, padding="failed"))
        _wait_until(lambda: manager.read_manifest()["failed"] == failed_count)
    assert manager.set_enabled(False, timeout_seconds=10.0)

    manifest = manager.read_manifest()
    assert manifest["attempted"] == 65
    assert manifest["completed"] == 32
    assert manifest["failed"] == 33
    assert len(manifest["attempts"]) == 64
    assert [
        attempt["capture_seq"]
        for attempt in manifest["attempts"]
        if attempt["status"] == "completed"
    ] == list(range(1, 33))
    assert [
        attempt["capture_seq"]
        for attempt in manifest["attempts"]
        if attempt["status"] == "failed"
    ] == list(range(34, 66))


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
    assert any("-00000003-" in path.name for path in files)


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


def test_retention_prunes_orphaned_empty_and_failed_run_manifests(
    tmp_path, monkeypatch
):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )

    manager.set_enabled(True)
    empty_run_id = manager._capture_run_id
    assert manager.set_enabled(False, timeout_seconds=5.0)

    original_publish = manager._publish_envelope

    def explode(_session):
        raise OSError("failed-only run")

    monkeypatch.setattr(manager, "_publish_envelope", explode)
    manager.set_enabled(True)
    failed_run_id = manager._capture_run_id
    manager.finish_cycle(_complete_session(manager))
    assert manager.set_enabled(False, timeout_seconds=5.0)

    monkeypatch.setattr(manager, "_publish_envelope", original_publish)
    manager.set_enabled(True)
    retained_run_id = manager._capture_run_id
    manager.finish_cycle(_complete_session(manager))
    assert manager.set_enabled(False, timeout_seconds=5.0)

    manifests = list(
        (tmp_path / "revenue_ops_fee_replay").glob("manifest-*.v0.json")
    )
    manifest_run_ids = {
        json.loads(path.read_text(encoding="utf-8"))["capture_run_id"]
        for path in manifests
    }
    assert manifest_run_ids == {retained_run_id}
    assert empty_run_id not in manifest_run_ids
    assert failed_run_id not in manifest_run_ids


def test_retention_keeps_manifest_for_each_globally_retained_envelope(tmp_path):
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kw: None
    )
    first_run_id = None
    for _ in range(33):
        manager.set_enabled(True)
        if first_run_id is None:
            first_run_id = manager._capture_run_id
        manager.finish_cycle(_complete_session(manager))
        assert manager.set_enabled(False, timeout_seconds=5.0)

    output_dir = tmp_path / "revenue_ops_fee_replay"
    captures = _capture_files(output_dir)
    manifests = list(output_dir.glob("manifest-*.v0.json"))
    capture_run_ids = {
        json.loads(path.read_text(encoding="utf-8"))["capture_run_id"]
        for path in captures
    }
    manifest_run_ids = {
        json.loads(path.read_text(encoding="utf-8"))["capture_run_id"]
        for path in manifests
    }
    assert len(captures) == 32
    assert len(manifests) == 32
    assert manifest_run_ids == capture_run_ids
    assert first_run_id not in manifest_run_ids
