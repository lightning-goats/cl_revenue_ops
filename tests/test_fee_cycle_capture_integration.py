import threading

import pytest

import modules.fee_cycle_capture as capture
from modules.fee_cycle_capture import (
    FeeCycleCaptureSession,
    bind_capture,
    current_capture,
    decision_gauss,
    decision_now,
    decision_random,
    record_effective_evidence,
)


@pytest.fixture
def capture_session():
    return FeeCycleCaptureSession(
        capture_run_id="a" * 32,
        capture_seq=1,
        cycle_id=f"{'a' * 32}:00000001",
        producer={"started_at": "2026-07-18T00:00:00+00:00"},
        configuration={"version": 1},
    )


def test_decision_context_delegates_and_records(monkeypatch, capture_session):
    monkeypatch.setattr(capture.time, "time", lambda: 1234.9)
    monkeypatch.setattr(capture.random, "random", lambda: 0.25)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, sigma: mu + sigma)

    with bind_capture(capture_session):
        assert decision_now("pid.calculate") == 1234
        assert decision_random("vegas.boost") == 0.25
        assert decision_gauss("thompson.posterior", 10.0, 2.0) == 12.0

    assert capture_session.observations["clock"] == [
        {"ordinal": 0, "label": "pid.calculate", "value": 1234}
    ]
    assert capture_session.observations["entropy"] == [
        {
            "ordinal": 0,
            "op": "random",
            "label": "vegas.boost",
            "args": [],
            "result": 0.25,
        },
        {
            "ordinal": 1,
            "op": "gauss",
            "label": "thompson.posterior",
            "args": [10.0, 2.0],
            "result": 12.0,
        },
    ]


def test_decision_context_does_not_leak_to_new_thread(capture_session):
    observed = []

    with bind_capture(capture_session):
        thread = threading.Thread(target=lambda: observed.append(current_capture()))
        thread.start()
        thread.join()
        assert current_capture() is capture_session

    assert observed == [None]
    assert current_capture() is None


def test_effective_evidence_records_result_and_preserves_exception(capture_session):
    result = {"rows": [1]}

    with bind_capture(capture_session):
        assert record_effective_evidence(
            "channel_states", {"limit": 1}, lambda: result
        ) is result
        result["rows"].append(2)

        original = ValueError("database unavailable")
        with pytest.raises(ValueError) as raised:
            record_effective_evidence(
                "flow_window",
                ["1x1x1", 1234],
                lambda: (_ for _ in ()).throw(original),
            )

    assert raised.value is original
    assert capture_session.observations["evidence"] == [
        {
            "ordinal": 0,
            "op": "channel_states",
            "args": {"limit": 1},
            "result": {"rows": [1]},
        },
        {
            "ordinal": 1,
            "op": "flow_window",
            "args": ["1x1x1", 1234],
            "error": {
                "category": "ValueError",
                "message": "database unavailable",
            },
        },
    ]


def test_recording_failure_never_changes_delegated_result(capture_session, monkeypatch):
    monkeypatch.setattr(capture.time, "time", lambda: 1234.9)

    def fail_record(*_args, **_kwargs):
        raise TypeError("malformed recorder")

    monkeypatch.setattr(capture_session, "record_observation", fail_record)
    sentinel = object()

    with bind_capture(capture_session):
        assert decision_now("pid.calculate") == 1234
        assert record_effective_evidence(
            "channel_states", object(), lambda: sentinel
        ) is sentinel

    assert capture_session.invalid_reason == "capture recorder failure: TypeError"


def test_malformed_bound_session_does_not_change_authority_behavior(monkeypatch):
    monkeypatch.setattr(capture.time, "time", lambda: 999.8)
    monkeypatch.setattr(capture.random, "random", lambda: 0.75)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, sigma: mu - sigma)

    with bind_capture(object()):
        assert decision_now("pid.calculate") == 999
        assert decision_random("vegas.boost") == 0.75
        assert decision_gauss("thompson.posterior", 10.0, 2.0) == 8.0
        assert record_effective_evidence("channel_states", None, lambda: 7) == 7


def test_no_bound_capture_preserves_delegation(monkeypatch):
    monkeypatch.setattr(capture.time, "time", lambda: 4321.7)
    monkeypatch.setattr(capture.random, "random", lambda: 0.125)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, sigma: mu + 2 * sigma)

    assert current_capture() is None
    assert decision_now("pid.calculate") == 4321
    assert decision_random("vegas.boost") == 0.125
    assert decision_gauss("thompson.posterior", 10.0, 2.0) == 14.0
    assert record_effective_evidence("channel_states", {}, lambda: {"ok": True}) == {
        "ok": True
    }
