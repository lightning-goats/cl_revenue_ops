import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.rebalance_cycle_capture import RebalanceCycleCaptureManager, project_cycle_result
from modules.rebalance_cycle_replay_wire import seal_envelope, verify_envelope
from modules.rebalance_engine_v2 import CycleResult
from modules.rebalance_types_v2 import PairCandidate
from modules.rebalance_execution import ExecutionResult
from modules.rebalance_state_v2 import ChannelState, StateSnapshot


def _configuration():
    return {
        "config_version": 1,
        "target_band_low": 0.35,
        "target_band_high": 0.65,
        "max_chunk_sats": 2_000_000,
        "max_pairs": 1,
        "pair_fee_cap_ppm": 1_000,
    }


def _pair(source="a", dest="b", rank=1):
    return PairCandidate(
        source_channel_id=source,
        dest_channel_id=dest,
        source_peer_id=f"{source}-peer",
        dest_peer_id=f"{dest}-peer",
        amount_sats=100,
        pair_budget_sats=20,
        source_excess_sats=100,
        dest_need_sats=100,
        max_chunk_sats=100,
        cheap_rank=rank,
        planner_selected=True,
    )


def test_disabled_capture_creates_no_directory(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)

    assert manager.begin_cycle(_configuration(), {"trigger": "automatic"}) is None
    assert not manager.output_dir.exists()


def test_enabled_capture_assigns_identity_and_seals_projected_cycle(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    assert reference is not None
    assert reference.capture_seq == 1
    assert reference.cycle_id == f"{reference.capture_run_id}:00000001"
    result = _strict_result(outcome={
        "source_channel_id": "a", "dest_channel_id": "b",
        "result": {"success": True},
    })

    manager.finish_cycle(reference, result)
    assert manager.set_enabled(False, timeout_seconds=2.0)
    import json
    envelope_paths = [
        path for path in manager.output_dir.glob("*.json")
        if not path.name.startswith("manifest-")
    ]
    assert len(envelope_paths) == 1
    envelope = json.loads(envelope_paths[0].read_text())
    assert envelope["funnel"]["generated_pairs"][0]["cheap_rank"] == 1
    assert envelope["execution"]["pair_outcomes"][0]["source_channel_id"] == "a"
    verify_envelope(envelope)

def test_pair_outcomes_keep_completed_future_pair_identity():
    first = _pair("source-a", "dest-a", 1)
    second = _pair("source-b", "dest-b", 2)
    result = CycleResult(
        considered_candidates=[first, second], candidates=[first, second],
        pair_outcomes=[
            {"source_channel_id": "source-b", "dest_channel_id": "dest-b", "result": {"success": True}},
            {"source_channel_id": "source-a", "dest_channel_id": "dest-a", "result": {"success": False}},
        ],
        snapshot=_strict_snapshot(),
        planner_bootstrap_evidence=[
            {"source_channel_id": p.source_channel_id, "dest_channel_id": p.dest_channel_id, "score_decomposition": {}}
            for p in (first, second)
        ],
    )

    body = project_cycle_result(_strict_reference(), result)

    assert [(row["source_channel_id"], row["dest_channel_id"]) for row in body["execution"]["pair_outcomes"]] == [
        ("source-b", "dest-b"), ("source-a", "dest-a"),
    ]

def test_finish_is_no_throw_when_writer_fails(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    ref = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    monkeypatch.setattr(manager, "_publish_envelope", lambda *_a: (_ for _ in ()).throw(OSError("blocked")))

    manager.finish_cycle(ref, CycleResult())

    assert manager.read_manifest()["failed"] >= 0


def test_projection_bounds_malformed_failure_metadata():
    with pytest.raises(ValueError, match="absent from final selection"):
        project_cycle_result(
            _strict_reference(),
            CycleResult(
                snapshot=_strict_snapshot(),
                pair_outcomes=[{"source_channel_id": "x", "dest_channel_id": "y", "result": object()}],
            ),
        )

def test_projection_keeps_complete_explicit_normalized_snapshot():
    channel = ChannelState(
        channel_id="a", peer_id="peer-a", capacity_sats=1_000, local_ratio=0.8,
        actual_inbound_fee_ppm=12, value_class="profitable", is_valuable=True,
        remaining_budget_sats=0, cooldown_active=True, source_eligible=True,
        dest_eligible=False, source_reason="", dest_reason="cooldown", dest_urgency=0.2,
        source_drain_score=0.4, budget_source="capex", local_out_fee_ppm=99,
        historical_direct_fee_ppm=1.5, historical_sourced_fee_ppm=2.5, is_active=True,
        realized_utilization=0.7, utilization_is_realized=True, activity_out_sats=3,
        activity_in_sats=4, target_band_low=0.3, target_band_high=0.7,
    )
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=_strict_reference().producer)

    body = project_cycle_result(reference, CycleResult(snapshot=StateSnapshot((channel,), 1_000, 0, 1)))

    captured = body["pre_state"]["normalized_snapshot"]["channels"][0]
    assert captured["peer_id"] == "peer-a"
    assert captured["remaining_budget_sats"] == 0
    assert captured["target_band_low"] == 0.3
    assert captured["utilization_is_realized"] is True


def test_projection_keeps_real_execution_result_with_explicit_allowlist():
    pair = _pair()
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=_strict_reference().producer)
    execution = ExecutionResult(success=True, amount_sats=100, fee_sats=2, route_type="native", failure_data={"secret": "not-copied"})

    body = project_cycle_result(reference, _strict_result(pair=pair, outcome={"source_channel_id": "a", "dest_channel_id": "b", "result": execution}))

    result = body["execution"]["pair_outcomes"][0]["result"]
    assert result == {"success": True, "amount_sats": 100, "fee_sats": 2, "fee_msat": 0, "fee_ppm": 0, "attempts": 0, "hops": 0, "parts": 1, "route_type": "native", "error": "", "excluded_channels": [], "failure_data": {}, "payment_pending": False}


def test_manifest_attempts_are_bounded(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    for _ in range(40):
        reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
        manager.finish_cycle(reference, CycleResult())
    assert len(manager.read_manifest()["attempts"]) <= 32


def test_projection_preserves_zero_budget_and_disabled_fee_cap():
    pair = _pair()
    pair.pair_budget_sats = 0
    configuration = _configuration()
    configuration["pair_fee_cap_ppm"] = 0
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=configuration, producer=_strict_reference().producer)

    body = project_cycle_result(reference, _strict_result(pair=pair))

    assert body["configuration"]["pair_fee_cap_ppm"] == 0
    assert body["funnel"]["generated_pairs"][0]["pair_budget_sats"] == 0
    seal_envelope(body)


class _CaptureManager:
    def __init__(self):
        self.begins = []
        self.finishes = []

    def begin_cycle(self, configuration, producer):
        self.begins.append((configuration, producer))
        return SimpleNamespace(capture_run_id="a" * 32, capture_seq=len(self.begins), cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=producer)

    def finish_cycle(self, reference, result, terminal_stage="completed"):
        self.finishes.append((reference, result, terminal_stage))


def test_finish_enqueues_only_an_immutable_projected_body(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})

    manager.finish_cycle(reference, _strict_result())
    queued = manager._queue.get_nowait()

    assert isinstance(queued[1], bytes)
    assert b"rebalance_cycle_replay" in queued[1]
    assert not hasattr(queued[1], "append")

def test_projection_rejects_duplicate_or_noncontiguous_generated_evidence():
    first = _pair("source-a", "dest-a", 4)
    duplicate = _pair("source-a", "dest-a", 9)
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=_strict_reference().producer)

    with pytest.raises(ValueError):
        project_cycle_result(reference, CycleResult(considered_candidates=[first, duplicate]))


def test_projection_preserves_timeout_status_for_linked_pair():
    pair = _pair()
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=_strict_reference().producer)

    body = project_cycle_result(reference, _strict_result(pair=pair, outcome={"source_channel_id": "a", "dest_channel_id": "b", "status": "still_running_timeout", "result": None}))

    assert body["execution"]["pair_outcomes"] == [{"source_channel_id": "a", "dest_channel_id": "b", "status": "still_running_timeout"}]


def test_retention_leaves_unowned_json_untouched_and_counts_owned_manifests(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    unrelated = manager.output_dir / "operator-note.json"
    unrelated.write_text("{}")
    for index in range(34):
        path = manager.output_dir / (("a" * 32) + f"-{index + 1:08d}-" + ("a" * 32) + f":{index + 1:08d}.json")
        path.write_text("{}")

    manager._rotate_capture_files()

    assert unrelated.exists()
    assert len(list(manager.output_dir.glob("*.json"))) <= 33



def test_engine_capture_configuration_preserves_disabled_pair_fee_cap():
    from modules.rebalance_engine_v2 import RebalanceEngine

    manager = _CaptureManager()
    engine = object.__new__(RebalanceEngine)
    engine.rebalance_capture_manager = manager
    engine.config = SimpleNamespace(snapshot=lambda: SimpleNamespace(low_liquidity_threshold=0.35, high_liquidity_threshold=0.65, rebalance_max_amount=100, pair_fee_cap_ppm=0))
    engine._max_concurrent_jobs = lambda cfg: 1

    engine._begin_rebalance_capture()

    assert manager.begins[0][0]["pair_fee_cap_ppm"] == 0


def test_projection_rejects_unbounded_snapshot_instead_of_eligible_truncation():
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=_strict_reference().producer)
    channels = tuple(ChannelState(channel_id=f"{index}x1x0", peer_id=f"peer-{index}", capacity_sats=1, local_ratio=0.5, actual_inbound_fee_ppm=0, value_class="neutral", is_valuable=False, remaining_budget_sats=0, cooldown_active=False) for index in range(1025))

    with pytest.raises(ValueError):
        project_cycle_result(reference, CycleResult(snapshot=StateSnapshot(channels)))


def test_generated_and_selected_evidence_include_route_quote_and_rejection_fields():
    pair = _pair()
    pair.route = [{"channel": "1x1x0", "amount_msat": 1_000}]
    pair.route_cost_sats = 2
    pair.score_decomposition = {"p_success": 0.9, "final_score_sats": 3.0, "effective_budget_sats": 0}
    pair.rejection_reason = "priced"
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=_strict_reference().producer)

    body = project_cycle_result(reference, _strict_result(pair=pair))

    generated = body["funnel"]["generated_pairs"][0]
    selected = body["funnel"]["final_selected_pairs"][0]
    assert generated["route_summary"] == [{"index": 0, "channel": "1x1x0", "direction": None, "id": "", "amount_msat": 1_000, "delay": None}]
    assert generated["route_cost_sats"] == 2
    assert generated["score_decomposition"]["p_success"] == 0.9
    assert selected["route_cost_sats"] == 2


def test_rotation_enforces_32_total_owned_files_without_deleting_unowned_json(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    unrelated = manager.output_dir / "operator-note.json"
    unrelated.write_text("{}")
    for index in range(40):
        (manager.output_dir / (("a" * 32) + f"-{index + 1:08d}-" + ("a" * 32) + f":{index + 1:08d}.json")).write_text("{}")

    manager._rotate_capture_files()

    owned = [path for path in manager.output_dir.glob("*.json") if path.name != unrelated.name]
    assert unrelated.exists()
    assert len(owned) <= 32



def test_projection_keeps_true_bootstrap_decomposition_separate_from_later_engine_state():
    pair = _pair()
    pair.score_decomposition = {"stage": "priced", "p_success": 0.9}
    result = _strict_result(pair=pair)
    result.planner_bootstrap_evidence[0]["score_decomposition"] = {
        "stage": "planner_pre_route", "p_success": 0.5,
    }

    body = project_cycle_result(_strict_reference(), result)

    generated = body["funnel"]["generated_pairs"][0]
    assert generated["bootstrap_score_decomposition"]["stage"] == "planner_pre_route"
    assert generated["score_decomposition"]["stage"] == "priced"

def test_finish_handoff_freezes_mutable_cycle_evidence_before_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    pair = _pair()
    result = _strict_result(pair=pair)

    manager.finish_cycle(reference, result)
    pair.score = 999.0
    result.candidates.clear()
    queued = manager._queue.get_nowait()

    import json
    frozen = json.loads(queued[1])
    assert len(frozen["funnel"]["final_selected_pairs"]) == 1
    assert frozen["funnel"]["generated_pairs"][0]["cheap_score"] != 999.0

def test_disabled_finish_is_fast_and_does_not_touch_filesystem(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    atomic = MagicMock()
    monkeypatch.setattr(manager, "_atomic_write", atomic)
    started = time.monotonic()
    assert manager.begin_cycle(_configuration(), {"trigger": "automatic"}) is None
    assert time.monotonic() - started < 0.05
    atomic.assert_not_called()
    assert not manager.output_dir.exists()


def test_queue_pressure_uses_the_required_two_slot_bound():
    from modules.rebalance_cycle_capture import WRITER_QUEUE_SIZE
    assert WRITER_QUEUE_SIZE == 2

def _wait_for_failed(manager, category):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        manifest = manager.read_manifest()
        if manifest.get("failed") == 1:
            assert manifest["writer_health"] == "degraded"
            assert manifest["last_error_category"] == category
            return
        time.sleep(0.01)
    raise AssertionError(manager.read_manifest())


@pytest.mark.parametrize("failure, category", [
    (lambda monkeypatch, manager: monkeypatch.setattr("modules.rebalance_cycle_capture.project_cycle_result", lambda *_a: (_ for _ in ()).throw(ValueError("projection"))), "ValueError"),
    (lambda monkeypatch, manager: monkeypatch.setattr(manager, "_atomic_write", lambda *_a: (_ for _ in ()).throw(OSError("write"))), "OSError"),
    (lambda monkeypatch, manager: monkeypatch.setattr("modules.rebalance_cycle_capture.MAX_ENVELOPE_BYTES", 1), "ValueError"),
])
def test_writer_failures_are_manifested_without_raising(tmp_path, monkeypatch, failure, category):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    failure(monkeypatch, manager)
    manager.finish_cycle(manager.begin_cycle(_configuration(), {"trigger": "automatic"}), _strict_result())
    _wait_for_failed(manager, category)



def test_enable_rejects_symlink_output_directory(tmp_path):
    target = tmp_path / "target"; target.mkdir()
    database = tmp_path / "revenue_ops.db"
    output = tmp_path / "revenue_ops_rebalance_replay"
    output.symlink_to(target, target_is_directory=True)
    manager = RebalanceCycleCaptureManager(database, lambda *_a, **_k: None)

    assert manager.set_enabled(True) is False
    assert manager.read_manifest() == {}


def test_atomic_write_rejects_symlink_destination_and_cleans_temp(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    target = manager.output_dir / "target"
    target.write_text("unchanged")
    destination = manager.output_dir / "capture.json"
    destination.symlink_to(target)

    with pytest.raises(OSError):
        manager._atomic_write(destination, b"bad")

    assert target.read_text() == "unchanged"
    assert not list(manager.output_dir.glob(".*.tmp"))



def test_many_enable_disable_cycles_close_without_writer_wedge(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    for _ in range(40):
        assert manager.set_enabled(True)
        assert manager.set_enabled(False, timeout_seconds=1.0)
        assert manager._writer is None or not manager._writer.is_alive()



def test_rotation_bounds_owned_artifacts_and_bytes_including_manifest(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    unrelated = manager.output_dir / "keep.json"; unrelated.write_text("{}")
    for index in range(40):
        path = manager.output_dir / (("a" * 32) + f"-{index + 1:08d}-" + ("a" * 32) + f":{index + 1:08d}.json")
        with path.open("wb") as handle: handle.truncate(10 * 1024 * 1024)
    manager._rotate_capture_files()
    owned = [path for path in manager.output_dir.glob("*.json") if path.name != "keep.json"]
    assert unrelated.exists()
    assert len(owned) <= 32
    assert sum(path.stat().st_size for path in owned) <= 256 * 1024 * 1024



def test_disable_timeout_rolls_back_active_then_clean_disable_reenables(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    old_writer = manager._writer
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    assert manager.set_enabled(False, timeout_seconds=0.0) is False
    second = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    assert second is not None
    manager.finish_cycle(reference, _strict_result(), "failed")
    manager.finish_cycle(second, _strict_result(), "failed")
    assert manager.set_enabled(False, timeout_seconds=1.0)
    assert old_writer is None or not old_writer.is_alive()
    assert manager.set_enabled(True)
    assert manager._writer is not old_writer

def _strict_reference(trigger="automatic"):
    return SimpleNamespace(
        capture_run_id="a" * 32,
        capture_seq=1,
        cycle_id=("a" * 32) + ":00000001",
        configuration=_configuration(),
        producer={
            "started_at": "2026-08-20T18:00:00+00:00",
            "completed_at": "2026-08-20T18:00:01+00:00",
            "python_commit": "abc123",
            "algorithm_version": "rebalance-v2-phase1a",
            "trigger": trigger,
        },
    )


def _strict_snapshot():
    channel = ChannelState(
        channel_id="a", peer_id="peer-a", capacity_sats=1_000,
        local_ratio=0.8, actual_inbound_fee_ppm=12,
        value_class="profitable", is_valuable=True,
        remaining_budget_sats=20, cooldown_active=False,
    )
    return StateSnapshot((channel,), 1_000, 20, 1)


def _strict_result(*, pair=None, outcome=None):
    pair = pair or _pair()
    outcomes = [] if outcome is None else [outcome]
    return CycleResult(
        considered_candidates=[pair],
        candidates=[pair],
        pair_outcomes=outcomes,
        snapshot=_strict_snapshot(),
        planner_bootstrap_evidence=[{
            "source_channel_id": pair.source_channel_id,
            "dest_channel_id": pair.dest_channel_id,
            "score_decomposition": {"stage": "planner_pre_route"},
        }],
    )


def test_projection_rejects_missing_partial_malformed_and_duplicate_snapshot_evidence():
    reference = _strict_reference()
    valid = _strict_snapshot().channels[0]
    partial = {"channel_id": "a"}
    malformed = {**valid.__dict__, "capacity_sats": "1000"}
    duplicate = StateSnapshot((valid, valid), 2_000, 40, 2)

    with pytest.raises(ValueError, match="snapshot"):
        project_cycle_result(reference, CycleResult(snapshot=None))
    with pytest.raises(ValueError, match="snapshot channel"):
        project_cycle_result(reference, CycleResult(snapshot=StateSnapshot((partial,))))
    with pytest.raises(ValueError, match="capacity_sats"):
        project_cycle_result(reference, CycleResult(snapshot=StateSnapshot((malformed,))))
    with pytest.raises(ValueError, match="duplicate snapshot"):
        project_cycle_result(reference, CycleResult(snapshot=duplicate))


@pytest.mark.parametrize(
    "field",
    ["amount_sats", "source_excess_sats", "dest_need_sats", "max_chunk_sats", "cheap_rank"],
)
def test_projection_rejects_zero_for_strictly_positive_generated_pair_facts(field):
    pair = _pair()
    setattr(pair, field, 0)

    with pytest.raises(ValueError, match=field):
        project_cycle_result(_strict_reference(), _strict_result(pair=pair))


def test_projection_rejects_absent_producer_facts_instead_of_synthesizing_them():
    reference = _strict_reference()
    del reference.producer["started_at"]

    with pytest.raises(ValueError, match="producer.started_at"):
        project_cycle_result(reference, _strict_result())


def test_projection_serializes_terminal_stage_and_nested_effective_budget():
    pair = _pair()
    pair.score_decomposition = {"inputs": {"effective_budget_sats": 17}}

    body = project_cycle_result(_strict_reference(), _strict_result(pair=pair), "planning_only")

    assert body["terminal_stage"] == "planning_only"
    assert body["completeness"]["eligible"] is False
    assert body["funnel"]["generated_pairs"][0]["effective_budget_sats"] == 17


def test_projection_prefers_explicit_effective_budget_and_keeps_route_direction():
    pair = _pair()
    pair.effective_budget_sats = 9
    pair.score_decomposition = {"inputs": {"effective_budget_sats": 17}}
    pair.route = [{"channel": "1x1x0", "direction": 1, "amount_msat": 1000, "delay": 12}]

    body = project_cycle_result(_strict_reference(), _strict_result(pair=pair))

    generated = body["funnel"]["generated_pairs"][0]
    assert generated["effective_budget_sats"] == 9
    assert generated["route_summary"] == [{
        "index": 0, "channel": "1x1x0", "direction": 1,
        "id": "", "amount_msat": 1000, "delay": 12,
    }]


def test_projection_failure_evidence_uses_safe_bounded_allowlist():
    pair = _pair()
    execution = ExecutionResult(
        success=False,
        amount_sats=100,
        error="temporary failure",
        excluded_channels=["1x1x0/1", "x" * 1000],
        failure_data={
            "failure_class": "liquidity",
            "erring_channel": "2x2x0/0",
            "retry_excluded_channels": ["3x3x0/1"],
            "payment_hash": "secret-hash",
            "payment_secret": "secret",
            "raw_rpc": {"bolt11": "secret-invoice"},
        },
    )
    result = _strict_result(
        pair=pair,
        outcome={"source_channel_id": "a", "dest_channel_id": "b", "result": execution},
    )

    projected = project_cycle_result(_strict_reference(), result)["execution"]["pair_outcomes"][0]["result"]

    assert projected["excluded_channels"][0] == "1x1x0/1"
    assert len(projected["excluded_channels"][1]) == 512
    assert projected["failure_data"] == {
        "failure_class": "liquidity",
        "erring_channel": "2x2x0/0",
        "retry_excluded_channels": ["3x3x0/1"],
    }
    assert "payment_hash" not in repr(projected)
    assert "secret" not in repr(projected)


def test_queue_full_is_detected_before_any_slow_copy_or_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    for _ in range(2):
        manager.finish_cycle(manager.begin_cycle(_configuration(), {"trigger": "automatic"}), _strict_result())
    original = __import__("modules.rebalance_cycle_capture", fromlist=["copy"]).copy.deepcopy

    def slow_copy(value):
        time.sleep(0.2)
        return original(value)

    monkeypatch.setattr("modules.rebalance_cycle_capture.copy.deepcopy", slow_copy)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    started = time.monotonic()
    manager.finish_cycle(reference, _strict_result())

    assert time.monotonic() - started < 0.05
    assert manager._manifest["dropped"] == 1


def test_forty_manifest_only_toggles_stay_within_owned_retention(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    manager.output_dir.mkdir()
    unrelated = manager.output_dir / "unrelated.json"
    unrelated.write_text("{}")

    for _ in range(40):
        assert manager.set_enabled(True)
        assert manager.set_enabled(False, timeout_seconds=1.0)

    owned = [path for path in manager.output_dir.glob("*.json") if path != unrelated]
    assert unrelated.exists()
    assert len(owned) <= 32
    assert sum(path.stat().st_size for path in owned) <= 256 * 1024 * 1024


def test_begin_records_producer_start_and_commit_without_completed_time(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    manager.python_commit = "commit-at-begin"
    assert manager.set_enabled(True)

    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})

    assert reference.producer["started_at"]
    assert reference.producer["python_commit"] == "commit-at-begin"
    assert "completed_at" not in reference.producer


def test_enabled_finish_does_not_write_filesystem_on_cycle_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    atomic = MagicMock()
    monkeypatch.setattr(manager, "_atomic_write", atomic)

    manager.finish_cycle(
        manager.begin_cycle(_configuration(), {"trigger": "automatic"}),
        _strict_result(),
    )

    atomic.assert_not_called()
    assert manager._queue.qsize() == 1


def test_hard_preflight_pair_cap_rejects_before_projection(tmp_path, monkeypatch):
    import modules.rebalance_cycle_capture as capture_module

    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    projected = MagicMock()
    monkeypatch.setattr(capture_module, "project_cycle_result", projected)
    result = _strict_result()
    result.considered_candidates = [result.considered_candidates[0]] * (
        capture_module.MAX_GENERATED_PAIRS + 1
    )

    manager.finish_cycle(
        manager.begin_cycle(_configuration(), {"trigger": "automatic"}),
        result,
    )

    projected.assert_not_called()
    attempt = manager.read_manifest()["attempts"][-1]
    assert attempt["status"] == "failed"
    assert attempt["error_category"] == "ValueError"


def test_incomplete_terminal_stage_is_preserved_in_manifest_without_synthesis(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)

    manager.finish_cycle(
        manager.begin_cycle(_configuration(), {"trigger": "planning"}),
        CycleResult(),
        "no_router",
    )

    attempt = manager.read_manifest()["attempts"][-1]
    assert attempt["terminal_stage"] == "no_router"
    assert attempt["eligible"] is False
    assert attempt["status"] == "failed"
