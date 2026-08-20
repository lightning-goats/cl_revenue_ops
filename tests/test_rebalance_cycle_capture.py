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
    result = CycleResult(
        considered_candidates=[_pair()],
        candidates=[_pair()],
        pair_outcomes=[
            {"source_channel_id": "a", "dest_channel_id": "b", "result": {"success": True}}
        ],
    )
    body = project_cycle_result(reference, result)

    assert body["funnel"]["generated_pairs"][0]["cheap_rank"] == 1
    assert body["execution"]["pair_outcomes"][0]["source_channel_id"] == "a"
    manager.finish_cycle(reference, result)
    assert manager.set_enabled(False, timeout_seconds=2.0)
    envelope_paths = list(manager.output_dir.glob("*.json"))
    assert envelope_paths
    import json
    envelope = json.loads(next(path for path in envelope_paths if not path.name.startswith("manifest-")).read_text())
    verify_envelope(envelope)


def test_pair_outcomes_keep_completed_future_pair_identity():
    first = _pair("source-a", "dest-a", 1)
    second = _pair("source-b", "dest-b", 2)
    result = CycleResult(
        considered_candidates=[first, second],
        candidates=[first, second],
        pair_outcomes=[
            {"source_channel_id": "source-b", "dest_channel_id": "dest-b", "result": {"success": True}},
            {"source_channel_id": "source-a", "dest_channel_id": "dest-a", "result": {"success": False}},
        ],
    )
    reference = SimpleNamespace(
        capture_run_id="a" * 32,
        capture_seq=1,
        cycle_id=f"{'a' * 32}:00000001",
        configuration=_configuration(),
        producer={"trigger": "automatic"},
    )

    body = project_cycle_result(reference, result)

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
    reference = SimpleNamespace(
        capture_run_id="a" * 32,
        capture_seq=1,
        cycle_id=f"{'a' * 32}:00000001",
        configuration=_configuration(),
        producer={"trigger": "automatic"},
    )
    result = CycleResult(pair_outcomes=[{"source_channel_id": "x", "dest_channel_id": "y", "result": object()}])

    body = project_cycle_result(reference, result)

    assert body["execution"]["pair_outcomes"] == []


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
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer={"trigger": "automatic"})

    body = project_cycle_result(reference, CycleResult(snapshot=StateSnapshot((channel,), 1_000, 0, 1)))

    captured = body["pre_state"]["normalized_snapshot"]["channels"][0]
    assert captured["peer_id"] == "peer-a"
    assert captured["remaining_budget_sats"] == 0
    assert captured["target_band_low"] == 0.3
    assert captured["utilization_is_realized"] is True


def test_projection_keeps_real_execution_result_with_explicit_allowlist():
    pair = _pair()
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer={"trigger": "automatic"})
    execution = ExecutionResult(success=True, amount_sats=100, fee_sats=2, route_type="native", failure_data={"secret": "not-copied"})

    body = project_cycle_result(reference, CycleResult(considered_candidates=[pair], candidates=[pair], pair_outcomes=[{"source_channel_id": "a", "dest_channel_id": "b", "result": execution}]))

    result = body["execution"]["pair_outcomes"][0]["result"]
    assert result == {"success": True, "amount_sats": 100, "fee_sats": 2, "fee_msat": 0, "fee_ppm": 0, "attempts": 0, "hops": 0, "parts": 1, "route_type": "native", "error": "", "payment_pending": False}


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
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=configuration, producer={"trigger": "automatic"})

    body = project_cycle_result(reference, CycleResult(considered_candidates=[pair], candidates=[pair]))

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


def test_finish_hands_off_without_projecting_or_manifest_io(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    monkeypatch.setattr("modules.rebalance_cycle_capture.project_cycle_result", lambda *_a: (_ for _ in ()).throw(AssertionError("caller projected")))

    manager.finish_cycle(reference, CycleResult())

    assert manager._queue.qsize() == 1


def test_projection_rejects_duplicate_or_noncontiguous_generated_evidence():
    first = _pair("source-a", "dest-a", 4)
    duplicate = _pair("source-a", "dest-a", 9)
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer={"trigger": "automatic"})

    with pytest.raises(ValueError):
        project_cycle_result(reference, CycleResult(considered_candidates=[first, duplicate]))


def test_projection_preserves_timeout_status_for_linked_pair():
    pair = _pair()
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer={"trigger": "automatic"})

    body = project_cycle_result(reference, CycleResult(considered_candidates=[pair], candidates=[pair], pair_outcomes=[{"source_channel_id": "a", "dest_channel_id": "b", "status": "still_running_timeout", "result": None}]))

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
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer={"trigger": "automatic"})
    channels = tuple(ChannelState(channel_id=f"{index}x1x0", peer_id=f"peer-{index}", capacity_sats=1, local_ratio=0.5, actual_inbound_fee_ppm=0, value_class="neutral", is_valuable=False, remaining_budget_sats=0, cooldown_active=False) for index in range(1025))

    with pytest.raises(ValueError):
        project_cycle_result(reference, CycleResult(snapshot=StateSnapshot(channels)))


def test_generated_and_selected_evidence_include_route_quote_and_rejection_fields():
    pair = _pair()
    pair.route = [{"channel": "1x1x0", "amount_msat": 1_000}]
    pair.route_cost_sats = 2
    pair.score_decomposition = {"p_success": 0.9, "final_score_sats": 3.0, "effective_budget_sats": 0}
    pair.rejection_reason = "priced"
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer={"trigger": "automatic"})

    body = project_cycle_result(reference, CycleResult(considered_candidates=[pair], candidates=[pair]))

    generated = body["funnel"]["generated_pairs"][0]
    selected = body["funnel"]["final_selected_pairs"][0]
    assert generated["route_summary"] == [{"channel": "1x1x0", "amount_msat": 1_000}]
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
    pair.bootstrap_score_decomposition = {"stage": "planner_pre_route", "p_success": 0.5}
    pair.score_decomposition = {"stage": "priced", "p_success": 0.9}
    reference = SimpleNamespace(capture_run_id="a" * 32, capture_seq=1, cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer={"trigger": "automatic"})

    body = project_cycle_result(reference, CycleResult(considered_candidates=[pair], candidates=[pair]))

    generated = body["funnel"]["generated_pairs"][0]
    assert generated["bootstrap_score_decomposition"]["stage"] == "planner_pre_route"
    assert generated["score_decomposition"]["stage"] == "priced"



def test_finish_handoff_freezes_mutable_cycle_evidence_before_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    pair = _pair()
    result = CycleResult(considered_candidates=[pair], candidates=[pair])

    manager.finish_cycle(reference, result)
    pair.score = 999.0
    result.candidates.clear()
    queued = manager._queue.get_nowait()

    frozen_result = queued[1]
    assert frozen_result.candidates[0].source_channel_id == "a"
    assert frozen_result.considered_candidates[0].score != 999.0



def test_disabled_finish_is_fast_and_does_not_touch_filesystem(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    atomic = MagicMock()
    monkeypatch.setattr(manager, "_atomic_write", atomic)
    started = time.monotonic()
    assert manager.begin_cycle(_configuration(), {"trigger": "automatic"}) is None
    assert time.monotonic() - started < 0.05
    atomic.assert_not_called()
    assert not manager.output_dir.exists()


def test_queue_full_handoff_is_fast_without_projection_or_filesystem(tmp_path, monkeypatch):
    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", lambda self: None)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    atomic = MagicMock(); monkeypatch.setattr(manager, "_atomic_write", atomic)
    monkeypatch.setattr("modules.rebalance_cycle_capture.project_cycle_result", lambda *_a: (_ for _ in ()).throw(AssertionError("projected")))
    for _ in range(2):
        manager.finish_cycle(manager.begin_cycle(_configuration(), {"trigger": "automatic"}), CycleResult())
    started = time.monotonic()
    manager.finish_cycle(manager.begin_cycle(_configuration(), {"trigger": "automatic"}), CycleResult())
    assert time.monotonic() - started < 0.05
    assert manager.read_manifest()["dropped"] == 1
    atomic.assert_not_called()


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
    manager.finish_cycle(manager.begin_cycle(_configuration(), {"trigger": "automatic"}), CycleResult())
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
