import json
import threading
import time
from dataclasses import FrozenInstanceError
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

    assert manager.prepare_cycle_result(reference, result) is True
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

def test_finish_is_no_throw_and_writer_failure_is_manifested(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    ref = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    monkeypatch.setattr(
        manager,
        "_publish_envelope",
        lambda *_a: (_ for _ in ()).throw(OSError("blocked")),
    )

    result = _strict_result()
    assert manager.prepare_cycle_result(ref, result) is True
    manager.finish_cycle(ref, result)

    _wait_for_failed(manager, "OSError")
    assert manager.set_enabled(False, timeout_seconds=1.0)


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
        configuration = configuration() if callable(configuration) else configuration
        producer = producer() if callable(producer) else producer
        self.begins.append((configuration, producer))
        return SimpleNamespace(capture_run_id="a" * 32, capture_seq=len(self.begins), cycle_id=("a" * 32) + ":00000001", configuration=_configuration(), producer=producer)

    def finish_cycle(self, reference, result, terminal_stage="completed"):
        self.finishes.append((reference, result, terminal_stage))


def _install_live_blocked_cycle_writer(monkeypatch):
    release = threading.Event()
    original_writer = RebalanceCycleCaptureManager._writer_main

    def blocked_writer(manager):
        assert release.wait(2.0)
        original_writer(manager)

    monkeypatch.setattr(RebalanceCycleCaptureManager, "_writer_main", blocked_writer)
    return release


def test_prepared_finish_enqueues_only_a_frozen_bounded_reference_handoff(tmp_path, monkeypatch):
    release_writer = _install_live_blocked_cycle_writer(monkeypatch)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    result = _strict_result()

    assert manager.prepare_cycle_result(reference, result) is True
    manager.finish_cycle(reference, result)
    handoff = manager._queue.get_nowait()

    assert handoff.reference is not reference
    assert handoff.result is not result
    assert type(handoff.result).__name__ == "SimpleNamespace"
    with pytest.raises(TypeError):
        reference.configuration["raw"] = "mutation"
    assert handoff.terminal_stage == "completed"
    with pytest.raises(FrozenInstanceError):
        handoff.result = None
    release_writer.set()
    assert manager.set_enabled(False, timeout_seconds=1.0)

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
    _wait_for_manifest_idle(manager)
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

def test_writer_freezes_mutable_cycle_evidence_before_publication(tmp_path, monkeypatch):
    published = []
    projection_complete = threading.Event()
    allow_publish = threading.Event()

    def blocked_publish(_reference, serialized):
        published.append(serialized)
        projection_complete.set()
        assert allow_publish.wait(1.0)

    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    monkeypatch.setattr(manager, "_publish_envelope", blocked_publish)
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    pair = _pair()
    result = _strict_result(pair=pair)

    assert manager.prepare_cycle_result(reference, result) is True
    manager.finish_cycle(reference, result)
    assert projection_complete.wait(1.0)
    pair.score = 999.0
    result.candidates.clear()
    allow_publish.set()
    assert manager.set_enabled(False, timeout_seconds=1.0)

    frozen = json.loads(published[0])
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

def _wait_for_manifest_idle(manager, timeout=1.0):
    deadline = time.monotonic() + timeout
    with manager._manifest_publish_condition:
        while (
            manager._pending_manifest_snapshots
            or manager._manifest_publish_inflight
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError("manifest publisher did not become idle")
            manager._manifest_publish_condition.wait(remaining)


def _wait_for_failed_attempt(manager):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        manifest = manager.read_manifest()
        if manifest.get("failed") == 1:
            assert manifest["writer_health"] == "degraded"
            return manifest, manifest["attempts"][-1]
        time.sleep(0.01)
    raise AssertionError(manager.read_manifest())


def _wait_for_failed(manager, category):
    manifest, _attempt = _wait_for_failed_attempt(manager)
    assert manifest["last_error_category"] == category


@pytest.mark.parametrize(
    "failure_kind, category, exact_error",
    [
        pytest.param("projection", "ValueError", "projection", id="projection"),
        pytest.param("write", "OSError", "write", id="write"),
        pytest.param(
            "size", "ValueError", "complete sealed envelope exceeds 32 MiB",
            id="size",
        ),
    ],
)
def test_writer_failures_are_manifested_without_raising(
    tmp_path, monkeypatch, failure_kind, category, exact_error,
):
    import modules.rebalance_cycle_capture as capture_module

    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    _wait_for_manifest_idle(manager)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    result = _strict_result()
    assert manager.prepare_cycle_result(reference, result) is True
    _wait_for_manifest_idle(manager)

    if failure_kind == "projection":
        injected = MagicMock(side_effect=ValueError("projection"))
        monkeypatch.setattr(capture_module, "project_cycle_result", injected)
    elif failure_kind == "write":
        injected = MagicMock(side_effect=OSError("write"))
        monkeypatch.setattr(manager, "_publish_envelope", injected)
    else:
        injected = MagicMock(wraps=capture_module.seal_envelope)
        monkeypatch.setattr(capture_module, "seal_envelope", injected)
        monkeypatch.setattr(capture_module, "MAX_ENVELOPE_BYTES", 1)

    manager.finish_cycle(reference, result)

    manifest, attempt = _wait_for_failed_attempt(manager)
    injected.assert_called_once()
    assert manifest["last_error_category"] == category
    assert attempt["error_category"] == category
    assert attempt["error"] == exact_error
    assert manager.set_enabled(False, timeout_seconds=1.0)



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
    _wait_for_manifest_idle(manager)
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
    _wait_for_manifest_idle(manager)
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


def test_blocked_writer_never_owns_raw_unprepared_or_prepared_metadata(
    tmp_path, monkeypatch,
):
    marker_text = "UNPREPARED-FORBIDDEN-MARKER"
    logs = []

    class HostileSecret:
        deepcopy_calls = 0

        def __deepcopy__(self, _memo):
            self.deepcopy_calls += 1
            raise AssertionError(marker_text)

        def __repr__(self):
            return marker_text

    def result_with_forbidden_metadata(hostile):
        return _strict_result(outcome={
            "source_channel_id": "a",
            "dest_channel_id": "b",
            "status": "returned",
            "result": {
                "success": False,
                "amount_sats": 100,
                "failure_data": {
                    "failure_class": "liquidity",
                    "payment_secret": hostile,
                    "raw_rpc": {"marker": hostile},
                },
                "payment_hash": hostile,
                "invoice": hostile,
            },
        })

    release_writer = _install_live_blocked_cycle_writer(monkeypatch)
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db",
        lambda *args, **kwargs: logs.append((args, kwargs)),
    )
    assert manager.set_enabled(True)

    unprepared_reference = manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    )
    unprepared_hostile = HostileSecret()
    manager.finish_cycle(
        unprepared_reference,
        result_with_forbidden_metadata(unprepared_hostile),
        "failed",
    )

    prepared_reference = manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    )
    prepared_hostile = HostileSecret()
    prepared_result = result_with_forbidden_metadata(prepared_hostile)
    assert manager.prepare_cycle_result(prepared_reference, prepared_result) is True
    manager.finish_cycle(prepared_reference, prepared_result, "completed")

    with manager._queue.mutex:
        handoffs = list(manager._queue.queue)
    assert len(handoffs) == 2
    assert type(handoffs[0].result).__name__ == "_CapturePreparationFailure"
    assert type(handoffs[1].result).__name__ == "SimpleNamespace"
    handoff_text = repr(handoffs)
    assert marker_text not in handoff_text
    for forbidden in ("payment_secret", "payment_hash", "invoice", "raw_rpc"):
        assert forbidden not in handoff_text
    assert unprepared_hostile.deepcopy_calls == 0
    assert prepared_hostile.deepcopy_calls == 0

    release_writer.set()
    assert manager.set_enabled(False, timeout_seconds=2.0)
    manifest = manager.read_manifest()
    assert manifest["failed"] == 1
    assert manifest["completed"] == 1
    assert [attempt["terminal_stage"] for attempt in manifest["attempts"]] == [
        "failed", "completed",
    ]
    assert manifest["attempts"][0]["error_category"] == "ValueError"
    manifest_text = repr(manifest)
    assert marker_text not in manifest_text
    for forbidden in ("payment_secret", "payment_hash", "invoice", "raw_rpc"):
        assert forbidden not in manifest_text
    assert marker_text not in repr(logs)


def test_forbidden_failure_metadata_never_crosses_handoff_or_publication(
    tmp_path, monkeypatch,
):
    marker_text = "FORBIDDEN-RAW-FAILURE-MARKER"
    logs = []

    class HostileSecret:
        deepcopy_calls = 0

        def __deepcopy__(self, _memo):
            self.deepcopy_calls += 1
            raise AssertionError(marker_text)

        def __str__(self):
            return marker_text

        def __repr__(self):
            return marker_text

    release_writer = _install_live_blocked_cycle_writer(monkeypatch)
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db",
        lambda *args, **kwargs: logs.append((args, kwargs)),
    )
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    hostile = HostileSecret()
    result = _strict_result(outcome={
        "source_channel_id": "a",
        "dest_channel_id": "b",
        "status": "returned",
        "result": {
            "success": False,
            "amount_sats": 100,
            "error": "temporary failure",
            "failure_data": {
                "failure_class": "liquidity",
                "payment_secret": hostile,
                "payment_hash": hostile,
                "invoice": hostile,
                "raw_rpc": {"marker": hostile},
            },
            "payment_secret": hostile,
            "payment_hash": hostile,
            "invoice": hostile,
            "raw_rpc": {"marker": hostile},
        },
    })

    assert manager.prepare_cycle_result(reference, result) is True
    manager.finish_cycle(reference, result)
    with manager._queue.mutex:
        handoff = manager._queue.queue[0]
        handoff_text = repr(handoff)
    assert marker_text not in handoff_text
    for forbidden in ("payment_secret", "payment_hash", "invoice", "raw_rpc"):
        assert forbidden not in handoff_text
    assert hostile.deepcopy_calls == 0

    release_writer.set()
    assert manager.set_enabled(False, timeout_seconds=2.0)
    envelope_path = next(
        path for path in manager.output_dir.glob("*.json")
        if not path.name.startswith("manifest-")
    )
    envelope_text = envelope_path.read_text(encoding="utf-8")
    manifest_text = "".join(
        path.read_text(encoding="utf-8")
        for path in manager.output_dir.glob("manifest-*.json")
    )
    combined_logs = repr(logs)
    for output_text in (envelope_text, manifest_text, combined_logs):
        assert marker_text not in output_text
        for forbidden in ("payment_secret", "payment_hash", "invoice", "raw_rpc"):
            assert forbidden not in output_text


def test_queue_full_reserves_before_copy_and_repeated_drops_do_not_leak(
    tmp_path, monkeypatch,
):
    release_writer = _install_live_blocked_cycle_writer(monkeypatch)
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    for _ in range(2):
        reference = manager.begin_cycle(
            _configuration(), {"trigger": "automatic"},
        )
        result = _strict_result()
        assert manager.prepare_cycle_result(reference, result) is True
        manager.finish_cycle(reference, result)
    assert manager._queue.qsize() == 2
    assert manager._prepared == {}

    original = __import__("modules.rebalance_cycle_capture", fromlist=["copy"]).copy.deepcopy
    copy_calls = 0

    def slow_copy(value):
        nonlocal copy_calls
        copy_calls += 1
        time.sleep(0.2)
        return original(value)

    monkeypatch.setattr("modules.rebalance_cycle_capture.copy.deepcopy", slow_copy)
    for _ in range(40):
        reference = manager.begin_cycle(
            _configuration(), {"trigger": "automatic"},
        )
        started = time.monotonic()
        assert manager.prepare_cycle_result(reference, _strict_result()) is False
        assert time.monotonic() - started < 0.05
        manager.finish_cycle(reference, _strict_result(), "planning_only")
        manager.finish_cycle(reference, _strict_result(), "completed")

    assert copy_calls == 0
    assert manager._manifest["dropped"] == 40
    assert manager._manifest["attempts"][-1]["terminal_stage"] == "planning_only"
    assert manager._prepared == {}
    assert manager._active == set()
    release_writer.set()
    assert manager.set_enabled(False, timeout_seconds=2.0)


def test_blocked_manifest_publisher_keeps_newest_begin_drop_and_terminal_truth(
    tmp_path, monkeypatch,
):
    release_writer = _install_live_blocked_cycle_writer(monkeypatch)
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    _wait_for_manifest_idle(manager)
    manifest_path = manager._manifest_path()
    assert json.loads(manifest_path.read_text())["attempted"] == 0

    original_write = manager._write_manifest_snapshot
    write_started = threading.Event()
    release_manifest = threading.Event()

    def blocked_manifest_write(path, data, revision):
        write_started.set()
        assert release_manifest.wait(2.0)
        return original_write(path, data, revision)

    monkeypatch.setattr(manager, "_write_manifest_snapshot", blocked_manifest_write)
    first = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    assert write_started.wait(1.0)
    assert manager.prepare_cycle_result(first, _strict_result()) is True
    manager.finish_cycle(first, _strict_result())
    second = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    assert manager.prepare_cycle_result(second, _strict_result()) is True
    manager.finish_cycle(second, _strict_result())
    third = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    assert manager.prepare_cycle_result(third, _strict_result()) is False
    manager.finish_cycle(third, _strict_result(), "planning_only")

    with manager._manifest_publish_condition:
        pending = manager._pending_manifest_snapshots[manifest_path]
        pending_manifest = json.loads(pending[1])
    assert pending_manifest["attempted"] == 3
    assert pending_manifest["dropped"] == 1
    assert pending_manifest["attempts"][-1]["status"] == "dropped"
    assert pending_manifest["attempts"][-1]["terminal_stage"] == "planning_only"
    assert json.loads(manifest_path.read_text())["attempted"] == 0

    release_manifest.set()
    _wait_for_manifest_idle(manager, timeout=2.0)
    durable = json.loads(manifest_path.read_text())
    assert durable["attempted"] == 3
    assert durable["dropped"] == 1
    assert durable["attempts"][-1]["terminal_stage"] == "planning_only"
    release_writer.set()
    assert manager.set_enabled(False, timeout_seconds=2.0)


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
    release_writer = _install_live_blocked_cycle_writer(monkeypatch)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    manager.python_commit = "commit-at-begin"
    assert manager.set_enabled(True)

    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})

    assert reference.producer["started_at"]
    assert reference.producer["python_commit"] == "commit-at-begin"
    assert "completed_at" not in reference.producer
    manager.finish_cycle(reference, _strict_result(), "planning_only")
    release_writer.set()
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_enabled_finish_does_not_write_filesystem_on_cycle_thread(tmp_path, monkeypatch):
    release_writer = _install_live_blocked_cycle_writer(monkeypatch)
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    _wait_for_manifest_idle(manager)
    atomic = MagicMock()
    monkeypatch.setattr(manager, "_atomic_write", atomic)

    manager.finish_cycle(
        manager.begin_cycle(_configuration(), {"trigger": "automatic"}),
        _strict_result(),
    )

    atomic.assert_not_called()
    assert manager._queue.qsize() == 1
    release_writer.set()
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_hard_preflight_pair_cap_rejects_before_projection(tmp_path, monkeypatch):
    import modules.rebalance_cycle_capture as capture_module

    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    projected = MagicMock()
    monkeypatch.setattr(capture_module, "project_cycle_result", projected)
    preflight = MagicMock(wraps=capture_module._preflight_cycle_result)
    monkeypatch.setattr(capture_module, "_preflight_cycle_result", preflight)
    result = _strict_result()
    result.considered_candidates = [result.considered_candidates[0]] * (
        capture_module.MAX_GENERATED_PAIRS + 1
    )

    reference = manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    )
    assert manager.prepare_cycle_result(reference, result) is False
    preflight.assert_called_once_with(result)
    projected.assert_not_called()
    preparation_failure = manager._prepared[reference.capture_seq]
    assert preparation_failure.error_category == "ValueError"
    assert preparation_failure.error == "considered_candidates exceeds capture bound"

    manager.finish_cycle(reference, result)

    manifest, attempt = _wait_for_failed_attempt(manager)
    preflight.assert_called_once_with(result)
    projected.assert_not_called()
    assert manifest["last_error_category"] == "ValueError"
    assert attempt["status"] == "failed"
    assert attempt["error_category"] == "ValueError"
    assert attempt["error"] == (
        "capture preparation failed: ValueError: "
        "considered_candidates exceeds capture bound"
    )
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_incomplete_terminal_stage_is_preserved_in_manifest_without_synthesis(tmp_path):
    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)

    manager.finish_cycle(
        manager.begin_cycle(_configuration(), {"trigger": "planning"}),
        CycleResult(),
        "no_router",
    )

    _wait_for_failed(manager, "ValueError")
    attempt = manager.read_manifest()["attempts"][-1]
    assert attempt["terminal_stage"] == "no_router"
    assert attempt["eligible"] is False
    assert attempt["status"] == "failed"
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_normal_finish_returns_before_slow_projection(tmp_path, monkeypatch):
    import modules.rebalance_cycle_capture as capture_module

    manager = RebalanceCycleCaptureManager(tmp_path / "revenue_ops.db", lambda *_a, **_k: None)
    assert manager.set_enabled(True)
    original = capture_module.project_cycle_result
    projection_started = threading.Event()

    def slow_projection(*args, **kwargs):
        projection_started.set()
        time.sleep(0.30)
        return original(*args, **kwargs)

    monkeypatch.setattr(capture_module, "project_cycle_result", slow_projection)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    result = _strict_result()
    assert manager.prepare_cycle_result(reference, result) is True
    started = time.monotonic()
    manager.finish_cycle(reference, result)
    elapsed = time.monotonic() - started

    assert elapsed < 0.05
    assert projection_started.wait(1.0)
    assert manager.set_enabled(False, timeout_seconds=1.0)


def _assert_failed_enable_is_fully_rolled_back(
    manager, preserve_existing_publication=False,
):
    assert manager._enabled is False
    assert manager._run_id is None
    assert manager._manifest is None
    assert manager._active == set()
    assert manager._prepared == {}
    assert manager._inflight == 0
    assert manager._writer is None
    assert manager._queue.empty()
    publisher = manager._manifest_publisher
    if preserve_existing_publication:
        assert manager._pending_manifest_snapshots
        assert publisher is not None and publisher.is_alive()
    else:
        assert not manager._pending_manifest_snapshots
        assert publisher is None or not publisher.is_alive()
    assert manager._slots.acquire(blocking=False)
    assert manager._slots.acquire(blocking=False)
    assert manager._slots.acquire(blocking=False) is False
    manager._slots.release()
    manager._slots.release()


def test_manifest_publisher_start_failure_rolls_back_then_retry_is_fresh(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original_start = threading.Thread.start
    failed = []

    def fail_first_publisher(thread):
        if (
            thread.name == "rebalance-cycle-capture-manifest-writer"
            and not failed
        ):
            failed.append(thread)
            raise RuntimeError("publisher start failed")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_publisher)

    assert manager.set_enabled(True) is False
    _assert_failed_enable_is_fully_rolled_back(manager)
    assert manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    ) is None

    assert manager.set_enabled(True) is True
    reference = manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    )
    assert reference is not None
    assert reference.capture_seq == 1
    assert manager._writer.is_alive()
    assert manager._manifest_publisher.is_alive()
    manager.finish_cycle(reference, _strict_result(), "planning_only")
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_manifest_publisher_that_never_becomes_live_rolls_back(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original_start = threading.Thread.start
    skipped = []

    def skip_first_publisher_start(thread):
        if (
            thread.name == "rebalance-cycle-capture-manifest-writer"
            and not skipped
        ):
            skipped.append(thread)
            return None
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", skip_first_publisher_start)

    assert manager.set_enabled(True) is False
    _assert_failed_enable_is_fully_rolled_back(manager)
    assert manager.set_enabled(True) is True
    assert manager._writer.is_alive()
    assert manager._manifest_publisher.is_alive()
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_cycle_writer_start_failure_stops_new_publisher_and_retry_succeeds(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original_start = threading.Thread.start
    failed = []
    created_publishers = []

    def fail_first_cycle_writer(thread):
        if thread.name == "rebalance-cycle-capture-manifest-writer":
            created_publishers.append(thread)
        if thread.name == "rebalance-cycle-capture-writer" and not failed:
            failed.append(thread)
            raise RuntimeError("cycle writer start failed")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_first_cycle_writer)

    assert manager.set_enabled(True) is False
    _assert_failed_enable_is_fully_rolled_back(manager)
    assert created_publishers
    assert all(not thread.is_alive() for thread in created_publishers)

    assert manager.set_enabled(True) is True
    reference = manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    )
    assert reference.capture_seq == 1
    manager.finish_cycle(reference, _strict_result(), "planning_only")
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_partial_cycle_writer_start_is_stopped_and_leaves_no_pending_state(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original_start = threading.Thread.start
    partially_started = []

    def start_then_fail_cycle_writer(thread):
        if thread.name == "rebalance-cycle-capture-writer" and not partially_started:
            original_start(thread)
            partially_started.append(thread)
            raise RuntimeError("cycle writer failed after start")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_then_fail_cycle_writer)

    assert manager.set_enabled(True) is False
    _assert_failed_enable_is_fully_rolled_back(manager)
    assert partially_started
    assert all(not thread.is_alive() for thread in partially_started)
    assert manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    ) is None

    assert manager.set_enabled(True) is True
    assert manager._writer.is_alive()
    assert manager._manifest_publisher.is_alive()
    assert manager.set_enabled(False, timeout_seconds=1.0)


@pytest.mark.parametrize("publication_point", ["rejected", "queued", "durable"])
def test_false_initial_publication_ack_always_terminalizes_owned_run(
    tmp_path, monkeypatch, publication_point,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original_publish = manager._publish_manifest_snapshot
    failed_path = []

    def false_ack_after_selected_point(snapshot):
        state = json.loads(snapshot[1])["state"]
        if state != "active":
            return original_publish(snapshot)
        failed_path.append(snapshot[0])
        if publication_point == "queued":
            with manager._manifest_publish_condition:
                manager._pending_manifest_snapshots[snapshot[0]] = snapshot
                manager._manifest_publish_condition.notify_all()
        elif publication_point == "durable":
            manager._write_manifest_snapshot(*snapshot)
        return False

    monkeypatch.setattr(
        manager, "_publish_manifest_snapshot", false_ack_after_selected_point,
    )

    assert manager.set_enabled(True) is False
    assert manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    ) is None
    _wait_for_manifest_idle(manager, timeout=2.0)

    assert len(failed_path) == 1
    persisted = json.loads(failed_path[0].read_text())
    assert persisted["state"] == "closed"
    assert persisted["writer_health"] == "degraded"
    assert persisted["last_error_category"] == "RuntimeError"
    assert manager._enabled is False
    assert manager._run_id is None
    assert manager._manifest is None
    assert manager._writer is None
    assert not manager._pending_manifest_snapshots



@pytest.mark.parametrize("publication_point", ["queued", "durable"])
def test_publisher_exit_before_initial_ack_terminalizes_exact_owned_run(
    tmp_path, monkeypatch, publication_point,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original_ensure = manager._ensure_manifest_publisher
    ensure_calls = []
    failed_path = []

    class ExitedPublisher:
        @staticmethod
        def is_alive():
            return False

    def exit_publisher_at_ack():
        ensure_calls.append(True)
        if len(ensure_calls) != 2:
            return original_ensure()
        with manager._manifest_publish_condition:
            assert len(manager._pending_manifest_snapshots) == 1
            path, snapshot = next(iter(manager._pending_manifest_snapshots.items()))
            failed_path.append(path)
            if publication_point == "durable":
                manager._pending_manifest_snapshots.pop(path)
            publisher = manager._manifest_publisher
            assert publisher is not None and publisher.is_alive()
            manager._manifest_publisher_stop = True
            manager._manifest_publish_condition.notify_all()
        publisher.join(1.0)
        assert not publisher.is_alive()
        if publication_point == "durable":
            manager._write_manifest_snapshot(*snapshot)
        return ExitedPublisher(), False

    monkeypatch.setattr(
        manager, "_ensure_manifest_publisher", exit_publisher_at_ack,
    )

    assert manager.set_enabled(True) is False
    assert manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    ) is None
    _wait_for_manifest_idle(manager, timeout=2.0)

    assert len(failed_path) == 1
    persisted = json.loads(failed_path[0].read_text())
    assert persisted["state"] == "closed"
    assert persisted["writer_health"] == "degraded"
    assert persisted["last_error_category"] == "RuntimeError"
    assert manager._enabled is False
    assert manager._run_id is None
    assert manager._manifest is None
    assert manager._writer is None
    assert not manager._pending_manifest_snapshots

def test_post_publication_cycle_writer_exit_persists_closed_degraded_rollback(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    stop_writer = threading.Event()
    terminal_write_started = threading.Event()
    release_terminal_write = threading.Event()
    original_atomic = manager._atomic_write

    def block_terminal_write(path, data):
        if (
            path.name.startswith("manifest-")
            and json.loads(data).get("state") == "closed"
        ):
            terminal_write_started.set()
            assert release_terminal_write.wait(1.0)
        return original_atomic(path, data)

    monkeypatch.setattr(manager, "_atomic_write", block_terminal_write)

    def exit_on_signal():
        assert stop_writer.wait(1.0)

    monkeypatch.setattr(manager, "_writer_main", exit_on_signal)
    original_publish = manager._publish_manifest_snapshot
    failed_path = []

    def persist_active_then_stop(snapshot):
        accepted = original_publish(snapshot)
        if json.loads(snapshot[1])["state"] == "active" and not failed_path:
            failed_path.append(snapshot[0])
            deadline = time.monotonic() + 1.0
            while not snapshot[0].exists():
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert json.loads(snapshot[0].read_text())["state"] == "active"
            stop_writer.set()
            manager._writer.join(1.0)
            assert not manager._writer.is_alive()
        return accepted

    monkeypatch.setattr(manager, "_publish_manifest_snapshot", persist_active_then_stop)

    started = time.monotonic()
    try:
        assert manager.set_enabled(True) is False
        assert time.monotonic() - started < 0.20
        assert terminal_write_started.wait(1.0)
        assert manager.begin_cycle(
            _configuration(), {"trigger": "automatic"},
        ) is None
    finally:
        release_terminal_write.set()
    _wait_for_manifest_idle(manager, timeout=2.0)

    persisted = json.loads(failed_path[0].read_text())
    assert persisted["state"] == "closed"
    assert persisted["writer_health"] == "degraded"
    assert persisted["last_error_category"] == "RuntimeError"
    assert manager._enabled is False
    assert manager._run_id is None
    assert manager._manifest is None
    assert manager._writer is None
    assert not manager._pending_manifest_snapshots


def test_post_publication_publisher_exit_restarts_for_terminal_rollback(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )

    def one_shot_publisher():
        current = threading.current_thread()
        with manager._manifest_publish_condition:
            while not manager._pending_manifest_snapshots:
                manager._manifest_publish_condition.wait(1.0)
            _path, snapshot = manager._pending_manifest_snapshots.popitem(last=False)
            manager._manifest_publish_inflight = True
        try:
            manager._write_manifest_snapshot(*snapshot)
        finally:
            with manager._manifest_publish_condition:
                manager._manifest_publish_inflight = False
                if manager._manifest_publisher is current:
                    manager._manifest_publisher = None
                manager._manifest_publish_condition.notify_all()

    monkeypatch.setattr(manager, "_manifest_publisher_main", one_shot_publisher)
    original_publish = manager._publish_manifest_snapshot
    failed_path = []

    def wait_for_publisher_exit(snapshot):
        accepted = original_publish(snapshot)
        if json.loads(snapshot[1])["state"] == "active" and not failed_path:
            failed_path.append(snapshot[0])
            publisher = manager._manifest_publisher
            if publisher is not None:
                publisher.join(1.0)
            assert snapshot[0].exists()
            assert json.loads(snapshot[0].read_text())["state"] == "active"
        return accepted

    monkeypatch.setattr(manager, "_publish_manifest_snapshot", wait_for_publisher_exit)

    assert manager.set_enabled(True) is False
    _wait_for_manifest_idle(manager, timeout=2.0)

    persisted = json.loads(failed_path[0].read_text())
    assert persisted["state"] == "closed"
    assert persisted["writer_health"] == "degraded"
    assert persisted["last_error_category"] == "RuntimeError"
    assert manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    ) is None
    assert manager._run_id is None
    assert manager._manifest is None
    assert not manager._pending_manifest_snapshots


def test_disabled_manager_construction_does_not_probe_git(tmp_path, monkeypatch):
    import modules.rebalance_cycle_capture as capture_module

    run = MagicMock(side_effect=AssertionError("disabled constructor probed git"))
    monkeypatch.setattr(capture_module.subprocess, "run", run)

    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )

    run.assert_not_called()
    assert manager.begin_cycle(_configuration(), {"trigger": "automatic"}) is None


def test_disabled_engine_capture_skips_config_snapshot_and_building():
    from modules.rebalance_engine_v2 import RebalanceEngine

    manager = RebalanceCycleCaptureManager(
        "/tmp/disabled-rebalance-capture.db", lambda *_a, **_k: None,
    )
    engine = object.__new__(RebalanceEngine)
    engine.rebalance_capture_manager = manager
    engine.config = MagicMock()
    engine.config.snapshot.side_effect = AssertionError("disabled capture read config")
    engine._max_concurrent_jobs = MagicMock(
        side_effect=AssertionError("disabled capture built configuration"),
    )

    assert engine._begin_rebalance_capture() is None
    engine.config.snapshot.assert_not_called()
    engine._max_concurrent_jobs.assert_not_called()


def test_manifest_filesystem_io_never_holds_cycle_intake_lock(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    first = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    second = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    original = manager._atomic_write
    manifest_write_started = threading.Event()
    allow_manifest_write = threading.Event()

    def slow_manifest(path, data):
        if path.name.startswith("manifest-"):
            manifest_write_started.set()
            allow_manifest_write.wait(0.30)
        return original(path, data)

    monkeypatch.setattr(manager, "_atomic_write", slow_manifest)
    manager.finish_cycle(first, _strict_result())
    assert manifest_write_started.wait(1.0)

    started = time.monotonic()
    manager.finish_cycle(second, _strict_result())
    elapsed = time.monotonic() - started
    allow_manifest_write.set()

    assert elapsed < 0.05
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_enable_manifest_io_does_not_block_begin_or_finish(tmp_path, monkeypatch):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original = manager._atomic_write
    write_started = threading.Event()
    release_write = threading.Event()
    enable_result = []

    def slow_initial_manifest(path, data):
        if path.name.startswith("manifest-") and not write_started.is_set():
            write_started.set()
            assert release_write.wait(1.0)
        return original(path, data)

    monkeypatch.setattr(manager, "_atomic_write", slow_initial_manifest)
    enable_thread = threading.Thread(
        target=lambda: enable_result.append(manager.set_enabled(True)),
    )
    enable_thread.start()
    assert write_started.wait(1.0)

    begin_result = []
    begin_done = threading.Event()

    def begin_while_manifest_is_slow():
        begin_result.append(manager.begin_cycle(
            _configuration(), {"trigger": "automatic"},
        ))
        begin_done.set()

    begin_thread = threading.Thread(target=begin_while_manifest_is_slow)
    begin_thread.start()
    assert begin_done.wait(0.05)
    assert begin_result[0] is not None

    finish_done = threading.Event()
    finish_thread = threading.Thread(
        target=lambda: (
            manager.finish_cycle(begin_result[0], _strict_result()),
            finish_done.set(),
        ),
    )
    finish_thread.start()
    assert finish_done.wait(0.05)

    release_write.set()
    enable_thread.join(1.0)
    begin_thread.join(1.0)
    finish_thread.join(1.0)
    assert enable_result == [True]
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_disable_manifest_io_does_not_block_existing_finish_or_new_begin(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    )
    original = manager._atomic_write
    write_started = threading.Event()
    release_write = threading.Event()
    disable_result = []

    def slow_draining_manifest(path, data):
        if path.name.startswith("manifest-") and not write_started.is_set():
            write_started.set()
            assert release_write.wait(1.0)
        return original(path, data)

    monkeypatch.setattr(manager, "_atomic_write", slow_draining_manifest)
    disable_thread = threading.Thread(
        target=lambda: disable_result.append(
            manager.set_enabled(False, timeout_seconds=1.0)
        ),
    )
    disable_thread.start()
    assert write_started.wait(1.0)

    finish_done = threading.Event()
    begin_done = threading.Event()
    begin_result = []
    finish_thread = threading.Thread(
        target=lambda: (
            manager.finish_cycle(reference, _strict_result()),
            finish_done.set(),
        ),
    )
    begin_thread = threading.Thread(
        target=lambda: (
            begin_result.append(manager.begin_cycle(
                _configuration(), {"trigger": "automatic"},
            )),
            begin_done.set(),
        ),
    )
    finish_thread.start()
    begin_thread.start()
    assert finish_done.wait(0.05)
    assert begin_done.wait(0.05)
    assert begin_result == [None]

    release_write.set()
    finish_thread.join(1.0)
    begin_thread.join(1.0)
    disable_thread.join(2.0)
    assert disable_result == [True]


def test_every_lifecycle_manifest_snapshot_is_enqueued_outside_condition(
    tmp_path, monkeypatch,
):
    import queue

    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original_publish = manager._publish_manifest_snapshot
    original_atomic = manager._atomic_write
    enqueued = []
    written_lock_states = []

    def assert_unlocked_enqueue(snapshot):
        if snapshot[1] is not None:
            enqueued.append((
                json.loads(snapshot[1])["state"],
                manager._condition._is_owned(),
            ))
        return original_publish(snapshot)

    def assert_unlocked_write(path, data):
        if path.name.startswith("manifest-"):
            written_lock_states.append(manager._condition._is_owned())
        return original_atomic(path, data)

    monkeypatch.setattr(manager, "_publish_manifest_snapshot", assert_unlocked_enqueue)
    monkeypatch.setattr(manager, "_atomic_write", assert_unlocked_write)
    assert manager.set_enabled(True)
    assert manager.set_enabled(False, timeout_seconds=1.0)

    assert manager.set_enabled(True)
    active_reference = manager.begin_cycle(
        _configuration(), {"trigger": "automatic"},
    )
    assert manager.set_enabled(False, timeout_seconds=0.01) is False
    manager.finish_cycle(active_reference, _strict_result(), "planning_only")
    assert manager.set_enabled(False, timeout_seconds=1.0)

    assert manager.set_enabled(True)
    writer_queue = manager._queue

    class SentinelFullQueue:
        @staticmethod
        def empty():
            return True

        @staticmethod
        def put_nowait(_item):
            raise queue.Full

    manager._queue = SentinelFullQueue()
    assert manager.set_enabled(False, timeout_seconds=1.0) is False
    manager._queue = writer_queue
    assert manager.set_enabled(False, timeout_seconds=1.0)
    _wait_for_manifest_idle(manager)

    states = [state for state, _owned in enqueued]
    assert {"active", "draining", "closed"} <= set(states)
    assert states.count("active") >= 3
    assert all(not owned for _state, owned in enqueued)
    assert written_lock_states
    assert all(not owned for owned in written_lock_states)


def test_disable_wall_clock_bound_includes_stuck_manifest_and_keeps_intake_free(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    first = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    blocker = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    original = manager._atomic_write
    write_started = threading.Event()
    release_write = threading.Event()

    def stuck_manifest(path, data):
        if path.name.startswith("manifest-") and not release_write.is_set():
            write_started.set()
            assert release_write.wait(1.0)
        return original(path, data)

    monkeypatch.setattr(manager, "_atomic_write", stuck_manifest)
    disable_result = []
    disable_done = threading.Event()
    started = time.monotonic()
    disable_thread = threading.Thread(
        target=lambda: (
            disable_result.append(manager.set_enabled(False, timeout_seconds=0.05)),
            disable_done.set(),
        ),
    )
    disable_thread.start()
    try:
        assert write_started.wait(1.0)
        finish_started = time.monotonic()
        manager.finish_cycle(first, _strict_result(), "planning_only")
        assert time.monotonic() - finish_started < 0.05
        begin_started = time.monotonic()
        assert manager.begin_cycle(
            _configuration(), {"trigger": "automatic"},
        ) is None
        assert time.monotonic() - begin_started < 0.05
        assert disable_done.wait(0.08)
        assert time.monotonic() - started < 0.10
        assert disable_result == [False]
        resumed = manager.begin_cycle(
            _configuration(), {"trigger": "automatic"},
        )
        assert resumed is not None
    finally:
        release_write.set()
        disable_thread.join(1.0)

    manager.finish_cycle(blocker, _strict_result(), "planning_only")
    manager.finish_cycle(resumed, _strict_result(), "planning_only")
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_disable_deadline_includes_lifecycle_lock_acquisition(tmp_path):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    manager._lifecycle_lock.acquire()
    started = time.monotonic()
    try:
        assert manager.set_enabled(False, timeout_seconds=0.05) is False
    finally:
        manager._lifecycle_lock.release()
    assert time.monotonic() - started < 0.08
    assert manager._enabled is True
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_disable_is_bounded_when_filesystem_lock_is_stuck(tmp_path):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    deadline = time.monotonic() + 1.0
    while not list(manager.output_dir.glob("manifest-*.json")):
        assert time.monotonic() < deadline
        time.sleep(0.005)

    manager._filesystem_lock.acquire()
    result = []
    done = threading.Event()
    started = time.monotonic()
    thread = threading.Thread(
        target=lambda: (
            result.append(manager.set_enabled(False, timeout_seconds=0.05)),
            done.set(),
        ),
    )
    thread.start()
    try:
        assert done.wait(0.08)
        assert time.monotonic() - started < 0.10
        assert result == [True]
    finally:
        manager._filesystem_lock.release()
        thread.join(1.0)


def test_blocked_run_keeps_closed_terminal_snapshot_across_next_run(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original = manager._atomic_write
    first_write_started = threading.Event()
    release_first_write = threading.Event()

    def block_first_manifest(path, data):
        if path.name.startswith("manifest-") and not first_write_started.is_set():
            first_write_started.set()
            assert release_first_write.wait(1.0)
        return original(path, data)

    monkeypatch.setattr(manager, "_atomic_write", block_first_manifest)
    try:
        assert manager.set_enabled(True)
        assert first_write_started.wait(1.0)
        first_path = manager._manifest_path()
        first_run_id = manager._run_id
        assert manager.set_enabled(False, timeout_seconds=0.05)
        assert manager.set_enabled(True)
        second_path = manager._manifest_path()
        second_run_id = manager._run_id
        assert second_run_id != first_run_id
        with manager._manifest_publish_condition:
            assert first_path in manager._pending_manifest_snapshots
            assert second_path in manager._pending_manifest_snapshots
            assert len(manager._pending_manifest_snapshots) <= 64
    finally:
        release_first_write.set()

    _wait_for_manifest_idle(manager, timeout=2.0)
    first_manifest = json.loads(first_path.read_text())
    second_manifest = json.loads(second_path.read_text())
    assert first_manifest["capture_run_id"] == first_run_id
    assert first_manifest["state"] == "closed"
    assert second_manifest["capture_run_id"] == second_run_id
    assert second_manifest["state"] == "active"
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_pending_run_bound_rejects_new_enable_without_evicting_closed_truth(
    tmp_path, monkeypatch,
):
    from modules.rebalance_cycle_capture import MAX_PENDING_MANIFEST_RUNS

    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original = manager._atomic_write
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    closed_runs = set()

    def observe_and_block_first(path, data):
        if path.name.startswith("manifest-"):
            manifest = json.loads(data)
            if not first_write_started.is_set():
                first_write_started.set()
                assert release_first_write.wait(2.0)
            if manifest.get("state") == "closed":
                closed_runs.add(manifest["capture_run_id"])
        return original(path, data)

    monkeypatch.setattr(manager, "_atomic_write", observe_and_block_first)
    assert manager.set_enabled(True)
    assert first_write_started.wait(1.0)
    successful_run_ids = []
    for index in range(MAX_PENDING_MANIFEST_RUNS):
        successful_run_ids.append(manager._run_id)
        assert manager.set_enabled(False, timeout_seconds=0.05)
        if index + 1 < MAX_PENDING_MANIFEST_RUNS:
            assert manager.set_enabled(True)

    assert manager.set_enabled(True) is False
    _assert_failed_enable_is_fully_rolled_back(
        manager, preserve_existing_publication=True,
    )
    with manager._manifest_publish_condition:
        assert len(manager._pending_manifest_snapshots) == MAX_PENDING_MANIFEST_RUNS
    release_first_write.set()
    _wait_for_manifest_idle(manager, timeout=4.0)
    assert closed_runs == set(successful_run_ids)

    assert manager.set_enabled(True)
    assert manager._run_id not in successful_run_ids
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_manifest_publication_coalesces_repeated_toggles_and_writes_newest_revision(
    tmp_path, monkeypatch,
):
    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    original = manager._atomic_write
    write_started = threading.Event()
    release_write = threading.Event()

    def stuck_first_manifest(path, data):
        if path.name.startswith("manifest-") and not write_started.is_set():
            write_started.set()
            assert release_write.wait(1.0)
        return original(path, data)

    monkeypatch.setattr(manager, "_atomic_write", stuck_first_manifest)
    enable_result = []
    enable_done = threading.Event()
    enable_thread = threading.Thread(
        target=lambda: (
            enable_result.append(manager.set_enabled(True)),
            enable_done.set(),
        ),
    )
    enable_thread.start()
    try:
        assert write_started.wait(1.0)
        assert enable_done.wait(0.08)
        assert enable_result == [True]
        publisher = manager._manifest_publisher
        for _ in range(40):
            started = time.monotonic()
            assert manager.set_enabled(False, timeout_seconds=0.05)
            assert time.monotonic() - started < 0.08
            assert manager.set_enabled(True)
            assert manager._manifest_publisher is publisher
        latest_path = manager._manifest_path()
        latest_run_id = manager._run_id
        with manager._manifest_publish_condition:
            pending = manager._pending_manifest_snapshots
            assert 1 <= len(pending) <= 64
            assert pending[latest_path][2] == manager._manifest_revision
    finally:
        release_write.set()
        enable_thread.join(1.0)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if latest_path.exists():
            persisted = json.loads(latest_path.read_text())
            if persisted.get("capture_run_id") == latest_run_id:
                break
        time.sleep(0.01)
    else:
        pytest.fail("newest coalesced manifest revision was not published")
    assert persisted["state"] == "active"
    assert manager._manifest_publisher is publisher
    assert publisher.is_alive()
    assert manager.set_enabled(False, timeout_seconds=1.0)


def test_handoff_graph_bound_rejects_before_deepcopy_and_manifests_failure(
    tmp_path,
):
    import modules.rebalance_cycle_capture as capture_module

    class OversizedMapping(dict):
        deepcopy_calls = 0

        def __len__(self):
            return capture_module.MAX_HANDOFF_GRAPH_NODES + 1

        def __deepcopy__(self, memo):
            self.deepcopy_calls += 1
            raise AssertionError("oversized graph was deep-copied")

    manager = RebalanceCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_a, **_k: None,
    )
    assert manager.set_enabled(True)
    reference = manager.begin_cycle(_configuration(), {"trigger": "automatic"})
    pair = _pair()
    oversized = OversizedMapping()
    pair.score_decomposition = oversized
    result = _strict_result(pair=pair)

    assert manager.prepare_cycle_result(reference, result) is False
    manager.finish_cycle(reference, result)

    _wait_for_failed(manager, "ValueError")
    assert oversized.deepcopy_calls == 0
    assert manager.set_enabled(False, timeout_seconds=1.0)
