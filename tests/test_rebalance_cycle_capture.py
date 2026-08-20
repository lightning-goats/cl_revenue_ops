from types import SimpleNamespace

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
