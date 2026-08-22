"""Black-box tests for the local-only rebalance planner replay CLI."""

from dataclasses import asdict
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from modules.rebalance_cycle_replay_wire import seal_envelope
from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_state_v2 import ChannelState, StateSnapshot


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "rebalance_replay.py"


def _channel(channel_id, peer_id, local_ratio, *, capacity_sats=1_000_000,
             remaining_budget_sats=1_000, is_valuable=True, is_active=True):
    return ChannelState(
        channel_id=channel_id, peer_id=peer_id, capacity_sats=capacity_sats,
        local_ratio=local_ratio, actual_inbound_fee_ppm=120,
        value_class="active" if is_valuable else "neutral", is_valuable=is_valuable,
        remaining_budget_sats=remaining_budget_sats, cooldown_active=False,
        source_eligible=True,
        dest_eligible=is_valuable and remaining_budget_sats > 0,
        source_reason="",
        dest_reason="" if is_valuable and remaining_budget_sats > 0 else "no_budget",
        dest_urgency=max(0.0, (0.35 - local_ratio) / 0.35),
        source_drain_score=max(0.0, (local_ratio - 0.65) / 0.35),
        budget_source="capex", local_out_fee_ppm=250,
        historical_direct_fee_ppm=80.0, historical_sourced_fee_ppm=20.0,
        is_active=is_active, realized_utilization=0.6,
        utilization_is_realized=True, activity_out_sats=500, activity_in_sats=250,
        target_band_low=0.35, target_band_high=0.65,
    )


def _planner_projection(pair):
    return {
        "source_channel_id": pair.source_channel_id,
        "dest_channel_id": pair.dest_channel_id,
        "planned_amount_sats": pair.amount_sats,
        "pair_budget_sats": pair.pair_budget_sats,
        "source_excess_sats": pair.source_excess_sats,
        "dest_need_sats": pair.dest_need_sats,
        "max_chunk_sats": pair.max_chunk_sats,
        "cheap_rank": pair.cheap_rank,
        "cheap_score": pair.score,
        "planner_selected": pair.planner_selected,
        "planner_rejection_reason": pair.planner_rejection_reason or None,
        "bootstrap_score_decomposition": pair.score_decomposition,
    }


def _sealed_envelope(*, neutral=False):
    channels = () if neutral else (
        _channel("src-a", "02" + "aa" * 32, 0.91),
        _channel("src-b", "02" + "bb" * 32, 0.78),
        _channel("dest-a", "02" + "cc" * 32, 0.09, capacity_sats=800_000),
        _channel("dest-b", "02" + "dd" * 32, 0.22, capacity_sats=600_000),
    )
    snapshot = StateSnapshot(
        channels=channels,
        total_capacity_sats=sum(channel.capacity_sats for channel in channels),
        total_remaining_budget_sats=sum(
            channel.remaining_budget_sats for channel in channels
        ),
        valuable_channel_count=sum(channel.is_valuable for channel in channels),
    )
    configuration = {
        "config_version": 1, "target_band_low": 0.35, "target_band_high": 0.65,
        "max_chunk_sats": 500_000, "max_pairs": 2, "pair_fee_cap_ppm": 1_000,
    }
    plan = RebalancePlanner(
        target_band_low=configuration["target_band_low"],
        target_band_high=configuration["target_band_high"],
        max_chunk_sats=configuration["max_chunk_sats"],
        max_pairs=configuration["max_pairs"],
        pair_fee_cap_ppm=configuration["pair_fee_cap_ppm"],
    ).plan(snapshot)
    generated = [_planner_projection(pair) for pair in plan.generated]
    body = {
        "schema_name": "rebalance_cycle_replay", "schema_version": 0,
        "capture_run_id": "a" * 32, "capture_seq": 1,
        "cycle_id": "a" * 32 + ":00000001", "terminal_stage": "completed",
        "producer": {
            "started_at": "2026-08-20T18:00:00+00:00",
            "completed_at": "2026-08-20T18:00:01+00:00",
            "python_commit": "7630b75", "algorithm_version": "rebalance-v2-phase1a",
            "trigger": "test",
        },
        "configuration": configuration,
        "pre_state": {"normalized_snapshot": {
            "channels": [asdict(channel) for channel in channels],
            "total_capacity_sats": snapshot.total_capacity_sats,
            "total_remaining_budget_sats": snapshot.total_remaining_budget_sats,
            "valuable_channel_count": snapshot.valuable_channel_count,
        }},
        "funnel": {
            "generated_pairs": [{
                **pair, "score_decomposition": {}, "route_cost_sats": None,
                "effective_budget_sats": None, "rejection_reason": "", "route_summary": [],
            } for pair in generated],
            "planner_selected_pairs": [{
                "source_channel_id": pair.source_channel_id,
                "dest_channel_id": pair.dest_channel_id,
            } for pair in plan.selected],
            "final_selected_pairs": [], "skipped": [],
        },
        "execution": {"pair_outcomes": []},
        "completeness": {
            "generated_pair_count": len(generated),
            "retained_generated_pair_count": len(generated),
            "planner_selected_pair_count": len(plan.selected),
            "final_selected_pair_count": 0, "execution_outcome_count": 0,
            "candidate_universe_truncated": False, "eligible": True,
        },
    }
    return seal_envelope(body)


def _write_envelope(tmp_path, envelope):
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def _run_tool(*arguments):
    return subprocess.run(
        [sys.executable, str(TOOL), *map(str, arguments)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _expected_match_output(envelope):
    return {
        "status": "match", "cycle_id": envelope["cycle_id"],
        "generated_pairs_match": True, "planner_selected_pairs_match": True,
        "ev_gate_matches": True,
        "ev_gate_pairs_checked": envelope["completeness"][
            "final_selected_pair_count"
        ],
        "mismatches": [],
    }


def test_replays_sealed_real_channel_state_with_byte_stable_structured_output(tmp_path):
    envelope = _sealed_envelope()
    result = _run_tool(_write_envelope(tmp_path, envelope))

    expected = _expected_match_output(envelope)
    assert result.returncode == 0, result.stderr
    assert result.stdout == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_pretty_output_is_structured_match_output(tmp_path):
    envelope = _sealed_envelope()
    result = _run_tool(_write_envelope(tmp_path, envelope), "--pretty")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == _expected_match_output(envelope)
    assert result.stdout.startswith("{\n")


@pytest.mark.parametrize("field", ["planned_amount_sats", "cheap_rank"])
def test_valid_envelope_with_tampered_expected_planner_output_returns_mismatch(tmp_path, field):
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["funnel"]["generated_pairs"][0][field] += 1
    if field == "cheap_rank":
        body["funnel"]["generated_pairs"][1]["cheap_rank"] -= 1
    tampered = seal_envelope(body)

    result = _run_tool(_write_envelope(tmp_path, tampered))

    output = json.loads(result.stdout)
    assert result.returncode == 1
    assert output["status"] == "mismatch"
    assert output["generated_pairs_match"] is False
    assert output["planner_selected_pairs_match"] is True
    assert output["mismatches"] == ["generated_pairs"]


def test_tampered_digest_is_a_controlled_input_failure(tmp_path):
    envelope = _sealed_envelope()
    envelope["payload_sha256"] = "0" * 64

    result = _run_tool(_write_envelope(tmp_path, envelope))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "payload digest mismatch" in result.stderr


def test_malformed_json_is_a_controlled_input_failure(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text("{", encoding="utf-8")

    result = _run_tool(path)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid JSON" in result.stderr


@pytest.mark.parametrize("option", ["--apply", "--execute", "--rpc", "--unknown"])
def test_action_and_unknown_options_are_rejected(tmp_path, option):
    result = _run_tool(_write_envelope(tmp_path, _sealed_envelope()), option)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage:" in result.stderr


def test_neutral_empty_snapshot_replays_without_action(tmp_path):
    envelope = _sealed_envelope(neutral=True)

    result = _run_tool(_write_envelope(tmp_path, envelope))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == _expected_match_output(envelope)


# --- Recorded-price EV gate replay ------------------------------------------


def _ev_decomposition(final_score_sats=1.0, beats_do_nothing=True):
    return {
        "model_version": "v2-sats-ev",
        "p_success": 0.75,
        "expected_fee_sats": 2,
        "rejection_reason": "",
        "expected_utilization": 0.6,
        "source_utilization": 0.6,
        "source_utilization_discount": 0.5,
        "activity_penalty_sats": 0.0,
        "final_score_sats": final_score_sats,
        "beats_do_nothing": beats_do_nothing,
        "inputs": {
            "dest_value_fee_ppm": 100.0,
            "source_historical_sourced_fee_ppm": 20.0,
            "source_opportunity_fee_ppm": 80.0,
            "dest_out_fee_ppm": 250,
            "failure_count": 0,
            "expected_fee_sats": 2,
            "pair_budget_sats": 1_000,
            "effective_budget_sats": 1_000,
        },
    }


def _envelope_with_ev_final_pair():
    """Seal a real planner envelope plus one recorded-price final pair.

    Uses the planner's own first selected pair so the recorded planner
    selection stays byte-identical under replay; only the recorded-price
    gate evidence is synthetic.
    """
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    selected = body["funnel"]["planner_selected_pairs"][0]
    identity = {
        "source_channel_id": selected["source_channel_id"],
        "dest_channel_id": selected["dest_channel_id"],
    }
    final_pair = {
        **identity,
        "planned_amount_sats": 100_000,
        "pair_budget_sats": 1_000,
        "source_excess_sats": 1,
        "dest_need_sats": 1,
        "max_chunk_sats": 500_000,
        "cheap_rank": 1,
        "cheap_score": 0.5,
        "planner_selected": True,
        "planner_rejection_reason": None,
        "bootstrap_score_decomposition": {},
        "score_decomposition": _ev_decomposition(),
        "route_cost_sats": 2,
        "effective_budget_sats": 1_000,
        "rejection_reason": "",
        "route_summary": [],
    }
    body["funnel"]["final_selected_pairs"] = [final_pair]
    body["completeness"]["planner_selected_pair_count"] = len(
        body["funnel"]["planner_selected_pairs"]
    )
    body["completeness"]["final_selected_pair_count"] = 1
    return seal_envelope(body)


def test_match_output_includes_ev_gate_fields():
    envelope = _sealed_envelope()
    expected = _expected_match_output(envelope)
    assert expected["ev_gate_matches"] is True
    assert expected["ev_gate_pairs_checked"] == 0


def test_sealed_ev_final_pair_replays_with_gate_verdict_match(tmp_path):
    envelope = _envelope_with_ev_final_pair()

    result = _run_tool(_write_envelope(tmp_path, envelope))

    output = json.loads(result.stdout)
    assert result.returncode == 0, result.stderr
    assert output["status"] == "match"
    assert output["ev_gate_matches"] is True
    assert output["ev_gate_pairs_checked"] == 1
    assert output["mismatches"] == []


def test_tampered_recorded_final_score_is_a_gate_mismatch(tmp_path):
    envelope = _envelope_with_ev_final_pair()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    decomposition = body["funnel"]["final_selected_pairs"][0][
        "score_decomposition"
    ]
    # Sealed envelopes store floats as binary64 tags; decode, tamper, retag.
    from modules.rebalance_cycle_replay_wire import (
        BINARY64_TAG_KEY, tag_floats,
    )
    import struct

    decoded = struct.unpack(
        ">d", bytes.fromhex(decomposition["final_score_sats"][BINARY64_TAG_KEY])
    )[0]
    decomposition["final_score_sats"] = tag_floats(decoded + 5.0)
    tampered = seal_envelope(body)

    result = _run_tool(_write_envelope(tmp_path, tampered))

    output = json.loads(result.stdout)
    assert result.returncode == 1
    assert output["status"] == "mismatch"
    assert output["generated_pairs_match"] is True
    assert output["planner_selected_pairs_match"] is True
    assert output["ev_gate_matches"] is False
    assert output["mismatches"] == ["ev_gate"]


def test_tampered_beats_do_nothing_is_a_gate_mismatch(tmp_path):
    envelope = _envelope_with_ev_final_pair()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    decomposition = body["funnel"]["final_selected_pairs"][0][
        "score_decomposition"
    ]
    # Recompute-consistent score but flipped verdict: only the verdict lies.
    decomposition["beats_do_nothing"] = not decomposition["beats_do_nothing"]
    tampered = seal_envelope(body)

    result = _run_tool(_write_envelope(tmp_path, tampered))

    output = json.loads(result.stdout)
    assert result.returncode == 1
    assert output["status"] == "mismatch"
    assert output["ev_gate_matches"] is False


def test_unknown_gate_model_version_is_a_controlled_input_failure(tmp_path):
    # Unknown model versions are rejected at wire validation; build the
    # envelope by bypassing validation to exercise the tool's defense.
    from modules.rebalance_cycle_replay_wire import (
        canonical_body_bytes, tag_floats,
    )
    import hashlib

    body = copy.deepcopy(_envelope_with_ev_final_pair())
    body.pop("payload_sha256")
    body["funnel"]["final_selected_pairs"][0]["score_decomposition"][
        "model_version"
    ] = "v9-future"
    digest = hashlib.sha256(canonical_body_bytes(tag_floats(body))).hexdigest()

    path = tmp_path / "unknown-model.json"
    path.write_text(
        json.dumps({**body, "payload_sha256": digest}), encoding="utf-8",
    )

    result = _run_tool(path)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "model" in result.stderr.lower()
