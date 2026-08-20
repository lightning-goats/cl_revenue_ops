"""Read a sealed rebalance capture and replay its pure planner result."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.rebalance_cycle_replay_wire import (  # noqa: E402
    BINARY64_TAG_KEY,
    MAX_ENVELOPE_BYTES,
    tag_floats,
    verify_envelope,
)
from modules.rebalance_planner_v2 import RebalancePlanner  # noqa: E402
from modules.rebalance_state_v2 import ChannelState, StateSnapshot  # noqa: E402


_CHANNEL_FIELDS = (
    "channel_id", "peer_id", "capacity_sats", "local_ratio",
    "actual_inbound_fee_ppm", "value_class", "is_valuable",
    "remaining_budget_sats", "cooldown_active", "source_eligible",
    "dest_eligible", "source_reason", "dest_reason", "dest_urgency",
    "source_drain_score", "budget_source", "local_out_fee_ppm",
    "historical_direct_fee_ppm", "historical_sourced_fee_ppm", "is_active",
    "realized_utilization", "utilization_is_realized", "activity_out_sats",
    "activity_in_sats", "target_band_low", "target_band_high",
)
_SNAPSHOT_FIELDS = (
    "total_capacity_sats", "total_remaining_budget_sats", "valuable_channel_count",
)
_CONFIGURATION_FIELDS = (
    "config_version", "target_band_low", "target_band_high", "max_chunk_sats",
    "max_pairs", "pair_fee_cap_ppm",
)


def _decode_floats(value):
    if isinstance(value, dict):
        if set(value) == {BINARY64_TAG_KEY}:
            return struct.unpack(">d", bytes.fromhex(value[BINARY64_TAG_KEY]))[0]
        return {key: _decode_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_floats(item) for item in value]
    return value


def _canonical_bytes(value):
    return json.dumps(
        tag_floats(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _generated_projection(pair):
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


def _captured_generated_projection(pair):
    fields = (
        "source_channel_id", "dest_channel_id", "planned_amount_sats",
        "pair_budget_sats", "source_excess_sats", "dest_need_sats",
        "max_chunk_sats", "cheap_rank", "cheap_score", "planner_selected",
        "planner_rejection_reason", "bootstrap_score_decomposition",
    )
    return {field: pair[field] for field in fields}


def _reconstruct_snapshot(envelope):
    snapshot = envelope["pre_state"]["normalized_snapshot"]
    channels = tuple(
        ChannelState(**{field: channel[field] for field in _CHANNEL_FIELDS})
        for channel in snapshot["channels"]
    )
    return StateSnapshot(
        channels=channels,
        **{field: snapshot[field] for field in _SNAPSHOT_FIELDS},
    )


def _planner(envelope):
    configuration = envelope["configuration"]
    captured = {field: configuration[field] for field in _CONFIGURATION_FIELDS}
    return RebalancePlanner(
        target_band_low=captured["target_band_low"],
        target_band_high=captured["target_band_high"],
        max_chunk_sats=captured["max_chunk_sats"],
        max_pairs=captured["max_pairs"],
        pair_fee_cap_ppm=captured["pair_fee_cap_ppm"],
    )


def replay(envelope):
    verify_envelope(envelope)
    decoded = _decode_floats(envelope)
    result = _planner(decoded).plan(_reconstruct_snapshot(decoded))
    generated = [_generated_projection(pair) for pair in result.generated]
    expected_generated = [
        _captured_generated_projection(pair)
        for pair in decoded["funnel"]["generated_pairs"]
    ]
    selected = [
        {"source_channel_id": pair.source_channel_id, "dest_channel_id": pair.dest_channel_id}
        for pair in result.selected
    ]
    expected_selected = decoded["funnel"]["planner_selected_pairs"]
    generated_matches = _canonical_bytes(generated) == _canonical_bytes(expected_generated)
    selected_matches = _canonical_bytes(selected) == _canonical_bytes(expected_selected)
    mismatches = []
    if not generated_matches:
        mismatches.append("generated_pairs")
    if not selected_matches:
        mismatches.append("planner_selected_pairs")
    return {
        "status": "match" if not mismatches else "mismatch",
        "cycle_id": decoded["cycle_id"],
        "generated_pairs_match": generated_matches,
        "planner_selected_pairs_match": selected_matches,
        "mismatches": mismatches,
    }


def _parse_arguments(arguments):
    if len(arguments) == 1 and not arguments[0].startswith("-"):
        return arguments[0], False
    if (
        len(arguments) == 2
        and not arguments[0].startswith("-")
        and arguments[1] == "--pretty"
    ):
        return arguments[0], True
    raise ValueError("usage: rebalance_replay.py <envelope.json> [--pretty]")


def _load_envelope(filename):
    path = Path(filename)
    if path.stat().st_size > MAX_ENVELOPE_BYTES:
        raise ValueError("input exceeds 32 MiB")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON") from error


def main(arguments=None):
    try:
        filename, pretty = _parse_arguments(sys.argv[1:] if arguments is None else arguments)
        output = replay(_load_envelope(filename))
    except (OSError, TypeError, ValueError, struct.error) as error:
        sys.stderr.write(f"rebalance replay error: {error}\n")
        return 2

    if pretty:
        sys.stdout.write(json.dumps(output, sort_keys=True, indent=2) + "\n")
    else:
        sys.stdout.write(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if output["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
