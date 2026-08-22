"""Read a sealed rebalance capture and replay its pure planner result."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import types


_ROOT = Path(__file__).resolve().parents[1]
_PURE_MODULES = (
    "utils",
    "rebalance_route_policy",
    "rebalance_types_v2",
    "rebalance_state_v2",
    "rebalance_planner_v2",
    "rebalance_cycle_replay_wire",
    "rebalance_ev_model",
)
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
_COMPONENTS = None


def _load_pure_components():
    """Load the planner's pure modules without executing modules/__init__.py."""
    if "modules" in sys.modules:
        raise ValueError("standalone replay refuses a preloaded modules package")
    package = types.ModuleType("modules")
    package.__package__ = "modules"
    package.__path__ = [str(_ROOT / "modules")]
    sys.modules["modules"] = package
    loaded = {}
    try:
        for name in _PURE_MODULES:
            module_name = f"modules.{name}"
            module = types.ModuleType(module_name)
            module.__file__ = str(_ROOT / "modules" / f"{name}.py")
            module.__package__ = "modules"
            sys.modules[module_name] = module
            source = Path(module.__file__).read_text(encoding="utf-8")
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            loaded[name] = module
    except BaseException:
        for name in _PURE_MODULES:
            sys.modules.pop(f"modules.{name}", None)
        sys.modules.pop("modules", None)
        raise
    return (
        loaded["rebalance_cycle_replay_wire"],
        loaded["rebalance_ev_model"],
        loaded["rebalance_planner_v2"].RebalancePlanner,
        loaded["rebalance_state_v2"].ChannelState,
        loaded["rebalance_state_v2"].StateSnapshot,
    )


def _components():
    global _COMPONENTS
    if _COMPONENTS is None:
        _COMPONENTS = _load_pure_components()
    return _COMPONENTS


def _decode_floats(value, binary64_tag_key):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numeric value")
        return value
    if isinstance(value, dict):
        if set(value) == {binary64_tag_key}:
            decoded = struct.unpack(">d", bytes.fromhex(value[binary64_tag_key]))[0]
            if not math.isfinite(decoded):
                raise ValueError("non-finite binary64 value")
            return decoded
        return {
            key: _decode_floats(item, binary64_tag_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_floats(item, binary64_tag_key) for item in value]
    return value


def _canonical_bytes(value, tag_floats):
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


def _reconstruct_snapshot(envelope, channel_state, state_snapshot):
    snapshot = envelope["pre_state"]["normalized_snapshot"]
    channels = tuple(
        channel_state(**{field: channel[field] for field in _CHANNEL_FIELDS})
        for channel in snapshot["channels"]
    )
    return state_snapshot(
        channels=channels,
        **{field: snapshot[field] for field in _SNAPSHOT_FIELDS},
    )


def _planner(envelope, planner_class):
    configuration = envelope["configuration"]
    captured = {field: configuration[field] for field in _CONFIGURATION_FIELDS}
    return planner_class(
        target_band_low=captured["target_band_low"],
        target_band_high=captured["target_band_high"],
        max_chunk_sats=captured["max_chunk_sats"],
        max_pairs=captured["max_pairs"],
        pair_fee_cap_ppm=captured["pair_fee_cap_ppm"],
    )


def _require_complete_capture(envelope):
    completeness = envelope["completeness"]
    if not completeness["eligible"]:
        raise ValueError("replay evidence is ineligible")
    if completeness["candidate_universe_truncated"]:
        raise ValueError("replay candidate universe is truncated")
    if (
        completeness["generated_pair_count"]
        != completeness["retained_generated_pair_count"]
    ):
        raise ValueError("replay generated-pair counts differ")


def replay(envelope):
    wire, ev_model, planner_class, channel_state, state_snapshot = _components()
    normalized = wire.verify_normalized_envelope(envelope)
    _require_complete_capture(normalized)
    decoded = _decode_floats(normalized, wire.BINARY64_TAG_KEY)
    result = _planner(decoded, planner_class).plan(
        _reconstruct_snapshot(decoded, channel_state, state_snapshot)
    )
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
    generated_matches = _canonical_bytes(generated, wire.tag_floats) == _canonical_bytes(
        expected_generated, wire.tag_floats
    )
    selected_matches = _canonical_bytes(selected, wire.tag_floats) == _canonical_bytes(
        expected_selected, wire.tag_floats
    )
    mismatches = []
    if not generated_matches:
        mismatches.append("generated_pairs")
    if not selected_matches:
        mismatches.append("planner_selected_pairs")

    gate_checked = 0
    gate_matches = True
    for index, final_pair in enumerate(decoded["funnel"]["final_selected_pairs"]):
        decomposition = final_pair.get("score_decomposition")
        if not isinstance(decomposition, dict):
            raise ValueError(f"final pair {index} has no score_decomposition")
        recomputed = ev_model.recompute_gate(
            decomposition,
            amount_sats=final_pair["planned_amount_sats"],
        )
        recorded_final_score_sats = decomposition["final_score_sats"]
        recorded_beats_do_nothing = decomposition["beats_do_nothing"]
        gate_checked += 1
        if (
            recomputed["final_score_sats"] != recorded_final_score_sats
            or recomputed["beats_do_nothing"] is not recorded_beats_do_nothing
        ):
            gate_matches = False
    if not gate_matches:
        mismatches.append("ev_gate")

    return {
        "status": "match" if not mismatches else "mismatch",
        "cycle_id": decoded["cycle_id"],
        "generated_pairs_match": generated_matches,
        "planner_selected_pairs_match": selected_matches,
        "ev_gate_matches": gate_matches,
        "ev_gate_pairs_checked": gate_checked,
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


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _load_envelope(filename, maximum_bytes):
    required_flags = ("O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_flags):
        raise ValueError("safe local-file reading is unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(os.fspath(filename), flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("input must be a regular file")
        payload = os.read(descriptor, maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise ValueError("input exceeds 32 MiB")
    try:
        return json.loads(
            payload.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid JSON") from error


def main(arguments=None):
    try:
        filename, pretty = _parse_arguments(sys.argv[1:] if arguments is None else arguments)
        wire, _ev_model, _planner_class, _channel_state, _state_snapshot = _components()
        output = replay(_load_envelope(filename, wire.MAX_ENVELOPE_BYTES))
    except (ArithmeticError, OSError, TypeError, ValueError, struct.error) as error:
        sys.stderr.write(f"rebalance replay error: {error}\n")
        return 2

    if pretty:
        sys.stdout.write(json.dumps(output, sort_keys=True, indent=2) + "\n")
    else:
        sys.stdout.write(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if output["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
