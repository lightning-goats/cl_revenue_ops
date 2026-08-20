"""Review regressions for the standalone rebalance replay tool."""

import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import modules.rebalance_cycle_replay_wire as replay_wire
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "task4_replay_fixture", ROOT / "tests" / "test_rebalance_replay.py"
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
TOOL = _fixture.TOOL
_sealed_envelope = _fixture._sealed_envelope
_write_envelope = _fixture._write_envelope


def _run_tool(*arguments, timeout=2):
    return subprocess.run(
        [sys.executable, str(TOOL), *map(str, arguments)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def _reseal_received_body(body):
    """Keep received integral floats while sealing their canonical normalized body."""
    return {**body, "payload_sha256": replay_wire.seal_envelope(body)["payload_sha256"]}


def test_wire_returns_the_verified_normalized_envelope_for_integral_json_floats():
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["capture_seq"] = 1.0
    body["configuration"]["config_version"] = 1.0
    body["configuration"]["max_chunk_sats"] = 500_000.0
    body["configuration"]["max_pairs"] = 2.0
    body["configuration"]["pair_fee_cap_ppm"] = 1_000.0
    body["pre_state"]["normalized_snapshot"]["total_capacity_sats"] = 3_400_000.0
    normalized = replay_wire.verify_normalized_envelope(_reseal_received_body(body))

    assert normalized["capture_seq"] == 1
    assert isinstance(normalized["configuration"]["max_pairs"], int)
    assert isinstance(
        normalized["pre_state"]["normalized_snapshot"]["total_capacity_sats"], int
    )
    normalized_body = dict(normalized)
    supplied = normalized_body.pop("payload_sha256")
    assert supplied == hashlib.sha256(
        replay_wire.canonical_body_bytes(normalized_body)
    ).hexdigest()


def test_integral_json_float_capture_replays_as_a_match(tmp_path):
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["configuration"]["max_pairs"] = 2.0
    body["funnel"]["generated_pairs"][0]["cheap_rank"] = 1.0
    envelope = _reseal_received_body(body)

    result = _run_tool(_write_envelope(tmp_path, envelope))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "match"


def test_clean_process_blocks_broad_package_and_forbidden_transitive_imports(tmp_path):
    envelope = _sealed_envelope()
    path = _write_envelope(tmp_path, envelope)
    runner = """
import runpy
import sys

blocked = ("modules", "pyln", "numpy", "sqlite3")
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "modules" or fullname.startswith("modules.") or fullname in blocked:
            raise ImportError("blocked import: " + fullname)
        return None

tool, envelope = sys.argv[1:]
sys.meta_path.insert(0, Blocker())
sys.argv = [tool, envelope]
runpy.run_path(tool, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", runner, str(TOOL), str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=2,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "match"


@pytest.mark.parametrize("truncated", [False, True])
def test_incomplete_capture_is_a_controlled_failure(tmp_path, truncated):
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["terminal_stage"] = "planning_only"
    body["completeness"]["eligible"] = False
    body["completeness"]["candidate_universe_truncated"] = truncated
    if truncated:
        body["completeness"]["generated_pair_count"] += 1
    envelope = replay_wire.seal_envelope(body)

    result = _run_tool(_write_envelope(tmp_path, envelope))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_original_and_retained_candidate_count_mismatch_is_controlled(tmp_path):
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["completeness"]["generated_pair_count"] += 1
    envelope = {**body, "payload_sha256": envelope["payload_sha256"]}

    result = _run_tool(_write_envelope(tmp_path, envelope))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_oversized_json_integer_is_a_controlled_failure(tmp_path):
    path = tmp_path / "oversized-number.json"
    path.write_text('{"capture_seq":' + "9" * 4001 + "}", encoding="utf-8")

    result = _run_tool(path)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_nonfinite_binary64_tag_is_a_controlled_failure(tmp_path):
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["configuration"]["target_band_low"] = {"__f64__": "7ff8000000000000"}
    envelope = replay_wire.seal_envelope(body)

    result = _run_tool(_write_envelope(tmp_path, envelope))

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_symlink_input_is_a_controlled_failure(tmp_path):
    target = _write_envelope(tmp_path, _sealed_envelope())
    link = tmp_path / "linked-envelope.json"
    link.symlink_to(target)

    result = _run_tool(link)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_fifo_input_is_a_bounded_controlled_failure(tmp_path):
    fifo = tmp_path / "envelope.fifo"
    os.mkfifo(fifo)

    result = _run_tool(fifo)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_raw_integral_float_digest_is_rejected_even_when_the_body_normalizes():
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["configuration"]["max_pairs"] = 2.0
    raw_digest_envelope = {
        **body,
        "payload_sha256": hashlib.sha256(
            replay_wire.canonical_body_bytes(body)
        ).hexdigest(),
    }

    with pytest.raises(ValueError, match="payload digest mismatch"):
        replay_wire.verify_normalized_envelope(raw_digest_envelope)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_digest_valid_raw_nonfinite_json_constant_is_controlled(tmp_path, constant):
    envelope = _sealed_envelope()
    body = copy.deepcopy(envelope)
    body.pop("payload_sha256")
    body["configuration"]["target_band_low"] = float(constant.replace("Infinity", "inf").replace("NaN", "nan"))
    envelope = {
        **body,
        "payload_sha256": hashlib.sha256(
            replay_wire.canonical_body_bytes(body)
        ).hexdigest(),
    }
    path = tmp_path / "raw-nonfinite.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    result = _run_tool(path)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
