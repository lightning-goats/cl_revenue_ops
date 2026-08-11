"""Offline production-cutover tool tests; no CLN calls are made."""

import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

from modules.database import Database


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools/liquidity_decommission_preflight.py"
RENDER = ROOT / "tools/render_liquidity_decommission_config.py"
PEER_A = "02" + "a" * 64
PEER_B = "02" + "b" * 64


def _db(tmp_path):
    path = tmp_path / "revenue.db"
    db = Database(str(path), MagicMock())
    db.initialize()
    return db, path


def _run(script, *args):
    return subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def _policy(conn, peer):
    conn.execute(
        "INSERT INTO peer_policies "
        "(peer_id,strategy,rebalance_mode,tags,updated_at) VALUES (?,?,?,?,?)",
        (peer, "dynamic", "enabled", json.dumps(["no_close"]), int(time.time())),
    )


def _contract(conn, *, status="active", outgoing=PEER_A, incoming=PEER_B):
    conn.execute(
        "INSERT INTO lnplus_swaps "
        "(swap_id,status,capacity_sats,duration_months,ends_at,outbound_peer,"
        "incoming_peer,applied_at,tag_added,incoming_tag_added) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("private-contract-id", status, 100_000, 3, int(time.time()) + 86400,
         outgoing, incoming, int(time.time()) - 100, 1, 1),
    )


def test_preflight_safe_contract_only_is_read_only_and_private(tmp_path):
    db, db_path = _db(tmp_path)
    conn = db._get_connection()
    _policy(conn, PEER_A)
    _policy(conn, PEER_B)
    _contract(conn)
    db.close_all_connections()
    before = (db_path.read_bytes(), db_path.stat().st_mtime_ns)
    output = tmp_path / "preflight.json"
    result = _run(PREFLIGHT, "--db", db_path, "--output", output)
    assert result.returncode == 0, result.stderr
    assert "private-contract-id" not in result.stdout + result.stderr
    assert (db_path.read_bytes(), db_path.stat().st_mtime_ns) == before
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["preconditions"]["safe"] is True
    assert "pending_boltz_rows" not in payload["preconditions"]
    assert {c["direction"] for c in payload["contracts"]} == {"outbound", "incoming"}
    assert {c["tag_removal_owner"] for c in payload["contracts"]} == {"operator"}


@pytest.mark.parametrize("status", ["applied", "opening", "opened"])
def test_preflight_rejects_noncontract_lnplus_states(tmp_path, status):
    db, db_path = _db(tmp_path)
    _contract(db._get_connection(), status=status)
    db.close_all_connections()
    result = _run(PREFLIGHT, "--db", db_path, "--output", tmp_path / "out.json")
    assert result.returncode != 0


def test_preflight_rejects_missing_contract_protection(tmp_path):
    db, db_path = _db(tmp_path)
    conn = db._get_connection()
    _policy(conn, PEER_A)
    _contract(conn)
    db.close_all_connections()
    result = _run(PREFLIGHT, "--db", db_path, "--output", tmp_path / "out.json")
    assert result.returncode != 0


@pytest.mark.parametrize("category", ["boltz", "lnplus", "channel_open", "channel_close"])
def test_preflight_rejects_active_retired_reservations(tmp_path, category):
    db, db_path = _db(tmp_path)
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO spend_reservations "
        "(reservation_id,category,reserved_sats,reserved_at,status) VALUES (?,?,?,?,?)",
        ("retired-r", category, 10, int(time.time()), "active"),
    )
    db.close_all_connections()
    result = _run(PREFLIGHT, "--db", db_path, "--output", tmp_path / "out.json")
    assert result.returncode != 0


def test_preflight_rejects_unsafe_existing_output(tmp_path):
    db, db_path = _db(tmp_path)
    db.close_all_connections()
    output = tmp_path / "out.json"
    output.write_text("old")
    output.chmod(0o644)
    result = _run(PREFLIGHT, "--db", db_path, "--output", output)
    assert result.returncode != 0
    assert output.read_text() == "old"


def test_config_renderer_removes_retired_options_and_builds_exact_rollback(tmp_path):
    source = tmp_path / "cln.conf"
    source_bytes = (
        b"# keep me\nnetwork=bitcoin\n"
        b" revenue-ops-boltz-enabled = true # duplicate one\n"
        b"revenue-ops-boltz-enabled=true\n"
        b"revenue-ops-lnplus-swaps-enabled=true\n"
        b"revenue-ops-planner-enabled=true\n"
        b"revenue-ops-expansion-treasury-enabled=true\n"
        b"plugin=/somewhere/else\n"
    )
    source.write_bytes(source_bytes)
    before_mtime = source.stat().st_mtime_ns
    active = tmp_path / "active.conf"
    rollback = tmp_path / "rollback.conf"
    result = _run(
        RENDER, "--input", source, "--active-output", active,
        "--rollback-output", rollback,
    )
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == source_bytes
    assert source.stat().st_mtime_ns == before_mtime
    assert stat.S_IMODE(active.stat().st_mode) == 0o600
    assert stat.S_IMODE(rollback.stat().st_mode) == 0o600
    active_text = active.read_text()
    assert "network=bitcoin" in active_text and "plugin=/somewhere/else" in active_text
    for prefix in (
        "revenue-ops-boltz-", "revenue-ops-lnplus-",
        "revenue-ops-planner-", "revenue-ops-expansion-treasury-",
    ):
        assert prefix not in active_text
    gates = {
        "revenue-ops-planner-enabled": "false",
        "revenue-ops-planner-dry-run": "true",
        "revenue-ops-planner-max-opens-per-cycle": "0",
        "revenue-ops-planner-execute-closes": "false",
        "revenue-ops-planner-max-closes-per-cycle": "0",
        "revenue-ops-boltz-enabled": "false",
        "revenue-ops-boltz-auto-cycle-enabled": "false",
        "revenue-ops-expansion-treasury-enabled": "false",
        "revenue-ops-lnplus-swaps-enabled": "false",
        "revenue-ops-lnplus-execute-applications": "false",
    }
    effective = {}
    for raw in rollback.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split("=", 1))
        effective.setdefault(key, []).append(value.split("#", 1)[0].strip())
    for key, value in gates.items():
        assert effective[key] == [value]


@pytest.mark.parametrize("which", ["input", "active", "rollback"])
def test_config_renderer_rejects_symlinks(tmp_path, which):
    source = tmp_path / "source.conf"
    source.write_text("network=bitcoin\n")
    active = tmp_path / "active.conf"
    rollback = tmp_path / "rollback.conf"
    target = tmp_path / "target"
    target.write_text("target")
    paths = {"input": source, "active": active, "rollback": rollback}
    paths[which].unlink(missing_ok=True)
    paths[which].symlink_to(target)
    result = _run(
        RENDER, "--input", paths["input"], "--active-output", paths["active"],
        "--rollback-output", paths["rollback"],
    )
    assert result.returncode != 0
    assert target.read_text() == "target"


def test_config_renderer_refuses_existing_output_and_malformed_lines(tmp_path):
    source = tmp_path / "source.conf"
    source.write_text("not an assignment\n")
    active = tmp_path / "active.conf"
    active.write_text("existing")
    result = _run(
        RENDER, "--input", source, "--active-output", active,
        "--rollback-output", tmp_path / "rollback.conf",
    )
    assert result.returncode != 0
    assert active.read_text() == "existing"


def _load_tool(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preflight_rejects_malformed_active_contract(tmp_path):
    db, db_path = _db(tmp_path)
    conn = db._get_connection()
    _policy(conn, PEER_A)
    _policy(conn, PEER_B)
    _contract(conn)
    conn.execute("UPDATE lnplus_swaps SET ends_at = ?", ("not-a-time",))
    db.close_all_connections()
    result = _run(PREFLIGHT, "--db", db_path, "--output", tmp_path / "out.json")
    assert result.returncode != 0


def test_preflight_exclusive_create_race_does_not_overwrite(tmp_path, monkeypatch):
    db, db_path = _db(tmp_path)
    db.close_all_connections()
    output = tmp_path / "out.json"
    tool = _load_tool(PREFLIGHT, "preflight_race_test")
    original_inspect = tool._inspect

    def race(conn):
        result = original_inspect(conn)
        output.write_text("racer")
        output.chmod(0o600)
        return result

    monkeypatch.setattr(tool, "_inspect", race)
    with pytest.raises(FileExistsError):
        tool.run(db_path, output)
    assert output.read_text() == "racer"


def test_config_renderer_rejects_malformed_retired_assignment(tmp_path):
    source = tmp_path / "source.conf"
    source.write_text("revenue-ops-boltz-enabled true\n")
    result = _run(
        RENDER, "--input", source, "--active-output", tmp_path / "active.conf",
        "--rollback-output", tmp_path / "rollback.conf",
    )
    assert result.returncode != 0
    assert not (tmp_path / "active.conf").exists()


def test_config_renderer_removes_first_output_if_second_write_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.conf"
    source.write_text("network=bitcoin\n")
    active = tmp_path / "active.conf"
    rollback = tmp_path / "rollback.conf"
    tool = _load_tool(RENDER, "renderer_pair_failure_test")
    original_write = tool._exclusive_write
    calls = []

    def fail_second(path, payload):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated paired write failure")
        original_write(path, payload)

    monkeypatch.setattr(tool, "_exclusive_write", fail_second)
    with pytest.raises(OSError):
        tool.run(source, active, rollback)
    assert not active.exists()
    assert not rollback.exists()


def test_config_renderer_cleans_partial_file_after_write_error(tmp_path, monkeypatch):
    tool = _load_tool(RENDER, "renderer_partial_write_test")
    output = tmp_path / "partial.conf"
    monkeypatch.setattr(tool.os, "write", MagicMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        tool._exclusive_write(output, b"payload")
    assert not output.exists()
