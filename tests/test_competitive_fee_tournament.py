import importlib.util
import sys
from pathlib import Path


def load_runner():
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "competitive_fee_tournament.py"
    spec = importlib.util.spec_from_file_location("competitive_fee_tournament", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_payment_succeeded_rejects_nonzero_lncli_failure_stdout():
    runner = load_runner()

    result = {
        "ok": False,
        "returncode": 1,
        "stdout": "Payment status: FAILED, reason: FAILURE_REASON_INSUFFICIENT_BALANCE\n[lncli] FAILED",
    }

    assert runner.payment_succeeded(result) is False


def test_payment_succeeded_accepts_success_stdout():
    runner = load_runner()

    result = {
        "ok": True,
        "stdout": "Payment status: SUCCEEDED",
    }

    assert runner.payment_succeeded(result) is True


def test_payment_succeeded_rejects_structured_payment_error():
    runner = load_runner()

    result = {
        "ok": True,
        "payment_error": "unable to find a path to destination",
    }

    assert runner.payment_succeeded(result) is False


def test_graph_policy_matches_expected_policy_with_string_fields():
    runner = load_runner()

    policy = {
        "fee_rate_milli_msat": "60",
        "fee_base_msat": "1",
        "time_lock_delta": 18,
        "disabled": False,
    }

    assert runner.graph_policy_matches(policy, ppm=60, base_fee_msat=1, cltv_delta=18)


def test_graph_policy_rejects_stale_policy():
    runner = load_runner()

    policy = {
        "fee_rate_milli_msat": "100",
        "fee_base_msat": "1",
        "time_lock_delta": 18,
        "disabled": False,
    }

    assert not runner.graph_policy_matches(policy, ppm=60, base_fee_msat=1, cltv_delta=18)


def test_node_policy_selects_policy_by_pubkey():
    runner = load_runner()

    chan_info = {
        "node1_pub": "a",
        "node1_policy": {"fee_rate_milli_msat": "75"},
        "node2_pub": "b",
        "node2_policy": {"fee_rate_milli_msat": "60"},
    }

    assert runner.node_policy(chan_info, "b") == {"fee_rate_milli_msat": "60"}


def test_force_fee_cycle_prefers_manual_cycle_rpc(monkeypatch):
    runner = load_runner()
    calls = []

    def fake_cln(node, *args, check=False):
        calls.append((node, args))
        if args == ("revenue-fee-cycle",):
            return {"ok": True, "adjusted_channels": 1}
        return {"ok": True}

    monkeypatch.setattr(runner, "cln", fake_cln)

    result = runner.force_fee_cycle("/tmp/plugin.py", wait_seconds=0)

    assert result["method"] == "revenue-fee-cycle"
    assert calls == [(runner.REVENUE, ("revenue-fee-cycle",))]


def test_collect_hive_snapshot_skips_when_disabled():
    runner = load_runner()

    assert runner.collect_hive_snapshot(False) == {
        "ok": True,
        "skipped": True,
        "reason": "cl-hive disabled",
    }


def test_collect_hive_snapshot_rejects_empty_datastore_when_export_fails(monkeypatch):
    runner = load_runner()

    def fake_cln(node, *args, check=False):
        if args == ("hive-status",):
            return {"ok": True}
        if args == ("hive-export-hints",):
            return {"ok": True, "error": "Not a Hive member"}
        if args == ("listdatastore", '["hive","hints"]'):
            return {"ok": True, "datastore": []}
        return {"ok": False}

    monkeypatch.setattr(runner, "cln", fake_cln)

    assert runner.collect_hive_snapshot(True)["ok"] is False


def test_push_hive_hints_datastore_writes_export_as_hex(monkeypatch):
    runner = load_runner()
    calls = []

    def fake_cln(node, *args, check=False):
        calls.append(args)
        if args == ("hive-export-hints",):
            return {"ok": True, "competition_bias": 0.125}
        if args[0] == "datastore":
            return {"ok": True}
        return {"ok": False}

    monkeypatch.setattr(runner, "cln", fake_cln)

    result = runner.push_hive_hints_datastore()

    assert result["ok"] is True
    assert calls[0] == ("hive-export-hints",)
    assert calls[1][0] == "datastore"
    assert calls[1][1] == 'key=["hive","hints"]'
    assert calls[1][2].startswith("hex=")
    assert calls[1][3] == "mode=create-or-replace"


def test_push_hive_hints_datastore_rejects_non_member_export(monkeypatch):
    runner = load_runner()
    calls = []

    def fake_cln(node, *args, check=False):
        calls.append(args)
        if args == ("hive-export-hints",):
            return {"ok": True, "error": "Not a Hive member"}
        return {"ok": True}

    monkeypatch.setattr(runner, "cln", fake_cln)

    result = runner.push_hive_hints_datastore()

    assert result["ok"] is False
    assert result["stage"] == "hive-export-hints"
    assert calls == [("hive-export-hints",)]


def test_clear_hive_hints_datastore_deletes_existing_entry(monkeypatch):
    runner = load_runner()
    calls = []
    list_calls = 0

    def fake_cln(node, *args, check=False):
        nonlocal list_calls
        calls.append(args)
        if args == ("listdatastore", '["hive","hints"]'):
            list_calls += 1
            if list_calls == 1:
                return {
                    "ok": True,
                    "datastore": [
                        {"key": ["hive", "hints"], "generation": 7, "hex": "7b7d"}
                    ],
                }
            return {"ok": True, "datastore": []}
        if args == ("deldatastore", '["hive","hints"]', "7"):
            return {"ok": True}
        return {"ok": False}

    monkeypatch.setattr(runner, "cln", fake_cln)

    result = runner.clear_hive_hints_datastore()

    assert result["ok"] is True
    assert ("deldatastore", '["hive","hints"]', "7") in calls


def test_disable_cl_hive_stops_plugin_and_clears_hints(monkeypatch):
    runner = load_runner()
    calls = []
    stopped = False

    def fake_cln(node, *args, check=False):
        nonlocal stopped
        calls.append(args)
        if args == ("hive-status",):
            return (
                {"ok": False, "stderr": "Unknown command 'hive-status'"}
                if stopped else
                {"ok": True}
            )
        if args == ("plugin", "stop", "/tmp/cl_hive/cl-hive.py"):
            stopped = True
            return {"ok": True}
        if args == ("listdatastore", '["hive","hints"]'):
            return {"ok": True, "datastore": []}
        return {"ok": False}

    monkeypatch.setattr(runner, "cln", fake_cln)

    result = runner.disable_cl_hive()

    assert result["ok"] is True
    assert ("plugin", "stop", "/tmp/cl_hive/cl-hive.py") in calls


def test_ensure_cl_hive_deploys_starts_and_runs_genesis(tmp_path, monkeypatch):
    runner = load_runner()
    (tmp_path / "modules").mkdir()
    (tmp_path / "cl-hive.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    active = False
    member = False
    cln_calls = []
    run_calls = []

    def fake_run(cmd, check=False):
        run_calls.append(cmd)
        return {"ok": True}

    def fake_cln(node, *args, check=False):
        nonlocal active, member
        cln_calls.append(args)
        if args == ("hive-status",):
            if active:
                return {"ok": True, "status": "active"}
            return {"ok": False, "stderr": "Unknown command 'hive-status'"}
        if args == ("plugin", "start", "/tmp/cl_hive/cl-hive.py"):
            active = True
            return {"ok": True}
        if args == ("hive-export-hints",):
            if member:
                return {"ok": True, "competition_bias": 0.0}
            return {"ok": True, "error": "Not a Hive member"}
        if args == ("hive-genesis", "test-hive"):
            member = True
            return {"ok": True, "hive_id": "test-hive"}
        return {"ok": True}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "cln", fake_cln)

    result = runner.ensure_cl_hive(host_path=tmp_path, hive_id="test-hive")

    assert result["ok"] is True
    assert result["push_datastore"]["ok"] is True
    assert ("plugin", "start", "/tmp/cl_hive/cl-hive.py") in cln_calls
    assert ("hive-genesis", "test-hive") in cln_calls
    assert any(call[:1] == ("datastore",) for call in cln_calls)
    assert any(cmd[:2] == ["docker", "cp"] for cmd in run_calls)


def test_ensure_cl_hive_skips_start_when_dependencies_missing(tmp_path, monkeypatch):
    runner = load_runner()
    (tmp_path / "modules").mkdir()
    (tmp_path / "cl-hive.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    cln_calls = []

    def fake_run(cmd, check=False):
        if cmd[:5] == ["docker", "exec", "polar-n1-revenue-node", "sh", "-lc"]:
            return {"ok": False, "stderr": "python3 missing"}
        return {"ok": True}

    def fake_cln(node, *args, check=False):
        cln_calls.append(args)
        if args == ("hive-status",):
            return {"ok": False, "stderr": "Unknown command 'hive-status'"}
        if args == ("hive-export-hints",):
            return {"ok": False, "stderr": "Unknown command 'hive-export-hints'"}
        return {"ok": True}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "cln", fake_cln)

    result = runner.ensure_cl_hive(host_path=tmp_path)

    assert result["dependencies"]["ok"] is False
    assert result["start"]["reason"] == "dependency_probe_failed"
    assert ("plugin", "start", "/tmp/cl_hive/cl-hive.py") not in cln_calls
