import json
from pathlib import Path

from tools import revenue_validation_collect as mod


def test_collect_node_day_writes_expected_snapshot_files(tmp_path: Path, monkeypatch) -> None:
    invoked_commands: list[str] = []

    def fake_run_json_rpc(node_cfg, command):
        invoked_commands.append(command)
        if command == "revenue-status":
            payload = _valid_payload(command)
            payload["operator_controls"]["values"].update(
                {
                    "paused": False,
                    "daily_budget_sats": 5000,
                    "weekly_budget_sats": 35000,
                    "risk_profile": "conservative",
                }
            )
        elif command == "revenue-dashboard 30":
            payload = _valid_payload(command)
        else:
            payload = _valid_payload(command)
        return mod.CommandResult(
            ok=True,
            stdout_json=payload,
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(mod, "run_json_rpc", fake_run_json_rpc)
    monkeypatch.setattr(
        mod,
        "run_text_command",
        lambda node_cfg, command: mod.common.RunResult([], True, "log", "", 0),
    )

    result = mod.collect_node_day(
        node_name="lnnode",
        node_cfg={
            "t0": "2026-04-23T00:00:00Z",
            "lightning_cli_prefix": "lightning-cli --lightning-dir=/data/lightningd",
            "log_extract_command": "read-only-log-query",
        },
        day_dir=tmp_path,
        run_date="2026-04-23",
    )

    assert result["status"] == "complete"
    assert (tmp_path / "revenue-dashboard-30.json").exists()
    assert (tmp_path / "revenue-report-summary.json").exists()
    assert (tmp_path / "revenue-profitability.json").exists()
    assert (tmp_path / "revenue-budget.json").exists()
    assert (tmp_path / "revenue-econ-reconcile.json").exists()
    assert "-k revenue-budget section=total_cost window_hours=24" in invoked_commands
    assert "revenue-budget" not in invoked_commands
    assert "hive-members" not in invoked_commands
    assert not (tmp_path / "hive-members.json").exists()

    reconcile_command = next(
        command for command in invoked_commands
        if command.startswith("-k revenue-econ-reconcile")
    )
    assert reconcile_command == (
        "-k revenue-econ-reconcile "
        "history_since=1776902400 history_until=1776988800 history_limit=24"
    )
    assert "apply=true" not in reconcile_command
    assert "apply=" not in reconcile_command


def test_collect_node_day_rejects_malformed_run_date_before_any_rpc(tmp_path: Path, monkeypatch) -> None:
    invoked_commands: list[str] = []

    def fake_run_json_rpc(node_cfg, command):
        invoked_commands.append(command)
        return mod.CommandResult(
            ok=True,
            stdout_json={},
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(mod, "run_json_rpc", fake_run_json_rpc)

    try:
        mod.collect_node_day(
            node_name="lnnode",
            node_cfg={
                "t0": "2026-04-23T00:00:00Z",
                "lightning_cli_prefix": "lightning-cli --lightning-dir=/data/lightningd",
            },
            day_dir=tmp_path,
            run_date="2026-04-23T00:00:00Z",
        )
    except ValueError as exc:
        assert "run_date" in str(exc)
    else:
        raise AssertionError("malformed run_date must raise ValueError")

    assert invoked_commands == []


def test_collect_all_nodes_writes_manifest_and_trend_row(tmp_path: Path, monkeypatch) -> None:
    def fake_collect_node_day(node_name, node_cfg, day_dir, run_date):
        (day_dir / "revenue-dashboard-30.json").write_text("{}", encoding="utf-8")
        return {
            "status": "complete",
            "trend_record": {
                "date": run_date,
                "node": node_name,
                "t0": node_cfg["t0"],
                "days_since_t0": 0,
                "gross_revenue_sats_30d": 18843,
                "net_profit_sats_30d": 12850,
                "opex_sats_30d": 5993,
                "forward_count_30d": 466,
                "volume_sats_30d": 68602516,
                "paused": False,
                "daily_budget_sats": 5000,
                "weekly_budget_sats": 35000,
                "risk_profile": "conservative",
            },
        }

    monkeypatch.setattr(mod, "collect_node_day", fake_collect_node_day)

    config = {
        "paths": {"results_root": str(tmp_path)},
        "nodes": {
            "lnnode": {
                "t0": "2026-04-23T00:00:00Z",
                "lightning_cli_prefix": "lightning-cli --lightning-dir=/data/lightningd",
            }
        },
    }

    manifest = mod.collect_all_nodes(config, run_date="2026-04-23")

    manifest_path = tmp_path / "manifests" / "2026-04-23.json"
    trend_path = tmp_path / "trends" / "lnnode.jsonl"

    assert manifest["nodes"]["lnnode"]["status"] == "complete"
    assert manifest_path.exists()
    assert trend_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["nodes"]["lnnode"]["status"] == "complete"
    assert json.loads(trend_path.read_text(encoding="utf-8").splitlines()[0])["node"] == "lnnode"


def _valid_payload(command: str) -> dict:
    if command == "revenue-dashboard 30":
        return {
            "financial_health": {"net_profit_sats": 100},
            "period": {
                "gross_revenue_sats": 120, "opex_sats": 20,
                "forward_count": 3, "volume_sats": 4000,
            },
        }
    if command == "revenue-report summary":
        return {"type": "summary", "policies": {}}
    if command == "revenue-profitability":
        return {"summary": {}, "channels_by_class": {}}
    if command == "revenue-status":
        return {
            "status": "running",
            "operator_controls": {"values": {
                "paused": False, "daily_budget_sats": 5000,
                "weekly_budget_sats": 35000, "risk_profile": "conservative",
                "authority_level": "capital",
            }},
        }
    if command == "revenue-config get":
        return {
            "config": {
                "daily_budget_sats": 5000, "weekly_budget_sats": 35000,
                "authority_level": "capital",
            },
            "version": 1,
        }
    if command == "listforwards":
        return {"forwards": []}
    if command == "listpays":
        return {"pays": []}
    if command == "listpeerchannels":
        return {"channels": []}
    if command.startswith("-k revenue-budget"):
        return {"coverage_status": "complete", "actual_spent_by_category": {}}
    if command.startswith("-k revenue-econ-reconcile"):
        return {"history": {
            "runs": [],
            "summary": {
                "expected_runs": 24, "complete": True, "all_clean": True,
                "fee_intent_complete": True,
            },
        }}
    if command == "feerates perkb":
        return {"perkb": 1000}
    raise AssertionError(f"unexpected command: {command}")


def _collect_with_failure(tmp_path, monkeypatch, failed_command: str, payload=None):
    def fake_run_json_rpc(node_cfg, command):
        if command == failed_command:
            if payload is not None:
                return mod.CommandResult(True, payload, "", 0)
            return mod.CommandResult(False, None, "not available", 1)
        return mod.CommandResult(True, _valid_payload(command), "", 0)

    monkeypatch.setattr(mod, "run_json_rpc", fake_run_json_rpc)
    monkeypatch.setattr(
        mod,
        "run_text_command",
        lambda node_cfg, command: mod.common.RunResult([], True, "log", "", 0),
    )
    return mod.collect_node_day(
        "lnnode",
        {
            "t0": "2026-04-23T00:00:00Z",
            "lightning_cli_prefix": "lightning-cli",
            "log_extract_command": "read-only-log-query",
        },
        tmp_path,
        "2026-04-23",
    )


def test_optional_diagnostic_failure_is_collection_warning(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(tmp_path, monkeypatch, "revenue-report summary")

    assert result["status"] == "collection_warning"
    assert result["errors"]["revenue-report-summary.json"]["role"] == "optional_diagnostic"


def test_dashboard_failure_is_required_economic_incomplete(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(tmp_path, monkeypatch, "revenue-dashboard 30")

    assert result["status"] == "incomplete"
    assert result["errors"]["revenue-dashboard-30.json"]["role"] == "required_for_economic_metrics"
    assert result["trend_record"] is None


def test_transport_success_empty_required_payload_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(tmp_path, monkeypatch, "listpays", payload={})

    assert result["status"] == "incomplete"
    assert result["errors"]["listpays.json"]["role"] == "required_for_economic_metrics"
    assert "missing required keys" in result["errors"]["listpays.json"]["stderr"]


def test_transport_success_wrong_shape_required_payload_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(tmp_path, monkeypatch, "listforwards", payload=[])

    assert result["status"] == "incomplete"
    assert result["errors"]["listforwards.json"]["role"] == "required_for_economic_metrics"
    assert "expected object" in result["errors"]["listforwards.json"]["stderr"]


def test_collect_all_nodes_records_unexpected_collector_failure(tmp_path: Path, monkeypatch) -> None:
    def fail_collect(*args, **kwargs):
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(mod, "collect_node_day", fail_collect)
    config = {
        "paths": {"results_root": str(tmp_path)},
        "nodes": {"lnnode": {"t0": "2026-04-23T00:00:00Z"}},
    }

    manifest = mod.collect_all_nodes(config, run_date="2026-04-23")

    node = manifest["nodes"]["lnnode"]
    assert node["status"] == "collection_failure"
    assert node["errors"]["collector"]["role"] == "required_for_completeness"


def test_transport_success_invalid_required_inner_type_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(tmp_path, monkeypatch, "listpays", payload={"pays": {}})

    assert result["status"] == "incomplete"
    assert "expected list" in result["errors"]["listpays.json"]["stderr"]


def test_invalid_required_mapping_type_is_incomplete_without_crash(tmp_path: Path, monkeypatch) -> None:
    payload = _valid_payload("revenue-config get")
    payload["config"] = []
    result = _collect_with_failure(
        tmp_path, monkeypatch, "revenue-config get", payload=payload
    )

    assert result["status"] == "incomplete"
    assert "config." in result["errors"]["revenue-config.json"]["stderr"]


def test_malformed_required_list_record_is_incomplete_without_crash(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(
        tmp_path, monkeypatch, "listpays", payload={"pays": [0]}
    )

    assert result["status"] == "incomplete"
    assert "pays[0]" in result["errors"]["listpays.json"]["stderr"]


def test_required_list_record_missing_consumed_field_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(
        tmp_path, monkeypatch, "listforwards", payload={"forwards": [{}]}
    )

    assert result["status"] == "incomplete"
    assert "forwards[0].status" in result["errors"]["listforwards.json"]["stderr"]


def test_trend_persistence_failure_becomes_collection_failure_manifest(tmp_path: Path, monkeypatch) -> None:
    def fake_collect_node_day(node_name, node_cfg, day_dir, run_date):
        return {
            "status": "complete",
            "errors": {},
            "trend_record": {"date": run_date, "node": node_name},
        }

    monkeypatch.setattr(mod, "collect_node_day", fake_collect_node_day)
    monkeypatch.setattr(
        mod,
        "append_trend_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    config = {
        "paths": {"results_root": str(tmp_path)},
        "nodes": {"lnnode": {"t0": "2026-04-23T00:00:00Z"}},
    }

    manifest = mod.collect_all_nodes(config, run_date="2026-04-23")

    node = manifest["nodes"]["lnnode"]
    assert node["status"] == "collection_failure"
    assert "trend persistence" in node["errors"]["collector"]["stderr"]
    assert (tmp_path / "manifests" / "2026-04-23.json").exists()


def test_rebalance_pay_missing_fee_evidence_is_incomplete(tmp_path: Path, monkeypatch) -> None:
    result = _collect_with_failure(
        tmp_path,
        monkeypatch,
        "listpays",
        payload={
            "pays": [{
                "label": "rebalance-test", "status": "complete",
                "created_at": 1,
            }]
        },
    )

    assert result["status"] == "incomplete"
    assert "fee evidence" in result["errors"]["listpays.json"]["stderr"]


def test_trend_and_manifest_preserve_versioned_evaluation_identity(tmp_path: Path, monkeypatch) -> None:
    evaluation = {
        "id": "optimization-phase0-measurement-preflight-v1",
        "version": 1,
        "state": "preflight",
        "formal_window_active": False,
        "t0": "2026-08-13T00:00:00Z",
    }

    def fake_run_json_rpc(node_cfg, command):
        return mod.CommandResult(True, _valid_payload(command), "", 0)

    monkeypatch.setattr(mod, "run_json_rpc", fake_run_json_rpc)
    monkeypatch.setattr(
        mod,
        "run_text_command",
        lambda node_cfg, command: mod.common.RunResult([], True, "log", "", 0),
    )
    config = {
        "paths": {"results_root": str(tmp_path)},
        "nodes": {
            "lnnode": {
                "evaluation": evaluation,
                "lightning_cli_prefix": "lightning-cli",
                "log_extract_command": "read-only-log-query",
            }
        },
    }

    manifest = mod.collect_all_nodes(config, run_date="2026-08-13")
    trend = json.loads((tmp_path / "trends" / "lnnode.jsonl").read_text())

    assert manifest["nodes"]["lnnode"]["evaluation"] == evaluation
    assert trend["evaluation_id"] == evaluation["id"]
    assert trend["evaluation_version"] == 1
    assert trend["t0"] == "2026-08-13T00:00:00Z"
