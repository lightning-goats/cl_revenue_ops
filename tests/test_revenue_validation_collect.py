import json
from pathlib import Path

from tools import revenue_validation_collect as mod


def test_collect_node_day_writes_expected_snapshot_files(tmp_path: Path, monkeypatch) -> None:
    invoked_commands: list[str] = []

    def fake_run_json_rpc(node_cfg, command):
        invoked_commands.append(command)
        if command == "revenue-status":
            payload = {
                "operator_controls": {
                    "values": {
                        "paused": False,
                        "daily_budget_sats": 5000,
                        "weekly_budget_sats": 35000,
                        "risk_profile": "conservative",
                    }
                }
            }
        elif command == "revenue-dashboard 30":
            payload = {"financial_health": {}, "period": {}}
        else:
            payload = {"command": command}
        return mod.CommandResult(
            ok=True,
            stdout_json=payload,
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(mod, "run_json_rpc", fake_run_json_rpc)

    result = mod.collect_node_day(
        node_name="lnnode",
        node_cfg={
            "t0": "2026-04-23T00:00:00Z",
            "lightning_cli_prefix": "lightning-cli --lightning-dir=/data/lightningd",
        },
        day_dir=tmp_path,
        run_date="2026-04-23",
    )

    assert result["status"] == "ok"
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
            "status": "ok",
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

    assert manifest["nodes"]["lnnode"]["status"] == "ok"
    assert manifest_path.exists()
    assert trend_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["nodes"]["lnnode"]["status"] == "ok"
    assert json.loads(trend_path.read_text(encoding="utf-8").splitlines()[0])["node"] == "lnnode"
