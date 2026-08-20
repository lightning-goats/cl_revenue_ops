from datetime import datetime, timezone
import json

from tools import revenue_validation_preflight as mod


def _clean_run(slot: int) -> dict:
    return {
        "reconciliation_id": f"reconcile-hour-{slot}",
        "slot_started_at": slot,
        "started_at": slot,
        "completed_at": slot + 1,
        "result": "clean",
        "unexplained_divergence_count": 0,
        "ledger_projection_status": "aligned",
        "fee_intent_completeness": "ok",
        "error": None,
    }


def test_reconciliation_command_is_bounded_and_read_only() -> None:
    command = mod.reconciliation_command(
        since=1786989600,
        until=1787248800,
        hours=72,
    )

    assert command == (
        "-k revenue-econ-reconcile apply=false "
        "history_since=1786989600 history_until=1787248800 history_limit=72"
    )


def test_trailing_clean_hours_stops_at_first_missing_slot() -> None:
    until = 10 * 3600
    payload = {
        "history": {
            "runs": [
                _clean_run(7 * 3600),
                _clean_run(8 * 3600),
                _clean_run(9 * 3600),
            ]
        }
    }

    assert mod.trailing_clean_hours(payload, until=until, hours=5) == 3


def test_archive_window_requires_three_complete_closed_utc_days() -> None:
    until = 1787184000
    coverage = []
    for day in range(until - 3 * 86400, until, 86400):
        coverage.append({
            "date_utc": day,
            "created_sync_complete": True,
            "updated_sync_complete": True,
            "aggregate_complete": True,
            "reconciliation_status": "complete",
            "reasons": [],
        })
    payload = {
        "history_since": until - 3 * 86400,
        "history_until": until,
        "coverage": coverage,
        "complete": True,
        "truncated": False,
    }

    assert mod.archive_window_complete(payload, until_day=until, days=3) is True
    coverage[1]["reasons"] = ["coverage_mismatch"]
    assert mod.archive_window_complete(payload, until_day=until, days=3) is False


def test_monitor_uses_only_diagnostic_rpc_commands(monkeypatch) -> None:
    commands = []
    until_hour = 1787248800
    until_day = 1787184000
    runs = [
        _clean_run(slot)
        for slot in range(until_hour - 72 * 3600, until_hour, 3600)
    ]
    coverage = [
        {
            "date_utc": day,
            "created_sync_complete": True,
            "updated_sync_complete": True,
            "aggregate_complete": True,
            "reconciliation_status": "complete",
            "reasons": [],
        }
        for day in range(until_day - 3 * 86400, until_day, 86400)
    ]

    def fake_rpc(_node_cfg, command):
        commands.append(command)
        if "revenue-econ-reconcile" in command:
            return mod.collect.CommandResult(
                ok=True,
                stdout_json={"history": {"runs": runs}},
                stderr="",
                returncode=0,
            )
        return mod.collect.CommandResult(
            ok=True,
            stdout_json={
                "history_since": until_day - 3 * 86400,
                "history_until": until_day,
                "coverage": coverage,
                "complete": True,
                "truncated": False,
            },
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(mod.collect, "run_json_rpc", fake_rpc)
    observed = datetime(2026, 8, 20, 18, 10, tzinfo=timezone.utc)

    result = mod.monitor_node("lnnode", {}, observed_at=observed, hours=72)

    assert result["status"] == "ready"
    assert result["consecutive_clean_hours"] == 72
    assert commands == [
        mod.reconciliation_command(
            since=until_hour - 72 * 3600,
            until=until_hour,
            hours=72,
        ),
        mod.forward_history_command(
            since=until_day - 3 * 86400,
            until=until_day,
        ),
    ]
    assert all("revenue-fee-cycle" not in command for command in commands)
    assert all("revenue-rebalance" not in command for command in commands)


def test_append_observation_preserves_jsonl_history(tmp_path) -> None:
    path = tmp_path / "preflight" / "lnnode.jsonl"

    mod.append_observation(path, {"status": "pending", "sequence": 1})
    mod.append_observation(path, {"status": "ready", "sequence": 2})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"sequence": 1, "status": "pending"},
        {"sequence": 2, "status": "ready"},
    ]


def test_monitor_rejects_malformed_success_payloads(monkeypatch) -> None:
    monkeypatch.setattr(
        mod.collect,
        "run_json_rpc",
        lambda *_args, **_kwargs: mod.collect.CommandResult(
            ok=True,
            stdout_json={},
            stderr="",
            returncode=0,
        ),
    )

    result = mod.monitor_node(
        "lnnode",
        {},
        observed_at=datetime(2026, 8, 20, 18, 10, tzinfo=timezone.utc),
        hours=72,
    )

    assert result["status"] == "error"
    assert "malformed reconciliation evidence" in result["errors"]
    assert "malformed forward archive evidence" in result["errors"]


def test_monitor_does_not_count_slots_before_frozen_boundary(monkeypatch) -> None:
    commands = []
    boundary = 1787248800
    observed = datetime(2026, 8, 20, 19, 10, tzinfo=timezone.utc)

    def fake_rpc(_node_cfg, command):
        commands.append(command)
        if "revenue-econ-reconcile" in command:
            return mod.collect.CommandResult(
                ok=True,
                stdout_json={"history": {"runs": [_clean_run(boundary)]}},
                stderr="",
                returncode=0,
            )
        return mod.collect.CommandResult(
            ok=True,
            stdout_json={
                "history_since": 1786924800,
                "history_until": 1787184000,
                "coverage": [
                    {
                        "date_utc": day,
                        "created_sync_complete": True,
                        "updated_sync_complete": True,
                        "aggregate_complete": True,
                        "reconciliation_status": "complete",
                        "reasons": [],
                    }
                    for day in range(1786924800, 1787184000, 86400)
                ],
                "complete": True,
                "truncated": False,
            },
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(mod.collect, "run_json_rpc", fake_rpc)

    result = mod.monitor_node(
        "lnnode",
        {},
        observed_at=observed,
        hours=72,
        not_before=boundary,
    )

    assert result["consecutive_clean_hours"] == 1
    assert result["reconciliation_since"] == boundary
    assert result["gate_not_before"] == boundary
    assert f"history_since={boundary}" in commands[0]


def test_monitor_waits_without_querying_empty_reconciliation_window(monkeypatch) -> None:
    commands = []
    boundary = 1787248800

    def fake_rpc(_node_cfg, command):
        commands.append(command)
        return mod.collect.CommandResult(
            ok=True,
            stdout_json={
                "history_since": 1786924800,
                "history_until": 1787184000,
                "coverage": [
                    {
                        "date_utc": day,
                        "created_sync_complete": True,
                        "updated_sync_complete": True,
                        "aggregate_complete": True,
                        "reconciliation_status": "complete",
                        "reasons": [],
                    }
                    for day in range(1786924800, 1787184000, 86400)
                ],
                "complete": True,
                "truncated": False,
            },
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(mod.collect, "run_json_rpc", fake_rpc)

    result = mod.monitor_node(
        "lnnode",
        {},
        observed_at=datetime(2026, 8, 20, 17, 10, tzinfo=timezone.utc),
        hours=72,
        not_before=boundary,
    )

    assert result["status"] == "pending"
    assert result["consecutive_clean_hours"] == 0
    assert result["reconciliation_since"] == 1787245200
    assert result["reconciliation_until"] == 1787245200
    assert commands == [
        mod.forward_history_command(
            since=1786924800,
            until=1787184000,
        )
    ]
