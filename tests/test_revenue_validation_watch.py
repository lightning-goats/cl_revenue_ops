import json
from datetime import datetime
from pathlib import Path

from tools import revenue_validation_watch as mod


def _unix(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())


def test_detects_red_flag_when_non_hive_channel_hits_zero_ppm() -> None:
    peerchannels = {
        "channels": [
            {
                "peer_id": "02peer",
                "short_channel_id": "1x1x1",
                "fee_proportional_millionths": 0,
            }
        ]
    }
    hive_members = {"members": []}

    result = mod.check_zero_ppm_non_hive(peerchannels, hive_members)

    assert result["severity"] == "red"
    assert result["count"] == 1


def test_detects_yellow_flag_for_traceback_burst() -> None:
    lines = ["Traceback: boom"] * 11

    result = mod.check_traceback_volume(lines)

    assert result["severity"] == "yellow"
    assert result["count"] == 11


def test_plugin_restart_count_ignores_non_lifecycle_plugin_lines() -> None:
    lines = [
        "2026-04-23T16:31:01.914Z INFO    plugin-cl-revenue-ops.py: Running scheduled fee adjustment...",
        "2026-04-23T16:31:02.914Z INFO    plugin-cl-revenue-ops.py: Profitability analysis complete",
        "2026-04-23T16:31:03.914Z INFO    plugin-cl-revenue-ops.py: Initializing cl-revenue-ops plugin...",
        "2026-04-23T16:31:04.914Z INFO    plugin-cl-revenue-ops.py: Initializing cl-revenue-ops plugin...",
    ]

    result = mod.check_plugin_restart_count(lines, limit=3)

    assert result["severity"] == "green"
    assert result["count"] == 2


def test_detects_red_flag_for_rebalance_success_rate_drop() -> None:
    pays = {
        "pays": [
            {"label": "rebalance-a", "status": "failed", "created_at": _unix("2026-05-01T00:00:00Z")},
            {"label": "rebalance-b", "status": "failed", "created_at": _unix("2026-05-01T01:00:00Z")},
            {"label": "rebalance-c", "status": "complete", "created_at": _unix("2026-05-01T02:00:00Z")},
        ]
    }

    result = mod.check_rebalance_success_rate(
        pays,
        run_date="2026-05-02",
        floor_pct=50,
    )

    assert result["severity"] == "red"
    assert result["attempt_count"] == 3
    assert result["success_count"] == 1


def test_revenue_drop_stays_green_until_post_window_is_complete() -> None:
    forwards = {
        "forwards": [
            {
                "status": "settled",
                "received_time": _unix("2026-04-20T12:00:00Z"),
                "fee_msat": "21000msat",
            }
        ]
    }

    result = mod.check_revenue_drop(
        forwards,
        t0="2026-04-23T16:31:01Z",
        run_date="2026-04-23",
        drop_pct=25,
    )

    assert result["severity"] == "green"
    assert result["window_complete"] is False


def test_evaluate_all_nodes_suppresses_restart_alarm_on_t0_day(tmp_path: Path) -> None:
    results_root = tmp_path
    day_dir = results_root / "2026-04-23" / "lnnode"
    day_dir.mkdir(parents=True)

    (results_root / "manifests").mkdir(parents=True)
    (results_root / "manifests" / "2026-04-23.json").write_text(
        json.dumps({"date": "2026-04-23", "nodes": {"lnnode": {"status": "ok", "errors": {}}}}),
        encoding="utf-8",
    )

    (day_dir / "listpeerchannels.json").write_text(json.dumps({"channels": []}), encoding="utf-8")
    (day_dir / "hive-members.json").write_text(json.dumps({"members": []}), encoding="utf-8")
    (day_dir / "listpays.json").write_text(json.dumps({"pays": []}), encoding="utf-8")
    (day_dir / "listforwards.json").write_text(json.dumps({"forwards": []}), encoding="utf-8")
    (day_dir / "revenue-config.json").write_text(json.dumps({"config": {"max_fee_ppm": 5000}}), encoding="utf-8")
    (day_dir / "rollback-watch.log").write_text(
        "\n".join(
            [
                "2026-04-23T16:31:01.914Z INFO    plugin-cl-revenue-ops.py: Initializing cl-revenue-ops plugin...",
                "2026-04-23T16:40:58.793Z INFO    plugin-cl-revenue-ops.py: Initializing cl-revenue-ops plugin...",
                "2026-04-23T17:06:53.525Z INFO    plugin-cl-revenue-ops.py: Initializing cl-revenue-ops plugin...",
                "2026-04-23T17:22:09.008Z INFO    plugin-cl-revenue-ops.py: Initializing cl-revenue-ops plugin...",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "paths": {"results_root": str(results_root)},
        "thresholds": {
            "rollback": {
                "plugin_restart_limit_24h": 3,
                "revenue_drop_pct": 25,
                "rebalance_success_floor_pct": 50,
            }
        },
        "nodes": {
            "lnnode": {
                "t0": "2026-04-23T16:31:01Z",
            }
        },
    }

    result = mod.evaluate_all_nodes(config, run_date="2026-04-23")

    restart_finding = next(
        finding
        for finding in result["nodes"]["lnnode"]["findings"]
        if finding["rule"] == "plugin_restart_count"
    )
    assert restart_finding["severity"] == "green"


def test_evaluate_all_nodes_writes_watch_file_and_marks_red(tmp_path: Path) -> None:
    results_root = tmp_path
    day_dir = results_root / "2026-04-23" / "lnnode"
    day_dir.mkdir(parents=True)

    (results_root / "manifests").mkdir(parents=True)
    (results_root / "manifests" / "2026-04-23.json").write_text(
        json.dumps({"date": "2026-04-23", "nodes": {"lnnode": {"status": "ok", "errors": {}}}}),
        encoding="utf-8",
    )

    (day_dir / "listpeerchannels.json").write_text(
        json.dumps(
            {
                "channels": [
                    {
                        "peer_id": "02peer",
                        "short_channel_id": "1x1x1",
                        "fee_proportional_millionths": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (day_dir / "hive-members.json").write_text(json.dumps({"members": []}), encoding="utf-8")
    (day_dir / "listpays.json").write_text(json.dumps({"pays": []}), encoding="utf-8")
    (day_dir / "listforwards.json").write_text(json.dumps({"forwards": []}), encoding="utf-8")
    (day_dir / "revenue-config.json").write_text(json.dumps({"config": {"max_fee_ppm": 5000}}), encoding="utf-8")
    (day_dir / "rollback-watch.log").write_text("", encoding="utf-8")

    config = {
        "paths": {"results_root": str(results_root)},
        "thresholds": {
            "rollback": {
                "plugin_restart_limit_24h": 3,
                "revenue_drop_pct": 25,
                "rebalance_success_floor_pct": 50,
            }
        },
        "nodes": {
            "lnnode": {
                "t0": "2026-04-23T00:00:00Z",
            }
        },
    }

    result = mod.evaluate_all_nodes(config, run_date="2026-04-23")

    assert result["status"] == "red"
    assert result["nodes"]["lnnode"]["highest_severity"] == "red"
    assert (results_root / "watch" / "2026-04-23.json").exists()
