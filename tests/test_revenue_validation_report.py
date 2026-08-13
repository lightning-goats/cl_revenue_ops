import json
from pathlib import Path

from tools import revenue_validation_report as mod


def test_generates_t14_report_when_checkpoint_reached(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    reports_root = tmp_path / "docs" / "reports"
    trends_dir = results_root / "trends"
    watch_dir = results_root / "watch"
    day_dir = results_root / "2026-05-07" / "lnnode"

    trends_dir.mkdir(parents=True)
    watch_dir.mkdir(parents=True)
    day_dir.mkdir(parents=True)
    reports_root.mkdir(parents=True)

    (trends_dir / "lnnode.jsonl").write_text(
        json.dumps(
            {
                "date": "2026-05-07",
                "node": "lnnode",
                "t0": "2026-04-23T00:00:00Z",
                "days_since_t0": 14,
                "gross_revenue_sats_30d": 18843,
                "net_profit_sats_30d": 12850,
                "opex_sats_30d": 5993,
                "forward_count_30d": 466,
                "volume_sats_30d": 68602516,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (watch_dir / "2026-05-07.json").write_text(
        json.dumps(
            {
                "date": "2026-05-07",
                "status": "green",
                "nodes": {
                    "lnnode": {
                        "highest_severity": "green",
                        "findings": [
                            {
                                "rule": "rebalance_success_rate",
                                "severity": "green",
                                "attempt_count": 3,
                                "success_count": 3,
                                "success_rate_pct": 100.0,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (day_dir / "rollback-watch.log").write_text(
        "\n".join(
            [
                "INITIAL_FEE Hive member: zero-fee fleet policy",
                "FEE: channel=1x1x1 blend:0.30",
                "competition_aware preserve 1x1x1",
            ]
        ),
        encoding="utf-8",
    )
    (day_dir / "listpeerchannels.json").write_text(
        json.dumps({"channels": [{"peer_id": "02hive", "fee_proportional_millionths": 0}]}),
        encoding="utf-8",
    )
    (day_dir / "hive-members.json").write_text(
        json.dumps({"members": [{"peer_id": "02hive"}]}),
        encoding="utf-8",
    )
    (day_dir / "listforwards.json").write_text(json.dumps({"forwards": []}), encoding="utf-8")
    (day_dir / "listpays.json").write_text(json.dumps({"pays": []}), encoding="utf-8")

    config = {
        "paths": {
            "results_root": str(results_root),
            "reports_root": str(reports_root),
        },
        "nodes": {
            "lnnode": {
                "t0": "2026-04-23T00:00:00Z",
            }
        },
    }

    generated = mod.generate_checkpoint_reports(config, run_date="2026-05-07")
    report_path = reports_root / "2026-05-07-production-t14-findings.md"

    assert report_path in generated
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Deploy timestamp confirmation" in report_text
    assert "lnnode" in report_text


def test_generates_t28_report_when_checkpoint_reached(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    reports_root = tmp_path / "docs" / "reports"
    trends_dir = results_root / "trends"
    watch_dir = results_root / "watch"
    day_dir = results_root / "2026-05-21" / "lnnode"

    trends_dir.mkdir(parents=True)
    watch_dir.mkdir(parents=True)
    day_dir.mkdir(parents=True)
    reports_root.mkdir(parents=True)

    (trends_dir / "lnnode.jsonl").write_text(
        json.dumps(
            {
                "date": "2026-05-21",
                "node": "lnnode",
                "t0": "2026-04-23T00:00:00Z",
                "days_since_t0": 28,
                "gross_revenue_sats_30d": 18843,
                "net_profit_sats_30d": 12850,
                "opex_sats_30d": 5993,
                "forward_count_30d": 466,
                "volume_sats_30d": 68602516,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (watch_dir / "2026-05-21.json").write_text(
        json.dumps(
            {
                "date": "2026-05-21",
                "status": "green",
                "nodes": {
                    "lnnode": {
                        "highest_severity": "green",
                        "findings": [
                            {
                                "rule": "revenue_drop",
                                "severity": "green",
                                "pre_14d_fee_sats": 1000,
                                "post_window_fee_sats": 1200,
                                "post_window_days": 14,
                                "pre_avg_sats_per_day": 71.4,
                                "post_avg_sats_per_day": 85.7,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (day_dir / "rollback-watch.log").write_text(
        "\n".join(
            [
                "INITIAL_FEE Hive member: zero-fee fleet policy",
                "competition_aware preserve 1x1x1",
                "estimated_closure_cost=1200",
            ]
        ),
        encoding="utf-8",
    )
    (day_dir / "listpeerchannels.json").write_text(
        json.dumps({"channels": [{"peer_id": "02hive", "fee_proportional_millionths": 0}]}),
        encoding="utf-8",
    )
    (day_dir / "hive-members.json").write_text(
        json.dumps({"members": [{"peer_id": "02hive"}]}),
        encoding="utf-8",
    )
    (day_dir / "listforwards.json").write_text(json.dumps({"forwards": []}), encoding="utf-8")
    (day_dir / "listpays.json").write_text(json.dumps({"pays": []}), encoding="utf-8")

    config = {
        "paths": {
            "results_root": str(results_root),
            "reports_root": str(reports_root),
        },
        "nodes": {
            "lnnode": {
                "t0": "2026-04-23T00:00:00Z",
            }
        },
    }

    generated = mod.generate_checkpoint_reports(config, run_date="2026-05-21")
    report_path = reports_root / "2026-05-21-production-t28-findings.md"

    assert report_path in generated
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Executive summary" in report_text
    assert "Per-PR hypothesis status" in report_text


def test_t28_missing_profit_is_investigate_not_ship() -> None:
    decision = mod._t28_decision(
        {"lnnode": {"net_profit_sats_30d": None}},
        {"lnnode": []},
    )

    assert decision == "investigate"


def test_latest_trend_rows_do_not_reinterpret_legacy_identity(tmp_path: Path) -> None:
    trends = tmp_path / "trends"
    trends.mkdir()
    (trends / "lnnode.jsonl").write_text(
        "\n".join([
            json.dumps({
                "date": "2026-08-14", "node": "lnnode",
                "t0": "2026-04-23T16:31:01Z", "days_since_t0": 111,
                "net_profit_sats_30d": 999,
            }),
            json.dumps({
                "date": "2026-08-13", "node": "lnnode",
                "evaluation_id": "optimization-phase0-measurement-preflight-v1",
                "evaluation_version": 1,
                "t0": "2026-08-13T00:00:00Z", "days_since_t0": 0,
                "net_profit_sats_30d": 100,
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    nodes = {
        "lnnode": {
            "evaluation": {
                "id": "optimization-phase0-measurement-preflight-v1",
                "version": 1,
                "state": "preflight",
                "formal_window_active": False,
                "t0": "2026-08-13T00:00:00Z",
            }
        }
    }

    rows = mod._latest_trend_rows(tmp_path, nodes, "2026-08-14")

    assert rows["lnnode"]["net_profit_sats_30d"] == 100
    assert rows["lnnode"]["days_since_t0"] == 0


def test_explicit_evaluation_gets_identity_scoped_report_name() -> None:
    config = {
        "nodes": {"lnnode": {"evaluation": {
            "id": "optimization-phase0-measurement-preflight-v1",
            "version": 1,
            "state": "preflight",
            "formal_window_active": False,
            "t0": "2026-08-13T00:00:00Z",
        }}}
    }

    assert mod._evaluation_report_slug(config) == (
        "optimization-phase0-measurement-preflight-v1-v1"
    )


def test_watch_history_rejects_mismatched_explicit_identity(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "2026-08-13.json").write_text(
        json.dumps({
            "date": "2026-08-13",
            "nodes": {"lnnode": {
                "evaluation": {
                    "id": "old-evaluation", "version": 1,
                    "state": "closed", "formal_window_active": False,
                    "t0": "2026-07-13T00:00:00Z",
                },
                "highest_severity": "red",
                "findings": [{"rule": "old_evaluation_red", "severity": "red"}],
            }},
        }) + "\n",
        encoding="utf-8",
    )
    expected = {
        "id": "new-evaluation", "version": 1,
        "state": "active", "formal_window_active": True,
        "t0": "2026-08-13T00:00:00Z",
    }

    history = mod._watch_history(
        tmp_path, "lnnode", mod.date(2026, 8, 13), mod.date(2026, 8, 13),
        expected,
    )

    assert history[0]["highest_severity"] == "red"
    assert history[0]["findings"][0]["rule"] == "evaluation_identity"
    assert "old_evaluation_red" not in json.dumps(history)


def test_watch_history_fails_closed_when_explicit_identity_is_missing(tmp_path: Path) -> None:
    expected = {
        "id": "new-evaluation", "version": 1,
        "state": "active", "formal_window_active": True,
        "t0": "2026-08-13T00:00:00Z",
    }

    history = mod._watch_history(
        tmp_path, "lnnode", mod.date(2026, 8, 13), mod.date(2026, 8, 13),
        expected,
    )

    assert history[0]["highest_severity"] == "red"
    assert history[0]["findings"][0]["rule"] == "evaluation_identity"


def test_preflight_identity_cannot_generate_formal_checkpoint_report(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    reports_root = tmp_path / "reports"
    trends = results_root / "trends"
    trends.mkdir(parents=True)
    evaluation = {
        "id": "optimization-phase0-measurement-preflight-v1",
        "version": 1,
        "state": "preflight",
        "formal_window_active": False,
        "t0": "2026-08-13T00:00:00Z",
    }
    (trends / "lnnode.jsonl").write_text(
        json.dumps({
            "date": "2026-09-10", "node": "lnnode",
            "evaluation_id": evaluation["id"], "evaluation_version": 1,
            "t0": evaluation["t0"], "days_since_t0": 28,
            "net_profit_sats_30d": 100,
        }) + "\n",
        encoding="utf-8",
    )
    config = {
        "paths": {
            "results_root": str(results_root),
            "reports_root": str(reports_root),
        },
        "nodes": {"lnnode": {"evaluation": evaluation}},
    }

    generated = mod.generate_checkpoint_reports(config, "2026-09-10")

    assert generated == []
    assert list(reports_root.glob("*.md")) == []
