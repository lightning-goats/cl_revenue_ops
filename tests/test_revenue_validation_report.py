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
                "INITIAL_FEE Hive member: 1-PPM fleet policy",
                "FEE: channel=1x1x1 blend:0.30",
                "competition_aware preserve 1x1x1",
            ]
        ),
        encoding="utf-8",
    )
    (day_dir / "listpeerchannels.json").write_text(
        json.dumps({"channels": [{"peer_id": "02hive", "fee_proportional_millionths": 1}]}),
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
                "INITIAL_FEE Hive member: 1-PPM fleet policy",
                "competition_aware preserve 1x1x1",
                "estimated_closure_cost=1200",
            ]
        ),
        encoding="utf-8",
    )
    (day_dir / "listpeerchannels.json").write_text(
        json.dumps({"channels": [{"peer_id": "02hive", "fee_proportional_millionths": 1}]}),
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
