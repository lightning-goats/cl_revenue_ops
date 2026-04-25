import json
import importlib.util
import sys
from pathlib import Path


def load_analyzer():
    path = Path(__file__).resolve().parents[1] / "tools" / "analyze_fee_tournament.py"
    spec = importlib.util.spec_from_file_location("analyze_fee_tournament", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_record(route="revenue", revenue_forwards=4, competitor_forwards=0, fee_ppm=93):
    return {
        "started": 123,
        "amount_sat": 20_000,
        "payments_succeeded": revenue_forwards + competitor_forwards,
        "payments_failed": [],
        "before": {
            "route": {"route": route, "total_fees_msat": "1601"},
            "revenue_fee_debug": {
                "channels": [
                    {
                        "channel_id": "277x1x0",
                        "last_broadcast_fee_ppm": fee_ppm,
                    }
                ]
            },
        },
        "after": {
            "route": {"route": route, "total_fees_msat": "1601"},
            "revenue_forwards_since_start": revenue_forwards,
            "competitor_forwards_since_start": competitor_forwards,
            "revenue_fee_debug": {
                "channels": [
                    {
                        "channel_id": "277x1x0",
                        "last_broadcast_fee_ppm": fee_ppm,
                    }
                ]
            },
        },
    }


def test_extract_metrics_detects_mission_control_masking():
    analyzer = load_analyzer()
    metric = analyzer.extract_metrics(
        Path("burst.json"),
        sample_record(route="competitor", revenue_forwards=4, competitor_forwards=0),
        "277x1x0",
    )

    assert metric.before_route == "competitor"
    assert metric.revenue_forwards == 4
    assert metric.competitor_forwards == 0
    assert metric.mission_control_masked is True
    assert metric.quote_forward_diverged is True


def test_extract_metrics_detects_non_mission_control_quote_divergence():
    analyzer = load_analyzer()
    metric = analyzer.extract_metrics(
        Path("burst.json"),
        sample_record(route="revenue", revenue_forwards=0, competitor_forwards=3),
        "277x1x0",
    )

    assert metric.before_route == "revenue"
    assert metric.revenue_forwards == 0
    assert metric.competitor_forwards == 3
    assert metric.mission_control_masked is False
    assert metric.quote_forward_diverged is True


def test_summary_ranks_candidates_and_recommends_boundary_controls():
    analyzer = load_analyzer()
    metrics = [
        analyzer.extract_metrics(
            Path("revenue.json"),
            sample_record(route="revenue", revenue_forwards=6, competitor_forwards=0, fee_ppm=79),
            "277x1x0",
        ),
        analyzer.extract_metrics(
            Path("competitor.json"),
            sample_record(route="competitor", revenue_forwards=0, competitor_forwards=4, fee_ppm=93),
            "277x1x0",
        ),
    ]

    summary = analyzer.summarize(metrics, boundary_ppm=80)

    assert summary["totals"]["revenue_forwards"] == 6
    assert summary["totals"]["competitor_forwards"] == 4
    assert summary["totals"]["payment_success_rate"] == 1.0
    assert summary["totals"]["forward_attribution_rate"] == 1.0
    assert summary["totals"]["revenue_share"] == 0.6
    assert summary["totals"]["observed_revenue_per_success_sat"] == 0.948
    assert summary["segments"][0]["payments_succeeded"] == 10
    assert summary["segments"][0]["observed_revenue_per_success_sat"] == 0.948
    assert summary["ranked_candidate_settings"]
    assert any("market-clearing signal" in rec for rec in summary["recommendations"])


def test_load_records_ignores_generated_summary_files(tmp_path):
    analyzer = load_analyzer()
    phase_path = tmp_path / "phase_real.json"
    summary_path = tmp_path / "analysis.json"
    phase_record = sample_record(route="revenue", revenue_forwards=1)
    phase_path.write_text(json.dumps(phase_record), encoding="utf-8")
    summary_path.write_text(json.dumps({"totals": {"payments_succeeded": 1}, "bursts": []}), encoding="utf-8")

    records = analyzer.load_records([summary_path, phase_path])

    assert records == [(phase_path, phase_record)]
