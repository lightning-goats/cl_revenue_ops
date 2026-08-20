from datetime import datetime, timezone

import pytest

from tools import revenue_validation_daily as mod


def test_daily_pipeline_runs_collect_watch_and_report_in_order(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(mod, "run_collect", lambda *a, **k: calls.append("collect") or 0)
    monkeypatch.setattr(mod, "run_watch", lambda *a, **k: calls.append("watch") or 0)
    monkeypatch.setattr(mod, "run_report", lambda *a, **k: calls.append("report") or 0)

    code = mod.main(["--config", "config/revenue_validation.yaml", "--date", "2026-04-23"])

    assert code == 0
    assert calls == ["collect", "watch", "report"]


def test_daily_pipeline_runs_all_steps_even_when_collect_fails(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(mod, "run_collect", lambda *a, **k: calls.append("collect") or 1)
    monkeypatch.setattr(mod, "run_watch", lambda *a, **k: calls.append("watch") or 0)
    monkeypatch.setattr(mod, "run_report", lambda *a, **k: calls.append("report") or 0)

    code = mod.main(["--config", "config/revenue_validation.yaml", "--date", "2026-04-23"])

    assert code == 1
    assert calls == ["collect", "watch", "report"]


def test_closed_utc_day_selects_previous_completed_day() -> None:
    now = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)

    assert mod.closed_utc_day(now) == "2026-08-19"


def test_closed_utc_day_requires_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        mod.closed_utc_day(datetime(2026, 8, 20, 12, 0))


def test_explicit_date_and_closed_utc_day_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        mod.parse_args([
            "--date", "2026-08-19", "--closed-utc-day",
        ])
