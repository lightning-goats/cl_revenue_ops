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
