from pathlib import Path


SYSTEMD_DIR = Path("tools/systemd")


def test_daily_service_collects_previous_closed_utc_day() -> None:
    service = (SYSTEMD_DIR / "revenue-validation-daily.service").read_text()

    assert "--closed-utc-day" in service


def test_preflight_timer_is_hourly_and_observational() -> None:
    timer = (SYSTEMD_DIR / "revenue-validation-preflight.timer").read_text()
    service = (SYSTEMD_DIR / "revenue-validation-preflight.service").read_text()

    assert "OnCalendar=*-*-* *:10:00" in timer
    assert "revenue_validation_preflight.py" in service
    assert "--not-before 2026-08-20T18:00:00Z" in service
    assert "revenue-fee-cycle" not in service
    assert "revenue-rebalance" not in service
