from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone

if __package__ in {None, ""}:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import revenue_validation_collect as collect
from tools import revenue_validation_report as report
from tools import revenue_validation_watch as watch


def closed_utc_day(now: datetime | None = None) -> str:
    observed = datetime.now(timezone.utc) if now is None else now
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return (observed.astimezone(timezone.utc).date() - timedelta(days=1)).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily revenue validation pipeline.")
    parser.add_argument(
        "--config",
        default="config/revenue_validation.yaml",
        help="Path to revenue validation config.",
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--date",
        help="Run date in YYYY-MM-DD format. Defaults to local today.",
    )
    date_group.add_argument(
        "--closed-utc-day",
        action="store_true",
        help="Collect the previous fully closed UTC calendar day.",
    )
    args = parser.parse_args(argv)
    if args.closed_utc_day:
        args.date = closed_utc_day()
    elif args.date is None:
        args.date = date.today().isoformat()
    return args


def run_collect(config_path: str, run_date: str) -> int:
    return collect.main(["--config", config_path, "--date", run_date])


def run_watch(config_path: str, run_date: str) -> int:
    return watch.main(["--config", config_path, "--date", run_date])


def run_report(config_path: str, run_date: str) -> int:
    return report.main(["--config", config_path, "--date", run_date])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    codes = [
        run_collect(args.config, args.date),
        run_watch(args.config, args.date),
        run_report(args.config, args.date),
    ]
    return 1 if any(code != 0 for code in codes) else 0


if __name__ == "__main__":
    sys.exit(main())
