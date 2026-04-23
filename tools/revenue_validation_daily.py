from __future__ import annotations

import argparse
import sys
from datetime import date

from tools import revenue_validation_collect as collect
from tools import revenue_validation_report as report
from tools import revenue_validation_watch as watch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily revenue validation pipeline.")
    parser.add_argument(
        "--config",
        default="config/revenue_validation.yaml",
        help="Path to revenue validation config.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Run date in YYYY-MM-DD format. Defaults to local today.",
    )
    return parser.parse_args(argv)


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
