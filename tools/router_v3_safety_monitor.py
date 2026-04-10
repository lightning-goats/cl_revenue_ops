#!/usr/bin/env python3
"""A/B safety monitor for the v3 rebalance router rollout.

Reads recent audit log lines, computes success-rate metrics grouped by
router version (via the `router=v2|v3` field introduced in Phase 1),
and if v3 regresses against a v2 baseline by more than a configurable
threshold, flips `rebalance-router` back to `v2` via `lightning-cli
setconfig` and exits non-zero.

This is a Phase 3 rollout safety tool. In Phase 1 it ships as a skeleton
that operators can adapt once an A/B baseline exists. Threshold and
time windows will be calibrated against real production data during the
phase-3 default-flip evaluation.

Usage:
    python3 tools/router_v3_safety_monitor.py --log /data/lightningd/cln.log \\
        --threshold 0.10 [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Tuple


REBAL_PICK_RE = re.compile(r"REBAL_PICK .* router=(v2|v3)")
REBAL_SKIP_RE = re.compile(r"REBAL_SKIP .* router=(v2|v3)")


def parse_log(path: str) -> Dict[str, Dict[str, int]]:
    """Return {router_version: {picks: int, skips: int}} from a CLN log file."""
    counters: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"picks": 0, "skips": 0}
    )
    if not os.path.exists(path):
        return counters
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = REBAL_PICK_RE.search(line)
            if m:
                counters[m.group(1)]["picks"] += 1
                continue
            m = REBAL_SKIP_RE.search(line)
            if m:
                counters[m.group(1)]["skips"] += 1
    return counters


def should_rollback(
    counters: Dict[str, Dict[str, int]], threshold: float
) -> Tuple[bool, str]:
    """Decide if v3 should be rolled back based on the success-rate comparison."""
    v2 = counters.get("v2", {"picks": 0, "skips": 0})
    v3 = counters.get("v3", {"picks": 0, "skips": 0})
    v2_total = v2["picks"] + v2["skips"]
    v3_total = v3["picks"] + v3["skips"]
    if v2_total == 0 or v3_total == 0:
        return False, "insufficient data for comparison"
    v2_success_rate = v2["picks"] / v2_total
    v3_success_rate = v3["picks"] / v3_total
    if v3_success_rate < v2_success_rate - threshold:
        return True, (
            f"v3 success rate {v3_success_rate:.2%} < v2 baseline "
            f"{v2_success_rate:.2%} - threshold {threshold:.2%}"
        )
    return False, (
        f"v3 {v3_success_rate:.2%} vs v2 {v2_success_rate:.2%} (ok)"
    )


def rollback_to_v2(dry_run: bool) -> None:
    cmd = [
        "lightning-cli",
        "-k",
        "setconfig",
        "config=revenue-ops-rebalance-router",
        "val=v2",
    ]
    if dry_run:
        print("[dry-run] would run:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    print("rolled back to rebalance-router=v2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        required=True,
        help="Path to CLN log file containing REBAL_PICK/REBAL_SKIP lines",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Success rate drop threshold that triggers rollback (default 0.10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without actually flipping setconfig",
    )
    args = parser.parse_args()

    counters = parse_log(args.log)
    print(f"counters: {dict(counters)}")
    rollback, reason = should_rollback(counters, args.threshold)
    print(f"decision: rollback={rollback}, reason={reason}")

    if rollback:
        rollback_to_v2(args.dry_run)
        return 1 if not args.dry_run else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
