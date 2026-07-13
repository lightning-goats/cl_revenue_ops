#!/usr/bin/env python3
"""Migration scanner for the 2026-08-12 compatibility-window removals
(gap-closure Phase H, PR 11).

Checks, per removal item, whether the system is READY for the deletion:

1. `rebalance_min_profit` (deprecated no-op) — must not be set in the
   CLN config file nor persisted in config_overrides.
2. Legacy `budget_reservations` table — must contain zero ACTIVE rows
   (settled/released history is archivable).
3. v0 schema emission cutover — informational: lists the emission
   touchpoints that switch to schema_version 1.

Exit 0 = ready for removal; 1 = blockers found (each printed with the
operator action that clears it). Read-only: scans, never mutates.

Usage:
    python3 tools/deprecation_scan.py [--db PATH] [--config PATH]
Defaults target the production layout.
"""
import argparse
import pathlib
import sqlite3
import sys

DEPRECATED_KEY = "rebalance_min_profit"
REPLACEMENT = "rebalance_hold_margin"


def scan_config_file(path: pathlib.Path) -> list:
    blockers = []
    if not path.exists():
        return blockers
    forms = (DEPRECATED_KEY, DEPRECATED_KEY.replace("_", "-"))
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if any(form in stripped for form in forms):
            blockers.append(
                f"config file {path}:{lineno} sets {DEPRECATED_KEY!r} — "
                f"remove the line (migrate intent to {REPLACEMENT!r})")
    return blockers


def scan_db(path: pathlib.Path) -> list:
    blockers = []
    if not path.exists():
        blockers.append(f"database not found: {path}")
        return blockers
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            rows = conn.execute(
                "SELECT value FROM config_overrides WHERE key = ?",
                (DEPRECATED_KEY,)).fetchall()
            if rows:
                blockers.append(
                    f"config_overrides persists {DEPRECATED_KEY!r}="
                    f"{rows[0][0]!r} — clear it before removal "
                    f"(revenue-config set is not needed; delete the "
                    f"override row after confirming {REPLACEMENT!r} "
                    f"carries the intent)")
        except sqlite3.OperationalError:
            pass  # no overrides table = nothing persisted
        try:
            active = conn.execute(
                "SELECT COUNT(*) FROM budget_reservations "
                "WHERE status = 'active'").fetchone()[0]
            if active:
                blockers.append(
                    f"budget_reservations has {active} ACTIVE legacy "
                    f"row(s) — wait for release/settle (or run the "
                    f"stale-reservation sweep) before dropping the "
                    f"dual-path")
        except sqlite3.OperationalError:
            pass  # table already gone = nothing to migrate
    finally:
        conn.close()
    return blockers


def emission_touchpoints() -> list:
    """v0 -> v1 emission cutover sites (informational)."""
    return [
        "modules/econ_snapshot.py: EconomicSnapshot wire "
        "(schema_version 0 -> 1; v1 is closed — additive fields become "
        "a v2 matter)",
        "modules/econ_intents.py: to_wire (schema_version 0 -> 1)",
        "tests/conformance corpus + validator accept both during the "
        "window; regenerate the corpus after cutover",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",
                        default="/data/lightningd/.lightning/"
                                "revenue_ops.db")
    parser.add_argument("--config",
                        default="/data/lightningd/config")
    args = parser.parse_args()

    blockers = []
    blockers += scan_config_file(pathlib.Path(args.config))
    blockers += scan_db(pathlib.Path(args.db))

    print("=== deprecation_scan: 2026-08-12 window readiness ===")
    if blockers:
        for blocker in blockers:
            print(f"BLOCKER: {blocker}")
    else:
        print("READY: no deprecated-key usage; no active legacy "
              "reservations.")
    print("\nv0->v1 emission touchpoints (cutover step, not a blocker):")
    for note in emission_touchpoints():
        print(f"  - {note}")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
