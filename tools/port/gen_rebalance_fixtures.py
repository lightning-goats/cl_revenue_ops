#!/usr/bin/env python3
"""Generate rebalance-stack parity fixtures for the Rust port (Phase 5).

Run from the repo root (~/bin/cl_revenue_ops-port, branch `port`):

    python3 tools/port/gen_rebalance_fixtures.py modes > fixtures/rebalance/modes.json

This is the ONE generator every Phase-5 task extends: each suite is a
subcommand printing JSON to stdout (the Rust repo commits the redirected
output under `fixtures/rebalance/<suite>.json`). Later tasks add subcommands
(planner, segstore, router, executor, ev, cooldowns, inbound_fee, defib) —
do not fork a second generator (the `gen_fees_fixtures.py` / `gen_flow_
fixtures.py` precedent).

The real module code is the oracle: `modules.rebalance_modes.MODES` /
`engine_kwargs` are called directly; nothing here reimplements the table.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.rebalance_modes import MODES, engine_kwargs  # noqa: E402

MODE_NAMES = ["normal", "hot_protection", "structural_drain", "manual", "diagnostic"]


def gen_modes() -> dict:
    """Dump the MODES dict + engine_kwargs for all 5 modes (Phase 5 Task 1).

    Feeds `crates/revops-rebalance/src/modes.rs` /
    `crates/revops-rebalance/tests/modes.rs` in the Rust port
    (cl-revenue-ops-r), committed there as `fixtures/rebalance/modes.json`.
    """
    assert set(MODES.keys()) == set(MODE_NAMES), (
        f"MODES keys changed: {sorted(MODES.keys())} vs expected {sorted(MODE_NAMES)}"
    )
    modes_out = {}
    kwargs_out = {}
    for name in MODE_NAMES:
        m = MODES[name]
        modes_out[name] = {
            "name": m.name,
            "priority": m.priority,
            "budget_bucket": m.budget_bucket,
            "reserve_on_rail": m.reserve_on_rail,
            "account_costs": m.account_costs,
            "deadline": m.deadline,
            "accounting_owner": m.accounting_owner,
        }
        kwargs_out[name] = engine_kwargs(name)
    return {"modes": modes_out, "engine_kwargs": kwargs_out}


SUITES = {
    "modes": gen_modes,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(SUITES.keys()))
    parser.add_argument(
        "outfile",
        nargs="?",
        default="-",
        help="output path, or '-' for stdout (default)",
    )
    args = parser.parse_args()

    out = SUITES[args.suite]()
    text = json.dumps(out, indent=1) + "\n"
    if args.outfile == "-":
        sys.stdout.write(text)
    else:
        Path(args.outfile).write_text(text)
        print(f"wrote {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
