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
from modules.segment_observations import SegmentObservationStore  # noqa: E402

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


def gen_segstore() -> dict:
    """Dump `SegmentObservationStore` bucket-boundary + scripted
    record/prune/export parity data (Phase 5 Task 3).

    Feeds `crates/revops-rebalance/src/segstore.rs` /
    inline unit tests in the Rust port (cl-revenue-ops-r), committed there
    as `fixtures/rebalance/segstore.json`. The real class is the oracle:
    `SegmentObservationStore.bucket_amount_sats` /
    `.record_failure` / `.export_snapshot` are called directly; nothing here
    reimplements bucket, validation, TTL-prune, or sort semantics.
    """
    edges = [
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
    ]
    amounts = {0, 1, 49_999, 50_000, 50_001, 10_000_000 + 1, 15_000_000, -5, -1}
    for e in edges:
        amounts.update({e - 1, e, e + 1})
    bucket_cases = [
        {"amount_sats": a, "bucket": SegmentObservationStore.bucket_amount_sats(a)}
        for a in sorted(amounts)
    ]

    # Scripted record/prune/export sequence: exercises bucket rejection
    # (amount_sats <= 0, sequence number NOT consumed), invalid-direction
    # exclusion, empty-short_channel_id exclusion, unknown-failure_class
    # normalization, confidence clamping to [0, 1], FIFO eviction beyond
    # max_observations, TTL-boundary pruning across two export calls, and
    # stable descending sort by observed_at (with observed_at ties).
    store = SegmentObservationStore(ttl_seconds=500, max_observations=4)
    record_calls = [
        dict(
            short_channel_id="111x1x0",
            direction=0,
            amount_sats=10_000,
            failure_class="liquidity",
            confidence=0.9,
            observed_at=1000,
        ),
        dict(
            short_channel_id="zero-amount",
            direction=0,
            amount_sats=0,
            failure_class="liquidity",
            confidence=0.5,
            observed_at=1005,
        ),
        dict(
            short_channel_id="neg-amount",
            direction=0,
            amount_sats=-5,
            failure_class="liquidity",
            confidence=0.5,
            observed_at=1006,
        ),
        dict(
            short_channel_id="222x2x1",
            direction=1,
            amount_sats=60_000,
            failure_class="fee",
            confidence=1.5,
            observed_at=1010,
        ),
        dict(
            short_channel_id="333x3x0",
            direction=2,
            amount_sats=200_000,
            failure_class="timeout",
            confidence=-0.3,
            observed_at=1020,
        ),
        dict(
            short_channel_id="444x4x1",
            direction=1,
            amount_sats=5_000_000,
            failure_class="not_a_real_class",
            confidence=2.0,
            observed_at=1030,
        ),
        dict(
            short_channel_id="",
            direction=0,
            amount_sats=80_000,
            failure_class="liquidity",
            confidence=0.4,
            observed_at=1040,
        ),
        dict(
            short_channel_id="555x5x0",
            direction=0,
            amount_sats=80_000,
            failure_class="liquidity",
            confidence=0.4,
            observed_at=1040,
        ),
    ]

    # `steps` is a SINGLE chronologically-ordered list (not separate
    # records/exports arrays) precisely because interleaving matters:
    # export_snapshot mutates the internal ring (prunes it permanently), so
    # a replaying test MUST perform record/export calls in exactly this
    # order, not all records followed by all exports.
    steps = []
    for call in record_calls:
        result = store.record_failure(**call)
        steps.append({"op": "record", "args": call, "result": result})

    steps.append(
        {
            "op": "export",
            "now": 1050,
            "snapshot": store.export_snapshot(observer_member_id="test-node", now=1050),
        }
    )

    late_call = dict(
        short_channel_id="666x6x0",
        direction=0,
        amount_sats=300_000,
        failure_class="liquidity",
        confidence=0.7,
        observed_at=1600,
    )
    steps.append(
        {"op": "record", "args": late_call, "result": store.record_failure(**late_call)}
    )

    steps.append(
        {
            "op": "export",
            "now": 1600,
            "snapshot": store.export_snapshot(observer_member_id="test-node", now=1600),
        }
    )

    return {
        "bucket_boundaries": bucket_cases,
        "sequence": {
            "init": {"ttl_seconds": 500, "max_observations": 4},
            "steps": steps,
        },
    }


SUITES = {
    "modes": gen_modes,
    "segstore": gen_segstore,
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
