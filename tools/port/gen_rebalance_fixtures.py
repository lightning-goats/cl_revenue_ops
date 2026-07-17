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
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.rebalance_modes import MODES, engine_kwargs  # noqa: E402
from modules.rebalance_planner_v2 import RebalancePlanner  # noqa: E402
from modules.rebalance_state_v2 import ChannelState, StateSnapshot  # noqa: E402
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



# ---------------------------------------------------------------------------
# planner suite (Phase 5 Task 2) — RebalancePlanner golden parity.
#
# The Rust `PlannerChannel` shape (crates/revops-rebalance/src/planner.rs)
# is a REDUCED subset of the real `ChannelState`: it carries the numbers a
# live snapshot builder (`rebalance_state_v2.build_state_snapshot`, ported
# later at plugin-wiring time — see the plan's "Explicitly Deferred"
# section) would already have computed — dest_urgency/source_drain_score as
# plain `urgency`/`drain` floats, remaining capex budget as
# `capex_remaining_sats`, and per-channel target band as `band_low`/
# `band_high` — rather than raw RPC fields. It does NOT carry
# source_eligible/dest_eligible/cooldown (that gating lives in the deferred
# state builder), so this generator always constructs ChannelState with
# source_eligible=dest_eligible=True: every T2 case exercises ONLY the
# band-classification + pairing + scoring math RebalancePlanner.plan()
# itself owns, never the upstream eligibility gate.
#
# local_ratio is derived from spendable_sats/capacity_sats and rounded to 6
# decimals, exactly mirroring build_state_snapshot's `round(local_ratio, 6)`
# (rebalance_state_v2.py ~380) — this matters for exact-tie band-edge cases.
# ---------------------------------------------------------------------------

VALUE_CLASSES = ["profitable", "active", "funded", "neutral"]
BAND_CHOICES = [(0.35, 0.65), (0.30, 0.70), (0.20, 0.80), (0.40, 0.60)]
CAPACITY_CHOICES = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
INBOUND_PPM_CHOICES = [0, 50, 250, 800, 1500, 3000, 5000, 6000, 12000]
CAPEX_CHOICES = [0, 1_000, 5_000, 20_000, 100_000]
MAX_CHUNK_CHOICES = [100_000, 500_000, 2_000_000, 10_000_000]
MAX_PAIRS_CHOICES = [1, 2, 3, 5, 10]
FEE_CAP_CHOICES = [0, 50, 500, 2_000]


def _channel_local_ratio(spendable_sats: int, capacity_sats: int) -> float:
    capacity = max(0, int(capacity_sats))
    spendable = max(0, int(spendable_sats))
    ratio = 0.0
    if capacity > 0:
        ratio = min(1.0, max(0.0, spendable / capacity))
    return round(ratio, 6)


def _channel_state_from_case(ch: dict) -> ChannelState:
    capacity_sats = max(0, int(ch["capacity_sats"]))
    return ChannelState(
        channel_id=ch["channel_id"],
        peer_id=ch["peer_id"],
        capacity_sats=capacity_sats,
        local_ratio=_channel_local_ratio(ch["spendable_sats"], capacity_sats),
        actual_inbound_fee_ppm=int(ch["inbound_ppm"]),
        value_class=ch["value_class"],
        is_valuable=True,
        remaining_budget_sats=max(0, int(ch["capex_remaining_sats"])),
        cooldown_active=False,
        source_eligible=True,
        dest_eligible=True,
        source_reason="",
        dest_reason="",
        dest_urgency=float(ch["urgency"]),
        source_drain_score=float(ch["drain"]),
        budget_source="capex" if int(ch["capex_remaining_sats"]) > 0 else "none",
        target_band_low=float(ch["band_low"]),
        target_band_high=float(ch["band_high"]),
    )


def _run_planner_case(case: dict) -> dict:
    snapshot = StateSnapshot(
        channels=tuple(_channel_state_from_case(c) for c in case["channels"])
    )
    params = case["params"]
    planner = RebalancePlanner(
        max_chunk_sats=int(params["max_chunk_sats"]),
        max_pairs=int(params["max_pairs"]),
        pair_fee_cap_ppm=int(params["pair_fee_cap_ppm"]),
    )
    result = planner.plan(snapshot)

    pairs = [
        {
            "source": p.source_channel_id,
            "dest": p.dest_channel_id,
            "amount_sats": p.amount_sats,
            "pair_budget_sats": p.pair_budget_sats,
            "pair_fee_cap_ppm": int(params["pair_fee_cap_ppm"]),
            "score": p.score,
        }
        for p in result.selected
    ]
    skips = [
        {
            "channel_id": s.channel_id,
            "reason": s.reason,
            "value_class": s.value_class,
            "remaining_budget_sats": s.remaining_budget_sats,
            "detail": s.detail,
        }
        for s in result.skipped
    ]
    dd = result.drain_demand
    drain_demand = [
        {
            "channel_id": e.channel_id,
            "peer_id": e.peer_id,
            "excess_sats": e.excess_sats,
            "drain_score": e.drain_score,
            "value_class": e.value_class,
        }
        for e in (dd.entries if dd is not None else [])
    ]
    return {"pairs": pairs, "skips": skips, "drain_demand": drain_demand}


def _random_channel(rng: random.Random, channel_id: str, peer_pool: list) -> dict:
    capacity_sats = rng.choice(CAPACITY_CHOICES)
    spendable_sats = rng.randint(0, capacity_sats)
    band_low, band_high = rng.choice(BAND_CHOICES)
    return {
        "channel_id": channel_id,
        "peer_id": rng.choice(peer_pool),
        "capacity_sats": capacity_sats,
        "spendable_sats": spendable_sats,
        "receivable_sats": max(0, capacity_sats - spendable_sats - rng.randint(0, capacity_sats // 20 or 1)),
        "band_low": band_low,
        "band_high": band_high,
        "inbound_ppm": rng.choice(INBOUND_PPM_CHOICES),
        "value_class": rng.choice(VALUE_CLASSES),
        "urgency": round(rng.uniform(0.0, 1.0), 6),
        "drain": round(rng.uniform(0.0, 1.0), 6),
        "capex_remaining_sats": rng.choice(CAPEX_CHOICES),
    }


def _random_case(rng: random.Random, i: int) -> dict:
    n = rng.randint(2, 8)
    peer_pool = [f"peer-{i}-{p}" for p in range(max(2, n // 2 + 1))]
    channels = [_random_channel(rng, f"chan-{i}-{j}", peer_pool) for j in range(n)]
    params = {
        "max_chunk_sats": rng.choice(MAX_CHUNK_CHOICES),
        "max_pairs": rng.choice(MAX_PAIRS_CHOICES),
        "pair_fee_cap_ppm": rng.choice(FEE_CAP_CHOICES),
    }
    return {"case_id": f"random_{i:03d}", "params": params, "channels": channels}


def _hand_built_cases() -> list:
    cases = []

    # 1. Band edges: local_ratio exactly == band_high / == band_low is
    # "inside_band" (python's classification is strict > / strict <).
    cases.append({
        "case_id": "band_edge_high_exact_inside_band",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 0},
        "channels": [
            {
                "channel_id": "edge-high", "peer_id": "peer-eh", "capacity_sats": 1_000_000,
                "spendable_sats": 650_000, "receivable_sats": 350_000,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 100,
                "value_class": "neutral", "urgency": 0.1, "drain": 0.1,
                "capex_remaining_sats": 0,
            },
        ],
    })
    cases.append({
        "case_id": "band_edge_low_exact_inside_band",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 0},
        "channels": [
            {
                "channel_id": "edge-low", "peer_id": "peer-el", "capacity_sats": 1_000_000,
                "spendable_sats": 350_000, "receivable_sats": 650_000,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 100,
                "value_class": "neutral", "urgency": 0.1, "drain": 0.1,
                "capex_remaining_sats": 0,
            },
        ],
    })

    # 2. Round-half-even proof: local_ratio=1.0, band_high=0.75 -> delta
    # exactly 0.25 (both exactly representable in binary), capacity chosen
    # so delta*capacity lands EXACTLY on an n.5 tie whose nearest EVEN
    # neighbor is the LOWER integer (0.5->0, 2.5->2, 4.5->4) -- the case a
    # naive "round half away from zero" (Rust's f64::round()) gets wrong.
    # Isolated as unpaired over-local channels (no over-remote counterpart)
    # so the tie shows up untouched in drain_demand's excess_sats.
    half_even_channels = []
    for idx, capacity in enumerate([2, 10, 18]):
        half_even_channels.append({
            "channel_id": f"tie-{capacity}", "peer_id": f"peer-tie-{idx}",
            "capacity_sats": capacity, "spendable_sats": capacity, "receivable_sats": 0,
            "band_low": 0.35, "band_high": 0.75, "inbound_ppm": 0,
            "value_class": "neutral", "urgency": 0.0, "drain": 0.5,
            "capex_remaining_sats": 0,
        })
    cases.append({
        "case_id": "half_even_rounding_case",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 0},
        "channels": half_even_channels,
    })

    # 3. Stable-tie case: 2 sources x 2 dests, all four pairs score
    # IDENTICALLY (same dest_urgency/value_class/source_drain/inbound_ppm on
    # every channel) so only a stable sort reproduces the expected greedy
    # selection: (src0,dest0) then (src1,dest1), in original nested-loop
    # generation order.
    stable_channels = []
    for i in range(2):
        stable_channels.append({
            "channel_id": f"stable-src-{i}", "peer_id": f"stable-src-peer-{i}",
            "capacity_sats": 1_000_000, "spendable_sats": 900_000, "receivable_sats": 100_000,
            "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 500,
            "value_class": "neutral", "urgency": 0.0, "drain": 0.4,
            "capex_remaining_sats": 0,
        })
        stable_channels.append({
            "channel_id": f"stable-dest-{i}", "peer_id": f"stable-dest-peer-{i}",
            "capacity_sats": 1_000_000, "spendable_sats": 100_000, "receivable_sats": 900_000,
            "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 500,
            "value_class": "active", "urgency": 0.6, "drain": 0.0,
            "capex_remaining_sats": 50_000,
        })
    cases.append({
        "case_id": "stable_tie_case",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 0},
        "channels": stable_channels,
    })

    # 4. Peer collision: the only source/dest pair shares a peer_id, so no
    # candidate pair is ever generated even though both are eligible and
    # imbalanced (python's `_generate_pairs` skips same-peer combinations).
    cases.append({
        "case_id": "peer_collision_no_pair",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 0},
        "channels": [
            {
                "channel_id": "pc-src", "peer_id": "shared-peer", "capacity_sats": 1_000_000,
                "spendable_sats": 900_000, "receivable_sats": 100_000,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 100,
                "value_class": "neutral", "urgency": 0.0, "drain": 0.5,
                "capex_remaining_sats": 0,
            },
            {
                "channel_id": "pc-dest", "peer_id": "shared-peer", "capacity_sats": 1_000_000,
                "spendable_sats": 100_000, "receivable_sats": 900_000,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 100,
                "value_class": "profitable", "urgency": 0.8, "drain": 0.0,
                "capex_remaining_sats": 20_000,
            },
        ],
    })

    # 5. max_pairs_reached: 3 valid, non-conflicting pairs but max_pairs=1.
    max_pairs_channels = []
    for i in range(3):
        max_pairs_channels.append({
            "channel_id": f"mp-src-{i}", "peer_id": f"mp-src-peer-{i}",
            "capacity_sats": 1_000_000, "spendable_sats": 900_000, "receivable_sats": 100_000,
            "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 100 * i,
            "value_class": "neutral", "urgency": 0.0, "drain": round(0.9 - 0.1 * i, 6),
            "capex_remaining_sats": 0,
        })
        max_pairs_channels.append({
            "channel_id": f"mp-dest-{i}", "peer_id": f"mp-dest-peer-{i}",
            "capacity_sats": 1_000_000, "spendable_sats": 100_000, "receivable_sats": 900_000,
            "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 0,
            "value_class": "active", "urgency": round(0.9 - 0.1 * i, 6), "drain": 0.0,
            "capex_remaining_sats": 10_000,
        })
    cases.append({
        "case_id": "max_pairs_reached_case",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 1, "pair_fee_cap_ppm": 0},
        "channels": max_pairs_channels,
    })

    # 6. Value-class table: one dest per value class, same source, disjoint
    # peers, to directly exercise {profitable:2, active:1, funded:1,
    # neutral:0} (+ an unmapped class falling back to 0 like "neutral").
    value_class_dests = []
    for i, vc in enumerate(["profitable", "active", "funded", "neutral", "totally_unmapped"]):
        value_class_dests.append({
            "channel_id": f"vc-dest-{vc}", "peer_id": f"vc-dest-peer-{i}",
            "capacity_sats": 1_000_000, "spendable_sats": 100_000, "receivable_sats": 900_000,
            "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 0,
            "value_class": vc, "urgency": 0.5, "drain": 0.0,
            "capex_remaining_sats": 30_000,
        })
    value_class_sources = []
    for i in range(len(value_class_dests)):
        value_class_sources.append({
            "channel_id": f"vc-src-{i}", "peer_id": f"vc-src-peer-{i}",
            "capacity_sats": 1_000_000, "spendable_sats": 900_000, "receivable_sats": 100_000,
            "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 0,
            "value_class": "neutral", "urgency": 0.0, "drain": 0.5,
            "capex_remaining_sats": 0,
        })
    cases.append({
        "case_id": "value_class_table_case",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 0},
        "channels": value_class_sources + value_class_dests,
    })

    # 7. Fee-cap-dominated pair budget: dest has zero capex remaining, so
    # pair_budget must come entirely from ceil(amount*pair_fee_cap_ppm/1e6).
    cases.append({
        "case_id": "fee_cap_dominates_pair_budget",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 2_500},
        "channels": [
            {
                "channel_id": "fc-src", "peer_id": "fc-src-peer", "capacity_sats": 2_000_000,
                "spendable_sats": 1_800_000, "receivable_sats": 200_000,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 0,
                "value_class": "neutral", "urgency": 0.0, "drain": 0.9,
                "capex_remaining_sats": 0,
            },
            {
                "channel_id": "fc-dest", "peer_id": "fc-dest-peer", "capacity_sats": 2_000_000,
                "spendable_sats": 200_000, "receivable_sats": 1_800_000,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 0,
                "value_class": "profitable", "urgency": 0.9, "drain": 0.0,
                "capex_remaining_sats": 0,
            },
        ],
    })

    # 8. Zero-capacity channel: guards the local_ratio divide-by-zero path
    # (capacity_sats<=0 -> local_ratio=0.0, never crashes).
    cases.append({
        "case_id": "zero_capacity_channel",
        "params": {"max_chunk_sats": 2_000_000, "max_pairs": 10, "pair_fee_cap_ppm": 0},
        "channels": [
            {
                "channel_id": "zc-chan", "peer_id": "zc-peer", "capacity_sats": 0,
                "spendable_sats": 0, "receivable_sats": 0,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 0,
                "value_class": "neutral", "urgency": 0.0, "drain": 0.0,
                "capex_remaining_sats": 0,
            },
            {
                "channel_id": "zc-partner-src", "peer_id": "zc-partner-peer",
                "capacity_sats": 1_000_000, "spendable_sats": 900_000, "receivable_sats": 100_000,
                "band_low": 0.35, "band_high": 0.65, "inbound_ppm": 0,
                "value_class": "neutral", "urgency": 0.0, "drain": 0.5,
                "capex_remaining_sats": 0,
            },
        ],
    })

    return cases


def gen_planner() -> dict:
    """Drive the REAL `RebalancePlanner` over ~30 randomized-but-seeded
    snapshots plus hand-built band-edge/tie/rounding cases (Phase 5 Task 2).

    Feeds `crates/revops-rebalance/src/planner.rs` /
    `crates/revops-rebalance/tests/planner.rs` in the Rust port
    (cl-revenue-ops-r), committed there as `fixtures/rebalance/planner.json`.
    """
    rng = random.Random(20260717)
    cases = _hand_built_cases()
    cases.extend(_random_case(rng, i) for i in range(30))

    seen_ids = set()
    for case in cases:
        assert case["case_id"] not in seen_ids, f"duplicate case_id {case['case_id']!r}"
        seen_ids.add(case["case_id"])
        case["expected"] = _run_planner_case(case)

    return {"cases": cases}
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


# ---------------------------------------------------------------------------
# router suite (Phase 5 Task 4)
# ---------------------------------------------------------------------------

# Deterministic ids: hex ordering drives CLN's channel direction bit
# (1 if start_node > end_node), so the set below pins both directions.
_OUR = "02" + "aa" * 32
_PEER_A = "02" + "cc" * 32  # source peer > our node  -> first-hop direction 0
_PEER_B = "02" + "44" * 32  # dest peer  < our node   -> final-hop direction 0
_PEER_C = "02" + "11" * 32  # source peer < our node  -> first-hop direction 1
_PEER_D = "02" + "ee" * 32  # dest peer  > our node   -> final-hop direction 1
_M1 = "03" + "11" * 32
_M2 = "03" + "22" * 32
_M3 = "03" + "33" * 32
_SRC_SCID = "800000x1000x1"
_DST_SCID = "800001x2000x1"
_SRC2_SCID = "800002x3000x0"
_DST2_SCID = "800003x4000x0"
_MID1 = "700000x100x0"  # peer_A -> M1
_MID2 = "700001x200x0"  # M1 -> peer_B (2-hop) / M1 -> M2 (3-hop)
_MID3 = "700002x300x0"  # M2 -> peer_B
_MID4 = "700003x400x0"  # peer_C -> peer_D (single-hop middle)


class _RouterRpcStub:
    """Scripted `plugin.rpc` double for RebalanceRouterV3 (data_service=None).

    Serves canned responses and records every askrene/getroutes interaction
    so the fixture pins the payload SHAPES (method, params — final_cltv
    arithmetic, layer lists) alongside the RouteResult outputs.
    """

    def __init__(self, setup):
        self._setup = setup
        self._getroutes_i = 0
        self.reset()

    def reset(self):
        self.getroutes_calls = []
        self.creates = []
        self.updates = {}  # layer -> [scid_dir, ...] in call order
        self.removes = []
        self.listlayers_calls = 0

    def call(self, method, params):
        if method == "askrene-listlayers":
            self.listlayers_calls += 1
            return {
                "layers": [
                    {"layer": n} for n in self._setup.get("live_layers", [])
                ]
            }
        if method == "askrene-create-layer":
            self.creates.append(params["layer"])
            return {}
        if method == "askrene-update-channel":
            self.updates.setdefault(params["layer"], []).append(
                params["short_channel_id_dir"]
            )
            assert params["enabled"] is False
            return {}
        if method == "askrene-remove-layer":
            self.removes.append(params["layer"])
            return {}
        raise AssertionError(f"unexpected rpc.call({method!r})")

    def getroutes(self, **kwargs):
        self.getroutes_calls.append(dict(kwargs))
        script = self._setup["getroutes"]
        entry = script[min(self._getroutes_i, len(script) - 1)]
        self._getroutes_i += 1
        if entry.get("error") is not None:
            raise Exception(entry["error"])
        return entry["result"]

    def listpeerchannels(self, peer_id=None):
        return {
            "channels": list(self._setup.get("peers", {}).get(peer_id, []))
        }

    def listchannels(self, source=None, short_channel_id=None):
        chans = self._setup.get("gossip_channels", [])
        if short_channel_id is not None:
            out = [
                c for c in chans if c.get("short_channel_id") == short_channel_id
            ]
        elif source is not None:
            out = [c for c in chans if c.get("source") == source]
        else:
            out = list(chans)
        return {"channels": out}

    def listconfigs(self):
        return {
            "configs": {
                "cltv-final": {"value_int": self._setup["invoice_final_cltv"]}
            }
        }


class _StubPlugin:
    def __init__(self, rpc):
        self.rpc = rpc

    def log(self, msg, level=None):
        pass


def _nolog(msg, level=None):
    pass


def _norm_layer(name, mapping):
    return mapping.get(name, name)


def _router_case_inputs():
    """Input-only case table; expected outputs come from the real Python."""
    remote_b = {
        "fee_proportional_millionths": 250,
        "fee_base_msat": 0,
        "cltv_expiry_delta": 34,
    }
    peers_b = {
        _PEER_B: [
            {
                "peer_id": _PEER_B,
                "short_channel_id": _DST_SCID,
                "updates": {"remote": dict(remote_b)},
            }
        ]
    }
    gossip_2hop = [
        # peer_A -> M1 (first middle edge; source-peer forwarding policy)
        {
            "short_channel_id": _MID1,
            "source": _PEER_A,
            "destination": _M1,
            "fee_per_millionth": 100,
            "base_fee_millisatoshi": 1000,
            "delay": 34,
        },
        # M1 -> peer_B (reprice edge)
        {
            "short_channel_id": _MID2,
            "source": _M1,
            "destination": _PEER_B,
            "fee_per_millionth": 150,
            "base_fee_millisatoshi": 0,
            "delay": 40,
        },
    ]
    path_2hop = [
        {
            "short_channel_id_dir": f"{_MID1}/1",
            "next_node_id": _M1,
            "amount_msat": 500_200_000,
            "delay": 92,
        },
        {
            "short_channel_id_dir": f"{_MID2}/0",
            "next_node_id": _PEER_B,
            "amount_msat": 500_125_000,
            "delay": 52,
        },
    ]
    route_2hop = {
        "probability_ppm": 875_000,
        "amount_msat": 500_125_000,
        "path": path_2hop,
    }
    base_pair = {
        "source_channel_id": _SRC_SCID,
        "dest_channel_id": _DST_SCID,
        "source_peer_id": _PEER_A,
        "dest_peer_id": _PEER_B,
        "amount_sats": 500_000,
    }
    base_setup = {
        "our_node_id": _OUR,
        "layer_names": ["xpay", "bimodal"],
        "invoice_final_cltv": 18,
        "live_layers": ["xpay", "auto.no_mpp_support"],
        "peers": peers_b,
        "gossip_channels": gossip_2hop,
    }

    def setup(**over):
        s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base_setup.items()}
        s.update(over)
        return s

    cases = []

    # 1. Cheapest-route selection: empty-path route (sentinel), expensive
    # route, cheap route — cheapest by (first-hop amt - delivered).
    expensive_path = [
        {
            "short_channel_id_dir": f"{_MID3}/0",
            "next_node_id": _M2,
            "amount_msat": 500_300_000,
            "delay": 100,
        },
        {
            "short_channel_id_dir": f"{_MID2}/0",
            "next_node_id": _PEER_B,
            "amount_msat": 500_125_000,
            "delay": 52,
        },
    ]
    cases.append(
        {
            "name": "multi_route_picks_cheapest_int_msat",
            "setup": setup(
                getroutes=[
                    {
                        "result": {
                            "routes": [
                                {"amount_msat": 500_125_000, "path": []},
                                {
                                    "probability_ppm": 900_000,
                                    "amount_msat": 500_125_000,
                                    "path": expensive_path,
                                },
                                route_2hop,
                            ]
                        }
                    }
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 2. "Nmsat" string amounts everywhere in the getroutes response.
    def _msat_str(route):
        r = json.loads(json.dumps(route))
        r["amount_msat"] = f"{r['amount_msat']}msat"
        for hop in r["path"]:
            hop["amount_msat"] = f"{hop['amount_msat']}msat"
        return r

    cases.append(
        {
            "name": "nmsat_string_amounts",
            "setup": setup(
                getroutes=[{"result": {"routes": [_msat_str(route_2hop)]}}]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 3. Missing probability_ppm defaults to 0.
    no_prob = json.loads(json.dumps(route_2hop))
    del no_prob["probability_ppm"]
    cases.append(
        {
            "name": "missing_probability_defaults_zero",
            "setup": setup(getroutes=[{"result": {"routes": [no_prob]}}]),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 4. Middle path routing through us -> rejected.
    loop_path = json.loads(json.dumps(path_2hop))
    loop_path[0]["next_node_id"] = _OUR
    cases.append(
        {
            "name": "loop_through_us_rejected",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [{"amount_msat": 500_125_000, "path": loop_path}]}}
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 5. Path not terminating at dest peer -> rejected.
    wrong_end = json.loads(json.dumps(path_2hop))
    wrong_end[-1]["next_node_id"] = _M2
    cases.append(
        {
            "name": "wrong_terminus_rejected",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [{"amount_msat": 500_125_000, "path": wrong_end}]}}
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 6. Empty routes list.
    cases.append(
        {
            "name": "empty_routes_list",
            "setup": setup(getroutes=[{"result": {"routes": []}}]),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 7. Only route has an empty path: sentinel keeps it "cheapest", then
    # validation rejects it.
    cases.append(
        {
            "name": "all_paths_empty_rejected",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [{"amount_msat": 1_000, "path": []}]}}
                ]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 8. getroutes error translation (full pipeline).
    cases.append(
        {
            "name": "getroutes_error_unknown_source",
            "setup": setup(
                getroutes=[{"error": f"Unknown source node {_PEER_A}"}]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )
    cases.append(
        {
            "name": "getroutes_error_child_died",
            "setup": setup(
                getroutes=[{"error": "askrene: child died with signal 6"}]
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 9. 'Unknown layer' invalidates the cycle caches: exclude layer is torn
    # down, listlayers re-probed, a fresh exclude layer minted on retry.
    cases.append(
        {
            "name": "unknown_layer_invalidates_cycle_caches",
            "setup": setup(
                getroutes=[
                    {"error": "Unknown layer 'rebalance-exclude-3-7'"},
                    {"result": {"routes": [route_2hop]}},
                ]
            ),
            "calls": [
                {"pair": dict(base_pair), "exclude": []},
                {"pair": dict(base_pair), "exclude": []},
            ],
        }
    )

    # 10. Final-hop policy from gossip fallback (no updates.remote).
    cases.append(
        {
            "name": "final_hop_policy_gossip_fallback",
            "setup": setup(
                peers={
                    _PEER_B: [
                        {"peer_id": _PEER_B, "short_channel_id": _DST_SCID}
                    ]
                },
                gossip_channels=gossip_2hop
                + [
                    {
                        "short_channel_id": _DST_SCID,
                        "source": _PEER_B,
                        "destination": _OUR,
                        "fee_per_millionth": 300,
                        "base_fee_millisatoshi": 0,
                        "delay": 34,
                    }
                ],
                getroutes=[{"result": {"routes": [route_2hop]}}],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 11. No final-hop policy anywhere.
    cases.append(
        {
            "name": "final_hop_policy_none",
            "setup": setup(peers={}, gossip_channels=[], getroutes=[]),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 12. Policy without cltv_expiry_delta -> dest-cltv fallback default 40
    # (pins final_cltv arithmetic: 40 + invoice_final_cltv).
    cases.append(
        {
            "name": "dest_cltv_default_40",
            "setup": setup(
                peers={
                    _PEER_B: [
                        {
                            "peer_id": _PEER_B,
                            "short_channel_id": _DST_SCID,
                            "updates": {
                                "remote": {
                                    "fee_proportional_millionths": 250,
                                    "fee_base_msat": 0,
                                }
                            },
                        }
                    ]
                },
                getroutes=[{"result": {"routes": [route_2hop]}}],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 13. Source peer forwarding policy missing -> hard failure.
    cases.append(
        {
            "name": "first_middle_policy_missing",
            "setup": setup(
                gossip_channels=[],
                getroutes=[{"result": {"routes": [route_2hop]}}],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 14. Reprice keeps the router amount when a middle policy is missing
    # (3-hop middle; M1->M2 edge absent from gossip).
    path_3hop = [
        {
            "short_channel_id_dir": f"{_MID1}/1",
            "next_node_id": _M1,
            "amount_msat": 500_260_000,
            "delay": 132,
        },
        {
            "short_channel_id_dir": f"{_MID2}/0",
            "next_node_id": _M2,
            "amount_msat": 500_190_000,
            "delay": 92,
        },
        {
            "short_channel_id_dir": f"{_MID3}/1",
            "next_node_id": _PEER_B,
            "amount_msat": 500_125_000,
            "delay": 52,
        },
    ]
    cases.append(
        {
            "name": "reprice_missing_policy_keeps_router_amount",
            "setup": setup(
                gossip_channels=[
                    dict(gossip_2hop[0]),  # peer_A -> M1 (first middle)
                    {
                        "short_channel_id": _MID3,
                        "source": _M2,
                        "destination": _PEER_B,
                        "fee_per_millionth": 200,
                        "base_fee_millisatoshi": 500,
                        "delay": 30,
                    },
                ],
                getroutes=[
                    {
                        "result": {
                            "routes": [
                                {
                                    "probability_ppm": 640_000,
                                    "amount_msat": 500_125_000,
                                    "path": path_3hop,
                                }
                            ]
                        }
                    }
                ],
            ),
            "calls": [{"pair": dict(base_pair), "exclude": []}],
        }
    )

    # 15. Retry exclusions: directional entry passes through as-is, bare SCID
    # disables both directions, local endpoint SCIDs always appended (dest
    # already present -> not duplicated).
    cases.append(
        {
            "name": "exclude_retry_directional_and_bare",
            "setup": setup(getroutes=[{"result": {"routes": [route_2hop]}}]),
            "calls": [
                {
                    "pair": dict(base_pair),
                    "exclude": ["600000x5x0/1", "600001x6x0", _DST_SCID],
                }
            ],
        }
    )

    # 16. Cycle cache: identical exclude sets share ONE throwaway layer.
    cases.append(
        {
            "name": "cycle_reuses_exclude_layer",
            "setup": setup(
                getroutes=[
                    {"result": {"routes": [route_2hop]}},
                    {"result": {"routes": [route_2hop]}},
                ]
            ),
            "calls": [
                {"pair": dict(base_pair), "exclude": []},
                {"pair": dict(base_pair), "exclude": []},
            ],
        }
    )

    # 17. Single-hop middle + opposite direction bits + standalone layers
    # (configured []) + final-hop base fee arithmetic.
    cases.append(
        {
            "name": "single_hop_middle_standalone_layers",
            "setup": setup(
                layer_names=[],
                peers={
                    _PEER_D: [
                        {
                            "peer_id": _PEER_D,
                            "short_channel_id": _DST2_SCID,
                            "updates": {
                                "remote": {
                                    "fee_proportional_millionths": 777,
                                    "fee_base_msat": 1500,
                                    "cltv_expiry_delta": 80,
                                }
                            },
                        }
                    ]
                },
                gossip_channels=[
                    {
                        "short_channel_id": _MID4,
                        "source": _PEER_C,
                        "destination": _PEER_D,
                        "fee_per_millionth": 50,
                        "base_fee_millisatoshi": 0,
                        "delay": 14,
                    }
                ],
                getroutes=[
                    {
                        "result": {
                            "routes": [
                                {
                                    "probability_ppm": 999_999,
                                    "amount_msat": 250_195_000,
                                    "path": [
                                        {
                                            "short_channel_id_dir": f"{_MID4}/0",
                                            "next_node_id": _PEER_D,
                                            "amount_msat": 250_195_000,
                                            "delay": 98,
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                ],
            ),
            "calls": [
                {
                    "pair": {
                        "source_channel_id": _SRC2_SCID,
                        "dest_channel_id": _DST2_SCID,
                        "source_peer_id": _PEER_C,
                        "dest_peer_id": _PEER_D,
                        "amount_sats": 250_000,
                    },
                    "exclude": [],
                }
            ],
        }
    )

    return cases


_EXCLUDE_LAYER_RE = __import__("re").compile(r"^rebalance-exclude-\d+-\d+$")


def gen_router() -> dict:
    """Drive the REAL RebalanceRouterV3 over scripted RPC doubles.

    Feeds `crates/revops-rebalance/src/router.rs` /
    `crates/revops-rebalance/tests/router.rs` (Phase 5 Task 4), committed in
    the Rust repo as `fixtures/rebalance/router.json`.
    """
    import dataclasses

    from modules.rebalance_router_v3 import (
        RebalanceRouterV3,
        _configured_layer_names,
        _translate_getroutes_error,
        GETROUTES_RPC_TIMEOUT_SECONDS,
    )
    from modules.rebalance_engine_v2 import RebalanceEngine

    out_cases = []
    for case in _router_case_inputs():
        s = case["setup"]
        stub = _RouterRpcStub(s)
        router = RebalanceRouterV3(
            _StubPlugin(stub), s["our_node_id"], list(s["layer_names"]), _nolog
        )
        # The init-time probe is outside the cycle window; the Rust
        # CycleRouter has no init probe, so capture from begin_cycle on.
        stub.reset()
        router.begin_cycle()
        calls_out = []
        for call in case["calls"]:
            before = len(stub.getroutes_calls)
            p = call["pair"]
            rr = router.price_pair(
                p["source_channel_id"],
                p["dest_channel_id"],
                p["source_peer_id"],
                p["dest_peer_id"],
                p["amount_sats"],
                exclude=list(call["exclude"]) or None,
            )
            new_grs = stub.getroutes_calls[before:]
            assert len(new_grs) <= 1
            calls_out.append(
                {
                    "pair": dict(p),
                    "exclude": list(call["exclude"]),
                    "expected": dataclasses.asdict(rr),
                    "expected_getroutes_params": new_grs[0] if new_grs else None,
                }
            )
        router.end_cycle()

        # Normalize the time+counter throwaway layer names, creation order.
        mapping = {}
        for i, name in enumerate(stub.creates, start=1):
            assert _EXCLUDE_LAYER_RE.match(name), name
            mapping[name] = f"<exclude-{i}>"
        for c in calls_out:
            params = c["expected_getroutes_params"]
            if params is not None:
                params["layers"] = [
                    _norm_layer(n, mapping) for n in params["layers"]
                ]
        out_cases.append(
            {
                "name": case["name"],
                "setup": {
                    k: s[k]
                    for k in (
                        "our_node_id",
                        "layer_names",
                        "invoice_final_cltv",
                        "live_layers",
                        "peers",
                        "gossip_channels",
                        "getroutes",
                    )
                },
                "calls": calls_out,
                "expected_layer_lifecycle": {
                    "creates": [_norm_layer(n, mapping) for n in stub.creates],
                    "removes": [_norm_layer(n, mapping) for n in stub.removes],
                    "updates": {
                        _norm_layer(k, mapping): v
                        for k, v in stub.updates.items()
                    },
                    "listlayers_calls": stub.listlayers_calls,
                },
            }
        )

    # --- error-translation table (real _translate_getroutes_error) ---
    error_inputs = [
        f"Unknown source node {_PEER_A}",
        f"Unknown destination node {_PEER_B}",
        "Unknown layer 'rebalance-exclude-1-2'",
        "askrene: child died with signal 11",
        "askrene: failed to fork child",
        "askrene: child produced no output",
        "askrene: failed to create pipes: EMFILE",
        "We could not find a usable set of paths",
        "RPC timeout after 45s on getroutes",
        "",
    ]
    error_translation = []
    for msg in error_inputs:
        reason, detail = _translate_getroutes_error(msg)
        error_translation.append(
            {"input": msg, "reason": reason, "detail": detail}
        )

    # --- configured-layer-names normalization table ---
    layer_config_inputs = [
        None,
        "",
        "   ",
        "standalone",
        "STANDALONE",
        "none",
        "off",
        "Disabled",
        "false",
        "0",
        "xpay",
        "xpay,bimodal",
        " a , b ,, c ",
    ]
    layer_config = [
        {"raw": raw, "expected": _configured_layer_names(raw)}
        for raw in layer_config_inputs
    ]

    # --- orphan sweep (real RebalanceEngine._sweep_orphan_exclude_layers) ---
    class _SweepHost:
        _data_service = None

        def __init__(self, rpc):
            self.plugin = _StubPlugin(rpc)

        def _log(self, msg, level="debug"):
            pass

    class _SweepRpc:
        def __init__(self, layers, fail_list=False, fail_remove=()):
            self._layers = layers
            self._fail_list = fail_list
            self._fail_remove = set(fail_remove)
            self.removes = []

        def call(self, method, params):
            if method == "askrene-listlayers":
                if self._fail_list:
                    raise Exception("askrene not loaded")
                return {"layers": [{"layer": n} for n in self._layers]}
            if method == "askrene-remove-layer":
                name = params["layer"]
                self.removes.append(name)
                if name in self._fail_remove:
                    raise Exception("Unknown layer")
                return {}
            raise AssertionError(method)

    sweep_inputs = [
        {
            "name": "removes_only_prefix",
            "live_layers": [
                "rebalance-exclude-1700000000-1",
                "xpay",
                "auto.no_mpp_support",
                "rebalance-exclude-1700000099-7",
                "my-rebalance-exclude-not-prefix",
            ],
            "fail_list": False,
            "fail_remove": [],
        },
        {
            "name": "listlayers_failure_returns_zero",
            "live_layers": [],
            "fail_list": True,
            "fail_remove": [],
        },
        {
            "name": "remove_failure_still_counted",
            "live_layers": [
                "rebalance-exclude-1700000000-1",
                "rebalance-exclude-1700000000-2",
            ],
            "fail_list": False,
            "fail_remove": ["rebalance-exclude-1700000000-1"],
        },
        {
            "name": "no_orphans",
            "live_layers": ["xpay"],
            "fail_list": False,
            "fail_remove": [],
        },
    ]
    sweep_cases = []
    for si in sweep_inputs:
        rpc = _SweepRpc(
            si["live_layers"],
            fail_list=si["fail_list"],
            fail_remove=si["fail_remove"],
        )
        count = RebalanceEngine._sweep_orphan_exclude_layers(_SweepHost(rpc))
        sweep_cases.append(
            {
                "name": si["name"],
                "live_layers": si["live_layers"],
                "fail_list": si["fail_list"],
                "fail_remove": si["fail_remove"],
                "expected_removed_count": count,
                "expected_remove_attempts": rpc.removes,
            }
        )

    return {
        "getroutes_rpc_timeout_seconds": GETROUTES_RPC_TIMEOUT_SECONDS,
        "cases": out_cases,
        "error_translation": error_translation,
        "configured_layer_names": layer_config,
        "orphan_sweep": sweep_cases,
    }


# ---------------------------------------------------------------------------
# executor suite (Phase 5 Task 5) — NativeRouteExecutor golden parity.
#
# Drives the REAL modules.rebalance_native_executor_v2.NativeRouteExecutor
# with a scripted payment-RPC double over every contract point: each strict
# route-validation failure expressible through the Rust typed SendpayHop seam
# (hop dicts always carry id/channel/amount_msat/delay as an int/str, so the
# Python-only hop_N_not_object / hop_N_missing_<field> branches are
# unreachable in the port and deliberately not fixtured), success each
# amount_sent shape, payment-pending each way (waitsendpay code 200, proxy
# timeout, waitsendpay_status=pending), terminal failures with/without
# erring_channel attribution, malformed invoice responses (NX-4), and the
# confidence-weighted segment observations (0.85 attributed, /2 when the
# direction is unknown, 0.85/n with a 0.2 floor when inferred).
#
# Determinism: time.time is patched per-case to an integer number of seconds
# (case_now_s), so the invoice label is exactly
# `rebal-native-{case_now_s*1000}-{dest_scid}` and segment observations get
# observed_at == case_now_s. The Rust ExecuteRequest carries now_ms =
# case_now_s*1000 and derives observed_at = now_ms / 1000.
#
# RPC error encoding (the seam convention the Rust PaymentRpc doubles and
# the future live impl must follow): a structured CLN RPC error (pyln
# RpcError.error dict) is carried to Rust as RpcFailure{ message:
# json.dumps(error_dict) } — the executor parses the message as JSON to
# recover code/message/data exactly like Python's _error_details reads
# exc.error. A plain-text failure stays plain text. Proxy timeouts are
# plain-text messages containing "rpc timeout" (Python detects the
# RPCTimeoutError CLASS first, then falls back to the same substring; the
# generator asserts every scripted timeout_error message contains the
# substring so both detectors agree).
# ---------------------------------------------------------------------------

_NX_OUR_ID = "020000000000000000000000000000000000000000000000000000000000000001"
_NX_PEER_SRC = "02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa02"
_NX_PEER_M1 = "03bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb03"
_NX_PEER_M2 = "03cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc04"
_NX_PEER_M3 = "03dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd05"
_NX_PEER_DST = "02eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee06"
_NX_SRC_SCID = "800000x1000x0"
_NX_DST_SCID = "800000x2000x1"
_NX_M1 = "810000x10x0"
_NX_M2 = "810000x20x1"
_NX_M3 = "810000x30x0"
_NX_NOW_BASE_S = 1750000000


class _NxRpcError(Exception):
    """Scripted stand-in for pyln RpcError: carries .error like the real one."""

    def __init__(self, error):
        super().__init__(str(error.get("message") or error))
        self.error = error


class RPCTimeoutError(Exception):
    """Name matters: NativeRouteExecutor._is_proxy_timeout matches the MRO
    class name (the main plugin's RPCTimeoutError without importing it)."""


class _NxPaymentRpc:
    """Scripted payment-RPC double with a normalized call log.

    Script specs per method:
      {"result": v}        -> return v (a dict)
      {"raw": v}           -> return v verbatim (non-dict, NX-4 shapes)
      {"error": obj}       -> raise _NxRpcError(obj)   (structured CLN error)
      {"error_text": s}    -> raise RuntimeError(s)    (plain-text failure)
      {"timeout_error": s} -> raise RPCTimeoutError(s) (proxy deadline)
    """

    def __init__(self, script):
        self.script = dict(script or {})
        self.calls = []

    def call(self, method, params, timeout=None):
        rec = {"method": method, "params": params, "timeout_kwarg": timeout}
        self.calls.append(rec)
        spec = self.script.get(method)
        if spec is None:
            raise AssertionError(f"unscripted RPC method: {method}")
        if "result" in spec:
            return spec["result"]
        if "raw" in spec:
            return spec["raw"]
        if "error" in spec:
            err = spec["error"]
            assert err.get("message"), (
                "seam convention: structured errors always carry a non-empty "
                f"'message' (got {err!r})"
            )
            raise _NxRpcError(err)
        if "error_text" in spec:
            raise RuntimeError(spec["error_text"])
        if "timeout_error" in spec:
            msg = spec["timeout_error"]
            assert "rpc timeout" in msg.lower(), (
                "seam convention: proxy-timeout messages must contain "
                f"'rpc timeout' so the Rust substring detector agrees ({msg!r})"
            )
            raise RPCTimeoutError(msg)
        raise AssertionError(f"bad script spec for {method}: {spec}")


def _nx_hop(node_id, channel, direction, delay, amount_msat):
    return {
        "id": node_id,
        "channel": channel,
        "direction": direction,
        "delay": delay,
        "amount_msat": amount_msat,
        "style": "tlv",
    }


def _nx_route_4hop(amount_sats=100_000):
    """us -> SRC peer -> M1 -> M2 -> us; planned fee 1500 msat."""
    amt = amount_sats * 1000
    return [
        _nx_hop(_NX_PEER_SRC, _NX_SRC_SCID, 0, 120, amt + 1500),
        _nx_hop(_NX_PEER_M1, _NX_M1, 1, 100, amt + 900),
        _nx_hop(_NX_PEER_M2, _NX_M2, 0, 80, amt + 300),
        _nx_hop(_NX_OUR_ID, _NX_DST_SCID, 1, 18, amt),
    ]


def _nx_invoice_result(case_i, with_bolt11=True, with_secret=True):
    out = {"payment_hash": f"ph{case_i:02d}" + "00" * 30}
    if with_bolt11:
        out["bolt11"] = f"lnbc1case{case_i:02d}"
    if with_secret:
        out["payment_secret"] = f"sec{case_i:02d}" + "11" * 30
    return out


def _nx_normalize_calls(calls, src, dst, amount_sats):
    """Reduce raw plugin.rpc.call records to the seam-visible argument set
    (the shape the Rust ScriptedPaymentRpc logs and the test compares)."""
    out = []
    for rec in calls:
        method, params = rec["method"], rec["params"]
        if method == "invoice":
            assert params["description"] == (
                f"cl_revenue_ops rebalance {src}->{dst}"
            ), params
            out.append(
                {
                    "method": "invoice",
                    "amount_msat": params["amount_msat"],
                    "label": params["label"],
                    "expiry": params["expiry"],
                }
            )
        elif method == "sendpay":
            # The frozen Rust seam has no amount_msat parameter: the live
            # impl derives it from the final hop, which validation pinned
            # to amount_sats*1000. Assert that derivation is sound.
            assert params["amount_msat"] == amount_sats * 1000
            assert params["route"][-1]["amount_msat"] == params["amount_msat"]
            out.append(
                {
                    "method": "sendpay",
                    "route": params["route"],
                    "payment_hash": params["payment_hash"],
                    "bolt11": params.get("bolt11", ""),
                    "payment_secret": params.get("payment_secret", ""),
                }
            )
        elif method == "waitsendpay":
            assert rec["timeout_kwarg"] == params["timeout"] == 60
            out.append(
                {
                    "method": "waitsendpay",
                    "payment_hash": params["payment_hash"],
                    "timeout": params["timeout"],
                }
            )
        elif method == "delpay":
            out.append(
                {
                    "method": "delpay",
                    "payment_hash": params["payment_hash"],
                    "status": params["status"],
                }
            )
        elif method == "delinvoice":
            out.append(
                {
                    "method": "delinvoice",
                    "label": params["label"],
                    "status": params["status"],
                }
            )
        else:
            raise AssertionError(f"unexpected RPC method {method}")
    return out


def _nx_case_inputs():
    """Input-only case table; expected outputs come from the real Python."""
    r4 = _nx_route_4hop
    wire_temp = "WIRE_TEMPORARY_CHANNEL_FAILURE: temporary_channel_failure"
    cases = []

    def add(name, route, script=None, amount_sats=100_000, max_fee_sats=10,
            source=_NX_SRC_SCID, dest=_NX_DST_SCID, invoice_i=None,
            expect_no_rpc=False):
        cases.append(
            {
                "name": name,
                "route": route,
                "script": script or {},
                "amount_sats": amount_sats,
                "max_fee_sats": max_fee_sats,
                "source_channel_id": source,
                "dest_channel_id": dest,
                "expect_no_rpc": expect_no_rpc,
            }
        )

    # --- strict validation failures (fail before ANY RPC) ---
    add("validation_invalid_amount_zero", r4(), amount_sats=0,
        expect_no_rpc=True)
    add("validation_invalid_amount_negative", r4(), amount_sats=-5,
        expect_no_rpc=True)
    add("validation_missing_route", [], expect_no_rpc=True)
    add("validation_first_hop_not_source_channel", r4(),
        source="999999x1x0", expect_no_rpc=True)
    add("validation_final_hop_not_dest_channel", r4(),
        dest="999999x2x1", expect_no_rpc=True)
    bad_final_node = r4()
    bad_final_node[-1]["id"] = _NX_PEER_M2
    add("validation_final_hop_not_our_node", bad_final_node,
        expect_no_rpc=True)
    bad_final_amt = r4()
    bad_final_amt[-1]["amount_msat"] = 100_000_000 - 1
    add("validation_final_amount_mismatch", bad_final_amt,
        expect_no_rpc=True)
    zero_first = r4()
    zero_first[0]["amount_msat"] = 0
    zero_first[-1]["amount_msat"] = 100_000_000
    add("validation_hop_0_invalid_amount", zero_first, expect_no_rpc=True)
    increasing = r4()
    increasing[2]["amount_msat"] = increasing[1]["amount_msat"] + 1
    add("validation_increasing_route_amount", increasing, expect_no_rpc=True)
    add("validation_over_budget", r4(), max_fee_sats=1, expect_no_rpc=True)

    # --- success shapes ---
    add("success_amount_sent_int", r4(), {
        "invoice": {"result": _nx_invoice_result(10)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"result": {
            "status": "complete", "amount_sent_msat": 100_001_500,
        }},
    })
    add("success_amount_sent_msat_string", r4(), {
        "invoice": {"result": _nx_invoice_result(11, with_secret=False)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"result": {
            "status": "complete", "amount_sent_msat": "100001500msat",
        }},
    })
    add("success_missing_amount_sent_falls_back_to_first_hop", r4(), {
        "invoice": {"result": _nx_invoice_result(12, with_bolt11=False,
                                                 with_secret=False)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"result": {"status": "complete"}},
    })
    add("success_status_missing_treated_complete", r4(), {
        "invoice": {"result": _nx_invoice_result(13)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"result": {"amount_sent_msat": 100_000_600}},
    })

    # --- malformed invoice responses (NX-4): terminal + best-effort delinvoice ---
    add("invoice_response_not_dict", r4(), {
        "invoice": {"raw": "garbage"},
        "delinvoice": {"result": {}},
    })
    add("invoice_response_missing_payment_hash", r4(), {
        "invoice": {"result": {"bolt11": "lnbc1nohash"}},
        "delinvoice": {"result": {}},
    })

    # --- payment_pending: abandon, never cancel ---
    add("pending_waitsendpay_code_200", r4(), {
        "invoice": {"result": _nx_invoice_result(16)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error": {
            "code": 200,
            "message": "Timed out while waiting for payment",
        }},
    })
    add("pending_proxy_timeout_on_sendpay", r4(), {
        "invoice": {"result": _nx_invoice_result(17)},
        "sendpay": {"timeout_error": "RPC timeout after 60.0s (sendpay)"},
    })
    add("pending_proxy_timeout_on_waitsendpay", r4(), {
        "invoice": {"result": _nx_invoice_result(18)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"timeout_error":
                        "RPC timeout after 60.0s (waitsendpay)"},
    })
    add("pending_waitsendpay_status_pending", r4(), {
        "invoice": {"result": _nx_invoice_result(19)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"result": {"status": "pending"}},
    })

    # --- terminal sendpay failures ---
    add("terminal_erring_channel_and_direction", r4(), {
        "invoice": {"result": _nx_invoice_result(20)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error": {
            "code": 204,
            "message": f"failed: {wire_temp} (reply from remote)",
            "data": {
                "erring_channel": _NX_M2,
                "erring_direction": 0,
                "erring_node": _NX_PEER_M1,
                "erring_index": 2,
                "failcode": 4103,
                "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
            },
        }},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    add("terminal_erring_direction_string", r4(), {
        "invoice": {"result": _nx_invoice_result(21)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error": {
            "code": 204,
            "message": f"failed: {wire_temp}",
            "data": {"erring_channel": _NX_M1, "erring_direction": "1"},
        }},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    add("terminal_erring_channel_unknown_direction", r4(), {
        "invoice": {"result": _nx_invoice_result(22)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error": {
            "code": 204,
            "message": f"failed: {wire_temp}",
            "data": {"erring_channel": _NX_M1, "failcode": 4103},
        }},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    add("terminal_liquidity_no_attribution_middle_hops", r4(), {
        "invoice": {"result": _nx_invoice_result(23)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error_text": wire_temp},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    # 5-hop route with a duplicated middle (scid, dir): dedup -> n=2.
    amt = 100_000 * 1000
    dup_middle = [
        _nx_hop(_NX_PEER_SRC, _NX_SRC_SCID, 0, 140, amt + 2000),
        _nx_hop(_NX_PEER_M1, _NX_M1, 1, 120, amt + 1200),
        _nx_hop(_NX_PEER_M1, _NX_M1, 1, 100, amt + 700),
        _nx_hop(_NX_PEER_M2, _NX_M2, 0, 80, amt + 300),
        _nx_hop(_NX_OUR_ID, _NX_DST_SCID, 1, 18, amt),
    ]
    add("terminal_fee_no_attribution_dedups_middle_hops", dup_middle, {
        "invoice": {"result": _nx_invoice_result(24)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error": {
            "code": 204,
            "message": "failed: WIRE_FEE_INSUFFICIENT: fee_insufficient",
        }},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    # 8-hop route, 6 distinct middles: 0.85/6 ~ 0.1417 -> floor 0.2.
    floor_route = [_nx_hop(_NX_PEER_SRC, _NX_SRC_SCID, 0, 220, amt + 6000)]
    middles = [
        (_NX_PEER_M1, _NX_M1, 1), (_NX_PEER_M2, _NX_M2, 0),
        (_NX_PEER_M3, _NX_M3, 0), (_NX_PEER_M1, "820000x40x1", 1),
        (_NX_PEER_M2, "820000x50x0", 0), (_NX_PEER_M3, "820000x60x1", 1),
    ]
    for i, (pid, scid, d) in enumerate(middles):
        floor_route.append(
            _nx_hop(pid, scid, d, 200 - i * 20, amt + 5000 - i * 800)
        )
    floor_route.append(_nx_hop(_NX_OUR_ID, _NX_DST_SCID, 1, 18, amt))
    add("terminal_inferred_confidence_floor", floor_route, {
        "invoice": {"result": _nx_invoice_result(25)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error_text": wire_temp},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    add("terminal_waitsendpay_status_failed_unknown_class", r4(), {
        "invoice": {"result": _nx_invoice_result(26)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"result": {"status": "failed"}},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    add("terminal_timeout_class_not_proxy_timeout", r4(), {
        "invoice": {"result": _nx_invoice_result(27)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error_text":
                        "deadline exceeded waiting for htlc resolution"},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    add("terminal_invoice_rpc_error", r4(), {
        "invoice": {"error": {"code": -32602, "message": "Invalid label"}},
        "delinvoice": {"result": {}},
    })
    add("terminal_invoice_code_200_not_pending", r4(), {
        "invoice": {"error": {
            "code": 200, "message": "Timed out creating invoice",
        }},
        "delinvoice": {"result": {}},
    })
    add("terminal_sendpay_structured_error", r4(), {
        "invoice": {"result": _nx_invoice_result(30)},
        "sendpay": {"error": {
            "code": 204,
            "message": f"failed: {wire_temp}",
            "data": {"erring_channel": _NX_M1, "erring_direction": 1},
        }},
        "delpay": {"result": {}},
        "delinvoice": {"result": {}},
    })
    add("terminal_cleanup_failures_swallowed", r4(), {
        "invoice": {"result": _nx_invoice_result(31)},
        "sendpay": {"result": {"status": "pending"}},
        "waitsendpay": {"error": {
            "code": 204,
            "message": f"failed: {wire_temp}",
            "data": {"erring_channel": _NX_M2, "erring_direction": 0},
        }},
        "delpay": {"error": {"message": "Payment with hash not found"}},
        "delinvoice": {"error": {"message": "Invoice not found"}},
    })
    return cases


def gen_executor() -> dict:
    """Drive the REAL NativeRouteExecutor over a scripted payment-RPC double.

    Feeds `crates/revops-rebalance/src/executor.rs` /
    `crates/revops-rebalance/tests/executor.rs` (Phase 5 Task 5), committed
    in the Rust repo as `fixtures/rebalance/executor.json`.
    """
    import copy
    import dataclasses
    import time as _time

    from modules.rebalance_native_executor_v2 import NativeRouteExecutor

    assert NativeRouteExecutor.ATTRIBUTED_CONFIDENCE == 0.85
    assert NativeRouteExecutor.INFERRED_CONFIDENCE_FLOOR == 0.2
    assert NativeRouteExecutor.INVOICE_EXPIRY_SEC == 300
    assert NativeRouteExecutor.SENDPAY_TIMEOUT_SEC == 60

    real_time = _time.time
    out_cases = []
    try:
        for i, case in enumerate(_nx_case_inputs()):
            now_s = _NX_NOW_BASE_S + i
            _time.time = lambda now_s=now_s: float(now_s)

            rpc = _NxPaymentRpc(case["script"])
            store = SegmentObservationStore()
            executor = NativeRouteExecutor(
                _StubPlugin(rpc),
                observation_store=store,
                our_id=_NX_OUR_ID,
            )
            route = copy.deepcopy(case["route"])
            result = executor.execute(
                route,
                case["amount_sats"],
                case["source_channel_id"],
                case["dest_channel_id"],
                case["max_fee_sats"],
            )
            if case["expect_no_rpc"]:
                assert rpc.calls == [], (case["name"], rpc.calls)

            expected = dataclasses.asdict(result)
            if expected["error"] == "":
                expected["error"] = None
            pending_hash = None
            if result.payment_pending:
                pending_hash = result.failure_data.get("payment_hash")
            snap = store.export_snapshot(
                observer_member_id="gen", now=now_s
            )
            out_cases.append(
                {
                    "name": case["name"],
                    "now_ms": now_s * 1000,
                    "request": {
                        "route": case["route"],
                        "amount_sats": case["amount_sats"],
                        "source_channel_id": case["source_channel_id"],
                        "dest_channel_id": case["dest_channel_id"],
                        "max_fee_sats": case["max_fee_sats"],
                        "our_id": _NX_OUR_ID,
                    },
                    "rpc_script": case["script"],
                    "expected": expected,
                    "expected_payment_hash": pending_hash,
                    "expected_calls": _nx_normalize_calls(
                        rpc.calls,
                        case["source_channel_id"],
                        case["dest_channel_id"],
                        case["amount_sats"],
                    ),
                    "expected_observations": snap["segment_observations"],
                }
            )
    finally:
        _time.time = real_time

    return {
        "invoice_expiry_sec": NativeRouteExecutor.INVOICE_EXPIRY_SEC,
        "sendpay_timeout_sec": NativeRouteExecutor.SENDPAY_TIMEOUT_SEC,
        "attributed_confidence": NativeRouteExecutor.ATTRIBUTED_CONFIDENCE,
        "inferred_confidence_floor":
            NativeRouteExecutor.INFERRED_CONFIDENCE_FLOOR,
        "cases": out_cases,
    }


# ---------------------------------------------------------------------------
# ev suite (Phase 5 Task 6) — sats-EV gate, per-attempt ceiling, fee
# escalation.
# ---------------------------------------------------------------------------


def _ev_host_class():
    """Build a stub `self` bound to the REAL `RebalanceEngine` instance
    methods the sats-EV math needs, per the plan's "drive the engine method
    with a stub self" technique (same as `gen_router`'s `_SweepHost`).

    Only state the real methods actually read is stubbed:
    `_pair_failures` (empty — see `gen_ev`'s scope-note docstring) and
    `_dest_success_rate_memo` (per-cycle memo, always empty here since each
    case is a fresh call).
    """
    from modules.rebalance_engine_v2 import RebalanceEngine

    class _EvHost:
        def __init__(self, cfg, database=None):
            self.config = cfg
            self.database = database
            self._dest_success_rate_memo = {}
            self._pair_failures = {}
            self._futility_window_sec = 1800.0

    _EvHost._pair_key = staticmethod(RebalanceEngine._pair_key)
    _EvHost._prune_pair_failures = RebalanceEngine._prune_pair_failures
    _EvHost._failure_count = RebalanceEngine._failure_count
    _EvHost._failure_penalty = RebalanceEngine._failure_penalty
    _EvHost._empirical_dest_success_rate = RebalanceEngine._empirical_dest_success_rate
    _EvHost._query_dest_success_rate = RebalanceEngine._query_dest_success_rate
    return _EvHost


class _EvStubDatabase:
    """Backs `_query_dest_success_rate`: `total`/`success_rate` mirror
    exactly the dict shape `get_channel_rebalance_success_rate` returns."""

    def __init__(self, total, success_rate):
        self._total = total
        self._success_rate = success_rate

    def get_channel_rebalance_success_rate(self, dest_channel_id):
        return {"total": self._total, "success_rate": self._success_rate}


def _ev_default_cfg(**overrides):
    from types import SimpleNamespace

    base = dict(
        high_liquidity_threshold=0.65,
        low_liquidity_threshold=0.35,
        max_fee_ppm=500,
        rebalance_activity_penalty_coeff=0.5,
        rebalance_activity_penalty_cap_frac=0.5,
        rebalance_hold_margin=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _ev_default_pair(**overrides):
    from types import SimpleNamespace
    from modules.rebalance_engine_v2 import EXPECTED_UTILIZATION

    base = dict(
        source_channel_id="111x1x0",
        dest_channel_id="222x2x0",
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        amount_sats=100_000,
        source_local_ratio=0.7,
        dest_local_ratio=0.3,
        dest_out_fee_ppm=200,
        source_out_fee_ppm=150,
        source_historical_direct_fee_ppm=0.0,
        source_historical_sourced_fee_ppm=0.0,
        dest_historical_direct_fee_ppm=0.0,
        dest_historical_sourced_fee_ppm=0.0,
        dest_fee_history_validated=False,
        dest_realized_utilization=EXPECTED_UTILIZATION,
        dest_utilization_is_realized=False,
        source_realized_utilization=EXPECTED_UTILIZATION,
        source_utilization_is_realized=False,
        source_activity_out_sats=0,
        dest_activity_in_sats=0,
        score=0.0,
        pair_budget_sats=5_000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _call_capturing_locals(func, *args, **kwargs):
    """Call `func`, returning `(result, locals_at_return)`.

    `_build_score_decomposition`'s OWN `final_score_sats` computation
    (`rebalance_engine_v2.py:556-563`) uses the RAW (full double-precision,
    pre-round) `expected_future_value_sats`/`source_opportunity_sats`
    locals — the dict it returns only exposes `round(x, 6)`'d COPIES of
    those same names for display. Feeding the Rust port the rounded display
    copies as its `EvInputs.efv_sats`/`source_opportunity_sats` re-rounds an
    already-lossy number and occasionally disagrees with the real function
    in the 6th decimal after `p_success` multiplies through the rounding
    error. `sys.settrace` on the function's `return` event recovers the
    exact pre-round locals without reimplementing any of the math — still
    driving the real function, just reading its stack frame instead of its
    return value for these two terms.
    """
    import sys

    captured = {}
    target_code = func.__code__

    def local_tracer(frame, event, arg):
        if event == "return":
            captured.update(frame.f_locals)
        return local_tracer

    def global_tracer(frame, event, arg):
        if event == "call" and frame.f_code is target_code:
            return local_tracer
        return global_tracer

    old_trace = sys.gettrace()
    sys.settrace(global_tracer)
    try:
        result = func(*args, **kwargs)
    finally:
        sys.settrace(old_trace)
    return result, captured


def _drive_ev_gate_case(
    case_id,
    *,
    probability_ppm,
    dest_attempts,
    dest_success_rate,
    route_cost_sats,
    effective_budget_sats,
    hold_margin_sats,
    cfg_overrides=None,
    pair_overrides=None,
):
    """Drive the REAL `_build_score_decomposition`, then apply the
    hold-margin gate transcribed verbatim from `find_candidates`
    (`rebalance_engine_v2.py:1468-1472`) — both operands of that boolean
    (`final_score_sats`, `expected_fee_sats`) are themselves oracle-derived
    just above, so this is a direct re-application of the exact inline
    condition, not a reimplementation of new math.

    `efv_sats`/`source_opportunity_sats` are the RAW pre-round locals
    (see `_call_capturing_locals`'s docstring), not the rounded dict
    values — this is what the real `final_score_sats` formula actually
    consumes.
    """
    from modules.rebalance_engine_v2 import RebalanceEngine

    host_cls = _ev_host_class()
    cfg = _ev_default_cfg(**(cfg_overrides or {}))
    pair = _ev_default_pair(**(pair_overrides or {}))
    database = _EvStubDatabase(dest_attempts, dest_success_rate)
    host = host_cls(cfg, database=database)

    decomp, raw_locals = _call_capturing_locals(
        RebalanceEngine._build_score_decomposition,
        host,
        pair,
        probability_ppm=probability_ppm,
        route_cost_sats=route_cost_sats,
        effective_budget_sats=effective_budget_sats,
    )
    efv_sats = raw_locals["expected_future_value_sats"]
    fee_sats = decomp["expected_fee_sats"]
    source_opportunity_sats = raw_locals["source_opportunity_sats"]
    activity_penalty_sats = decomp["activity_penalty_sats"]
    final_score_sats = decomp["final_score_sats"]
    rejected = fee_sats > 0 and final_score_sats < hold_margin_sats

    return {
        "case_id": case_id,
        "inputs": {
            "probability_ppm": probability_ppm,
            "dest_attempts": dest_attempts,
            "dest_success_rate": dest_success_rate,
            "efv_sats": efv_sats,
            "fee_sats": fee_sats,
            "source_opportunity_sats": source_opportunity_sats,
            "activity_penalty_sats": activity_penalty_sats,
            "hold_margin_sats": hold_margin_sats,
        },
        "expected": {
            "pass": not rejected,
            "final_score_sats": final_score_sats,
            "reject_reason": "below_hold_margin" if rejected else None,
        },
    }


def _ev_hand_built_gate_cases():
    cases = []

    # Zero-cost routes ALWAYS pass, however negative the sats score (the
    # zero-budget equalization invariant) — a huge hold_margin would reject
    # any priced route but must not touch a free one.
    cases.append(
        _drive_ev_gate_case(
            "zero_cost_route_always_passes",
            probability_ppm=100_000,
            dest_attempts=0,
            dest_success_rate=0.0,
            route_cost_sats=0,
            effective_budget_sats=5_000,
            hold_margin_sats=1_000.0,
            pair_overrides=dict(dest_out_fee_ppm=0, source_out_fee_ppm=2_000),
        )
    )
    cases.append(
        _drive_ev_gate_case(
            "zero_cost_route_negative_efv_still_passes",
            probability_ppm=50_000,
            dest_attempts=0,
            dest_success_rate=0.0,
            route_cost_sats=0,
            effective_budget_sats=5_000,
            hold_margin_sats=50.0,
            pair_overrides=dict(dest_out_fee_ppm=0, source_out_fee_ppm=5_000, amount_sats=500_000),
        )
    )

    # Prior-blend boundary: <3 dest attempts keeps the flat 0.5 prior; >=3
    # blends in the empirical rate (`_query_dest_success_rate`'s own `total
    # < 3` guard).
    for attempts, cid in ((2, "prior_blend_boundary_below_3"), (3, "prior_blend_boundary_at_3")):
        cases.append(
            _drive_ev_gate_case(
                cid,
                probability_ppm=0,
                dest_attempts=attempts,
                dest_success_rate=0.8,
                route_cost_sats=40,
                effective_budget_sats=5_000,
                hold_margin_sats=0.0,
            )
        )

    # Empirical blend rate range extremes (rate clamped to [0,1] upstream by
    # `_query_dest_success_rate`, so 0.5*(0.5+rate) never itself breaches the
    # [0.05, 0.95] clamp — these pin the two range extremes anyway).
    for rate, cid in ((0.0, "blend_rate_zero"), (1.0, "blend_rate_one")):
        cases.append(
            _drive_ev_gate_case(
                cid,
                probability_ppm=0,
                dest_attempts=5,
                dest_success_rate=rate,
                route_cost_sats=40,
                effective_budget_sats=5_000,
                hold_margin_sats=0.0,
            )
        )

    # probability_ppm clamp edges: [0.05, 0.99].
    for ppm, cid in (
        (1, "clamp_low_probability"),
        (50_000, "clamp_low_boundary_exact_0_05"),
        (990_000, "clamp_high_boundary_exact_0_99"),
        (999_999, "clamp_high_probability"),
    ):
        cases.append(
            _drive_ev_gate_case(
                cid,
                probability_ppm=ppm,
                dest_attempts=0,
                dest_success_rate=0.0,
                route_cost_sats=40,
                effective_budget_sats=5_000,
                hold_margin_sats=0.0,
            )
        )

    # Hold-margin exact tie: `final_score < hold_margin` is a STRICT
    # inequality, so an exact tie must still PASS.
    probe = _drive_ev_gate_case(
        "hold_margin_tie_probe",
        probability_ppm=500_000,
        dest_attempts=0,
        dest_success_rate=0.0,
        route_cost_sats=50,
        effective_budget_sats=5_000,
        hold_margin_sats=0.0,
    )
    tie_margin = probe["expected"]["final_score_sats"]
    cases.append(
        _drive_ev_gate_case(
            "hold_margin_exact_tie_passes",
            probability_ppm=500_000,
            dest_attempts=0,
            dest_success_rate=0.0,
            route_cost_sats=50,
            effective_budget_sats=5_000,
            hold_margin_sats=tie_margin,
        )
    )
    cases.append(
        _drive_ev_gate_case(
            "hold_margin_just_above_rejects",
            probability_ppm=500_000,
            dest_attempts=0,
            dest_success_rate=0.0,
            route_cost_sats=50,
            effective_budget_sats=5_000,
            hold_margin_sats=tie_margin + 1e-6,
        )
    )

    # Activity-penalty cap binds: a large helpful_flow drives raw penalty
    # past `activity_cap_frac * expected_future_value_sats`.
    cases.append(
        _drive_ev_gate_case(
            "activity_penalty_cap_binds",
            probability_ppm=500_000,
            dest_attempts=0,
            dest_success_rate=0.0,
            route_cost_sats=10,
            effective_budget_sats=5_000,
            hold_margin_sats=0.0,
            pair_overrides=dict(source_activity_out_sats=1_000_000, dest_activity_in_sats=1_000_000),
        )
    )

    # Validated dest fee history uses the direct realized ppm outright
    # (no 0.5 discount) instead of the discounted advertised fee.
    cases.append(
        _drive_ev_gate_case(
            "dest_fee_validated_uses_direct_ppm",
            probability_ppm=500_000,
            dest_attempts=0,
            dest_success_rate=0.0,
            route_cost_sats=10,
            effective_budget_sats=5_000,
            hold_margin_sats=0.0,
            pair_overrides=dict(
                dest_fee_history_validated=True,
                dest_historical_direct_fee_ppm=300.0,
                dest_out_fee_ppm=1_000,
            ),
        )
    )

    # A zero-configured max_fee_ppm zeroes every historical-fee-derived term
    # (P4-012: no fee headroom means no historical EV benefit either).
    cases.append(
        _drive_ev_gate_case(
            "historical_fee_cap_zero",
            probability_ppm=500_000,
            dest_attempts=0,
            dest_success_rate=0.0,
            route_cost_sats=10,
            effective_budget_sats=5_000,
            hold_margin_sats=0.0,
            cfg_overrides=dict(max_fee_ppm=0),
            pair_overrides=dict(
                dest_historical_direct_fee_ppm=900.0,
                source_historical_direct_fee_ppm=900.0,
                source_historical_sourced_fee_ppm=900.0,
            ),
        )
    )

    return cases


def _ev_random_gate_case(rng, i):
    probability_ppm = 0 if rng.random() < 0.3 else rng.randint(1, 1_000_000)
    dest_attempts = rng.randint(0, 10)
    dest_success_rate = round(rng.uniform(0.0, 1.0), 6)
    route_cost_sats = 0 if rng.random() < 0.1 else rng.randint(1, 20_000)
    effective_budget_sats = route_cost_sats + rng.randint(0, 10_000)
    max_fee_ppm = rng.choice([0, 100, 500, 2_000])

    pair_overrides = dict(
        source_capacity_sats=rng.randint(100_000, 5_000_000),
        dest_capacity_sats=rng.randint(100_000, 5_000_000),
        amount_sats=rng.randint(1_000, 500_000),
        source_local_ratio=round(rng.uniform(0.0, 1.0), 6),
        dest_local_ratio=round(rng.uniform(0.0, 1.0), 6),
        dest_out_fee_ppm=rng.randint(0, 2_000),
        source_out_fee_ppm=rng.randint(0, 2_000),
        source_historical_direct_fee_ppm=float(rng.choice([0, rng.randint(0, 2_000)])),
        source_historical_sourced_fee_ppm=float(rng.choice([0, rng.randint(0, 2_000)])),
        dest_historical_direct_fee_ppm=float(rng.choice([0, rng.randint(0, 2_000)])),
        dest_historical_sourced_fee_ppm=float(rng.choice([0, rng.randint(0, 2_000)])),
        dest_fee_history_validated=rng.random() < 0.3,
        dest_utilization_is_realized=rng.random() < 0.3,
        dest_realized_utilization=round(rng.uniform(0.0, 1.0), 6),
        source_utilization_is_realized=rng.random() < 0.3,
        source_realized_utilization=round(rng.uniform(0.0, 1.0), 6),
        source_activity_out_sats=0 if rng.random() < 0.6 else rng.randint(0, 200_000),
        dest_activity_in_sats=0 if rng.random() < 0.6 else rng.randint(0, 200_000),
        pair_budget_sats=rng.randint(0, 20_000),
    )

    # hold_margin_sats is picked AFTER driving the case (below) so it can
    # straddle final_score_sats for good pass/fail/tie coverage; the probe
    # call uses hold_margin_sats=0.0 as a placeholder (the gate math never
    # feeds back into the decomposition).
    probe = _drive_ev_gate_case(
        f"random-{i:04d}",
        probability_ppm=probability_ppm,
        dest_attempts=dest_attempts,
        dest_success_rate=dest_success_rate,
        route_cost_sats=route_cost_sats,
        effective_budget_sats=effective_budget_sats,
        hold_margin_sats=0.0,
        cfg_overrides=dict(max_fee_ppm=max_fee_ppm),
        pair_overrides=pair_overrides,
    )
    final_score_sats = probe["expected"]["final_score_sats"]
    roll = rng.random()
    if roll < 0.15:
        hold_margin_sats = final_score_sats
    elif roll < 0.55:
        hold_margin_sats = round(final_score_sats + rng.uniform(0.000001, 500.0), 6)
    else:
        hold_margin_sats = round(final_score_sats - rng.uniform(0.000001, 500.0), 6)

    return _drive_ev_gate_case(
        f"random-{i:04d}",
        probability_ppm=probability_ppm,
        dest_attempts=dest_attempts,
        dest_success_rate=dest_success_rate,
        route_cost_sats=route_cost_sats,
        effective_budget_sats=effective_budget_sats,
        hold_margin_sats=hold_margin_sats,
        cfg_overrides=dict(max_fee_ppm=max_fee_ppm),
        pair_overrides=pair_overrides,
    )


def _gen_per_attempt_ceiling_cases():
    from modules.rebalance_engine_v2 import RebalanceEngine
    from types import SimpleNamespace

    cases = []
    grid = [
        # (prob_adjusted_budget_sats, amount_sats, pair_fee_cap_ppm, name)
        (5_000, 100_000, 1_000, "ceiling_binds"),
        (100, 100_000, 1_000, "budget_binds"),
        (5_000, 100_000, 0, "ppm_disabled_budget_only"),
        (5_000, 0, 1_000, "amount_zero_disabled"),
        (5_000, -10, 1_000, "amount_negative_disabled"),
        (0, 100_000, 1_000, "zero_budget_stays_zero"),
        (5_000, 1_000_000, 5, "ceil_exact_divisible"),  # 1_000_000*5/1e6=5 exact
        (5_000, 1_000_001, 5, "ceil_rounds_up_by_one"),
        (5_000, 1, 1_000_000, "ceil_tiny_amount"),
        (10_000_000, 2_000_000, 2_500, "large_values"),
    ]
    for budget, amount, ppm, name in grid:
        cfg = SimpleNamespace(pair_fee_cap_ppm=ppm)
        expected = RebalanceEngine._per_attempt_fee_ceiling(None, amount, budget, cfg)
        cases.append(
            {
                "case_id": name,
                "prob_adjusted_budget_sats": budget,
                "amount_sats": amount,
                "pair_fee_cap_ppm": ppm,
                "expected": expected,
            }
        )
    return cases


def _gen_fee_escalation_cases():
    from modules.rebalancer import EVRebalancer

    cases = []
    grid = [
        # (last_attempted_sats, ev_max_sats, name) — driven with fail_count=1
        # and last_attempted>0 (the reduced 2-arg Rust interface's implicit
        # precondition; the fail_count==0/last<=0 short-circuit branches are
        # the CALLER's responsibility per the frozen 2-arg signature — see
        # ev.rs's module doc comment).
        (100, 1_000, "escalates_below_max"),
        (100, 120, "escalates_capped_at_max"),
        (200, 100, "last_already_at_or_above_max"),
        (80, 120, "exact_boundary_last_times_1_5_eq_max"),  # 80*1.5=120
        (81, 120, "truncation_last_times_1_5_not_integer"),  # 81*1.5=121.5->121
        (1, 1_000_000, "tiny_last_attempted"),
        (1_000_000, 1_000_000, "last_equals_max"),
        (0, 1_000, "last_zero_still_driven_with_failcount_1"),
    ]
    for last, ev_max, name in grid:
        expected = EVRebalancer._apply_fee_escalation(ev_max, 1, last)
        cases.append(
            {
                "case_id": name,
                "last_attempted_sats": last,
                "ev_max_sats": ev_max,
                "expected": expected,
            }
        )
    return cases


def gen_ev() -> dict:
    """Drive the REAL sats-EV gate math (Phase 5 Task 6) over ~500
    randomized-but-seeded input vectors plus edge cases, plus the
    per-attempt-ceiling and fee-escalation tables.

    Feeds `crates/revops-rebalance/src/ev.rs` /
    `crates/revops-rebalance/tests/ev.rs` in the Rust port
    (cl-revenue-ops-r), committed there as `fixtures/rebalance/ev.json`.

    Scope note (documented, not an oversight): the Rust `EvInputs` interface
    takes ALREADY-COMPUTED sats terms (`efv_sats`/`source_opportunity_sats`/
    `activity_penalty_sats`) rather than the raw ppm/utilization/activity
    inputs the real `_build_score_decomposition` reduces them from — that
    raw-to-sats reduction (`EXPECTED_UTILIZATION`/
    `SOURCE_UTILIZATION_DISCOUNT`/`UNVALIDATED_ADVERTISED_FEE_DISCOUNT`) is
    exercised HERE, in the generator, by driving the real function with
    synthetic pair/cfg stubs; only the resulting sats terms cross the Rust
    interface boundary. `EvInputs` also carries no `failure_count`: every
    driven case uses a stub host with an EMPTY in-cycle failure history
    (`_pair_failures = {}`), so `failure_penalty_sats` is always exactly 0.0
    and drops out of the real 5-term formula, leaving the reduced 4-term
    formula `ev.rs` implements (`p*efv - fee - source_opp -
    activity_penalty`) byte-identical to the real one for every fixture case
    (failure-count-driven futility breaking is `cooldowns.rs`'s
    `PairFutility`/`DestFutility`, a separate concern with its own
    constants, not fixture-driven — see `cooldowns.rs`'s doc comment).
    """
    rng = random.Random(20260717)
    gate_cases = _ev_hand_built_gate_cases()
    seen_ids = {c["case_id"] for c in gate_cases}
    for i in range(500):
        case = _ev_random_gate_case(rng, i)
        assert case["case_id"] not in seen_ids
        seen_ids.add(case["case_id"])
        gate_cases.append(case)

    return {
        "constants": {
            "expected_utilization": 0.5,
            "source_utilization_discount": 0.5,
            "failure_cost_rate": 0.25,
            "unvalidated_advertised_fee_discount": 0.5,
        },
        "gate_cases": gate_cases,
        "per_attempt_ceiling_cases": _gen_per_attempt_ceiling_cases(),
        "fee_escalation_cases": _gen_fee_escalation_cases(),
    }


# ---------------------------------------------------------------------------
# cooldowns suite (Phase 5 Task 6) — persisted per-kind cooldown backoff,
# fill-fraction dest cooldown (audit F7), drift override.
# ---------------------------------------------------------------------------


def _gen_persisted_cooldown_cases():
    """Drive the REAL `Database.record_pair_rebalance_failure` (SQL-embedded
    backoff math, `database.py:2657-2733`) repeatedly to build up
    `failure_count`, reading `cooldown_until - last_failure_at` back as the
    oracle for `persisted_cooldown_secs(kind, failure_count)`. Base seconds
    per kind transcribed from `rebalance_engine_v2.py:178-186`
    (`_pair_failure_cooldowns`, frozen already at `errors.rs::
    cooldown_base_secs` — Task 1).
    """
    import tempfile

    from modules.database import Database

    class _FakePlugin:
        def log(self, *a, **kw):
            pass

    base_by_kind = {
        "temporary_channel_failure": 300,
        "fee_insufficient": 1800,
        "incorrect_cltv_expiry": 3600,
        "permanent_failure": 21_600,
        "payment_pending_timeout": 3600,
        "local_execution_failed": 600,
        "other_retriable": 600,
    }

    cases = []
    with tempfile.TemporaryDirectory() as td:
        for kind, base in base_by_kind.items():
            db = Database(os.path.join(td, f"cooldown_{kind}.db"), _FakePlugin())
            db.initialize()
            src, dst = "111x1x0", "222x2x0"
            last = None
            for count in range(1, 9):
                last = db.record_pair_rebalance_failure(
                    src, dst, kind, base, now=1_700_000_000
                )
                assert last["failure_count"] == count
                secs = last["cooldown_until"] - last["last_failure_at"]
                cases.append(
                    {
                        "kind": kind,
                        "base_secs": base,
                        "failure_count": count,
                        "expected_secs": secs,
                    }
                )
    return cases


def _gen_dest_cooldown_cases():
    """Drive the REAL `RebalanceEngine._effective_dest_cooldown_secs` (audit
    F7 fill-fraction scaling, `rebalance_engine_v2.py:876-927`) with an
    explicit `anchor` dict, bypassing the DB lookup — the reduced
    `dest_cooldown_secs(base_secs, amount_sats, remaining_band_gap_sats)`
    Rust signature takes the remaining gap directly rather than deriving it
    from `capacity_sats`/`local_sats`/`target_band_low`, so `capacity_sats`/
    `target_band_low` are chosen here purely as an algebraic vehicle:
    `target_band_low=1.0, local_sats=0, capacity_sats=<desired gap>` makes
    `remaining_gap == capacity_sats` exactly (int(1.0 * gap) - 0 == gap).
    """
    from modules.rebalance_engine_v2 import RebalanceEngine

    def _drive(base_secs, amount_sats, remaining_gap_sats):
        if remaining_gap_sats <= 0:
            # Zero/negative gap: capacity_sats must still be > 0 to reach the
            # gap<=0 branch (rather than the capacity_sats<=0 short-circuit).
            capacity_sats = 1_000
            target_band_low = 0.0
        else:
            capacity_sats = remaining_gap_sats
            target_band_low = 1.0
        return RebalanceEngine._effective_dest_cooldown_secs(
            None,
            "111x1x0",
            capacity_sats=capacity_sats,
            local_sats=0,
            base_cooldown_secs=base_secs,
            target_band_low=target_band_low,
            anchor={"amount_sats": amount_sats} if amount_sats > 0 else None,
        )

    cases = []
    grid = [
        # (base_secs, amount_sats, remaining_gap_sats, name)
        (86_400, 500, 0, "zero_gap_full_cooldown"),
        (86_400, 0, 1_000, "zero_amount_no_anchor_amount_full_cooldown"),
        (86_400, 500, 1_000, "partial_fill_scaled"),
        (86_400, 1_000, 1_000, "exact_half_fill_boundary"),  # fraction=0.5 *2=1.0 -> full
        (86_400, 5_000, 1_000, "over_half_fill_full_cooldown"),
        (86_400, 1, 1_000_000, "tiny_fill_short_cooldown"),
        (3_600, 250, 250, "small_base_half_fill"),
        (300, 100, 400, "fill_fraction_one_fifth"),
    ]
    for base_secs, amount_sats, remaining_gap, name in grid:
        expected = _drive(base_secs, amount_sats, remaining_gap)
        cases.append(
            {
                "case_id": name,
                "base_secs": base_secs,
                "amount_sats": amount_sats,
                "remaining_band_gap_sats": remaining_gap,
                "expected_secs": expected,
            }
        )
    # No-anchor case: base_secs unmodified regardless of remaining_gap.
    for base_secs, remaining_gap, name in (
        (86_400, 1_000, "no_anchor_full_cooldown_gap_positive"),
        (86_400, 0, "no_anchor_full_cooldown_gap_zero"),
    ):
        expected = _drive(base_secs, 0, remaining_gap)
        cases.append(
            {
                "case_id": name,
                "base_secs": base_secs,
                "amount_sats": 0,
                "remaining_band_gap_sats": remaining_gap,
                "expected_secs": expected,
            }
        )
    return cases


def _gen_drift_override_cases():
    """`drift_override(anchor_ratio, current_ratio) -> anchor - current >=
    0.30`, transcribed verbatim from the single inline boolean at
    `rebalance_engine_v2.py:1166` (`anchor_ratio - current_ratio >=
    drift_threshold`, `drift_threshold` defaulting to the module's
    `rebalance_drift_override_ratio` config default of 0.30) — there is no
    standalone function to drive; this IS the whole function body.
    """
    cases = []
    grid = [
        # 0.7 - 0.4 == 0.29999999999999993 in float64 (NOT exactly 0.30) —
        # a genuine binary-fp-precision edge, not a mislabeled boundary: it
        # pins that the Rust port must reproduce Python's float subtraction
        # bit-for-bit rather than "helpfully" rounding first.
        (0.70, 0.40, "near_threshold_fp_precision_no_trigger"),
        (0.70, 0.41, "just_below_threshold_no_trigger"),  # 0.29
        (0.70, 0.39, "just_above_threshold_triggers"),  # 0.31 (approx, fp)
        (0.50, 0.50, "no_drift_no_trigger"),
        (0.40, 0.70, "negative_drift_no_trigger"),
        (1.0, 0.70, "large_drift_triggers"),
        (0.30, 0.0, "exact_threshold_from_zero"),  # 0.3 - 0.0 == 0.3 exactly
    ]
    for anchor_ratio, current_ratio, name in grid:
        expected = (anchor_ratio - current_ratio) >= 0.30
        cases.append(
            {
                "case_id": name,
                "anchor_ratio": anchor_ratio,
                "current_ratio": current_ratio,
                "expected": expected,
            }
        )
    return cases


def gen_cooldowns() -> dict:
    """Dump persisted per-kind cooldown backoff, fill-fraction dest
    cooldown (audit F7), and drift-override parity data (Phase 5 Task 6).

    Feeds `crates/revops-rebalance/src/cooldowns.rs` /
    `crates/revops-rebalance/tests/cooldowns.rs` in the Rust port
    (cl-revenue-ops-r), committed there as `fixtures/rebalance/
    cooldowns.json`. `PairFutility` (3 failures / 1800s window) and
    `DestFutility` (4/10 thresholds, keyed on `to_peer_id` — DEF-063) are
    NOT fixture-driven here: both are documented, fully-specified-by-the-
    plan in-memory state machines (constants already frozen in the brief),
    proven instead by targeted Rust unit tests in `cooldowns.rs` — there is
    no batch input/output table to golden-pin for a stateful breaker in the
    way there is for the pure functions below.
    """
    return {
        "persisted_cooldown_cases": _gen_persisted_cooldown_cases(),
        "dest_cooldown_cases": _gen_dest_cooldown_cases(),
        "drift_override_cases": _gen_drift_override_cases(),
    }


def gen_partial_amounts() -> dict:
    """Dump `RebalanceEngine._native_partial_amounts` over a boundary +
    seeded-random amount grid (Phase 5 Task 7).

    Feeds `crates/revops-rebalance/src/engine.rs::native_partial_amounts` /
    `tests/engine.rs::native_partial_amounts_replays_python_fixture` in the
    Rust port (cl-revenue-ops-r), committed there as `fixtures/rebalance/
    partial_amounts.json`. The ladder is the ONE new pure function T7
    introduces (floor `min(orig-1, max(1000, min(5000, orig//2)))`, halving
    steps, max 7 amounts); everything else in the engine is orchestration
    proven by scripted-double tests. The static method is driven directly on
    the class — no engine instance (and hence no plugin/db stub) is needed.
    """
    from modules.rebalance_engine_v2 import RebalanceEngine

    fn = RebalanceEngine._native_partial_amounts
    amounts = [
        0, 1, 2, 3, 999, 1_000, 1_001, 1_999, 2_000, 2_001, 3_999, 4_000,
        4_999, 5_000, 5_001, 9_999, 10_000, 10_001, 20_000, 50_000, 99_999,
        100_000, 100_001, 640_000, 1_000_000, 2_000_000, 10_000_000,
    ]
    rng = random.Random(0x5EBA1A7C)
    amounts += sorted(rng.randrange(1, 5_000_000) for _ in range(25))
    return {
        "cases": [
            {"amount_sats": a, "amounts": fn(a)} for a in amounts
        ],
    }


SUITES = {
    "modes": gen_modes,
    "planner": gen_planner,
    "segstore": gen_segstore,
    "router": gen_router,
    "executor": gen_executor,
    "ev": gen_ev,
    "cooldowns": gen_cooldowns,
    "partial_amounts": gen_partial_amounts,
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
