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


SUITES = {
    "modes": gen_modes,
    "planner": gen_planner,
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
