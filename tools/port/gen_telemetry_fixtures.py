#!/usr/bin/env python3
"""Generate frozen datastore-telemetry byte-shape fixtures for the Rust port.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_telemetry_fixtures.py <output.json>

The output feeds `crates/revops-analytics/src/telemetry.rs` /
`crates/revops-analytics/tests/telemetry.rs` in the Rust port
(cl-revenue-ops-r), committed there as `fixtures/telemetry.json` (Phase 3
Task 8). Every "expected_json" string below is the REAL
`json.dumps(payload)` output produced by constructing real
`ChannelProfitability`/`ChannelRevenue`/`ChannelCosts` objects and running
the *exact* dict-construction lines of
`modules/profitability_analyzer.py::_push_profitability_summary`
(lines ~700-733) -- copied verbatim into `build_channel_entry()` below, with
ONE deliberate substitution: `fee_multiplier` is a value INJECTED into that
method (`self.get_fee_multiplier(ch_id)`, a whole fee-stack computation that
is Phase 4 evidence, out of scope for this task) rather than computed here;
the fixture instead passes an explicit `fee_multiplier` float per entry, and
`build_channel_entry` applies the same `round(m, 2)` Python does. Every
other line matches the source byte for byte.

Ground-truth notes:

- `json.dumps(payload)` (Python's DEFAULT arguments) keeps dict insertion
  order (NOT sorted), uses separators `(", ", ": ")`, `ensure_ascii=True`
  (would `\\uXXXX`-escape any non-ASCII, though this payload's fields are
  IDs/numbers/enum-values -- ASCII only, by construction of Lightning
  channel/peer IDs), and renders floats via `repr(float)`.
- `roi_pct` / `fee_multiplier` in the payload are `round(x, 2)` of a float
  that may carry binary-repr noise (e.g. `10.555000000000001`); the
  generator deliberately includes such noisy floats pre-round so the Rust
  port's `py_round` + `py_repr` combination is exercised on the exact
  CPython-computed post-round value, not a hand-typed one.
- The `channels` dict in the real code is built by iterating
  `results.items()` (a plain dict), so key/insertion order in the payload
  is simply the order channels were added to that dict -- reproduced here
  by iterating the `entries` list in the order given.
- Fields NOT part of the datastore payload (`capacity_sats`,
  `cost_per_sat_routed`, `fee_per_sat_routed`, `last_routed`,
  `marginal_profit_30d_sats`, `opener`) are dataclass-required but
  payload-irrelevant; the generator pins them to fixed placeholder values
  (mirrored exactly by the Rust test's construction helper) so both sides
  build byte-identical `ChannelProfitability` objects without those fields
  needing per-case variation.
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules import profitability_analyzer as pa  # noqa: E402
from modules import classification as cl  # noqa: E402
from modules.utils import sats_to_base  # noqa: E402

# Placeholders for dataclass fields the datastore payload never reads.
PLACEHOLDER_CAPACITY_SATS = 1_000_000
PLACEHOLDER_COST_PER_SAT_ROUTED = 0.0
PLACEHOLDER_FEE_PER_SAT_ROUTED = 0.0
PLACEHOLDER_LAST_ROUTED = None
PLACEHOLDER_MARGINAL_PROFIT_30D_SATS = 0
PLACEHOLDER_OPENER = "local"


def bits(v: float) -> str:
    return struct.pack("<d", float(v)).hex()


def make_profitability(spec: dict) -> "pa.ChannelProfitability":
    costs = pa.ChannelCosts(
        channel_id=spec["channel_id"],
        peer_id=spec["peer_id"],
        open_cost_sats=spec["open_cost_sats"],
        rebalance_cost_sats=spec["rebalance_cost_sats"],
    )
    revenue = pa.ChannelRevenue(
        channel_id=spec["channel_id"],
        fees_earned_msat=spec["fees_earned_msat"],
        volume_routed_msat=spec["volume_routed_msat"],
        forward_count=spec["forward_count"],
        sourced_volume_msat=spec["sourced_volume_msat"],
        sourced_fee_contribution_msat=spec["sourced_fee_contribution_msat"],
        sourced_forward_count=spec["sourced_forward_count"],
    )
    return pa.ChannelProfitability(
        channel_id=spec["channel_id"],
        peer_id=spec["peer_id"],
        capacity_sats=PLACEHOLDER_CAPACITY_SATS,
        costs=costs,
        revenue=revenue,
        net_profit_sats=spec["net_profit_sats"],
        roi_percent=spec["roi_percent"],
        classification=pa.ProfitabilityClass[spec["classification"]],
        cost_per_sat_routed=PLACEHOLDER_COST_PER_SAT_ROUTED,
        fee_per_sat_routed=PLACEHOLDER_FEE_PER_SAT_ROUTED,
        days_open=spec["days_open"],
        last_routed=PLACEHOLDER_LAST_ROUTED,
        marginal_profit_30d_sats=PLACEHOLDER_MARGINAL_PROFIT_30D_SATS,
        rebalance_cost_30d_sats=spec["rebalance_cost_30d_sats"],
        opener=PLACEHOLDER_OPENER,
        contribution_30d_msat=spec["contribution_30d_msat"],
        fees_earned_30d_msat=spec["fees_earned_30d_msat"],
        sourced_fee_30d_msat=spec["sourced_fee_30d_msat"],
        forward_count_30d=spec["forward_count_30d"],
        sourced_forward_count_30d=spec["sourced_forward_count_30d"],
        window_30d_available=spec["window_30d_available"],
    )


def build_channel_entry(ch_id: str, p: "pa.ChannelProfitability", fee_multiplier: float) -> dict:
    """Verbatim copy of `_push_profitability_summary`'s per-channel dict
    (modules/profitability_analyzer.py lines ~700-733), with
    `self.get_fee_multiplier(ch_id)` replaced by the injected
    `fee_multiplier` parameter (see module docstring)."""
    return {
        "channel_id": ch_id,
        "peer_id": p.peer_id,
        "class": p.classification.value,
        "net_profit_sats": p.net_profit_sats,
        "roi_pct": round(p.roi_percent, 2),
        "days_open": p.days_open,
        "role": p.channel_role.value,
        "fee_multiplier": round(fee_multiplier, 2),
        "forward_count": p.revenue.forward_count,
        "sourced_forward_count": p.revenue.sourced_forward_count,
        "total_forward_count": p.revenue.total_forward_count,
        "fees_earned_msat": p.revenue.fees_earned_msat,
        "fees_earned_sats": p.revenue.fees_earned_sats,
        "volume_routed_msat": p.revenue.volume_routed_msat,
        "sourced_volume_msat": p.revenue.sourced_volume_msat,
        "sourced_fee_contribution_msat": p.revenue.sourced_fee_contribution_msat,
        "sourced_fee_contribution_sats": p.revenue.sourced_fee_contribution_sats,
        "total_contribution_msat": p.revenue.total_contribution_msat,
        "total_contribution_sats": p.revenue.total_contribution_sats,
        "open_cost_msat": sats_to_base(p.costs.open_cost_sats),
        "rebalance_cost_msat": sats_to_base(p.costs.rebalance_cost_sats),
        "net_pnl_msat": p.revenue.total_contribution_msat - sats_to_base(p.costs.total_cost_sats),
        "contribution_30d_msat": p.contribution_30d_msat,
        "fees_earned_30d_msat": p.fees_earned_30d_msat,
        "sourced_fee_30d_msat": p.sourced_fee_30d_msat,
        "forward_count_30d": p.forward_count_30d,
        "sourced_forward_count_30d": p.sourced_forward_count_30d,
        "role_30d": p.role_30d.value,
        "marginal_roi_reliable": p.marginal_roi_reliable,
    }


def build_payload(timestamp: int, entries: list) -> dict:
    """Verbatim shape of the `payload` dict at the end of
    `_push_profitability_summary` -- `{"timestamp": ..., "channels": ...}`."""
    channels = {}
    for ch_id, p, fee_multiplier in entries:
        channels[ch_id] = build_channel_entry(ch_id, p, fee_multiplier)
    return {"timestamp": timestamp, "channels": channels}


def spec_to_fixture_entry(spec: dict, fee_multiplier: float) -> dict:
    """Encode one channel's input spec for the Rust side, with float fields
    pinned as `struct.pack('<d', v).hex()` bit patterns."""
    out = dict(spec)
    out["roi_percent_bits"] = bits(spec["roi_percent"])
    del out["roi_percent"]
    out["fee_multiplier_bits"] = bits(fee_multiplier)
    return out


def gen_payload_cases() -> list:
    cases = []

    # Case 1: single profitable channel, clean round numbers.
    specs = [
        dict(channel_id="103x1x0", peer_id="02" + "a1" * 32, open_cost_sats=5000,
             rebalance_cost_sats=200, fees_earned_msat=1_250_000, volume_routed_msat=50_000_000,
             forward_count=25, sourced_volume_msat=0, sourced_fee_contribution_msat=0,
             sourced_forward_count=0, net_profit_sats=1000, roi_percent=20.0,
             classification="PROFITABLE", days_open=90, contribution_30d_msat=400_000,
             fees_earned_30d_msat=400_000, sourced_fee_30d_msat=0, forward_count_30d=12,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=150, window_30d_available=True)
    ]
    fee_mults = [0.95]
    cases.append({
        "id": "single_profitable_channel",
        "timestamp": 1_752_400_000,
        "specs": specs,
        "fee_mults": fee_mults,
    })

    # Case 2: float-artifact ROI pre-round (10.555000000000001), and a
    # fee_multiplier that also carries binary noise.
    specs = [
        dict(channel_id="203x2x1", peer_id="03" + "b2" * 32, open_cost_sats=10_000,
             rebalance_cost_sats=0, fees_earned_msat=333_333, volume_routed_msat=3_000_000,
             forward_count=8, sourced_volume_msat=200_000, sourced_fee_contribution_msat=15_000,
             sourced_forward_count=3, net_profit_sats=323_333, roi_percent=10.555000000000001,
             classification="BREAK_EVEN", days_open=15, contribution_30d_msat=50_000,
             fees_earned_30d_msat=40_000, sourced_fee_30d_msat=15_000, forward_count_30d=8,
             sourced_forward_count_30d=3, rebalance_cost_30d_sats=0, window_30d_available=True)
    ]
    fee_mults = [1.0 + 0.1 / 3.0]  # 1.0333333333333334 -> round(.,2) == 1.03
    cases.append({
        "id": "float_artifact_roi_preround",
        "timestamp": 1_752_400_100,
        "specs": specs,
        "fee_mults": fee_mults,
    })

    # Case 3: negative net_profit / negative net_pnl_msat (underwater
    # channel whose rebalance cost dwarfs revenue).
    specs = [
        dict(channel_id="303x0x2", peer_id="02" + "c3" * 32, open_cost_sats=20_000,
             rebalance_cost_sats=15_000, fees_earned_msat=50_000, volume_routed_msat=1_000_000,
             forward_count=4, sourced_volume_msat=0, sourced_fee_contribution_msat=0,
             sourced_forward_count=0, net_profit_sats=-34_950, roi_percent=-99.857142857142861,
             classification="UNDERWATER", days_open=200, contribution_30d_msat=0,
             fees_earned_30d_msat=0, sourced_fee_30d_msat=0, forward_count_30d=0,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=15_000, window_30d_available=True)
    ]
    fee_mults = [1.10]
    cases.append({
        "id": "negative_pnl_underwater",
        "timestamp": 1_752_400_200,
        "specs": specs,
        "fee_mults": fee_mults,
    })

    # Case 4: multi-channel dict to pin iteration/insertion order (3
    # channels, deliberately NOT alphabetically or numerically sorted so a
    # sort-based serializer would be caught immediately).
    specs = [
        dict(channel_id="900x1x1", peer_id="02" + "f0" * 32, open_cost_sats=1_000,
             rebalance_cost_sats=100, fees_earned_msat=80_000, volume_routed_msat=2_000_000,
             forward_count=15, sourced_volume_msat=500_000, sourced_fee_contribution_msat=20_000,
             sourced_forward_count=20, net_profit_sats=79_000 - 1_100, roi_percent=71.8,
             classification="PROFITABLE", days_open=30, contribution_30d_msat=100_000,
             fees_earned_30d_msat=80_000, sourced_fee_30d_msat=20_000, forward_count_30d=15,
             sourced_forward_count_30d=20, rebalance_cost_30d_sats=100, window_30d_available=True),
        dict(channel_id="100x2x0", peer_id="03" + "0e" * 32, open_cost_sats=8_000,
             rebalance_cost_sats=0, fees_earned_msat=0, volume_routed_msat=0,
             forward_count=0, sourced_volume_msat=0, sourced_fee_contribution_msat=0,
             sourced_forward_count=0, net_profit_sats=-8_000, roi_percent=-100.0,
             classification="STAGNANT_CANDIDATE", days_open=120, contribution_30d_msat=0,
             fees_earned_30d_msat=0, sourced_fee_30d_msat=0, forward_count_30d=0,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=0, window_30d_available=True),
        dict(channel_id="500x0x3", peer_id="02" + "77" * 32, open_cost_sats=3_000,
             rebalance_cost_sats=3_500, fees_earned_msat=200_000, volume_routed_msat=6_000_000,
             forward_count=60, sourced_volume_msat=0, sourced_fee_contribution_msat=0,
             sourced_forward_count=0, net_profit_sats=193_500, roi_percent=2978.46,
             classification="PROFITABLE", days_open=400, contribution_30d_msat=10_000,
             fees_earned_30d_msat=10_000, sourced_fee_30d_msat=0, forward_count_30d=5,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=99, window_30d_available=True),
    ]
    fee_mults = [0.95, 1.0, 0.95]
    cases.append({
        "id": "multi_channel_insertion_order",
        "timestamp": 1_752_400_300,
        "specs": specs,
        "fee_mults": fee_mults,
    })

    # Case 5: marginal_roi_reliable boundary (rebalance_cost_30d_sats
    # exactly at / just under / just over the 100-sat reliability floor),
    # plus a role_30d fallback to lifetime role (window_30d_available False).
    specs = [
        dict(channel_id="600x1x0", peer_id="03" + "aa" * 32, open_cost_sats=500,
             rebalance_cost_sats=100, fees_earned_msat=100_000, volume_routed_msat=1_000_000,
             forward_count=11, sourced_volume_msat=0, sourced_fee_contribution_msat=0,
             sourced_forward_count=0, net_profit_sats=99_500, roi_percent=19900.0,
             classification="PROFITABLE", days_open=10, contribution_30d_msat=100_000,
             fees_earned_30d_msat=100_000, sourced_fee_30d_msat=0, forward_count_30d=11,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=100, window_30d_available=True),
        dict(channel_id="600x2x0", peer_id="03" + "ab" * 32, open_cost_sats=500,
             rebalance_cost_sats=99, fees_earned_msat=100_000, volume_routed_msat=1_000_000,
             forward_count=11, sourced_volume_msat=0, sourced_fee_contribution_msat=0,
             sourced_forward_count=0, net_profit_sats=99_401, roi_percent=19880.2,
             classification="PROFITABLE", days_open=10, contribution_30d_msat=100_000,
             fees_earned_30d_msat=100_000, sourced_fee_30d_msat=0, forward_count_30d=11,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=99, window_30d_available=True),
        dict(channel_id="600x3x0", peer_id="03" + "ac" * 32, open_cost_sats=500,
             rebalance_cost_sats=0, fees_earned_msat=0, volume_routed_msat=0,
             forward_count=2, sourced_volume_msat=0, sourced_fee_contribution_msat=200_000,
             sourced_forward_count=9, net_profit_sats=-500, roi_percent=-100.0,
             classification="BREAK_EVEN", days_open=5, contribution_30d_msat=0,
             fees_earned_30d_msat=0, sourced_fee_30d_msat=0, forward_count_30d=0,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=0, window_30d_available=False),
    ]
    fee_mults = [1.0, 1.0, 1.05]
    cases.append({
        "id": "marginal_roi_reliable_boundary_and_role30d_fallback",
        "timestamp": 1_752_400_400,
        "specs": specs,
        "fee_mults": fee_mults,
    })

    # Case 6: empty channels dict (no channels analyzed this cycle).
    cases.append({
        "id": "empty_channels",
        "timestamp": 1_752_400_500,
        "specs": [],
        "fee_mults": [],
    })

    # Case 7: inbound/outbound gateway roles + zombie classification, to
    # exercise every ProfitabilityClass/ChannelRole wire-value pair at least
    # once across the corpus.
    specs = [
        dict(channel_id="700x1x0", peer_id="02" + "d1" * 32, open_cost_sats=2_000,
             rebalance_cost_sats=500, fees_earned_msat=10_000, volume_routed_msat=100_000,
             forward_count=2, sourced_volume_msat=9_000_000, sourced_fee_contribution_msat=450_000,
             sourced_forward_count=30, net_profit_sats=-2_490, roi_percent=-99.55,
             classification="ZOMBIE", days_open=500, contribution_30d_msat=0,
             fees_earned_30d_msat=0, sourced_fee_30d_msat=0, forward_count_30d=0,
             sourced_forward_count_30d=0, rebalance_cost_30d_sats=500, window_30d_available=True),
        dict(channel_id="700x2x0", peer_id="03" + "d2" * 32, open_cost_sats=1_000,
             rebalance_cost_sats=0, fees_earned_msat=900_000, volume_routed_msat=20_000_000,
             forward_count=40, sourced_volume_msat=100_000, sourced_fee_contribution_msat=5_000,
             sourced_forward_count=2, net_profit_sats=899_000, roi_percent=89900.0,
             classification="PROFITABLE", days_open=60, contribution_30d_msat=900_000,
             fees_earned_30d_msat=900_000, sourced_fee_30d_msat=5_000, forward_count_30d=40,
             sourced_forward_count_30d=2, rebalance_cost_30d_sats=0, window_30d_available=True),
    ]
    fee_mults = [1.10, 0.95]
    cases.append({
        "id": "gateway_roles_and_zombie",
        "timestamp": 1_752_400_600,
        "specs": specs,
        "fee_mults": fee_mults,
    })

    fixtures = []
    for case in cases:
        entries = []
        fixture_entries = []
        for spec, fee_mult in zip(case["specs"], case["fee_mults"]):
            p = make_profitability(spec)
            entries.append((spec["channel_id"], p, fee_mult))
            fixture_entries.append(spec_to_fixture_entry(spec, fee_mult))
        payload = build_payload(case["timestamp"], entries)
        expected_json = json.dumps(payload)
        # Sanity: re-decode and compare structurally (catches accidental
        # typos in this generator itself, independent of the Rust side).
        assert json.loads(expected_json) == payload
        fixtures.append({
            "id": case["id"],
            "timestamp": case["timestamp"],
            "entries": fixture_entries,
            "expected_json": expected_json,
        })
    return fixtures


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
        sys.exit(1)

    out = {
        "meta": {
            "source": "modules/profitability_analyzer.py::_push_profitability_summary "
                       "+ modules/data_service.py::datastore_push",
        },
        "datastore_key": ["revenue", "profitability-summary"],
        "datastore_max_bytes": 60000,
        "payload_cases": gen_payload_cases(),
    }
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {sys.argv[1]} ({len(out['payload_cases'])} payload cases)")


if __name__ == "__main__":
    main()
