# Design: Adopt three upstream-`rebalance` patterns into the cl_revenue_ops rebalancer

Date: 2026-07-02. Status: **DESIGN — awaiting operator review.** Author: Hex.
Source analysis: the comparison of `lightningd/plugins/rebalance` (askrene-native, CLN ≥25.09)
against our rebalance stack. This spec turns the three adoptable patterns into an implementation
design. The implementation plan (writing-plans) is generated only after this spec is approved.

## Goal

Make our EV-driven rebalancer smarter about **when** and **whether** to spend, by adopting three
things upstream does better, without regressing any safety property from the deep audit.

1. **Live-activity coordination (#1)** — stop spending to rebalance a channel that live forwarding
   is already moving the helpful way.
2. **Realized-utilization EV basis (#2)** — replace the hardcoded `EXPECTED_UTILIZATION = 0.5`
   with each channel's *measured* forwarding utilization (closes audit hypotheses RE-H1/H2).
3. **Size-tiered ideal-ratio targets (#3)** — replace the flat 0.35–0.65 target band with
   per-channel targets derived from the node's channel-size distribution.

## Operator decisions locked for this design

- **Rollout: ship live, default ON** (operator, 2026-07-02). No shadow phase. Mitigated by:
  (a) every feature's parameters are config-tunable so behavior is adjustable/reversible without a
  code change, and (b) the **cross-category atomic budget rail is unchanged** — see Safety.
- **#1 action:** deprioritize via a soft score penalty (no hard skip).
- **#2 utilization:** per-channel realized utilization, clamped, with a thin-history fallback to
  the 0.5 prior.
- **#3 targets:** size-tiered ideal ratio replacing the flat band.

*(These are the recommended options, chosen while the operator was away; revisable on review.)*

## Non-negotiable safety invariant (why default-ON is acceptable)

**None of these three features can cause an overspend.** All spend still flows through the
audited unified cross-category budget reservation (`reserve_budget` inside `BEGIN IMMEDIATE`,
RB-I1/I3/I4, guarded by `test_all_spenders_atomic.py`). These features only change:
- which candidate pairs the planner generates (#3),
- the EV score used to rank/gate them (#2, #1).
The worst case is *suboptimal pair selection within budget*, never runaway spend. No feature here
touches a spend path, a reservation, or the budget math. This is the property that makes
"ship live, default ON" safe, and it is asserted by keeping the existing budget/atomicity tests
green (regression gate at every merge).

## Architecture: one shared signal layer, three consumers

The three features share a need for **per-channel windowed forwarding facts**. Rather than three
ad-hoc queries, add one small, well-bounded data unit and have all three consume it.

### New unit: `ChannelFlowFacts` (data access + snapshot fields)

A per-channel, per-cycle fact bundle computed once from the existing `forwards` table
(`out_msat`/`in_msat`, already indexed; reuse the `get_volume_since` / histogram query style in
`database.py`). Computed for all channels at snapshot-build time and attached to `ChannelState`
(`rebalance_state_v2.py`).

Fields:
- `out_sats_window` / `in_sats_window` — forwarded sats over a **short** window (default 3600s,
  config `rebalance_activity_window_seconds`) — feeds #1.
- `realized_utilization` — outbound forwarded volume ÷ capacity (turnover fraction) over a
  **long** window (default 7d, config `rebalance_utilization_window_days`), clamped to
  `[U_FLOOR, U_CEIL]` (defaults 0.05, 1.0; config) — feeds #2.
- `forward_count_window` — count over the long window; if `< util_min_forwards` (default 5) the
  channel is "thin-history" and #2 falls back to the 0.5 prior — feeds #2.

Contract: pure function of (channel_id, now, config) → facts; no writes; unit-testable in
isolation with a seeded `forwards` fixture. This is the one place forwarding history is read for
rebalancing, so it is the one place to test the SQL/rounding.

### Consumer #2 — realized-utilization EV (engine)

In `rebalance_engine_v2.py`, replace the three `EXPECTED_UTILIZATION` multiplications
(lines ~503/509/518) with the pair's destination `realized_utilization`:
- `u = dest.realized_utilization` if `dest.forward_count_window >= util_min_forwards`, else
  `EXPECTED_UTILIZATION` (0.5 prior).
- Value term becomes `amount_sats * dest_value_fee_ppm / 1e6 * u` (and the matching source/cost
  terms use their channel's `u` consistently).
- `EXPECTED_UTILIZATION` is retained as the named fallback constant (not deleted).
- `score_decomposition` gains `realized_utilization` + `utilization_source` ("realized"|"prior")
  for observability (the dict already exists at ~571).

Effect: a channel that never forwards scores its value term near the floor (was 0.5 → often
over-valued); a genuinely hot channel scores higher. Directly empirical, closes RE-H1/H2.

### Consumer #1 — live-activity penalty (engine)

Add a soft penalty to `final_score_sats`:
- "Helpful net-flow" = live forwarding already moving the channel toward its target the same way a
  rebalance would. For a **source** (over-local, rebalanced by draining outbound): helpful =
  `out_sats_window`. For a **destination** (over-remote, filled inbound): helpful =
  `in_sats_window`.
- `activity_penalty_sats = activity_penalty_coeff * helpful_net_flow_sats * dest_value_fee_ppm/1e6`
  (config `rebalance_activity_penalty_coeff`, default TBD-small e.g. 0.5), **capped** at
  `activity_penalty_cap_frac` of the pair's gross value term (default 0.5) so a strongly-EV pair
  can still run.
- Subtracted in the score composition; surfaced in `score_decomposition` as `activity_penalty_sats`.

No hard skip; no new skip reason (keeps `VALID_SKIP_REASONS` / the RA2-1 drift guard untouched).

*Fee-cycle coordination (secondary):* we ARE the fee controller (no separate `feeadjuster`), so
upstream's "toggle feeadjuster off during a run" maps to **ordering**: the rebalance cycle already
reads a fresh config/state snapshot per cycle, so it prices against current fees. We add one
lightweight guard — the activity signal is computed from the same snapshot — and document that the
fee cycle and rebalance cycle must not be collapsed into one lock. No behavioral fee change.

### Consumer #3 — size-tiered targets (planner)

In `rebalance_planner_v2.py`, replace the scalar `target_band_low/high` classification
(lines ~129/139) with **per-channel** `(band_low, band_high)`:
- Compute the node's "enough liquidity" reference from the channel-size distribution at
  snapshot-build (median/percentile capacity; mirror upstream `enough_liquidity` intent without
  its binary search — a percentile is sufficient and cheaper).
- Small channels (capacity ≤ reference) → tight band around 0.5 (default ±0.15 → 0.35–0.65,
  i.e. current behavior preserved for small channels).
- Large channels (capacity > reference) → asymmetric band allowing them to hold the residual
  (target skewed so a big channel is a liquidity buffer, not force-balanced).
- The flat band remains the config default fallback (`rebalance_size_tiered_targets` toggle,
  default ON; when OFF, current flat behavior). Targets are bounded to sane limits.

Per-channel `(band_low, band_high)` is computed in the state/snapshot layer and consumed by the
planner, keeping the planner a pure classifier.

## Data flow (per cycle)

```
snapshot build ──> ChannelFlowFacts (per channel, from forwards table)
                     │                         │                        │
                     ▼                         ▼                        ▼
   ChannelState.realized_utilization   ChannelState.{out,in}_sats   ChannelState.(band_low,high)
                     │                         │                        │
                     ▼                         ▼                        ▼
        engine EV value term (#2)     engine activity penalty (#1)   planner classify (#3)
                     └──────────────► final_score_sats ◄─────────────┘  (which pairs exist)
                                          │
                                          ▼
                             existing EV gate + UNCHANGED budget rail ──► execute
```

## Error handling / degradation

- `ChannelFlowFacts` failure (DB error, missing rows) → return neutral facts (utilization = 0.5
  prior, net-flow = 0, flat band) so the rebalancer degrades to *current behavior*, never crashes.
  Fail-open to today's logic, consistent with the plugin's fail-open posture.
- All new config values validated in `config.py` with the existing range/type checks; out-of-range
  → clamped/rejected like other options.
- No new RPC, no new spend path, no new lock.

## Testing (TDD, red-first per feature)

- **`ChannelFlowFacts`**: unit tests over a seeded `forwards` fixture — utilization math (rounding,
  clamp, thin-history fallback), directional net-flow, window boundaries, empty/zero-capacity.
- **#2**: EV score uses realized u for a hot channel and the 0.5 prior for a thin-history channel;
  `score_decomposition.utilization_source` correct. **Mutation:** flip realized↔prior, drop the
  clamp — tests must fail.
- **#1**: a pair whose channel has helpful live net-flow scores lower than an identical pair
  without it; the penalty is capped (a strongly-positive pair still passes the gate).
- **#3**: a large channel gets an asymmetric band and is NOT classified over_local at a ratio a
  small channel would be; toggle OFF restores the flat band exactly.
- **Regression gate (every merge):** full suite + `scorecard.py --deep-only` +
  `test_all_spenders_atomic.py` green — proves the budget/atomicity invariant is untouched.

## Build order

0. `ChannelFlowFacts` data unit + `ChannelState` fields + config options (shared foundation).
1. #2 (realized utilization) — smallest, highest-confidence, closes an audit hypothesis.
2. #3 (size-tiered targets) — changes which pairs exist; validate planner classification.
3. #1 (activity penalty) — layered on the engine score last.
4. Integration + live-node verification (`revenue-status` / `score_decomposition` inspection on
   hive-nexus-02; confirm penalties/utilization appear and budget behavior is unchanged).

Each step is its own commit(s), red-first, merged behind the green regression gate.

## Out of scope (YAGNI / anti-patterns not adopted)

- Upstream's unbounded `rebalanceall` spend model, statelessness, `maxparts=1`-as-constraint, and
  hours-long blocking thread — explicitly rejected in the analysis; not built.
- A single node-wide liquidity-health scalar (the trivial 4th idea) — nice-to-have, not part of
  this behavioral change; can be a separate small observability task.
- Any change to fee computation, the budget rail, reservations, or spend paths.

## Open parameters for operator review

Defaults chosen conservatively; all config-tunable: activity window (3600s), penalty coeff (~0.5)
and cap (0.5×), utilization window (7d), utilization clamp [0.05, 1.0], thin-history threshold
(5 forwards), size reference percentile, and size-tiered small-channel band (±0.15). These are the
knobs to sanity-check on review.
