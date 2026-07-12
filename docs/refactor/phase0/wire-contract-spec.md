# Draft wire-contract specification v0 (Phase 0 deliverable)

Status: DRAFT for operator review. Instantiates refactor.md Workstream
J1–J4 with concrete choices. Nothing here is enforced yet.

## Schema versioning (J1)

- Every canonical payload carries `schema_name` (string, e.g.
  "economic_snapshot") and `schema_version` (integer, starts at 0 while
  draft; 1 = first frozen).
- Encoding: JSON Schema draft 2020-12, one file per payload type at
  `schemas/<name>.v<version>.schema.json`.
- Unknown-field rule: readers MUST ignore unknown fields
  (`additionalProperties: true`) until version 1 freezes; version 1
  revisits per payload.
- Enum wire values: stable UPPER_SNAKE strings (`"INBOUND_GATEWAY"`),
  never ordinals or Python enum names implicitly.
- Change classes: backward-compatible (add optional field), forward-
  compatible (documented default), breaking (new schema_version +
  migration fixtures).

## Numeric and monetary semantics (J2)

- Money: integer millisatoshi. Canonical payloads encode msat as JSON
  integers; validators MUST reject non-integral or out-of-range values.
  Range: unsigned values in [0, 2^63-1] (checked u64 in Rust); signed
  P&L/deltas in [-(2^63), 2^63-1] (checked i64).
- Overflow: any checked-arithmetic failure in an authorization-relevant
  computation fails closed (reason code ARITHMETIC_OVERFLOW).
- Ratios/confidence/multipliers: scaled fixed-point integers,
  denominator 1_000_000 (field suffix `_ppm` for rates, `_micro` for
  generic ratios: confidence 0..1 → 0..1_000_000 micro). Binary floats
  are permitted ONLY in fields explicitly marked non-authoritative
  (suffix `_diag`).
- msat→sat conversions are DIRECTION-SPECIFIC (three canonical rules in
  `modules/utils.py`, plus one local exception) and must be encoded as
  conformance vectors:
  - CEILING (`base_to_sats_ceil`, utils.py:73): fees, budgets, costs —
    never undercharge or underbudget; ALSO revenue reporting, so sub-sat
    earnings stay visible instead of truncating to zero.
  - FLOOR (`base_to_sats_floor`, utils.py:81): capacity and balances —
    never overstate what is spendable.
  - TOWARD-ZERO (`base_delta_to_sats_toward_zero`, utils.py:91): signed
    deltas.
  - Local exception: `_close_protection_reason` floors
    `sourced_fee_30d_msat // 1000` (`modules/capacity_planner.py:1167`)
    before its >100-sat protection threshold — pinned by golden fixture;
    Phase 1 must preserve or explicitly migrate it.
  - sats→msat: exact `* 1000` (`modules/utils.py:98` `sats_to_base`).
- Division by zero / missing denominator: defined per field; default is
  "signal absent" (null + lower confidence), NEVER silent zero
  (refactor invariant 7). Precedent already pinned by golden fixture
  `profitability/marginal_roi_zero_cost_positive_profit`: zero 30d cost
  with positive profit = exactly 1.0.
- Timestamps: integer unix seconds UTC (`_at` suffix); durations
  integer seconds.

## Determinism rules (J3)

- Canonical serialization for hashing: JSON with lexicographically
  sorted keys, no insignificant whitespace, UTF-8, integers only for
  authoritative numerics.
- Idempotency key: `sha256(canonical_json(envelope-subset))` where the
  subset is (intent type, target id, amount, snapshot id, budget
  bucket) — draft; finalize in Phase 1.
- Tie-break sequence (refactor.md J3): precedence class → requested
  priority → expected value → confidence → capital committed → stable
  target identifier → intent ID. Current violations (single-float-key
  sorts): see `portability-hazards.md` §4.
- Clock: cycle context supplies `cycle_time_at`; policies MUST NOT read
  wall clock (hazards §1 lists current violations).
- Randomness: explicit seed in cycle context, recorded in ledger
  evidence (current violation: DTS posterior sampling, hazards §2).
- DB queries feeding decisions carry explicit ORDER BY.

## Reason-code catalog v0 (J4)

Seed catalog (code → owning layer → kind):

| Code | Layer | Kind |
|---|---|---|
| BUDGET_EXHAUSTED | governor | rejection |
| AUTHORITY_LEVEL_BLOCKED | governor | rejection |
| PAUSED | governor | rejection |
| INTENT_STALE | arbiter | deferral |
| INTENT_SUPERSEDED | arbiter | rejection |
| CHANNEL_PROTECTED | governor | rejection |
| CONTRACT_OBLIGATION | arbiter | rejection |
| EV_BELOW_HOLD_MARGIN | policy | hold |
| INSUFFICIENT_CONFIDENCE | governor | hold |
| FEE_RAIL_CLAMPED | policy | clamp |
| COOLDOWN_ACTIVE | policy | hold |
| CONFLICT_CLOSE_REBALANCE | arbiter | rejection |
| EXTERNAL_CIRCUIT_BREAKER | executor | deferral |
| EXTERNAL_OUTCOME_UNKNOWN | reconciliation | unknown |
| ARITHMETIC_OVERFLOW | any | failure |
| SCHEMA_INVALID | any | failure |

Existing string reasons to map into the catalog (all now pinned by
Phase 0 golden fixtures):

- Close protection: `KALMAN_LOW_CONFIDENCE`, `INBOUND_GATEWAY`,
  `SOURCED_FEE_CONTRIBUTION`, `REVENUE_ROUTE`
  (`tests/golden/fixtures/close_protection/`)
- LN+ gate prefixes: `fill_state:`, `terms:`, `peer_quality:`,
  `own_node:` (`tests/golden/fixtures/lnplus/`)
- Planner skip reasons: `inside_band`, `no_partner`, `cooldown_active`,
  `outcompeted`, `source_ineligible`
  (`tests/golden/fixtures/rebalance/`)
- Fee damping cap reasons: `normal_cycle_delta_cap`,
  `wake_cycle_delta_cap`, `none` (`tests/golden/fixtures/fee/`)
- Boltz mode reasons: `standing_onchain_reserve_below_target`,
  `onchain_reserve_healthy_use_balance_mode`,
  `no_eligible_boltz_actions` (`tests/golden/fixtures/boltz/`)

Each code defines its owning layer, required context fields, and
hold/rejection/deferral/clamp/failure/unknown kind in Phase 1.

## Domain wrappers (J2, Phase 1 implementation)

`Msat`, `Sat`, `Ppm`, `FixedRatio(micro)`, `UnixTime`, `IntentId`,
`ChannelId` (short-channel-id string form `NNNxNNNxN`), `PeerId`
(66-hex-char compressed pubkey, `^0[23][0-9a-f]{64}$`).
