# Python-portability hazard inventory (baseline 5e8f747)

Hazards that block cross-language decision parity (refactor.md
Workstream J / invariant 14). Counts from `grep -c` sweeps; verify with
the commands in each section.

## 1. Wall-clock reads in decision code (`time.time(`)

| Module | count | Notes |
|---|---|---|
| modules/database.py | 82 | timestamps for spend windows — AUTHORIZATION-RELEVANT (budget window boundaries) |
| modules/fee_controller.py | 37 | cycle cadence, cooldowns, zero-flow streaks |
| cl-revenue-ops.py | 29 | loop scheduling, heartbeats |
| modules/profitability_analyzer.py | 24 | recency vs last_routed in classification (`_classify_channel` computes days_inactive from `time.time()` — the Phase 0 golden tests freeze it) |
| modules/lnplus_swaps.py | 11 | breaker windows, deadline math |
| modules/policy_manager.py | 10 | policy timestamps |
| capacity_planner / flow_analysis / rebalance_engine_v2 | 9 each | — |
| rebalancer.py 6 · boltz_manager.py 4 · data_service.py 3 · others ≤2 | | |

Verify: `grep -c "time.time(" modules/*.py cl-revenue-ops.py`
Refactor rule (J3): policies receive cycle time; direct reads must move
to snapshot/cycle context.

## 2. Randomness

- Loop-interval jitter: `random.randint` in every background loop tail
  (`cl-revenue-ops.py:3051,3091,3131,3191,3231,3289,3361`) — scheduling
  only, NOT decision-relevant; may remain.
- `modules/fee_controller.py` (5 uses): **Gaussian-Thompson posterior
  sampling inside `_adjust_channel_fee`** — DECISION-RELEVANT unseeded
  randomness. This is why Phase 0 goldens pin the damping/floor/deadband
  stages, not the raw DTS target. J3 requires seed injection recorded in
  cycle evidence before the fee target itself can be conformance-tested.
- `uuid` in `modules/boltz_manager.py` (2) — swap/reservation IDs;
  idempotency keys must become deterministic (J3). Reservation/event IDs
  elsewhere are built from `time.time()` + counters — also
  non-deterministic, same fix.

## 3. Binary floating point in authoritative paths (J2 violations)

- ROI/marginal_roi, confidence, kalman ratios, multipliers are Python
  floats end-to-end: `ChannelProfitability.marginal_roi` (float division),
  `_close_protection_reason` ROI thresholds (-30.0/-50.0 float compares),
  planner pair `score` floats rounded to 6dp
  (`rebalance_planner_v2._bootstrap_score_decomposition`).
- `_classify_channel` thresholds are float fractions (`roi < -0.10`) and
  are WIDENED by float multiplication (0.5×/1.5×) when the DTS posterior
  variance is low — spend-adjacent float math.
- Fee floor math divides float chain costs into ppm (`_calculate_floor`).
- Budget/spend amounts ARE integer sats/msat (good), but window boundary
  comparisons mix float `time.time()`.

## 4. Unordered/underspecified ordering feeding results

Confirmed sites where ranking uses a SINGLE float key with no total
tie-break (Python's stable sort makes ties fall back to construction
order — not a portable contract; J3 requires the documented tie-break
sequence):

- `modules/rebalance_planner_v2.py:156` — `pairs.sort(key=lambda p:
  p.score, reverse=True)`
- `modules/rebalance_planner_v2.py:223` — drain demand by `drain_score`
- `modules/capacity_planner.py:395,439,564,579,613,1436,1447,1506,1559,
  1593` — candidate/loser/winner rankings keyed on bare
  `roi`/`score`/`marginal_roi`

Dict iteration feeding these lists (22 `.items()`/`.values()` sites in
planner+capacity modules) inherits insertion order from RPC/db result
construction — same fix: sort by stable keys before ranking (J3).

## 5. Other hazards

- Untyped dicts cross every subsystem boundary (channel_info, swap
  dicts, plan dicts, RPC results) — the intent/schema work is the fix.
- Enum serialization: `ChannelRole`/`ProfitabilityClass`/`ChannelState`
  are Python enums; wire values must become stable strings (J1).
- Duck-typed `getattr(prof, 'role_30d', None)` fallbacks in
  capacity_planner tolerate legacy objects — schema versioning replaces
  this.
- Truthy string coercion at decision inputs:
  `_compute_dynamic_htlcmax_msat` accepts `enable_dynamic_htlcmax` as
  bool OR string ("true"/"1"/"yes") — wire contract must fix the type.
- `schema_version` DB table is write-only (no version gate) — replay/
  migration tooling cannot rely on it.
- Mutable module-level globals in `cl-revenue-ops.py` (managers wired at
  init; tests monkeypatch them) — Workstream H cycle context replaces.
- Boltz adapter shells out to `boltzcli` (subprocess) — outcome parsing
  is text/JSON from CLI; unknown-outcome handling must go through
  reconciliation (Workstream G).

## Hazards found while building the Phase 0 goldens

- `_classify_channel` required a frozen `time.time()` to be
  deterministic (see `tests/golden/test_golden_profitability.py`); it
  also silently swallows fee-state parsing errors (`except Exception:
  pass`) — a masked-evidence hazard (invariant 7).
- `ChannelProfitability.total_forward_count_30d` is a derived property;
  external writers can only set the component fields — good (single
  source), but the schema must model it as derived.
- The Boltz auto-cycle result embeds free-text `reason` strings that
  double as machine-checked selection reasons
  (`no_eligible_boltz_actions`) — reason-code catalog (J4) formalizes.
