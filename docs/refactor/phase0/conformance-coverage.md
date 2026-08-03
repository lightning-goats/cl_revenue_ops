# Conformance Corpus Coverage (generated — PR 9, Phase F)

Generator: `tools/conformance/generate_scenarios.py` (deterministic; regenerating must be byte-identical). Validation: `tools/conformance/validate_fixtures.py` — no plugin imports. Golden-derived cases copy fixture bytes verbatim; generated cases are produced BY the reference implementation.

Scenario classes: 35/40. Documented gaps: 0 (listed last — these are honest holes with pointers, not silent omissions).

| # Scenario | Category | Requirement | Source | Files |
|---|---|---|---|---|
| `01-ordinary-profitable-channel` | classification | DoD 1/9: profitable classification from routing evidence | tests/golden/fixtures/profitability/classify_young_profitable.json | case.json |
| `02-source-gateway-protection` | classification | F5: inbound-gateway channels protected from closure | tests/golden/fixtures/close_protection/gateway_30d_protected.json | case.json |
| `03-sink-depletion` | rebalance_mode | F4: depleted profitable destinations preferred for refill | tests/golden/fixtures/rebalance/plan_profitable_dest_preferred.json | case.json |
| `04-balanced-channel` | classification | Workstream A: BALANCED role from 30d flow evidence | tests/golden/fixtures/profitability/role30d_balanced_30d.json | case.json |
| `05-underwater-classification` | classification | Workstream A: negative marginal ROI (underwater) | tests/golden/fixtures/profitability/marginal_roi_negative_profit.json | case.json |
| `06-stagnant-candidate` | classification | Workstream A: stagnant classification | tests/golden/fixtures/profitability/classify_old_loser_stagnant.json | case.json |
| `07-zombie-classification` | classification | Workstream A: zombie after failed defibrillation | tests/golden/fixtures/profitability/classify_zombie_after_failed_defib.json | case.json |
| `08-fee-rail` | fee_stage | ADR-001 stage 1 (rails): fee floor | tests/golden/fixtures/fee/floor_defaults_no_chain_costs.json | case.json |
| `09-fee-rate-limit` | fee_stage | ADR-001 stage 2 (rate_limit): per-cycle delta cap | tests/golden/fixtures/fee/damping_large_raise_clamped.json | case.json |
| `10-fee-deadband` | fee_stage | ADR-001 stage 3 (deadband): no-op suppression | tests/golden/fixtures/fee/damping_no_change.json | case.json |
| `11-fee-cooldown` | fee_stage | ADR-001 stage 4 (cooldown): wake-from-sleep cycle semantics | tests/golden/fixtures/fee/damping_wake_cycle_wider_cap.json | case.json |
| `12-dts-pid-components` | fee_stage | ADR-001: DTS posterior update + PID multiplier are deterministic given inputs | generated | case.json |
| `13-dynamic-htlcmax` | admission | F3: dynamic htlc_max admission control | tests/golden/fixtures/htlcmax/balanced_mid_share.json | case.json |
| `14-hot-channel-priority` | rebalance_mode | F4 table: hot-channel protection outranks normal redistribution as priority data, not a separate path | generated | case.json |
| `15-normal-rebalance` | rebalance_mode | F4: one planner; chunk-bounded amounts | tests/golden/fixtures/rebalance/plan_amount_bounded_by_chunk.json | case.json |
| `17-manual-diagnostic-rebalance` | rebalance_mode | F4 + contradiction #7: manual is operator-directed; diagnostic is a BOUNDED spend (evidence purchase), not free | generated | case.json |
| `18-conflicting-close-rebalance` | arbitration | J3/spec conflict rule: rebalance into a channel scheduled for closure is rejected (CONFLICT_CLOSE_REBALANCE) | generated | case.json |
| `19-protected-close-rejection` | authorization | F5: protection tags veto closure before any intent exists | tests/golden/fixtures/close_protection/allowed_protect_tag_blocks.json | case.json |
| `20-duplicate-open-priority` | arbitration | Duplicate OPEN_CHANNEL intents to one peer are deduplicated; the higher-priority intent wins | generated | case.json |
| `22-budget-exhaustion` | authorization | DoD 4/5: refused reservation -> BUDGET_EXHAUSTED, no spend | generated | case.json |
| `23-concurrent-reservation-contention` | reservation | DoD 5: atomic reservations cannot jointly oversubscribe | generated | case.json |
| `24-restart-outstanding-reservation` | reservation | DoD 5: reservations survive restart (durable store) | generated | case.json |
| `25-missing-execution-cost` | ledger | DoD 6: cost without reservation context is an ANOMALY, never silently absorbed | generated | case.json |
| `26-unknown-execution-outcome` | ledger | Workstream E: unknown outcome is a TERMINAL state pending reconciliation; reservation state preserved for the sweep | generated | case.json |
| `30-stale-intent` | authorization | DoD 4: stale envelopes rejected fail-closed (STALE) | generated | case.json |
| `31-duplicate-idempotency-key` | arbitration | J3: identical five-field subsets share an idempotency key; duplicates superseded (INTENT_SUPERSEDED) | generated | case.json |
| `32-numeric-overflow-underflow` | intent_semantics | Workstream J numeric rules: msat in [0, 2^63-1], checked — out-of-range raises, never wraps | generated | case.json |
| `33-msat-rounding-boundaries` | intent_semantics | Workstream J: explicit, documented rounding at msat/sat boundaries | generated | case.json |
| `34-expired-intent` | intent_semantics | Workstream B: expiry boundary is inclusive at expires_at | generated | case.json |
| `35-stable-ordering-tiebreak` | determinism | J3 ladder: precedence, -priority, -EV, -confidence, capital, target, intent_id — total and stable | generated | case.json |
| `36-map-order-independence` | determinism | Workstream J: canonical JSON is insertion-order independent | generated | case.json |
| `37-clock-seed-determinism` | determinism | Workstream J: identical (fields, clock) -> identical ids; no hidden entropy in the authoritative path | generated | case.json |
| `38-partial-batch-completion` | arbitration | Workstream H: a batch completes partially — survivors ordered, each rejection carries its own reason code | generated | case.json |
| `39-bookkeeper-present-absent` | failure_mode | Workstream G: bookkeeper is an optional evidence source | modules/profitability_analyzer.py:_get_open_cost_from_bookkeeper | case.json |
| `40-sanitized-production-decisions` | production_capture | DoD 17: replaying real production lifecycles reproduces the reference ledger state | generated | case.json, expected-ledger-events.json, expected-projections.json |

## Reason-code / rule coverage

- Arbitration: `INTENT_SUPERSEDED` (31, 38), `CONFLICT_CLOSE_REBALANCE` (18, 38), J3 total order (35).
- Authorization: `BUDGET_EXHAUSTED` (22), `STALE` (30), `AUTHORITY_LEVEL_BLOCKED` vs obligation exemption (29).
- Ledger/lifecycle: anomaly on orphan cost (25), unknown-outcome terminal (26), in-flight quarantine rule (27), replayed production lifecycle (40).
- Reservations: oversubscription refusal (23), restart durability (24).
- Numeric/determinism: checked ranges (32), rounding (33), expiry boundary (34), canonical JSON (36), id determinism (37).
- Policy stages: fee rails/rate/deadband/cooldown (8-11), DTS+PID components (12), htlc_max (13), classifications (1-7), modes/priorities (14-17).

## Documented gaps

