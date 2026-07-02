# Hive-Hints Audit — Report

Date: 2026-07-02. Method: 4 parallel read-only auditors (producer catalog, consumer catalog,
correctness deep-dive, contract+reconciliation) + a fresh-context adversarial refuter. Repos:
cl-hive (producer), cl_revenue_ops (consumer). Ledger: `findings-ledger.md`.

## Direct answer: is revenue_ops acting on hive hints correctly? — YES.

For **every money-affecting hint**, revenue_ops acts correctly, and the verdict SURVIVED
adversarial refutation. No High/Critical/Medium defect exists in how the CONSUMER acts:
- **Caps bind.** Fee authority: hive fee_bias (folding metabolic/immune, each re-clamped ±5%)
  × temporal, product clamped ONCE to [0.9,1.1] before a single application; exploration
  [0.75,2.0] only widens DTS draw-noise (never the fee), every draw rail-clamped
  [min_fee_ppm,max_fee_ppm]. Rebalance bias [0.85,1.15]; segment ±0.12; fleet_fee/optimal
  range-rejected at [1,10000].
- **Directions correct.** owner/secondary fee bias, sink/source rebalance bias, peak/quiet
  temporal, reputation open-mult — all the right sign (the prior traffic-layer sign error is
  fixed on the producer).
- **Staleness neutralizes at READ time** (not just poll): only fee_bias/rebalance_bias survive
  the bounded-fallback window; metabolic/immune/sections/membership neutralize when stale.
- **No double-count on money layers.** Each hive signal is applied on exactly one layer;
  drain_direction is routing-layer only (fee controller abstains, confirmed no callers). The
  single fee "double-touch" (peak_hours → temporal + DTS context) is real but bounded and
  rail-clamped (~+26% worst case, learner-data not hive authority) — Low.
- **Fail-open / poison-resistant.** Finite/range guards reject NaN/Inf/out-of-range pre-math;
  oversized/over-peer-count payloads rejected pre-parse; hive carries NO amount authority
  (rebalance amounts/budgets computed locally); a poisoned datastore blob or askrene layer
  cannot crash a loop or drive a fee/spend past the rails.
- **The sats-EV spend gate is independent of hive influence** — compounded pair.score reorders
  selection only; every executed rebalance still independently clears the hold-margin gate.

## Findings (16): 2 Medium, 12 Low, 2 Info — none overturn the verdict

Both Mediums are **producer-side / documentation**, not consumer-acting-wrongly:
- **HH-1 (Medium, producer):** cl-hive immune `_fresh_enough` uses `any(fresh)` vs the
  contract's all-fresh; a partially-stale immune effect can pass the consumer gate. REFUTER:
  **nil money exposure** — immune fee bias is hardwired to 1.0 (producer emits 0), immune
  rebalance bias only feeds ordering-only pair.score. Fix on the producer for hygiene.
- **HH-2 (Medium, doc):** the IMMUNE_INFLUENCE contract documents no numeric caps (the same
  P1-028/P6-006 doc-gap class that was fixed for exploration/fleet_fee_prior). Consumer/producer
  enforce caps; the DOC just doesn't state them, so the three-way check isn't doc-verifiable.
  Fix: document the immune caps (fee ±0.05, rebal ±0.15, open [-0.15,+0.10], closure ±0.15).

Actionable Lows worth the operator's eye:
- **HH-3:** metabolic+immune FEE bias is inert today — producer emits fee_bias_delta=0, so that
  path is dormant (rebalance/open influence flows). Intended default-off, or a gap?
- **HH-4:** metabolic/immune action_constraints (max_rebalance_burn_sats, allowed flags) are
  telemetry-only, never enforced. Deliberate (budget_authority: cl_revenue_ops) — confirm split.
- **HH-8:** hive-fleet askrene layer 15-min age cutoff vs hourly refresh (freshness intent inert).
- **HH-9:** capacity_planner reuses get_rebalance_bias as a revenue-forecast multiplier (wrong
  signal, bounded).
The rest are doc-naming drift (HH-10/11), a defense-in-depth note (HH-12), askrene
no-re-validation (HH-13), a minor TOCTOU (HH-14), and dead/orphan getters (HH-16, harmless).

## Signal-integrity confirmations (positive)
- All 7 hive contracts CONFORM functionally; no unit/scale/key mismatch; no producer→consumer
  clamp disagreement.
- Two intentional no-ops POSITIVELY confirm the design: drain_direction (routing-layer only) and
  optimal_fee_estimate (debug-only) — abstaining is correct; acting on them would be the defect.
- Every produced field cl-hive writes is read somewhere; the only true orphan is a double-dead
  segment_observations path (producer never emits + zero consumers).

## Recommended follow-ups (operator decides; none urgent)
1. Producer HH-1: change immune `_fresh_enough` to all-fresh (hygiene; nil money impact today).
2. Doc HH-2/HH-10/HH-11: document immune caps; correct fleet_fee_prior→fleet_fee_median naming;
   document metabolic closure-watch cap. (Cheap doc fixes; close the P1-028-class gap on immune.)
3. Confirm the intended-abstain items (HH-3 dormant fee bias, HH-4 unenforced constraints) are
   the desired producer/consumer authority split, and prune the dead getters (HH-16).
