# EV Coverage Matrix — Workstream F1 Audit (PR 5, 2026-07-13)

Phase E of `docs/planning/refactor-gap-closure.md`. Documents, per
action class, the REAL economics the repo computes today against the
spec's common contract:

```text
expected_value = expected_incremental_revenue
               - expected_execution_cost
               - expected_capital_cost
               - risk_premium
```

Finding in one sentence: **every spend path already computes a real,
often richer EV — none of it reaches the intent envelope** (all
`expected_benefit_msat`/`confidence_micro` are zeros; gap-closure Gap
6). This audit maps each path's terms to the contract and defines the
PR 6 population plan. Documentation only — no behavior changes.

## Matrix

### 1. Rebalance (normal redistribution)

| Contract term | Implementation (rebalance_engine_v2.py:565–621, gate :1450–1478) |
|---|---|
| Incremental revenue | `expected_future_value_sats` = destination-refill value + source-drain value, forward-history-derived |
| Execution cost | `expected_fee_sats` (route-priced) |
| Capital cost | `capital_risk_penalty` |
| Risk premium | `failure_penalty` + `p_success` weighting |
| EV | `final_score_sats` (model `v2-sats-ev`) |
| Hold margin | `rebalance_hold_margin` (sats) — `below_hold_margin` rejection :1478 |
| Confidence | `p_success` (attempt-history posterior) |
| Hard constraints | unified budget reservation (governor, authority `liquidity`), close-conflict registry, cooldowns, protection overrides |
| EV role | **AUTHORIZES** (hard gate) + ranks (`pair.score` ordering) |

Zero-cost moves bypass the gate by construction (`beats_do_nothing`,
:577 — "can never lose to do-nothing").

### 2. Channel open

| Contract term | Implementation (capacity_planner.py:2859–2930, :594–600) |
|---|---|
| Incremental revenue | expected lifetime revenue: closed-channel profit inheritance, else observed node daily-ppm anchor × `NEW_PEER_DISCOUNT`, ceiling-clamped |
| Execution cost | on-chain open(+close) cost estimate |
| Capital cost | expected rebalance costs over lifetime (netted in EV) |
| Risk premium | new-peer discount + reputation/uptime score shaping (`_score_candidate`:2110) |
| EV | `_calculate_open_ev` return (sats): revenue − on_chain − rebalance costs |
| Hold margin | EV > 0 |
| Confidence | implicit (discounts); no explicit term |
| Hard constraints | feerate/mempool gate, per-peer exposure cap, failed-open backoff, portfolio-balance governor, capital reservation (authority `capital`) |
| EV role | **AUTHORIZES** + ranks candidates |

### 3. Channel close / recycle

| Contract term | Implementation (capacity_planner.py:402–463) |
|---|---|
| Incremental revenue | REDEPLOYMENT framing: freed capital's value at winners (F4a `_apply_redeployment_ev_demotion` :406) |
| Execution cost | close+reopen chain cost (priced in redeployment demotion) |
| Capital cost | the loser's negative `marginal_roi` (bleed rate) |
| Risk premium | none explicit; Kalman-confidence protection acts as an evidence gate instead |
| EV | ranking metric: worst `marginal_roi` first (:463), demoted when redeployment EV is not positive |
| Hold margin | protection gates + dead-capital staging thresholds |
| Confidence | Kalman flow confidence (protection veto when unreliable) |
| Hard constraints | protection service (inbound gateway, sourced fees, route pairs, 30d-window close gate) — VETOES, not EV terms |
| EV role | **RANKS** (worst-first) with EV-based demotion; protections authorize |

### 4. Boltz swap-out / swap-in

| Contract term | Implementation (cl-revenue-ops.py:9240–9390) |
|---|---|
| Incremental revenue | `expected_gross_uplift_sats` (DTS-uplift volume model) + `structural_uplift_sats` (amortized structural credit, :9290–9306) |
| Execution cost | `estimated_fee_sats` (live quote via `get_boltz_cost_components`) |
| Capital cost | bounded by daily structural envelope / unified budget; no per-swap term |
| Risk premium | loop-in `non_pinned_penalty` 0.7 → `risk_adjusted_net_sats` (:9315) |
| EV | `expected_net_sats` = gross uplift − fee |
| Hold margin | `required_profit_threshold_sats`; loop-in additionally requires risk-adjusted net > 0 |
| Confidence | posterior-std gate on the structural premium (loose posterior → no credit, :9290) |
| Hard constraints | capex/treasury budget reservation (governor, authority `capital`), cooldowns, band separation |
| EV role | **AUTHORIZES** (`passes_profit_guard`) + multigoal ranking |

### 5. LN+ participation

| Contract term | Implementation (lnplus_swaps.py:514–563) |
|---|---|
| Incremental revenue | `outbound_ev` (reuses open-EV) + `inbound_credit × reliability × lnplus_inbound_credit_factor` |
| Execution cost | open cost (netted inside `_calculate_open_ev`; capex gates use it directly) |
| Capital cost + risk premium | `lockup_haircut = P_UNDERPERFORM × best_regular_ev × duration/12` — opportunity cost of locked capital, blended with risk |
| EV | `_swap_ev` value |
| Hold margin | must beat `best_regular_ev` by `lnplus_swap_preference_margin` |
| Confidence | `reliability` (ratings floor + Tor discount) |
| Hard constraints | gates 0–9 (feerate ceiling, peer quality veto, ring size, ban veto); post-acceptance = CONTRACTUAL OBLIGATION |
| EV role | **AUTHORIZES + RANKS** pre-application; NONE post-acceptance (obligation — exception class 2) |

### 6. Hot-channel protection / budget uplift

Mode descriptor `hot_protection`, priority 90 (rebalance_modes.py:48–60,
Phase 3D): same `v2-sats-ev` model as class 1; difference expressed as
PRIORITY and budget allocation per the spec's F4 table, not a separate
EV. EV role: ranks within the shared model; priority modifies ordering.

### 7. Structural drain

Class 4's structural credit path (`structural_uplift_sats`, scarcity ×
amortization horizons) plus the structural budget envelope; Boltz
structural loop-outs and drain-mode rebalances share the class 1/4
models with mode priority. EV role: **AUTHORIZES** via profit guard
with the structural credit term.

### 8. Growth experiments

Growth budget machinery (`growth_budget_*`: earned fraction, experiment
fraction, max extra, hard ceiling). DELIBERATELY not EV-gated:
exploration exists to buy evidence where EV estimates have none.
Bounded instead by hard budget caps and the earned-fraction coupling.
EV role: **NONE (deliberate)** — bounded-exploration exception,
analogous to DTS exploration in fees.

### 9. Reserve maintenance (expansion treasury)

Deficit-driven (`expansion_treasury_min_deficit_sats` threshold toward
`onchain_target_sats`), with treasury filters and failsafe. Solvency
maintenance, not yield-seeking: EV role **NONE (threshold-gated)** —
classified reversible-solvency maintenance; swap execution still passes
the class 4 cost machinery and governor reservation.

### 10. Fee / HTLC-policy actions

Zero-cost reversible mutations (ADR-001): execution and capital cost
are zero; the DTS posterior IS the revenue-EV estimator (sampling
optimizes expected revenue directly); PID manages liquidity risk.
`htlc_max` admission control is liquidity/safety, not EV. EV role:
**controller-internal**; envelope EV deliberately 0 — exception class
4 (reversible policy maintenance). Governor contributes pause/authority
gating and audit trail, no budget reservation (zero-cost skip).

## Exception classification (spec-required)

| Exception class | Members |
|---|---|
| Protocol safety | htlc_max reserve floors; fee rails; anchor-reserve floors in budget caps |
| Existing contractual obligation | LN+ accepted-swap fulfillment (`CONTRACT_OBLIGATION` reason code; ungated by pause/authority per invariant 6; still governed + ledgered) |
| Reconciliation / recovery | econ reconciler corrections; startup reservation cleanup; Boltz journal reconciliation |
| Reversible policy maintenance | SET_FEE, htlc_max, gossip refresh (zero-cost governor path) |
| Operator-directed manual | manual RPCs (revenue-set-fee manual=True, revenue-rebalance, manual swaps) — bypass EV, never bypass ledger/audit conventions |
| Bounded exploration (documented addition) | growth experiments; DTS exploration windows; diagnostic (defibrillation) rebalances bounded by `diagnostic_rebalance_max_fee_sats` — spend bounded evidence purchases, not EV claims |

## Envelope population plan (feeds PR 6, flag `econ_ev_populated`)

| Path | expected_benefit_msat | confidence_micro | notes |
|---|---|---|---|
| Rebalance | `expected_future_value_sats × 1000` (signed) | `p_success × 1e6` | max_cost/capital already real |
| Open (planner) | `_calculate_open_ev × 1000` | 0 (no explicit term — documented) | |
| Close (planner) | redeployment-EV delta where computed; else 0 with reason code | Kalman confidence when available | needs definition pass in PR 6 |
| Boltz | `risk_adjusted_net_sats × 1000` | posterior-tightness proxy | |
| LN+ | `_swap_ev × 1000` | `reliability × 1e6` | obligation path stays exception |
| Fees | 0 (exception class 4) | 0 | ADR-001 |

**Ordering impact (the PR 6 flip risk — CORRECTED during
implementation):** the J3 ladder sorts `-EV` before `-confidence` and
target, but only the REBALANCE loop consumes J3 output order — the
Boltz and planner stages deliberately preserve legacy plan order among
survivors. So flipping `econ_ev_populated` reorders execution in the
rebalance loop only (richest-EV first, pinned by test); for Boltz and
planner closes the populated EV is evidence, order unchanged (pinned).
Adopting J3 order for those two loops is a separate, explicit decision.
The flip still requires operator approval (gap-closure §F.2).

**PR 6 implementation record (2026-07-13):** `modules/econ_ev.py`
(checked contract helpers, conservative missing-data rules) +
`econ_ev_populated` flag (default off, count 60). Populated:
rebalance batch + governed rebalance reservations
(`final_score_sats`/`p_success` from `pair.score_decomposition`),
Boltz batch (`risk_adjusted_net_sats`). Deliberately zero: planner
closes (benefit undefined pending a definition pass), planner opens
(EV not threaded to `_execute_open` — candidate for a later pass),
LN+ obligation and fee paths (exception classes 2 and 4). All
directive property tests pass (monotonicity ×4, conservative missing
data, checked integer semantics).

## Duplicate "worth doing" booleans (retirement candidates)

`beats_do_nothing` (rebalance), `passes_profit_guard` (Boltz), `EV > 0`
(planner opens), preference-margin comparison (LN+). Per the directive,
these may be removed ONLY where the common contract fully replaces them
without behavior change. Audit verdict: **retain all four** until the
populated envelope EV has run governed in production and the arbiter's
EV gate provably reproduces each verdict (dual-run comparison); the
booleans and the envelope share source data, so premature removal risks
silent semantic drift (e.g. the zero-cost bypass in class 1 and the
loop-in risk adjustment in class 4 are NOT plain `EV > margin` checks).

## Required PR 6 property tests (from the directive)

Monotonicity: higher execution cost, capital cost, or risk premium can
never raise EV; a higher capital hurdle can never make an identical
open more attractive. Conservative failure: missing cost or confidence
fields fail toward rejection/zero-benefit. Exceptions stay explicit and
governed. All EV envelope fields use checked `SignedMsat`/`Micro`
fixed-point (econ_types) — never floats on the wire.
