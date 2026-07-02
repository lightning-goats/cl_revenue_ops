# Hive-Hints Audit — Plan

Date: 2026-07-02. Scope: verify the full hive-hint signal chain across two repos —
**producer** `/home/sat/bin/cl-hive` (the organism) → **transport** (CLN datastore /
hive-datastore keys) → **consumer** `/home/sat/bin/cl_revenue_ops` (fee controller,
rebalancer, routers). Adversarial posture: treat every "acting correctly" as a claim to refute.

## The question
1. **Production** — how is every hive hint produced? (source computation, schema, units,
   value bounds/caps, TTL, update cadence, datastore key.)
2. **Consumption** — how does revenue_ops read each hint? (getter, validation/clamp/TTL,
   neutralization on missing/poisoned, which decision it feeds: fee / rebalance / router.)
3. **Correctness** — is revenue_ops **acting on the hints correctly**?
   - documented bias caps enforced (metabolic, immune, exploration, fleet_fee_prior, etc.)
   - direction correct (a hint that says "raise/drain inbound" moves fees/liquidity the right way)
   - staleness/TTL honored (a stale hint neutralizes, doesn't drive a live action)
   - **no double-counting** (each hint applied on exactly ONE layer — e.g. `drain_direction`
     is routing-layer only, the fee controller deliberately abstains; verify every such split)
   - **fail-open / poison-resistance** (a missing/malformed/hostile hint can't crash a loop or
     drive an absurd fee/spend; absolute values bounded)
   - **no orphans** (produced-but-ignored hints that should drive action; consumed-but-never-
     produced getters that are dead)
   - **influence reaches the decision** (not computed-then-dropped, like an audit-flagged
     dead-signal).

## Method (parallel read-only auditors + findings ledger + refutation)
- Findings → `docs/audit/hive-hints/findings-ledger.md`
  (`ID | severity | dimension | repo:file:line | description | status`). Read-only audit;
  fixes are a follow-up the operator rules on (unless trivial-and-authorized).
- Refutation gate: a fresh-context adversary re-derives ≥2 "correct" verdicts and attacks the
  cap/TTL/fail-open/no-double-count claims. A confirmed miss ≥ Medium re-opens that area.

## Agents
- **A — Producer catalog (cl-hive).** Enumerate EVERY hint the organism writes: type, datastore
  key/namespace, schema, units, value bounds/caps, TTL, cadence, computation source. Flag any
  producer emitting out-of-contract values (wrong unit/sign, unbounded, never-refreshed).
- **B — Consumer catalog (cl_revenue_ops).** Enumerate EVERY hint revenue_ops reads
  (hive_hints.py + any other consumer): getter, validation/clamp/TTL/neutralization, and the
  exact decision site each feeds (fee_controller / rebalancer / routers / planner).
- **C — Correctness-of-action deep dive (cl_revenue_ops).** For each consumed hint answer the
  Correctness checklist above with code evidence: caps enforced, direction, TTL, double-count,
  fail-open, orphan, influence-reaches-decision. This is the core deliverable.
- **D — Contract conformance + reconciliation.** The hive contracts under
  `docs/contracts/*.md` (HIVE_HINTS, METABOLIC_INFLUENCE, IMMUNE_INFLUENCE,
  HIVE_REBALANCE_REPORTING, REVENUE_* producers) — do producer and consumer agree on
  units/TTL/caps/fields? Cross-map A↔B: orphans in both directions.
- **R — Refuter.** After A–D, attack the clean verdicts.

## Deliverable
`findings-ledger.md` + a report answering **"is revenue_ops acting on hive hints correctly?"**
with a per-hint verdict table (hint → produced? → consumed? → acted-on-correctly? → evidence),
and every gap/defect as a ledger finding.
