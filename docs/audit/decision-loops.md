# Phase 3 Rollup — End-to-End Decision-Loop Verification (completed 2026-07-01)

Per-loop reports: docs/audit/decision-loops/{fee-loop,rebalance-loop,planner-boltz-loops}.md.
Sweeps: tools/audit/loop_sweep_{fee,rebalance,planner}.py (all reproduce deterministically
over the frozen corpus). Method per loop: handoff-integrity checks at snapshot granularity,
4-10 cross-module invariants swept with vacuity labels, and hand-audited real episodes.

## Loop verdicts

| Loop | Verdict |
|---|---|
| Fee (flow_analysis → fee_controller → gossip → forwards) | **COHERENT** — every observable handoff agrees; n2 exact (407/407, 3,689/3,689, 453/453 chain links); the only violation cluster is an external fee writer, not the plugin |
| Rebalance: budget half | **WORKS end-to-end** — its one live stress event (operator mass close, 06-13) bred the correct fleet-wide suppression and resumption; no enforcement hole |
| Rebalance: liquidity half | **INERT** — run_cycle selected/executed 0 pairs in 1,227 cycles; 177/178 candidates below_hold_margin; genuine automated attempts in corpus = 1 |
| Planner: close pipeline | **Intended gating** — planner_max_closes_per_cycle=0 everywhere makes execute_closes=true inert; 0/62 closes completed by design |
| Planner: fee_reduce delegation | **WORKS** — 14/17 observable delegations descended within 24h (one up-ratchet counter-case = known pending-fix failure mode) |
| Planner: defibrillation | **BROKEN as a diagnostic** — 0/22 recorded shocks delivered liquidity, yet all marked "completed" |
| Boltz | **Vacuous** — zero activity in 2,454 surfaces; rests entirely on Phase 2 tests+code |

## New defects found only at loop level (add to follow-up fix list)

1. **Defibrillation placebo + status lie** — shocks fail (route 101–363 sats vs 100-sat
   cap; WIRE_TEMPORARY; capital-control blocks) but the planner records
   `completed` even for blocked/failed shocks (rebalancer.py:2398-2402, :2442-2447,
   :2485-2488) and `actual_cost_sats` is never backfilled. The 100-sat cap is below the
   observed market price of probing suspected-dead channels — the probe confirms the
   illiquidity it tests, paying attempt costs for zero information gain recorded.
2. **Planner status surface echoes raw flags** — `execute_closes: true` shown while
   effective close execution is disabled by max_closes_per_cycle=0.
3. **fee-debug lies about unmanaged channels** — a channel that silently left the
   managed set (restart, 06-15) is still reported "ready" with growing lb staleness
   117+ h; nothing records management opt-outs.
4. **`guard=zero_flow_downshift` stamps floor-driven raises** — the guard's floor arm
   emits the downshift tag on upward moves; should emit `floor_override`.
5. **`recent_fee_changes` (10-row RPC echo) loses 61% of nexus-01's change records** —
   any auditing/scorecard must read the fee_changes DB table, not the RPC echo.
6. **Twin-row recording defect** (pre-06-10) observed firing in-corpus; the 62ae545 fix
   verified holding. Historical stats must dedup: "4 ev_positive normals" is really 1.

## Decisive observational evidence (Phase 4 inputs)

- **Fee elasticity is real and the controller overprices dead channels**: DTS twice
  laddered 946890x2272x0 to 2000–2300 ppm where forwards ceased; an external reprice to
  ~100–250 ppm returned it to the earning region (~2,900 sats) while the controller's
  belief went stale. FC-H1-adjacent evidence with the *treatment done by a third party*.
- **Zero-flow ratchet fix (071a5b3) verified live**: first guard change 86 s after the
  deploy stamp; 244/246 subsequent guard changes in the correct direction; a clean
  fleet-wide unwind episode (720→61 ppm in 15 exact ×0.85 steps).
- **CP-H3 (defib efficacy) is untestable as designed**: no shock ever delivered
  liquidity, and the false-positive risk is proven (941347x1139x0 settled 57 forwards
  in the 5 days after a *failed* defib; nexus-02's 7×-defibbed pair never routed).
- **RB/RE contribution hypotheses are vacuous** at n=1 genuine automated attempt; the
  live question is hold-margin calibration (177/178 rejections under an 84%-sink
  liquidity distribution), not execution correctness.
- **D1 exposure (fleet-union membership): 7 member defibrillations + 22 member
  fee_reduce delegations** — larger than Phase 2's time-matched count.
- Budget suppression semantics: all 550 "suppressed" decisions in the corpus are one
  event (the mass close). Contribution analysis must not read suppression frequency
  as routine budget exhaustion (corrects a Phase 2 anomaly note).

## Evidence limits

Snapshot window ~12 days (06-09→06-20 + one 07-01 capture), 10-day hole, nexus-02
routed nothing, spend ledger all zeros, boltz dormant, pause/concurrency/congestion
paths never fired. Cross-module invariants that were vacuous are labeled so in each
sweep; nothing vacuous was counted as positive evidence.
