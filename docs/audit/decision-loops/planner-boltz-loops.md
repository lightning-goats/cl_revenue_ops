# Phase 3 — Decision-Loop Verification: Capacity Planner (+ Boltz vacuity note)

Scope: end-to-end loops candidates → planner actions → execution/delegation →
outcomes, on HEAD cdb536a against the frozen hermes corpus
(`/home/sat/cl-mycelium-hermes`, snapshots 2026-06-09 → 2026-06-20 plus one
2026-07-01 snapshot; hole 06-21..06-30; planner ledger ids 953–962 fall in the
hole but are recovered via the 2026-07-01 snapshot's history window).
Sweep: `tools/audit/loop_sweep_planner.py` (all counts below reproduce from it).
Corpus-first; code cited to explain corpus behavior. Builds on
docs/audit/verification/capacity_planner.md (post-refutation) and
docs/audit/verification/boltz_manager.md.

Ledger reconstruction (union by action id across all revenue-planner-history +
revenue-planner-status snapshots): nexus-01 93 actions — 44 close/recommended,
18 defibrillate/completed, 31 fee_reduce/delegated; nexus-02 35 actions —
18 close/dry_run, 7 defibrillate/completed, 10 fee_reduce/delegated.
(= the known 0 opens, 44+18 closes, 25 defibs, 41 delegations.)

## Loop verdicts (summary)

| Loop | Verdict |
|---|---|
| Candidates → OPEN execution | **Vacuous** — 0 opens corpus-wide; candidate pool present (≤32, Phase 2) but never converted. Nothing end-to-end checkable. |
| Losers → CLOSE execution | **Intended config gating, not a broken handoff.** `planner_max_closes_per_cycle=0` in all 388 listconfigs snapshots on both nodes disables execution regardless of `execute_closes=true`. Details below. |
| Losers → FEE_REDUCE delegation → fee controller | **Functioning.** "Delegated" is a ledger-record-only handoff to the fee controller's independent zero-revenue descent; 14 of the 17 observable delegations where descent was possible produced a downward fee change within 24h. Details below. |
| Losers → DEFIBRILLATE → diagnostic rebalance → forwarding | **Handoff works; actuator never fires.** 22/25 completed defibs have the expected 1:1 diagnostic rebalance row (the 3 without are proven blocked shocks), but **0 of 22 recorded shocks succeeded** — the "active shock" never delivered liquidity once in the entire corpus. Outcome table below (CP-H3 input). |
| Boltz swap loops | **Vacuous** — zero swap activity in all 2,454 spend-ledger/budget snapshots. See final section. |

---

## 1. The stalled close pipeline (top priority)

**Question:** why did 44 close recommendations on nexus-01 never execute despite
`execute_closes=true` for 118 snapshots?

**Verdict: intended gating by `planner_max_closes_per_cycle = 0` (the config
default), not a broken handoff.** Evidence:

- Code: `_close_execution_enabled` (modules/capacity_planner.py:232-236) returns
  True only when **both** `planner_execute_closes` is set **and**
  `planner_max_closes_per_cycle > 0`. The default is 0
  (modules/config.py:570 and :1013 — the cron-job baseline never raises it).
  With the gate off, `_execute_close` (:3550-3565) records the action and sets
  `status="recommended"` before any RPC; the budget check, rebalancer-job stop
  and `close` RPC (:3567-3654) are all downstream of the gate.
- Corpus: `revenue-ops-planner-max-closes-per-cycle = 0` in **all 194 + 194
  listconfigs snapshots** on both nodes (sweep check
  "L1: planner-max-closes-per-cycle == 0 …": 388 checked, 0 exceptions).
  nexus-01's `revenue-ops-planner-execute-closes` flipped false→true across
  76→118 snapshots (first `true` snapshot 2026-06-12T00:36Z) — the flip was
  **inert** because the second conjunct never held.
- Discriminator against the "gates blocked every attempt" hypothesis (Phase 2
  anomaly 2): all 44 nexus-01 close actions are `recommended` and **zero are
  `failed`**. Had `_close_execution_enabled` ever returned True, a blocked
  attempt would surface as `failed` (budget path :3572-3588) or `completed`.
  The recommendation path also runs the per-peer cooldown (:440-447) but not
  the budget/fee-gate guards, consistent with what the ledger shows.
- nexus-02's 18 close/dry_run actions all date 2026-05-13/14 — a month before
  snapshot coverage — and are consistent with a `planner_dry_run=true` posture
  at that time (dry-run gate :3536-3548 sits above the execution gate); the
  corpus-era config is dry_run=false, execute-closes=false, max-closes 0.

**Secondary finding — staging dried up after the 2026-06-11 dead-capital
rework (e5297c7, "Turn dead-capital staging from timers into priced
interventions"):** the nexus-01 staging timeline is 06-07 ×20 DEAD_CAPITAL,
06-09 ×19 DEAD_CAPITAL + 1 STAGNANT+HARD_REBAL, 06-10 ×1 DEAD_CAPITAL, then
only STAGNANT+HARD_REBAL ×1 on each of 06-11/06-12/06-13, and **nothing after
06-13**. Zero DEAD_CAPITAL close stagings exist after the rework landed. Only
2 of the 44 recommendations postdate the execute_closes flip. So even raising
`planner_max_closes_per_cycle` today would execute almost nothing: post-rework,
the dead-capital machine holds channels at DEFIBRILLATE (F4b demotion
:1288-1319 — CLOSE requires an executed diagnostic attempt in 14 days — plus
`close_protection` and the priced staging introduced by e5297c7). Which of
those three holds each channel is not observable from the corpus (open question
for Phase 4; note F4b's `_dead_capital_defib_attempted` :1168-1183 counts
*attempts* ≥1, and failed diagnostic rows do exist, so F4b alone cannot explain
zero post-rework CLOSE stagings).

**Anomaly (operator surface):** `revenue-planner-status` reports
`execute_closes: true` — a raw echo of the flag (capacity_planner.py:190) —
while effective close execution is disabled by the other config. The status
surface should expose `_close_execution_enabled` (and the per-cycle limit) or
operators will keep flipping an inert switch.

## 2. FEE_REDUCE delegation handoff

**"Delegated" to what?** To nothing that receives a message. F4c
(`_record_fee_reduce_delegation`, capacity_planner.py:1185-1233, called on the
none→fee_reduction stage entry at :1279) only writes a planner_actions row
(`status="delegated"`, estimated_cost 0) and anchors the dead-capital stage
timer. There is no RPC/API call into the fee controller. The actual fee
reduction is the fee controller's **zero-revenue descent**
(modules/fee_controller.py DTS `zero_revenue_streak` machinery, :668-707),
which walks fees toward `min_fee_ppm` on any channel earning nothing —
independently of the planner. The loop contract is therefore: *the channel is
dead capital, so the descent should already be acting on it*.

**Corpus quantification (41 delegations: 31 nexus-01, 10 nexus-02).** Primary
evidence: union of `recent_fee_changes` rows from all revenue-status snapshots
(nexus-01: 3,610 rows, ids 88018–97169, 5,542 in-range ids unobserved —
burst overflow of the 10-row window plus the hole; nexus-02: 455 rows, 1
missing). Fallback: per-channel `fee_proportional_millionths` timelines from
613/614 listpeerchannels snapshots (~30 min cadence). Fee floors from
listconfigs: nexus-01 `min-fee-ppm=60`, nexus-02 `=10`.

| Outcome within 24h of delegation | n1 | n2 | total |
|---|---|---|---|
| fee moved **down** | 12 | 2 | **14** |
| no change — already at the configured floor (60 ppm) | 7 | 0 | 7 |
| no change, above floor | 0 | 2 | 2 |
| changed but **not** down | 1 | 0 | 1 |
| unobservable (24h window in/next to the corpus hole) | 11 | 6 | 17 |
| **total** | 31 | 10 | 41 |

Of the 17 observable delegations where descent was *possible* (excluding the 7
already sitting at nexus-01's 60 ppm floor), **14 (82%) produced a downward fee
change within 24h** — the delegate does its job. The 3 exceptions:

- **n2 ids 293, 294** (2026-06-17, member channels 944921x2901x0/2899x0 at
  60 ppm vs floor 10): no change for 24h. Descent runs on the controller's own
  sampling cadence; both channels had already descended after their first
  delegations (291/292 on 06-13: 1517→1211, 115→66). Not a broken handoff,
  but "within 24h" is not guaranteed.
- **n1 id 947** (2026-06-19, member channel 944921x2901x0 at 60 ppm): the only
  *contradictory* case — fee change rows 96690/96691 (06-20 04:05/04:31) moved
  the fee **up** 60→110→210 via `dts_pid_sample` on a dormant zero-revenue
  channel ~23h after a "descent" delegation, with no zero-flow guard token in
  the reason string (other rows in the same window show
  `guard=zero_flow_ratchet_guard`/`zero_flow_downshift`). This is precisely the
  behavior targeted by the pending zero-flow ratchet work
  (docs/prompts/fix-dts-zero-flow-fee-ratchet-codex-2026-06-15.md). The
  delegation reason string ("fee reduction delegated to … zero-revenue
  descent") over-promises what the delegate does: DTS can sample upward on a
  dead channel.

Sanity checks passed: all 41 delegation records are zero-cost (estimated_cost
0, no amount), matching the record-only semantics. Note id 980 (n1, 07-01,
member channel 933791x3241x0) descended 60→**0** — the hive-member zero-fee
gate (8630ca6), not the descent floor, by then active on nexus-01.

## 3. Defibrillation outcomes (CP-H3 pre-work)

Code path: `_execute_defibrillation` (capacity_planner.py:3434-3512) →
`rebalancer.diagnostic_rebalance` (modules/rebalancer.py:2366-2495): set
`bounded_low_fee` probe flag (passive lure), then a 50k-sat "active shock"
capped at 100 sats fee / 2000 ppm, recorded as a `rebalance_type='diagnostic'`,
`reason_code='defibrillator'` history row. **Critical code caveat:**
`diagnostic_rebalance` returns `success: True` — hence planner
`status="completed"` — even when the shock is *blocked* (capital controls
:2442-2447, or no source :2398-2402), and in those paths **no rebalance row is
recorded**. "completed" means "defibrillator sequence triggered", not "shock
delivered".

**Join results (25 completed defibs → diagnostic rebalance rows,** union of
`recent_rebalances` across all revenue-status snapshots; nexus-01 rows
442–478 with only id 468 (≈06-21, in the hole) missing, nexus-02 rows 3–17
complete — so id continuity proves where no row exists):

- 22/25 matched exactly 1:1 (to_channel match, |Δt| ≤ 600s; nexus-01 15/18,
  nexus-02 7/7).
- 3/25 recorded **no row** (all nexus-01): ids 933 and 937 are corroborated
  capital-control blocks — the total-cost-budget surface within ~90 min shows
  24h spend 2,958 and 2,803 sats against an effective budget of 612
  (remaining 0); id 975 (06-28) sits in the snapshot hole (no budget surface)
  but rows 475 (06-27) and 476 (06-29) are contiguous, so no row exists —
  blocked shock, cause unobservable.
- **Shock success rate: 0/22.** Every recorded diagnostic attempt failed:
  10× `native_route_over_budget` (cheapest route 118–363 sats > the 100-sat
  cap), 10× `native_sendpay_error` WIRE_TEMPORARY_CHANNEL_FAILURE, 2×
  `route_pricing_failed` (askrene RPC, nexus-02 June 2–3). No diagnostic row
  has a settled fee — the defib program spent ~0 sats and delivered 0 sats of
  liquidity corpus-wide.
  **ADDRESSED (operator ruling D4, 2026-07-01):** the 100-sat cap finding is fixed —
  the cap is now configurable (`diagnostic_rebalance_max_fee_sats`, default 400 sats,
  covering the observed 118–363 sat routes) with the ppm ceiling derived from it;
  see docs/audit/operator-decisions.md D4 and RB-I10 in contracts/rebalancer.md.

**Per-channel outcome table** (settled forwards from the deduplicated
listforwards chain, lossless through 2026-07-01T20:35Z, so 7d windows are fully
covered for defibs up to ~06-24 and partially after; `out`=out_channel
forwards, `in`=in_channel forwards; fee in msat):

nexus-01 (18 defib actions, 11 distinct channels):

| action | channel | defib at | shock | fw ≤7d out/in | fw ≤14d out/in (out-fee) | 7d/14d coverage |
|---|---|---|---|---|---|---|
| 921 | 950276x2878x0 | 06-11 01:25 | failed route_over_budget 126>100 | 0/0 | 0/0 | full/full |
| 923 | 938435x2419x2 | 06-12 01:25 | failed route_over_budget 363>100 | 0/0 | 0/0 | full/full |
| 928 | 941523x1334x0 | 06-13 01:25 | failed WIRE_TEMPORARY | 0/0 | 0/0 | full/full |
| 933 | 941347x1139x0 | 06-14 00:27 | **NO ROW — budget exhausted (2958>612)** | 0/0 | 11/10 (49,290) | full/full |
| 937 | 944754x1796x0 | 06-14 23:23 | **NO ROW — budget exhausted (2803>612)** | 1/19 | 1/19 (150,012) | full/full |
| 940 | 941347x1139x0 | 06-15 23:23 | failed WIRE_TEMPORARY | 11/10 | 11/10 (49,290) | full/full |
| 944 | 953567x1632x0 | 06-16 22:12 | failed route_over_budget 184>100 | 0/0 | 0/0 | full/full |
| 946 | 944346x755x2 | 06-18 00:16 | failed WIRE_TEMPORARY | 0/1 | 0/1 | full/partial |
| 951 | 944346x755x2 | 06-19 04:57 | failed WIRE_TEMPORARY | 0/1 | 0/1 | full/partial |
| 952 | 931308x1256x0 | 06-20 00:43 | failed route_over_budget 227>100 | 1/0 | 4/4 (28,806) | full/partial |
| 963 | 931308x1256x0 | 06-24 16:30 | failed route_over_budget 224>100 | 4/2 | 4/4 (28,806) | full/partial |
| 964 | 931308x1256x0 | 06-25 21:15 | failed route_over_budget 222>100 | 4/4 | 4/4 (28,806) | partial/partial |
| 968 | 953567x1632x0 | 06-26 17:30 | failed route_over_budget 149>100 | 0/0 | 0/0 | partial/partial |
| 970 | 953567x1631x0 | 06-27 12:54 | failed route_over_budget 118>100 | 0/0 | 0/0 | partial/partial |
| 975 | 935804x2243x0 | 06-28 12:54 | **NO ROW — hole (no budget surface); id continuity ⇒ no row** | 0/1 | 0/1 | partial/partial |
| 978 | 955688x1618x0 | 06-29 12:54 | failed route_over_budget 269>100 | 0/0 | 0/0 | partial/partial |
| 979 | 944346x755x2 | 06-30 10:51 | failed WIRE_TEMPORARY | 0/0 | 0/0 | partial/partial |
| 981 | 955688x1618x0 | 07-01 10:51 | failed route_over_budget 267>100 | 0/0 | 0/0 | partial/partial |

nexus-02 (7 defib actions, 2 distinct channels — both hive-member channels,
D1): actions 289 (06-02), 290 (06-03), 295 (06-18), 296 (06-19), 297 (06-20),
298 (06-21), 305 (06-30); shocks: 2× route_pricing_failed (askrene), 5×
WIRE_TEMPORARY_CHANNEL_FAILURE; **forwards after every defib: 0/0 in both 7d
and 14d windows** (nexus-02's whole forwards chain contains 29 rows).

**CP-H3 implications for Phase 4:**

- Since 0 shocks succeeded, *any* post-defib forwarding is attributable at most
  to the passive lure (bounded_low_fee probe flag) or the concurrent fee
  descent — never to delivered liquidity. CP-H3 as "shock revives channels" is
  untestable on this corpus; only "defib episode (flag+descent) precedes
  forwarding" is testable.
- Signal available: 5 of 11 distinct nexus-01 defibbed channels saw settled
  forwards within 14d of a defib (3 with out-forwards earning 49.3 + 150.0 +
  28.8 ≈ 228 sats of fees: 941347x1139x0, 944754x1796x0, 931308x1256x0);
  6 of 11 saw nothing; nexus-02 0 of 2.
- Repeat-defib pattern (931308x1256x0 ×3, 944346x755x2 ×3, 953567x1632x0 ×2,
  955688x1618x0 ×2, n2 944921x2901x0 ×5): because the shock never lands, the
  stage machine re-defibrillates the same channels every 1–5 days
  indefinitely.

## 4. Cross-module invariant sweep

All from `tools/audit/loop_sweep_planner.py` (n1 = nexus-01, n2 = nexus-02):

| Invariant | Result | Vacuity/notes |
|---|---|---|
| (a) every defibrillate/completed has a matching diagnostic rebalance row | 22/25 matched; 3 unmatched all **explained** (2 proven budget-exhaustion blocks, 1 in-hole block proven row-less by ledger-id continuity) | Not vacuous. Root cause of the mismatch class is code, not data: `diagnostic_rebalance` reports success before `record_rebalance` on blocked paths (rebalancer.py:2398-2402, :2442-2447) — planner `completed` overstates execution |
| (b) no close ever completed (consistency with cooldown-consumed `recommended` rows) | PASS — 62/62 close actions are recommended/dry_run; 0 completed, 0 failed | Execution half vacuous **by configuration** (max_closes_per_cycle=0), which is itself the verified explanation |
| (c) no phantom action targets: action channel_id appears in listpeerchannels | PASS — 88/88 in-coverage actions (73 n1 + 15 n2) | 20 n2 actions (the 05-13/14 dry-run closes on 4 channels) predate snapshot coverage and are excluded — those channels were already gone by 06-09 |
| (d) member protection: no CLOSE at any stage/status on any ever-member peer, **fleet-wide** membership union (each node's hive-export-hints exclude the node itself, so per-node sets miss sibling-node peers; union = 5 peers) | PASS — 0/62 | Supersedes the Phase 2 per-node re-check with a strictly larger member set |
| (e) stage ordering: completed defib preceded by a FEE_REDUCE delegation for the same channel | 11/25 verified ordered (6 n1 + 5 n2); 0 provable violations | **Largely indeterminate**: F4c recording landed 2026-06-11 (e5297c7) and fires only on the none→fee_reduction ENTRY, so the 12 n1 defibs on channels staged before the deploy and 2 pre-F4c n2 defibs legitimately lack records |
| (L5) boltz spend/events zero in every snapshot | PASS — 2,454/2,454 (1,226 n1 + 1,228 n2) | Confirms **vacuity**, not correctness |

**D1 exposure update (fleet-union membership, supersedes Phase 2's
nearest-snapshot counts of 3 defibs + 13 delegations):** member
defibrillations = **7** (all on nexus-02: 289, 290, 295, 296, 297, 298, 305 —
both defib targets on n2 are member channels; the 4 extra vs Phase 2 are the 2
pre-coverage and 2 in-hole actions its 24h-nearest-member-snapshot method could
not check); member FEE_REDUCE delegations = **22** (12 n1 + 10 n2 — 10 of 10
n2 delegations are member channels). D1's blast radius is materially larger
than Phase 2 reported; the removal-candidate ruling stands, reinforced.

## 5. Boltz loop note (dormant corpus-wide)

Zero Boltz activity anywhere: `spent_by_category.boltz` and
`event_count_by_category.boltz` are 0/absent in all 1,227 spend-ledger
snapshots and the boltz budget component is 0 in all budget snapshots (both
nodes; re-confirmed by this sweep's L5 check, 2,454 surfaces). Consequently
**every observational BM loop handoff is unverifiable on this corpus**:

- quote → loop_in/loop_out execution (and rejected-status wiring, BM-I2);
- unified-budget gate under the swap-creation lock (BM-I3/BM-H2 — Phase 2's
  BM-H2 "pass" is trivial);
- tactical/channel-capex gating and the structural-envelope bypass
  (BM-I4/I5/I6), including the **chainswap gate-bypass caveat** (unified gate
  only, no tactical/channel gates — untested anywhere in tests/);
- exactly-once spend recording into the "boltz" category (BM-I7) and
  pending-swap reservation (BM-I8);
- external-pay pinning/retry with budget re-check (BM-I12) and the
  over-payment guard (BM-I13).

What already covers these: docs/audit/verification/boltz_manager.md — all 14
BM invariants verified on tests+code (184 tests passing across 7 files), with
BM-I1/I10/I11/I14 code-only (no covering tests) and the BM-I3 serialization
test's vacuous-pass caveat. If Boltz is ever enabled, this loop needs a fresh
corpus before any end-to-end claim can be made; nothing further is checkable
now.

## Gaps / Anomalies

1. **Inert operator switch + misleading status surface.** nexus-01 ran
   `execute_closes=true` for 118 snapshots with zero effect (gated by the
   max-closes default 0), and `revenue-planner-status` echoes the raw flag
   (capacity_planner.py:190) rather than effective enablement. Fix candidates:
   surface `_close_execution_enabled` + limit in status; or warn at startup
   when execute_closes is set with a zero limit.
2. **Planner "completed" overstates defib execution.** Blocked and failed
   shocks both yield planner-visible success semantics ("completed" covers:
   shock failed, shock blocked by budget, no source). Any Phase 4 CP-H3 test
   keyed on planner status alone will overcount interventions by 100% of
   observed cases (0 real shocks vs 25 "completed").
3. **The shock is under-budgeted for its own targets.** 10 of 22 recorded
   failures are route_over_budget with cheapest routes 118–363 sats vs the
   hardcoded 100-sat cap (rebalancer.py:2431) — for these dead channels the
   defibrillator as parameterized *cannot* deliver, and the stage machine
   re-burns a defib slot + peer cooldown on the same channels every few days.
4. **Budget-overspend tension resurfaces (feeds the budget-loop verifier).**
   The total-cost surface showed 24h spend 2,958/2,803 sats vs effective
   budget 612 around 06-14/15 — same class as Phase 2's unresolved 129-snapshot
   finding — and here it demonstrably *did* gate: two defib shocks were blocked
   by capital controls at exactly those times. Whatever overspent, it was not
   the diagnostic path.
5. **Delegation delegate can move fees the wrong way.** n1 id 947: 60→110→210
   dts_pid up-samples on a dormant zero-revenue member channel within 24h of a
   "descent" delegation (no zero-flow guard token on those rows) — the pending
   zero-flow ratchet fix is the relevant work item.
6. **Post-rework close staging is fully dark.** Zero DEAD_CAPITAL CLOSE
   stagings after e5297c7 (06-11) despite defib attempts satisfying F4b's
   attempt_count≥1; whether `close_protection` or the new priced staging holds
   each channel at DEFIBRILLATE is not corpus-observable. Phase 4 should decide
   whether that is intended (needs code-level review of the e5297c7 EV gates).
7. Evidence-quality: the n1 fee-change ledger reconstruction misses 5,542
   in-range ids (10-row window overflow + hole) — delegation outcomes lean on
   the listpeerchannels fee timelines (~30 min cadence), which cannot see
   intra-snapshot churn; 17 of 41 delegation windows and all 06-21..06-30
   planner activity except via the 07-01 history window are unobservable.
