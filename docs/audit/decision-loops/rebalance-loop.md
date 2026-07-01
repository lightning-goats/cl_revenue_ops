# Phase 3 — Rebalance + Budget Loop (end-to-end)

Verifier: Phase 3 loop auditor, 2026-07-01, HEAD cdb536a. Corpus: frozen hermes
snapshots `/home/sat/cl-mycelium-hermes` (5,196 snapshot dirs, 2,598/node;
surfaces 2026-06-09 → 06-20 + one 2026-07-01 pair; rebalance_history rows reach
back to 2026-04-14 via the persisted table read through `recent_rebalances`).
Sweep: `tools/audit/loop_sweep_rebalance.py` (all numbers below reproduced by it
unless cited otherwise).

## Loop verdict

**The budget half of the loop works end-to-end and is genuinely enforced; the
liquidity-moving half of the loop is inert.** State → decision → execution →
accounting → feedback all connect and behaved correctly in the one live stress
event the corpus contains (the 06-13 mass-close budget saturation). But the
engine's automated path (`run_cycle`) **executed zero rebalances in 1,227
observed cycles** — every history row in the snapshot window came from planner
defibrillation, manual RPC, or a since-fixed double-recording defect. The loop
as deployed is a *diagnostic/containment* system (zombie probing + spend
ceiling), not a liquidity allocator.

## 1. Budget reconciliation (the Phase 2 "129 snapshots" tension) — RESOLVED

**Verdict: no enforcement hole. Different categories on the same surface —
and the breach itself closed the loop by suppressing rebalancing.**

Mechanism (code):

- `_compute_total_cost_budget_status` (cl-revenue-ops.py:6523-6610) sums five
  categories into `actual_spent_sats`: rebalance, boltz, **open**
  (`get_opening_costs_since`), **close** (`get_closure_costs_since`,
  database.py:5525), ledger. Open/close are *retrospective on-chain accounting*
  — nothing reserves them against the daily budget before they confirm.
- The rebalancer's gate `_check_capital_controls` (rebalancer.py:2608-2676)
  blocks when `rebalance_fees_24h + external_spent >= effective_budget`, where
  `external_spent` comes from `_non_rebalance_liquidity_cost_components`
  (cl-revenue-ops.py:6628-6640) = `actual_spent_sats − rebalance category` —
  **so close costs feed straight into the rebalance gate.**

Evidence (corpus):

- All 129 over-budget snapshots (`actual_spent_sats > effective_budget_sats`)
  are on **nexus-01, 06-13 20:22 → 06-15 16:12**, and in every one of them the
  entire overage is the **`close` category** (760–2,958 sats vs 612 budget);
  rebalance/boltz/ledger are 0 (LP-I1a, 129/129).
- Cause: an operator-driven **mass mutual close of 11 channels at 06-13
  19:43Z** (listpeerchannels diff: 950127x1815x0, 949288x1886x0, 948030x1078x0,
  951283x3783x0, 949527x1260x0, 950276x2878x0, 945529x2331x0, 949958x2088x0,
  941604x3312x0, 949283x1800x0, 951165x2981x0 → CLOSINGD_COMPLETE/ONCHAIN).
  Several were standing planner `close/DEAD_CAPITAL` recommendations (all
  planner close actions remain status `recommended` — executed outside the
  plugin). Close fees accumulated 155 → 2,198 → 2,958 sats in the 24h window.
- The budget surface reported the breach honestly: `remaining_sats == 0` in all
  129 (LP-I1b), and the **same snapshot's `rebalance_decision` is
  `suppressed / capital_controls_blocked / budget_blocked=true` in all 129**
  (LP-I1c). All 550 suppressed decisions in the whole corpus are exactly this
  event (nexus-01, 06-13: 46 / 06-14: 299 / 06-15: 205; the 129 vs 550 count
  difference is capture frequency — budget surface is snapshotted ~4× less
  often than revenue-status).
- Enforcement reached the write path: **no normal or diagnostic history row was
  created inside the saturated interval** (LP-I2, 41/41 rows checked). Two
  planner defibrillations *inside* the window — actions 933 (941347x1139x0,
  06-14 00:27) and 937 (944754x1796x0, 06-14 23:23) — completed with **no
  history row**; row-id contiguity (462→463 spans both) proves the shock was
  never attempted: that is `diagnostic_rebalance`'s capital-controls block
  (rebalancer.py:2442-2447) firing in production. The only rows near the window
  are the five **manual** attempts 458-462 at 06-13 19:47-19:55 — manual
  bypasses capital controls by design (RB-I9) and all failed with zero spend.
- Phase 2's RB-I1c ("rebalance-category 24h spend ≤ effective budget",
  1,227/1,227) does not contradict any of this because it is **vacuous**: the
  max rebalance-category spend anywhere in the corpus is **0 sats** (LP-I3) —
  both successes carried fee 0 and failed attempts spend nothing.

Ruled-out hypotheses from the Phase 2 handoff:

- *Hot-channel budget raises*: never fired — `effective_budget_sats ==
  daily_budget_sats` in all 1,227 budget snapshots (mode `fixed` throughout);
  the hot-channel override only widens the per-candidate reservation limit
  (rebalancer.py:1980-1996), never the reported effective budget.
- *Effective-budget redefinition mid-corpus*: the only config change is
  nexus-01's `daily_budget_sats` 500 → 612 at 06-13T19:43 (same snapshot the
  first close cost lands — operator action during the close session). It
  neither created nor masked the breach (2,958 ≫ 612).
- *Manual+diagnostic spend*: no — manual/diagnostic fees were all zero; the
  overage is 100% close category.

**Classification: the daily total-cost budget is *prospectively enforced* on
discretionary liquidity spend (rebalance reservations, boltz, defib shocks) and
*retrospectively counted* for on-chain open/close costs. On-chain costs can
therefore breach the printed ceiling (they are settlements of on-chain reality,
not gateable by this surface), but the breach immediately starves all further
discretionary spend for the remainder of the rolling window. Enforced, with a
documented-by-behavior retrospective category — not a hole.** (The one genuine
hole in this neighborhood remains the *code-level* CB-4/RB fail-open on DB
error, Phase 2 finding — not observed firing in the corpus.)

## 2. Failure dissection — why 2 successes / 51 rows

Token × type taxonomy (51 deduplicated final rows):

| token | diagnostic | normal | manual | class |
|---|---|---|---|---|
| native_route_over_budget | 14 | 1 | 3 | pre-flight (economic: priced route > fee cap) |
| native_sendpay_error (all WIRE_TEMPORARY_CHANNEL_FAILURE) | 12 | 3 | 2 | on-path (liquidity) |
| route_pricing_failed (askrene-create-layer RPC timeout) | 7 | — | — | pre-flight (infra) |
| sling_preflight_error (sling-once RPC, legacy executor era 04-15/16) | 4 | — | — | pre-flight (infra) |
| no_fleet_route / retriable_failure(NoCandidates) | — | — | 2 | pre-flight (routing) |
| engine_busy | — | — | 1 | contention (single-flight lock) |
| success | — | — | 2 | — |

Split: **pre-flight 31 / on-path 17 / contention 1 / success 2.** The majority
of failures never risked funds; the budget/pricing gates and the 100-sat defib
fee cap did the rejecting.

Root cause of the completion rate, layer by layer:

1. **The automated engine never executes.** Across all 1,227 debug snapshots:
   `selected_pairs` total = 0, `execution_count` total = 0. All 178 considered-
   candidate exports (14 distinct hourly resamples, 3 distinct pairs — every
   one targeting dest 936056x1037x0) were rejected: 177 `below_hold_margin`,
   1 `route_over_budget`. Hold diagnostics show `source_inside_band` in 1,222
   snapshots — 84% of channel-states are `sink`, so almost no source ever
   leaves its band. The EV gate is working as specified; the fleet's liquidity
   distribution simply never produced a pair that cleared it.
2. **What does execute is designed to fail often.** 37 of 51 rows are
   defibrillator shocks into channels *suspected dead* with a hard 100-sat fee
   cap (RB-I10). Priced routes into those channels cost 101–363 sats
   (`route_over_budget: 101/118/126/149/180/184/222-227/263-269/363 > 100`) or
   die on-path with TEMPORARY_CHANNEL_FAILURE — i.e. the probe *confirming*
   the illiquidity it was sent to test. A failed defib is diagnostic signal,
   not a malfunction.
3. **The 2 successes are manual, zero-fee, hive-internal** (nexus-02 rows 5-6,
   04-14, 944921x2901x0 → 944951x1646x0, 300k + 100k sats at fee 0) —
   pre-dating snapshot coverage.
4. **The 4 "normal ev_positive" rows are not four automated attempts.** Row 442
   (05-30) is the only genuine engine-path attempt visible; rows 448/451/453
   are duplicate recordings of defib shocks 447/450/452 (below).

**Twin-row defect (found, already fixed).** Rows 447+448, 450+451, 452+453 are
pairwise identical to the second (same pair, amount, max_fee, timestamp, same
error text) — one physical attempt recorded as both a `diagnostic/defibrillator`
row and a `normal/ev_positive` row. All three episodes predate commit
**62ae545** (2026-06-10, "Guard rebalance cycles and fix budget/accounting
distortions"), which made the engine honor the caller's `rebalance_id`
(`_execute_pair`, rebalance_engine_v2.py:2726-2729) instead of inserting its
own pending row. Zero duplicates after 06-10 (LP-I5, 48/48). This also explains
Phase 2's "failed row with actual_fee_sats(101) > max_fee_sats(100)" anomaly
(row 453 = the engine-side twin bookkeeping the rejected route's price) and
deflates "ev_positive normals: 4" → **1**.

**The 63.120-sat duplicate-forward pair is NOT rebalance-related.** Forwards
updated_index 91,151/91,152 (nexus-01, 06-13T07:10, 931199x1231x0 →
946890x2272x0) are two equal 485,608.620-sat MPP shards of a *routed
third-party payment* (both settled, 63.120 sats fee each). No rebalance row
exists within ±12h on either node, no rebalance in the corpus moves ~485k sats,
and a self-rebalance never appears in our own listforwards. The connection to
this loop is zero; the plugin-side ledger dropping one shard is the forwarding-
ledger dedup bug already logged in data-quality-acceptance-20260701.md.

**Failures do feed segment_observations.** 40 of 41 exported observations join
a rebalance_history row on (node, src→dst, ±10 min) via their `correlation_id`;
every matched observation belongs to a **native_sendpay_error** attempt (10
distinct attempt contexts; route_over_budget rejections send nothing, so they
emit no observations). The reverse direction is lossy: several sendpay rows
(e.g. 447/448, nexus-02 15/16) have no surviving observations — the datastore
export is a last-write TTL snapshot, so observation evidence between captures
can be overwritten. Hop-level
detail: repeated blame concentrates on 927328x2085x1, 924769x798x1,
950017x3386x1, 933791x3241x0; within an episode the retry-with-exclusions
produced 3–8 observations over 3–40s, sometimes re-blaming the *same* hop
(928171x1180x10 five times on 06-07) — partial-amount retries re-traverse
previously failed hops. The 1 unmatched observation
(`obs-1776182577-1`, nexus-02, 944921x2899x0→944951x1646x0:1776182575,
04-14 16:02) has **provably no history row** (row-id sequence 6→7 is contiguous
across its timestamp): an attempt path emitted an observation without creating
a history row. Pre-coverage, single instance, cause not corpus-determinable —
see Gaps.

## 3. Cross-module invariant sweep

`tools/audit/loop_sweep_rebalance.py` (read-only; run 2026-07-01):

| Check | Result | n | Vacuity/notes |
|---|---|---|---|
| LP-I1a over-budget overage is entirely open/close | PASS | 129/129 | non-vacuous (close 760–2,958 sats) |
| LP-I1b over-budget ⇒ remaining_sats == 0 | PASS | 129/129 | |
| LP-I1c over-budget ⇒ same-snapshot decision suppressed+budget_blocked | PASS | 129/129 | the loop's feedback edge, observed live |
| LP-I2 no normal/diagnostic row inside saturated interval (manual exempt) | PASS | 41/41 rows | non-vacuous: 2 defibs blocked in-window, row absence id-proven |
| LP-I4a completed defib action → diagnostic row ∨ budget-blocked ∨ hole | PASS | 25/25 | 1 in coverage hole (action 975, 06-28; row absence proven, cause unobservable) |
| LP-I4b diagnostic row → defib action (within planner 7d visibility) | PASS | 25/25 | planner-history RPC is a bounded window; rows 469-471 (06-21..23) fall outside 07-01's 7-day visibility — artifact, not mismatch |
| LP-I5 one history row per (from,to,timestamp) | PASS | 48/48 post-fix | 3 pre-62ae545 twin pairs reported as known fixed defect |
| LP-I6 segment observation joins an attempt row (±10 min) | **1 VIOLATION** | 40/41 | the 04-14 pre-coverage observation above |
| LP-I7 success fee ≤ max_fee | PASS | 2/2 | **balance-shift leg VACUOUS**: both successes predate snapshot coverage — no listpeerchannels pair to diff |
| LP-I8 suppressed windows contain no new normal OR diagnostic row | PASS | 549/549 | now non-vacuous (37 diagnostic rows in play), unlike Phase 2's RB-I1b |
| (b from task) debug-selected candidate that executed appears in history | VACUOUS | 0 | run_cycle selected/executed nothing, ever |
| (d from task) cooldown honored across snapshots | consistent, weak | — | min non-manual same-pair gap = 25h ≫ 36h-max persisted cooldown never binding; defib/manual paths skip cooldown gates by design (RE-I13), so only report-grade |

## 4. Episode audits

1. **The two successes (nexus-02 rows 5, 6 — manual, 04-14 14:42/14:45).**
   944921x2901x0 → 944951x1646x0, 300k then 100k sats, max_fee 3/2, actual fee
   **0** (zero-fee hive-internal path). Immediately preceded by failures id 3
   (`no_fleet_route`: askrene getroutes RPC failure) and id 4
   (`retriable_failure: NoCandidates`) — the operator retried until the fleet
   route resolved. Budget accounting: fee 0 ⇒ no ledger movement (consistent
   with the all-zero rebalance category). Balance shift: not corpus-observable
   (predates first snapshot 06-09) — the only corpus-era liquidity evidence is
   that 944951x1646x0 later routed on nexus-02: none (nexus-02 routed nothing
   corpus-wide), so even the successful refills produced no measured revenue.
2. **native_route_over_budget failure (rows 452+453, 06-09 01:24:56).**
   Planner defib (dest 945529x2331x0, DEAD_CAPITAL) → diagnostic row 452
   (amount 50k, cap 100) → engine prices route at **101 sats** → executor
   rejects pre-send, `route_over_budget: 101 > 100`; twin row 453 records the
   rejected price as `actual_fee_sats=101` on a *failed* row (pre-fix
   bookkeeping; no spend — ledger stayed 0). No observations (nothing sent).
   Aftermath: no retry of the pair; **945529x2331x0 was mutually closed 06-13
   and gone from listpeerchannels by 06-14** — defib-fail → zombie-confirm →
   close is the loop working end-to-end (close executed by operator, not
   planner).
3. **native_sendpay_error failure (row 463, 06-15 23:23).** Planner defib 940 →
   953555x1338x0 → 941347x1139x0, 50k sats, cap 100. Route priced ≤100, sendpay
   failed on-path: WIRE_TEMPORARY_CHANNEL_FAILURE. Three route attempts in 9s
   (observations obs-…23:23:24/29/33 blaming hops 927328x2085x1 then
   924769x798x1 then 927328x2085x1 again — exclusion retry re-blamed the first
   hop). Failure persisted a 5-min-class cooldown (temporary_channel_failure);
   no retry of the pair occurred within the corpus. First defib of this dest
   (action 933, 06-14 00:27) had been *budget-blocked* by the close-cost
   saturation — this 06-15 shock is the deferred retry the planner scheduled
   the next day, executing only after the window unsaturated (16:12).
4. **Defibrillation with forwarding aftermath (941347x1139x0, CP-H3
   material).** Shock failed (episode 3) — but the channel was **not dead**:
   80 settled forwards before the defib, and in the ~5 observable post-defib
   days (collection halts 06-20) it settled **57 more** (18 with the channel as
   inbound hop, 39 as outbound hop). A failed 50k active shock at cap 100 is weak evidence of
   zombie-hood for a channel that demonstrably routes smaller amounts — direct
   corpus support for CP-H3's false-positive concern. Contrast the nexus-02
   self-pair 944921x2899x0 ↔ 944921x2901x0: defibbed **7 times** (06-02 →
   06-30, alternating directions, every shock failed on fee/liquidity;
   observations classify hops as `fee` and `liquidity`), and nexus-02 routed
   zero forwards all corpus — repeated defibrillation there confirmed dead and
   changed nothing. Also note 953567x1631x0/1632x0 were defibbed as
   DEAD_CAPITAL **2–13 days after opening** (opened 06-14), 0 forwards ever.
5. **Manual rebalance burst (rows 458-462, nexus-01 06-13 19:47-19:55).**
   Minutes after the mass close (budget remaining 457 → 0), the operator tried
   935804x2243x0 → 936056x1037x0 five times: 100k@25 → `route_over_budget
   70 > 25`; 50k@12 → `36 > 12`; 50k@40 → `engine_busy` (single-flight lock —
   RE-I11 observed live); 50k@40 and 25k@25 → on-path TEMPORARY failures with
   `partial_retry_failed: route_over_budget`. Eight segment observations
   (19:52-19:55, hops 927328x2085x1 / 924769x798x1). Manual bypassed the
   just-saturated budget exactly as RB-I9 specifies; fees recorded: none (all
   failed). Notably the dest is the *same channel the engine considered 178
   times and always held below margin* — the manual attempts effectively
   market-tested the engine's refusal, and the market agreed (36–70 sats to
   refill a channel the engine valued below its hold margin).
6. **Budget saturation event (06-13 → 06-15, nexus-01) — the loop's one live
   stress test.** 11 mutual closes at 19:43 (several long-standing planner
   DEAD_CAPITAL close recommendations, executed out-of-band); operator raised
   daily_budget 500 → 612 in the same snapshot; close fees 155 → 2,958 sats;
   budget surface pinned at remaining 0 for ~44h of snapshots; decision surface
   suppressed/budget_blocked for the whole span (550 decisions); two scheduled
   defib shocks silently skipped (no rows — id-contiguity proven); zero
   automated or diagnostic spend leaked; normal operation resumed 06-15
   evening (defib 940 executed 23:23). Every edge of the budget loop fired in
   the right order.

## 5. Gaps / Anomalies

1. **Planner "completed" ≠ executed.** `_execute_defibrillation`
   (capacity_planner.py:3484-3491) marks the action `completed` whenever
   `diagnostic_rebalance` returns `success: True` — which includes the
   *blocked-by-capital-controls* and *no-sources* early returns
   (rebalancer.py:2398-2402, 2442-2447). Actions 933, 937 (and probably 975,
   in the coverage hole) are "completed" defibrillations whose shock never
   happened. Consumers of planner history (including `get_diagnostic_rebalance
   _stats`-driven zombie confirmation, which reads *history rows*, not actions)
   see divergent truths. Recommend a distinct `blocked` status.
2. **`actual_cost_sats` is never backfilled on defib actions** (all 25
   completed actions carry `actual_cost_sats: null` even where a history row
   exists) — planner-side spend attribution for defibs is dead.
3. **Observation-without-row** (LP-I6 violation, 04-14, pre-coverage, row
   absence proven by id contiguity): some path emitted a segment observation
   for an attempt that never created a history row. Single instance, cause not
   determinable from the corpus; worth a code trace in Phase 4 (candidate:
   `_record_rebalance_pending` returning None on DB error while execution
   proceeds — that path is silent by design, rebalance_engine_v2.py:2804-2810).
4. **Twin-row defect** (3 episodes, 06-05/07/09): fixed by 62ae545 on 06-10;
   corpus shows both the defect firing and the fix holding. Phase 2/4 stats
   must not count rows 448/451/453 as independent automated attempts, and
   Phase 2's "4 ev_positive normals" is really **1** (row 442).
5. **Sweep-vacuity corrections to Phase 2 claims**: RB-I1c was vacuous
   (rebalance-category spend is 0 corpus-wide, LP-I3); RB-I1b remains
   weightless as stated; the "diagnostic 37 / manual 10 / normal 4" split is
   really 37/10/1(+3 duplicates); "49 failed vs 2 success" stands.
6. **On-chain close costs are the only observed budget consumer** in the whole
   corpus. Every budget number except the 06-13 close event is zero — daily
   ceilings, weekly ceilings, reservations, hot-channel overrides and the
   capex tiers all remain corpus-unexercised beyond that single event. Budget
   arithmetic under *concurrent discretionary* spend is still test-only.
7. **Defib targets include 2-day-old channels** (953567x1631x0/1632x0, opened
   06-14, defibbed 06-16/26/27 as DEAD_CAPITAL with zero lifetime forwards) —
   aggressive for channels that had no chance to route; interacts with CP-H3.
8. **Coverage limits honestly stated**: balance-shift verification for
   successes is impossible (both predate snapshots); the 06-21..06-30 hole
   hides the budget context of defib 975 and rows 469-478's decision states;
   forwarding-aftermath windows truncate at 06-20 (collection halt).
