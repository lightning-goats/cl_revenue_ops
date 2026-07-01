# Phase 3 — Fee Decision Loop, End-to-End (2026-07-01)

Loop under test: **flow_analysis state → fee_controller decision (policy_manager /
hive_hints inputs) → setchannel → gossip-visible fee → recorded change → forwarding
outcome.** Phase 2 verified the modules in isolation; this verifies the handoffs
between them at snapshot granularity over the full frozen hermes corpus
(hive-nexus-01 + hive-nexus-02, 2026-06-09 → 06-20 + one 2026-07-01 snapshot,
10-day hole 06-21..06-30). Sweep: `tools/audit/loop_sweep_fee.py` (read-only,
full corpus, both nodes). Code references are HEAD cdb536a.

## Loop verdict

**Coherent.** Everywhere the corpus lets us see both sides of a handoff, the sides
agree — on the complete-record node (nexus-02) *exactly*: 407/407 gossip-visible
fee changes reconcile to recorded decision chains, 100% of decisions carry the
flow state the snapshots show, all fees in bounds, member channels pinned to zero
on schedule, all deltas within caps, gossip gate never underrun. The one hard
violation cluster (LF-7, 28 hits) is a single-channel episode traced to
**operator intervention outside the plugin** (external `setchannel` + the channel
leaving the managed set at the 06-15 restart) — the plugin's control loop was not
broken; it was bypassed. Two genuine telemetry/observability defects were found
(fee-debug "ready" for an unmanaged channel; `guard=zero_flow_downshift` labelling
an *upward* floor-driven move), plus one structural evidence limit: the
`recent_fee_changes` window loses 61% of nexus-01's change records, so
completeness there rests on the 39% recovered plus nexus-02's 100%.

## Data model and completeness (read this before the table)

- **Recorded changes** are recovered by deduplicating the rolling 10-row
  `recent_fee_changes` window across 2,594/2,597 revenue-status snapshots
  (~5-min cadence). The DB id sequence exposes losses: a 30-min fee cycle that
  touches more than 10 of nexus-01's 38 channels overflows the window between
  snapshots. Recovered: **n1 3,610 of 9,152 ids (39.4%; 5,542 lost across 367
  gaps)**, **n2 455 of 456 (99.8%)**. Every completeness check below classifies
  misses as *gap-explained* (an unrecovered-id interval overlaps the check
  window) vs *unexplained*. Gap-explained is an upper bound on benign misses
  (gaps are not per-channel attributable); the discriminating evidence is
  nexus-02, where the window never overflowed and every check is exact.
- **Advertised fees**: 613/614 hourly listpeerchannels snapshots
  (local `fee_proportional_millionths` / `fee_base_msat`).
- **Records are best-effort post-RPC** (fee_controller.py:7197, try/except after
  the setchannel at :7164): a gossip change without a record is possible by
  design. Measured, not assumed — see LF-1.
- Deploy epochs recovered from the corpus itself: **071a5b3 (zero-flow ratchet,
  245ac12)** first stamped 20260615T192238Z, first guard-tagged change 86 s later
  (19:23:39Z); **8630ca6 (member zero-fee)** deploy epoch 2026-06-27T22:27:01Z
  (earliest `hive_member_zero_fee` record, n2). Caveat: metadata.json code stamps
  are the collector host's working tree (identical for both nodes, `dirty: true`),
  so stamps are corroborating, not authoritative — the guard-change timing is.

## Handoff integrity findings

1. **flow_analysis → fee_controller (state at decision time): clean.** Every
   dts_pid_sample decision embeds the flow state it acted on (`state=X` in the
   reason). 4,020/4,020 checkable decisions match the channel_states row in a
   bracketing revenue-status snapshot (3,998 match the snapshot before the
   decision, 22 only the one after — consistent with the decision cycle itself
   refreshing the state; 11 had no bracketing row, all at the 07-01 hole edge).
   Zero cases where the decision's state matches neither neighbor. The
   reason-internal `flow=` (DTS input) and `state=` tokens agree in **all 4,031**
   decisions. The debug surface agrees too: fee-debug `flow_state` ==
   channel_states state in 8,222/8,222 same-snapshot joins.
2. **fee_controller decision → gossip: exact where visible.** See LF-1/LF-2:
   every recovered record chain reproduces the LPC-observed fee trajectory
   endpoint-for-endpoint; on n2, 100% of gossip diffs have full chains.
3. **policy_manager overrides → fees: vacuous.** The corpus has no policies
   artifact and no STATIC/PASSIVE policy is corpus-observable; 0 `policy_static`
   records exist. The one *inferred* policy event (a channel leaving the managed
   set on 06-15, below) is invisible in any artifact except the
   `revenue-health fees.managed_channels` count. **PM→FC handoff cannot be
   verified on this corpus** (Phase 2 anticipated this; unchanged).
4. **hive membership → zero-fee pinning (8630ca6): holds, thinly.** n2 recorded
   both member transitions (60→0 ppm, `hive_member_zero_fee`, 06-27T22:27:01);
   n1's transition records fell in the 454-id recovery hole, but the 07-01
   listpeerchannels shows all 5 member channels (3 on n1, 2 on n2) at
   0 ppm / 0 base msat, and hive-export-hints confirms all 5 peers member:true.
   Converse holds corpus-wide: **zero non-member zero-fee channels** in 27,012
   advertised-fee observations. Pre-deploy, the same member channels carried
   normal DTS fees (06-16: n1 164/60/60 ppm, n2 582/61 ppm) — the flip is the
   deploy, not a coincidence.

## Invariant sweep (loop_sweep_fee.py, full corpus, both nodes)

| Inv | Claim (cross-module) | Checked | Violations | Vacuity / notes |
|---|---|---|---|---|
| LF-1 | every gossip-visible fee change has a matching recorded chain (first.old == observed old, links continuous, last.new == observed new) | 8,624 LPC diffs | **0 unexplained** | n2: 407/407 full match (window never overflowed). n1: 3,337 full match; 4,789 misses + 91 partials, **all** overlapping unrecovered-id gaps (lost to the 10-row window, quantified miss rate 58% of n1 diffs — an evidence limit, not a plugin defect) |
| LF-2 | advertised fee == last recorded change's new value until the next change | 3,732 record→next-LPC checks | **0 unexplained** | 3,689 exact (n1 3,290, n2 399); 43 mismatches all gap-explained (lost later change) |
| LF-2b | record chain self-continuity: next.old == prev.new for id-adjacent records | 4,013 pairs | **0** | 3,298 continuous (n2 453/453); 715 discontinuities, every one separated by an id gap |
| LF-3 | no fee change during pause/suppression windows | 0 paused snapshots | 0 | **VACUOUS** — operators never paused in the window (matches Phase 2 FC-I2) |
| LF-4 | member channels pinned 0 ppm/0 base after 8630ca6 deploy; zero-fee ⇒ member | 5 post-deploy member checks; 27,012 zero-fee-converse checks | **0** | **NEAR-VACUOUS on the pinning side**: deploy epoch (06-27T22:27Z) is inside the corpus hole; only the single 07-01 snapshot per node is post-deploy. Converse (no non-member at 0) is corpus-wide and real |
| LF-5 | recorded new fees and advertised fees within snapshot-local [min_fee_ppm, max_fee_ppm] | 4,045 records + 27,012 advertised | **0** | member zero-fee exempted (2 records, 5 advertised); config was 60/3500 throughout |
| LF-6 | decision-embedded flow state matches channel_states at decision time; `flow=` == `state=` | 4,031 decisions | **0** | 11 no-bracketing-row (07-01 edge); "before/after snapshot" granularity ±6 min |
| LF-7 | debug surface coherent: fee-debug `last_broadcast_fee_ppm` == advertised; fee-debug `flow_state` == channel_states | 8,267 lb joins + 8,222 state joins | **28** (lb≠adv) | all 28 are one channel/one episode (946890x2272x0, 06-16→06-20, lb=2306 vs adv=250) — see Episode E2b; flow_state side 8,222/8,222 clean |
| LF-8 | ratchet guard changes only post-deploy; downshift lowers the fee | 246 guard-tagged records | **0** hard | 244 correct-direction post-deploy (first 86 s after code stamp); 2 floor-arm cases where `guard=zero_flow_downshift` is stamped on an *upward* move to the effective floor (design-consistent: `max(floor, min(target, 0.85·current))`, fee_controller.py:5241-5252 — but the telemetry is misleading; see Anomaly 3) |

Sweep runtime ~3 s; output reproduced twice byte-identically.

## Episode audits

### E1 — zero-flow downshift ratchet: fleet-wide discovery unwind (n1, 941573x2327x0)

Deploy 06-15T19:22Z; first cycle 19:23:39 touched ~28 of 38 channels with
`guard=zero_flow_downshift`, then every ~30 min cycle continued.

| time (Z) | input state | decision | advertised | forwards next window |
|---|---|---|---|---|
| 06-15T19:23 | zero-rev streak ≥ 24, rate 0 | 720→612 (0.850) downshift | 612 ✓ | none |
| 19:48 → 03:30 (14 cycles) | streak persists | 612→…→61, each ×0.845-0.850 | tracks each step ✓ | 2 dust probes |
| 06-18T03:33-06:35 | 3 settled forwards land (~0.3 sats) | — | 61 | — |
| 06-18T06:51 | rate 0.22 sats/hr, guard **disarms** (`zero_flow_guard:none`) | 61→70 (+15%) | 70 ✓ | forward @09:39 |
| 06-18T15:17 | 88.7M msat forward, **6.57 sats fee** | — | 74 | — |
| 06-18T20:00 | rate=0.00 (window empty), streak ≥ 24 **again** | 74→66 downshift | 66 ✓ | — |
| 06-19 | oscillates 60↔75, alternating climbs and downshifts | | ✓ | trickle |

Verdict: **loop-coherent, including the surprise.** The 06-18T20:00 downshift
right after a 6.5-sat forward is the trickle guard by design: revenue below
`TRICKLE_RESET_FRAC ×` the decayed positive-rate reference *extends* the zero
streak instead of resetting it (fee_controller.py:661-683), and the immediate
observation window was empty. Consequence worth knowing: the ratchet will
undercut a fee level that is producing (trickle) revenue. The 15-step 720→61
unwind (~8 h) also shows the ratchet's intended speed: a fleet that had climbed
on extrapolated posteriors was repriced to the floor region in one evening.

### E2 — DTS reprice vs forwarding outcome (n1, 946890x2272x0, 06-13 → 06-15)

The channel's real demand lives at 60-130 ppm (settled forwards every day at
70-130 ppm through 06-09; 06-02..06-05 earned 700-2,100 sats/day there). DTS
discovery ladders climbed it anyway:

| time (Z) | decision (recorded) | advertised | settled forwards at that fee |
|---|---|---|---|
| 06-13T17:17→06-14T07:26 | 130→230→…→2046 (12 steps, ~×1.5/cycle, `state=source`) | tracks ✓ | 8 settled @130 before the climb; **zero** above ~300 |
| 06-14T17:43 | *(no record — external)* | 2046→**100** | — |
| 06-14T17:48→06-15T02:49 | 100→200→…→2306 (13 steps, recorded, chain exact) | tracks ✓ | 9 settled @200 early in climb; **zero** above ~300 |
| 06-15T19:35 | *(no record — external, at restart)* | 2306→**250** | 11 settled @250 on 06-16, 19 on 06-17, ~2,900 sats total 06-16..06-22 |

Verdict: the decision→gossip→record legs are exact (every recorded step matches
gossip). The decision→outcome leg shows textbook elasticity the optimizer kept
overshooting: volume/revenue responded *immediately* when the fee returned to
the supported region — but it was an **external hand, not the plugin**, that
returned it there (twice). The zero-flow ratchet (deployed exactly at the second
reset) is the in-plugin answer to this failure mode.

### E2b — the LF-7 violation cluster: unmanaged-channel desync (n1, 946890x2272x0, 06-15 → 07-01)

- 06-15T19:22 restart: `revenue-health fees.managed_channels` drops 38 → **37**
  and stays there through 06-20. The missing channel is this one (its DTS state
  never re-enters the live `_channel_fee_states` map; managed_channels counts
  that map, cl-revenue-ops.py:4742-4757).
- 06-15T19:35: advertised 2306→250 with no record, no `manual=1` row, and —
  decisive — fee-debug `last_broadcast_fee_ppm` stays 2306: **every** plugin
  write path syncs or reconciles that field, so the 250 was set outside
  cl_revenue_ops (raw `setchannel` / hive tooling). Same signature as the
  06-12T12:40 (2468→138) and 06-14T17:43 (2046→100) resets on the same channel;
  no other channel shows it (both single-channel LPC diffs).
- 06-16→06-20: fee-debug reports the channel `status: "ready", skip_reason: null`
  with `hours_since_update` growing 6.5 → 117+ and `zero_revenue_streak` frozen
  at 198 — the controller never processes it (consistent with an operator
  PASSIVE/ignore policy; the corpus cannot show policies directly).
- 07-01: managed again (records #97161 65→60, #97169 60→66); the 07-01
  `managed_channels` 33/36 gap is exactly the 3 zero-fee member channels.

Verdict: **not a control-loop break — an operator bypass**, but it exposes two
real observability defects: (a) revenue-fee-debug computes status from timers
only and reports a policy-skipped/unmanaged channel as "ready" with a stale
last_broadcast (28 sweep violations are all this); (b) nothing in any operator
surface records "this channel left management" — it had to be inferred from a
count. Also note the economics: the externally-pinned 250 ppm earned ~2,900 sats
while the controller's stale belief was 2306 — the operator out-priced the
optimizer on this channel.

### E3 — hive-member zero-fee transition (8630ca6, both nodes)

| | pre-deploy (06-16) | transition | post-deploy (07-01) |
|---|---|---|---|
| n2 944921x2901x0 (peer = nexus-01) | 582 ppm (DTS) | recorded 60→0 `hive_member_zero_fee` 06-27T22:27:01 | 0 ppm / 0 base ✓ |
| n2 944921x2899x0 | 61 ppm | recorded 60→0, same second | 0 / 0 ✓ |
| n1 944921x2901x0, 940132x2695x0, 933791x3241x0 | 164 / 60 / 60 ppm | records lost in the 06-20..07-01 recovery hole | 0 / 0 ✓ (all member:true) |

Forwarding outcome: near-vacuous — n2 routed nothing all corpus; n1's three
member channels settled only dust (1-22 sats volume each) in the 4 observable
post-deploy days. No measurable flow response to the zero-fee policy yet.

### E4 — largest single-cycle fee moves: which cap applied?

- **Largest overall:** #94679 `channel_open` 10→3500 (n1, 06-13T22:22) — initial
  fee at channel open, outside the damper by design (FC-I10 exception);
  3500 == max_fee_ppm, in bounds.
- **Largest DTS move:** #88809 (n1, 951165x2981x0) 2065→1470 (−595, `state=sink`,
  99% liquidity). Reason decomposes exactly: target dts:113 → post_pid:82 →
  blended 2065 + 0.30·(82−2065) = 1470; `cap:none(1033ppm)` — the −595 step sat
  *under* the wake:none cap max(⌈0.5·2065⌉,100)=1033, so no cap applied. Coherent.
- **Largest capped move:** #6694 (n2, 944921x2901x0) 707→1061 = **+354 =
  max(⌈0.5·707⌉,100) exactly**, reason says `cap:normal_cycle_delta_cap(354ppm)`
  and `bound:ceiling` (raw target 2070 first clipped to the 2000 discovery
  ceiling). The cap fired, said so, and the arithmetic checks.

### E5 — high-churn channel: gates over 185 changes (n2, 944921x2901x0)

The corpus's best-instrumented channel: 185 recorded changes with **zero record
loss**. Median inter-change spacing 33 min (the 30-min cycle; occasional
shorter wake cycles, min 7.3 min). **0/185** changes below the 5% gossip-gate
band; **0/185** below the alpha-guard minimum (3% / 5 ppm); every change lands
on the previous one's value (453/453 chain links). LF-1/LF-2 reconcile all 407
of n2's gossip diffs to these records. The gossip gate limits broadcast rate
without ever pinning the price level — the 60→1061→0 full range was traversed.

## Gaps / Anomalies

1. **(evidence) `recent_fee_changes` is not a fee-change ledger.** At 38 channels
   per cycle vs a 10-row window, nexus-01 lost 5,542 of 9,152 records (61%) to
   window overflow — including the entire fleet-downshift burst cycles and n1's
   member zero-fee transitions. Any future phase needing complete change history
   must snapshot the DB table (`fee_changes`) directly, not the RPC echo. All
   "gap-explained" classifications inherit this ceiling.
2. **(defect, observability) revenue-fee-debug lies about unmanaged channels.**
   Status is computed from sleep/interval timers only; a channel outside
   management (policy-skipped / not in `_channel_fee_states`) reports
   `ready, skip_reason: null` with stale `last_broadcast_fee_ppm` indefinitely
   (E2b; 28 sweep hits). Fix candidate: have the debug surface consult
   policy_manager and flag `unmanaged` (and/or reconcile lb against
   listpeerchannels when reporting).
3. **(defect, telemetry) `guard=zero_flow_downshift` can stamp an upward move.**
   When the effective floor (config min + rebalance-cost floor) exceeds the
   current fee, the guard's `max(floor, …)` arm raises the fee while the reason
   string still claims a downshift (2 cases, both 60→66 with a 66-ppm cost
   floor; #96680, #97169). Design-consistent ("hard floors win"), but log
   consumers bucketing by guard tag will misread it. Cheap fix: emit a distinct
   `guard=floor_override` (or suppress the tag) when the floor arm fires.
4. **(behavior worth an operator note) the trickle guard undercuts earning fees.**
   E1: a 6.5-sat forward within the ratchet window did not stop the downshift,
   because trickle revenue (< frac of the decayed positive-rate reference)
   extends the zero streak by design. Correct per code intent, surprising in a
   timeline; deserves a line in the operator docs.
5. **(unattributed) three external fee writes on one channel** (06-12, 06-14,
   06-15; 2468→138, 2046→100, 2306→250) plus an inferred management opt-out at
   the 06-15 restart. All evidence says outside-the-plugin (no records, no
   manual rows, lb never synced, single-channel). The corpus cannot say *who*
   (raw lightning-cli, hive tooling, or operator agent). If this was fleet
   tooling, the fee loop has a second writer it never hears about — worth
   confirming operationally, since gossip-refresh only reconciles channels the
   controller still manages.
6. **(vacuous checks, explicit)** LF-3 pause (0 paused snapshots), policy→fee
   handoff (no policies artifact, no STATIC observable), LF-4 pinning
   persistence (single post-deploy snapshot per node; the 06-27→07-01 pinning
   *interval* is unobservable), member zero-fee outcome leg (n2 routed nothing
   corpus-wide; n1 member channels dust-only), and congestion/pause/concurrency
   decision paths (never fired, unchanged from Phase 2).
7. **(context for Phase 4)** n1's forward outcomes are measurable (17,475-forward
   lossless chain; E2 shows a clean elasticity signal on at least one channel);
   n2 has zero forwards, so every outcome-leg hypothesis is n1-only. Fee-change
   power is also lopsided (3,610 vs 455 recovered records).

## Files

- Sweep: `tools/audit/loop_sweep_fee.py` (this phase; read-only)
- This report: `docs/audit/decision-loops/fee-loop.md`
