# Hermes Corpus Data Quality — Phase 0 of the Module Verification Campaign

Date: 2026-06-12
Scope: `/home/sat/cl-mycelium-hermes` observational corpus (hive-nexus-01, hive-nexus-02),
collector `/home/sat/bin/cl-mycelium-hermes-collector.py`, hermes-agent cron scripts.

## Findings and fixes

### 1. Forwards cursor poisoned by index-family confusion (FIXED)

The nexus-01 `listforwards` window cursor was seeded at **94,961** — derived from the
`created_index` family (May 20 cumulative capture: `created_max` 94,960) — while windows
query `index=updated`, whose May 20 max was **79,823**. Every hourly window since
windowed collection began (2026-06-09) silently returned zero forwards and would have
kept doing so for years.

- One-time repair: cursor reset to 79,824 (backup:
  `state/hive-nexus-01/listforwards-updated-index.json.bak-20260612-created-index-poisoning`).
- Backfill verified: the next full sample recovered **10,256 forwards (805 settled,
  16,711 sats fees) spanning updated_index 79,824–90,079**, i.e. all forwarding activity
  since 2026-05-20. The May 20 → June 9 attribution hole is closed.
- Guard added to the collector: each window now probes the node's live max
  `updated_index` (`wait subsystem=forwards indexname=updated nextvalue=0`, read-only)
  and raises a non-ignorable `cursor_ahead_of_live` quality error — with
  `usable_for_value_window: false` — if the cursor ever points past reality again.
  Window metadata now records `live_updated_index`.
  (Backup: `cl-mycelium-hermes-collector.py.bak-20260612-cursor-fix`; contract tests pass.)

nexus-02's cursor (84) was seeded consistently (young node: created_max = updated_max
= 83) and needed no repair.

### 2. `full-daily` profile was never scheduled (FIXED)

The daily cumulative `listforwards` ground-truth capture stopped after the manual
2026-05-19/20 runs because no hermes cron job ran the `full-daily` profile. The
daily-rollup script (`~/.hermes/scripts/cl-mycelium-hermes-daily-rollup.sh`) now runs
`collector.py full-daily` (non-fatally) before the rollup, restoring a daily cumulative
capture that lets cursor drift be detected against ground truth.

### 3. "Insufficient coverage" verdicts are a plugin-surface gap, not a sampling gap (RECORDED, not fixed here)

Report-level sampling coverage is 1.0; collector cron jobs are healthy (~300 snapshot
dirs/day/node; interval jobs drift by their own runtime, which is benign). The
`evidence_inconclusive` classifications instead stem from the plugin's metabolism
ledger reporting `covered_hours: null / status: unknown / freshness: unknown` for
**every** window (1h–30d), so the rollup can never certify any horizon
(`usable_for_value_proof: false`).

Also suspicious: all five ledger windows report **identical** intake/reserve values on
both nodes — and nexus-02 reports 285 sats of "1h window" intake while having had
**zero forward updates since 2026-05-20** (proven by the repaired cursor + live-index
probe). The ledger windows appear to carry lifetime values, not windowed ones.

→ This is a cl_revenue_ops finding (capital_efficiency / metabolism ledger surface),
in-scope for Phase 2 verification of that module's contract. No plugin change is made
during the campaign.

### 4. Segment observations are not sporadic (NO FIX NEEDED)

`revenue-segment-observations` is captured hourly (≈23 samples/day) since 2026-06-02 —
which is when the surface first existed. Earlier absence is feature age, not collection
failure.

## What is now measurable that wasn't

- Exact per-channel forwarding revenue for 2026-05-20 → present on nexus-01
  (contiguous deduplicated updated-index windows; chain verified by
  `tools/audit/check_hermes_forwards_chain.py`).
- Fee-decision → forwarding-outcome joins over the same period (hourly
  `revenue-status` decisions + lossless forwards).
- A proven negative: nexus-02 has routed **nothing** since 2026-05-20 — a hard input
  for its contribution analysis.

## Acceptance check still pending

≥95% expected sample coverage and forwards-vs-dashboard reconciliation over a full
week of post-fix data (re-check ~2026-06-19; the Phase 5 scorecard should encode both).
