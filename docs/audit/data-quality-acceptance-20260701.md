# Hermes Corpus Data Quality — Phase 0 Acceptance Check

Date: 2026-07-01 (due ~2026-06-19; overdue)
Baseline: `docs/audit/data-quality-20260612.md` (fixes landed 2026-06-12, deployed ~14:00Z)
Corpus: `/home/sat/cl-mycelium-hermes/{hive-nexus-01,hive-nexus-02}`
Auditor: `tools/audit/check_hermes_forwards_chain.py` (run 2026-07-01)

## Verdict

| # | Criterion | Verdict |
|---|-----------|---------|
| 1 | ≥95% expected sample coverage (some full post-fix week) | **FAIL** |
| 2 | Lossless forwards (contiguous chains, no unretried truncation) | **PASS** |
| 3 | Corpus-vs-plugin revenue reconciliation | **PASS** (residual explained) |
| 4 | full-daily ground truth daily + zero cursor drift | **PASS** (halted with collection) |
| 5 | cursor_ahead_of_live guard + live_updated_index metadata | **PASS** |

**Overall: FAIL — Phase 0 is not done.** The quality of everything that *was*
collected is excellent (criteria 2–5 all pass, with a perfect zero-drift
ground-truth match), but the collection pipeline itself degraded on 2026-06-15/16
and **stopped entirely at 2026-06-20T15:11Z**. As of this check there are **11
consecutive days with zero snapshots** (2026-06-21 → 2026-06-30). No post-fix week
achieves ≥95% coverage. First action: diagnose/restart the hermes collector cron
on `ogsatoth` before re-running this acceptance check.

---

## Criterion 1 — Sample coverage: FAIL

Expected cadence, per snapshot metadata (unchanged throughout): health every 5m,
compact every ~30m, full every 60m, full-daily 1/day. Interval jobs self-drift by
their own runtime, so the design expectation is the healthy-day baseline of
**~300 snapshot dirs/day/node** (mean of 2026-06-09→12 = 300.75), consistent with
the 06-12 audit doc. 95% threshold = **285 dirs/day**.

Both nodes are collected by the same runner and have byte-identical daily counts.

| Day (2026) | dirs/day/node | % of 300 | ≥95%? | by type (compact/full/full-daily/health) |
|-----------|---------------|----------|-------|------------------------------------------|
| 06-12 | 298 | 99.3% | yes | 45/25/0/228 |
| 06-13 | 299 | 99.7% | yes | 47/24/1/227 |
| 06-14 | 299 | 99.7% | yes | 46/23/1/229 |
| 06-15 | 255 | 85.0% | no | 40/18/1/196 |
| 06-16 | 117 | 39.0% | no | 23/4/1/89 |
| 06-17 | 117 | 39.0% | no | 24/4/1/88 |
| 06-18 | 115 | 38.3% | no | 24/4/1/86 |
| 06-19 | 116 | 38.7% | no | 23/4/1/88 |
| 06-20 | 76 (ends 15:11Z) | 25.3% | no | 15/3/2/56 |
| 06-21 → 06-30 | **0** | 0% | no | — |

Only 3 post-fix days pass; **no full week achieves ≥95%**.

Cause, as far as visible from the corpus:

- 06-15 (partial) → 06-16 onward: declared cadences in `metadata.json` are
  unchanged ("every 5m" / "every 60m"), but effective intervals stretched ~3x
  (health lands every ~16 min instead of 5; full drops from 23–24/day to 4/day).
  Snapshots still span all 24h — this is collector-runtime drift/slowdown, not a
  scheduling change and not host downtime.
- 2026-06-20T15:11:45Z: last snapshot ever (a full-daily run). It completed
  cleanly (`errors.json`: no failed commands, only the long-standing
  `hive-organism-drift-report` unavailable), then collection stopped. Weekly
  rollups in `reports/` also end at `weekly-20260620` — the whole hermes cron
  stack went silent after 06-20.
- `quarantine/` contains nothing post-fix (only the 2026-05-20 transport-failure
  entry), so the gap is missing runs, not quarantined ones.

## Criterion 2 — Lossless forwards: PASS

`check_hermes_forwards_chain.py` (2026-07-01):

| Node | Verdict | Windows | Gaps | Overlaps | Trunc. unretried | updated_index span | Dedup fwds | Settled | Settled fees |
|------|---------|---------|------|----------|------------------|--------------------|-----------|---------|--------------|
| hive-nexus-01 | CONTIGUOUS | 178 | 0 | 1 (benign) | 0 | 79,824 – 97,298 | 17,475 (raw=dedup) | 1,033 | **20,329.884 sat** |
| hive-nexus-02 | CONTIGUOUS | 178 | 0 | 0 | 0 | 84 – 87 | 4 | 0 | 0.000 sat |

No unreadable windows, no unretried truncation. The chain is lossless from the
2026-05-20 backfill point up to the collection halt; it necessarily ends at
2026-06-20T15:11Z (criterion 1 failure), so forwards after that moment are
unobserved, not lost — the cursor state file will resume from 97,298.

## Criterion 3 — Reconciliation vs plugin revenue reporting: PASS

Clean common window: **2026-06-13T05:55:46Z → 2026-06-20T05:55:58Z** (the first
and last post-fix full-daily snapshots), reconciling deduplicated windowed
settled forwards (attributed by `resolved_time`) against the delta of the
plugin's `revenue-history.json` lifetime counters between those two instants.

| Node | Plugin Δ lifetime_revenue | Plugin Δ forward_count | Corpus settled fees | Corpus settled count | Residual |
|------|---------------------------|------------------------|---------------------|----------------------|----------|
| hive-nexus-01 | 3,056 sat | 196 | 3,121.494 sat | 198 | +65.494 sat (2.1%) |
| hive-nexus-02 | 0 sat | 0 | 0 sat | 0 | 0 (exact) |

Daily decomposition (nexus-01, full-daily → full-daily):

| Window | Plugin ΔRev | ΔCnt | Corpus settled | Corpus fees | Diff |
|--------|------------|------|----------------|-------------|------|
| 06-13→14 | 691 | 27 | 28 | 754.542 | **+63.542** |
| 06-14→15 | 346 | 13 | 14 | 347.405 | +1.405 |
| 06-15→16 | 453 | 20 | 20 | 453.699 | +0.699 |
| 06-16→17 | 530 | 28 | 28 | 529.385 | −0.615 |
| 06-17→18 | 791 | 50 | 50 | 790.706 | −0.294 |
| 06-18→19 | 166 | 29 | 29 | 166.276 | +0.276 |
| 06-19→20 | 79 | 29 | 29 | 79.481 | +0.481 |

Residual explanation:

- **63.120 sat / 1 forward**: on 06-13T07:10:30 two *identical* settled forwards
  landed in the same second (updated_index 91,151 and 91,152; both
  931199x1231x0 → 946890x2272x0, both 63.120 sat, distinct HTLCs). The plugin's
  ledger recorded only one of the pair — matching both its −1 forward count and
  ~−63 sat that day. The corpus keys by `updated_index` and provably has both;
  the corpus figure is the correct one (confirmed by the zero-drift ground-truth
  match in criterion 4).
- **~±0.3–1.4 sat/day**: sub-sat (msat) rounding — the plugin reports integer
  sats; corpus totals are msat-exact. Six of seven days reconcile within ±1.5 sat.
- No window-edge effects: no settled forward resolved within 1 hour of either
  boundary; `received_time` vs `resolved_time` attribution gives identical totals.

Conclusion: the corpus reconciles with, and is strictly more accurate than, the
plugin's own reporting. The 2.1% residual is a plugin-side bookkeeping artifact
(duplicate-forward dedup + integer rounding), not corpus loss.

## Criterion 4 — full-daily ground truth + zero cursor drift: PASS

Cumulative `listforwards.json.gz` captures (via
`~/.hermes/scripts/cl-mycelium-hermes-daily-rollup.sh`) appear **daily on both
nodes from 2026-06-13 through 2026-06-20** (~05:55Z; 06-20 has a second capture
at 15:11Z — the final collector run). None on 06-12 itself (fix landed mid-day;
first scheduled run was the next rollup) and none after 06-20 (collection halt —
criterion 1, not a full-daily defect).

Cursor-vs-ground-truth drift, comparing the last cumulative capture
(06-20T15:11Z; nexus-01: 113,301 lifetime forwards) against the deduplicated
window chain over the common updated_index span:

| Node | Common span | Ground truth | Windows | Only-in-either | Status mismatches | Settled fees (both) | Drift |
|------|-------------|--------------|---------|----------------|-------------------|---------------------|-------|
| hive-nexus-01 | 79,824–97,298 | 17,475 | 17,475 | 0 / 0 | 0 | 20,329.884 sat | **0** |
| hive-nexus-02 | 84–87 | 4 | 4 | 0 / 0 | 0 | 0 sat | **0** |

Byte-level agreement on membership, status, and fees. The windowed pipeline is
provably lossless against node ground truth.

## Criterion 5 — cursor_ahead_of_live guard: PASS

All 236 `listforwards-window.json.gz` files from 2026-06-12 onward were scanned:

- `cursor_ahead_of_live=true`: **0 occurrences** (both nodes).
- `_window.live_updated_index`: present in all **208** windows from
  2026-06-12T14:05:33Z onward. The 28 windows earlier that day
  (00:34Z–13:49Z, both nodes) predate the fix deployment (~14:00Z per the
  06-12 audit) and lack the field, as expected; every post-deploy window
  carries it.

## Bottom line

Fixes #1 (cursor repair + guard) and #2 (full-daily scheduling) from 2026-06-12
are verified working, and the forwards pipeline is demonstrably lossless and
more accurate than the plugin's own revenue ledger. Phase 0 nonetheless **fails
acceptance** on coverage: the collector degraded to ~39% cadence on 2026-06-16
and died completely on 2026-06-20T15:11Z, leaving an 11-day (and counting) hole.
Until the cron stack on `ogsatoth` is restarted and a full ≥95% week is banked,
Phase 0 remains open. Re-run this check after one full week of restored
collection; the updated-index cursor (state at 97,298) will resume the chain
without loss, but the 06-20→restart gap in *non-forwards* surfaces (revenue
decisions, health, segments) is unrecoverable.
