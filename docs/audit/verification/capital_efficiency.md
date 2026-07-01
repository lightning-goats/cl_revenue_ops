# Phase 2 Verification — capital_efficiency.py

Contract: docs/audit/contracts/capital_efficiency.md (CE-1..CE-6, + metabolism-ledger
seam section). Drift check: `git diff f905cfd..HEAD -- modules/capital_efficiency.py`
is empty — module is byte-identical to the contract-authoring commit; all cited line
numbers reconfirmed directly against HEAD. Corpus: `tools/audit/sweep_data_budget.py`
over 5,196 snapshots (both nodes, ~~2026-05-19 → 2026-07-01~~ **corrected 2026-07-01:
corpus spans 2026-06-09 → 2026-06-20 plus one 2026-07-01 snapshot pair; May data
quarantined**); this module emits no RPC/
datastore surface of its own, so corpus evidence here is limited to the metabolism-ledger
seam (`ML-*` checks) plus cross-checks in tests/test_capital_efficiency.py (~~31 tests~~
**corrected: 12 tests**, all pass on HEAD).

| Invariant | Verdict | Evidence |
|---|---|---|
| CE-1 RPSD = `fees_earned_msat*1000/capacity_sats`, 0.0 at zero capacity | **verified** | code confirmed (`_calculate_rpsd`, capital_efficiency.py:151-164); test_analyze_computes_rpsd_rank_and_fleet_summary, test_rpsd_uses_msat_precision_not_ceiling_sat_boundary |
| CE-2 windowed-net blend activates only when every channel exposes numeric `marginal_profit_30d_sats`; one missing field disables it fleet-wide | **verified** | code confirmed (analyze() loop :91-103 breaks to `{}` on first `None`; `_calculate_windowed_net_rpsd` :166-182); test_blend_ordering_demotes_stale_lifetime_leader, test_windowed_net_rpsd_exposed_in_snapshot, test_no_windowed_signal_falls_back_to_pure_lifetime_rank, test_dead_capital_detection_unchanged_by_blend |
| CE-3 percentile ranks tie-averaged in [0,1]; single channel ranks 1.0 | **verified** | code confirmed (`_calculate_percentile_ranks` :184-207, single-item early return :190-192); test_equal_rpsd_channels_share_same_percentile_rank, test_analyze_computes_rpsd_rank_and_fleet_summary (single non-tied case) |
| CE-4 dead capital requires flow-present + zero forwards + `days_open > grace_days` + not hive member; absence of flow data never treated as death | **verified** | code confirmed (`_is_dead_capital` :209-227, `flow_metrics is None → False` at :211-212); test_dead_capital_excludes_young_channels_and_hive_members, test_missing_stage_defaults_to_none |
| CE-5 fleet totals are non-negative sums; dead-capital sats only for dead-classified channels | **verified** | code confirmed (:112-118 total_deployed_sats, :134-136 dead_capital accumulation gated on `is_dead_capital`); test_analyze_computes_rpsd_rank_and_fleet_summary asserts exact totals (`dead_capital_sats == 2_000_000` for the one dead channel only, `total_deployed_sats == 12_500_000` = sum of all three) — non-tautological, real arithmetic checked |
| CE-6 SCID-format drift cannot orphan flow/stage lookups (normalize_scid aliasing) | **verified** | code confirmed (:69-78 `flow_by_normalized_channel`/`stages` built via `normalize_scid`, :122-127 three-way fallback chain `flow.get(channel_id) or flow.get(normalized) or flow_by_normalized_channel.get(normalized)`, :145 stage lookup via normalized key); test_stage_lookup_normalizes_scid_keys, test_flow_lookup_normalizes_scid_keys, test_stage_lookup_degrades_cleanly_on_db_failure (DB exception → `raw_stages = {}`, code confirmed :65-68) |

All six cited test files/functions were executed directly (`pytest tests/test_capital_efficiency.py tests/test_dead_capital_protections.py`) and pass on HEAD (~~31 + 19 = 50 passed~~ **corrected: 12 + 19 = 31 passed — the original text mistook the combined-run total (31) for the CE file's own count and then double-added**). `test_dead_capital_protections.py` exercises `capacity_planner`'s stage-progression gates (a *consumer* of `dead_capital_stage`, not this module's own classification) — included for context but not counted as direct CE-4 coverage.

## Metabolism-ledger seam — DB-7/covered_hours drift, verified fixed

The contract's "Observable surface" section asserted `Database.get_spend_ledger_summary`
"never emits" `covered_hours`/`coverage_hours`, so cl-hive's `_ledger_window_coverage`
could only ever report `"unknown"`. **That premise is now false on HEAD.** Commit
`2247370` ("Expose freshness metadata for revenue evidence") added
`timestamp`, `generated_at`, `ttl_seconds`, `coverage_hours`, `covered_hours`,
`coverage_status` to the result dict (modules/database.py:3960-3970 on HEAD; confirmed
by direct read).

**Seam-side finding**: the defect lived entirely on the cl_revenue_ops (producer) side.
cl-hive's consumer, `_ledger_window_coverage` (cl-hive/modules/organism/runtime.py:2677,
current line — moved from the contract's cited 2240-2262, cl-hive has also changed
independently since the contract was authored; see Anomalies below), already reads
**either** field name defensively:
```python
raw_covered_hours = spend_payload.get("covered_hours", spend_payload.get("coverage_hours"))
```
No cl-hive change was required to consume the fix — it was purely waiting on the
producer to emit the field. Corpus confirms the wiring works: in the single
post-deploy snapshot pair (`hive-nexus-01` and `hive-nexus-02`,
`20260701/20260701T203541Z`), all 5 windows (1h/6h/24h/7d/30d) under
`goal_state.metabolism.ledger.windows.*.coverage` now show `status: "complete"` with
`covered_hours == required_hours` (verified directly against the raw JSON, not just the
sweep tally).

**Correction to the sweep-note lead**: the prompt's hint that the sweep's `ML-COVER`
`pass=141` count is "likely the 2026-07-01 post-fix" rows is only **partially** right.
Breaking down all 23,000 `ML-COVER` window-checks by `coverage.status` across the full
corpus: `unknown=22,859`, `insufficient_coverage=131`, `complete=10`. Only the **10**
`complete` rows (2 nodes × 5 windows, the one post-deploy snapshot pair) are the
covered_hours fix landing. The other **131** `insufficient_coverage` rows are a
*pre-existing, unrelated* code path from 2026-06-09 (`hive-nexus-01/20260609/...`):
`_ledger_window_coverage` returns `insufficient_coverage`/`covered_hours: 0` whenever
`spend_payload` is not a dict at all (i.e. the per-window spend-ledger call inside
`_build_canonical_metabolism_ledger` produced nothing), which is the *same* fallback
mechanism already documented for `ML-BURN-IDENT`/`ML-DEV-IDENT` — confirmed directly:
those 2026-06-09 windows have `source_notes` showing `burn:revenue_profitability_summary`
(lifetime fallback engaged, `has_spend_ledger=False`), not the covered_hours field being
present-but-insufficient. So `insufficient_coverage` and `complete` are two structurally
different code paths in `_ledger_window_coverage`, and only `complete` is attributable to
commit 2247370.

**New anomaly introduced by the fix** (not present in the original contract, since the
field didn't exist yet): `covered_hours`/`coverage_hours` in `get_spend_ledger_summary`
is a **hardcoded echo of the requested `window_hours` parameter**, not a measurement of
actual data span (modules/database.py:3960-3966 — `"coverage_hours": window_hours,
"covered_hours": window_hours, "coverage_status": "complete"`, unconditional). There is
no earliest-`spend_events`-timestamp or plugin-uptime check anywhere in database.py
(confirmed: no `MIN(timestamp)`/`first_seen`/`plugin_start` tracking exists). Consequently
`_ledger_window_coverage`'s `covered_hours >= window_hours` test is now a tautology —
it will **always** evaluate `True` and report `"complete"`, even for, e.g., a 30-day
window on a plugin that restarted 2 hours ago with an empty `spend_events` table for
the other 718 hours. The original problem (opaque, permanently `"unknown"`) is fixed;
a new one (falsely-confident `"complete"`) has replaced it. This degrades the metabolism
ledger's confidence signal from "we don't know" to "we're sure" without adding the
underlying measurement — arguably worse for downstream consumers that gate behavior on
`coverage_status`.

## Gaps

- CE-1..CE-6 have no dedicated hermes/RPC surface (contract: "No RPC surface and no
  datastore key of its own") — all direct verification is code+test, none corpus-observable.
- CE-6's degrade-on-DB-failure path (`test_stage_lookup_degrades_cleanly_on_db_failure`)
  covers `get_dead_capital_stages()` raising, but no test exercises `flow_analyzer`
  raising from `analyze_all_channels()` — `analyze()` has no try/except around that call
  (capital_efficiency.py:64), so a flow-analyzer exception would propagate uncaught;
  not contradicted by the contract (which doesn't claim resilience there) but worth
  flagging as **untestable-with-current-data** / unaudited blast radius.
- `forward_velocity`'s `flow_window_days` triple-fallback (config → flow analyzer config →
  literal 7, capital_efficiency.py:106-110) has no dedicated test; not independently
  verified beyond code read.

## Anomalies

1. **cl-hive line-number drift**: the contract cites `_ledger_window_coverage` at
   runtime.py:2240-2262 and `_build_canonical_metabolism_ledger` at :2265-2372; on
   current cl-hive HEAD these live at :2677-2698 and :2702-2372(+offset) respectively —
   cl-hive has changed independently since the contract's 2026-06-12 audit date (new
   code — an `_intervention_verdict_details`/pathology-record block — was inserted
   earlier in the file, pushing the metabolism functions down ~440 lines). Logic itself
   is unchanged in substance (confirmed by direct read); only line numbers moved. Anyone
   re-auditing cl-hive directly should re-grep by function name, not trust these line
   numbers.
2. **Fix is partial, not complete** (see "New anomaly" above): `covered_hours` now
   exists but is a tautological echo of the request parameter, not a real coverage
   measurement. Recommend flagging to whoever owns modules/database.py: computing a
   real `covered_hours` (e.g. `min(window_hours, (now - earliest_spend_event_ts)/3600)`,
   falling back to plugin-start time when no spend_events exist) would let
   `coverage_status` mean something. This is a code-change recommendation, not applied
   here (read-only per audit rules).
3. The still-standing metabolism-ledger anomalies (`ML-INTAKE-IDENT`,
   `ML-RESERVE-IDENT` both 0/4,600 pass; `ML-BURN-IDENT`/`ML-DEV-IDENT` both 2,315/4,600
   fail) are **unaffected by the covered_hours fix** and remain entirely on the cl-hive
   side: intake/reserves are computed once outside the per-window loop and copied into
   all 5 windows by construction (`_build_canonical_metabolism_ledger`, confirmed at
   current HEAD lines ~2716-2723, logic matches contract's description of the old
   :2280-2301/:2335-2348 region); burn/development fall back to lifetime
   profitability/dashboard values whenever a window's spend-ledger call yields
   `has_spend_ledger=False` (confirmed at `_ledger_spend_totals_msat`, current HEAD
   :2498-2517, logic unchanged from contract's :2061-2080 description). None of this is
   capital_efficiency.py's responsibility — this module supplies no metabolism-ledger
   inputs at all.

## Refutation pass (2026-07-01)

Adversarial re-verification on HEAD (plugin cdb536a-era tree; cl-hive read-only at
53bc7c1). `git diff f905cfd..HEAD -- modules/capital_efficiency.py` re-confirmed empty.

**No verdict flipped.** CE-1..CE-6 survived direct attack — the full module (227 lines)
was re-read and every cited line range matches: `_calculate_rpsd` :151-164 (0.0 at zero
capacity :154-155), windowed-blend break-to-`{}` :92-96 with blend gate :98-103,
`_calculate_windowed_net_rpsd` :166-182 (None on missing/bool/non-numeric field),
`_calculate_percentile_ranks` :184-207 (single-item 1.0 :190-192, tie-averaging
:197-206), `_is_dead_capital` :209-227 (flow-None → False :211-212, grace :217, hive
bypass :220-225), fleet totals :112-118/:134-136, normalize_scid maps :69-78 and
three-way fallback :123-127, DB-degrade :65-68, unguarded flow-analyzer call :63-64
(gap stands). All 12 cited test functions exist and were re-run (12 + 19 pass);
`test_analyze_computes_rpsd_rank_and_fleet_summary` re-read — asserts exact fleet
arithmetic (dead 2,000,000 / total 12,500,000 / median 50.0 / per-channel rpsd and
ranks), genuinely non-tautological.

**Seam attribution attack (the load-bearing claim) — survived, verified precisely on
cl-hive HEAD 53bc7c1:**
- `_ledger_window_coverage` at runtime.py:2677-2699 exactly as cited; defensive
  either-key read at :2684; own-status computation at :2696 (producer's
  `coverage_status` ignored); not-a-dict → `insufficient_coverage`/`covered_hours: 0`
  at :2678-2683.
- `_build_canonical_metabolism_ledger` at :2702; intake computed once outside the
  window loop (:2716-2725), reserves once (:2730-2736), both copied verbatim into all 5
  windows (:2774, :2780) — ML-INTAKE-IDENT / ML-RESERVE-IDENT (0/4,600 pass each,
  reproduced by re-running the sweep) are cl-hive constructions, exactly as attributed.
- burn/development lifetime fallback at :2757-2761 with `burn:` source_notes :2767;
  `_ledger_spend_totals_msat` at :2498. ML-BURN/DEV-IDENT 2,285/2,315 reproduced.
- The `unknown=22,859 / insufficient_coverage=131 / complete=10` breakdown was
  independently recomputed from the raw corpus JSON and matches exactly; the 10
  `complete` rows are solely the two 20260701T203541Z snapshots (2 nodes × 5 windows).
  The doc's correction of the "141 = post-fix" hint is itself correct.
- The covered_hours echo anomaly re-confirmed, and strengthened: `coverage_status:
  "complete"` is a hardcoded literal at *both* plugin writers (modules/database.py:3971
  and cl-revenue-ops.py:6590 — the total-cost-budget surface echoes `wh` the same way),
  so the falsely-confident-"complete" concern applies to every coverage-bearing surface
  the plugin emits, not just the spend ledger.

**Corrections (evidence hygiene, verdicts unaffected):**

1. Test-count arithmetic fixed inline: tests/test_capital_efficiency.py has 12 tests,
   not 31; the combined run is 31, not 50.
2. Corpus window fixed inline (2026-06-09 → 06-20 + 07-01; May quarantined).
3. The 131 `insufficient_coverage` windows are not only "from 2026-06-09": they span
   2026-06-09 → 2026-06-13 plus 2026-06-18, all on hive-nexus-01 (spot-verified
   source_notes on 20260609T034120Z show `burn:revenue_profitability_summary`, i.e. the
   spend-payload-missing fallback, exactly the mechanism the doc describes — the
   mechanism claim is right, the date scoping was narrow).
