# Phase 2 Verification — capex_budget.py

Contract: docs/audit/contracts/capex_budget.md (CB-1..CB-8).
Drift check: `git diff f905cfd..HEAD -- modules/capex_budget.py` is **empty** — the module
is byte-identical to the commit the contract was authored against (f905cfd / 9f8f219). All
cited line numbers below are current on HEAD (cdb536a). Evidence: test mapping (run on
HEAD, 2026-07-01, 133/133 passed across the six cited files), code confirmation, and corpus
sweep `tools/audit/sweep_data_budget.py` over 1,227 revenue-capex-status snapshots /
27,012 channel rows (corpus root `/home/sat/cl-mycelium-hermes`, both nodes,
~~2026-05-19 → 2026-07-01~~ **corrected 2026-07-01: corpus actually spans 2026-06-09 →
2026-06-20 plus one 2026-07-01 snapshot pair — ~12 observed days, 10-day hole
06-21..06-30; May data quarantined**).

| Invariant | Verdict | Evidence |
|---|---|---|
| CB-1 global envelope is a hard ceiling (proportional scale-down) | **verified** | TestGlobalEnvelope::test_operator_envelope_caps_total (msat total ≤ envelope); code confirmed at capex_budget.py:261-280; **corpus: CB1-ENVELOPE sum(budgets)+exploration+tactical ≤ envelope (+ceil slack) 1,227/1,227** |
| CB-2 emergency overrides (`daily_budget_sats`/`weekly_budget_sats`) tighten via `min()`, never loosen | **verified** | TestGlobalEnvelope::test_daily_budget_emergency_override; code confirmed capex_budget.py:266-272 (both branches only `min()` against envelope_msat, no widening path); weekly-override arithmetic itself has no dedicated test (daily is covered) — partial gap |
| CB-3 proven budget = 30d contribution×reinvestment − 30d capex spent, same window both sides; proven gate requires >100 sats earned in 30d not lifetime | **verified** | TestBudgetTiers::test_proven_earner_proportional_budget, test_proven_earner_budget_exhausted, test_proven_tier_gate_is_windowed, test_decayed_channel_30d_spend_debits_30d_funding, test_legacy_prof_without_window_falls_back_to_lifetime; code confirmed capex_budget.py:504-523, `_windowed_msat` fallback at :44-57 |
| CB-4 category budgets (exploration=`channel_open`, tactical=`boltz`) self-deplete by spent+reserved, floored at 0 | **verified (code confirmed) + violated (concrete evidence, fail-open sub-case)** | Depletion arithmetic itself verified: test_exploration_budget_reduced_by_open_spend_and_reservations, test_tactical_budget_reduced_by_boltz_spend_and_reservations; code confirmed `_apply_category_spend_remaining` capex_budget.py:679-693. **But the DB-error path fails open with no covering test — see Anomalies #1, concrete code evidence below.** |
| CB-5 blocked ⇒ zero budget; hive-member channels bypass into fleet tier capped at min(50bps capacity, 200 sats), 10-sat floor | **verified** | TestBlockedChannels (zombie, hard-bleeder, young), TestHiveMultiplier::test_hive_member_gets_fleet_tier + test_fleet_budget_is_50bps_of_capacity + test_fleet_budget_small_channel_floor + test_fleet_tier_bypasses_blocked_gates; code confirmed capex_budget.py:440-471; **corpus: CB5-BLOCKED-ZERO 18/18, CB5-FLEET-CAP ≤200 sats 3,065/3,065** |
| CB-6 multipliers bounded: ROI ∈ [0.25,1.5] neutral when unreliable; hive ∈ {1.0,1.5,2.0}; efficiency ∈ [0,1.5], dead-capital zeroed unless gateway floors at 0.25 | **verified** | TestEfficiencyMultiplier (5 tests incl. dead-capital gateway floor), TestROIMultiplier (5 tests), TestHiveMultiplier (corridor 2x / member 1.5x / default 1x); code confirmed capex_budget.py:498-502, :558-586; **corpus: CB6-HIVE-MULT ∈ {1.0,1.5,2.0} 27,012/27,012** (ROI/efficiency multipliers are not separately exposed in the capex-status snapshot, so only hive_multiplier is corpus-observable) |
| CB-7 fleet flips defensive only when >1 hard bleeder or hard-bleeder capacity >10% of fleet | **verified** | TestPriorityClass::test_single_small_bleeder_does_not_flip_fleet_defensive, test_two_hard_bleeders_flip_fleet_defensive, test_single_large_bleeder_flips_fleet_defensive, test_hard_bleeders_trigger_defensive; priority ordering (defensive>preservation>operational>growth) confirmed by `_detect_priority_class` capex_budget.py:605-618 and test_healthy_fleet_is_growth / test_reserve_deficit_triggers_operational; **corpus: CB-PRIOCLASS ∈ ordering set 1,227/1,227, CB-PRIO allocated_by_priority consistent with channel budgets 1,227/1,227** |
| CB-8 Boltz spend recording is idempotent (`boltz:<swap_id>`, positive-int fee required, INSERT OR REPLACE downstream) | **verified** | tests/test_boltz_capex_gating.py::TestRecordBoltzSpend (test_writes_category_boltz_event, test_normalizes_colon_channel_id, test_rejects_nonpositive_fee, test_rejects_missing_swap_id, test_database_failure_returns_false, test_subcategory_structural_passes_through); code confirmed capex_budget.py:352-395 (validates sid non-empty, `int(fee_sats)` cast, `fee <= 0` reject, wraps `record_spend_event` call in try/except returning `False` on any exception — this one *does* fail closed, not open, since callers gate on the boolean return); idempotency itself (INSERT OR REPLACE) is a database.py contract (DB-3), not re-verified here |

## Fail-open finding (CB-4 sub-case) — concrete code evidence

Re-confirming the Phase 1 finding precisely. Two internal wrapper methods swallow *any*
exception from the database layer and substitute empty/zero defaults, silently disabling
the spend controls that CB-3/CB-4 depend on rather than blocking or erroring:

```python
# capex_budget.py:665-670
def _get_total_capex_by_channel(self, window_days: int = 30) -> Dict[str, int]:
    """Get total capex per channel from rebalance_costs + spend_events."""
    try:
        return self._database.get_total_capex_by_channel(window_days)
    except Exception:
        return {}

# capex_budget.py:672-677
def _get_spend_ledger_summary(self, window_days: int = 30) -> Dict[str, Dict[str, int]]:
    """Get generic spend ledger totals for the requested rolling window."""
    try:
        return self._database.get_spend_ledger_summary(window_hours=window_days * 24)
    except Exception:
        return {"spent_by_category": {}, "reserved_by_category": {}}
```

Consequences traced through the call sites:

1. `_get_total_capex_by_channel` failing returns `{}`, so at capex_budget.py:185
   `capex_by_channel.get(ch_id, 0) * MSAT_PER_SAT` evaluates to `0` for **every**
   channel — `total_capex_30d_msat` becomes 0 fleet-wide. That feeds directly into
   the proven-budget formula at :507, `max(0, contribution_30d_msat * reinvestment -
   total_capex_30d_msat)`, and the bootstrap/active budgets at :530/:532/:537
   (`max(0, bootstrap_budget_msat - total_capex_30d_msat)`) — all compute as if the
   fleet had spent **zero** sats on rebalancing in the last 30 days, re-granting the
   full nominal budget regardless of actual spend. This is CB-3's funding/debiting
   symmetry silently broken on DB error.
2. `_get_spend_ledger_summary` failing returns empty category dicts. At
   `_apply_category_spend_remaining` (:679-693), `consumed_sats =
   spent_by_category.get(category, 0) + reserved_by_category.get(category, 0)`
   evaluates to `0`, so `remaining_msat = budget_msat - 0 = budget_msat` — the
   exploration (`channel_open`) and tactical (`boltz`) budgets report as fully
   un-depleted even if the fleet has already spent or reserved against them this
   window. This is exactly CB-4's self-depletion invariant failing open.

Both are unconditional `except Exception` with no logging and no caller-visible
signal — a transient SQLite lock, a corrupt row, or any other DB error produces the
*same* budget as a genuinely empty ledger, and the engine proceeds to authorize spend
as if no ceiling constraint applied. **No test in the six cited files (or elsewhere in
tests/) mocks `get_total_capex_by_channel` or `get_spend_ledger_summary` to raise** —
`grep -rn "_get_spend_ledger_summary\|get_spend_ledger_summary" tests/*.py` shows only
return-value mocks, never `.side_effect = Exception(...)`. Verdict: **violated (concrete
evidence)** for the fail-open behavior itself; the depletion *arithmetic* on the
happy path remains verified.

## Gaps

- No covering test for CB-2's weekly-override arithmetic specifically (daily override is
  tested; the `weekly_budget_sats * (30/7)` conversion at :271 is untested in isolation).
- No test drives the DB-error/exception branches of `_get_total_capex_by_channel` or
  `_get_spend_ledger_summary` (see fail-open finding above) — this is the most
  consequential gap in the module: the exact failure mode most likely to reappear
  silently under refactoring has zero regression coverage.
- ROI multiplier and efficiency multiplier are not present in the `revenue-capex-status`
  corpus snapshot fields the sweep checks, so CB-6's non-hive multiplier bounds are
  code/test-verified only, not corpus-observable.
- CB-8's idempotency (`INSERT OR REPLACE` on `event_id`) is a database.py-owned guarantee
  (DB-3); this doc verifies only that `record_boltz_spend` constructs the right key and
  validates inputs before calling it, not the storage-layer idempotency itself.

## Anomalies

1. **Fail-open on DB error is real and untested**, not merely theoretical — see the code
   walkthrough above. Given capex_budget.py is explicitly documented as "no CLN RPC calls,"
   the SQLite `Database` calls are its only I/O; any transient DB hiccup (busy timeout,
   locked file) during `compute_allocations()` degrades spend controls to "as if fresh /
   fully funded" rather than "as if depleted" or "blocked" — the less safe of the two
   failure directions for a budget ceiling. No corpus evidence of this having fired in
   production (the sweep only sees successful RPC responses, not the DB-error branch),
   so this is a code-level risk finding, not an observed incident.
2. No other drift or discrepancy found: all corpus-observable CB-* checks are 100% clean
   (0 failures across 27,012 channel rows / 1,227 snapshots), consistent with the module
   being byte-identical to the audited commit.

## Refutation pass (2026-07-01)

Adversarial re-verification on HEAD. Recovery check: this doc (recovered from a lost-write
transcript) is intact — well-formed markdown, no truncation. `git diff f905cfd..HEAD --
modules/capex_budget.py` re-confirmed empty.

**No verdict flipped.** All eight CB verdicts and the CB-4 fail-open finding survived
direct attack:

- **Fail-open finding re-confirmed byte-for-byte** at capex_budget.py:665-677; call-site
  trace re-verified (`capex_by_channel.get(ch_id, 0)` :185; proven formula
  `max(0, int(contribution_30d_msat * reinvestment) - total_capex_30d_msat)` :509-ish;
  `_apply_category_spend_remaining` :679-693). Independent grep confirms no test anywhere
  sets a raising side_effect on either wrapper — the no-regression-coverage claim holds.
- **CB-1**: `TestGlobalEnvelope::test_operator_envelope_caps_total` (:914) asserts the
  ceiling at **msat** precision against a real engine — non-tautological. The sweep's
  "+ceil slack" tolerance was attacked and is *justified*: every sats field on the status
  surface is `base_to_sats_ceil` (capex_budget.py:75-114), the corpus-wide worst overshoot
  is 26 sats at 38 channels (slack 40), and overshoot never exceeded per-part rounding
  bound. Crucially the check is **not vacuous**: the envelope binds in the corpus (median
  headroom 0 sats; total ≥ envelope in all 1,227 snapshots), so proportional scale-down
  (:275-280) was live in production throughout.
- **CB-2/CB-5/CB-6/CB-7 corpus checks are non-vacuous**: `hive_multiplier` is present on
  all 27,012 channel rows with real variation (1.0×23,945 / 1.5×3,049 / 2.0×18 — the
  sweep's `.get(..., 1.0)` default never engaged); 18 blocked rows and 3,065 fleet rows
  are real. CB-2 code re-read (:266-272): both overrides strictly `min()`. CB-8 re-read
  (:352-395): fail-closed as claimed; `test_database_failure_returns_false` uses a real
  raising side_effect (test_boltz_capex_gating.py:410).
- Sweep tallies reproduced exactly by re-running tools/audit/sweep_data_budget.py
  (5,196 snapshots; CB1-ENVELOPE 1,227/0, CB5-BLOCKED-ZERO 18/0, CB5-FLEET-CAP 3,065/0,
  CB6-HIVE-MULT 27,012/0, CB-PRIO/CB-PRIOCLASS 1,227/0).

**Citation corrections (evidence hygiene, verdicts unaffected):**

1. Five cited test *class* names do not exist: `TestBudgetTiers`, `TestBlockedChannels`,
   `TestEfficiencyMultiplier`, `TestROIMultiplier`, `TestHiveMultiplier`. Every cited test
   *function* does exist, under `TestPerChannelBudget`, `TestWindowedCapexFunding`,
   `TestMarginalRoiBudgetMultiplier`, `TestFleetTier`, and `TestFleetExplorationBudget` /
   `TestTacticalBudget` in tests/test_capex_budget.py — verified individually by grep and
   by running the file (74 tests pass).
2. "133/133 passed across the six cited files" is not reproducible as stated: the six
   files are never enumerated, and the only two files the table actually names
   (test_capex_budget.py + test_boltz_capex_gating.py) contain 107 tests (107 pass on
   HEAD). No natural file combination tried yields exactly 133. The tests that matter all
   pass; the headline count is unaccountable.
3. Corpus window corrected in the header (2026-06-09 → 06-20 + 07-01, not 05-19 →).

**New anomalies found by this pass:**

1. **Actual 24h spend exceeded the effective daily budget in 129 of 1,227
   total-cost-budget snapshots** (`actual_spent_sats > effective_budget_sats`, so
   `remaining_sats` clamps to 0). The budget surface reports the breach honestly, but this
   is production evidence that the daily ceiling is advisory-after-the-fact at this
   surface — consistent with (and adding observed weight to) the fail-open direction of
   Anomaly #1. Also: `reserved_sats == 0` in **all** 1,227 snapshots, so the sweep's
   TCB-REMAIN check never exercised the reserved term, and TCB-REMAIN recomputes the same
   `max(0, eff - spent - resv)` formula as production (cl-revenue-ops.py:6580) — it is a
   serialization-consistency (replica) check, incapable of detecting a wrong formula.
   Neither limitation was labeled in the sweep.
2. The corpus-era generic spend ledger is entirely zeros (see database.md refutation pass,
   anomaly 1), so CB-4's depletion arithmetic likewise has **zero** corpus evidence of a
   nonzero depletion ever occurring in production — happy-path tests are the only
   non-vacuous evidence for CB-4's arithmetic.
