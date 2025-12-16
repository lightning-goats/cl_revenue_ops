# TODO Explanations

This document explains the open TODO items listed in `ROADMAP.md` (updated: December 16, 2025): what each item means, why it matters, where to implement it, and how to verify it’s done.

## Align wallet reserve definition (Phase 1)

**Problem**
- The config comment suggests reserve is “on-chain + receivable” (`modules/config.py`: `min_wallet_reserve`), but the current enforcement logic in `EVRebalancer._check_capital_controls()` computes reserve as **confirmed on-chain outputs + our spendable channel balance** (`modules/rebalancer.py`: `_check_capital_controls`).

**Why it matters**
- Reserve protection is a safety invariant. A mismatch between documentation/config semantics and enforcement can cause unexpected aborts (too strict) or unsafe operation (too lax).

**Likely touch-points**
- `modules/config.py`: `min_wallet_reserve` description/meaning.
- `modules/rebalancer.py`: `EVRebalancer._check_capital_controls()` reserve calculation.
- `README.md`: reserve/budget documentation.

**Done when**
- Documentation and code agree on what counts toward reserve.
- `min_wallet_reserve` is described in the same terms everywhere (config + README + logs).

**AI Prompt**
```text
Repo: cl_revenue_ops (Python Core Lightning plugin).

Task: Align the definition of “wallet reserve” across code and docs.

Context:
- `modules/config.py` has `min_wallet_reserve` with comment “Min sats (on-chain + receivable) before ABORT”.
- `modules/rebalancer.py` `EVRebalancer._check_capital_controls()` computes reserve as confirmed on-chain outputs + our local spendable channel balance (our_amount_msat on CHANNELD_NORMAL).

Requirements:
1) Decide and implement ONE canonical definition (either update docs/comments to match current behavior, or update the computation to match the documented meaning). Prefer the smallest change that removes ambiguity.
2) Ensure logs and README wording match the chosen definition and units (sats).
3) Do not introduce new features or settings; keep changes surgical.

Acceptance:
- No contradictions between `modules/config.py`, `README.md`, and runtime log message “CAPITAL CONTROL: Wallet reserve …”.
- `python -m py_compile` succeeds for touched files.

Deliverable:
- Briefly state the definition chosen and list the files changed.
```

## Fix revenue-history pruning issue (Phase 2)

**Problem**
- Lifetime revenue is aggregated from the `forwards` table (used by `revenue-history`), but the database prunes old rows (`modules/database.py`: `cleanup_old_data()` includes `DELETE FROM forwards WHERE timestamp < ?`).

**Why it matters**
- “Lifetime” reporting becomes “last N days” once old rows are pruned, silently undermining accounting accuracy.

**Likely touch-points**
- `modules/database.py`: `cleanup_old_data()` pruning policy and call sites.
- `modules/database.py`: `get_lifetime_stats()` (source of `revenue-history`).
- `modules/profitability_analyzer.py`: `get_lifetime_report()`.
- `cl-revenue-ops.py`: where cleanup is scheduled/invoked.

**Done when**
- `revenue-history` remains correct even after pruning.
- No double-counting across rollups + remaining raw rows.
- A migration exists if schema changes are required.

**AI Prompt**
```text
Repo: cl_revenue_ops.

Task: Make `revenue-history` truly lifetime-correct even when old `forwards` rows are pruned.

Context:
- `modules/database.py:cleanup_old_data()` deletes old rows from `forwards`.
- `modules/database.py:get_lifetime_stats()` sums revenue from `forwards` for `revenue-history`.
- RPC `revenue-history` is in `cl-revenue-ops.py` and calls `modules/profitability_analyzer.py:get_lifetime_report()`.

Requirements:
1) Keep pruning (for DB size), but preserve lifetime revenue correctness.
2) Prefer a minimal, maintainable approach:
   - Add a rollup table (e.g., daily forward fee totals) that is updated before pruning, OR
   - Add a dedicated “lifetime aggregates” table that is incrementally maintained.
3) Add a migration under `migrations/` if you add tables/columns.
4) Update README if behavior changes (what lifetime means, retention windows).

Acceptance:
- After calling `cleanup_old_data()`, `revenue-history` outputs the same lifetime totals as before cleanup (within any explicitly documented rounding behavior).
- No double-counting across rollups + remaining raw rows.
- `python -m py_compile` succeeds for touched files.

Deliverable:
- List schema changes, where rollups are updated, and how pruning interacts with lifetime accounting.
```

## Record rebalance_costs on success (Phase 2)

**Problem**
- Costs for reporting are summed from `rebalance_costs`, but many execution paths only update `rebalance_history.actual_fee_sats` via `Database.update_rebalance_result()` (`modules/rebalancer.py` calls). That can undercount costs in `revenue-history`.

**Why it matters**
- If costs are undercounted, ROI and net profit are overstated.

**Likely touch-points**
- `modules/rebalancer.py`: success/finalization path(s).
- `modules/database.py`: `record_rebalance_cost()`.
- `modules/profitability_analyzer.py`: reconciliation between db and bookkeeper.

**Done when**
- Every successful rebalance persists its fee into the canonical cost store used by reporting.
- `revenue-history` costs align with `rebalance_history` / bookkeeper within expected tolerances.

**AI Prompt**
```text
Repo: cl_revenue_ops.

Task: Ensure successful rebalances are reflected in lifetime cost accounting.

Context:
- `modules/database.py` supports `record_rebalance_cost(channel_id, peer_id, cost_sats, amount_sats, timestamp)` and reporting sums from `rebalance_costs`.
- Rebalancer paths commonly call `Database.update_rebalance_result(..., actual_fee_sats=...)` which updates `rebalance_history` but may not write into `rebalance_costs`.

Requirements:
1) Confirm and document the canonical source for rebalance costs used by reporting.
2) On rebalance success, write a single entry into `rebalance_costs` with:
   - `channel_id`
   - `peer_id`
   - `cost_sats` (actual_fee_sats)
   - `amount_sats`
   - `timestamp`
3) Make it idempotent: do not double-record the same rebalance.

Acceptance:
- After a successful rebalance, `rebalance_costs` increases accordingly and `revenue-history` reflects it.
- No duplicates for a single rebalance attempt.
- `python -m py_compile` succeeds for touched files.

Deliverable:
- Point to the exact code path(s) changed and explain the idempotency strategy.
```

## Reconcile README option names (Phase 2)

**Problem**
- Option/setting names and their meanings can drift between `README.md` and implementation (`modules/config.py` + `cl-revenue-ops.py`).

**Why it matters**
- Incorrect option names cause misconfiguration and confusing behavior.

**Likely touch-points**
- `README.md`: “Options/Configuration” sections.
- `modules/config.py`: config fields and defaults.
- `cl-revenue-ops.py`: option parsing and plugin initialization.

**Done when**
- Every documented option matches a real config key and behavior.
- Defaults and units (sats, ppm, msat, hours/days) are consistent.

**AI Prompt**
```text
Repo: cl_revenue_ops.

Task: Reconcile documented options with actual implementation.

Requirements:
1) Audit `README.md` options/config sections and compare against:
   - `modules/config.py` fields + defaults
   - `cl-revenue-ops.py` plugin option parsing / initialization
2) Fix mismatches by updating documentation first (prefer docs-only changes unless an option is truly missing but referenced).
3) Ensure units are stated consistently (sats vs msat, ppm, thresholds as floats, time windows in days/hours).

Acceptance:
- Every option mentioned in README exists and is spelled identically in code.
- Defaults in README match `modules/config.py`.

Deliverable:
- Provide a short list of mismatches you fixed and where.
```

## Apply congested-state protections (Phase 3)

**Problem**
- Channels can be marked congested (HTLC slot utilization threshold), but downstream decisions (fee changes, rebalancer selection) may not consistently treat congestion as a protection condition.

**Why it matters**
- Rebalancing or fee adjustments on congested channels can worsen HTLC contention and reduce forwarding success.

**Likely touch-points**
- `modules/flow_analysis.py`: congestion detection/representation.
- `modules/rebalancer.py`: candidate selection and execution guards.
- `modules/fee_controller.py`: fee update logic.

**Done when**
- Congested channels are handled according to an explicit policy, and the behavior is visible via logs/metrics.

**AI Prompt**
```text
Repo: cl_revenue_ops.

Task: Apply consistent protections when a channel is marked congested (HTLC slots).

Context:
- `modules/config.py` defines `htlc_congestion_threshold`.
- Flow/analysis marks channels as CONGESTED.

Requirements:
1) Choose a minimal protection policy and implement it consistently. Examples:
   - Rebalancer: do not pick congested channels as destinations; optionally also avoid as sources.
   - Fee controller: reduce fee churn on congested channels (skip updates or cap change rate).
2) Implement with the smallest number of conditionals at the decision points.
3) Add clear logs and/or Prometheus metrics so operators can tell congestion is gating behavior.
4) Update README to describe the policy.

Acceptance:
- Congested channels are visibly treated differently in logs/metrics.
- No change in behavior for non-congested channels.
- `python -m py_compile` succeeds for touched files.

Deliverable:
- List the exact guards added (file/function) and the chosen policy.
```

## Align reputation default weighting docs (Phase 3)

**Problem**
- Documentation can drift from the actual defaults and formula used for reputation-weighted volume (e.g., neutral fallback score, Laplace smoothing formula, decay factor).

**Why it matters**
- Reputation weighting impacts fee control and rebalancer sizing; inaccurate docs make tuning difficult.

**Likely touch-points**
- `modules/config.py`: `enable_reputation`, `reputation_decay`.
- `modules/database.py`: `get_weighted_volume_since()` and reputation scoring/decay.
- `README.md`: reputation documentation.

**Done when**
- README matches implementation for:
  - Default neutral score when no reputation exists
  - The Laplace smoothing formula used
  - The decay schedule and factor

**AI Prompt**
```text
Repo: cl_revenue_ops.

Task: Align README documentation with reputation-weighted volume implementation defaults.

Context:
- `modules/config.py` has `enable_reputation` and `reputation_decay`.
- `modules/database.py:get_weighted_volume_since()` uses Laplace smoothing and a neutral fallback score (0.5) when no reputation exists.

Requirements:
1) Update `README.md` to reflect the actual defaults and formulas used in code.
2) Prefer docs-only changes unless you find a genuine bug/inconsistency in code comments.
3) Clearly state units and timing (e.g., decay applied per flow interval).

Acceptance:
- README matches `modules/config.py` defaults and `modules/database.py` behavior.

Deliverable:
- Quote the final documented formula/behavior and point to the corresponding code locations.
```

## Clarify bookkeeper vs listforwards usage (Phase 4)

**Problem**
- The codebase uses bookkeeper RPCs when available with fallbacks to `listforwards`, and profitability mixes db-derived forwards/costs with bookkeeper-derived events. Without a clear policy, users won’t know which data source is authoritative.

**Why it matters**
- Different RPC sources can disagree (coverage windows, channel id formats, accounting granularity). Reporting should be explainable and predictable.

**Likely touch-points**
- `modules/flow_analysis.py`: bookkeeper availability checks and fallbacks.
- `modules/profitability_analyzer.py`: bookkeeper summation logic.
- `README.md`: data sources and prerequisites.

**Done when**
- README documents a stable hierarchy of sources (bookkeeper vs `listforwards` vs local DB) that matches actual behavior.
- Reporting commands mention their data sources.

**AI Prompt**
```text
Repo: cl_revenue_ops.

Task: Clearly document and (if needed) standardize the hierarchy of data sources: bookkeeper RPCs vs `listforwards` vs local DB.

Context:
- `modules/flow_analysis.py` uses bookkeeper when available and falls back to `listforwards`.
- `modules/profitability_analyzer.py` uses bookkeeper summation logic for batch tx fees and mixes sources.

Requirements:
1) Write a short “Data sources” section in `README.md` that explains:
   - Which RPCs are used for flow/volume, fees, and costs
   - When fallbacks are used
   - Known differences (coverage windows, channel id formats)
2) If the code contradicts the documented policy, make the smallest code change necessary OR adjust the policy to match reality.
3) Ensure reporting commands (`revenue-profitability`, `revenue-history`) mention their data sources.

Acceptance:
- README provides an unambiguous hierarchy and matches actual behavior.

Deliverable:
- Provide the final hierarchy bullets and point to the key code paths.
```
