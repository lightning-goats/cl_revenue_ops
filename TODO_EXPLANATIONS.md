# TODO Explanations

This document explains the open TODO items listed in `ROADMAP.md` (updated: December 16, 2025): what each item means, why it matters, where to implement it, and how to verify it’s done.

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
