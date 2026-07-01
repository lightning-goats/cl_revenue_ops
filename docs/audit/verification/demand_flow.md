# Phase 2 Verification — demand_flow.py

Contract: docs/audit/contracts/demand_flow.md (DF-1..DF-6).
Module byte-identical to contract commit f905cfd (no drift on HEAD cdb536a).
Evidence: unit-test mapping + run (tests/test_demand_flow.py 24 tests,
tests/test_demand_flow_integration.py 4 tests — all pass on HEAD, 2026-07-01),
code confirmation on HEAD. **Not corpus-observable**: no hermes artifact
isolates this module (its fingerprints land in `planner_candidates` DB rows,
and advisor.db is not collected); the sweep has no DF checks, so no invariant
here has corpus evidence — verdicts rest on tests and code only.

| Invariant | Verdict | Evidence |
|---|---|---|
| DF-1 flow roles at ±0.3 net-flow ratio; zero volume → unknown/0.0 | **verified** | TestPeerFlowProfiles (6 tests) PITs source (ratio 0.6), sink (−0.8), router (0.0), multi-channel aggregation (0.667), zero-volume → unknown with confidence 0.0, empty input → {}. Code confirmed (demand_flow.py:69-95). Exact boundary (ratio == ±0.3 → router) untested — minor gap |
| DF-2 flow confidence = clamp(0.3·log10(total)/log10(1e6), 0.1, 0.9) | **verified (code-only)** | Code confirmed (:85-86, order is min-then-max so both clamps bind). **No test asserts the formula or either clamp** (the only confidence assertion is the DF-1 zero-volume 0.0 path, which bypasses the formula). Contract note re-confirmed: 0.9 requires total = 1e18 sats; realized ceiling at 1M sats of volume is 0.3 |
| DF-3 gossip role = normalized argmax; zero score → ("unknown", 0.0) | **verified** | TestGossipHeuristics PITs role selection per keyword class (exchange→source, sink alias→sink, LSP→router, case-insensitive) and zero-signal → unknown. Confidence-share arithmetic not numerically asserted (partial). Code confirmed (:170-183; ties resolve source > sink > router by the ≥ chain). **Caveat: this is production-dead code — see anomaly 1** |
| DF-4 malformed gossip cannot raise | **verified** | 4 tests PIT it: non-string alias, string-msat amounts, unparseable amounts, string fee fields — all classify without raising. Code confirmed (`_safe_float` :33-40, `parse_msat`, dict-typed active filter :128) |
| DF-5 sink-adjacency proposes only new, active peers; top-5 sinks, ≤10 candidates | **verified** | TestSinkAdjacentDiscovery PITs existing-peer exclusion, ≤10 cap (20 offered → ≤10), empty input → []. Code confirmed (:203-233). Untested halves: inactive-channel skip (:218), top-5 sink cap (:207), cross-sink `seen` dedupe (:216). Production wiring confirmed on HEAD: capacity_planner.py:2094 (classify_peers), :2110 (find_sink_adjacent_candidates), :2934 (discovery), :2392-2395 (cached profile annotation) |
| DF-6 score = 0.4·conf·(1+(n−rank)/n), bounded ≤ 0.8 | **verified** | TestSinkAdjacentDiscovery::test_scores_by_sink_confidence PITs monotonicity in sink confidence; formula confirmed at :221 (rank 0 of n gives factor 2 → max 0.8·confidence; ≤ 0.72 under DF-2's 0.9 cap). Exact value / upper bound not asserted by any test (partial) |

## Gaps

- **DF-2 has no covering test at all** — the volume-scaled confidence formula
  and its [0.1, 0.9] clamps are code-only. Since DF-6 scores and the planner's
  candidate ranking multiply through this confidence, a silent formula change
  would re-rank open candidates with zero test signal.
- Boundary conditions untested across the module: ratio exactly ±0.3 (DF-1),
  inactive-channel skip / top-5 sink cap / dedupe (DF-5), score bound (DF-6).
- No corpus visibility: `planner_candidates` rows with source="demand_flow"
  live only in advisor.db (not collected), so whether this path has *ever*
  produced a production candidate is unverifiable from the frozen corpus.
  Phase 3/4 cannot evaluate the sink-adjacency theory from this dataset.

## Anomalies

1. **Half the module is production-dead, but fully tested**: `classify_candidate`
   (:97-191) and the three keyword lists (:16-30) have no caller outside
   tests/test_demand_flow.py (re-confirmed on HEAD via grep of modules/ and
   cl-revenue-ops.py). 12 of the 24 unit tests exercise dead code — coverage
   numbers overstate live-path assurance. If gossip-classification of open
   candidates is intended, the planner wiring is missing; if not, this is
   removable surface.
2. **`fee_extractive` is a dead signal inside the dead code**: recorded into
   gossip_signals at −0.2 (:159-164) but never added to any role score, so it
   can never influence the DF-3 argmax. Contract uncertainty confirmed on HEAD.
3. DF-2's practical confidence range (~0.1-0.5 for realistic volumes) caps
   DF-6 scores near ~0.4, well under the planner's other candidate sources'
   score scales (STRATEGY_WEIGHTS treats demand_flow at 1.0,
   test_demand_flow_integration.py) — sink-adjacent candidates are structurally
   faint. Not a defect, but relevant when Phase 3 asks why demand_flow
   candidates rarely win.

## Refutation pass (2026-07-01)

Adversarial re-verification on HEAD. `git diff f905cfd..HEAD -- modules/demand_flow.py`
re-confirmed empty; the full module (233 lines) was re-read line by line.

**No verdict flipped.** DF-1..DF-6 survived direct attack:

- **DF-1**: thresholds :78-83 (`> 0.3` / `< -0.3`, so exact ±0.3 → router — the doc's
  boundary-untested note is accurate), zero-volume early-out :73-75. All 6
  TestPeerFlowProfiles tests exist and pass.
- **DF-2**: formula :85-86 re-read — `min(0.9, ·)` then `max(0.1, ·)`, both clamps bind;
  0.9 ⇔ total = 1e18 sats and the 1M-sat realized ceiling of 0.3 both recomputed and
  correct. The "no covering test at all" claim was hunted for counterexamples and holds:
  no test asserts any confidence value from the volume formula (the only confidence
  assertion anywhere is the zero-volume 0.0 path).
- **DF-3**: argmax :170-183 with the `≥` chain resolving ties source > sink > router as
  stated; zero-score → ("unknown", 0.0) :171-173.
- **DF-4**: `_safe_float` :33-40, `parse_msat` usage :132/:152, dict-typed active filter
  :128; the four malformed-gossip tests exist (test_demand_flow.py :139, :145, :155, :171).
- **DF-5/DF-6**: :203-233 re-read — top-5 slice :207, active skip :218, `seen` dedupe
  :216, ≤10 truncation :233, score formula :221 (rank 0 of n gives factor 2 → 0.8·conf).
  Planner wiring re-confirmed at capacity_planner.py:2094 (classify_peers), :2110
  (find_sink_adjacent_candidates), :2934 (strategy 6 discovery), :2392-2395 (sink-adjacent
  1.4x boost / cached profile annotation).
- **Anomaly 1 (production-dead classify_candidate) re-confirmed by independent grep**:
  the only references to `classify_candidate` outside the module itself are in
  tests/test_demand_flow.py — no caller in modules/ or cl-revenue-ops.py. **Anomaly 2
  (fee_extractive dead signal) re-confirmed**: recorded at :164 into `signals` only; no
  score variable is ever decremented, so it cannot influence the argmax.

**Citation correction (evidence hygiene, verdicts unaffected):** tests/test_demand_flow.py
contains 23 tests, not 24 (23 + 4 integration = 27 pass on HEAD). Consequently the
"12 of the 24 unit tests exercise dead code" tally in Anomaly 1 is 12 of 23 —
the conclusion (coverage overstates live-path assurance) is unchanged.
