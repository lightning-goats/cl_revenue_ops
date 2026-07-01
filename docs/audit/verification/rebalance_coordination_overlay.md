# Verification: modules/rebalance_coordination_overlay.py

Phase 2 — Tier 2 (targeted). Verified 2026-07-01 at HEAD 61b031c against
`docs/audit/contracts/rebalance_coordination_overlay.md` (authored at f905cfd).
Module unchanged since the contract commit (`git diff f905cfd HEAD --
modules/rebalance_coordination_overlay.py` is empty), so all module-internal
line references in the contract are still exact. Engine call-site references
drifted after 441b8e3 (see Anomalies).

Test run: `.venv/bin/python -m pytest tests/test_rebalance_coordination_overlay.py
tests/test_rebalance_economics_fixes.py -q` → green (part of a 116-passed batch);
engine integration tests `tests/test_rebalance_engine_v2.py -k "coordination or
overlay or segment_score"` → 12 passed.

Corpus context (frozen sweep, `tools/audit/sweep_routing_stack.py`, results in
scratchpad `sweep_routing_stack.out`): **zero coordinated episodes** in the
whole corpus — `coordination_unavailable` appears 788 times in skip histograms
(the unavailable-skip path at :251-297 is the single most-exercised overlay
path), while `coordination_unresolvable_endpoint`, `not_designated_executor`,
`lease_conflict`, `fleet_lease_held`, and `coordination_preempted` never appear
and no history row carries `reason_code=coordinated_rebalance` (history reason
codes: defibrillator 37, manual 10, ev_positive 4). Consequently **every
happy-path invariant below is corpus-vacuous**; verdicts rest on tests + code.

## Invariants

- **RCO-1 (no wildcard endpoint rebinding)** — **verified.** Code:
  `_resolve_endpoint` :120-152 (unknown SCID → `(None, True)` at :141;
  no SCID + no peer → :143-145), skip emission :235-249. Tests (all genuinely
  pit — each feeds a hostile hint and asserts the skip reason):
  `tests/test_rebalance_coordination_overlay.py::`
  `test_overlay_skips_foreign_scid_hint_with_empty_peers`,
  `::test_overlay_skips_hint_with_no_endpoints_at_all`,
  `::test_overlay_skips_campaign_without_endpoints_or_active_chunk`,
  `::test_overlay_does_not_wildcard_when_scid_unknown_but_peer_given` (SCID
  unknown but valid peer given — proves SCID mismatch is terminal, no peer
  fallback), plus positive control
  `::test_overlay_resolves_endpoint_by_peer_when_scid_missing`. Corpus:
  reason never observed (vacuous). Run: pass.
- **RCO-2 (executor designation gating)** — **verified.** Code:
  `_executor_gate_skip` :155-184 (empty primary → no gating :167-169; we are
  primary → :170-171; fallback membership → :173-178). Tests:
  `::test_overlay_skips_hint_designating_another_executor` (us absent from
  fallbacks → `not_designated_executor`),
  `::test_overlay_allows_fallback_executor`,
  `::test_overlay_no_executor_gating_when_fields_missing` (back-compat empty
  string). Corpus-vacuous. Run: pass.
- **RCO-3 (amount = min(chunk, excess, need, hint); distinct endpoints)** —
  **verified (code-only).** Code confirms every clause: distinct channel AND
  peer check :258-264 (skip `coordination_unavailable` detail
  `invalid_pairing`); amount min :284-290 (hinted amount falls back to
  `min(excess, need)` when absent); non-positive skip :291-297. **No test pits
  any of these clauses**: no overlay test asserts `amount_sats` on a built
  pair, none constructs a same-peer source/sink hint, none drives
  `amount_not_viable`. (Planner-side amount arithmetic is tested, but not this
  module's.) Corpus-vacuous.
- **RCO-4 (planner-unit scores, bounded priority multiplier, no double
  segment bias)** — **verified.** Code: :266-283 (0.30 × `_refill_urgency` +
  0.20 × `_drain_score`, priority via `_priority_score` :110-117 clamped to
  `MAX_HINT_PRIORITY_SCORE = 100.0`, rebalance_route_policy.py:19, multiplier
  ≤ ×1.15); deliberate no-bias comment :340-343. Tests:
  `::test_overlay_score_is_normalized_to_planner_units` (exact expected value
  base × 1.135), `::test_overlay_score_priority_multiplier_is_bounded`
  (priority 1e6 → exactly base × 1.15),
  `::test_overlay_does_not_apply_segment_bias_itself` (segment-aware vs plain
  hints produce identical scores),
  `::test_low_merit_coordination_score_comparable_with_planner_scores`, and
  `tests/test_rebalance_economics_fixes.py::test_f4_overlay_score_matches_planner_coefficients`,
  `::test_f4_overlay_priority_multiplier_preserved`,
  `::test_f4_identical_state_planner_and_overlay_scores_comparable` (planner −
  overlay == exactly the dest-value term on identical state). Run: pass.
- **RCO-5 (foreign active leases suppress; ours never; terminal/id-less
  ignored)** — **verified** for owner matching, **verified (code-only)** for
  the terminal-status/no-`lease_id` clause. Code: `_lease_is_active` :350-354
  (terminal set + `lease_id` required), owner skip :374-380, segment
  intersection :382. Tests:
  `::test_overlay_skips_candidate_when_foreign_lease_overlaps_route_segments`
  (skip reason `lease_conflict`, lease id in detail),
  `::test_overlay_keeps_pair_when_overlapping_lease_is_ours`; engine reuse
  pitted by `tests/test_rebalance_engine_v2.py::`
  `test_fleet_lease_suppresses_matching_planner_pair` (reason
  `fleet_lease_held`) and
  `::test_fleet_lease_owned_by_us_does_not_suppress_planner_pair`. **No test
  feeds an expired/released lease or a lease without `lease_id`** and asserts
  the pair proceeds. Corpus-vacuous. Run: pass.
- **RCO-6 (reserved-slot merge; strict planner cap; dedup by max-score;
  one role per channel; preempted skips)** — **verified** for the slot
  arithmetic and coordination precedence, **verified (code-only)** for the
  duplicate-key merge and the `coordination_preempted` skip record. Code:
  slot logic :492-551 (coord overflow competes in planner pool :533-541),
  duplicate-key max-score merge :504-527, used-source/dest sets :528-529,
  preempted skip :571-581. Tests:
  `::test_engine_preserves_coordinated_candidate_even_when_local_pair_score_is_lower`
  (score-10 planner pair displaced by score-0.1 coordination pair at
  max_pairs=1 — genuinely pits precedence),
  `::test_reserved_slots_zero_preserves_strict_cap`,
  `::test_reserved_slots_let_coordination_bypass_cap`,
  `::test_plan_pairs_still_respect_max_pairs_when_reserved_unused`,
  `::test_coordination_cannot_exceed_reserved_plus_max` (10 coord,
  reserved=2, max=3 → exactly 5), `::test_negative_reserved_clamped_to_zero`;
  engine wiring `tests/test_rebalance_engine_v2.py::`
  `test_engine_merges_coordination_pairs_before_pair_cap`. **No test asserts
  the max-score merge of a duplicate (source,dest) key, the once-per-channel
  source/dest exclusion, or that a displaced planner pair yields a
  `coordination_preempted` SkipRecord** (the strict-cap test checks selection
  only, not `plan.skipped`). Run: pass.
- **RCO-7 (campaign status admission; per-build dedup)** — **verified
  (code-only).** Code: status gate :96-99 (blank or {active, running,
  pending}), `seen_pairs` dedup :415-436. **No test feeds a
  completed/cancelled campaign to the overlay, and no test feeds duplicate
  hints in one build.** Note the adapter interplay: `hive_hints` adapter tests
  (tests/test_hive_hints.py:1750) show status `"queued"` campaigns pass the
  adapter — the overlay's own gate then silently drops them (no skip record
  is emitted for a status-rejected campaign, it is filtered in
  `_coordination_entries` before pair building). Corpus-vacuous.

## Purpose-section claims

- Pure functions, no RPC/state: **verified** — imports are types/helpers only
  (:1-23).
- Config `rebalance_coordination_reserved_slots` default 2, clamp 0-10:
  **verified** — modules/config.py:486 (default 2), :278 (clamp (0, 10)),
  threading at rebalance_engine_v2.py:1375-1380.
- `build_coordination_pairs` dead code: **confirmed** — no caller in
  `modules/` or `cl-revenue-ops.py`; only tests use it (as a selected-only
  convenience wrapper). Contract's uncertainty stands.
- Engine applies `pair_segment_bias_multiplier` uniformly: **verified** —
  rebalance_engine_v2.py:1440 (`_apply_segment_score_bias` per selected pair,
  defined :1746) + `test_engine_applies_segment_score_bias_to_pair_score`
  (which, however, only asserts `score > 100.0` — see Gaps).

## Gaps

1. RCO-3 is entirely test-uncovered at this module's level: no assertion on
   built-pair `amount_sats`, no same-peer pairing test, no `amount_not_viable`
   test.
2. RCO-5's fail-open clauses (`_lease_is_active`: terminal statuses, missing
   `lease_id`) are unpitted — a regression inverting the status set would pass
   the suite.
3. RCO-6's duplicate-key max-score merge, once-per-channel constraint, and
   `coordination_preempted` skip emission are unpitted.
4. RCO-7's campaign status gate and per-build dedup are unpitted at the
   overlay level.
5. `test_engine_applies_segment_score_bias_to_pair_score` is nearly
   tautological (`score > 100.0`): it does not pin the bound, so the ±12%/±10%
   discrepancy below is invisible to the suite.
6. Corpus contains zero coordinated episodes and zero occurrences of five of
   the six overlay skip reasons, so no production evidence exists for any
   selection-path invariant; only `coordination_unavailable` (788) is
   production-exercised.

## Anomalies

1. **Advertised bias bound unreachable**: contract and function docstring say
   "bounded ±12%", but `pair_segment_bias_multiplier` clamps
   `average_utility * 0.10` to ±0.12 (:64) where `average_utility` ∈ [-1, 1] —
   the effective bound is ±10% and the ±0.12 clamp can never bind. Cosmetic,
   fail-safe direction, but the documented constant is wrong.
2. **Engine call-site line drift** (contract authored pre-441b8e3):
   overlay build/merge now at rebalance_engine_v2.py:1362-1381 (was
   :1256-1275), lease reuse :1403-1411 (was :1297-1308), segment bias
   application :1440 / :1746-1760 (was :1332-1334 / :1640-1650), executor
   factory :2366-2373 (was :2260-2268). Module-internal references unaffected.
3. **Skip-reason vocabulary drift** (cross-reference, already recorded in
   docs/audit/verification/rebalance_audit_v2.md RA2-1): none of
   `coordination_unresolvable_endpoint`, `coordination_unavailable`,
   `not_designated_executor`, `lease_conflict`, `fleet_lease_held`,
   `coordination_preempted` is in `VALID_SKIP_REASONS`
   (modules/rebalance_audit_v2.py:29-49). Not re-derived here.
4. Status-rejected campaigns and per-build duplicate hints are dropped
   silently (no SkipRecord), unlike every other rejection path in the module —
   contradicts the "every decision explained" spirit but matches the contract
   text, which only promises skips for the enumerated reasons.

## Refutation pass (2026-07-01)

Adversarial re-verification at HEAD dac9b48 (module byte-identical to f905cfd
through HEAD, matching this doc's drift check). Method: mutation testing in a
scratch copy + frozen-corpus re-sweep.

- Attacked: RCO-1, RCO-2, RCO-4, RCO-5, RCO-6, purpose claims, corpus
  statements.
- Survived — every decisive mutation was killed:
  RCO-1 (letting an unknown SCID fall through to peer binding kills
  `test_overlay_does_not_wildcard_when_scid_unknown_but_peer_given`);
  RCO-2 (blanking `primary_executor_member_id` handling kills
  `test_overlay_skips_hint_designating_another_executor`);
  RCO-4 coefficients (0.30 → 0.35 kills the normalized-score test and an F4
  economics test — the exact-value assertions are real);
  RCO-5 (ignoring lease ownership kills
  `test_overlay_keeps_pair_when_overlapping_lease_is_ours`).
  RCO-3/RCO-7 code-only cites re-read exact (amount-min clause,
  `invalid_pairing`, `amount_not_viable`, status gate :96-99); the
  documented test-coverage gaps stand.
- RCO-4 note: removing the `min(priority, MAX_HINT_PRIORITY_SCORE)` clamp in
  the multiplier expression survives the bounded-multiplier test — but only
  because `_priority_score` (:110-117) independently clamps to the same
  bound; the invariant is protected by redundant defense-in-depth, and
  removing both clamps is caught. Not a refutation; recorded so a future
  "simplification" removing one clamp is not mistaken for safe.
- RCO-6 note: a mutation forcing coordination pairs through the planner-pool
  capacity branch survives all 25 overlay tests, but analysis shows it is
  semantically equivalent in every tested configuration (coordination pairs
  merge before planner pairs, so `plan_count == 0` during coordination
  admission and the two branches compute identical outcomes; with
  reserved=0 the branches are textually identical). The precedence and
  reserved-slot tests genuinely pit observable behavior; no flip.
- Corpus: this doc's numbers match the frozen sweep exactly (re-run
  2026-07-01: coordination_unavailable 788, zero coordinated episodes, zero
  occurrences of the other five skip reasons, history reason codes
  defibrillator 37 / manual 10 / ev_positive 4). The doc correctly labels
  every happy-path invariant corpus-vacuous — the standard this campaign
  requires.

Counts: attacked 7 invariants + 4 purpose claims; survived all; refuted 0.
