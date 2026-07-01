# Verification: modules/rebalance_planner_v2.py

Phase 2 — Tier 2 (targeted). Verified 2026-07-01 at HEAD 61b031c against
`docs/audit/contracts/rebalance_planner_v2.md` (authored at f905cfd against
9f8f219).

**Drift check (441b8e3, 2026-06-27 "Improve rebalance EV criteria")**: the
planner diff is +13/−2 (15 changed lines; refutation pass corrected the
"+15/−2" miscount against `git diff --stat f905cfd..HEAD`) — (a) removed
the unused `src_value` lookup, (b)
threads four new historical role-fee fields
(`source/dest_historical_direct_fee_ppm`, `source/dest_historical_sourced_fee_ppm`)
from `ChannelState` into `PairCandidate` (new fields in
modules/rebalance_types_v2.py:41-47; snapshot side +30 lines in
rebalance_state_v2.py). **No RP2 invariant semantics changed.** All RP2 claims
below re-verified on HEAD. Line drift: RP2-6's cited helpers moved —
`_normalize_rebalance_bias` is now :390-396 (was :379-385) and
`_pair_hint_multiplier` :398-411 (was :387-400); `value_score` diagnostic now
:330 (was :331). All other cited ranges are still exact. The new fields are
consumed by the engine's sats-EV gate and pitted by
`tests/test_rebalance_economics_fixes.py::test_f2f_historical_destination_and_source_value_feed_sats_ev`
and `tests/test_rebalance_engine_v2.py::test_v2_planner_carries_historical_profitability_role_metrics`.

Test run: `.venv/bin/python -m pytest tests/test_rebalance_planner_v2.py
tests/test_drain_demand.py tests/test_rebalance_economics_fixes.py -q` → green
(part of a 116-passed batch); `tests/test_rebalance_engine_v2.py -k
"rebalance_bias or historical or overlay"` → passed.

Corpus context (frozen sweep, `tools/audit/sweep_routing_stack.py`): planner
skip reasons dominate the corpus — `inside_band` 3,316, `no_partner` 2,984,
`outcompeted` 1,032 (`max_pairs_reached` absent; `below_hold_margin` 177 is
the engine's EV gate, not this module). All 178 priced candidates carry
`reason_code=ev_positive`, `route_policy=market_only`. 0 sweep violations.

## Invariants

- **RP2-1 (band classification precedes role eligibility)** — **verified.**
  Code: :128-155 (band test first, then `source_eligible`/`dest_eligible`,
  reason passthrough from state with `source_ineligible`/`dest_ineligible`
  fallbacks). Tests (all pit): `tests/test_rebalance_planner_v2.py::`
  `TestSkipReasons::test_skips_inside_band_channels`,
  `::test_over_local_in_cooldown_skipped_as_protected_source` (source_reason
  passthrough), `::test_destination_skipped_when_not_valuable`,
  `::test_destination_skipped_when_no_budget`,
  `::test_over_local_neutral_is_eligible_source` and
  `::test_over_local_no_budget_is_eligible_source` (role asymmetry — the
  same disqualifiers do NOT block the source role). Corpus: `inside_band`
  3,316 — the most-exercised skip path in production. Run: pass.
- **RP2-2 (every channel explained)** — **verified** for each reason path;
  the whole-set coverage property itself is **verified (code-only).** Code:
  :176-210 — each unpaired over_local/over_remote channel gets exactly one of
  `no_partner`/`max_pairs_reached`/`outcompeted`; combined with :128-155
  every classified channel appears exactly once in selected-or-skipped.
  Tests: `::test_emits_no_partner_skip_when_no_opposite_side` (both sides),
  `::test_emits_outcompeted_skip_for_losing_sources`,
  `::test_emits_max_pairs_reached_skip`,
  `TestScoring::test_polar_s9_shape_recognizes_neutral_sources` (mixed
  reasons in one plan). No test asserts the exhaustiveness property as a set
  equation (skipped ∪ selected endpoints == all channels). Corpus:
  `no_partner` 2,984 / `outcompeted` 1,032 observed. Run: pass.
- **RP2-3 (one role per channel; ≤ max_pairs; score-descending)** —
  **verified.** Code: :158-174 (`pairs.sort` desc :163, `paired_sources`/
  `paired_dests` refusal, `len(candidates) >= self.max_pairs` break). Tests:
  `::test_emits_max_pairs_reached_skip` (max_pairs=1 → 1 selected),
  `TestScoring::test_scores_hive_channels_higher` (one dest cannot pair
  twice; higher-scoring source wins),
  `::test_cheaper_return_source_wins_when_other_terms_equal` (insertion order
  explicitly ruled out). Run: pass.
- **RP2-4 (no self-pairing; bounded positive amounts)** — **verified** for
  the amount arithmetic; the same-peer skip is **verified (code-only).**
  Code: peer check :248-249, amount = min(excess, need, max_chunk) > 0
  :251-264. Test: `TestPairGeneration::test_computes_correct_transfer_amount`
  (min(250k, 125k, 2M) = 125k ± 1 sat int truncation). **No test constructs
  two channels to the same peer on opposite band sides** to pit :248-249.
  Run: pass.
- **RP2-5 (destination authorizes spend)** — **verified.** Code: :273-278
  (`pair_budget = dest.remaining_budget_sats`, optional
  `ceil(amount × ppm / 1e6)` floor). Tests (all pit exact values):
  `::test_destination_budget_authorizes_pair_spend` (source 5000 vs dest 500
  → 500), `::test_pair_budget_uses_fee_cap_ppm_when_capex_too_low` (max(200,
  250) = 250), `::test_pair_budget_keeps_capex_when_higher_than_fee_cap`,
  `::test_pair_budget_zero_fee_cap_keeps_legacy_capex_only_behavior`,
  `::test_source_safety_controls_survive_destination_led_budget`. Downstream
  the envelope is pitted by tests/test_rebalance_economics_fixes.py F1 tests
  (`test_f1_budget_does_not_double_as_fee_envelope` etc.). Run: pass.
- **RP2-6 (hive bias clamped to ±15%, source-inverted)** — **verified** for
  the inversion and combined multiplier; the clamp boundaries are only
  pitted upstream. Code: `_normalize_rebalance_bias` :390-396 (clamp
  [0.85, 1.15]), `_pair_hint_multiplier` :398-411 (source inversion +
  combined clamp). Test: `tests/test_rebalance_engine_v2.py::`
  `test_v2_planner_uses_hive_rebalance_bias_for_pair_roles` — source bias
  0.95 / dest 1.05 → multiplier exactly 1.10, source-preferred channel wins,
  decomposition fields carried, `score > pre_hint_pair_score`. **No test
  feeds this module an out-of-range raw bias** (e.g. 2.0 → expect 1.15) or a
  combination whose raw multiplier exceeds 1.15 (dest 1.15 + source 0.85 →
  raw 1.30 → clamp 1.15); the [0.85, 1.15] range is enforced-and-tested at
  the HiveHintAdapter (tests/test_hive_hints.py:733,922;
  test_cross_plugin_contracts.py:182), so the planner clamp is currently
  redundant defense-in-depth — unpitted at this layer. Run: pass.
- **RP2-7 (drain demand = unpaired over-local residual)** — **verified.**
  Code: :212-234 (entries only for `over_local` not in `paired_sources`,
  excess via `_sats_from_ratio_delta(local_ratio − band_high, capacity)`,
  drain-score sort, totals). Tests (tests/test_drain_demand.py, all pit):
  `::test_unpaired_over_local_channels_become_drain_demand` (paired source
  excluded, sort order, total == sum),
  `::test_fully_source_heavy_node_publishes_all_as_demand`,
  `::test_balanced_node_has_empty_drain_demand`,
  `::test_excess_sats_measured_against_band_high` (0.97 on 1M, band 0.65 →
  exactly 320,000); engine pruning of overlay-claimed sources pitted by
  `tests/test_rebalance_engine_v2.py::test_overlay_claimed_source_removed_from_drain_demand`.
  Run: pass.

## Purpose-section claims

- Pure function (no RPC/DB/clock): **verified** — imports are types + state
  only (:9-20).
- Score = sum of four additive role terms: **verified** —
  `TestScoring::test_additive_score_decomposition_exposes_role_terms`
  (sum equals `pair.score` to 1e-6, pre-hint),
  `::test_destination_drives_value_term_not_source` (dest-led value),
  `::test_more_drained_source_wins_when_value_and_return_tied`.
- Engine construction refs (rebalance_engine_v2.py:1247-1253 in the
  contract): drifted with 441b8e3's +122 engine lines; planner construction
  and `get_drain_demand`/pruning confirmed live on HEAD (`get_drain_demand`
  now rebalance_engine_v2.py:713, was :642; overlay-claimed-source pruning
  now :1424-1436, was :1318-1327 — pitted by tests/test_drain_demand.py
  engine tests).

## Gaps

1. RP2-4's same-peer self-pairing guard (:248-249) is unpitted — a genuinely
   dangerous regression (circular rebalance through one peer) would pass the
   suite.
2. RP2-6's clamps are unpitted at this layer (both per-side normalize on
   malformed/out-of-range input and the combined-multiplier clamp); the suite
   only exercises in-range biases through the planner.
3. RP2-2's exhaustiveness is never asserted as a set property; individual
   reason tests would not catch a channel silently dropped from both lists.
4. Contract uncertainties re-checked and still open on HEAD: `p_success`
   hard-coded 0.5 labeled `final_score`/`beats_do_nothing` (:62-77) —
   unchanged, still a potential audit-tooling trap (the engine's F6 tests
   show the real p_success is overlaid later); O(sources × dests) pair
   generation still uncapped (:246-247).

## Anomalies

1. **Skip-reason vocabulary drift** (cross-reference, recorded in
   docs/audit/verification/rebalance_audit_v2.md RA2-1, not re-derived): the
   planner's `source_ineligible`/`dest_ineligible` fallbacks and the
   state-layer passthrough reasons emitted at :133-148 are absent from
   `VALID_SKIP_REASONS` (modules/rebalance_audit_v2.py:29-49).
2. 441b8e3 widened `PairCandidate` (types_v2 +7 lines) without touching the
   contract's Inputs/Outputs section — the contract's `PairCandidate`
   description is now incomplete (missing the four historical fee-ppm
   fields); scoring semantics unaffected because the new fields feed the
   engine's downstream sats-EV gate, not the planner score.
3. `_sats_from_ratio_delta` uses `round()` (:34) while the coordination
   overlay's `_channel_excess_sats` uses `int()` truncation
   (rebalance_coordination_overlay.py:69) — the two modules can disagree by
   1 sat on the same channel's excess. Harmless at current amounts, but the
   asymmetry is undocumented.
4. Corpus: `max_pairs_reached` never appears in 1,227 debug snapshots — the
   fleet has never saturated `max_pairs` in the frozen window; that skip path
   is test-verified only.

## Refutation pass (2026-07-01)

Adversarial re-verification at HEAD dac9b48 (only planner/types drifted since
f905cfd, exactly as this doc's drift note records; drifted line cites
:390-396/:398-411 re-read exact). Method: mutation testing in a scratch copy
+ frozen-corpus re-sweep.

- Attacked: drift note, RP2-1..RP2-7, purpose claims, corpus statements.
- Survived — every decisive mutation was killed by the cited test:
  RP2-1 (letting source-eligibility bypass the band test kills a
  TestSkipReasons case); RP2-3 (sorting ascending kills the
  higher-scoring-source test — insertion order genuinely ruled out); RP2-5
  (substituting the source's budget for the destination's kills
  `test_destination_budget_authorizes_pair_spend` plus two siblings — the
  destination-authorizes-spend envelope is among the best-pitted claims in
  the routing stack); RP2-6 (removing the source-side inversion kills
  `test_v2_planner_uses_hive_rebalance_bias_for_pair_roles`, exact 1.10
  multiplier); RP2-7 (measuring excess against band_low instead of band_high
  kills `test_excess_sats_measured_against_band_high`). RP2-4's same-peer
  guard re-read at :248-249 (code-only gap stands); RP2-6 clamp bounds
  re-read (:390-396, :398-411; unpitted-at-this-layer gap stands).
- Refuted: no verdicts. One arithmetic nit corrected inline (drift note said
  "+15/−2"; actual diff is 13 insertions / 2 deletions).
- Corpus: this doc's numbers match the frozen sweep exactly (re-run
  2026-07-01: inside_band 3,316 / no_partner 2,984 / outcompeted 1,032 /
  below_hold_margin 177; max_pairs_reached absent; 178 candidate rows all
  ev_positive+market_only — note those rows are resamples of 14 distinct
  candidates, the sweep does not dedup debug candidates).

Counts: attacked 7 invariants + 3 purpose claims + drift note; survived all
verdicts; refuted 0 (1 numeric correction).
