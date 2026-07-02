# Operator Decisions — Module Verification Campaign

Rulings from Sat on questions raised by the Phase 1 intent contracts. Per campaign
rules, these are **follow-up work items**, not mid-campaign code changes: no
cl_revenue_ops runtime code is modified while the campaign runs. Phase 2+ verifies
current behavior as-implemented and reports these as confirmed intentional-behavior
gaps to fix afterward.

## D1 — Hive-member defibrillation (capacity_planner, CP-I5 caveat)

**Question:** The dead-capital pipeline runs before the hive-member skip, so a hive
member can be emitted as a DEAD_CAPITAL loser and receive a real fee-spending
defibrillation (capacity_planner.py: pipeline at :875 vs member skip at :897,
FEE_REDUCE/DEFIBRILLATE allowed for protected channels :1248-1251). Intended?

**Ruling (Sat, 2026-06-12):** Not intended as designed — "it probably can be
removed." Hive-member protection should short-circuit dead-capital staging entirely.

**Follow-up:** remove the hive-member DEFIBRILLATE/FEE_REDUCE path from the
dead-capital pipeline (or gate the whole pipeline behind the member skip), and fix
the fail-open `is_hive_member` exception swallowing noted in CP-I5 at the same time.
Until then, Phase 2 verifies CP-I5 against current (permissive) behavior and flags
any corpus episode where a member was actually defibrillated.

## D3 — Hermes study terminated (2026-07-01)

**Ruling (Sat):** "We're no longer doing the hermes study." The hermes gateway /
collection stack is not being restored. The observational corpus is frozen as-is:
2026-05-19 → 2026-06-20 continuous, plus a one-shot 2026-07-01 capture that closed
the forwards backlog losslessly.

**Consequences for the campaign:** Phase 0's pending acceptance criterion (a ≥95%
post-fix coverage week) is moot — the corpus quality findings stand as final. Phases
2–4 proceed against the frozen corpus. Phase 5 (ongoing scorecard *in the hermes
report pipeline*) must be re-scoped: the sweep scripts in tools/audit/ remain
runnable standalone against any future data source, but there is no hermes cadence
to wire them into.

## D2 — Fleet-loss masking in profitability (profitability_analyzer)

**Question:** Structural protection upgrades UNDERWATER → BREAK_EVEN for hive
members / corridor owners / centrality > 0.03 (profitability_analyzer.py:2693-2701),
silently hiding losses on fleet channels from close recommendations and loss
reporting. Intended interaction with the sovereignty revenue target?

**Ruling (Sat, 2026-06-12):** Should be removed.

**Follow-up:** remove the UNDERWATER → BREAK_EVEN reclassification. If fleet
channels need close protection, express it as an explicit close-protection reason
(as capacity_planner already does with `_close_protection_reason` → HIVE_MEMBER)
rather than by falsifying the profitability class. Until then, Phase 2/4 must treat
BREAK_EVEN on hive-member/corridor/central channels as potentially masked
UNDERWATER, and the contribution analysis (Phase 4) should quantify how many sats of
loss the mask currently hides.

## D4 — Defibrillation fee cap (rebalancer, RB-I10)

**Question:** The diagnostic ("defibrillator") shock's fee envelope was hardcoded at
max_budget_sats=100 AND max_fee_ppm=2000 — both bounds bind at exactly 100 sats on
the 50k shock. The corpus shows every priced route into a defib target cost 118–363
sats, so all priced shocks were rejected `route_over_budget` (0/22 recorded shocks
succeeded): the diagnostic could never fire. Raise the cap?

**Ruling (Sat, 2026-07-01):** Raise the defib cap so the diagnostic functions.
Default 400 sats. Rationale: observed market route prices were 118–363 sats; the
ppm ceiling is now derived from the sat cap (ceil(cap/amount×1e6), 8000 ppm at
defaults) so the sat cap is the single binding knob.

**Implementation:** new config option `diagnostic_rebalance_max_fee_sats`
(plugin option `revenue-ops-diagnostic-rebalance-max-fee-sats`, default 400),
range-validated to [1, 10,000] and clamped at use to
[1, min(daily_budget_sats, 10,000)] so a typo cannot authorize huge diagnostic
spend. Shock amount (50k sats) and capital-controls gating unchanged. Pinned by
TestDiagnosticFeeCap in tests/test_rebalancer_module.py; RB-I10 amended in
docs/audit/contracts/rebalancer.md.
