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
