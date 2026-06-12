# Intent Contract: modules/rebalance_planner_v2.py

Tier 2 — medium treatment. Audited 2026-06-12 against commit 9f8f219.

## Purpose

`RebalancePlanner` (modules/rebalance_planner_v2.py:98) is the single pair-based planning
stage of the v2 rebalance engine. Given a `StateSnapshot` of per-channel state, it
classifies channels against a liquidity band (default 0.35–0.65 local ratio), pairs
over-local sources with over-remote destinations, scores pairs with an explicit additive
role-aware model (dest urgency, source drain, dest value class, cheap-return bonus,
hive-hint multiplier), selects up to `max_pairs` non-overlapping pairs, and emits a
`SkipRecord` for every channel it did not select. It also publishes `DrainDemand` — the
over-local residual the circular path could not place — which is the only inventory the
Boltz structural loop-out is allowed to act on. Pure function of the snapshot: no RPC,
no DB, no clock.

## Inputs / Outputs

- **Constructed by**: `RebalanceEngine` per cycle (modules/rebalance_engine_v2.py:1247–1253)
  with band/chunk/max-pairs/`pair_fee_cap_ppm` from config.
- **Input**: `StateSnapshot` / `ChannelState` (modules/rebalance_state_v2.py) — each
  channel carries `local_ratio`, eligibility flags + reasons, `value_class`
  (hive/profitable/active/funded/neutral), `remaining_budget_sats` (capex-derived,
  modules/rebalance_state_v2.py:267), `dest_urgency`, `source_drain_score`,
  `rebalance_bias` (hive hints), fee ppms.
- **Output**: `PlanResult(selected, skipped, drain_demand)` (types in
  modules/rebalance_types_v2.py). `PairCandidate` includes a full
  `score_decomposition` dict (`_bootstrap_score_decomposition`, :37–95) that the engine
  later overlays with route success/fee terms. `drain_demand` consumed by
  `RebalanceEngine.get_drain_demand` (modules/rebalance_engine_v2.py:642) and pruned of
  selected sources at modules/rebalance_engine_v2.py:1318–1327 before Boltz sees it.
- **No datastore keys, no DB tables, no RPC** of its own.

## Invariants

- **RP2-1** Band classification precedes eligibility: a channel is considered a source
  only if `local_ratio > target_band_high` AND `source_eligible`, a dest only if
  `local_ratio < target_band_low` AND `dest_eligible`; everything else gets a skip record
  (`inside_band` or the role-specific reason) (:128–155).
- **RP2-2** Every channel is explained: each eligible-but-unpaired channel receives a
  skip record with reason `no_partner`, `max_pairs_reached`, or `outcompeted`
  (:176–210). Number of skip records + selected source/dest channels covers all
  classified channels.
- **RP2-3** One role per channel per plan: selection iterates score-descending and
  refuses a pair whose source or dest already appears in a selected pair
  (`paired_sources`/`paired_dests`, :158–174); at most `max_pairs` pairs.
- **RP2-4** Self-pairing is impossible and amounts are bounded:
  `src.peer_id == dest.peer_id` pairs are skipped; amount =
  min(source excess, dest need, `max_chunk_sats`) and must be > 0 (:248–264).
- **RP2-5** The destination authorizes spend: `pair_budget` starts from
  `dest.remaining_budget_sats` only (source capex budget never enters), optionally raised
  to `ceil(amount * pair_fee_cap_ppm / 1e6)` when `pair_fee_cap_ppm > 0` (:269–278).
- **RP2-6** Hive hints can shift a pair score by at most ±15%: per-peer biases are
  clamped to [0.85, 1.15] and the combined pair multiplier
  `1 + (dest_bias-1) - (source_bias-1)` is clamped to the same range
  (`_normalize_rebalance_bias` :379–385, `_pair_hint_multiplier` :387–400). Note the
  intentional source-side inversion documented at :389–396.
- **RP2-7** Drain demand is exactly the unpaired over-local residual: entries only for
  over-local channels not selected as sources, excess =
  `(local_ratio - band_high) * capacity`, sorted by drain score (:215–234).

## Revenue role

Indirect. It spends nothing itself, but its scoring decides where rebalance budget is
risked; RP2-5 is the claim that ties every planned fee to a destination that earned (or
was granted) the budget, and RP2-7 keeps Boltz loop-outs subordinate to free internal
placement.

## Observable surface

Not directly observable as a standalone artifact. Its decisions surface through the
engine's rebalance audit/coordination overlay output, `rebalance_history` rows (selected
pairs that execute), and indirectly through `revenue-spend-ledger.json` (fees from
executed plans). Score decompositions appear in engine logs/audit records
(`stage: "planner_pre_route"`).

## Uncertainties

- `p_success` in the bootstrap decomposition is a hard-coded 0.5 (:62) — diagnostic only,
  but it is labeled `final_score`/`beats_do_nothing`, which could mislead audit tooling
  into reading it as a calibrated probability.
- `imbalance_score` is computed and exported but explicitly not a scoring input
  (:304–308); legacy `value_score` in the decomposition is dest-only (:331) — consumers
  assuming src+dest semantics would misread it.
- Pair generation is O(sources × dests) with no cap before sorting; fine at current fleet
  size, untested at hundreds of imbalanced channels.
- Whether `funded` value class (capex bootstrap inventory, :28) should score equal to
  `active` (both 1) is a policy question, not verified against outcomes.
