# Phase 3 de-hive map (revenue-critical core)

Status: generated 2026-07-09 by a read-only mapping pass. Execution spec for
removing all remaining hive/coordination/zero-fee-corridor code from the
rebalance / capacity / fee core. Every hive branch is gated on
`self.hive_hints`/`self._hive_hints`/`self.hive_router`/`self._hive_router`/
`is_hive_member`, all permanently None/False — so each branch below is provably
dead and its removal preserves the non-hive path.

## Guiding decision: keep inert vestigial plumbing
Where a `hive_*`/`coordination_*` **dataclass field** is always its default now
and removing it has wide blast radius into scoring math (esp. `rebalance_bias`
and the `PairCandidate` hive fields), KEEP it inert to bound risk; a later
cosmetic pass can rename/drop it. Prioritize revenue-engine correctness over
grep-purity. Remove all hive **methods, branches, and hive-only enum members**.

## Execution order (each step = one green commit unless noted)
1. **fee_controller.py** — self-contained (uses `getattr` for DB methods). RISK AREA. Do with care + deploy-and-watch. (Plan says fee_controller LAST; we honor that by ordering it after the mechanical modules but before/at the final deploy.)
2. **capacity_planner.py** — self-contained; only DB-hive call is inside the deleted `_is_protected_hive_member`.
3. **rebalancer.py** — self-contained; coordination/report/estimate cluster.
4. **database.py** (+ cl-revenue-ops corridor cleanup) — after 1-2 so nothing calls the hive methods.
5. **ATOMIC single commit** (mutual import coupling): `rebalance_route_policy` (prune enum+helpers+kwarg) + delete `rebalance_coordination_overlay.py` + delete `rebalance_hive_router.py` + `rebalance_engine_v2` + `rebalance_state_v2` + `rebalance_planner_v2` (+ optional `rebalance_types_v2`). Must land together or collection hits ImportError / `AttributeError: HIVE_ONLY`.
6. Tests deleted/pruned in the same commit as the code that breaks them. Extend `test_architecture_guard.py` to assert deleted modules/symbols are gone. Keep `test_standalone_independence.py`.
7. Afterward: dead config.py keys, cl-revenue-ops hive globals/RPC (`revenue_hive_hints_status`, `_corridor_classify_forward`), then deploy + watch fee/rebalance cycles.

## Per-module symbol removal (see git history / agent map for exact line #s)

### rebalance_hive_router.py — DELETE WHOLE
Importer: `rebalance_engine_v2.py:27` + dead construction 225-232. Tests: delete `test_rebalance_hive_router.py`; prune `test_rebalance_perf_fixes.py`, `test_router_v3_engine.py`.

### rebalance_coordination_overlay.py — DELETE WHOLE
`coordination_reserved_slots` reserve is behavior-neutral (planner enforces max_pairs). `_drain_score`/`_refill_urgency` live in state_v2 and stay. Importer: engine 18-23 + call sites. Tests: delete `test_rebalance_coordination_overlay.py`, `test_hive_hints_finite_hardening.py`; prune `test_rebalance_economics_fixes.py`.

### rebalance_route_policy.py — prune to a single MARKET_ONLY decision
Remove enum members `RoutePolicy.HIVE_ONLY`, `RoutePriority.COORDINATED`, `RoutePriority.HIVE_EQUALIZATION`, const `MAX_HINT_PRIORITY_SCORE`; helpers `_is_hive_member`, `_hints_fresh`, `_entries`, `_entry_view`, `_normalize_value`, `_normalize_route_segment`, `_pair_segments`, `_entry_segments`, `_entry_amount_sats`, `_priority_score`, `_match_entry`; in `decide_route_policy` drop `hive_hints` kwarg + all hive branches → body becomes `return RouteDecision(policy=MARKET_ONLY, priority=EV_POSITIVE, reason=reason_code or "ev_positive")`. KEEP `RouteDecision`, `RoutePolicy.HYBRID/MARKET_ONLY`, `RoutePriority.EV_POSITIVE/BACKGROUND`. Consumers: engine import (drop `_entries`), engine `decide_route_policy(...)` calls drop `hive_hints=`. Tests: prune `test_rebalance_route_policy.py`.

### rebalance_types_v2.py — optional field drop
`PairCandidate.hive_source_rebalance_bias/hive_dest_rebalance_bias/hive_hint_score_multiplier/coordination_hint_type/coordination_hint_id/coordination_rank_bonus`. RECOMMEND keep inert unless doing a same-commit sweep of planner_v2/engine/rebalancer consumers.

### rebalance_state_v2.py
`_value_class`: drop `if is_hive_member: return "hive"`. Drop `ChannelInput.is_hive_member` field + normalizer read. `build_state_snapshot`: drop `hive_bootstrap_budget_sats` kwarg + bootstrap block. KEEP `_drain_score`/`_refill_urgency`/`rebalance_bias`.

### rebalance_planner_v2.py
Drop `_VALUE_SCORES["hive"]=3`. KEEP bias plumbing inert.

### rebalance_engine_v2.py — the crux
Remove overlay/hive_router/`_entries` imports; `__init__` hive_router/`_membership_router`/`_hive_router` + RebalanceHiveRouter build; `_is_hive_member`, `_get_hive_rebalance_bias`, `_hive_equalization_overlay`, `_apply_segment_score_bias`, `_hybrid_choice`, `_fail_closed_on_route_failure`; `is_hive_member`/`hive_bootstrap_budget_sats` in normalized dict; overlay+merge+lease-suppress call sites; priority dict drop COORDINATED/HIVE_EQUALIZATION; hive_router begin/end blocks; retry guards; `_route_pair` collapses to MARKET_ONLY branch. Drop `hive_hints=`/`hive_router=` in the `cl-revenue-ops.py` RebalanceEngine construction. KEEP `_market_price_pair`, RouteResult, router_v3, metabolic/immune methods (inert, out of scope).

### rebalancer.py
Remove reason codes `HIVE_EQUALIZATION/HIVE_PUSH/COORDINATED_REBALANCE`; `hive_router`/`hive_hints` attrs + property/setter; `_is_hive_member`, `_fresh_hive_entries`, coordination entry/segment/priority helpers, `_is_coordinated_candidate`, `_get_coordination_execution_context`, `_report_coordination_intent/outcome`, `_build_hive_liquidity_state_payload`, `_report_hive_liquidity_state`; hive route-estimate branch. Simplify every `if self._is_coordinated_candidate(candidate):` to the non-coordinated branch. KEEP `get_boltz_coordination` (Boltz, not hive). Optional: `dest_is_hive_member`/coordination fields. Tests: delete `test_hive_liquidity_state_report.py`; prune `test_rebalancer_module.py`.

### capacity_planner.py
Drop hive value weights (`"hive":0.9/0.3/0.09`); `self.hive_hints` + all `if self.hive_hints is not None:` blocks; `_is_protected_hive_member` (→callers 945/1182/2258/3722 drop the gate); `_discover_from_hive*`, `_dedupe_hive_candidates`, `_is_hive_topology_witness`, `_get_hive_open_score_multiplier` + their call sites; `RESERVED["hive"]`. Frees `db.hive_member_last_confirmed`. Tests: delete `test_hive_discovery.py`; prune `test_capacity_planner.py` close-protection. RISK: fleet peers no longer close-protected (intended standalone behavior).

### fee_controller.py — RISK AREA
Remove `FeeReasonCode.HIVE_MEMBER_ZERO_FEE`; all `__init__` hive state; methods `_get_hive_fee_bias`, `_hive_hint_effective_ttl`, `_cached_hive_membership_active`, `_remember_hive_member`, `_clear_hive_member_cache`, `_get_hive_membership_status`, `_hive_member_zero_fee_active`, `_hive_zero_fee_stale_grace_seconds`, `_hive_zero_fee_grace_active`, `_confirm_hive_membership_db`, `_log_hive_grace_hold`, `_classify_channel_role`, `_get_hive_exploration_multiplier`, `_check_hive_member_fee`, `_consume_hive_member_release`, `_consume_hive_member_advisory`. Pricing simplifications (PRESERVE the clamp, only drop the ≡1.0 factor):
- `composite_hint_bias = clamp(hive_fee_bias * temporal_adj)` → `clamp(temporal_adj)` (keep HIVE_HINT_TOTAL_BIAS_MIN/MAX bounds).
- base-fee: keep the adaptive `return _cfg_int('base_fee_msat',0)` early-return; drop the `_classify_channel_role` intra_fleet/non_hive branch.
- `force_reprice_reason` from release/advisory → always None; inline None.
- remove `apply_hive_zero_fee` blocks + zero-fee initial-fee block + `_check_hive_member_fee` block.
KEEP `_drain_fee_multiplier` (general drain pricing) and `CycleState.last_corridor_role`/`_corridor_role` (DTS "P/S" context bucket, NOT hive). Tests: delete `test_hive_zero_fee_grace.py`, `test_p4_001_hive_zero_fee_locked_read.py`; prune fee_controller tests asserting HIVE_MEMBER_ZERO_FEE/apply_hive_zero_fee/intra_fleet.

### database.py
Drop `hive_member_confirmations` table + `hive_member_confirm`/`hive_member_last_confirmed`; `corridor_flow_daily` table + `_CORRIDOR_FLOW_KLASSES`/`corridor_flow_record`/`corridor_flow_summary` (settled-forward hive classification). KEEP `planner_recycle_ops`. Consumers use `getattr`/`callable` guards (graceful). cl-revenue-ops: delete `_corridor_classify_forward` + call sites (6628/6742-6746/5886). Tests: delete `test_corridor_flow.py`.

### Out-of-scope consumers to flag
config.py dead keys (`hive_equalization_*`, `hive_push_*`, `hive_zero_fee_stale_grace_seconds`, `hive_rebalance_bootstrap_budget_sats`, `base_fee_msat_intra_fleet`, `base_fee_msat_non_hive`, `rebalance_coordination_reserved_slots`); `rebalance_audit_v2.py:53` `"hive_equalization_cooldown"` string; cl-revenue-ops `revenue_hive_hints_status` RPC + hive globals. `pair_fee_cap_ppm` is NOT hive — keep.

## Risk callouts (non-hive behavior that removal must NOT change)
1. fee_controller composite bias: keep the clamp, only drop the ≡1.0 factor.
2. fee_controller base fee: keep the adaptive early-return.
3. fee_controller force_reprice_reason: confirm only hive set it (it does).
4. engine `_route_pair` collapse: MARKET_ONLY is the live path today; verify no non-hive producer of HYBRID/HIVE_ONLY.
5. engine merge_coordination_pairs removal: neutral only because planner enforces max_pairs.
6. engine priority sort: keep EV_POSITIVE/BACKGROUND so lambda never KeyErrors.
7. rebalance_bias plumbing: keep inert (always 1.0) to avoid touching scoring.
8. capacity_planner close-protection: fleet peers no longer protected (intended standalone change).
9. metabolic/immune bias methods: inert; safe to leave as no-ops.
