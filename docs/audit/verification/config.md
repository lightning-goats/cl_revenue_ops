# Verification: modules/config.py (Tier 3)

Contract: docs/audit/contracts/config.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. Mutable `Config` dataclass (line 318), frozen `ConfigSnapshot` (line 853),
transactional `update_runtime` (line 739: validate → DB write → read-back → in-memory update
under `_lock`), `ChainCostDefaults` (line 1068), `LiquidityBuckets` (line 1117).

## Invariant verdicts
- **CFG-1 — verified.** `IMMUTABLE_CONFIG_KEYS = {db_path, dry_run}` (lines 22-25); rejected at
  line 751 before any write. Exercised empirically: `update_runtime(db, 'dry_run', 'true')` →
  `{"error": "Key 'dry_run' cannot be changed at runtime"}`.
- **CFG-2 — verified.** Read-back at line 831; mismatch triggers `delete_config_override`
  rollback and error (lines 832-838). Exercised with a lying fake DB: returned
  `"Database write verification failed (Ghost Config prevention)"`, DB override rolled back,
  in-memory value unchanged.
- **CFG-3 — verified.** All five contract-listed cross-field pairs enforced inside `with
  self._lock` (lines 792-823), plus `hive_equalization_low/high_pct` (not in the contract list).
  Exercised: min_fee_ppm > max_fee_ppm, sink >= source, receivable floor > target all rejected.
- **CFG-4 — verified.** `from_config` (lines 1046-1064) falls back to snapshot-field
  default/default_factory when `Config` lacks the field (lines 1060-1063); sanity one-liner
  prints version 0 without raising.
- **CFG-5 — verified.** `math.isfinite` rejection in `update_runtime` (line 767, error return)
  and `_apply_override` (line 716, warning + skip). Exercised: `'nan'` and `'inf'` both rejected
  at runtime.

## Tests
No dedicated `tests/test_config.py` (as the contract states). Coverage is incidental:
`tests/test_operator_surface.py` (52 passed, 0.23s — but it *mocks* `update_runtime` at line
748, so it exercises the RPC plumbing, not the transaction), `tests/test_capex_budget.py` and
`tests/test_fee_controller.py` consume `ConfigSnapshot`. The transactional invariants above
were verified in this pass by direct empirical exercise against a fake database (all passed).

## Liveness
LIVE. Imported by `cl-revenue-ops.py`, `modules/fee_controller.py`, `modules/rebalancer.py`,
`modules/capacity_planner.py`, and re-exported by `modules/__init__.py`.

## Gaps
- No test exercises the real `update_runtime` transaction (immutable-key rejection, read-back
  rollback, cross-field TOCTOU guards). It is currently verified only by this audit's ad-hoc
  exercise; a regression could land silently.
- `Config`/`ConfigSnapshot` field alignment is by hand; CFG-4's silent default fallback means
  drift is invisible by design (contract acknowledges this).

## Anomalies
- None beyond the contract's own notes. Cross-field enforcement is a superset of the contract
  list (hive_equalization pair also guarded) — benign.
