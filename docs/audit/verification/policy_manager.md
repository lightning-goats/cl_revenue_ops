# Phase 2 Verification — policy_manager.py

Contract: docs/audit/contracts/policy_manager.md (PM-I1..PM-I13).
Evidence: test mapping + code confirmation on HEAD cdb536a. Drift since contract commit
f905cfd is confined to `get_policy_suggestions` (:915-1057, channel-scope metadata,
cdb536a) — additive only, no invariant semantics changed; citations at/after :915
re-located on HEAD (see Drift notes). Not corpus-observable: the hermes corpus has no
policies-table artifact, so all verdicts are code+test (contract section 6 anticipated
this). All 189 tests in the 8 cited files pass on HEAD (2026-07-01):
test_policy_manager.py, test_policy_change_wake.py, test_database_policies.py,
test_low_severity_fixes.py, test_thompson_rebalancer_policy_bugs.py,
test_operator_surface.py, test_fee_controller.py, test_cross_plugin_contracts.py.

| Invariant | Verdict | Evidence |
|---|---|---|
| PM-I1 peer-id validation on every write path | **open (refuted — validator edge hole)** | Write-path coverage itself holds: policy_manager.py:300-313 (`_validate_peer_id`), used at :572 (set), :712 (delete), :1195 (batch); test_policy_manager.py::TestValidation::test_invalid_peer_id_rejected, ::TestPolicyBatchOperations::test_batch_update_invalid_peer_id_raises. Note: contract cites test_database_policies.py, but the DB layer does NOT validate (accepts peer_id "expired") — validation lives only in PolicyManager. REFUTED: the validator itself has a hole — `PEER_ID_PATTERN = re.compile(r'^[0-9a-fA-F]{66}$')` (:149) uses `$`, which in Python matches before a trailing newline, so the 67-char input `'a'*66 + '\n'` passes `_validate_peer_id`; proven end-to-end: the real `set_policy` persists it (`database.upsert_policy` called with the newline peer_id) and notifies subscribers. Reachable via JSON RPC (JSON strings carry `\n`). Practical damage is low (junk row under a key no lookup matches), but "validates a 66-char hex peer_id and raises ValueError otherwise" is false as stated. Fix: `\Z` or `fullmatch`. Also an evidence mis-citation: test_revenue_unignore_rejects_malformed_peer_id lives in test_low_severity_fixes.py:57 (not test_operator_surface.py) and mocks PolicyManager entirely — it pits the RPC shim's own check, not `_validate_peer_id`. |
| PM-I2 STATIC requires a target | **violated** (Phase 1 gap re-confirmed on HEAD) | Single path enforces (:606-609 raises); batch path :1193-1267 has NO static-target check. Live repro: `set_policies_batch([{peer_id, strategy: "static"}])` persists STATIC with fee_ppm_target=None; fee controller null-checks (fee_controller.py:4638, :7510 on HEAD) silently fall through to dynamic management — the exact failure the invariant exists to prevent. Single path pitted by test_low_severity_fixes.py::test_set_policy_rejects_static_without_fee_target. **The gap is entrenched by a test**: test_policy_manager.py::TestPolicyBatchOperations::test_batch_update_applies_rebalance_modes (line 110) sets batch `static` with no target and expects success — it will fail when the gap is fixed. test_fee_controller.py::TestStaticStrategy::test_static_strategy_requires_fee_ppm is misnamed (happy path only, never asserts the raise). |
| PM-I3 fee target bounds (0..100000 int) | **verified** | Single :601-605, batch :1218-1223; applied-static clamp fee_controller.py:4642-4645 (HEAD); test_thompson_rebalancer_policy_bugs.py::TestPolicyBatchValidation::test_batch_rejects_negative_fee_ppm + ::test_batch_rejects_excessive_fee_ppm (batch), test_low_severity_fixes.py::test_set_policy_accepts_static_with_zero_target (zero accepted). Single-path bounds rejection untested (code-only half). |
| PM-I4 multiplier bounds sane | **verified** | Write validation single :621-640, batch :1230-1250 (incl. min>max rejection both paths); read-side clamp+swap :112-129; TestPolicyBatchValidation::test_batch_rejects_out_of_range_multiplier_min/_max, test_policy_manager.py::test_peer_policy_fee_multiplier_bounds_clamped, ::test_set_policy_persists_fee_autoband_multipliers. Gap: no test pits the min>max ValueError on either path. |
| PM-I5 expiry bounded and honored | **verified (code-only)** at manager level | 30-day cap :648-651 (single), :1259-1262 (batch); is_expired :106-110; get_policy lazy expiry → default + delete :441-468, :470-476; cache-load skip :350-353. DB-level deletion pitted by test_database_policies.py::TestDeleteExpiredPolicies (3 tests); **no test exercises manager-level lazy expiry, cache-load skip, or the 30-day cap**. |
| PM-I6 rate limit counts only committed changes (single path) | **verified (code-only)** | _check_rate_limit :259-288 (no recording), _record_rate_limit_change :290-298; set_policy checks :655-660, records only after DB write :669-670. Batch asymmetry re-confirmed on HEAD: read-only check :1275-1282, timestamps recorded :1283-1288 BEFORE upsert :1291 (contract caveat holds). **TestRateLimiting is tautological** (test_batch_bypasses_rate_limit wraps in try/except-pass, asserts nothing); nothing pits the 11th-change RuntimeError or count-after-commit ordering. |
| PM-I7 default is permissive | **verified** | :461-468, DEFAULT_POLICY :182-188; test_policy_manager.py::test_hive_member_does_not_create_static_zero_fee_policy (get_policy on unstored peer → DYNAMIC/ENABLED/no tags), ::test_peer_policy_default_values |
| PM-I8 automation never overwrites operator intent | **verified** | HEAD lines: manual-tag skip :1093-1095; member auto_fleet-only deletion :1100-1106; stored-policy/auto_corridor guard :1107-1129; role-loss deletes only auto_corridor :1130-1145. test_policy_manager.py::TestCorridorAutoPolicies (10 tests: never_clobbers_operator_policy, role_loss_spares_operator_policy, role_loss_spares_manual_auto_corridor_policy, deletes_legacy_auto_fleet, without_stored_policy_is_noop, ...). Contract caveat (auto_fleet deletion keyed on tag alone; only 'manual' protects) still true on HEAD :1104-1106. |
| PM-I9 committed changes notify — except lazy expiry | **verified (code-only)** for dispatch | Notify sites on HEAD: set :696, delete :723-729, batch :1314, cleanup_expired :1343-1348; lazy-expiry path :446-458/:470-476 evicts + deletes with NO notify (exception confirmed). test_policy_change_wake.py (5 tests) pits the fee controller's `_handle_policy_change` handler, not PolicyManager's `_notify_change` dispatch; registration at fee_controller.py:2573-2575 (HEAD). **No test pits the dispatch itself.** |
| PM-I10 batch is validate-first | **verified** | Validation loop :1193-1267 completes before rate-limit block :1269-1288 and DB write :1291; fail-whole-batch pitted by ::test_batch_update_invalid_strategy_raises, ::test_batch_update_invalid_peer_id_raises. Partial gap: no test asserts no-partial-persistence (upsert never called) after mid-batch validation failure — ordering code-confirmed only. Known deviations from set_policy parity: PM-I2 missing static check, PM-I6 record-before-write. |
| PM-I11 is_peer_ignored narrower than PASSIVE | **verified (code-only)** | :1362-1374 (PASSIVE AND DISABLED) vs should_manage_fees :825-838 (PASSIVE alone); **no covering test of the conjunction**. |
| PM-I12 batch size bounded (100) | **verified** | :1178-1181 (check precedes validation loop); ::test_batch_update_exceeds_max_size_raises (101 entries → ValueError) |
| PM-I13 corrupt rows degrade to safe defaults | **open (refuted)** | The three named guards are real: :362-373 (tags JSON → []), :388-396 (strategy → DYNAMIC), :398-406 (mode → ENABLED), all log-and-continue; no test feeds a corrupt row through _row_to_policy. REFUTED: only those 3 of the row's fields are guarded, and the invariant's conclusion "a corrupted row can neither brick policy reads nor silently freeze a peer" is false. `expires_at` passes through `_row_to_policy` raw (:383, :417); a corrupt TEXT value (SQLite columns are dynamically typed) reaches `is_expired()` → `int(time.time()) > 'garbage'` → TypeError (proven executable; ENABLE_AUTO_EXPIRY=True at :43 so the comparison is live). That call sits inside `_load_cache` at :352 with no try/except, and `get_policy` calls `_load_cache` unguarded — ONE corrupt `expires_at` row makes EVERY `get_policy`/`get_all_policies` call raise, for all peers: exactly the brick-the-reads failure mode the invariant exists to prevent. (`fee_ppm_target`/`updated_at` are also unguarded, with per-consumer blast radius.) The verified verdict generalized from the 3 guarded fields to the whole row. |

## Gaps

- **No covering tests for PM-I5 (manager-level expiry), PM-I6 (rate-limit semantics),
  PM-I9 (notify dispatch), PM-I11, PM-I13** — five code-only verdicts. PM-I6 and PM-I9
  are the consequential ones: the count-after-commit ordering and the callback wiring
  are exactly the kind of behavior that silently inverts under refactoring, and the
  existing TestRateLimiting suite is tautological.
- Untested halves: PM-I3 single-path bounds rejection, PM-I4 min>max ValueError,
  PM-I10 no-partial-persistence.
- Nothing here is corpus-observable (no policies-table artifact; no STATIC/PASSIVE
  policies evidently active in the window — the 5 zero-fee channels trace to the hive
  fleet-member path in the fee controller, not to STATIC policies). PM-H1/H2/H3 remain
  untestable on this corpus, as the contract anticipated.

## Anomalies

1. **PM-I2 violation is entrenched by the test suite**:
   test_policy_manager.py::TestPolicyBatchOperations::test_batch_update_applies_rebalance_modes
   asserts success for a batch STATIC entry with no target. Any fix restoring
   batch/single parity must update this test or it will fail as a false regression.
2. test_fee_controller.py::TestStaticStrategy::test_static_strategy_requires_fee_ppm
   asserts only the happy path despite its name — a misleading coverage signal for the
   one violated invariant in this module.
3. Contract's PM-I1 test citation (test_database_policies.py) is wrong in direction:
   the Database layer performs no peer-id validation; only PolicyManager does.

## Drift notes

- All drift f905cfd → cdb536a is inside `get_policy_suggestions` (channel-scoped
  suggestion metadata, commit cdb536a) — additive fields only (scope, channel_id,
  policy_evidence_scope, suggested_channel_rebalance_mode); no invariant semantics
  changed. New behavior covered by
  test_policy_manager.py::TestPolicySuggestions::test_bleeder_suggestion_carries_channel_scope.
- Re-located citations (contract → HEAD): batch validation :1184-1207 → :1193-1267;
  batch upsert :1275-1279 → :1291; rate-limit record :1253-1272 → :1269-1288; batch
  notify :1298 → :1314; cleanup notify :1327-1332 → :1343-1348; is_peer_ignored
  :1346-1358 → :1362-1374; batch-size check :1162-1165 → :1178-1181; corridor block
  :1047-1139 → :1063-1155.
- Pre-:915 citations verified stable (PM-I1..PM-I7 write-path lines unchanged).
- fee_controller consumer null-checks moved: :4644 → :4638, :7391 → :7510 (HEAD).

## Refutation pass (2026-07-01)

Adversarial re-verification of the 12 non-violated verdicts (PM-I2's "violated"
adjudication was re-checked for accuracy, not attacked), with all five code-only
verdicts given mandatory deep treatment and executable probes where reading was
inconclusive. All 189 cited tests re-run and pass on HEAD. Attacked: 12. Survived: 10
(PM-I3, I4, I5, I6, I7, I8, I9, I10, I11, I12). Refuted: PM-I1 (validator accepts a
trailing-newline 67-char peer_id, persisted end-to-end through the real set_policy)
and PM-I13 (unguarded `expires_at` bricks every policy read on one corrupt row) —
see table rows.

Code-only verdicts that survived deep checks: PM-I5 (batch expiry cap real at
:1259-1262; lazy expiry really deletes DB rows via `delete_expired_policies`,
transactional), PM-I6 (`upsert_policy` is a bare sqlite execute that raises on
failure — no return-code path can record budget for an uncommitted write; window
pruning correct; the doc's TestRateLimiting-is-tautological claim re-confirmed
verbatim, try/except-pass asserting nothing), PM-I9 (delete notifies only when a row
was actually deleted; callbacks individually try/except-wrapped in `_notify_change`
:246-253 so one raising subscriber cannot starve the rest; lazy-expiry no-notify
exception confirmed), PM-I11 (trivially correct), PM-I7 (get_policy constructs a
fresh default PeerPolicy per call — no shared-object aliasing; `DEFAULT_POLICY` :182
is dead code, defined and never used). PM-I2's violation description re-verified
accurate, including the entrenching test at test_policy_manager.py:104-117.

New anomalies (no verdict impact):
1. **Vacuous mock assertions in TestCorridorAutoPolicies**:
   `mock_database.set_policy.assert_not_called()` (test_policy_manager.py:282, 319,
   374, 406, 428) asserts on a method PolicyManager never calls — production writes
   go through `upsert_policy`, so these asserts always pass. The tests still pit
   PM-I8 via `applied` counts and cache-visible policy state, but the asserts are
   dead weight giving false comfort.
2. NaN multiplier bypass: `fee_multiplier_min/max=float('nan')` passes write
   validation on both paths (every range comparison is False for NaN); read-side
   `get_fee_multiplier_bounds` degrades it to (0.1, 0.1) so the effect stays inside
   global bounds — PM-I4's effect survives, but NaN storage is unintended.
3. `expires_in_hours` has no type validation: a string raises TypeError at the
   `<= 0` comparison (:645/:1256), not the documented ValueError.
4. `get_policy` returns cached PeerPolicy objects by reference despite the
   "immutable snapshot" framing (dataclass not frozen); a mutating consumer would
   corrupt the shared cache.
