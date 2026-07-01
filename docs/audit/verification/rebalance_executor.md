# Verification: modules/rebalance_executor.py

> **REMOVED (2026-07-01, commit 9bc0953)** — the dead-code finding below was acted
> on: `modules/rebalance_executor.py` was deleted together with
> `modules/rebalance_memory.py` (whose sole importer it was) and its dedicated test
> file `tests/test_rebalance_executor.py` (44 tests). The 13-line
> `modules/rebalance_executor_v2.py` shim was kept (it imports only the live
> `rebalance_execution`/`rebalance_native_executor_v2` modules). This also retires
> Anomaly 2 below: the legacy 6-reason `stable_failure_reason` vocabulary is gone
> and only the live 5-reason map in `modules/rebalance_execution.py` remains.
> Everything below describes the module as it existed before removal.

Phase 2 — Tier 2. Verified 2026-07-01 at HEAD 6740115 against
`docs/audit/contracts/rebalance_executor.md`. Full verification performed by a
prior Phase 2 agent; this doc consolidates that report after independent
spot-checks (dead-code call-site grep, tautological-test inspection,
stable_failure_reason divergence re-read — all three confirmed).

## Liveness (Phase 1 finding)

**Verdict: violated-as-suspected / dead code CONFIRMED.** Repo-wide grep at
HEAD: the only imports of `modules.rebalance_executor` are in
`tests/test_rebalance_executor.py:9` (and a local re-import at :575).
`modules/rebalance_executor_v2.py:11` aliases `NativeRouteExecutor` (from
`rebalance_native_executor_v2.py`) as `RebalanceExecutor` — a name collision,
not a use. `cl-revenue-ops.py:1810` contains only the log string
`"rebalance_executor=native, "`. No hits in scripts/ or tools/ beyond the audit
sweep's own docstring. Nothing in the live plugin constructs or calls this
module.

## Invariants

- **RX-1 (one active job per destination SCID)** — **verified (code-only).**
  Lock + normalized-SCID dedup at modules/rebalance_executor.py:1293-1297
  (`execute` returns `job_already_active`) and :1347-1351 (`execute_async`
  returns `""`). **No genuine test coverage**: no test calls
  execute/execute_async twice on the same channel;
  `TestStableFailureReasonMapping` only checks the error-string mapping.
- **RX-2 (no malformed route reaches sendpay)** — **verified.**
  `_validate_sendpay_route` :428-443, called at :1022 pre-sendpay. Genuinely
  pitted by `tests/test_rebalance_executor.py::TestExecuteFailure::`
  `test_rejects_malformed_route_before_sendpay` (increasing-amount route →
  error + `sendpay.assert_not_called()`). Empty-route / non-positive /
  insufficient-first-hop branches untested.
- **RX-3 (real budget enforced despite inflated discovery maxfee)** —
  **verified (code-only).** `fleet_maxfee = max(job.max_fee_msat,
  required_amount_msat // 100)` :841; hard gate `total_fee > job.max_fee_msat
  → route_over_budget` :1036-1041 before sendpay. **No genuine test**: the two
  budget-inflation tests in `TestWireFeeInsufficientHandling` are tautological
  (see Anomalies); no test drives `execute()` into an over-budget route.
  Nuance: the 1% floor is on `required_amount_msat` (delivery + return-hop
  fee), not the raw amount as the contract phrases it — behaviorally the same
  claim.
- **RX-4 (hive_equalization requires pure-hive path, never falls back)** —
  **verified.** `_validate_pure_hive_path` :306-314, invoked :888-889;
  no-fallback guard :1001-1002; ordinary-fleet fallback :1003-1009. Pitted by
  `TestHiveEqualizationRouteValidation::test_equalization_rejects_non_hive_intermediate`
  (`getroute.assert_not_called()` proves no network fallback) and
  `::test_equalization_accepts_all_hive_intermediates`.
- **RX-5 (fleet route starts on selected source + contains a fleet hop)** —
  **first half verified, second half verified (code-only).**
  `fleet_source_mismatch` :884-887 pitted by
  `::test_compute_fleet_route_rejects_unplanned_first_hop`; the
  `no_fleet_route` check :895-901 has **no covering test** (every
  hive-router-equipped test stubs `is_hive_member` to True for path nodes).
- **RX-6 (valid askrene inform semantics only)** — **verified** for the
  prefix/erring-hop/nothing-downstream sequence; **REFUTED as test-pitted,
  downgraded to verified (code-only), for the code-204-only gate**
  (refutation pass 2026-07-01: mutating `failure.get("code") != 204` to
  inform on every code survives all 44 tests — no test feeds a non-204
  failure and asserts no inform). `_inform_result` :473-481,
  `_inform_failure` :483-522 (code-204 only,
  prefix `unconstrained` / erring hop `constrained`, nothing downstream),
  whitelist :449-454. The sequence half is genuinely pitted by
  `TestInformChannel::test_informs_on_success`,
  `::test_informs_on_failure_with_valid_askrene_semantics` (exact sequence
  asserted; mutation informing downstream hops is killed), and two
  `TestHiveEqualizationRouteValidation` tests. The
  raise-on-invalid-value branch of `_inform_channel` itself is untested.
- **RX-7 (no auto.sourcefree; phantom first-hop fee stripped with cascade)** —
  **first half verified, second half verified (code-only).** `_get_layers`
  :175-200 plus defensive filter :833-834, pitted by
  `TestLayerSelection::test_fleet_layers_include_hive` (explicit
  `"auto.sourcefree" not in layers`). The strip-cascade :916-931 has **no
  covering test** (the one fleet-getroutes test returns a first hop exactly
  equal to `required_amount_msat`, so the branch never fires).
- **RX-8 (retry only on grown excludes or fleet final-hop temp failure; fee
  inflation 20%/2x)** — **verified** for the positive retry paths:
  `::test_retries_on_route_failure_with_exclude` (real 204 → grown exclude →
  retry + delpay), `::test_fleet_retry_after_send_failure_stays_on_fleet_path`,
  `::test_fleet_retries_on_fee_insufficient_without_network_fallback`. Code at
  :1172-1217 (`MAX_ATTEMPTS = 3` :80). The no-retry-when-excludes-static
  negative case and the MAX_ATTEMPTS=3 cap have no genuine end-to-end test;
  all four `TestWireFeeInsufficientHandling` tests are tautological.

Corpus: not observable — module is dead code; the sweep
(`tools/audit/sweep_routing_stack.py`, 2026-07-01 run) found zero
executor-attributable artifacts, consistent with dead-code status.

Test run: `.venv/bin/python -m pytest tests/test_rebalance_executor.py -q` →
44 passed (re-run 2026-07-01).

## Gaps

1. Invariants with **no genuine test coverage** despite a 44-green suite:
   RX-1 (job dedup), RX-3 (budget gate), RX-5 second half (`no_fleet_route`),
   RX-7 second half (phantom-fee strip cascade), RX-8 negatives
   (static-excludes no-retry, MAX_ATTEMPTS cap).
2. Tautological tests: `TestWireFeeInsufficientHandling::`
   `test_fee_budget_inflates_by_20_percent`, `::test_fee_budget_capped_at_2x_original`,
   `::test_destination_hop_fee_error_triggers_retry`,
   `::test_destination_hop_non_fee_error_stays_terminal` copy the module's
   logic into the test body ("# Simulate the inflation logic",
   tests/test_rebalance_executor.py:1787) and assert on local state — they
   pass regardless of module behavior. Confirmed by direct read 2026-07-01.
3. Corpus cannot exercise anything here (dead code) — all "verified" verdicts
   above are code+test only.

## Anomalies

1. **Dead module retained with a live-sounding name** while
   `rebalance_executor_v2.py` aliases the *different* live executor as
   `RebalanceExecutor` — high confusion risk for future maintenance.
2. **`stable_failure_reason` has diverged** between
   modules/rebalance_executor.py:108-145 and
   modules/rebalance_execution.py:28-49 (confirmed by side-by-side read
   2026-07-01): execution.py lacks the `no_viable_hive_path` family entirely
   (`no_route_back`/`no_fleet_route`/`fleet_self_route`/`non_pure_hive_route`
   → `local_execution_failed` there); `local_policy_block` triggers differ
   (`job_already_active` vs `native_route_invalid:`); executor maps
   `constrained_route` → `route_segment_exhausted`, execution does not;
   execution additionally maps `incorrect_cltv_expiry`,
   `native_route_over_budget:`, `retriable_failure:`. Since only
   execution.py's mapping is live, hive coordination reporting has silently
   lost the `no_viable_hive_path` taxonomy.
3. Non-atomic class-attribute counter `_exclude_layer_counter`
   (rebalance_executor.py:88, increment :239) — the live v3 router replaced
   this with `itertools.count` and its comment calls the old pattern a bug.
   Harmless only because the module is dead.
4. `self.database` assigned at :93 and never read (vestigial).
5. Contract line references verified accurate at HEAD except `delpay` at :549
   (contract says :548).

## Refutation pass (2026-07-01)

Adversarial re-verification at HEAD dac9b48 (module byte-identical to
f905cfd/6740115; all line cites re-checked). Method: mutation testing in a
scratch copy — break each claimed invariant, run the cited tests.

- Attacked: liveness (dead-code grep repeated on HEAD), RX-1..RX-8, the
  tautology findings, corpus statements, anomalies 1-5.
- Survived: liveness (grep clean: only tests + the v2 alias name collision +
  a log string); RX-2 (removing the `_validate_sendpay_route` call kills
  `test_rejects_malformed_route_before_sendpay`); RX-4 (disabling
  `_validate_pure_hive_path` kills the equalization-reject test); RX-5 first
  half (disabling the `fleet_source_mismatch` raise kills
  `test_compute_fleet_route_rejects_unplanned_first_hop`); RX-7 first half
  (adding `auto.sourcefree` in `_get_layers` kills two TestLayerSelection
  tests); RX-8 positive paths. Code-only cites RX-1 (:1293-1297/:1347-1351)
  and RX-3 (:841/:1036-1041) re-read and exact. Tautology finding confirmed:
  tests/test_rebalance_executor.py:1787 literally copies the inflation logic
  under "# Simulate the inflation logic". `stable_failure_reason` divergence
  re-confirmed (rebalance_execution.py has zero `no_viable_hive_path` /
  `no_fleet_route` hits). Doc-claimed negatives independently reproduced:
  `should_retry = True` (static-excludes no-retry removed) survives all 44
  tests, confirming the RX-8 negative-case gap as stated.
- Refuted: RX-6's code-204-only clause (see inline note) — one clause
  downgraded from test-pitted to code-only; the invariant's code is intact.
- New anomaly: the corpus was NOT final when this doc's sweep ran — a
  termination capture (20260701T203541Z, one per node) landed afterward;
  frozen-corpus history is 51 deduped entries (not 38). No new violations on
  re-sweep; dead-code status makes the numbers immaterial here.

Counts: attacked 8 invariants + 5 anomalies + liveness; survived 13;
refuted 1 clause (RX-6 code-204 gate → code-only).
