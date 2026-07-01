# Verification: modules/rebalance_native_executor_v2.py

> **Remediation (2026-07-01, commit d14256e):** the NX-4 malformed-invoice cleanup
> gap (Gap 1 / Anomaly 1 below) is FIXED — both malformed-response early returns
> (pre-fix :422-428) now route through `_fail_malformed_invoice_response`, which
> attempts the same best-effort `_cleanup_failed_payment("", label)` as the
> thrown-exception path and stamps `failure_class` into `failure_data`. Three
> pitting tests added in tests/test_rebalance_native_executor_v2.py
> (`test_native_executor_cleans_up_malformed_invoice_response`,
> `test_native_executor_cleans_up_invoice_response_missing_payment_hash`,
> `test_malformed_invoice_cleanup_is_best_effort`) — grep `native_invoice_error`
> in tests/ is no longer zero. Line references below describe the pre-fix code.

Phase 2 — Tier 2 (targeted). Verified 2026-07-01 at HEAD 61b031c against
`docs/audit/contracts/rebalance_native_executor_v2.md` (authored at f905cfd
against 9f8f219). Module unchanged since the contract commit
(`git diff f905cfd HEAD` empty for this file) — all contract line references
are exact on HEAD.

Test run: `.venv/bin/python -m pytest tests/test_rebalance_native_executor_v2.py
tests/test_payment_pending_settlement.py -q` → green (part of a 116-passed
batch). Corpus (frozen sweep, `tools/audit/sweep_routing_stack.py`): status
error tokens `native_route_over_budget` 18, `native_sendpay_error` 17 across
5,191 revenue-status snapshots; 41 segment observations all `router_kind=v3`
(40 market_only, 1 hive_only); history 49 failed / 2 success; 0 sweep
violations.

## Invariants

- **NX-1 (no execution without full route validation)** — **verified** for
  the pre-invoice ordering and two branches; **verified (code-only)** for the
  remaining validation branches. Code: `_validate_route` :277-332 covers every
  claimed check (first hop == source :300-301, final hop == dest :302-303,
  final id == our node :305-307, final msat == amount*1000 :309-312,
  non-increasing :314-321, per-hop required fields :293-298), called before
  any RPC at :395-407. Tests:
  `tests/test_rebalance_native_executor_v2.py::`
  `test_native_executor_rejects_missing_route_before_invoice` (asserts zero
  RPC calls — genuinely pits the ordering) and
  `::test_native_executor_rejects_route_over_budget_before_invoice`. **No
  test pits `first_hop_not_source_channel`, `final_hop_not_dest_channel`,
  `final_hop_not_our_node`, `final_amount_mismatch`,
  `increasing_route_amount`, or missing-field rejection** (the
  `increasing_route_amount` test at tests/test_rebalance_executor.py:481
  targets the legacy executor, not this module). Run: pass.
- **NX-2 (fee budget is a hard pre-send ceiling)** — **verified.** Code:
  planned fee = first-hop msat − delivered msat :323-324, rejection :325-331,
  `native_route_over_budget` prefix :403. Test:
  `::test_native_executor_rejects_route_over_budget_before_invoice` — fee 3 >
  cap 1, exact error string `"native_route_over_budget: route_over_budget:
  3 > 1"`, `fee_sats == 3` surfaced, only `getinfo` called. Corpus: 18
  `native_route_over_budget` error tokens — the gate visibly fires in
  production (it is the top rebalance error token). Run: pass.
- **NX-3 (unresolved payments never reported as failures)** — **verified**
  for the code-200 and proxy-timeout triggers; **verified (code-only)** for
  the `waitsendpay_status=pending` string trigger. Code:
  `_is_payment_unresolved` :342-363 (requires `payment_attempted`), pending
  handler :474-494 (sets `payment_pending`, keeps invoice/payment, no
  exclusions, surfaces planned fee). Tests
  (tests/test_payment_pending_settlement.py, all genuinely pit):
  `::test_waitsendpay_cln_code_200_marks_payment_pending` (also asserts no
  delpay/delinvoice and empty exclusions),
  `::test_waitsendpay_proxy_timeout_marks_payment_pending`,
  `::test_sendpay_proxy_timeout_marks_payment_pending`,
  `::test_pending_result_carries_planned_fee_for_budget_hold` (fee_msat ==
  planned 1000), and the pre-sendpay guard
  `::test_invoice_failure_is_not_payment_pending` (invoice timeout →
  `payment_pending=False` because `payment_attempted` is still False —
  correctly not pending, since no HTLC can be in flight). Engine side of the
  claim pitted by `::test_finish_execution_budget_keeps_reservation_on_pending`,
  `::test_record_rebalance_result_marks_pending_settlement`,
  `::test_retries_skipped_when_payment_pending`, and the four
  `reconcile_*` tests. The third trigger — waitsendpay returning
  `status="pending"` in-band (raise at :450-451, matched at :363) — has no
  test. Corpus: no `pending_settlement` rows and no `payment_pending_timeout`
  tokens in the frozen window — corpus-vacuous. Run: pass.
- **NX-4 (terminal failures cleaned up; pending never; malformed-invoice gap)**
  — **verified**, and the contract's documented gap is **confirmed present on
  HEAD and untested.** Code: `_cleanup_failed_payment` :365-375, called only
  from the terminal branch of the exception handler :508 (pending branch
  returns at :494 without cleanup — pitted by the three pending tests'
  delpay/delinvoice assertions). Tests:
  `tests/test_rebalance_native_executor_v2.py::`
  `test_native_executor_cleans_failed_sendpay_attempt` (asserts exact RPC
  sequence ends `delpay, delinvoice`),
  `tests/test_payment_pending_settlement.py::`
  `test_terminal_waitsendpay_failure_still_cleans_up`. **Gap confirmed**: the
  malformed-invoice-response early returns at :422-424 (`invoice` not a dict)
  and :425-428 (missing `payment_hash`) return from inside the `try` without
  raising, so neither `_cleanup_failed_payment` nor the exception handler
  runs — if CLN actually created the invoice server-side despite the mangled
  proxy response, it lingers under label `rebal-native-<ms>-<scid>` until the
  300 s expiry. No test drives either early return
  (grep `native_invoice_error` in tests/: zero hits). Note the asymmetry with
  the invoice-*exception* path (`test_invoice_failure_is_not_payment_pending`),
  which does reach the handler and attempts `delinvoice` — the malformed-
  *response* path is strictly worse-handled than the thrown-error path.
  Severity low (300 s self-expiry, unpaid invoice holds no funds), but it is
  a genuine cleanup hole. Run: pass.
- **NX-5 (attribution-scaled observation confidence)** — **verified.** Code:
  constants :205-206, tiering :208-257 (channel+direction → 0.85; channel
  only → both directions 0.425; inferred → 0.85/n with 0.2 floor;
  undirected inferred entries dropped :244-252). Tests (all pit exact
  values): `::test_attributed_failure_with_direction_records_full_confidence`,
  `::test_attributed_failure_without_direction_records_both_at_half_confidence`,
  `::test_unattributed_failure_splits_confidence_across_middle_hops`
  (0.85/4), `::test_unattributed_failure_confidence_floor_is_02` (6 middles →
  0.2 > 0.85/6), `::test_single_middle_hop_fallback_keeps_full_confidence`.
  Corpus: 41 segment observations written, consistent with the store path
  being live. Run: pass.
- **NX-6 (inferred exclusions middle-hops-only; attributed verbatim)** —
  **verified** for the two main branches, **verified (code-only)** for the
  failure-class restriction and the own-hop-verbatim edge. Code:
  `_exclude_from_failure` :156-194 (verbatim attributed :162-167; class gate
  :169-171; `route[1:-1]` fallback :179-194). Tests:
  `::test_native_executor_extracts_erring_channel_from_rpc_error` (verbatim
  `150x1x0/1`), `::test_native_executor_fallback_excludes_attempted_middle_path`
  (middle hop only; route summary intact). **No test asserts** (a) that a
  non-liquidity/fee unattributed failure yields zero exclusions, or (b) the
  documented sharp edge that an attributed `erring_channel` naming our own
  pinned source/dest hop is excluded verbatim
  (`test_terminal_waitsendpay_failure_still_cleans_up` supplies erring own-hop
  data but never inspects `excluded_channels`). Run: pass.
- **NX-7 (success fees are actuals)** — **REFUTED as test-pitted at this
  module's level, downgraded to verified (code-only)** for the
  `amount_sent_msat` derivation; **verified (code-only)** for the
  first-hop fallback. Refutation pass 2026-07-01: mutating the derivation to
  always use the route-estimate path (`parse_msat(route[0]["amount_msat"])`)
  survives the entire 19-test battery — the cited test's fixture route has
  first hop 101_000 msat, identical to the mocked
  `amount_sent_msat: "101000msat"`, so `fee_msat == 1000` holds on both
  paths and the parenthetical "(not the route estimate path)" was false: the
  test cannot distinguish actuals from the planned estimate. Code: :453-465.
  Test:
  `::test_native_executor_executes_priced_route_with_sendpay` — waitsendpay
  reports `amount_sent_msat: "101000msat"`, asserted `fee_msat == 1000`,
  `fee_sats == 1`. The fallback
  `parse_msat(route[0]["amount_msat"])` when `amount_sent_msat` is absent
  (:458) is untested — and because planned == sent in every fixture, so is
  the primary path's distinctness. Engine-side actuals also pitted by
  `tests/test_payment_pending_settlement.py::`
  `test_reconcile_settled_payment_records_cost_and_marks_spent` (actual 5000
  msat from listsendpays). Corpus: prior sweep S1 (fee ≤ max_fee on
  successes) 0 violations over the deduped history; only 2 successes in the
  frozen window, so weak corpus power. Run: pass.

## Purpose-section claims

- Constructed by `RebalanceEngine._make_executor` with cached node id:
  **verified** — now at rebalance_engine_v2.py:2366-2373 (contract cites
  :2260-2268; drift from 441b8e3's +122 engine lines), `our_id=self._our_id`
  injected. Legacy alias re-export confirmed at
  modules/rebalance_executor_v2.py:11.
- Timeout-kwarg proxy fallback (:36-48): **verified** —
  `::test_waitsendpay_passes_timeout_kwarg_to_proxy` (== 60) and
  `::test_waitsendpay_timeout_kwarg_falls_back_for_legacy_rpc`.
- `stable_failure_reason` mapping: spot-checked by
  tests/test_rebalance_execution.py:21 (`native_route_invalid:` →
  `local_policy_block`); taxonomy-match uncertainty stands.

## Gaps

1. **NX-4 malformed-invoice early returns (:422-428) have no cleanup and no
   test** — the one concrete correctness hole in this module. A minimal fix
   would route both early returns through `_cleanup_failed_payment("", label)`;
   finding only, no fix applied.
2. NX-1: six of eight validation branches unpitted; a regression that, e.g.,
   dropped the `final_hop_not_our_node` check (funds would leave the node)
   would pass the current suite.
3. NX-3: the in-band `waitsendpay_status=pending` trigger (:450-451/:363) is
   unpitted.
4. NX-6: class-gate (no exclusions for timeout/unknown) and own-hop-verbatim
   attribution are unpitted.
5. NX-7: first-hop fee fallback when `amount_sent_msat` is missing is
   unpitted.
6. Contract uncertainties re-checked and still open: invoice expiry (300 s) vs
   HTLC-in-flight interplay untested; duplicate-label millisecond collision
   unenforced.

## Anomalies

1. The malformed-invoice early returns also leave `result.failure_data`
   without a `failure_class` key (only the initial `route_summary` from :393)
   — every other failure path sets one; downstream consumers keying on
   `failure_class` see an inconsistent shape for this path.
2. Corpus shows `native_route_over_budget` (18) as the top rebalance error
   token — NX-2 doing real work — while `native_sendpay_error` (17) confirms
   the terminal-failure path runs in production; the pending path (NX-3) has
   zero corpus occurrences, so its correctness rests entirely on the
   (thorough) test suite.
3. Engine call-site line references in the contract drifted after 441b8e3
   (see Purpose-section claims); module-internal references are exact.

## Refutation pass (2026-07-01)

Adversarial re-verification at HEAD dac9b48 (module byte-identical to f905cfd
through HEAD, matching this doc's drift check; suites re-run: 19 passed).
Method: mutation testing in a scratch copy + frozen-corpus re-sweep.

- Attacked: NX-1..NX-7, purpose claims, corpus statements.
- Survived — every decisive mutation was killed by the cited test:
  NX-1 ordering (accepting an empty route kills
  `test_native_executor_rejects_missing_route_before_invoice`, whose
  zero-RPC assertion is real); NX-2 (disabling the budget comparison kills
  `test_native_executor_rejects_route_over_budget_before_invoice`); NX-3
  (treating code 200 as terminal kills
  `test_waitsendpay_cln_code_200_marks_payment_pending`); NX-4 (gutting
  `_cleanup_failed_payment` kills both cleanup tests); NX-5 (0.85 → 0.5
  kills the full-confidence test; floor 0.2 → 0.05 kills the floor test —
  the exact-value assertions are real); NX-6 (excluding the full route
  instead of `route[1:-1]` kills four tests). `_validate_route` re-read:
  every claimed branch present (:277-332); the six-unpitted-branches gap
  and the malformed-invoice cleanup hole (:422-428) stand as documented
  (grep `native_invoice_error` in tests/: still zero hits).
- Refuted: NX-7's `amount_sent_msat` derivation as test-pitted (inline note)
  — the fixture makes planned fee == actual fee, so the "actuals, not
  estimate" property is code-only at this module's level. The engine-side
  reconcile test (actual 5000 msat from listsendpays) still pits actuals at
  the engine layer, which is why this is a clause downgrade rather than an
  invariant refutation.
- Corpus: this doc's numbers match the frozen sweep exactly (re-run
  2026-07-01: over-budget 18, sendpay_error 17, 41 observations, history
  49 failed / 2 success, 0 violations); NX-3 correctly labeled
  corpus-vacuous.

Counts: attacked 7 invariants + 3 purpose claims; survived 9; refuted 1
clause (NX-7 actuals derivation → code-only).
