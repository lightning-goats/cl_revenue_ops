# Phase 2 Verification — boltz_manager.py

Contract: docs/audit/contracts/boltz_manager.md (BM-I1..BM-I14).
Evidence: test mapping (subagent, spot-checked), code confirmation on current HEAD
(module unchanged since contract commit f905cfd — `git diff f905cfd..HEAD` empty, so
all contract line citations remain valid), corpus sweep
`tools/audit/sweep_planner_boltz_hints.py` over 613 (hive-nexus-01) + 614
(hive-nexus-02) revenue-spend-ledger snapshots. CORRECTED (refutation pass): snapshot
coverage is 2026-06-08 → 2026-06-20 plus a single terminal snapshot on 2026-07-01 —
NOT "2026-05-19 → 2026-07-01"; 2026-06-21..06-30 has zero snapshots.
All cited test files pass on HEAD: 184 passed (test_boltz_manager.py,
test_boltz_capex_gating.py, test_boltz_integration.py, test_boltz_invoice_validation.py,
test_capex_boltz.py, test_budget_recursion_fix.py, test_boltz_structural_loopout.py).

**Corpus caveat (dominates every corpus column below): zero Boltz activity in the
entire corpus.** Max rolling-24h spent_by_category is empty and the boltz event
count is 0 in all 1,227 ledger snapshots on both nodes. BM-H2 passes (0 violations
in 613+614 snapshots) but trivially. Every BM invariant is corpus-vacuous; verdicts
rest on tests and code.

| Invariant | Verdict | Evidence |
|---|---|---|
| BM-I1 no subprocess while disabled (`_run` → `_ensure_enabled`) | **verified (code-only)** | code confirmed (:220-233: every command path funnels through `_run`, which raises before subprocess when cfg.enabled is false); **no covering test** (no test constructs cfg.enabled=False) |
| BM-I2 enforce_budget rejects when estimated fee > remaining unified 24h budget; same estimator for gate and accounting | **verified** | test_boltz_integration.py::TestBoltzCostComponents::test_budget_enforcement_blocks_over_limit / _allows_within_limit (helper `_enforce_budget_for_quote` level); loop wiring exercised indirectly via test_boltz_capex_gating.py::TestRetryBudgetUsesNestedQuote; code confirmed (:1306-1324 helper uses `_estimate_swap_fee_sats`, same function as `_record_swap_result` accounting; wired at loop_in :1384-1391, loop_out :1471-1478). Partial: the loop_in/loop_out "status: rejected" wiring and the same-estimator symmetry claim have no direct test |
| BM-I3 budget-check + createswap atomic under `_swap_creation_lock` | **verified** | test_boltz_manager.py::test_concurrent_loop_in_serialized (interleaving pitted) + ::test_swap_creation_lock_exists (tautological — attribute existence only); code confirmed (:54-55, loop_in `with self._swap_creation_lock:` :1382, loop_out :1462-1465, chainswap :1848). Caveat: the serialization test's key assertion is inside `if len(enforce_starts) >= 2:` and can pass vacuously if a thread errors early |
| BM-I4 treasury swaps gated by tactical budget; **chainswap bypasses tactical + channel capex gates** | **verified** (caveat re-confirmed) | test_capex_boltz.py::TestTacticalGate (zero-budget blocks, sufficient allows, channel-targeted bypasses tactical) + test_boltz_capex_gating.py::test_treasury_swap_not_gated_by_channel_budget; code confirmed (:76-106, called at :1394-1403, :1481-1490). **Phase 1 finding re-confirmed on HEAD: `chainswap` (:1837-1877) runs only the unified-budget gate under the creation lock (:1848-1859) — no `check_tactical_budget`, no `check_channel_capex_budget` — while its fee is recorded into the same "boltz" category (`chainswap` ∈ `_SPEND_RECORD_SOURCES`, :1156). Zero tests mention chainswap anywhere in tests/** |
| BM-I5 channel-targeted swaps ≤ channel capex budget, fail closed; structural-envelope bypass only | **verified** | test_boltz_capex_gating.py::TestChannelGate (over-budget rejects, unknown channel rejects, lookup failure fails closed, scid normalization) + loop_in/loop_out rejection wiring tests + TestStructuralBypass (bypass and non-structural still gated); code confirmed (:123-189 fail-closed, :1499-1518 bypass gated on `structural and self._structural_envelope_sats() > 0`) |
| BM-I6 structural bypass fails closed (no provider / error / non-positive → 0) | **verified** | test_boltz_capex_gating.py::TestStructuralBypass (envelope zero, no provider, provider failure) + test_boltz_structural_loopout.py (envelope-spent block, fail-closed on spend-query error); code confirmed (:108-121) |
| BM-I7 executed swap fee recorded exactly once to "boltz" category (structural/swap_fee subcategory) | **verified** | test_boltz_capex_gating.py::TestSpendRecording (loop_in records; probe/error/zero-fee don't; structural vs swap_fee subcategory) + TestRecordBoltzSpend (event written, non-positive/missing-id rejected, db-failure False, tactical depleted); code confirmed (:1193-1217; idempotency via event_id `boltz:{sid}` INSERT OR REPLACE, capex_budget.py record_boltz_spend). Partial: no duplicate-call test pins the exactly-once (overwrite-not-accumulate) mechanism |
| BM-I8 pending swaps count as reserved; reserved capped at remaining budget | **verified** | test_boltz_manager.py (pending counted, error swap not reserved, old pending not reserved) + TestBudgetStatus::test_pending_swap_reserves_count_toward_budget; code confirmed (:898-923, cap at min(boltz cfg budget, passed unified cap) minus spent). Partial: the cap-clamp leg (over-estimation cannot wedge) is never exercised — fixtures keep reserves under budget |
| BM-I9 get_boltz_cost_components never calls global_budget_limit_provider | **verified** | test_budget_recursion_fix.py (never-call, explicit cap used, wiring does not recurse) — directly pits the mutual-recursion guard; code confirmed (:855-868) |
| BM-I10 journal/ignore-list read-modify-write serialized under locks | **verified (code-only)** | code confirmed (`_record_swap_result` under `_journal_lock` :1158-1161; `manage_external_pay_ignores` under `_ignored_swaps_lock` :1087-1092); **no covering concurrency test** (TestPreClaim tests a different lock) |
| BM-I11 amount_sats ≤ 0 raises before any gate/subprocess | **verified (code-only)** | code confirmed (quote :1340-1342, loop_in :1375-1377, loop_out :1457-1459, chainswap :1839-1841); **no covering test** passes non-positive amounts |
| BM-I12 chanIds capability handling: cached-False → external-pay first-hop pinning; unknown → attempt, on rejection retry unpinned with warning + budget re-check | **verified** | test_boltz_integration.py::TestExternalPayFallback (detection/extraction helpers), test_boltz_capex_gating.py::TestRetryBudgetUsesNestedQuote::test_exception_retry_checks_budget_on_nested_quote (sync-exception retry + budget re-check on nested quote, no double count), test_boltz_invoice_validation.py::test_loop_out_external_pay_threads_expected_amount (cached-False path); code confirmed (:1654-1656 unreachable-in-practice branch, sync :1676-1685, async swapinfo probe :1686-1716, exception :1717-1728 — all record warnings and re-check budget before the second creation). Partial: the async probe retry path is untested |
| BM-I13 external-pay never pays more principal than requested | **verified** | test_boltz_invoice_validation.py (rejects over-amount invoice, rejects amountless invoice when amount expected, accepts matching, msat string format, skip when no expectation) — reject occurs before `pay`; code confirmed (:580-594 raises BoltzCliError) |
| BM-I14 principal movements (withdraw/deposit/refund/claim) pass no budget gate | **verified (code-only)** | code confirmed (refund :1824-1827, claim :1829-1835, withdraw :1879-1901, deposit_address — all go straight to `_run` with no gate); **no covering test**; documented-as-designed, BM-I13 is the only principal guard |

## Gaps

- **No covering tests for BM-I1, BM-I10, BM-I11, BM-I14** (code-only verdicts).
  BM-I1 is the most consequential: nothing pins that a disabled integration cannot
  shell out to boltzcli.
- **The BM-I4 chainswap bypass — a known Phase 1 finding — remains completely
  unpinned**: zero tests reference chainswap at all, so neither the current bypass
  nor a future fix would be caught by the suite.
- Partial legs untested: BM-I2 loop-level rejection wiring and gate/accounting
  estimator symmetry; BM-I3's serialization test can pass vacuously (conditional
  assertion) and its companion lock test is tautological; BM-I7's exactly-once
  idempotency (no duplicate-recording test); BM-I8's reserved-cap clamp; BM-I12's
  async swapinfo-probe retry.
- Corpus is uninformative for this module: zero swaps executed in the study
  window, so no invariant has production evidence. BM-H1/BM-H3 are dead on this
  corpus (n=0 identifiable swap events); BM-H2 passes only vacuously.

## Anomalies

1. **Boltz was effectively dormant in production during the observed window**
   (0 boltz spend events across 1,227 ledger snapshots on both nodes). CORRECTED
   (refutation pass): the ledgers are rolling-24h views and the corpus has no
   snapshots for 2026-06-21..06-30, so dormancy is established for ~12 observed days
   plus the 24h before 2026-07-01, not the "entire study window". Either the
   integration/auto-cycle is disabled or it never selected an action. This answers
   the contract's first uncertainty in the negative for this corpus and should be
   recorded as scoping input for Phase 3/4: the module's budget machinery is
   exercised only by tests today.
2. The `_SPEND_RECORD_SOURCES` set includes `chainswap`, so chainswap fees deplete
   the same tactical "boltz" category whose creation-time gate they bypass —
   the asymmetry is one-directional (spend counted, gate skipped), which makes the
   bypass budget-visible after the fact but unenforced at creation.

## Refutation pass (2026-07-01)

Adversarial re-verification on HEAD (dac9b48; `git diff f905cfd..HEAD` on the module
still empty). All 14 verdicts attacked; **0 refuted, 14 survived**. Method: re-ran all
7 cited test files (184 passed reproduced), re-ran the sweep (BM-H2 vacuous-pass and
zero-activity inventory reproduced), independently re-confirmed every code-only claim
on HEAD, and audited the vacuous-corpus handling.

Findings:

1. **Code-only verdicts (BM-I1, I10, I11, I14) independently re-confirmed.**
   `subprocess` is invoked at exactly one site in the module (:236, inside `_run`),
   with no `Popen`/`os.system`/`check_*` anywhere, and `_run` calls `_ensure_enabled`
   first — BM-I1's funnel claim is structurally airtight, not just path-sampled.
   Journal/ignore locks (:1158-1161, :1087-1092), amount_sats<=0 raises in
   quote/loop_in/loop_out/chainswap, and gate-free refund/claim/withdraw/deposit_address
   all match the cited lines. The "no covering test" labels are accurate.
2. **BM-I4 chainswap bypass direction confirmed on HEAD**: chainswap (:1848-1859)
   runs `_enforce_budget_for_quote` only — no `check_tactical_budget`, no
   `check_channel_capex_budget` — and records into the "boltz" category
   (`chainswap` ∈ `_SPEND_RECORD_SOURCES` :1156). `grep -rn chainswap tests/` returns
   nothing, confirming zero test coverage. One sharpening note: chainswap feeds the
   raw `boltzcli quote … chain` JSON to the gate (loop_in/loop_out feed the nested
   `quote["quote"]`); `_estimate_swap_fee_sats` (:738) tolerates unknown shapes via
   its recursive fee-key fallback, so the unified gate is not vacuous, but the
   chain-quote shape is exercised by no test either.
3. **BM-I3 caveat verified by reading the test**: the serialization assertion in
   test_concurrent_loop_in_serialized is inside
   `if len(enforce_starts) >= 2 and len(enforce_ends) >= 1:` exactly as documented —
   an early thread error yields a silent pass.
4. **No BM verdict leans on the vacuous corpus as positive evidence** — every
   invariant row cites tests/code only, the header flags the vacuity up front, and
   BM-H2 is labeled trivially-passing. Confirmed clean.
5. Corpus-window correction (header, Anomaly 1): dormancy is evidenced for the
   observed ~12 days + the final 24h window, not a continuous 6-week study window.
