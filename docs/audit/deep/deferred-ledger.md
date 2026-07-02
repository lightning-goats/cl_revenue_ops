# Deferred-Findings Ledger — Deep Audit Phase 0B

Exhumed prior-era audit findings that were **deferred**, **mitigated-not-fixed**, or
otherwise **left open**, so the deep audit (Phases 5/6) can re-adjudicate each against
current code. Every row is an OPEN item for Phase 5/6 adjudication unless its
`current_status` already reads LIKELY-FIXED (still requires code-re-derivation, never
trusted from the doc) or CONTRADICTION (Phase 6 doc-conformance).

**Trust posture (per campaign rules):** prior audit docs are used only as *risk maps* and
*claims to re-verify*. A `LIKELY-FIXED-<commit>` status is a lead, not evidence — Phase
4/5 must re-derive from code. UNRECOVERABLE means the source could not be mined.

Format:
`DEF-NNN | era | original_id | severity | dimension | module | description | current_status`

---

## Summary counts

- **Total deferred entries: 94** (DEF-001 … DEF-094)

### By era
| Era | Range | Count |
|-----|-------|-------|
| Jan-2026 (Red Team / Zero-Tolerance — recovered from deleted files) | DEF-001..018 | 18 |
| Mar-2026 (Sessions 3–6 + fee-controller + rebalancer module audits) | DEF-019..067 | 49 |
| Jun/Jul-2026 (verification campaign: operator-decisions + phase2-summary) | DEF-068..088 | 21 |
| Doc contradictions (stale-doc inventory, Phase 6) | DEF-089..094 | 6 |

### By dimension (approx.)
| Dimension | Count |
|-----------|-------|
| Concurrency / thread-safety / atomicity | 18 |
| Security / RPC & input validation | 7 |
| Resource / DB growth / FD leak | 6 |
| Data integrity / financial-correctness | 20 |
| Algorithm / economic-model | 14 |
| Test coverage (phantom/tautological/missing) | 11 |
| Daemon survival / lifecycle | 3 |
| Docs conformance / contradiction | 15 |

### Recovery / disposition notes
- The **Jan-2026 Red Team reports WERE recoverable** from git history — deleted in commit
  `fe38b1b` ("Remove outdated documentation…"), mined read-only via `git show fe38b1b^:<path>`
  for `docs/audits/2026-01-09-zero-tolerance-security-audit.md`,
  `docs/audits/PHASE7_RED_TEAM_REPORT.md`, and `docs/audits/ZERO_TOLERANCE_AUDIT.md`.
  Files were NOT restored to the tree. **Security dimension is therefore baselined, not un-baselined.**
- **Already-FIXED items are recorded as `LIKELY-FIXED-<commit>` and kept in the ledger** so
  Phase 5/6 do not re-litigate them but still re-derive the fix from code.
- The **2026-03-27 full-plugin audit** (4 findings) and the **2026-05-19 standalone
  independence audit** are *not carried forward as open*: all 4 of the former were resolved
  and re-verified in-session; the latter reported only positive/neutral findings. See the
  "Closed-in-era (no carry-forward)" note at the bottom.

---

## Jan-2026 — Red Team / Zero-Tolerance (recovered from deleted files)

Sources: `2026-01-09-zero-tolerance-security-audit.md` (blocking criticals + majors),
`PHASE7_RED_TEAM_REPORT.md` (7 spec-era vulns + 2 v1.4 deferrals),
`ZERO_TOLERANCE_AUDIT.md` (Jan-1 companion; claims most items fixed — treat as claim).
Note: several targets (sling, CLBOSS `_wait_for_job`) predate the native-rebalance/DTS
rewrites, so status leans LIKELY-FIXED-by-rewrite but must be re-derived.

```
DEF-001 | Jan-2026 | ZT CRITICAL-01 | High     | concurrency        | rebalancer            | Daily-budget check + spend not atomic (_check_budget_constraint): concurrent jobs both pass, overspend | OPEN (atomic reserve_budget likely present now — re-derive; Phase2 flags reserve-budget path phantom-tested)
DEF-002 | Jan-2026 | ZT CRITICAL-02 | High     | econ/validation    | fee_controller        | Fee-floor fallback returns 1 PPM for zero-revenue channels; ChainCostDefaults.calculate_floor_ppm not called on that path | OPEN (re-verify floor path)
DEF-003 | Jan-2026 | ZT CRITICAL-03 | High     | resource/daemon    | rebalancer            | Unbounded retry in _wait_for_job (while True + 30s sleep); sling_job_timeout not enforced | LIKELY-FIXED (sling path superseded by native rebalance_execution — re-verify no unbounded loop remains)
DEF-004 | Jan-2026 | ZT MAJOR-01    | Medium   | concurrency        | clboss_manager        | TOCTOU in ensure_unmanaged_for_channel: CLBOSS re-manages between unmanage and setchannel | OPEN
DEF-005 | Jan-2026 | ZT MAJOR-02    | Medium   | security/RPC       | cl-revenue-ops.py     | force=true bypasses cooldown/budget/deadband on multiple RPCs; no trust-boundary guard | OPEN (Phase 1 cross-check)
DEF-006 | Jan-2026 | ZT MAJOR-03    | Low-Med  | data-integrity     | database              | Forward re-ingestion (INSERT OR IGNORE) can mislead volume calc via returned_rows count | OPEN
DEF-007 | Jan-2026 | ZT MAJOR-04    | Low-Med  | resource           | database              | Thread-local sqlite connections never explicitly cleaned; FD accumulation over uptime (== Session3 I-6) | OPEN (dup DEF-021)
DEF-008 | Jan-2026 | ZT MAJOR-05    | Medium   | correctness        | config                | ConfigSnapshot omits hive_fee_ppm / hive_rebalance_tolerance → inconsistent hive fee behavior mid-update | OPEN (re-verify current snapshot field coverage)
DEF-009 | Jan-2026 | ZT MAJOR-06    | Medium   | algorithm/econ     | rebalancer            | Kelly sizing uses daily_budget as bankroll (not routing capital) — misapplies Kelly formula | OPEN
DEF-010 | Jan-2026 | P7 CRITICAL-02 | High     | concurrency        | config                | Config hot-swap "torn read": worker binds old budget mid-cycle. Mitigation = ConfigSnapshot | OPEN (partial — ConfigSnapshot exists but see C-3/TS-2 direct-read gap DEF-032/044)
DEF-011 | Jan-2026 | P7 CRITICAL-03 | High     | correctness        | config                | Config persistence "ghost state": memory updated before disk flush; failed write → config vanishes on restart. Mitigation = transactional read-back-verify | LIKELY-FIXED (Session4 confirms transactional update_runtime DB-before-memory — re-verify read-back)
DEF-012 | Jan-2026 | P7 v1.4-defer  | High     | security/econ      | fee_controller/hive   | Peer-Syncing Arbitrage "Anchor & Drain": drain small channels via large-channel fee sync. Explicitly deferred to v1.4 "Floor-Only sync" | OPEN
DEF-013 | Jan-2026 | P7 v1.4-defer  | Medium   | algorithm/econ     | flow_analysis         | Flow-asymmetry false positives double-tax valid circular rebalances. Deferred pending traffic-analysis module | OPEN
DEF-014 | Jan-2026 | P7 CRITICAL-01 | High     | econ/DoS           | fee_controller        | Vegas Reflex "latch bomb": 4h routing paralysis via cheap mempool spike. Spec mitigation = exponential decay state | OPEN (re-verify VegasReflex decay in code)
DEF-015 | Jan-2026 | P7 HIGH-01     | High     | econ/DoS           | fee_controller        | Dust-flood slot stuffing triggers scarcity pricing. Spec mitigation = value-weighted utilization MAX(slots,VaR) | OPEN (re-verify utilization metric)
DEF-016 | Jan-2026 | P7 HIGH-02     | High     | algorithm/econ     | rebalancer/fee        | Scarcity deadlock: rebalance into channel pushes it over 35% util, pricing out the bought liquidity. Spec mitigation = predictive eviction / post_rebalance_utilization forecast | OPEN
DEF-017 | Jan-2026 | P7 HIGH-03     | High     | econ               | fee_controller        | Confirmation-window front-running (200–400% mempool spike). Spec mitigation = probabilistic early trigger | OPEN (re-verify)
DEF-018 | Jan-2026 | P7 MEDIUM-01   | Medium   | algorithm          | fee_controller        | Symmetric EMA lags on congestion clearing → revenue loss. Spec mitigation = asymmetric EMA (up 0.4 / down 0.1) | OPEN (re-verify EMA constants)
```

---

## Mar-2026 — Session 3–6 + module audits (2026-03-02)

### Session 3 — database.py + profitability_analyzer.py
```
DEF-019 | Mar-2026 | S3 C-1(db)     | Critical | concurrency        | database               | Autocommit + manual BEGIN → corruption risk on rollback failure. Mitigated by single-threaded write pattern (untested assumption — Phase 2 target) | OPEN
DEF-020 | Mar-2026 | S3 I-5         | Important| data-integrity     | database               | get_total_routing_revenue double-count at rollup boundary if cleanup crashes mid-transaction | OPEN
DEF-021 | Mar-2026 | S3 I-6         | Important| resource           | database               | _thread_connections grows unbounded; daemon threads never close_connection (== Jan MAJOR-04) | OPEN (dup DEF-007)
DEF-022 | Mar-2026 | S3 I-7         | Important| security/validation| database               | Unclamped limit in get_rebalance_history_by_peer (bypasses SEC-10 clamping) → memory exhaustion | OPEN
DEF-023 | Mar-2026 | S3 I-8         | Important| performance        | database               | Missing index on rebalance_costs(channel_id, timestamp); full-scan on large tables (Phase 7) | OPEN
DEF-024 | Mar-2026 | S3 I-9         | Important| concurrency        | database               | increment_failure_count non-atomic read-after-write | OPEN
DEF-025 | Mar-2026 | S3 S-1..S-8    | Suggestion| data-integrity    | database/profitability | 8 suggestions: RoC metric mismatch, 50/50 split, _bleeder_cache non-atomic, record_mempool_fee prune+insert non-atomic, 50K fee cap, lifetime-report formula divergence, no FKs, portfolio_metrics DDL embedded in _migrate_kalman_schema | OPEN (group)
```

### Session 4 — flow_analysis.py + portfolio_optimizer.py + config.py
```
DEF-026 | Mar-2026 | S4 flow C-1    | Critical | algorithm          | flow_analysis          | has_observation=True feeds zero observations to idle channels (design tradeoff — alt caused convergence to 0.0) | OPEN
DEF-027 | Mar-2026 | S4 flow I-2    | Important| concurrency        | flow_analysis          | Race between concurrent analyze_channel + analyze_all_channels. Mitigated by single-threaded timer | OPEN
DEF-028 | Mar-2026 | S4 flow I-4    | Important| algorithm          | flow_analysis          | 24h window + hourly updates → correlated observations | OPEN
DEF-029 | Mar-2026 | S4 flow I-9    | Important| concurrency        | flow_analysis          | analyze_channel RPC can double-update Kalman filter (== later FC-I16 family) | OPEN
DEF-030 | Mar-2026 | S4 port C-1    | Critical | correctness        | portfolio_optimizer    | Variance inconsistency between channel_stats and covariance-matrix diagonal (advisory module) | OPEN
DEF-031 | Mar-2026 | S4 port I-3..10| Important| algorithm          | portfolio_optimizer    | 6 issues: scale-dependent convergence tol, Sharpe ignores correlation, Gershgorin over-inflation, 100% idiosyncratic single-channel, missing-bucket vs zero-revenue, simplex tolerance too tight | OPEN (group)
DEF-032 | Mar-2026 | S4 config C-3  | Critical | concurrency        | rebalancer/config      | 52 direct self.config.* reads bypass snapshot pattern (torn reads); large refactor. == Session6 TS-2 == Jan P7 CRITICAL-02 residue | OPEN (Phase 2 must CLOSE, not re-defer)
DEF-033 | Mar-2026 | S4 config I-6  | Important| error-handling     | config                 | _apply_override silently swallows errors | OPEN
DEF-034 | Mar-2026 | S4 config I-7  | Important| correctness        | config                 | expansion_treasury_min_source_local_pct uses 0-100 scale unlike other pct fields | OPEN
```

### Session 5 — Tier 3/4 (policy_manager, hive_bridge, capacity_planner, clboss_manager, utils, main)
```
DEF-035 | Mar-2026 | S5 HB-6        | Low      | validation         | hive_bridge            | execute_circular_rebalance no amount validation (requires hive coordinator) | OPEN
DEF-036 | Mar-2026 | S5 PM-1        | Important| concurrency        | policy_manager         | _load_cache can overwrite concurrent _update_cache write-through (narrow race) | OPEN
DEF-037 | Mar-2026 | S5 PM-4        | Important| ordering           | policy_manager         | Batch rate-limit timestamps written before DB COMMIT | OPEN
DEF-038 | Mar-2026 | S5 CP-1        | Suggestion| compat            | capacity_planner       | Uses deprecated listpeers API (still functional) | OPEN
DEF-039 | Mar-2026 | S5 CP-2        | Suggestion| robustness        | capacity_planner       | peer_id can be None in channel records (defensive gap) | OPEN
DEF-040 | Mar-2026 | S5 CB-1        | Suggestion| concurrency        | clboss_manager         | _clboss_available not thread-safe (single-writer mitigates) | OPEN
DEF-041 | Mar-2026 | S5 U-1         | Suggestion| validation         | utils                  | parse_msat silently converts booleans to 1/0 msat | OPEN
```

### Session 6 — Cross-cutting spec alignment
```
DEF-042 | Mar-2026 | S6 defaults    | Important| docs/config        | config vs main options | config.py defaults differ from option defaults: enable_kelly, min_fee_ppm, kelly_fraction, max_fee_ppm, scarcity_threshold (option wins at runtime; affects tests) | OPEN (Phase 6 code-as-truth)
DEF-043 | Mar-2026 | S6 conf.full   | Important| docs               | conf.full vs code      | conf.full drift: low/high_liquidity_threshold, proportional_budget_pct, rebalance_min_profit, min_wallet_reserve, max_fee_ppm 2500, scarcity 0.15 vs code | CONTRADICTION (Phase 6)
DEF-044 | Mar-2026 | S6 TS-2        | High     | concurrency        | rebalancer/fee/main    | 52+ direct config reads in background threads w/o snapshot (== C-3 DEF-032) | OPEN (cross-ref)
DEF-045 | Mar-2026 | S6 TS-0        | Medium   | daemon-lifecycle   | cl-revenue-ops.py      | Background loops start before init() completes; mitigated by 10/60/120s delays, no hard sync | OPEN (Phase 1/2)
DEF-046 | Mar-2026 | S6 err-handling| Important| error-handling     | cl-revenue-ops.py      | revenue-hive-status unguarded; revenue-capacity-report raises RpcError vs error dict; 14+ bare except swallowing without logging | OPEN
```

### Fee-controller module audit (Session 2)
```
DEF-047 | Mar-2026 | FC I-10        | Important| algorithm/econ     | fee_controller         | Fleet-data injection can displace local observations (MAX_OBSERVATIONS=200; fleet_weight=0.25). == campaign FC-I6 family | OPEN (see DEF-073)
DEF-048 | Mar-2026 | FC I-11        | Important| algorithm          | fee_controller         | AIMD success_score denominator (*10.0) too demanding; channels perpetually failure/neutral | OPEN
DEF-049 | Mar-2026 | FC I-12        | Important| validation         | fee_controller         | No per-channel rate limit in fee_controller (only in policy_manager) | OPEN
DEF-050 | Mar-2026 | FC I-13        | Important| algorithm          | fee_controller         | AIMD recovery extremely slow (~250 successes 0.5→1.0); fees depressed up to 50% for ~25h | OPEN
DEF-051 | Mar-2026 | FC I-14        | Important| concurrency        | fee_controller         | RLock (_state_lock) held across DB I/O for entire fee cycle; blocks RPC set_channel_fee | OPEN (also FC S-3)
DEF-052 | Mar-2026 | FC I-15        | Important| resource/DB        | fee_controller         | Dual state duplication (HillClimbState + ThompsonAIMDState) doubles DB writes | LIKELY-FIXED-73a7b09 (DTS+PID-only refactor purged HC/Thompson generations — re-verify no dual state remains)
DEF-053 | Mar-2026 | FC I-16        | Important| concurrency        | fee_controller         | Non-atomic state save (memory before DB); mitigated by single-threaded fee cycle | OPEN
DEF-054 | Mar-2026 | FC I-17        | Important| spec-gap           | fee_controller         | HIVE_COORDINATED not implemented as per-peer policy (global toggle only) | OPEN
DEF-055 | Mar-2026 | FC S-1..S-12   | Suggestion| mixed             | fee_controller         | 12 suggestions incl. ABS_MIN_FEE_PPM=0 surprise, phantom observation on RPC failure (S-2), _state_lock across cycle (S-3), stale HillClimbing log/docstring refs (S-5/S-10), gossip hysteresis untested (S-6), O(n) _recompute_posterior (S-11) | OPEN (group; some stale-ref items → Phase 6)
```

### Rebalancer module audit (Session 1) — deferred Important/Suggestion
Note: many sling-specific items may be moot post native-rebalance rewrite — re-derive.
```
DEF-056 | Mar-2026 | RB I-5         | Important| data-integrity     | rebalancer             | Balance-delta false positive under concurrent forwarding → premature success detection | OPEN (sling-era; re-verify)
DEF-057 | Mar-2026 | RB I-6         | Important| robustness         | rebalancer             | Bulk sling-stats fallback wrong structure → jobs appear stuck until timeout | LIKELY-FIXED (sling superseded — re-verify)
DEF-058 | Mar-2026 | RB I-7         | Important| concurrency        | rebalancer             | Raw DB access in cleanup_orphans bypasses transaction safety (UPDATE + budget release not atomic) | OPEN
DEF-059 | Mar-2026 | RB I-9         | Important| robustness         | rebalancer             | execute_once auto-heal deletes legitimate background job | OPEN (sling-era; re-verify)
DEF-060 | Mar-2026 | RB I-13        | Important| concurrency        | rebalancer             | _budget_hot_channel_only flag unprotected across threads | OPEN
DEF-061 | Mar-2026 | RB I-14        | Important| concurrency        | rebalancer             | _fee_cache unprotected (mitigated by GIL) | OPEN
DEF-062 | Mar-2026 | RB I-15        | Important| algorithm          | rebalancer             | Fixed 48h futility cooldown regardless of failure count | OPEN
DEF-063 | Mar-2026 | RB I-16        | Important| data-integrity     | rebalancer             | SCID-keyed failure counts reset by splice → chronic problems evade breaker | OPEN
DEF-064 | Mar-2026 | RB I-18        | Important| spec-gap           | rebalancer             | Predictive rebalancing (should_preemptive_rebalance) not implemented | OPEN
DEF-065 | Mar-2026 | RB I-19        | Important| spec-gap           | rebalancer             | HIVE_COORDINATED fee strategy not handled (only FeeStrategy.HIVE) | OPEN
DEF-066 | Mar-2026 | RB I-21..I-28  | Important| test-coverage      | rebalancer             | Zero coverage: MCF targets, NNLB opportunity exec, peer-quality gating, hive outcome args, fleet mutual-benefit, AskRene-Kelly blend, partial-timeout handler, budget-exceeded handler | OPEN (group)
DEF-067 | Mar-2026 | RB S-1..S-16   | Suggestion| mixed             | rebalancer             | 16 suggestions incl. reservation-not-counted (S-2), fee not recorded on stop_all_jobs (S-3), inbound-fee double-count (S-4), _peer_inbound_fees unlocked (S-9), N+1 query (S-13), 20% savings-gate missing (S-14), phase-number misalignment (S-16) | OPEN (group)
```

---

## Jun/Jul-2026 — Verification campaign (operator-decisions + phase2-summary)

### Operator decisions (follow-up work; not applied mid-campaign per rules)
```
DEF-068 | Jul-2026 | D1             | Important| correctness/safety | capacity_planner       | Hive-member DEAD_CAPITAL defibrillation/FEE_REDUCE fires before member-skip; fail-open is_hive_member. Ruling: remove path. Fired LIVE (3 defibs nexus-02 + 13 FEE_REDUCE) | OPEN (BEHAVIORAL-HOLD candidate)
DEF-069 | Jul-2026 | D2             | Important| financial-correct  | profitability_analyzer | UNDERWATER→BREAK_EVEN reclassification hides fleet-channel losses. Ruling: remove. Fired LIVE (chan 940304x912x0, roi -19.49%) | OPEN (BEHAVIORAL-HOLD candidate)
DEF-070 | Jul-2026 | D4 (RB-I10)    | Important| econ               | rebalancer             | Defib fee cap hardcoded 100 sats → 0/22 shocks succeeded. Ruling: raise to 400. New option diagnostic_rebalance_max_fee_sats + TestDiagnosticFeeCap | LIKELY-FIXED (implemented per operator-decisions.md; confirm commit + test)
```

### Phase 2 confirmed violations (code fixes needed)
```
DEF-071 | Jul-2026 | PM-I2          | Important| data-integrity     | policy_manager         | set_policies_batch persists STATIC policy w/ no fee target → silent dynamic fallback; ENTRENCHED by test asserting success for invalid input (fix must change test) | OPEN
DEF-072 | Jul-2026 | CB-4           | High     | fail-open/safety   | capex_budget           | capex_budget.py:665-677 returns empty dicts on any DB exception → re-grants full budgets fleet-wide; raising path untested | OPEN
DEF-073 | Jul-2026 | FC-I6          | Important| algorithm/econ     | fee_controller         | "Bounded hive authority ±10%" claim FALSE: exploration multiplier [0.75,2.0] DTS draw-noise + fleet fee prior seeding are influence channels outside clamp. Either bound or amend contract | OPEN (immune-caps cross-ref DEF-092)
DEF-074 | Jul-2026 | PM-I13         | Important| robustness         | policy_manager         | One corrupt expires_at TEXT row raises in _load_cache → get_policy broken for ALL peers (no per-row isolation) | OPEN
DEF-075 | Jul-2026 | FC-I16         | Important| concurrency/data   | fee_controller         | Gossip-refresh no-nudge/RPC-failure paths return before observation-cursor reset → consumed window re-ingested into DTS posterior (main-broadcast path resets correctly) | OPEN
DEF-076 | Jul-2026 | PM-I1          | Medium   | security/validation| policy_manager         | Peer-id regex accepts 66 hex + trailing newline; persists end-to-end | OPEN
DEF-077 | Jul-2026 | RA2-1          | Medium   | data-integrity     | routing (planner_v2/overlay/engine) | ≥10 production skip reasons missing from VALID_SKIP_REASONS; only router reasons test-guarded → log-consumer bucketing broken | OPEN
DEF-078 | Jul-2026 | NX-4           | Minor    | robustness         | rebalance_native_executor_v2 | Malformed-invoice early returns skipped invoice cleanup + failure_class (:422-428) | LIKELY-FIXED-d14256e (both early returns clean up + set failure_class; pinning tests added)
DEF-079 | Jul-2026 | stable_failure_reason | Minor | consistency     | rebalance_executor (legacy) | Legacy vs live failure-reason vocabularies diverge (legacy dead code) | LIKELY-FIXED-9bc0953 (RETIRED — legacy module removed; only rebalance_execution.py vocabulary remains)
```

### Contract drift (contracts to refresh)
```
DEF-080 | Jul-2026 | RE-I3 / 441b8e3| Important| algorithm          | rebalance engine/planner| 441b8e3 flipped two gate boundaries: hold-margin <=→<, beats_do_nothing >→>= (exact-break-even positive-cost pairs now execute); RE-I3 refuted as stated | OPEN (re-adjudicate contract)
DEF-081 | Jul-2026 | FC-I1 / 8630ca6| Important| algorithm/econ     | fee_controller         | Hive-member zero-fee gate overrides even manual enforce_limits fee sets — a 100% override outside FC-I6's hint-multiplier framing | OPEN
DEF-082 | Jul-2026 | 2247370        | Important| data-integrity     | database + main        | spend-ledger coverage_status was hardcoded literal "complete" at both writers (database.py:3971, cl-revenue-ops.py:6590) → false confidence | LIKELY-FIXED-9ad0b59 (both writers now measure vs oldest cost-evidence row; emit covered_hours=null/"unknown" when no basis). Note: cl-hive-side ML-*-IDENT defects (runtime.py:2702) remain
```

### Biggest test gaps (regressions could slip silently)
```
DEF-083 | Jul-2026 | TG-db          | High     | test-coverage      | database               | reserve-budget ceiling/rollback (_reserve_budget_atomic), spend-event replay dedup, amount/fee sanitizers — all phantom-cited, zero real coverage | OPEN (Phase 3/4 mutation target)
DEF-084 | Jul-2026 | TG-routing     | Medium   | test-coverage      | hive_router/routing    | Mutation-proven survivable: HR-1 availability gate, HR-4 ownership split, R3-6 cheapest-selection (min→max survives 80 tests), R2-5 negative clamp | OPEN (tautological coverage)
DEF-085 | Jul-2026 | TG-rebal       | Medium   | test-coverage      | rebalancer             | RB-I2 fail-open/fail-closed asymmetry; hot-channel budget cap (deleting it fails no test); rejected-hive-intent blocking | OPEN
DEF-086 | Jul-2026 | TG-feestack    | Medium   | test-coverage      | fee/profitability/planner| FC-I3/FC-I13 (code-only), PA-I10 stampede lock, CP fail-open member guard, CP-I14 recommended/delegated cooldown leg, RE-I10 p_success boundaries | OPEN
```

### Unresolved tension / dead code
```
DEF-087 | Jul-2026 | budget-tension | High     | financial-correct  | database/rebalancer/budget | 129 snapshots with actual 24h spend > effective daily budget on total-cost-budget surface, while sweep_rebalancer RB-I1c passed 1227/1227 on rebalance-category spend ≤ budget. Reconcile (fields/categories, hot-channel raises, or real enforcement hole) | OPEN (Phase 3/4)
DEF-088 | Jul-2026 | demand_flow    | Low      | dead-code          | demand_flow            | classify_candidate + keyword scoring production-dead yet carries 12/23 module tests; fee_extractive signal dead within dead code | OPEN
```

---

## Doc contradictions (stale-doc inventory — Phase 6)

```
DEF-089 | inventory | fee_interval   | Low      | docs-conformance   | config/cl-revenue-ops.conf.full | fee_interval drift: conf.full internally inconsistent — 1800 (lines 37,166) vs 3600 (line 175). CODE GROUND TRUTH = 1800 (option default cl-revenue-ops.py:532, config.py:335). Historical 600 was fixed (S6 D-1) | CONTRADICTION
DEF-090 | inventory | license        | Low      | docs-conformance   | LICENSE / README / all docs | MIT-vs-BSD: LICENSE=BSD 3-Clause, README=BSD — currently consistent, but sweep all docs/headers for stray MIT references. LICENSE is truth | CONTRADICTION (verify no stray MIT)
DEF-091 | inventory | stale-algo-ref | Low      | docs-conformance   | README.md / AGENTS.md  | Stale HillClimbing / Sling / CLBoss references remain in README + AGENTS.md; controller is DTS+PID-only (73a7b09), sling superseded by native rebalance. Annotate historical docs "superseded" | CONTRADICTION
DEF-092 | inventory | immune-caps    | Medium   | docs-conformance   | IMMUNE_INFLUENCE_LEVEL2C_AUDIT.md vs fee_controller | Immune-influence caps ARE numerically pinned in doc (fee [0.95,1.05], rebalance [0.85,1.15], planner [0.85,1.10], closure [0.85,1.15]) — but FC-I6 (DEF-073) shows fee-influence channels OUTSIDE the clamp. Verify code binds the documented caps | CONTRADICTION (cross-ref DEF-073)
DEF-093 | inventory | stale-doc-exec | Low      | docs-conformance   | docs/audit/{contracts,verification}/rebalance_executor.md | Contract + verification docs describe modules/rebalance_executor.py which was DELETED (commit 9bc0953). Mark docs stale | CONTRADICTION
DEF-094 | inventory | stale-doc-mem  | Low      | docs-conformance   | docs/audit/{contracts,verification}/rebalance_memory.md  | Contract + verification docs describe modules/rebalance_memory.py which was DELETED (commit 9bc0953). Mark docs stale | CONTRADICTION
```

---

## Closed-in-era (no carry-forward, recorded for completeness)

- **2026-03-27 full-plugin audit** — 4 findings, all Resolved + re-verified in-session:
  planner unified-budget fail-open (now fails closed, capacity_planner.py:1105); unified-budget
  double-count of planner opens (excludes canonical channel_open/close, cl-revenue-ops.py:5016);
  planner close-protection fail-open (returns False,"Policy unavailable", capacity_planner.py:1525);
  cl-hive stubbed hive-rebalance-recommendations RPC removed. Follow-up was optional Codex-config cleanup only.
- **2026-05-19 standalone independence audit** — all findings positive/neutral (standalone invariant
  holds; neutral hint lookups on adapter/datastore absence). Only note: older builds may expose
  revenue-hive-hints-status without `diagnostics_version` — production parity check, not a code defect.
- **Session-era FIXED (not deferred):** S3 C-1(budget atomicity) FIXED; S4 config C-1/C-2 FIXED;
  S6 D-1 fee_interval config.py 600→1800 FIXED; all "Fixes Applied" tables in each session doc.
```
