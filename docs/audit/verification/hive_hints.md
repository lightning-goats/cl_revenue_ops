# Phase 2 Verification — hive_hints.py

Contract: docs/audit/contracts/hive_hints.md (HH-I1..HH-I12).
Evidence: test mapping (subagent, spot-checked), code confirmation on current HEAD
(module unchanged since contract commit f905cfd — `git diff f905cfd..HEAD` empty, so
all contract line citations remain valid), corpus sweep
`tools/audit/sweep_planner_boltz_hints.py` over 2,595 (hive-nexus-01) + 2,598
(hive-nexus-02) revenue-hive-hints-status snapshots plus offline replay of the
adapter's bias getters over 123 producer payloads per node (stride 5).
CORRECTED (refutation pass): snapshot coverage is 2026-06-08 → 2026-06-20 plus a
single terminal snapshot on 2026-07-01 — NOT "2026-05-19 → 2026-07-01";
2026-06-21..06-30 has zero snapshots. All cited test files pass on HEAD: 179 passed
(test_hive_hints.py, test_hint_hardening.py, test_hive_hints_finite_hardening.py,
test_metabolic_influence_hints.py, test_immune_influence_hints.py,
test_metabolic_level2c_integration.py, plus companions
test_hive_hint_freshness_rpc_diagnostics.py and
test_hive_hints_diagnostics_regression.py — the exact set reproducing 179).

Production posture (from the sweep): hints were **fresh in 5,193/5,193 snapshots**
on both nodes; stale_fallback_active in 0; m2_scope always `legacy_seed_only`
(legacy producer, M2 scoping never engaged); source datastore 2,451+2,572 vs
hive_export_rpc 144+26; effective TTL observed at 300s and 900s.

| Invariant | Verdict | Evidence |
|---|---|---|
| HH-I1 TTL: fresh only within [-300s skew, effective TTL]; TTL capped at 86400 | **verified** | test_hint_hardening.py (far-future not fresh/not usable, small skew still fresh, oversized producer ttl capped) + test_hive_hints.py::TestFreshness (fresh/stale/ttl_override/no-snapshot); code confirmed (:184-185, :426-453); **corpus: fresh⇒age∈[-300, ttl] consistent in 2,595+2,598 snapshots, 0 violations**. Minor gap: ttl_override > 86400 (override clamped to ceiling, :427-428) untested |
| HH-I2 stale fallback serves only fee_bias/rebalance_bias, window max(6h, min(24×TTL, 48h)), re-checked per read | **verified** | test_hive_hints.py (49h>48h neutral, 5999s within 6h-min window still serves, ttl=86400 stops at 48h not 24 days, bounded_bias allows only the two fields); code confirmed (:60-73, :459-479 window arithmetic, :617-633 per-read recency re-check); **corpus vacuous: stale fallback never activated (0/5,193)**. One adjacent test (::test_fallback_max_seconds_constant_is_48h) is tautological alone but carried by its behavioral neighbors |
| HH-I3 bias bounds: fee ∈[0.9,1.1], rebalance ∈[0.85,1.15], corridor-util ∈[0.9,1.1] | **verified** | test_hive_hints.py::TestSafetyRails adversarial payload sweeps for fee and rebalance bias (+ hard-cap tests); code confirmed (:21-23 caps, clamp-then-1.0+bias construction :882-904, :910-938, :944-972 — output bounded by construction regardless of payload); **corpus: replayed getters over 123+123 real producer payloads, all outputs in bounds, 0 violations**. Gap: corridor-utilization bias lacks an adversarial cap sweep (only direction/neutral/NaN tests) |
| HH-I4 neutral on absence (1.0 / {} / [] / 50 / 0.5 / 0.0) | **verified** | pervasive unknown-peer/stale/no-snapshot neutral tests in every getter class + finite-hardening exact-neutral test; code confirmed (`_get_peer_hint` :695-709 returns {} unless allowed and in scope; every getter's fallback branch) |
| HH-I5 non-finite rejection in depth — with the contract's two documented gaps | **verified (as-implemented, gaps confirmed)** | tests: NaN/Inf JSON literals via datastore hex neutralize (parse hook :152-166 confirmed), non-finite generated_at/ttl rejected at validation, non-finite confidence → exact neutral, `_clamp_float` NaN/Inf → default. **Both contract gaps confirmed in code on HEAD and pinned by no test:** (1) the RPC fallback returns pre-parsed data through pyln (:266) with no strict-literal rejection — only the datastore transport gets `_json_loads_strict`; (2) segment validators use bare `float()` with two-sided clamps (:1227/:1246 observation confidence, :1257-1260/:1277-1280 scores) where `max(lo, min(hi, nan))` evaluates to the **upper bound** — NaN confidence/net_utility becomes 1.0 instead of being rejected. Outputs remain bounded (nothing unbounds or crashes), but "NaN never influences a decision" fails for segment intelligence via the RPC transport, exactly as the contract states |
| HH-I6 M2 scope enforced consumer-side; all_hints demoted without operator opt-in | **verified** | test_hive_hints.py (channel_and_fleet / channel_peers / legacy_seed_only scoping, all_hints neutral without enablement, allowed with it, unknown scope → safe default) + metabolic/immune scope tests; code confirmed (:514-524 demotion, `_peer_in_m2_scope` gating in `_get_peer_hint` :704-708); **corpus: M2 never engaged in production (m2_scope=legacy_seed_only in all 5,193 snapshots — legacy producer)** |
| HH-I7 open candidates fresh-only, re-validated per hint | **verified** | test_hive_hints.py (fresh returns candidates, stale → [], no snapshot → [], invalid enum values dropped, partial fields, stale open-hint → {}); code confirmed (:1435-1457 freshness under lock, every result re-validated through get_channel_open_hint :1412-1433 with enumerated vocab + clamped confidence) |
| HH-I8 absolute values rejected, not clamped (fleet fee prior [1,10000]; fleet balance sanity) | **verified** | test_hive_hints_finite_hardening.py::TestFleetFeePriorBounded (absurd rejected-not-clamped, boundary accepted, just-above rejected, non-finite/non-positive/bool rejected) + test_hint_hardening.py fleet-balance tests (negative, available>capacity, absurd capacity rejected; valid passes); code confirmed (:186-193, :1965-1978 returns None outside [1, 10000]; get_fleet_balance full sanity chain incl. 21M-BTC cap and avail≤cap); **corpus: replayed fleet-fee-prior over 123+123 producer payloads, always None or in [1,10000], 0 violations** |
| HH-I9 no authority: additional_permission=False always; deltas capped; confidence ≥ 0.50 | **verified** | metabolic/immune/level2c tests assert additional_permission is False in all states and pin the exact caps (fee ±5%, rebalance ±15%, open −15%/+10%, closure-watch ±15%) and low-confidence neutralization; code confirmed (:26-44 constants, :1939-1959 hard-coded False/authority strings) |
| HH-I10 closure hints neutralized under stale fallback; planner prefers fresh variant | **verified** | test_hive_hints.py (is_closure_recommended False under bounded_bias stale fallback; True under full_legacy contrast; stale → both variants False); code confirmed (closure_recommended ∈ STALE_FALLBACK_NEUTRALIZED_FIELDS :61-73; capacity_planner.py:924-927 prefers is_closure_recommended_fresh) |
| HH-I11 snapshot atomicity under the lock | **verified** | test_hint_hardening.py::test_no_crash_if_snapshot_cleared_after_freshness_check (simulated clear-between-check-and-read race); code confirmed (all getters snapshot under `self._lock`; `_store_snapshot`/`_clear_snapshot` :275-280/:350-355 atomic; get_open_candidates captures items under lock :1439-1444). Partial: single simulated race, no true concurrency test — structurally enforced |
| HH-I12 datastore-first; live RPC only when datastore missing/invalid/stale | **verified** | test_hive_hints.py (fresh datastore ⇒ **rpc.call asserted not called**; fallback on stale/invalid datastore; recent-stale kept when RPC refresh fails; ancient-stale ignored); code confirmed (poll :293-348: RPC path entered only when `candidate is None or stale_candidate is not None`); **corpus: source=datastore in 2,451/2,595 (n1) and 2,572/2,598 (n2) snapshots, RPC fallback 144/26 — datastore-first visibly operating** |

## Gaps

- **HH-I5 is the only invariant with material untested surface, and both documented
  gaps are unpinned**: no test sends NaN/Inf JSON literals through the RPC
  transport (existing RPC tests inject already-parsed float("nan") objects), and
  zero tests touch the segment-score/segment-observation validators with NaN — the
  clamp-to-upper-bound behavior is neither pinned as known-flawed nor would a fix
  or regression be caught. Exploitability still hinges on the contract's open
  question of whether lightningd's C JSON parser would ever deliver such literals.
- HH-I3: get_corridor_utilization_bias has no adversarial hard-cap sweep analogous
  to TestSafetyRails for the other two biases.
- HH-I1: operator ttl_override values above the 86400 ceiling are untested.
- HH-I11: atomicity has one simulated race only.
- Corpus cannot observe: HH-I2 (stale fallback never fired), HH-I6 M2 scoping
  (legacy producer only — the M2 half of the adapter has zero production
  exposure), HH-I5 (no poisoned payload appeared), HH-I9/I10 internals.

## Anomalies

1. **The stale-fallback machinery never ran in the observed corpus** (0 activations
   in 5,193 snapshots; hints fresh in every observed snapshot — note the observation
   window is ~12 days plus one terminal snapshot, not a continuous 6-week study). Combined with the contract's
   note that stale_fallback_policy is not wired to any CLN option (production is
   pinned to bounded_bias, and diagnostics_only / full_legacy_fallback are dead
   configuration), a meaningful slice of this module — and of its test suite — is
   exercised only in tests. HH-H1's "not-fresh node-hours" comparison group is
   empty on this corpus.
2. **M2 scoping is production-dead today**: every snapshot reports
   m2_scope=legacy_seed_only, i.e. a legacy producer. The M2 privacy enforcement
   (HH-I6) is verified against tests only; when cl-mycelium starts emitting
   explicit m2_scope, this verification should be re-run against real payloads.
3. TTL flapped between 300s and 900s within the corpus (n1: 2,111 vs 484; n2:
   1,191 vs 1,407) — producer-driven ttl_seconds changes, handled correctly by the
   effective-TTL logic, but worth knowing when interpreting freshness-window
   analyses in Phase 4.
4. The RPC fallback path did fire in production (144 snapshots on n1, 26 on n2),
   so the HH-I5 RPC-transport gap is on a *live* code path, not a theoretical one
   — the datastore-first design reduces but does not eliminate exposure.

## Refutation pass (2026-07-01)

Adversarial re-verification on HEAD (dac9b48; `git diff f905cfd..HEAD` on the module
still empty). All 12 verdicts attacked; **0 refuted, 12 survived**. Method: re-ran the
cited test files (179 reproduced only after identifying the two unnamed "companion"
files — now listed in the header), re-ran the sweep including the offline replay
(123+123 payloads through the real `HiveHintAdapter`, 0 bias/prior violations
reproduced), and re-read every load-bearing code citation.

Findings:

1. **HH-I3 bounds hold by construction and by genuine adversarial pitting.**
   TestSafetyRails sweeps role × competition × confidence (fee) and preference ×
   quality × confidence (rebalance) grids including out-of-range values (±50, 100.0)
   and asserts the closed bounds — not mock echoes. The documented corridor-bias gap
   (no analogous sweep) is accurate. The sweep replay is a real-code replay:
   `_validate_and_normalize_snapshot` + `_store_snapshot` + the actual getters, with
   only freshness pinned — a getter exception would have crashed the sweep, so the
   0-violation result is load-bearing.
2. **HH-I5's two documented gaps re-confirmed in code**: the RPC transport (:266)
   returns pyln-pre-parsed data with no `_json_loads_strict`; segment validators use
   bare `float()` and `max(lo, min(hi, x))` where NaN propagates to the upper bound
   (Python `min(1.0, nan)` returns 1.0). "verified (as-implemented, gaps confirmed)"
   is the correct verdict framing; neither gap is pinned by any test, as stated.
3. **Membership source sanity (feeds CP-I5/D1)**: the sweep derives member sets from
   per-peer `member` booleans in hive-export-hints payloads (not a counts-only
   surface); all 613+614 payload snapshots yield a stable 4-member set, and the
   detected member episodes reproduce exactly.
4. **Corpus-window correction (header, Anomaly 1)**: freshness/M2/fallback
   production-vacuity claims are established over ~12 observed days
   (2026-06-08 → 2026-06-20) plus one 2026-07-01 snapshot; the 06-21..06-30 hole
   means "never fired in production" cannot be asserted for that interval.
5. HH-I12's datastore-first test genuinely pits (`rpc.call.assert_not_called()`
   after a fresh hex datastore poll), and HH-I1's ttl_override ceiling clamp
   (:427-428) matches the code; the untested-override gap note is accurate.
