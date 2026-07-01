# Verification: modules/rebalance_hive_router.py

Phase 2 — Tier 2. Verified 2026-07-01 at HEAD 6740115 against
`docs/audit/contracts/rebalance_hive_router.md`.

Test run: `.venv/bin/python -m pytest tests/test_rebalance_hive_router.py -q`
→ 14 passed (plus RHR-relevant cases in tests/test_rebalance_perf_fixes.py,
run green in the perf-fix suite). Corpus sweep
(`tools/audit/sweep_routing_stack.py`, 2026-07-01): 0 violations.

## Purpose-section claims

**Verified with line drift.** Engine constructs it iff askrene probe
succeeded and hive hints exist — now at
`modules/rebalance_engine_v2.py:209-216` (contract: :172-179); pricing call
now at :1967 (contract: :1861-1868). Duck-typing that keeps the plugin-level
`HiveRouter` out of pricing is now :140-145 (`price_pair` attribute check).
Layer set: `ROUTE_LAYER_NAMES` at modules/rebalance_hive_router.py:66-73,
composed from live listlayers only —
`test_hive_router_uses_all_live_hive_and_revenue_layers` and
`test_hive_router_excludes_expansion_test_layers_from_live_routes`
(expansion-test layer present in listlayers but never routed over) genuinely
pit this.

## Invariants

- **RHR-1 (first hop must use selected source SCID else
  fleet_source_mismatch)** — **verified (code-only).** Code: :608-610.
  **No test anywhere produces `fleet_source_mismatch`** (repo-wide grep of
  tests/ excluding the dead executor's suite: zero hits). Corpus: sweep C3
  (route_summary first hop == source_channel_id) has 0 violations over 178
  route-bearing candidates — but all 178 are market_only (v3) routes, so
  this corroborates source pinning generally, not this module's check
  specifically.
- **RHR-2 (HIVE_ONLY rejects non-member intermediates; HYBRID requires >=1
  member hop)** — **verified.** Code: `_validate_hive_only_path` :479-482,
  applied :612-616; HYBRID gate :617-618. Tests (genuine):
  `tests/test_rebalance_hive_router.py::`
  `test_hive_router_rejects_non_hive_intermediate_for_hive_only` (member
  set {src,dst}, non-member mid hop → `non_hive_intermediate`) and
  `::test_hive_router_returns_no_fleet_route_when_hybrid_path_has_no_hive_hops`.
  Run: pass. **Nuance:** the HIVE_ONLY check iterates *every* hop's
  `next_node_id`, including the final hop to the dest peer — stricter than
  the contract's "every intermediate"; a HIVE_ONLY decision toward a
  non-member dest would fail even with an all-member middle. Corpus:
  no hive_only/hybrid candidates in the 2026-06-09→20 window (route_policy
  hist: 178/178 market_only; segment obs: 32 market_only, 1 hive_only), so
  corpus-unobserved.
- **RHR-3 (pinned-source layer composition with retry-excludes LAST;
  fallback to legacy merged-exclude layer)** — **verified.** Code: :521-560
  (pinned compose :524-530, excludes appended last :553-560; fallback
  :531-548). Tests: `::test_hive_router_pinned_layer_order_in_getroutes`
  (asserts gossip/hive layers < LOCAL_DISABLE_LAYER < source-enable, exclude
  layer strictly last, and that exactly one enable targets the selected
  source half) and
  `::test_hive_router_falls_back_to_legacy_excludes_when_pinning_fails`.
  Run: pass.
- **RHR-4 (stale rebalance-local-disable rebuilt clean; add-only
  reconciliation)** — **verified.** Code: `_ensure_local_disable_layer`
  :350-373 (create-failure → remove + recreate :359-366; add-only :369-372).
  Tests: `::test_hive_router_recreates_stale_base_layer_from_prior_run`,
  `::test_hive_router_base_layer_reconciles_new_channels`,
  `::test_hive_router_base_layer_survives_across_cycles`. Run: pass.
- **RHR-5 (Unknown-layer error → exactly one retry with refreshed layers;
  identical refreshed set re-raises)** — **retry half verified; re-raise
  half verified (code-only).** Code: :571-595 (`_is_unknown_layer_error`
  :265-267; cache invalidation :580; pinned-layer rebuild :582-589;
  identical-set re-raise :592-593). Test:
  `::test_hive_router_retries_once_when_expansion_layers_rotate` (layer
  rotation: second getroutes call runs without the vanished layer, result
  succeeds). **The identical-refreshed-set re-raise branch has no test.**
  Run: pass.
- **RHR-6 (cycle state thread-local; end_cycle removes every cached
  layer)** — **cycle semantics verified; thread isolation verified
  (code-only).** Code: `threading.local()` :96, cycle dicts :103-131,
  `end_cycle` best-effort removal :133-148. Tests:
  `::test_hive_router_reuses_pinned_layers_within_cycle` and
  `::test_hive_router_without_cycle_tears_down_enable_layer_each_call`
  pit the in-cycle reuse and out-of-cycle teardown; no test runs two
  threads, so cross-thread isolation rests on `threading.local`.
- **RHR-7 (discovery maxfee = max(1% required, pair budget); reported cost =
  first hop − delivery; gating downstream)** — **verified (code-only)** for
  the maxfee formula (:507 — no test asserts the `maxfee_msat` kwarg);
  **verified** for cost accounting: :638-639, pitted end-to-end by
  `::test_hive_router_reprices_prefix_amounts_from_live_forwarding_policies`
  (exact repriced amounts asserted, via the shared v2 helper). Corpus: sweep
  C5 (implied vs reported cost) 0 violations; downstream budget gate visible
  as `native_route_over_budget` (9) in revenue-status error tokens and S1
  (fee <= max on success) 0 violations.
- **RHR-8 (unique throwaway layer names; half-built layers removed before
  error propagation)** — **verified (code-only).** Code: `_exclude_counter`
  :51-55, `_layer_name` :336-337 (itertools.count + timestamp);
  cleanup-on-failure `_build_enable_layer` :385-390 and
  `_build_exclude_layer` :441-446. No test drives a mid-build failure.

## Additional verified behavior (not in contract invariants)

- `GETROUTES_TIMEOUT_SEC = 30` actually reaches the RPC:
  `::test_hive_router_get_routes_passes_timeout` (:214-223).
- Broadcast-cache return-hop policy path:
  `tests/test_rebalance_perf_fixes.py::test_hive_return_hop_policy_uses_broadcast_cache`
  (fee 1000 base + 250 ppm arithmetic asserted) and
  `::test_hive_return_hop_policy_falls_back_to_per_peer_rpc_when_absent`.

## Gaps

1. RHR-1 (`fleet_source_mismatch`) and the `fleet_invalid_amount` guard
   (:500-501) have zero test coverage.
2. RHR-7 maxfee formula unpinned by tests (same gap as R3-8 / HR-3 —
   a systematic blind spot across the routing stack: no test in the repo
   asserts any router's `maxfee_msat`).
3. RHR-5 identical-set re-raise, RHR-8 half-built cleanup: code-only.
4. Corpus contains no HIVE_ONLY/HYBRID priced candidates in the observed
   window, so policy enforcement is corpus-untestable with current data;
   exactly 1 `no_fleet_route` error message in 38 deduped
   `recent_rebalances` entries is the sole corpus trace of this module.

## Anomalies

1. **Contract Uncertainty confirmed:** `_return_hop_policy` (:280-319)
   treats `fee_ppm == 0 and base == 0` as unknown and falls back to gossip
   (:302-315) — a genuinely zero-fee fleet dest takes the gossip path every
   time. Additionally, when *both* lookups fail it silently proceeds with
   `fee=0, cltv_delta=6` defaults rather than failing — unlike v2/v3's
   R2-1 "fail, don't guess" final-hop rule. A hive route to a peer whose
   policy is unreadable is priced assuming a zero-fee, cltv-6 return hop.
   This asymmetry with R2-1 is worth an explicit contract note.
2. HIVE_ONLY validates the dest hop too (stricter than documented) — see
   RHR-2 nuance.
3. Line drift vs contract in the engine (construction :209-216, call :1967);
   module-internal citations all accurate at HEAD.
