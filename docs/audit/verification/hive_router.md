# Verification: modules/hive_router.py

Phase 2 — Tier 2. Verified 2026-07-01 at HEAD 6740115 against
`docs/audit/contracts/hive_router.md`.

Test run: `.venv/bin/python -m pytest tests/test_hive_router.py -q` →
32 passed. Corpus: layer contents and discover_route results are not hermes
artifacts (contract agrees); the sweep found nothing attributable to this
module, so all verdicts here are code+test.

## Purpose-section claims

**Verified with line drift.** Instantiated once at `cl-revenue-ops.py:2109`
(contract: :2098-2110 window — still accurate). The engine's
membership-only duck-typing has moved: it is now
`modules/rebalance_engine_v2.py:140-145` (`_membership_router` = object with
`is_hive_member`; `_hive_router` requires `price_pair`, which `HiveRouter`
lacks) — contract cites :103-107 (drift, same semantics). Live consumers
confirmed: `modules/hive_runtime.py:34-52` (refresh_layer /
refresh_fleet_balances / clear_route_cache), `modules/rebalancer.py:1639`
(discover_route for inbound fee estimation), `cl-revenue-ops.py:8072` (Boltz
loop-out first hop; `BOLTZ HIVE ROUTE` log now at :8076, contract :8024 —
drift).

## Invariants

- **HR-1 (discover_route None unless available; per-peer 60 s cache incl.
  None failures)** — **availability gate verified; cache verified
  (code-only) with an accuracy nuance.** Gate: modules/hive_router.py:276-277,
  pitted by `tests/test_hive_router.py::TestHiveRouterDiscover::`
  `test_discover_returns_none_when_unavailable`. Cache: 60 s window
  :282-287, success cached :381, exception-failure None cached :386,
  `clear_route_cache` :393-396. **Nuance:** the "including None failures"
  claim holds only for the *exception* path — empty-`routes` (:332-334),
  empty-`path` (:337-339), and missing-our-id (:289-291) returns are NOT
  cached, so those repeat the getroutes call within a cycle. No test covers
  any caching behavior.
- **HR-2 (never auto.sourcefree; layer list from live listlayers; failed
  getroutes retried once with auto-only layers)** — **first two clauses
  verified; retry clause verified (code-only).** Code: :296-330 (layer list
  from listlayers :301-311; retry :325-330). Test:
  `::test_discover_avoids_auto_sourcefree_crash_vector` asserts
  `auto.sourcefree` absent, `auto.localchans`/`auto.no_mpp_support` present,
  `maxparts == 1`, `final_cltv == 18`. **No test exercises the
  retry-with-auto-only path.** Note the retry fires on *any* getroutes
  exception, not only unknown-layer — broader than the contract's TOCTOU
  rationale but strictly safer.
- **HR-3 (discovery fee capped at 1%)** — **verified (code-only).**
  `max_fee_msat = amount_msat // 100` at :294. No test asserts the
  `maxfee_msat` kwarg.
- **HR-4 (layer ownership defers to cl-hive; standalone creation only when
  absent; 0-fee both directions + node bias +5)** — **ownership split
  verified; parameter details verified (code-only).** Code: `refresh_layer`
  :119-148 (managed detect :134-143), `_create_standalone_layer` :165-258
  (0-fee both dirs :197-217, bias +5 in/out :219-240). Tests:
  `TestHiveRouterLayerDetection::test_detects_cl_hive_managed_layer` and
  `::test_falls_back_to_standalone_when_no_cl_hive`, plus
  `TestHiveRouterRefresh::test_refresh_skips_remove_when_layer_missing` /
  `::test_recreate_layer_reuses_existing_layer_without_remove`
  (non-destructive create). **No test asserts the 0-fee/cltv-6
  askrene-update-channel params or the +5 bias value** — those exact numbers
  are code-verified only.
- **HR-5 (max_rebalance_through_member = min(25% capacity, liquidity above
  40% floor); 0 when unknown)** — **verified (code-only).** :435-461 matches
  the formula exactly. Zero genuine tests (the only test double lives in the
  dead `tests/test_rebalance_executor.py` mock), and zero live callers — a
  dead surface (see Anomalies).
- **HR-6 (reserve/unreserve submit only normalized entries, return False on
  error; reserve_for_job accepts only directions 0/1)** — **verified.** Code:
  `_normalize_path` :582-594, `reserve_path`/`unreserve_path` :702-732,
  `_coerce_graph_direction` :668-678. Tests (all genuine):
  `TestReservations::test_reserve_path_normalizes_to_exact_hops`,
  `::test_unreserve_path_normalizes_to_exact_hops`,
  `::test_reserve_uses_explicit_graph_direction`,
  `::test_reserve_rejects_ambiguous_pull_push_direction` (and unreserve
  twins), `::test_reserve_returns_false_on_error`,
  `::test_unreserve_returns_false_on_error`. Run: pass. Note these are also
  dead surface (no live callers).
- **HR-7 (revenue-local biases exactly {profitable:+3, break_even:0,
  underwater:-3, stagnant_candidate:-5, zombie:-8}; zero-bias skipped)** —
  **verified.** Code: `PROFITABILITY_BIAS` :596-602, skip-zero :634-636,
  both directions :638-656. Tests:
  `TestLocalLayer::test_profitable_channel_gets_positive_bias` (bias == 3,
  exactly 2 calls = both directions),
  `::test_underwater_channel_gets_negative_bias`,
  `::test_zombie_channel_gets_strong_negative_bias`,
  `::test_break_even_channel_gets_no_bias` (0 bias calls),
  `::test_no_profitability_analyzer_returns_false`. `stagnant_candidate`
  (-5) is the one class without its own test. Run: pass.

## Gaps

1. HR-1 route-cache behavior (60 s TTL, None-failure caching,
   clear_route_cache interaction) is completely untested.
2. HR-2 auto-only retry path untested; HR-3 1% cap unasserted; HR-4 standalone
   layer parameters (0 fee, cltv 6, bias +5) unasserted.
3. HR-5 and HR-6 protect dead surface only — tests exist for HR-6 but the
   invariants have no production consumer to protect.
4. Not corpus-observable at all: no hermes artifact captures
   askrene-listlayers, layer biases, or discover_route outputs; even the
   indirect surfaces (INBOUND FEE EST log lines, BOLTZ HIVE ROUTE) are logs,
   which hermes does not collect. Corpus verdict for every HR invariant:
   untestable-with-current-data.

## Anomalies

1. **Dead surface confirmed by grep at HEAD** (contract Uncertainty #1
   upheld): `reserve_path`, `unreserve_path`, `reserve_for_job`,
   `unreserve_for_job`, `max_rebalance_through_member`,
   `suggest_fleet_rebalance_chunks`, `get_fleet_member_balance`, and
   `fleet_member_can_route` have zero callers in modules/ or
   cl-revenue-ops.py outside this module itself; only the dead
   rebalance_executor and its tests reference them.
2. HR-1 cache-miss nuance (empty-routes results not cached) — the contract's
   "caches ... (including None failures)" is over-broad; only
   exception-failures are negatively cached.
3. `score_channel_for_hive` non-fleet branch still the acknowledged
   always-1.0 stub (:573-576); `discover_route` still hardcodes
   `final_cltv=18` (:322) — both contract Uncertainties confirmed present,
   unchanged.
4. Line drift vs contract: engine duck-typing moved to
   rebalance_engine_v2.py:140-145; Boltz hive-route usage moved to
   cl-revenue-ops.py:8072-8076. No semantic drift found.
