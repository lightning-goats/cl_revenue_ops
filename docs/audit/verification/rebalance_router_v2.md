# Verification: modules/rebalance_router_v2.py

Phase 2 — Tier 2. Verified 2026-07-01 at HEAD 6740115 against
`docs/audit/contracts/rebalance_router_v2.md`.

Test run: `.venv/bin/python -m pytest tests/test_rebalance_router_v2.py -q` →
13 passed. Corpus sweep: `tools/audit/sweep_routing_stack.py` (validated, see
verification/rebalance_hive_router.md appendix note), full-corpus run
2026-07-01: 0 violations.

## Retirement claim (Purpose section)

**Verified.** `modules/config.py:312` — `'rebalance_router': ('v3',)` in
`STRING_ENUM_VALID_VALUES`; raise "rebalance_router only supports 'v3';
legacy 'v2' routing was removed" at modules/config.py:637-642.
`modules/rebalance_engine_v2.py:35` imports only `RouteResult`. Live imports
of the class are exactly the helper-mode embeddings:
`modules/rebalance_router_v3.py:22` and
`modules/rebalance_hive_router.py:18`. `price_pair` has no live caller.
Corpus corroboration: all 33 deduped segment observations carry
`router_kind: "v3"` (sweep O1, 0 violations); all 178 debug candidates
carry `route_policy` set by the v3/hive stack.

## Invariants

- **R2-1 (never assume 0-ppm final hop; fail instead of guess)** —
  **verified.** Code: `_get_final_hop_policy` returns None when both
  listpeerchannels (:137-158) and the listchannels fallback (:160-179) yield
  nothing; `price_pair` fails at :410-415 with "cannot determine final-hop
  fee". Test: `tests/test_rebalance_router_v2.py::TestFailureOnNoRoute::`
  `test_router_returns_failure_when_fee_unknown` (empty RPC world → failure,
  exact error asserted). Run: pass.
- **R2-2 (final-hop policy read for the specific dest channel)** —
  **violated (narrow, code-demonstrated).** The priority-1 path honors
  `dest_channel_id` via `_channel_matches_scid` (:81-95, applied :142) and is
  genuinely pitted by `tests/test_low_severity_fixes.py::`
  `test_final_hop_policy_matches_requested_channel` (parallel channels: asks
  for 200x1x0, gets its 50 ppm, not the first channel's 1000 ppm; run: pass).
  **But the priority-2 listchannels fallback (:160-177) does not filter by
  SCID at all** — it returns the first `dest_peer -> us` channel with a
  fee_ppm. Reproduced 2026-07-01: with two parallel channels where the
  requested 200x2x0 (10 ppm) lacks `updates.remote`, the fallback returned
  the *other* channel's 5000 ppm. So the contract's "parallel channels ...
  cannot mis-price the route" holds only when listpeerchannels carries
  `updates.remote` for the requested channel. Same narrow issue applies to
  `_get_dest_channel_cltv`'s default (returns 40 when no match, :327-346),
  which is at least conservative.
- **R2-3 (middle amounts recomputed backwards from live policy; router
  amounts advisory)** — **verified.** Code: `_reprice_middle_route_amounts`
  :268-302 (backwards loop :286-300; keep-router-amount only on policy-lookup
  failure via `continue` at :290-295); applied in `price_pair` :461-464.
  Tests: `TestFullRoute::test_first_hop_amount_includes_fee_for_first_middle_edge`
  pits the repriced arithmetic end-to-end (exact 50_024_002 msat asserted);
  the multi-hop backward recursion is pitted through the hive router's use of
  the same helper (`tests/test_rebalance_hive_router.py:127`,
  `test_hive_router_reprices_prefix_amounts_from_live_forwarding_policies`).
  The keep-advisory-on-policy-failure branch has no direct test. Run: pass.
- **R2-4 (first hop adds source peer's fee + CLTV for first middle edge;
  direct pair skips getroute)** — **first half verified, second half verified
  (code-only).** Code: :473-509 (fee at :489-495, delay at :496-498); direct
  pair guard `if source_peer_id != dest_peer_id:` at :432 with the
  else-branch :499-501. Tests:
  `test_first_hop_amount_includes_fee_for_first_middle_edge` and
  `test_first_hop_delay_includes_cltv_delta_for_first_middle_edge` (delay
  58+12=70 asserted) genuinely pit the first half. **No test prices a direct
  pair** (`source_peer == dest_peer`), so the skip-getroute branch is
  code-verified only. Run: pass.
- **R2-5 (route_cost_sats = max(0, ceil(...)), never negative)** —
  **verified.** Code: :521-525 and `_route_fee_sats` :203-217 (both
  `max(0, ...)`). Tests: `TestZeroFeePeer::test_zero_ppm_peer_costs_zero`
  (cost exactly 0, not clamped to 1) and
  `test_router_includes_final_hop_base_fee_in_middle_amount` (cost 1 from
  base fee). Corpus: sweep C1 (route_cost_sats >= 0) and C5 (implied cost ==
  reported ±1) over 178 route-bearing candidates: 0 violations — the cost
  formula holds on every corpus route the v2 arithmetic priced (via v3).
- **R2-6 (invoice final CLTV from listconfigs, cached, default 18; final hop
  uses it)** — **verified.** Code: `_get_invoice_final_cltv` :348-373
  (process-lifetime cache :357, default 18 at :372); final hop delay :517.
  Tests: `TestFullRoute::test_route_uses_invoice_cltv_and_explicit_directions`
  (getroute `cltv == 58` = dest 40 + invoice 18; final hop delay == 18) and
  `TestDataServiceRouting::test_router_prefers_data_service_for_reads_and_route_lookup`
  (`get_configs.assert_called_once_with()` — pits the cache within one
  price_pair). The default-18-on-error branch is untested. Run: pass.

## Corpus notes

The corpus (2026-06-09 → 2026-06-20 only; see Anomalies) contains 1225
`revenue-rebalance-debug.json` snapshots with 178 route-bearing candidates,
all priced by v3-over-v2-helpers: sweep checks C2 (monotone non-increasing
hop amounts), C4 (last hop == delivery amount), C5 (cost formula) all pass
with 0 violations — consistent with R2-3/R2-5 in helper mode. No corpus
artifact can isolate v2's own `price_pair` (retired), as the contract states.

## Gaps

1. R2-2 fallback branch: no test covers listchannels fallback with parallel
   channels (the existing fallback test uses a single channel), which is
   exactly where the mis-pricing hides.
2. R2-4 direct-pair branch (`source_peer == dest_peer`, :499-501) untested.
3. `_reprice_middle_route_amounts` keep-advisory-amount branch (:290-295)
   untested; its `(None, error)` failure contract (checked at :465-466) is
   unreachable in the current implementation — the function never returns
   None, making the price_pair error path at :466 dead.
4. `_get_invoice_final_cltv` default-on-exception branch untested.

## Anomalies

1. **R2-2 fallback mis-pricing** (see above) — the one concrete deviation
   from the contract text. Practical exposure is low (listpeerchannels
   normally carries `updates.remote`; the regression comment at
   tests/test_rebalance_router_v2.py:176-179 records that the fallback
   *has* fired in production when a pyln kwarg bug broke priority 1).
2. `_get_final_hop_fee_ppm` (:183-189) is a `@staticmethod` taking `self` —
   broken for instance calls, and grep shows **zero callers anywhere**
   (modules or tests). Contract flagged it as suspect; it is confirmed dead
   and should be a deletion candidate.
3. Corpus span is 2026-06-09 → 2026-06-20 for both nodes — narrower than the
   campaign brief's "2026-05-19 → present", and collection apparently stopped
   2026-06-20. All corpus-based statements in this campaign cover ~12 days.
