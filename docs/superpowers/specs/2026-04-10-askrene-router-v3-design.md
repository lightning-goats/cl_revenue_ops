# Rebalance Router V3 — Askrene + Layers + xpay Design

**Date:** 2026-04-10
**Status:** Approved (design only; research phase not started)
**Worktree:** `.worktrees/askrene-router-v3-20260410`
**Branch:** `feature/askrene-router-v3`
**Upstream issue that motivated this work:** https://github.com/ElementsProject/lightning/issues/9032#issuecomment-4196841896 (Lagrang3 suggests a rebalancing plugin built on `getroutes` and `xpay`, with clever layer usage for circular routes)

## Goal

Upgrade cl-revenue-ops's rebalance route discovery from CLN's legacy `getroute` to askrene's `getroutes` with explicit layer support, and (pending research) replace the v2 executor's hand-rolled `invoice` / `sendpay` / `waitsendpay` / retry loop with `xpay`.

The engine, planner, state, audit, types, and dry-run semantics of v2 remain unchanged. Operators get:

- better route selection (layer-aware, fleet-biased, bad-peer-avoiding when cl-hive is present)
- more robust payment execution (xpay's built-in MPP + retry, if research approves)
- standalone-first behavior: works without cl-hive on any CLN 24.11+ node
- a runtime switch (`lightning-cli setconfig rebalance-router v3|v2`) for production A/B testing without plugin restart

## Non-Goals

1. Redesigning the planner, state snapshot, or eligibility model. The v2 design's contract — "for every valuable imbalanced channel, did we rebalance it, and if not, exactly why not?" — is preserved verbatim.
2. Requiring cl-hive. cl-revenue-ops must run standalone with askrene alone. Fleet layers are additive enrichment, not dependencies.
3. Requiring CLN ≥ 24.11. Nodes on older CLN keep the v2 router via a runtime-detected fallback. No one is locked out.
4. Introducing a new standalone rebalancing plugin in this worktree. The door is left open (by isolating v3 in its own file), but extraction is a future task.
5. Removing or modifying `rebalance_router_v2.py`. It becomes the fallback path and stays untouched.
6. Changing the planner's interface to the router. `price_pair(...) -> RouteResult` stays; v3 implements the same interface.

## Architecture

### Module layout

```
modules/
├── rebalance_router_v2.py      # unchanged — fallback path (getroute-based)
├── rebalance_router_v3.py      # NEW — askrene getroutes + layers
├── rebalance_executor_v2.py    # unchanged in phase 1; may be replaced in phase 2
├── rebalance_executor_v3.py    # NEW (phase 2, after xpay research)
├── rebalance_engine_v2.py      # +router factory + runtime dispatch (~40 lines)
└── config.py                   # +4 config keys
```

### Config keys (all opt-in; defaults are safe for standalone)

| Key | Default | Purpose |
|---|---|---|
| `rebalance-router` | `"v2"` | `"v2"` or `"v3"`. Opt-in because v3 is new. Runtime-switchable via `setconfig`. |
| `askrene-layers` | `"hive-fleet"` | CSV of layer names to pass to `getroutes`. Missing layers are silently ignored by askrene. |
| `rebalance-executor` | `"v2"` | `"v2"` or `"v3"` (xpay). In phase 1 the config validator rejects `"v3"` unconditionally. In phase 2 the validator is relaxed to accept `"v3"` iff `_probe_xpay(plugin)` returns True. |

### Runtime router switch (dispatch per cycle, not per call)

The engine initializes **both** routers at init when askrene is available, and dispatches per cycle based on live config:

```python
class RebalanceEngineV2:
    def __init__(self, ...):
        ...
        self.router_v2 = RebalanceRouter(plugin, our_node_id)
        self.router_v3 = None
        if _probe_askrene(plugin):
            layer_names = _parse_layer_names(config.askrene_layers)
            self.router_v3 = RebalanceRouterV3(
                plugin, our_node_id,
                layer_names=layer_names, log=log,
            )

    def _active_router(self):
        want = self.config.rebalance_router
        if want == "v3" and self.router_v3 is not None:
            return self.router_v3
        return self.router_v2

    def run_cycle(self):
        self._cycle_router = self._active_router()  # captured for atomicity
        ...
```

**Per-cycle atomicity:** the router selected at the start of a cycle stays selected for the entire cycle. If config flips mid-cycle, the change takes effect at the next cycle boundary.

**Config validation:** if an operator sets `rebalance-router = "v3"` but askrene is unavailable, the config setter rejects the value with a clear error. The dispatcher never sees an invalid state.

**No silent fallback on v3 runtime errors:** if v3 raises an unexpected exception (not a clean `RouteResult(success=False)`), the engine logs `[router] v3 errored, cycle aborted` and returns cleanly. It does NOT retry the same cycle with v2. Silent fallback would corrupt A/B measurements.

**Audit records carry router version:** every `REBAL_PICK` and `REBAL_SKIP` line gets a `router=v2|v3` field so post-hoc A/B analysis can bucket results cleanly.

### Decision ownership

The engine factory is the only place that decides which router to use. Neither router contains fallback branching internally; each is the unconditional implementation of its strategy. This keeps each file readable and testable in isolation.

## V3 Router Interface & Layer Integration

### Interface contract (identical to v2)

```python
class RebalanceRouterV3:
    def __init__(self, plugin, our_node_id,
                 layer_names: list[str], log: Callable[[str, str], None]):
        ...

    def price_pair(
        self,
        source_channel_id: str,
        dest_channel_id: str,
        source_peer_id: str,
        dest_peer_id: str,
        amount_sats: int,
        exclude: Optional[List[str]] = None,
    ) -> RouteResult: ...
```

`RouteResult` stays exactly as v2 defines it (`success`, `route_cost_sats`, `final_hop_fee_ppm`, `hops`, `route`, `error`). The planner doesn't care which router produced it.

### Pair pinning (reused from v2)

V3 reuses v2's core insight: circular routes are discovered by asking the route engine for `source_peer → dest_peer` only, then prepending our source hop and appending our dest hop manually. This sidesteps askrene's source≠destination constraint without any layer trickery. Lagrang3's "use layers cleverly" suggestion is satisfied by this pair-pinning pattern; layers then become pure middle-path biasing, not circular-route hacks.

### The single real change from v2 to v3

```python
# v2:
self.plugin.rpc.getroute(
    node_id=dest_peer_id,
    amount_msat=route_amount_msat,
    riskfactor=10,
    fromid=source_peer_id,
    exclude=exclude,
)

# v3:
self.plugin.rpc.getroutes(
    source=source_peer_id,
    destination=dest_peer_id,
    amount_msat=route_amount_msat,
    layers=self.layer_names,       # e.g. ["hive-fleet"]
    maxfee_msat=pair_budget_msat,  # NEW: push budget into askrene
    final_cltv=dest_cltv,
    maxdelay=2016,                 # ~2 weeks, sensible ceiling
)
```

`getroutes` returns a list of routes (askrene supports MPP). The router picks the cheapest single-path route that fits the pair budget. Multi-path is deferred to phase 2 (xpay handles it natively).

### Fee and CLTV math (identical to v2)

`_get_final_hop_fee_ppm`, `_compute_final_hop_fee_sats`, `_route_fee_sats`, `_get_source_channel_policy`, `_get_dest_channel_cltv` are imported from `rebalance_router_v2` as module-level helpers. That's the only cross-router dependency, one-way (v3 uses v2's helpers; v2 never touches v3). When the standalone plugin extraction happens later, those helpers move into a shared `rebalance_route_common.py` and both routers import from it.

### Exclude handling

V2 passes `exclude` as a list of SCIDs. V3's **preferred** pattern — to be validated by research — is per-query excluded layers: build a throwaway layer `rebalance-exclude-<cycle>` containing the failed channels, pass it in `layers=[…, "rebalance-exclude-<cycle>"]` for the retry, tear it down via `askrene-remove-layer` at cycle end. If research measures layer create/remove cost >50ms per cycle, fall back to v3-internal exclude translation.

### maxfee_msat as a hard planner budget

V2 prices a route and then checks it against `pair_budget_sats`, throwing away any route that doesn't fit. V3 passes `maxfee_msat = pair_budget_msat` directly to askrene, letting it either return an affordable route or nothing. Net: fewer wasted RPC round-trips, same final behavior, but moves the budget check one layer down. The planner's `route_over_budget` skip reason still fires — it just triggers on askrene returning empty instead of on v2's post-hoc comparison.

### Layer name parsing & standalone safety

`_parse_layer_names("hive-fleet,hive-reputation")` splits on comma, strips whitespace, drops empties. Empty list is legal (passes `layers=[]` → askrene uses its built-in gossip view only — exactly what a standalone operator without cl-hive gets). At init, v3 calls `askrene-listlayers` once, logs which of the requested layers were actually found, and proceeds regardless. Missing layers never crash.

## Executor — Phase 1 vs Phase 2

### Phase 1: v2 executor unchanged

When `rebalance-router = "v3"` and `rebalance-executor = "v2"` (the only allowed combination in phase 1), the engine wires v3 router + v2 executor. The router produces a pre-built `route` list; the existing `RebalanceExecutorV2` takes that route and runs its normal `invoice` / `sendpay` / `waitsendpay` / retry-with-exclude loop against it. No executor code changes.

This is the safe shipping state. Phase 1 can go to production behind `rebalance-router = "v3"` and be meaningfully tested without touching any payment code.

### Phase 2: xpay executor — TBD after research

The design includes a `rebalance_executor_v3.py` slot but does not specify its implementation. The spec is explicit: **"Use xpay for execution; exact integration depth TBD based on xpay API research."**

Research must answer:

1. Can `xpay` accept a precomputed route (preserving source/dest channel pinning), or does it always discover its own?
2. Does `xpay` use askrene layers automatically, or do we need to pass them explicitly?
3. When `xpay` fails a hop, does it retry internally with excludes, or does it return failure and expect the caller to retry? What's the failure taxonomy?
4. Does `xpay` automatically split amounts across multiple paths? Is MPP safe for circular rebalancing? Can we disable MPP for the debugging period?
5. Is `xpay`'s `maxfee` a soft budget or a hard stop? Does it honor `pair_budget_sats` strictly?
6. Do we still generate a local invoice and point `xpay` at it, or does `xpay` want a bolt11 generated differently for self-pays?

Based on findings, phase 2 picks one of:

- **(a) Full xpay takeover**: `rebalance_executor_v3.py` is a ~100-line wrapper that creates an invoice and calls `xpay` with layers. Delete hand-rolled retry logic.
- **(b) xpay with pinned route**: use `xpay`'s route-hint mechanism (if it exists) to force the planner's pair selection. Keep most of the retry control, gain MPP.
- **(c) Keep v2 executor, make v3 executor a thin xpay alternative**: both coexist; operator picks via `rebalance-executor` config.

Phase 2 does not start until the research doc is committed and approved. The spec is explicit that we may decide, after research, that xpay is not ready for our use case and we keep the v2 executor permanently. That is an acceptable outcome and still leaves v3's router as a meaningful deliverable.

## Standalone Operation & Graceful Degradation

### Init-time probes (run once, cached for process lifetime)

1. `_probe_askrene(plugin)` — calls `plugin.rpc.help("getroutes")`. Returns True iff the method exists.
2. `_probe_layers(plugin, requested)` — calls `askrene-listlayers`, intersects with requested, returns the found set. Called once by the v3 router constructor. Missing layers are logged once at `info`, never re-probed.
3. `_probe_xpay(plugin)` (phase 2 only) — same pattern.

### Degradation ladder

| Environment | Router | Layers | Executor | Result |
|---|---|---|---|---|
| CLN 23.05, no askrene, no cl-hive | v2 | n/a | v2 | Current behavior, zero changes |
| CLN 24.11, askrene only, no cl-hive | v3 | `[]` | v2 (phase 1) / TBD (phase 2) | Layer-unbiased askrene routes. Standalone mode. |
| CLN 24.11, askrene + cl-hive | v3 | `["hive-fleet"]` default | v2 (phase 1) / TBD (phase 2) | Full fleet intelligence. |
| CLN 24.11, full layer set opt-in | v3 | `["hive-fleet","hive-reputation","hive-corridors","hive-traffic"]` | v2 (phase 1) / TBD (phase 2) | Maximum layer consumption. |
| CLN 24.11, operator opts out | v2 | n/a | v2 | `rebalance-router = "v2"` forces fallback. |

### Required startup log lines

```
[router] askrene detected: <yes|no>
[router] v3 selected: <yes|no>  reason=<askrene_unavailable|user_opted_out|enabled>
[router-v3] requested layers=[hive-fleet]  found=[hive-fleet]
[router-v3] requested layers=[hive-fleet,hive-reputation]  found=[hive-fleet]  missing=[hive-reputation]
```

Exactly four line patterns. Asking operators for "the first four router log lines from startup" must answer every config question.

### Layer churn

V3 does NOT re-probe layers per cycle. Missing layers at runtime just mean askrene ignores them silently; new layers appear automatically on the next query. No state synchronization needed. cl-revenue-ops treats the layer set as best-effort enrichment, not a contract.

### Anti-requirements

1. No automatic creation or management of askrene layers. That's cl-hive's job. v3 is a pure consumer.
2. No heuristic "upgrade me to v3" nudges. Operator must opt in via config.
3. No per-cycle askrene health checks. One probe at init, trust the result.
4. No layer hot-reload. If operators change `askrene-layers`, they restart the plugin.
5. No telemetry on which layers influenced which routes. Askrene doesn't expose that and we're not going to infer it.

## Research Phase & Reference Sources

### Status gate

No v3 code is written until `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` is committed to the worktree and approved.

### Primary reference source

**CLN upstream at https://github.com/ElementsProject/lightning**. Specific files to read, cite, and attach line-refs in the research doc:

| Topic | CLN source path |
|---|---|
| `askrene-listlayers` schema | `doc/schemas/askrene-listlayers.json` |
| `askrene-create-layer` schema | `doc/schemas/askrene-create-layer.json` |
| `askrene-remove-layer` schema | `doc/schemas/askrene-remove-layer.json` |
| `askrene-update-channel` schema | `doc/schemas/askrene-update-channel.json` |
| `askrene-bias-channel` / `askrene-bias-node` | `doc/schemas/askrene-bias-*.json` |
| `askrene-inform-channel` | `doc/schemas/askrene-inform-channel.json` |
| `getroutes` schema (request + response) | `doc/schemas/getroutes.json` |
| askrene plugin implementation | `plugins/askrene/*.c` (flow.c, reserve.c, layer.c, mcf.c) |
| askrene README | `plugins/askrene/README` or equivalent |
| xpay command docs + schema | `doc/schemas/xpay.json` |
| xpay plugin implementation | `plugins/xpay.c` |
| xpay waitpayment semantics | `plugins/xpay.c` + `doc/schemas/waitpayment*.json` |
| setconfig RPC for runtime switch | `doc/schemas/setconfig.json` |

Each citation in the research doc has the form `ElementsProject/lightning@<sha>:<path>#L<start>-L<end>`. No guessing, no memory, no hallucinated field names. When in doubt, `gh api repos/ElementsProject/lightning/contents/<path>` pulls the actual bytes.

### Research deliverables

One doc, `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md`, with these numbered sections:

1. **`getroutes` contract.** Full request params, response shape, error modes, timeout behavior. Cite CLN source lines.
2. **Layer lifecycle.** How create/remove/update/inform/bias work, who owns the layer, persistence across restarts, multi-plugin concurrency. Confirm that cl-revenue-ops reading cl-hive's layers is safe (no writes, no reservations on shared state).
3. **Layer semantics under pair pinning.** Experimental: with `source=peer_A, destination=peer_B` and `layers=["hive-fleet"]`, does askrene respect the layer's channel constraints for middle hops? Prove it by running `getroutes` with and without a layer containing a known fleet channel and diffing the routes.
4. **Exclude-via-layer pattern.** Measure the cost of creating + removing a throwaway exclude layer per retry. If >50ms, fall back to v3-internal exclude translation.
5. **`xpay` API surface.** Full request/response, route-pinning capability, layer support, retry semantics, failure taxonomy, MPP behavior, maxfee enforcement, self-pay/circular-pay support. Each claim backed by a source citation or a live RPC transcript.
6. **`xpay` vs `sendpay`+`waitsendpay` behavior diff for circular self-pays.** Run both paths against the same pair on a real node (tiny amount, e.g. 1000 sats), capture full RPC transcripts, compare latency / retry count / failure-recovery behavior / audit-log noise.
7. **`setconfig` runtime-switch verification.** Confirm that `lightning-cli setconfig rebalance-router v3` is honored hot without plugin restart. Capture actual behavior — does CLN persist the new value? Does `pyln-client` get notified?
8. **Failure-mode taxonomy.** Enumerate every error code / error string that `getroutes` and `xpay` can return, mapped to v2 router's existing skip reasons (`no_route`, `route_over_budget`, `cannot_determine_fee`, etc.). Gaps become new skip reasons.
9. **Decision records.** For Q4 (xpay integration depth a/b/c) and the exclude-via-layer vs internal-exclude choice, pick one and record the reasoning. The implementation plan starts from this section.

### Research methodology

1. **Prefer upstream source over docs.** CLN's upstream C code and JSON schemas are authoritative. `docs.corelightning.org` lags. Cite source files with commit SHAs.
2. **Every behavioral claim has a receipt.** Either a source citation or a captured RPC transcript from a real node.
3. **Live node for behavioral questions.** The sat's own CLN node is the reference. Research doc includes commands + redacted transcripts.
4. **No implementation peeking.** Don't start `rebalance_router_v3.py` while writing research.
5. **Research doc gets committed and reviewed before the phase 1 plan is written.** Explicit gate.

Research phase estimated scope: 4-8 hours of reading + 1-3 hours of live-node experiments + 1-2 hours of write-up.

## Testing Strategy

### Unit tests — `rebalance_router_v3.py`

Pure logic + mocked RPCs:

| Test | What it proves |
|---|---|
| `test_v3_price_pair_calls_getroutes_with_layers` | Correct RPC method and args |
| `test_v3_price_pair_prepends_source_hop_and_appends_final_hop` | Pair pinning preserved |
| `test_v3_price_pair_returns_route_result_shape_matching_v2` | Router interface contract identical to v2 |
| `test_v3_picks_cheapest_route_when_askrene_returns_multiple` | Single-path phase-1 selection |
| `test_v3_returns_failure_when_getroutes_returns_empty` | Maps to `no_route` skip reason |
| `test_v3_returns_failure_when_final_hop_fee_unknown` | Reuses v2's fee-lookup helpers |
| `test_v3_honors_pair_budget_via_maxfee_msat` | Budget enforced at askrene layer |
| `test_v3_exclude_list_translates_to_throwaway_layer` | Exclude pattern works (or internal translation if research chose that) |
| `test_v3_drops_missing_requested_layers_silently` | Standalone-mode correctness |
| `test_v3_empty_layer_list_passes_empty_to_getroutes` | True standalone |
| `test_v3_init_logs_found_and_missing_layers_once` | Four-log-line startup contract |

### Unit tests — engine factory & runtime switch

In `test_rebalance_engine_v2.py`:

| Test | What it proves |
|---|---|
| `test_engine_builds_v2_router_when_askrene_unavailable` | Graceful degradation |
| `test_engine_builds_both_routers_when_askrene_available` | Runtime-switch prerequisite |
| `test_engine_active_router_respects_config_toggle` | `setconfig rebalance-router v3` flips dispatch |
| `test_engine_captures_router_at_cycle_start` | Per-cycle atomicity |
| `test_engine_rejects_v3_config_when_askrene_unavailable` | Config validator safety rail |
| `test_audit_record_carries_router_version_field` | A/B analysis tagging |
| `test_engine_v3_runtime_exception_aborts_cycle_cleanly` | No silent fallback on v3 errors |

### Integration tests — live CLN RPC surface

`tests/integration/test_router_v3_live.py`, gated behind `CLN_INTEGRATION=1`:

| Test | What it proves |
|---|---|
| `test_live_askrene_listlayers_succeeds` | Probe works against real CLN |
| `test_live_getroutes_returns_route_for_known_pair` | End-to-end route discovery |
| `test_live_getroutes_respects_hive_fleet_layer` | Layer bias actually influences route choice |
| `test_live_getroutes_standalone_no_layers` | Works with `layers=[]` |
| `test_live_setconfig_runtime_switch_honored` | Config hot-reload confirms research finding |
| `test_live_exclude_pattern_chosen_by_research` | Whichever exclude approach won actually works |

Harness: `conftest.py` helper `live_plugin()` connects to `$LIGHTNING_RPC` socket (default `~/.lightning/bitcoin/lightning-rpc`), skips tests with a clear message if unavailable. No mocking.

### Replay tests — captured snapshots

Fixtures in `tests/fixtures/router_v3/`:

- `getroutes_response_direct_pair.json`
- `getroutes_response_multi_hop.json`
- `getroutes_response_with_fleet_layer.json`
- `getroutes_response_empty.json`
- `getroutes_response_multi_path.json`

Each captured live during research via `tools/capture_getroutes_fixture.py`. Replay test `test_router_v3_replay_fixture_picks_matching_single_path` loads each fixture, feeds it to a mocked `plugin.rpc.getroutes`, asserts the router's `RouteResult` matches a committed golden output. Regenerating goldens is explicit (`pytest --regen-goldens`), not automatic.

### A/B field test

After phases 1-2 ship behind the runtime switch, before v3 becomes default:

1. **Warm-up.** 24h on `v2`, capture baseline audit log.
2. **A.** 48h on `v2`. Metrics: avg `route_cost_sats`, success rate, `no_route` count, cycles aborted on exception.
3. **B.** 48h on `v3`. Same metrics.
4. **Flip-back safety.** Any day where v3 success rate drops >10% vs. v2 baseline triggers automatic rollback via `tools/router_v3_safety_monitor.py` (calls `setconfig rebalance-router v2` and alerts). Script and thresholds committed.
5. **Promotion criterion.** v3 becomes default only when: success rate ≥ v2, avg cost ≤ v2, zero unexplained cycle aborts, operator signs off in writing.

### What is NOT tested

- askrene's internal MCF solver correctness — CLN's responsibility
- cl-hive's layer publishing correctness — cl-hive's responsibility
- xpay internal retry logic (phase 2) — trust CLN, test only the integration boundary
- Multi-node network topology scenarios — we're not re-testing CLN's gossip

## Migration & Rollout

### Phase 0 — Research (no code)

**Deliverable**: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` with the nine sections above.

**Gate to Phase 1**: research doc committed, user-reviewed, user-approved. Q4 (xpay depth) and exclude-handling choice resolved. CLN source citations present for every behavioral claim.

**If research kills the project** (xpay too immature AND layer-aware getroutes has no observable benefit on this node's topology): spec is closed as "researched, no action taken," worktree discarded, v2 remains production.

### Phase 1 — v3 router + v2 executor (shippable slice)

**Scope**:
- New `modules/rebalance_router_v3.py`
- Engine factory + runtime-switch plumbing in `rebalance_engine_v2.py`
- 4 new config keys in `modules/config.py` + validator
- Audit record `router=` field in `modules/rebalance_audit_v2.py`
- All unit tests under "Unit tests — `rebalance_router_v3.py`" and "Unit tests — engine factory & runtime switch"
- Replay fixtures + replay tests under "Replay tests — captured snapshots"
- Integration tests under "Integration tests — live CLN RPC surface" (skipped unless `CLN_INTEGRATION=1`)
- A/B monitoring script skeleton `tools/router_v3_safety_monitor.py`

**Explicit non-scope in Phase 1**: no xpay code; no `rebalance_router_v2.py` modifications; no default flip.

**Gate to Phase 2**: all Phase 1 tests green, dry-run cycle against real node produces identical candidate selection to v2 for a known-stable pair, A/B baseline captured, operator signs off on the audit-log diff.

### Phase 2 — xpay executor (conditional)

Scope gates on research outcome. Only runs if research concluded xpay is usable. Scope matches whichever of Q4 options (a)/(b)/(c) the research picked.

**Deliverable shape**:
- New `modules/rebalance_executor_v3.py` (size and shape TBD)
- `rebalance-executor` config key unlocked
- Unit + integration + replay tests for the executor
- Audit record `executor=` field alongside `router=`
- A/B comparison framework extended to tag executor version

**Gate to Phase 3**: all Phase 2 tests green, real-node A/B for the 2×2 matrix completes, operator signs off.

**If Phase 2 is skipped** (research rejects xpay): spec transitions directly from Phase 1 → Phase 3.

### Phase 3 — Default flip

**Scope**:
- `rebalance-router` default changes from `"v2"` to `"v3"` in `modules/config.py`
- `CLAUDE.md` updates to mention the new default
- Changelog entry
- v2 router code stays, clearly commented as `# fallback path for CLN < 24.11 or operator opt-out`

**Gate to Phase 4**: Phase 1+2 shipped in production with the runtime switch, at least 2 weeks of v3-default operation with no rollback events, audit logs reviewed, operator signs off in writing.

### Phase 4 — v2 router deprecation (out of scope for this spec)

Explicitly listed to mark the horizon. When CLN 24.11+ is the overwhelming baseline and the v2 router has seen zero fallback activity for 6+ months, a future spec may deprecate and delete the v2 router. **This spec does not authorize that work.**

### Cross-phase artifacts

- **Worktree** `.worktrees/askrene-router-v3-20260410` is the sole workspace until Phase 3 merges. Phase 0 → Phase 3 all commit here on `feature/askrene-router-v3`.
- **Decision log** lives in the research doc's Section 9 and is appended with phase outcomes. No separate decision log file.
- **Rollback plan** for every phase is "flip `rebalance-router` back to `v2` via setconfig." No database migrations, no schema changes, no state to unwind.

## Expected Outcome

After Phase 3, cl-revenue-ops routes every circular rebalance through askrene with optional fleet-layer biasing, executes via xpay (or v2 executor if research rejected xpay), supports runtime A/B toggle between v2 and v3 routers, and runs healthy on a standalone CLN 24.11+ node with no cl-hive. The v2 router stays as the fallback for CLN < 24.11 and as the escape hatch for any future v3 regression.

The engine preserves v2's explainability contract: for every valuable imbalanced channel, every cycle, the audit log still answers "did we rebalance it, and if not, exactly why not?" — now with a `router=` field that tells you which strategy produced the answer.
