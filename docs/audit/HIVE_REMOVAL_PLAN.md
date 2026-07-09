# Plan: make cl_revenue_ops truly standalone (remove all hive / cl-mycelium code)

Status: proposed 2026-07-09 · Context: cl-mycelium retired and disabled on both
fleet nodes. cl_revenue_ops is the keeper; this plan removes every hive / mycelium
dependency so it stands on its own with no dead coordination code.

## Starting reality (evidence)

- **cl_revenue_ops is already *functionally* standalone.** When cl-hive is absent,
  `HiveHintAdapter` init fails and the code falls back to `hive_hints = None`
  (`cl-revenue-ops.py:2995`); every consumer reads hints through `getattr(hive_hints, …)`
  with neutral defaults. The nodes are running this way right now, healthy. So this is a
  **code-cleanup / de-risking effort, not a functional fix** — nothing here changes revenue
  behavior (the one behaviorally-significant consumer, member→zero-fee, is already inert
  with no fleet members).
- **Surface to remove:**
  - **4 dedicated modules, ~4,032 lines** (wholesale delete): `hive_hints.py` (2582),
    `hive_router.py` (732), `rebalance_hive_router.py` (664), `hive_runtime.py` (54).
  - **Embedded references across 15 modules** (~320 total), by density:
    `fee_controller.py` (125), `rebalance_engine_v2.py` (54), `capacity_planner.py` (38),
    `rebalancer.py` (24), `rebalance_coordination_overlay.py` (16),
    `rebalance_route_policy.py` (14), `lnplus_swaps.py` (12), `config.py` (12),
    `capex_budget.py` (8), `policy_manager.py` (5), `database.py` (5),
    `rebalance_state_v2.py` (4), `capital_efficiency.py` (4), `profitability_analyzer.py` (1).
  - **Injection**: `cl-revenue-ops.py:2988-3006` builds `HiveHintAdapter` and injects it into
    6 components (`fee_controller`, `rebalancer`, `profitability_analyzer`, `policy_manager`,
    `capacity_planner`, and the v2 engine).
  - **3 RPC calls to cl-hive**: `hive-export-hints` (`hive_hints.py:306`),
    `hive-report-rebalance-intent` (`rebalancer.py:957`),
    `hive-report-rebalance-outcome` (`rebalancer.py:1070,1149`).
  - **askrene default layer**: `rebalance_router_v3.py:45` `DEFAULT_ASKRENE_LAYERS="hive-fleet"`
    + `OBSERVED_LIQUIDITY_LAYER="hive-observed-liquidity"`; the whole `hive_router.py` /
    `rebalance_hive_router.py` layer machinery.
  - **6 config options**: `revenue-ops-hive-rebalance-bootstrap-budget-sats`,
    `revenue-ops-lnplus-fleet-pubkeys` (drop — fleet retired), `revenue-ops-hive-zero-fee-stale-grace`,
    `revenue-ops-hive-hints-enabled`, `revenue-ops-hive-hints-ttl`,
    `revenue-ops-hive-hints-allow-all-hints-m2-scope`.
  - **~80 test files** reference hive; **~133 docs** mention hive/mycelium.

## The one strategic choice

**Neutralize vs. excise.** We could stop at "neutralize" (force `hive_hints=None`, drop the
RPCs, flip the askrene default) — fast, ~zero risk, but leaves ~4,000 lines of inert code.
The directive is *truly standalone with all hive/mycelium removed*, so this plan **excises**,
in phases ordered by risk so we never do the dangerous part (fee_controller) without a green
suite behind us. Each phase ends with the full test suite passing and, for the behavioral
phases, a deploy + live check.

## Phasing (each phase: TDD/edit → full suite green → commit; deploy where noted)

**Phase 0 — Cut the live wires (low risk, do first). ✅ DONE 2026-07-09 (v2.14.0).**
Made it standalone at the seams without touching the consumers yet:
- `hive_hints = None` unconditional (`cl-revenue-ops.py`), injection kept.
- All 3 `rpc.call("hive-…")` sites in `rebalancer.py` neutralized (no-ops).
- `DEFAULT_ASKRENE_LAYERS` + config defaults flipped to the `"standalone"` sentinel;
  fixed `_configured_layer_names` ordering so blank→default→sentinel resolves to `[]`.
- **Deferred:** removing the 6 hive config *option registrations* — dropping an option
  a node's config still references is restart-fatal (we hit exactly this on nexus-01).
  Options stay registered-but-unused until the config files are cleaned first.
- Suite green (3657 passed). Original Phase-0 scope, for reference:
- Replace the `HiveHintAdapter` construction with `hive_hints = None` unconditionally; keep the
  attribute injection so consumers still no-op through their neutral `getattr` paths.
- Remove the 3 `rpc.call("hive-…")` sites in `rebalancer.py` (intent/outcome become no-ops).
- Flip `DEFAULT_ASKRENE_LAYERS` to CLN's default (no hive layer) and stop calling the
  hive_router layer creation.
- Delete the 6 hive config options + their `config.py` plumbing.
- **Deploy + verify** both nodes: fees/rebalances/opens unchanged, no "hive-export-hints not
  found" log noise. This alone delivers "truly standalone" operationally.

**Phase 1 — Delete the dedicated modules. ✅ DONE 2026-07-09 (v2.15.0).**
Re-scoped from "4 modules" to **3**: `hive_hints.py`, `hive_router.py`, `hive_runtime.py`
(~3,368 lines). Removed the imports (`cl-revenue-ops.py:44-46`), the `HiveRouter`
construction/injection, all four `refresh_hive_runtime` call sites, and the hive_refresh
debug block. `hive_hints`/`hive_router` remain as permanently-`None` globals (neutral seams);
their guarded consumer branches no-op until Phase 2/3. Deleted 18 dedicated hive test modules
and pruned 5 mixed ones (kept non-hive coverage). Suite green (3330 passed).
- **`rebalance_hive_router.py` deferred to Phase 3**: it is a *module-level* import of
  `rebalance_engine_v2.py` (`from .rebalance_hive_router import RebalanceHiveRouter`), so it
  cannot be cleanly removed before the engine is de-hived. It moves into Phase 3 with the engine.
- **`tools/audit/*` hive sweeps deferred to Phase 5** (dev-only, not runtime/suite).

**Phase 2 — Surgical de-hive of the easy modules (low density first).**
`growth_budget` (0) → `profitability_analyzer` (1) → `capital_efficiency` (4) →
`rebalance_state_v2` (4) → `database` (5) → `policy_manager` (5) → `capex_budget` (8) →
`rebalance_route_policy` (14) → `rebalance_coordination_overlay` (16). For each: delete the
hint-bias code path and the now-dead branches, keeping the underlying decision on its
non-hive inputs. TDD each; the coordination-overlay lease/campaign/segment logic goes entirely
(it required a fleet).

**Phase 3 — The dense, revenue-critical modules (highest care).**
`rebalancer` (24) → `capacity_planner` (38) → `rebalance_engine_v2` (54) →
`fee_controller` (125, last). fee_controller is the revenue engine and holds the member→zero-fee
corridor logic, competition/traffic/quality biases, and the fleet-fee-median prior — all of
which collapse to their local-only behavior once hive inputs are gone. Do this module with a
dedicated test pass and a **deploy + multi-day fee/rebalance watch**, since it's the only place
removal could shift live pricing (even though it's inert without a fleet today).

**Phase 4 — LN+ swaps de-hive.** `lnplus_swaps` (12): LN+ swap *automation stays* (it's a real
standalone feature); only the fleet-topology *swap hints* consumption goes.

**Phase 5 — Tests + docs.** Delete/trim the ~80 hive-referencing test files (many are whole
hive test modules — delete; others need the hive assertions pruned). Update the ~133 docs:
delete the hive/mycelium contracts and audit docs, scrub hive references from README/AGENTS and
the operator/config docs, and update the RPC/option inventories. Rename any remaining
"fleet/hive" operator language to plain node-local terms.

## Verification gates
- Full `pytest` green after every phase.
- After Phase 0 and Phase 3: deploy to both nodes, confirm `revenue-status` healthy,
  fee/rebalance/open cycles run, and no hive-related errors in the log.
- Grep gate at the end: `grep -rniE "hive|mycelium" modules/ cl-revenue-ops.py` returns only
  incidental matches (ideally zero).

## What explicitly stays
Everything that makes cl_revenue_ops money on its own: the fee controller's local DTS/PID
pricing, native + v2/v3 rebalancing (minus the hive layer), the capacity planner, LN+ swap
automation, Boltz, capex/budget, profitability analysis, policy manager. Only the coordination
inputs and the code that consumed them are removed.

## Effort / sequencing note
Phases 0-1 are a day and deliver the operational win (standalone, dead wires cut, 4k lines
gone). Phase 2 is mechanical. Phase 3 (esp. fee_controller) is the real work and the only place
to be slow and careful. Phases 4-5 are cleanup. Recommend executing 0-1 immediately, then 2-3
with a deploy-and-watch between fee_controller and calling it done.
