# Boltz, LN+, and Capacity Planner Decommission Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:test-driven-development for each behavior change and superpowers:verification-before-completion before every completion claim.

**Goal:** Remove Boltz, LN+, and CapacityPlanner execution authority from `cl_revenue_ops` while preserving revenue reporting, fee control, generic channel policy, and ordinary budget-constrained rebalancing.

**Architecture:** Work forward from approved Python `main` in an isolated worktree. Delete retired executor roots and their exclusive dependencies, preserve historical schema and generic ledger data, and pin the smaller authority boundary with source-level architecture tests plus retained-behavior tests. Production cutover is a separate Tier 1 step after Python and Rust target review, using read-only preflight evidence and explicitly safe rollback configuration.

**Tech Stack:** Python 3, Core Lightning Python plugin API, SQLite, pytest, shell deployment tooling, Hexmem task routing.

---

## Global constraints

- Base the worktree on commit `14f46fb` or a later reviewed descendant of `origin/main` that contains the approved design.
- Do not reset or wholesale-revert to `320fa438ebe787105630a895e8732db7c85c1356`; use it only to identify LN+-specific additions.
- Do not drop or rewrite historical LN+, Boltz, planner, diagnostic, or spend-ledger rows.
- Keep generic `no_close` policy rows and treat retired LN+ contract tags as operator-owned.
- Keep these reporting and diagnostic surfaces functional: `revenue-status`, `revenue-profitability`, `revenue-history`, `revenue-report`, `revenue-dashboard`, `revenue-health`, `revenue-budget`, and `revenue-rebalance-debug`.
- Keep ordinary `revenue-rebalance-cycle` and explicitly requested `revenue-rebalance` behavior. Both remain subordinate to pause/authority/policy checks, atomic spend reservation, daily and weekly limits, channel capex limits, and the unified budget ledger.
- Remove only planner defibrillation rebalancing and planner-driven fee reduction.
- Historical spend categorized as `boltz` remains included in generic total-cost reporting until it naturally ages out of the selected reporting window. A historical enum or category string may remain only when required to decode old ledger rows; it must not authorize execution.
- The retained plugin must start without `boltzcli`, `boltzd`, a Boltz datadir, or LN+ network access.
- No Hive, Mycelium, fleet coordinator, Sling dependency, or replacement autonomous channel executor may be introduced.
- Tests use local fakes and temporary SQLite databases only. They must never call live CLN open, close, connect, sign, pay, withdraw, swap, planner, LN+, or Boltz action RPCs.
- Production deployment is Tier 1: the owner cannot pass the independent `review` criterion.
- Do not deploy Python until the Rust whole-plugin target and active Hexmem routing can no longer reintroduce these capabilities.

## Task 1: Establish the retained boundary and safe cutover tools

**Files:**

- Create: `tests/test_retired_liquidity_authority.py`
- Create: `tests/test_retained_revenue_core.py`
- Create: `tests/test_liquidity_decommission_tools.py`
- Create: `tools/liquidity_decommission_preflight.py`
- Create: `tools/render_liquidity_decommission_config.py`
- Modify: `tests/test_architecture_guard.py`

### Step 1: Add a passing authority-inventory harness

In `tests/test_retired_liquidity_authority.py`, read explicit tracked Python paths through `pathlib.Path`. Add separate inventory functions for module imports/files, RPC names, option names, direct CLN/subprocess mutation verbs, retained historical DDL, and retired state-transition helpers. Add characterization tests proving the current audited roots are found:

- the five retiring module roots are present;
- each retired RPC/option family is nonempty;
- each audited mutation verb is assigned to the expected LN+, Boltz, or planner owner;
- historical table declarations and generic ledger/policy helpers are present.

Use literal source paths and symbols. Do not scan comments/docs or infer ownership by fuzzy substrings. Tasks 2, 3, and 4 each replace only their owner's positive characterization with a failing absence assertion before deleting implementation, then leave the absence assertion green.

Extend `tests/test_architecture_guard.py` so the existing no-Sling and no-Hive/Mycelium assertions cover the final tracked Python source set after deletions.

```bash
pytest -q tests/test_retired_liquidity_authority.py tests/test_architecture_guard.py
```

Expected: the audited current inventory and existing no-Sling/no-coordinator guards pass.

### Step 2: Pin retained reporting and budgeted rebalancing

In `tests/test_retained_revenue_core.py`, use the repository's existing plugin-loading fixture and fake RPC/database collaborators. Add tests that assert:

1. The registered method inventory contains these retained names: `revenue-status`, `revenue-profitability`, `revenue-history`, `revenue-report`, `revenue-dashboard`, `revenue-health`, `revenue-budget`, `revenue-rebalance-debug`, `revenue-rebalance-cycle`, and `revenue-rebalance`.
2. Calling each read-only surface with empty database rows, empty gossip, and collaborator RPC errors returns a neutral/error-bearing response without calling any fake mutation method.
3. Pre-existing generic `no_close` tags survive database initialization and plugin reload.
4. An ordinary rebalance is rejected when paused, when policy disallows it, when the channel daily cap is exhausted, when the weekly/global cap is exhausted, and when atomic reservation fails.
5. A permitted rebalance reserves the estimated amount atomically before executor delegation, settles actual cost afterward, and releases the reservation on executor failure.
6. A rebalance-cycle read/plan phase cannot bypass the same authority and budget gates.
7. Historical ledger rows in category `boltz` contribute to generic total cost.

```bash
pytest -q tests/test_retained_revenue_core.py
```

Expected: all retained-baseline tests pass before any deletion. Task 2 strengthens the tag test to startup without LN+; Task 3 adds the manager-free historical-report assertion; Task 5 makes the retained RPC set exact by proving retired families are absent.

### Step 3: Implement and test the read-only production preflight

Create `tools/liquidity_decommission_preflight.py` with this CLI:

```text
usage: liquidity_decommission_preflight.py --db PATH --output PATH
```

Requirements:

- Open `--db` with SQLite URI `mode=ro`; never migrate or write it.
- Reject an existing output path unless it is a regular file owned by the current user and already mode 0600.
- Query current LN+ application/contract rows and retired spend reservations using the actual column names in `modules/database.py`.
- Fail nonzero if any LN+ row is in an application, pending, opening, funding-unknown, or otherwise nonterminal/non-contract state.
- Fail nonzero if any retired Boltz/LN+/planner reservation is active, or if a pending Boltz swap is represented in current schema.
- For every active imported contract, verify the generic policy table contains `no_close` for the peer. Cover both outbound and known incoming contract peers.
- Write a JSON document with `schema_version`, `generated_at`, database identity, precondition results, and contracts containing peer id, direction, stored expiry, and `tag_removal_owner: "operator"`.
- Create the output with mode 0600 using an exclusive create. Do not log contract identifiers to stdout.
- Exit 0 only when every fail-closed condition is satisfied.

In `tests/test_liquidity_decommission_tools.py`, create temporary databases from the real schema initializer and cover: safe contract-only state, every forbidden state, missing `no_close`, malformed rows, active retired reservation, existing unsafe output file, exclusive-create race, mode 0600, and proof that the input database bytes and modification time remain unchanged.

Run:

```bash
pytest -q tests/test_liquidity_decommission_tools.py -k preflight
```

### Step 4: Implement and test deterministic config rendering

Create `tools/render_liquidity_decommission_config.py` with this CLI:

```text
usage: render_liquidity_decommission_config.py --input PATH --active-output PATH --rollback-output PATH
```

The active output removes every option whose normalized name starts with:

```text
revenue-ops-boltz-
revenue-ops-lnplus-
revenue-ops-planner-
revenue-ops-expansion-treasury-
```

The rollback output preserves other configuration and contains exactly one effective assignment for each old-release safety gate:

```text
revenue-ops-planner-enabled=false
revenue-ops-planner-dry-run=true
revenue-ops-planner-max-opens-per-cycle=0
revenue-ops-planner-execute-closes=false
revenue-ops-planner-max-closes-per-cycle=0
revenue-ops-boltz-enabled=false
revenue-ops-boltz-auto-cycle-enabled=false
revenue-ops-expansion-treasury-enabled=false
revenue-ops-lnplus-swaps-enabled=false
revenue-ops-lnplus-execute-applications=false
```

Preserve unrelated comments and option order. Reject symlink inputs/outputs, do not edit the input, exclusive-create both outputs with mode 0600, and remove a partially created paired output if the second create fails.

Test duplicate retired options, whitespace/comments, malformed lines, symlinks, existing outputs, paired-write failure, exact safety-gate values, mode 0600, and unchanged input bytes.

Run:

```bash
pytest -q tests/test_liquidity_decommission_tools.py -k config
```

### Step 5: Commit the boundary and tooling checkpoint

Run:

```bash
pytest -q tests/test_liquidity_decommission_tools.py tests/test_retained_revenue_core.py tests/test_retired_liquidity_authority.py tests/test_architecture_guard.py
git diff --check
git add tests/test_retired_liquidity_authority.py tests/test_retained_revenue_core.py tests/test_liquidity_decommission_tools.py tests/test_architecture_guard.py tools/liquidity_decommission_preflight.py tools/render_liquidity_decommission_config.py
git commit -m "test: pin retired liquidity authority boundary"
```

Every test must pass at this checkpoint. Do not commit a failing retirement assertion.

## Task 2: Remove LN+ acquisition and lifecycle authority

**Files:**

- Delete: `modules/lnplus_swaps.py`
- Modify: `cl-revenue-ops.py`
- Modify: `modules/config.py`
- Modify: `modules/database.py`
- Modify: `modules/data_service.py`
- Modify: LN+-specific tests, fixtures, and scenario/catalog files found by the inventory command below

### Step 1: Inventory the exact LN+ ownership graph

Run and save the output in the implementation review notes:

```bash
rg -n -i 'lnplus|ln\+|lightningnetwork\.plus|connect_peer|sign_message' --glob '*.py' --glob '*.json' --glob '*.md'
git diff 320fa438ebe787105630a895e8732db7c85c1356..HEAD -- modules/lnplus_swaps.py cl-revenue-ops.py modules/config.py modules/database.py modules/data_service.py
```

Classify each hit as live executor, historical schema/data, generic policy, retained reporting, or documentation/test. Put every live executor hit in the deletion set. Never delete a generic `no_close` row or the table declarations required to read old data.

### Step 2: Remove LN+ construction, loops, and public surfaces

First replace the LN+ positive characterization assertions in `tests/test_retired_liquidity_authority.py` with exact absence assertions for its module, RPCs, options, network/signing verbs, and state writers. Run `pytest -q tests/test_retired_liquidity_authority.py -k lnplus` and observe failure for the audited live surfaces before editing implementation.

Delete from `cl-revenue-ops.py`:

- imports, singleton/global state, construction, dependency injection, and shutdown hooks for LN+;
- the LN+ watcher/background loop and all scheduler registration;
- every `revenue-lnplus-*` RPC and dispatcher verb;
- status aggregation and health fields that require a live LN+ evaluator;
- planner/capex/rebalancer injection that exists only for LN+.

Delete LN+ option registration and dynamic/runtime config keys from `modules/config.py`. Stale DB overrides must be treated by the existing unknown-key path; do not map them to a compatibility implementation.

### Step 3: Remove LN+-exclusive persistence and service verbs

In `modules/database.py`, delete helpers that create, transition, backfill, rate, notify, reserve for, or otherwise advance LN+ live state. Keep table/index DDL and generic policy helpers unchanged.

In `modules/data_service.py`, delete `connect_peer` and `sign_message` only after `rg` proves no retained caller. Do not delete invoice, sendpay, waitsendpay, listpeerchannels, or other verbs used by ordinary rebalancing/reporting.

Delete `modules/lnplus_swaps.py`. Delete or rewrite tests and fixtures that promise LN+ execution; retain schema round-trip tests for historical rows and the new generic-policy survival test.

### Step 4: Verify LN+ removal and retained behavior

Run:

```bash
pytest -q tests/test_retired_liquidity_authority.py -k lnplus
pytest -q tests/test_retained_revenue_core.py
pytest -q tests/test_database.py tests/test_config_contradictions.py tests/test_p1_008_numeric_options.py tests/test_p1_026_enum_options.py tests/test_architecture_guard.py
rg -n -i 'lightningnetwork\.plus|revenue-lnplus-|revenue-ops-lnplus-|modules\.lnplus_swaps' --glob '*.py'
git diff --check
```

Expected: no live Python hit remains; historical schema strings may remain only in `modules/database.py` and tests that prove inert compatibility.

### Step 5: Commit the LN+ checkpoint

```bash
git add -A
git commit -m "refactor: remove LN+ execution authority"
```
## Task 3: Remove Boltz execution and wallet authority

**Files:**

- Delete: `modules/boltz_manager.py`
- Modify: `cl-revenue-ops.py`
- Modify: `modules/config.py`
- Modify: `modules/database.py`
- Modify: `modules/data_service.py`
- Modify: `modules/rebalance_engine_v2.py`
- Modify: `modules/capex_budget.py`

### Step 1: Inventory the complete Boltz reachability graph

Run:

```bash
rg -n -i 'boltz|boltzd|boltzcli|DrainDemand|drain_demand|structural.credit|expansion.treasury' --glob '*.py' --glob '*.json' --glob '*.toml' --glob '*.md'
rg -n 'get_boltz_coordination|set_boltz|actual_spent_by_category|tactical' modules cl-revenue-ops.py tests
```

Classify generic historical cost-category reads separately from live management. Preserve only the generic ledger reader needed to include an old `boltz` category in total spend.

### Step 2: Delete manager construction, action RPCs, and loop owners

First replace the Boltz positive characterization assertions in `tests/test_retired_liquidity_authority.py` with exact absence assertions for its module, RPCs, options, subprocess/wallet verbs, and state writers. Run `pytest -q tests/test_retired_liquidity_authority.py -k boltz` and observe failure before editing implementation.

Delete from `cl-revenue-ops.py` the `BoltzManager` import/global/construction, option refresh, shutdown, locks, caches, version probes, subprocess reachability, action RPCs, dispatcher verbs and deprecated aliases, schedulers, and manager-dependent status/health branches. This includes quote, loop-in, loop-out, chain-swap, claim, refund, withdraw, deposit, backup, external-pay-ignore, balance-cycle, auto-cycle, and expansion-treasury authority.

Delete the live option/runtime-key families from `modules/config.py`, including master, auto, CLI/datadir, structural-budget, treasury, quote, timeout, and scheduler keys. Unknown persisted keys remain inert. Delete `modules/boltz_manager.py` and tests whose contract is executable swap/wallet behavior.

### Step 3: Remove exclusive coordination and mutation verbs

From `modules/rebalance_engine_v2.py`, delete Boltz coordination getters/setters, structural drain-demand production, exhaustion counters used only by Boltz, and the `DrainDemand` return type. Keep ordinary source/destination selection, fee limits, routing, and budget gates.

From `modules/capex_budget.py`, delete Boltz structural/tactical credit and expansion-treasury methods. Keep unified cost attribution and per-channel/global rebalance budget methods.

From `modules/database.py`, delete Boltz live-state transition and dedicated live-budget helpers only after checking callers. Keep generic ledger methods and historical DDL.

From `modules/data_service.py`, delete `pay` only if `rg -n '\\.pay\\(' modules cl-revenue-ops.py` proves Boltz was its sole caller. Preserve invoice/sendpay/waitsendpay methods used by circular rebalancing.

### Step 4: Make reporting historical and manager-free

Add focused tests, then update retained RPC serializers so `revenue-budget` reports unified rebalance budgets and generic spend totals with no live Boltz block; `revenue-status` and `revenue-health` make no Boltz manager/scheduler claim; `revenue-rebalance-debug` uses the generic ledger; historical `boltz` rows count in `total_cost`; malformed historical rows degrade neutrally and never instantiate an executor.

### Step 5: Verify and commit the Boltz checkpoint

```bash
pytest -q tests/test_retired_liquidity_authority.py -k boltz
pytest -q tests/test_retained_revenue_core.py
pytest -q tests/test_capex_budget.py tests/test_rebalance_engine_v2.py tests/test_database.py tests/test_config_contradictions.py tests/test_p1_008_numeric_options.py tests/test_p1_026_enum_options.py tests/test_architecture_guard.py
rg -n -i 'boltzcli|boltzd|revenue-boltz-|revenue-ops-boltz-|modules\\.boltz_manager|DrainDemand' --glob '*.py'
git diff --check
git add -A
git commit -m "refactor: remove Boltz execution authority"
```

Expected: remaining case-insensitive `boltz` hits are limited to historical schema/category compatibility and tests proving that compatibility. No executable, datadir, HTTP, subprocess, wallet, or scheduler path remains.
## Task 4: Remove CapacityPlanner and planner-only defibrillation

**Files:**

- Delete: `modules/capacity_planner.py`
- Delete: `modules/demand_flow.py`
- Delete: `modules/protection_service.py`
- Delete: `tools/capex_planner_loop.py`
- Delete: `tools/audit/loop_sweep_planner.py`
- Modify: `cl-revenue-ops.py`
- Modify: `modules/config.py`
- Modify: `modules/database.py`
- Modify: `modules/data_service.py`
- Modify: `modules/rebalance_engine_v2.py`
- Modify: `modules/capex_budget.py`
- Modify: planner-specific tests, fixtures, scenario catalogs, and packaging files

### Step 1: Prove exclusive dependencies before deletion

Run:

```bash
rg -n 'CapacityPlanner|capacity_planner|DemandFlow|demand_flow|ProtectionService|protection_service|diagnostic_rebalance|fund_channel|close_channel' --glob '*.py' --glob '*.json' --glob '*.md'
```

Confirm `demand_flow.py` and `protection_service.py` have no retained caller outside CapacityPlanner. If a generic data calculation has a retained reporting caller, move that pure calculation into the existing reporting/profitability module with a focused parity test before deleting the planner dependency; do not keep a planner-shaped wrapper.

### Step 2: Delete planner wiring, methods, options, and loops

First replace the planner positive characterization assertions in `tests/test_retired_liquidity_authority.py` with exact absence assertions for planner modules, RPCs, options, loops, `fundchannel`, `close`, and diagnostic rebalance. Run `pytest -q tests/test_retired_liquidity_authority.py -k 'planner or channel_mutation'` and observe failure before editing implementation.

Delete from `cl-revenue-ops.py`:

- the CapacityPlanner import/global/construction and dependency injections;
- planner periodic/background loop ownership and wake/shutdown hooks;
- `revenue-planner-execute` and planner status/candidate/source/report/history RPCs;
- dispatcher verbs and deprecated aliases;
- health/status sections that advertise planner readiness or candidate state;
- automatic open, close, dual-fund, loser staging, defibrillation, and fee-reduction delegation.

Delete every `revenue-ops-planner-*` option and public/runtime planner key from `modules/config.py`. Do not retain a hidden default or alias capable of rebuilding the planner.

Delete `modules/capacity_planner.py`, `modules/demand_flow.py`, `modules/protection_service.py`, and tools whose sole purpose is planner candidate/action management.

### Step 3: Remove planner-exclusive database and CLN mutation verbs

From `modules/database.py`, delete planner candidate/action/recycle writer helpers and readers used only by removed RPCs. Preserve DDL for `planner_actions`, `planner_candidates`, and any historical diagnostic table.

From `modules/data_service.py`, delete `fund_channel` and `close_channel` only after source inventory proves there is no retained caller. The plugin must have no CLN channel-open or channel-close wrapper after this step.

From `modules/rebalance_engine_v2.py`, delete `diagnostic_rebalance` and planner setter/callbacks. Retain ordinary rebalance methods and their budget/authority integration.

From `modules/capex_budget.py`, remove planner open/close/exploration allocation APIs. Retain the smallest budget engine that supports per-channel, daily, weekly/global rebalance caps, atomic reservation, settlement, release, stale-release safety, and generic historical cost totals.

### Step 4: Add explicit negative and positive authority tests

Before implementation changes, add tests to `tests/test_retained_revenue_core.py` proving:

- no retained RPC, config update, startup row, or historical planner row can call a fake `fundchannel` or `close` method;
- no status/health/report call can start a background loop or diagnostic rebalance;
- malformed and absent planner history does not crash generic reporting;
- a normal budget-approved manual rebalance still reaches the rebalance executor;
- the same rebalance is denied by each retained authority/budget gate and cannot reserve twice.

### Step 5: Verify and commit the planner checkpoint

Run:

```bash
pytest -q tests/test_retired_liquidity_authority.py -k 'planner or channel_mutation'
pytest -q tests/test_retained_revenue_core.py
pytest -q tests/test_capex_budget.py tests/test_rebalance_engine_v2.py tests/test_profitability_analyzer.py tests/test_database.py tests/test_config_contradictions.py tests/test_p1_008_numeric_options.py tests/test_p1_026_enum_options.py tests/test_architecture_guard.py
rg -n -i 'CapacityPlanner|revenue-planner-|revenue-ops-planner-|diagnostic_rebalance|fundchannel|close_channel' --glob '*.py'
git diff --check
git add -A
git commit -m "refactor: remove capacity planner authority"
```

Expected: no planner live source hit remains. Historical table declarations and neutral historical readers may remain only where a retained report test requires them.
## Task 5: Re-pin public contracts, inventories, documentation, and version

**Files:**

- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `cl-revenue-ops.py` (`PLUGIN_VERSION`)
- Modify: `config/cl-revenue-ops.conf.full`
- Modify: `config/cl-revenue-ops.conf.minimal`
- Modify: `docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md`
- Modify: `docs/refactor/phase0/compatibility-catalog.md`
- Modify: `tests/test_rpc_surface_inventory.py`
- Modify: `tools/audit/deep_manifest.py`
- Modify: `docs/audit/deep/coverage-manifest.md`
- Modify: `tests/test_deep_manifest_p5005_coverage.py`
- Modify: `tests/test_deep_manifest_p5006_check.py`
- Modify: `tests/test_daemon_survival.py`
- Modify: `tests/test_drain_demand.py`
- Modify: `tests/test_explainability.py`
- Modify: `tests/test_loop_heartbeat_surface.py`
- Modify: `tests/test_phase_c_dispatchers.py`
- Modify: `tests/test_plugin_listing_compat.py`
- Modify: `tests/test_pyln_integration.py`
- Modify: `tests/test_rebalance_engine_v2.py`
- Modify: `tests/test_revenue_validation_collect.py`
- Modify: `tests/test_rpc_surface_inventory.py`
- Create: `docs/operations/LIQUIDITY_EXECUTOR_DECOMMISSION_RUNBOOK.md`

### Step 1: Update contract tests before documentation

Update report/dashboard/status/health/budget snapshot tests to assert the retained contract: fee decisions/outcomes; profitability/contribution and flow evidence; generic policy/protection; unified spend and rebalance budget state; and ordinary rebalance diagnostics all remain visible. Assert that no retired manager, candidate, scheduler, action, treasury, or external-service readiness field is advertised.

If deleting a field is intentionally breaking, record it in the runbook and version notes. Do not leave a success-shaped `disabled` tombstone implying that a retired subsystem is supported.

Find and execute the relevant contract files:

```bash
rg -l 'revenue-status|revenue-report|revenue-dashboard|revenue-health|revenue-budget|revenue-rebalance-debug' tests
```

### Step 2: Update source-of-truth documentation and generated inventories

Update `README.md`, `AGENTS.md`, sample configs, RPC inventories, feature-story/audit catalogs, and package metadata to describe a standalone plugin with retained fee/revenue/profitability/flow/reporting capability and retained ordinary, explicitly authorized, budget-constrained rebalancing. It has no Boltz, LN+, CapacityPlanner, automatic channel open/close, planner defibrillation, Sling, or external coordinator.

Remove retired action RPCs from the AGENTS action warning because they no longer exist; retain warnings for real rebalance, fee, policy, budget, and other mutation RPCs.

Regenerate committed inventories with their repository scripts. Review every generated deletion; do not hand-edit generated output when a generator owns it.

### Step 3: Write the operational runbook

In `docs/operations/LIQUIDITY_EXECUTOR_DECOMMISSION_RUNBOOK.md`, record these production paths:

- Lightning directory: `/data/lightningd`
- active config: `/data/lightningd/config`
- plugin entrypoint: `/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py`
- revenue database: `/data/lightningd/.lightning/revenue_ops.db`
- private contract record: `/data/lightningd/cl_revenue_ops-retired-lnplus-contracts.json`

The runbook must include read-only preflight and failure interpretation; active/rollback config rendering and mode checks; artifact hashes and backups; the established stop/install/start lifecycle; read-only retained RPC verification; `help` inventory verification for removed methods; thread/process inspection; before/after database evidence proving no retired economic action; a `no_close` recheck for every private-record peer; and rollback gates that keep every old executor gate false.

Do not include node-specific peer identifiers or private preflight output in the repository.

### Step 4: Apply the breaking version bump

Set `PLUGIN_VERSION` in `cl-revenue-ops.py` to `3.0.0` and update all tests/docs that assert it. This is an intentional action-surface contraction, not a patch-level compatibility change.

### Step 5: Verify and commit contracts/docs

```bash
pytest -q tests/test_retained_revenue_core.py tests/test_retired_liquidity_authority.py tests/test_architecture_guard.py
rg -n -i 'revenue-(lnplus|boltz|planner)-|revenue-ops-(lnplus|boltz|planner)-|lightningnetwork\.plus|boltzcli|boltzd' README.md AGENTS.md docs tests tools cl-revenue-ops.py modules
git diff --check
git status --short
git add -A
git commit -m "docs: define post-planner revenue operations surface"
```

Review every remaining search hit and label it in the commit notes as approved historical compatibility, safe rollback documentation, or a test proving absence. Any other hit blocks completion.

## Task 6: Run full Python verification and independent code review

**Files:**

- Modify only files required by verified failures or review findings
- Create: a private or non-secret review evidence note outside committed node-specific data

### Step 1: Run the complete local verification set

From the isolated implementation worktree:

```bash
python -m compileall -q cl-revenue-ops.py modules tools
pytest -q
git diff --check
git status --short
```

Also run the repository's canonical generated-inventory/check commands documented in `README.md` or CI. If optional integration dependencies cause skips, list every skip and prove it is unrelated to the retired or retained authority boundary.

### Step 2: Perform explicit reachability scans

```bash
rg -n -i 'lightningnetwork\.plus|boltzcli|boltzd|CapacityPlanner|modules\.(lnplus_swaps|boltz_manager|capacity_planner|demand_flow|protection_service)' --glob '*.py'
rg -n 'fundchannel|close_channel|diagnostic_rebalance|get_boltz_coordination|set_boltz' --glob '*.py'
rg -n -i 'hive|mycelium|sling' --glob '*.py'
git ls-files 'modules/*.py' 'tools/*.py' 'tests/*.py' | sort
```

For every permitted historical hit, cite the exact retained test that proves it is inert. Any unclassified live hit fails the review.

### Step 3: Request independent code review

Provide the verifier the approved design, this plan, the commit range, test output, mutation-path inventory, remaining-hit classifications, and the retained-reporting/rebalance test names. The verifier must be an agent other than the implementation owner and must specifically verify:

- reporting was not accidentally hollowed out;
- ordinary rebalancing remains functional and cannot bypass budgets;
- no removed API, subprocess, thread, dispatcher alias, config alias, stale override, or historical row can reactivate authority;
- old LN+ `no_close` protection survives;
- startup has no Boltz binary/datadir or LN+ network dependency;
- no Sling or external coordinator returned.

Address findings with TDD, rerun focused and full verification, and obtain a fresh independent approval. Do not self-pass the Tier 1 `review` criterion.

### Step 4: Commit review fixes

If review required changes:

```bash
git add -A
git commit -m "fix: close liquidity decommission review findings"
pytest -q
git diff --check
```

The Python implementation is review-complete only with a clean worktree, full passing suite, classified reachability scan, and independent approval.
## Task 7: Retire the Rust targets and reconcile Hexmem routing

**Repositories and durable state:**

- Python repository: `/home/sat/bin/cl_revenue_ops`
- Rust repository: `/home/sat/bin/cl-revenue-ops-r`
- Shared protocol: `/home/sat/.claude/skills/superagent/SKILL.md`
- Existing production-shutdown task: Hexmem task 84

### Step 1: Publish and acknowledge a fresh team-capacity view

Before coordination, read the shared protocol completely. Publish Codex's capacity snapshot or explicit unknown, obtain the Python/Rust snapshots or explicit unknown, acknowledge the same fresh team view, and protect independent-review capacity. Store content in Hexmem and send only task-id pointers through tmux.

### Step 2: Search before changing durable routing

Use `hexmem_search` and task queries to find all active tasks whose goal ports, wires, tests, deploys, or promotes Rust Boltz, LN+, CapacityPlanner, planner mutations, or their live adapters. Classify each as retained read-only/reporting or ordinary rebalance work; retired executor work to supersede/cancel; or mixed work needing a narrowed replacement. Give feedback on an existing durable decision instead of creating a duplicate. Do not mark an implementation complete merely because its target was retired.

### Step 3: Create or update the separately owned Rust decommission task

Route the Rust source change as `owner=rust; verifier=codex; tier=1`. Its criteria require:

- live and shadow RPC manifests omit Boltz, LN+, and planner actions;
- no mutation adapter, subprocess, HTTP client, loop owner, config key, compatibility alias, or promotion gate can arm the retired targets;
- reporting and ordinary budget-constrained rebalancing targets remain;
- fixture/parity trackers explicitly classify historical data as inert;
- focused/workspace tests, fmt, clippy, diff check, exact changed-file review, and independent Codex review pass.

The Python implementation can proceed in isolation, but production cutover is blocked until this Rust task is independently approved.

### Step 4: Reconcile Hexmem task 84 without self-review

Task 84 records the narrow production Boltz shutdown and currently routes `owner=codex; verifier=rust; tier=1`. Codex may re-derive and attach implementation evidence, but must not pass its own `review` criterion.

Before the broader cutover, have Rust independently verify the effective and persistent three-gate Boltz shutdown, plugin health, and absence of swap action. If the old criterion text no longer matches the authorized verifier or broadened decommission scope, record a correction/supersession explicitly; do not force a false pass. Close or supersede task 84 before deployment so it cannot remain an ambiguous live-policy task.

### Step 5: Record routing outcome

After independent reviews, call `hexmem_observation_add` with `category="routing"`, `action_type="task_handoff"`, an action summary naming the Python and Rust task ids, context containing owner/verifier/tier reason codes, `outcome` set from observed success or failure, `outcome_details` naming the review result, and `outcome_source="review"`. Store durable pointers, not capacity percentages or secrets.

## Task 8: Perform the Tier 1 lnnode cutover and rollback verification

**Production paths:**

- `/data/lightningd/config`
- `/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py`
- `/data/lightningd/.lightning/revenue_ops.db`
- `/data/lightningd/cl_revenue_ops-retired-lnplus-contracts.json`

### Step 1: Reconfirm authority and save a private before-snapshot

Require explicit operator authority for the production window if execution does not occur in the same approved change session. Record exact reviewed Python and Rust commits, artifact hashes, active/rollback config hashes, database copy/hash, plugin help inventory, process/thread inventory, row counts/latest timestamps, generic policy tags, and effective runtime config.

All inspection commands are read-only. Do not call `revenue-rebalance-cycle`, `revenue-rebalance`, fee-cycle/action methods, planner actions, Boltz actions, LN+ actions, CLN open/close/pay/withdraw, or any mutation probe.

### Step 2: Persist fail-closed old-release overrides

While the reviewed old plugin is still installed, use `revenue-config set` only for these dynamic keys and values:

```text
planner_enabled=false
planner_dry_run=true
planner_max_opens_per_cycle=0
planner_execute_closes=false
planner_max_closes_per_cycle=0
boltz_auto_cycle_enabled=false
lnplus_swaps_enabled=false
lnplus_execute_applications=false
```

Verify each readback and the corresponding `config_overrides` row. Confirm the Lightning config already has `revenue-ops-boltz-enabled=false`, `revenue-ops-boltz-auto-cycle-enabled=false`, and `revenue-ops-expansion-treasury-enabled=false`. If the running old plugin still caches a true master/treasury option, restart only `cl_revenue_ops` under the approved production window and re-derive all gates. This config mutation and scoped restart are the only old-release mutations authorized by this step; they must not execute an economic cycle or action.

### Step 3: Run fail-closed preflight

Run the reviewed preflight against `/data/lightningd/.lightning/revenue_ops.db` and create `/data/lightningd/cl_revenue_ops-retired-lnplus-contracts.json` at mode 0600.

Stop if planner gates are not exactly false/true/zero/false/zero; any Boltz master/auto/treasury gate is true; either LN+ runtime gate is true; the rollback-critical `config_overrides` rows do not match Step 2; the L-BTC wallet is nonzero or a swap is pending/unknown; an LN+ row has an unresolved obligation; an active contract peer lacks generic `no_close`; a retired reservation is active; or the private artifact is incomplete/nonexclusive/not 0600. Resolve failures under the old reviewed release. Never bypass a failure by editing evidence.

### Step 4: Render and inspect active and rollback configs

Verify the active file has no retired option family. Verify the rollback file contains the ten exact safety assignments from Task 1. Compare unrelated options byte-for-byte after normalizing only retired lines.

Back up the current config and plugin tree to explicit timestamped paths. Record ownership, permissions, and hashes. Do not use destructive cleanup.

### Step 5: Stop, install, and start through the established lifecycle

Stop only `cl_revenue_ops`, atomically activate the reviewed plugin tree and cleaned config using the runbook, then start only `cl_revenue_ops`. Do not restart lightningd or unrelated plugins without separately authorized blast-radius expansion.

If startup fails, capture logs read-only and use the prepared rollback only after rechecking all old executor gates remain false.

### Step 6: Verify retained revenue service without economic action

Use read-only calls to verify plugin health/version `3.0.0`; fee observation/status; profitability, history, report, dashboard, and flow evidence; generic policy and every private-record `no_close` tag; retained unified/channel/global budget status; and rebalance debug/status with its authority/budget posture.

Do not prove rebalancing by executing one in production. Functionality is established by reviewed tests and read-only live wiring/manifest evidence; any real rebalance is a separately authorized economic action.

### Step 7: Verify removed reachability and zero unintended action

Confirm through `help`, logs, process/thread inspection, and before/after database snapshots that retired RPCs/config are absent; no Boltz executable, LN+ client, retired scheduler/watcher/planner loop starts; and no new planner action, LN+ transition, Boltz swap, channel open, channel close, payment, withdrawal, diagnostic rebalance, or read-only-surface mutation occurred.

### Step 8: Obtain independent production review

The independent verifier re-runs read-only gates from fresh shell state, compares evidence to reviewed commits/runbook, and marks the Tier 1 review criterion. The implementation owner must not self-pass it.

If review fails, keep the plugin stopped or roll back to the pre-agreed safe state; never reactivate old executor defaults merely to restore service.

### Step 9: Finalize durable outcomes

After approval and verification, search Hexmem before remembering the durable decision; give feedback to an existing matching record or add one atomic private record with commits, outcome, boundary, and runbook pointer; supersede tasks that would restore retired authority; add the required routing observation; and defer inert-schema/config-override cleanup until after the rollback window.

## Final completion gate

The work is complete only when:

1. Python source, tests, docs, config catalog, and RPC manifest contain no reachable retired executor.
2. Revenue reporting and fee/profitability/flow evidence pass focused/full tests and respond live.
3. Ordinary manual/cycle rebalancing passes positive-path tests and every authority, atomic-reservation, daily, weekly, channel, and global budget-denial test.
4. Historical schema and generic cost/policy data remain readable and non-authorizing.
5. Rust targets and durable task routing cannot reintroduce retired authority.
6. Production preflight, cutover, zero-action comparison, and independent Tier 1 review pass.
7. Worktrees are clean and exact reviewed commits are recorded.
