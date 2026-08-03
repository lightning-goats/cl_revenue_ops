# Boltz, LN+, and Planner Decommission Design

Date: 2026-08-03

Status: operator-approved design; implementation plan pending written-spec review

## 1. Objective

Reduce `cl_revenue_ops` to the local revenue functions that have demonstrated
value: fee control, flow analysis, profitability analysis, policy protection,
budget accounting, and explicitly requested rebalancing. Remove autonomous
capital-allocation and external-swap authority that has not justified its
complexity or attack surface.

This decommission has three coupled targets:

1. Remove every Boltz swap and wallet-execution path from the plugin.
2. Remove every LN+ remote API, application, and lifecycle-execution path while
   preserving existing manually arranged contract-channel protection.
3. Remove the capacity planner's autonomous open, close, and defibrillation
   machinery while retaining human-facing profitability and flow evidence.

The change is a surgical removal from current `main`. Historical trees are
comparison evidence, not rollback targets. In particular, the parent of the
first LN+ commit, `320fa438ebe787105630a895e8732db7c85c1356`, is the semantic
reference for identifying LN+-specific surfaces. The repository must not be
reset or wholesale-reverted to that tree because later commits contain
unrelated correctness, safety, standalone-architecture, and refactor work.

## 2. Production Preconditions

Implementation and deployment must fail closed unless all of these statements
are freshly true:

- The capacity planner remains fully shut down: `planner_enabled=false`,
  `planner_dry_run=true`, `planner_max_opens_per_cycle=0`,
  `planner_execute_closes=false`, and `planner_max_closes_per_cycle=0`.
- Boltz automatic, master, and expansion-treasury controls are false in the
  persistent configuration and effective runtime.
- The integrated L-BTC wallet has no remaining balance and no pending swap.
- LN+ has no row in an application, pending, opening, funding-unknown, or other
  non-contract obligation state.
- Every active imported LN+ contract peer that the current module owns has its
  generic `no_close` policy tag present.
- A mode-0600 private operator record at
  /data/lightningd/cl_revenue_ops-retired-lnplus-contracts.json contains each
  active contract's peer, direction, stored expiry, and tag-removal
  responsibility. No node-specific contract identifiers are committed to this
  repository.

If an LN+ opening obligation, ambiguous funding result, pending Boltz swap, or
wallet balance appears during the cutover, deployment stops. The operator must
resolve that state under the old release before removal continues.

## 3. Target Architecture

### 3.1 Retained authority

The plugin retains only these relevant responsibilities:

- local forwarding-fee observation and fee decisions;
- local flow and demand analysis;
- channel profitability and contribution analysis;
- generic peer policy, including operator-owned `no_close` tags;
- read-only status and diagnostics for retained subsystems;
- governed budget and historical cost reporting;
- existing rebalancing behavior outside planner defibrillation, subject to its
  own authority, safety, and budget gates.

Profitability and flow evidence advises the human operator. It does not create
an implicit replacement channel-opening or channel-closing executor.

### 3.2 Removed authority

After deployment the plugin has no code path that can:

- spawn `boltzcli`, communicate with `boltzd`, create or manage a Boltz swap,
  or send from an integrated Boltz wallet;
- contact `lightningnetwork.plus`, request or sign an LN+ challenge, apply to a
  swap, backfill a remote obligation, open a channel for LN+, complete an
  application, rate a peer, mark notifications, or mutate ignore policy from
  LN+ data;
- autonomously call `fundchannel` or `close` as a capacity-planner action;
- trigger diagnostic rebalances as a planner defibrillation stage;
- run a Boltz scheduler, LN+ watcher, or capacity-planner background loop;
- restore any of those capabilities through a retained dynamic configuration
  key, deprecated alias, dispatcher verb, or hidden fallback.

No external coordinator, Hive/Mycelium integration, or Sling dependency is
introduced as a replacement.

## 4. LN+ Decommission

### 4.1 Contract safety boundary

All observed active LN+ contracts were arranged manually and imported into the
plugin. The valuable current behavior is close protection, not acquisition or
execution. Existing `no_close` tags are generic persistent policy data, so they
remain after the LN+ module is deleted.

Before removal, deployment tooling produces a private contract-expiry checklist
and verifies both outbound and known incoming tagged peers. Those tags become
operator-owned. No new background guardian is introduced. The operator removes
each tag manually only after the corresponding stored contract expiry or a
separately verified early termination.

Failing to remove a tag on time temporarily overprotects a channel; it cannot
spend funds or breach an LN+ contract. Removing a tag too early can breach a
contract, so the runbook is intentionally fail-closed toward overprotection.

### 4.2 Code and interface removal

Delete the LN+ client, evaluator, lifecycle watcher, construction wiring,
threads, configuration fields and options, public/runtime config keys, status
and action RPCs, dispatch entries, budget reservations, planner coupling,
database write helpers used only by live LN+ behavior, tests that specify LN+
execution, and operator documentation advertising the feature.

The existing LN+ tables and rows remain inert for historical audit and rollback
compatibility. No destructive schema migration or data deletion occurs in this
release. Generic policy rows containing `no_close` remain active.

## 5. Boltz Decommission

Delete the Boltz manager and every plugin surface that can reach it, including:

- subprocess command construction and version probing;
- quote, loop-in, loop-out, chain-swap, refund, claim, withdraw, deposit,
  backup, external-pay-ignore, balance-cycle, auto-cycle, and treasury RPCs;
- primary dispatcher verbs and deprecated aliases;
- balance and treasury schedulers, locks, caches, coordination objects, and
  startup wiring;
- Boltz-specific config options, public/runtime keys, profile settings, and
  dynamic refresh code;
- planner/rebalancer coordination and structural-credit paths whose only
  consumer is Boltz;
- tests and documentation that promise Boltz execution.

Historical Boltz swap records, spend-ledger categories, and schema remain
readable as historical data. They do not imply a live manager, executable, or
RPC. The plugin must start and serve retained read-only surfaces when
`boltzcli`, `boltzd`, and their datadir do not exist.

The discovered v2.12 sweep-wrapper mismatch is removed with the entire wallet
execution surface rather than patched: the current wrapper appends a zero
amount in sweep mode, while the installed CLI selects sweep mode only when no
amount positional is present.

## 6. Capacity Planner Decommission

Delete `CapacityPlanner` and its autonomous allocation behavior rather than
leaving a permanently disabled executor. Remove:

- open-candidate discovery and staging;
- automatic channel opening and dual-fund selection;
- loser staging and channel-close execution;
- defibrillation and planner-driven fee-reduction delegation;
- the planner thread and manual execution RPC;
- planner configuration options and runtime controls;
- planner status, candidate, sources, report, and history RPCs;
- rebalancer, capex, Boltz, and LN+ injection or coordination that exists only
  to support the planner.

The `planner_actions` and `planner_candidates` tables remain inert and
queryable directly for historical audit. The plugin no longer updates them or
advertises a planner surface. Existing generic profitability classifications,
flow analysis, protection policy, capex cost attribution, and retained budget
reporting remain intact.

Removing the planner must not remove manual CLN channel-management capability;
it removes only authority owned by this plugin. Human operators continue to
open and close channels using their established external workflow.

## 7. Compatibility and Data Policy

This is an intentional operator-surface contraction. Action RPCs and their
deprecated aliases are removed, not retained as dormant implementations.
Unknown-method errors are preferable to success-shaped tombstones that can be
mistaken for a functioning subsystem.

Database migrations are additive/non-destructive in this release:

- do not drop LN+, Boltz, or planner tables;
- do not rewrite historical spend or action rows;
- do not delete generic `no_close` policy rows;
- remove only live code that writes or acts from those rows.

The old release therefore remains a technical rollback candidate without a
reverse database migration. A rollback is allowed only after rechecking that it
will not reactivate Boltz, LN+, or planner authority from old defaults or stale
runtime configuration.

The live Lightning configuration requires an explicit cutover because the new
plugin will no longer register the retired options. Deployment prepares two
recoverable configuration artifacts:

- the active configuration with every retired Boltz, LN+, and planner plugin
  option removed; and
- a rollback configuration for the old release with every master/execution
  gate explicitly false, including LN+ gates whose old defaults were true.

Persisted database overrides remain inert and are not deleted in this release.
The new plugin may report them as unknown historical overrides, but it must not
apply them. A later schema-cleanup release may delete them after the rollback
window closes.

## 8. Rust-Port Alignment

The intended Rust whole-plugin target must be changed before any cutover can
occur. Rust LN+, Boltz, and planner-action implementations are not alternate
owners after Python removal. Their mutation capabilities, loop owners, RPCs,
manifest entries, configuration, and cutover requirements must be removed or
explicitly classified as historical/read-only.

Hexmem tasks whose purpose is to port these removed capabilities must be
superseded rather than completed and deployed. This cross-repository work is a
separate reviewed task; the Python repository change must not edit the Rust
repository implicitly.

## 9. Implementation Structure

Use an isolated worktree from current `origin/main`. Implement as reviewable
checkpoints in this order:

1. Characterization and architecture tests for retained behavior, active
   `no_close` persistence, startup without external swap dependencies, and the
   complete mutation-surface inventory.
2. LN+ contract-protection handoff and LN+ code/interface deletion.
3. Boltz code/interface and scheduler deletion.
4. Capacity-planner code/interface and loop deletion.
5. Cross-cutting config, database-helper, budget, docs, compatibility-catalog,
   package, and test cleanup.
6. Rust-port target and Hexmem-task supersession in a separately owned change.

Each checkpoint must leave the Python test suite runnable. Do not combine this
work with unrelated refactors or schema cleanup.

## 10. Verification

Verification is evidence-first and must include:

- focused tests for each removal checkpoint;
- the full Python test suite;
- architecture guards proving no Hive/Mycelium, Sling, or external coordinator
  dependency;
- mutation-path inventory proving no removed Boltz, LN+, or planner executor is
  registered or reachable;
- config and RPC inventories proving removed keys and methods are absent;
- startup tests with no `boltzcli`, `boltzd`, or LN+ network access;
- malformed and absent-data tests for retained read-only surfaces;
- policy tests proving pre-existing generic `no_close` tags survive plugin
  initialization without the LN+ module;
- tests proving no live action occurs from status, diagnostics, migration, or
  historical data;
- clean-tree and exact changed-file review.

Tests use mocks and local fakes only. They must not call live planner, LN+,
Boltz, CLN open/close, payment, withdrawal, or swap RPCs.

## 11. Deployment and Rollback

Deployment is a Tier 1 production change with an independent verifier.

1. Re-run every production precondition and save private evidence.
2. Write and verify the mode-0600 LN+ contract-expiry record.
3. Prepare the active cleaned configuration and the rollback configuration with
   all old execution gates explicitly false.
4. Stop the old plugin, install the reviewed commit series, activate the cleaned
   configuration, and start the new plugin through the established lifecycle.
5. Verify retained plugin health and fee/profitability/flow surfaces.
6. Verify removed RPC names are absent from `help` and no removed thread or
   external process is launched by the plugin.
7. Verify contract peers retain `no_close` protection.
8. Verify no new planner action, Boltz swap, LN+ row transition, channel open,
   channel close, payment, or withdrawal occurred during deployment.

Rollback reinstalls the previous reviewed plugin revision, but it must keep the
five planner shutdown overrides and all Boltz/LN+ execution controls false.
Rollback must not proceed if the old release would resume an obligation or
wallet action from ambiguous state. Because historical schemas and rows are
preserved, rollback requires no destructive data restoration.

## 12. Completion Criteria

The decommission is complete only when:

- current source contains no Boltz execution module, LN+ execution module, or
  capacity-planner module;
- no removed background loop, RPC, config key, subprocess, HTTP client, signing
  path, `fundchannel`, or planner `close` call is reachable;
- active manual LN+ contract channels remain protected by generic policy;
- retained fee, profitability, flow, policy, budget, and rebalancing behavior
  passes focused and full verification;
- the deployed node is healthy and independent review confirms zero unintended
  economic action;
- the Rust target architecture and durable task routing can no longer
  reintroduce the retired capabilities.
