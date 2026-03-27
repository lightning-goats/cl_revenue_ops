# Full Plugin Audit Design

**Date:** 2026-03-27
**Scope:** Full correctness, safety, and logic audit of [cl-revenue-ops.py](/home/sat/bin/cl_revenue_ops/cl-revenue-ops.py) and [cl-hive.py](/home/sat/bin/cl-hive/cl-hive.py), using current MCP read-only evidence from `hive-nexus-01` and code review as the source of truth.
**Approach:** Contract-led split audit. Build a code-defined RPC inventory first, compare it to MCP-observable behavior on the live node, then do targeted static review of the highest-risk subsystems and write a severity-ranked report with evidence quality noted for each finding.

## Current Constraints

- The only live node is `hive-nexus-01`; fleet-style assumptions are no longer valid.
- Read-only MCP probes are allowed. No live mutating RPCs should be used during the audit.
- `revenue-status` and `hive-status` are available and useful for runtime evidence.
- `revenue_ops_health` is currently degraded because `revenue-dashboard` is timing out.
- Proactive advisor surfaces are not currently available, so advisor state is not reliable live evidence.

## Audit Goals

The audit should answer five questions:

1. Which RPCs and internal decision paths exist in each plugin?
2. Which live read-only surfaces behave differently from the implementation contract?
3. Where can the code make a wrong decision even when the RPC call succeeds?
4. Where do safety controls fail open, race, over-trust stale data, or silently degrade correctness?
5. Which existing tests cover the risky paths, and where are the coverage gaps?

## Evidence Rules

Every finding must be tagged with one of these evidence levels:

- **Live + code:** confirmed by MCP behavior and traced to code.
- **Code-only:** defensible by code inspection, but not directly exercised live.
- **Runtime-only:** visible through MCP behavior, but root cause still ambiguous in code.

Every finding must also be tagged by impact:

- **Bug:** deterministic incorrect behavior.
- **Logic flaw:** design/decision path can choose the wrong action under valid inputs.
- **Safety flaw:** incorrect failure mode, missing guardrail, or live mutation risk.
- **Correctness drift:** code contract and live observable behavior disagree.
- **Test gap:** risky path lacks direct regression coverage.

## Audit Structure

### Phase 1: RPC And Test Inventory

Build the inventory for both plugins from the codebases:

- All `@plugin.method` and `@plugin.async_method` surfaces.
- The handler function for each surface.
- Whether the RPC is read-only or mutating.
- The internal subsystems each handler touches.
- Existing direct tests covering the handler or subsystem.

This phase produces the audit map and prevents random reading.

### Phase 2: Live Contract Audit Through MCP

Use MCP only for read-only verification on `hive-nexus-01`:

- Status and health surfaces
- Debug and explainability surfaces
- Financial and spend surfaces
- Hive membership and topology surfaces
- Planner/reporting/recommendation surfaces
- Boltz visibility surfaces

The goal is not to test every method live. The goal is to identify contract drift, timeouts, stale-data hazards, and surprising runtime behavior.

### Phase 3: Static Deep Audit Of `cl-revenue-ops`

Prioritize the highest-risk subsystems:

- Fee controller and fee-setting execution
- Rebalancer candidate selection, EV filtering, and execution gating
- Unified cost budget and spend ledger reservation flow
- Boltz reserve/treasury/balance cycle logic
- Capacity planner opens/closes and recommendation-vs-execution safety
- Runtime config and policy writes
- Explainability/status/debug surfaces

Key questions:

- Can the plugin over-spend, double-count, or misreport budget state?
- Can stale or sparse data produce unsafe fee or rebalance decisions?
- Do debug/status surfaces accurately reflect live decision state?
- Do dry-run or recommendation-only paths actually stay non-mutating?

### Phase 4: Static Deep Audit Of `cl-hive`

Prioritize the highest-risk subsystems:

- Membership and governance state transitions
- Channel planning/open/close recommendation logic
- Fee coordination and corridor ownership logic
- Gossip/state synchronization and stale-state handling
- Connectivity/rationalization/positioning recommendations
- Health/status/reporting surfaces

Key questions:

- Can membership, governance, or planner paths create inconsistent state?
- Can recommendation logic silently drift from the live single-node reality?
- Are coordination decisions based on stale, missing, or invalid data without clear fail-closed behavior?

### Phase 5: Cross-Cutting Safety Review

For both plugins, explicitly review:

- Fail-open vs fail-closed behavior
- Timeout and retry semantics
- Concurrency and shared-state mutation
- Stale snapshot handling
- Idempotency of mutating flows
- Approval gates and recommendation-only promises
- Input validation and schema assumptions
- Logging/explainability mismatches

### Phase 6: Findings Synthesis

Produce:

- A single severity-ranked report
- An RPC drift matrix
- A list of missing tests by subsystem
- A short remediation sequence ordered by risk reduction

## Deliverables

1. `docs/audits/2026-03-27-full-plugin-audit-report.md`
2. `docs/audits/2026-03-27-plugin-rpc-matrix.md`
3. A short executive summary of top risks and immediate fixes

## Recommendation

Use the contract-led split audit, keep all live probes read-only, and treat the code as authoritative when MCP surfaces are stale or partially unavailable. This keeps the report defensible and avoids conflating observability issues with plugin correctness.
