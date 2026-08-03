# AGENTS.md — cl_revenue_ops

## Project identity

`cl_revenue_ops` is an independent, standalone Core Lightning plugin for local revenue operations.

It owns local execution:

- fee decisioning
- rebalance decisioning/execution
- rebalance budget allocation
- revenue reporting
- profitability analysis
- budget enforcement

It has no external coordinator: the cl-mycelium/cl-hive hint integration was
retired and fully removed in 2026-07 (see `docs/audit/HIVE_REMOVAL_PLAN.md`).

## Core invariants

- `cl_revenue_ops` is fully independent — all decisions run on local evidence
  (own forwards, gossip, node state) only.
- No hive/mycelium/fleet coordination code may return
  (`tests/test_architecture_guard.py` pins this).
- No Sling dependency.
- Do not trigger live action in tests unless explicitly scoped.
- Hermes/data collection must be read-only.

## Required reading before tasks

Read:

- `README.md`
- rebalance engine, capex budget, profitability, and reporting modules
- existing revenue RPC tests

## Action RPC warning

The following are action/mutation RPCs and must not be called in read-only tests or Hermes tasks:

- `revenue-rebalance-cycle`
- `revenue-fee-cycle`
- `revenue-set-fee`
- `revenue-rebalance`
- `revenue-spend-reserve`
- `revenue-spend-release`
- `revenue-spend-release-stale`
- `revenue-spend-settle`
- `revenue-analyze`
- `revenue-wake-all`
- `revenue-ignore`
- `revenue-unignore`
- `revenue-cleanup-closed`
- `revenue-clear-reservations`
- `revenue-policy set`
- `revenue-config set`
- any direct CLN mutation RPC; this plugin has no open, close, or withdraw caller

## Required tests

When touching decision paths, add tests for:

- neutral/absent-data fallback (missing DB rows, empty gossip, RPC errors)
- no crash on malformed inputs
- no live action triggered from read-only surfaces

## Required report format

For every Codex task, report:

- files changed
- tests run
- no-Sling confirmation
- no action RPCs triggered
- production compatibility notes
- follow-up risks
