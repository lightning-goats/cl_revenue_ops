# AGENTS.md — cl_revenue_ops

## Project identity

`cl_revenue_ops` is an independent Core Lightning plugin for local revenue operations.

It owns local execution:

- fee decisioning
- rebalance decisioning/execution
- planner/capex
- profitability analysis
- budget enforcement

cl-mycelium may enhance it through bounded hints, but `cl_revenue_ops` must run safely without cl-mycelium.

## Core invariants

- `cl_revenue_ops` must remain fully independent.
- It must run when cl-hive/cl-mycelium is absent.
- Missing hints must fall back to neutral behavior.
- Stale hints must fall back safely.
- Malformed hints must not crash the plugin.
- cl-mycelium hints may influence but must not command.
- No Sling dependency.
- Do not introduce a dependency on cl-mycelium.
- Do not trigger live action in tests unless explicitly scoped.
- Hermes/data collection must be read-only.

## Required reading before tasks

Read:

- `docs/plans/cl_mycelium_revenue_integrated_plan_v3.md`
- `docs/prompts/cl_mycelium_revenue_codex_prompt_pack_v3.md`
- `modules/hive_hints.py`
- `modules/lnplus_swaps.py`
- rebalance engine / planner / capex / profitability modules
- existing revenue RPC tests

## Action RPC warning

The following are action/mutation RPCs and must not be called in read-only tests or Hermes tasks:

- `revenue-rebalance-cycle`
- `revenue-fee-cycle`
- `revenue-planner-execute`
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
- `revenue-lnplus-abandon`
- `revenue-lnplus-breaker-clear`
- `revenue-lnplus-backfill`
- any Boltz action RPC
- any CLN open/close/pay/withdraw RPC

## Hint integration requirement

`modules/hive_hints.py` is the integration boundary.

Required behavior:

```text
no cl-hive -> neutral hints
missing datastore -> neutral hints
unknown hive-export-hints -> neutral hints
stale hints -> safe stale/neutral fallback
malformed hints -> safe neutral fallback
valid hints -> bounded influence
```

## Bounded influence

Hive/cl-mycelium hints may bias fee/rebalance decisions, but within bounded caps.

Do not allow hints to override local budget, safety, or executor policy.

## Required tests

When touching hint integration, add tests for:

- no cl-hive
- no `["hive","hints"]`
- stale hints
- malformed hints
- valid classic hints
- valid cl-mycelium hints
- neutral fallback
- no crash
- no live action triggered

## Required report format

For every Codex task, report:

- files changed
- tests run
- standalone behavior without cl-hive
- stale/malformed hint behavior
- no-Sling confirmation
- no action RPCs triggered
- production compatibility notes
- follow-up risks
