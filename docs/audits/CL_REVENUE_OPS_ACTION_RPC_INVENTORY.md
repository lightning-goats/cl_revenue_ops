# cl_revenue_ops action RPC inventory

Current contract: plugin version 3.0.0, 2026-08-03.

This inventory describes the registered runtime surface after removal of
CapacityPlanner, automatic channel open/close, planner defibrillation, Boltz,
and LN+. The plugin registers 39 methods. Removed method families are not
deprecated aliases and are not safe to probe by invoking them.

## Classification

### External economic or node mutations

These methods may change fees, send a rebalance payment, or dispatch a cycle
that can do so. They are never safe for read-only collectors:

- `revenue-rebalance-cycle`
- `revenue-fee-cycle`
- `revenue-set-fee`
- `revenue-rebalance`
- `revenue-wake-all`
- `revenue-cycle` when the selected phase can execute

The rebalance paths remain governed by pause, authority, policy, route,
profitability, and atomic daily/weekly/global/channel budget checks.

### Local state mutations or mixed read/write methods

These methods mutate plugin policy, configuration, accounting state, or
observational shadow state. A method with a read action is still mixed and
must not be whitelisted wholesale:

- `revenue-analyze`
- `revenue-ignore`
- `revenue-unignore`
- `revenue-ban`
- `revenue-unban`
- `revenue-policy`
- `revenue-config`
- `revenue-hot-channel-protection-peers`
- `revenue-econ-reconcile` when `apply=true`
- `revenue-econ-cycle`
- `revenue-cleanup-closed`
- `revenue-clear-reservations`
- `revenue-spend-reserve`
- `revenue-spend-release`
- `revenue-spend-release-stale`
- `revenue-spend-settle`
- `revenue-capex-status` because its retained read also refreshes datastore telemetry
- `revenue-budget` with no section because it includes the capex telemetry refresh

### Read-only methods

These methods do not intentionally execute a payment, fee change, channel
lifecycle action, swap, or local policy mutation:

- `revenue-status`
- `revenue-rebalance-debug`
- `revenue-fee-debug`
- `revenue-fee-authority-status`
- `revenue-profitability`
- `revenue-history`
- `revenue-list-ignored`
- `revenue-list-banned`
- `revenue-report`
- `revenue-dashboard`
- `revenue-health`
- `revenue-econ-snapshot`
- `revenue-profile-preview`
- `revenue-total-cost-budget`
- `revenue-spend-ledger`
- `revenue-budget section=ledger`

Collectors should still use a deny-by-default allowlist and pin the method
name and subaction rather than accepting arbitrary arguments.

## Removed authority

The following families must be absent from `lightning-cli help` after a v3
cutover:

- every `revenue-planner*` method
- every `revenue-boltz*` method
- every `revenue-lnplus*` method
- planner phases in `revenue-cycle`
- planner or Boltz sections in `revenue-budget`

The plugin has no `fundchannel`, `close`, or wallet-withdraw execution
caller. Historical `planner_*`, `lnplus_*`, and spend-ledger rows for
`channel_open`, `channel_close`, or `boltz` remain readable for audit and
accounting only; they cannot authorize work.

## Architecture constraints

- No Sling dependency.
- No hive, mycelium, or fleet coordinator.
- Reporting and profitability remain available.
- Circular rebalancing remains functional and budget constrained.
- Read-only tests and Hermes collection must not call action or mixed methods.
