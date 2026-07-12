# EconomicSnapshot v0 → current data sources

Schema: `schemas/economic_snapshot.v0.schema.json` (draft; validity test
`tests/test_schema_validity.py`). Every field mapped to the code that
produces the equivalent value today.

| Field | Current source | Notes |
|---|---|---|
| channel_id / peer_id / capacity_msat / local_msat / spendable_msat / receivable_msat | `listpeerchannels` via data_service cached reads (`modules/data_service.py`) | remote_msat = capacity − local |
| exit_revenue_msat | `ChannelRevenue.fees_earned_msat` (`modules/profitability_analyzer.py:~300`; db `forwards`/`lifetime_aggregates`) | |
| sourced_value_msat | `ChannelRevenue.sourced_fee_contribution_msat` | 30d window: `ChannelProfitability.sourced_fee_30d_msat` |
| rebalance_cost_msat | `ChannelCosts.rebalance_cost_sats * 1000` (db `rebalance_costs`) | plus `effective_rebalance_cost_sats` variant |
| capital_cost_msat | `ChannelCosts.open_cost_sats * 1000` + capital-efficiency carry (`modules/capital_efficiency.py`) | |
| net_value_msat | `ChannelProfitability.net_profit_sats * 1000` | today computed in sats — sub-sat precision is lost upstream; v1 decides where rounding happens |
| exit_volume_msat / sourced_volume_msat | `ChannelRevenue.volume_routed_msat` / sourced volume fields | |
| forward_count / sourced_forward_count | `ChannelRevenue.forward_count`, `sourced_forward_count_30d` etc. | `total_forward_count_30d` is DERIVED (property, profitability_analyzer.py:391) — schema models components only |
| role | UNIFICATION REQUIRED: profitability `ChannelRole`/`role_30d` (revenue-based) vs `flow_analysis.ChannelState` (balance/flow-based) — two live authorities (`decision-owners.md`) | schema enum is the UNION of both vocabularies; v1 narrows once the single classification authority exists (Workstream A) |
| lifecycle | DOES NOT EXIST today — derivable: `dead_capital_stage` → RECYCLING/CLOSING, `planner_actions` → OPENING, LN+/no_close tags → PROTECTED | Workstream F5 introduces the explicit model |
| protections | policy tags (`peer_policies`), LN+ contract windows (`lnplus_swaps`), `hot_channel_protection_overrides` | become owned, expiring `Protection` records |
| confidence_micro | flow-analysis kalman confidence (float 0..1 → micro fixed-point) | float→micro conversion rule in `wire-contract-spec.md` |
| node.total_local_msat / total_remote_msat | `listpeerchannels` aggregation (data_service cached) | |
| node.receivable_objective_msat | `receivable_ratio_target` config × capacity (Boltz balance planning inputs) | |
| node.onchain_confirmed_msat | `listfunds` confirmed outputs | |
| node.reserved_msat / daily_budget | db `get_budget_status` / spend ledger (`spend_reservations` + `spend_events`) | FOUR budget systems today — Workstream D unifies before v1 freeze |
| node.pending_operations | in-flight rebalances (`rebalance_history` open rows), Boltz journal, planner staged actions | |
| node.external_obligations | `lnplus_swaps` in-flight + Boltz journal | invariant 6 |
| snapshot_id / observed_at / evidence_window_seconds | NEW: cycle context (Workstream H / J3) | replaces scattered `time.time()` reads (`portability-hazards.md` §1) |
