# 2026-03-27 Plugin RPC Matrix

## Scope

- Audit date: 2026-03-27
- Local node under test: `hive-nexus-01`
- Codebases:
  - `cl-revenue-ops` in `/home/sat/bin/cl_revenue_ops`
  - `cl-hive` in `/home/sat/bin/cl-hive`
- Method:
  - Extract the full RPC inventory from `@plugin.method(...)` decorators.
  - Cross-check a read-only live subset with MCP on `hive-nexus-01`.
  - Re-baseline after remediation so the matrix reflects the current source tree.

## Inventory Summary

| Plugin | RPC count | Test files | Primary entrypoint |
| --- | ---: | ---: | --- |
| `cl-revenue-ops` | 53 | 35 | `cl-revenue-ops.py` |
| `cl-hive` | 92 | 39 | `cl-hive.py` |

## Verification Snapshot

| Surface | Source status | Verification | Notes |
| --- | --- | --- | --- |
| `revenue-status` | Present | MCP OK | Returned running state and operator controls for `hive-nexus-01`. |
| `revenue-dashboard` | Present | MCP OK | Returned 30-day P&L and bleeder warning data. |
| `revenue-total-cost-budget` | Present | MCP OK + source re-audited | Budget path now excludes canonical open/close categories from generic ledger accounting in source. |
| `revenue-spend-ledger` | Present | MCP OK | Returned current spend/reservation view. |
| `hive-status` | Present | MCP OK | Returned governance mode and membership summary. |
| `hive-members` | Present | MCP OK | Returned member rows successfully. |
| `hive-rationalization-summary` | Present | MCP OK | Returned redundant/orphan counts successfully. |
| `hive-positioning-summary` | Present | MCP OK | Returned successfully. |
| `hive-rebalance-recommendations` | Removed from source | Source tests OK + MCP probe OK | Removed from `cl-hive` export surface, and a fresh MCP `tools/list` probe confirmed it is no longer advertised. |
| `advisor_get_status` | MCP wrapper surface | Unavailable | Still returned `Proactive advisor modules not available`; excluded from plugin findings. |

## Runtime Note

The active Codex MCP config still contains a stale approval stanza for `rebalance_recommendations`, but a fresh stdio MCP server spawned from that same config now reports 137 advertised tools and does not include `rebalance_recommendations` or `hive_rebalance_recommendations`. The live wrapper surface is therefore aligned with the updated source tree.

## Full `cl-revenue-ops` RPC Inventory

| RPC | Handler | Line |
| --- | --- | ---: |
| `revenue-status` | `revenue_status` | `1948` |
| `revenue-rebalance-debug` | `revenue_rebalance_debug` | `2001` |
| `revenue-fee-debug` | `revenue_fee_debug` | `2325` |
| `revenue-analyze` | `revenue_analyze` | `2423` |
| `revenue-wake-all` | `revenue_wake_all` | `2449` |
| `revenue-capacity-report` | `revenue_capacity_report` | `2470` |
| `revenue-planner-status` | `revenue_planner_status` | `2488` |
| `revenue-planner-candidates` | `revenue_planner_candidates` | `2496` |
| `revenue-planner-execute` | `revenue_planner_execute` | `2505` |
| `revenue-planner-history` | `revenue_planner_history` | `2513` |
| `revenue-set-fee` | `revenue_set_fee` | `2522` |
| `revenue-rebalance` | `revenue_rebalance` | `2566` |
| `revenue-profitability` | `revenue_profitability` | `2627` |
| `revenue-history` | `revenue_history` | `2796` |
| `revenue-ignore` | `revenue_ignore` | `2823` |
| `revenue-unignore` | `revenue_unignore` | `2861` |
| `revenue-list-ignored` | `revenue_list_ignored` | `2889` |
| `revenue-policy` | `revenue_policy` | `2927` |
| `revenue-report` | `revenue_report` | `3183` |
| `revenue-hot-channel-protection-peers` | `revenue_hot_channel_protection_peers` | `3324` |
| `revenue-config` | `revenue_config` | `3380` |
| `revenue-dashboard` | `revenue_dashboard` | `3462` |
| `revenue-health` | `revenue_health` | `3554` |
| `revenue-cleanup-closed` | `revenue_cleanup_closed` | `3730` |
| `revenue-clear-reservations` | `revenue_clear_reservations` | `3864` |
| `revenue-total-cost-budget` | `revenue_total_cost_budget` | `4644` |
| `revenue-spend-ledger` | `revenue_spend_ledger` | `4653` |
| `revenue-spend-reserve` | `revenue_spend_reserve` | `4673` |
| `revenue-spend-release` | `revenue_spend_release` | `4732` |
| `revenue-spend-release-stale` | `revenue_spend_release_stale` | `4743` |
| `revenue-spend-settle` | `revenue_spend_settle` | `4770` |
| `revenue-boltz-quote` | `revenue_boltz_quote` | `4793` |
| `revenue-boltz-loop-out` | `revenue_boltz_loop_out` | `4801` |
| `revenue-boltz-loop-in` | `revenue_boltz_loop_in` | `4813` |
| `revenue-boltz-status` | `revenue_boltz_status` | `4824` |
| `revenue-boltz-history` | `revenue_boltz_history` | `4832` |
| `revenue-boltz-external-pay-ignores` | `revenue_boltz_external_pay_ignores` | `4840` |
| `revenue-boltz-budget` | `revenue_boltz_budget` | `4848` |
| `revenue-boltz-wallet` | `revenue_boltz_wallet` | `4856` |
| `revenue-boltz-refund` | `revenue_boltz_refund` | `4864` |
| `revenue-boltz-claim` | `revenue_boltz_claim` | `4872` |
| `revenue-boltz-chainswap` | `revenue_boltz_chainswap` | `4880` |
| `revenue-boltz-withdraw` | `revenue_boltz_withdraw` | `4891` |
| `revenue-boltz-deposit` | `revenue_boltz_deposit` | `4903` |
| `revenue-boltz-backup` | `revenue_boltz_backup` | `4911` |
| `revenue-boltz-backup-verify` | `revenue_boltz_backup_verify` | `4919` |
| `revenue-boltz-balance-recommendations` | `revenue_boltz_balance_recommendations` | `5854` |
| `revenue-boltz-auto-cycle-status` | `revenue_boltz_auto_cycle_status` | `5896` |
| `revenue-boltz-auto-cycle-run-now` | `revenue_boltz_auto_cycle_run_now` | `5916` |
| `revenue-boltz-balance-cycle` | `revenue_boltz_balance_cycle` | `5927` |
| `revenue-boltz-expansion-treasury-status` | `revenue_boltz_expansion_treasury_status` | `6140` |
| `revenue-boltz-expansion-treasury-recommendations` | `revenue_boltz_expansion_treasury_recommendations` | `6165` |
| `revenue-boltz-expansion-treasury-cycle` | `revenue_boltz_expansion_treasury_cycle` | `6202` |

## Full `cl-hive` RPC Inventory

| RPC | Handler | Line |
| --- | --- | ---: |
| `hive-connect` | `hive_connect` | `1165` |
| `hive-health` | `hive_health` | `1176` |
| `hive-status` | `hive_status` | `1184` |
| `hive-config` | `hive_config` | `1193` |
| `hive-reload-config` | `hive_reload_config` | `1210` |
| `hive-reinit-bridge` | `hive_reinit_bridge` | `1231` |
| `hive-members` | `hive_members` | `1242` |
| `hive-topology` | `hive_topology` | `1251` |
| `hive-expansion-recommendations` | `hive_expansion_recommendations` | `1262` |
| `hive-channel-closed` | `hive_channel_closed` | `1281` |
| `hive-channel-opened` | `hive_channel_opened` | `1364` |
| `hive-peer-events` | `hive_peer_events` | `1413` |
| `hive-peer-quality` | `hive_peer_quality` | `1497` |
| `hive-quality-check` | `hive_quality_check` | `1578` |
| `hive-calculate-size` | `hive_calculate_size` | `1628` |
| `hive-planner-log` | `hive_planner_log` | `1770` |
| `hive-planner-ignore` | `hive_planner_ignore` | `1784` |
| `hive-planner-unignore` | `hive_planner_unignore` | `1841` |
| `hive-planner-ignored-peers` | `hive_planner_ignored_peers` | `1884` |
| `hive-test-intent` | `hive_test_intent` | `1920` |
| `hive-intent-status` | `hive_intent_status` | `1981` |
| `hive-fee-profiles` | `hive_fee_profiles` | `1994` |
| `hive-fee-recommendation` | `hive_fee_recommendation` | `2039` |
| `hive-fee-intelligence` | `hive_fee_intelligence` | `2081` |
| `hive-aggregate-fees` | `hive_aggregate_fees` | `2117` |
| `hive-fee-intel-query` | `hive_fee_intel_query` | `2146` |
| `hive-report-fee-observation` | `hive_report_fee_observation` | `2212` |
| `hive-trigger-fee-broadcast` | `hive_trigger_fee_broadcast` | `2298` |
| `hive-trigger-health-report` | `hive_trigger_health_report` | `2326` |
| `hive-trigger-all` | `hive_trigger_all` | `2363` |
| `hive-nnlb-status` | `hive_nnlb_status` | `2419` |
| `hive-member-health` | `hive_member_health` | `2443` |
| `hive-report-health` | `hive_report_health` | `2513` |
| `hive-calculate-health` | `hive_calculate_health` | `2590` |
| `hive-peer-reputations` | `hive_peer_reputations` | `2654` |
| `hive-reputation-stats` | `hive_reputation_stats` | `2714` |
| `hive-liquidity-needs` | `hive_liquidity_needs` | `2737` |
| `hive-liquidity-status` | `hive_liquidity_status` | `2772` |
| `hive-liquidity-state` | `hive_liquidity_state` | `2795` |
| `hive-report-liquidity-state` | `hive_report_liquidity_state` | `2833` |
| `hive-update-rebalancing-activity` | `hive_update_rebalancing_activity` | `2878` |
| `hive-check-rebalance-conflict` | `hive_check_rebalance_conflict` | `2912` |
| `hive-report-traffic-profile` | `hive_report_traffic_profile` | `2926` |
| `hive-traffic-intelligence` | `hive_traffic_intelligence` | `2951` |
| `hive-fleet-demand-forecast` | `hive_fleet_demand_forecast` | `2962` |
| `hive-export-hints` | `hive_export_hints` | `2969` |
| `hive-bump-version` | `hive_bump_version` | `2993` |
| `hive-gossip-stats` | `hive_gossip_stats` | `3037` |
| `hive-ban` | `hive_ban` | `3083` |
| `hive-leave` | `hive_leave` | `3172` |
| `hive-remove-member` | `hive_remove_member` | `3239` |
| `hive-contribution` | `hive_contribution` | `3364` |
| `hive-network-metrics` | `hive_network_metrics` | `3380` |
| `hive-rebalance-hubs` | `hive_rebalance_hubs` | `3398` |
| `hive-fleet-health` | `hive_fleet_health` | `3424` |
| `hive-connectivity-alerts` | `hive_connectivity_alerts` | `3439` |
| `hive-member-connectivity` | `hive_member_connectivity` | `3456` |
| `hive-fee-reports` | `hive_fee_reports` | `3471` |
| `hive-yield-metrics` | `hive_yield_metrics` | `3517` |
| `hive-yield-summary` | `hive_yield_summary` | `3532` |
| `hive-velocity-prediction` | `hive_velocity_prediction` | `3546` |
| `hive-critical-velocity` | `hive_critical_velocity` | `3561` |
| `hive-coord-fee-recommendation` | `hive_coord_fee_recommendation` | `3578` |
| `hive-egress-desaturation-bias` | `hive_egress_desaturation_bias` | `3613` |
| `hive-corridor-assignments` | `hive_corridor_assignments` | `3637` |
| `hive-record-routing-outcome` | `hive_record_routing_outcome` | `3651` |
| `hive-ban-candidates` | `hive_ban_candidates` | `3696` |
| `hive-get-routing-intelligence` | `hive_get_routing_intelligence` | `3718` |
| `hive-fee-coordination-status` | `hive_fee_coordination_status` | `3732` |
| `hive-coverage-analysis` | `hive_coverage_analysis` | `3751` |
| `hive-close-recommendations` | `hive_close_recommendations` | `3768` |
| `hive-rationalization-summary` | `hive_rationalization_summary` | `3783` |
| `hive-rationalization-status` | `hive_rationalization_status` | `3797` |
| `hive-valuable-corridors` | `hive_valuable_corridors` | `3814` |
| `hive-exchange-coverage` | `hive_exchange_coverage` | `3831` |
| `hive-positioning-recommendations` | `hive_positioning_recommendations` | `3845` |
| `hive-positioning-summary` | `hive_positioning_summary` | `3860` |
| `hive-positioning-status` | `hive_positioning_status` | `3873` |
| `hive-genesis` | `hive_genesis` | `3884` |
| `hive-repair-member` | `hive_repair_member` | `3914` |
| `hive-join` | `hive_join` | `4040` |
| `hive-approve` | `hive_approve` | `4090` |
| `hive-pending` | `hive_pending` | `4164` |
| `hive-time-fee-status` | `hive_time_fee_status` | `4191` |
| `hive-time-fee-adjustment` | `hive_time_fee_adjustment` | `4207` |
| `hive-time-peak-hours` | `hive_time_peak_hours` | `4227` |
| `hive-time-low-hours` | `hive_time_low_hours` | `4251` |
| `hive-routing-intelligence-status` | `hive_routing_intelligence_status` | `4275` |
| `hive-get-peer-quality` | `hive_get_peer_quality` | `4284` |
| `hive-get-channel-flags` | `hive_get_channel_flags` | `4305` |
| `hive-get-nnlb-opportunities` | `hive_get_nnlb_opportunities` | `4325` |
| `hive-get-channel-ages` | `hive_get_channel_ages` | `4347` |
