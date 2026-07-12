# Decision-owner matrix (baseline 5e8f747)

Prior art: `docs/audit/decision-loops.md` (loop verdicts),
`docs/audit/operator-decisions.md` + `docs/audit/deep/operator-decisions-deep.md`
(operator rulings D1–D4, DD1+).

Each row: the decision, its single current owner (or duplicated owners —
a refactor target), the exact entry-point seam, and what it consumes.

| Decision | Owner | Entry seam | Consumes |
|---|---|---|---|
| Fee target per channel | `FeeController` | `_adjust_channel_fee` (`modules/fee_controller.py:5465`); cycle `adjust_all_fees` (:4379) | db (DTS/cycle state), config snapshot, channel_info, profitability |
| Fee damping/rails/deadband | `FeeController` | `_apply_damped_fee_target` (:5241), `_get_fee_step_cap` (:5085), `_apply_zero_flow_ratchet_guard` (:5306), `_calculate_floor` (:7957) | fee profile, chain costs |
| Dynamic htlc_max | `FeeController` (embedded — Workstream F3 wants it out) | `_compute_dynamic_htlcmax_msat` (:2874), deadband `_htlcmax_delta_exceeds_deadband` (:2913) | cfg pcts, channel_info |
| Rebalance pair selection | `RebalanceEngine` + `RebalancePlanner` | `RebalanceEngine.find_candidates` (`modules/rebalance_engine_v2.py:1217`) → `RebalancePlanner.plan` (`modules/rebalance_planner_v2.py:110`) | StateSnapshot (pure), capex budgets, profitability |
| Rebalance execution & max-cost | `RebalanceEngine` | `execute_candidate` (:2879), `run_cycle` (:3025); `_pair_policy_allowed` (:2393); `_pair_max_fee_sats` (:1829) | policy_manager, budget ledger |
| Profitability class & ROI | `ChannelProfitabilityAnalyzer` | `analyze_channel` (`modules/profitability_analyzer.py:760`); `_classify_channel` (:2656) | db P&L, bookkeeper, config |
| Channel economic role (revenue) | `ChannelProfitabilityAnalyzer` | `ChannelRole` (:151); 30d window `ChannelProfitability.role_30d` (:396) | forward counts 30d/lifetime |
| Channel flow/balance state | `FlowAnalyzer` (**duplicate classification authority** vs profitability role — Workstream A target) | `_analyze_channel_impl` (`modules/flow_analysis.py:1792`), `_classify_balance_position` (:1904), `ChannelState` enum (:652) | kalman ratio, db flow state |
| Open candidates & ranking | `CapacityPlanner` | `generate_report` (`modules/capacity_planner.py:210`), `_score_candidate` (:2188), `get_candidate_sources` (:2384) | profitability, flow, policy |
| Close recommendation & protection | `CapacityPlanner` | `_close_protection_reason` (:1096, single source of truth), `_check_close_allowed` (:3426, policy tags, fail-closed), exec gate in `execute_cycle` (:339/:450) | profitability `role_30d`, flow confidence, policy tags |
| Boltz swap mode/plan | module-level in `cl-revenue-ops.py` (**not a class** — adapter boundary target) | `_run_boltz_auto_cycle_once` (:2019), `_select_boltz_auto_cycle_mode` (:1926), `_build_boltz_expansion_treasury_plan` (:8297), `_build_boltz_balance_plan` (:8475) | config snapshot, boltz_manager, planner |
| Boltz execution | `BoltzCliManager` | `loop_in` (:1751), `loop_out` (:1851), budget `check_tactical_budget` (:289) (`modules/boltz_manager.py`) | boltzcli subprocess, swap journal |
| LN+ qualification | `SwapEvaluator` | `run_cycle` (`modules/lnplus_swaps.py:268`), `_filter_swap` (:313), `_check_participants` (:345), `_select_and_apply` (:526) | LN+ HTTP client, db, policy bans, planner reputation |
| Budgets | FOUR implementations (see mutation-paths.md §budget) — Workstream D unifies | db spend ledger / db rebalance reservations / capex_budget / growth_budget | — |
| Protections | Policy tags (`PolicyManager`, no_close), close-protection gates, LN+ contract windows, hot-channel overrides (`hot_channel_protection_overrides` table) — DISTRIBUTED, Workstream F5 unifies | — | — |

## Known duplications (refactor targets, confirmed at baseline)

1. Channel classification: `FlowAnalyzer.ChannelState` (flow/balance) vs
   `ChannelRole`/`role_30d` (revenue) — two authorities, different enums.
2. Budgets: four implementations (above).
3. Rebalance modes: hot-channel protection, normal, structural drain,
   manual, diagnostic have distinct paths into the engine (Workstream F4).
4. Boltz decision logic lives in the plugin entry file, not the manager.
