# Config Field Classification (Phase D, PR 7 — 2026-07-13)

Source of truth: `modules/risk_profiles.py:FIELD_CLASSIFICATION`
(coverage-pinned by `tests/test_risk_profiles.py` — a new Config
field cannot ship unclassified). Generated table; regenerate on
change.

## safety_invariant (13) — Hard rails, reserves, ceilings — profiles NEVER touch these

| Field | Default |
|---|---|
| `diagnostic_rebalance_max_fee_sats` | `400` |
| `expansion_treasury_exclude_protected` | `True` |
| `expansion_treasury_min_source_local_pct` | `80.0` |
| `growth_budget_hard_ceiling_sats` | `10000` |
| `lnplus_apply_feerate_ceiling` | `5000` |
| `max_fee_ppm` | `2000` |
| `min_fee_ppm` | `10` |
| `min_fee_ppm_saturated` | `0` |
| `min_wallet_reserve` | `1000000` |
| `planner_close_fee_cap_sats` | `0` |
| `planner_close_fee_reserve_multiplier` | `2.0` |
| `planner_max_fee_rate_sat_vb` | `50.0` |
| `receivable_ratio_floor` | `0.2` |

## authority_control (23) — What the node MAY do — pause, dry-run, authority level, capability gates

| Field | Default |
|---|---|
| `authority_level` | `'capital'` |
| `boltz_auto_cycle_enabled` | `False` |
| `dry_run` | `False` |
| `econ_arbiter_enabled` | `False` |
| `econ_cycle_boltz_enabled` | `False` |
| `econ_cycle_planner_enabled` | `False` |
| `econ_cycle_rebalance_enabled` | `False` |
| `econ_ev_populated` | `False` |
| `econ_governor_boltz_enabled` | `False` |
| `econ_governor_fees_enabled` | `False` |
| `econ_governor_lnplus_enabled` | `False` |
| `econ_governor_planner_enabled` | `False` |
| `econ_governor_rebalance_enabled` | `False` |
| `econ_shadow_enabled` | `False` |
| `expansion_treasury_enabled` | `False` |
| `fee_authority_enabled` | `True` |
| `lnplus_execute_applications` | `True` |
| `lnplus_swaps_enabled` | `True` |
| `paused` | `False` |
| `planner_dry_run` | `False` |
| `planner_enabled` | `False` |
| `planner_execute_closes` | `False` |
| `risk_profile` | `'custom'` |

## economic_risk (26) — Risk preferences — the ONLY profile-bundleable class

| Field | Default |
|---|---|
| `boltz_structural_budget_sats_per_day` | `0` |
| `capex_exploration_rate` | `0.1` |
| `capex_global_envelope_sats` | `0` |
| `capex_probability_budget_bonus` | `0.0` |
| `capex_reinvestment_rate` | `0.5` |
| `capex_tactical_rate` | `0.15` |
| `daily_budget_sats` | `5000` |
| `expansion_treasury_min_deficit_sats` | `250000` |
| `expansion_treasury_onchain_target_sats` | `5000000` |
| `fee_profile` | `'active'` |
| `growth_budget_earned_fraction` | `0.25` |
| `growth_budget_enabled` | `False` |
| `growth_budget_experiment_fraction` | `0.1` |
| `growth_budget_max_extra_sats` | `2000` |
| `lnplus_max_duration_months` | `3` |
| `lnplus_swap_preference_margin` | `0.2` |
| `pair_fee_cap_ppm` | `1000` |
| `planner_max_channel_sats` | `10000000` |
| `planner_max_closes_per_cycle` | `0` |
| `planner_max_opens_per_cycle` | `1` |
| `planner_min_annual_roi_pct` | `1.0` |
| `planner_min_channel_sats` | `500000` |
| `rebalance_hold_margin` | `0.0` |
| `rebalance_max_amount` | `5000000` |
| `receivable_ratio_target` | `0.3` |
| `weekly_budget_sats` | `35000` |

## operational_timing (15) — Schedules, pacing, timeouts

| Field | Default |
|---|---|
| `boltz_auto_cycle_interval_minutes` | `15` |
| `boltz_auto_cycle_max_actions` | `1` |
| `boltz_auto_cycle_startup_delay_seconds` | `120` |
| `capex_grace_days` | `14` |
| `expansion_treasury_max_actions` | `1` |
| `fee_interval` | `1800` |
| `flow_interval` | `3600` |
| `lnplus_pending_timeout_days` | `7` |
| `lnplus_watcher_interval` | `3600` |
| `max_concurrent_jobs` | `5` |
| `planner_interval` | `21600` |
| `rebalance_cooldown_hours` | `24` |
| `rebalance_interval` | `900` |
| `reservation_timeout_hours` | `4` |
| `rpc_timeout_seconds` | `15` |

## external_integration (5) — Paths, routers, currencies

| Field | Default |
|---|---|
| `askrene_layer` | `'xpay'` |
| `askrene_layers` | `'standalone'` |
| `db_path` | `'~/.lightning/revenue_ops.db'` |
| `expansion_treasury_preferred_currency` | `'BTC'` |
| `rebalance_router` | `'v3'` |

## advanced_expert (59) — Algorithm tuning — expert-only overrides

| Field | Default |
|---|---|
| `allow_zero_cost_auto_rebalance_when_budget_zero` | `False` |
| `base_fee_msat` | `0` |
| `base_fee_policy` | `'off'` |
| `capex_bootstrap_bps` | `10` |
| `capex_bootstrap_max_sats` | `200` |
| `drain_fee_discount_max` | `0.0` |
| `enable_dynamic_htlcmax` | `False` |
| `enable_kelly` | `False` |
| `enable_reputation` | `True` |
| `enable_vegas_reflex` | `True` |
| `estimated_open_cost_sats` | `5000` |
| `fee_market_boundary_cache_seconds` | `60` |
| `fee_market_boundary_enabled` | `False` |
| `fee_market_boundary_margin_ppm` | `5` |
| `fee_market_boundary_margin_ratio` | `0.05` |
| `fee_market_boundary_max_downshift_ratio` | `0.35` |
| `fee_market_boundary_min_competitors` | `3` |
| `flow_window_days` | `7` |
| `high_liquidity_threshold` | `0.7` |
| `hot_channel_protection_enabled` | `True` |
| `hot_channel_protection_max_chunk_multiplier` | `4.0` |
| `hot_channel_protection_min_cooldown_hours` | `1.0` |
| `hot_channel_protection_min_marginal_roi` | `0.2` |
| `hot_channel_protection_min_velocity` | `0.2` |
| `hot_channel_protection_override_peers` | `''` |
| `hot_channel_protection_profit_budget_pct` | `0.75` |
| `htlc_congestion_threshold` | `0.8` |
| `htlcmax_balanced_pct` | `0.45` |
| `htlcmax_sink_pct` | `0.25` |
| `htlcmax_source_pct` | `0.5` |
| `inbound_fee_estimate_ppm` | `50` |
| `lnplus_inbound_credit_factor` | `0.5` |
| `lnplus_max_participants` | `4` |
| `lnplus_min_participants` | `3` |
| `lnplus_min_peer_positive_ratings` | `5` |
| `lnplus_min_peer_rank` | `8` |
| `low_liquidity_threshold` | `0.3` |
| `market_fee_mode` | `'undercut'` |
| `neighbor_median_min_competitors` | `2` |
| `node_drain_bias_enabled` | `False` |
| `node_drain_bias_max` | `0.3` |
| `planner_close_feerange_enabled` | `False` |
| `rebalance_activity_penalty_cap_frac` | `0.5` |
| `rebalance_activity_penalty_coeff` | `0.5` |
| `rebalance_activity_window_seconds` | `3600` |
| `rebalance_drift_override_ratio` | `0.3` |
| `rebalance_emergency_local_ratio` | `0.1` |
| `rebalance_size_reference_percentile` | `0.5` |
| `rebalance_size_tiered_targets` | `True` |
| `rebalance_small_channel_band_half_width` | `0.15` |
| `rebalance_utilization_ceiling` | `1.0` |
| `rebalance_utilization_floor` | `0.05` |
| `rebalance_utilization_min_forwards` | `5` |
| `rebalance_utilization_window_days` | `7` |
| `reputation_decay` | `0.98` |
| `sink_threshold` | `-0.05` |
| `source_threshold` | `0.05` |
| `thompson_prior_std_fee` | `100` |
| `vegas_decay_rate` | `0.85` |

## deprecated_transition (1) — Announced-removal compatibility shims

| Field | Default |
|---|---|
| `rebalance_min_profit` | `10` |

## Profile bundles (economic_risk keys only)

| Key | preserve | conservative (=defaults) | balanced | growth |
|---|---|---|---|---|
| `daily_budget_sats` | `2000` | `5000` | `8000` | `12000` |
| `growth_budget_earned_fraction` | `0.1` | `0.25` | `0.25` | `0.4` |
| `growth_budget_enabled` | `False` | `False` | `True` | `True` |
| `growth_budget_experiment_fraction` | `0.05` | `0.1` | `0.1` | `0.2` |
| `growth_budget_max_extra_sats` | `1000` | `2000` | `2000` | `5000` |
| `lnplus_swap_preference_margin` | `0.5` | `0.2` | `0.2` | `0.1` |
| `planner_max_closes_per_cycle` | `0` | `0` | `1` | `1` |
| `planner_max_opens_per_cycle` | `0` | `1` | `1` | `2` |
| `planner_min_annual_roi_pct` | `5.0` | `1.0` | `1.0` | `0.5` |
| `rebalance_hold_margin` | `5.0` | `0.0` | `0.0` | `0.0` |
| `weekly_budget_sats` | `14000` | `35000` | `56000` | `84000` |

`custom` derives nothing (exact current behavior; the default
and the migration value for existing deployments). Precedence:
explicit override > profile bundle > dataclass default. Applied
at startup only; runtime profile changes take effect at plugin
restart, after preview/diff (PR 8). Activation of a non-custom
profile on production requires separate operator direction.
