# Public compatibility catalog (liquidity executors retired)

What the refactor MUST keep working (refactor invariants 2 and 3).
Pin test: `tests/test_rpc_surface_inventory.py` (37 retained methods).

## RPC surface

Primary operator surfaces (must remain schema-compatible; per
refactor.md Workstream I these become facades over projections):
`revenue-status`, `revenue-fee-authority-status`, `revenue-fee-debug`,
`revenue-rebalance-debug`, `revenue-forward-history`,
`revenue-config get|set`, `revenue-profitability`, `revenue-analyze`,
`revenue-wake-all`, `revenue-dashboard`, `revenue-health`.

Action/mutation RPCs (AGENTS.md list; execution-gated): see AGENTS.md
"Action RPC warning".
Classification per method: `docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md`
(current v3 inventory, refreshed 2026-08-13).

Full 37-method list: `EXPECTED_RPC_METHODS` in the pin test is normative.

Phase C retained operator-surface dispatchers: `revenue-cycle <subsystem>`,
`revenue-budget [section]`, and the diagnostic `revenue-policy` actions are the
primary operator names.

## Config surface

Owner: `modules/config.py` — a single `Config` dataclass (102 fields)
plus the `config_overrides` table (`revenue-config set` persists
overrides; precedence documented in README.md §revenue-config).

- Runtime-settable keys: `PUBLIC_RUNTIME_KEYS` (38, listed below).
- Immutable at runtime: `db_path`, `dry_run` (`IMMUTABLE_CONFIG_KEYS` —
  dry_run is immutable so enabling it cannot HIDE actions).
- Version 3.0.0 intentionally removes the planner, Boltz, and LN+ option
  families; retained keys remain governed by the compatibility policy.

The full generated field table below is the baseline enumeration
(regenerate with `dataclasses.fields(Config)` if it drifts; the
dataclass is normative).

## Datastore telemetry contracts (current, tested)

- `["revenue","profitability-summary"]`, `["revenue","capex-summary"]`
  (TTL 1800s), `["revenue","segment-observations"]` (schema_version 1)
- Docs: `docs/contracts/*.md`; conformance test:
  `tests/test_cross_plugin_contracts.py`; stale/malformed semantics are
  part of the contract (refactor invariant 3).
- Three additional UNDOCUMENTED mirror keys exist
  (`["revenue","status"|"fee-bounds"|"dashboard"]`) — see
  `persistence-map.md` §datastore; Workstream I must either document or
  explicitly mark them internal.

## External obligations

The v3 plugin creates no external swap or channel-lifecycle obligation.
Historical planner and LN+ rows survive restart for audit compatibility but
cannot schedule work or authorize action.

---

### Runtime-settable keys (PUBLIC_RUNTIME_KEYS, 38)

- `paused`
- `daily_budget_sats`
- `weekly_budget_sats`
- `growth_budget_enabled`
- `growth_budget_earned_fraction`
- `growth_budget_experiment_fraction`
- `growth_budget_max_extra_sats`
- `growth_budget_hard_ceiling_sats`
- `min_fee_ppm`
- `min_fee_ppm_saturated`
- `acquisition_experiment_enabled`
- `max_fee_ppm`
- `fee_profile`
- `fee_market_boundary_enabled`
- `fee_market_boundary_min_competitors`
- `fee_market_boundary_margin_ppm`
- `fee_market_boundary_margin_ratio`
- `fee_market_boundary_max_downshift_ratio`
- `fee_market_boundary_cache_seconds`
- `capex_probability_budget_bonus`
- `receivable_ratio_target`
- `receivable_ratio_floor`
- `drain_fee_discount_max`
- `node_drain_bias_enabled`
- `node_drain_bias_max`
- `enable_dynamic_htlcmax`
- `htlcmax_source_pct`
- `htlcmax_sink_pct`
- `htlcmax_balanced_pct`
- `econ_shadow_enabled` (added 2026-07-12, Phase 1 wiring)
- `econ_governor_rebalance_enabled` (added 2026-07-12, Phase 2D)
- `econ_governor_fees_enabled` (added 2026-07-12, Phase 2H)
- `econ_arbiter_enabled` (added 2026-07-13, Phase 3F)
- `econ_cycle_rebalance_enabled` (added 2026-07-13, Workstream H cutover)
- `econ_ev_populated` (added 2026-07-13, Phase E PR 6)
- `econ_conflict_rules_extended` (added 2026-07-13, Phase G PR 10)
- `authority_level` (added 2026-07-13, Phase 4 Workstream I; observe/fees/liquidity/capital, default `capital`)
- `risk_profile` (added 2026-07-13, Phase D PR 7; default `custom`)

### Full Config dataclass surface (104 fields with defaults)

| Field | Default | Runtime-settable |
|---|---|---|
| `db_path` | `'~/.lightning/revenue_ops.db'` |  |
| `flow_interval` | `3600` |  |
| `fee_interval` | `1800` |  |
| `rebalance_interval` | `900` |  |
| `fee_authority_enabled` | `True` |  |
| `fee_replay_capture_enabled` | `False` |  |
| `rebalance_replay_capture_enabled` | `False` |  |
| `hot_channel_protection_enabled` | `True` |  |
| `hot_channel_protection_override_peers` | `''` |  |
| `hot_channel_protection_min_velocity` | `0.2` |  |
| `hot_channel_protection_min_marginal_roi` | `0.2` |  |
| `hot_channel_protection_profit_budget_pct` | `0.75` |  |
| `hot_channel_protection_max_chunk_multiplier` | `4.0` |  |
| `hot_channel_protection_min_cooldown_hours` | `1.0` |  |
| `receivable_ratio_target` | `0.3` | yes |
| `receivable_ratio_floor` | `0.2` | yes |
| `drain_fee_discount_max` | `0.0` | yes |
| `node_drain_bias_enabled` | `False` | yes |
| `node_drain_bias_max` | `0.3` | yes |
| `enable_dynamic_htlcmax` | `True` | yes |
| `htlcmax_source_pct` | `0.5` | yes |
| `htlcmax_sink_pct` | `0.25` | yes |
| `htlcmax_balanced_pct` | `0.45` | yes |
| `econ_shadow_enabled` | `True` | yes |
| `econ_governor_rebalance_enabled` | `True` | yes |
| `econ_governor_fees_enabled` | `True` | yes |
| `econ_arbiter_enabled` | `True` | yes |
| `econ_cycle_rebalance_enabled` | `True` | yes |
| `econ_ev_populated` | `True` | yes |
| `econ_conflict_rules_extended` | `True` | yes |
| `authority_level` | `'capital'` | yes |
| `risk_profile` | `'custom'` | yes |
| `flow_window_days` | `7` |  |
| `source_threshold` | `0.05` |  |
| `sink_threshold` | `-0.05` |  |
| `min_fee_ppm` | `50` | yes |
| `min_fee_ppm_saturated` | `0` | yes |
| `acquisition_experiment_enabled` | `False` | yes |
| `max_fee_ppm` | `2000` | yes |
| `base_fee_msat` | `0` |  |
| `base_fee_policy` | `'off'` |  |
| `neighbor_median_min_competitors` | `2` |  |
| `market_fee_mode` | `'undercut'` |  |
| `fee_profile` | `'active'` | yes |
| `fee_market_boundary_enabled` | `False` | yes |
| `fee_market_boundary_min_competitors` | `3` | yes |
| `fee_market_boundary_margin_ppm` | `5` | yes |
| `fee_market_boundary_margin_ratio` | `0.05` | yes |
| `fee_market_boundary_max_downshift_ratio` | `0.35` | yes |
| `fee_market_boundary_cache_seconds` | `60` | yes |
| `rebalance_min_profit` | `10` |  |
| `rebalance_max_amount` | `5000000` |  |
| `low_liquidity_threshold` | `0.3` |  |
| `high_liquidity_threshold` | `0.7` |  |
| `rebalance_cooldown_hours` | `24` |  |
| `rebalance_emergency_local_ratio` | `0.1` |  |
| `rebalance_drift_override_ratio` | `0.3` |  |
| `rebalance_hold_margin` | `0.0` |  |
| `pair_fee_cap_ppm` | `1000` |  |
| `rebalance_activity_window_seconds` | `3600` |  |
| `rebalance_activity_penalty_coeff` | `0.5` |  |
| `rebalance_activity_penalty_cap_frac` | `0.5` |  |
| `rebalance_utilization_window_days` | `7` |  |
| `rebalance_utilization_floor` | `0.05` |  |
| `rebalance_utilization_ceiling` | `1.0` |  |
| `rebalance_utilization_min_forwards` | `5` |  |
| `rebalance_size_tiered_targets` | `True` |  |
| `rebalance_size_reference_percentile` | `0.5` |  |
| `rebalance_small_channel_band_half_width` | `0.15` |  |
| `inbound_fee_estimate_ppm` | `50` |  |
| `estimated_open_cost_sats` | `5000` |  |
| `daily_budget_sats` | `5000` | yes |
| `growth_budget_enabled` | `False` | yes |
| `growth_budget_earned_fraction` | `0.25` | yes |
| `growth_budget_experiment_fraction` | `0.1` | yes |
| `growth_budget_max_extra_sats` | `2000` | yes |
| `growth_budget_hard_ceiling_sats` | `10000` | yes |
| `allow_zero_cost_auto_rebalance_when_budget_zero` | `False` |  |
| `weekly_budget_sats` | `35000` | yes |
| `min_wallet_reserve` | `1000000` |  |
| `rpc_timeout_seconds` | `15` |  |
| `reservation_timeout_hours` | `4` |  |
| `htlc_congestion_threshold` | `0.8` |  |
| `enable_reputation` | `True` |  |
| `reputation_decay` | `0.98` |  |
| `enable_kelly` | `False` |  |
| `max_concurrent_jobs` | `5` |  |
| `askrene_layer` | `'xpay'` |  |
| `rebalance_router` | `'v3'` |  |
| `askrene_layers` | `'standalone'` |  |
| `paused` | `False` | yes |
| `dry_run` | `False` |  |
| `enable_vegas_reflex` | `True` |  |
| `vegas_decay_rate` | `0.85` |  |
| `thompson_prior_std_fee` | `100` |  |
| `capex_reinvestment_rate` | `0.5` |  |
| `capex_bootstrap_bps` | `10` |  |
| `capex_bootstrap_max_sats` | `200` |  |
| `capex_grace_days` | `14` |  |
| `capex_global_envelope_sats` | `0` |  |
| `capex_probability_budget_bonus` | `0.0` | yes |
| `_version` | `0` |  |
| `_lock` | `factory:allocate_lock` |  |
| `_override_warnings` | `factory:list` |  |
