"""Risk profiles (gap-closure Phase D, PR 7).

The spec's normal operator surface includes ``risk_profile`` ∈
{preserve, conservative, balanced, growth, custom} — coherent defaults
for ECONOMIC-RISK PREFERENCES only. Safety invariants, authority
controls, external integrations, and expert overrides are never
profile-bundled (constraint: profiles must not weaken protocol safety,
contractual obligations, capital-at-risk ceilings, or hard rails).

Semantics:
- ``custom`` (the default, and the migration value for every existing
  deployment) derives NOTHING: effective configuration is exactly the
  explicit configuration. Behavior-preserving by construction.
- Non-custom profiles resolve to a bundle of economic-risk values
  applied at STARTUP (Config.load_overrides) to keys the operator has
  NOT explicitly overridden. Precedence (documented, tested):
      explicit override (DB)  >  profile bundle  >  dataclass default
- Unknown/invalid profile names resolve like ``custom`` (fail
  conservative — no derived values).
- Policies never see profile names: they read effective settings only.

Field classification (spec Phase D): every Config field is assigned
exactly one category below; the coverage pin in
tests/test_risk_profiles.py forces new fields to be classified here.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable

PROFILE_NAMES = ("preserve", "conservative", "balanced", "growth",
                 "custom")

CATEGORIES: FrozenSet[str] = frozenset({
    "safety_invariant",      # hard rails/reserves — profiles NEVER touch
    "authority_control",     # what the node MAY do (gates, pause, levels)
    "economic_risk",         # profile-bundleable risk preferences
    "operational_timing",    # schedules, pacing, timeouts
    "external_integration",  # paths, routers, currencies
    "advanced_expert",       # algorithm tuning; expert-only
    "deprecated_transition", # announced-removal shims
})

FIELD_CLASSIFICATION: Dict[str, str] = {
    # --- external integration ------------------------------------------
    "db_path": "external_integration",
    "askrene_layer": "external_integration",
    "askrene_layers": "external_integration",
    "rebalance_router": "external_integration",
    "expansion_treasury_preferred_currency": "external_integration",
    # --- operational timing --------------------------------------------
    "flow_interval": "operational_timing",
    "fee_interval": "operational_timing",
    "rebalance_interval": "operational_timing",
    "planner_interval": "operational_timing",
    "boltz_auto_cycle_interval_minutes": "operational_timing",
    "boltz_auto_cycle_startup_delay_seconds": "operational_timing",
    "boltz_auto_cycle_max_actions": "operational_timing",
    "expansion_treasury_max_actions": "operational_timing",
    "rebalance_cooldown_hours": "operational_timing",
    "reservation_timeout_hours": "operational_timing",
    "rpc_timeout_seconds": "operational_timing",
    "max_concurrent_jobs": "operational_timing",
    "lnplus_watcher_interval": "operational_timing",
    "lnplus_pending_timeout_days": "operational_timing",
    "capex_grace_days": "operational_timing",
    # --- authority controls --------------------------------------------
    "paused": "authority_control",
    "dry_run": "authority_control",
    "authority_level": "authority_control",
    "risk_profile": "authority_control",
    "econ_shadow_enabled": "authority_control",
    "econ_governor_rebalance_enabled": "authority_control",
    "econ_governor_planner_enabled": "authority_control",
    "econ_governor_lnplus_enabled": "authority_control",
    "econ_governor_boltz_enabled": "authority_control",
    "econ_governor_fees_enabled": "authority_control",
    "econ_arbiter_enabled": "authority_control",
    "econ_cycle_rebalance_enabled": "authority_control",
    "econ_cycle_planner_enabled": "authority_control",
    "econ_cycle_boltz_enabled": "authority_control",
    "econ_ev_populated": "authority_control",
    "planner_enabled": "authority_control",
    "planner_dry_run": "authority_control",
    "planner_execute_closes": "authority_control",
    "boltz_auto_cycle_enabled": "authority_control",
    "lnplus_swaps_enabled": "authority_control",
    "lnplus_execute_applications": "authority_control",
    "expansion_treasury_enabled": "authority_control",
    # --- safety invariants ---------------------------------------------
    "min_fee_ppm": "safety_invariant",
    "min_fee_ppm_saturated": "safety_invariant",
    "max_fee_ppm": "safety_invariant",
    "min_wallet_reserve": "safety_invariant",
    "growth_budget_hard_ceiling_sats": "safety_invariant",
    "planner_max_fee_rate_sat_vb": "safety_invariant",
    "planner_close_fee_reserve_multiplier": "safety_invariant",
    "planner_close_fee_cap_sats": "safety_invariant",
    "diagnostic_rebalance_max_fee_sats": "safety_invariant",
    "lnplus_apply_feerate_ceiling": "safety_invariant",
    "receivable_ratio_floor": "safety_invariant",
    "expansion_treasury_min_source_local_pct": "safety_invariant",
    "expansion_treasury_exclude_protected": "safety_invariant",
    # --- economic-risk preferences (profile-bundleable) ----------------
    "daily_budget_sats": "economic_risk",
    "weekly_budget_sats": "economic_risk",
    "rebalance_hold_margin": "economic_risk",
    "rebalance_max_amount": "economic_risk",
    "pair_fee_cap_ppm": "economic_risk",
    "growth_budget_enabled": "economic_risk",
    "growth_budget_earned_fraction": "economic_risk",
    "growth_budget_experiment_fraction": "economic_risk",
    "growth_budget_max_extra_sats": "economic_risk",
    "planner_min_annual_roi_pct": "economic_risk",
    "planner_max_opens_per_cycle": "economic_risk",
    "planner_max_closes_per_cycle": "economic_risk",
    "planner_min_channel_sats": "economic_risk",
    "planner_max_channel_sats": "economic_risk",
    "boltz_structural_budget_sats_per_day": "economic_risk",
    "lnplus_swap_preference_margin": "economic_risk",
    "lnplus_max_duration_months": "economic_risk",
    "capex_reinvestment_rate": "economic_risk",
    "capex_exploration_rate": "economic_risk",
    "capex_tactical_rate": "economic_risk",
    "capex_global_envelope_sats": "economic_risk",
    "capex_probability_budget_bonus": "economic_risk",
    "receivable_ratio_target": "economic_risk",
    "fee_profile": "economic_risk",
    "expansion_treasury_onchain_target_sats": "economic_risk",
    "expansion_treasury_min_deficit_sats": "economic_risk",
    # --- deprecated / transition ---------------------------------------
    "rebalance_min_profit": "deprecated_transition",
    # --- advanced expert overrides --------------------------------------
    "hot_channel_protection_enabled": "advanced_expert",
    "hot_channel_protection_override_peers": "advanced_expert",
    "hot_channel_protection_min_velocity": "advanced_expert",
    "hot_channel_protection_min_marginal_roi": "advanced_expert",
    "hot_channel_protection_profit_budget_pct": "advanced_expert",
    "hot_channel_protection_max_chunk_multiplier": "advanced_expert",
    "hot_channel_protection_min_cooldown_hours": "advanced_expert",
    "drain_fee_discount_max": "advanced_expert",
    "node_drain_bias_enabled": "advanced_expert",
    "node_drain_bias_max": "advanced_expert",
    "enable_dynamic_htlcmax": "advanced_expert",
    "htlcmax_source_pct": "advanced_expert",
    "htlcmax_sink_pct": "advanced_expert",
    "htlcmax_balanced_pct": "advanced_expert",
    "flow_window_days": "advanced_expert",
    "source_threshold": "advanced_expert",
    "sink_threshold": "advanced_expert",
    "base_fee_msat": "advanced_expert",
    "base_fee_policy": "advanced_expert",
    "neighbor_median_min_competitors": "advanced_expert",
    "market_fee_mode": "advanced_expert",
    "fee_market_boundary_enabled": "advanced_expert",
    "fee_market_boundary_min_competitors": "advanced_expert",
    "fee_market_boundary_margin_ppm": "advanced_expert",
    "fee_market_boundary_margin_ratio": "advanced_expert",
    "fee_market_boundary_max_downshift_ratio": "advanced_expert",
    "fee_market_boundary_cache_seconds": "advanced_expert",
    "planner_close_feerange_enabled": "advanced_expert",
    "low_liquidity_threshold": "advanced_expert",
    "high_liquidity_threshold": "advanced_expert",
    "rebalance_emergency_local_ratio": "advanced_expert",
    "rebalance_drift_override_ratio": "advanced_expert",
    "rebalance_activity_window_seconds": "advanced_expert",
    "rebalance_activity_penalty_coeff": "advanced_expert",
    "rebalance_activity_penalty_cap_frac": "advanced_expert",
    "rebalance_utilization_window_days": "advanced_expert",
    "rebalance_utilization_floor": "advanced_expert",
    "rebalance_utilization_ceiling": "advanced_expert",
    "rebalance_utilization_min_forwards": "advanced_expert",
    "rebalance_size_tiered_targets": "advanced_expert",
    "rebalance_size_reference_percentile": "advanced_expert",
    "rebalance_small_channel_band_half_width": "advanced_expert",
    "inbound_fee_estimate_ppm": "advanced_expert",
    "estimated_open_cost_sats": "advanced_expert",
    "allow_zero_cost_auto_rebalance_when_budget_zero": "advanced_expert",
    "htlc_congestion_threshold": "advanced_expert",
    "enable_reputation": "advanced_expert",
    "reputation_decay": "advanced_expert",
    "enable_kelly": "advanced_expert",
    "enable_vegas_reflex": "advanced_expert",
    "vegas_decay_rate": "advanced_expert",
    "thompson_prior_std_fee": "advanced_expert",
    "capex_bootstrap_bps": "advanced_expert",
    "capex_bootstrap_max_sats": "advanced_expert",
    "lnplus_min_peer_positive_ratings": "advanced_expert",
    "lnplus_min_peer_rank": "advanced_expert",
    "lnplus_max_participants": "advanced_expert",
    "lnplus_min_participants": "advanced_expert",
    "lnplus_inbound_credit_factor": "advanced_expert",
}

# Bundles touch ONLY economic_risk keys (pinned by test). `conservative`
# restates today's dataclass defaults — selecting it reproduces guarded
# default operation; `custom` derives nothing. Values chosen inside
# CONFIG_FIELD_RANGES; growth_budget_hard_ceiling_sats (safety) is
# deliberately absent from every bundle.
PROFILE_BUNDLES: Dict[str, Dict[str, Any]] = {
    "custom": {},
    "preserve": {
        "daily_budget_sats": 2000,
        "weekly_budget_sats": 14000,
        "rebalance_hold_margin": 5.0,
        "growth_budget_enabled": False,
        "growth_budget_earned_fraction": 0.1,
        "growth_budget_experiment_fraction": 0.05,
        "growth_budget_max_extra_sats": 1000,
        "planner_min_annual_roi_pct": 5.0,
        "planner_max_opens_per_cycle": 0,
        "planner_max_closes_per_cycle": 0,
        "lnplus_swap_preference_margin": 0.5,
    },
    "conservative": {
        "daily_budget_sats": 5000,
        "weekly_budget_sats": 35000,
        "rebalance_hold_margin": 0.0,
        "growth_budget_enabled": False,
        "growth_budget_earned_fraction": 0.25,
        "growth_budget_experiment_fraction": 0.1,
        "growth_budget_max_extra_sats": 2000,
        "planner_min_annual_roi_pct": 1.0,
        "planner_max_opens_per_cycle": 1,
        "planner_max_closes_per_cycle": 0,
        "lnplus_swap_preference_margin": 0.2,
    },
    "balanced": {
        "daily_budget_sats": 8000,
        "weekly_budget_sats": 56000,
        "rebalance_hold_margin": 0.0,
        "growth_budget_enabled": True,
        "growth_budget_earned_fraction": 0.25,
        "growth_budget_experiment_fraction": 0.1,
        "growth_budget_max_extra_sats": 2000,
        "planner_min_annual_roi_pct": 1.0,
        "planner_max_opens_per_cycle": 1,
        "planner_max_closes_per_cycle": 1,
        "lnplus_swap_preference_margin": 0.2,
    },
    "growth": {
        "daily_budget_sats": 12000,
        "weekly_budget_sats": 84000,
        "rebalance_hold_margin": 0.0,
        "growth_budget_enabled": True,
        "growth_budget_earned_fraction": 0.4,
        "growth_budget_experiment_fraction": 0.2,
        "growth_budget_max_extra_sats": 5000,
        "planner_min_annual_roi_pct": 0.5,
        "planner_max_opens_per_cycle": 2,
        "planner_max_closes_per_cycle": 1,
        "lnplus_swap_preference_margin": 0.1,
    },
}


def resolve_profile(profile: Any,
                    explicit_keys: Iterable[str]) -> Dict[str, Any]:
    """Profile-derived values for keys the operator has NOT explicitly
    overridden. Unknown/invalid profile -> {} (behaves like custom)."""
    name = str(profile or "").strip().lower()
    bundle = PROFILE_BUNDLES.get(name, {})
    explicit = set(explicit_keys or ())
    return {key: value for key, value in bundle.items()
            if key not in explicit}


def preview_profile(current_values: Dict[str, Any], profile: Any,
                    explicit_keys: Iterable[str]) -> Dict[str, Any]:
    """PR 8: read-only preview of what selecting `profile` would change
    at the next restart, against the CURRENT effective values.

    Per bundle key: current value, profile value, whether an explicit
    override blocks the profile (precedence), and whether the key would
    actually change. Includes a contradiction pre-check over the merged
    result for the known cross-field pair the bundles can affect
    (daily vs weekly budget). Never mutates anything."""
    name = str(profile or "").strip().lower()
    if name not in PROFILE_BUNDLES:
        return {"error": f"unknown profile: {profile!r}",
                "valid_profiles": list(PROFILE_NAMES)}
    explicit = set(explicit_keys or ())
    bundle = PROFILE_BUNDLES[name]
    changes, blocked, unchanged = [], [], []
    merged = dict(current_values)
    for key in sorted(bundle):
        profile_value = bundle[key]
        current = current_values.get(key)
        entry = {"key": key, "current": current,
                 "profile_value": profile_value}
        if key in explicit:
            entry["blocked_by"] = "explicit_override"
            blocked.append(entry)
        elif current == profile_value:
            unchanged.append(entry)
        else:
            merged[key] = profile_value
            changes.append(entry)
    contradictions = []
    try:
        daily = merged.get("daily_budget_sats")
        weekly = merged.get("weekly_budget_sats")
        if isinstance(daily, (int, float)) and isinstance(weekly,
                                                          (int, float)) \
                and daily > weekly:
            contradictions.append(
                f"daily_budget_sats ({daily}) > weekly_budget_sats "
                f"({weekly}) in the merged result; the weekly cap "
                f"binds first")
    except Exception:
        pass
    return {
        "profile": name,
        "would_change": changes,
        "blocked_by_explicit_override": blocked,
        "already_equal": unchanged,
        "contradiction_precheck": contradictions,
        "activation": "takes effect at plugin restart after "
                      "`revenue-config set risk_profile " + name + "`",
    }


def preview_all(current_values: Dict[str, Any],
                explicit_keys: Iterable[str]) -> Dict[str, Any]:
    """PR 8 observe-only comparison: every profile's delta from the
    current effective configuration, side by side."""
    return {name: preview_profile(current_values, name, explicit_keys)
            for name in PROFILE_NAMES if name != "custom"}
