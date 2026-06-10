"""
Configuration module for cl-revenue-ops

Contains the Config dataclass that holds all tunable parameters
for the Revenue Operations plugin.

Includes ConfigSnapshot for immutable, thread-safe cycle execution
and runtime configuration updates via RPC.
"""

import math
import threading
import dataclasses
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, FrozenSet, TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database


# Immutable keys that cannot be changed at runtime
IMMUTABLE_CONFIG_KEYS: FrozenSet[str] = frozenset({
    'db_path',
    'dry_run',  # Safety: don't allow enabling dry_run to hide actions
})

PUBLIC_RUNTIME_KEYS = (
    'paused',
    'daily_budget_sats',
    'min_fee_ppm',
    'max_fee_ppm',
    'fee_profile',
    'fee_market_boundary_enabled',
    'fee_market_boundary_min_competitors',
    'fee_market_boundary_margin_ppm',
    'fee_market_boundary_margin_ratio',
    'fee_market_boundary_max_downshift_ratio',
    'fee_market_boundary_cache_seconds',
    'planner_enabled',
    'planner_dry_run',
    'planner_execute_closes',
    'planner_max_opens_per_cycle',
    'planner_max_closes_per_cycle',
    'planner_min_annual_roi_pct',
    # V3 router probability-aware budget relaxation (default 0.0 = off).
    # Exposed so operators running the askrene router can enable the
    # reliability-weighted budget bonus without editing code or the database.
    'capex_probability_budget_bonus',
    # Source-heavy drain rollout knobs. Plugin options in the config file
    # only load at lightningd startup; these are exposed at runtime so the
    # drain can be enabled and tuned via revenue-config without a daemon
    # restart. All default to off/neutral.
    'boltz_auto_cycle_enabled',
    'boltz_structural_budget_sats_per_day',
    'receivable_ratio_target',
    'receivable_ratio_floor',
    'drain_fee_discount_max',
)

# Type mapping for config fields (for validation)
CONFIG_FIELD_TYPES: Dict[str, type] = {
    'flow_interval': int,
    'fee_interval': int,
    'rebalance_interval': int,
    'paused': bool,
    'min_fee_ppm': int,
    'max_fee_ppm': int,
    'fee_profile': str,
    'fee_market_boundary_enabled': bool,
    'fee_market_boundary_min_competitors': int,
    'fee_market_boundary_margin_ppm': int,
    'fee_market_boundary_margin_ratio': float,
    'fee_market_boundary_max_downshift_ratio': float,
    'fee_market_boundary_cache_seconds': int,
    'daily_budget_sats': int,
    'allow_zero_cost_auto_rebalance_when_budget_zero': bool,
    'weekly_budget_sats': int,
    'hot_channel_protection_enabled': bool,
    'hot_channel_protection_override_peers': str,
    'hot_channel_protection_min_velocity': float,
    'hot_channel_protection_min_marginal_roi': float,
    'hot_channel_protection_profit_budget_pct': float,
    'hot_channel_protection_max_chunk_multiplier': float,
    'hot_channel_protection_min_cooldown_hours': float,
    'hot_channel_protection_max_rebalance_fee_ppm': int,
    'boltz_auto_cycle_enabled': bool,
    'boltz_auto_cycle_interval_minutes': int,
    'boltz_auto_cycle_max_actions': int,
    'boltz_auto_cycle_startup_delay_seconds': int,
    'receivable_ratio_target': float,
    'receivable_ratio_floor': float,
    'boltz_structural_budget_sats_per_day': int,
    'drain_fee_discount_max': float,
    'expansion_treasury_enabled': bool,
    'expansion_treasury_onchain_target_sats': int,
    'expansion_treasury_min_deficit_sats': int,
    'expansion_treasury_preferred_currency': str,
    'expansion_treasury_max_actions': int,
    'expansion_treasury_min_source_local_pct': float,
    'expansion_treasury_exclude_protected': bool,
    'min_wallet_reserve': int,
    'low_liquidity_threshold': float,
    'high_liquidity_threshold': float,
    'htlc_congestion_threshold': float,
    'enable_reputation': bool,
    'enable_kelly': bool,
    'kelly_fraction': float,
    'reputation_decay': float,
    'max_concurrent_jobs': int,
    'askrene_layer': str,
    'askrene_layers': str,
    'askrene_max_age_sec': int,
    'rebalance_router': str,
    'rebalance_min_profit': int,
    'rebalance_min_profit_ppm': int,
    'rebalance_max_amount': int,
    'rebalance_min_amount': int,
    'rebalance_cooldown_hours': int,
    'rebalance_emergency_local_ratio': float,
    'rebalance_drift_override_ratio': float,
    'rebalance_hold_margin': float,
    'pair_fee_cap_ppm': int,
    'hive_equalization_enabled': bool,
    'hive_equalization_low_pct': float,
    'hive_equalization_high_pct': float,
    'hive_equalization_cooldown_hours': int,
    'hive_equalization_max_candidates_per_cycle': int,
    'hive_rebalance_bootstrap_budget_sats': int,
    'rebalance_coordination_reserved_slots': int,
    'hive_push_enabled': bool,
    'hive_push_trigger_ratio': float,
    'hive_push_target_ratio': float,
    'futility_cooldown_hours': int,
    'inbound_fee_estimate_ppm': int,
    # Vegas Reflex
    'enable_vegas_reflex': bool,
    'vegas_decay_rate': float,
    # Operational Hardening
    'rpc_timeout_seconds': int,
    'rpc_circuit_breaker_seconds': int,
    'rpc_pool_size': int,
    'reservation_timeout_hours': int,
    # Issue #28: Revenue rate smoothing
    'ema_smoothing_alpha': float,
    # Issue #30: Velocity gate for rebalancing
    'enable_velocity_gate': bool,
    'min_velocity_threshold': float,
    'new_channel_grace_days': int,
    # DTS (Discounted Thompson Sampling) parameters
    'thompson_prior_std_fee': int,
    'thompson_observation_decay_hours': int,
    'thompson_max_observations': int,
    'thompson_min_observations': int,
    # Routing Intelligence Integration
    # Fields present in CONFIG_FIELD_RANGES that need type registration
    'base_fee_msat': int,
    'fee_ppm_intra_fleet': int,
    'neighbor_median_min_competitors': int,
    'flow_window_days': int,
    'estimated_open_cost_sats': int,
    'target_flow': int,
    'source_threshold': float,
    'sink_threshold': float,
    # Capacity Planner
    'planner_enabled': bool,
    'planner_interval': int,
    'planner_dry_run': bool,
    'planner_execute_closes': bool,
    'planner_max_opens_per_cycle': int,
    'planner_max_closes_per_cycle': int,
    'planner_close_fee_reserve_multiplier': float,
    'planner_close_fee_cap_sats': int,
    'planner_close_feerange_enabled': bool,
    'planner_min_channel_sats': int,
    'planner_max_channel_sats': int,
    'planner_min_channel_age_days': int,
    'planner_min_peer_uptime_pct': float,
    'planner_max_fee_rate_sat_vb': float,
    'planner_min_annual_roi_pct': float,
    # Hive Hints
    'hive_hints_enabled': bool,
    'hive_hints_ttl_seconds': int,
    'hive_hints_allow_all_hints_m2_scope': bool,
    # Unified Capex Budget Engine
    'capex_reinvestment_rate': float,
    'capex_bootstrap_bps': int,
    'capex_bootstrap_max_sats': int,
    'capex_grace_days': int,
    'capex_probability_budget_bonus': float,
    'capex_exploration_rate': float,
    'capex_tactical_rate': float,
    'capex_global_envelope_sats': int,
    'capex_cost_efficiency_weight': float,
    'capex_drain_benefit_weight': float,
}

# Explicit migration shims only. Non-public keys remain internal until they are
# intentionally exposed as deprecated compatibility controls.
DEPRECATED_RUNTIME_KEYS: FrozenSet[str] = frozenset()

# Range constraints for numeric fields
CONFIG_FIELD_RANGES: Dict[str, tuple] = {
    'min_fee_ppm': (5, 100000),  # CRITICAL-02 FIX: Minimum 5 PPM to ensure economic viability
    'max_fee_ppm': (1, 100000),
    'fee_market_boundary_min_competitors': (1, 100),
    'fee_market_boundary_margin_ppm': (0, 10000),
    'fee_market_boundary_margin_ratio': (0.0, 0.50),
    'fee_market_boundary_max_downshift_ratio': (0.05, 1.0),
    'fee_market_boundary_cache_seconds': (10, 3600),
    'daily_budget_sats': (0, 10000000),
    'weekly_budget_sats': (0, 70_000_000),
    'boltz_auto_cycle_interval_minutes': (1, 1440),
    'boltz_auto_cycle_max_actions': (1, 10),
    'boltz_auto_cycle_startup_delay_seconds': (0, 3600),
    'receivable_ratio_target': (0.0, 1.0),
    'receivable_ratio_floor': (0.0, 1.0),
    'boltz_structural_budget_sats_per_day': (0, 1_000_000),
    'drain_fee_discount_max': (0.0, 0.5),
    'min_wallet_reserve': (0, 100000000),
    'low_liquidity_threshold': (0.0, 1.0),
    'high_liquidity_threshold': (0.0, 1.0),
    'htlc_congestion_threshold': (0.0, 1.0),
    'reputation_decay': (0.0, 1.0),
    'kelly_fraction': (0.0, 1.0),
    'vegas_decay_rate': (0.0, 1.0),
    'rebalance_min_profit_ppm': (0, 100000),
    'rpc_timeout_seconds': (1, 300),
    'rpc_circuit_breaker_seconds': (0, 3600),
    'rpc_pool_size': (1, 8),
    'reservation_timeout_hours': (1, 24),
    # Issue #28: Revenue rate smoothing
    'ema_smoothing_alpha': (0.1, 0.9),
    # Issue #30: Velocity gate for rebalancing
    'min_velocity_threshold': (0.0, 1.0),
    'new_channel_grace_days': (0, 30),
    # DTS (Discounted Thompson Sampling) parameters
    'thompson_prior_std_fee': (10, 500),
    'thompson_observation_decay_hours': (24, 720),  # 1 day to 30 days
    'thompson_max_observations': (50, 500),
    'thompson_min_observations': (1, 20),
    # Routing Intelligence Integration
    # Additional range validations
    'flow_interval': (60, 86400),
    'fee_interval': (60, 86400),
    'rebalance_interval': (60, 86400),
    'max_concurrent_jobs': (1, 20),
    'askrene_max_age_sec': (10, 86400),
    'base_fee_msat': (0, 10000),
    'fee_ppm_intra_fleet': (0, 1000),
    'neighbor_median_min_competitors': (2, 50),
    'rebalance_min_profit': (0, 1000000),
    'rebalance_min_amount': (1000, 50000000),
    'rebalance_max_amount': (10000, 100000000),
    'flow_window_days': (1, 365),
    # AUDIT FIX C-2/I-4: Missing range validation for float/int fields
    'source_threshold': (-1.0, 1.0),
    'sink_threshold': (-1.0, 1.0),
    'expansion_treasury_min_source_local_pct': (0.0, 100.0),
    'hot_channel_protection_max_chunk_multiplier': (1.0, 20.0),
    'hot_channel_protection_min_cooldown_hours': (0.0, 168.0),
    'hot_channel_protection_min_marginal_roi': (0.0, 10.0),
    'hot_channel_protection_min_velocity': (0.0, 1.0),
    'hot_channel_protection_profit_budget_pct': (0.0, 1.0),
    'inbound_fee_estimate_ppm': (0, 5000),
    'rebalance_cooldown_hours': (1, 168),
    'rebalance_emergency_local_ratio': (0.0, 1.0),
    'rebalance_drift_override_ratio': (0.0, 1.0),
    'rebalance_hold_margin': (0.0, 1.0),
    'pair_fee_cap_ppm': (0, 100000),
    'hive_equalization_low_pct': (0.0, 1.0),
    'hive_equalization_high_pct': (0.0, 1.0),
    'hive_equalization_cooldown_hours': (1, 168),
    'hive_equalization_max_candidates_per_cycle': (1, 10),
    'hive_rebalance_bootstrap_budget_sats': (0, 10_000),
    'rebalance_coordination_reserved_slots': (0, 10),
    'futility_cooldown_hours': (1, 168),
    'target_flow': (1000, 100000000),
    'estimated_open_cost_sats': (0, 1000000),
    'expansion_treasury_max_actions': (1, 10),
    'expansion_treasury_min_deficit_sats': (0, 100000000),
    'expansion_treasury_onchain_target_sats': (0, 1000000000),
    'hot_channel_protection_max_rebalance_fee_ppm': (0, 100000),
    # Capacity Planner
    'planner_interval': (600, 604800),
    'planner_max_opens_per_cycle': (0, 10),
    'planner_max_closes_per_cycle': (0, 10),
    'planner_min_channel_sats': (100000, 100000000),
    'planner_max_channel_sats': (500000, 1677721500),
    'planner_min_channel_age_days': (1, 365),
    'planner_min_peer_uptime_pct': (0.0, 100.0),
    'planner_max_fee_rate_sat_vb': (1.0, 1000.0),
    'planner_min_annual_roi_pct': (0.0, 100.0),
    'hive_hints_ttl_seconds': (60, 7200),
    # Unified Capex Budget Engine
    'capex_reinvestment_rate': (0.0, 1.0),
    'capex_bootstrap_bps': (0, 100),
    'capex_bootstrap_max_sats': (0, 10000),
    'capex_grace_days': (0, 90),
    'capex_exploration_rate': (0.0, 1.0),
    'capex_tactical_rate': (0.0, 1.0),
    'capex_probability_budget_bonus': (0.0, 1.0),
    'capex_global_envelope_sats': (0, 100_000_000),
}

# Valid values for string enum fields
STRING_ENUM_VALID_VALUES: Dict[str, tuple] = {
    'expansion_treasury_preferred_currency': ('BTC', 'LBTC', 'L-BTC', 'btc', 'lbtc', 'l-btc'),
    'fee_profile': ('active', 'conservative'),
    'rebalance_router': ('v3',),
    'market_fee_mode': ('undercut', 'match', 'premium', 'competition_aware'),
}


@dataclass
class Config:
    """
    Configuration container for the Revenue Operations plugin.
    
    All values can be set via plugin options at startup.
    """
    
    # Database path
    db_path: str = '~/.lightning/revenue_ops.db'
    
    # Timer intervals (in seconds)
    flow_interval: int = 3600      # 1 hour
    fee_interval: int = 1800       # 30 minutes (matches option default)
    rebalance_interval: int = 900  # 15 minutes
    # Hot-channel protection (aggressiveness for fast-draining, high-profit channels)
    hot_channel_protection_enabled: bool = True
    hot_channel_protection_override_peers: str = ''  # CSV fallback; DB override table preferred
    hot_channel_protection_min_velocity: float = 0.20
    hot_channel_protection_min_marginal_roi: float = 0.20
    hot_channel_protection_profit_budget_pct: float = 0.75
    hot_channel_protection_max_chunk_multiplier: float = 4.0
    hot_channel_protection_min_cooldown_hours: float = 1.0
    hot_channel_protection_max_rebalance_fee_ppm: int = 2000
    boltz_auto_cycle_enabled: bool = False  # Run profit-gated Boltz auto-balance cycle in background (opt-in)
    boltz_auto_cycle_interval_minutes: int = 15  # Scheduler cadence for Boltz auto-cycle
    boltz_auto_cycle_max_actions: int = 1   # Max actions per scheduled cycle
    boltz_auto_cycle_startup_delay_seconds: int = 120  # Delay before first Boltz auto-cycle
    # Source-heavy drain: node-level receivable objective and envelopes.
    # receivable_ratio = total receivable / total capacity across channels.
    receivable_ratio_target: float = 0.30   # structural credit scales to 0 here
    receivable_ratio_floor: float = 0.20    # below this the node is "starved"
    # Daily cap (sats of swap fees) for loop-outs that only pass the profit
    # guard via the structural credit. 0 = structural loop-outs disabled.
    boltz_structural_budget_sats_per_day: int = 0
    # Max bounded fee discount applied to stagnant over-local channels.
    # 0.0 = disabled. 0.10 means fees may be biased down by at most 10%.
    drain_fee_discount_max: float = 0.0
    # Expansion treasury mode (reverse swaps to build on-chain funds for channel opens)
    expansion_treasury_enabled: bool = False
    expansion_treasury_onchain_target_sats: int = 5_000_000
    expansion_treasury_min_deficit_sats: int = 250_000
    expansion_treasury_preferred_currency: str = 'BTC'
    expansion_treasury_max_actions: int = 1
    expansion_treasury_min_source_local_pct: float = 80.0  # I-7: 0-100 scale (not 0-1)
    expansion_treasury_exclude_protected: bool = True
    
    # Flow analysis parameters
    target_flow: int = 100000      # Target sats routed per day per channel
    flow_window_days: int = 7      # Days to analyze for flow calculation
    
    # Flow ratio thresholds for classification
    source_threshold: float = 0.5   # FlowRatio > 0.5 = Source (draining)
    sink_threshold: float = -0.5    # FlowRatio < -0.5 = Sink (filling)
    
    # Fee parameters
    min_fee_ppm: int = 10          # Floor fee in PPM (matches plugin option default)
    max_fee_ppm: int = 5000        # Ceiling fee in PPM
    base_fee_msat: int = 0         # Base fee fallback when base_fee_policy = "off"

    # Adaptive base_fee (Upgrade A, 2026-04-22) — per-channel-role base_fee_msat.
    # Motivated by the 168x per-forward fee gap observed between clboss-ivan
    # (15,307 msat base) and hive (0 msat base) in the 2026-04-21 tier runs.
    # policy = "off" -> use base_fee_msat (legacy). "adaptive" -> classify each
    # peer and apply role-specific base fee. V1 classification is two-bucket:
    # hive fleet members get base_fee_msat_intra_fleet (0); everyone else gets
    # base_fee_msat_non_hive. Gateway/leaf split is deferred until data shows
    # a single non-hive value is insufficient.
    base_fee_policy: str = "off"            # off | adaptive
    base_fee_msat_intra_fleet: int = 0
    base_fee_msat_non_hive: int = 1000      # conservative default per advisor calibration

    # Intra-fleet proportional fee (Path B Step 3, 2026-04-22). The prior
    # 0-PPM fleet policy was a revenue leak: 2 of 3 hive channels earned
    # nothing, and external traffic transiting the hive mesh got free
    # hops it could not distinguish from member-to-member flow. A small
    # nonzero value (default 1 ppm) preserves "cheapest path for members"
    # (1 ppm is ~50-500× below typical competitor rates) while extracting
    # revenue on external transit. Set to 0 to restore legacy 0-PPM policy.
    fee_ppm_intra_fleet: int = 1

    # Minimum competitor count for _get_neighbor_fee_median to return a
    # value. The original threshold of 3 was too strict for small labs /
    # sparse gossip: peers with only 2 other inbound channels produced
    # None and skipped market-fee-mode entirely. Default 2 is the lowest
    # value that still requires *some* competition — a single outlier
    # cannot drag the median. Production nodes with dense gossip may
    # prefer 3 for smoother medians.
    neighbor_median_min_competitors: int = 2

    # Market-fee mode: how we price relative to the weighted-median of neighbor
    # competitors' fees. Added 2026-04-21 to close the head-to-head-vs-clboss
    # gap. Default "undercut" preserves existing behavior; "premium" prices
    # above median in inelastic markets where the hive's coordinated routing
    # means we lose less volume to a price increase than a single operator would.
    #   - "undercut": price below the median (existing behavior)
    #   - "match":    price at the median
    #   - "premium":  price above the median by the same per-corridor weight
    market_fee_mode: str = "undercut"
    fee_profile: str = 'active'    # Fee-controller aggressiveness profile
    # Experimental competitive boundary guard. Disabled by default because a
    # market cap derived from peer gossip is too easy to anchor on cheap
    # outliers and can synchronize unrelated channels around one floor.
    fee_market_boundary_enabled: bool = False
    fee_market_boundary_min_competitors: int = 3
    fee_market_boundary_margin_ppm: int = 5
    fee_market_boundary_margin_ratio: float = 0.05
    fee_market_boundary_max_downshift_ratio: float = 0.35
    fee_market_boundary_cache_seconds: int = 60
    
    # Rebalancing parameters
    rebalance_min_profit: int = 10     # Min profit in sats to trigger (legacy, used when ppm=0)
    rebalance_min_profit_ppm: int = 0  # Min profit in PPM (0 = use sats threshold, >0 = use ppm)
                                        # Recommended: 20 ppm (~10 sats per 500k chunk)
    rebalance_max_amount: int = 5000000  # Max rebalance amount in sats
    rebalance_min_amount: int = 50000    # Min rebalance amount in sats
    low_liquidity_threshold: float = 0.3  # Below 30% = low outbound
    high_liquidity_threshold: float = 0.7 # Above 70% = high outbound
    rebalance_cooldown_hours: int = 24   # Don't re-rebalance same channel for 24h
    # Phase 3 drift override: a destination below this local ratio is still
    # refill-eligible even when the channel-level cooldown is active. Set to
    # 0 to disable the override and keep the strict cooldown gate.
    rebalance_emergency_local_ratio: float = 0.10
    # Phase 3.3 anchor-state drift override: when a destination's local ratio
    # has dropped by at least this much since the last successful rebalance,
    # the cooldown gate is bypassed. Set to 0 to disable.
    rebalance_drift_override_ratio: float = 0.30
    # Phase 4.3 do_nothing hold gate: priced pairs with engine final_score
    # at or below this margin are rejected with reason='below_hold_margin'.
    # 0 leaves the legacy "any positive score" behavior. Use a small positive
    # value to require pairs to clear a meaningful EV bar before executing.
    rebalance_hold_margin: float = 0.0
    # Iter1 pair fee budget: layered on top of the destination's capex
    # bootstrap budget so a small/new channel can still pay enough route
    # fee for the selected route. pair_budget_sats =
    #   max(dest.remaining_capex_sats, ceil(amount * pair_fee_cap_ppm / 1M)).
    # Default 1000 ppm = 0.1% of rebalance amount. 0 disables the layer
    # and keeps the Phase 5 capex-only behavior.
    pair_fee_cap_ppm: int = 1000
    hive_equalization_enabled: bool = True  # Fallback pure-hive inventory equalization
    hive_equalization_low_pct: float = 0.35  # Lower bound for hive balance band
    hive_equalization_high_pct: float = 0.65  # Upper bound for hive balance band
    hive_equalization_cooldown_hours: int = 48  # Longer than standard rebalance cooldown
    hive_equalization_max_candidates_per_cycle: int = 1
    # Conservative fee budget for active hive-member channels whose capex
    # allocation has not appeared yet. Global/weekly budget reservations still
    # enforce aggregate spend; 0 disables this bootstrap path.
    hive_rebalance_bootstrap_budget_sats: int = 300
    hive_push_enabled: bool = True            # Deploy capital to fleet member channels
    hive_push_trigger_ratio: float = 0.60     # Push when local ratio exceeds this
    hive_push_target_ratio: float = 0.50      # Push balance toward this ratio

    # Reserved slots for coordination pairs on top of the planner's
    # max_pairs cap (Phase B.f, 2026-04-23). cl-hive publishes
    # rebalance_recommendations / rebalance_campaigns via hive-export-hints,
    # and the documented contract is that those pairs should "materialize
    # before the normal pair cap is applied." Before this default, they
    # competed for the planner's 10-slot cap and could be squeezed out
    # entirely by a crop of EV-positive planner pairs. Default 2 lets up
    # to two hive-blessed coordination pairs bypass the normal cap without
    # letting coordination dominate arbitrarily. Set to 0 to restore the
    # strict-cap behavior.
    rebalance_coordination_reserved_slots: int = 2

    futility_cooldown_hours: int = 48   # Hours before retrying after 10+ consecutive failures
    inbound_fee_estimate_ppm: int = 50  # Route cost buffer added on top of last-hop fee (PPM)
    
    # Profitability tracking
    estimated_open_cost_sats: int = 5000  # Estimated on-chain fee for channel open
    
    # Global Capital Controls
    daily_budget_sats: int = 5000          # Max rebalancing fees per 24h period (fixed floor)
    allow_zero_cost_auto_rebalance_when_budget_zero: bool = False
    weekly_budget_sats: int = 35000        # Max rebalancing fees per 7-day window (hard ceiling)
    min_wallet_reserve: int = 1_000_000    # Min sats (confirmed on-chain + channel spendable) before ABORT

    # RPC Hardening
    rpc_timeout_seconds: int = 15
    rpc_circuit_breaker_seconds: int = 60
    rpc_pool_size: int = 5             # Number of RPC worker processes
    reservation_timeout_hours: int = 4  # Hours before stale budget reservations auto-release
    
    # HTLC Congestion threshold
    htlc_congestion_threshold: float = 0.8  # Mark channel as CONGESTED if >80% HTLC slots used
    
    # Reputation-weighted volume
    enable_reputation: bool = True  # If True, weight volume by peer success rate
    reputation_decay: float = 0.98  # Decay factor per flow_interval (default hourly)
                                     # 0.98^24 ≈ 0.61, meaning old data loses ~40% weight daily

    # Kelly Criterion Position Sizing
    enable_kelly: bool = False       # If True, scale rebalance budget by Kelly fraction (opt-in)
    kelly_fraction: float = 0.5      # Multiplier for Kelly fraction (0.5 = Half Kelly)
                                      # Full Kelly (1.0) maximizes growth but has high volatility
                                      # Half Kelly (0.5) reduces volatility drag significantly
    
    # Async Job Queue
    max_concurrent_jobs: int = 5              # Max number of concurrent rebalance jobs

    # AskRene (xpay) constraint integration
    askrene_layer: str = 'xpay'               # Layer name for askrene-listlayers
    askrene_max_age_sec: int = 900            # Max constraint age (seconds) to consider fresh

    # V3 rebalance router (askrene getroutes + cl-hive layers)
    rebalance_router: str = 'v3'              # only 'v3' is supported
    askrene_layers: str = 'hive-fleet'        # CSV of layers to pass to v3 router's getroutes calls

    # Safety flags
    paused: bool = False           # If True, suppress automated executor actions
    dry_run: bool = False          # If True, log but don't execute

    # Vegas Reflex (mempool spike defense)
    enable_vegas_reflex: bool = True       # Mempool spike defense
    vegas_decay_rate: float = 0.85         # Per-cycle decay (~30min half-life)
    
    # Deferred features

    # Issue #28: Revenue rate EMA smoothing
    # EMA formula: new_ema = alpha * current + (1 - alpha) * old_ema
    # Lower alpha = slower response (more smoothing), higher = faster response
    ema_smoothing_alpha: float = 0.3       # Default 0.3 balances responsiveness and stability

    # Issue #30: Velocity gate for rebalancing
    # Prevents overfilling channels with no routing history
    enable_velocity_gate: bool = True      # Require minimum velocity before full rebalancing
    min_velocity_threshold: float = 0.01   # Min daily_volume/capacity ratio (1% daily turnover)
    new_channel_grace_days: int = 7        # Days before velocity gate applies to new channels

    # ==========================================================================
    # DTS (Discounted Thompson Sampling) Parameters
    # ==========================================================================
    thompson_prior_std_fee: int = 100         # Default prior uncertainty in ppm
    thompson_observation_decay_hours: int = 168  # 7-day half-life for observations
    thompson_max_observations: int = 200      # Bounded memory per channel
    thompson_min_observations: int = 3        # Minimum before trusting posterior


    # ==========================================================================
    # Routing Intelligence Integration
    # ==========================================================================

    # Capacity Planner
    planner_enabled: bool = False
    planner_interval: int = 21600               # 6 hours
    planner_dry_run: bool = False
    planner_execute_closes: bool = False
    planner_max_opens_per_cycle: int = 1
    planner_max_closes_per_cycle: int = 0
    planner_close_fee_reserve_multiplier: float = 2.0
    planner_close_fee_cap_sats: int = 0
    planner_close_feerange_enabled: bool = False
    planner_min_channel_sats: int = 500000      # 500k sats
    planner_max_channel_sats: int = 10000000    # 10M sats
    planner_min_channel_age_days: int = 30
    planner_min_peer_uptime_pct: float = 95.0
    planner_max_fee_rate_sat_vb: float = 50.0
    planner_min_annual_roi_pct: float = 1.0
    # Hive Hints integration
    hive_hints_enabled: bool = True
    hive_hints_ttl_seconds: int = 0  # 0 = use snapshot's ttl_seconds
    hive_hints_allow_all_hints_m2_scope: bool = False
    # Unified Capex Budget Engine
    capex_reinvestment_rate: float = 0.50       # Fraction of channel contribution for all capex
    capex_bootstrap_bps: int = 10               # Bootstrap: basis points of capacity per 30d
    capex_bootstrap_max_sats: int = 200         # Bootstrap cap per channel per 30d
    capex_grace_days: int = 14                  # Days before bootstrap activates
    capex_exploration_rate: float = 0.10        # Fleet contribution fraction for opens/growth
    capex_tactical_rate: float = 0.15           # Fleet contribution fraction for Boltz treasury
    capex_global_envelope_sats: int = 0         # Global cap (0 = auto-computed)
    capex_cost_efficiency_weight: float = 0.5   # Weight for cost-efficiency in dual-benefit score
    capex_drain_benefit_weight: float = 0.5     # Weight for drain-benefit in dual-benefit score
    # Probability-aware budget relaxation. When a router reports a route
    # probability (v3/askrene does; v2/getroute returns 0), the engine allows
    # the route to exceed the raw pair budget by up to (probability * bonus)
    # fraction. Default 0.0 = disabled, preserving v2 behavior exactly.
    capex_probability_budget_bonus: float = 0.0

    # Internal version tracking (not a user-configurable option)
    _version: int = field(default=0, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _override_warnings: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate cross-field invariants on direct construction."""
        if self.hive_equalization_low_pct >= self.hive_equalization_high_pct:
            raise ValueError(
                "hive_equalization_low_pct must be less than "
                "hive_equalization_high_pct"
            )
        # Source-heavy drain fields: enforce ranges at construction time so a
        # bad receivable objective or discount cap can never be instantiated.
        for _key in (
            'receivable_ratio_target',
            'receivable_ratio_floor',
            'boltz_structural_budget_sats_per_day',
            'drain_fee_discount_max',
        ):
            _min_val, _max_val = CONFIG_FIELD_RANGES[_key]
            _val = getattr(self, _key)
            if not (_min_val <= _val <= _max_val):
                raise ValueError(
                    f"{_key} must be between {_min_val} and {_max_val}, got {_val}"
                )
        if self.receivable_ratio_floor > self.receivable_ratio_target:
            raise ValueError(
                "receivable_ratio_floor must not exceed receivable_ratio_target"
            )
        profile = str(self.fee_profile or "active").lower()
        if profile not in STRING_ENUM_VALID_VALUES["fee_profile"]:
            raise ValueError(
                "fee_profile must be one of: "
                + ", ".join(STRING_ENUM_VALID_VALUES["fee_profile"])
            )
        self.fee_profile = profile
        router = str(self.rebalance_router or "v3").lower()
        if router not in STRING_ENUM_VALID_VALUES["rebalance_router"]:
            raise ValueError(
                "rebalance_router only supports 'v3'; legacy 'v2' routing was removed"
            )
        self.rebalance_router = router
    
    def snapshot(self) -> 'ConfigSnapshot':
        """
        Create an immutable snapshot for cycle execution.
        
        All worker cycles MUST capture a snapshot at cycle start and use
        only that snapshot for the duration of the cycle. This prevents
        torn reads when config is updated mid-cycle.
        """
        return ConfigSnapshot.from_config(self)

    def public_runtime_keys(self) -> list[str]:
        """Return the supported public runtime controls."""
        return list(PUBLIC_RUNTIME_KEYS)

    @classmethod
    def is_public_runtime_key(cls, key: str) -> bool:
        """Return True when the key is part of the supported operator surface."""
        return key in PUBLIC_RUNTIME_KEYS

    def public_runtime_dict(self) -> Dict[str, Any]:
        """Return only the supported public runtime controls and their values."""
        return {key: getattr(self, key) for key in PUBLIC_RUNTIME_KEYS}

    @classmethod
    def classify_runtime_key(cls, key: str) -> str:
        """Classify a runtime key for operator-surface decisions."""
        if cls.is_public_runtime_key(key):
            return "public"
        if key in DEPRECATED_RUNTIME_KEYS:
            return "deprecated"
        return "internal"
    
    def load_overrides(self, database: 'Database') -> list:
        """Load config overrides from database on startup. Returns list of warnings."""
        self._override_warnings.clear()
        overrides = database.get_all_config_overrides()
        for key, value in overrides.items():
            if hasattr(self, key) and key not in IMMUTABLE_CONFIG_KEYS:
                self._apply_override(key, value)
        self._version = database.get_config_version()
        # M-R5-2 FIX: Post-load cross-field invariant repair.
        # Overrides applied individually may violate cross-field constraints
        # (e.g., min_fee_ppm > max_fee_ppm from TOCTOU race or manual DB edits).
        if self.min_fee_ppm > self.max_fee_ppm:
            self.min_fee_ppm = self.max_fee_ppm
        if self.rebalance_min_amount > self.rebalance_max_amount:
            self.rebalance_min_amount = self.rebalance_max_amount
        if hasattr(self, 'low_liquidity_threshold') and hasattr(self, 'high_liquidity_threshold'):
            if self.low_liquidity_threshold >= self.high_liquidity_threshold:
                # M-R6-1 FIX: Clamp to 0.0 to prevent negative values when
                # high_liquidity_threshold is very small (e.g., < 0.05).
                self.low_liquidity_threshold = max(0.0, self.high_liquidity_threshold - 0.05)
        if hasattr(self, 'hive_equalization_low_pct') and hasattr(self, 'hive_equalization_high_pct'):
            if self.hive_equalization_low_pct >= self.hive_equalization_high_pct:
                self.hive_equalization_low_pct = max(
                    0.0, self.hive_equalization_high_pct - 0.05
                )
        if self.receivable_ratio_floor > self.receivable_ratio_target:
            self.receivable_ratio_floor = self.receivable_ratio_target
        return list(self._override_warnings)

    def _apply_override(self, key: str, value: str) -> None:
        """Apply a single override with type conversion and range validation."""
        field_type = CONFIG_FIELD_TYPES.get(key, str)
        try:
            if field_type == bool:
                typed_value = value.lower() in ('true', '1', 'yes', 'on')
            elif field_type == int:
                typed_value = int(value)
            elif field_type == float:
                typed_value = float(value)
                # AUDIT FIX C-2: Reject NaN/Infinity for float fields
                if not math.isfinite(typed_value):
                    self._override_warnings.append(f"Skipped non-finite override for {key}: {value}")
                    return
            else:
                typed_value = value
            # Range validation (matching update_runtime behavior)
            if key in CONFIG_FIELD_RANGES:
                min_val, max_val = CONFIG_FIELD_RANGES[key]
                if not (min_val <= typed_value <= max_val):
                    self._override_warnings.append(f"Skipped out-of-range override for {key}: {typed_value} not in [{min_val}, {max_val}]")
                    return
            # String enum validation (matching update_runtime behavior)
            if key in STRING_ENUM_VALID_VALUES:
                valid_values = STRING_ENUM_VALID_VALUES[key]
                if typed_value not in valid_values and (not isinstance(typed_value, str) or typed_value.lower() not in [v.lower() for v in valid_values]):
                    self._override_warnings.append(f"Skipped invalid enum override for {key}: {typed_value}")
                    return
                if isinstance(typed_value, str):
                    typed_value = typed_value.lower()
            setattr(self, key, typed_value)
        except (ValueError, TypeError, AttributeError) as e:
            self._override_warnings.append(f"Override conversion failed for {key}={value}: {e}")
    
    def update_runtime(self, database: 'Database', key: str, value: str) -> Dict[str, Any]:
        """
        Transactional runtime update: Validate → Write DB → Read-Back → Update Memory.
        
        This implements the CRITICAL-02/CRITICAL-03 defenses from the Red Team report:
        - ConfigSnapshot pattern prevents torn reads
        - Transactional update prevents Ghost Config
        
        Returns:
            Dict with status, old_value, new_value, version
        """
        # 1. VALIDATE: Check if key exists and is mutable
        if key in IMMUTABLE_CONFIG_KEYS:
            return {"error": f"Key '{key}' cannot be changed at runtime"}
        
        if not hasattr(self, key) or key.startswith('_'):
            return {"error": f"Unknown config key: {key}"}
        
        # 2. VALIDATE: Type check
        field_type = CONFIG_FIELD_TYPES.get(key, str)
        try:
            if field_type == bool:
                typed_value = value.lower() in ('true', '1', 'yes', 'on')
            elif field_type == int:
                typed_value = int(value)
            elif field_type == float:
                typed_value = float(value)
                # AUDIT FIX C-2: Reject NaN/Infinity for float fields
                if not math.isfinite(typed_value):
                    return {"error": f"Value {value} is not a finite number for {key}"}
            else:
                typed_value = value
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid value for {key} (expected {field_type.__name__}): {e}"}
        
        # 3. VALIDATE: Range check
        if key in CONFIG_FIELD_RANGES:
            min_val, max_val = CONFIG_FIELD_RANGES[key]
            if not (min_val <= typed_value <= max_val):
                return {"error": f"Value {typed_value} out of range [{min_val}, {max_val}] for {key}"}

        # 3b. VALIDATE: String enum check
        if key in STRING_ENUM_VALID_VALUES:
            valid_values = STRING_ENUM_VALID_VALUES[key]
            # Case-insensitive comparison for string enums
            if typed_value not in valid_values and (not isinstance(typed_value, str) or typed_value.lower() not in[v.lower() for v in valid_values]):
                valid = ', '.join(valid_values)
                return {"error": f"Invalid value '{typed_value}' for {key}. Valid values: {valid}"}
            # Normalize string enums to lowercase for consistent consumer comparisons
            if isinstance(typed_value, str):
                typed_value = typed_value.lower()

        # 4-6. WRITE + VERIFY + UPDATE under lock to prevent TOCTOU races
        with self._lock:
            # M-R5-1 FIX: Cross-field consistency checks INSIDE lock to prevent
            # TOCTOU race where a concurrent update changes the companion field
            # between our check and our write.
            if key == 'min_fee_ppm' and typed_value > self.max_fee_ppm:
                return {"error": f"min_fee_ppm ({typed_value}) cannot exceed current max_fee_ppm ({self.max_fee_ppm})"}
            if key == 'max_fee_ppm' and typed_value < self.min_fee_ppm:
                return {"error": f"max_fee_ppm ({typed_value}) cannot be less than current min_fee_ppm ({self.min_fee_ppm})"}
            if key == 'rebalance_min_amount' and typed_value > self.rebalance_max_amount:
                return {"error": f"rebalance_min_amount ({typed_value}) cannot exceed rebalance_max_amount ({self.rebalance_max_amount})"}
            if key == 'rebalance_max_amount' and typed_value < self.rebalance_min_amount:
                return {"error": f"rebalance_max_amount ({typed_value}) cannot be less than rebalance_min_amount ({self.rebalance_min_amount})"}
            # M-R5-4 FIX: Also validate liquidity threshold cross-field consistency
            if key == 'low_liquidity_threshold' and typed_value >= self.high_liquidity_threshold:
                return {"error": f"low_liquidity_threshold ({typed_value}) must be less than high_liquidity_threshold ({self.high_liquidity_threshold})"}
            if key == 'high_liquidity_threshold' and typed_value <= self.low_liquidity_threshold:
                return {"error": f"high_liquidity_threshold ({typed_value}) must be greater than low_liquidity_threshold ({self.low_liquidity_threshold})"}
            if key == 'hive_equalization_low_pct' and typed_value >= self.hive_equalization_high_pct:
                return {"error": f"hive_equalization_low_pct ({typed_value}) must be less than hive_equalization_high_pct ({self.hive_equalization_high_pct})"}
            if key == 'hive_equalization_high_pct' and typed_value <= self.hive_equalization_low_pct:
                return {"error": f"hive_equalization_high_pct ({typed_value}) must be greater than hive_equalization_low_pct ({self.hive_equalization_low_pct})"}
            # AUDIT FIX I-5: Validate sink/source threshold cross-field consistency
            if key == 'sink_threshold' and typed_value >= self.source_threshold:
                return {"error": f"sink_threshold ({typed_value}) must be less than source_threshold ({self.source_threshold})"}
            if key == 'source_threshold' and typed_value <= self.sink_threshold:
                return {"error": f"source_threshold ({typed_value}) must be greater than sink_threshold ({self.sink_threshold})"}
            # Receivable objective: floor must never exceed target (mirrors
            # the __post_init__ invariant for construction-time values).
            if key == 'receivable_ratio_floor' and typed_value > self.receivable_ratio_target:
                return {"error": f"receivable_ratio_floor ({typed_value}) cannot exceed receivable_ratio_target ({self.receivable_ratio_target})"}
            if key == 'receivable_ratio_target' and typed_value < self.receivable_ratio_floor:
                return {"error": f"receivable_ratio_target ({typed_value}) cannot be less than receivable_ratio_floor ({self.receivable_ratio_floor})"}

            old_value = getattr(self, key)

            # 4. WRITE to database
            new_version = database.set_config_override(key, value)

            # 5. READ-BACK verification (prevents Ghost Config - CRITICAL-03)
            read_back = database.get_config_override(key)
            if read_back != value:
                # Roll back the DB write to prevent phantom config on restart
                try:
                    database.delete_config_override(key)
                except Exception:
                    pass  # Best-effort cleanup
                return {"error": "Database write verification failed (Ghost Config prevention)"}

            # 6. UPDATE in-memory
            setattr(self, key, typed_value)
            self._version = new_version

        return {
            "status": "success",
            "key": key,
            "old_value": old_value,
            "new_value": typed_value,
            "version": new_version
        }


@dataclass(frozen=True)
class ConfigSnapshot:
    """
    Immutable configuration snapshot for thread-safe cycle execution.
    
    All worker cycles MUST capture a snapshot at cycle start and use
    only that snapshot for the duration of the cycle. This prevents
    torn reads when config is updated mid-cycle (CRITICAL-02 defense).
    
    Usage:
        def run_cycle(self):
            cfg = self.config.snapshot()  # Immutable for this cycle
            # All logic uses cfg, never self.config directly
    """
    # Database path
    db_path: str
    
    # Timer intervals (in seconds)
    flow_interval: int
    fee_interval: int
    rebalance_interval: int
    paused: bool
    hot_channel_protection_enabled: bool
    hot_channel_protection_override_peers: str
    hot_channel_protection_min_velocity: float
    hot_channel_protection_min_marginal_roi: float
    hot_channel_protection_profit_budget_pct: float
    hot_channel_protection_max_chunk_multiplier: float
    hot_channel_protection_min_cooldown_hours: float
    hot_channel_protection_max_rebalance_fee_ppm: int
    boltz_auto_cycle_enabled: bool
    boltz_auto_cycle_interval_minutes: int
    boltz_auto_cycle_max_actions: int
    boltz_auto_cycle_startup_delay_seconds: int
    expansion_treasury_enabled: bool
    expansion_treasury_onchain_target_sats: int
    expansion_treasury_min_deficit_sats: int
    expansion_treasury_preferred_currency: str
    expansion_treasury_max_actions: int
    expansion_treasury_min_source_local_pct: float
    expansion_treasury_exclude_protected: bool
    
    # Flow analysis parameters
    target_flow: int
    flow_window_days: int
    
    # Flow ratio thresholds for classification
    source_threshold: float
    sink_threshold: float
    
    # Fee parameters
    min_fee_ppm: int
    max_fee_ppm: int
    base_fee_msat: int
    market_fee_mode: str
    base_fee_policy: str
    base_fee_msat_intra_fleet: int
    base_fee_msat_non_hive: int
    fee_ppm_intra_fleet: int
    neighbor_median_min_competitors: int
    fee_profile: str
    fee_market_boundary_enabled: bool
    fee_market_boundary_min_competitors: int
    fee_market_boundary_margin_ppm: int
    fee_market_boundary_margin_ratio: float
    fee_market_boundary_max_downshift_ratio: float
    fee_market_boundary_cache_seconds: int
    # Rebalancing parameters
    rebalance_min_profit: int
    rebalance_min_profit_ppm: int
    rebalance_max_amount: int
    rebalance_min_amount: int
    low_liquidity_threshold: float
    high_liquidity_threshold: float
    rebalance_cooldown_hours: int
    rebalance_emergency_local_ratio: float
    rebalance_drift_override_ratio: float
    rebalance_hold_margin: float
    pair_fee_cap_ppm: int
    hive_equalization_enabled: bool
    hive_equalization_low_pct: float
    hive_equalization_high_pct: float
    hive_equalization_cooldown_hours: int
    hive_equalization_max_candidates_per_cycle: int
    hive_rebalance_bootstrap_budget_sats: int
    hive_push_enabled: bool
    hive_push_trigger_ratio: float
    hive_push_target_ratio: float
    rebalance_coordination_reserved_slots: int
    futility_cooldown_hours: int
    inbound_fee_estimate_ppm: int

    # Profitability tracking
    estimated_open_cost_sats: int
    
    # Global Capital Controls
    daily_budget_sats: int
    allow_zero_cost_auto_rebalance_when_budget_zero: bool
    min_wallet_reserve: int

    # HTLC Congestion threshold
    htlc_congestion_threshold: float
    
    # Reputation-weighted volume
    enable_reputation: bool
    reputation_decay: float

    # Kelly Criterion Position Sizing
    enable_kelly: bool
    kelly_fraction: float
    
    # Async Job Queue
    max_concurrent_jobs: int

    # Safety flags
    dry_run: bool

    # Vegas Reflex (mempool spike defense)
    enable_vegas_reflex: bool
    vegas_decay_rate: float
    
    # Deferred features

    # RPC Hardening
    rpc_timeout_seconds: int
    rpc_circuit_breaker_seconds: int
    rpc_pool_size: int
    reservation_timeout_hours: int

    # Issue #28: Revenue rate EMA smoothing
    ema_smoothing_alpha: float

    # Issue #30: Velocity gate for rebalancing
    enable_velocity_gate: bool
    min_velocity_threshold: float
    new_channel_grace_days: int

    # DTS (Discounted Thompson Sampling) parameters
    thompson_prior_std_fee: int
    thompson_observation_decay_hours: int
    thompson_max_observations: int
    thompson_min_observations: int
    # Routing Intelligence Integration

    # M-27: xpay/askrene parameters (were missing from snapshot)
    askrene_layer: str = 'xpay'
    askrene_max_age_sec: int = 900

    # V3 rebalance router (askrene getroutes + cl-hive layers)
    rebalance_router: str = 'v3'
    askrene_layers: str = 'hive-fleet'

    # Weekly budget cap (hard ceiling over daily burst)
    weekly_budget_sats: int = 35000

    # Capacity Planner
    planner_enabled: bool = False
    planner_interval: int = 21600
    planner_dry_run: bool = False
    planner_execute_closes: bool = False
    planner_max_opens_per_cycle: int = 1
    planner_max_closes_per_cycle: int = 0
    planner_close_fee_reserve_multiplier: float = 2.0
    planner_close_fee_cap_sats: int = 0
    planner_close_feerange_enabled: bool = False
    planner_min_channel_sats: int = 500000
    planner_max_channel_sats: int = 10000000
    planner_min_channel_age_days: int = 30
    planner_min_peer_uptime_pct: float = 95.0
    planner_max_fee_rate_sat_vb: float = 50.0
    planner_min_annual_roi_pct: float = 1.0
    # Hive Hints
    hive_hints_enabled: bool = True
    hive_hints_ttl_seconds: int = 0
    # Unified Capex Budget Engine
    capex_reinvestment_rate: float = 0.50
    capex_bootstrap_bps: int = 10
    capex_bootstrap_max_sats: int = 200
    capex_grace_days: int = 14
    capex_exploration_rate: float = 0.10
    capex_tactical_rate: float = 0.15
    capex_global_envelope_sats: int = 0
    capex_cost_efficiency_weight: float = 0.5
    capex_drain_benefit_weight: float = 0.5
    capex_probability_budget_bonus: float = 0.0
    # Structural loop-out / drain-demand fields
    receivable_ratio_target: float = 0.30
    receivable_ratio_floor: float = 0.20
    boltz_structural_budget_sats_per_day: int = 0
    drain_fee_discount_max: float = 0.0
    # Version tracking
    version: int = 0
    
    @classmethod
    def from_config(cls, config: 'Config') -> 'ConfigSnapshot':
        """Create snapshot from mutable Config. Auto-maps matching field names.

        Backward-compatibility: if ConfigSnapshot gains new fields before the mutable
        Config dataclass is updated in a partial deployment, fall back to the snapshot
        field's declared default/default_factory instead of raising TypeError.
        """
        with config._lock:
            kwargs = {}
            for f in dataclasses.fields(cls):
                if f.name == 'version':
                    kwargs['version'] = config._version
                elif hasattr(config, f.name):
                    kwargs[f.name] = getattr(config, f.name)
                elif f.default is not dataclasses.MISSING:
                    kwargs[f.name] = f.default
                elif getattr(f, 'default_factory', dataclasses.MISSING) is not dataclasses.MISSING:
                    kwargs[f.name] = f.default_factory()
        return cls(**kwargs)


# Default chain cost assumptions for fee floor calculation
class ChainCostDefaults:
    """
    Default assumptions for calculating the economic fee floor.
    
    The floor is calculated as:
    floor_ppm = (channel_open_cost + channel_close_cost) / estimated_lifetime_volume * 1_000_000
    
    This ensures we never charge less than what it costs us to maintain the channel.
    """
    
    # Estimated on-chain costs in sats
    CHANNEL_OPEN_COST_SATS: int = 5000      # ~$3-5 at typical fee rates
    CHANNEL_CLOSE_COST_SATS: int = 3000     # Usually cheaper than open


    # Estimated channel lifetime
    CHANNEL_LIFETIME_DAYS: int = 365        # 1 year average
    
    # Estimated routing volume per day (conservative)
    DAILY_VOLUME_SATS: int = 1000000        # 1M sats/day
    
    @classmethod
    def calculate_floor_ppm(cls, capacity_sats: int, opener: str = "local") -> int:
        """
        Calculate the economic floor fee for a channel.

        Args:
            capacity_sats: Channel capacity in satoshis
            opener: Who opened the channel ('local' or 'remote').
                    Remote-opened channels exclude the open cost since we didn't pay it.

        Returns:
            Minimum fee in PPM that covers channel costs
        """
        if opener == "remote":
            total_chain_cost = cls.CHANNEL_CLOSE_COST_SATS  # We didn't pay to open
        else:
            total_chain_cost = cls.CHANNEL_OPEN_COST_SATS + cls.CHANNEL_CLOSE_COST_SATS
        estimated_lifetime_volume = cls.DAILY_VOLUME_SATS * cls.CHANNEL_LIFETIME_DAYS
        
        # Calculate minimum fee to break even
        # floor_ppm = cost / volume * 1_000_000
        if estimated_lifetime_volume > 0:
            floor_ppm = (total_chain_cost / estimated_lifetime_volume) * 1_000_000
            return max(1, int(floor_ppm))
        return 1


# Liquidity bucket definitions for fee tiers
class LiquidityBuckets:
    """
    Define liquidity buckets for tiered fee strategies.
    
    Different liquidity levels warrant different fee approaches:
    - Very low outbound: High fees (scarce resource)
    - Low outbound: Above average fees
    - Balanced: Target fees
    - High outbound: Below average fees  
    - Very high outbound: Low fees (encourage usage)
    """
    
    VERY_LOW = 0.1    # < 10% outbound
    LOW = 0.25        # 10-25% outbound
    BALANCED_LOW = 0.4   # 25-40% outbound
    BALANCED_HIGH = 0.6  # 40-60% outbound (ideal)
    HIGH = 0.75       # 60-75% outbound
    VERY_HIGH = 0.9   # > 75% outbound
    
    @classmethod
    def get_bucket(cls, outbound_ratio: float) -> str:
        """
        Classify a channel by its outbound liquidity ratio.
        
        Args:
            outbound_ratio: outbound_sats / capacity_sats
            
        Returns:
            Bucket name string
        """
        if outbound_ratio < cls.VERY_LOW:
            return "very_low"
        elif outbound_ratio < cls.LOW:
            return "low"
        elif outbound_ratio < cls.BALANCED_LOW:
            return "balanced_low"
        elif outbound_ratio < cls.BALANCED_HIGH:
            return "balanced"
        elif outbound_ratio < cls.HIGH:
            return "balanced_high"
        elif outbound_ratio < cls.VERY_HIGH:
            return "high"
        else:
            return "very_high"
    
    @classmethod
    def get_fee_multiplier(cls, bucket: str) -> float:
        """
        Get fee multiplier for a liquidity bucket.
        
        Args:
            bucket: Bucket name from get_bucket()
            
        Returns:
            Multiplier to apply to base fee
        """
        multipliers = {
            "very_low": 3.0,      # Triple fees when nearly depleted
            "low": 2.0,           # Double fees when low
            "balanced_low": 1.25, # Slightly above average
            "balanced": 1.0,      # Target fee
            "balanced_high": 0.85,# Slightly below average
            "high": 0.7,          # Reduced fees to encourage routing
            "very_high": 0.5      # Half fees when overloaded
        }
        return multipliers.get(bucket, 1.0)
