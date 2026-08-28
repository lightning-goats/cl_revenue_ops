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
    # E-3 (2026-07 econ audit): weekly cap must be raisable live — daily was
    # raised to 5000 but the weekly ceiling silently capped it.
    'weekly_budget_sats',
    'growth_budget_enabled',
    'growth_budget_earned_fraction',
    'growth_budget_experiment_fraction',
    'growth_budget_max_extra_sats',
    'growth_budget_hard_ceiling_sats',
    'min_fee_ppm',
    # E-2 (2026-07 econ audit): class-aware floor — saturated/source channels
    # may advertise below min_fee_ppm (fee-band decompression). 0 = allow
    # true cheap egress on saturated edges (the default).
    'min_fee_ppm_saturated',
    # Default-off, single-lane bounded zero-fee market-acquisition experiment.
    'acquisition_experiment_enabled',
    'max_fee_ppm',
    'fee_profile',
    'fee_market_boundary_enabled',
    'fee_market_boundary_min_competitors',
    'fee_market_boundary_margin_ppm',
    'fee_market_boundary_margin_ratio',
    'fee_market_boundary_max_downshift_ratio',
    'fee_market_boundary_cache_seconds',
    # V3 router probability-aware budget relaxation (default 0.0 = off).
    # Exposed so operators running the askrene router can enable the
    # reliability-weighted budget bonus without editing code or the database.
    'capex_probability_budget_bonus',
    # Source-heavy drain rollout knobs. Plugin options in the config file
    # only load at lightningd startup; these are exposed at runtime so the
    # drain can be enabled and tuned via revenue-config without a daemon
    # restart. All default to off/neutral.
    'receivable_ratio_target',
    'receivable_ratio_floor',
    'drain_fee_discount_max',
    'node_drain_bias_enabled',
    'node_drain_bias_max',
    # Dynamic htlc_max flow valve (H-2, 2026-07-03 audit): scales each
    # channel's max HTLC by flow state so sinks throttle drain-through.
    # CLN has no negative inbound fees; this is the inbound-side control
    # surface. Default off.
    'enable_dynamic_htlcmax',
    'htlcmax_source_pct',
    'htlcmax_sink_pct',
    'htlcmax_balanced_pct',
    # Refactor Phase 1 shadow (docs/planning/2026-07-12-refactor-phase1-
    # wiring.md): observe-mode intent recording + snapshot preview.
    # Read-only wrt node state; default off.
    'econ_shadow_enabled',
    # Refactor Phase 2D: rebalance reservations gated by the governor
    # facade (same reserve_budget accounting, new authorization
    # boundary). Instant rollback by setting false. Default off.
    'econ_governor_rebalance_enabled',
    # Refactor Phase 2H: automated fee broadcasts gated by the governor
    # (paused/stale + audit trail; zero-cost, no reservation). Manual
    # revenue-set-fee stays operator-direct. Default off.
    'econ_governor_fees_enabled',
    # Refactor Phase 3F: live conflict arbitration at the governor
    # (duplicate suppression for retained intents). Default off.
    'econ_arbiter_enabled',
    # Workstream H cutover: the rebalance loop's execution list passes
    # through cycle intent generation + batch arbitration. Default off.
    'econ_cycle_rebalance_enabled',
    # PR 6 (gap-closure Phase E): populate real EV/confidence in intent
    # envelopes. CAUTION: flipping changes J3 batch-arbitration ORDER in
    # the cutover loops (EV sorts before target). Default off = zeros.
    'econ_ev_populated',
    # PR 10 (gap-closure Phase G): extra arbiter conflict rules —
    # retained extended intent-conflict rules. Default off preserves the
    # original arbitration set.
    'econ_conflict_rules_extended',
    # Phase 4 (Workstream I): global authority level — observe < fees <
    # liquidity < capital. Governed actions above the level are blocked
    # with AUTHORITY_LEVEL_BLOCKED. Default 'capital' preserves current
    # behavior; operators dial DOWN.
    'authority_level',
    # PR 7 (gap-closure Phase D): coherent economic-risk profile.
    # 'custom' (default) derives nothing — exact current behavior.
    # Non-custom bundles apply AT STARTUP to keys without explicit
    # overrides (precedence: explicit > profile > default).
    'risk_profile',
)

# Type mapping for config fields (for validation)
CONFIG_FIELD_TYPES: Dict[str, type] = {
    'flow_interval': int,
    'fee_interval': int,
    'rebalance_interval': int,
    'paused': bool,
    'min_fee_ppm': int,
    'min_fee_ppm_saturated': int,
    'acquisition_experiment_enabled': bool,
    'max_fee_ppm': int,
    'fee_profile': str,
    'fee_market_boundary_enabled': bool,
    'fee_market_boundary_min_competitors': int,
    'fee_market_boundary_margin_ppm': int,
    'fee_market_boundary_margin_ratio': float,
    'fee_market_boundary_max_downshift_ratio': float,
    'fee_market_boundary_cache_seconds': int,
    'daily_budget_sats': int,
    'growth_budget_enabled': bool,
    'growth_budget_earned_fraction': float,
    'growth_budget_experiment_fraction': float,
    'growth_budget_max_extra_sats': int,
    'growth_budget_hard_ceiling_sats': int,
    'allow_zero_cost_auto_rebalance_when_budget_zero': bool,
    'weekly_budget_sats': int,
    'hot_channel_protection_enabled': bool,
    'hot_channel_protection_override_peers': str,
    'hot_channel_protection_min_velocity': float,
    'hot_channel_protection_min_marginal_roi': float,
    'hot_channel_protection_profit_budget_pct': float,
    'hot_channel_protection_max_chunk_multiplier': float,
    'hot_channel_protection_min_cooldown_hours': float,
    'receivable_ratio_target': float,
    'receivable_ratio_floor': float,
    'drain_fee_discount_max': float,
    'node_drain_bias_enabled': bool,
    'node_drain_bias_max': float,
    'enable_dynamic_htlcmax': bool,
    'htlcmax_source_pct': float,
    'htlcmax_sink_pct': float,
    'htlcmax_balanced_pct': float,
    'econ_shadow_enabled': bool,
    'econ_governor_rebalance_enabled': bool,
    'econ_governor_fees_enabled': bool,
    'econ_arbiter_enabled': bool,
    'econ_cycle_rebalance_enabled': bool,
    'econ_ev_populated': bool,
    'econ_conflict_rules_extended': bool,
    'authority_level': str,
    'risk_profile': str,
    'min_wallet_reserve': int,
    'low_liquidity_threshold': float,
    'high_liquidity_threshold': float,
    'htlc_congestion_threshold': float,
    'enable_reputation': bool,
    'enable_kelly': bool,
    'reputation_decay': float,
    'max_concurrent_jobs': int,
    'askrene_layer': str,
    'askrene_layers': str,
    'rebalance_router': str,
    'rebalance_min_profit': int,
    'rebalance_max_amount': int,
    'rebalance_cooldown_hours': int,
    'rebalance_emergency_local_ratio': float,
    'rebalance_drift_override_ratio': float,
    'rebalance_hold_margin': float,
    'pair_fee_cap_ppm': int,
    # Upstream rebalancer patterns
    'rebalance_activity_window_seconds': int,
    'rebalance_activity_penalty_coeff': float,
    'rebalance_activity_penalty_cap_frac': float,
    'rebalance_utilization_window_days': int,
    'rebalance_utilization_floor': float,
    'rebalance_utilization_ceiling': float,
    'rebalance_utilization_min_forwards': int,
    'rebalance_size_tiered_targets': bool,
    'rebalance_size_reference_percentile': float,
    'rebalance_small_channel_band_half_width': float,
    'inbound_fee_estimate_ppm': int,
    # Vegas Reflex
    'enable_vegas_reflex': bool,
    'vegas_decay_rate': float,
    # Operational Hardening
    'rpc_timeout_seconds': int,
    'reservation_timeout_hours': int,
    # Issue #28: Revenue rate smoothing
    # Issue #30: Velocity gate for rebalancing
    # DTS (Discounted Thompson Sampling) parameters
    'thompson_prior_std_fee': int,
    # Routing Intelligence Integration
    # Fields present in CONFIG_FIELD_RANGES that need type registration
    'base_fee_msat': int,
    'neighbor_median_min_competitors': int,
    'flow_window_days': int,
    'estimated_open_cost_sats': int,
    'source_threshold': float,
    'sink_threshold': float,
    # Unified Capex Budget Engine
    'capex_reinvestment_rate': float,
    'capex_bootstrap_bps': int,
    'capex_bootstrap_max_sats': int,
    'capex_grace_days': int,
    'capex_probability_budget_bonus': float,
    'capex_global_envelope_sats': int,
}

# Explicit migration shims only. Non-public keys remain internal until they are
# intentionally exposed as deprecated compatibility controls.
# E-4.5 (2026-07 econ audit): rebalance_min_profit was parsed and echoed but
# enforced nowhere; the sats-EV gate's rebalance_hold_margin covers the
# semantics. Kept as a deprecated compatibility shim so config files load.
DEPRECATED_RUNTIME_KEYS: FrozenSet[str] = frozenset({'rebalance_min_profit'})

# Replacement guidance surfaced in the startup deprecation warning.
DEPRECATED_KEY_REPLACEMENTS: Dict[str, str] = {
    'rebalance_min_profit': 'rebalance_hold_margin',
}

# Workstream I: settings whose effect is entirely gated behind a boolean
# flag. An explicit override of a gated key while its gate is off is
# "shadowed" — legal, but it does nothing until the gate is enabled, so
# startup warns instead of leaving the operator to discover it.
SHADOWED_SETTING_GATES: Dict[str, tuple] = {
    'growth_budget_enabled': (
        'growth_budget_earned_fraction',
        'growth_budget_experiment_fraction',
        'growth_budget_max_extra_sats',
        'growth_budget_hard_ceiling_sats',
    ),
    'fee_market_boundary_enabled': (
        'fee_market_boundary_min_competitors',
        'fee_market_boundary_margin_ppm',
    ),
    # Phase B (2026-08-01 surface reduction, section 5.4): the structural
    # cycle — the exact shadow found in production.
    # per-cycle open/close limits are inert until the planner is on.
}

# Reverse map for set-time shadowed-gate warnings on revenue-config set:
# gated key -> the boolean gate that must be on for it to have any effect.
_GATE_FOR_GATED_KEY: Dict[str, str] = {
    gated_key: gate
    for gate, gated_keys in SHADOWED_SETTING_GATES.items()
    for gated_key in gated_keys
}

# Phase B (2026-08-01 surface reduction): keys that still WORK but are
# scheduled for removal in the announced Phase C window. Unlike
# DEPRECATED_RUNTIME_KEYS these are NOT no-ops yet — the warning must not
# claim otherwise. revenue-config set returns the note; startup warns when
# an override row exists.
SOFT_DEPRECATED_RUNTIME_KEYS: Dict[str, str] = {
    'capex_probability_budget_bonus': (
        'capex_probability_budget_bonus is deprecated and scheduled for '
        'removal in the announced surface-reduction window (Phase C); it '
        'is still enforced until then'),
}

# Range constraints for numeric fields
CONFIG_FIELD_RANGES: Dict[str, tuple] = {
    'min_fee_ppm': (5, 100000),  # CRITICAL-02 FIX: Minimum 5 PPM to ensure economic viability
    # E-2: class floor for saturated/source channels. 0 (default) allows true
    # cheap egress; it only ever LOWERS the floor (values >= min_fee_ppm are
    # ignored at use), so the CRITICAL-02 global minimum is not weakened for
    # any other class.
    'min_fee_ppm_saturated': (0, 1000),
    'max_fee_ppm': (1, 100000),
    'fee_market_boundary_min_competitors': (1, 100),
    'fee_market_boundary_margin_ppm': (0, 10000),
    'fee_market_boundary_margin_ratio': (0.0, 0.50),
    'fee_market_boundary_max_downshift_ratio': (0.05, 1.0),
    'fee_market_boundary_cache_seconds': (10, 3600),
    'daily_budget_sats': (0, 10000000),
    'growth_budget_earned_fraction': (0.0, 1.0),
    'growth_budget_experiment_fraction': (0.0, 1.0),
    'growth_budget_max_extra_sats': (0, 1_000_000),
    'growth_budget_hard_ceiling_sats': (0, 10_000_000),
    'weekly_budget_sats': (0, 70_000_000),
    'receivable_ratio_target': (0.0, 1.0),
    'receivable_ratio_floor': (0.0, 1.0),
    'drain_fee_discount_max': (0.0, 0.5),
    'node_drain_bias_max': (0.0, 0.5),
    'htlcmax_source_pct': (0.01, 1.0),
    'htlcmax_sink_pct': (0.01, 1.0),
    'htlcmax_balanced_pct': (0.01, 1.0),
    'min_wallet_reserve': (0, 100000000),
    'low_liquidity_threshold': (0.0, 1.0),
    'high_liquidity_threshold': (0.0, 1.0),
    'htlc_congestion_threshold': (0.0, 1.0),
    'reputation_decay': (0.0, 1.0),
    'vegas_decay_rate': (0.0, 1.0),
    'rpc_timeout_seconds': (1, 300),
    'reservation_timeout_hours': (1, 24),
    # Issue #28: Revenue rate smoothing
    # Issue #30: Velocity gate for rebalancing
    # DTS (Discounted Thompson Sampling) parameters
    'thompson_prior_std_fee': (10, 500),
    # Routing Intelligence Integration
    # Additional range validations
    # Polar and other small fast-settling graphs may safely use compressed
    # cycles. Production defaults remain unchanged; 15s is an explicit
    # operator choice, not a new default.
    'flow_interval': (15, 86400),
    'fee_interval': (15, 86400),
    'rebalance_interval': (15, 86400),
    'max_concurrent_jobs': (1, 20),
    'base_fee_msat': (0, 10000),
    'neighbor_median_min_competitors': (2, 50),
    'rebalance_min_profit': (0, 1000000),
    'rebalance_max_amount': (10000, 100000000),
    'flow_window_days': (1, 365),
    # AUDIT FIX C-2/I-4: Missing range validation for float/int fields
    # F1: full ±1.0 range still allowed, but defaults are ±0.05 — values
    # beyond ~±0.2 are effectively dead for typical daily turnover.
    'source_threshold': (-1.0, 1.0),
    'sink_threshold': (-1.0, 1.0),
    'hot_channel_protection_max_chunk_multiplier': (1.0, 20.0),
    'hot_channel_protection_min_cooldown_hours': (0.0, 168.0),
    'hot_channel_protection_min_marginal_roi': (0.0, 10.0),
    'hot_channel_protection_min_velocity': (0.0, 1.0),
    'hot_channel_protection_profit_budget_pct': (0.0, 1.0),
    'inbound_fee_estimate_ppm': (0, 5000),
    'rebalance_cooldown_hours': (1, 168),
    'rebalance_emergency_local_ratio': (0.0, 1.0),
    'rebalance_drift_override_ratio': (0.0, 1.0),
    # final_score is denominated in SATS of expected net value since the
    # sats-EV scoring change (not a 0-1 normalized score), so the hold margin
    # must admit sat-scale bars; 1000 sats is already a very aggressive hold.
    'rebalance_hold_margin': (0.0, 1000.0),
    'pair_fee_cap_ppm': (0, 100000),
    # Upstream rebalancer patterns
    'rebalance_activity_window_seconds': (60, 86400),
    'rebalance_activity_penalty_coeff': (0.0, 1.0),
    'rebalance_activity_penalty_cap_frac': (0.0, 1.0),
    'rebalance_utilization_window_days': (1, 90),
    'rebalance_utilization_floor': (0.0, 1.0),
    'rebalance_utilization_ceiling': (0.0, 1.0),
    'rebalance_utilization_min_forwards': (0, 1000),
    'rebalance_size_reference_percentile': (0.0, 1.0),
    'rebalance_small_channel_band_half_width': (0.0, 0.5),
    'estimated_open_cost_sats': (0, 1000000),
    # Z-2 (2026-07-08): 0 disables the grace fallback entirely (immediate
    # release on hint staleness); ceiling is 30 days.
    # Unified Capex Budget Engine
    'capex_reinvestment_rate': (0.0, 1.0),
    'capex_bootstrap_bps': (0, 100),
    'capex_bootstrap_max_sats': (0, 10000),
    'capex_grace_days': (0, 90),
    'capex_probability_budget_bonus': (0.0, 1.0),
    'capex_global_envelope_sats': (0, 100_000_000),
}

# Valid values for string enum fields
STRING_ENUM_VALID_VALUES: Dict[str, tuple] = {
    'fee_profile': ('active', 'conservative'),
    'rebalance_router': ('v3',),
    'market_fee_mode': ('undercut', 'match', 'premium', 'competition_aware'),
    'risk_profile': ('preserve', 'conservative', 'balanced', 'growth',
                     'custom'),
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
    # Python remains the fee authority unless an explicit handoff disables it.
    fee_authority_enabled: bool = True
    # Internal fee-cycle replay instrumentation. Disabled by default and
    # observational only; enabling affects the next naturally scheduled cycle.
    fee_replay_capture_enabled: bool = False
    # Internal, default-off rebalance-cycle replay instrumentation.
    rebalance_replay_capture_enabled: bool = False
    # Hot-channel protection (aggressiveness for fast-draining, high-profit channels)
    hot_channel_protection_enabled: bool = True
    hot_channel_protection_override_peers: str = ''  # CSV fallback; DB override table preferred
    hot_channel_protection_min_velocity: float = 0.20
    hot_channel_protection_min_marginal_roi: float = 0.20
    hot_channel_protection_profit_budget_pct: float = 0.75
    hot_channel_protection_max_chunk_multiplier: float = 4.0
    hot_channel_protection_min_cooldown_hours: float = 1.0
    # Source-heavy drain: node-level receivable objective and envelopes.
    # receivable_ratio = total receivable / total capacity across channels.
    receivable_ratio_target: float = 0.30   # structural credit scales to 0 here
    receivable_ratio_floor: float = 0.20    # below this the node is "starved"
    # Daily cap (sats of swap fees) for loop-outs that only pass the profit
    # guard via the structural credit. 0 = structural loop-outs disabled.
    # Max bounded fee discount applied to stagnant over-local channels.
    # 0.0 = disabled. 0.10 means fees may be biased down by at most 10%.
    drain_fee_discount_max: float = 0.0
    # Node-liquidity-aware auto-drain-bias: when the node as a whole is
    # source-heavy (receivable ratio below target), automatically scale a
    # drain discount on over-local channels' fees. Default OFF; the feature
    # is inert until enabled (P6-002: must be wired end-to-end, not a
    # silent no-op).
    node_drain_bias_enabled: bool = False
    # Max node-scaled drain discount (0.0-0.5); only active when enabled.
    node_drain_bias_max: float = 0.3
    # Dynamic htlc_max flow valve (H-2, 2026-07-03 audit). Phase B
    # (2026-08-01 surface reduction): default ON — prod-proven, bounded
    # valve. Max HTLC = capacity * pct by flow state (sinks tightest:
    # throttle drain-through; CLN's only inbound-side control surface).
    # Bounds in fee_controller keep it >= 10k sats. The CLN option
    # revenue-ops-enable-dynamic-htlcmax feeds this at plugin init and
    # shares this default.
    enable_dynamic_htlcmax: bool = True
    htlcmax_source_pct: float = 0.50
    htlcmax_sink_pct: float = 0.25
    htlcmax_balanced_pct: float = 0.45
    # econ_* rollout flags — Phase B (2026-08-01 surface reduction, section
    # 2e/3): all 12 default TRUE so fresh nodes run the governed paths that
    # scheduled for removal (with the legacy branches) after one stable
    # default-True release (Phase D).
    # Refactor Phase 1 shadow: observe-mode intent recording into
    # econ_ledger.db + revenue-econ-snapshot preview. Read-only wrt node
    # state (docs/planning/2026-07-12-refactor-phase1-wiring.md).
    econ_shadow_enabled: bool = True
    # Refactor Phase 2D: governor-gated rebalance reservations (see
    # docs/planning/2026-07-12-refactor-phase2-governed-rebalance.md).
    econ_governor_rebalance_enabled: bool = True
    # Refactor Phase 2H: governor-gated automated fee broadcasts.
    econ_governor_fees_enabled: bool = True
    # Refactor Phase 3F: live governor-boundary arbitration.
    econ_arbiter_enabled: bool = True
    # Workstream H: cycle-arbitrated rebalance execution list.
    econ_cycle_rebalance_enabled: bool = True
    econ_ev_populated: bool = True
    econ_conflict_rules_extended: bool = True
    # Phase 4: global authority level (observe|fees|liquidity|capital).
    authority_level: str = "capital"
    risk_profile: str = "custom"
    # Flow analysis parameters
    flow_window_days: int = 7      # Days to analyze for flow calculation
    
    # Flow ratio thresholds for classification (net daily flow / capacity).
    # F1 (2026-06 audit): the old ±0.5 defaults required HALF the channel
    # capacity to net-flow per day before firing; typical channels run
    # 0.01-0.10, so the thresholds were dead and classification always fell
    # through to the instantaneous balance position. ±0.05 lets the Kalman
    # flow estimate classify first.
    source_threshold: float = 0.05   # FlowRatio > +0.05/day = Source (draining)
    sink_threshold: float = -0.05    # FlowRatio < -0.05/day = Sink (filling)
    
    # Fee parameters
    # Phase B (2026-08-01 surface reduction, section 3): 10 -> 50. A 10 PPM
    # floor invites unprofitable forwards; 50 matches the production
    # operator value. The CLN option revenue-ops-min-fee-ppm feeds this
    # at plugin init and shares this default.
    min_fee_ppm: int = 50          # Floor fee in PPM
    # E-2 (2026-07 econ audit): class-aware min-fee floor. Channels the
    # controller classifies as saturated (outbound_ratio >= 0.85) or source
    # (flow_state == 'source') use min(min_fee_ppm, min_fee_ppm_saturated)
    # as their config floor so pinned-at-floor channels can advertise
    # genuinely cheap egress. Default 0 = true cheap egress allowed. The
    # rebalance-cost floor / chain-cost floor / vegas floor still compose
    # via max() on top — this only replaces the min_fee_ppm term.
    min_fee_ppm_saturated: int = 0
    # Default-off. The fee controller owns fixed safety caps; this switch is
    # intentionally the only public tuning surface for the first rollout.
    acquisition_experiment_enabled: bool = False
    max_fee_ppm: int = 2000        # Ceiling fee in PPM (matches revenue-ops-max-fee-ppm option default; P6-010/DEF-042)
    base_fee_msat: int = 0         # Base fee fallback when base_fee_policy = "off"

    # Base-fee policy (Upgrade A, 2026-04-22). The per-channel-role split
    # ("adaptive": intra-fleet vs everyone else) retired with cl-mycelium;
    # both values now apply base_fee_msat to every channel.
    base_fee_policy: str = "off"            # off | adaptive

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
    # above median in inelastic markets where we lose less volume to a
    # price increase than the median-follower would.
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
    # E-4.5 (2026-07 econ audit): rebalance_min_profit is a DEPRECATED
    # no-op compatibility field — it was parsed and echoed but enforced
    # nowhere. Its semantics (absolute sats profit floor) are covered by
    # rebalance_hold_margin, which the v2 engine's sats-EV do-nothing gate
    # actually enforces. Kept only so existing config files load cleanly
    # (mirrors the fee-market-boundary deprecation pattern).
    rebalance_min_profit: int = 10     # DEPRECATED no-op (use rebalance_hold_margin)
                                        # Recommended: 20 ppm (~10 sats per 500k chunk)
    rebalance_max_amount: int = 5000000  # Max rebalance amount in sats
    low_liquidity_threshold: float = 0.3  # Below 30% = low outbound
    high_liquidity_threshold: float = 0.7 # Above 70% = high outbound
    rebalance_cooldown_hours: int = 24   # Don't re-rebalance same channel for 24h
    # Phase 3 drift override: a destination below this local ratio is still
    # refill-eligible even when the channel-level cooldown is active. Set to
    # 0 to disable the override and keep the strict cooldown gate.
    rebalance_emergency_local_ratio: float = 0.20
    # Phase 3.3 anchor-state drift override: when a destination's local ratio
    # has dropped by at least this much since the last successful rebalance,
    # the cooldown gate is bypassed. Set to 0 to disable.
    rebalance_drift_override_ratio: float = 0.30
    # Phase 4.3 do_nothing hold gate: priced pairs with engine final_score
    # at or below this margin are rejected with reason='below_hold_margin'.
    # final_score is denominated in SATS of expected net value (sats-EV
    # scoring), so this is a sats bar. 0 leaves the legacy "any positive
    # score" behavior; a small positive value requires pairs to clear a
    # meaningful EV bar before executing.
    rebalance_hold_margin: float = 0.0
    # Iter1 pair fee budget: layered on top of the destination's capex
    # bootstrap budget so a small/new channel can still pay enough route
    # fee for the selected route. pair_budget_sats =
    #   max(dest.remaining_capex_sats, ceil(amount * pair_fee_cap_ppm / 1M)).
    # Default 1000 ppm = 0.1% of rebalance amount. 0 disables the layer
    # and keeps the Phase 5 capex-only behavior.
    pair_fee_cap_ppm: int = 1000
    # Rebalance flow-fact, EV, utilization, and size-tier tuning.
    # Window over which recent forwarding activity is measured for the
    # activity-recency penalty (seconds). Default 3600 = 1 hour.
    rebalance_activity_window_seconds: int = 3600
    # Coefficient applied to the activity-recency penalty when scoring a
    # rebalance candidate. 0 disables the penalty.
    rebalance_activity_penalty_coeff: float = 0.5
    # Upper bound (fraction of score) that the activity-recency penalty may
    # ever remove, so a stale channel is deprioritized but never zeroed out.
    rebalance_activity_penalty_cap_frac: float = 0.5
    # Trailing window (days) over which forwarding utilization is measured
    # for the utilization-based sizing/target logic.
    rebalance_utilization_window_days: int = 7
    # Minimum utilization ratio floor used when normalizing observed flow.
    rebalance_utilization_floor: float = 0.05
    # Maximum utilization ratio ceiling used when normalizing observed flow.
    rebalance_utilization_ceiling: float = 1.0
    # Minimum number of forwards in the utilization window required before
    # utilization-based sizing is trusted; below this, fall back to defaults.
    rebalance_utilization_min_forwards: int = 5
    # Enable size-tiered rebalance targets (bucket target amounts by channel
    # capacity percentile instead of a single flat target).
    rebalance_size_tiered_targets: bool = True
    # Percentile (0.0-1.0) of the channel-capacity distribution used as the
    # reference point for size-tiered target amounts.
    rebalance_size_reference_percentile: float = 0.5
    # Half-width (fraction) of the "small channel" band around the reference
    # percentile; channels within this band are treated as small/reference
    # sized for target-sizing purposes.
    rebalance_small_channel_band_half_width: float = 0.15

    inbound_fee_estimate_ppm: int = 50  # Route cost buffer added on top of last-hop fee (PPM)
    
    # Profitability tracking
    estimated_open_cost_sats: int = 5000  # Estimated on-chain fee for channel open
    
    # Global Capital Controls
    daily_budget_sats: int = 5000          # Max rebalancing fees per 24h period (fixed floor)
    growth_budget_enabled: bool = False    # Opt-in dynamic budget: base + bounded earned/growth credit
    growth_budget_earned_fraction: float = 0.25
    growth_budget_experiment_fraction: float = 0.10
    growth_budget_max_extra_sats: int = 2_000
    growth_budget_hard_ceiling_sats: int = 10_000
    allow_zero_cost_auto_rebalance_when_budget_zero: bool = False
    weekly_budget_sats: int = 35000        # Max rebalancing fees per 7-day window (hard ceiling)
    min_wallet_reserve: int = 1_000_000    # Min sats (confirmed on-chain + channel spendable) before ABORT

    # RPC Hardening
    rpc_timeout_seconds: int = 15
    reservation_timeout_hours: int = 4  # Hours before stale budget reservations auto-release
    
    # HTLC Congestion threshold
    htlc_congestion_threshold: float = 0.8  # Mark channel as CONGESTED if >80% HTLC slots used
    
    # Reputation-weighted volume
    enable_reputation: bool = True  # If True, weight volume by peer success rate
    reputation_decay: float = 0.98  # Decay factor per flow_interval (default hourly)
                                     # 0.98^24 ≈ 0.61, meaning old data loses ~40% weight daily

    # Kelly Criterion Position Sizing
    # (kelly_fraction knob removed 2026-06: it had zero consumers — only the
    # enable_kelly flag is read.)
    enable_kelly: bool = False       # If True, scale rebalance budget by Kelly fraction (opt-in)
    
    # Async Job Queue
    max_concurrent_jobs: int = 5              # Max number of concurrent rebalance jobs

    # AskRene (xpay) constraint integration
    askrene_layer: str = 'xpay'               # Layer name for askrene-listlayers

    # V3 rebalance router (askrene getroutes)
    rebalance_router: str = 'v3'              # only 'v3' is supported
    askrene_layers: str = 'standalone'        # CSV of layers to pass to v3 router's getroutes calls ('standalone' = none)

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

    # Issue #30: Velocity gate for rebalancing
    # Prevents overfilling channels with no routing history

    # ==========================================================================
    # DTS (Discounted Thompson Sampling) Parameters
    # ==========================================================================
    thompson_prior_std_fee: int = 100         # Default prior uncertainty in ppm


    # ==========================================================================
    # Routing Intelligence Integration
    # ==========================================================================

    # Unified Capex Budget Engine
    capex_reinvestment_rate: float = 0.50       # Fraction of channel contribution for all capex
    capex_bootstrap_bps: int = 10               # Bootstrap: basis points of capacity per 30d
    capex_bootstrap_max_sats: int = 200         # Bootstrap cap per channel per 30d
    capex_grace_days: int = 14                  # Days before bootstrap activates
    capex_global_envelope_sats: int = 0         # Global cap (0 = auto-computed)
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
        # Source-heavy drain fields: enforce ranges at construction time so a
        # bad receivable objective or discount cap can never be instantiated.
        for _key in (
            'receivable_ratio_target',
            'receivable_ratio_floor',
            'drain_fee_discount_max',
            'node_drain_bias_max',
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
            if key.startswith('_'):
                # Internal marker rows (e.g. _closure_sweep_tripped,
                # internal migration markers share the override
                # table but are not config keys.
                continue
            if hasattr(self, key) and key not in IMMUTABLE_CONFIG_KEYS:
                self._apply_override(key, value)
            elif not hasattr(self, key):
                # Phase B (2026-08-01 surface reduction, section 5.5): a
                # silently skipped unknown row rots invisibly — the
                # stale production keys can otherwise dangle for months.
                self._override_warnings.append(
                    f"config override '{key}' does not match any known key "
                    f"— ignored (stale after an upgrade?)")
        self._version = database.get_config_version()
        # PR 7 (Phase D): apply the selected risk profile's derived values
        # to keys WITHOUT explicit overrides (precedence: explicit >
        # profile > default). 'custom'/unknown derive nothing. Runs before
        # detection/repairs so derived values face the same contradiction
        # checks as any other setting. A profile change at runtime takes
        # effect at plugin restart (preview/diff first — PR 8).
        try:
            profile_name = str(getattr(self, "risk_profile", "custom")
                               or "custom").strip().lower()
            if profile_name != "custom":
                from .risk_profiles import resolve_profile
                derived = resolve_profile(profile_name, set(overrides))
                for key, value in sorted(derived.items()):
                    setattr(self, key, value)
                    self._override_warnings.append(
                        f"Profile '{profile_name}' derived {key}={value} "
                        f"(no explicit override)")
        except Exception as e:
            self._override_warnings.append(
                f"Risk-profile resolution failed open (custom behavior "
                f"kept): {e}")
        # Workstream I: shadowed/deprecated detection runs against the keys
        # the operator EXPLICITLY overrode, before repairs mutate anything.
        self._detect_shadowed_and_deprecated(set(overrides))
        # M-R5-2 FIX: Post-load cross-field invariant repair.
        # Overrides applied individually may violate cross-field constraints
        # (e.g., min_fee_ppm > max_fee_ppm from TOCTOU race or manual DB edits).
        # Workstream I: each repair now also WARNS — silent repair left the
        # operator unaware their persisted settings contradict each other.
        # The crossed fee pair is repaired UPWARD. Lowering min_fee_ppm to a
        # persisted max_fee_ppm of 1-4 is individually in range but drags the
        # floor under its own CONFIG_FIELD_RANGES minimum (CRITICAL-02), and
        # the two bounds have different lower limits so no downward repair can
        # hold both invariants. Raising the ceiling honors the operator's
        # stated floor and only widens a cap. Matches the init-time repair in
        # cl-revenue-ops.py::_enforce_fee_bound_invariant.
        if self.min_fee_ppm > self.max_fee_ppm:
            self._override_warnings.append(
                f"Contradictory settings: min_fee_ppm ({self.min_fee_ppm}) > "
                f"max_fee_ppm ({self.max_fee_ppm}); repaired max_fee_ppm to "
                f"{self.min_fee_ppm}")
            self.max_fee_ppm = self.min_fee_ppm
        if hasattr(self, 'low_liquidity_threshold') and hasattr(self, 'high_liquidity_threshold'):
            if self.low_liquidity_threshold >= self.high_liquidity_threshold:
                # M-R6-1 FIX: Clamp to 0.0 to prevent negative values when
                # high_liquidity_threshold is very small (e.g., < 0.05).
                repaired = max(0.0, self.high_liquidity_threshold - 0.05)
                self._override_warnings.append(
                    f"Contradictory settings: low_liquidity_threshold "
                    f"({self.low_liquidity_threshold}) >= high_liquidity_threshold "
                    f"({self.high_liquidity_threshold}); repaired low to {repaired}")
                self.low_liquidity_threshold = repaired
        if hasattr(self, 'rebalance_utilization_floor') and hasattr(self, 'rebalance_utilization_ceiling'):
            if self.rebalance_utilization_floor >= self.rebalance_utilization_ceiling:
                # Mirrors the low_liquidity_threshold/high_liquidity_threshold
                # repair: an inverted floor/ceiling pair would otherwise pin
                # realized utilization to the (wrong) floor for every channel.
                repaired = max(0.0, self.rebalance_utilization_ceiling - 0.05)
                self._override_warnings.append(
                    f"Contradictory settings: rebalance_utilization_floor "
                    f"({self.rebalance_utilization_floor}) >= ceiling "
                    f"({self.rebalance_utilization_ceiling}); repaired floor to "
                    f"{repaired}")
                self.rebalance_utilization_floor = repaired
        if self.receivable_ratio_floor > self.receivable_ratio_target:
            self._override_warnings.append(
                f"Contradictory settings: receivable_ratio_floor "
                f"({self.receivable_ratio_floor}) > receivable_ratio_target "
                f"({self.receivable_ratio_target}); repaired floor to target")
            self.receivable_ratio_floor = self.receivable_ratio_target
        # A crossed persisted budget pair must fail closed. The weekly
        # value is the longer-window hard ceiling, so startup preserves it
        # and lowers the daily cap instead of expanding spend authority.
        if self.daily_budget_sats > self.weekly_budget_sats:
            self._override_warnings.append(
                f"Contradictory settings: daily_budget_sats "
                f"({self.daily_budget_sats}) > weekly_budget_sats "
                f"({self.weekly_budget_sats}); repaired daily_budget_sats "
                f"to {self.weekly_budget_sats}")
            self.daily_budget_sats = self.weekly_budget_sats
        return list(self._override_warnings)

    def _detect_shadowed_and_deprecated(self, explicit_keys: set) -> None:
        """Workstream I startup detection: explicitly-set options that are
        deprecated no-ops, or whose effect is shadowed by an off gate flag."""
        for key in sorted(explicit_keys & DEPRECATED_RUNTIME_KEYS):
            replacement = DEPRECATED_KEY_REPLACEMENTS.get(key)
            hint = f"; use {replacement}" if replacement else ""
            self._override_warnings.append(
                f"Deprecated option {key} is set — it is a no-op{hint} "
                f"(scheduled for removal after the announced compatibility "
                f"window)")
        # Phase B soft deprecations: still functional (NOT no-ops), but
        # scheduled for Phase C removal — warn when an override row exists.
        for key in sorted(explicit_keys & set(SOFT_DEPRECATED_RUNTIME_KEYS)):
            self._override_warnings.append(
                f"Deprecated option {key} is set — "
                f"{SOFT_DEPRECATED_RUNTIME_KEYS[key]}")
        for gate, gated_keys in SHADOWED_SETTING_GATES.items():
            if getattr(self, gate, False):
                continue
            for key in gated_keys:
                if key in explicit_keys:
                    self._override_warnings.append(
                        f"Shadowed setting: {key} is set but {gate} is "
                        f"false — it has no effect until the gate is enabled")

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
                lowered = value.lower()
                if lowered in ('true', '1', 'yes', 'on'):
                    typed_value = True
                elif lowered in ('false', '0', 'no', 'off'):
                    typed_value = False
                else:
                    # A typo ("treu") silently parsed as False and reported
                    # success — reject instead.
                    return {"error": f"Invalid boolean for {key}: {value!r} (use true/false)"}
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
            # M-R5-4 FIX: Also validate liquidity threshold cross-field consistency
            if key == 'low_liquidity_threshold' and typed_value >= self.high_liquidity_threshold:
                return {"error": f"low_liquidity_threshold ({typed_value}) must be less than high_liquidity_threshold ({self.high_liquidity_threshold})"}
            if key == 'high_liquidity_threshold' and typed_value <= self.low_liquidity_threshold:
                return {"error": f"high_liquidity_threshold ({typed_value}) must be greater than low_liquidity_threshold ({self.low_liquidity_threshold})"}
            # Utilization floor/ceiling: mirrors the low/high_liquidity_threshold
            # guard above. Without this, an inverted pair (e.g. floor=0.9,
            # ceiling=0.1) silently pins realized utilization to 0.9 for every
            # channel via modules/rebalance_flow_facts.py's clamp.
            if key == 'rebalance_utilization_floor' and typed_value >= self.rebalance_utilization_ceiling:
                return {"error": f"rebalance_utilization_floor ({typed_value}) must be less than rebalance_utilization_ceiling ({self.rebalance_utilization_ceiling})"}
            if key == 'rebalance_utilization_ceiling' and typed_value <= self.rebalance_utilization_floor:
                return {"error": f"rebalance_utilization_ceiling ({typed_value}) must be greater than rebalance_utilization_floor ({self.rebalance_utilization_floor})"}
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
            # Reject crossed budget pairs at set-time; startup also
            # preserves the weekly ceiling by clamping daily down.
            if key == 'daily_budget_sats' and typed_value > self.weekly_budget_sats:
                return {"error": f"daily_budget_sats ({typed_value}) cannot exceed current weekly_budget_sats ({self.weekly_budget_sats})"}
            if key == 'weekly_budget_sats' and typed_value < self.daily_budget_sats:
                return {"error": f"weekly_budget_sats ({typed_value}) cannot be less than current daily_budget_sats ({self.daily_budget_sats})"}

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

            result = {
                "status": "success",
                "key": key,
                "old_value": old_value,
                "new_value": typed_value,
                "version": new_version
            }
            # Phase B (2026-08-01 surface reduction, section 5.4): set-time
            # shadowed-gate warning — the write succeeded, but the key's
            # effect is entirely gated behind a flag that is currently off,
            # so tell the operator NOW instead of at the next restart.
            gate = _GATE_FOR_GATED_KEY.get(key)
            if gate is not None and not getattr(self, gate, False):
                result["warning"] = (
                    f"{key} is set, but {gate} is currently false — it has "
                    f"no effect until the gate is enabled")
            # Phase B soft deprecation note (still enforced; Phase C removes).
            if key in SOFT_DEPRECATED_RUNTIME_KEYS:
                result["deprecation"] = SOFT_DEPRECATED_RUNTIME_KEYS[key]

        return result


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
    
    # Flow analysis parameters
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
    rebalance_max_amount: int
    low_liquidity_threshold: float
    high_liquidity_threshold: float
    rebalance_cooldown_hours: int
    rebalance_emergency_local_ratio: float
    rebalance_drift_override_ratio: float
    rebalance_hold_margin: float
    pair_fee_cap_ppm: int
    inbound_fee_estimate_ppm: int
    # Upstream rebalancer patterns
    rebalance_activity_window_seconds: int
    rebalance_activity_penalty_coeff: float
    rebalance_activity_penalty_cap_frac: float
    rebalance_utilization_window_days: int
    rebalance_utilization_floor: float
    rebalance_utilization_ceiling: float
    rebalance_utilization_min_forwards: int
    rebalance_size_tiered_targets: bool
    rebalance_size_reference_percentile: float
    rebalance_small_channel_band_half_width: float

    # Profitability tracking
    estimated_open_cost_sats: int
    
    # Global Capital Controls
    daily_budget_sats: int
    growth_budget_enabled: bool
    growth_budget_earned_fraction: float
    growth_budget_experiment_fraction: float
    growth_budget_max_extra_sats: int
    growth_budget_hard_ceiling_sats: int
    allow_zero_cost_auto_rebalance_when_budget_zero: bool
    min_wallet_reserve: int

    # HTLC Congestion threshold
    htlc_congestion_threshold: float
    
    # Reputation-weighted volume
    enable_reputation: bool
    reputation_decay: float

    # Kelly Criterion Position Sizing
    enable_kelly: bool
    
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
    reservation_timeout_hours: int

    # Issue #28: Revenue rate EMA smoothing

    # Issue #30: Velocity gate for rebalancing

    # DTS (Discounted Thompson Sampling) parameters
    thompson_prior_std_fee: int
    # Routing Intelligence Integration

    # M-27: xpay/askrene parameters (were missing from snapshot)
    askrene_layer: str = 'xpay'

    # Python fee-authority handoff state. Default-on preserves current behavior.
    fee_authority_enabled: bool = True

    # Internal, default-off fee-cycle replay instrumentation.
    fee_replay_capture_enabled: bool = False

    # Internal, default-off rebalance-cycle replay instrumentation.
    rebalance_replay_capture_enabled: bool = False

    # V3 rebalance router (askrene getroutes)
    rebalance_router: str = 'v3'
    askrene_layers: str = 'standalone'

    # Weekly budget cap (hard ceiling over daily burst)
    weekly_budget_sats: int = 35000

    # Unified Capex Budget Engine
    capex_reinvestment_rate: float = 0.50
    capex_bootstrap_bps: int = 10
    capex_bootstrap_max_sats: int = 200
    capex_grace_days: int = 14
    capex_global_envelope_sats: int = 0
    capex_probability_budget_bonus: float = 0.0
    # Receivable-ratio and fee drain-bias fields
    receivable_ratio_target: float = 0.30
    receivable_ratio_floor: float = 0.20
    drain_fee_discount_max: float = 0.0
    node_drain_bias_enabled: bool = False
    node_drain_bias_max: float = 0.3
    # Dynamic htlc_max flow valve (H-2, 2026-07-03 audit). Phase B
    # (2026-08-01 surface reduction): default ON, mirroring Config.
    enable_dynamic_htlcmax: bool = True
    htlcmax_source_pct: float = 0.50
    htlcmax_sink_pct: float = 0.25
    htlcmax_balanced_pct: float = 0.45
    # econ_* rollout flags — mirrored from Config (a key missing from the
    # snapshot reads as absent in production). Phase B: default TRUE.
    econ_shadow_enabled: bool = True
    econ_governor_rebalance_enabled: bool = True
    econ_governor_fees_enabled: bool = True
    econ_arbiter_enabled: bool = True
    econ_cycle_rebalance_enabled: bool = True
    econ_ev_populated: bool = True
    econ_conflict_rules_extended: bool = True
    authority_level: str = "capital"
    risk_profile: str = "custom"
    # E-2: class-aware min-fee floor for saturated/source channels (0 =
    # allow true cheap egress). Missing-from-snapshot kills the feature in
    # production (getattr fallback), so it MUST be mirrored here.
    min_fee_ppm_saturated: int = 0
    acquisition_experiment_enabled: bool = False
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
