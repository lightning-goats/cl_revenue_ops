"""
Configuration module for cl-revenue-ops

Contains the Config dataclass that holds all tunable parameters
for the Revenue Operations plugin.

Phase 7 additions:
- ConfigSnapshot: Immutable snapshot for thread-safe cycle execution
- Runtime configuration updates via RPC
- Vegas Reflex and Scarcity Pricing settings
"""

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

# Type mapping for config fields (for validation)
CONFIG_FIELD_TYPES: Dict[str, type] = {
    'flow_interval': int,
    'fee_interval': int,
    'rebalance_interval': int,
    'min_fee_ppm': int,
    'max_fee_ppm': int,
    'daily_budget_sats': int,
    'min_wallet_reserve': int,
    'revenue_boltz_auto': bool,
    'boltz_loop_in_max_sats': int,
    'boltz_loop_in_daily_cap_sats': int,
    'boltz_loop_in_min_confirmations': int,
    'swap_daily_budget_sats': int,
    'swap_max_fee_ppm': int,
    'swap_min_amount_sats': int,
    'swap_max_amount_sats': int,
    'low_liquidity_threshold': float,
    'high_liquidity_threshold': float,
    'htlc_congestion_threshold': float,
    'enable_reputation': bool,
    'enable_kelly': bool,
    'kelly_bypass_for_fleet': bool,
    'enable_proportional_budget': bool,
    'proportional_budget_pct': float,
    'kelly_fraction': float,
    'reputation_decay': float,
    'max_concurrent_jobs': int,
    'sling_job_timeout_seconds': int,
    'sling_chunk_size_sats': int,
    'sling_max_hops': int,
    'askrene_layer': str,
    'askrene_max_age_sec': int,
    'sling_parallel_jobs': int,
    'sling_target_sink': float,
    'sling_target_source': float,
    'sling_target_balanced': float,
    'sling_outppm_fallback': int,
    'sling_deplete_pct_sink': float,
    'sling_deplete_pct_source': float,
    'sling_deplete_pct_balanced': float,
    'rebalance_min_profit': int,
    'rebalance_min_profit_ppm': int,
    'rebalance_max_amount': int,
    'rebalance_min_amount': int,
    'rebalance_cooldown_hours': int,
    'inbound_fee_estimate_ppm': int,
    # Phase 7 additions
    'enable_vegas_reflex': bool,
    'vegas_decay_rate': float,
    'enable_scarcity_pricing': bool,
    'scarcity_threshold': float,
    # Hive Parameters
    'hive_enabled': str,  # "auto", "true", "false"
    'hive_fee_ppm': int,
    'hive_rebalance_tolerance': int,
    # Phase 1: Operational Hardening
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
    # Thompson Sampling + AIMD (v1.7.0)
    'thompson_prior_std_fee': int,
    'thompson_observation_decay_hours': int,
    'thompson_max_observations': int,
    'thompson_min_observations': int,
    'aimd_failure_threshold': int,
    'aimd_success_threshold': int,
    'aimd_multiplicative_decrease': float,
    'aimd_additive_increase_ppm': int,
    'aimd_min_decrease_interval': int,
    'hive_prior_weight': float,
    'hive_min_confidence_for_prior': float,
    # Routing Intelligence Integration
    'routing_intelligence_enabled': bool,
    'routing_intelligence_cache_seconds': int,
    # Comprehensive Hive Data Integration
    'hive_defense_status_enabled': bool,
    'hive_defense_status_cache_seconds': int,
    'hive_peer_quality_enabled': bool,
    'hive_peer_quality_cache_seconds': int,
    'hive_decision_history_enabled': bool,
    'hive_decision_history_days': int,
    'hive_channel_flags_enabled': bool,
    'hive_mcf_targets_enabled': bool,
    'hive_mcf_targets_cache_seconds': int,
    'hive_nnlb_enabled': bool,
    'hive_nnlb_min_amount': int,
    'hive_nnlb_auto_execute': bool,
    'hive_channel_ages_enabled': bool,
    'hive_channel_ages_cache_seconds': int,
    # Boltz swap currency
    'swap_currency': str,
}

# Range constraints for numeric fields
CONFIG_FIELD_RANGES: Dict[str, tuple] = {
    'min_fee_ppm': (5, 100000),  # CRITICAL-02 FIX: Minimum 5 PPM to ensure economic viability
    'max_fee_ppm': (1, 100000),
    'daily_budget_sats': (0, 10000000),
    'min_wallet_reserve': (0, 100000000),
    'boltz_loop_in_max_sats': (25000, 100000000),
    'boltz_loop_in_daily_cap_sats': (25000, 250000000),
    'boltz_loop_in_min_confirmations': (1, 100),
    'swap_daily_budget_sats': (0, 1000000),
    'swap_max_fee_ppm': (100, 50000),
    'swap_min_amount_sats': (10000, 100000000),
    'swap_max_amount_sats': (10000, 100000000),
    'low_liquidity_threshold': (0.0, 1.0),
    'high_liquidity_threshold': (0.0, 1.0),
    'htlc_congestion_threshold': (0.0, 1.0),
    'reputation_decay': (0.0, 1.0),
    'proportional_budget_pct': (0.0, 1.0),
    'kelly_fraction': (0.0, 1.0),
    'vegas_decay_rate': (0.0, 1.0),
    'scarcity_threshold': (0.0, 1.0),
    'hive_fee_ppm': (0, 100000),
    'hive_rebalance_tolerance': (0, 100000),
    'sling_chunk_size_sats': (1, 50000000),
    'sling_max_hops': (2, 20),
    'sling_parallel_jobs': (1, 10),
    'sling_target_sink': (0.1, 0.9),
    'sling_target_source': (0.1, 0.9),
    'sling_target_balanced': (0.1, 0.9),
    'sling_outppm_fallback': (0, 10000),
    'sling_deplete_pct_sink': (0.01, 0.50),
    'sling_deplete_pct_source': (0.01, 0.50),
    'sling_deplete_pct_balanced': (0.01, 0.50),
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
    # Thompson Sampling + AIMD (v1.7.0)
    'thompson_prior_std_fee': (10, 500),
    'thompson_observation_decay_hours': (24, 720),  # 1 day to 30 days
    'thompson_max_observations': (50, 500),
    'thompson_min_observations': (1, 20),
    'aimd_failure_threshold': (1, 10),
    'aimd_success_threshold': (3, 30),
    'aimd_multiplicative_decrease': (0.5, 0.95),
    'aimd_additive_increase_ppm': (1, 20),
    'aimd_min_decrease_interval': (300, 86400),  # 5 min to 24 hours
    'hive_prior_weight': (0.0, 1.0),
    'hive_min_confidence_for_prior': (0.0, 1.0),
    # Routing Intelligence Integration
    'routing_intelligence_cache_seconds': (60, 3600),  # 1 min to 1 hour
    # Comprehensive Hive Data Integration
    'hive_defense_status_cache_seconds': (10, 300),    # 10 sec to 5 min
    'hive_peer_quality_cache_seconds': (60, 1800),     # 1 min to 30 min
    'hive_decision_history_days': (1, 90),             # 1 to 90 days
    'hive_mcf_targets_cache_seconds': (60, 1800),      # 1 min to 30 min
    'hive_nnlb_min_amount': (10000, 10000000),         # 10k to 10M sats
    'hive_channel_ages_cache_seconds': (300, 86400),   # 5 min to 24 hours
    # swap_currency: validated in boltz_swaps.py (string enum, not numeric range)
    # Additional range validations
    'flow_interval': (60, 86400),
    'fee_interval': (60, 86400),
    'rebalance_interval': (60, 86400),
    'max_concurrent_jobs': (1, 20),
    'sling_job_timeout_seconds': (60, 7200),
    'askrene_max_age_sec': (10, 86400),
    'base_fee_msat': (0, 10000),
    'rebalance_min_profit': (0, 1000000),
    'rebalance_min_amount': (1000, 50000000),
    'rebalance_max_amount': (10000, 100000000),
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
    fee_interval: int = 600        # 10 minutes
    rebalance_interval: int = 900  # 15 minutes
    
    # Flow analysis parameters
    target_flow: int = 100000      # Target sats routed per day per channel
    flow_window_days: int = 7      # Days to analyze for flow calculation
    
    # Flow ratio thresholds for classification
    source_threshold: float = 0.5   # FlowRatio > 0.5 = Source (draining)
    sink_threshold: float = -0.5    # FlowRatio < -0.5 = Sink (filling)
    
    # Fee parameters
    min_fee_ppm: int = 25          # Floor fee in PPM
    max_fee_ppm: int = 5000        # Ceiling fee in PPM
    base_fee_msat: int = 0         # Base fee (we focus on PPM)
    
    # Rebalancing parameters
    rebalance_min_profit: int = 10     # Min profit in sats to trigger (legacy, used when ppm=0)
    rebalance_min_profit_ppm: int = 0  # Min profit in PPM (0 = use sats threshold, >0 = use ppm)
                                        # Recommended: 20 ppm (~10 sats per 500k chunk)
    rebalance_max_amount: int = 5000000  # Max rebalance amount in sats
    rebalance_min_amount: int = 50000    # Min rebalance amount in sats
    low_liquidity_threshold: float = 0.2  # Below 20% = low outbound
    high_liquidity_threshold: float = 0.8 # Above 80% = high outbound
    rebalance_cooldown_hours: int = 24   # Don't re-rebalance same channel for 24h
    inbound_fee_estimate_ppm: int = 500  # Network routing cost estimate in PPM
    
    # clboss integration
    clboss_enabled: bool = True    # Whether to use clboss-unmanage
    clboss_unmanage_duration_hours: int = 24  # Keep unmanaged after rebalance
    
    # Rebalancer plugin selection
    rebalancer_plugin: str = 'sling'  # Only sling is supported
    
    # Profitability tracking
    estimated_open_cost_sats: int = 5000  # Estimated on-chain fee for channel open
    
    # Global Capital Controls
    daily_budget_sats: int = 5000          # Max rebalancing fees per 24h period (fixed floor)
    min_wallet_reserve: int = 1_000_000    # Min sats (confirmed on-chain + channel spendable) before ABORT

    # Boltz loop-in auto-funding safety controls (legacy, kept for backward compat)
    revenue_boltz_auto: bool = True                 # Kill-switch for automatic loop-in funding
    boltz_loop_in_max_sats: int = 10_000_000        # Per-swap max auto-funded amount
    boltz_loop_in_daily_cap_sats: int = 25_000_000  # 24h aggregate auto-funding cap
    boltz_loop_in_min_confirmations: int = 1        # CLN wallet UTXO minconf for withdraw

    # Boltz swap budget controls (boltzcli integration)
    swap_daily_budget_sats: int = 50_000            # Max daily spend on swap fees
    swap_max_fee_ppm: int = 5_000                   # Max acceptable fee rate for any single swap
    swap_min_amount_sats: int = 100_000             # Minimum swap amount
    swap_max_amount_sats: int = 10_000_000          # Maximum swap amount
    swap_currency: str = 'lbtc'                       # 'btc' or 'lbtc' (Liquid, lower fees)
    
    # Revenue-Proportional Budget (Phase 7: Dynamic Budget Scaling)
    enable_proportional_budget: bool = True   # Scale daily budget based on revenue (Issue #22)
    proportional_budget_pct: float = 0.30     # Budget = max(daily_budget_sats, revenue_24h * pct)
                                               # Default 30% of 24h revenue
    
    # Phase 1: Operational Hardening
    rpc_timeout_seconds: int = 15
    rpc_circuit_breaker_seconds: int = 60
    rpc_pool_size: int = 5             # Number of RPC worker processes (Phase 2)
    reservation_timeout_hours: int = 4  # Hours before stale budget reservations auto-release
    
    # HTLC Congestion threshold
    htlc_congestion_threshold: float = 0.8  # Mark channel as CONGESTED if >80% HTLC slots used
    
    # Reputation-weighted volume
    enable_reputation: bool = True  # If True, weight volume by peer success rate
    reputation_decay: float = 0.98  # Decay factor per flow_interval (default hourly)
                                     # 0.98^24 ≈ 0.61, meaning old data loses ~40% weight daily

    # Kelly Criterion Position Sizing (Phase 4: Risk Management)
    enable_kelly: bool = True        # If True, scale rebalance budget by Kelly fraction
    kelly_bypass_for_fleet: bool = True  # If True, skip Kelly for hive/fleet destinations
                                          # Fleet paths are ~free (0 ppm internal), so Kelly's
                                          # EV gate is counterproductive — it kills candidates
                                          # before the fleet path optimizer can apply zero-fee routing.
    kelly_fraction: float = 0.6      # Multiplier for Kelly fraction (0.6 = "Half-Plus Kelly")
                                      # Full Kelly (1.0) maximizes growth but has high volatility
                                      # Half Kelly (0.5) reduces volatility drag significantly
    
    # Async Job Queue (Phase 4: Stability & Scaling)
    max_concurrent_jobs: int = 5              # Max number of concurrent sling rebalance jobs
    sling_job_timeout_seconds: int = 7200     # Timeout for sling jobs (2 hours default)
    sling_chunk_size_sats: int = 500000       # Amount per sling rebalance attempt (500k sats)

    # AskRene (xpay) constraint integration
    askrene_layer: str = 'xpay'               # Layer name for askrene-listlayers
    askrene_max_age_sec: int = 900            # Max constraint age (seconds) to consider fresh

    # Enhanced Sling Integration (Phase 6)
    sling_max_hops: int = 5                   # Max route hops (shorter = faster, more reliable)
    sling_parallel_jobs: int = 2              # Concurrent route attempts per job
    sling_target_sink: float = 0.40           # Balance target for sink channels (want more inbound)
    sling_target_source: float = 0.65         # Balance target for source channels (want more outbound)
    sling_target_balanced: float = 0.50       # Balance target for balanced channels
    sling_outppm_fallback: int = 500          # Max fee PPM for outppm fallback (0 = disabled)
    sling_deplete_pct_sink: float = 0.10      # Aggressive drain for sink sources
    sling_deplete_pct_source: float = 0.35    # Protective for source channels
    sling_deplete_pct_balanced: float = 0.20  # Sling default

    # Safety flags
    dry_run: bool = False          # If True, log but don't execute
    
    # Runtime dependency flags (set during init based on listplugins)
    sling_available: bool = True   # Set to False if sling plugin not detected
    
    # Phase 7 additions (v1.3.0)
    enable_vegas_reflex: bool = True       # Mempool spike defense
    vegas_decay_rate: float = 0.85         # Per-cycle decay (~30min half-life)
    enable_scarcity_pricing: bool = True   # HTLC slot scarcity pricing
    scarcity_threshold: float = 0.35       # Start pricing at 35% utilization
    
    # Hive Parameters (v1.4.0 - Strategic Rebalance Exemption)
    # v1.6.0 - Added hive_enabled for standalone/hive mode control
    hive_enabled: str = 'auto'         # "auto" = detect cl-hive, "true" = require hive, "false" = standalone
    hive_fee_ppm: int = 0              # The fee we charge fleet members (default 0)
    hive_rebalance_tolerance: int = 50 # Max sats loss allowed per rebalance to keep channels earning
    
    # Deferred (v1.4.0)
    enable_flow_asymmetry: bool = False    # Rare liquidity premium
    enable_peer_sync: bool = False         # Peer-level fee syncing

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
    # Thompson Sampling + AIMD Fee Optimization (v1.7.0)
    # ==========================================================================
    # Primary algorithm: Gaussian Thompson Sampling with contextual bandits
    # Defense layer: AIMD for rapid response to routing failures
    #
    # Thompson Sampling parameters
    thompson_prior_std_fee: int = 100         # Default prior uncertainty in ppm
    thompson_observation_decay_hours: int = 168  # 7-day half-life for observations
    thompson_max_observations: int = 200      # Bounded memory per channel
    thompson_min_observations: int = 3        # Minimum before trusting posterior

    # AIMD Defense parameters
    aimd_failure_threshold: int = 3           # Failures before multiplicative decrease
    aimd_success_threshold: int = 10          # Successes before additive increase
    aimd_multiplicative_decrease: float = 0.85  # 15% reduction on failure streak
    aimd_additive_increase_ppm: int = 5       # +5 ppm per success streak
    aimd_min_decrease_interval: int = 3600    # 1 hour cooldown between decreases

    # Hive Prior Integration
    hive_prior_weight: float = 0.6            # Weight for hive-informed priors
    hive_min_confidence_for_prior: float = 0.3  # Min confidence to use hive data

    # ==========================================================================
    # Routing Intelligence Integration (cl-hive pheromone/corridor data)
    # ==========================================================================
    # When enabled, Thompson sampling priors are weighted by pheromone data
    # from cl-hive's routing intelligence system.
    routing_intelligence_enabled: bool = False    # Opt-in feature (off by default)
    routing_intelligence_cache_seconds: int = 300  # Cache TTL for routing intel

    # ==========================================================================
    # Comprehensive Hive Data Integration (v1.8.0)
    # ==========================================================================
    # Integration with cl-hive for enhanced fee optimization and rebalancing.
    # Each integration can be individually enabled/disabled.

    # Defense status: Prevent overriding defensive fees during attacks
    hive_defense_status_enabled: bool = True
    hive_defense_status_cache_seconds: int = 60     # Short TTL - attacks are time-sensitive

    # Peer quality: Adjust optimization intensity based on peer quality
    hive_peer_quality_enabled: bool = True
    hive_peer_quality_cache_seconds: int = 300      # 5 minute cache

    # Decision history: Learn from past fee changes
    hive_decision_history_enabled: bool = True
    hive_decision_history_days: int = 30            # Days of history to consider

    # Channel flags: Identify hive-internal channels
    hive_channel_flags_enabled: bool = True

    # MCF targets: Use multi-commodity flow analysis for rebalancing
    hive_mcf_targets_enabled: bool = False          # Opt-in, may conflict with manual rebalancing
    hive_mcf_targets_cache_seconds: int = 300       # 5 minute cache

    # NNLB opportunities: Low-cost hive-internal rebalancing
    hive_nnlb_enabled: bool = False                 # Opt-in
    hive_nnlb_min_amount: int = 50000               # Minimum sats to consider
    hive_nnlb_auto_execute: bool = False            # Require manual trigger by default

    # Channel ages: Exploration/exploitation based on maturity
    hive_channel_ages_enabled: bool = True
    hive_channel_ages_cache_seconds: int = 3600     # 1 hour cache - ages change slowly

    # Internal version tracking (not a user-configurable option)
    _version: int = field(default=0, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    
    def snapshot(self) -> 'ConfigSnapshot':
        """
        Create an immutable snapshot for cycle execution.
        
        All worker cycles MUST capture a snapshot at cycle start and use
        only that snapshot for the duration of the cycle. This prevents
        torn reads when config is updated mid-cycle.
        """
        return ConfigSnapshot.from_config(self)
    
    def load_overrides(self, database: 'Database') -> None:
        """Load config overrides from database on startup."""
        overrides = database.get_all_config_overrides()
        for key, value in overrides.items():
            if hasattr(self, key) and key not in IMMUTABLE_CONFIG_KEYS:
                self._apply_override(key, value)
        self._version = database.get_config_version()
    
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
            else:
                typed_value = value
            # Range validation (matching update_runtime behavior)
            if key in CONFIG_FIELD_RANGES:
                min_val, max_val = CONFIG_FIELD_RANGES[key]
                if not (min_val <= typed_value <= max_val):
                    return  # Skip out-of-range override, keep default
            setattr(self, key, typed_value)
        except (ValueError, TypeError):
            pass  # Keep default if conversion fails
    
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
            else:
                typed_value = value
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid value for {key} (expected {field_type.__name__}): {e}"}
        
        # 3. VALIDATE: Range check
        if key in CONFIG_FIELD_RANGES:
            min_val, max_val = CONFIG_FIELD_RANGES[key]
            if not (min_val <= typed_value <= max_val):
                return {"error": f"Value {typed_value} out of range [{min_val}, {max_val}] for {key}"}
        
        old_value = getattr(self, key)
        
        # 4. WRITE to database
        new_version = database.set_config_override(key, value)
        
        # 5. READ-BACK verification (prevents Ghost Config - CRITICAL-03)
        read_back = database.get_config_override(key)
        if read_back != value:
            return {"error": "Database write verification failed (Ghost Config prevention)"}
        
        # 6. UPDATE in-memory (atomic under lock)
        with self._lock:
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
    
    # Rebalancing parameters
    rebalance_min_profit: int
    rebalance_min_profit_ppm: int
    rebalance_max_amount: int
    rebalance_min_amount: int
    low_liquidity_threshold: float
    high_liquidity_threshold: float
    rebalance_cooldown_hours: int
    inbound_fee_estimate_ppm: int
    
    # clboss integration
    clboss_enabled: bool
    clboss_unmanage_duration_hours: int
    
    # Rebalancer plugin selection
    rebalancer_plugin: str
    
    # Profitability tracking
    estimated_open_cost_sats: int
    
    # Global Capital Controls
    daily_budget_sats: int
    min_wallet_reserve: int
    revenue_boltz_auto: bool
    boltz_loop_in_max_sats: int
    boltz_loop_in_daily_cap_sats: int
    boltz_loop_in_min_confirmations: int
    swap_daily_budget_sats: int
    swap_max_fee_ppm: int
    swap_min_amount_sats: int
    swap_max_amount_sats: int
    swap_currency: str
    
    # Revenue-Proportional Budget
    enable_proportional_budget: bool
    proportional_budget_pct: float
    
    # HTLC Congestion threshold
    htlc_congestion_threshold: float
    
    # Reputation-weighted volume
    enable_reputation: bool
    reputation_decay: float

    # Kelly Criterion Position Sizing
    enable_kelly: bool
    kelly_bypass_for_fleet: bool
    kelly_fraction: float
    
    # Async Job Queue
    max_concurrent_jobs: int
    sling_job_timeout_seconds: int
    sling_chunk_size_sats: int

    # Enhanced Sling Integration (Phase 6)
    sling_max_hops: int
    sling_parallel_jobs: int
    sling_target_sink: float
    sling_target_source: float
    sling_target_balanced: float
    sling_outppm_fallback: int
    sling_deplete_pct_sink: float
    sling_deplete_pct_source: float
    sling_deplete_pct_balanced: float

    # Safety flags
    dry_run: bool
    
    # Runtime dependency flags
    sling_available: bool
    
    # Phase 7 additions (v1.3.0)
    enable_vegas_reflex: bool
    vegas_decay_rate: float
    enable_scarcity_pricing: bool
    scarcity_threshold: float
    
    # Deferred (v1.4.0)
    enable_flow_asymmetry: bool
    enable_peer_sync: bool

    # Phase 1: Operational Hardening
    rpc_timeout_seconds: int
    rpc_circuit_breaker_seconds: int
    rpc_pool_size: int
    reservation_timeout_hours: int

    # Hive Parameters (v1.4.0) - MAJOR-12 FIX: Added missing fields
    # v1.6.0 - Added hive_enabled for standalone/hive mode control
    hive_enabled: str
    hive_fee_ppm: int
    hive_rebalance_tolerance: int

    # Issue #28: Revenue rate EMA smoothing
    ema_smoothing_alpha: float

    # Issue #30: Velocity gate for rebalancing
    enable_velocity_gate: bool
    min_velocity_threshold: float
    new_channel_grace_days: int

    # Thompson Sampling + AIMD (v1.7.0)
    thompson_prior_std_fee: int
    thompson_observation_decay_hours: int
    thompson_max_observations: int
    thompson_min_observations: int
    aimd_failure_threshold: int
    aimd_success_threshold: int
    aimd_multiplicative_decrease: float
    aimd_additive_increase_ppm: int
    aimd_min_decrease_interval: int
    hive_prior_weight: float
    hive_min_confidence_for_prior: float

    # Routing Intelligence Integration
    routing_intelligence_enabled: bool
    routing_intelligence_cache_seconds: int

    # Version tracking
    version: int = 0
    
    @classmethod
    def from_config(cls, config: 'Config') -> 'ConfigSnapshot':
        """Create snapshot from mutable Config. Auto-maps matching field names."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name == 'version':
                kwargs['version'] = config._version
            elif hasattr(config, f.name):
                kwargs[f.name] = getattr(config, f.name)
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
    SPLICE_COST_SATS: int = 2000            # Splice tx fee (similar to single input/output)

    # Estimated channel lifetime
    CHANNEL_LIFETIME_DAYS: int = 365        # 1 year average
    
    # Estimated routing volume per day (conservative)
    DAILY_VOLUME_SATS: int = 1000000        # 1M sats/day
    
    @classmethod
    def calculate_floor_ppm(cls, capacity_sats: int) -> int:
        """
        Calculate the economic floor fee for a channel.
        
        Args:
            capacity_sats: Channel capacity in satoshis
            
        Returns:
            Minimum fee in PPM that covers channel costs
        """
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
