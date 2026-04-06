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
)

# Type mapping for config fields (for validation)
CONFIG_FIELD_TYPES: Dict[str, type] = {
    'flow_interval': int,
    'fee_interval': int,
    'rebalance_interval': int,
    'paused': bool,
    'min_fee_ppm': int,
    'max_fee_ppm': int,
    'daily_budget_sats': int,
    'weekly_budget_sats': int,
    'total_cost_budget_mode': str,
    'total_cost_budget_profit_pct': float,
    'total_cost_budget_profit_pct_cap': float,
    'total_cost_budget_window_hours': int,
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
    'enable_proportional_budget': bool,
    'proportional_budget_pct': float,
    'kelly_fraction': float,
    'reputation_decay': float,
    'max_concurrent_jobs': int,
    'askrene_layer': str,
    'askrene_max_age_sec': int,
    'rebalance_min_profit': int,
    'rebalance_min_profit_ppm': int,
    'rebalance_max_amount': int,
    'rebalance_min_amount': int,
    'rebalance_cooldown_hours': int,
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
    'routing_intelligence_enabled': bool,
    'routing_intelligence_cache_seconds': int,
    # Fields present in CONFIG_FIELD_RANGES that need type registration
    'base_fee_msat': int,
    'flow_window_days': int,
    'estimated_open_cost_sats': int,
    'target_flow': int,
    'source_threshold': float,
    'sink_threshold': float,
    'enable_flow_asymmetry': bool,
    'enable_peer_sync': bool,
    # Capacity Planner
    'planner_enabled': bool,
    'planner_interval': int,
    'planner_dry_run': bool,
    'planner_execute_closes': bool,
    'planner_max_opens_per_cycle': int,
    'planner_max_closes_per_cycle': int,
    'planner_min_channel_sats': int,
    'planner_max_channel_sats': int,
    'planner_min_channel_age_days': int,
    'planner_min_peer_uptime_pct': float,
    'planner_max_fee_rate_sat_vb': float,
    # Hive Hints
    'hive_hints_enabled': bool,
    'hive_hints_ttl_seconds': int,
    # Capex-Aware Rebalancer
    'rebalance_reinvestment_rate': float,
    'rebalance_bootstrap_bps': int,
    'rebalance_bootstrap_max_sats': int,
    'rebalance_grace_days': int,
}

# Explicit migration shims only. Non-public keys remain internal until they are
# intentionally exposed as deprecated compatibility controls.
DEPRECATED_RUNTIME_KEYS: FrozenSet[str] = frozenset()

# Range constraints for numeric fields
CONFIG_FIELD_RANGES: Dict[str, tuple] = {
    'min_fee_ppm': (5, 100000),  # CRITICAL-02 FIX: Minimum 5 PPM to ensure economic viability
    'max_fee_ppm': (1, 100000),
    'daily_budget_sats': (0, 10000000),
    'weekly_budget_sats': (0, 70_000_000),
    'total_cost_budget_profit_pct': (0.0, 1.0),
    'total_cost_budget_profit_pct_cap': (0.0, 1.0),
    'total_cost_budget_window_hours': (1, 168),
    'boltz_auto_cycle_interval_minutes': (1, 1440),
    'boltz_auto_cycle_max_actions': (1, 10),
    'boltz_auto_cycle_startup_delay_seconds': (0, 3600),
    'min_wallet_reserve': (0, 100000000),
    'low_liquidity_threshold': (0.0, 1.0),
    'high_liquidity_threshold': (0.0, 1.0),
    'htlc_congestion_threshold': (0.0, 1.0),
    'reputation_decay': (0.0, 1.0),
    'proportional_budget_pct': (0.0, 1.0),
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
    'routing_intelligence_cache_seconds': (60, 3600),  # 1 min to 1 hour
    # Additional range validations
    'flow_interval': (60, 86400),
    'fee_interval': (60, 86400),
    'rebalance_interval': (60, 86400),
    'max_concurrent_jobs': (1, 20),
    'askrene_max_age_sec': (10, 86400),
    'base_fee_msat': (0, 10000),
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
    'hive_hints_ttl_seconds': (60, 7200),
    # Capex-Aware Rebalancer
    'rebalance_reinvestment_rate': (0.0, 1.0),
    'rebalance_bootstrap_bps': (0, 100),
    'rebalance_bootstrap_max_sats': (0, 10000),
    'rebalance_grace_days': (0, 90),
}

# Valid values for string enum fields
STRING_ENUM_VALID_VALUES: Dict[str, tuple] = {
    'expansion_treasury_preferred_currency': ('BTC', 'LBTC', 'L-BTC', 'btc', 'lbtc', 'l-btc'),
    'total_cost_budget_mode': ('fixed', 'profit_pct'),
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
    boltz_auto_cycle_enabled: bool = True   # Run profit-gated Boltz auto-balance cycle in background
    boltz_auto_cycle_interval_minutes: int = 15  # Scheduler cadence for Boltz auto-cycle
    boltz_auto_cycle_max_actions: int = 1   # Max actions per scheduled cycle
    boltz_auto_cycle_startup_delay_seconds: int = 120  # Delay before first Boltz auto-cycle
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
    base_fee_msat: int = 0         # Base fee (we focus on PPM)
    
    # Rebalancing parameters
    rebalance_min_profit: int = 10     # Min profit in sats to trigger (legacy, used when ppm=0)
    rebalance_min_profit_ppm: int = 0  # Min profit in PPM (0 = use sats threshold, >0 = use ppm)
                                        # Recommended: 20 ppm (~10 sats per 500k chunk)
    rebalance_max_amount: int = 5000000  # Max rebalance amount in sats
    rebalance_min_amount: int = 50000    # Min rebalance amount in sats
    low_liquidity_threshold: float = 0.3  # Below 30% = low outbound
    high_liquidity_threshold: float = 0.7 # Above 70% = high outbound
    rebalance_cooldown_hours: int = 24   # Don't re-rebalance same channel for 24h
    futility_cooldown_hours: int = 48   # Hours before retrying after 10+ consecutive failures
    inbound_fee_estimate_ppm: int = 50  # Route cost buffer added on top of last-hop fee (PPM)
    
    # Profitability tracking
    estimated_open_cost_sats: int = 5000  # Estimated on-chain fee for channel open
    
    # Global Capital Controls
    daily_budget_sats: int = 5000          # Max rebalancing fees per 24h period (fixed floor)
    weekly_budget_sats: int = 35000        # Max rebalancing fees per 7-day window (hard ceiling)
    total_cost_budget_mode: str = 'fixed'  # 'fixed' or 'profit_pct' (global liquidity spend gate)
    total_cost_budget_profit_pct: float = 0.30  # Percent of net profit allocated to spend budget when mode=profit_pct
    total_cost_budget_profit_pct_cap: float = 0.75  # Hard cap for pct input (operator guard)
    total_cost_budget_window_hours: int = 24  # Window for profit-based budget calculation
    min_wallet_reserve: int = 1_000_000    # Min sats (confirmed on-chain + channel spendable) before ABORT

    # Revenue-Proportional Budget
    enable_proportional_budget: bool = True   # Scale daily budget based on revenue (Issue #22)
    proportional_budget_pct: float = 0.30     # Budget = max(daily_budget_sats, revenue_24h * pct)
                                               # Default 30% of 24h revenue

    # Capex-Aware Rebalancer
    rebalance_reinvestment_rate: float = 0.50   # Fraction of channel contribution for rebalance budget
    rebalance_bootstrap_bps: int = 10           # Bootstrap budget: basis points of capacity per 30d
    rebalance_bootstrap_max_sats: int = 200     # Max bootstrap budget per channel per 30d
    rebalance_grace_days: int = 14              # Days before bootstrap activates for new channels

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

    # Safety flags
    paused: bool = False           # If True, suppress automated executor actions
    dry_run: bool = False          # If True, log but don't execute

    # Vegas Reflex (mempool spike defense)
    enable_vegas_reflex: bool = True       # Mempool spike defense
    vegas_decay_rate: float = 0.85         # Per-cycle decay (~30min half-life)
    
    # Deferred features
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
    # DTS (Discounted Thompson Sampling) Parameters
    # ==========================================================================
    thompson_prior_std_fee: int = 100         # Default prior uncertainty in ppm
    thompson_observation_decay_hours: int = 168  # 7-day half-life for observations
    thompson_max_observations: int = 200      # Bounded memory per channel
    thompson_min_observations: int = 3        # Minimum before trusting posterior


    # ==========================================================================
    # Routing Intelligence Integration
    # ==========================================================================
    routing_intelligence_enabled: bool = False    # Opt-in feature (off by default)
    routing_intelligence_cache_seconds: int = 300  # Cache TTL for routing intel

    # Capacity Planner
    planner_enabled: bool = False
    planner_interval: int = 21600               # 6 hours
    planner_dry_run: bool = False
    planner_execute_closes: bool = False
    planner_max_opens_per_cycle: int = 1
    planner_max_closes_per_cycle: int = 0
    planner_min_channel_sats: int = 500000      # 500k sats
    planner_max_channel_sats: int = 10000000    # 10M sats
    planner_min_channel_age_days: int = 30
    planner_min_peer_uptime_pct: float = 95.0
    planner_max_fee_rate_sat_vb: float = 50.0
    # Hive Hints integration
    hive_hints_enabled: bool = True
    hive_hints_ttl_seconds: int = 0  # 0 = use snapshot's ttl_seconds
    # Internal version tracking (not a user-configurable option)
    _version: int = field(default=0, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _override_warnings: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate cross-field invariants on direct construction."""
        pass
    
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
            # AUDIT FIX I-5: Validate sink/source threshold cross-field consistency
            if key == 'sink_threshold' and typed_value >= self.source_threshold:
                return {"error": f"sink_threshold ({typed_value}) must be less than source_threshold ({self.source_threshold})"}
            if key == 'source_threshold' and typed_value <= self.sink_threshold:
                return {"error": f"source_threshold ({typed_value}) must be greater than sink_threshold ({self.sink_threshold})"}

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
    
    # Rebalancing parameters
    rebalance_min_profit: int
    rebalance_min_profit_ppm: int
    rebalance_max_amount: int
    rebalance_min_amount: int
    low_liquidity_threshold: float
    high_liquidity_threshold: float
    rebalance_cooldown_hours: int
    futility_cooldown_hours: int
    inbound_fee_estimate_ppm: int

    # Profitability tracking
    estimated_open_cost_sats: int
    
    # Global Capital Controls
    daily_budget_sats: int
    total_cost_budget_mode: str
    total_cost_budget_profit_pct: float
    total_cost_budget_profit_pct_cap: float
    total_cost_budget_window_hours: int
    min_wallet_reserve: int

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
    kelly_fraction: float
    
    # Async Job Queue
    max_concurrent_jobs: int

    # Safety flags
    dry_run: bool

    # Vegas Reflex (mempool spike defense)
    enable_vegas_reflex: bool
    vegas_decay_rate: float
    
    # Deferred features
    enable_flow_asymmetry: bool
    enable_peer_sync: bool

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
    routing_intelligence_enabled: bool
    routing_intelligence_cache_seconds: int

    # M-27: xpay/askrene parameters (were missing from snapshot)
    askrene_layer: str = 'xpay'
    askrene_max_age_sec: int = 900

    # Weekly budget cap (hard ceiling over daily burst)
    weekly_budget_sats: int = 35000

    # Capacity Planner
    planner_enabled: bool = False
    planner_interval: int = 21600
    planner_dry_run: bool = False
    planner_execute_closes: bool = False
    planner_max_opens_per_cycle: int = 1
    planner_max_closes_per_cycle: int = 0
    planner_min_channel_sats: int = 500000
    planner_max_channel_sats: int = 10000000
    planner_min_channel_age_days: int = 30
    planner_min_peer_uptime_pct: float = 95.0
    planner_max_fee_rate_sat_vb: float = 50.0
    # Hive Hints
    hive_hints_enabled: bool = True
    hive_hints_ttl_seconds: int = 0
    # Capex-Aware Rebalancer
    rebalance_reinvestment_rate: float = 0.50
    rebalance_bootstrap_bps: int = 10
    rebalance_bootstrap_max_sats: int = 200
    rebalance_grace_days: int = 14
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
