"""
Configuration module for cl-revenue-ops

Contains the Config dataclass that holds all tunable parameters
for the Revenue Operations plugin.

Phase 7 additions:
- ConfigSnapshot: Immutable snapshot for thread-safe cycle execution
- Runtime configuration updates via RPC
- Vegas Reflex and Scarcity Pricing settings
"""

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
    'low_liquidity_threshold': float,
    'high_liquidity_threshold': float,
    'htlc_congestion_threshold': float,
    'enable_reputation': bool,
    'enable_prometheus': bool,
    'enable_kelly': bool,
    'enable_proportional_budget': bool,
    'proportional_budget_pct': float,
    'kelly_fraction': float,
    'reputation_decay': float,
    'max_concurrent_jobs': int,
    'sling_job_timeout_seconds': int,
    'sling_chunk_size_sats': int,
    'sling_max_hops': int,
    'sling_parallel_jobs': int,
    'sling_target_sink': float,
    'sling_target_source': float,
    'sling_target_balanced': float,
    'sling_outppm_fallback': int,
    'rebalance_min_profit': int,
    'rebalance_min_profit_ppm': int,
    'rebalance_max_amount': int,
    'rebalance_min_amount': int,
    'rebalance_cooldown_hours': int,
    'inbound_fee_estimate_ppm': int,
    'prometheus_port': int,
    # Phase 7 additions
    'enable_vegas_reflex': bool,
    'vegas_decay_rate': float,
    'enable_scarcity_pricing': bool,
    'scarcity_threshold': float,
    # Hive Parameters
    'hive_fee_ppm': int,
    'hive_rebalance_tolerance': int,
    # Phase 1: Operational Hardening
    'rpc_timeout_seconds': int,
    'rpc_circuit_breaker_seconds': int,
    'reservation_timeout_hours': int,
}

# Range constraints for numeric fields
CONFIG_FIELD_RANGES: Dict[str, tuple] = {
    'min_fee_ppm': (5, 100000),  # CRITICAL-02 FIX: Minimum 5 PPM to ensure economic viability
    'max_fee_ppm': (1, 100000),
    'daily_budget_sats': (0, 10000000),
    'min_wallet_reserve': (0, 100000000),
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
    'rebalance_min_profit_ppm': (0, 100000),
    'rpc_timeout_seconds': (1, 300),
    'rpc_circuit_breaker_seconds': (0, 3600),
    'reservation_timeout_hours': (1, 24),
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
    fee_interval: int = 1800       # 30 minutes
    rebalance_interval: int = 900  # 15 minutes
    
    # Flow analysis parameters
    target_flow: int = 100000      # Target sats routed per day per channel
    flow_window_days: int = 7      # Days to analyze for flow calculation
    
    # Flow ratio thresholds for classification
    source_threshold: float = 0.5   # FlowRatio > 0.5 = Source (draining)
    sink_threshold: float = -0.5    # FlowRatio < -0.5 = Sink (filling)
    
    # Fee parameters
    min_fee_ppm: int = 10          # Floor fee in PPM
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
    
    # Revenue-Proportional Budget (Phase 7: Dynamic Budget Scaling)
    enable_proportional_budget: bool = True   # Scale daily budget based on revenue (Issue #22)
    proportional_budget_pct: float = 0.30     # Budget = max(daily_budget_sats, revenue_24h * pct)
                                               # Default 30% of 24h revenue
    
    # Phase 1: Operational Hardening
    rpc_timeout_seconds: int = 15
    rpc_circuit_breaker_seconds: int = 60
    reservation_timeout_hours: int = 4  # Hours before stale budget reservations auto-release
    
    # HTLC Congestion threshold
    htlc_congestion_threshold: float = 0.8  # Mark channel as CONGESTED if >80% HTLC slots used
    
    # Reputation-weighted volume
    enable_reputation: bool = True  # If True, weight volume by peer success rate
    reputation_decay: float = 0.98  # Decay factor per flow_interval (default hourly)
                                     # 0.98^24 ≈ 0.61, meaning old data loses ~40% weight daily
    
    # Prometheus Metrics (Phase 2: Observability)
    enable_prometheus: bool = False  # If True, start Prometheus metrics exporter (disabled by default)
    prometheus_port: int = 9800      # Port for Prometheus HTTP server
    
    # Kelly Criterion Position Sizing (Phase 4: Risk Management)
    enable_kelly: bool = False       # If True, scale rebalance budget by Kelly fraction
    kelly_fraction: float = 0.5      # Multiplier for Kelly fraction ("Half Kelly" is standard)
                                      # Full Kelly (1.0) maximizes growth but has high volatility
                                      # Half Kelly (0.5) reduces volatility drag significantly
    
    # Async Job Queue (Phase 4: Stability & Scaling)
    max_concurrent_jobs: int = 5              # Max number of concurrent sling rebalance jobs
    sling_job_timeout_seconds: int = 7200     # Timeout for sling jobs (2 hours default)
    sling_chunk_size_sats: int = 500000       # Amount per sling rebalance attempt (500k sats)

    # Enhanced Sling Integration (Phase 6)
    sling_max_hops: int = 5                   # Max route hops (shorter = faster, more reliable)
    sling_parallel_jobs: int = 1              # Concurrent route attempts per job
    sling_target_sink: float = 0.35           # Balance target for sink channels (want more inbound)
    sling_target_source: float = 0.65         # Balance target for source channels (want more outbound)
    sling_target_balanced: float = 0.50       # Balance target for balanced channels
    sling_outppm_fallback: int = 500          # Max fee PPM for outppm fallback (0 = disabled)

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
    hive_fee_ppm: int = 0              # The fee we charge fleet members (default 0)
    hive_rebalance_tolerance: int = 50 # Max sats we are willing to LOSE to balance a friend
    
    # Deferred (v1.4.0)
    enable_flow_asymmetry: bool = False    # Rare liquidity premium
    enable_peer_sync: bool = False         # Peer-level fee syncing
    
    # Internal version tracking (not a user-configurable option)
    _version: int = field(default=0, repr=False, compare=False)
    
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
        """Apply a single override with type conversion."""
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
    
    # Revenue-Proportional Budget
    enable_proportional_budget: bool
    proportional_budget_pct: float
    
    # HTLC Congestion threshold
    htlc_congestion_threshold: float
    
    # Reputation-weighted volume
    enable_reputation: bool
    reputation_decay: float
    
    # Prometheus Metrics
    enable_prometheus: bool
    prometheus_port: int
    
    # Kelly Criterion Position Sizing
    enable_kelly: bool
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
    reservation_timeout_hours: int

    # Hive Parameters (v1.4.0) - MAJOR-12 FIX: Added missing fields
    hive_fee_ppm: int
    hive_rebalance_tolerance: int

    # Version tracking
    version: int = 0
    
    @classmethod
    def from_config(cls, config: 'Config') -> 'ConfigSnapshot':
        """Create snapshot from mutable Config."""
        return cls(
            db_path=config.db_path,
            flow_interval=config.flow_interval,
            fee_interval=config.fee_interval,
            rebalance_interval=config.rebalance_interval,
            target_flow=config.target_flow,
            flow_window_days=config.flow_window_days,
            source_threshold=config.source_threshold,
            sink_threshold=config.sink_threshold,
            min_fee_ppm=config.min_fee_ppm,
            max_fee_ppm=config.max_fee_ppm,
            base_fee_msat=config.base_fee_msat,
            rebalance_min_profit=config.rebalance_min_profit,
            rebalance_min_profit_ppm=config.rebalance_min_profit_ppm,
            rebalance_max_amount=config.rebalance_max_amount,
            rebalance_min_amount=config.rebalance_min_amount,
            low_liquidity_threshold=config.low_liquidity_threshold,
            high_liquidity_threshold=config.high_liquidity_threshold,
            rebalance_cooldown_hours=config.rebalance_cooldown_hours,
            inbound_fee_estimate_ppm=config.inbound_fee_estimate_ppm,
            clboss_enabled=config.clboss_enabled,
            clboss_unmanage_duration_hours=config.clboss_unmanage_duration_hours,
            rebalancer_plugin=config.rebalancer_plugin,
            estimated_open_cost_sats=config.estimated_open_cost_sats,
            daily_budget_sats=config.daily_budget_sats,
            min_wallet_reserve=config.min_wallet_reserve,
            enable_proportional_budget=config.enable_proportional_budget,
            proportional_budget_pct=config.proportional_budget_pct,
            htlc_congestion_threshold=config.htlc_congestion_threshold,
            enable_reputation=config.enable_reputation,
            reputation_decay=config.reputation_decay,
            enable_prometheus=config.enable_prometheus,
            prometheus_port=config.prometheus_port,
            enable_kelly=config.enable_kelly,
            kelly_fraction=config.kelly_fraction,
            max_concurrent_jobs=config.max_concurrent_jobs,
            sling_job_timeout_seconds=config.sling_job_timeout_seconds,
            sling_chunk_size_sats=config.sling_chunk_size_sats,
            sling_max_hops=config.sling_max_hops,
            sling_parallel_jobs=config.sling_parallel_jobs,
            sling_target_sink=config.sling_target_sink,
            sling_target_source=config.sling_target_source,
            sling_target_balanced=config.sling_target_balanced,
            sling_outppm_fallback=config.sling_outppm_fallback,
            dry_run=config.dry_run,
            sling_available=config.sling_available,
            enable_vegas_reflex=config.enable_vegas_reflex,
            vegas_decay_rate=config.vegas_decay_rate,
            enable_scarcity_pricing=config.enable_scarcity_pricing,
            scarcity_threshold=config.scarcity_threshold,
            enable_flow_asymmetry=config.enable_flow_asymmetry,
            enable_peer_sync=config.enable_peer_sync,
            rpc_timeout_seconds=config.rpc_timeout_seconds,
            rpc_circuit_breaker_seconds=config.rpc_circuit_breaker_seconds,
            reservation_timeout_hours=config.reservation_timeout_hours,
            hive_fee_ppm=config.hive_fee_ppm,
            hive_rebalance_tolerance=config.hive_rebalance_tolerance,
            version=config._version,
        )


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