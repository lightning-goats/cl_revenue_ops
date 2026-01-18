# Phase 7 v1.3.0 Implementation Plan

## Technical Implementation Guide — "The 1% Node" Defense

| Field | Value |
|-------|-------|
| **Date** | January 3, 2026 |
| **Target Version** | cl-revenue-ops v1.3.0 |
| **Status** | ✅ COMPLETED |
| **Specification** | [`PHASE7_SPECIFICATION.md`](../specs/PHASE7_SPECIFICATION.md) |
| **Red Team Report** | [`PHASE7_RED_TEAM_REPORT.md`](../audits/PHASE7_RED_TEAM_REPORT.md) |

---

## 1. Executive Summary

This document provides step-by-step implementation details for Phase 7 v1.3.0.

| Priority | Feature | Vulnerabilities Addressed | Dependency |
|----------|---------|---------------------------|------------|
| 1 | Dynamic Runtime Configuration | CRITICAL-02, CRITICAL-03 | Foundation |
| 2 | Mempool Acceleration (Vegas Reflex) | CRITICAL-01, HIGH-03 | Requires Feature 1 |
| 3 | HTLC Slot Scarcity Pricing | HIGH-01, HIGH-02, MEDIUM-01 | Requires Feature 2 patterns |
| 4 | Liquidity Efficiency Suite | Capital Efficiency / Resource Waste | Independent |

---

## 2. Feature 1: Dynamic Runtime Configuration

### 2.1 Objective

Allow operators to tune algorithmic thresholds via RPC without restarting the plugin, while preventing race conditions and configuration corruption.

### 2.2 Files to Modify

| File | Changes |
|------|---------|
| `modules/database.py` | Add `config_overrides` table and helper methods |
| `modules/config.py` | Add `ConfigSnapshot` pattern, version tracking, runtime update logic |
| `cl-revenue-ops.py` | Add `revenue-config` RPC command |

### 2.3 Database Schema

**Location:** `modules/database.py` → `initialize()`

```python
# Config overrides table (Phase 7: Dynamic Runtime Configuration)
# Stores operator overrides that persist across restarts
conn.execute("""
    CREATE TABLE IF NOT EXISTS config_overrides (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        updated_at INTEGER NOT NULL
    )
""")
```

### 2.4 Database Helper Methods

**Location:** `modules/database.py`

```python
def get_config_override(self, key: str) -> Optional[str]:
    """Get a single config override value."""
    conn = self._get_connection()
    row = conn.execute(
        "SELECT value FROM config_overrides WHERE key = ?", (key,)
    ).fetchone()
    return row['value'] if row else None

def set_config_override(self, key: str, value: str) -> int:
    """
    Set a config override with transactional safety.
    
    Returns:
        New version number after update
    """
    conn = self._get_connection()
    now = int(time.time())
    
    # Get current max version
    row = conn.execute("SELECT MAX(version) as max_v FROM config_overrides").fetchone()
    new_version = (row['max_v'] or 0) + 1
    
    conn.execute("""
        INSERT OR REPLACE INTO config_overrides (key, value, version, updated_at)
        VALUES (?, ?, ?, ?)
    """, (key, value, new_version, now))
    
    return new_version

def get_all_config_overrides(self) -> Dict[str, str]:
    """Get all config overrides as a dictionary."""
    conn = self._get_connection()
    rows = conn.execute("SELECT key, value FROM config_overrides").fetchall()
    return {row['key']: row['value'] for row in rows}

def get_config_version(self) -> int:
    """Get current config version (max version in table)."""
    conn = self._get_connection()
    row = conn.execute("SELECT MAX(version) as max_v FROM config_overrides").fetchone()
    return row['max_v'] or 0

def delete_config_override(self, key: str) -> bool:
    """Delete a config override, returning to default."""
    conn = self._get_connection()
    cursor = conn.execute("DELETE FROM config_overrides WHERE key = ?", (key,))
    return cursor.rowcount > 0
```

### 2.5 ConfigSnapshot Pattern

**Location:** `modules/config.py`

```python
from dataclasses import dataclass, asdict, field
from typing import FrozenSet

# Immutable keys that cannot be changed at runtime
IMMUTABLE_CONFIG_KEYS: FrozenSet[str] = frozenset({
    'db_path',
    'dry_run',  # Safety: don't allow enabling dry_run to hide actions
})

# Type mapping for config fields (for validation)
CONFIG_FIELD_TYPES = {
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
    'enable_kelly': bool,
    'enable_proportional_budget': bool,
    'proportional_budget_pct': float,
    'kelly_fraction': float,
    'reputation_decay': float,
    'max_concurrent_jobs': int,
    'sling_job_timeout_seconds': int,
    'sling_chunk_size_sats': int,
    # Phase 7 additions
    'enable_vegas_reflex': bool,
    'vegas_decay_rate': float,
    'enable_scarcity_pricing': bool,
    'scarcity_threshold': float,
}

# Range constraints for numeric fields
CONFIG_FIELD_RANGES = {
    'min_fee_ppm': (0, 100000),
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
}


@dataclass(frozen=True)
class ConfigSnapshot:
    """
    Immutable configuration snapshot for thread-safe cycle execution.
    
    All worker cycles MUST capture a snapshot at cycle start and use
    only that snapshot for the duration of the cycle. This prevents
    torn reads when config is updated mid-cycle.
    """
    # Copy all fields from Config...
    db_path: str
    flow_interval: int
    fee_interval: int
    rebalance_interval: int
    # ... (all other fields)
    
    # Version tracking
    version: int = 0
    
    @classmethod
    def from_config(cls, config: 'Config', version: int = 0) -> 'ConfigSnapshot':
        """Create snapshot from mutable Config."""
        return cls(**asdict(config), version=version)


@dataclass
class Config:
    """Configuration container with runtime update support."""
    
    # ... existing fields ...
    
    # Phase 7 additions (v1.3)
    enable_vegas_reflex: bool = True
    vegas_decay_rate: float = 0.85
    enable_scarcity_pricing: bool = True
    scarcity_threshold: float = 0.35
    
    # Internal version tracking
    _version: int = field(default=0, repr=False)
    
    def snapshot(self) -> ConfigSnapshot:
        """Create an immutable snapshot for cycle execution."""
        d = asdict(self)
        d.pop('_version', None)
        return ConfigSnapshot(**d, version=self._version)
    
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
        
        Returns:
            Dict with status, old_value, new_value, version
        """
        # 1. VALIDATE: Check if key exists and is mutable
        if key in IMMUTABLE_CONFIG_KEYS:
            return {"error": f"Key '{key}' cannot be changed at runtime"}
        
        if not hasattr(self, key):
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
        
        # 5. READ-BACK verification (prevents Ghost Config)
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
```

### 2.6 RPC Commands

**Location:** `cl-revenue-ops.py`

```python
@plugin.method("revenue-config")
def revenue_config(plugin: Plugin, action: str, key: str = None, value: str = None) -> Dict[str, Any]:
    """
    Get or set runtime configuration.
    
    Usage:
      lightning-cli revenue-config get           # Get all config
      lightning-cli revenue-config get <key>     # Get specific key
      lightning-cli revenue-config set <key> <value>  # Set key
      lightning-cli revenue-config reset <key>   # Reset to default
      lightning-cli revenue-config list-mutable  # List changeable keys
    """
    if config is None or database is None:
        return {"error": "Plugin not initialized"}
    
    if action == "get":
        if key:
            if not hasattr(config, key):
                return {"error": f"Unknown config key: {key}"}
            return {
                "key": key,
                "value": getattr(config, key),
                "version": config._version
            }
        else:
            # Return all config as dict
            snapshot = config.snapshot()
            return {
                "config": asdict(snapshot),
                "version": config._version
            }
    
    elif action == "set":
        if not key or value is None:
            return {"error": "Usage: revenue-config set <key> <value>"}
        
        result = config.update_runtime(database, key, str(value))
        
        if result.get("status") == "success":
            plugin.log(
                f"CONFIG UPDATE: {key} changed from {result['old_value']} "
                f"to {result['new_value']} (v{result['version']})",
                level='info'
            )
        
        return result
    
    elif action == "reset":
        if not key:
            return {"error": "Usage: revenue-config reset <key>"}
        
        if database.delete_config_override(key):
            # Reload from defaults (would need plugin restart or default lookup)
            return {"status": "success", "message": f"Override for {key} removed. Restart plugin to apply default."}
        return {"error": f"No override found for {key}"}
    
    elif action == "list-mutable":
        from modules.config import CONFIG_FIELD_TYPES, IMMUTABLE_CONFIG_KEYS
        mutable = [k for k in CONFIG_FIELD_TYPES.keys() if k not in IMMUTABLE_CONFIG_KEYS]
        return {"mutable_keys": sorted(mutable), "count": len(mutable)}
    
    else:
        return {"error": f"Unknown action: {action}. Use 'get', 'set', 'reset', or 'list-mutable'"}
```

### 2.7 Integration Points

Update worker cycles to use snapshots:

**Location:** `modules/fee_controller.py` → `adjust_all_fees()`

```python
def adjust_all_fees(self) -> List[FeeAdjustment]:
    """Adjust fees for all channels using Hill Climbing optimization."""
    # Capture immutable config snapshot for this cycle
    cfg = self.config.snapshot()
    
    # Use cfg instead of self.config throughout this method
    # ...
```

**Location:** `modules/rebalancer.py` → `find_rebalance_candidates()`

```python
def find_rebalance_candidates(self) -> List[RebalanceCandidate]:
    """Find channels that would benefit from rebalancing."""
    # Capture immutable config snapshot for this cycle
    cfg = self.config.snapshot()
    
    # Use cfg instead of self.config throughout this method
    # ...
```

---

## 3. Feature 2: Mempool Acceleration (Vegas Reflex)

### 3.1 Objective

Protect against arbitrageurs draining channels during high on-chain fee spikes by dynamically raising fee floors. Uses exponential decay (not fixed latch) to prevent DoS attacks.

### 3.2 Files to Modify

| File | Changes |
|------|---------|
| `modules/fee_controller.py` | Add `VegasReflexState` class, integrate into `_calculate_floor()` |
| `modules/database.py` | Add `mempool_fee_history` table for MA calculation |

### 3.3 VegasReflexState Class

**Location:** `modules/fee_controller.py` (after `HillClimbState`)

```python
@dataclass
class VegasReflexState:
    """
    State for Vegas Reflex mempool acceleration.
    
    Defenses implemented:
    - CRITICAL-01: Exponential decay prevents permanent latch
    - HIGH-03: Probabilistic early trigger at 200-400% spikes
    """
    intensity: float = 0.0          # Range: 0.0 to 1.0
    decay_rate: float = 0.85        # Per-cycle decay (~30min half-life at 30min intervals)
    last_sat_vb: float = 1.0        # Last observed sat/vB
    last_update: int = 0            # Unix timestamp
    consecutive_spikes: int = 0     # For confirmation window
    
    def update(self, current_sat_vb: float, ma_sat_vb: float) -> None:
        """
        Update intensity based on mempool spike ratio.
        
        Args:
            current_sat_vb: Current mempool fee rate in sat/vB
            ma_sat_vb: Moving average fee rate (24h)
        """
        if ma_sat_vb <= 0:
            ma_sat_vb = 1.0  # Prevent division by zero
        
        spike_ratio = current_sat_vb / ma_sat_vb
        
        # Track consecutive spikes for confirmation window
        if spike_ratio >= 2.0:
            self.consecutive_spikes += 1
        else:
            self.consecutive_spikes = 0
        
        if spike_ratio >= 4.0:
            # Immediate trigger: set intensity to max (>400% spike)
            self.intensity = 1.0
        elif spike_ratio >= 2.0:
            # HIGH-03 Defense: Probabilistic boost for 200-400% spikes
            # Either 2 consecutive spikes OR random chance proportional to spike
            import random
            boost = (spike_ratio - 2.0) / 2.0  # 0.0 to 1.0
            
            if self.consecutive_spikes >= 2 or random.random() < boost * 0.5:
                self.intensity = min(1.0, self.intensity + boost * 0.3)
        
        # Always decay toward zero (CRITICAL-01 defense)
        self.intensity *= self.decay_rate
        self.last_sat_vb = current_sat_vb
        self.last_update = int(time.time())
    
    def get_floor_multiplier(self) -> float:
        """
        Get fee floor multiplier based on intensity.
        
        Returns:
            Multiplier from 1.0x (calm) to 3.0x (max intensity)
        """
        if self.intensity < 0.01:
            return 1.0
        # Smooth curve using square root for gradual response
        return 1.0 + (self.intensity ** 0.5) * 2.0
```

### 3.4 Mempool History Tracking

**Location:** `modules/database.py` → `initialize()`

```python
# Mempool fee history (for Vegas Reflex MA calculation)
conn.execute("""
    CREATE TABLE IF NOT EXISTS mempool_fee_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sat_per_vbyte REAL NOT NULL,
        timestamp INTEGER NOT NULL
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_mempool_time ON mempool_fee_history(timestamp)")
```

**Location:** `modules/database.py` (helper methods)

```python
def record_mempool_fee(self, sat_per_vbyte: float) -> None:
    """Record current mempool fee rate."""
    conn = self._get_connection()
    now = int(time.time())
    conn.execute(
        "INSERT INTO mempool_fee_history (sat_per_vbyte, timestamp) VALUES (?, ?)",
        (sat_per_vbyte, now)
    )
    # Prune old entries (keep 48h)
    conn.execute(
        "DELETE FROM mempool_fee_history WHERE timestamp < ?",
        (now - 172800,)
    )

def get_mempool_ma(self, window_seconds: int = 86400) -> float:
    """Get moving average of mempool fees over window."""
    conn = self._get_connection()
    cutoff = int(time.time()) - window_seconds
    row = conn.execute(
        "SELECT AVG(sat_per_vbyte) as avg_fee FROM mempool_fee_history WHERE timestamp >= ?",
        (cutoff,)
    ).fetchone()
    return row['avg_fee'] if row and row['avg_fee'] else 1.0
```

### 3.5 Integration into FeeController

**Location:** `modules/fee_controller.py` → `__init__()`

```python
def __init__(self, plugin: Plugin, config: Config, database: Database, ...):
    # ... existing init ...
    
    # Phase 7: Vegas Reflex state (global, not per-channel)
    self._vegas_state = VegasReflexState(decay_rate=config.vegas_decay_rate)
```

**Location:** `modules/fee_controller.py` → `adjust_all_fees()` (before channel loop)

```python
def adjust_all_fees(self) -> List[FeeAdjustment]:
    adjustments = []
    cfg = self.config.snapshot()
    
    # ... existing setup ...
    
    # Phase 7: Update Vegas Reflex state
    if cfg.enable_vegas_reflex:
        chain_costs = self._get_dynamic_chain_costs()
        if chain_costs:
            current_sat_vb = chain_costs.get("sat_per_vbyte", 1.0)
            
            # Record for MA calculation
            self.database.record_mempool_fee(current_sat_vb)
            
            # Get 24h moving average
            ma_sat_vb = self.database.get_mempool_ma(86400)
            
            # Update Vegas state
            self._vegas_state.update(current_sat_vb, ma_sat_vb)
            
            if self._vegas_state.intensity > 0.1:
                self.plugin.log(
                    f"VEGAS REFLEX: Mempool spike detected. "
                    f"Current: {current_sat_vb:.1f} sat/vB, MA: {ma_sat_vb:.1f} sat/vB, "
                    f"Intensity: {self._vegas_state.intensity:.2f}, "
                    f"Floor multiplier: {self._vegas_state.get_floor_multiplier():.2f}x",
                    level='info'
                )
    
    # ... continue with channel loop ...
```

**Location:** `modules/fee_controller.py` → `_calculate_floor()` (at end, before return)

```python
def _calculate_floor(self, capacity_sats: int, ...) -> int:
    # ... existing floor calculation ...
    
    # Phase 7: Apply Vegas Reflex multiplier
    if self.config.enable_vegas_reflex and hasattr(self, '_vegas_state'):
        multiplier = self._vegas_state.get_floor_multiplier()
        if multiplier > 1.01:
            self.plugin.log(
                f"VEGAS REFLEX: Applying {multiplier:.2f}x floor multiplier",
                level='debug'
            )
            floor_ppm = int(floor_ppm * multiplier)
    
    return max(1, int(floor_ppm))
```

---

## 4. Feature 3: HTLC Slot Scarcity Pricing

### 4.1 Objective

Exponentially price capacity as HTLC slots fill up to prevent channel paralysis. Uses value-weighted utilization (prevents Dust Flood attack) with asymmetric EMA (fast up, slow down).

### 4.2 Files to Modify

| File | Changes |
|------|---------|
| `modules/fee_controller.py` | Add `ScarcityState` class, per-channel state tracking, integration |
| `modules/rebalancer.py` | Add `_check_scarcity_safe()` pre-flight check |

### 4.3 ScarcityState Class

**Location:** `modules/fee_controller.py` (after `VegasReflexState`)

```python
@dataclass
class ScarcityState:
    """
    State for HTLC slot scarcity pricing per channel.
    
    Defenses implemented:
    - HIGH-01: Value-weighted utilization prevents Dust Flood
    - MEDIUM-01: Asymmetric EMA prevents premature relaxation
    """
    max_htlcs: int = 483            # Standard max concurrent HTLCs
    utilization_ema: float = 0.0    # Smoothed utilization
    alpha_up: float = 0.4           # Fast response to increases
    alpha_down: float = 0.1         # Slow response to decreases
    last_update: int = 0
    
    def update(self, htlcs: List[Dict[str, Any]]) -> float:
        """
        Update utilization EMA based on current pending HTLCs.
        
        Uses value-weighting: 1M sat HTLC = 10 slots, 1K sat = 0.01 slots.
        This prevents Dust Flood attacks where an adversary fills slots
        with tiny HTLCs.
        
        Args:
            htlcs: List of pending HTLCs with 'amount_msat' field
            
        Returns:
            Current utilization EMA (0.0 to 1.0)
        """
        weighted_slots = 0.0
        
        for htlc in htlcs:
            amount_msat = htlc.get("amount_msat", 0)
            if isinstance(amount_msat, str):
                amount_msat = int(amount_msat.replace("msat", ""))
            
            # Value weighting: 100M msat (100k sats) = 1.0 slot
            # Clamp to 0.01 (dust) to 10.0 (whale) range
            slot_weight = min(10.0, max(0.01, amount_msat / 100_000_000))
            weighted_slots += slot_weight
        
        raw_utilization = min(1.0, weighted_slots / self.max_htlcs)
        
        # Asymmetric EMA: fast up (defense), slow down (stability)
        if raw_utilization > self.utilization_ema:
            alpha = self.alpha_up
        else:
            alpha = self.alpha_down
        
        self.utilization_ema = alpha * raw_utilization + (1 - alpha) * self.utilization_ema
        self.last_update = int(time.time())
        
        return self.utilization_ema
    
    def get_multiplier(self) -> float:
        """
        Get fee multiplier based on scarcity.
        
        Returns:
            Multiplier from 1.0x (low util) to 3.0x (full util)
        """
        threshold = 0.35  # Start pricing at 35% utilization
        
        if self.utilization_ema <= threshold:
            return 1.0
        
        # Quadratic curve: 1.0x at threshold, up to 3.0x at 100%
        progress = (self.utilization_ema - threshold) / (1.0 - threshold)
        return 1.0 + (progress ** 2) * 2.0
    
    def calculate_raw_utilization(self, htlcs: List[Dict[str, Any]]) -> float:
        """
        Calculate raw (non-EMA) utilization for forecasting.
        
        Used by rebalancer to simulate post-rebalance state.
        """
        weighted_slots = 0.0
        for htlc in htlcs:
            amount_msat = htlc.get("amount_msat", 0)
            if isinstance(amount_msat, str):
                amount_msat = int(amount_msat.replace("msat", ""))
            slot_weight = min(10.0, max(0.01, amount_msat / 100_000_000))
            weighted_slots += slot_weight
        return min(1.0, weighted_slots / self.max_htlcs)
```

### 4.4 Per-Channel State Tracking

**Location:** `modules/fee_controller.py` → `__init__()`

```python
def __init__(self, plugin: Plugin, config: Config, database: Database, ...):
    # ... existing init ...
    
    # Phase 7: Per-channel scarcity states
    self._scarcity_states: Dict[str, ScarcityState] = {}
```

**Location:** `modules/fee_controller.py` (new helper method)

```python
def _get_scarcity_state(self, channel_id: str) -> ScarcityState:
    """Get or create scarcity state for a channel."""
    if channel_id not in self._scarcity_states:
        self._scarcity_states[channel_id] = ScarcityState()
    return self._scarcity_states[channel_id]

def _get_pending_htlcs(self, channel_id: str) -> List[Dict[str, Any]]:
    """Get pending HTLCs for a channel from listpeerchannels."""
    try:
        peers = self.plugin.rpc.listpeerchannels()
        for channel in peers.get("channels", []):
            scid = channel.get("short_channel_id", "").replace(":", "x")
            if scid == channel_id.replace(":", "x"):
                return channel.get("htlcs", [])
    except Exception:
        pass
    return []
```

### 4.5 Integration into Fee Adjustment

**Location:** `modules/fee_controller.py` → `_adjust_channel_fee()` (after floor calculation, before Hill Climbing)

```python
def _adjust_channel_fee(self, channel_id: str, peer_id: str, state: Dict, 
                        channel_info: Dict, chain_costs: Optional[Dict] = None) -> Optional[FeeAdjustment]:
    # ... existing code up to floor calculation ...
    
    floor_ppm = self._calculate_floor(capacity, chain_costs=chain_costs, peer_id=peer_id)
    
    # Phase 7: Scarcity Pricing
    scarcity_multiplier = 1.0
    if self.config.enable_scarcity_pricing:
        scarcity_state = self._get_scarcity_state(channel_id)
        htlcs = self._get_pending_htlcs(channel_id)
        utilization = scarcity_state.update(htlcs)
        scarcity_multiplier = scarcity_state.get_multiplier()
        
        if scarcity_multiplier > 1.05:
            self.plugin.log(
                f"SCARCITY PRICING: {channel_id[:12]}... "
                f"utilization={utilization:.1%}, multiplier={scarcity_multiplier:.2f}x",
                level='debug'
            )
    
    # ... existing Alpha Sequence logic ...
    
    # Apply scarcity multiplier to final fee (before clamping)
    if scarcity_multiplier > 1.0 and new_fee_ppm > floor_ppm:
        new_fee_ppm = int(new_fee_ppm * scarcity_multiplier)
    
    # ... continue with fee clamping and broadcast ...
```

### 4.6 Rebalancer Scarcity Guard

**Location:** `modules/rebalancer.py` (new method)

```python
def _check_scarcity_safe(self, channel_id: str, rebalance_amount_sats: int) -> bool:
    """
    Check if rebalance would trigger scarcity pricing trap.
    
    HIGH-02 Defense: Prevents "Trap & Trap" deadlock where rebalancing
    into a channel triggers scarcity pricing, making the channel
    unprofitable.
    
    Args:
        channel_id: Target channel for rebalance
        rebalance_amount_sats: Amount being rebalanced
        
    Returns:
        True if safe to proceed, False if would exceed threshold
    """
    UTILIZATION_THRESHOLD = 0.6  # Block if would exceed 60%
    
    try:
        # Get current pending HTLCs
        htlcs = self._get_pending_htlcs(channel_id)
        
        # Simulate adding rebalance as pending HTLC
        simulated_htlc = {"amount_msat": rebalance_amount_sats * 1000}
        simulated_htlcs = htlcs + [simulated_htlc]
        
        # Calculate simulated utilization (value-weighted)
        weighted_slots = 0.0
        for htlc in simulated_htlcs:
            amount_msat = htlc.get("amount_msat", 0)
            if isinstance(amount_msat, str):
                amount_msat = int(amount_msat.replace("msat", ""))
            slot_weight = min(10.0, max(0.01, amount_msat / 100_000_000))
            weighted_slots += slot_weight
        
        simulated_utilization = weighted_slots / 483
        
        if simulated_utilization > UTILIZATION_THRESHOLD:
            self.plugin.log(
                f"SCARCITY GUARD: Rebalance to {channel_id[:12]}... blocked. "
                f"Would push utilization to {simulated_utilization:.1%} "
                f"(threshold: {UTILIZATION_THRESHOLD:.0%})",
                level='warn'
            )
            return False
        
        return True
        
    except Exception as e:
        self.plugin.log(f"Scarcity check error: {e}", level='debug')
        return True  # Fail open if we can't check

def _get_pending_htlcs(self, channel_id: str) -> List[Dict[str, Any]]:
    """Get pending HTLCs for a channel."""
    try:
        peers = self.plugin.rpc.listpeerchannels()
        for channel in peers.get("channels", []):
            scid = channel.get("short_channel_id", "").replace(":", "x")
            if scid == channel_id.replace(":", "x"):
                return channel.get("htlcs", [])
    except Exception:
        pass
    return []
```

**Location:** `modules/rebalancer.py` → `execute_rebalance()` (add guard before sling job)

```python
def execute_rebalance(self, candidate: RebalanceCandidate, ...) -> Dict[str, Any]:
    # ... existing validation ...
    
    # Phase 7: Scarcity safety check
    if self.config.enable_scarcity_pricing:
        if not self._check_scarcity_safe(candidate.to_channel, candidate.amount_sats):
            return {
                "success": False,
                "reason": "scarcity_guard",
                "message": f"Rebalance blocked: would trigger scarcity pricing on {candidate.to_channel}"
            }
    
    # ... continue with sling job submission ...
```

---

## 5. Feature 4: Liquidity Efficiency Suite

### 5.1 Objective
Optimize how capital is deployed (Smart Allocation) and stop wasting resources on broken paths (Futility).

### 5.2 Files to Modify
| File | Changes |
|------|---------|
| `modules/rebalancer.py` | Update `_analyze_rebalance_ev` (Targets) and `find_rebalance_candidates` (Futility) |

### 5.3 Volume-Weighted Targets (Smart Allocation)

**Location:** `modules/rebalancer.py` → `_analyze_rebalance_ev`

**Algorithm:**
Instead of a blind 50% target, we calculate:
1.  **Velocity:** Average daily volume over the last 7 days.
2.  **Inventory Goal:** Enough liquidity for 3 days of flow.
3.  **Cap:** Never exceed 50% of channel capacity (don't overfill).
4.  **Floor:** Never drop below `rebalance_min_amount` (burst buffer).

```python
# Pseudo-code logic insertion
daily_vol = (state.sats_in + state.sats_out) / 7
vol_target = daily_vol * 3
cap_target = capacity * 0.5
target_spendable = max(min(cap_target, vol_target), config.rebalance_min_amount)
```

### 5.4 Futility Circuit Breaker

**Location:** `modules/rebalancer.py` → `find_rebalance_candidates`

**Logic:**
Check the existing `channel_failures` table.
- If `failure_count > 10` AND `last_failure < 48 hours ago`:
- **SKIP** candidate. Do not calculate EV. Do not query graph.

---

## 6. Configuration Additions

**Location:** `modules/config.py` → `Config` class

```python
@dataclass
class Config:
    # ... existing fields ...
    
    # Phase 7 additions (v1.3.0)
    enable_vegas_reflex: bool = True       # Mempool spike defense
    vegas_decay_rate: float = 0.85         # Per-cycle decay (~30min half-life)
    enable_scarcity_pricing: bool = True   # HTLC slot scarcity pricing
    scarcity_threshold: float = 0.35       # Start pricing at 35% utilization
    
    # Deferred (v1.4.0)
    enable_flow_asymmetry: bool = False    # Rare liquidity premium
    enable_peer_sync: bool = False         # Peer-level fee syncing
```

**Location:** `cl-revenue-ops.py` → Plugin options

```python
plugin.add_option(
    "revenue-vegas-reflex",
    True,
    "Enable Vegas Reflex mempool spike defense",
    opt_type="bool"
)
plugin.add_option(
    "revenue-vegas-decay",
    0.85,
    "Vegas Reflex decay rate per cycle (0.0-1.0)",
    opt_type="float"
)
plugin.add_option(
    "revenue-scarcity-pricing",
    True,
    "Enable HTLC slot scarcity pricing",
    opt_type="bool"
)
plugin.add_option(
    "revenue-scarcity-threshold",
    0.35,
    "Utilization threshold to start scarcity pricing (0.0-1.0)",
    opt_type="float"
)
```

---

## 7. Testing Plan

### 7.1 Unit Tests

| Test ID | Feature | Description |
|---------|---------|-------------|
| T1.1 | Config | Transactional update succeeds |
| T1.2 | Config | Immutable key rejection |
| T1.3 | Config | Type validation (int, float, bool) |
| T1.4 | Config | Range validation |
| T1.5 | Config | Ghost Config prevention (DB read-back) |
| T2.1 | Vegas | Intensity decay over cycles |
| T2.2 | Vegas | No permanent latch at 1.0 |
| T2.3 | Vegas | Probabilistic trigger at 200-400% |
| T2.4 | Vegas | Floor multiplier calculation |
| T3.1 | Scarcity | Value-weighted utilization |
| T3.2 | Scarcity | Dust Flood resistance (1000 tiny HTLCs ≠ 1000 slots) |
| T3.3 | Scarcity | Asymmetric EMA (fast up, slow down) |
| T3.4 | Scarcity | Rebalancer forecast blocks high-util |
| T4.1 | Targets | High volume channel targets 50% capacity |
| T4.2 | Targets | Low volume channel targets 3x daily volume |
| T4.3 | Targets | Dead channel respects min_amount floor |
| T4.4 | Futility | Channel with 11 fails is skipped |
| T4.5 | Futility | Channel with 11 fails (old) is retried |

### 7.2 Integration Tests

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| I1 | Set config via RPC, verify in-memory | Value matches |
| I2 | Set config, restart plugin, verify loaded | Persisted value loaded |
| I3 | Simulate mempool spike, verify floor increase | Floor multiplied |
| I4 | Simulate spike removal, verify decay | Intensity decreases each cycle |
| I5 | Fill channel with HTLCs, verify fee increase | Scarcity multiplier applied |
| I6 | Attempt rebalance to 70% utilized channel | Rebalance blocked |
| I7 | Rebalance dead channel 10x | 11th attempt is silenced (no log/activity) |
| I8 | Check large low-vol channel | Target liquidity drops, capital freed |

### 7.3 Adversarial Tests

| Test ID | Attack | Defense | Expected Result |
|---------|--------|---------|-----------------|
| A1 | Vegas Latch Bomb (rapid spikes) | Exponential decay | Intensity returns to 0 |
| A2 | Dust Flood (1000 tiny HTLCs) | Value-weighting | Utilization stays low |
| A3 | Config Race (update mid-cycle) | ConfigSnapshot | Cycle uses consistent values |
| A4 | Ghost Config (DB write fails) | Read-back verify | Error returned, memory unchanged |

---

## 8. Implementation Timeline

| Week | Feature | Tasks |
|------|---------|-------|
| 1 | Feature 1 | Database schema, Config methods, RPC commands |
| 2 | Feature 2 | VegasReflexState, mempool history, integration |
| 3 | Feature 3 | ScarcityState, rebalancer guard, integration |
| 4 | Feature 4 | Smart Targets, Futility Circuit Breaker |
| 5 | Testing | Unit tests, integration tests, adversarial tests |
| 6 | Release | Documentation, changelog, v1.3.0 release |

---

## 9. Rollback Plan

All features have enable flags that default to `True` but can be disabled:

```bash
# Disable Vegas Reflex
lightning-cli revenue-config set enable_vegas_reflex false

# Disable Scarcity Pricing
lightning-cli revenue-config set enable_scarcity_pricing false
```

Database schema changes are additive (new tables/columns) and do not break backward compatibility.

---

*Document Author: cl-revenue-ops Development Team*  
*Last Updated: January 3, 2026*
