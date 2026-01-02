# Phase 7: "The 1% Node" Strategy Path

## Technical Specification — Version 1.3 (Red Team Hardened)

| Field | Value |
|-------|-------|
| **Date** | January 1, 2026 |
| **Target Version** | cl-revenue-ops v1.3.0 |
| **Status** | ✅ IMPLEMENTED — All 3 core features deployed |

---

## 1. Executive Summary

Phase 7 transitions from an "Optimizing Node" to a "Market Making Node." This specification has been hardened following adversarial review to address:

| Threat Class | Defense | Status |
|--------------|---------|--------|
| **L1 Variance** | Vegas Reflex | ✅ Implemented |
| **Resource Exhaustion** | Scarcity Pricing | ✅ Implemented |
| **Config Race Conditions** | Dynamic Config | ✅ Implemented |
| **Cross-Module Deadlock** | Rebalancer Integration | Deferred to v1.4 |

### Implementation Priority (Strict Order)

1. **Feature 1: Dynamic Runtime Configuration** — ✅ Complete
2. **Feature 2: Mempool Acceleration (Vegas Reflex)** — ✅ Complete
3. **Feature 3: Scarcity Pricing (Balance-Based)** — ✅ Complete

**Deferred to v1.4:** Flow Asymmetry, Peer-Level Syncing, HTLC Slot Value-Weighted Scarcity

---

## 2. Feature Specifications (Hardened)

---

### Feature 1: Dynamic Runtime Configuration (Safe Hot-Swap)

**Objective:** Allow operators to tune algorithmic thresholds via RPC without restarting the plugin.

#### Architecture

| Component | Implementation |
|-----------|----------------|
| **Database** | New table `config_overrides` (`key TEXT`, `value TEXT`, `version INTEGER`) |
| **Object Model** | `ConfigSnapshot` immutable object; global `config` refreshed atomically |
| **RPC Command** | `revenue-config set <key> <value>` and `revenue-config get` |

#### Hardened Logic Flow (Transactional)

```
┌─────────────────────────────────────────────────────────────┐
│  1. VALIDATE input (type-check, range-check)                │
│  2. WRITE to SQLite (config_overrides table)                │
│  3. READ-BACK from SQLite (verify persistence)              │
│  4. UPDATE in-memory ConfigSnapshot (atomic swap)           │
│  5. LOG audit entry                                         │
└─────────────────────────────────────────────────────────────┘
```

> **CRITICAL:** Step 3 (Read-Back) prevents "Ghost Config" — where memory believes a value is set but DB write failed silently.

#### ConfigSnapshot Pattern

All module cycles MUST capture a snapshot at cycle start:

```python
def run_cycle(self):
    cfg = config.snapshot()  # Immutable for this cycle
    # All logic uses cfg, never global config directly
```

#### Safety Guards

| Guard | Purpose |
|-------|---------|
| **Immutable Core** | `database_path`, `network` cannot change at runtime |
| **Version Monotonic** | Each config update increments version; stale snapshots detectable |
| **Transactional Write** | DB write verified before memory update |

---

### Feature 2: Mempool Acceleration ("Vegas Reflex")

**Objective:** Protect against arbitrageurs draining channels during high on-chain fee spikes.

**Integration:** `modules/fee_controller.py` → `_calculate_floor()`

#### Hardened Algorithm (Exponential Decay)

**REMOVED:** Fixed 4-hour binary latch (vulnerable to Latch Bomb DoS)

**REPLACED WITH:** Continuous exponential decay state

```python
class VegasReflexState:
    def __init__(self):
        self.intensity = 0.0  # Range: 0.0 to 1.0
        self.decay_rate = 0.85  # Per-cycle decay (30min half-life)
    
    def update(self, current_sat_vb: float, ma_sat_vb: float):
        spike_ratio = current_sat_vb / ma_sat_vb
        
        if spike_ratio >= 4.0:
            # Immediate trigger: set intensity to max
            self.intensity = 1.0
        elif spike_ratio >= 2.0:
            # Proportional boost
            boost = (spike_ratio - 2.0) / 2.0  # 0.0 to 1.0
            self.intensity = min(1.0, self.intensity + boost * 0.3)
        
        # Always decay toward zero
        self.intensity *= self.decay_rate
    
    def get_floor_multiplier(self) -> float:
        # Smooth curve from 1.0x to 3.0x based on intensity
        return 1.0 + (self.intensity ** 0.5) * 2.0
```

#### Fee Floor Calculation

```python
defensive_floor_ppm = (current_sat_vb * 150 * 0.05) / avg_htlc_size
floor_ppm = max(standard_floor, defensive_floor_ppm * vegas.get_floor_multiplier())
```

#### Safety Guards

| Guard | Purpose |
|-------|---------|
| **Exponential Decay** | No permanent latch; intensity naturally fades |
| **Confirmation Window** | 2 consecutive cycles OR >400% spike for immediate trigger |
| **Intensity Cap** | Maximum 1.0; prevents runaway multipliers |

---

### Feature 3: HTLC Slot Scarcity Pricing

**Objective:** Exponentially price capacity as HTLC slots fill up to prevent channel paralysis.

**Integration:** `modules/fee_controller.py` → `_adjust_channel_fee()`

#### Hardened Algorithm (Value-Weighted)

**REMOVED:** Simple `active_htlcs / max_htlcs` count (vulnerable to Dust Flood)

**REPLACED WITH:** Value-weighted utilization with asymmetric EMA

```python
class ScarcityState:
    def __init__(self, max_htlcs: int = 483):
        self.max_htlcs = max_htlcs
        self.utilization_ema = 0.0
        self.alpha_up = 0.4    # Fast response to increases
        self.alpha_down = 0.1  # Slow response to decreases
    
    def update(self, htlcs: List[HTLC]) -> float:
        # Value-weighted: 1M sat HTLC = 10 slots, 1K sat HTLC = 0.01 slots
        weighted_slots = sum(
            min(10.0, max(0.01, htlc.amount_msat / 100_000_000))
            for htlc in htlcs
        )
        raw_utilization = min(1.0, weighted_slots / self.max_htlcs)
        
        # Asymmetric EMA: fast up, slow down
        if raw_utilization > self.utilization_ema:
            alpha = self.alpha_up
        else:
            alpha = self.alpha_down
        
        self.utilization_ema = alpha * raw_utilization + (1 - alpha) * self.utilization_ema
        return self.utilization_ema
    
    def get_multiplier(self) -> float:
        if self.utilization_ema <= 0.35:
            return 1.0
        # Exponential curve: 1.0x at 35%, up to 3.0x at 100%
        return 1.0 + ((self.utilization_ema - 0.35) / 0.65) ** 2 * 2.0
```

#### Rebalancer Integration (Trap & Trap Prevention)

**CRITICAL:** Rebalancer MUST forecast post-rebalance utilization before executing:

```python
# In rebalancer.py, before executing rebalance:
def _check_scarcity_safe(self, channel_id: str, rebalance_amount: int) -> bool:
    """Ensure rebalance won't trigger scarcity pricing trap."""
    current_htlcs = self._get_pending_htlcs(channel_id)
    
    # Simulate adding rebalance HTLC
    simulated_htlcs = current_htlcs + [SimulatedHTLC(rebalance_amount)]
    simulated_utilization = scarcity_state.calculate_utilization(simulated_htlcs)
    
    if simulated_utilization > 0.6:  # Buffer threshold
        log.warning(f"Rebalance blocked: would push utilization to {simulated_utilization:.1%}")
        return False
    return True
```

#### Safety Guards

| Guard | Purpose |
|-------|---------|
| **Value Weighting** | 1000 dust HTLCs ≠ 1000 slots; prevents Dust Flood |
| **Asymmetric EMA** | Fast up (0.4), slow down (0.1); prevents premature relaxation |
| **Rebalancer Forecast** | Prevents "Trap & Trap" deadlock with scarcity pricing |
| **Hard Cap at 80%** | Existing CONGESTION logic takes over |

---

## 3. Deferred Features (v1.4)

The following features require additional data collection and are deferred:

### Feature 4: Flow Asymmetry (Rare Liquidity Premium)
- **Reason for Deferral:** Requires 30+ days of flow data to identify stable "one-way streets"
- **Dependency:** Scarcity Pricing metrics must be validated first

### Feature 5: Peer-Level Atomic Fee Syncing
- **Reason for Deferral:** High-01 "Anchor & Drain" attack vector requires floor-only architecture
- **Requirement:** Must implement as "raise floor to highest peer channel" not "sync to average"

---

## 4. Implementation Checklist

### Modified Files

| File | Changes |
|------|---------|
| `cl-revenue-ops.py` | Add `revenue-config` RPC command |
| `modules/config.py` | Add `ConfigSnapshot`, transactional update, version tracking |
| `modules/database.py` | Add `config_overrides` table with version column |
| `modules/fee_controller.py` | `VegasReflexState` class, `ScarcityState` class |
| `modules/rebalancer.py` | Add `_check_scarcity_safe()` pre-flight check |

### Configuration Schema Additions

```python
@dataclass
class Config:
    # ... existing ...
    
    # Phase 7 additions (v1.3)
    enable_vegas_reflex: bool = True
    vegas_decay_rate: float = 0.85
    enable_scarcity_pricing: bool = True
    scarcity_threshold: float = 0.35
    
    # Deferred (v1.4)
    enable_flow_asymmetry: bool = False
    enable_peer_sync: bool = False
```

---

## 5. Security Mitigations Summary

This specification addresses all **7 vulnerabilities** identified in the consolidated Red Team assessment.

### Critical Severity (3)

| ID | Vulnerability | Mitigation |
|----|---------------|------------|
| **CRITICAL-01** | Vegas Latch Bomb | Exponential decay state, no fixed latch |
| **CRITICAL-02** | Config Torn Read | ConfigSnapshot pattern, version monotonic |
| **CRITICAL-03** | Ghost Config | Transactional: Write → Read-Back → Update Memory |

### High Severity (3)

| ID | Vulnerability | Mitigation |
|----|---------------|------------|
| **HIGH-01** | Dust Flood | Value-weighted HTLC utilization |
| **HIGH-02** | Trap & Trap Deadlock | Rebalancer forecasts post-rebalance utilization |
| **HIGH-03** | Confirmation Front-Run | Probabilistic early trigger at >200% spike |

### Medium Severity (1)

| ID | Vulnerability | Mitigation |
|----|---------------|------------|
| **MEDIUM-01** | EMA Downward Lag | Asymmetric EMA: α_up=0.4, α_down=0.1 |

### Deferred Risks (v1.4)

| Feature | Risk | Action |
|---------|------|--------|
| Peer-Level Syncing | Anchor & Drain arbitrage | Deferred until floor-only architecture designed |
| Flow Asymmetry | False positive taxation | Deferred until traffic analysis improved |

---

## 6. Deployment Recommendation

### v1.3.0 — Immediate Release
- ✅ Feature 1: Dynamic Runtime Configuration (Transactional)
- ✅ Feature 2: Mempool Acceleration (Exponential Decay)
- ✅ Feature 3: HTLC Slot Scarcity Pricing (Value-Weighted + Rebalancer Integration)

### v1.4.0 — Deferred (Post-Validation)
- ⏳ Feature 4: Flow Asymmetry
- ⏳ Feature 5: Peer-Level Syncing (Floor-Only Architecture Required)

---

*Specification Author: Lightning Goats Team*  
*Red Team Review: PASSED — See [`PHASE7_RED_TEAM_REPORT.md`](../audits/PHASE7_RED_TEAM_REPORT.md) (Final Consolidated)*  
*Approval: Senior Red Team Lead*  
*Last Updated: January 1, 2026*
