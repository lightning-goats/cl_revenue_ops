# Plan: Reinforcement Learning (DQN/PPO) for cl-revenue-ops and cl-hive

## Overview

Add Reinforcement Learning capabilities to learn exploration policies from experience, focusing on **Fee Controller** as the primary candidate due to clear reward signals and high impact.

**Key Constraints:**
- Minimal dependencies (pure Python, numpy only if needed)
- Backward compatible (Thompson Sampling preserved, RL toggled on/off)
- Safety-first (action bounds, exploration limits, fallback policies)
- Use existing SQLite infrastructure for experience replay

---

## Target Components (Prioritized)

| Component | Plugin | Current Method | RL Enhancement | Reward Signal |
|-----------|--------|---------------|----------------|---------------|
| Fee Controller | cl-revenue-ops | Thompson + AIMD | DQN/SAC learns optimal fee | Revenue/hour |
| Rebalancer | cl-revenue-ops | EV calculation | DQN learns when to rebalance | Profit/loss per rebalance |
| Planner | cl-hive | Weighted scoring | Value estimation for targets | Channel success (sparse) |

---

## Architecture

```
cl-revenue-ops/modules/
├── rl/
│   ├── __init__.py
│   ├── base.py              # Abstract RLAgent, ReplayBuffer
│   ├── networks.py          # Pure Python neural networks
│   ├── dqn_agent.py         # DQN implementation
│   ├── experience_buffer.py # SQLite-backed replay
│   ├── state_encoder.py     # Feature engineering
│   └── reward_calculator.py # Reward shaping
├── fee_controller.py        # Add RLFeeOptimizer
└── rebalancer.py            # Add RLRebalanceAdvisor
```

Integration:
```
HillClimbingFeeController
    ├── ThompsonAIMDState (existing - Bayesian)
    └── RLFeeOptimizer (new - toggleable)
         ├── Shares observations with Thompson
         └── SQLite experience replay
```

---

## Fee Controller RL Design

### State Space (~20 dimensions)

```python
@dataclass
class FeeRLState:
    # Channel state (normalized 0-1)
    outbound_ratio: float          # local_balance / capacity
    inbound_ratio: float           # remote_balance / capacity
    htlc_utilization: float

    # Flow metrics (from Kalman filter)
    kalman_velocity: float
    kalman_uncertainty: float
    flow_ratio: float

    # Revenue signals
    revenue_rate_ema: float        # Revenue/hour (normalized)
    forward_count_recent: float
    turnover_rate: float

    # Market context
    elasticity_estimate: float
    peer_avg_fee: float

    # Temporal (cyclical encoding)
    hour_of_day_sin: float
    hour_of_day_cos: float
    day_of_week_sin: float
    day_of_week_cos: float

    # Hive context
    pheromone_level: float
    is_primary_corridor: float
    fleet_threat_active: float

    # Policy state
    current_fee_ppm_normalized: float
    time_at_current_fee: float
```

### Action Space (Discretized)

```python
FEE_BUCKETS = [10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
```

### Reward Function

```python
def calculate_reward(revenue_delta, observation_hours, forward_count, routing_failures=0):
    # Primary: revenue per hour (normalized to cap at 1000 sat/hr)
    revenue_rate = revenue_delta / max(observation_hours, 1.0)
    normalized_revenue = min(1.0, revenue_rate / 1000.0)

    # Activity bonus
    activity_bonus = 0.1 * min(1.0, forward_count / 10.0)

    # Failure penalty
    failure_penalty = 0.05 * min(1.0, routing_failures / 5.0)

    return normalized_revenue + activity_bonus - failure_penalty
```

---

## Database Schema

```sql
-- Experience replay for fee controller
CREATE TABLE IF NOT EXISTS rl_fee_experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    action_fee_ppm INTEGER NOT NULL,
    action_type TEXT NOT NULL,  -- 'thompson', 'rl', 'manual'
    reward REAL NOT NULL,
    next_state_json TEXT,
    is_terminal INTEGER DEFAULT 0,
    timestamp INTEGER NOT NULL,
    observation_hours REAL NOT NULL,
    forward_count INTEGER DEFAULT 0,
    revenue_sats INTEGER DEFAULT 0,
    td_error REAL DEFAULT 0.0,
    priority REAL DEFAULT 1.0
);

CREATE INDEX idx_rl_fee_exp_channel ON rl_fee_experience(channel_id, timestamp);
CREATE INDEX idx_rl_fee_exp_priority ON rl_fee_experience(priority DESC);

-- Experience replay for rebalancer
CREATE TABLE IF NOT EXISTS rl_rebalance_experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dest_channel_id TEXT NOT NULL,
    source_channel_ids TEXT NOT NULL,
    state_json TEXT NOT NULL,
    action_amount_bucket INTEGER NOT NULL,
    action_execute INTEGER NOT NULL,
    reward REAL NOT NULL,
    next_state_json TEXT,
    success INTEGER,
    actual_cost_sats INTEGER,
    amount_transferred INTEGER,
    timestamp INTEGER NOT NULL,
    priority REAL DEFAULT 1.0
);

-- Model checkpoints
CREATE TABLE IF NOT EXISTS rl_model_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    optimizer_state_json TEXT,
    episode_count INTEGER,
    total_reward REAL,
    timestamp INTEGER NOT NULL
);
```

---

## Safety Mechanisms

### 1. Action Clipping

```python
class SafeRLWrapper:
    def __init__(self, min_fee=10, max_fee=5000, max_change_pct=0.5):
        ...

    def apply_action(self, rl_fee, current_fee, thompson_suggestion):
        # 1. Absolute bounds
        bounded = max(self.min_fee, min(self.max_fee, rl_fee))

        # 2. Rate limiting (max 50% change per step)
        max_delta = int(current_fee * self.max_change_pct)
        rate_limited = clamp(bounded, current_fee - max_delta, current_fee + max_delta)

        # 3. Sanity check against Thompson (stay within 30% band)
        if abs(rate_limited - thompson_suggestion) > thompson_suggestion * 0.3:
            rate_limited = int(0.7 * thompson_suggestion + 0.3 * rate_limited)

        return rate_limited
```

### 2. Warmup and Fallback

```python
MIN_EXPERIENCES_BEFORE_TRAINING = 100
MIN_CHANNEL_AGE_DAYS = 7
THOMPSON_BASELINE_PROB = 0.2  # 20% always use Thompson

def select_fee_with_fallback(channel_id, rl_agent, thompson_state, config):
    # Fallback to Thompson if:
    # 1. RL not enabled
    # 2. Not enough training data
    # 3. Random baseline check (20% of time)

    if not config.enable_rl_fee_optimization:
        return thompson_fee, "thompson"
    if rl_agent.total_experiences < 100:
        return thompson_fee, "thompson_warmup"
    if random.random() < 0.2:
        return thompson_fee, "thompson_baseline"

    # Use RL with safety wrapper
    rl_fee = rl_agent.select_action(state)
    return SafeRLWrapper().apply_action(rl_fee, current_fee, thompson_fee), "rl"
```

---

## Configuration

```python
@dataclass
class Config:
    # RL Master Switch (off by default)
    enable_rl_fee_optimization: bool = False
    enable_rl_rebalance_advisor: bool = False

    # Training Parameters
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.99
    rl_epsilon_start: float = 1.0
    rl_epsilon_end: float = 0.05
    rl_epsilon_decay: float = 0.995
    rl_batch_size: int = 32
    rl_buffer_size: int = 10000
    rl_update_frequency: int = 4
    rl_target_update_frequency: int = 100

    # Safety Parameters
    rl_max_change_pct: float = 0.5
    rl_thompson_blend: float = 0.3
    rl_warmup_experiences: int = 100
    rl_fallback_prob: float = 0.2

    # Persistence
    rl_checkpoint_interval: int = 100
    rl_load_checkpoint: bool = True
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `cl-revenue-ops/modules/rl/__init__.py` | New - package init |
| `cl-revenue-ops/modules/rl/base.py` | New - abstract classes |
| `cl-revenue-ops/modules/rl/networks.py` | New - pure Python neural nets |
| `cl-revenue-ops/modules/rl/dqn_agent.py` | New - DQN implementation |
| `cl-revenue-ops/modules/rl/experience_buffer.py` | New - SQLite replay buffer |
| `cl-revenue-ops/modules/rl/state_encoder.py` | New - feature engineering |
| `cl-revenue-ops/modules/rl/reward_calculator.py` | New - reward shaping |
| `cl-revenue-ops/modules/fee_controller.py` | Add RLFeeOptimizer integration |
| `cl-revenue-ops/modules/rebalancer.py` | Add RLRebalanceAdvisor integration |
| `cl-revenue-ops/modules/database.py` | Add RL tables and migrations |
| `cl-revenue-ops/modules/config.py` | Add RL config options |
| `cl-revenue-ops/cl-revenue-ops.py` | Wire up RL components |

---

## Phased Rollout

### Phase 1: Infrastructure (Week 1-2)
- Create `modules/rl/` directory structure
- Implement base classes and SQLite experience buffer
- Implement state encoder
- Add config options (disabled by default)
- Add database schema migrations

### Phase 2: DQN for Fee Controller (Week 3-4)
- Implement pure-Python DQN network (2-layer MLP)
- Implement `RLFeeOptimizer` class
- Integrate with `HillClimbingFeeController`
- Add safety wrapper
- Unit tests and simulation tests

### Phase 3: Offline Training (Week 5)
- Implement `train_from_history()` function
- Add CLI command `revenue-rl train`
- Validate on historical data

### Phase 4: Online Training (Week 6-7)
- Enable online learning with experience collection
- Implement checkpoint save/load
- Add monitoring metrics
- A/B comparison logging

### Phase 5: Rebalancer RL (Week 8-9)
- Implement `RLRebalanceAdvisor`
- Integrate with `EVRebalancer`
- Experience collection for rebalance decisions

### Phase 6: cl-hive Integration (Week 10+)
- Share learned policies across fleet (optional)
- Fleet-aggregated reward signals
- Planner value estimation

---

## Verification

### Unit Tests
```bash
# Test DQN components
python3 -m pytest tests/test_rl_agent.py -v

# Test experience buffer
python3 -m pytest tests/test_experience_buffer.py -v

# Test safety wrapper
python3 -m pytest tests/test_rl_safety.py -v
```

### Simulation Tests
```python
# Synthetic demand curve: volume = 1000 * exp(-fee/500)
# Agent should converge to optimal fee ~500 ppm
def test_fee_optimization_simulation():
    agent = RLFeeOptimizer()
    for episode in range(100):
        fee = FEE_BUCKETS[agent.select_action(state)]
        volume = demand_model(fee)
        reward = calculate_reward(volume * fee / 1_000_000, 1.0, volume)
        agent.update(...)

    final_fee = FEE_BUCKETS[agent.select_action(state, training=False)]
    assert 400 <= final_fee <= 600
```

### Integration Verification
1. Enable RL with `enable_rl_fee_optimization: true`
2. Monitor via MCP tool: `hive rl_status`
3. Compare Thompson vs RL fees in logs
4. Check experience buffer growth: `SELECT COUNT(*) FROM rl_fee_experience`
5. Verify checkpoint saves: `SELECT * FROM rl_model_checkpoints ORDER BY timestamp DESC LIMIT 1`

### A/B Comparison
```
# Logs should show:
[FeeController] channel=abc... thompson=250 rl=300 selected=rl revenue_rate=45.2
```

---

## Algorithm Details

### Why DQN First (not PPO)

1. **Simpler implementation**: No policy gradient computation, no advantage estimation
2. **Pure Python feasible**: Forward pass + experience replay only
3. **Sample efficient for small action space**: 14 fee buckets is tractable
4. **Off-policy**: Can learn from Thompson's historical decisions

### Network Architecture

```python
class DQNNetwork:
    """2-layer MLP for Q-value estimation."""
    def __init__(self, state_dim=20, hidden_dim=64, n_actions=14):
        # Layer 1: state_dim -> hidden_dim (ReLU)
        # Layer 2: hidden_dim -> hidden_dim (ReLU)
        # Output: hidden_dim -> n_actions (linear)

    def forward(self, state):
        """Returns Q-values for each action."""
        h1 = relu(state @ W1 + b1)
        h2 = relu(h1 @ W2 + b2)
        q_values = h2 @ W3 + b3
        return q_values
```

### Upgrade Path to SAC

If continuous fees needed later:
1. Implement simple autograd (or use numpy gradients)
2. Actor outputs mean/std of fee distribution
3. Critic estimates Q(s,a) for continuous a
4. Entropy bonus for exploration
