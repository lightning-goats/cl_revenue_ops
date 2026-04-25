# Plan: Advanced Fee Optimization for cl-revenue-ops and cl-hive

## Overview

Enhance fee optimization with **Causal Inference** and **Game-Theoretic** reasoning to account for competitor reactions. Primary focus on modeling strategic interactions rather than treating the market as a stationary optimization problem.

**Key Insight:** Current Thompson Sampling assumes competitors don't react to our fee changes. In reality, competitors observe and respond strategically.

**Key Constraints:**
- Minimal dependencies (pure Python, standard library + existing SQLite)
- Backward compatible (Thompson Sampling preserved, enhancements toggled on/off)
- Interpretable models (linear regression, not black-box neural nets)
- Use existing hive fee intelligence infrastructure

---

## Approaches (Prioritized)

| Approach | Complexity | Value | Status |
|----------|-----------|-------|--------|
| **Causal Inference** | Medium | High - models competitor reactions | **Primary focus** |
| Reinforcement Learning | High | Medium - black-box optimization | Future consideration |
| Gaussian Processes | Medium | Medium - non-linear demand curves | Deferred |

---

## Architecture (Causal Inference)

```
cl-revenue-ops/modules/
├── causal_fee.py            # NEW: Causal inference components
│   ├── CompetitorReactionModel   # β coefficients, classification
│   ├── CausalFeeEstimator        # Counterfactual estimation
│   └── CausalThompsonState       # Strategic sampling extension
├── fee_controller.py        # Integrate causal models
├── database.py              # Add tracking tables
└── config.py                # Add causal config options

cl-hive/modules/
└── fee_intelligence.py      # Track competitor fee reactions
```

Integration:
```
HillClimbingFeeController
    ├── ThompsonAIMDState (existing - Bayesian exploration)
    └── CausalThompsonState (new - strategic adjustment)
         ├── CompetitorReactionModel per peer
         ├── Predicts competitor response to our fee
         └── Adjusts Thompson sample based on game theory
```

---

## Problem Statement

Current Thompson Sampling treats the market as **stationary** - it learns fee-demand curves assuming competitors don't react. In reality:

1. **Competitors observe our fees** (via gossip, probing)
2. **Competitors adjust their fees** in response
3. **Our optimal fee depends on their reaction**

This creates a **strategic game**, not just an optimization problem.

## Current System Gaps

| What We Have | What's Missing |
|--------------|----------------|
| Observe competitor fees (fee_intelligence.py) | Model how they'll **react** to our changes |
| Elasticity estimation (local demand curve) | **Counterfactual** reasoning ("what if we'd set X?") |
| Heuristic undercutting (market share < 20%) | **Best response** function estimation |
| Pheromone-based coordination (internal) | **Nash equilibrium** analysis (external) |

## Causal Inference Framework

### 1. Structural Causal Model (SCM)

```
Our Fee (F_us) ──────────────────────┐
      │                              │
      ▼                              ▼
Competitor Fee (F_them) ────► Market Volume (V)
      │                              │
      └──────────────────────────────┘
              (confounded by demand shocks)
```

**Key insight**: Observed correlation between `F_us` and `V` is confounded by:
- Demand shocks (holidays, market conditions) affect both fees and volume
- Competitor reactions create feedback loops

### 2. Intervention vs Observation

**Observational**: P(V | F_us = 200) - "What volume did we see when fee was 200?"
**Interventional**: P(V | do(F_us = 200)) - "What volume would we get if we SET fee to 200?"

The difference matters because:
- Observationally, low fees correlate with low demand (we lower fees when things are slow)
- Interventionally, low fees might **cause** high demand (attract more routing)

### 3. Competitor Reaction Function

Model competitor's best response:
```python
F_them(t+1) = β₀ + β₁·F_us(t) + β₂·F_them(t) + ε

Where:
- β₁ > 0: Competitor matches our fee changes (strategic complement)
- β₁ < 0: Competitor counters our fee changes (strategic substitute)
- β₁ ≈ 0: Competitor ignores us (independent pricing)
```

### 4. Causal Effect Estimation

**Average Treatment Effect (ATE)** of fee change:
```
ATE = E[V | do(F_us = high)] - E[V | do(F_us = low)]
```

**Methods to estimate:**
1. **Difference-in-Differences**: Compare before/after fee change, controlling for time trends
2. **Instrumental Variables**: Use exogenous shocks (on-chain fees, hive expansion) as instruments
3. **Propensity Score Matching**: Match fee changes with similar market conditions
4. **Regression Discontinuity**: If fee changes at balance thresholds, exploit discontinuity

## Game-Theoretic Fee Setting

### 1. Best Response Function

Given competitor's current fee `F_them`, our best response is:
```python
BR_us(F_them) = argmax_f  E[Revenue(f, F_them, V(f, F_them))]
```

### 2. Nash Equilibrium

Find (F_us*, F_them*) where neither wants to deviate:
```
F_us* = BR_us(F_them*)
F_them* = BR_them(F_us*)
```

### 3. Stackelberg Leadership

If we move first and competitor reacts:
```python
F_us* = argmax_f  E[Revenue(f, BR_them(f))]
```

This accounts for competitor's anticipated reaction.

## Implementation Design

### Data Requirements

```sql
-- Track fee changes with context for causal analysis
CREATE TABLE IF NOT EXISTS fee_change_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    peer_id TEXT NOT NULL,

    -- Our action
    our_old_fee INTEGER NOT NULL,
    our_new_fee INTEGER NOT NULL,

    -- Competitor state at change time
    their_fee_at_change INTEGER,
    their_fee_1h_later INTEGER,
    their_fee_24h_later INTEGER,

    -- Market context (for controlling confounds)
    mempool_fee_rate INTEGER,
    network_capacity_btc REAL,
    time_of_day INTEGER,
    day_of_week INTEGER,

    -- Outcomes
    volume_before_24h INTEGER,
    volume_after_24h INTEGER,
    revenue_before_24h INTEGER,
    revenue_after_24h INTEGER,

    timestamp INTEGER NOT NULL
);

-- Competitor reaction tracking
CREATE TABLE IF NOT EXISTS competitor_reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id TEXT NOT NULL,

    -- Our change that triggered reaction
    our_fee_change_id INTEGER REFERENCES fee_change_events(id),

    -- Their reaction
    their_old_fee INTEGER NOT NULL,
    their_new_fee INTEGER NOT NULL,
    reaction_delay_hours REAL,

    -- Classification
    reaction_type TEXT,  -- 'match', 'undercut', 'ignore', 'counter'

    timestamp INTEGER NOT NULL
);
```

### Causal Estimator Class

```python
@dataclass
class CompetitorReactionModel:
    """Estimates how competitors react to our fee changes."""

    peer_id: str

    # Reaction function coefficients: F_them(t+1) = β₀ + β₁·F_us(t) + β₂·F_them(t)
    beta_0: float = 0.0        # Intercept
    beta_1: float = 0.0        # Response to our fee
    beta_2: float = 0.95       # Persistence of their own fee

    # Classification
    reaction_type: str = "unknown"  # 'matcher', 'undercutter', 'ignorer', 'leader'
    reaction_delay_hours: float = 6.0
    confidence: float = 0.0

    # Sample size
    observations: int = 0

    def predict_their_fee(self, our_fee: int, their_current_fee: int) -> int:
        """Predict competitor's fee after we set ours."""
        predicted = self.beta_0 + self.beta_1 * our_fee + self.beta_2 * their_current_fee
        return max(1, int(predicted))

    def update_from_observation(self, our_fee: int, their_before: int, their_after: int):
        """Update model from observed reaction."""
        # Online linear regression update
        ...


class CausalFeeEstimator:
    """Estimates causal effect of fee changes accounting for competitor reactions."""

    def __init__(self, database: Database):
        self.db = database
        self.reaction_models: Dict[str, CompetitorReactionModel] = {}

    def estimate_causal_effect(
        self,
        channel_id: str,
        proposed_fee: int,
        current_fee: int
    ) -> Dict[str, Any]:
        """
        Estimate causal effect of fee change, accounting for:
        1. Direct effect on demand
        2. Competitor reaction
        3. Equilibrium volume
        """
        peer_id = self._get_peer_id(channel_id)
        reaction_model = self.reaction_models.get(peer_id)

        # Predict competitor reaction
        if reaction_model and reaction_model.confidence > 0.5:
            their_current = self._get_competitor_fee(peer_id)
            their_predicted = reaction_model.predict_their_fee(proposed_fee, their_current)
        else:
            their_predicted = None  # Unknown reaction

        # Estimate demand under intervention
        # Using difference-in-differences on historical data
        ate = self._estimate_ate(channel_id, proposed_fee, their_predicted)

        return {
            "proposed_fee": proposed_fee,
            "predicted_competitor_fee": their_predicted,
            "estimated_volume_change": ate["volume_change"],
            "estimated_revenue_change": ate["revenue_change"],
            "confidence": ate["confidence"],
            "is_nash_equilibrium": self._check_nash(proposed_fee, their_predicted),
        }

    def find_stackelberg_optimal(
        self,
        channel_id: str,
        fee_range: Tuple[int, int]
    ) -> int:
        """
        Find optimal fee assuming Stackelberg leadership.
        We move first, competitor reacts optimally.
        """
        best_fee = None
        best_revenue = -float('inf')

        for fee in range(fee_range[0], fee_range[1], 10):
            effect = self.estimate_causal_effect(channel_id, fee, current_fee)
            expected_revenue = effect["estimated_revenue_change"]

            if expected_revenue > best_revenue:
                best_revenue = expected_revenue
                best_fee = fee

        return best_fee
```

### Integration with Thompson Sampling

```python
class CausalThompsonState(GaussianThompsonState):
    """Thompson Sampling enhanced with causal competitor modeling."""

    def __init__(self):
        super().__init__()
        self.causal_estimator: Optional[CausalFeeEstimator] = None
        self.competitor_model: Optional[CompetitorReactionModel] = None

    def sample_fee_strategic(self, floor: int, ceiling: int) -> int:
        """
        Sample fee accounting for competitor reactions.

        1. Thompson samples candidate fee
        2. Predict competitor reaction
        3. Estimate equilibrium outcome
        4. Adjust if strategic consideration warrants
        """
        # Standard Thompson sample
        thompson_fee = self.sample_fee(floor, ceiling)

        # If no competitor model, use Thompson directly
        if not self.competitor_model or self.competitor_model.confidence < 0.3:
            return thompson_fee

        # Predict competitor reaction to Thompson's choice
        their_current = self.fleet_avg_fee or thompson_fee
        their_predicted = self.competitor_model.predict_their_fee(thompson_fee, their_current)

        # Check if we should adjust based on game theory
        reaction_type = self.competitor_model.reaction_type

        if reaction_type == "matcher":
            # They'll match us - no advantage to undercutting
            # Raise fee slightly to increase margin
            return min(ceiling, int(thompson_fee * 1.05))

        elif reaction_type == "undercutter":
            # They'll undercut - don't start price war
            # Stay at current level or slightly above
            return thompson_fee

        elif reaction_type == "ignorer":
            # They ignore us - optimize freely
            return thompson_fee

        elif reaction_type == "leader":
            # They set prices, we follow
            # Stay competitive with their fee
            return min(ceiling, max(floor, int(their_current * 0.95)))

        return thompson_fee
```

## Competitor Classification

```python
def classify_competitor(reaction_model: CompetitorReactionModel) -> str:
    """
    Classify competitor based on reaction function.

    β₁ = response coefficient to our fee changes
    """
    beta_1 = reaction_model.beta_1

    if reaction_model.observations < 5:
        return "unknown"

    if abs(beta_1) < 0.1:
        return "ignorer"      # Doesn't react to us
    elif beta_1 > 0.5:
        return "matcher"      # Follows our fee up/down
    elif beta_1 < -0.3:
        return "undercutter"  # Counters our moves
    elif beta_1 > 0.1:
        return "follower"     # Weakly follows
    else:
        return "leader"       # We should follow them
```

## When to Use Causal vs Standard Thompson

```python
def should_use_causal_model(peer_id: str, reaction_model: CompetitorReactionModel) -> bool:
    """Decide whether causal model adds value over standard Thompson."""

    # Need enough observations
    if reaction_model.observations < 10:
        return False

    # Need confident reaction estimate
    if reaction_model.confidence < 0.5:
        return False

    # Only valuable if competitor actually reacts
    if reaction_model.reaction_type == "ignorer":
        return False  # Standard Thompson is fine

    # Valuable for matchers/undercutters where strategy matters
    return reaction_model.reaction_type in ("matcher", "undercutter", "leader")
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `cl-revenue-ops/modules/database.py` | Add `fee_change_events` and `competitor_reactions` tables |
| `cl-revenue-ops/modules/causal_fee.py` | New - CompetitorReactionModel, CausalFeeEstimator |
| `cl-revenue-ops/modules/fee_controller.py` | Integrate CausalThompsonState |
| `cl-revenue-ops/modules/config.py` | Add causal inference config options |
| `cl-hive/modules/fee_intelligence.py` | Add competitor reaction tracking |

---

## Phased Implementation

### Phase 1: Data Collection
- Add `fee_change_events` table to track our fee changes with context
- Add `competitor_reactions` table to track how peers respond
- Instrument fee changes to record before/after competitor fees

### Phase 2: Reaction Model
- Implement `CompetitorReactionModel` dataclass
- Online linear regression to estimate β coefficients
- Classify competitors (matcher, undercutter, ignorer, leader)

### Phase 3: Thompson Integration
- Implement `CausalThompsonState` extending `GaussianThompsonState`
- Add `sample_fee_strategic()` method
- Adjust sampling based on predicted competitor reaction

### Phase 4: Stackelberg Optimization (Optional)
- Implement `find_stackelberg_optimal()` for leadership pricing
- Nash equilibrium detection

---

## Verification

### Unit Tests
```bash
python3 -m pytest tests/test_causal_fee.py -v
```

### Test Cases
1. `test_reaction_model_update` - verify β coefficients update correctly
2. `test_competitor_classification` - verify matcher/undercutter/ignorer detection
3. `test_strategic_sampling` - verify fee adjustment for different competitor types
4. `test_causal_effect_estimation` - verify counterfactual calculations

### Integration Verification
1. Enable causal mode: `enable_causal_fee_optimization: true`
2. Monitor competitor classifications in logs
3. Check reaction model confidence: `SELECT * FROM competitor_reactions`
4. Verify strategic adjustments: `[FeeController] peer=02abc... type=matcher adjusted=+5%`

---

## Future Considerations

**Reinforcement Learning** may be added later for:
- Multi-channel joint optimization
- Non-linear dynamics causal model can't capture
- When competitor behavior is too complex for linear models

**Gaussian Processes** may be added for:
- Non-linear fee-demand curve estimation
- When demand elasticity varies non-linearly with fee level
