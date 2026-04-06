# Capital Efficiency & Smart Allocation Design

## Problem Statement

The fleet has 45 channels with 201M sats deployed. 27 channels (86.6M sats, 43% of capital) have zero forwards over 30 days. The top 3 channels handle 75.6% of volume. The capacity planner cannot detect dead capital, has no revenue-per-sat metric, and the channel open strategy doesn't follow proven demand corridors effectively.

Three problems to solve:
1. Dead capital detection — zero-flow channels are invisible to the planner
2. Capital efficiency ranking — no metric exists to compare channels by capital productivity
3. Smarter channel open strategy — follow demand corridors by examining high-performing peers' neighbors with multi-signal scoring

## Architecture

New module `modules/capital_efficiency.py` computes per-channel efficiency metrics once per planner cycle. Pure calculation layer — no RPC, no database writes. Consumed by the capex engine (budget multiplier), planner losers (dead capital detection), and planner opens (patron pool selection).

---

## 1. Capital Efficiency Analyzer (`modules/capital_efficiency.py`)

### Inputs (injected)
- Profitability analyzer cache (`analyze_all_channels()`)
- Flow analyzer data (per-channel flow state with `sats_in`, `sats_out`, `forward_count`, `daily_volume`)
- Channel capacity from profitability data

### Core Metric — Revenue Per Sat Deployed (RPSD)

```
rpsd = (fees_earned_30d_sats / capacity_sats) * 1_000_000
```

Ppm-scale number: for every sat of capital deployed, how many microsats of revenue it earned over 30 days. A 10M sat channel earning 500 sats/month has RPSD = 50. A 500k sat channel earning 100 sats/month has RPSD = 200 — 4x more efficient.

### Per-Channel Output — `ChannelEfficiency` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `channel_id` | `str` | Channel short ID |
| `rpsd` | `float` | Revenue per sat deployed (ppm-scale) |
| `efficiency_rank` | `float` | 0.0 to 1.0 percentile within fleet |
| `forward_velocity` | `float` | Forwards per day (demand signal) |
| `is_dead_capital` | `bool` | Zero forwards + past grace period |
| `dead_capital_stage` | `str` | "none" / "fee_reduction" / "defibrillation" / "close" |

### Fleet-Level Output — `FleetEfficiency` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `channel_efficiencies` | `Dict[str, ChannelEfficiency]` | Per-channel metrics |
| `median_rpsd` | `float` | Fleet median RPSD |
| `dead_capital_count` | `int` | Number of dead capital channels |
| `dead_capital_sats` | `int` | Total sats in dead capital |
| `total_deployed_sats` | `int` | Total fleet capital |

### Dead Capital Classification

A channel is dead capital if ALL of:
- `forward_count == 0` over the flow analysis window
- `days_open > capex_grace_days` (config value, currently 14 days)
- Channel peer is not a hive member (checked via hive_hints if available)

Stage progression is persisted in the database (see Section 2).

---

## 2. Dead Capital Graduated Response

### Integration Point

Capacity planner's `_identify_losers` method. Dead capital check runs BEFORE the existing Kalman confidence gate so that zero-flow channels without reliable flow_metrics are still detected.

### Stage Pipeline

| Stage | Loser Action | What Happens | Advance Condition |
|-------|-------------|--------------|-------------------|
| `fee_reduction` | `FEE_REDUCE` | Drop fees to floor (10 ppm) for one cycle | No forwards AND `current_time - entered_at > 24 hours` |
| `defibrillation` | `DEFIBRILLATE` | Diagnostic rebalance to test path viability | No forwards AND `current_time - entered_at > 24 hours` |
| `close` | `CLOSE` | Close and redeploy capital | Terminal stage |

Stage advancement uses the `entered_at` timestamp. Each stage lasts at least 24 hours to give the intervention time to attract traffic. Remote-opened channels get 48 hours per stage (double the normal interval) since they represent free capital and deserve more patience.

### Stage Persistence

New database table:

```sql
CREATE TABLE IF NOT EXISTS dead_capital_stage (
    channel_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL DEFAULT 'fee_reduction',
    entered_at INTEGER NOT NULL
)
```

The efficiency analyzer reads this table to populate `dead_capital_stage` on each `ChannelEfficiency`. The planner advances stages when a channel remains dead capital across cycles.

Channels that gain any forward activity are removed from the table (recovered).

### Protections

- Hive member channels are never classified as dead capital
- Remote-opened channels get an extra cycle at each stage before advancing (free capital — slower to abandon)
- Channels younger than `capex_grace_days` are excluded
- Existing loser protections (route-pair, inbound gateway, sourced fee contribution) still apply at the `close` stage — a dead capital channel on a revenue route won't be closed

---

## 3. Enhanced Neighbor Open Strategy

### 3a. Wider Patron Pool

Replace "top 3 by marginal ROI" with a combined, deduplicated pool (capped at 10):

| Source | Count | Signal |
|--------|-------|--------|
| Top by RPSD | 5 | Capital efficiency — channels working hardest per sat |
| Top by forward volume | 5 | Demand corridors — where traffic actually flows |
| Top by marginal ROI | 3 | Profitability (existing signal) |

A channel routing 7.7M sats at thin margins becomes a patron because it reveals a high-demand corridor, even if its ROI is moderate.

### 3b. Multi-Signal Neighbor Scoring

For each patron's neighbors (via `listchannels(source=patron_peer_id)`):

```
base_score = patron_efficiency_rank    # 0.0-1.0, from RPSD percentile

# Well-connected = better routing partner
channel_count_bonus = min(channel_count / 20, 1.0) * 0.3

# Many large channels = routing hub, not leaf node
# Median channel size > 3M sats signals serious routing node
median_size_bonus = min(median_channel_sats / 5_000_000, 1.0) * 0.4

# Low fees = volume-optimized
fee_bonus = 0.2 if avg_fee_ppm < 200
           else 0.1 if avg_fee_ppm < 500
           else 0.0

# Connects to multiple of our patrons = routing crossroads
crossroads_bonus = 0.15 * patron_connection_count

score = base_score + channel_count_bonus + median_size_bonus + fee_bonus + crossroads_bonus
```

**Channel size distribution** is the strongest signal (0.4 weight) per the design decision that a node with 20 channels averaging 5M+ sats is a much better routing partner than a node with 2 channels of 50M sats.

### 3c. 2nd-Degree Exploration

For the top 3 scored 1st-degree neighbors, also examine their neighbors:
- 2nd-degree candidates get a 0.5x score dampening factor
- Capped at 3 candidates per 1st-degree neighbor
- Same multi-signal scoring formula applies

### RPC Budget

- 10 patrons x 1 `listchannels` call = 10 calls
- 3 second-degree explorations x 1 call = 3 calls
- Total: 13 calls worst case, acceptable for a 15-minute planner cycle

### Strategy Weight

Increase from 0.7 to 0.9. Neighbor discovery grounded in efficiency data is a stronger signal than the current ROI-only version.

### Existing filters preserved

- Skip peers we already have channels with
- Skip neighbors with fees > 1500 ppm
- Skip neighbors with channels < 200k sats capacity

---

## 4. Capex Engine Efficiency Multiplier

### Current Budget Formula

```
proven:    budget = contribution * reinvestment_rate - capex_spent
active:    budget = max(proven, bootstrap - capex_spent)
bootstrap: budget = bootstrap_bps * capacity - capex_spent

budget_msat = raw_budget * discount * hive_mult
```

### Enhanced Formula

```
budget_msat = raw_budget * discount * hive_mult * efficiency_mult
```

### Efficiency Multiplier Calculation

| Condition | Formula | Example |
|-----------|---------|---------|
| RPSD >= median | `1.0 + min(0.5, (rpsd / median - 1.0) * 0.25)` | 2x median → 1.25x budget; 4x median → 1.5x (cap) |
| RPSD < median | `max(0.5, rpsd / median)` | Half median → 0.5x budget (floor) |
| Dead capital | `0.0` | Zero budget — no rebalancing for zero-flow channels |

**Effect on current fleet**:
- Top 3 channels (75% of volume): ~1.3-1.5x their current budgets
- 27 zero-flow channels: 0x — capex engine stops wasting budget
- Long tail of low-flow channels: proportional reduction (0.5x-1.0x)

### Injection

`CapexBudgetEngine.__init__` gains an optional `capital_efficiency` parameter (same injection pattern as `hive_hints`). When available, `_compute_channel_budget` applies the multiplier. When unavailable, `efficiency_mult = 1.0` (no change).

---

## 5. Integration and Data Flow

### Cycle Sequence

```
1. Profitability analyzer runs (existing)
2. Flow analyzer runs (existing)
3. Capital efficiency analyzer computes metrics (NEW)
   Input: profitability cache + flow data
   Output: FleetEfficiency with per-channel RPSD, rankings, dead capital flags
4. Capex engine receives efficiency data (MODIFIED)
   compute_allocations() applies efficiency multiplier to budgets
5. Planner _identify_winners (existing, unchanged)
6. Planner _identify_losers (MODIFIED)
   Dead capital check runs FIRST, before Kalman confidence gate
   Existing loser logic unchanged for non-dead-capital channels
7. Planner open strategies (MODIFIED)
   _discover_from_neighbors uses efficiency-ranked patron pool
   Multi-signal scoring with 2nd-degree exploration
   Other 5 strategies unchanged
```

### Dependency Injection (`cl-revenue-ops.py`)

```python
capital_efficiency = CapitalEfficiencyAnalyzer(profitability_analyzer, flow_analyzer, database, hive_hints)
capex_engine = CapexBudgetEngine(profitability_analyzer, database, config, hive_hints, capital_efficiency)
capacity_planner.set_capital_efficiency(capital_efficiency)
```

### Files

**New**:
- `modules/capital_efficiency.py` (~150 lines) — analyzer module
- `tests/test_capital_efficiency.py` — unit tests

**Modified**:
- `modules/capex_budget.py` — add efficiency_multiplier to budget formula
- `modules/capacity_planner.py` — dead capital in losers, enhanced neighbor strategy
- `modules/database.py` — dead_capital_stage table (CREATE, read, upsert, delete)
- `cl-revenue-ops.py` — construct and inject CapitalEfficiencyAnalyzer

**Unchanged**:
- `modules/rebalancer.py`
- `modules/rebalance_executor.py`
- `modules/fee_controller.py`
- `modules/hive_router.py`
- `modules/hive_hints.py`
- 5 of 6 open strategies (winner, route_pair, hive, demand_flow, graph)
