# Capacity Planner Refactor Design

**Date:** 2026-03-19
**Scope:** Transform capacity planner from advisory report generator into automated channel lifecycle manager with peer discovery. Complete splice removal.

## Goals

1. Automate channel opens (to existing winners and discovered new peers) via `fundchannel`/`multifundchannel`
2. Automate channel closes (conservative, with drain-then-close lifecycle) via `close`
3. Remove all splice functionality (not well supported by other Lightning implementations)
4. Integrate existing underutilized data signals across all modules

## Architecture

The capacity planner becomes a **peer** of the rebalancer and boltz_manager — not a subordinate. It participates in the unified budget system via `reserve_spend()`, follows the same background loop pattern, and coordinates with the rebalancer through the exclusion system and a shared pending-operations interface.

### Execution Model

- Direct execution via CLN RPCs (`fundchannel`, `multifundchannel`, `close`)
- Timer-driven background loop (default 6 hours, configurable)
- Defaults to `enabled=False` and `dry_run=True` — requires explicit opt-in
- Conservative safety guards: 1 open + 1 close per cycle, wallet reserve, fee gate, cooldowns

### Module Structure

```
CapacityPlanner
├── generate_report()                    # Advisory report (existing, cleaned)
├── execute_cycle()                      # Timer-driven main loop
├── _identify_winners()                  # Existing, splice removed, data-enriched
├── _identify_losers()                   # Existing, splice removed, bleeder-aware
├── _discover_peers()                    # 3-strategy ensemble
│   ├── _discover_from_winners()         # Strategy 1: existing high-ROI peers
│   ├── _discover_from_neighbors()       # Strategy 2: neighbors of top earners
│   └── _discover_from_graph()           # Strategy 3: network centrality
├── _score_candidate()                   # Composite scoring with existing data
├── _size_channel()                      # ROI-proportional sizing
├── _calculate_open_ev()                 # EV-based open decision
├── _execute_open()                      # fundchannel/multifundchannel wrapper
├── _execute_close()                     # close RPC with drain lifecycle
├── _initiate_drain()                    # Phase 1: passive + source_only policy
├── _check_drain_complete()              # Monitor drain progress
├── _check_fee_gate()                    # Percentile-based fee hysteresis
├── _check_safety_guards()              # Reserve, cooldown, budget checks
├── _get_mempool_recommendation()        # Existing
├── _generate_recommendations()          # Existing, splice text removed
└── get_status()                         # Planner state for RPC queries
```

---

## Peer Discovery: 3-Strategy Ensemble

Inspired by CLBOSS's multi-strategy approach, but using our richer data.

### Strategy 1: Existing Winners

From current `_identify_winners()` — channels with marginal ROI > 20%, high turnover, strong flow. Open additional parallel channel to same peer for capital injection. Enriched with DTS posterior mean, sourced fee contribution, and Kalman velocity.

### Strategy 2: Fee-Weighted Neighbors (from CLBOSS ChannelFinderByEarnedFee)

Find our top-earning peers → propose their neighbors as candidates via `listchannels`. Rationale: neighbors of profitable peers are likely good routing targets. Filter by fee competitiveness and capacity.

### Strategy 3: Graph Centrality (improved CLBOSS ChannelFinderByPopularity)

`listnodes`/`listchannels` graph analysis:
- Score = peer_count * median_capacity * fee_competitiveness_factor
- Filter: no existing channel, reasonable fees, sufficient capacity
- IP-bin diversity reordering (from CLBOSS) to reduce correlated failure risk
- Minimum graph knowledge gate: 800+ known nodes before proposing

### Candidate Scoring

Composite score using existing data:

| Signal | Source | Weight |
|--------|--------|--------|
| Peer reputation | `get_peer_reputation()` | HIGH — Laplace-smoothed routing success rate |
| Peer uptime | `get_peer_uptime_percent()` | HIGH — require > 95% for opens |
| Closed channel P&L | `get_peer_closed_channel_profit_summary()` | HIGH — profit inheritance |
| Network centrality | `listnodes` peer count * capacity | MEDIUM |
| Fee competitiveness | `listchannels` fee analysis | MEDIUM |
| DTS posterior (existing peers) | `get_fee_strategy_state()` | MEDIUM — proven fee-earner signal |

### Candidate Pool

Persistent SQLite table (`planner_candidates`) with scored candidates. Hourly connect probes accumulate uptime evidence. Pool size: min 8, max 32 candidates (from CLBOSS).

---

## Channel Sizing: ROI-Proportional + Available Funds

```
available = onchain_balance - wallet_reserve
per_candidate_max = available * 0.5        # never more than half remaining
roi_weight = candidate.roi / sum(all_candidate.roi)
channel_size = clamp(available * roi_weight, min_channel, max_channel)
```

- `min_channel`: 500k sats (configurable, default from CLBOSS)
- `max_channel`: 16.7M sats (wumbo, configurable)
- For new peers (no ROI data): use route-capacity estimation as proxy
- If < 2 total channels: split funds to ensure diversity (from CLBOSS `at_least_2` rule)

---

## EV-Based Channel Open Decision

Extends the rebalancer's EV framework:

```
EV_open = expected_lifetime_revenue - on_chain_cost - expected_rebalance_costs

where:
  expected_lifetime_revenue = daily_revenue_estimate * channel_lifetime_days
  on_chain_cost = open_cost + close_cost       # from _get_dynamic_chain_costs()
  expected_rebalance_costs = avg_cost_ppm * expected_volume * lifetime
```

Data sources (all already collected):
- `daily_revenue_estimate`: From `get_peer_closed_channel_profit_summary()` for returning peers, DTS posterior mean for existing peers, graph-based estimate for new peers
- `on_chain_cost`: From `FeeController._get_dynamic_chain_costs()` (~140 vbytes * sat/vB open + ~200 vbytes * sat/vB close)
- `expected_rebalance_costs`: From `get_channel_rebalance_success_rate().avg_cost_ppm`
- `channel_lifetime_days`: Historical from `closed_channels` table

Only open when `EV_open > 0`.

---

## Close Decision Framework

### Enhanced Loser Identification

Layer existing classifications with new signals:

| Signal | Source | Effect |
|--------|--------|--------|
| Hard bleeder | `identify_bleeders_v2()` | Bypass defibrillation gate → direct close |
| Futility `no_route` | `get_failure_metadata()` | Structurally dead → strong close signal |
| Channel role | `ChannelProfitability.channel_role` | Protect INBOUND_GATEWAYs (higher bar) |
| Kalman regime change | `FlowMetrics.kalman_regime_change` | Defer close — situation may be improving |
| Kalman confidence | `FlowMetrics.confidence` | Don't close if confidence < 0.5 |
| Avg rebalance cost > fee | `avg_cost_ppm` vs `broadcast_fee_ppm` | Structurally unprofitable signal |
| Peer uptime | `get_peer_uptime_percent()` | < 80% uptime + poor ROI → strong close |

### Drain-Then-Close Lifecycle

```
Phase 1: DRAIN (72h time-limited policy)
  - policy_manager.set_policy(peer_id,
      strategy="passive",
      rebalance_mode="source_only",
      tags=["closing", "drain_phase"],
      expires_in_hours=72)
  - Rebalancer naturally drains the channel as a source
  - Monitor local balance decrease

Phase 2: CLOSE (after drain succeeds or 72h timeout)
  - Check fee timing via mempool MA ratio
  - Block if current_sat_vb / ma_24h > 2.0 (spike)
  - Prefer if ratio < 0.5 (dip)
  - Execute close RPC when conditions met

Phase 3: ARCHIVE
  - Existing _archive_closed_channel() handles P&L recording
  - record_closed_channel_history() persists to closed_channels table
```

### Policy Awareness

Never close:
- Peers with `strategy="static"` (manually pinned fee)
- Peers tagged `"protect"` or `"no_close"`
- Channels with `opener == 'remote'` unless deeply underwater (marginal_roi < -75%)

### Boltz Soft-Close Alternative

For channels where peer is valuable but oversized, or fees are high:
- `boltz_manager.loop_out(channel_id=closing_channel)` extracts capital via submarine swap
- Avoids both close AND re-open on-chain fees
- Considered before on-chain close when fee gate would block

---

## Safety Guards

| Guard | Default | Rationale |
|-------|---------|-----------|
| `planner_enabled` | `False` | Opt-in only |
| `planner_dry_run` | `True` | Must explicitly enable execution |
| Max opens per cycle | 1 | Prevent over-commitment |
| Max closes per cycle | 1 | Prevent fleet shrinkage |
| Min channel age for close | 30 days | Allow channels to mature |
| Min wallet reserve | Existing `min_wallet_reserve` config | Never touch reserve |
| Fee gate | `sat/vB < 50` with CLBOSS-style percentile hysteresis | Don't transact in high-fee environment |
| Fee gate override | If onchain > 25% of total funds | Don't let capital sit idle (from CLBOSS) |
| Defibrillation gate | 2 failed diagnostic rebalances (existing) | Prevent premature closure |
| Drain timeout | 72 hours | Don't hold channels in limbo forever |
| Cooldown | 24h between actions on same peer | Prevent thrashing |
| Min peer uptime | 95% for opens | Only open to reliable peers |
| Min peer reputation | 0.4 (Laplace-smoothed) | Only open to peers with routing track record |
| EV gate | `EV_open > 0` | Only open if expected profitable |

---

## Fee-Rate Hysteresis (from CLBOSS)

Rolling window of fee samples (2 weeks, via existing `record_mempool_fee()`):
- `lo_to_hi` threshold: 23rd percentile
- `hi_to_lo` threshold: 17th percentile
- Asymmetric transitions prevent oscillation
- Override: onchain > 25% of total funds → open anyway

---

## Budget Integration

Use the **generic spend ledger** (`database.reserve_spend(category="channel_open")`), NOT the rebalancer-specific budget reservation. Channel open/close costs automatically appear in the unified `revenue-report costs` dashboard via existing `get_opening_costs_since()` and `get_closure_costs_since()`.

---

## Rebalancer Coordination

1. Before closing: check `job_manager.has_active_job()`, stop active jobs, add sling exclusion
2. Expose `is_pending_close(channel_id)` for rebalancer to skip in candidate selection
3. After opening: rebalancer auto-detects via `_on_channel_state_changed_impl()`, `new_channel_grace_days` (7d) exempts from velocity gate

---

## Splice Removal Scope

Remove all splice references:
- `capacity_planner.py`: `_get_peer_splice_map()`, `peer_supports_splice` fields
- `cl-revenue-ops.py`: `_handle_splice_completion()` (~70 lines), `_get_splice_costs_from_bookkeeper()` (~90 lines), splice cost reporting, splice detection
- `database.py`: `splice_costs` table creation, `record_splice()`, splice query methods (~150 lines)
- `config.py`: `SPLICE_COST_SATS` constant
- Tests: splice-related test cases

Total: ~400 lines removed across 4 files.

---

## New Database Tables

### `planner_candidates`
```sql
CREATE TABLE IF NOT EXISTS planner_candidates (
    peer_id TEXT PRIMARY KEY,
    score REAL,
    source TEXT,              -- 'winner', 'neighbor', 'graph', 'manual'
    last_evaluated INTEGER,
    capacity_recommendation_sats INTEGER,
    connect_successes INTEGER DEFAULT 0,
    connect_failures INTEGER DEFAULT 0,
    metadata_json TEXT
);
```

### `planner_actions`
```sql
CREATE TABLE IF NOT EXISTS planner_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT,          -- 'open', 'close', 'drain', 'soft_close'
    peer_id TEXT,
    channel_id TEXT,
    amount_sats INTEGER,
    estimated_cost_sats INTEGER,
    actual_cost_sats INTEGER,
    status TEXT,               -- 'planned', 'draining', 'reserved', 'executing', 'completed', 'failed'
    created_at INTEGER,
    completed_at INTEGER,
    reason TEXT,
    metadata_json TEXT
);
```

---

## Config Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `revenue-ops-planner-enabled` | bool | false | Enable automated capacity planner |
| `revenue-ops-planner-interval` | int | 21600 | Seconds between evaluation cycles |
| `revenue-ops-planner-dry-run` | bool | true | Log decisions without executing |
| `revenue-ops-planner-max-opens` | int | 1 | Max channel opens per cycle |
| `revenue-ops-planner-max-closes` | int | 1 | Max channel closes per cycle |
| `revenue-ops-planner-min-channel-sats` | int | 500000 | Minimum channel size |
| `revenue-ops-planner-max-channel-sats` | int | 10000000 | Maximum channel size |
| `revenue-ops-planner-min-channel-age-days` | int | 30 | Min age before close eligible |
| `revenue-ops-planner-min-peer-uptime` | float | 95.0 | Min uptime % for open candidates |
| `revenue-ops-planner-max-fee-rate` | float | 50.0 | Max sat/vB for on-chain operations |
| `revenue-ops-planner-drain-timeout-hours` | int | 72 | Max drain phase duration |

---

## RPC Commands

| Command | Purpose |
|---------|---------|
| `revenue-capacity-report` | Existing advisory report (cleaned up) |
| `revenue-planner-status` | Current planner state, pending actions, last cycle |
| `revenue-planner-candidates` | Scored peer candidate list |
| `revenue-planner-execute` | Manually trigger a planner cycle |
| `revenue-planner-history` | Audit log of past actions |

---

## Files Touched

| File | Changes |
|------|---------|
| `modules/capacity_planner.py` | Major rewrite: background loop, peer discovery, execution, EV scoring, drain lifecycle |
| `modules/config.py` | ~11 new config fields |
| `modules/database.py` | 2 new tables, remove splice tables/methods (~150 lines) |
| `cl-revenue-ops.py` | New thread, ~5 RPC commands, remove splice handling (~200 lines), wire planner to rebalancer |
| `modules/rebalancer.py` | Add `set_capacity_planner()`, pending-close check |
| Tests | New test file + remove splice tests |
