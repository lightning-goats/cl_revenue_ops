# Capacity Planner Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the capacity planner from an advisory report generator into an automated channel lifecycle manager with peer discovery, and completely remove all splice functionality.

**Architecture:** The capacity planner becomes a peer of the rebalancer and boltz_manager. It runs on a timer-driven background loop (6h default), uses the generic spend ledger for budget tracking, and coordinates with the rebalancer via a pending-close interface. Peer discovery uses a 3-strategy ensemble (existing winners, fee-weighted neighbors, graph centrality). Channel closes follow a drain-then-close lifecycle via policy manager integration.

**Tech Stack:** Python 3.10+, pyln-client, SQLite (WAL mode), CLN RPCs (fundchannel, multifundchannel, close, listnodes, listchannels, listfunds, feerates)

**Design Doc:** `docs/plans/2026-03-19-capacity-planner-refactor-design.md`

---

## Phase 0: Splice Removal (~400 lines removed)

Clean foundation before building new functionality. Each task removes splice from one logical area. Run full test suite after each task.

### Task 1: Remove splice from capacity_planner.py and config.py

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `modules/config.py:782`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write failing test confirming splice removal**

In `tests/test_capacity_planner.py`, add a test verifying `peer_supports_splice` is no longer in output:

```python
def test_no_splice_fields_in_output(self):
    """Verify splice fields are completely removed from planner output."""
    # Run generate_report and verify no splice references
    report = self.planner.generate_report()
    for winner in report.get("winners", []):
        assert "peer_supports_splice" not in winner
    for loser in report.get("losers", []):
        assert "peer_supports_splice" not in loser
```

**Step 2: Remove splice from capacity_planner.py**

Remove the following:
- `_get_peer_splice_map()` method (lines 88-123) — entire method
- `peer_splice_map` parameter from `generate_report()` call chain:
  - Line 44: `peer_splice_map = self._get_peer_splice_map()` — delete line
  - Lines 46-47: Remove `peer_splice_map` argument from `_identify_winners()` and `_identify_losers()` calls
- `peer_splice_map` parameter from `_identify_winners()` signature (line 125) and `_identify_losers()` signature (line 174)
- `"peer_supports_splice"` key from winner dicts (line 168) and loser dicts (lines 258, 273)
- Splice-aware recommendation text in `_generate_recommendations()` (lines 304-329):
  - Replace splice-aware branching with simple recommendations
  - Remove `has_splice = winner.get('peer_supports_splice', False)` (line 304)
  - Simplify all recommendation strings to remove splice language
- Update module docstring (lines 1-6) to remove splice references
- Update `_get_mempool_recommendation()` text (line 80): "splicing" → "channel operations"

**Step 3: Remove SPLICE_COST_SATS from config.py**

- Delete line 782: `SPLICE_COST_SATS: int = 2000`
- Update comment on line 252 to remove "splices" mention

**Step 4: Update tests/test_capacity_planner.py**

- Remove `peer_splice_map` from all test calls to `_identify_winners()` and `_identify_losers()` (lines 88, 97, 131, 140, 147)
- Remove assertions about `peer_supports_splice`

**Step 5: Run tests**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
```

**Step 6: Commit**

```bash
git add modules/capacity_planner.py modules/config.py tests/test_capacity_planner.py
git commit -m "refactor: remove splice from capacity planner and config"
```

---

### Task 2: Remove splice from cl-revenue-ops.py

**Files:**
- Modify: `cl-revenue-ops.py`

**Step 1: Remove _handle_splice_completion and _get_splice_costs_from_bookkeeper**

Delete these two functions entirely:
- `_handle_splice_completion()`: lines 4115-4183 (~69 lines)
- `_get_splice_costs_from_bookkeeper()`: lines 4186-4276 (~91 lines)

**Step 2: Remove splice detection in channel state handler**

Lines 3773-3784: Remove the `CHANNELD_AWAITING_SPLICE` detection block:
```python
if old_state == 'CHANNELD_AWAITING_SPLICE' and new_state == 'CHANNELD_NORMAL':
    ...
    _handle_splice_completion(channel_id, peer_id)
    return
```

**Step 3: Remove splice cost reporting from revenue-report costs**

In the `costs` subcommand handler (around lines 2945-2974):
- Remove `splice_costs_day/week/month/total` queries (lines 2945-2948)
- Remove `splice_summary` query (lines 2950-2951)
- Remove `ChainCostDefaults.SPLICE_COST_SATS` from estimated costs dict (line 2958)
- Remove `"splice_costs"` key from return dict (lines 2969-2974)

**Step 4: Remove splice from _total_cost_budget_status()**

- Remove line 4691: `splice_cost_sats = int(database.get_splice_costs_since(since)) if database else 0`
- Remove line 4698: `"splice": splice_cost_sats,` from `actual_by_category`
- Remove line 4750: `"splice_cost_sats": splice_cost_sats,` from components

**Step 5: Remove splice from revenue-report summary**

- Remove line 3206: `"splice_cost_sats": pnl.get("splice_cost_sats", 0),`

**Step 6: Update comments and help text**

- Line 11: Remove "splices" from bookkeeper comment
- Line 718: Remove "splices" from option description
- Line 988: Remove "splices" from log message
- Lines 2841, 2847: Update help text to remove "splice"
- Line 2933: Update comment to remove "splice"
- Line 4305: Update docstring to remove "splices"

**Step 7: Run tests**

```bash
python3 -m pytest tests/ -x -q
```

**Step 8: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "refactor: remove splice handling from main plugin"
```

---

### Task 3: Remove splice from database.py, profitability_analyzer.py, and tests

**Files:**
- Modify: `modules/database.py`
- Modify: `modules/profitability_analyzer.py`
- Modify: `modules/rebalancer.py:1869-1870` (comment only)
- Modify: `tests/test_profitability_fixes.py`
- Modify: `tests/test_session3_audit_regressions.py`

**Step 1: Remove splice from database.py**

Remove the following:
- `splice_costs` table creation and indexes (lines 899-919)
- `record_splice()` method (lines 4558-4650)
- Section header comment (lines 4555-4556)
- `get_channel_splice_history()` method (lines 4652-4669)
- `get_total_splice_costs()` method (lines 4671-4684)
- `get_splice_costs_since()` method (lines 4686-4703)
- `get_splice_summary()` method (lines 4705-4732)
- `get_lifetime_stats()` splice query (lines 4001-4005) and return key (line 4028)
- Update comments (lines 390, 489, 803) to remove splice mentions

**Step 2: Remove splice from profitability_analyzer.py**

- Remove `splice_cost_sats: int = 0` from `ChannelCosts` dataclass (line 73)
- Update `total_cost` property (line 77) to remove `self.splice_cost_sats`
- Remove splice from `get_lifetime_pnl()` (lines 1003, 1010, 1031)
- Remove splice from `get_pnl_summary()` (lines 1076, 1079, 1097)
- Remove splice from `_get_channel_costs()` (lines 1682-1684, 1712)
- Update docstrings (lines 986, 988, 1000, 1050, 1073)

**Step 3: Update rebalancer.py comment**

- Update comment at lines 1869-1870 to remove splice mention

**Step 4: Update tests**

In `tests/test_profitability_fixes.py`:
- Remove `splice` parameter from `_make_costs()` helper (lines 37-44)
- Remove `TestSpliceCosts` class entirely (lines 141-186)
- Remove `splice=0` from all `_make_costs()` calls (lines 334, 367, 412, 576)
- Remove `get_channel_splice_history.return_value = []` mocks (lines 433, 468, 497)

In `tests/test_session3_audit_regressions.py`:
- Remove `get_total_splice_costs.return_value = 0` mocks (lines 307, 336, 363)

**Step 5: Run tests**

```bash
python3 -m pytest tests/test_profitability_fixes.py tests/test_session3_audit_regressions.py -v
python3 -m pytest tests/ -x -q
```

**Step 6: Commit**

```bash
git add modules/database.py modules/profitability_analyzer.py modules/rebalancer.py tests/test_profitability_fixes.py tests/test_session3_audit_regressions.py
git commit -m "refactor: remove splice from database, profitability analyzer, and tests"
```

---

## Phase 1: Foundation (Config + Database)

### Task 4: Add planner config fields

**Files:**
- Modify: `modules/config.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_config.py` (if exists, otherwise inline verification)

**Step 1: Add fields to CONFIG_FIELD_TYPES (before line 131)**

```python
    'planner_enabled': bool,
    'planner_interval': int,
    'planner_dry_run': bool,
    'planner_max_opens_per_cycle': int,
    'planner_max_closes_per_cycle': int,
    'planner_min_channel_sats': int,
    'planner_max_channel_sats': int,
    'planner_min_channel_age_days': int,
    'planner_min_peer_uptime_pct': float,
    'planner_max_fee_rate_sat_vb': float,
    'planner_drain_timeout_hours': int,
```

**Step 2: Add fields to CONFIG_FIELD_RANGES (before line 214)**

```python
    'planner_interval': (600, 604800),          # 10 min to 7 days
    'planner_max_opens_per_cycle': (0, 10),
    'planner_max_closes_per_cycle': (0, 10),
    'planner_min_channel_sats': (100000, 100000000),  # 100k to 1 BTC
    'planner_max_channel_sats': (500000, 1677721500),  # 500k to wumbo
    'planner_min_channel_age_days': (1, 365),
    'planner_min_peer_uptime_pct': (0.0, 100.0),
    'planner_max_fee_rate_sat_vb': (1.0, 1000.0),
    'planner_drain_timeout_hours': (1, 720),     # 1 hour to 30 days
```

**Step 3: Add fields to Config dataclass (before line 387, the internal fields section)**

```python
    # Capacity Planner
    planner_enabled: bool = False
    planner_interval: int = 21600               # 6 hours
    planner_dry_run: bool = True
    planner_max_opens_per_cycle: int = 1
    planner_max_closes_per_cycle: int = 1
    planner_min_channel_sats: int = 500000      # 500k sats
    planner_max_channel_sats: int = 10000000    # 10M sats
    planner_min_channel_age_days: int = 30
    planner_min_peer_uptime_pct: float = 95.0
    planner_max_fee_rate_sat_vb: float = 50.0
    planner_drain_timeout_hours: int = 72
```

**Step 4: Add fields to ConfigSnapshot (before line 743, before version)**

```python
    # Capacity Planner
    planner_enabled: bool = False
    planner_interval: int = 21600
    planner_dry_run: bool = True
    planner_max_opens_per_cycle: int = 1
    planner_max_closes_per_cycle: int = 1
    planner_min_channel_sats: int = 500000
    planner_max_channel_sats: int = 10000000
    planner_min_channel_age_days: int = 30
    planner_min_peer_uptime_pct: float = 95.0
    planner_max_fee_rate_sat_vb: float = 50.0
    planner_drain_timeout_hours: int = 72
```

**Step 5: Add plugin.add_option() calls in cl-revenue-ops.py (after line 755)**

```python
plugin.add_option(
    name='revenue-ops-planner-enabled',
    default='false',
    description='Enable automated capacity planner for channel opens/closes (default: false)'
)
plugin.add_option(
    name='revenue-ops-planner-interval',
    default='21600',
    description='Seconds between capacity planner evaluation cycles (default: 21600 = 6 hours)'
)
plugin.add_option(
    name='revenue-ops-planner-dry-run',
    default='true',
    description='Log planner decisions without executing (default: true)'
)
plugin.add_option(
    name='revenue-ops-planner-min-channel-sats',
    default='500000',
    description='Minimum channel size in sats for automated opens (default: 500000)'
)
plugin.add_option(
    name='revenue-ops-planner-max-channel-sats',
    default='10000000',
    description='Maximum channel size in sats for automated opens (default: 10000000)'
)
plugin.add_option(
    name='revenue-ops-planner-max-fee-rate',
    default='50.0',
    description='Maximum on-chain fee rate (sat/vB) for automated opens/closes (default: 50.0)'
)
```

**Step 6: Parse options in init() (before line 903)**

```python
        planner_enabled=options.get('revenue-ops-planner-enabled', 'false').lower() in ('true', '1', 'yes'),
        planner_interval=_safe_int('revenue-ops-planner-interval'),
        planner_dry_run=options.get('revenue-ops-planner-dry-run', 'true').lower() in ('true', '1', 'yes'),
        planner_min_channel_sats=_safe_int('revenue-ops-planner-min-channel-sats'),
        planner_max_channel_sats=_safe_int('revenue-ops-planner-max-channel-sats'),
        planner_max_fee_rate_sat_vb=_safe_float('revenue-ops-planner-max-fee-rate'),
```

**Step 7: Run tests**

```bash
python3 -m pytest tests/ -x -q
```

**Step 8: Commit**

```bash
git add modules/config.py cl-revenue-ops.py
git commit -m "feat: add capacity planner config fields"
```

---

### Task 5: Add planner database tables

**Files:**
- Modify: `modules/database.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests for new database methods**

```python
class TestPlannerDatabase:
    def test_record_planner_candidate(self):
        """Test recording and retrieving planner candidates."""
        db.record_planner_candidate("peer1", score=0.8, source="winner",
                                     capacity_recommendation_sats=2000000)
        candidates = db.get_planner_candidates(min_score=0.5)
        assert len(candidates) == 1
        assert candidates[0]["peer_id"] == "peer1"
        assert candidates[0]["score"] == 0.8

    def test_record_planner_action(self):
        """Test recording and retrieving planner actions."""
        action_id = db.record_planner_action(
            action_type="open", peer_id="peer1",
            amount_sats=2000000, estimated_cost_sats=5000,
            reason="High ROI winner"
        )
        assert action_id > 0
        actions = db.get_planner_actions(limit=10)
        assert len(actions) == 1

    def test_update_planner_action_status(self):
        """Test updating action status through lifecycle."""
        action_id = db.record_planner_action(
            action_type="close", peer_id="peer2",
            amount_sats=1000000, estimated_cost_sats=3000,
            reason="Zombie channel"
        )
        db.update_planner_action(action_id, status="executing")
        db.update_planner_action(action_id, status="completed",
                                  actual_cost_sats=2800)
        action = db.get_planner_action(action_id)
        assert action["status"] == "completed"
        assert action["actual_cost_sats"] == 2800
```

**Step 2: Add table creation to database.py initialize() (after existing tables, ~line 1098)**

```python
        # Capacity Planner tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS planner_candidates (
                peer_id TEXT PRIMARY KEY,
                score REAL NOT NULL DEFAULT 0.0,
                source TEXT NOT NULL,
                last_evaluated INTEGER NOT NULL,
                capacity_recommendation_sats INTEGER,
                connect_successes INTEGER DEFAULT 0,
                connect_failures INTEGER DEFAULT 0,
                metadata_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_planner_candidates_score ON planner_candidates(score)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS planner_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                peer_id TEXT NOT NULL,
                channel_id TEXT,
                amount_sats INTEGER,
                estimated_cost_sats INTEGER,
                actual_cost_sats INTEGER,
                status TEXT NOT NULL DEFAULT 'planned',
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                reason TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_planner_actions_status ON planner_actions(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_planner_actions_peer ON planner_actions(peer_id)")
```

**Step 3: Add database methods**

Add CRUD methods for both tables:
- `record_planner_candidate(peer_id, score, source, capacity_recommendation_sats, metadata)` — INSERT OR REPLACE
- `get_planner_candidates(min_score=0.0, source=None, limit=32)` — SELECT with filters
- `update_candidate_score(peer_id, delta)` — increment score
- `delete_planner_candidate(peer_id)` — DELETE
- `record_planner_action(action_type, peer_id, amount_sats, estimated_cost_sats, reason, channel_id, metadata)` — INSERT, returns id
- `update_planner_action(action_id, status, actual_cost_sats, channel_id, completed_at)` — UPDATE
- `get_planner_action(action_id)` — SELECT by id
- `get_planner_actions(status=None, limit=20)` — SELECT with optional status filter
- `get_recent_planner_actions(peer_id, hours=24)` — for cooldown check

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
```

**Step 5: Commit**

```bash
git add modules/database.py tests/test_capacity_planner.py
git commit -m "feat: add planner_candidates and planner_actions database tables"
```

---

## Phase 2: Enhanced Analysis

### Task 6: Enrich winner identification with existing data signals

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests for enriched winner scoring**

```python
def test_winner_includes_kalman_velocity(self):
    """Winners with high kalman_velocity are flagged as urgently capacity-constrained."""
    # Setup flow_metrics with kalman_velocity > 0.1
    # Assert winner dict includes velocity_urgency field

def test_winner_includes_sourced_fee_contribution(self):
    """Winners scoring includes inbound fee contribution to other channels."""
    # Setup profitability with high sourced_fee_contribution_sats
    # Assert scoring weights this appropriately

def test_congested_winner_flagged_urgent(self):
    """Congested winners (HTLC slots >80%) are flagged for immediate action."""
    # Setup flow_metrics with is_congested=True
    # Assert winner has congestion_urgent=True

def test_dts_posterior_enriches_winner_scoring(self):
    """Winners with high DTS posterior mean are scored higher."""
    # Mock database.get_fee_strategy_state() returning high posterior mean
```

**Step 2: Enrich _identify_winners()**

Add to the winner scoring pipeline:
- Query `FlowMetrics.kalman_velocity` — positive velocity on source = draining faster
- Query `FlowMetrics.is_congested` — congested winner needs immediate capacity
- Query `database.get_fee_strategy_state()` for DTS posterior mean — high mean = proven fee earner
- Include `revenue.sourced_fee_contribution_sats` in effective ROI calculation
- Add `channel_role` from profitability data for downstream prioritization

**Step 3: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: enrich winner identification with Kalman velocity, DTS, congestion data"
```

---

### Task 7: Enrich loser identification with bleeders, channel role, Kalman data

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests for enriched loser detection**

```python
def test_hard_bleeder_bypasses_defibrillation_gate(self):
    """Hard bleeders go straight to CLOSE, skipping defibrillation requirement."""
    # Setup profitability.identify_bleeders_v2() returning hard bleeder
    # Assert loser action == "CLOSE" even with attempt_count < 2

def test_inbound_gateway_protected_from_closure(self):
    """INBOUND_GATEWAY channels have higher bar for closure."""
    # Setup channel_role == INBOUND_GATEWAY
    # Assert channel is NOT in losers despite poor ROI

def test_kalman_regime_change_defers_close(self):
    """Channels with recent regime change are deferred."""
    # Setup FlowMetrics.kalman_regime_change = True
    # Assert loser action == "DEFIBRILLATE" instead of "CLOSE"

def test_low_confidence_prevents_close(self):
    """Channels with confidence < 0.5 are not recommended for closure."""
    # Setup FlowMetrics.confidence = 0.3
    # Assert channel not in losers

def test_futility_no_route_strengthens_close_signal(self):
    """Futility breaker with no_route errors is a strong close signal."""
    # Mock database.get_failure_metadata() returning no_route failures

def test_low_uptime_close_signal(self):
    """Peers with < 80% uptime and poor ROI are strong close candidates."""
    # Mock database.get_peer_uptime_percent() returning 60%
```

**Step 2: Enrich _identify_losers()**

Add to the loser detection pipeline:
- Call `self.profitability.identify_bleeders_v2()` — hard bleeders bypass defibrillation gate
- Check `ChannelProfitability.channel_role` — protect INBOUND_GATEWAYs (require marginal_roi < -50%)
- Check `FlowMetrics.kalman_regime_change` — defer close if regime change detected
- Check `FlowMetrics.confidence` — skip closure if confidence < 0.5
- Query `database.get_failure_metadata()` for `no_route` futility — strong close signal
- Query `database.get_peer_uptime_percent()` — < 80% + poor ROI = close candidate
- Compare `avg_cost_ppm` from rebalance success rate vs broadcast fee — structurally unprofitable check

**Step 3: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: enrich loser detection with bleeders, channel role, Kalman, uptime"
```

---

## Phase 3: Peer Discovery

### Task 8: Strategy 1 (existing winners) and Strategy 2 (fee-weighted neighbors)

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
class TestPeerDiscovery:
    def test_discover_from_winners_returns_existing_peers(self):
        """Strategy 1: existing winners are proposed for capital injection."""
        # Setup winners list with peer_ids
        # Assert discover returns those peer_ids with source="winner"

    def test_discover_from_neighbors_finds_adjacent_peers(self):
        """Strategy 2: neighbors of top earners are proposed."""
        # Mock listchannels for a top-earning peer
        # Assert returned candidates are that peer's channel partners
        # Assert candidates exclude peers we already have channels with

    def test_discover_excludes_existing_peers(self):
        """Discovered candidates exclude peers with existing channels."""
        # Setup existing channel with peer_id_x
        # Assert peer_id_x not in discovered candidates

    def test_discover_from_neighbors_limits_candidates(self):
        """Strategy 2 returns at most 5 candidates per top earner."""
```

**Step 2: Implement _discover_from_winners()**

```python
def _discover_from_winners(self, winners: List[Dict]) -> List[Dict]:
    """Strategy 1: Propose existing winners for additional channel opens."""
    candidates = []
    for winner in winners:
        if winner["roi"] > 30.0:  # Only very strong winners
            candidates.append({
                "peer_id": winner["peer_id"],
                "source": "winner",
                "score": winner["roi"] / 100.0,
                "reason": f"Existing winner with {winner['roi']:.1f}% ROI",
            })
    return candidates
```

**Step 3: Implement _discover_from_neighbors()**

```python
def _discover_from_neighbors(self, all_profitability) -> List[Dict]:
    """Strategy 2: Find neighbors of top-earning peers (CLBOSS-inspired)."""
    # 1. Sort existing channels by daily fee income, take top 3
    # 2. For each top earner, call listchannels(source=peer_id)
    # 3. Extract destination node_ids as candidates
    # 4. Filter: no existing channel, not self
    # 5. Score by the patron's ROI (proxy for neighborhood quality)
    # Return max 5 candidates per top earner, 10 total
```

**Step 4: Implement _discover_peers() orchestrator**

```python
def _discover_peers(self, winners, all_profitability, all_flow) -> List[Dict]:
    """Run all discovery strategies and merge candidates."""
    candidates = []
    candidates.extend(self._discover_from_winners(winners))
    candidates.extend(self._discover_from_neighbors(all_profitability))
    # Strategy 3 added in Task 9
    # Deduplicate by peer_id, keeping highest score
    seen = {}
    for c in candidates:
        pid = c["peer_id"]
        if pid not in seen or c["score"] > seen[pid]["score"]:
            seen[pid] = c
    return list(seen.values())
```

**Step 5: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: peer discovery strategies 1 (winners) and 2 (neighbors)"
```

---

### Task 9: Strategy 3 (graph centrality) and candidate scoring

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
def test_discover_from_graph_scores_by_centrality(self):
    """Strategy 3: peers scored by channel count * capacity * fee competitiveness."""
    # Mock listnodes and listchannels
    # Assert returned candidates are scored by composite centrality metric
    # Assert candidates exclude peers with existing channels

def test_discover_from_graph_requires_min_nodes(self):
    """Strategy 3: requires knowledge of 800+ nodes before proposing."""
    # Mock listnodes returning only 100 nodes
    # Assert returns empty list

def test_candidate_scoring_integrates_reputation(self):
    """Candidate score includes peer reputation from database."""
    # Mock get_peer_reputation returning score 0.9
    # Assert candidate score is boosted

def test_candidate_scoring_integrates_uptime(self):
    """Candidates below min_peer_uptime are filtered out."""
    # Mock get_peer_uptime_percent returning 80%
    # Assert candidate is excluded (below 95% default threshold)

def test_candidate_scoring_profit_inheritance(self):
    """Candidates with prior profitable channels score higher."""
    # Mock get_peer_closed_channel_profit_summary returning positive ROI
    # Assert candidate score is boosted
```

**Step 2: Implement _discover_from_graph()**

```python
def _discover_from_graph(self, existing_peer_ids: set) -> List[Dict]:
    """Strategy 3: Network centrality scoring via listnodes/listchannels."""
    try:
        nodes = self.plugin.rpc.listnodes().get("nodes", [])
    except Exception:
        return []

    if len(nodes) < 800:
        return []  # Insufficient graph knowledge

    # Score each node: peer_count * sqrt(total_capacity) * fee_factor
    # Filter: not in existing_peer_ids, not self
    # Sort by score descending, return top 10
```

**Step 3: Implement _score_candidate()**

Composite scoring integrating all available data:

```python
def _score_candidate(self, peer_id: str, base_score: float) -> float:
    """Enrich candidate score with reputation, uptime, and profit history."""
    score = base_score

    # Peer reputation (Laplace-smoothed success rate)
    rep = self.profitability.database.get_peer_reputation(peer_id)
    if rep:
        rep_score = (rep.get('successes', 0) + 1) / (rep.get('total', 0) + 2)
        score *= rep_score  # 0.0-1.0 multiplier

    # Profit inheritance from closed channels
    closed_summary = self.profitability.database.get_peer_closed_channel_profit_summary(peer_id)
    if closed_summary and closed_summary.get('marginal_roi_proxy', 0) > 0:
        score *= 1.5  # Boost for proven profitable peer

    return score
```

**Step 4: Implement candidate pool persistence**

```python
def _update_candidate_pool(self, candidates: List[Dict]):
    """Persist scored candidates to database for cross-cycle tracking."""
    for candidate in candidates:
        self.profitability.database.record_planner_candidate(
            peer_id=candidate["peer_id"],
            score=candidate["score"],
            source=candidate["source"],
            capacity_recommendation_sats=candidate.get("capacity_recommendation_sats"),
        )
    # Prune pool to max 32 candidates
    # Remove candidates with score < -3.0
```

**Step 5: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: graph centrality discovery and composite candidate scoring"
```

---

## Phase 4: Execution Engine

### Task 10: Safety guards (fee gate, reserve, cooldown, budget)

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
class TestSafetyGuards:
    def test_fee_gate_blocks_high_fees(self):
        """Channel ops blocked when sat/vB > max_fee_rate."""
        # Mock feerates returning 100 sat/vB
        # Assert _check_fee_gate() returns False

    def test_fee_gate_allows_low_fees(self):
        """Channel ops allowed when sat/vB < max_fee_rate."""
        # Mock feerates returning 10 sat/vB
        # Assert _check_fee_gate() returns True

    def test_reserve_check_blocks_insufficient_funds(self):
        """Channel open blocked when on-chain balance < reserve + channel_size."""
        # Mock listfunds returning low confirmed balance
        # Assert _check_reserve() returns False

    def test_cooldown_blocks_recent_peer_action(self):
        """Actions blocked if same peer had action in last 24h."""
        # Insert recent planner_action for peer
        # Assert _check_cooldown(peer_id) returns False

    def test_dry_run_logs_but_does_not_execute(self):
        """In dry_run mode, decisions are logged but not executed."""
        # Set planner_dry_run = True
        # Call execute_cycle
        # Assert no RPC calls made
        # Assert planner_actions recorded with status="dry_run"

    def test_budget_check_respects_unified_budget(self):
        """Channel open blocked when unified budget exhausted."""
```

**Step 2: Implement safety guard methods**

```python
def _check_fee_gate(self, cfg) -> tuple[bool, str]:
    """Check on-chain fee rate is acceptable. Returns (ok, reason)."""
    try:
        feerates = self.plugin.rpc.feerates(style="perkb")
        opening_kvb = feerates.get("perkb", {}).get("opening", 1000)
        sat_per_vb = opening_kvb / 1000.0
        if sat_per_vb > cfg.planner_max_fee_rate_sat_vb:
            return False, f"Fee rate {sat_per_vb:.0f} sat/vB exceeds max {cfg.planner_max_fee_rate_sat_vb}"
        return True, f"Fee rate {sat_per_vb:.0f} sat/vB acceptable"
    except Exception as e:
        return False, f"Cannot check feerates: {e}"

def _check_reserve(self, cfg, required_sats: int) -> tuple[bool, str]:
    """Check on-chain balance has sufficient reserve."""
    try:
        funds = self.plugin.rpc.listfunds()
        confirmed = sum(o["amount_msat"] // 1000 for o in funds.get("outputs", [])
                       if o.get("status") == "confirmed")
        available = confirmed - cfg.min_wallet_reserve
        if available < required_sats:
            return False, f"Insufficient funds: {available} < {required_sats} required"
        return True, f"Available: {available} sats"
    except Exception as e:
        return False, f"Cannot check funds: {e}"

def _check_cooldown(self, peer_id: str) -> tuple[bool, str]:
    """Check 24h cooldown per peer."""
    recent = self.database.get_recent_planner_actions(peer_id, hours=24)
    if recent:
        return False, f"Cooldown: {len(recent)} action(s) in last 24h"
    return True, "No recent actions"

def _check_safety_guards(self, cfg, action_type: str, peer_id: str,
                          amount_sats: int = 0) -> tuple[bool, str]:
    """Run all safety checks. Returns (ok, reason)."""
    if cfg.planner_dry_run:
        return True, "dry_run"  # Allow analysis, flag at execution time

    fee_ok, fee_reason = self._check_fee_gate(cfg)
    if not fee_ok:
        return False, fee_reason

    if action_type == "open":
        reserve_ok, reserve_reason = self._check_reserve(cfg, amount_sats)
        if not reserve_ok:
            return False, reserve_reason

    cooldown_ok, cooldown_reason = self._check_cooldown(peer_id)
    if not cooldown_ok:
        return False, cooldown_reason

    return True, "All guards passed"
```

**Step 3: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: capacity planner safety guards — fee gate, reserve, cooldown, budget"
```

---

### Task 11: Channel sizing (ROI-proportional) and EV-based open decision

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
class TestChannelSizing:
    def test_roi_proportional_sizing(self):
        """Higher ROI candidates get proportionally larger channels."""
        # Two candidates: ROI 50% and ROI 25%
        # Available: 6M sats
        # Assert first gets ~4M, second gets ~2M (proportional to ROI)

    def test_size_clamped_to_min_max(self):
        """Channel size clamped to [min_channel, max_channel]."""
        # Candidate with very high ROI but available funds limited
        # Assert size <= max_channel and size >= min_channel

    def test_never_more_than_half_remaining(self):
        """No single channel takes more than 50% of available funds."""
        # Single candidate with high ROI, 4M available
        # Assert size <= 2M

class TestOpenEV:
    def test_positive_ev_allows_open(self):
        """Channel open proceeds when EV > 0."""
        # Mock revenue estimate, chain costs, rebalance costs
        # Assert _calculate_open_ev() > 0

    def test_negative_ev_blocks_open(self):
        """Channel open blocked when EV < 0."""
        # High chain costs + low revenue estimate
        # Assert _calculate_open_ev() <= 0

    def test_profit_inheritance_for_returning_peers(self):
        """Returning peers use historical daily revenue for EV."""
        # Mock get_peer_closed_channel_profit_summary with positive data
        # Assert EV uses historical revenue, not estimate
```

**Step 2: Implement _size_channel()**

```python
def _size_channel(self, candidate: Dict, all_candidates: List[Dict],
                   available_sats: int, cfg) -> int:
    """ROI-proportional channel sizing."""
    total_roi = sum(max(c.get("score", 0.01), 0.01) for c in all_candidates)
    roi_weight = max(candidate.get("score", 0.01), 0.01) / total_roi

    raw_size = int(available_sats * roi_weight)

    # Never more than half remaining
    raw_size = min(raw_size, available_sats // 2)

    # Clamp to config bounds
    return max(cfg.planner_min_channel_sats,
               min(raw_size, cfg.planner_max_channel_sats))
```

**Step 3: Implement _calculate_open_ev()**

```python
def _calculate_open_ev(self, peer_id: str, channel_size_sats: int, cfg) -> float:
    """EV-based channel open decision. Returns expected profit in sats."""
    # Estimate daily revenue
    closed_summary = self.profitability.database.get_peer_closed_channel_profit_summary(peer_id)
    if closed_summary and closed_summary.get("daily_net_est_sats", 0) > 0:
        daily_revenue = closed_summary["daily_net_est_sats"]
    else:
        # Estimate from network-average fee rate and channel capacity
        daily_revenue = channel_size_sats * 0.5 * 150 / 1e6  # 50% utilization * 150 PPM

    # Estimate costs
    try:
        feerates = self.plugin.rpc.feerates(style="perkb")
        sat_per_vb = feerates.get("perkb", {}).get("opening", 1000) / 1000.0
        open_cost = int(sat_per_vb * 140)   # ~140 vbytes for open
        close_cost = int(sat_per_vb * 200)  # ~200 vbytes for close
    except Exception:
        open_cost = ChainCostDefaults.CHANNEL_OPEN_COST_SATS
        close_cost = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS

    on_chain_cost = open_cost + close_cost
    lifetime_days = 180  # Conservative 6-month estimate

    expected_revenue = daily_revenue * lifetime_days
    return expected_revenue - on_chain_cost
```

**Step 4: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: ROI-proportional channel sizing and EV-based open decisions"
```

---

### Task 12: Channel open execution

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
class TestChannelOpen:
    def test_execute_open_calls_fundchannel(self):
        """Successful open calls plugin.rpc.fundchannel with correct params."""
        # Mock fundchannel returning success
        # Assert called with peer_id and amount

    def test_execute_open_records_action(self):
        """Open execution records action in planner_actions table."""
        # Mock fundchannel success
        # Assert planner_action recorded with status="completed"

    def test_execute_open_connects_first(self):
        """Open attempts to connect to peer before funding."""
        # Mock connect and fundchannel
        # Assert connect called before fundchannel

    def test_execute_open_handles_failure(self):
        """Failed fundchannel records action as failed."""
        # Mock fundchannel raising exception
        # Assert planner_action recorded with status="failed"

    def test_execute_open_reserves_budget(self):
        """Open reserves budget via generic spend ledger before executing."""
        # Mock database.reserve_spend
        # Assert called with category="channel_open"

    def test_dry_run_does_not_call_fundchannel(self):
        """Dry run mode logs but does not execute."""
        # Set dry_run=True
        # Assert fundchannel NOT called
        # Assert action logged with status="dry_run"
```

**Step 2: Implement _execute_open()**

```python
def _execute_open(self, peer_id: str, amount_sats: int, cfg, reason: str) -> Dict:
    """Execute a channel open via fundchannel RPC."""
    action_id = self.database.record_planner_action(
        action_type="open", peer_id=peer_id,
        amount_sats=amount_sats,
        estimated_cost_sats=self._estimate_open_cost(),
        reason=reason,
    )

    if cfg.planner_dry_run:
        self.database.update_planner_action(action_id, status="dry_run")
        self.plugin.log(f"[DRY RUN] Would open {amount_sats} sat channel to {peer_id[:16]}...")
        return {"action_id": action_id, "status": "dry_run"}

    # Reserve budget
    reservation_id = f"planner-open-{peer_id[:16]}-{int(time.time())}"
    self.database.reserve_spend(
        reservation_id=reservation_id,
        amount_sats=self._estimate_open_cost(),
        category="channel_open",
    )

    try:
        # Connect first
        try:
            self.plugin.rpc.connect(peer_id)
        except Exception:
            pass  # May already be connected

        # Fund channel
        result = self.plugin.rpc.fundchannel(
            id=peer_id,
            amount=amount_sats,
            announce=True,
        )

        self.database.update_planner_action(
            action_id, status="completed",
            channel_id=result.get("channel_id"),
        )
        self.database.mark_spend_reservation_spent(
            reservation_id=reservation_id,
            actual_spent_sats=result.get("fee_sats", 0),
            source="capacity_planner",
        )
        return {"action_id": action_id, "status": "completed", "result": result}

    except Exception as e:
        self.database.update_planner_action(action_id, status="failed")
        self.database.release_spend_reservation(reservation_id)
        self.plugin.log(f"Channel open failed for {peer_id[:16]}: {e}", level='error')
        return {"action_id": action_id, "status": "failed", "error": str(e)}
```

**Step 3: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: channel open execution via fundchannel with budget reservation"
```

---

### Task 13: Drain-then-close lifecycle and close execution

**Files:**
- Modify: `modules/capacity_planner.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
class TestDrainAndClose:
    def test_initiate_drain_sets_policy(self):
        """Drain phase sets peer policy to passive + source_only."""
        # Mock policy_manager.set_policy
        # Assert called with strategy="passive", rebalance_mode="source_only"

    def test_drain_records_action_as_draining(self):
        """Drain initiation records action with status='draining'."""

    def test_check_drain_complete_by_balance(self):
        """Drain completes when local balance < 10% capacity."""
        # Mock listpeerchannels showing low local balance
        # Assert _check_drain_complete returns True

    def test_drain_timeout_proceeds_to_close(self):
        """After drain_timeout_hours, close proceeds regardless."""

    def test_execute_close_calls_close_rpc(self):
        """Successful close calls plugin.rpc.close with channel_id."""
        # Mock close returning success
        # Assert called with correct channel_id

    def test_execute_close_stops_rebalancer_jobs(self):
        """Close stops any active rebalancer jobs on the channel."""
        # Mock rebalancer.job_manager.has_active_job returning True
        # Assert stop_job called

    def test_close_respects_static_policy(self):
        """Channels with static policy are never closed."""
        # Mock get_policy returning strategy="static"
        # Assert close is blocked

    def test_close_respects_protect_tag(self):
        """Channels tagged 'protect' are never closed."""

    def test_dry_run_close_does_not_execute(self):
        """Dry run mode logs close decision but does not execute."""
```

**Step 2: Implement drain lifecycle**

```python
def _initiate_drain(self, peer_id: str, channel_id: str, cfg, reason: str) -> int:
    """Phase 1: Set policy to drain, record action."""
    action_id = self.database.record_planner_action(
        action_type="close", peer_id=peer_id,
        channel_id=channel_id,
        estimated_cost_sats=ChainCostDefaults.CHANNEL_CLOSE_COST_SATS,
        reason=reason,
    )

    if cfg.planner_dry_run:
        self.database.update_planner_action(action_id, status="dry_run")
        self.plugin.log(f"[DRY RUN] Would drain and close {channel_id}")
        return action_id

    # Set drain policy
    if self.policy_manager:
        self.policy_manager.set_policy(
            peer_id, strategy="passive",
            rebalance_mode="source_only",
            tags=["closing", "drain_phase"],
            expires_in_hours=cfg.planner_drain_timeout_hours,
        )

    self.database.update_planner_action(action_id, status="draining")
    self._pending_closes[channel_id] = int(time.time())
    return action_id

def _check_drain_complete(self, channel_id: str, cfg) -> bool:
    """Check if drain phase is complete (low local balance or timeout)."""
    # Check timeout
    drain_start = self._pending_closes.get(channel_id)
    if drain_start:
        elapsed_hours = (time.time() - drain_start) / 3600
        if elapsed_hours > cfg.planner_drain_timeout_hours:
            return True

    # Check balance
    try:
        channels = self.plugin.rpc.listpeerchannels()
        for ch in channels.get("channels", []):
            if ch.get("short_channel_id") == channel_id or ch.get("channel_id") == channel_id:
                local = ch.get("spendable_msat", 0)
                if isinstance(local, str):
                    local = int(local.replace("msat", ""))
                capacity = ch.get("total_msat", 1)
                if isinstance(capacity, str):
                    capacity = int(capacity.replace("msat", ""))
                if capacity > 0 and (local / capacity) < 0.1:
                    return True
    except Exception:
        pass

    return False

def _execute_close(self, channel_id: str, peer_id: str, action_id: int, cfg) -> Dict:
    """Phase 2: Execute channel close."""
    # Stop rebalancer jobs if any
    if self.rebalancer and hasattr(self.rebalancer, 'job_manager'):
        if self.rebalancer.job_manager.has_active_job(channel_id):
            self.rebalancer.job_manager.stop_job(channel_id, reason="planner_close")

    try:
        result = self.plugin.rpc.close(id=channel_id)
        self.database.update_planner_action(action_id, status="completed")
        del self._pending_closes[channel_id]
        return {"action_id": action_id, "status": "completed", "result": result}
    except Exception as e:
        self.database.update_planner_action(action_id, status="failed")
        self.plugin.log(f"Channel close failed for {channel_id}: {e}", level='error')
        return {"action_id": action_id, "status": "failed", "error": str(e)}
```

**Step 3: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: drain-then-close lifecycle with policy coordination"
```

---

## Phase 5: Integration

### Task 14: Background loop and execute_cycle()

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
class TestExecuteCycle:
    def test_execute_cycle_skips_when_disabled(self):
        """Cycle does nothing when planner_enabled=False."""

    def test_execute_cycle_opens_best_candidate(self):
        """Cycle opens channel to highest-scoring candidate when guards pass."""

    def test_execute_cycle_closes_worst_loser(self):
        """Cycle closes worst loser when guards pass."""

    def test_execute_cycle_respects_max_opens_per_cycle(self):
        """At most max_opens_per_cycle opens per invocation."""

    def test_execute_cycle_progresses_draining_channels(self):
        """Cycle checks drain progress on pending closes."""

    def test_execute_cycle_returns_summary(self):
        """Cycle returns structured summary of decisions made."""
```

**Step 2: Implement execute_cycle()**

```python
def execute_cycle(self, cfg=None) -> Dict[str, Any]:
    """Main timer-driven cycle. Evaluates and executes open/close decisions."""
    if cfg is None:
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

    if not cfg.planner_enabled:
        return {"skipped": True, "reason": "planner disabled"}

    summary = {"opens": [], "closes": [], "drains_progressed": [], "skipped_reasons": []}

    # 1. Check fee gate
    fee_ok, fee_reason = self._check_fee_gate(cfg)
    if not fee_ok:
        summary["skipped_reasons"].append(fee_reason)

    # 2. Fetch analysis data
    all_profitability = self.profitability.analyze_all_channels()
    all_flow = self.flow.analyze_all_channels()

    # 3. Identify winners and losers
    winners = self._identify_winners(all_profitability, all_flow)
    losers = self._identify_losers(all_profitability, all_flow)

    # 4. Progress existing drains
    for channel_id in list(self._pending_closes.keys()):
        if self._check_drain_complete(channel_id, cfg):
            # Find matching action
            # Execute close
            pass

    # 5. Initiate new closes (up to max_closes_per_cycle)
    closes_this_cycle = 0
    closeable = [l for l in losers if l.get("action") == "CLOSE"]
    sorted_closeable = sorted(closeable, key=lambda x: x.get("marginal_roi", 0))
    for loser in sorted_closeable:
        if closes_this_cycle >= cfg.planner_max_closes_per_cycle:
            break
        if loser.get("scid") in self._pending_closes:
            continue
        # Check policy, guards
        self._initiate_drain(loser["peer_id"], loser["scid"], cfg, loser["reason"])
        closes_this_cycle += 1

    # 6. Discover and open channels (up to max_opens_per_cycle)
    if fee_ok:
        candidates = self._discover_peers(winners, all_profitability, all_flow)
        # Score, size, EV-check, execute
        opens_this_cycle = 0
        for candidate in sorted(candidates, key=lambda c: c["score"], reverse=True):
            if opens_this_cycle >= cfg.planner_max_opens_per_cycle:
                break
            # Size channel
            # Check EV
            # Check safety guards
            # Execute open
            opens_this_cycle += 1

    # 7. Update candidate pool
    self._update_candidate_pool(candidates if fee_ok else [])

    summary["timestamp"] = int(time.time())
    return summary
```

**Step 3: Add background loop to cl-revenue-ops.py**

Add after line 1534 (after `boltz_auto_cycle_loop` thread):

```python
def capacity_planner_loop():
    """Background loop for automated capacity planning."""
    if not config.planner_enabled:
        plugin.log("Capacity planner disabled, loop not started")
        return

    # Startup delay: wait for flow + profitability data to warm up
    startup_delay = 300
    if shutdown_event.wait(startup_delay):
        return

    while not shutdown_event.is_set():
        try:
            plugin.log("Running scheduled capacity planner cycle...")
            result = capacity_planner.execute_cycle()
            if result.get("skipped"):
                plugin.log(f"Planner cycle skipped: {result.get('reason')}", level='debug')
            else:
                opens = len(result.get("opens", []))
                closes = len(result.get("closes", []))
                plugin.log(f"Planner cycle complete: {opens} opens, {closes} closes")
        except Exception as e:
            plugin.log(f"Error in capacity planner cycle: {e}", level='error')

        cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
        interval = max(600, cfg_snap.planner_interval)
        jitter = int(interval * 0.2)
        sleep_time = interval + random.randint(-jitter, jitter)
        if shutdown_event.wait(sleep_time):
            break
```

Then register the thread:

```python
threading.Thread(target=capacity_planner_loop, daemon=True, name="capacity-planner").start()
```

**Step 4: Wire planner with config and database references**

In `init()`, update the `CapacityPlanner` constructor call (line 1135) to pass config and database:

```python
capacity_planner = CapacityPlanner(
    safe_plugin, profitability_analyzer, flow_analyzer,
    policy_manager=policy_manager, config=config, database=database
)
```

**Step 5: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py cl-revenue-ops.py tests/test_capacity_planner.py
git commit -m "feat: capacity planner background loop and execute_cycle orchestrator"
```

---

### Task 15: Rebalancer coordination

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `modules/rebalancer.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_capacity_planner.py`

**Step 1: Write tests**

```python
def test_pending_close_interface(self):
    """Rebalancer can query capacity planner for pending closes."""
    planner._pending_closes["123x1x0"] = int(time.time())
    assert planner.is_pending_close("123x1x0")
    assert not planner.is_pending_close("456x2x0")

def test_rebalancer_skips_pending_close_channels(self):
    """Rebalancer skips channels pending closure by planner."""
    # Mock capacity_planner.is_pending_close returning True
    # Assert channel excluded from rebalance candidates
```

**Step 2: Add pending-close interface to CapacityPlanner**

```python
def is_pending_close(self, channel_id: str) -> bool:
    """Check if a channel is pending close by the planner."""
    return channel_id in self._pending_closes
```

**Step 3: Add set_capacity_planner to rebalancer**

In `modules/rebalancer.py`, add method (follow `set_profitability_analyzer` pattern):

```python
def set_capacity_planner(self, planner):
    """Set reference to capacity planner for coordination."""
    self._capacity_planner = planner
```

**Step 4: Add pending-close check in rebalancer candidate selection**

In `find_rebalance_candidates()` (around line 2043), add check:

```python
# Skip channels pending closure by capacity planner
if hasattr(self, '_capacity_planner') and self._capacity_planner:
    if self._capacity_planner.is_pending_close(dest_channel):
        continue
```

**Step 5: Wire in cl-revenue-ops.py init()**

After the existing `rebalancer.set_profitability_analyzer(profitability_analyzer)` call:

```python
rebalancer.set_capacity_planner(capacity_planner)
```

**Step 6: Run tests and commit**

```bash
python3 -m pytest tests/test_capacity_planner.py tests/test_rebalancer_module.py -v
python3 -m pytest tests/ -x -q
git add modules/capacity_planner.py modules/rebalancer.py cl-revenue-ops.py tests/test_capacity_planner.py
git commit -m "feat: rebalancer-planner coordination via pending-close interface"
```

---

### Task 16: RPC commands

**Files:**
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_rpc_commands_audit.py` (if applicable)

**Step 1: Add RPC commands to cl-revenue-ops.py**

Add after the existing `revenue-capacity-report` command:

```python
@plugin.method("revenue-planner-status")
def revenue_planner_status(plugin: Plugin) -> Dict[str, Any]:
    """Get capacity planner status — pending actions, last cycle result, config."""
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}
    return capacity_planner.get_status()

@plugin.method("revenue-planner-candidates")
def revenue_planner_candidates(plugin: Plugin, limit: int = 20) -> Dict[str, Any]:
    """List scored peer candidates for channel opens."""
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}
    candidates = database.get_planner_candidates(limit=limit)
    return {"candidates": candidates, "count": len(candidates)}

@plugin.method("revenue-planner-execute")
def revenue_planner_execute(plugin: Plugin) -> Dict[str, Any]:
    """Manually trigger a capacity planner cycle."""
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}
    return capacity_planner.execute_cycle()

@plugin.method("revenue-planner-history")
def revenue_planner_history(plugin: Plugin, limit: int = 20) -> Dict[str, Any]:
    """Get audit log of past planner actions."""
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}
    actions = database.get_planner_actions(limit=limit)
    return {"actions": actions, "count": len(actions)}
```

**Step 2: Update revenue-capacity-report docstring**

Update the docstring at line 2148-2152 to remove splice references:

```python
    """
    Generate a strategic capital redeployment report.

    Identifies "Winner" channels for capital injection
    and "Loser" channels for capital extraction or closure.
    """
```

**Step 3: Implement get_status() on CapacityPlanner**

```python
def get_status(self) -> Dict[str, Any]:
    """Return planner status for RPC query."""
    return {
        "enabled": self.config.planner_enabled if hasattr(self.config, 'planner_enabled') else False,
        "dry_run": self.config.planner_dry_run if hasattr(self.config, 'planner_dry_run') else True,
        "pending_closes": len(self._pending_closes),
        "pending_close_channels": list(self._pending_closes.keys()),
        "candidate_pool_size": len(self.database.get_planner_candidates()),
        "recent_actions": self.database.get_planner_actions(limit=5),
    }
```

**Step 4: Run tests and commit**

```bash
python3 -m pytest tests/ -x -q
git add cl-revenue-ops.py modules/capacity_planner.py
git commit -m "feat: add planner RPC commands — status, candidates, execute, history"
```

---

### Task 17: Final integration test and cleanup

**Files:**
- Modify: `modules/capacity_planner.py` (constructor, __init__)
- Test: `tests/test_capacity_planner.py`

**Step 1: Write integration test**

```python
class TestPlannerIntegration:
    def test_full_cycle_dry_run(self):
        """End-to-end test: full cycle in dry_run mode produces valid report."""
        # Setup all mocks (profitability, flow, listchannels, listnodes, feerates)
        # Call execute_cycle with dry_run=True
        # Assert structured summary returned
        # Assert planner_actions recorded with status="dry_run"
        # Assert no RPC mutations (no fundchannel/close calls)

    def test_generate_report_still_works(self):
        """Advisory report generation is not broken by refactor."""
        # Call generate_report
        # Assert valid structured output
        # Assert no splice fields
```

**Step 2: Verify __init__ signature and state initialization**

Ensure `CapacityPlanner.__init__` initializes all new state:

```python
def __init__(self, plugin, profitability_analyzer, flow_analyzer,
             policy_manager=None, config=None, database=None, rebalancer=None):
    self.plugin = plugin
    self.profitability = profitability_analyzer
    self.flow = flow_analyzer
    self.policy_manager = policy_manager
    self.config = config
    self.database = database or (profitability_analyzer.database if profitability_analyzer else None)
    self.rebalancer = rebalancer
    self._pending_closes: Dict[str, int] = {}  # channel_id -> drain_start_timestamp
```

**Step 3: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```

Verify total test count and all passing.

**Step 4: Commit**

```bash
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: capacity planner integration tests and constructor cleanup"
```

---

## Summary

| Phase | Tasks | Commits | Key Changes |
|-------|-------|---------|-------------|
| Phase 0: Splice Removal | 1-3 | 3 | ~400 lines removed across 6 files |
| Phase 1: Foundation | 4-5 | 2 | 11 config fields, 2 database tables |
| Phase 2: Enhanced Analysis | 6-7 | 2 | Enriched winner/loser detection with 10+ existing data signals |
| Phase 3: Peer Discovery | 8-9 | 2 | 3-strategy ensemble, composite scoring, candidate pool |
| Phase 4: Execution Engine | 10-13 | 4 | Safety guards, sizing, EV, open execution, drain-close lifecycle |
| Phase 5: Integration | 14-17 | 4 | Background loop, rebalancer coordination, RPC commands |
| **Total** | **17** | **17** | Major refactor: advisory → automated lifecycle manager |
