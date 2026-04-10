# Capital Efficiency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add capital-efficiency analysis that detects dead capital, feeds planner actions, improves neighbor discovery, and gates capex budgets toward productive channels.

**Architecture:** Introduce a pure `CapitalEfficiencyAnalyzer` that composes profitability, flow, database stage state, and optional hive hints into per-channel and fleet efficiency snapshots. Inject that snapshot into the capex budget engine and capacity planner so dead-capital handling, efficiency multipliers, and neighbor discovery all share one calculation source instead of duplicating heuristics.

**Tech Stack:** Python 3.10+, SQLite, pyln-client, pytest

**Spec:** `docs/superpowers/specs/2026-04-06-capital-efficiency-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `modules/capital_efficiency.py` | Create | Capital-efficiency analyzer and dataclasses |
| `modules/database.py` | Modify | Dead-capital stage schema and CRUD helpers |
| `modules/capex_budget.py` | Modify | Apply efficiency multiplier to per-channel budgets |
| `modules/capacity_planner.py` | Modify | Dead-capital loser pipeline and enhanced neighbor discovery |
| `cl-revenue-ops.py` | Modify | Construct and inject `CapitalEfficiencyAnalyzer` |
| `tests/test_capital_efficiency.py` | Create | Analyzer unit tests |
| `tests/test_database.py` | Modify | Dead-capital persistence tests |
| `tests/test_capex_budget.py` | Modify | Efficiency-multiplier tests |
| `tests/test_capacity_planner.py` | Modify | Dead-capital and neighbor-discovery tests |

### Task 1: Add dead-capital stage persistence to the database

**Files:**
- Modify: `modules/database.py`
- Modify: `tests/test_database.py`

**Step 1: Write the failing tests**

```python
def test_dead_capital_stage_round_trip(tmp_path):
    db = _make_db(tmp_path)
    db.upsert_dead_capital_stage("100x1x0", "fee_reduction", 123)
    assert db.get_dead_capital_stages()["100x1x0"]["stage"] == "fee_reduction"

def test_delete_dead_capital_stage(tmp_path):
    db = _make_db(tmp_path)
    db.upsert_dead_capital_stage("100x1x0", "close", 123)
    db.delete_dead_capital_stage("100x1x0")
    assert "100x1x0" not in db.get_dead_capital_stages()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_database.py -k dead_capital -v`
Expected: FAIL with missing table or missing `Database` methods

**Step 3: Write minimal implementation**

```python
conn.execute("""
    CREATE TABLE IF NOT EXISTS dead_capital_stage (
        channel_id TEXT PRIMARY KEY,
        stage TEXT NOT NULL DEFAULT 'fee_reduction',
        entered_at INTEGER NOT NULL
    )
""")

def get_dead_capital_stages(self) -> Dict[str, Dict[str, int | str]]:
    ...

def upsert_dead_capital_stage(self, channel_id: str, stage: str, entered_at: int) -> None:
    ...

def delete_dead_capital_stage(self, channel_id: str) -> None:
    ...
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_database.py -k dead_capital -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/database.py tests/test_database.py
git commit -m "feat: persist dead capital stages"
```

### Task 2: Add the capital-efficiency analyzer

**Files:**
- Create: `modules/capital_efficiency.py`
- Create: `tests/test_capital_efficiency.py`

**Step 1: Write the failing tests**

```python
def test_rpsd_and_percentile_rank_are_computed():
    fleet = analyzer.analyze()
    assert fleet.channel_efficiencies["100x1x0"].rpsd == 50.0
    assert 0.0 <= fleet.channel_efficiencies["100x1x0"].efficiency_rank <= 1.0

def test_dead_capital_excludes_young_channels_and_hive_members():
    fleet = analyzer.analyze()
    assert fleet.channel_efficiencies["young"].is_dead_capital is False
    assert fleet.channel_efficiencies["member"].is_dead_capital is False

def test_stage_defaults_to_none_when_channel_not_tracked():
    fleet = analyzer.analyze()
    assert fleet.channel_efficiencies["100x1x0"].dead_capital_stage == "none"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_capital_efficiency.py -v`
Expected: FAIL with missing module or missing analyzer API

**Step 3: Write minimal implementation**

```python
@dataclass
class ChannelEfficiency:
    channel_id: str
    rpsd: float
    efficiency_rank: float
    forward_velocity: float
    is_dead_capital: bool
    dead_capital_stage: str

@dataclass
class FleetEfficiency:
    channel_efficiencies: Dict[str, ChannelEfficiency]
    median_rpsd: float
    dead_capital_count: int
    dead_capital_sats: int
    total_deployed_sats: int

class CapitalEfficiencyAnalyzer:
    def analyze(self) -> FleetEfficiency:
        ...
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capital_efficiency.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/capital_efficiency.py tests/test_capital_efficiency.py
git commit -m "feat: add capital efficiency analyzer"
```

### Task 3: Apply efficiency multipliers inside the capex budget engine

**Files:**
- Modify: `modules/capex_budget.py`
- Modify: `tests/test_capex_budget.py`

**Step 1: Write the failing tests**

```python
def test_dead_capital_channel_gets_zero_budget():
    alloc = engine.compute_allocations()
    assert alloc.channel_budgets["100x1x0"].budget_sats == 0

def test_above_median_efficiency_increases_budget():
    alloc = engine.compute_allocations()
    assert alloc.channel_budgets["100x1x0"].budget_sats > 300

def test_missing_efficiency_data_is_neutral():
    alloc = engine.compute_allocations()
    assert alloc.channel_budgets["100x1x0"].budget_sats == 300
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_capex_budget.py -k efficiency -v`
Expected: FAIL because budgets ignore efficiency inputs

**Step 3: Write minimal implementation**

```python
class CapexBudgetEngine:
    def __init__(..., capital_efficiency=None):
        self._capital_efficiency = capital_efficiency

    def _get_efficiency_multiplier(...):
        ...

    def _compute_channel_budget(...):
        budget_msat = int(raw_budget_msat * discount * hive_mult * efficiency_mult)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capex_budget.py -k efficiency -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/capex_budget.py tests/test_capex_budget.py
git commit -m "feat: apply capital efficiency to capex budgets"
```

### Task 4: Add planner dead-capital loser handling and stage advancement

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

```python
def test_dead_capital_channel_enters_fee_reduction_stage_first():
    losers = planner._identify_losers(all_prof, all_flow)
    assert losers[0]["reason"] == "DEAD_CAPITAL"
    assert losers[0]["action"] == "FEE_REDUCE"

def test_dead_capital_advances_to_defibrillate_after_stage_timeout():
    losers = planner._identify_losers(all_prof, all_flow)
    assert losers[0]["action"] == "DEFIBRILLATE"

def test_dead_capital_recovery_clears_stage_tracking():
    planner._identify_losers(all_prof, recovered_flow)
    db.delete_dead_capital_stage.assert_called_once_with("100x1x0")
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_capacity_planner.py -k dead_capital -v`
Expected: FAIL because planner does not consult capital-efficiency state

**Step 3: Write minimal implementation**

```python
def set_capital_efficiency(self, analyzer):
    self._capital_efficiency = analyzer

def _identify_losers(...):
    fleet_efficiency = self._capital_efficiency.analyze() if self._capital_efficiency else None
    dead = fleet_efficiency.channel_efficiencies.get(scid)
    if dead and dead.is_dead_capital:
        ...
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capacity_planner.py -k dead_capital -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: add planner dead capital staging"
```

### Task 5: Replace ROI-only neighbor discovery with efficiency-ranked patron scoring

**Files:**
- Modify: `modules/capacity_planner.py`
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

```python
def test_neighbor_discovery_uses_combined_patron_pool():
    candidates = planner._discover_from_neighbors(all_profitability)
    assert candidates
    assert any(c["source"] == "neighbor" for c in candidates)

def test_second_degree_candidates_are_dampened():
    candidates = planner._discover_from_neighbors(all_profitability)
    assert any(c.get("degree") == 2 and c["score"] < 1.0 for c in candidates)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_capacity_planner.py -k neighbor -v`
Expected: FAIL because current strategy only uses top-ROI patrons and first-degree scoring

**Step 3: Write minimal implementation**

```python
STRATEGY_WEIGHTS["neighbor"] = 0.9

def _discover_from_neighbors(self, all_profitability):
    patrons = self._build_neighbor_patron_pool(...)
    ...
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capacity_planner.py -k neighbor -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: improve planner neighbor discovery"
```

### Task 6: Wire the analyzer into plugin initialization and run focused regression coverage

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `modules/capacity_planner.py`
- Modify: `modules/capex_budget.py`
- Modify: `tests/test_capex_budget.py`
- Modify: `tests/test_capacity_planner.py`
- Modify: `tests/test_database.py`
- Create: `tests/test_capital_efficiency.py`

**Step 1: Write the failing integration test**

```python
def test_capex_engine_and_planner_receive_capital_efficiency():
    ...
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_capex_budget.py tests/test_capacity_planner.py tests/test_database.py tests/test_capital_efficiency.py -v`
Expected: FAIL until plugin wiring and consumers are connected

**Step 3: Write minimal implementation**

```python
capital_efficiency = CapitalEfficiencyAnalyzer(
    profitability_analyzer,
    flow_analyzer,
    database,
    hive_hints,
    config,
)
capex_engine = CapexBudgetEngine(..., capital_efficiency=capital_efficiency)
capacity_planner.set_capital_efficiency(capital_efficiency)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capex_budget.py tests/test_capacity_planner.py tests/test_database.py tests/test_capital_efficiency.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/capacity_planner.py modules/capex_budget.py modules/database.py modules/capital_efficiency.py tests/test_capex_budget.py tests/test_capacity_planner.py tests/test_database.py tests/test_capital_efficiency.py
git commit -m "feat: add capital efficiency planning and budgeting"
```
