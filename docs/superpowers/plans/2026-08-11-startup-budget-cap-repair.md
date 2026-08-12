# Startup Budget Cap Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure startup contradiction recovery never widens persisted daily or weekly rebalance spending authority.

**Architecture:** Keep enforcement at `Config.load_overrides()`, the boundary that first combines individually valid persisted rows. Preserve the weekly ceiling and clamp the daily cap down before any immutable `ConfigSnapshot` or rebalance reservation consumer can observe the crossed pair.

**Tech Stack:** Python 3.12, dataclasses, SQLite-backed `Database`, pytest 9.

## Global Constraints

- No startup repair may increase `daily_budget_sats` or `weekly_budget_sats`.
- Ordered budget pairs and `Config.update_runtime()` rejection semantics remain unchanged.
- No live CLN action RPC, deployment, or plugin restart is permitted.
- No Sling, Hive, Mycelium, LN+, Boltz, or capacity-planner authority may be introduced.
- Tests run from the worktree-local environment installed with `pip install --require-hashes -r requirements.lock`.

---

### Task 1: Encode fail-closed startup behavior

**Files:**
- Modify: `tests/test_config_contradictions.py`
- Modify: `tests/test_surface_reduction_phase_b.py`

**Interfaces:**
- Consumes: `Config.load_overrides(database) -> list[str]` and `Database.reserve_budget(...) -> tuple[bool, int]`.
- Produces: regression coverage for in-memory repair and the real persisted-row-to-reservation boundary.

- [ ] **Step 1: Replace upward-repair expectations with the approved clamp**

```python
def test_daily_budget_above_weekly_clamps_daily_to_weekly(self):
    cfg, warnings = _load({"daily_budget_sats": "5000",
                           "weekly_budget_sats": "1000"})
    assert _matching(warnings, "Contradictory", "daily_budget_sats",
                     "weekly_budget_sats", "repaired daily_budget_sats")
    assert cfg.daily_budget_sats == 1000
    assert cfg.weekly_budget_sats == 1000
```

- [ ] **Step 2: Add zero-cap and malformed-input safety cases**

```python
def test_zero_weekly_cap_clamps_daily_to_zero(self):
    cfg, _ = _load({"daily_budget_sats": "5000",
                    "weekly_budget_sats": "0"})
    assert (cfg.daily_budget_sats, cfg.weekly_budget_sats) == (0, 0)

def test_malformed_daily_override_cannot_bypass_restrictive_weekly_cap(self):
    cfg, warnings = _load({"daily_budget_sats": "not-an-int",
                           "weekly_budget_sats": "1000"})
    assert _matching(warnings, "Override conversion failed",
                     "daily_budget_sats")
    assert (cfg.daily_budget_sats, cfg.weekly_budget_sats) == (1000, 1000)
```

- [ ] **Step 3: Add the real persistence and reservation regression**

```python
def test_startup_crossed_budget_never_widens_reservation_authority(tmp_path):
    db = Database(str(tmp_path / "crossed-budget.db"), MagicMock())
    db.initialize()
    db.set_config_override("daily_budget_sats", "9000")
    db.set_config_override("weekly_budget_sats", "1000")

    cfg = Config()
    cfg.load_overrides(db)
    now = int(time.time())
    ok, remaining = db.reserve_budget(
        reservation_id="startup-cap-regression",
        amount_sats=5000,
        channel_id="test-channel",
        budget_limit=cfg.daily_budget_sats,
        since_timestamp=now - 86400,
        weekly_budget_limit=cfg.weekly_budget_sats,
        weekly_since_timestamp=now - 7 * 86400,
    )

    assert ok is False
    assert remaining == 1000
```

- [ ] **Step 4: Run the regression tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_config_contradictions.py tests/test_surface_reduction_phase_b.py -q`

Expected: failures show vulnerable code produced `daily=5000, weekly=5000` (or admitted the 5000-sat reservation) instead of preserving the 1000-sat weekly ceiling.

- [ ] **Step 5: Commit the failing regression proof**

```bash
git add tests/test_config_contradictions.py tests/test_surface_reduction_phase_b.py
git commit -m "test(security): require fail-closed startup budget repair"
```

### Task 2: Implement the minimal startup clamp

**Files:**
- Modify: `modules/config.py:820-837`
- Test: `tests/test_config_contradictions.py`
- Test: `tests/test_surface_reduction_phase_b.py`

**Interfaces:**
- Consumes: individually converted and range-validated `Config` fields after persisted override loading.
- Produces: an ordered `(daily_budget_sats, weekly_budget_sats)` pair where neither cap exceeds the persisted weekly ceiling because of repair.

- [ ] **Step 1: Replace upward ceiling repair with a downward daily clamp**

```python
if self.daily_budget_sats > self.weekly_budget_sats:
    self._override_warnings.append(
        f"Contradictory settings: daily_budget_sats "
        f"({self.daily_budget_sats}) > weekly_budget_sats "
        f"({self.weekly_budget_sats}); repaired daily_budget_sats "
        f"to {self.weekly_budget_sats}")
    self.daily_budget_sats = self.weekly_budget_sats
```

- [ ] **Step 2: Remove obsolete planner-repair comments**

Delete the comments after the startup budget branch and the empty planner comment before `old_value = getattr(self, key)` in `update_runtime()`.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_config_contradictions.py tests/test_surface_reduction_phase_b.py -q`

Expected: all focused tests pass.

- [ ] **Step 4: Confirm the test detects regression by temporarily reversing the clamp**

Temporarily restore the vulnerable assignment, run the real reservation test and observe failure, then restore the security fix and rerun it green. Do not commit the temporary reversal.

- [ ] **Step 5: Commit the production fix**

```bash
git add modules/config.py
git commit -m "fix(security): keep startup budget repair fail-closed"
```

### Task 3: Verify closure and compatibility

**Files:**
- Verify: `modules/config.py`
- Verify: `modules/rebalance_engine_v2.py`
- Verify: `modules/rebalancer.py`
- Verify: `modules/database.py`

**Interfaces:**
- Consumes: final branch diff and the original source-to-sink finding.
- Produces: ordered verification evidence and an independently reviewed Tier 1 branch.

- [ ] **Step 1: Inspect scope and syntax**

Run: `git diff --check origin/main...HEAD`

Run: `.venv/bin/python -m py_compile modules/config.py`

Expected: both exit zero.

- [ ] **Step 2: Run focused security and compatibility tests**

Run: `.venv/bin/python -m pytest tests/test_config_contradictions.py tests/test_surface_reduction_phase_b.py tests/test_reservations_unification.py tests/test_p4_017_active_reservation_window.py -q`

Expected: all selected tests pass.

- [ ] **Step 3: Run the full hash-locked suite**

Run: `.venv/bin/python -m pytest -q`

Expected: zero failures; only documented environment skips and scheduled expected failures remain.

- [ ] **Step 4: Perform change-aware bypass review**

Verify statically that startup cannot assign a larger weekly cap, every current rebalance reservation path reads the clamped snapshot value, valid ordered pairs remain untouched, `update_runtime()` still rejects crossed pairs, and no retired planner code or action RPC surface was introduced.

- [ ] **Step 5: Obtain independent Tier 1 review**

Give the verifier the Hexmem task id and base/head commits. The verifier must independently confirm security closure, compatibility, test adequacy, no Sling, and no live action RPC use, then pass or fail the `review` criterion.
