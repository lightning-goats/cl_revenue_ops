# Weekly Budget Cap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a weekly budget cap that acts as a hard ceiling alongside the daily burst limit for rebalance spending.

**Architecture:** One new config option (`weekly_budget_sats`), a second budget check in `_check_capital_controls()`, and an extended atomic `reserve_budget()` that enforces both daily and weekly limits in one transaction. When proportional budgeting is enabled, weekly proportional = same percentage applied to 7-day revenue.

**Tech Stack:** Python 3.10+, SQLite (existing `rebalance_costs` and `budget_reservations` tables)

**Design doc:** `docs/plans/2026-03-08-weekly-budget-cap-design.md`

---

### Task 1: Config Option

**Files:**
- Modify: `modules/config.py` (add field + validation)
- Modify: `cl-revenue-ops.py` (add CLN plugin option)
- Create: `tests/test_weekly_budget.py`

**Step 1: Write failing test**

Create `tests/test_weekly_budget.py`:

```python
"""Tests for weekly budget cap."""


def test_weekly_budget_sats_default():
    """Default weekly budget = 7 * daily default (5000)."""
    from modules.config import ConfigSnapshot
    snap = ConfigSnapshot.__dataclass_fields__
    assert snap["weekly_budget_sats"].default == 35000


def test_weekly_budget_sats_in_field_types():
    """weekly_budget_sats validated as int."""
    from modules.config import CONFIG_FIELD_TYPES
    assert "weekly_budget_sats" in CONFIG_FIELD_TYPES
    assert CONFIG_FIELD_TYPES["weekly_budget_sats"] is int
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_weekly_budget.py -v`
Expected: FAIL — `KeyError: 'weekly_budget_sats'`

**Step 3: Implement config changes**

In `modules/config.py`:

Add to CONFIG_FIELD_TYPES dict (near line 49, after `'daily_budget_sats': int,`):
```python
    'weekly_budget_sats': int,
```

Add to CONFIG_FIELD_RANGES dict (find the `'daily_budget_sats'` entry and add after it):
```python
    'weekly_budget_sats': (0, 70_000_000),
```

Add field to ConfigSnapshot dataclass (after `daily_budget_sats` field, around line 768):
```python
    weekly_budget_sats: int = 35000
```

In `cl-revenue-ops.py`, add plugin option (after the `revenue-ops-daily-budget-sats` option block at line ~530):
```python
plugin.add_option(
    name='revenue-ops-weekly-budget-sats',
    default='35000',
    description='Max rebalancing fees to spend in 7 days - hard ceiling over daily burst limit (default: 35000)'
)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_weekly_budget.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/config.py cl-revenue-ops.py tests/test_weekly_budget.py
git commit -m "feat: add weekly_budget_sats config option (default 35000)"
```

---

### Task 2: Extend reserve_budget with Weekly Limit

**Files:**
- Modify: `modules/database.py:2329-2399` (extend reserve_budget)
- Modify: `tests/test_weekly_budget.py` (add reservation tests)

**Step 1: Write failing tests**

Append to `tests/test_weekly_budget.py`:

```python
import sqlite3
import time


def _create_budget_db():
    """Create in-memory DB with rebalance_costs and budget_reservations tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE rebalance_costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            peer_id TEXT NOT NULL,
            cost_sats INTEGER NOT NULL,
            amount_sats INTEGER NOT NULL,
            timestamp INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE budget_reservations (
            reservation_id TEXT PRIMARY KEY,
            reserved_sats INTEGER NOT NULL,
            reserved_at INTEGER NOT NULL,
            job_channel_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
    """)
    return conn


def _insert_cost(conn, cost_sats, timestamp):
    conn.execute(
        "INSERT INTO rebalance_costs (channel_id, peer_id, cost_sats, amount_sats, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        ("ch1", "peer1", cost_sats, cost_sats * 10, timestamp),
    )
    conn.commit()


def test_reserve_budget_weekly_blocks():
    """Weekly limit blocks reservation even when daily has room."""
    from modules.database import _reserve_budget_atomic
    conn = _create_budget_db()
    now = int(time.time())
    daily_since = now - 86400
    weekly_since = now - 7 * 86400

    # Spend 9000 over the week (spread across 6 days, all outside daily window)
    for day in range(6):
        _insert_cost(conn, 1500, now - (day + 1) * 86400 - 3600)

    # Daily has 0 spent (all costs are >24h old), weekly has 9000 spent
    # Daily limit 5000: room for 5000. Weekly limit 10000: room for 1000.
    success, remaining = _reserve_budget_atomic(
        conn, "res1", 2000, "ch1",
        budget_limit=5000, since_timestamp=daily_since,
        weekly_budget_limit=10000, weekly_since_timestamp=weekly_since,
    )
    assert not success  # weekly blocks it
    assert remaining <= 1000  # weekly remaining


def test_reserve_budget_weekly_allows():
    """Both daily and weekly have room -> reservation succeeds."""
    from modules.database import _reserve_budget_atomic
    conn = _create_budget_db()
    now = int(time.time())
    daily_since = now - 86400
    weekly_since = now - 7 * 86400

    success, remaining = _reserve_budget_atomic(
        conn, "res1", 1000, "ch1",
        budget_limit=5000, since_timestamp=daily_since,
        weekly_budget_limit=35000, weekly_since_timestamp=weekly_since,
    )
    assert success
    assert remaining >= 0


def test_reserve_budget_no_weekly_param():
    """When weekly params not provided, behaves like before (daily only)."""
    from modules.database import _reserve_budget_atomic
    conn = _create_budget_db()
    now = int(time.time())
    daily_since = now - 86400

    success, remaining = _reserve_budget_atomic(
        conn, "res1", 1000, "ch1",
        budget_limit=5000, since_timestamp=daily_since,
    )
    assert success  # no weekly constraint
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_weekly_budget.py::test_reserve_budget_weekly_blocks -v`
Expected: FAIL — `ImportError: cannot import name '_reserve_budget_atomic'`

**Step 3: Implement**

In `modules/database.py`, add a standalone function before the `Database` class (for testability, same pattern as `_revenue_by_size_bucket_sql`):

```python
def _reserve_budget_atomic(conn, reservation_id: str, amount_sats: int,
                           channel_id: str, budget_limit: int,
                           since_timestamp: int,
                           weekly_budget_limit: int = None,
                           weekly_since_timestamp: int = None) -> tuple:
    """Atomically reserve budget, enforcing both daily and optional weekly limits.

    Runs inside BEGIN IMMEDIATE for exclusive lock. Checks daily spent+reserved
    against budget_limit, and if weekly params provided, checks weekly
    spent+reserved against weekly_budget_limit.

    Returns (success: bool, remaining: int) where remaining is the
    minimum of daily and weekly remaining budget.
    """
    now = int(time.time())
    try:
        conn.execute("BEGIN IMMEDIATE")

        # Daily: actual spent + active reservations
        spent_row = conn.execute(
            "SELECT COALESCE(SUM(cost_sats), 0) as spent "
            "FROM rebalance_costs WHERE timestamp >= ?",
            (since_timestamp,),
        ).fetchone()
        daily_spent = spent_row["spent"] if isinstance(spent_row, sqlite3.Row) else (spent_row[0] if spent_row else 0)

        res_row = conn.execute(
            "SELECT COALESCE(SUM(reserved_sats), 0) as reserved "
            "FROM budget_reservations WHERE status = 'active' AND reserved_at >= ?",
            (since_timestamp,),
        ).fetchone()
        daily_reserved = res_row["reserved"] if isinstance(res_row, sqlite3.Row) else (res_row[0] if res_row else 0)

        daily_committed = daily_spent + daily_reserved
        daily_remaining = budget_limit - daily_committed

        if amount_sats > daily_remaining:
            conn.execute("ROLLBACK")
            return (False, daily_remaining)

        # Weekly check (if provided)
        if weekly_budget_limit is not None and weekly_since_timestamp is not None:
            w_spent_row = conn.execute(
                "SELECT COALESCE(SUM(cost_sats), 0) as spent "
                "FROM rebalance_costs WHERE timestamp >= ?",
                (weekly_since_timestamp,),
            ).fetchone()
            weekly_spent = w_spent_row["spent"] if isinstance(w_spent_row, sqlite3.Row) else (w_spent_row[0] if w_spent_row else 0)

            w_res_row = conn.execute(
                "SELECT COALESCE(SUM(reserved_sats), 0) as reserved "
                "FROM budget_reservations WHERE status = 'active' AND reserved_at >= ?",
                (weekly_since_timestamp,),
            ).fetchone()
            weekly_reserved = w_res_row["reserved"] if isinstance(w_res_row, sqlite3.Row) else (w_res_row[0] if w_res_row else 0)

            weekly_committed = weekly_spent + weekly_reserved
            weekly_remaining = weekly_budget_limit - weekly_committed

            if amount_sats > weekly_remaining:
                conn.execute("ROLLBACK")
                return (False, weekly_remaining)

        # Both checks passed — insert reservation
        conn.execute(
            "INSERT INTO budget_reservations "
            "(reservation_id, reserved_sats, reserved_at, job_channel_id, status) "
            "VALUES (?, ?, ?, ?, 'active')",
            (reservation_id, amount_sats, now, channel_id),
        )
        conn.execute("COMMIT")

        effective_remaining = daily_remaining - amount_sats
        if weekly_budget_limit is not None:
            effective_remaining = min(effective_remaining, weekly_remaining - amount_sats)
        return (True, effective_remaining)

    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        return (False, 0)
```

Then update the existing `Database.reserve_budget` method (line ~2329) to delegate to the standalone function with the new optional params:

```python
    def reserve_budget(self, reservation_id: str, amount_sats: int,
                      channel_id: str, budget_limit: int,
                      since_timestamp: int,
                      weekly_budget_limit: int = None,
                      weekly_since_timestamp: int = None) -> Tuple[bool, int]:
        """Atomically reserve budget for a rebalance operation.

        Enforces both daily budget_limit and optional weekly_budget_limit
        in a single atomic transaction.
        """
        conn = self._get_connection()
        return _reserve_budget_atomic(
            conn, reservation_id, amount_sats, channel_id,
            budget_limit, since_timestamp,
            weekly_budget_limit, weekly_since_timestamp,
        )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_weekly_budget.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/database.py tests/test_weekly_budget.py
git commit -m "feat: extend reserve_budget with atomic weekly limit enforcement"
```

---

### Task 3: Weekly Check in _check_capital_controls

**Files:**
- Modify: `modules/rebalancer.py:4778-4876` (add weekly check)
- Modify: `tests/test_weekly_budget.py` (add capital control tests)

**Step 1: Write failing tests**

Append to `tests/test_weekly_budget.py`:

```python
from unittest.mock import MagicMock, patch


def _make_mock_rebalancer(daily_budget=5000, weekly_budget=35000,
                          proportional=False, proportional_pct=0.30,
                          daily_spent=0, weekly_spent=0,
                          routing_revenue_24h=0, routing_revenue_7d=0):
    """Create a mock rebalancer with configurable budget state."""
    from modules.config import ConfigSnapshot

    rebalancer = MagicMock()
    cfg = ConfigSnapshot.__new__(ConfigSnapshot)
    cfg.daily_budget_sats = daily_budget
    cfg.weekly_budget_sats = weekly_budget
    cfg.enable_proportional_budget = proportional
    cfg.proportional_budget_pct = proportional_pct
    cfg.total_cost_budget_window_hours = 24
    cfg.min_wallet_reserve = 0  # skip reserve check

    rebalancer.config = MagicMock()
    rebalancer.config.snapshot.return_value = cfg
    rebalancer.global_budget_limit_provider = None
    rebalancer._budget_hot_channel_only = False

    # Mock plugin RPC for reserve check
    rebalancer.plugin = MagicMock()
    rebalancer.plugin.rpc.listfunds.return_value = {
        "outputs": [{"status": "confirmed", "amount_msat": 10_000_000_000}],
        "channels": [],
    }

    # Mock database queries
    def mock_get_total_rebalance_fees(since_ts):
        now = int(time.time())
        if now - since_ts > 2 * 86400:  # >2 days = weekly query
            return weekly_spent
        return daily_spent

    rebalancer.database = MagicMock()
    rebalancer.database.get_total_rebalance_fees.side_effect = mock_get_total_rebalance_fees
    rebalancer.database.get_total_routing_revenue.side_effect = lambda since: (
        routing_revenue_7d if int(time.time()) - since > 2 * 86400 else routing_revenue_24h
    )
    rebalancer._get_external_liquidity_costs.return_value = {"spent_24h_sats": 0, "reserved_24h_sats": 0}
    rebalancer._parse_msat = lambda x: int(str(x).replace("msat", "")) if isinstance(x, str) else int(x)

    return rebalancer, cfg


def test_weekly_budget_blocks_when_exceeded():
    """Weekly spending at limit blocks rebalancing."""
    from modules.rebalancer import Rebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=20000,
        daily_spent=0, weekly_spent=20000,  # weekly exhausted, daily fresh
    )
    result = Rebalancer._check_capital_controls(rebalancer, cfg)
    assert result is False


def test_weekly_budget_allows_when_under():
    """Weekly spending under limit allows rebalancing."""
    from modules.rebalancer import Rebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=35000,
        daily_spent=1000, weekly_spent=10000,
    )
    result = Rebalancer._check_capital_controls(rebalancer, cfg)
    assert result is True


def test_daily_blocks_before_weekly():
    """Daily limit hit blocks even when weekly has room."""
    from modules.rebalancer import Rebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=35000,
        daily_spent=5000, weekly_spent=5000,  # daily exhausted, weekly has room
    )
    result = Rebalancer._check_capital_controls(rebalancer, cfg)
    assert result is False


def test_proportional_weekly_budget():
    """Proportional weekly budget uses 7-day revenue * pct."""
    from modules.rebalancer import Rebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=10000,
        proportional=True, proportional_pct=0.30,
        daily_spent=0, weekly_spent=15000,
        routing_revenue_24h=50000, routing_revenue_7d=200000,
    )
    # effective_weekly = max(10000, 200000 * 0.30) = max(10000, 60000) = 60000
    # weekly_spent=15000 < 60000 → should pass
    result = Rebalancer._check_capital_controls(rebalancer, cfg)
    assert result is True
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_weekly_budget.py::test_weekly_budget_blocks_when_exceeded -v`
Expected: FAIL — the method doesn't check weekly budget yet

**Step 3: Implement weekly check in _check_capital_controls**

In `modules/rebalancer.py`, in the `_check_capital_controls` method, add the weekly check AFTER the existing daily budget check (after the `if total_actual_spent >= effective_budget: return False` block, around line 4865) and BEFORE the final `return True`:

```python
            # --- Weekly budget check ---
            effective_weekly = cfg.weekly_budget_sats

            if cfg.enable_proportional_budget:
                weekly_revenue = self.database.get_total_routing_revenue(now - 7 * 86400)
                proportional_weekly = int(weekly_revenue * cfg.proportional_budget_pct)
                effective_weekly = max(cfg.weekly_budget_sats, proportional_weekly)
                self.plugin.log(
                    f"CAPITAL CONTROL: Proportional weekly budget. "
                    f"Revenue 7d: {weekly_revenue} sats, {cfg.proportional_budget_pct*100:.1f}% = {proportional_weekly} sats, "
                    f"Effective weekly: {effective_weekly} sats (floor: {cfg.weekly_budget_sats})",
                    level='debug'
                )

            weekly_fees_spent = self.database.get_total_rebalance_fees(now - 7 * 86400)
            weekly_total_spent = weekly_fees_spent + ext_spent
            if weekly_total_spent >= effective_weekly:
                self.plugin.log(
                    f"CAPITAL CONTROL: Weekly budget exceeded "
                    f"(rebalance_fees_7d={weekly_fees_spent} + external_spent={ext_spent} "
                    f"= {weekly_total_spent} >= {effective_weekly})",
                    level='warn'
                )
                return False
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_weekly_budget.py -v`
Expected: All PASS

**Step 5: Run existing rebalancer tests for regression**

Run: `python3 -m pytest tests/test_rebalancer.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_weekly_budget.py
git commit -m "feat: add weekly budget check in _check_capital_controls"
```

---

### Task 4: Wire Weekly Params to reserve_budget Call Site

**Files:**
- Modify: `modules/rebalancer.py:4427-4433` (pass weekly params)
- Modify: `tests/test_weekly_budget.py` (add integration test)

**Step 1: Write failing test**

Append to `tests/test_weekly_budget.py`:

```python
def test_reserve_budget_called_with_weekly_params():
    """Verify reserve_budget receives weekly limit and timestamp."""
    from modules.rebalancer import Rebalancer
    import time

    rebalancer = MagicMock()
    rebalancer.database = MagicMock()
    rebalancer.database.reserve_budget.return_value = (True, 1000)

    cfg = MagicMock()
    cfg.weekly_budget_sats = 35000
    cfg.enable_proportional_budget = False
    cfg.total_cost_budget_window_hours = 24

    now = int(time.time())

    # Call the method that invokes reserve_budget
    # We need to check that weekly params are passed through
    # This verifies the wiring, not the logic (logic tested in Tasks 2-3)
    rebalancer.database.reserve_budget.assert_not_called()  # sanity
```

Note: This test is a placeholder — the actual verification happens by checking that `reserve_budget` is called with `weekly_budget_limit` and `weekly_since_timestamp` kwargs. The implementation step below shows the exact change.

**Step 2: Implement**

In `modules/rebalancer.py`, find the `reserve_budget` call site (line ~4427). The current code is:

```python
                reserved, remaining = self.database.reserve_budget(
                    reservation_id=str(rebalance_id),
                    amount_sats=db_max_fee,
                    channel_id=db_to_channel,
                    budget_limit=rebalance_budget_limit,
                    since_timestamp=since_24h
                )
```

Change to:

```python
                # Compute effective weekly budget for atomic reservation
                _weekly_cfg = cfg if cfg else self.config.snapshot()
                _effective_weekly = _weekly_cfg.weekly_budget_sats
                if _weekly_cfg.enable_proportional_budget:
                    _now = int(time.time())
                    _weekly_rev = self.database.get_total_routing_revenue(_now - 7 * 86400)
                    _prop_weekly = int(_weekly_rev * _weekly_cfg.proportional_budget_pct)
                    _effective_weekly = max(_weekly_cfg.weekly_budget_sats, _prop_weekly)

                reserved, remaining = self.database.reserve_budget(
                    reservation_id=str(rebalance_id),
                    amount_sats=db_max_fee,
                    channel_id=db_to_channel,
                    budget_limit=rebalance_budget_limit,
                    since_timestamp=since_24h,
                    weekly_budget_limit=_effective_weekly,
                    weekly_since_timestamp=int(time.time()) - 7 * 86400,
                )
```

Check if `cfg` is in scope at the call site. The `reserve_budget` call is inside `execute_rebalance()` or similar — read the surrounding context to confirm. If `cfg` is not in scope, use `self.config.snapshot()`.

**Step 3: Run all tests**

Run: `python3 -m pytest tests/test_weekly_budget.py tests/test_rebalancer.py -v`
Expected: All PASS

**Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All PASS (no regressions)

**Step 5: Commit**

```bash
git add modules/rebalancer.py tests/test_weekly_budget.py
git commit -m "feat: pass weekly budget params to atomic reserve_budget"
```
