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


def test_weekly_budget_sats_range():
    """weekly_budget_sats has valid range constraint."""
    from modules.config import CONFIG_FIELD_RANGES
    assert "weekly_budget_sats" in CONFIG_FIELD_RANGES
    low, high = CONFIG_FIELD_RANGES["weekly_budget_sats"]
    assert low == 0
    assert high == 70_000_000


# --- Task 2: reserve_budget with weekly limit ---

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


# --- Task 3: _check_capital_controls weekly check ---

from unittest.mock import MagicMock, patch


def _make_mock_rebalancer(daily_budget=5000, weekly_budget=35000,
                          proportional=False, proportional_pct=0.30,
                          daily_spent=0, weekly_spent=0,
                          routing_revenue_24h=0, routing_revenue_7d=0):
    """Create a mock rebalancer with configurable budget state."""
    from modules.config import ConfigSnapshot

    rebalancer = MagicMock()
    cfg = ConfigSnapshot.__new__(ConfigSnapshot)
    # ConfigSnapshot is frozen, use object.__setattr__ to bypass
    object.__setattr__(cfg, 'daily_budget_sats', daily_budget)
    object.__setattr__(cfg, 'weekly_budget_sats', weekly_budget)
    object.__setattr__(cfg, 'enable_proportional_budget', proportional)
    object.__setattr__(cfg, 'proportional_budget_pct', proportional_pct)
    object.__setattr__(cfg, 'total_cost_budget_window_hours', 24)
    object.__setattr__(cfg, 'min_wallet_reserve', 0)  # skip reserve check

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
    from modules.rebalancer import EVRebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=20000,
        daily_spent=0, weekly_spent=20000,  # weekly exhausted, daily fresh
    )
    result = EVRebalancer._check_capital_controls(rebalancer, cfg)
    assert result is False


def test_weekly_budget_allows_when_under():
    """Weekly spending under limit allows rebalancing."""
    from modules.rebalancer import EVRebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=35000,
        daily_spent=1000, weekly_spent=10000,
    )
    result = EVRebalancer._check_capital_controls(rebalancer, cfg)
    assert result is True


def test_daily_blocks_before_weekly():
    """Daily limit hit blocks even when weekly has room."""
    from modules.rebalancer import EVRebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=35000,
        daily_spent=5000, weekly_spent=5000,  # daily exhausted, weekly has room
    )
    result = EVRebalancer._check_capital_controls(rebalancer, cfg)
    assert result is False


def test_proportional_weekly_budget():
    """Proportional weekly budget uses 7-day revenue * pct."""
    from modules.rebalancer import EVRebalancer
    rebalancer, cfg = _make_mock_rebalancer(
        daily_budget=5000, weekly_budget=10000,
        proportional=True, proportional_pct=0.30,
        daily_spent=0, weekly_spent=15000,
        routing_revenue_24h=50000, routing_revenue_7d=200000,
    )
    # effective_weekly = max(10000, 200000 * 0.30) = max(10000, 60000) = 60000
    # weekly_spent=15000 < 60000 → should pass
    result = EVRebalancer._check_capital_controls(rebalancer, cfg)
    assert result is True


# --- Task 4: Wire weekly params to reserve_budget call site ---


def test_reserve_budget_called_with_weekly_params():
    """Verify reserve_budget receives weekly limit and timestamp."""
    from modules.rebalancer import EVRebalancer
    import inspect

    # Verify the wiring exists by checking the source code contains
    # the weekly params in the reserve_budget call.
    source = inspect.getsource(EVRebalancer)
    assert "weekly_budget_limit=" in source
    assert "weekly_since_timestamp=" in source
