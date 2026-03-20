"""
Database module for cl-revenue-ops

Handles SQLite persistence for:
- Channel flow states and history
- Fee controller state
- Fee change history
- Rebalance history

Thread Safety:
- Uses threading.local() to provide each thread with its own SQLite connection
- Prevents race conditions during concurrent writes (e.g., Rebalancer + Fee Controller)
"""

import sqlite3
import os
import time
import json
import math
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .utils import normalize_scid


# Size bucket labels matching fee_controller.SIZE_BUCKET_LABELS
_SIZE_BUCKET_LABELS = ("micro", "small", "medium", "large", "whale")


def _revenue_by_size_bucket_sql(conn, channel_id: str, window_days: int = 7) -> dict:
    """Query forwards table and return {label: total_fee_msat} grouped by payment size bucket.

    Note: Only accurate within the forwards retention window (~8 days).
    Older forwards are pruned by cleanup_old_data() and rolled up into
    daily_forwarding_stats which lacks per-payment size granularity.

    This is a standalone function that accepts a raw sqlite3 connection so it
    can be tested with in-memory databases without instantiating Database.
    """
    cutoff = int(time.time()) - window_days * 86400
    cursor = conn.execute(
        """
        SELECT
            CASE
                WHEN out_msat / 1000 < 10000 THEN 0
                WHEN out_msat / 1000 < 100000 THEN 1
                WHEN out_msat / 1000 < 500000 THEN 2
                WHEN out_msat / 1000 < 5000000 THEN 3
                ELSE 4
            END AS bucket_idx,
            SUM(fee_msat) AS total_fee
        FROM forwards
        WHERE out_channel = ? AND timestamp >= ?
        GROUP BY bucket_idx
        """,
        (channel_id, cutoff),
    )
    result = {label: 0 for label in _SIZE_BUCKET_LABELS}
    for row in cursor:
        idx, total_fee = row
        if 0 <= idx < len(_SIZE_BUCKET_LABELS):
            result[_SIZE_BUCKET_LABELS[idx]] = int(total_fee or 0)
    return result


def _reserve_budget_atomic(conn, reservation_id: str, amount_sats: int,
                           channel_id: str, budget_limit: int,
                           since_timestamp: int,
                           weekly_budget_limit: int = None,
                           weekly_since_timestamp: int = None) -> tuple:
    """Atomically reserve budget, enforcing both daily and optional weekly limits.

    This is a standalone function that accepts a raw sqlite3 connection so it
    can be tested with in-memory databases without instantiating Database.

    Args:
        conn: sqlite3 connection (may have row_factory=sqlite3.Row or not)
        reservation_id: Unique ID for this reservation
        amount_sats: Amount to reserve
        channel_id: Channel this reservation is for
        budget_limit: Maximum daily budget in sats
        since_timestamp: Start of daily budget period (e.g., 24h ago)
        weekly_budget_limit: Optional maximum weekly budget in sats
        weekly_since_timestamp: Optional start of weekly budget period (e.g., 7d ago)

    Returns:
        Tuple of (success: bool, remaining_budget: int)
    """
    now = int(time.time())

    conn.execute("BEGIN IMMEDIATE")

    # --- Daily check ---
    spent_row = conn.execute(
        "SELECT COALESCE(SUM(cost_sats), 0) FROM rebalance_costs WHERE timestamp >= ?",
        (since_timestamp,),
    ).fetchone()
    daily_spent = spent_row[0] if spent_row else 0

    reserved_row = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats), 0) FROM budget_reservations "
        "WHERE status = 'active' AND reserved_at >= ?",
        (since_timestamp,),
    ).fetchone()
    daily_reserved = reserved_row[0] if reserved_row else 0

    daily_committed = daily_spent + daily_reserved
    daily_remaining = budget_limit - daily_committed

    if amount_sats > daily_remaining:
        conn.execute("ROLLBACK")
        return (False, daily_remaining)

    # --- Weekly check (optional) ---
    if weekly_budget_limit is not None and weekly_since_timestamp is not None:
        weekly_spent_row = conn.execute(
            "SELECT COALESCE(SUM(cost_sats), 0) FROM rebalance_costs WHERE timestamp >= ?",
            (weekly_since_timestamp,),
        ).fetchone()
        weekly_spent = weekly_spent_row[0] if weekly_spent_row else 0

        weekly_reserved_row = conn.execute(
            "SELECT COALESCE(SUM(reserved_sats), 0) FROM budget_reservations "
            "WHERE status = 'active' AND reserved_at >= ?",
            (weekly_since_timestamp,),
        ).fetchone()
        weekly_reserved = weekly_reserved_row[0] if weekly_reserved_row else 0

        weekly_committed = weekly_spent + weekly_reserved
        weekly_remaining = weekly_budget_limit - weekly_committed

        if amount_sats > weekly_remaining:
            conn.execute("ROLLBACK")
            return (False, weekly_remaining)
    else:
        weekly_remaining = None

    # --- Insert reservation ---
    conn.execute(
        "INSERT INTO budget_reservations "
        "(reservation_id, reserved_sats, reserved_at, job_channel_id, status) "
        "VALUES (?, ?, ?, ?, 'active')",
        (reservation_id, amount_sats, now, channel_id),
    )
    conn.execute("COMMIT")

    # Return remaining as min(daily, weekly) - amount_sats
    after_daily = daily_remaining - amount_sats
    if weekly_remaining is not None:
        after_weekly = weekly_remaining - amount_sats
        return (True, min(after_daily, after_weekly))
    return (True, after_daily)


def _hourly_forward_histogram_sql(conn, channel_id: str, window_days: int = 7) -> list:
    """Compute per-hour flow histogram for a channel from the forwards table.

    Returns a list of 24 dicts (one per hour UTC), each with:
        out_sats: average outflow sats per day at this hour
        in_sats: average inflow sats per day at this hour
        count: average forward count per day at this hour

    Uses a 7-day window. Averages are per-day to normalize for varying window sizes.
    """
    now = int(time.time())
    since = now - window_days * 86400

    # Count distinct days with data for normalization
    day_count_row = conn.execute(
        "SELECT COUNT(DISTINCT timestamp / 86400) FROM forwards "
        "WHERE timestamp >= ? AND (in_channel = ? OR out_channel = ?)",
        (since, channel_id, channel_id),
    ).fetchone()
    days_with_data = max(1, day_count_row[0] if day_count_row else 1)

    rows = conn.execute("""
        SELECT
            hour_utc,
            SUM(out_sats) as total_out,
            SUM(in_sats) as total_in,
            SUM(cnt) as total_count
        FROM (
            SELECT
                CAST(((timestamp % 86400) / 3600) AS INTEGER) AS hour_utc,
                0 AS out_sats,
                SUM(in_msat) / 1000 AS in_sats,
                COUNT(*) AS cnt
            FROM forwards
            WHERE timestamp >= ? AND in_channel = ?
            GROUP BY hour_utc

            UNION ALL

            SELECT
                CAST(((timestamp % 86400) / 3600) AS INTEGER) AS hour_utc,
                SUM(out_msat) / 1000 AS out_sats,
                0 AS in_sats,
                COUNT(*) AS cnt
            FROM forwards
            WHERE timestamp >= ? AND out_channel = ?
            GROUP BY hour_utc
        )
        GROUP BY hour_utc
        ORDER BY hour_utc
    """, (since, channel_id, since, channel_id)).fetchall()

    # Initialize 24 hourly buckets
    result = [{"out_sats": 0, "in_sats": 0, "count": 0} for _ in range(24)]
    for row in rows:
        h = int(row[0]) % 24
        result[h]["out_sats"] = int(row[1] or 0) // days_with_data
        result[h]["in_sats"] = int(row[2] or 0) // days_with_data
        result[h]["count"] = int(row[3] or 0) // days_with_data

    return result


class Database:
    """
    SQLite database manager for the Revenue Operations plugin.
    
    Provides persistence for:
    - Channel states (source/sink/balanced classification)
    - Flow metrics history
    - Fee controller state
    - Fee change audit log
    - Rebalance history
    
    Thread Safety:
    - Each thread gets its own isolated SQLite connection via threading.local()
    - WAL mode enabled for better concurrent read/write performance
    """
    
    def __init__(self, db_path: str, plugin):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file
            plugin: Reference to the pyln Plugin for logging
        """
        self.db_path = os.path.expanduser(db_path)
        self.plugin = plugin
        # Thread-local storage for connections (Thread Safety)
        self._local = threading.local()
        # EH-6: Track all thread connections for shutdown cleanup
        self._thread_connections: list = []
        self._thread_conn_lock = threading.Lock()
        
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create a thread-local database connection.
        
        Each thread gets its own isolated connection to prevent race conditions
        during concurrent database operations (e.g., when Rebalancer and Fee
        Controller run simultaneously on different timer threads).
        
        Returns:
            sqlite3.Connection: Thread-local database connection
        """
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            return self._local.conn

        # Ensure directory exists (guard against bare filename with no directory)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Create new connection for this thread in a local variable so that
        # a PRAGMA failure does not leave a half-configured connection in
        # thread-local storage (connection leak prevention).
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row

            # Enable Write-Ahead Logging for better multi-thread concurrency
            # WAL allows readers and writers to operate concurrently
            wal_result = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
            if wal_result and wal_result[0] != 'wal':
                self.plugin.log(
                    f"Database: WAL mode not enabled (got '{wal_result[0]}'), "
                    f"falling back to default journal mode",
                    level='warn'
                )

            # Reduce "database is locked" errors under contention
            conn.execute("PRAGMA busy_timeout=5000;")
            # Reasonable durability/performance tradeoff for WAL mode
            conn.execute("PRAGMA synchronous=NORMAL;")
            # D3 FIX: Enforce foreign key constraints (defined but previously unenforced)
            conn.execute("PRAGMA foreign_keys=ON;")
            # Enable incremental auto-vacuum so PRAGMA incremental_vacuum works.
            # Only takes effect on a fresh (empty) database; harmless no-op otherwise.
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")
        except Exception:
            conn.close()
            raise

        # EH-6: Track connection for shutdown cleanup — only after full success
        with self._thread_conn_lock:
            self._thread_connections.append(conn)
        self._local.conn = conn
        self.plugin.log(
            f"Database: Created new thread-local connection (thread={threading.current_thread().name})",
            level='debug'
        )
        return conn

    def close_connection(self) -> None:
        """
        Close the thread-local database connection.

        MAJOR-11 FIX: Explicit cleanup to prevent connection leaks in
        long-running plugins with many threads.

        Should be called when a thread is about to exit or during shutdown.
        """
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            conn = self._local.conn
            try:
                conn.close()
                self.plugin.log(
                    f"Database: Closed thread-local connection (thread={threading.current_thread().name})",
                    level='debug'
                )
            except Exception as e:
                self.plugin.log(f"Error closing connection: {e}", level='debug')
            finally:
                self._local.conn = None
                # L-21: Remove from tracking list to prevent leaks
                with self._thread_conn_lock:
                    try:
                        self._thread_connections.remove(conn)
                    except ValueError:
                        pass

    def close_all_connections(self) -> None:
        """
        EH-6: Close all known thread-local connections during shutdown.

        Daemon threads don't run atexit handlers, so we track connections
        and close them explicitly from the shutdown handler.
        """
        # Checkpoint WAL BEFORE closing connections so that the main database
        # file is up-to-date and the WAL doesn't grow across restarts.
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                self._local.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self.plugin.log("Database: Shutdown checkpoint complete", level='info')
            except Exception as e:
                self.plugin.log(f"Error during shutdown checkpoint: {e}", level='warn')

        with self._thread_conn_lock:
            connections = list(self._thread_connections)
            self._thread_connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass
        # Also close this thread's connection
        self.close_connection()

    def close_main_connection(self) -> None:
        """
        Close the main thread's connection and checkpoint WAL.

        Should be called during plugin shutdown to ensure clean state.
        """
        conn = getattr(self._local, 'conn', None)
        if conn:
            try:
                # Checkpoint WAL to ensure all writes are in main database
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.close()
                self.plugin.log("Database: Shutdown checkpoint complete", level='info')
            except Exception as e:
                self.plugin.log(f"Error during shutdown: {e}", level='warn')
            finally:
                self._local.conn = None

    # =========================================================================
    # Security: Input Validation Constants and Methods (Accounting v2.0)
    # =========================================================================

    # Maximum reasonable fee for a single on-chain operation (50,000 sats ~ $50)
    MAX_FEE_SATS: int = 50000
    # Maximum reasonable capacity amount (100 BTC in sats)
    MAX_AMOUNT_SATS: int = 10_000_000_000
    # Channel ID format: short_channel_id like "123x456x789" or "123456x789x0"
    CHANNEL_ID_PATTERN = r'^\d+x\d+x\d+$'
    # Peer ID format: 66 hex characters (33 bytes public key)
    PEER_ID_PATTERN = r'^[0-9a-fA-F]{66}$'

    def _validate_channel_id(self, channel_id: str) -> bool:
        """
        Validate channel ID format (Security: input validation).

        Args:
            channel_id: The channel short ID to validate

        Returns:
            True if valid format, False otherwise
        """
        import re
        if not channel_id or not isinstance(channel_id, str):
            return False
        return bool(re.match(self.CHANNEL_ID_PATTERN, channel_id))

    def _validate_peer_id(self, peer_id: str) -> bool:
        """
        Validate peer ID format (Security: input validation).

        Args:
            peer_id: The peer node ID to validate

        Returns:
            True if valid format, False otherwise
        """
        import re
        if not peer_id or not isinstance(peer_id, str):
            return False
        return bool(re.match(self.PEER_ID_PATTERN, peer_id))

    def _sanitize_fee(self, fee_sats: int, field_name: str = "fee") -> int:
        """
        Sanitize fee value with bounds checking (Security: input validation).

        Args:
            fee_sats: The fee value in satoshis
            field_name: Name of the field for logging

        Returns:
            Sanitized fee value (clamped to valid range)
        """
        if not isinstance(fee_sats, (int, float)):
            self.plugin.log(
                f"Security: Invalid {field_name} type {type(fee_sats)}, defaulting to 0",
                level='warn'
            )
            return 0
        if isinstance(fee_sats, float) and (math.isnan(fee_sats) or math.isinf(fee_sats)):
            self.plugin.log(
                f"Security: Non-finite {field_name} value {fee_sats}, defaulting to 0",
                level='warn'
            )
            return 0
        fee_sats = int(fee_sats)
        if fee_sats < 0:
            self.plugin.log(
                f"Security: Negative {field_name} {fee_sats}, clamping to 0",
                level='warn'
            )
            return 0
        if fee_sats > self.MAX_FEE_SATS:
            self.plugin.log(
                f"Security: Excessive {field_name} {fee_sats}, clamping to {self.MAX_FEE_SATS}",
                level='warn'
            )
            return self.MAX_FEE_SATS
        return fee_sats

    def _sanitize_amount(self, amount_sats: int, field_name: str = "amount") -> int:
        """
        Sanitize amount value with bounds checking (Security: input validation).

        Args:
            amount_sats: The amount value in satoshis
            field_name: Name of the field for logging

        Returns:
            Sanitized amount value (clamped to valid range)
        """
        if not isinstance(amount_sats, (int, float)):
            self.plugin.log(
                f"Security: Invalid {field_name} type {type(amount_sats)}, defaulting to 0",
                level='warn'
            )
            return 0
        if isinstance(amount_sats, float) and (math.isnan(amount_sats) or math.isinf(amount_sats)):
            self.plugin.log(
                f"Security: Non-finite {field_name} value {amount_sats}, defaulting to 0",
                level='warn'
            )
            return 0
        amount_sats = int(amount_sats)
        # Allow negative amounts, but clamp magnitude
        if abs(amount_sats) > self.MAX_AMOUNT_SATS:
            sign = 1 if amount_sats >= 0 else -1
            self.plugin.log(
                f"Security: Excessive {field_name} {amount_sats}, clamping to {sign * self.MAX_AMOUNT_SATS}",
                level='warn'
            )
            return sign * self.MAX_AMOUNT_SATS
        return amount_sats

    def initialize(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()

        # Schema version tracking for migrations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 1,
                updated_at INTEGER
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO schema_version (id, version, updated_at) VALUES (1, 1, ?)
        """, (int(time.time()),))

        # Channel states table - stores current classification
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_states (
                channel_id TEXT PRIMARY KEY,
                peer_id TEXT NOT NULL,
                state TEXT NOT NULL,  -- 'source', 'sink', 'balanced'
                flow_ratio REAL NOT NULL,
                sats_in INTEGER NOT NULL,
                sats_out INTEGER NOT NULL,
                capacity INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)
        
        
        # Fee Strategy State table for fee controller
        # Stores state for the revenue-maximizing Perturb & Observe algorithm
        # UPDATED: Uses last_revenue_rate (REAL) for rate-based feedback instead of
        # last_revenue_sats to measure revenue per hour since last fee change.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fee_strategy_state (
                channel_id TEXT PRIMARY KEY,
                last_revenue_rate REAL NOT NULL DEFAULT 0.0,
                last_fee_ppm INTEGER NOT NULL DEFAULT 0,
                trend_direction INTEGER NOT NULL DEFAULT 1,  -- 1 = increase, -1 = decrease
                step_ppm INTEGER NOT NULL DEFAULT 50,  -- Current step size (for dampening)
                consecutive_same_direction INTEGER NOT NULL DEFAULT 0,
                last_update INTEGER NOT NULL DEFAULT 0,
                last_broadcast_fee_ppm INTEGER NOT NULL DEFAULT 0,
                is_sleeping INTEGER NOT NULL DEFAULT 0,
                sleep_until INTEGER NOT NULL DEFAULT 0,
                stable_cycles INTEGER NOT NULL DEFAULT 0
            )
        """)
        
        # Fee changes audit log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fee_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                peer_id TEXT NOT NULL,
                old_fee_ppm INTEGER NOT NULL,
                new_fee_ppm INTEGER NOT NULL,
                reason TEXT,
                manual INTEGER NOT NULL DEFAULT 0,
                timestamp INTEGER NOT NULL
            )
        """)
        
        # Rebalance history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rebalance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_channel TEXT NOT NULL,
                to_channel TEXT NOT NULL,
                amount_sats INTEGER NOT NULL,
                max_fee_sats INTEGER NOT NULL,
                actual_fee_sats INTEGER,
                expected_profit_sats INTEGER NOT NULL,
                actual_profit_sats INTEGER,
                status TEXT NOT NULL,  -- 'pending', 'success', 'failed'
                rebalance_type TEXT NOT NULL DEFAULT 'normal',  -- 'normal', 'diagnostic'
                error_message TEXT,
                timestamp INTEGER NOT NULL
            )
        """)
        
        # Real-time forwards tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forwards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                in_channel TEXT NOT NULL,
                out_channel TEXT NOT NULL,
                in_msat INTEGER NOT NULL,
                out_msat INTEGER NOT NULL,
                fee_msat INTEGER NOT NULL,
                resolution_time REAL DEFAULT 0,
                timestamp INTEGER NOT NULL,
                resolved_time INTEGER DEFAULT 0
            )
        """)
        # Ensure forwards schema is idempotent and restart-safe
        self._migrate_forwards_schema(conn)

        
        # Channel open costs tracking (for profitability analysis)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_costs (
                channel_id TEXT PRIMARY KEY,
                peer_id TEXT NOT NULL,
                open_cost_sats INTEGER NOT NULL DEFAULT 0,
                capacity_sats INTEGER NOT NULL,
                opened_at INTEGER NOT NULL
            )
        """)
        
        # Rebalance costs tracking (cumulative per channel)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rebalance_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                peer_id TEXT NOT NULL,
                cost_sats INTEGER NOT NULL,
                amount_sats INTEGER NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        
        # Channel failure tracking for adaptive backoff (persisted across restarts)
        # This prevents "retry storms" after plugin restart
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_failures (
                channel_id TEXT PRIMARY KEY,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_failure_time INTEGER NOT NULL DEFAULT 0,
                last_attempted_ppm INTEGER NOT NULL DEFAULT 0,
                last_attempted_amount INTEGER NOT NULL DEFAULT 0,
                last_error_type TEXT NOT NULL DEFAULT ''
            )
        """)
        
        # Peer reputation tracking for routing success rates
        # Used to evaluate peer reliability for traffic intelligence
        conn.execute("""
            CREATE TABLE IF NOT EXISTS peer_reputation (
                peer_id TEXT PRIMARY KEY,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_update INTEGER NOT NULL DEFAULT 0
            )
        """)
        
        # Peer connection history for uptime/stability tracking
        # Logs connect/disconnect events to calculate historical uptime
        conn.execute("""
            CREATE TABLE IF NOT EXISTS peer_connection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        
        # Lifetime aggregates - stores cumulative totals before pruning
        # This ensures revenue-history remains accurate even after old forwards are deleted
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lifetime_aggregates (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                pruned_revenue_msat INTEGER NOT NULL DEFAULT 0,
                pruned_forward_count INTEGER NOT NULL DEFAULT 0,
                last_prune_timestamp INTEGER NOT NULL DEFAULT 0
            )
        """)
        # ensure exactly one row exists
        conn.execute("""
            INSERT OR IGNORE INTO lifetime_aggregates (id, pruned_revenue_msat, pruned_forward_count, last_prune_timestamp)
            VALUES (1, 0, 0, 0)
        """)

        # Channel exploration flags. Legacy rows may still carry probe_type='zero_fee',
        # but the fee controller now interprets the flag as bounded low-fee exploration.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_probes (
                channel_id TEXT PRIMARY KEY,
                probe_type TEXT NOT NULL,  -- legacy 'zero_fee' or current 'bounded_low_fee'
                started_at INTEGER NOT NULL
            )
        """)
        
        # Ignored peers table (Blacklist)
        # Prevents cl-revenue-ops from managing fees or rebalancing for specific peers
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ignored_peers (
                peer_id TEXT PRIMARY KEY,
                reason TEXT,
                ignored_at INTEGER NOT NULL
            )
        """)
        
        # Peer policies table (v1.4: Policy-Driven Architecture)
        # Replaces ignored_peers with full strategy/mode control
        conn.execute("""
            CREATE TABLE IF NOT EXISTS peer_policies (
                peer_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL DEFAULT 'dynamic',
                rebalance_mode TEXT NOT NULL DEFAULT 'enabled',
                fee_ppm_target INTEGER,
                tags TEXT,
                updated_at INTEGER NOT NULL
            )
        """)
        
        # Hot-channel protection peer overrides (operator-managed)
        # Peers in this table are always eligible for hot-channel protection logic.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hot_channel_protection_overrides (
                peer_id TEXT PRIMARY KEY,
                added_at INTEGER NOT NULL,
                note TEXT,
                min_depletion_trigger_pct REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hot_channel_protection_overrides_added_at ON hot_channel_protection_overrides(added_at)")
        try:
            conn.execute("ALTER TABLE hot_channel_protection_overrides ADD COLUMN min_depletion_trigger_pct REAL")
            self.plugin.log("Added min_depletion_trigger_pct column to hot_channel_protection_overrides")
        except sqlite3.OperationalError:
            pass

        # Config overrides table (Dynamic Runtime Configuration)
        # Stores operator overrides that persist across restarts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config_overrides (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                updated_at INTEGER NOT NULL
            )
        """)
        
        # Mempool fee history (Vegas Reflex MA calculation)
        # Tracks on-chain fee rates for detecting spikes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mempool_fee_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sat_per_vbyte REAL NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)

        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fee_changes_channel ON fee_changes(channel_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forwards_time ON forwards(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forwards_channels ON forwards(in_channel, out_channel)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_costs_channel ON rebalance_costs(channel_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_costs_channel_time ON rebalance_costs(channel_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_states_peer ON channel_states(peer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_connection_history_peer_time ON peer_connection_history(peer_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mempool_time ON mempool_fee_history(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_history_time ON rebalance_history(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_history_to_channel ON rebalance_history(to_channel, timestamp)")
        
        # Composite index for get_volume_since optimization (TODO #17)
        # Fee Controller queries by out_channel + timestamp every 30min
        # This changes query complexity from O(N) to O(log N)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forwards_out_channel_time ON forwards(out_channel, timestamp)")
        
        # Daily aggregated forwarding stats (Granular History)
        # Replacing the single 'lifetime_aggregates' counter with daily resolution
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_forwarding_stats (
                channel_id TEXT NOT NULL,
                date INTEGER NOT NULL,  -- Unix timestamp of midnight (UTC)
                total_in_msat INTEGER NOT NULL DEFAULT 0,
                total_out_msat INTEGER NOT NULL DEFAULT 0,
                total_fee_msat INTEGER NOT NULL DEFAULT 0,
                forward_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (channel_id, date)
            )
        """)

        # Daily aggregated entry-side forwarding stats (Profitability v2.0)
        # Preserves "in_channel" contribution metrics when raw forwards are pruned.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_forwarding_stats_inbound (
                channel_id TEXT NOT NULL,
                date INTEGER NOT NULL,  -- Unix timestamp of midnight (UTC)
                total_in_msat INTEGER NOT NULL DEFAULT 0,
                total_fee_msat INTEGER NOT NULL DEFAULT 0,
                forward_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (channel_id, date)
            )
        """)
        
        # Budget reservations table for atomic budget management (CRITICAL-01 fix)
        # Prevents race conditions where multiple concurrent jobs can overspend
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_reservations (
                reservation_id TEXT PRIMARY KEY,
                reserved_sats INTEGER NOT NULL,
                reserved_at INTEGER NOT NULL,
                job_channel_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'  -- 'active', 'spent', 'released'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_budget_reservations_status ON budget_reservations(status, reserved_at)")

        # Generic spend reservations/events for unified total-cost budget gating.
        # Used by actions outside the rebalance engine (e.g. channel open/close proposals)
        # to reserve and optionally record spend against the same budget envelope.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spend_reservations (
                reservation_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                subcategory TEXT,
                reserved_sats INTEGER NOT NULL,
                reserved_at INTEGER NOT NULL,
                reference_id TEXT,
                channel_id TEXT,
                status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'spent', 'released'
                metadata_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spend_reservations_status ON spend_reservations(status, reserved_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spend_reservations_category ON spend_reservations(category, reserved_at)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS spend_events (
                event_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                subcategory TEXT,
                amount_sats INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                reference_id TEXT,
                channel_id TEXT,
                source TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spend_events_time ON spend_events(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spend_events_category ON spend_events(category, timestamp)")

        # Financial Snapshots for P&L Dashboard
        # Records daily node state for TLV tracking and trend analysis
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_snapshots (
                timestamp INTEGER PRIMARY KEY,
                total_local_balance_sats INTEGER NOT NULL,
                total_remote_balance_sats INTEGER NOT NULL,
                total_onchain_sats INTEGER NOT NULL,
                total_capacity_sats INTEGER NOT NULL,
                total_revenue_accumulated_sats INTEGER NOT NULL,
                total_rebalance_cost_accumulated_sats INTEGER NOT NULL,
                channel_count INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_financial_snapshots_time ON financial_snapshots(timestamp)")

        # Channel Closure Costs table (Accounting v2.0)
        # Tracks on-chain fees paid when channels close (mutual or force close)
        # This is the missing cost category that was causing overstated P&L
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_closure_costs (
                channel_id TEXT PRIMARY KEY,
                peer_id TEXT NOT NULL,
                close_type TEXT NOT NULL,  -- 'mutual', 'local_unilateral', 'remote_unilateral', 'unknown'
                closure_fee_sats INTEGER NOT NULL DEFAULT 0,
                htlc_sweep_fee_sats INTEGER NOT NULL DEFAULT 0,
                penalty_fee_sats INTEGER NOT NULL DEFAULT 0,
                total_closure_cost_sats INTEGER NOT NULL DEFAULT 0,
                funding_txid TEXT,
                closing_txid TEXT,
                closed_at INTEGER NOT NULL,
                resolution_complete INTEGER NOT NULL DEFAULT 0  -- 1 when all outputs resolved
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_closure_costs_peer ON channel_closure_costs(peer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_closure_costs_time ON channel_closure_costs(closed_at)")

        # Closed Channels History table (Accounting v2.0)
        # Preserves complete P&L history for channels after they close
        # Without this, closing a channel would orphan its accounting data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS closed_channels (
                channel_id TEXT PRIMARY KEY,
                peer_id TEXT NOT NULL,
                capacity_sats INTEGER NOT NULL,
                opened_at INTEGER,
                closed_at INTEGER NOT NULL,
                close_type TEXT NOT NULL,
                open_cost_sats INTEGER NOT NULL DEFAULT 0,
                closure_cost_sats INTEGER NOT NULL DEFAULT 0,
                total_revenue_sats INTEGER NOT NULL DEFAULT 0,
                total_rebalance_cost_sats INTEGER NOT NULL DEFAULT 0,
                forward_count INTEGER NOT NULL DEFAULT 0,
                net_pnl_sats INTEGER NOT NULL DEFAULT 0,
                days_open INTEGER NOT NULL DEFAULT 0,
                funding_txid TEXT,
                closing_txid TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_closed_channels_peer ON closed_channels(peer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_closed_channels_time ON closed_channels(closed_at)")

        # Schema migration: Add deadband hysteresis columns to fee_strategy_state
        # SQLite doesn't support IF NOT EXISTS for columns, so we wrap in try/except
        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN is_sleeping INTEGER DEFAULT 0")
            self.plugin.log("Added is_sleeping column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN sleep_until INTEGER DEFAULT 0")
            self.plugin.log("Added sleep_until column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Schema migration: Add stable_cycles counter for hysteresis
        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN stable_cycles INTEGER DEFAULT 0")
            self.plugin.log("Added stable_cycles column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        # Schema migration: Add resolution_time to forwards for risk premium
        try:
            conn.execute("ALTER TABLE forwards ADD COLUMN resolution_time REAL DEFAULT 0")
            self.plugin.log("Added resolution_time column to forwards")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Schema migration: Add last_broadcast_fee_ppm to fee_strategy_state
        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN last_broadcast_fee_ppm INTEGER DEFAULT 0")
            self.plugin.log("Added last_broadcast_fee_ppm column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Schema migration: Add last_state to fee_strategy_state
        # NOTE: Must be separate try/except to ensure it runs even if above column exists
        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN last_state TEXT DEFAULT 'balanced'")
            self.plugin.log("Added last_state column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Schema migration: Add rebalance_type to rebalance_history
        try:
            conn.execute("ALTER TABLE rebalance_history ADD COLUMN rebalance_type TEXT DEFAULT 'normal'")
            self.plugin.log("Added rebalance_type column to rebalance_history")
        except sqlite3.OperationalError:
            pass

        # v2.0 Migration: Add columns for fee algorithm improvements
        # - forward_count_since_update: Dynamic observation windows (Improvement #2)
        # - last_volume_sats: For elasticity tracking
        # - v2_state_json: JSON blob for DTS+PID state (posterior, PID integral/derivative)
        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN forward_count_since_update INTEGER DEFAULT 0")
            self.plugin.log("Added forward_count_since_update column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN last_volume_sats INTEGER DEFAULT 0")
            self.plugin.log("Added last_volume_sats column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE fee_strategy_state ADD COLUMN v2_state_json TEXT DEFAULT '{}'")
            self.plugin.log("Added v2_state_json column to fee_strategy_state")
        except sqlite3.OperationalError:
            pass

        # v2.0 Migration: Add peer_policies columns for Policy Manager v2.0
        # - fee_multiplier_min: Per-peer fee multiplier floor
        # - fee_multiplier_max: Per-peer fee multiplier ceiling
        # - expires_at: Unix timestamp for time-limited policies
        try:
            conn.execute("ALTER TABLE peer_policies ADD COLUMN fee_multiplier_min REAL")
            self.plugin.log("Added fee_multiplier_min column to peer_policies")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE peer_policies ADD COLUMN fee_multiplier_max REAL")
            self.plugin.log("Added fee_multiplier_max column to peer_policies")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE peer_policies ADD COLUMN expires_at INTEGER")
            self.plugin.log("Added expires_at column to peer_policies")
        except sqlite3.OperationalError:
            pass

        # v2.1 Migration: Add closer column to closed_channels
        # Tracks who initiated the closure: 'local', 'remote', or 'mutual'
        try:
            conn.execute("ALTER TABLE closed_channels ADD COLUMN closer TEXT DEFAULT 'unknown'")
            self.plugin.log("Added closer column to closed_channels")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # =================================================================
        # EXPLAINABILITY MIGRATIONS: Reason codes and heuristic modifiers
        # =================================================================
        # These columns enable structured decision explanations for debugging,
        # auditing, and fleet-wide analysis of fee and rebalance behavior.
        # =================================================================

        # fee_changes table: Add reason_code and heuristic_modifiers columns
        try:
            conn.execute("ALTER TABLE fee_changes ADD COLUMN reason_code TEXT")
            self.plugin.log("Added reason_code column to fee_changes")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            conn.execute("ALTER TABLE fee_changes ADD COLUMN heuristic_modifiers TEXT")
            self.plugin.log("Added heuristic_modifiers column to fee_changes")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # rebalance_history table: Add reason_code and bleeder_status columns
        try:
            conn.execute("ALTER TABLE rebalance_history ADD COLUMN reason_code TEXT")
            self.plugin.log("Added reason_code column to rebalance_history")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            conn.execute("ALTER TABLE rebalance_history ADD COLUMN bleeder_status TEXT")
            self.plugin.log("Added bleeder_status column to rebalance_history")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # v1.4 Migration: Migrate ignored_peers to peer_policies
        # This migrates legacy "ignored" peers to the new policy system
        self._migrate_ignored_peers_to_policies(conn)

        # v1.6 Migration: Add flow analysis v2.0 columns
        self._migrate_flow_v2_schema(conn)

        # v2.0 Migration: Add Kalman filter columns and table
        self._migrate_kalman_schema(conn)

        # v2.1 Migration: Add temporal profile column to channel_states
        self._migrate_temporal_profile_schema(conn)

        # Flow analysis cleanup: remove dead flow_history table
        try:
            conn.execute("DROP TABLE IF EXISTS flow_history")
        except sqlite3.OperationalError:
            pass

        # Drop portfolio_metrics table (module removed)
        try:
            conn.execute("DROP TABLE IF EXISTS portfolio_metrics")
        except Exception as e:
            self.plugin.log(f"DB migration warning: {e}", level='debug')

        # Drop clboss_unmanaged table (clboss_manager module removed)
        try:
            conn.execute("DROP TABLE IF EXISTS clboss_unmanaged")
        except Exception as e:
            self.plugin.log(f"DB migration warning: {e}", level='debug')

        # Rebalancer efficiency: failure-informed routing columns
        for col, col_type, default in [
            ("last_attempted_ppm", "INTEGER", "0"),
            ("last_attempted_amount", "INTEGER", "0"),
            ("last_error_type", "TEXT", "''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE channel_failures ADD COLUMN {col} {col_type} NOT NULL DEFAULT {default}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        self.plugin.log("Database initialized successfully")
    

    def _migrate_forwards_schema(self, conn: sqlite3.Connection) -> None:
        """
        Make forwards ingestion idempotent and restart-safe.

        - Adds resolved_time column if missing
        - Backfills resolved_time from timestamp + resolution_time (best-effort)
        - Deduplicates existing rows prior to enforcing uniqueness
        - Creates a UNIQUE index so INSERT OR IGNORE can safely skip duplicates
        """
        # D4 FIX: Wrap migration in transaction for atomicity
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(forwards)").fetchall()]
                if "resolved_time" not in cols:
                    self.plugin.log("DB migration: adding forwards.resolved_time column", level="info")
                    conn.execute("ALTER TABLE forwards ADD COLUMN resolved_time INTEGER")

                # Backfill resolved_time where missing/zero (best-effort).
                # resolution_time is stored as REAL seconds; round down to int seconds for stable keying.
                conn.execute("""
                    UPDATE forwards
                    SET resolved_time = COALESCE(resolved_time, 0)
                    WHERE resolved_time IS NULL
                """)
                conn.execute("""
                    UPDATE forwards
                    SET resolved_time = timestamp + CAST(resolution_time AS INTEGER)
                    WHERE (resolved_time IS NULL OR resolved_time = 0)
                      AND resolution_time IS NOT NULL
                      AND resolution_time > 0
                """)

                # Deduplicate rows before adding a UNIQUE index (keep earliest id).
                # Key includes resolved_time to reduce collision risk within the same second.
                self.plugin.log("DB migration: deduplicating forwards table (if needed)", level="debug")
                conn.execute("""
                    DELETE FROM forwards
                    WHERE id NOT IN (
                        SELECT MIN(id)
                        FROM forwards
                        GROUP BY
                            in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp,
                            COALESCE(resolved_time, 0)
                    )
                """)

                # Enforce idempotency: duplicates become no-ops.
                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_forwards_unique
                    ON forwards(in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp, resolved_time)
                """)

                # Helpful indexes for per-channel lookups (keeps queries fast as forwards grow)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_forwards_in_time ON forwards(in_channel, timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_forwards_out_time ON forwards(out_channel, timestamp)")

                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        except Exception as e:
            # Non-fatal: plugin can still run, but may double-dip without uniqueness.
            self.plugin.log(f"DB migration warning: forwards schema migration failed: {e}", level="warn")


    def _migrate_ignored_peers_to_policies(self, conn):
        """
        Migrate legacy ignored_peers table to peer_policies.
        
        Conversion:
        - ignored_peers entries become PASSIVE strategy + DISABLED rebalance
        - Original ignore reason stored in tags as 'migrated_ignore'
        - Original table renamed to _backup_ignored_peers
        """
        # Check if ignored_peers table exists and has data to migrate
        try:
            rows = conn.execute("SELECT peer_id, reason, ignored_at FROM ignored_peers").fetchall()
        except sqlite3.OperationalError:
            return  # Table doesn't exist, nothing to migrate
        
        if not rows:
            return  # No data to migrate
        
        # Check if we've already migrated (look for _backup_ignored_peers)
        try:
            conn.execute("SELECT 1 FROM _backup_ignored_peers LIMIT 1")
            return  # Backup exists, migration already done
        except sqlite3.OperationalError:
            pass  # Backup doesn't exist, proceed with migration
        
        conn.execute("BEGIN IMMEDIATE")
        try:
            migrated_count = 0
            for row in rows:
                peer_id = row['peer_id']
                reason = row['reason'] or 'migrated_ignore'
                ignored_at = row['ignored_at']

                # Check if peer already has a policy (don't overwrite)
                existing = conn.execute(
                    "SELECT 1 FROM peer_policies WHERE peer_id = ?", (peer_id,)
                ).fetchone()

                if not existing:
                    import json
                    tags = json.dumps(['migrated_ignore', reason] if reason != 'migrated_ignore' else ['migrated_ignore'])
                    conn.execute("""
                        INSERT INTO peer_policies
                            (peer_id, strategy, rebalance_mode, fee_ppm_target, tags, updated_at)
                        VALUES (?, 'passive', 'disabled', NULL, ?, ?)
                    """, (peer_id, tags, ignored_at))
                    migrated_count += 1

            # Rename old table to backup (preserve data for safety).
            # Always rename when rows were processed, even if all peers already had
            # policies (migrated_count == 0). Otherwise the non-empty table persists
            # and triggers re-migration on every startup.
            if migrated_count > 0 or len(rows) > 0:
                conn.execute("ALTER TABLE ignored_peers RENAME TO _backup_ignored_peers")
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        if migrated_count > 0:
            self.plugin.log(
                f"v1.4 Migration: Migrated {migrated_count} ignored peers to peer_policies. "
                f"Old table renamed to _backup_ignored_peers.",
                level='info'
            )

    def _migrate_flow_v2_schema(self, conn: sqlite3.Connection) -> None:
        """
        v1.6 Migration: Add flow analysis v2.0 columns to channel_states.

        New columns:
        - confidence: Flow confidence score (0.1 to 1.0)
        - velocity: Rate of change of flow_ratio
        - flow_multiplier: Graduated multiplier for fee adjustments
        - ema_decay: Adaptive decay factor used
        - forward_count: Number of forwards in analysis window
        """
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(channel_states)").fetchall()]

            new_cols = [
                ("confidence", "REAL DEFAULT 1.0"),
                ("velocity", "REAL DEFAULT 0.0"),
                ("flow_multiplier", "REAL DEFAULT 1.0"),
                ("ema_decay", "REAL DEFAULT 0.8"),
                ("forward_count", "INTEGER DEFAULT 0"),
            ]

            for col_name, col_def in new_cols:
                if col_name not in cols:
                    self.plugin.log(f"DB migration: adding channel_states.{col_name}", level="info")
                    conn.execute(f"ALTER TABLE channel_states ADD COLUMN {col_name} {col_def}")

        except Exception as e:
            self.plugin.log(f"DB migration warning: flow v2 schema migration failed: {e}", level="warn")

    def _migrate_kalman_schema(self, conn: sqlite3.Connection) -> None:
        """
        v2.0 Migration: Add Kalman filter columns to channel_states and create kalman_state table.

        New columns in channel_states:
        - kalman_flow_ratio: Kalman-filtered flow ratio estimate
        - kalman_velocity: Kalman-estimated velocity
        - kalman_uncertainty: Estimation uncertainty

        New table kalman_state:
        - Stores full Kalman filter state for persistence across restarts
        """
        try:
            # Add columns to channel_states
            cols = [r[1] for r in conn.execute("PRAGMA table_info(channel_states)").fetchall()]

            kalman_cols = [
                ("kalman_flow_ratio", "REAL DEFAULT 0.0"),
                ("kalman_velocity", "REAL DEFAULT 0.0"),
                ("kalman_uncertainty", "REAL DEFAULT 0.1"),
            ]

            for col_name, col_def in kalman_cols:
                if col_name not in cols:
                    self.plugin.log(f"DB migration: adding channel_states.{col_name}", level="info")
                    conn.execute(f"ALTER TABLE channel_states ADD COLUMN {col_name} {col_def}")

            # Create kalman_state table for full filter state persistence
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kalman_state (
                    channel_id TEXT PRIMARY KEY,
                    flow_ratio REAL DEFAULT 0.0,
                    flow_velocity REAL DEFAULT 0.0,
                    variance_ratio REAL DEFAULT 0.1,
                    variance_velocity REAL DEFAULT 0.1,
                    covariance REAL DEFAULT 0.0,
                    last_update INTEGER DEFAULT 0,
                    innovation_variance REAL DEFAULT 0.01,
                    last_innovation REAL DEFAULT 0.0
                )
            """)

            # Migration: add last_innovation column if missing
            try:
                ks_cols = {row[1] for row in conn.execute("PRAGMA table_info(kalman_state)").fetchall()}
                if 'last_innovation' not in ks_cols:
                    conn.execute("ALTER TABLE kalman_state ADD COLUMN last_innovation REAL DEFAULT 0.0")
                if 'velocity_unit' not in ks_cols:
                    conn.execute("ALTER TABLE kalman_state ADD COLUMN velocity_unit TEXT DEFAULT 'per_day'")
                if 'observation_count' not in ks_cols:
                    conn.execute("ALTER TABLE kalman_state ADD COLUMN observation_count INTEGER DEFAULT 0")
            except Exception as e:
                self.plugin.log(f"Kalman migration warning: {e}", level='debug')

        except Exception as e:
            self.plugin.log(f"DB migration warning: Kalman schema migration failed: {e}", level="warn")

    def _migrate_temporal_profile_schema(self, conn: sqlite3.Connection) -> None:
        """Add temporal_profile_json column to channel_states."""
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(channel_states)").fetchall()]
            if "temporal_profile_json" not in cols:
                self.plugin.log("DB migration: adding channel_states.temporal_profile_json", level="info")
                conn.execute("ALTER TABLE channel_states ADD COLUMN temporal_profile_json TEXT DEFAULT NULL")
                conn.commit()
        except Exception as e:
            self.plugin.log(f"Migration temporal_profile_json failed: {e}", level="warn")


    # =========================================================================
    # Channel State Methods
    # =========================================================================

    def update_channel_state(
        self, channel_id: str, peer_id: str, state: str,
        flow_ratio: float, sats_in: int, sats_out: int, capacity: int,
        # v2.0 fields (optional for backwards compatibility)
        confidence: float = 1.0,
        velocity: float = 0.0,
        flow_multiplier: float = 1.0,
        ema_decay: float = 0.8,
        forward_count: int = 0,
        # v2.1 Kalman fields
        kalman_flow_ratio: float = 0.0,
        kalman_velocity: float = 0.0,
        kalman_uncertainty: float = 0.1
    ):
        """Update the current state of a channel (v2.1: includes Kalman filter metrics)."""
        conn = self._get_connection()
        now = int(time.time())

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("""
                INSERT INTO channel_states
                (channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, updated_at,
                 confidence, velocity, flow_multiplier, ema_decay, forward_count,
                 kalman_flow_ratio, kalman_velocity, kalman_uncertainty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    peer_id=excluded.peer_id, state=excluded.state,
                    flow_ratio=excluded.flow_ratio, sats_in=excluded.sats_in,
                    sats_out=excluded.sats_out, capacity=excluded.capacity,
                    updated_at=excluded.updated_at, confidence=excluded.confidence,
                    velocity=excluded.velocity, flow_multiplier=excluded.flow_multiplier,
                    ema_decay=excluded.ema_decay, forward_count=excluded.forward_count,
                    kalman_flow_ratio=excluded.kalman_flow_ratio,
                    kalman_velocity=excluded.kalman_velocity,
                    kalman_uncertainty=excluded.kalman_uncertainty
            """, (channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, now,
                  confidence, velocity, flow_multiplier, ema_decay, forward_count,
                  kalman_flow_ratio, kalman_velocity, kalman_uncertainty))

            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception as rb_err:
                self.plugin.log(f"ROLLBACK failed in update_channel_state: {rb_err}", level='error')
            raise

    def get_channel_state(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a channel."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM channel_states WHERE channel_id = ?", 
            (channel_id,)
        ).fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_all_channel_states(self) -> List[Dict[str, Any]]:
        """Get states of all tracked channels."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM channel_states ORDER BY state, flow_ratio DESC").fetchall()
        return [dict(row) for row in rows]
    
    def get_channels_by_state(self, state: str) -> List[Dict[str, Any]]:
        """Get all channels with a specific state.

        For backward compatibility, querying "balanced" returns both
        "balanced" and "balanced_active" channels.
        """
        conn = self._get_connection()
        if state == "balanced":
            rows = conn.execute(
                "SELECT * FROM channel_states WHERE state IN ('balanced', 'balanced_active')"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM channel_states WHERE state = ?",
                (state,)
            ).fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Temporal Profile Methods (v2.1)
    # =========================================================================

    def save_temporal_profile(self, channel_id: str, profile_json: str) -> None:
        """Save temporal profile JSON for a channel."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE channel_states SET temporal_profile_json = ? WHERE channel_id = ?",
            (profile_json, channel_id),
        )
        conn.commit()

    def load_temporal_profile(self, channel_id: str) -> Optional[str]:
        """Load temporal profile JSON for a channel. Returns None if absent."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT temporal_profile_json FROM channel_states WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        if row and row[0]:
            return row[0]
        return None

    def get_hourly_forward_histogram(self, channel_id: str, window_days: int = 7) -> list:
        """Get per-hour flow histogram for a channel."""
        conn = self._get_connection()
        return _hourly_forward_histogram_sql(conn, channel_id, window_days)

    # =========================================================================
    # Kalman Filter State Methods (v2.1)
    # =========================================================================

    def get_kalman_state(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get Kalman filter state for a channel."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM kalman_state WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()

        if row:
            return dict(row)
        return None

    def save_kalman_state(self, channel_id: str, state: Dict[str, Any]) -> None:
        """Save Kalman filter state for a channel.

        Rejects NaN/Inf values to prevent poisoning the persisted state.
        """
        float_fields = [
            ("flow_ratio", 0.0), ("flow_velocity", 0.0),
            ("variance_ratio", 0.1), ("variance_velocity", 0.1),
            ("covariance", 0.0), ("innovation_variance", 0.01),
            ("last_innovation", 0.0),
        ]
        values = []
        for key, default in float_fields:
            v = state.get(key, default)
            if isinstance(v, float) and not math.isfinite(v):
                self.plugin.log(
                    f"Kalman NaN/Inf in {key} for {channel_id}, skipping persist",
                    level='warn'
                )
                return
            values.append(v)

        conn = self._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO kalman_state
            (channel_id, flow_ratio, flow_velocity, variance_ratio, variance_velocity,
             covariance, last_update, innovation_variance, last_innovation, velocity_unit,
             observation_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            channel_id,
            values[0],  # flow_ratio
            values[1],  # flow_velocity
            values[2],  # variance_ratio
            values[3],  # variance_velocity
            values[4],  # covariance
            state.get("last_update", 0),
            values[5],  # innovation_variance
            values[6],  # last_innovation
            "per_hour",
            state.get("observation_count", 0),
        ))

    def get_all_kalman_states(self) -> List[Dict[str, Any]]:
        """Get Kalman filter states for all channels."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM kalman_state").fetchall()
        return [dict(row) for row in rows]

    def reset_all_kalman_states(self) -> int:
        """Delete all Kalman filter states, forcing fresh re-convergence.

        Returns:
            Number of rows deleted.
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM kalman_state")
        return cursor.rowcount

    def kalman_purge_needed(self) -> bool:
        """Check whether the one-time Kalman purge has already run."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_flags (
                flag TEXT PRIMARY KEY,
                value TEXT,
                created_at INTEGER
            )
        """)
        row = conn.execute(
            "SELECT 1 FROM plugin_flags WHERE flag = 'kalman_observation_purge_v2'"
        ).fetchone()
        return row is None

    def mark_kalman_purge_done(self) -> None:
        """Record that the one-time Kalman purge has completed."""
        conn = self._get_connection()
        conn.execute(
            "INSERT OR IGNORE INTO plugin_flags (flag, value, created_at) "
            "VALUES ('kalman_observation_purge_v2', 'done', ?)",
            (int(time.time()),)
        )


    # =========================================================================
    # Channel Probe Methods
    # =========================================================================

    def set_channel_probe(self, channel_id: str, probe_type: str = 'bounded_low_fee'):
        """Sets the bounded-exploration flag for a channel.

        The legacy probe_type='zero_fee' value is still accepted for backward
        compatibility with older callers and persisted rows.
        """
        conn = self._get_connection()
        now = int(time.time())
        conn.execute("""
            INSERT OR REPLACE INTO channel_probes (channel_id, probe_type, started_at)
            VALUES (?, ?, ?)
        """, (channel_id, probe_type, now))

    def get_channel_probe(self, channel_id: str, max_age_seconds: int = 86400) -> Optional[Dict[str, Any]]:
        """Gets the bounded-exploration flag for a channel, returning None if expired.

        Args:
            channel_id: Channel to check
            max_age_seconds: Maximum age of exploration before auto-expiry (default 24h)
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM channel_probes WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        if row is None:
            return None
        probe = dict(row)
        # Auto-expire stale flags to prevent permanent exploration mode.
        started_at = probe.get("started_at", 0)
        if started_at > 0 and (int(time.time()) - started_at) > max_age_seconds:
            conn.execute("DELETE FROM channel_probes WHERE channel_id = ?", (channel_id,))
            return None
        return probe

    def clear_channel_probe(self, channel_id: str):
        """Clears the bounded-exploration flag for a channel."""
        conn = self._get_connection()
        conn.execute("DELETE FROM channel_probes WHERE channel_id = ?", (channel_id,))
    
    # =========================================================================
    # Fee Strategy State Methods
    # =========================================================================
    
    def get_fee_strategy_state(self, channel_id: str) -> Dict[str, Any]:
        """
        Get fee strategy state for a channel.

        Used by the revenue-maximizing Perturb & Observe algorithm.

        Args:
            channel_id: Channel to get state for

        Returns:
            Dict with last_revenue_rate, last_fee_ppm, trend_direction, step_ppm,
            is_sleeping, sleep_until, stable_cycles, v2.0 fields, etc.
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM fee_strategy_state WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()

        if row:
            result = dict(row)
            # Handle migration: old schema had last_revenue_sats (int)
            # New schema uses last_revenue_rate (float)
            if 'last_revenue_sats' in result and 'last_revenue_rate' not in result:
                result['last_revenue_rate'] = float(result.get('last_revenue_sats', 0))
            # Ensure step_ppm is present (may be missing from old schema)
            if 'step_ppm' not in result:
                result['step_ppm'] = 50  # Default step size
            # Ensure hysteresis fields are present (may be missing from old schema)
            if 'is_sleeping' not in result:
                result['is_sleeping'] = 0
            if 'sleep_until' not in result:
                result['sleep_until'] = 0
            if 'stable_cycles' not in result:
                result['stable_cycles'] = 0
            if 'last_broadcast_fee_ppm' not in result:
                result['last_broadcast_fee_ppm'] = result.get('last_fee_ppm', 0)
            # v2.0 fields
            if 'forward_count_since_update' not in result:
                result['forward_count_since_update'] = 0
            if 'last_volume_sats' not in result:
                result['last_volume_sats'] = 0
            if 'v2_state_json' not in result:
                result['v2_state_json'] = '{}'
            return result

        # Return default state if not found
        return {
            'channel_id': channel_id,
            'last_revenue_rate': 0.0,  # Revenue rate in sats/hour
            'last_fee_ppm': 0,
            'last_broadcast_fee_ppm': 0,
            'trend_direction': 1,  # Default: try increasing fee
            'step_ppm': 50,  # Default step size for dampening
            'consecutive_same_direction': 0,
            'last_update': 0,
            'last_state': 'unknown',
            'is_sleeping': 0,
            'sleep_until': 0,
            'stable_cycles': 0,
            # v2.0 fields
            'forward_count_since_update': 0,
            'last_volume_sats': 0,
            'v2_state_json': '{}'
        }
    
    def update_fee_strategy_state(self, channel_id: str, last_revenue_rate: float,
                                   last_fee_ppm: int, trend_direction: int,
                                   step_ppm: int = 50,
                                   consecutive_same_direction: int = 0,
                                   last_broadcast_fee_ppm: int = 0,
                                   last_state: str = 'unknown',
                                   is_sleeping: int = 0,
                                   sleep_until: int = 0,
                                   stable_cycles: int = 0,
                                   forward_count_since_update: int = 0,
                                   last_volume_sats: int = 0,
                                   v2_state_json: str = '{}',
                                   last_update: int = None):
        """
        Update fee strategy state for a channel.

        Called after each fee adjustment iteration to record the state
        for the next observation period.

        Args:
            channel_id: Channel to update
            last_revenue_rate: Revenue rate in sats/hour observed since last change
            last_fee_ppm: Fee that was in effect
            trend_direction: Direction we were moving (1 = up, -1 = down)
            step_ppm: Current step size (for wiggle dampening)
            consecutive_same_direction: How many times we've moved same way
            last_broadcast_fee_ppm: The last fee PPM broadcasted to the network
            last_state: The state classification during the last broadcast
            is_sleeping: Deadband hysteresis sleep state (0 = awake, 1 = sleeping)
            sleep_until: Unix timestamp when to wake up from sleep mode
            stable_cycles: Number of consecutive stable cycles (for hysteresis)
            forward_count_since_update: v2.0 - Forwards since last fee change
            last_volume_sats: v2.0 - Volume during last period (for elasticity)
            v2_state_json: v2.0 - JSON blob for DTS+PID state (posterior, PID integral/derivative)
        """
        conn = self._get_connection()
        ts = last_update if last_update is not None else int(time.time())

        conn.execute("""
            INSERT OR REPLACE INTO fee_strategy_state
            (channel_id, last_revenue_rate, last_fee_ppm, trend_direction,
             step_ppm, consecutive_same_direction, last_update,
             last_broadcast_fee_ppm, last_state, is_sleeping, sleep_until, stable_cycles,
             forward_count_since_update, last_volume_sats, v2_state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (channel_id, last_revenue_rate, last_fee_ppm, trend_direction,
              step_ppm, consecutive_same_direction, ts,
              last_broadcast_fee_ppm, last_state, is_sleeping, sleep_until, stable_cycles,
              forward_count_since_update, last_volume_sats, v2_state_json))
    
    def get_all_fee_strategy_states(self) -> List[Dict[str, Any]]:
        """Get fee strategy state for all channels."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM fee_strategy_state").fetchall()
        return [dict(row) for row in rows]
    
    def reset_fee_strategy_state(self, channel_id: str):
        """
        Reset the fee strategy state for a channel.
        
        Use when manually intervening or if the controller is behaving erratically.
        """
        conn = self._get_connection()
        conn.execute(
            "DELETE FROM fee_strategy_state WHERE channel_id = ?",
            (channel_id,)
        )
        self.plugin.log(f"Reset fee strategy state for {channel_id}")
    
    # =========================================================================
    # Fee Change Methods
    # =========================================================================
    
    def record_fee_change(self, channel_id: str, peer_id: str, old_fee_ppm: int,
                          new_fee_ppm: int, reason: str, manual: bool = False,
                          reason_code: Optional[str] = None,
                          heuristic_modifiers: Optional[str] = None):
        """
        Record a fee change for audit purposes.

        Args:
            channel_id: Channel short ID
            peer_id: Peer node ID
            old_fee_ppm: Previous fee in PPM
            new_fee_ppm: New fee in PPM
            reason: Human-readable reason string
            manual: True if this was a manual change (not algorithmic)
            reason_code: Structured FeeReasonCode value (for explainability)
            heuristic_modifiers: JSON string of HeuristicModifiers (for explainability)
        """
        conn = self._get_connection()
        now = int(time.time())

        conn.execute("""
            INSERT INTO fee_changes
            (channel_id, peer_id, old_fee_ppm, new_fee_ppm, reason, manual, timestamp,
             reason_code, heuristic_modifiers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (channel_id, peer_id, old_fee_ppm, new_fee_ppm, reason, 1 if manual else 0, now,
              reason_code, heuristic_modifiers))
    
    def get_recent_fee_changes(self, limit: int = 10, channel_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent fee changes, optionally filtered by channel."""
        limit = max(1, min(int(limit), 10000))  # SEC-10: Clamp limit
        conn = self._get_connection()
        
        if channel_id:
            rows = conn.execute("""
                SELECT * FROM fee_changes 
                WHERE channel_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (channel_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM fee_changes 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Rebalance History Methods
    # =========================================================================
    
    def record_rebalance(self, from_channel: str, to_channel: str, amount_sats: int,
                         max_fee_sats: int, expected_profit_sats: int,
                         status: str = 'pending', **kwargs) -> int:
        """
        Record a rebalance attempt and return its ID.

        Args:
            from_channel: Source channel SCID
            to_channel: Destination channel SCID
            amount_sats: Amount to rebalance in satoshis
            max_fee_sats: Maximum fee budget in satoshis
            expected_profit_sats: Expected profit from the rebalance
            status: Initial status ('pending', 'success', 'failed')
            **kwargs:
                rebalance_type: Type of rebalance ('normal', 'diagnostic')
                reason_code: Structured RebalanceReasonCode value (for explainability)
                bleeder_status: Bleeder classification ('hard', 'soft', 'none')

        Returns:
            Database row ID for the rebalance record
        """
        conn = self._get_connection()
        now = int(time.time())

        params = {
            'from_channel': from_channel,
            'to_channel': to_channel,
            'amount_sats': amount_sats,
            'max_fee_sats': max_fee_sats,
            'expected_profit_sats': expected_profit_sats,
            'status': status,
            'rebalance_type': kwargs.get('rebalance_type', 'normal'),
            'reason_code': kwargs.get('reason_code'),
            'bleeder_status': kwargs.get('bleeder_status'),
            'timestamp': now
        }

        cursor = conn.execute("""
            INSERT INTO rebalance_history
            (from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats,
             status, rebalance_type, reason_code, bleeder_status, timestamp)
            VALUES (:from_channel, :to_channel, :amount_sats, :max_fee_sats, :expected_profit_sats,
                    :status, :rebalance_type, :reason_code, :bleeder_status, :timestamp)
        """, params)

        return cursor.lastrowid
    
    def update_rebalance_result(self, rebalance_id: int, status: str,
                                actual_fee_sats: Optional[int] = None,
                                actual_profit_sats: Optional[int] = None,
                                error_message: Optional[str] = None):
        """Update a rebalance record with the result."""
        conn = self._get_connection()
        
        conn.execute("""
            UPDATE rebalance_history 
            SET status = ?, actual_fee_sats = ?, actual_profit_sats = ?, error_message = ?
            WHERE id = ?
        """, (status, actual_fee_sats, actual_profit_sats, error_message, rebalance_id))
    
    def get_recent_rebalances(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent rebalance attempts."""
        limit = max(1, min(int(limit), 10000))  # SEC-10: Clamp limit
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT * FROM rebalance_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]
    
    def get_last_rebalance_time(self, channel_id: str) -> Optional[int]:
        """
        Get the timestamp of the last successful rebalance for a channel.
        
        Used to enforce rebalance cooldown periods to prevent thrashing.
        
        Args:
            channel_id: The destination channel of the rebalance
            
        Returns:
            Unix timestamp of last successful rebalance, or None if never
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT MAX(timestamp) as last_time
            FROM rebalance_history 
            WHERE to_channel = ? AND status = 'success'
        """, (channel_id,)).fetchone()
        
        if row and row['last_time']:
            return row['last_time']
        return None
    
    def get_diagnostic_rebalance_stats(self, channel_id: str, days: int = 14) -> Dict[str, Any]:
        """
        Get stats for diagnostic rebalance attempts for a channel.
        
        Args:
            channel_id: The destination channel SCID
            days: Lookback window in days
            
        Returns:
            Dict with count and last_success_time
        """
        conn = self._get_connection()
        since = int(time.time()) - (days * 86400)
        
        row = conn.execute("""
            SELECT 
                COUNT(*) as attempt_count,
                MAX(CASE WHEN status = 'success' THEN timestamp ELSE 0 END) as last_success
            FROM rebalance_history 
            WHERE to_channel = ? AND rebalance_type = 'diagnostic' AND timestamp >= ?
        """, (channel_id, since)).fetchone()
        
        return {
            "attempt_count": int(row['attempt_count']) if row else 0,
            "last_success_time": int(row['last_success']) if row and row['last_success'] and row['last_success'] > 0 else None
        }
    
    def get_total_rebalance_fees(self, since_timestamp: int) -> int:
        """
        Get the total rebalancing fees spent since a given timestamp.
        
        Used for Global Capital Controls to enforce daily budget limits.
        Uses the rebalance_costs table as the source of truth because async Sling
        jobs may record costs there even when rebalance_history.actual_fee_sats is
        missing or delayed.
        
        Args:
            since_timestamp: Unix timestamp to start summing from
            
        Returns:
            Total fees spent in sats (0 if none)
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(cost_sats), 0) as total_fees
            FROM rebalance_costs
            WHERE timestamp >= ?
        """, (since_timestamp,)).fetchone()
        
        return row['total_fees'] if row else 0
    
    def get_total_routing_revenue(self, since_timestamp: int) -> int:
        """
        Get total routing revenue (fees earned) since a given timestamp.
        
        Used for Revenue-Proportional Budget calculation.
        Sums fee_msat from the forwards table since the timestamp, plus
        rolled-up daily aggregates from pruned forwards.
        
        Args:
            since_timestamp: Unix timestamp to start summing from
            
        Returns:
            Total routing fees earned in sats (0 if none)
        """
        conn = self._get_connection()
        since_day = (int(since_timestamp) // 86400) * 86400

        # M-16: Fix day-boundary double-counting.
        # Only count rolled-up days that DON'T overlap with raw forwards.
        # daily_forwarding_stats: completed days only (date >= since_day AND date < today_start)
        # forwards: recent data (timestamp >= since_timestamp) covers the current partial day.
        today_start = (int(time.time()) // 86400) * 86400

        row = conn.execute("""
            SELECT
                (
                    SELECT COALESCE(SUM(fee_msat), 0)
                    FROM forwards
                    WHERE timestamp >= ?
                ) +
                (
                    SELECT COALESCE(SUM(total_fee_msat), 0)
                    FROM daily_forwarding_stats
                    WHERE date >= ? AND date < ?
                ) AS total_fees_msat
        """, (since_timestamp, since_day, today_start)).fetchone()

        # Convert msat to sats
        return int((row['total_fees_msat'] or 0) // 1000) if row else 0

    # =========================================================================
    # Financial Snapshots for P&L Dashboard
    # =========================================================================

    def record_financial_snapshot(self, local_balance_sats: int, remote_balance_sats: int,
                                   onchain_sats: int, capacity_sats: int,
                                   revenue_accumulated_sats: int,
                                   rebalance_cost_accumulated_sats: int,
                                   channel_count: int) -> bool:
        """
        Record a daily financial snapshot for P&L tracking.

        This captures the node's financial state at a point in time for
        trend analysis and Net Worth tracking.

        Args:
            local_balance_sats: Total local balance across all channels
            remote_balance_sats: Total remote balance across all channels
            onchain_sats: Total confirmed on-chain balance
            capacity_sats: Total channel capacity
            revenue_accumulated_sats: Lifetime routing revenue
            rebalance_cost_accumulated_sats: Lifetime rebalance costs
            channel_count: Number of active channels

        Returns:
            True if snapshot recorded, False on error
        """
        conn = self._get_connection()
        now = int(time.time())

        try:
            conn.execute("""
                INSERT OR REPLACE INTO financial_snapshots
                (timestamp, total_local_balance_sats, total_remote_balance_sats,
                 total_onchain_sats, total_capacity_sats,
                 total_revenue_accumulated_sats, total_rebalance_cost_accumulated_sats,
                 channel_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, local_balance_sats, remote_balance_sats, onchain_sats,
                  capacity_sats, revenue_accumulated_sats,
                  rebalance_cost_accumulated_sats, channel_count))

            self.plugin.log(
                f"Financial snapshot recorded: TLV={local_balance_sats + onchain_sats} sats, "
                f"{channel_count} channels",
                level='debug'
            )
            return True

        except Exception as e:
            self.plugin.log(f"Error recording financial snapshot: {e}", level='error')
            return False

    def get_financial_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent financial snapshots for trend analysis.

        Args:
            limit: Maximum number of snapshots to return (default 30 days)

        Returns:
            List of snapshot dicts, most recent first
        """
        limit = max(1, min(int(limit), 10000))  # SEC-10: Clamp limit
        conn = self._get_connection()

        rows = conn.execute("""
            SELECT timestamp, total_local_balance_sats, total_remote_balance_sats,
                   total_onchain_sats, total_capacity_sats,
                   total_revenue_accumulated_sats, total_rebalance_cost_accumulated_sats,
                   channel_count
            FROM financial_snapshots
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]

    def get_latest_financial_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent financial snapshot.

        Returns:
            Snapshot dict or None if no snapshots exist
        """
        conn = self._get_connection()

        row = conn.execute("""
            SELECT timestamp, total_local_balance_sats, total_remote_balance_sats,
                   total_onchain_sats, total_capacity_sats,
                   total_revenue_accumulated_sats, total_rebalance_cost_accumulated_sats,
                   channel_count
            FROM financial_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """).fetchone()

        return dict(row) if row else None

    # NOTE: get_lifetime_stats() is defined later in this file (line ~1954)
    # with the complete implementation that includes opening costs.

    def get_channel_pnl(self, channel_id: str, window_days: int = 30) -> Dict[str, Any]:
        """
        Get P&L breakdown for a specific channel.

        Args:
            channel_id: The channel SCID
            window_days: Time window for calculations (default 30 days)

        Returns:
            Dict with revenue_sats, rebalance_cost_sats, net_pnl_sats, forward_count
        """
        conn = self._get_connection()
        since = int(time.time()) - (window_days * 86400)
        since_day = (since // 86400) * 86400

        # DB-1: Use today_start boundary to avoid double-counting.
        # daily_forwarding_stats aggregates completed days; forwards has live data.
        # Exclude the current partial day from daily_forwarding_stats since those
        # forwards are still in the live forwards table.
        today_start = (int(time.time()) // 86400) * 86400

        # Revenue from this channel (as outbound)
        rev_row = conn.execute("""
            SELECT
                (
                    SELECT COALESCE(SUM(fee_msat), 0)
                    FROM forwards
                    WHERE out_channel = ? AND timestamp >= ?
                ) +
                (
                    SELECT COALESCE(SUM(total_fee_msat), 0)
                    FROM daily_forwarding_stats
                    WHERE channel_id = ? AND date >= ? AND date < ?
                ) AS revenue_msat,
                (
                    SELECT COALESCE(COUNT(*), 0)
                    FROM forwards
                    WHERE out_channel = ? AND timestamp >= ?
                ) +
                (
                    SELECT COALESCE(SUM(forward_count), 0)
                    FROM daily_forwarding_stats
                    WHERE channel_id = ? AND date >= ? AND date < ?
                ) AS forward_count
        """, (
            channel_id, since,
            channel_id, since_day, today_start,
            channel_id, since,
            channel_id, since_day, today_start,
        )).fetchone()

        # C6 FIX: Use unpruned rebalance_costs table instead of rebalance_history.
        # rebalance_history is pruned to 90 days by cleanup_old_data(), but
        # rebalance_costs is never pruned and serves as the source of truth
        # for lifetime cost accounting. This matters for channels open > 90 days.
        cost_row = conn.execute("""
            SELECT COALESCE(SUM(cost_sats), 0) as cost_sats
            FROM rebalance_costs
            WHERE channel_id = ? AND timestamp >= ?
        """, (channel_id, since)).fetchone()

        revenue_sats = (rev_row['revenue_msat'] // 1000) if rev_row else 0
        cost_sats = cost_row['cost_sats'] if cost_row else 0
        forward_count = rev_row['forward_count'] if rev_row else 0

        return {
            'channel_id': channel_id,
            'window_days': window_days,
            'revenue_sats': revenue_sats,
            'rebalance_cost_sats': cost_sats,
            'net_pnl_sats': revenue_sats - cost_sats,
            'forward_count': forward_count
        }

    def get_channel_inbound_contribution(self, channel_id: str, window_days: int = 30) -> Dict[str, Any]:
        """
        Get inbound contribution metrics for a channel.

        This calculates the value a channel provides by SOURCING inbound volume,
        even when it doesn't earn direct fees as the exit channel.

        For each forward where this channel was the in_channel (entry point),
        we track:
        - sourced_volume_sats: Total sats that entered through this channel
        - sourced_fee_contribution_sats: Fees earned on the exit channels
        - sourced_forward_count: Number of forwards sourced

        Args:
            channel_id: The channel SCID
            window_days: Time window for calculations (default 30 days)

        Returns:
            Dict with sourced_volume_sats, sourced_fee_contribution_sats,
            sourced_forward_count
        """
        conn = self._get_connection()
        since = int(time.time()) - (window_days * 86400)
        since_day = (since // 86400) * 86400
        # DB-1: Exclude current partial day from daily stats to avoid double-counting
        today_start = (int(time.time()) // 86400) * 86400

        # Get inbound contribution (where this channel was the entry point)
        inbound_row = conn.execute("""
            SELECT
                (
                    SELECT COALESCE(SUM(in_msat), 0)
                    FROM forwards
                    WHERE in_channel = ? AND timestamp >= ?
                ) +
                (
                    SELECT COALESCE(SUM(total_in_msat), 0)
                    FROM daily_forwarding_stats_inbound
                    WHERE channel_id = ? AND date >= ? AND date < ?
                ) AS sourced_volume_msat,
                (
                    SELECT COALESCE(SUM(fee_msat), 0)
                    FROM forwards
                    WHERE in_channel = ? AND timestamp >= ?
                ) +
                (
                    SELECT COALESCE(SUM(total_fee_msat), 0)
                    FROM daily_forwarding_stats_inbound
                    WHERE channel_id = ? AND date >= ? AND date < ?
                ) AS sourced_fee_msat,
                (
                    SELECT COALESCE(COUNT(*), 0)
                    FROM forwards
                    WHERE in_channel = ? AND timestamp >= ?
                ) +
                (
                    SELECT COALESCE(SUM(forward_count), 0)
                    FROM daily_forwarding_stats_inbound
                    WHERE channel_id = ? AND date >= ? AND date < ?
                ) AS sourced_forward_count
        """, (
            channel_id, since,
            channel_id, since_day, today_start,
            channel_id, since,
            channel_id, since_day, today_start,
            channel_id, since,
            channel_id, since_day, today_start,
        )).fetchone()

        sourced_volume_sats = (inbound_row['sourced_volume_msat'] // 1000) if inbound_row else 0
        sourced_fee_sats = (inbound_row['sourced_fee_msat'] // 1000) if inbound_row else 0
        sourced_forward_count = inbound_row['sourced_forward_count'] if inbound_row else 0

        return {
            'channel_id': channel_id,
            'window_days': window_days,
            'sourced_volume_sats': sourced_volume_sats,
            'sourced_fee_contribution_sats': sourced_fee_sats,
            'sourced_forward_count': sourced_forward_count
        }

    def get_channel_full_pnl(self, channel_id: str, window_days: int = 30) -> Dict[str, Any]:
        """
        Get complete P&L breakdown including inbound contribution value.

        This combines:
        1. Direct revenue (fees earned as exit channel)
        2. Inbound contribution (fees enabled by sourcing volume)
        3. Costs (rebalancing)

        The "contribution_value" metric provides a fuller picture of channel
        profitability by crediting channels for the volume they source.

        Args:
            channel_id: The channel SCID
            window_days: Time window for calculations (default 30 days)

        Returns:
            Dict with complete P&L including contribution metrics
        """
        # Get direct P&L (exit channel fees)
        direct_pnl = self.get_channel_pnl(channel_id, window_days)

        # Get inbound contribution
        inbound = self.get_channel_inbound_contribution(channel_id, window_days)

        # FIX: Apply the 50/50 split here to prevent double-counting in the dictionary return
        total_contribution = (direct_pnl['revenue_sats'] + inbound['sourced_fee_contribution_sats']) // 2

        return {
            'channel_id': channel_id,
            'window_days': window_days,
            # Direct metrics (as exit channel)
            'direct_revenue_sats': direct_pnl['revenue_sats'],
            'direct_forward_count': direct_pnl['forward_count'],
            # Inbound contribution metrics (as entry channel)
            'sourced_volume_sats': inbound['sourced_volume_sats'],
            'sourced_fee_contribution_sats': inbound['sourced_fee_contribution_sats'],
            'sourced_forward_count': inbound['sourced_forward_count'],
            # Combined metrics
            'total_contribution_sats': total_contribution,
            'rebalance_cost_sats': direct_pnl['rebalance_cost_sats'],
            'net_pnl_sats': total_contribution - direct_pnl['rebalance_cost_sats'],
            # Legacy fields for backward compatibility
            'revenue_sats': direct_pnl['revenue_sats'],
            'forward_count': direct_pnl['forward_count']
        }

    def get_last_forward_time_any_direction(self, channel_id: str) -> Optional[int]:
        """
        Get the timestamp of the most recent forward involving a channel.

        Considers both roles:
        - out_channel (exit)
        - in_channel (entry)

        Also considers rolled-up daily stats when raw forwards were pruned.
        Daily stats don't preserve time-of-day, so we approximate the most recent
        activity within a day as (midnight + 86399).
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT MAX(ts) as last_ts
            FROM (
                SELECT MAX(timestamp) as ts
                FROM forwards
                WHERE out_channel = ? OR in_channel = ?
                UNION ALL
                SELECT MAX(date) + 86399 as ts
                FROM daily_forwarding_stats
                WHERE channel_id = ? AND forward_count > 0
                UNION ALL
                SELECT MAX(date) + 86399 as ts
                FROM daily_forwarding_stats_inbound
                WHERE channel_id = ? AND forward_count > 0
            )
        """, (channel_id, channel_id, channel_id, channel_id)).fetchone()

        return int(row["last_ts"]) if row and row["last_ts"] else None

    def get_channel_revenue_totals(self, channel_id: str) -> Dict[str, int]:
        """
        Get all-time routing metrics for a channel (since plugin started tracking).

        Combines:
        - Raw forwards (recent, not yet pruned)
        - Rolled-up daily_forwarding_stats (exit-side)
        - Rolled-up daily_forwarding_stats_inbound (entry-side)

        Returns a dict with values in sats and counts as integers.
        """
        conn = self._get_connection()

        # Exclude current day from daily stats to prevent double-counting
        # with forwards that haven't been pruned yet (boundary day fix).
        today_start = (int(time.time()) // 86400) * 86400

        exit_row = conn.execute("""
            SELECT
                (
                    SELECT COALESCE(SUM(fee_msat), 0)
                    FROM forwards
                    WHERE out_channel = ?
                ) +
                (
                    SELECT COALESCE(SUM(total_fee_msat), 0)
                    FROM daily_forwarding_stats
                    WHERE channel_id = ? AND date < ?
                ) AS fee_msat,
                (
                    SELECT COALESCE(SUM(out_msat), 0)
                    FROM forwards
                    WHERE out_channel = ?
                ) +
                (
                    SELECT COALESCE(SUM(total_out_msat), 0)
                    FROM daily_forwarding_stats
                    WHERE channel_id = ? AND date < ?
                ) AS out_msat,
                (
                    SELECT COALESCE(COUNT(*), 0)
                    FROM forwards
                    WHERE out_channel = ?
                ) +
                (
                    SELECT COALESCE(SUM(forward_count), 0)
                    FROM daily_forwarding_stats
                    WHERE channel_id = ? AND date < ?
                ) AS forward_count
        """, (channel_id, channel_id, today_start,
              channel_id, channel_id, today_start,
              channel_id, channel_id, today_start)).fetchone()

        inbound_row = conn.execute("""
            SELECT
                (
                    SELECT COALESCE(SUM(in_msat), 0)
                    FROM forwards
                    WHERE in_channel = ?
                ) +
                (
                    SELECT COALESCE(SUM(total_in_msat), 0)
                    FROM daily_forwarding_stats_inbound
                    WHERE channel_id = ? AND date < ?
                ) AS in_msat,
                (
                    SELECT COALESCE(SUM(fee_msat), 0)
                    FROM forwards
                    WHERE in_channel = ?
                ) +
                (
                    SELECT COALESCE(SUM(total_fee_msat), 0)
                    FROM daily_forwarding_stats_inbound
                    WHERE channel_id = ? AND date < ?
                ) AS fee_msat,
                (
                    SELECT COALESCE(COUNT(*), 0)
                    FROM forwards
                    WHERE in_channel = ?
                ) +
                (
                    SELECT COALESCE(SUM(forward_count), 0)
                    FROM daily_forwarding_stats_inbound
                    WHERE channel_id = ? AND date < ?
                ) AS forward_count
        """, (channel_id, channel_id, today_start,
              channel_id, channel_id, today_start,
              channel_id, channel_id, today_start)).fetchone()

        return {
            "fees_earned_sats": int((exit_row["fee_msat"] or 0) // 1000) if exit_row else 0,
            "volume_routed_sats": int((exit_row["out_msat"] or 0) // 1000) if exit_row else 0,
            "forward_count": int(exit_row["forward_count"] or 0) if exit_row else 0,
            "sourced_volume_sats": int((inbound_row["in_msat"] or 0) // 1000) if inbound_row else 0,
            "sourced_fee_contribution_sats": int((inbound_row["fee_msat"] or 0) // 1000) if inbound_row else 0,
            "sourced_forward_count": int(inbound_row["forward_count"] or 0) if inbound_row else 0,
        }

    def get_all_channels_revenue_totals(self) -> Dict[str, Dict[str, int]]:
        """
        Get all-time routing metrics for all channels (since plugin started tracking).

        Returns:
            Dict[channel_id] -> dict with:
              fees_earned_sats, volume_routed_sats, forward_count,
              sourced_volume_sats, sourced_fee_contribution_sats, sourced_forward_count
        """
        conn = self._get_connection()

        # Exclude current day from daily stats to prevent double-counting
        # with forwards that haven't been pruned yet (boundary day fix).
        today_start = (int(time.time()) // 86400) * 86400

        revenue: Dict[str, Dict[str, int]] = {}

        # Exit-side aggregation (out_channel)
        exit_rows = conn.execute("""
            SELECT channel_id,
                   COALESCE(SUM(fee_msat), 0) as fee_msat,
                   COALESCE(SUM(out_msat), 0) as out_msat,
                   COALESCE(SUM(cnt), 0) as forward_count
            FROM (
                SELECT out_channel as channel_id,
                       SUM(fee_msat) as fee_msat,
                       SUM(out_msat) as out_msat,
                       COUNT(*) as cnt
                FROM forwards
                GROUP BY out_channel
                UNION ALL
                SELECT channel_id as channel_id,
                       SUM(total_fee_msat) as fee_msat,
                       SUM(total_out_msat) as out_msat,
                       SUM(forward_count) as cnt
                FROM daily_forwarding_stats
                WHERE date < ?
                GROUP BY channel_id
            )
            GROUP BY channel_id
        """, (today_start,)).fetchall()

        for r in exit_rows:
            cid = r["channel_id"]
            if not cid:
                continue
            revenue[cid] = {
                "fees_earned_sats": int((r["fee_msat"] or 0) // 1000),
                "volume_routed_sats": int((r["out_msat"] or 0) // 1000),
                "forward_count": int(r["forward_count"] or 0),
                "sourced_volume_sats": 0,
                "sourced_fee_contribution_sats": 0,
                "sourced_forward_count": 0,
            }

        # Entry-side aggregation (in_channel)
        inbound_rows = conn.execute("""
            SELECT channel_id,
                   COALESCE(SUM(in_msat), 0) as in_msat,
                   COALESCE(SUM(fee_msat), 0) as fee_msat,
                   COALESCE(SUM(cnt), 0) as forward_count
            FROM (
                SELECT in_channel as channel_id,
                       SUM(in_msat) as in_msat,
                       SUM(fee_msat) as fee_msat,
                       COUNT(*) as cnt
                FROM forwards
                GROUP BY in_channel
                UNION ALL
                SELECT channel_id as channel_id,
                       SUM(total_in_msat) as in_msat,
                       SUM(total_fee_msat) as fee_msat,
                       SUM(forward_count) as cnt
                FROM daily_forwarding_stats_inbound
                WHERE date < ?
                GROUP BY channel_id
            )
            GROUP BY channel_id
        """, (today_start,)).fetchall()

        for r in inbound_rows:
            cid = r["channel_id"]
            if not cid:
                continue
            if cid not in revenue:
                revenue[cid] = {
                    "fees_earned_sats": 0,
                    "volume_routed_sats": 0,
                    "forward_count": 0,
                    "sourced_volume_sats": 0,
                    "sourced_fee_contribution_sats": 0,
                    "sourced_forward_count": 0,
                }
            revenue[cid]["sourced_volume_sats"] = int((r["in_msat"] or 0) // 1000)
            revenue[cid]["sourced_fee_contribution_sats"] = int((r["fee_msat"] or 0) // 1000)
            revenue[cid]["sourced_forward_count"] = int(r["forward_count"] or 0)

        return revenue

    def get_revenue_by_size_bucket(self, channel_id: str, window_days: int = 7) -> dict:
        """Return {label: total_fee_msat} grouped by payment size bucket."""
        return _revenue_by_size_bucket_sql(self._get_connection(), channel_id, window_days)

    # =========================================================================
    # Atomic Budget Reservation System (CRITICAL-01 fix)
    # =========================================================================

    def reserve_budget(self, reservation_id: str, amount_sats: int,
                      channel_id: str, budget_limit: int,
                      since_timestamp: int,
                      weekly_budget_limit: int = None,
                      weekly_since_timestamp: int = None) -> Tuple[bool, int]:
        """
        Atomically reserve budget for a rebalance operation.

        This prevents race conditions where multiple concurrent jobs could
        all pass budget validation before any of them records their spend.

        The reservation is atomic: it checks current spend + active reservations
        and only creates the reservation if within budget, all in one transaction.

        Args:
            reservation_id: Unique ID for this reservation (e.g., rebalance_id)
            amount_sats: Amount to reserve
            channel_id: Channel this reservation is for
            budget_limit: Maximum daily budget in sats
            since_timestamp: Start of budget period (e.g., 24h ago)
            weekly_budget_limit: Optional maximum weekly budget in sats
            weekly_since_timestamp: Optional start of weekly budget period (e.g., 7d ago)

        Returns:
            Tuple of (success: bool, remaining_budget: int)
        """
        conn = self._get_connection()
        try:
            return _reserve_budget_atomic(
                conn, reservation_id, amount_sats, channel_id,
                budget_limit, since_timestamp,
                weekly_budget_limit, weekly_since_timestamp,
            )
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception as rb_err:
                self.plugin.log(f"ROLLBACK failed in reserve_budget: {rb_err}", level='error')
            self.plugin.log(f"Budget reservation failed: {e}", level='error')
            return (False, 0)

    def release_budget_reservation(self, reservation_id: str) -> bool:
        """
        Release a budget reservation (job failed/timed out without spending).

        Args:
            reservation_id: The reservation to release

        Returns:
            True if released, False if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            UPDATE budget_reservations
            SET status = 'released'
            WHERE reservation_id = ? AND status = 'active'
        """, (reservation_id,))
        return cursor.rowcount > 0

    def mark_budget_spent(self, reservation_id: str, actual_spent: int) -> bool:
        """
        Mark a reservation as spent (job completed successfully).

        The actual spend is recorded in rebalance_history, so we just
        update the reservation status to prevent double-counting.

        Args:
            reservation_id: The reservation that was spent
            actual_spent: Actual amount spent (for logging)

        Returns:
            True if marked, False if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            UPDATE budget_reservations
            SET status = 'spent'
            WHERE reservation_id = ? AND status = 'active'
        """, (reservation_id,))
        return cursor.rowcount > 0

    def cleanup_stale_reservations(self, max_age_seconds: int = 14400) -> int:
        """
        Clean up stale reservations older than max_age.

        Reservations that are still 'active' after max_age are likely
        from crashed jobs and should be released.

        Args:
            max_age_seconds: Maximum age before auto-release (default 4 hours)

        Returns:
            Number of stale reservations cleaned up
        """
        conn = self._get_connection()
        cutoff = int(time.time()) - max_age_seconds
        cursor = conn.execute("""
            UPDATE budget_reservations
            SET status = 'released'
            WHERE status = 'active' AND reserved_at < ?
        """, (cutoff,))
        count = cursor.rowcount
        if count > 0:
            self.plugin.log(f"Cleaned up {count} stale budget reservations", level='info')
        return count

    def count_stale_reservations(self, max_age_seconds: int = 14400) -> int:
        """
        Count stale reservations without releasing them (Issue #24).

        Useful for monitoring/debug output to see how many reservations
        are stale without triggering cleanup.

        Args:
            max_age_seconds: Age threshold for "stale" (default 4 hours)

        Returns:
            Number of stale active reservations
        """
        conn = self._get_connection()
        cutoff = int(time.time()) - max_age_seconds
        result = conn.execute("""
            SELECT COUNT(*) as count
            FROM budget_reservations
            WHERE status = 'active' AND reserved_at < ?
        """, (cutoff,)).fetchone()
        return result['count'] if result else 0

    def clear_all_reservations(self) -> Dict[str, Any]:
        """
        Clear ALL active budget reservations (Issue #33).

        Use this to reset the reservation system when sling jobs are
        manually stopped or stuck. Releases all active reservations
        regardless of age.

        Returns:
            Dict with count of cleared reservations and total amount released
        """
        conn = self._get_connection()

        # AUDIT FIX I-1: Wrap in BEGIN IMMEDIATE to prevent TOCTOU race.
        # Without this, a new reservation created between the read and update
        # would be silently released.
        try:
            conn.execute("BEGIN IMMEDIATE")

            # First get stats on what we're clearing
            stats = conn.execute("""
                SELECT COUNT(*) as count, COALESCE(SUM(reserved_sats), 0) as total_sats
                FROM budget_reservations
                WHERE status = 'active'
            """).fetchone()

            count = stats['count'] if stats else 0
            total_sats = stats['total_sats'] if stats else 0

            # Release all active reservations
            conn.execute("""
                UPDATE budget_reservations
                SET status = 'released'
                WHERE status = 'active'
            """)

            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        if count > 0:
            self.plugin.log(
                f"Cleared {count} active budget reservations ({total_sats} sats)",
                level='info'
            )

        return {
            "cleared_count": count,
            "released_sats": total_sats
        }

    # =========================================================================
    # Generic Spend Ledger (Unified Budget Gating for all spending categories)
    # =========================================================================

    def reserve_spend(self, reservation_id: str, amount_sats: int, category: str,
                     subcategory: Optional[str] = None, reference_id: Optional[str] = None,
                     channel_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a generic spend reservation (best-effort, caller enforces budget)."""
        conn = self._get_connection()
        now = int(time.time())
        amount = self._sanitize_amount(amount_sats, "reserved_sats")
        if amount <= 0:
            return False
        rid = str(reservation_id or "").strip()
        if not rid:
            return False
        cat = str(category or "").strip().lower()
        if not cat:
            return False
        meta_json = json.dumps(metadata or {}, sort_keys=True) if metadata else None
        try:
            conn.execute("""
                INSERT OR REPLACE INTO spend_reservations
                (reservation_id, category, subcategory, reserved_sats, reserved_at, reference_id, channel_id, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """, (rid, cat, subcategory, amount, now, reference_id, channel_id, meta_json))
            return True
        except Exception as e:
            self.plugin.log(f"Spend reservation failed: {e}", level='error')
            return False

    def release_spend_reservation(self, reservation_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.execute("""
            UPDATE spend_reservations
            SET status = 'released'
            WHERE reservation_id = ? AND status = 'active'
        """, (str(reservation_id),))
        return cursor.rowcount > 0

    def mark_spend_reservation_spent(self, reservation_id: str, actual_spent_sats: Optional[int] = None,
                                     source: Optional[str] = None, record_event: bool = False) -> bool:
        """Mark reservation spent; optionally also record a spend event."""
        conn = self._get_connection()
        rid = str(reservation_id or "").strip()
        if not rid:
            return False
        row = conn.execute("""
            SELECT * FROM spend_reservations WHERE reservation_id = ?
        """, (rid,)).fetchone()
        if not row:
            return False
        cursor = conn.execute("""
            UPDATE spend_reservations
            SET status = 'spent'
            WHERE reservation_id = ? AND status = 'active'
        """, (rid,))
        changed = cursor.rowcount > 0
        if changed and record_event:
            amount = self._sanitize_amount(actual_spent_sats if actual_spent_sats is not None else row["reserved_sats"], "actual_spent_sats")
            self.record_spend_event(
                event_id=f"resv:{rid}",
                category=row["category"],
                amount_sats=amount,
                subcategory=row["subcategory"],
                reference_id=row["reference_id"],
                channel_id=row["channel_id"],
                source=source or "reservation_settlement",
                metadata={"reservation_id": rid},
            )
        return changed

    def record_spend_event(self, event_id: str, category: str, amount_sats: int,
                          subcategory: Optional[str] = None, timestamp: Optional[int] = None,
                          reference_id: Optional[str] = None, channel_id: Optional[str] = None,
                          source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a generic spend event for categories not covered by canonical tables."""
        conn = self._get_connection()
        eid = str(event_id or "").strip()
        cat = str(category or "").strip().lower()
        if not eid or not cat:
            return False
        ts = int(timestamp or time.time())
        amount = self._sanitize_amount(amount_sats, "amount_sats")
        meta_json = json.dumps(metadata or {}, sort_keys=True) if metadata else None
        try:
            conn.execute("""
                INSERT OR REPLACE INTO spend_events
                (event_id, category, subcategory, amount_sats, timestamp, reference_id, channel_id, source, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (eid, cat, subcategory, amount, ts, reference_id, channel_id, source, meta_json))
            return True
        except Exception as e:
            self.plugin.log(f"Record spend event failed: {e}", level='error')
            return False

    def cleanup_stale_spend_reservations(self, max_age_seconds: int = 86400, category: Optional[str] = None) -> int:
        conn = self._get_connection()
        cutoff = int(time.time()) - max_age_seconds
        params = [cutoff]
        category_clause = ""
        if category:
            category_clause = " AND category = ?"
            params.append(str(category).strip().lower())
        cursor = conn.execute(f"""
            UPDATE spend_reservations
            SET status = 'released'
            WHERE status = 'active' AND reserved_at < ?{category_clause}
        """, tuple(params))
        return cursor.rowcount

    def release_spend_reservations(
        self,
        category: Optional[str] = None,
        older_than_seconds: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Release active spend reservations with optional filters for safe operational recovery."""
        conn = self._get_connection()
        where = ["status = 'active'"]
        params: List[Any] = []

        if category:
            where.append("category = ?")
            params.append(str(category).strip().lower())

        if older_than_seconds is not None and int(older_than_seconds) > 0:
            cutoff = int(time.time()) - int(older_than_seconds)
            where.append("reserved_at < ?")
            params.append(cutoff)

        where_sql = " AND ".join(where)
        limit_sql = f" LIMIT {int(limit)}" if limit is not None and int(limit) > 0 else ""

        rows = conn.execute(
            f"""
            SELECT reservation_id, reserved_sats
            FROM spend_reservations
            WHERE {where_sql}
            ORDER BY reserved_at ASC{limit_sql}
            """,
            tuple(params),
        ).fetchall()

        if not rows:
            return {"released_count": 0, "released_sats": 0, "reservation_ids": []}

        ids = [str(r["reservation_id"]) for r in rows]
        released_sats = int(sum(int(r["reserved_sats"] or 0) for r in rows))

        qmarks = ",".join(["?"] * len(ids))
        conn.execute(
            f"UPDATE spend_reservations SET status = 'released' WHERE reservation_id IN ({qmarks})",
            tuple(ids),
        )
        return {
            "released_count": len(ids),
            "released_sats": released_sats,
            "reservation_ids": ids,
        }

    def get_spend_ledger_summary(
        self,
        window_hours: int = 24,
        include_reservations: bool = False,
        reservation_limit: int = 50,
    ) -> Dict[str, Any]:
        """Summarize generic spend ledger events + reservations for unified budget accounting."""
        conn = self._get_connection()
        cutoff = int(time.time()) - (int(window_hours) * 3600)

        spent_row = conn.execute("""
            SELECT COALESCE(SUM(amount_sats), 0) AS total_spent
            FROM spend_events
            WHERE timestamp >= ?
        """, (cutoff,)).fetchone()

        reserved_row = conn.execute("""
            SELECT COALESCE(SUM(reserved_sats), 0) AS total_reserved
            FROM spend_reservations
            WHERE status = 'active' AND reserved_at >= ?
        """, (cutoff,)).fetchone()

        by_cat_events = conn.execute("""
            SELECT category, COALESCE(SUM(amount_sats), 0) AS total
            FROM spend_events
            WHERE timestamp >= ?
            GROUP BY category
        """, (cutoff,)).fetchall()
        by_cat_resv = conn.execute("""
            SELECT category, COALESCE(SUM(reserved_sats), 0) AS total
            FROM spend_reservations
            WHERE status = 'active' AND reserved_at >= ?
            GROUP BY category
        """, (cutoff,)).fetchall()

        spent_by_category = {str(r["category"]): int(r["total"] or 0) for r in by_cat_events}
        reserved_by_category = {str(r["category"]): int(r["total"] or 0) for r in by_cat_resv}

        result = {
            "window_hours": int(window_hours),
            "spent_24h_sats": int((spent_row["total_spent"] if spent_row else 0) or 0),
            "reserved_24h_sats": int((reserved_row["total_reserved"] if reserved_row else 0) or 0),
            "spent_by_category": spent_by_category,
            "reserved_by_category": reserved_by_category,
        }

        if include_reservations:
            rows = conn.execute(
                """
                SELECT reservation_id, category, subcategory, reserved_sats, reserved_at,
                       reference_id, channel_id, status, metadata_json
                FROM spend_reservations
                WHERE status = 'active' AND reserved_at >= ?
                ORDER BY reserved_at ASC
                LIMIT ?
                """,
                (cutoff, max(1, int(reservation_limit))),
            ).fetchall()
            now = int(time.time())
            result["active_reservations"] = [
                {
                    "reservation_id": str(r["reservation_id"]),
                    "category": str(r["category"]),
                    "subcategory": r["subcategory"],
                    "reserved_sats": int(r["reserved_sats"] or 0),
                    "reserved_at": int(r["reserved_at"] or 0),
                    "age_seconds": max(0, now - int(r["reserved_at"] or now)),
                    "reference_id": r["reference_id"],
                    "channel_id": r["channel_id"],
                    "status": r["status"],
                    "metadata_json": r["metadata_json"],
                }
                for r in rows
            ]

        return result

    def get_daily_rebalance_spend(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get rebalance spending summary for the specified window (Issue #23).

        Provides a comprehensive view of rebalance spending for debugging
        and monitoring the capital controls system.

        Args:
            window_hours: Time window in hours (default 24)

        Returns:
            Dict with:
            - total_spent_sats: Actual fees paid for successful rebalances
            - total_reserved_sats: Currently reserved budget (active jobs)
            - job_count: Total number of rebalance attempts
            - success_count: Number of successful rebalances
            - failed_count: Number of failed rebalances
            - success_rate: Percentage of successful rebalances
            - stale_reservations: Count of reservations older than timeout
            - window_hours: The time window used
        """
        conn = self._get_connection()
        cutoff = int(time.time()) - (window_hours * 3600)

        # Get job counts from rebalance history (status lifecycle source of truth)
        stats = conn.execute("""
            SELECT
                COUNT(*) as job_count,
                COALESCE(SUM(CASE WHEN status IN ('success', 'failed', 'partial', 'timeout') THEN 1 ELSE 0 END), 0) as completed_count,
                COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) as success_count,
                COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) as failed_count
            FROM rebalance_history
            WHERE timestamp >= ?
        """, (cutoff,)).fetchone()

        # Get actual spend from rebalance_costs (source of truth for fee accounting)
        spent_row = conn.execute("""
            SELECT COALESCE(SUM(cost_sats), 0) as total_spent
            FROM rebalance_costs
            WHERE timestamp >= ?
        """, (cutoff,)).fetchone()

        # Get current active reservations
        reserved = conn.execute("""
            SELECT COALESCE(SUM(reserved_sats), 0) as total_reserved
            FROM budget_reservations
            WHERE status = 'active' AND reserved_at >= ?
        """, (cutoff,)).fetchone()

        # Get stale reservation count (Issue #24)
        stale_count = self.count_stale_reservations()

        total_spent = spent_row['total_spent'] if spent_row else 0
        job_count = stats['job_count'] if stats else 0
        completed_count = stats['completed_count'] if stats else 0
        success_count = stats['success_count'] if stats else 0
        failed_count = stats['failed_count'] if stats else 0
        total_reserved = reserved['total_reserved'] if reserved else 0

        # Calculate success rate based on completed jobs only (exclude pending_async)
        success_rate = (success_count / completed_count * 100) if completed_count > 0 else 0.0

        return {
            'total_spent_sats': total_spent,
            'spend_source': 'rebalance_costs',
            'total_reserved_sats': total_reserved,
            'job_count': job_count,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': round(success_rate, 1),
            'stale_reservations': stale_count,
            'window_hours': window_hours
        }

    def get_budget_status(self, since_timestamp: int) -> Dict[str, int]:
        """
        Get current budget status including reservations.

        Args:
            since_timestamp: Start of budget period

        Returns:
            Dict with 'spent', 'reserved', 'total_committed' keys
        """
        conn = self._get_connection()

        # AUDIT FIX C-1: Wrap in transaction for consistent snapshot.
        # Without this, a rebalance completing between the two reads
        # could move sats from reserved to spent, skewing the total.
        try:
            conn.execute("BEGIN")
            spent_row = conn.execute("""
                SELECT COALESCE(SUM(cost_sats), 0) as spent
                FROM rebalance_costs
                WHERE timestamp >= ?
            """, (since_timestamp,)).fetchone()

            reserved_row = conn.execute("""
                SELECT COALESCE(SUM(reserved_sats), 0) as reserved
                FROM budget_reservations
                WHERE status = 'active' AND reserved_at >= ?
            """, (since_timestamp,)).fetchone()
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        spent = spent_row['spent'] if spent_row else 0
        reserved = reserved_row['reserved'] if reserved_row else 0

        return {
            'spent': spent,
            'reserved': reserved,
            'total_committed': spent + reserved
        }

    def get_rebalance_history_by_peer(self, peer_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get rebalance history for channels belonging to a specific peer.

        Joins rebalance_history with channel_states to find channels
        for the given peer, then returns rebalances to those channels.

        Args:
            peer_id: The peer node ID
            limit: Maximum records to return (clamped to [1, 1000])

        Returns:
            List of rebalance records with fee and amount info.
            Note: fee_paid_msat is in millisatoshis (actual_fee_sats * 1000)
        """
        limit = max(1, min(limit, 1000))
        conn = self._get_connection()
        
        # First get all channels for this peer
        peer_channels = conn.execute("""
            SELECT channel_id FROM channel_states WHERE peer_id = ?
        """, (peer_id,)).fetchall()
        
        if not peer_channels:
            return []
        
        channel_ids = [row['channel_id'] for row in peer_channels]
        placeholders = ','.join('?' * len(channel_ids))
        
        # Get rebalances to these channels
        # Note: actual_fee_sats is stored in sats, convert to msat for fee_paid_msat
        rows = conn.execute(f"""
            SELECT
                to_channel,
                amount_sats,
                max_fee_sats,
                COALESCE(actual_fee_sats, 0) * 1000 as fee_paid_msat,
                amount_sats * 1000 as amount_msat,
                status,
                timestamp
            FROM rebalance_history
            WHERE to_channel IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT ?
        """, (*channel_ids, limit)).fetchall()
        
        return [dict(row) for row in rows]

    def get_historical_inbound_fee_ppm(self, peer_id: str, window_days: int = 30,
                                        min_samples: int = 3) -> Optional[Dict[str, Any]]:
        """
        Get historical average fee PPM paid to rebalance TO a peer.

        Uses actual fees from successful rebalances to estimate inbound routing
        costs more accurately than heuristics. Returns None if insufficient data.

        Args:
            peer_id: The destination peer node ID
            window_days: Lookback window in days (default 30)
            min_samples: Minimum successful rebalances required (default 3)

        Returns:
            Dict with:
                - avg_fee_ppm: Weighted average fee in PPM
                - median_fee_ppm: Median fee in PPM (more robust to outliers)
                - sample_count: Number of successful rebalances
                - total_volume_sats: Total sats rebalanced
                - confidence: 'high' (10+ samples), 'medium' (5-9), 'low' (3-4)
            Or None if insufficient data.
        """
        conn = self._get_connection()
        since = int(time.time()) - (window_days * 86400)

        # Get channels for this peer
        peer_channels = conn.execute("""
            SELECT channel_id FROM channel_states WHERE peer_id = ?
        """, (peer_id,)).fetchall()

        if not peer_channels:
            return None

        channel_ids = [row['channel_id'] for row in peer_channels]
        placeholders = ','.join('?' * len(channel_ids))

        # Get successful rebalances with fee data
        rows = conn.execute(f"""
            SELECT
                amount_sats,
                actual_fee_sats,
                (actual_fee_sats * 1000000) / NULLIF(amount_sats, 0) as fee_ppm
            FROM rebalance_history
            WHERE to_channel IN ({placeholders})
              AND status = 'success'
              AND actual_fee_sats IS NOT NULL
              AND actual_fee_sats > 0
              AND amount_sats > 0
              AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (*channel_ids, since)).fetchall()

        if len(rows) < min_samples:
            return None

        # Calculate weighted average (by volume)
        total_fees = sum(row['actual_fee_sats'] for row in rows)
        total_volume = sum(row['amount_sats'] for row in rows)

        if total_volume == 0:
            return None

        avg_fee_ppm = (total_fees * 1_000_000) // total_volume

        # Calculate median (more robust to outliers)
        fee_ppms = sorted([row['fee_ppm'] for row in rows if row['fee_ppm'] is not None])
        if fee_ppms:
            mid = len(fee_ppms) // 2
            if len(fee_ppms) % 2 == 0:
                median_fee_ppm = (fee_ppms[mid - 1] + fee_ppms[mid]) // 2
            else:
                median_fee_ppm = fee_ppms[mid]
        else:
            median_fee_ppm = avg_fee_ppm

        # Confidence based on sample size
        sample_count = len(rows)
        if sample_count >= 10:
            confidence = 'high'
        elif sample_count >= 5:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'avg_fee_ppm': int(avg_fee_ppm),
            'median_fee_ppm': int(median_fee_ppm),
            'sample_count': sample_count,
            'total_volume_sats': total_volume,
            'confidence': confidence
        }

    # =========================================================================
    # Forward Tracking Methods
    # =========================================================================

    def get_latest_forward_timestamp(self) -> Optional[int]:
        """
        Get the timestamp of the most recent forward in the database.
        
        Used for hydration on startup to determine how far back to query
        listforwards RPC to fill any gaps while the plugin was offline.
        
        Returns:
            Unix timestamp of the latest forward, or None if table is empty
        """
        conn = self._get_connection()
        row = conn.execute("SELECT MAX(timestamp) as max_ts FROM forwards").fetchone()
        return row['max_ts'] if row and row['max_ts'] else None
    
    
    def bulk_insert_forwards(self, forwards: list) -> int:
            """
            Bulk insert forwards from RPC hydration.

            Idempotent insert using INSERT OR IGNORE under a UNIQUE index.
            Wrapped in a single transaction for performance (10-100x faster than
            autocommit per-row on large forward sets).

            Args:
                forwards: List of dicts with keys:
                          in_channel, out_channel, in_msat, out_msat, fee_msat,
                          received_time (timestamp), resolved_time (optional),
                          resolution_time (optional)

            Returns:
                Number of forwards inserted (best-effort count)
            """
            conn = self._get_connection()
            inserted = 0

            conn.execute("BEGIN IMMEDIATE")
            try:
                for fwd in forwards:
                    try:
                        in_chan = normalize_scid(fwd.get('in_channel', '') or '')
                        out_chan = normalize_scid(fwd.get('out_channel', '') or '')
                        ts = int(fwd.get('received_time', 0) or 0)
                        rt = int(fwd.get('resolved_time', 0) or 0)
                        res_dur = float(fwd.get('resolution_time', 0) or 0)
                        if rt <= 0 and res_dur and ts:
                            rt = ts + int(res_dur)

                        cur = conn.execute("""
                            INSERT OR IGNORE INTO forwards
                            (in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time, timestamp, resolved_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            in_chan,
                            out_chan,
                            int(fwd.get('in_msat', 0) or 0),
                            int(fwd.get('out_msat', 0) or 0),
                            int(fwd.get('fee_msat', 0) or 0),
                            res_dur,
                            ts,
                            rt
                        ))
                        # sqlite3 cursor.rowcount is 1 for inserted, 0 for ignored
                        if getattr(cur, "rowcount", 0) == 1:
                            inserted += 1
                    except Exception as e:
                        self.plugin.log(f"bulk_insert_forwards: skipped invalid record: {e}", level='debug')
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception as rb_err:
                    self.plugin.log(f"ROLLBACK failed in bulk_insert_forwards: {rb_err}", level='error')
                raise

            return inserted

    def get_daily_flow_buckets(self, window_days: int = 7, channel_id: Optional[str] = None) -> Dict[str, list]:
        """
        Get daily flow buckets from the local forwards table.

        This replaces the listforwards RPC call for flow analysis, providing
        the same data structure but from local SQLite aggregation.

        v2.0: Now also returns count and last_ts per bucket for confidence calculation.

        Args:
            window_days: Number of days to look back
            channel_id: Optional specific channel to query (None = all channels)

        Returns:
            Dict mapping channel_id to a list of daily buckets:
            {'scid': [{'in': 100, 'out': 50, 'count': 5, 'last_ts': 1234567890}, ...]}
            where index 0 = today, index 1 = yesterday, etc.
        """
        conn = self._get_connection()
        now = int(time.time())
        start_time = now - (window_days * 86400)

        flow_data: Dict[str, list] = {}

        def init_bucket():
            """Initialize a single day bucket with v2.0 fields."""
            return {'in': 0, 'out': 0, 'count': 0, 'last_ts': 0}

        # SQL aggregation (keeps flow analysis CPU/memory bounded on high-volume nodes).
        #
        # Each forward contributes to two channels:
        # - in_channel: add to 'in'
        # - out_channel: add to 'out'
        #
        # We UNION the two projections and then group.
        if channel_id:
            # Filter to a single channel efficiently (avoid scanning other channel_ids).
            query = """
                WITH flow AS (
                    SELECT
                        in_channel AS channel_id,
                        CAST((? - timestamp) / 86400 AS INTEGER) AS age_days,
                        SUM(in_msat) AS in_msat,
                        0 AS out_msat,
                        COUNT(*) AS cnt,
                        MAX(timestamp) AS last_ts
                    FROM forwards
                    WHERE timestamp >= ? AND in_channel = ?
                    GROUP BY in_channel, age_days

                    UNION ALL

                    SELECT
                        out_channel AS channel_id,
                        CAST((? - timestamp) / 86400 AS INTEGER) AS age_days,
                        0 AS in_msat,
                        SUM(out_msat) AS out_msat,
                        COUNT(*) AS cnt,
                        MAX(timestamp) AS last_ts
                    FROM forwards
                    WHERE timestamp >= ? AND out_channel = ?
                    GROUP BY out_channel, age_days
                )
                SELECT
                    channel_id,
                    age_days,
                    SUM(in_msat) AS in_msat,
                    SUM(out_msat) AS out_msat,
                    SUM(cnt) AS count,
                    MAX(last_ts) AS last_ts
                FROM flow
                WHERE age_days >= 0 AND age_days < ?
                GROUP BY channel_id, age_days
            """
            params = (now, start_time, channel_id, now, start_time, channel_id, window_days)
        else:
            query = """
                WITH flow AS (
                    SELECT
                        in_channel AS channel_id,
                        CAST((? - timestamp) / 86400 AS INTEGER) AS age_days,
                        SUM(in_msat) AS in_msat,
                        0 AS out_msat,
                        COUNT(*) AS cnt,
                        MAX(timestamp) AS last_ts
                    FROM forwards
                    WHERE timestamp >= ?
                    GROUP BY in_channel, age_days

                    UNION ALL

                    SELECT
                        out_channel AS channel_id,
                        CAST((? - timestamp) / 86400 AS INTEGER) AS age_days,
                        0 AS in_msat,
                        SUM(out_msat) AS out_msat,
                        COUNT(*) AS cnt,
                        MAX(timestamp) AS last_ts
                    FROM forwards
                    WHERE timestamp >= ?
                    GROUP BY out_channel, age_days
                )
                SELECT
                    channel_id,
                    age_days,
                    SUM(in_msat) AS in_msat,
                    SUM(out_msat) AS out_msat,
                    SUM(cnt) AS count,
                    MAX(last_ts) AS last_ts
                FROM flow
                WHERE age_days >= 0 AND age_days < ?
                GROUP BY channel_id, age_days
            """
            params = (now, start_time, now, start_time, window_days)

        rows = conn.execute(query, params).fetchall()
        for row in rows:
            cid = row["channel_id"]
            if not cid:
                continue
            age_days = int(row["age_days"])
            if age_days < 0 or age_days >= window_days:
                continue
            if cid not in flow_data:
                flow_data[cid] = [init_bucket() for _ in range(window_days)]
            bucket = flow_data[cid][age_days]
            bucket["in"] += int(row["in_msat"] or 0) // 1000
            bucket["out"] += int(row["out_msat"] or 0) // 1000
            bucket["count"] += int(row["count"] or 0)
            last_ts = int(row["last_ts"] or 0)
            if last_ts > bucket["last_ts"]:
                bucket["last_ts"] = last_ts

        return flow_data

    def get_continuous_net_flow_all(self, window_hours: int = 168) -> Dict[str, List[Dict[str, Any]]]:
        """Per-forward net flow with timestamps for all channels.

        Returns {channel_id: [{timestamp, net_msat}, ...]} sorted by timestamp DESC.
        Unlike get_daily_flow_buckets_all() (calendar-day grouping), this returns
        individual forwards for continuous-time weighting in the Kalman filter.
        """
        conn = self._get_connection()
        cutoff = int(time.time()) - window_hours * 3600

        rows = conn.execute("""
            SELECT out_channel AS channel_id, timestamp, out_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            UNION ALL
            SELECT in_channel AS channel_id, timestamp, -in_msat AS net_msat
            FROM forwards WHERE timestamp >= ?
            ORDER BY channel_id, timestamp DESC
        """, (cutoff, cutoff)).fetchall()

        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            cid = row["channel_id"]
            if not cid:
                continue
            if cid not in result:
                result[cid] = []
            result[cid].append({
                "timestamp": int(row["timestamp"]),
                "net_msat": int(row["net_msat"]),
            })
        return result

    def get_continuous_net_flow_channel(self, channel_id: str, window_hours: int = 168) -> List[Dict[str, Any]]:
        """Per-forward net flow for a single channel.

        More efficient than get_continuous_net_flow_all() when only one
        channel is needed (e.g., analyze_channel single-channel path).
        """
        conn = self._get_connection()
        cutoff = int(time.time()) - window_hours * 3600

        rows = conn.execute("""
            SELECT timestamp, out_msat AS net_msat
            FROM forwards WHERE out_channel = ? AND timestamp >= ?
            UNION ALL
            SELECT timestamp, -in_msat AS net_msat
            FROM forwards WHERE in_channel = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (channel_id, cutoff, channel_id, cutoff)).fetchall()

        return [{"timestamp": int(r["timestamp"]), "net_msat": int(r["net_msat"])} for r in rows]


    
    def record_forward(self, in_channel: str, out_channel: str,
                       in_msat: int, out_msat: int, fee_msat: int, *args) -> None:
        """
        Record a completed forward for real-time tracking.

        Use canonical forward times (received_time/resolved_time) and
        INSERT OR IGNORE under a UNIQUE index to prevent double-dips on restart.

        Backward compatible:
          - Legacy call: record_forward(in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time)
          - Canonical call: record_forward(in_channel, out_channel, in_msat, out_msat, fee_msat,
                                         received_time, resolved_time[, resolution_time])
        """
        # Parse legacy vs canonical call patterns
        received_time: int = 0
        resolved_time: int = 0
        resolution_time: float = 0.0

        if len(args) == 1:
            # Legacy path: only resolution_time provided; use wall-clock as best-effort
            resolution_time = float(args[0] or 0)
            received_time = int(time.time())
            resolved_time = received_time + int(resolution_time) if resolution_time > 0 else 0

            # One-time warning to surface version skew without breaking ingestion
            if not getattr(self, "_warned_legacy_record_forward", False):
                self._warned_legacy_record_forward = True
                self.plugin.log(
                    "Warning: legacy Database.record_forward() call signature detected "
                    "(missing received_time/resolved_time). Using wall-clock timestamps; "
                    "please update cl-revenue-ops.py to pass canonical times.",
                    level="warn"
                )

        elif len(args) == 2:
            received_time = int(args[0] or 0)
            resolved_time = int(args[1] or 0)
            resolution_time = 0.0

        elif len(args) >= 3:
            received_time = int(args[0] or 0)
            resolved_time = int(args[1] or 0)
            resolution_time = float(args[2] or 0)

        else:
            raise TypeError("record_forward() missing required timing arguments")

        conn = self._get_connection()

        # Normalize SCIDs for consistency
        in_channel = (in_channel or "").replace(":", "x")
        out_channel = (out_channel or "").replace(":", "x")

        # Best-effort derive times if missing
        ts = int(received_time or 0) or int(time.time())
        rt = int(resolved_time or 0)
        if rt <= 0 and resolution_time and resolution_time > 0:
            rt = ts + int(resolution_time)

        # MAJOR-10 FIX: Return whether this was a new insert or duplicate
        # Use cursor.rowcount to detect if INSERT OR IGNORE actually inserted
        cursor = conn.execute("""
            INSERT OR IGNORE INTO forwards
            (in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time, timestamp, resolved_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (in_channel, out_channel, int(in_msat), int(out_msat), int(fee_msat),
                float(resolution_time or 0), ts, rt))

        # Log duplicate detection for observability
        if cursor.rowcount == 0:
            self.plugin.log(
                f"Forward duplicate detected (already recorded): "
                f"{in_channel[:12]}... -> {out_channel[:12]}... "
                f"(received={received_time}, resolved={resolved_time})",
                level='debug'
            )

    def get_channel_forwards(self, channel_id: str, since_timestamp: int) -> Dict[str, int]:
        """Get aggregate forward stats for a channel since a timestamp."""
        conn = self._get_connection()
        
        # Get inbound flow (channel received HTLCs)
        row_in = conn.execute("""
            SELECT COALESCE(SUM(in_msat), 0) as total_in_msat
            FROM forwards
            WHERE in_channel = ? AND timestamp >= ?
        """, (channel_id, since_timestamp)).fetchone()
        
        # Get outbound flow (channel sent HTLCs)
        row_out = conn.execute("""
            SELECT COALESCE(SUM(out_msat), 0) as total_out_msat
            FROM forwards
            WHERE out_channel = ? AND timestamp >= ?
        """, (channel_id, since_timestamp)).fetchone()
        
        return {
            'in_msat': row_in['total_in_msat'] if row_in else 0,
            'out_msat': row_out['total_out_msat'] if row_out else 0
        }
    
    def get_volume_since(self, channel_id: str, timestamp: int) -> int:
        """
        Get total outbound volume for a channel since a specific timestamp.
        
        This is used by the Fee Controller to measure volume specifically
        since the last fee change, rather than using a 7-day average which
        dilutes the signal of recent fee changes.
        
        Args:
            channel_id: Channel to get volume for
            timestamp: Unix timestamp to start counting from
            
        Returns:
            Total outbound volume in satoshis since the timestamp
        """
        conn = self._get_connection()
        
        row = conn.execute("""
            SELECT COALESCE(SUM(out_msat), 0) as total_out_msat
            FROM forwards
            WHERE out_channel = ? AND timestamp > ?
        """, (channel_id, timestamp)).fetchone()

        # Convert msat to sats
        return (row['total_out_msat'] // 1000) if row else 0

    def get_forward_count_since(self, channel_id: str, timestamp: int) -> int:
        """
        Get number of forwards for a channel since a specific timestamp.

        Used by dynamic observation windows (Improvement #2) to determine
        when we have enough data points for statistically significant fee decisions.

        Args:
            channel_id: Channel to count forwards for
            timestamp: Unix timestamp to start counting from

        Returns:
            Number of forwards since the timestamp
        """
        conn = self._get_connection()

        row = conn.execute("""
            SELECT COUNT(*) as forward_count
            FROM forwards
            WHERE out_channel = ? AND timestamp > ?
        """, (channel_id, timestamp)).fetchone()

        return row['forward_count'] if row else 0

    def get_last_forward_time(self, channel_id: str) -> Optional[int]:
        """
        Get the timestamp of the most recent forward through a channel.

        Used by flow-based ceiling reduction (Issue #20) to determine
        how long a channel has been without routing activity.

        Args:
            channel_id: Channel to check

        Returns:
            Unix timestamp of last forward, or None if no forwards found
        """
        conn = self._get_connection()

        row = conn.execute("""
            SELECT MAX(timestamp) as last_ts
            FROM forwards
            WHERE out_channel = ?
        """, (channel_id,)).fetchone()

        return row['last_ts'] if row and row['last_ts'] else None

    def get_weighted_volume_since(self, channel_id: str, timestamp: int) -> int:
        """
        Get reputation-weighted outbound volume for a channel since a timestamp.
        
        This method weights each forward by the reputation score of the incoming
        peer. Peers with high failure rates have their volume discounted, preventing
        spammy peers from influencing fee decisions.
        
        Effective Volume = Raw Volume * Peer_Success_Rate
        
        The calculation:
        1. Select forwards where out_channel = channel_id and time > timestamp
        2. Join forwards.in_channel with channel_states.channel_id to find peer_id
        3. Join peer_id with peer_reputation to get success/failure counts
        4. Calculate score = success_count / (success_count + failure_count)
        5. Edge case: If total == 0 or peer not found, score = 1.0 (innocent until proven guilty)
        6. Return SUM(forwards.out_msat * score) in satoshis
        
        Args:
            channel_id: Channel to get weighted volume for
            timestamp: Unix timestamp to start counting from
            
        Returns:
            Total reputation-weighted outbound volume in satoshis
        """
        conn = self._get_connection()
        
        # Advanced SQL JOIN query using Laplace Smoothing (add-one smoothing)
        # Formula: Score = (success_count + 1) / (total_count + 2)
        # 
        # Laplace smoothing effects:
        # - 0/0 -> (0+1)/(0+2) = 0.5 (Neutral start for new peers)
        # - 0/1 -> (0+1)/(1+2) = 0.33 (Not harshly punished)
        # - 100/100 -> (100+1)/(100+2) = 0.99 (Not perfect 1.0)
        #
        # For peers not in peer_reputation table, we use 0.5 (neutral)
        row = conn.execute("""
            SELECT COALESCE(
                SUM(
                    f.out_msat * 
                    CASE 
                        WHEN pr.success_count IS NULL THEN 0.5
                        ELSE CAST(pr.success_count + 1 AS REAL) / 
                             CAST(pr.success_count + pr.failure_count + 2 AS REAL)
                    END
                ),
                0
            ) as weighted_out_msat
            FROM forwards f
            LEFT JOIN channel_states cs ON f.in_channel = cs.channel_id
            LEFT JOIN peer_reputation pr ON cs.peer_id = pr.peer_id
            WHERE f.out_channel = ? AND f.timestamp > ?
        """, (channel_id, timestamp)).fetchone()
        
        # Convert msat to sats
        return int(row['weighted_out_msat'] // 1000) if row else 0
    
    def get_daily_volume(self, days: int = 7) -> int:
        """Get total routing volume over the past N days."""
        conn = self._get_connection()
        since = int(time.time()) - (days * 86400)
        
        row = conn.execute("""
            SELECT COALESCE(SUM(out_msat), 0) as total_volume_msat
            FROM forwards
            WHERE timestamp >= ?
        """, (since,)).fetchone()
        
        return row['total_volume_msat'] // 1000 if row else 0

    def get_total_volume_since(self, since_timestamp: int) -> int:
        """Get total routing volume since a timestamp (in sats)."""
        conn = self._get_connection()
        
        row = conn.execute("""
            SELECT COALESCE(SUM(out_msat), 0) as total_volume_msat
            FROM forwards
            WHERE timestamp >= ?
        """, (since_timestamp,)).fetchone()
        
        return row['total_volume_msat'] // 1000 if row else 0

    def get_total_forward_count_since(self, since_timestamp: int) -> int:
        """Get total forward count since a timestamp."""
        conn = self._get_connection()
        
        row = conn.execute("""
            SELECT COUNT(*) as count
            FROM forwards
            WHERE timestamp >= ?
        """, (since_timestamp,)).fetchone()
        
        return row['count'] if row else 0
    
    def get_peer_latency_stats(self, peer_id: str, window_seconds: int = 86400) -> Dict[str, float]:
        """
        Calculate latency (resolution_time) statistics for a peer.
        
        Args:
            peer_id: The peer to analyze
            window_seconds: Lookback window in seconds (default 24h)
            
        Returns:
            Dict with 'avg' and 'std' of resolution_time
        """
        conn = self._get_connection()
        since = int(time.time()) - window_seconds
        
        # Join forwards with channel_states to filter by peer_id
        # We look at out_channel because that's where the capital was tied up
        rows = conn.execute("""
            SELECT f.resolution_time
            FROM forwards f
            JOIN channel_states cs ON f.out_channel = cs.channel_id
            WHERE cs.peer_id = ? AND f.timestamp >= ?
        """, (peer_id, since)).fetchall()
        
        if not rows:
            return {'avg': 0.0, 'std': 0.0}
            
        times = [row['resolution_time'] for row in rows if row['resolution_time'] and row['resolution_time'] > 0]
        n = len(times)
        if n == 0:
            return {'avg': 0.0, 'std': 0.0}
        avg = sum(times) / n

        if n < 2:
            return {'avg': avg, 'std': 0.0}

        variance = sum((x - avg) ** 2 for x in times) / (n - 1)
        std = math.sqrt(variance)
        
        return {'avg': avg, 'std': std}
    
    # =========================================================================
    # Profitability Tracking Methods
    # =========================================================================
    
    def record_channel_open_cost(self, channel_id: str, peer_id: str,
                                  open_cost_sats: int, capacity_sats: int,
                                  timestamp: Optional[int] = None):
        """Record the cost to open a channel."""
        conn = self._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO channel_costs 
            (channel_id, peer_id, open_cost_sats, capacity_sats, opened_at)
            VALUES (?, ?, ?, ?, ?)
        """, (channel_id, peer_id, open_cost_sats, capacity_sats, 
              timestamp or int(time.time())))

    def get_channel_open_cost(self, channel_id: str) -> Optional[int]:
        """Get the recorded open cost for a channel."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT open_cost_sats FROM channel_costs WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        return row["open_cost_sats"] if row else None

    def get_channel_cost(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        C5 FIX: Get full channel cost record for closure archival.

        Returns the complete channel_costs row as a dict, used by
        _archive_closed_channel() in cl-revenue-ops.py.

        Args:
            channel_id: The channel short ID

        Returns:
            Dict with keys: channel_id, peer_id, open_cost_sats, capacity_sats, opened_at
            or None if no record exists.
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM channel_costs WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        return dict(row) if row else None
    
    def record_rebalance_cost(self, channel_id: str, peer_id: str,
                              cost_sats: int, amount_sats: int,
                              timestamp: Optional[int] = None):
        """Record a rebalance cost for a channel."""
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO rebalance_costs 
            (channel_id, peer_id, cost_sats, amount_sats, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (channel_id, peer_id, cost_sats, amount_sats,
              timestamp or int(time.time())))

    def get_channel_rebalance_costs(self, channel_id: str) -> int:
        """Get total rebalance costs for a channel."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_sats), 0) as total FROM rebalance_costs WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        return row["total"] if row else 0
    
    def get_channel_cost_history(self, channel_id: str) -> List[Dict[str, Any]]:
        """Get detailed cost history for a channel."""
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT * FROM rebalance_costs 
            WHERE channel_id = ? 
            ORDER BY timestamp DESC
        """, (channel_id,)).fetchall()
        return [dict(row) for row in rows]
    
    def get_channel_rebalance_success_rate(
        self, channel_id: str, window_days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get per-channel rebalance success rate and cost efficiency.

        Uses existing rebalance_history table.  No schema changes needed.

        Returns:
            Dict with total, successes, failures, success_rate (0.0-1.0),
            avg_cost_ppm, and avg_amount_sats.
            Returns None if no rebalance history for this channel.
        """
        conn = self._get_connection()
        cutoff = int(time.time()) - (window_days * 86400)

        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
                AVG(CASE WHEN status = 'success' AND amount_sats > 0
                    THEN (actual_fee_sats * 1000000.0 / amount_sats)
                    ELSE NULL END) as avg_cost_ppm,
                AVG(CASE WHEN status = 'success' THEN amount_sats ELSE NULL END) as avg_amount
            FROM rebalance_history
            WHERE to_channel = ? AND timestamp >= ?
        """, (channel_id, cutoff)).fetchone()

        if not row or row['total'] == 0:
            return None

        total = row['total']
        successes = row['successes'] or 0
        return {
            'total': total,
            'successes': successes,
            'failures': row['failures'] or 0,
            'success_rate': successes / total if total > 0 else 0.0,
            'avg_cost_ppm': int(row['avg_cost_ppm'] or 0),
            'avg_amount_sats': int(row['avg_amount'] or 0),
        }

    def get_all_channel_costs(self) -> Dict[str, Dict[str, int]]:
        """Get summary of costs for all channels."""
        conn = self._get_connection()
        
        result = {}
        
        # Get open costs
        open_rows = conn.execute("SELECT * FROM channel_costs").fetchall()
        for row in open_rows:
            channel_id = row["channel_id"]
            result[channel_id] = {
                "open_cost_sats": row["open_cost_sats"],
                "rebalance_cost_sats": 0,
                "total_cost_sats": row["open_cost_sats"]
            }
        
        # Get rebalance costs
        rebalance_rows = conn.execute("""
            SELECT channel_id, SUM(cost_sats) as total 
            FROM rebalance_costs 
            GROUP BY channel_id
        """).fetchall()
        
        for row in rebalance_rows:
            channel_id = row["channel_id"]
            if channel_id not in result:
                result[channel_id] = {
                    "open_cost_sats": 0,
                    "rebalance_cost_sats": 0,
                    "total_cost_sats": 0
                }
            result[channel_id]["rebalance_cost_sats"] = row["total"]
            result[channel_id]["total_cost_sats"] = (
                result[channel_id]["open_cost_sats"] + row["total"]
            )
        
        return result
    
    def get_lifetime_stats(self) -> Dict[str, int]:
        """
        Get lifetime aggregate financial statistics.
        
        Returns aggregate data from ALL channels, including closed ones,
        to provide a true "Lifetime P&L" view.
        
        This combines:
        - Current data in the forwards table (recent, not yet pruned)
        - Historical aggregates from lifetime_aggregates (data from pruned rows)
        
        Returns:
            Dictionary with:
                - total_revenue_msat: Sum of all routing fees earned (including pruned)
                - total_rebalance_cost_sats: Sum of all rebalancing fees paid
                - total_opening_cost_sats: Sum of all channel opening costs
                - total_forwards: Count of all completed forwards (including pruned)
        """
        conn = self._get_connection()

        # Exclude current day from daily stats to prevent double-counting
        # with forwards that haven't been pruned yet (boundary day fix).
        today_start = (int(time.time()) // 86400) * 86400

        # Get pruned historical aggregates from lifetime_aggregates table
        lifetime_row = conn.execute(
            "SELECT pruned_revenue_msat, pruned_forward_count FROM lifetime_aggregates WHERE id = 1"
        ).fetchone()
        pruned_revenue_msat = lifetime_row["pruned_revenue_msat"] if lifetime_row else 0
        pruned_forward_count = lifetime_row["pruned_forward_count"] if lifetime_row else 0

        # Current revenue from forwards table (in msat) - not yet pruned
        revenue_row = conn.execute(
            "SELECT COALESCE(SUM(fee_msat), 0) as total FROM forwards"
        ).fetchone()
        current_revenue_msat = revenue_row["total"] if revenue_row else 0

        # Rolled-up revenue from daily_forwarding_stats (exclude today to avoid
        # double-counting with forwards on the boundary day)
        rollup_revenue_row = conn.execute(
            "SELECT COALESCE(SUM(total_fee_msat), 0) as total FROM daily_forwarding_stats WHERE date < ?",
            (today_start,)
        ).fetchone()
        rollup_revenue_msat = rollup_revenue_row["total"] if rollup_revenue_row else 0

        # Combine pruned (legacy) + rolled-up + current
        total_revenue_msat = pruned_revenue_msat + rollup_revenue_msat + current_revenue_msat
        
        # Total rebalance costs from rebalance_costs table (aggregated source of truth)
        rebalance_row = conn.execute(
            "SELECT COALESCE(SUM(cost_sats), 0) as total FROM rebalance_costs"
        ).fetchone()
        total_rebalance_cost_sats = rebalance_row["total"] if rebalance_row else 0
        
        # Total opening costs from channel_costs table
        opening_row = conn.execute(
            "SELECT COALESCE(SUM(open_cost_sats), 0) as total FROM channel_costs"
        ).fetchone()
        total_opening_cost_sats = opening_row["total"] if opening_row else 0

        # Total closure costs from channel_closure_costs table (Accounting v2.0)
        closure_row = conn.execute(
            "SELECT COALESCE(SUM(total_closure_cost_sats), 0) as total FROM channel_closure_costs"
        ).fetchone()
        total_closure_cost_sats = closure_row["total"] if closure_row else 0

        # Current forward count from forwards table
        count_row = conn.execute(
            "SELECT COUNT(*) as total FROM forwards"
        ).fetchone()
        current_forwards = count_row["total"] if count_row else 0
        
        # Rolled-up forward count (exclude today, same boundary fix as revenue)
        rollup_count_row = conn.execute(
            "SELECT COALESCE(SUM(forward_count), 0) as total FROM daily_forwarding_stats WHERE date < ?",
            (today_start,)
        ).fetchone()
        rollup_forwards = rollup_count_row["total"] if rollup_count_row else 0
        
        # Combine pruned (legacy) + rolled-up + current
        total_forwards = pruned_forward_count + rollup_forwards + current_forwards
        
        return {
            "total_revenue_msat": total_revenue_msat,
            "total_rebalance_cost_sats": total_rebalance_cost_sats,
            "total_opening_cost_sats": total_opening_cost_sats,
            "total_closure_cost_sats": total_closure_cost_sats,  # Accounting v2.0
            "total_forwards": total_forwards
        }
    
    # =========================================================================
    # Channel Closure Cost Tracking (Accounting v2.0)
    # =========================================================================

    def record_channel_closure(
        self,
        channel_id: str,
        peer_id: str,
        close_type: str,
        closure_fee_sats: int,
        htlc_sweep_fee_sats: int = 0,
        penalty_fee_sats: int = 0,
        funding_txid: Optional[str] = None,
        closing_txid: Optional[str] = None
    ) -> bool:
        """
        Record channel closure costs for accurate P&L accounting.

        This method is called when a channel transitions to ONCHAIN or CLOSED state.
        It queries bookkeeper for actual on-chain fees and stores them.

        Args:
            channel_id: The channel short ID (SCID)
            peer_id: The peer node ID
            close_type: Type of closure ('mutual', 'local_unilateral', 'remote_unilateral', 'unknown')
            closure_fee_sats: Base on-chain fee for the closing transaction
            htlc_sweep_fee_sats: Additional fees for sweeping pending HTLCs
            penalty_fee_sats: Penalty fees (if we were penalized - rare)
            funding_txid: The original funding transaction ID
            closing_txid: The closing transaction ID

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            # Security: Input validation
            if not self._validate_channel_id(channel_id):
                self.plugin.log(
                    f"Security: Invalid channel_id format '{channel_id}', skipping closure record",
                    level='warn'
                )
                return False

            if not self._validate_peer_id(peer_id):
                self.plugin.log(
                    f"Security: Invalid peer_id format for {channel_id}, rejecting closure record",
                    level='warn'
                )
                return False

            # Security: Sanitize fee values
            closure_fee_sats = self._sanitize_fee(closure_fee_sats, "closure_fee")
            htlc_sweep_fee_sats = self._sanitize_fee(htlc_sweep_fee_sats, "htlc_sweep_fee")
            penalty_fee_sats = self._sanitize_fee(penalty_fee_sats, "penalty_fee")

            # Validate close_type
            valid_close_types = {'mutual', 'local_unilateral', 'remote_unilateral', 'unknown'}
            if close_type not in valid_close_types:
                self.plugin.log(
                    f"Security: Invalid close_type '{close_type}', defaulting to 'unknown'",
                    level='warn'
                )
                close_type = 'unknown'

            conn = self._get_connection()
            now = int(time.time())

            total_closure_cost = closure_fee_sats + htlc_sweep_fee_sats + penalty_fee_sats

            conn.execute("""
                INSERT OR REPLACE INTO channel_closure_costs
                (channel_id, peer_id, close_type, closure_fee_sats, htlc_sweep_fee_sats,
                 penalty_fee_sats, total_closure_cost_sats, funding_txid, closing_txid,
                 closed_at, resolution_complete)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (channel_id, peer_id, close_type, closure_fee_sats, htlc_sweep_fee_sats,
                  penalty_fee_sats, total_closure_cost, funding_txid, closing_txid, now))

            self.plugin.log(
                f"Recorded closure cost for {channel_id}: {total_closure_cost} sats "
                f"(type={close_type}, base={closure_fee_sats}, htlc_sweep={htlc_sweep_fee_sats})",
                level='info'
            )
            return True

        except Exception as e:
            self.plugin.log(f"Error recording closure cost for {channel_id}: {e}", level='error')
            return False

    def update_closure_resolution(self, channel_id: str, additional_fees: int = 0) -> bool:
        """
        Update closure record when additional resolution occurs (HTLC sweeps, etc.).

        Called when the bookkeeper reports additional on-chain activity for a closed channel.

        Args:
            channel_id: The channel short ID
            additional_fees: Additional fees incurred during resolution

        Returns:
            True if updated successfully
        """
        try:
            conn = self._get_connection()
            additional_fees = self._sanitize_fee(additional_fees, "additional_fees")

            if additional_fees > 0:
                conn.execute("""
                    UPDATE channel_closure_costs
                    SET htlc_sweep_fee_sats = htlc_sweep_fee_sats + ?,
                        total_closure_cost_sats = total_closure_cost_sats + ?
                    WHERE channel_id = ?
                """, (additional_fees, additional_fees, channel_id))

            return True

        except Exception as e:
            self.plugin.log(f"Error updating closure resolution for {channel_id}: {e}", level='error')
            return False

    def mark_closure_complete(self, channel_id: str) -> bool:
        """
        Mark a channel closure as fully resolved (all outputs swept).

        Called when bookkeeper indicates all on-chain outputs are resolved.

        Args:
            channel_id: The channel short ID

        Returns:
            True if marked successfully
        """
        try:
            conn = self._get_connection()
            conn.execute("""
                UPDATE channel_closure_costs
                SET resolution_complete = 1
                WHERE channel_id = ?
            """, (channel_id,))

            self.plugin.log(f"Marked closure complete for {channel_id}", level='debug')
            return True

        except Exception as e:
            self.plugin.log(f"Error marking closure complete for {channel_id}: {e}", level='error')
            return False

    def get_channel_closure_cost(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get closure cost details for a specific channel.

        Args:
            channel_id: The channel short ID

        Returns:
            Dict with closure cost details, or None if not found
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT * FROM channel_closure_costs WHERE channel_id = ?
        """, (channel_id,)).fetchone()

        return dict(row) if row else None

    def get_total_closure_costs(self) -> int:
        """
        Get the total closure costs across all channels.

        Returns:
            Total closure costs in sats
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(total_closure_cost_sats), 0) as total
            FROM channel_closure_costs
        """).fetchone()

        return row["total"] if row else 0

    def get_opening_costs_since(self, since_timestamp: int) -> int:
        """
        Get channel opening costs since a specific timestamp.

        Args:
            since_timestamp: Unix timestamp to query from

        Returns:
            Total channel open costs in sats since the timestamp
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(open_cost_sats), 0) as total
            FROM channel_costs
            WHERE opened_at >= ?
        """, (since_timestamp,)).fetchone()
        return row["total"] if row else 0

    def get_closure_costs_since(self, since_timestamp: int) -> int:
        """
        Get closure costs since a specific timestamp (for windowed P&L).

        Args:
            since_timestamp: Unix timestamp to query from

        Returns:
            Total closure costs in sats since the timestamp
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(total_closure_cost_sats), 0) as total
            FROM channel_closure_costs
            WHERE closed_at >= ?
        """, (since_timestamp,)).fetchone()

        return row["total"] if row else 0

    def record_closed_channel_history(
        self,
        channel_id: str,
        peer_id: str,
        capacity_sats: int,
        opened_at: Optional[int],
        closed_at: int,
        close_type: str,
        open_cost_sats: int,
        closure_cost_sats: int,
        total_revenue_sats: int,
        total_rebalance_cost_sats: int,
        forward_count: int,
        funding_txid: Optional[str] = None,
        closing_txid: Optional[str] = None,
        closer: str = 'unknown'
    ) -> bool:
        """
        Record complete P&L history for a closed channel.

        This preserves the accounting data for a channel after it closes,
        ensuring accurate lifetime P&L calculations.

        Args:
            channel_id: The channel short ID
            peer_id: The peer node ID
            capacity_sats: Channel capacity
            opened_at: Unix timestamp when channel opened (may be None)
            closed_at: Unix timestamp when channel closed
            close_type: Type of closure
            open_cost_sats: On-chain fees paid to open
            closure_cost_sats: On-chain fees paid to close
            total_revenue_sats: Total routing fees earned
            total_rebalance_cost_sats: Total rebalancing fees paid
            forward_count: Number of successful forwards
            funding_txid: Funding transaction ID
            closing_txid: Closing transaction ID
            closer: Who initiated closure: 'local', 'remote', 'mutual', or 'unknown'

        Returns:
            True if recorded successfully
        """
        try:
            conn = self._get_connection()

            # Input validation (consistent with record_channel_closure)
            capacity_sats = self._sanitize_amount(capacity_sats, "capacity_sats")
            open_cost_sats = self._sanitize_amount(open_cost_sats, "open_cost_sats")
            closure_cost_sats = self._sanitize_amount(closure_cost_sats, "closure_cost_sats")
            total_revenue_sats = self._sanitize_amount(total_revenue_sats, "total_revenue_sats")
            total_rebalance_cost_sats = self._sanitize_amount(total_rebalance_cost_sats, "total_rebalance_cost_sats")
            forward_count = max(0, int(forward_count or 0))
            # M12 FIX: Include 'local_unilateral' and 'remote_unilateral' to match
            # _determine_close_type() output and record_channel_closure() validation.
            _VALID_CLOSE_TYPES = ('mutual', 'local_unilateral', 'remote_unilateral',
                                 'unilateral', 'force', 'onchain', 'unknown')
            if close_type not in _VALID_CLOSE_TYPES:
                close_type = 'unknown'
            _VALID_CLOSERS = ('local', 'remote', 'mutual', 'unknown')
            if closer not in _VALID_CLOSERS:
                closer = 'unknown'

            # Calculate derived values
            # M15 FIX: Clamp to zero to handle clock skew where closed_at < opened_at
            days_open = max(0, ((closed_at - opened_at) // 86400)) if opened_at else 0
            net_pnl = total_revenue_sats - (open_cost_sats + closure_cost_sats + total_rebalance_cost_sats)

            conn.execute("""
                INSERT OR REPLACE INTO closed_channels
                (channel_id, peer_id, capacity_sats, opened_at, closed_at, close_type,
                 open_cost_sats, closure_cost_sats, total_revenue_sats, total_rebalance_cost_sats,
                 forward_count, net_pnl_sats, days_open, funding_txid, closing_txid, closer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (channel_id, peer_id, capacity_sats, opened_at, closed_at, close_type,
                  open_cost_sats, closure_cost_sats, total_revenue_sats, total_rebalance_cost_sats,
                  forward_count, net_pnl, days_open, funding_txid, closing_txid, closer))

            self.plugin.log(
                f"Recorded closed channel history for {channel_id}: "
                f"net_pnl={net_pnl} sats, days_open={days_open}, closer={closer}",
                level='info'
            )
            return True

        except Exception as e:
            self.plugin.log(f"Error recording closed channel history for {channel_id}: {e}", level='error')
            return False

    def get_closed_channel_history(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get P&L history for a closed channel.

        Args:
            channel_id: The channel short ID

        Returns:
            Dict with closed channel history, or None if not found
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT * FROM closed_channels WHERE channel_id = ?
        """, (channel_id,)).fetchone()

        return dict(row) if row else None

    def get_all_closed_channels(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history for all closed channels.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of closed channel dicts, ordered by closure date (most recent first)
        """
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT * FROM closed_channels
            ORDER BY closed_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]

    def get_closed_channels_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all closed channels.

        Returns:
            Dict with aggregate statistics for closed channels
        """
        conn = self._get_connection()

        row = conn.execute("""
            SELECT
                COUNT(*) as channel_count,
                COALESCE(SUM(capacity_sats), 0) as total_capacity,
                COALESCE(SUM(open_cost_sats), 0) as total_open_costs,
                COALESCE(SUM(closure_cost_sats), 0) as total_closure_costs,
                COALESCE(SUM(total_revenue_sats), 0) as total_revenue,
                COALESCE(SUM(total_rebalance_cost_sats), 0) as total_rebalance_costs,
                COALESCE(SUM(forward_count), 0) as total_forwards,
                COALESCE(SUM(net_pnl_sats), 0) as total_net_pnl,
                COALESCE(AVG(days_open), 0) as avg_days_open
            FROM closed_channels
        """).fetchone()

        return dict(row) if row else {
            "channel_count": 0,
            "total_capacity": 0,
            "total_open_costs": 0,
            "total_closure_costs": 0,
            "total_revenue": 0,
            "total_rebalance_costs": 0,
            "total_forwards": 0,
            "total_net_pnl": 0,
            "avg_days_open": 0
        }

    def get_peer_closed_channel_profit_summary(
        self,
        peer_id: str,
        lookback_days: int = 30,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Summarize recent closed-channel profitability for a peer.

        This supports "hot/profit inheritance" when a peer replaces/closes a
        channel and the new channel has little history yet.
        """
        if not peer_id:
            return {"peer_id": "", "count": 0}

        lookback_days = max(1, min(int(lookback_days or 30), 365))
        limit = max(1, min(int(limit or 5), 20))
        cutoff = int(time.time()) - (lookback_days * 86400)
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT *
            FROM closed_channels
            WHERE peer_id = ? AND closed_at >= ?
            ORDER BY closed_at DESC
            LIMIT ?
        """, (peer_id, cutoff, limit)).fetchall()
        items = [dict(r) for r in rows]
        if not items:
            return {"peer_id": str(peer_id), "count": 0, "lookback_days": lookback_days}

        total_revenue = sum(int(i.get("total_revenue_sats", 0) or 0) for i in items)
        total_rebal_cost = sum(int(i.get("total_rebalance_cost_sats", 0) or 0) for i in items)
        total_net = sum(int(i.get("net_pnl_sats", 0) or 0) for i in items)
        total_days = sum(max(0, int(i.get("days_open", 0) or 0)) for i in items)
        total_forwards = sum(max(0, int(i.get("forward_count", 0) or 0)) for i in items)
        remote_close_count = sum(1 for i in items if str(i.get("closer", "unknown")) == "remote")
        most_recent_closed_at = max(int(i.get("closed_at", 0) or 0) for i in items)

        effective_days = max(1, total_days)
        daily_revenue_est_sats = int(total_revenue / effective_days)
        daily_net_est_sats = int(total_net / effective_days)
        if total_rebal_cost <= 0:
            marginal_roi_proxy = 1.0 if total_revenue > 0 else 0.0
        else:
            marginal_roi_proxy = (total_revenue - total_rebal_cost) / max(1, total_rebal_cost)

        return {
            "peer_id": str(peer_id),
            "count": len(items),
            "lookback_days": lookback_days,
            "total_revenue_sats": total_revenue,
            "total_rebalance_cost_sats": total_rebal_cost,
            "total_net_pnl_sats": total_net,
            "total_days_open": total_days,
            "total_forward_count": total_forwards,
            "daily_revenue_est_sats": daily_revenue_est_sats,
            "daily_net_est_sats": daily_net_est_sats,
            "marginal_roi_proxy": float(marginal_roi_proxy),
            "remote_close_count": remote_close_count,
            "most_recent_closed_at": most_recent_closed_at,
        }

    def remove_closed_channel_data(self, channel_id: str, peer_id: Optional[str] = None) -> Dict[str, int]:
        """
        Remove a closed channel from active tracking tables.

        After a channel is archived to closed_channels, we should clean up
        the active tracking data to prevent stale state and reduce db bloat.

        Args:
            channel_id: The channel short ID
            peer_id: Optional peer ID

        Returns:
            Dict with counts of deleted rows per table
        """
        deleted = {
            "channel_states": 0,
            "channel_failures": 0,
            "channel_probes": 0,
            "kalman_state": 0
        }

        try:
            conn = self._get_connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Remove from channel_states
                cursor = conn.execute(
                    "DELETE FROM channel_states WHERE channel_id = ?",
                    (channel_id,)
                )
                deleted["channel_states"] = cursor.rowcount

                # Remove from channel_failures
                cursor = conn.execute(
                    "DELETE FROM channel_failures WHERE channel_id = ?",
                    (channel_id,)
                )
                deleted["channel_failures"] = cursor.rowcount

                # Remove from channel_probes
                cursor = conn.execute(
                    "DELETE FROM channel_probes WHERE channel_id = ?",
                    (channel_id,)
                )
                deleted["channel_probes"] = cursor.rowcount

                # Remove Kalman filter state for closed channel
                cursor = conn.execute(
                    "DELETE FROM kalman_state WHERE channel_id = ?",
                    (channel_id,)
                )
                deleted["kalman_state"] = cursor.rowcount

                # L11 FIX: Clean up fee_strategy_state to prevent
                # orphaned rows from accumulating on nodes with frequent channel turnover.
                try:
                    cursor = conn.execute(
                        "DELETE FROM fee_strategy_state WHERE channel_id = ?",
                        (channel_id,)
                    )
                    deleted["fee_strategy_state"] = cursor.rowcount
                except Exception:
                    pass  # Table may not exist on older schemas

                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

            total = sum(deleted.values())
            if total > 0:
                self.plugin.log(
                    f"Cleaned up closed channel {channel_id}: {deleted}",
                    level='info'
                )

            return deleted

        except Exception as e:
            self.plugin.log(
                f"Error cleaning up closed channel {channel_id}: {e}",
                level='error'
            )
            return deleted

    # =========================================================================
    # Channel Failure Tracking Methods (Persistent Backoff)
    # =========================================================================

    def get_failure_count(self, channel_id: str) -> Tuple[int, int]:
        """
        Get the failure count and last failure time for a channel.
        
        Used by the rebalancer for adaptive backoff logic.
        
        Args:
            channel_id: Channel to query
            
        Returns:
            Tuple of (failure_count, last_failure_time)
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT failure_count, last_failure_time FROM channel_failures WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        
        if row:
            return (row["failure_count"], row["last_failure_time"])
        return (0, 0)
    
    def increment_failure_count(self, channel_id: str,
                                attempted_ppm: int = 0,
                                attempted_amount: int = 0,
                                error_type: str = "") -> int:
        """
        Increment the failure count for a channel and update last failure time.

        Called when a rebalance attempt fails. Uses atomic upsert to avoid
        read-then-write race conditions.

        Args:
            channel_id: Channel that failed
            attempted_ppm: Fee PPM used in the failed attempt
            attempted_amount: Amount in msat of the failed attempt
            error_type: Classification of the failure (e.g. "no_route", "timeout")

        Returns:
            New failure count
        """
        conn = self._get_connection()
        now = int(time.time())

        # Use RETURNING for atomic upsert+read (SQLite 3.35+)
        row = conn.execute("""
            INSERT INTO channel_failures (channel_id, failure_count, last_failure_time,
                                          last_attempted_ppm, last_attempted_amount, last_error_type)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                failure_count = failure_count + 1,
                last_failure_time = ?,
                last_attempted_ppm = ?,
                last_attempted_amount = ?,
                last_error_type = ?
            RETURNING failure_count
        """, (channel_id, now, attempted_ppm, attempted_amount, error_type,
              now, attempted_ppm, attempted_amount, error_type)).fetchone()
        return row[0] if row else 1
    
    def reset_failure_count(self, channel_id: str) -> None:
        """
        Reset the failure count for a channel (e.g., after successful rebalance).

        Args:
            channel_id: Channel to reset
        """
        conn = self._get_connection()
        conn.execute(
            "DELETE FROM channel_failures WHERE channel_id = ?",
            (channel_id,)
        )

    def get_failure_metadata(self, channel_id: str) -> dict:
        """Get failure tracking metadata for a channel.

        Args:
            channel_id: Channel to query

        Returns:
            Dict with last_attempted_ppm, last_attempted_amount, last_error_type
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT last_attempted_ppm, last_attempted_amount, last_error_type "
            "FROM channel_failures WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        if row:
            return {
                "last_attempted_ppm": row["last_attempted_ppm"],
                "last_attempted_amount": row["last_attempted_amount"],
                "last_error_type": row["last_error_type"],
            }
        return {"last_attempted_ppm": 0, "last_attempted_amount": 0, "last_error_type": ""}

    def get_all_failure_counts(self) -> Dict[str, Tuple[int, int]]:
        """
        Get failure counts for all channels with recorded failures.
        
        Returns:
            Dict mapping channel_id to (failure_count, last_failure_time)
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM channel_failures").fetchall()
        return {row["channel_id"]: (row["failure_count"], row["last_failure_time"]) for row in rows}
    
    # =========================================================================
    # Peer Reputation Methods
    # =========================================================================
    
    def update_peer_reputation(self, peer_id: str, is_success: bool):
        """
        Update peer reputation based on forward success or failure.
        
        Tracks success/failure counts for each peer to calculate routing
        success rates. This data is used for traffic intelligence to
        identify unreliable peers.
        
        Args:
            peer_id: The peer's node ID
            is_success: True if forward settled, False if failed
        """
        conn = self._get_connection()
        now = int(time.time())
        
        if is_success:
            conn.execute("""
                INSERT INTO peer_reputation (peer_id, success_count, failure_count, last_update)
                VALUES (?, 1, 0, ?)
                ON CONFLICT(peer_id) DO UPDATE SET
                    success_count = success_count + 1,
                    last_update = excluded.last_update
            """, (peer_id, now))
        else:
            conn.execute("""
                INSERT INTO peer_reputation (peer_id, success_count, failure_count, last_update)
                VALUES (?, 0, 1, ?)
                ON CONFLICT(peer_id) DO UPDATE SET
                    failure_count = failure_count + 1,
                    last_update = excluded.last_update
            """, (peer_id, now))
    
    def get_peer_reputation(self, peer_id: str) -> Dict[str, Any]:
        """
        Get reputation statistics for a peer.
        
        Uses Laplace Smoothing (add-one smoothing) for the score calculation:
        Score = (success_count + 1) / (total_count + 2)
        
        This provides better handling of edge cases:
        - 0/0 -> 0.5 (Neutral start for new peers)
        - 0/1 -> 0.33 (Not harshly punished for single failure)
        - 100/100 -> 0.99 (Never reaches perfect 1.0)
        
        Args:
            peer_id: The peer's node ID
            
        Returns:
            Dict with 'successes', 'failures', and 'score' (Laplace-smoothed rate 0.0-1.0)
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT success_count, failure_count, last_update FROM peer_reputation WHERE peer_id = ?",
            (peer_id,)
        ).fetchone()
        
        if row:
            successes = row["success_count"]
            failures = row["failure_count"]
            # Laplace smoothing: (successes + 1) / (total + 2)
            score = (successes + 1) / (successes + failures + 2)
            return {
                'successes': successes,
                'failures': failures,
                'score': score,
                'last_update': row["last_update"]
            }
        
        # No data yet - return default with Laplace smoothing: (0+1)/(0+2) = 0.5
        return {
            'successes': 0,
            'failures': 0,
            'score': 0.5,
            'last_update': 0
        }
    
    def get_all_peer_reputations(self) -> List[Dict[str, Any]]:
        """
        Get reputation statistics for all peers.
        
        Uses Laplace Smoothing (add-one smoothing) for score calculation:
        Score = (success_count + 1) / (total_count + 2)
        
        Returns:
            List of dicts with peer_id, successes, failures, and score (Laplace-smoothed)
        """
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT peer_id, success_count, failure_count, last_update FROM peer_reputation ORDER BY success_count + failure_count DESC"
        ).fetchall()
        
        results = []
        for row in rows:
            successes = row["success_count"]
            failures = row["failure_count"]
            # Laplace smoothing: (successes + 1) / (total + 2)
            score = (successes + 1) / (successes + failures + 2)
            results.append({
                'peer_id': row["peer_id"],
                'successes': successes,
                'failures': failures,
                'score': score,
                'last_update': row["last_update"]
            })
        
        return results
    
    def decay_reputation(self, decay_factor: float):
        """
        Apply time-based decay to all peer reputation counts.
        
        This implements reputation decay (time-windowing) so that recent behavior
        matters more than ancient history. Should be called periodically (e.g., hourly).
        
        The decay formula multiplies all counts by the decay_factor:
        - success_count = CAST(success_count * decay_factor AS INTEGER)
        - failure_count = CAST(failure_count * decay_factor AS INTEGER)
        
        With decay_factor = 0.98 applied hourly:
        - Daily decay: 0.98^24 ≈ 0.61 (old data loses ~40% weight daily)
        - Weekly decay: 0.98^168 ≈ 0.03 (data from a week ago is nearly gone)
        
        This allows peers to recover from past failures relatively quickly while
        still maintaining meaningful reputation data.
        
        Args:
            decay_factor: Multiplier to apply to counts (e.g., 0.98)
        """
        conn = self._get_connection()

        # Wrap in transaction so a concurrent update_peer_reputation between
        # the UPDATE and DELETE cannot have its fresh data immediately deleted.
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Apply decay to both success and failure counts
            # Using CAST to INTEGER naturally floors the result
            conn.execute("""
                UPDATE peer_reputation
                SET success_count = CAST(success_count * ? AS INTEGER),
                    failure_count = CAST(failure_count * ? AS INTEGER)
            """, (decay_factor, decay_factor))

            # Clean up peers that have decayed to (0, 0)
            # This prevents the table from growing indefinitely with stale entries
            conn.execute("""
                DELETE FROM peer_reputation
                WHERE success_count = 0 AND failure_count = 0
            """)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception as rb_err:
                self.plugin.log(f"ROLLBACK failed in apply_reputation_decay: {rb_err}", level='error')
            raise

    # =========================================================================
    # Cleanup Methods
    # =========================================================================
    
    def cleanup_old_data(self, days_to_keep: int = 8):
        """
        Remove old data to prevent database bloat.
        
        AGGRESSIVE PRUNING (Day 2 Task 3):
        The forwards table grows very fast on high-traffic nodes. We only need
        data for the flow_window_days (default 7 days), so we default to keeping
        8 days (7 + 1 buffer) instead of the previous 30 days.
        
        LIFETIME PRESERVATION:
        Before deleting old forwards, we aggregate their revenue and count into
        the lifetime_aggregates table. This ensures revenue-history remains
        accurate even after pruning.
        
        Args:
            days_to_keep: Number of days of data to retain (default 8)
        """
        conn = self._get_connection()
        now = int(time.time())
        cutoff = now - (days_to_keep * 86400)

        forwards_count = 0
        pruned_revenue = 0
        pruned_count = 0

        # Atomic aggregation + deletion to avoid double-counting if interrupted.
        # If we update lifetime_aggregates but fail before deleting forwards, we'd
        # count the same forwards again later. Using a transaction prevents this.
        # NOTE: We use explicit BEGIN/COMMIT because isolation_level=None
        # (autocommit) makes `with conn:` a no-op for transaction control.
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Count rows before deletion for logging
            forwards_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM forwards WHERE timestamp < ?", (cutoff,)
            ).fetchone()["cnt"]

            # LIFETIME PRESERVATION: Aggregate revenue from forwards about to be pruned
            # and store in daily_forwarding_stats for granular history.
            if forwards_count > 0:
                # Group by channel and day (86400s)
                # SQLite integer division floor handles the day bucket
                rows = conn.execute("""
                    SELECT 
                        out_channel,
                        (timestamp / 86400) * 86400 as day_ts,
                        COALESCE(SUM(in_msat), 0) as sum_in,
                        COALESCE(SUM(out_msat), 0) as sum_out,
                        COALESCE(SUM(fee_msat), 0) as sum_fee,
                        COUNT(*) as count
                    FROM forwards 
                    WHERE timestamp < ?
                    GROUP BY out_channel, day_ts
                """, (cutoff,)).fetchall()
                
                for r in rows:
                    pruned_revenue += r['sum_fee']
                    pruned_count += r['count']
                    
                    # Upsert into daily stats
                    conn.execute("""
                        INSERT INTO daily_forwarding_stats 
                        (channel_id, date, total_in_msat, total_out_msat, total_fee_msat, forward_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(channel_id, date) DO UPDATE SET
                            total_in_msat = total_in_msat + excluded.total_in_msat,
                            total_out_msat = total_out_msat + excluded.total_out_msat,
                            total_fee_msat = total_fee_msat + excluded.total_fee_msat,
                            forward_count = forward_count + excluded.forward_count
                    """, (r['out_channel'], r['day_ts'], r['sum_in'], r['sum_out'], r['sum_fee'], r['count']))

                # Also preserve entry-side contribution metrics (in_channel aggregation).
                inbound_rows = conn.execute("""
                    SELECT
                        in_channel,
                        (timestamp / 86400) * 86400 as day_ts,
                        COALESCE(SUM(in_msat), 0) as sum_in,
                        COALESCE(SUM(fee_msat), 0) as sum_fee,
                        COUNT(*) as count
                    FROM forwards
                    WHERE timestamp < ?
                    GROUP BY in_channel, day_ts
                """, (cutoff,)).fetchall()

                for r in inbound_rows:
                    conn.execute("""
                        INSERT INTO daily_forwarding_stats_inbound
                        (channel_id, date, total_in_msat, total_fee_msat, forward_count)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(channel_id, date) DO UPDATE SET
                            total_in_msat = total_in_msat + excluded.total_in_msat,
                            total_fee_msat = total_fee_msat + excluded.total_fee_msat,
                            forward_count = forward_count + excluded.forward_count
                    """, (r['in_channel'], r['day_ts'], r['sum_in'], r['sum_fee'], r['count']))
                
                # We NO LONGER update lifetime_aggregates for new data, 
                # but we leave the table alone as it contains legacy history.

            conn.execute("DELETE FROM forwards WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM peer_connection_history WHERE timestamp < ?", (cutoff,))

            # AUDIT LOG CLEANUP: Prevent unbounded growth of operational logs
            # Keep 90 days of audit history (vs 8 days for flow data)
            audit_cutoff = now - (90 * 86400)
            conn.execute("DELETE FROM fee_changes WHERE timestamp < ?", (audit_cutoff,))
            conn.execute("DELETE FROM rebalance_history WHERE timestamp < ?", (audit_cutoff,))

            # SNAPSHOT CLEANUP: Keep 1 year of financial snapshots for trend analysis
            snapshot_cutoff = now - (365 * 86400)
            conn.execute("DELETE FROM financial_snapshots WHERE timestamp < ?", (snapshot_cutoff,))

            conn.execute("COMMIT")
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            self.plugin.log(f"cleanup_old_data transaction failed: {e}", level='error')
            return  # Skip vacuum and stats logging on failure

        # L-20: Use incremental_vacuum instead of full VACUUM to avoid blocking readers.
        # Full VACUUM requires exclusive lock and copies the entire database.
        # incremental_vacuum frees pages without blocking concurrent reads.
        if forwards_count > 0:
            try:
                conn.execute("PRAGMA incremental_vacuum(100)")
                self.plugin.log("Database incremental_vacuum completed")
            except Exception as e:
                self.plugin.log(f"incremental_vacuum failed (non-fatal): {e}", level='warn')

            self.plugin.log(
                f"Preserved {pruned_revenue // 1000} sats revenue from {pruned_count} forwards before pruning"
            )

            self.plugin.log(
                f"Cleaned up data older than {days_to_keep} days: "
                f"{forwards_count} forwards rows"
            )
    
    # =========================================================================
    # Peer Connection History Methods
    # =========================================================================
    
    def record_connection_event(self, peer_id: str, event_type: str):
        """
        Record a peer connection event.
        
        Args:
            peer_id: The peer's node ID
            event_type: One of 'connected', 'disconnected', or 'snapshot'
        """
        conn = self._get_connection()
        now = int(time.time())
        
        conn.execute("""
            INSERT INTO peer_connection_history (peer_id, event_type, timestamp)
            VALUES (?, ?, ?)
        """, (peer_id, event_type, now))
    
    def has_recent_connection_history(self, peer_id: str, seconds: int) -> bool:
        """
        Check if a peer has any connection history within the given time window.
        
        Used to avoid duplicate 'snapshot' events on restart if history already exists.
        
        Args:
            peer_id: The peer's node ID
            seconds: Time window in seconds to check
            
        Returns:
            True if any events exist within the window
        """
        conn = self._get_connection()
        now = int(time.time())
        cutoff = now - seconds
        
        row = conn.execute("""
            SELECT 1 FROM peer_connection_history 
            WHERE peer_id = ? AND timestamp >= ?
            LIMIT 1
        """, (peer_id, cutoff)).fetchone()
        
        return row is not None
    
    def get_peer_uptime_percent(self, peer_id: str, duration_seconds: int) -> float:
        """
        Calculate the uptime percentage for a peer over a time window.
        
        Walks through connection events and sums time spent connected.
        Handles open-ended intervals (e.g., currently connected).
        
        Args:
            peer_id: The peer's node ID
            duration_seconds: Time window to analyze (seconds before now)
            
        Returns:
            Uptime percentage (0.0 to 100.0)
        """
        conn = self._get_connection()
        now = int(time.time())
        window_start = now - duration_seconds
        
        # 1. Determine state at window start
        # Look for the most recent event BEFORE the window to know initial state
        prior_event = conn.execute("""
            SELECT event_type FROM peer_connection_history
            WHERE peer_id = ? AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (peer_id, window_start)).fetchone()
        
        # Start state: connected if prior event was 'connected' or 'snapshot'
        is_connected = prior_event is not None and prior_event['event_type'] in ('connected', 'snapshot')
        last_interval_start = window_start
        
        # 2. Get all events in the window
        rows = conn.execute("""
            SELECT event_type, timestamp FROM peer_connection_history
            WHERE peer_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (peer_id, window_start)).fetchall()
        
        # COLD START: If no history at all (neither prior nor in window), assume 100% uptime
        if prior_event is None and not rows:
            return 100.0
            
        # Determine effective observation window
        # If we have history before the window, we use the full window.
        # If history starts inside the window, we only count time since that first event.
        if prior_event:
            effective_start = window_start
        else:
            effective_start = rows[0]['timestamp']
            
        actual_duration = now - effective_start
        
        # Avoid noise from very short windows (e.g. startup)
        if actual_duration < 60:
            return 100.0
            
        total_connected_time = 0
        last_interval_start = effective_start
        
        for row in rows:
            event_type = row['event_type']
            timestamp = row['timestamp']
            
            if is_connected:
                # We were connected until this event
                total_connected_time += timestamp - last_interval_start
            
            # Update state for next interval
            is_connected = event_type in ('connected', 'snapshot')
            last_interval_start = timestamp
        
        # 3. Handle interval from last event until now
        if is_connected:
            total_connected_time += now - last_interval_start
        
        if actual_duration <= 0:
            return 0.0
        
        uptime_pct = (total_connected_time / actual_duration) * 100.0
        return min(100.0, max(0.0, uptime_pct))
    
    # =========================================================================
    # Hot-Channel Protection Peer Overrides (operator-managed persistent set)
    # =========================================================================

    def list_hot_channel_protection_override_peers(self) -> List[Dict[str, Any]]:
        """List peers explicitly forced into hot-channel protection."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT peer_id, added_at, note, min_depletion_trigger_pct FROM hot_channel_protection_overrides ORDER BY added_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_hot_channel_protection_override_peer(self, peer_id: str, note: str = '', min_depletion_trigger_pct: Optional[float] = None) -> bool:
        """Add/update a peer override for hot-channel protection."""
        if not self._validate_peer_id(peer_id):
            raise ValueError(f"Invalid peer_id: {peer_id}")
        conn = self._get_connection()
        trigger_pct = None if min_depletion_trigger_pct is None else float(min_depletion_trigger_pct)
        if trigger_pct is not None and not (0.0 < trigger_pct <= 100.0):
            raise ValueError("min_depletion_trigger_pct must be between 0 and 100")
        conn.execute(
            "INSERT OR REPLACE INTO hot_channel_protection_overrides (peer_id, added_at, note, min_depletion_trigger_pct) VALUES (?, ?, ?, ?)",
            (peer_id, int(time.time()), str(note or ''), trigger_pct)
        )
        return True

    def remove_hot_channel_protection_override_peer(self, peer_id: str) -> bool:
        """Remove a peer override. Returns True if removed."""
        conn = self._get_connection()
        cur = conn.execute(
            "DELETE FROM hot_channel_protection_overrides WHERE peer_id = ?",
            (peer_id,)
        )
        return cur.rowcount > 0

    # =========================================================================
    # Config Overrides Methods (Dynamic Runtime Configuration)
    # =========================================================================
    
    def get_config_override(self, key: str) -> Optional[str]:
        """Get a single config override value."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT value FROM config_overrides WHERE key = ?", (key,)
        ).fetchone()
        return row['value'] if row else None

    def set_config_override(self, key: str, value: str) -> int:
        """
        Set a config override with transactional safety.

        Uses BEGIN IMMEDIATE to prevent TOCTOU race on version number.
        Without it, two concurrent calls could read the same MAX(version),
        both write version N+1 for different keys, causing the config poller
        to miss one change.

        Returns:
            New version number after update
        """
        conn = self._get_connection()
        now = int(time.time())

        conn.execute("BEGIN IMMEDIATE")
        try:
            # M-13 v2 FIX: Compute version BEFORE INSERT OR REPLACE.
            # INSERT OR REPLACE deletes the conflicting row first, then inserts.
            # If the deleted row had the highest version, MAX(version) drops and
            # the new version can equal or regress, causing missed change detection.
            current_max = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM config_overrides"
            ).fetchone()[0]
            new_version = current_max + 1
            conn.execute("""
                INSERT OR REPLACE INTO config_overrides (key, value, version, updated_at)
                VALUES (?, ?, ?, ?)
            """, (key, value, new_version, now))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Read back the version (outside transaction; reflects committed state)
        row = conn.execute("SELECT MAX(version) as max_v FROM config_overrides").fetchone()
        return row['max_v'] or 0

    def get_all_config_overrides(self) -> Dict[str, str]:
        """Get all config overrides as a dictionary.

        Filters out internal sentinel rows (keys starting with '_') to prevent
        them from leaking into user-facing displays or config application.
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT key, value FROM config_overrides").fetchall()
        return {row['key']: row['value'] for row in rows if not row['key'].startswith('_')}

    def get_config_version(self) -> int:
        """Get current config version (max version in table)."""
        conn = self._get_connection()
        row = conn.execute("SELECT MAX(version) as max_v FROM config_overrides").fetchone()
        return row['max_v'] or 0

    def delete_config_override(self, key: str) -> bool:
        """Delete a config override, returning to default."""
        conn = self._get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.execute("DELETE FROM config_overrides WHERE key = ?", (key,))
            if cursor.rowcount > 0:
                # Bump version so config poller detects the deletion.
                # Insert a sentinel row that will be overwritten on next set_config_override.
                current_max = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM config_overrides"
                ).fetchone()[0]
                new_version = current_max + 1
                now = int(time.time())
                conn.execute("""
                    INSERT OR REPLACE INTO config_overrides (key, value, version, updated_at)
                    VALUES ('_version_bump', '', ?, ?)
                """, (new_version, now))
                conn.execute("COMMIT")
                return True
            conn.execute("COMMIT")
            return False
        except Exception:
            conn.execute("ROLLBACK")
            raise
    
    # =========================================================================
    # Mempool Fee History Methods (Vegas Reflex)
    # =========================================================================
    
    def record_mempool_fee(self, sat_per_vbyte: float) -> None:
        """
        Record current mempool fee rate.
        
        Automatically prunes entries older than 48 hours to prevent bloat.
        """
        conn = self._get_connection()
        now = int(time.time())
        conn.execute(
            "INSERT INTO mempool_fee_history (sat_per_vbyte, timestamp) VALUES (?, ?)",
            (sat_per_vbyte, now)
        )
        # Prune old entries (keep 48h)
        conn.execute(
            "DELETE FROM mempool_fee_history WHERE timestamp < ?",
            (now - 172800,)
        )

    def get_mempool_ma(self, window_seconds: int = 86400) -> float:
        """
        Get moving average of mempool fees over window.
        
        Args:
            window_seconds: Time window for average (default 24h)
            
        Returns:
            Average sat/vB over the window, or 1.0 if no data
        """
        conn = self._get_connection()
        cutoff = int(time.time()) - window_seconds
        row = conn.execute(
            "SELECT AVG(sat_per_vbyte) as avg_fee FROM mempool_fee_history WHERE timestamp >= ?",
            (cutoff,)
        ).fetchone()
        return row['avg_fee'] if row and row['avg_fee'] else 1.0
    
    def close(self):
        """Close the thread-local database connection (if any).

        M14 FIX: Delegate to close_connection() which also removes the
        connection from _thread_connections tracking list.
        """
        self.close_connection()
