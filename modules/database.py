"""
Database module for cl-revenue-ops

Handles SQLite persistence for:
- Channel flow states and history
- Hill Climbing fee controller state
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


class Database:
    """
    SQLite database manager for the Revenue Operations plugin.
    
    Provides persistence for:
    - Channel states (source/sink/balanced classification)
    - Flow metrics history
    - Hill Climbing fee controller state
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
        # Thread-local storage for connections (Phase 5.5: Database Thread Safety)
        self._local = threading.local()
        
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create a thread-local database connection.
        
        Each thread gets its own isolated connection to prevent race conditions
        during concurrent database operations (e.g., when Rebalancer and Fee
        Controller run simultaneously on different timer threads).
        
        Returns:
            sqlite3.Connection: Thread-local database connection
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Create new connection for this thread
            self._local.conn = sqlite3.connect(
                self.db_path,
                isolation_level=None  # Autocommit mode
            )
            self._local.conn.row_factory = sqlite3.Row
            
            # Enable Write-Ahead Logging for better multi-thread concurrency
            # WAL allows readers and writers to operate concurrently
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            
            # Reduce "database is locked" errors under contention
            self._local.conn.execute("PRAGMA busy_timeout=5000;")
            # Reasonable durability/performance tradeoff for WAL mode
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
            self.plugin.log(
                f"Database: Created new thread-local connection (thread={threading.current_thread().name})",
                level='debug'
            )
        return self._local.conn

    def close_connection(self) -> None:
        """
        Close the thread-local database connection.

        MAJOR-11 FIX: Explicit cleanup to prevent connection leaks in
        long-running plugins with many threads.

        Should be called when a thread is about to exit or during shutdown.
        """
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                self._local.conn.close()
                self.plugin.log(
                    f"Database: Closed thread-local connection (thread={threading.current_thread().name})",
                    level='debug'
                )
            except Exception as e:
                self.plugin.log(f"Error closing connection: {e}", level='debug')
            finally:
                self._local.conn = None

    def close_all_connections(self) -> None:
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
    # Maximum reasonable splice/capacity amount (100 BTC in sats)
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
        amount_sats = int(amount_sats)
        # Allow negative for splice_out, but clamp magnitude
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
        
        # Flow history table - tracks flow over time
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flow_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                sats_in INTEGER NOT NULL,
                sats_out INTEGER NOT NULL,
                flow_ratio REAL NOT NULL,
                state TEXT NOT NULL
            )
        """)
        
        # PID state table - stores controller state per channel
        # LEGACY: Kept for backward compatibility, but Hill Climbing uses fee_strategy_state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pid_state (
                channel_id TEXT PRIMARY KEY,
                integral REAL NOT NULL DEFAULT 0,
                last_error REAL NOT NULL DEFAULT 0,
                last_fee_ppm INTEGER NOT NULL DEFAULT 0,
                last_update INTEGER NOT NULL
            )
        """)
        
        # NEW: Fee Strategy State table for Hill Climbing controller
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
        # Phase 2: Ensure forwards schema is idempotent and restart-safe
        self._migrate_forwards_schema(conn)

        
        # Clboss unmanage tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clboss_unmanaged (
                peer_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                unmanaged_at INTEGER NOT NULL,
                PRIMARY KEY (peer_id, tag)
            )
        """)
        
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
                last_failure_time INTEGER NOT NULL DEFAULT 0
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

        # Channel Probes table for Zero-Fee Probe Defibrillator
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_probes (
                channel_id TEXT PRIMARY KEY,
                probe_type TEXT NOT NULL,  -- 'zero_fee'
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
        
        # Config overrides table (Phase 7: Dynamic Runtime Configuration)
        # Stores operator overrides that persist across restarts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config_overrides (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                updated_at INTEGER NOT NULL
            )
        """)
        
        # Mempool fee history (Phase 7: Vegas Reflex MA calculation)
        # Tracks on-chain fee rates for detecting spikes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mempool_fee_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sat_per_vbyte REAL NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        
        # Create indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_history_channel ON flow_history(channel_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fee_changes_channel ON fee_changes(channel_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forwards_time ON forwards(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forwards_channels ON forwards(in_channel, out_channel)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rebalance_costs_channel ON rebalance_costs(channel_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_states_peer ON channel_states(peer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_connection_history_peer_time ON peer_connection_history(peer_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mempool_time ON mempool_fee_history(timestamp)")
        
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

        # Phase 8: Financial Snapshots for P&L Dashboard
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

        # Splice Costs table (Accounting v2.0)
        # Tracks on-chain fees for splice-in and splice-out operations
        # Splices modify channel capacity without closing/reopening
        conn.execute("""
            CREATE TABLE IF NOT EXISTS splice_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                peer_id TEXT NOT NULL,
                splice_type TEXT NOT NULL,  -- 'splice_in' or 'splice_out'
                amount_sats INTEGER NOT NULL,  -- Amount added (positive) or removed (negative)
                fee_sats INTEGER NOT NULL,
                old_capacity_sats INTEGER,
                new_capacity_sats INTEGER,
                txid TEXT,
                timestamp INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_splice_costs_channel ON splice_costs(channel_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_splice_costs_time ON splice_costs(timestamp)")
        # Prevent duplicate splice records when txid is known (Security: idempotency)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_splice_costs_unique ON splice_costs(channel_id, txid) WHERE txid IS NOT NULL")

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
        # - v2_state_json: JSON blob for complex state (historical curve, elasticity, Thompson)
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

        # v1.4 Migration: Migrate ignored_peers to peer_policies
        # This migrates legacy "ignored" peers to the new policy system
        self._migrate_ignored_peers_to_policies(conn)

        # v1.6 Migration: Add flow analysis v2.0 columns
        self._migrate_flow_v2_schema(conn)

        self.plugin.log("Database initialized successfully")
    

    def _migrate_forwards_schema(self, conn: sqlite3.Connection) -> None:
        """
        Phase 2: Make forwards ingestion idempotent and restart-safe.

        - Adds resolved_time column if missing
        - Backfills resolved_time from timestamp + resolution_time (best-effort)
        - Deduplicates existing rows prior to enforcing uniqueness
        - Creates a UNIQUE index so INSERT OR IGNORE can safely skip duplicates
        """
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
        
        # Rename old table to backup (preserve data for safety)
        if migrated_count > 0:
            conn.execute("ALTER TABLE ignored_peers RENAME TO _backup_ignored_peers")
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
        forward_count: int = 0
    ):
        """Update the current state of a channel (v2.0: includes flow metrics)."""
        conn = self._get_connection()
        now = int(time.time())

        conn.execute("""
            INSERT OR REPLACE INTO channel_states
            (channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, updated_at,
             confidence, velocity, flow_multiplier, ema_decay, forward_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, now,
              confidence, velocity, flow_multiplier, ema_decay, forward_count))

        # Also record in history
        conn.execute("""
            INSERT INTO flow_history
            (channel_id, timestamp, sats_in, sats_out, flow_ratio, state)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (channel_id, now, sats_in, sats_out, flow_ratio, state))
    
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
        """Get all channels with a specific state."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM channel_states WHERE state = ?",
            (state,)
        ).fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Channel Probe Methods
    # =========================================================================

    def set_channel_probe(self, channel_id: str, probe_type: str = 'zero_fee'):
        """Sets the probe flag for a channel."""
        conn = self._get_connection()
        now = int(time.time())
        conn.execute("""
            INSERT OR REPLACE INTO channel_probes (channel_id, probe_type, started_at)
            VALUES (?, ?, ?)
        """, (channel_id, probe_type, now))

    def get_channel_probe(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Gets the probe flag for a channel."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM channel_probes WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        return dict(row) if row else None

    def clear_channel_probe(self, channel_id: str):
        """Clears the probe flag for a channel."""
        conn = self._get_connection()
        conn.execute("DELETE FROM channel_probes WHERE channel_id = ?", (channel_id,))
    
    # =========================================================================
    # PID State Methods (DEPRECATED - table kept for migration compatibility)
    # The fee controller now uses Hill Climbing, not PID. These methods and
    # the pid_state table are retained only to prevent errors on upgrade.
    # =========================================================================
    
    def get_pid_state(self, channel_id: str) -> Dict[str, Any]:
        """Get PID controller state for a channel (DEPRECATED - unused)."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM pid_state WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        
        if row:
            return dict(row)
        
        # Return default state if not found
        return {
            'channel_id': channel_id,
            'integral': 0.0,
            'last_error': 0.0,
            'last_fee_ppm': 0,
            'last_update': 0
        }
    
    def update_pid_state(self, channel_id: str, integral: float, last_error: float, 
                         last_fee_ppm: int):
        """Update PID controller state for a channel (DEPRECATED - unused)."""
        conn = self._get_connection()
        now = int(time.time())
        
        conn.execute("""
            INSERT OR REPLACE INTO pid_state 
            (channel_id, integral, last_error, last_fee_ppm, last_update)
            VALUES (?, ?, ?, ?, ?)
        """, (channel_id, integral, last_error, last_fee_ppm, now))
    
    # =========================================================================
    # Fee Strategy State Methods (NEW - Hill Climbing Controller)
    # =========================================================================
    
    def get_fee_strategy_state(self, channel_id: str) -> Dict[str, Any]:
        """
        Get Hill Climbing fee strategy state for a channel.

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
                                   v2_state_json: str = '{}'):
        """
        Update Hill Climbing fee strategy state for a channel.

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
            v2_state_json: v2.0 - JSON blob for historical curve, elasticity, Thompson state
        """
        conn = self._get_connection()
        now = int(time.time())

        conn.execute("""
            INSERT OR REPLACE INTO fee_strategy_state
            (channel_id, last_revenue_rate, last_fee_ppm, trend_direction,
             step_ppm, consecutive_same_direction, last_update,
             last_broadcast_fee_ppm, last_state, is_sleeping, sleep_until, stable_cycles,
             forward_count_since_update, last_volume_sats, v2_state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (channel_id, last_revenue_rate, last_fee_ppm, trend_direction,
              step_ppm, consecutive_same_direction, now,
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
                          new_fee_ppm: int, reason: str, manual: bool = False):
        """Record a fee change for audit purposes."""
        conn = self._get_connection()
        now = int(time.time())
        
        conn.execute("""
            INSERT INTO fee_changes 
            (channel_id, peer_id, old_fee_ppm, new_fee_ppm, reason, manual, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (channel_id, peer_id, old_fee_ppm, new_fee_ppm, reason, 1 if manual else 0, now))
    
    def get_recent_fee_changes(self, limit: int = 10, channel_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent fee changes, optionally filtered by channel."""
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
        """Record a rebalance attempt and return its ID."""
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
            'timestamp': now
        }

        cursor = conn.execute("""
            INSERT INTO rebalance_history 
            (from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats,
             status, rebalance_type, timestamp)
            VALUES (:from_channel, :to_channel, :amount_sats, :max_fee_sats, :expected_profit_sats, 
                    :status, :rebalance_type, :timestamp)
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
        Sums actual_fee_sats from successful rebalances since the timestamp.
        
        Args:
            since_timestamp: Unix timestamp to start summing from
            
        Returns:
            Total fees spent in sats (0 if none)
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(actual_fee_sats), 0) as total_fees
            FROM rebalance_history 
            WHERE timestamp >= ? AND status = 'success' AND actual_fee_sats IS NOT NULL
        """, (since_timestamp,)).fetchone()
        
        return row['total_fees'] if row else 0
    
    def get_total_routing_revenue(self, since_timestamp: int) -> int:
        """
        Get total routing revenue (fees earned) since a given timestamp.
        
        Used for Revenue-Proportional Budget calculation.
        Sums fee_msat from the forwards table since the timestamp.
        
        Args:
            since_timestamp: Unix timestamp to start summing from
            
        Returns:
            Total routing fees earned in sats (0 if none)
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(fee_msat), 0) as total_fees_msat
            FROM forwards
            WHERE timestamp >= ?
        """, (since_timestamp,)).fetchone()
        
        # Convert msat to sats
        return (row['total_fees_msat'] // 1000) if row else 0

    # =========================================================================
    # Phase 8: Financial Snapshots for P&L Dashboard
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
                level='info'
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

        # Revenue from this channel (as outbound)
        rev_row = conn.execute("""
            SELECT COALESCE(SUM(fee_msat), 0) as revenue_msat,
                   COUNT(*) as forward_count
            FROM forwards
            WHERE out_channel = ? AND timestamp >= ?
        """, (channel_id, since)).fetchone()

        # Rebalance costs for this channel
        cost_row = conn.execute("""
            SELECT COALESCE(SUM(actual_fee_sats), 0) as cost_sats
            FROM rebalance_history
            WHERE to_channel = ? AND timestamp >= ? AND status = 'success'
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

        # Get inbound contribution (where this channel was the entry point)
        inbound_row = conn.execute("""
            SELECT COALESCE(SUM(in_msat), 0) as sourced_volume_msat,
                   COALESCE(SUM(fee_msat), 0) as sourced_fee_msat,
                   COUNT(*) as sourced_forward_count
            FROM forwards
            WHERE in_channel = ? AND timestamp >= ?
        """, (channel_id, since)).fetchone()

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

        # Calculate total contribution value
        # Direct fees + sourced fee contribution (what we helped earn elsewhere)
        total_contribution = direct_pnl['revenue_sats'] + inbound['sourced_fee_contribution_sats']

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

    # =========================================================================
    # Atomic Budget Reservation System (CRITICAL-01 fix)
    # =========================================================================

    def reserve_budget(self, reservation_id: str, amount_sats: int,
                      channel_id: str, budget_limit: int,
                      since_timestamp: int) -> Tuple[bool, int]:
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

        Returns:
            Tuple of (success: bool, remaining_budget: int)
        """
        conn = self._get_connection()
        now = int(time.time())

        try:
            # Use a transaction to ensure atomicity
            conn.execute("BEGIN IMMEDIATE")

            # Get actual spent (from completed successful rebalances)
            spent_row = conn.execute("""
                SELECT COALESCE(SUM(actual_fee_sats), 0) as spent
                FROM rebalance_history
                WHERE timestamp >= ? AND status = 'success' AND actual_fee_sats IS NOT NULL
            """, (since_timestamp,)).fetchone()
            actual_spent = spent_row['spent'] if spent_row else 0

            # Get active reservations (not yet spent or released)
            reserved_row = conn.execute("""
                SELECT COALESCE(SUM(reserved_sats), 0) as reserved
                FROM budget_reservations
                WHERE status = 'active' AND reserved_at >= ?
            """, (since_timestamp,)).fetchone()
            active_reserved = reserved_row['reserved'] if reserved_row else 0

            # Calculate total committed budget
            total_committed = actual_spent + active_reserved
            remaining = budget_limit - total_committed

            # Check if we have room for this reservation
            if amount_sats > remaining:
                conn.execute("ROLLBACK")
                return (False, remaining)

            # Create the reservation
            conn.execute("""
                INSERT INTO budget_reservations
                (reservation_id, reserved_sats, reserved_at, job_channel_id, status)
                VALUES (?, ?, ?, ?, 'active')
            """, (reservation_id, amount_sats, now, channel_id))

            conn.execute("COMMIT")
            return (True, remaining - amount_sats)

        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except:
                pass
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

        # First get stats on what we're clearing
        stats = conn.execute("""
            SELECT COUNT(*) as count, COALESCE(SUM(amount_sats), 0) as total_sats
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

        if count > 0:
            self.plugin.log(
                f"Cleared {count} active budget reservations ({total_sats} sats)",
                level='info'
            )

        return {
            "cleared_count": count,
            "released_sats": total_sats
        }

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

        # Get spending and job counts from rebalance history
        stats = conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN status = 'success' THEN actual_fee_sats ELSE 0 END), 0) as total_spent,
                COUNT(*) as job_count,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count
            FROM rebalance_history
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

        total_spent = stats['total_spent'] if stats else 0
        job_count = stats['job_count'] if stats else 0
        success_count = stats['success_count'] if stats else 0
        failed_count = stats['failed_count'] if stats else 0
        total_reserved = reserved['total_reserved'] if reserved else 0

        # Calculate success rate
        success_rate = (success_count / job_count * 100) if job_count > 0 else 0.0

        return {
            'total_spent_sats': total_spent,
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

        spent_row = conn.execute("""
            SELECT COALESCE(SUM(actual_fee_sats), 0) as spent
            FROM rebalance_history
            WHERE timestamp >= ? AND status = 'success' AND actual_fee_sats IS NOT NULL
        """, (since_timestamp,)).fetchone()

        reserved_row = conn.execute("""
            SELECT COALESCE(SUM(reserved_sats), 0) as reserved
            FROM budget_reservations
            WHERE status = 'active' AND reserved_at >= ?
        """, (since_timestamp,)).fetchone()

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
            limit: Maximum records to return
            
        Returns:
            List of rebalance records with fee and amount info.
            Note: fee_paid_msat is in millisatoshis (actual_fee_sats * 1000)
        """
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

            Phase 2: Idempotent insert using INSERT OR IGNORE under a UNIQUE index.

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

            for fwd in forwards:
                try:
                    in_chan = (fwd.get('in_channel', '') or '').replace(':', 'x')
                    out_chan = (fwd.get('out_channel', '') or '').replace(':', 'x')
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
                except Exception:
                    # Skip invalid records
                    pass

            return inserted

    def get_daily_flow_buckets(self, window_days: int = 7, channel_id: Optional[str] = None) -> Dict[str, list]:
        """
        Get daily flow buckets from the local forwards table.

        This replaces the listforwards RPC call for flow analysis, providing
        the same data structure but from local SQLite aggregation.

        TODO #19: This eliminates the heaviest RPC call in the plugin,
        reducing CPU usage by ~90% during flow analysis cycles.

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

        # Build query based on whether we're filtering by channel
        if channel_id:
            query = """
                SELECT in_channel, out_channel, in_msat, out_msat, timestamp
                FROM forwards
                WHERE timestamp >= ? AND (in_channel = ? OR out_channel = ?)
            """
            params = (start_time, channel_id, channel_id)
        else:
            query = """
                SELECT in_channel, out_channel, in_msat, out_msat, timestamp
                FROM forwards
                WHERE timestamp >= ?
            """
            params = (start_time,)

        rows = conn.execute(query, params).fetchall()

        def init_bucket():
            """Initialize a single day bucket with v2.0 fields."""
            return {'in': 0, 'out': 0, 'count': 0, 'last_ts': 0}

        for row in rows:
            in_channel = row['in_channel']
            out_channel = row['out_channel']
            in_msat = row['in_msat'] or 0
            out_msat = row['out_msat'] or 0
            timestamp = row['timestamp']

            # Calculate age in days (0 = today/last 24h)
            age_days = int((now - timestamp) // 86400)
            if age_days >= window_days:
                continue

            # Initialize bucket lists if needed (v2.0: with count and last_ts)
            if in_channel and in_channel not in flow_data:
                flow_data[in_channel] = [init_bucket() for _ in range(window_days)]
            if out_channel and out_channel not in flow_data:
                flow_data[out_channel] = [init_bucket() for _ in range(window_days)]

            # Add to appropriate day bucket (convert msat to sats)
            if in_channel:
                bucket = flow_data[in_channel][age_days]
                bucket['in'] += in_msat // 1000
                bucket['count'] += 1
                if timestamp > bucket['last_ts']:
                    bucket['last_ts'] = timestamp

            if out_channel:
                bucket = flow_data[out_channel][age_days]
                bucket['out'] += out_msat // 1000
                bucket['count'] += 1
                if timestamp > bucket['last_ts']:
                    bucket['last_ts'] = timestamp

        return flow_data
    
    
    def record_forward(self, in_channel: str, out_channel: str,
                       in_msat: int, out_msat: int, fee_msat: int, *args) -> None:
        """
        Record a completed forward for real-time tracking.

        Phase 2: Use canonical forward times (received_time/resolved_time) and
        INSERT OR IGNORE under a UNIQUE index to prevent double-dips on restart.

        Backward compatible:
          - Legacy call: record_forward(in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time)
          - Phase 2 call: record_forward(in_channel, out_channel, in_msat, out_msat, fee_msat,
                                         received_time, resolved_time[, resolution_time])
        """
        # Parse legacy vs Phase 2 call patterns
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
            
        times = [row['resolution_time'] for row in rows]
        n = len(times)
        avg = sum(times) / n
        
        if n < 2:
            return {'avg': avg, 'std': 0.0}
            
        variance = sum((x - avg) ** 2 for x in times) / (n - 1)
        std = math.sqrt(variance)
        
        return {'avg': avg, 'std': std}
    
    # =========================================================================
    # Clboss Unmanage Tracking
    # =========================================================================
    
    def record_unmanage(self, peer_id: str, tag):
        """Record that we unmanaged a peer/tag from clboss."""
        conn = self._get_connection()
        now = int(time.time())

        # Normalize tag: convert list to comma-separated string
        tag_str = ",".join(tag) if isinstance(tag, list) else tag

        conn.execute("""
            INSERT OR REPLACE INTO clboss_unmanaged
            (peer_id, tag, unmanaged_at)
            VALUES (?, ?, ?)
        """, (peer_id, tag_str, now))

    def remove_unmanage(self, peer_id: str, tag=None):
        """Remove unmanage record (when remanaging)."""
        conn = self._get_connection()

        if tag:
            # Normalize tag: convert list to comma-separated string
            tag_str = ",".join(tag) if isinstance(tag, list) else tag
            conn.execute(
                "DELETE FROM clboss_unmanaged WHERE peer_id = ? AND tag = ?",
                (peer_id, tag_str)
            )
        else:
            conn.execute(
                "DELETE FROM clboss_unmanaged WHERE peer_id = ?",
                (peer_id,)
            )

    def is_unmanaged(self, peer_id: str, tag) -> bool:
        """Check if a peer/tag is currently unmanaged."""
        conn = self._get_connection()

        # Normalize tag: convert list to comma-separated string
        tag_str = ",".join(tag) if isinstance(tag, list) else tag

        row = conn.execute(
            "SELECT 1 FROM clboss_unmanaged WHERE peer_id = ? AND tag = ?",
            (peer_id, tag_str)
        ).fetchone()
        return row is not None
    
    def get_all_unmanaged(self) -> List[Dict[str, Any]]:
        """Get all unmanaged peer/tag combinations."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM clboss_unmanaged").fetchall()
        return [dict(row) for row in rows]
    
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
        conn.commit()
    
    def get_channel_open_cost(self, channel_id: str) -> Optional[int]:
        """Get the recorded open cost for a channel."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT open_cost_sats FROM channel_costs WHERE channel_id = ?",
            (channel_id,)
        ).fetchone()
        return row["open_cost_sats"] if row else None
    
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
        conn.commit()
    
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
        
        # Rolled-up revenue from daily_forwarding_stats
        rollup_revenue_row = conn.execute(
            "SELECT COALESCE(SUM(total_fee_msat), 0) as total FROM daily_forwarding_stats"
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

        # Total splice costs from splice_costs table (Accounting v2.0)
        splice_row = conn.execute(
            "SELECT COALESCE(SUM(fee_sats), 0) as total FROM splice_costs"
        ).fetchone()
        total_splice_cost_sats = splice_row["total"] if splice_row else 0

        # Current forward count from forwards table
        count_row = conn.execute(
            "SELECT COUNT(*) as total FROM forwards"
        ).fetchone()
        current_forwards = count_row["total"] if count_row else 0
        
        # Rolled-up forward count
        rollup_count_row = conn.execute(
            "SELECT COALESCE(SUM(forward_count), 0) as total FROM daily_forwarding_stats"
        ).fetchone()
        rollup_forwards = rollup_count_row["total"] if rollup_count_row else 0
        
        # Combine pruned (legacy) + rolled-up + current
        total_forwards = pruned_forward_count + rollup_forwards + current_forwards
        
        return {
            "total_revenue_msat": total_revenue_msat,
            "total_rebalance_cost_sats": total_rebalance_cost_sats,
            "total_opening_cost_sats": total_opening_cost_sats,
            "total_closure_cost_sats": total_closure_cost_sats,  # Accounting v2.0
            "total_splice_cost_sats": total_splice_cost_sats,  # Accounting v2.0
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
                    f"Security: Invalid peer_id format for {channel_id}, using sanitized value",
                    level='warn'
                )
                peer_id = "invalid_peer_id"

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
            conn.commit()

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

            if additional_fees > 0:
                conn.execute("""
                    UPDATE channel_closure_costs
                    SET htlc_sweep_fee_sats = htlc_sweep_fee_sats + ?,
                        total_closure_cost_sats = total_closure_cost_sats + ?
                    WHERE channel_id = ?
                """, (additional_fees, additional_fees, channel_id))

            conn.commit()
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
            conn.commit()

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

            # Calculate derived values
            days_open = ((closed_at - opened_at) // 86400) if opened_at else 0
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
            conn.commit()

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

    def remove_closed_channel_data(self, channel_id: str, peer_id: Optional[str] = None) -> Dict[str, int]:
        """
        Remove a closed channel from active tracking tables.

        After a channel is archived to closed_channels, we should clean up
        the active tracking data to prevent stale state and reduce db bloat.

        Args:
            channel_id: The channel short ID
            peer_id: Optional peer ID (used for clboss_unmanaged cleanup)

        Returns:
            Dict with counts of deleted rows per table
        """
        deleted = {
            "channel_states": 0,
            "channel_failures": 0,
            "channel_probes": 0,
            "clboss_unmanaged": 0
        }

        try:
            conn = self._get_connection()

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

            # Remove from clboss_unmanaged if peer_id provided
            if peer_id:
                cursor = conn.execute(
                    "DELETE FROM clboss_unmanaged WHERE peer_id = ?",
                    (peer_id,)
                )
                deleted["clboss_unmanaged"] = cursor.rowcount

            conn.commit()

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
    # Splice Cost Tracking (Accounting v2.0)
    # =========================================================================

    def record_splice(
        self,
        channel_id: str,
        peer_id: str,
        splice_type: str,
        amount_sats: int,
        fee_sats: int,
        old_capacity_sats: Optional[int] = None,
        new_capacity_sats: Optional[int] = None,
        txid: Optional[str] = None
    ) -> bool:
        """
        Record a splice operation and its on-chain cost.

        Splices modify channel capacity without closing/reopening the channel.
        This tracks the on-chain fees for accurate P&L accounting.

        Args:
            channel_id: The channel short ID (SCID)
            peer_id: The peer node ID
            splice_type: Type of splice ('splice_in' or 'splice_out')
            amount_sats: Amount added (positive for splice_in) or removed (negative for splice_out)
            fee_sats: On-chain fee paid for the splice transaction
            old_capacity_sats: Channel capacity before splice
            new_capacity_sats: Channel capacity after splice
            txid: The splice transaction ID

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            # Security: Input validation
            if not self._validate_channel_id(channel_id):
                self.plugin.log(
                    f"Security: Invalid channel_id format '{channel_id}', skipping splice record",
                    level='warn'
                )
                return False

            if not self._validate_peer_id(peer_id):
                self.plugin.log(
                    f"Security: Invalid peer_id format for {channel_id}, using sanitized value",
                    level='warn'
                )
                peer_id = "invalid_peer_id"

            # Security: Sanitize fee and amount values
            fee_sats = self._sanitize_fee(fee_sats, "splice_fee")
            amount_sats = self._sanitize_amount(amount_sats, "splice_amount")
            if old_capacity_sats is not None:
                old_capacity_sats = self._sanitize_amount(old_capacity_sats, "old_capacity")
            if new_capacity_sats is not None:
                new_capacity_sats = self._sanitize_amount(new_capacity_sats, "new_capacity")

            # Validate splice_type
            valid_splice_types = {'splice_in', 'splice_out'}
            if splice_type not in valid_splice_types:
                self.plugin.log(
                    f"Security: Invalid splice_type '{splice_type}', defaulting to 'splice_in'",
                    level='warn'
                )
                splice_type = 'splice_in'

            conn = self._get_connection()
            now = int(time.time())

            # Use INSERT OR IGNORE to prevent duplicate records when txid is known
            # The UNIQUE index on (channel_id, txid) WHERE txid IS NOT NULL handles idempotency
            conn.execute("""
                INSERT OR IGNORE INTO splice_costs
                (channel_id, peer_id, splice_type, amount_sats, fee_sats,
                 old_capacity_sats, new_capacity_sats, txid, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (channel_id, peer_id, splice_type, amount_sats, fee_sats,
                  old_capacity_sats, new_capacity_sats, txid, now))

            # Check if insert was actually performed (rowcount > 0) or ignored (duplicate)
            if conn.total_changes > 0:
                conn.commit()
                self.plugin.log(
                    f"Recorded splice for {channel_id}: type={splice_type}, "
                    f"amount={amount_sats} sats, fee={fee_sats} sats",
                    level='info'
                )
            else:
                self.plugin.log(
                    f"Duplicate splice ignored for {channel_id} txid={txid}",
                    level='debug'
                )
            return True

        except Exception as e:
            self.plugin.log(f"Error recording splice for {channel_id}: {e}", level='error')
            return False

    def get_channel_splice_history(self, channel_id: str) -> List[Dict[str, Any]]:
        """
        Get splice history for a specific channel.

        Args:
            channel_id: The channel short ID

        Returns:
            List of splice records, ordered by timestamp (most recent first)
        """
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT * FROM splice_costs
            WHERE channel_id = ?
            ORDER BY timestamp DESC
        """, (channel_id,)).fetchall()

        return [dict(row) for row in rows]

    def get_total_splice_costs(self) -> int:
        """
        Get the total splice costs across all channels.

        Returns:
            Total splice costs in sats
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(fee_sats), 0) as total
            FROM splice_costs
        """).fetchone()

        return row["total"] if row else 0

    def get_splice_costs_since(self, since_timestamp: int) -> int:
        """
        Get splice costs since a specific timestamp (for windowed P&L).

        Args:
            since_timestamp: Unix timestamp to query from

        Returns:
            Total splice costs in sats since the timestamp
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(fee_sats), 0) as total
            FROM splice_costs
            WHERE timestamp >= ?
        """, (since_timestamp,)).fetchone()

        return row["total"] if row else 0

    def get_splice_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all splice operations.

        Returns:
            Dict with aggregate statistics for splices
        """
        conn = self._get_connection()

        row = conn.execute("""
            SELECT
                COUNT(*) as splice_count,
                COALESCE(SUM(CASE WHEN splice_type = 'splice_in' THEN 1 ELSE 0 END), 0) as splice_in_count,
                COALESCE(SUM(CASE WHEN splice_type = 'splice_out' THEN 1 ELSE 0 END), 0) as splice_out_count,
                COALESCE(SUM(CASE WHEN splice_type = 'splice_in' THEN amount_sats ELSE 0 END), 0) as total_splice_in_sats,
                COALESCE(SUM(CASE WHEN splice_type = 'splice_out' THEN ABS(amount_sats) ELSE 0 END), 0) as total_splice_out_sats,
                COALESCE(SUM(fee_sats), 0) as total_fees_sats
            FROM splice_costs
        """).fetchone()

        return dict(row) if row else {
            "splice_count": 0,
            "splice_in_count": 0,
            "splice_out_count": 0,
            "total_splice_in_sats": 0,
            "total_splice_out_sats": 0,
            "total_fees_sats": 0
        }

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
    
    def increment_failure_count(self, channel_id: str) -> int:
        """
        Increment the failure count for a channel and update last failure time.
        
        Called when a rebalance attempt fails.
        
        Args:
            channel_id: Channel that failed
            
        Returns:
            New failure count
        """
        conn = self._get_connection()
        now = int(time.time())
        
        # Get current count
        current_count, _ = self.get_failure_count(channel_id)
        new_count = current_count + 1
        
        conn.execute("""
            INSERT OR REPLACE INTO channel_failures 
            (channel_id, failure_count, last_failure_time)
            VALUES (?, ?, ?)
        """, (channel_id, new_count, now))
        
        return new_count
    
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
        
        # Apply decay to both success and failure counts
        # Using CAST to INTEGER naturally floors the result
        conn.execute("""
            UPDATE peer_reputation 
            SET success_count = CAST(success_count * ? AS INTEGER),
                failure_count = CAST(failure_count * ? AS INTEGER)
        """, (decay_factor, decay_factor))
        
        # Optionally clean up peers that have decayed to (0, 0)
        # This prevents the table from growing indefinitely with stale entries
        conn.execute("""
            DELETE FROM peer_reputation 
            WHERE success_count = 0 AND failure_count = 0
        """)
    
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

        flow_count = 0
        forwards_count = 0
        pruned_revenue = 0
        pruned_count = 0

        # Atomic aggregation + deletion to avoid double-counting if interrupted.
        # If we update lifetime_aggregates but fail before deleting forwards, we'd
        # count the same forwards again later. Using a transaction prevents this.
        with conn:
            # Count rows before deletion for logging
            flow_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM flow_history WHERE timestamp < ?", (cutoff,)
            ).fetchone()["cnt"]
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
                
                # We NO LONGER update lifetime_aggregates for new data, 
                # but we leave the table alone as it contains legacy history.

            conn.execute("DELETE FROM flow_history WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM forwards WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM peer_connection_history WHERE timestamp < ?", (cutoff,))

        # VACUUM to reclaim disk space after pruning
        # SQLite DELETE only marks pages as free; VACUUM actually shrinks the file.
        # This is safe to run from a background thread (blocking is acceptable).
        if flow_count > 0 or forwards_count > 0:
            try:
                conn.execute("VACUUM")
                self.plugin.log("Database VACUUM completed to reclaim disk space")
            except Exception as e:
                self.plugin.log(f"VACUUM failed (non-fatal): {e}", level='warn')

        if forwards_count > 0:
            self.plugin.log(
                f"Preserved {pruned_revenue // 1000} sats revenue from {pruned_count} forwards before pruning"
            )

        if flow_count > 0 or forwards_count > 0:
            self.plugin.log(
                f"Cleaned up data older than {days_to_keep} days: "
                f"{flow_count} flow_history rows, {forwards_count} forwards rows"
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
    # Ignored Peers (Blacklist) Methods
    # =========================================================================
    
    def add_ignored_peer(self, peer_id: str, reason: str = "manual"):
        """Add a peer to the ignore list."""
        conn = self._get_connection()
        now = int(time.time())
        conn.execute("""
            INSERT OR REPLACE INTO ignored_peers (peer_id, reason, ignored_at)
            VALUES (?, ?, ?)
        """, (peer_id, reason, now))

    def remove_ignored_peer(self, peer_id: str):
        """Remove a peer from the ignore list."""
        conn = self._get_connection()
        conn.execute("DELETE FROM ignored_peers WHERE peer_id = ?", (peer_id,))

    def is_peer_ignored(self, peer_id: str) -> bool:
        """Check if a peer is ignored."""
        conn = self._get_connection()
        row = conn.execute("SELECT 1 FROM ignored_peers WHERE peer_id = ?", (peer_id,)).fetchone()
        return row is not None

    def get_ignored_peers(self) -> List[Dict[str, Any]]:
        """Get list of all ignored peers."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM ignored_peers ORDER BY ignored_at DESC").fetchall()
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Config Overrides Methods (Phase 7: Dynamic Runtime Configuration)
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
        
        Returns:
            New version number after update
        """
        conn = self._get_connection()
        now = int(time.time())
        
        # Get current max version
        row = conn.execute("SELECT MAX(version) as max_v FROM config_overrides").fetchone()
        new_version = (row['max_v'] or 0) + 1
        
        conn.execute("""
            INSERT OR REPLACE INTO config_overrides (key, value, version, updated_at)
            VALUES (?, ?, ?, ?)
        """, (key, value, new_version, now))
        
        return new_version

    def get_all_config_overrides(self) -> Dict[str, str]:
        """Get all config overrides as a dictionary."""
        conn = self._get_connection()
        rows = conn.execute("SELECT key, value FROM config_overrides").fetchall()
        return {row['key']: row['value'] for row in rows}

    def get_config_version(self) -> int:
        """Get current config version (max version in table)."""
        conn = self._get_connection()
        row = conn.execute("SELECT MAX(version) as max_v FROM config_overrides").fetchone()
        return row['max_v'] or 0

    def delete_config_override(self, key: str) -> bool:
        """Delete a config override, returning to default."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM config_overrides WHERE key = ?", (key,))
        return cursor.rowcount > 0
    
    # =========================================================================
    # Mempool Fee History Methods (Phase 7: Vegas Reflex)
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
        """Close the thread-local database connection (if any)."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None