#!/usr/bin/env python3
"""
cl-revenue-ops: A Revenue Operations Plugin for Core Lightning

This plugin acts as a "Revenue Operations" layer that sits on top of the clboss 
automated manager. While clboss handles channel creation and node reliability,
this plugin overrides clboss for fee setting and rebalancing decisions to 
maximize profitability based on economic principles rather than heuristics.

MANAGER-OVERRIDE PATTERN:
-------------------------
Before changing any channel state, this plugin checks if the peer is managed 
by clboss. If it is, we issue the `clboss-unmanage` command for that specific 
peer and tag (e.g., lnfee) to prevent clboss from reverting our changes.

This allows us to:
1. Let clboss handle what it's good at (channel creation, peer selection)
2. Take over the economic decisions (fee setting, rebalancing) where we can
   apply more sophisticated algorithms

Dependencies:
- pyln-client: Core Lightning plugin framework
- bookkeeper plugin (built-in): On-chain cost attribution (opens/closes/splices) and accounting-grade events
- Local forwards table (SQLite): Routing history for flow analysis (hydrated once on startup)
- External rebalancer (sling): Executes rebalance payments

Author: Lightning Goats Team
License: MIT
"""

import os
import time
import json
import random
import threading
import signal
import atexit
import re
import dataclasses
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import traceback
from pyln.client import Plugin, RpcError

# Import our modules
from modules.flow_analysis import FlowAnalyzer, ChannelState
from modules.fee_controller import PIDFeeController
from modules.rebalancer import EVRebalancer
from modules.clboss_manager import ClbossManager
from modules.config import Config
from modules.database import Database
from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
from modules.capacity_planner import CapacityPlanner
from modules.policy_manager import PolicyManager, FeeStrategy, RebalanceMode, PeerPolicy
from modules.hive_bridge import HiveFeeIntelligenceBridge
from modules.boltz_manager import BoltzCliManager, BoltzCliConfig, BoltzCliError
from modules.utils import normalize_scid, parse_msat


# =============================================================================
# PLUGIN VERSION
# =============================================================================
# v2.2.4: Stability + correctness fixes (DB rollups, policy precedence, rebalancer reliability)
# v2.1.0: Kalman Filter for Flow State Estimation
#   - Replaces EMA with Kalman filter for optimal state estimation
#   - Faster regime change detection via innovation monitoring
#   - Adaptive process noise based on flow volatility
#   - Confidence-weighted measurement noise
#   - Velocity tracking built into state vector
#   - Persistent filter state across restarts
# v2.0.0: Thompson Sampling + AIMD Fee Controller
#   - Replaces Hill Climbing with Gaussian Thompson Sampling
#   - AIMD defense layer for rapid failure response
#   - Fleet-informed priors from hive intelligence
#   - Contextual posteriors (balance, pheromone, time, corridor role)
#   - Stigmergic modulation for exploration/exploitation
#   - P2 fleet integration: elasticity sharing, curve aggregation,
#     regime coordination, competition avoidance, profitability weighting
PLUGIN_VERSION = "2.2.4"


# =============================================================================
# RATE LIMITER FOR FORCE OPERATIONS (MAJOR-09 FIX)
# =============================================================================
# Prevents abuse of force=true parameters which bypass safety checks.
# Implements a simple sliding window rate limiter per command.

class ForceRateLimiter:
    """
    Rate limiter for force=true RPC operations.

    Prevents abuse by limiting how often force operations can be called.
    Uses a sliding window algorithm with configurable limits.
    """

    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum force calls allowed per window
            window_seconds: Window duration in seconds
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: Dict[str, list] = {}  # command -> list of timestamps
        self._lock = threading.Lock()

    def check_rate_limit(self, command: str) -> Tuple[bool, str]:
        """
        Check if a force operation is allowed.

        Args:
            command: The RPC command name

        Returns:
            Tuple of (allowed: bool, message: str)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # Get or create timestamp list for this command
            if command not in self._timestamps:
                self._timestamps[command] = []

            # Clean old timestamps
            self._timestamps[command] = [
                ts for ts in self._timestamps[command] if ts > cutoff
            ]

            # Check limit
            if len(self._timestamps[command]) >= self.max_calls:
                remaining = self._timestamps[command][0] + self.window_seconds - now
                return (False, f"Rate limit exceeded for force={command}. "
                              f"Try again in {int(remaining)}s. "
                              f"({self.max_calls} calls per {self.window_seconds}s)")

            # Record this call
            self._timestamps[command].append(now)
            return (True, "")

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            status = {}
            for cmd, timestamps in self._timestamps.items():
                recent = [ts for ts in timestamps if ts > cutoff]
                status[cmd] = {
                    "calls_in_window": len(recent),
                    "max_calls": self.max_calls,
                    "window_seconds": self.window_seconds
                }
            return status


# Global rate limiter for force operations (10 calls per 60 seconds)
force_rate_limiter = ForceRateLimiter(max_calls=10, window_seconds=60)


# Initialize the plugin
plugin = Plugin()

# =============================================================================
# GRACEFUL SHUTDOWN SUPPORT (Plugin Lifecycle Management)
# =============================================================================
# This event is used to signal all background threads to exit cleanly.
# When `lightning-cli plugin stop cl-revenue-ops` is called, CLN sends SIGTERM.
# We catch this signal and set the event, causing all loops to exit immediately
# instead of waiting for their sleep timers (which could be 30+ minutes).

shutdown_event = threading.Event()

# =============================================================================
# THREAD-SAFE RPC WRAPPER (Phase 5.5: High-Uptime Stability)
# =============================================================================

# =============================================================================
# CL-HIVE AVAILABILITY CACHE (Performance Optimization)
# =============================================================================
# Caches the cl-hive plugin availability check to avoid expensive
# plugin("list") RPC calls on every channel event. TTL: 60 seconds.

class HiveAvailabilityCache:
    """Thread-safe cache for cl-hive plugin availability."""

    def __init__(self, ttl_seconds: int = 60):
        self._available: Optional[bool] = None
        self._last_check: float = 0
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def is_available(self, rpc) -> bool:
        """
        Check if cl-hive plugin is available (cached).

        Args:
            rpc: RPC interface for plugin list call

        Returns:
            True if cl-hive is active, False otherwise
        """
        now = time.time()

        with self._lock:
            # Return cached value if still valid
            if self._available is not None and (now - self._last_check) < self._ttl:
                return self._available

        # Cache miss or expired - fetch fresh
        try:
            plugins = rpc.plugin("list")
            available = False
            for p in plugins.get('plugins', []):
                if 'cl-hive' in p.get('name', '') and p.get('active', False):
                    available = True
                    break

            with self._lock:
                self._available = available
                self._last_check = now

            return available

        except Exception:
            # On error, assume unavailable but don't cache failure long
            with self._lock:
                self._available = False
                self._last_check = now - (self._ttl - 5)  # Retry after 5s
            return False

    def invalidate(self):
        """Force cache refresh on next check."""
        with self._lock:
            self._available = None
            self._last_check = 0


# Global cache for cl-hive availability (60 second TTL)
hive_availability_cache = HiveAvailabilityCache(ttl_seconds=60)


class RPCTimeoutError(RpcError):
    """Exception raised when an RPC call times out."""
    def __init__(self, method):
        self.method = method
        # Initialize RpcError with compatible fields
        super().__init__(method, {}, f"RPC timeout for method: {method}")


class RPCBreakerOpen(RPCTimeoutError):
    """Kept for backward compatibility — callers catch this alongside RPCTimeoutError."""
    pass


class RPCOverloadedError(RPCTimeoutError):
    """Raised when the RPC worker pool is saturated."""
    def __init__(self, method):
        self.method = method
        RpcError.__init__(self, method, {}, f"RPC worker pool saturated for method: {method}")


class ThreadSafeRpcProxy:
    """
    Thread-safe RPC proxy using a ThreadPoolExecutor for timeout protection.

    pyln-client opens a new Unix domain socket per call — it's inherently
    thread-safe and supports unlimited concurrency. No subprocess pool needed.

    Calls run in a bounded worker pool. If a call hangs past the timeout, the
    caller gets an RPCTimeoutError immediately. The underlying worker may still
    finish later, so submissions are also backpressured with bounded queues.

    No circuit breaker: individual calls either succeed or fail on their own.
    One slow call can't poison 60+ other RPC methods for 60 seconds.

    Explicit fire-and-forget calls use a separate 4-thread pool so slow
    informational hive pushes can't starve the main pool.
    """

    def __init__(self, plugin_instance: Plugin):
        from concurrent.futures import ThreadPoolExecutor
        # Python 3.10: concurrent.futures.TimeoutError does NOT inherit from
        # builtins.TimeoutError.  Fixed in 3.11+.  Capture both for compat.
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        self._FuturesTimeoutError = FuturesTimeoutError
        self._plugin = plugin_instance
        self._rpc = plugin_instance.rpc
        self._executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="rpc")
        self._async_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rpc_async")
        # Bound queued submissions so timed-out/hung calls can't accumulate an
        # unbounded work queue and exhaust memory over time.
        self._main_submit_slots = threading.Semaphore(16 + 32)   # workers + bounded queue
        self._async_submit_slots = threading.Semaphore(4 + 64)   # explicit async pushes only
        self._async_fail_count = 0
        self._async_lock = threading.Lock()  # L-4: Protect _async_fail_count

    def __getattr__(self, name):
        if name in ("_plugin", "_rpc", "_executor", "_async_executor",
                     "_main_submit_slots", "_async_submit_slots",
                     "_async_fail_count", "_async_lock",
                     "_submit_main", "_submit_async",
                     "call", "fire_and_forget"):
            return super().__getattribute__(name)

        fn = getattr(self._rpc, name)

        def wrapper(*args, **kwargs):
            timeout = 30
            if config:
                timeout = config.rpc_timeout_seconds
            future = self._submit_main(fn, name, timeout, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except (TimeoutError, self._FuturesTimeoutError):
                self._plugin.log(f"RPC timeout after {timeout}s on {name}", level="warn")
                raise RPCTimeoutError(name)

        return wrapper

    def _submit_main(self, fn, method_name: str, timeout: int, *args, **kwargs):
        acquire_timeout = min(max(float(timeout), 0.1), 1.0)
        if not self._main_submit_slots.acquire(timeout=acquire_timeout):
            self._plugin.log(
                f"RPC pool saturated on {method_name} (submission queue full)",
                level="warn"
            )
            raise RPCOverloadedError(method_name)
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except Exception:
            self._main_submit_slots.release()
            raise
        future.add_done_callback(lambda _f: self._main_submit_slots.release())
        return future

    def _submit_async(self, fn, method_name: str):
        if not self._async_submit_slots.acquire(blocking=False):
            with self._async_lock:
                self._async_fail_count += 1
                count = self._async_fail_count
            self._plugin.log(
                f"fire_and_forget {method_name} dropped (async queue full, streak={count})",
                level="warn"
            )
            return False
        try:
            future = self._async_executor.submit(fn)
        except Exception:
            self._async_submit_slots.release()
            raise
        future.add_done_callback(lambda _f: self._async_submit_slots.release())
        return True

    def call(self, method_name: str, payload: Any = None, **kwargs):
        timeout = 30
        if config:
            timeout = config.rpc_timeout_seconds
        # Cross-plugin RPCs (hive-*) relay through CLN and are inherently
        # slower — give them 2x the normal timeout to avoid spurious failures.
        if method_name.startswith("hive-"):
            timeout *= 2
        future = self._submit_main(
            self._rpc.call,
            method_name,
            timeout,
            method_name,
            payload if payload is not None else {},
        )
        try:
            return future.result(timeout=timeout)
        except (TimeoutError, self._FuturesTimeoutError):
            self._plugin.log(f"RPC timeout after {timeout}s on {method_name}", level="warn")
            raise RPCTimeoutError(method_name)

    def fire_and_forget(self, method_name: str, payload: Any = None):
        """Submit an RPC call to the async pool without waiting.

        Uses a separate 4-thread pool so hung hive calls can't starve
        synchronous RPCs.  Tracks failures so the hive bridge circuit
        breaker still has signal.
        """
        def _run():
            try:
                self._rpc.call(method_name, payload if payload is not None else {})
                # L-4: Protect _async_fail_count with lock
                with self._async_lock:
                    self._async_fail_count = 0
            except Exception as e:
                with self._async_lock:
                    self._async_fail_count += 1
                    count = self._async_fail_count
                self._plugin.log(
                    f"fire_and_forget {method_name} failed "
                    f"(streak={count}): {e}", level="debug"
                )
        return self._submit_async(_run, method_name)


class ThreadSafePluginProxy:
    """
    A proxy for the Plugin object that provides thread-safe resilient RPC access.
    """

    def __init__(self, plugin_instance: Plugin):
        """Wrap the original plugin with a resilient RPC proxy."""
        self._plugin = plugin_instance
        self.rpc = ThreadSafeRpcProxy(plugin_instance)

    def log(self, message, level='info'):
        """Delegate logging to the original plugin."""
        self._plugin.log(message, level=level)

    def __getattr__(self, name):
        """Delegate all other attribute access to the original plugin."""
        return getattr(self._plugin, name)

# Global instances (initialized in init)
flow_analyzer: Optional[FlowAnalyzer] = None
fee_controller: Optional[PIDFeeController] = None
rebalancer: Optional[EVRebalancer] = None
clboss_manager: Optional[ClbossManager] = None
database: Optional[Database] = None
config: Optional[Config] = None
profitability_analyzer: Optional[ChannelProfitabilityAnalyzer] = None
capacity_planner: Optional[CapacityPlanner] = None
safe_plugin: Optional['ThreadSafePluginProxy'] = None  # Thread-safe plugin wrapper
policy_manager: Optional[PolicyManager] = None  # v1.4: Peer policy management
hive_bridge: Optional[HiveFeeIntelligenceBridge] = None  # v1.6: Hive intelligence
boltz_manager: Optional[BoltzCliManager] = None  # Boltz CLI integration (optional)
_boltz_balance_last_action: Dict[str, int] = {}  # channel_id -> unix ts of last Boltz balance action
_boltz_balance_lock = threading.Lock()
_boltz_auto_cycle_run_lock = threading.Lock()
_boltz_auto_cycle_state_lock = threading.Lock()
_boltz_auto_cycle_state: Dict[str, Any] = {
    'enabled': False,
    'thread_started': False,
    'running': False,
    'next_run_ts': None,
    'last_trigger': None,
    'last_started_ts': None,
    'last_finished_ts': None,
    'last_duration_ms': None,
    'last_result': None,
    'last_error': None,
    'consecutive_errors': 0,
}

# SCID to Peer ID cache for reputation tracking
# Maps short_channel_id -> peer_id for quick lookups
# Cache is cleared periodically to prevent stale mappings from corrupting reputation
_scid_to_peer_cache: Dict[str, str] = {}
_scid_cache_last_cleared: float = 0.0
_SCID_CACHE_TTL_SECONDS: int = 3600  # Clear cache every hour
_scid_cache_lock = threading.Lock()


# =============================================================================
# PLUGIN OPTIONS
# =============================================================================

plugin.add_option(
    name='revenue-ops-db-path',
    default='~/.lightning/revenue_ops.db',
    description='Path to the SQLite database for storing state'
)

plugin.add_option(
    name='revenue-ops-flow-interval',
    default='3600',
    description='Interval in seconds for flow analysis (default: 1 hour)'
)

plugin.add_option(
    name='revenue-ops-fee-interval',
    default='1800',
    description='Interval in seconds for fee adjustments (default: 30 min)'
)

plugin.add_option(
    name='revenue-ops-rebalance-interval',
    default='900',
    description='Interval in seconds for rebalance checks (default: 15 min)'
)

plugin.add_option(
    name='revenue-ops-target-flow',
    default='100000',
    description='Target daily flow in sats per channel (default: 100,000)'
)

plugin.add_option(
    name='revenue-ops-min-fee-ppm',
    default='10',
    description='Minimum fee floor in PPM (default: 10)'
)

plugin.add_option(
    name='revenue-ops-max-fee-ppm',
    default='5000',
    description='Maximum fee ceiling in PPM (default: 5000)'
)

plugin.add_option(
    name='revenue-ops-rebalance-min-profit',
    default='10',
    description='Minimum profit in sats to trigger rebalance (default: 10)'
)

plugin.add_option(
    name='revenue-ops-flow-window-days',
    default='7',
    description='Number of days to analyze for flow calculation (default: 7)'
)

plugin.add_option(
    name='revenue-ops-clboss-enabled',
    default='true',
    description='Whether to interact with clboss for unmanage commands (default: true)'
)

plugin.add_option(
    name='revenue-ops-rebalancer',
    default='sling',
    description='Rebalancer plugin to use (default: sling)'
)

plugin.add_option(
    name='revenue-ops-daily-budget-sats',
    default='5000',
    description='Max rebalancing fees to spend in 24 hours - acts as floor when proportional budget enabled (default: 5000)'
)

plugin.add_option(
    name='revenue-ops-total-cost-budget-mode',
    default='fixed',
    description="Unified spend gate mode for all liquidity costs: 'fixed' or 'profit_pct' (default: fixed)"
)

plugin.add_option(
    name='revenue-ops-total-cost-budget-profit-pct',
    default='0.30',
    description='When total-cost budget mode=profit_pct, use this fraction of net profit as spend budget (default: 0.30)'
)

plugin.add_option(
    name='revenue-ops-total-cost-budget-profit-pct-cap',
    default='0.75',
    description='Safety cap for total-cost profit percentage (default: 0.75 = 75%)'
)

plugin.add_option(
    name='revenue-ops-total-cost-budget-window-hours',
    default='24',
    description='Window for unified spend budget accounting and profit-based budget calculation (default: 24h)'
)

plugin.add_option(
    name='revenue-ops-min-wallet-reserve',
    default='1000000',
    description='Minimum total funds (on-chain + off-chain) to keep in reserve (default: 1,000,000)'
)

plugin.add_option(
    name='revenue-ops-proportional-budget',
    default='true',
    description='If true, scale daily budget based on 24h revenue (default: true)'
)

plugin.add_option(
    name='revenue-ops-proportional-budget-pct',
    default='0.30',
    description='Percentage of 24h revenue to use as budget when proportional budget enabled (default: 0.30 = 30%)'
)

plugin.add_option(
    name='revenue-ops-dry-run',
    default='false',
    description='If true, log actions but do not execute (default: false)'
)

plugin.add_option(
    name='revenue-ops-htlc-congestion-threshold',
    default='0.8',
    description='HTLC slot utilization threshold (0.0-1.0) above which channel is considered congested (default: 0.8)'
)

plugin.add_option(
    name='revenue-ops-enable-reputation',
    default='true',
    description='If true, weight volume by peer reputation (success rate) in fee decisions (default: true)'
)

plugin.add_option(
    name='revenue-ops-reputation-decay',
    default='0.98',
    description='Reputation decay factor applied per flow-interval (default: 0.98). 0.98^24 ≈ 0.61 daily decay.'
)

plugin.add_option(
    name='revenue-ops-enable-kelly',
    default='false',
    description='If true, scale rebalance budget using Kelly Criterion based on peer reputation (default: false)'
)

plugin.add_option(
    name='revenue-ops-kelly-bypass-fleet',
    default='true',
    description='If true, bypass Kelly Criterion for hive/fleet destinations where zero-fee internal paths exist (default: true)'
)

plugin.add_option(
    name='revenue-ops-kelly-fraction',
    default='0.5',
    description='Multiplier for Kelly fraction (default: 0.5 = Half Kelly). Full Kelly (1.0) maximizes growth but has high volatility.'
)

# Phase 7 options (v1.3.0)
plugin.add_option(
    name='revenue-ops-vegas-reflex',
    default='true',
    description='Enable Vegas Reflex mempool spike defense (default: true)'
)

plugin.add_option(
    name='revenue-ops-vegas-decay',
    default='0.85',
    description='Vegas Reflex decay rate per cycle, 0.0-1.0 (default: 0.85 = ~30min half-life)'
)

plugin.add_option(
    name='revenue-ops-scarcity-pricing',
    default='true',
    description='Enable HTLC slot scarcity pricing (default: true)'
)

plugin.add_option(
    name='revenue-ops-scarcity-threshold',
    default='0.35',
    description='Utilization threshold to start scarcity pricing, 0.0-1.0 (default: 0.35)'
)

plugin.add_option(
    name='revenue-ops-hive-enabled',
    default='auto',
    description='Hive mode: "auto" (detect cl-hive), "true" (require hive), "false" (standalone only)'
)

plugin.add_option(
    name='revenue-ops-hive-fee-ppm',
    default='0',
    description='Fee rate charged to Hive fleet members (default: 0)',
    opt_type='int'
)

plugin.add_option(
    name='revenue-ops-hive-rebalance-tolerance',
    default='50',
    description='Max sats loss tolerance per rebalance to keep channels balanced and earning (default: 50)',
    opt_type='int'
)

plugin.add_option(
    name='revenue-ops-rpc-timeout-seconds',
    default='15',
    description='Hard timeout for all RPC calls to lightningd (default: 15)'
)

plugin.add_option(
    name='revenue-ops-rpc-circuit-breaker-seconds',
    default='60',
    description='Cooldown period after an RPC timeout for that method group (default: 60)'
)

plugin.add_option(
    name='revenue-ops-reservation-timeout-hours',
    default='4',
    description='Hours before stale budget reservations are auto-released (default: 4)',
    opt_type='int'
)


plugin.add_option(
    name='revenue-ops-hot-channel-protection-enabled',
    default='true',
    description='Enable aggressive Sling protection for fast-draining high-profit channels (default: true)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-min-velocity',
    default='0.20',
    description='Minimum daily turnover ratio to qualify for hot-channel protection (default: 0.20)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-min-marginal-roi',
    default='0.20',
    description='Minimum marginal ROI (decimal) for hot-channel protection (default: 0.20)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-profit-budget-pct',
    default='0.75',
    description='Max fraction of daily channel contribution spendable on protective rebalancing (default: 0.75)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-max-chunk-multiplier',
    default='4.0',
    description='Max multiplier on sling chunk size for hot-channel protection (default: 4.0)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-min-cooldown-hours',
    default='1.0',
    description='Minimum cooldown hours for protected hot channels (default: 1.0)'
)

plugin.add_option(
    name='revenue-ops-boltz-enabled',
    default='false',
    description='Enable Boltz CLI integration for revenue-boltz-* RPCs (default: false)'
)

plugin.add_option(
    name='revenue-ops-boltz-cli-path',
    default='/usr/local/bin/boltzcli',
    description='Path to boltzcli binary (default: /usr/local/bin/boltzcli)'
)

plugin.add_option(
    name='revenue-ops-boltz-datadir',
    default='/var/lib/boltz',
    description='Boltz data dir for boltzd client auth (default: /var/lib/boltz)'
)

plugin.add_option(
    name='revenue-ops-boltz-use-sudo',
    default='false',
    description='Run boltzcli via sudo -n -u <user> (default: false)'
)

plugin.add_option(
    name='revenue-ops-boltz-sudo-user',
    default='boltz',
    description='User for sudo boltzcli execution when enabled (default: boltz)'
)

plugin.add_option(
    name='revenue-ops-boltz-timeout-seconds',
    default='60',
    description='boltzcli command timeout in seconds (default: 60)'
)

plugin.add_option(
    name='revenue-ops-boltz-daily-budget-sats',
    default='3000',
    description='Daily Boltz swap fee budget used by revenue-boltz loop methods (default: 3000)'
)

plugin.add_option(
    name='revenue-ops-boltz-enforce-budget',
    default='true',
    description='If true, reject Boltz swaps when estimated fee exceeds remaining daily budget (default: true)'
)

plugin.add_option(
    name='revenue-ops-boltz-btc-wallet',
    default='CLN',
    description='Preferred boltzd BTC wallet name (default: CLN)'
)

plugin.add_option(
    name='revenue-ops-boltz-lbtc-wallet',
    default='LOOP-LBTC',
    description='Preferred boltzd LBTC wallet name (default: LOOP-LBTC)'
)

plugin.add_option(
    name='revenue-ops-boltz-auto-cycle-enabled',
    default='true',
    description='Enable in-plugin periodic profit-gated Boltz balance cycles (default: true)'
)

plugin.add_option(
    name='revenue-ops-boltz-auto-cycle-interval-minutes',
    default='15',
    description='Minutes between scheduled Boltz auto-balance cycles (default: 15)'
)

plugin.add_option(
    name='revenue-ops-boltz-auto-cycle-max-actions',
    default='1',
    description='Maximum Boltz actions per scheduled auto-cycle (default: 1)'
)

plugin.add_option(
    name='revenue-ops-boltz-auto-cycle-startup-delay-seconds',
    default='120',
    description='Startup delay before first Boltz auto-cycle run (default: 120)'
)


# =============================================================================
# INITIALIZATION
# =============================================================================

@plugin.init()
def init(options: Dict[str, Any], configuration: Dict[str, Any], plugin: Plugin, **kwargs):
    """
    Initialize the Revenue Operations plugin.
    
    This is called once when the plugin starts. We:
    1. Parse and validate options
    2. Initialize the database
    3. Create instances of our analysis modules
    4. Set up timers for periodic execution
    """
    global flow_analyzer, fee_controller, rebalancer, clboss_manager, database, config, profitability_analyzer, capacity_planner, safe_plugin, policy_manager, hive_bridge, boltz_manager
    
    plugin.log("Initializing cl-revenue-ops plugin...")

    # M-10: Register SIGTERM handler early, before component initialization.
    # The handler checks `if rebalancer` and `if database` with None guards, so it's safe.
    def handle_shutdown_signal(signum, frame):
        """
        Handle SIGTERM for graceful shutdown.

        CLN sends SIGTERM when `lightning-cli plugin stop cl-revenue-ops` is called.
        Set the shutdown event and exit the process so Python runs atexit
        cleanup handlers. We avoid doing heavyweight cleanup in the signal
        handler itself.
        """
        shutdown_event.set()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    # M-7/C-1/L-26: Cleanup runs via atexit (safe outside signal handler context)
    def _shutdown_cleanup():
        """Perform cleanup after shutdown_event is set. Runs via atexit."""
        if safe_plugin and hasattr(safe_plugin, 'rpc'):
            try:
                safe_plugin.rpc._executor.shutdown(wait=False)
                safe_plugin.rpc._async_executor.shutdown(wait=False)
            except Exception:
                pass

        if rebalancer and rebalancer.job_manager:
            try:
                stopped = rebalancer.job_manager.stop_all_jobs(reason="plugin_shutdown")
                if stopped > 0:
                    plugin.log(f"Stopped {stopped} active rebalance jobs", level='info')
            except Exception:
                pass

        if database:
            try:
                database.close_all_connections()
                database.close_main_connection()
            except Exception:
                pass

    atexit.register(_shutdown_cleanup)

    # Build configuration from options. Filter kwargs against the imported Config
    # dataclass fields so partial deployments (new main file + older modules/config.py)
    # don't crash during init. Unknown fields are dropped with a warning.
    config_kwargs = dict(
        db_path=os.path.expanduser(options['revenue-ops-db-path']),
        flow_interval=int(options['revenue-ops-flow-interval']),
        fee_interval=int(options['revenue-ops-fee-interval']),
        rebalance_interval=int(options['revenue-ops-rebalance-interval']),
        hot_channel_protection_enabled=options.get('revenue-ops-hot-channel-protection-enabled', 'true').lower() == 'true',
        hot_channel_protection_min_velocity=float(options.get('revenue-ops-hot-channel-protection-min-velocity', '0.20')),
        hot_channel_protection_min_marginal_roi=float(options.get('revenue-ops-hot-channel-protection-min-marginal-roi', '0.20')),
        hot_channel_protection_profit_budget_pct=float(options.get('revenue-ops-hot-channel-protection-profit-budget-pct', '0.75')),
        hot_channel_protection_max_chunk_multiplier=float(options.get('revenue-ops-hot-channel-protection-max-chunk-multiplier', '4.0')),
        hot_channel_protection_min_cooldown_hours=float(options.get('revenue-ops-hot-channel-protection-min-cooldown-hours', '1.0')),
        boltz_auto_cycle_enabled=options.get('revenue-ops-boltz-auto-cycle-enabled', 'true').lower() == 'true',
        boltz_auto_cycle_interval_minutes=int(options.get('revenue-ops-boltz-auto-cycle-interval-minutes', '15')),
        boltz_auto_cycle_max_actions=int(options.get('revenue-ops-boltz-auto-cycle-max-actions', '1')),
        boltz_auto_cycle_startup_delay_seconds=int(options.get('revenue-ops-boltz-auto-cycle-startup-delay-seconds', '120')),
        target_flow=int(options['revenue-ops-target-flow']),
        min_fee_ppm=int(options['revenue-ops-min-fee-ppm']),
        max_fee_ppm=int(options['revenue-ops-max-fee-ppm']),
        rebalance_min_profit=int(options['revenue-ops-rebalance-min-profit']),
        flow_window_days=int(options['revenue-ops-flow-window-days']),
        clboss_enabled=options['revenue-ops-clboss-enabled'].lower() == 'true',
        rebalancer_plugin=options['revenue-ops-rebalancer'],
        daily_budget_sats=int(options['revenue-ops-daily-budget-sats']),
        total_cost_budget_mode=options.get('revenue-ops-total-cost-budget-mode', 'fixed').lower(),
        total_cost_budget_profit_pct=float(options.get('revenue-ops-total-cost-budget-profit-pct', '0.30')),
        total_cost_budget_profit_pct_cap=float(options.get('revenue-ops-total-cost-budget-profit-pct-cap', '0.75')),
        total_cost_budget_window_hours=int(options.get('revenue-ops-total-cost-budget-window-hours', '24')),
        min_wallet_reserve=int(options['revenue-ops-min-wallet-reserve']),
        enable_proportional_budget=options['revenue-ops-proportional-budget'].lower() == 'true',
        proportional_budget_pct=float(options['revenue-ops-proportional-budget-pct']),
        dry_run=options['revenue-ops-dry-run'].lower() == 'true',
        htlc_congestion_threshold=float(options['revenue-ops-htlc-congestion-threshold']),
        enable_reputation=options['revenue-ops-enable-reputation'].lower() == 'true',
        reputation_decay=float(options['revenue-ops-reputation-decay']),
        enable_kelly=options['revenue-ops-enable-kelly'].lower() == 'true',
        kelly_bypass_for_fleet=options['revenue-ops-kelly-bypass-fleet'].lower() == 'true',
        kelly_fraction=float(options['revenue-ops-kelly-fraction']),
        # Phase 7 options (v1.3.0)
        enable_vegas_reflex=options['revenue-ops-vegas-reflex'].lower() == 'true',
        vegas_decay_rate=float(options['revenue-ops-vegas-decay']),
        enable_scarcity_pricing=options['revenue-ops-scarcity-pricing'].lower() == 'true',
        scarcity_threshold=float(options['revenue-ops-scarcity-threshold']),
        rpc_timeout_seconds=int(options['revenue-ops-rpc-timeout-seconds']),
        rpc_circuit_breaker_seconds=int(options['revenue-ops-rpc-circuit-breaker-seconds']),
        reservation_timeout_hours=int(options['revenue-ops-reservation-timeout-hours']),
        # Phase 9: Hive Integration (cl-hive fleet coordination)
        hive_enabled=options['revenue-ops-hive-enabled'].lower(),
        hive_fee_ppm=int(options['revenue-ops-hive-fee-ppm']),
        hive_rebalance_tolerance=int(options['revenue-ops-hive-rebalance-tolerance'])
    )
    try:
        config_fields = {f.name for f in dataclasses.fields(Config)}
    except Exception:
        config_fields = set(config_kwargs.keys())
    dropped = [k for k in config_kwargs.keys() if k not in config_fields]
    if dropped:
        plugin.log(
            f"Config compatibility: dropping unsupported Config fields during init: {', '.join(sorted(dropped))}",
            level='warn'
        )
    config = Config(**{k: v for k, v in config_kwargs.items() if k in config_fields})
    
    plugin.log(f"Configuration loaded: target_flow={config.target_flow}, "
               f"fee_range=[{config.min_fee_ppm}, {config.max_fee_ppm}], "
               f"dry_run={config.dry_run}")
    
    # Create thread-safe RPC proxy (Phase 5.5: High-Uptime Stability)
    # pyln-client opens a new Unix socket per call — thread-safe by design.
    # ThreadSafeRpcProxy adds timeout protection via ThreadPoolExecutor.
    safe_plugin = ThreadSafePluginProxy(plugin)
    plugin.log("Thread-safe RPC proxy initialized", level="info")

    # Optional Boltz CLI integration (manual quotes/swaps, wallet ops).
    try:
        boltz_cfg = BoltzCliConfig(
            enabled=options.get('revenue-ops-boltz-enabled', 'false').lower() == 'true',
            cli_path=options.get('revenue-ops-boltz-cli-path', '/usr/local/bin/boltzcli'),
            datadir=options.get('revenue-ops-boltz-datadir', '/var/lib/boltz'),
            use_sudo=options.get('revenue-ops-boltz-use-sudo', 'false').lower() == 'true',
            sudo_user=options.get('revenue-ops-boltz-sudo-user', 'boltz'),
            timeout_seconds=int(options.get('revenue-ops-boltz-timeout-seconds', '60')),
            daily_budget_sats=int(options.get('revenue-ops-boltz-daily-budget-sats', '3000')),
            enforce_budget=options.get('revenue-ops-boltz-enforce-budget', 'true').lower() == 'true',
            btc_wallet=options.get('revenue-ops-boltz-btc-wallet', 'CLN'),
            lbtc_wallet=options.get('revenue-ops-boltz-lbtc-wallet', 'LOOP-LBTC'),
        )
        boltz_manager = BoltzCliManager(safe_plugin, safe_plugin.rpc, boltz_cfg)
        if boltz_cfg.enabled:
            plugin.log(
                f"Boltz CLI integration enabled (datadir={boltz_cfg.datadir}, "
                f"sudo={boltz_cfg.use_sudo}, budget={boltz_cfg.daily_budget_sats} sats/day)",
                level='info'
            )
        else:
            plugin.log("Boltz CLI integration disabled (set revenue-ops-boltz-enabled=true to enable)", level='debug')
    except Exception as e:
        boltz_manager = None
        plugin.log(f"Warning: Failed to initialize Boltz CLI integration: {e}", level='warn')

    # =========================================================================
    # STARTUP DEPENDENCY CHECKS (Phase 4: Stability & Scaling)
    # Verify external plugins are available before initializing dependent modules
    # =========================================================================
    try:
        # Try modern 'plugin list' command first, fallback to 'listplugins' for older nodes
        try:
            # Modern CLN (v23.08+)
            plugins_result = safe_plugin.rpc.plugin("list")
        except RpcError:
            # Fallback for older CLN versions
            plugins_result = safe_plugin.rpc.listplugins()
            
        active_plugins = [p.get("name", "").lower() for p in plugins_result.get("plugins", [])]
        
        # Check for sling plugin
        sling_found = any("sling" in name for name in active_plugins)
        if not sling_found:
            plugin.log(
                "Dependency 'sling' not found. Rebalancing module disabled. "
                "Install cln-sling to enable rebalancing.",
                level='warn'
            )
            config.sling_available = False
        else:
            plugin.log("Dependency check: sling plugin detected")
            config.sling_available = True
        
        # Check for bookkeeper plugin
        bookkeeper_found = any("bookkeeper" in name for name in active_plugins)
        if not bookkeeper_found:
            plugin.log(
                "Dependency 'bookkeeper' not found. Flow analysis uses the local forwards table (hydrated once at startup). "
                "Bookkeeper is still recommended for accurate on-chain cost tracking (opens/closes/splices).",
                level='info'
            )
        else:
            plugin.log("Dependency check: bookkeeper plugin detected")
            
    except Exception as e:
        plugin.log(f"Error checking plugin dependencies: {e}", level='warn')
        # EH-10: Fail-closed — assume sling unavailable if check fails
        config.sling_available = False
    
    
    # Initialize database
    database = Database(config.db_path, safe_plugin)
    database.initialize()

    # Issue #24: Clean up stale budget reservations on startup
    # Reservations from crashed jobs should be released immediately
    timeout_seconds = config.reservation_timeout_hours * 3600
    cleaned = database.cleanup_stale_reservations(timeout_seconds)
    if cleaned > 0:
        plugin.log(f"Startup cleanup: Released {cleaned} stale budget reservations")

    # Phase 7: Load config overrides from database (persisted runtime changes)
    try:
        config.load_overrides(database)
        if config._version > 0:
            plugin.log(f"Loaded config overrides from database (version {config._version})")
    except Exception as e:
        plugin.log(f"Warning: Could not load config overrides: {e}", level='warn')
    
    # =========================================================================
    # FORWARDS TABLE HYDRATION (TODO #19: Double-Dip Fix)
    # =========================================================================
    # The forwards table is populated in real-time by forward_event hook.
    # However, when the plugin restarts, we may have gaps in the data.
    # This hydration fills those gaps by calling listforwards RPC ONCE on startup.
    # After this, flow_analysis.py uses only local DB (no more RPC calls).
    # =========================================================================
    try:
        # Check DB head: get timestamp of the most recent forward
        last_forward_ts = database.get_latest_forward_timestamp()
        now = int(time.time())

        if last_forward_ts is None:
            # Empty database - hydrate from flow_window_days ago (or 14 days default)
            hydrate_days = max(config.flow_window_days, 14)
            start_time = now - (hydrate_days * 86400)
            plugin.log(f"Forwards table empty. Hydrating last {hydrate_days} days of forwards...")
        elif (now - last_forward_ts) < 600:
            # Table was updated within the last 10 minutes — the real-time
            # forward_event hook kept it current.  Skip the expensive
            # listforwards RPC (can return millions of rows on busy nodes).
            plugin.log("Forwards table is current; skipping hydration", level='debug')
            start_time = None
        else:
            # Have data but a gap exists — fetch what we missed while offline
            start_time = max(0, last_forward_ts - 3600)
            plugin.log(f"Hydrating forwards since {time.strftime('%Y-%m-%d %H:%M', time.localtime(start_time))}...")

        if start_time is not None:
            # Fetch from RPC - this is the ONLY listforwards call we make.
            # EH-12: Limit to recent forwards to avoid unbounded memory usage on large nodes.
            # CLN v23.08+ supports index-based pagination; fall back to full fetch if unavailable.
            # CLN's listforwards `start` param expects a created_index (sequential
            # counter), NOT a Unix timestamp.  Passing a timestamp silently returns
            # zero results on any node with < 1.7 billion forwards.  Use unfiltered
            # fetch and rely on the post-filter at received_time > start_time below.
            try:
                result = safe_plugin.rpc.listforwards(status="settled")
            except Exception:
                result = {"forwards": []}
            forwards_to_insert = []

            for fwd in result.get("forwards", []):
                received_time = fwd.get("received_time", 0)
                if received_time > start_time:
                    forwards_to_insert.append({
                        'in_channel': fwd.get("in_channel", ""),
                        'out_channel': fwd.get("out_channel", ""),
                        'in_msat': _parse_msat(fwd.get("in_msat", fwd.get("in_msatoshi", 0))),
                        'out_msat': _parse_msat(fwd.get("out_msat", fwd.get("out_msatoshi", 0))),
                        'fee_msat': _parse_msat(fwd.get("fee_msat", fwd.get("fee_msatoshi", 0))),
                        'resolution_time': (fwd.get("resolved_time", 0) - received_time) if fwd.get("resolved_time") else 0,
                        'received_time': received_time,
                        'resolved_time': int(fwd.get("resolved_time", 0) or 0)
                    })

            if forwards_to_insert:
                inserted = database.bulk_insert_forwards(forwards_to_insert)
                if inserted > 0:
                    plugin.log(f"Hydration complete: inserted {inserted} forwards into local database")
                else:
                    plugin.log(f"Hydration: {len(forwards_to_insert)} forwards already in database", level='debug')
            else:
                plugin.log("Hydration complete: no new forwards to insert", level='debug')

    except Exception as e:
        plugin.log(f"Warning: Forwards hydration failed: {e}", level='warn')
        # Non-fatal - flow analysis will work with whatever data we have
    
    # Snapshot currently connected peers for baseline state on restart
    # This establishes a known state for uptime tracking after plugin restarts
    try:
        peers = safe_plugin.rpc.listpeers()
        total_peers = len(peers.get("peers", []))
        connected_peers = 0
        snapshot_count = 0
        
        plugin.log(f"Checking {total_peers} peers for connection snapshot...")
        
        for peer in peers.get("peers", []):
            if peer.get("connected", False):
                connected_peers += 1
                peer_id = peer["id"]
                # Only insert snapshot if no recent history exists (within 1 hour)
                has_recent = database.has_recent_connection_history(peer_id, 3600)
                plugin.log(f"Peer {peer_id[:12]}... is connected, has_recent_history={has_recent}", level='debug')
                if not has_recent:
                    database.record_connection_event(peer_id, "snapshot")
                    snapshot_count += 1
        
        plugin.log(f"Connection baseline: {connected_peers} connected peers, snapshotted {snapshot_count} new peers")
    except Exception as e:
        plugin.log(f"Error snapshotting peer connections: {e}", level='warn')
        plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')  # SEC-2: Stack traces at debug level
    
    # Initialize clboss manager (handles unmanage commands)
    clboss_manager = ClbossManager(safe_plugin, config)
    
    # Initialize policy manager (v1.4: Policy-Driven Architecture)
    policy_manager = PolicyManager(database, safe_plugin)
    plugin.log("PolicyManager initialized for peer-level fee/rebalance policies")

    # Initialize hive bridge for competitor intelligence and NNLB health (v1.6)
    # Respect hive_enabled setting: "auto", "true", "false"
    if config.hive_enabled == 'false':
        # Standalone mode - no hive integration
        hive_bridge = None
        plugin.log("=" * 60)
        plugin.log("STANDALONE MODE: Hive integration disabled (hive-enabled=false)")
        plugin.log("All fee optimization and rebalancing will use local-only algorithms")
        plugin.log("To join a hive, set revenue-ops-hive-enabled=auto or true")
        plugin.log("=" * 60)
    else:
        # Auto or required hive mode
        # During plugin init, CLN holds a lock that blocks plugin-to-plugin
        # RPCs (like hive-status), so we can only check plugin("list") here.
        # Membership is verified lazily by background threads after init.
        hive_bridge = HiveFeeIntelligenceBridge(safe_plugin, database, config=config)
        hive_bridge._init_complete = False  # Block hive calls until init finishes
        hive_loaded = False
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                plugins = safe_plugin.rpc.plugin("list")
                hive_loaded = any(
                    "cl-hive" in p.get("name", "") and p.get("active", False)
                    for p in plugins.get("plugins", [])
                )
                if hive_loaded:
                    break
            except Exception as e:
                plugin.log(f"Plugin list check failed: {e}", level="debug")
            if attempt < max_attempts - 1:
                wait = 3 if attempt < 2 else 5
                plugin.log(f"Waiting for cl-hive (attempt {attempt + 1}/{max_attempts})...")
                # EH-4: Use shutdown_event.wait() so init retry loop is interruptible
                shutdown_event.wait(wait)

        if hive_loaded:
            # cl-hive is loaded — report hive mode for startup logging.
            # Don't seed _hive_available: plugin-to-plugin RPCs are blocked
            # during init (CLN holds a lock), so background threads must not
            # attempt hive calls until init completes.  Leave the cache
            # empty so the first post-init is_available() probe will verify
            # membership via plugin list.
            plugin.log("=" * 60)
            plugin.log("HIVE MODE ACTIVE: cl-hive detected")
            plugin.log("Hive features enabled:")
            plugin.log("  - Coordinated fee recommendations")
            plugin.log("  - Fleet-wide fee intelligence")
            plugin.log("  - Rebalancing conflict detection")
            plugin.log("  - Collective defense against drain attacks")
            plugin.log("  - Anticipatory liquidity predictions")
            plugin.log("Membership will be verified after init completes")
            plugin.log("=" * 60)
        elif config.hive_enabled == 'true':
            # Required mode but hive not available - warn but continue
            plugin.log("=" * 60, level='warn')
            plugin.log("WARNING: hive-enabled=true but cl-hive not loaded!", level='warn')
            plugin.log("Possible reasons:", level='warn')
            plugin.log("  - cl-hive plugin not installed or failed to start", level='warn')
            plugin.log("Hive features will activate when cl-hive becomes available", level='warn')
            plugin.log("Plugin will continue in standalone mode", level='warn')
            plugin.log("=" * 60, level='warn')
        else:
            plugin.log("=" * 60)
            plugin.log("STANDALONE MODE: cl-hive not detected (hive-enabled=auto)")
            plugin.log("All fee optimization and rebalancing will use local-only algorithms")
            plugin.log("To join a hive: open a channel to any hive member")
            plugin.log("=" * 60)

    # Initialize profitability analyzer with hive bridge for NNLB health reporting
    profitability_analyzer = ChannelProfitabilityAnalyzer(
        safe_plugin, config, database, hive_bridge=hive_bridge
    )

    # Initialize analysis modules with profitability analyzer and hive bridge
    flow_analyzer = FlowAnalyzer(safe_plugin, config, database)
    capacity_planner = CapacityPlanner(safe_plugin, config, profitability_analyzer, flow_analyzer)
    fee_controller = PIDFeeController(safe_plugin, config, database, clboss_manager, policy_manager, profitability_analyzer, hive_bridge)
    rebalancer = EVRebalancer(
        safe_plugin, config, database, clboss_manager, policy_manager,
        hive_bridge=hive_bridge
    )
    rebalancer.set_profitability_analyzer(profitability_analyzer)
    # Unified liquidity-cost accounting:
    # - Rebalancer sees Boltz spend as external liquidity cost
    # - Boltz manager sees rebalance spend/reservations as external liquidity cost
    if rebalancer is not None:
        rebalancer.external_liquidity_cost_provider = _non_rebalance_liquidity_cost_components
        rebalancer.global_budget_limit_provider = _total_cost_budget_limit_provider
    if boltz_manager is not None:
        boltz_manager.external_liquidity_cost_provider = _non_boltz_liquidity_cost_components
        boltz_manager.global_budget_limit_provider = _total_cost_budget_limit_provider

    # =========================================================================
    # Hive Settlement / Yield Reporting (Issue #42)
    # =========================================================================
    # cl-hive supports settlement based on net yield (revenue - costs). We report
    # routing revenue and operating costs periodically so the fleet can settle
    # fairly across members with different rebalance spend profiles.
    YIELD_REPORT_WINDOW_DAYS = 30
    YIELD_REPORT_INTERVAL_SECONDS = 6 * 3600  # report at most every 6 hours
    last_yield_report_time = 0

    def _maybe_report_yield_and_costs() -> None:
        nonlocal last_yield_report_time
        if not hive_bridge:
            return
        try:
            if not hive_bridge.is_available():
                return
        except Exception:
            return

        now = int(time.time())
        if last_yield_report_time and (now - last_yield_report_time) < YIELD_REPORT_INTERVAL_SECONDS:
            return

        try:
            tlv = profitability_analyzer.get_tlv().get("tlv_sats", 0)
            pnl = profitability_analyzer.get_pnl_summary(window_days=YIELD_REPORT_WINDOW_DAYS)
            hive_bridge.report_yield_and_costs(
                tlv_sats=int(tlv or 0),
                operating_costs_sats=int(pnl.get("opex_sats", 0) or 0),
                routing_revenue_sats=int(pnl.get("gross_revenue_sats", 0) or 0),
                rebalance_costs_sats=int(pnl.get("rebalance_cost_sats", 0) or 0),
                period_days=YIELD_REPORT_WINDOW_DAYS,
            )
            last_yield_report_time = now
        except Exception as e:
            plugin.log(f"Hive yield/cost report failed (non-fatal): {e}", level="debug")
    
    # Set up periodic background tasks using threading
    # Note: plugin.log() is safe to call from threads in pyln-client
    # We use daemon threads so they don't block shutdown
    
    def flow_analysis_loop():
        """Background loop for flow analysis."""
        # Initial delay to let lightningd fully start (interruptible)
        if shutdown_event.wait(10):
            plugin.log("Flow analysis loop cancelled during startup delay")
            return
        
        while not shutdown_event.is_set():
            try:
                plugin.log("Running scheduled flow analysis...")
                run_flow_analysis()
                
                # Run cleanup on each iteration (it's a fast DELETE query)
                # Keeps history tables from growing unbounded over months
                # Use flow_window_days + 1 day buffer, minimum 8 days
                if database:
                    days_to_keep = max(8, config.flow_window_days + 1)
                    database.cleanup_old_data(days_to_keep=days_to_keep)
                
            except (RPCTimeoutError, RPCBreakerOpen) as e:
                plugin.log(f"RPC degraded in flow analysis: {e}. Skipping this cycle.", level='warn')
            except Exception as e:
                plugin.log(f"Error in flow analysis: {e}", level='error')
            
            # Calculate +/- 20% jitter (minimum 60s to prevent busy loop)
            interval = max(60, config.flow_interval)
            jitter_seconds = int(interval * 0.2)
            sleep_time = interval + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Flow analysis sleeping for {sleep_time}s")
            
            # Interruptible sleep: wait for timeout OR shutdown signal
            if shutdown_event.wait(sleep_time):
                plugin.log("Flow analysis loop stopping due to shutdown signal")
                break
    
    def fee_adjustment_loop():
        """Background loop for fee adjustment."""
        # Initial delay to let flow analysis run first (interruptible)
        if shutdown_event.wait(60):
            plugin.log("Fee adjustment loop cancelled during startup delay")
            return

        while not shutdown_event.is_set():
            try:
                plugin.log("Running scheduled fee adjustment...")
                run_fee_adjustment()
                _maybe_report_yield_and_costs()
            except (RPCTimeoutError, RPCBreakerOpen) as e:
                plugin.log(f"RPC degraded in fee adjustment: {e}. Skipping this cycle.", level='warn')
            except Exception as e:
                plugin.log(f"Error in fee adjustment: {e}", level='error')

            # Calculate +/- 20% jitter (minimum 60s to prevent busy loop)
            interval = max(60, config.fee_interval)
            jitter_seconds = int(interval * 0.2)
            sleep_time = interval + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Fee adjustment sleeping for {sleep_time}s")

            # Interruptible sleep: wait for timeout OR shutdown signal
            if shutdown_event.wait(sleep_time):
                plugin.log("Fee adjustment loop stopping due to shutdown signal")
                break
    
    def rebalance_check_loop():
        """Background loop for rebalance checks."""
        # Skip rebalancing entirely if sling is not available
        if not config.sling_available:
            plugin.log("Rebalance loop disabled: sling plugin not found")
            return
        
        # Initial delay to let other analyses run first (interruptible)
        if shutdown_event.wait(120):
            plugin.log("Rebalance check loop cancelled during startup delay")
            return
        
        while not shutdown_event.is_set():
            try:
                plugin.log("Running scheduled rebalance check...")
                run_rebalance_check()
            except (RPCTimeoutError, RPCBreakerOpen) as e:
                plugin.log(f"RPC degraded in rebalance check: {e}. Skipping this cycle.", level='warn')
            except Exception as e:
                plugin.log(f"Error in rebalance check: {e}", level='error')
            
            # Calculate +/- 20% jitter (minimum 60s to prevent busy loop)
            interval = max(60, config.rebalance_interval)
            jitter_seconds = int(interval * 0.2)
            sleep_time = interval + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Rebalance check sleeping for {sleep_time}s")
            
            # Interruptible sleep: wait for timeout OR shutdown signal
            if shutdown_event.wait(sleep_time):
                plugin.log("Rebalance check loop stopping due to shutdown signal")
                break
    
    def _boltz_auto_cycle_mark_state(**updates):
        with _boltz_auto_cycle_state_lock:
            _boltz_auto_cycle_state.update(updates)

    def _run_boltz_auto_cycle_once(trigger: str = "manual", force: bool = False) -> Dict[str, Any]:
        """Run one in-plugin Boltz auto-cycle using existing RPC logic (single-flight)."""
        if boltz_manager is None or not getattr(boltz_manager, 'enabled', False):
            result = {
                'status': 'disabled',
                'reason': 'boltz integration disabled',
                'trigger': trigger,
            }
            _boltz_auto_cycle_mark_state(last_result=result, last_error=None, enabled=False)
            return result

        cfg = config.snapshot() if config else None
        enabled = bool(getattr(cfg, 'boltz_auto_cycle_enabled', True)) if cfg else True
        _boltz_auto_cycle_mark_state(enabled=enabled)
        if not enabled and not force:
            result = {
                'status': 'disabled',
                'reason': 'boltz auto-cycle disabled by config',
                'trigger': trigger,
            }
            _boltz_auto_cycle_mark_state(last_result=result, last_error=None)
            return result

        acquired = _boltz_auto_cycle_run_lock.acquire(blocking=False)
        if not acquired:
            result = {'status': 'skipped', 'reason': 'auto-cycle already running', 'trigger': trigger}
            _boltz_auto_cycle_mark_state(last_result=result)
            return result

        started = int(time.time())
        start_monotonic = time.monotonic()
        _boltz_auto_cycle_mark_state(
            running=True,
            last_trigger=trigger,
            last_started_ts=started,
            last_error=None,
        )
        try:
            max_actions = max(1, int(getattr(cfg, 'boltz_auto_cycle_max_actions', 1) if cfg else 1))
            result = revenue_boltz_balance_cycle(
                plugin=plugin,
                dry_run=False,
                max_actions=max_actions,
                allow_concurrent_swaps=False,
                loop_in_currency='LBTC',
                loop_out_currency='LBTC',
            )
            status = str(result.get('status') or 'unknown') if isinstance(result, dict) else 'unknown'
            if isinstance(result, dict) and 'error' in result:
                with _boltz_auto_cycle_state_lock:
                    _boltz_auto_cycle_state['consecutive_errors'] = int(_boltz_auto_cycle_state.get('consecutive_errors', 0) or 0) + 1
                _boltz_auto_cycle_mark_state(last_error=str(result.get('error')))
            else:
                _boltz_auto_cycle_mark_state(last_error=None)
                with _boltz_auto_cycle_state_lock:
                    _boltz_auto_cycle_state['consecutive_errors'] = 0
            return result if isinstance(result, dict) else {'status': status, 'result': result}
        except Exception as e:
            with _boltz_auto_cycle_state_lock:
                _boltz_auto_cycle_state['consecutive_errors'] = int(_boltz_auto_cycle_state.get('consecutive_errors', 0) or 0) + 1
            _boltz_auto_cycle_mark_state(last_error=str(e))
            raise
        finally:
            finished = int(time.time())
            duration_ms = int((time.monotonic() - start_monotonic) * 1000)
            with _boltz_auto_cycle_state_lock:
                _boltz_auto_cycle_state['running'] = False
                _boltz_auto_cycle_state['last_finished_ts'] = finished
                _boltz_auto_cycle_state['last_duration_ms'] = duration_ms
                # Preserve last_result if set by success/skip/disabled path, otherwise leave as-is.
            _boltz_auto_cycle_run_lock.release()

    def boltz_auto_cycle_loop():
        """Background loop for profit-gated Boltz auto-balance cycles."""
        if boltz_manager is None or not getattr(boltz_manager, 'enabled', False):
            _boltz_auto_cycle_mark_state(enabled=False, thread_started=False, next_run_ts=None)
            plugin.log("Boltz auto-cycle loop disabled: Boltz CLI integration not enabled", level='debug')
            return

        _boltz_auto_cycle_mark_state(enabled=bool(getattr(config, 'boltz_auto_cycle_enabled', True)), thread_started=True)

        startup_delay = max(0, int(getattr(config, 'boltz_auto_cycle_startup_delay_seconds', 120) or 0))
        if startup_delay > 0:
            _boltz_auto_cycle_mark_state(next_run_ts=int(time.time()) + startup_delay)
            if shutdown_event.wait(startup_delay):
                plugin.log("Boltz auto-cycle loop cancelled during startup delay")
                _boltz_auto_cycle_mark_state(next_run_ts=None)
                return

        while not shutdown_event.is_set():
            enabled = bool(getattr(config, 'boltz_auto_cycle_enabled', True))
            _boltz_auto_cycle_mark_state(enabled=enabled)

            if enabled:
                try:
                    result = _run_boltz_auto_cycle_once(trigger='scheduler')
                    _boltz_auto_cycle_mark_state(last_result=result)
                    summary = ''
                    if isinstance(result, dict):
                        summary = f"status={result.get('status')} executed={result.get('executed_count', 0)} skipped={result.get('skipped_count', 0)}"
                    plugin.log(f"Boltz auto-cycle completed ({summary})", level='debug')
                except Exception as e:
                    plugin.log(f"Error in Boltz auto-cycle: {e}", level='warn')
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
            else:
                _boltz_auto_cycle_mark_state(last_result={'status': 'disabled', 'reason': 'boltz auto-cycle disabled by config', 'trigger': 'scheduler'})

            interval_min = max(1, int(getattr(config, 'boltz_auto_cycle_interval_minutes', 15) or 15))
            interval_sec = interval_min * 60
            jitter = max(0, int(interval_sec * 0.1))
            sleep_time = interval_sec + (random.randint(-jitter, jitter) if jitter > 0 else 0)
            _boltz_auto_cycle_mark_state(next_run_ts=int(time.time()) + max(1, sleep_time))
            if shutdown_event.wait(max(1, sleep_time)):
                plugin.log("Boltz auto-cycle loop stopping due to shutdown signal")
                break

        _boltz_auto_cycle_mark_state(next_run_ts=None, running=False)

    def snapshot_peers_delayed():
        """
        One-time delayed snapshot of connected peers.
        
        Sleeps to allow lightningd to establish connections, then records
        a snapshot for all currently connected peers. Exits after completion.
        """
        delay_seconds = 60
        plugin.log(f"Startup snapshot: waiting {delay_seconds}s for network connections...")
        
        # Interruptible delay
        if shutdown_event.wait(delay_seconds):
            plugin.log("Startup snapshot cancelled due to shutdown signal")
            return
        
        try:
            peers = safe_plugin.rpc.listpeers()
            connected_count = 0
            snapshot_count = 0
            
            for peer in peers.get("peers", []):
                if peer.get("connected", False):
                    connected_count += 1
                    peer_id = peer["id"]
                    # Only snapshot if no recent history exists
                    if not database.has_recent_connection_history(peer_id, 3600):
                        database.record_connection_event(peer_id, "snapshot")
                        snapshot_count += 1
            
            plugin.log(f"Startup snapshot: Recorded {snapshot_count} of {connected_count} connected peers")
        except Exception as e:
            plugin.log(f"Error in delayed snapshot: {e}", level='warn')
            plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')  # SEC-2: Stack traces at debug level

    def financial_snapshot_loop():
        """
        Background loop for daily financial snapshots (Phase 8: Dashboard).

        Takes a snapshot of TLV, balances, and accumulated P&L metrics
        once every 24 hours for historical trend analysis.
        """
        SNAPSHOT_INTERVAL = 86400  # 24 hours in seconds

        # Initial delay: wait 5 minutes to let everything stabilize
        if shutdown_event.wait(300):
            plugin.log("Financial snapshot loop cancelled during startup delay")
            return

        # Take an initial snapshot on startup
        try:
            _take_financial_snapshot()
        except Exception as e:
            plugin.log(f"Error taking initial financial snapshot: {e}", level='warn')

        while not shutdown_event.is_set():
            # Calculate +/- 10% jitter (about 2.4 hours variance)
            jitter_seconds = int(SNAPSHOT_INTERVAL * 0.1)
            sleep_time = SNAPSHOT_INTERVAL + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Financial snapshot sleeping for {sleep_time // 3600}h {(sleep_time % 3600) // 60}m")

            # Interruptible sleep
            if shutdown_event.wait(sleep_time):
                plugin.log("Financial snapshot loop stopping due to shutdown signal")
                break

            try:
                _take_financial_snapshot()
            except (RPCTimeoutError, RPCBreakerOpen) as e:
                plugin.log(f"RPC degraded in financial snapshot: {e}. Skipping this cycle.", level='warn')
            except Exception as e:
                plugin.log(f"Error in financial snapshot: {e}", level='error')

    def _take_financial_snapshot():
        """Take a single financial snapshot and record it to the database."""
        if database is None or profitability_analyzer is None:
            plugin.log("Cannot take financial snapshot: components not initialized", level='warn')
            return

        # Get current TLV data
        tlv_data = profitability_analyzer.get_tlv()

        # Get lifetime accumulated stats
        lifetime_stats = database.get_lifetime_stats()

        # Convert revenue from msat to sats (get_lifetime_stats returns msat)
        revenue_msat = lifetime_stats.get("total_revenue_msat", 0)
        revenue_sats = revenue_msat // 1000

        # Record the snapshot
        local_bal = tlv_data.get("local_balance_sats", 0)
        remote_bal = tlv_data.get("remote_balance_sats", 0)
        database.record_financial_snapshot(
            local_balance_sats=local_bal,
            remote_balance_sats=remote_bal,
            onchain_sats=tlv_data.get("onchain_sats", 0),
            capacity_sats=local_bal + remote_bal,
            revenue_accumulated_sats=revenue_sats,
            rebalance_cost_accumulated_sats=lifetime_stats.get("total_rebalance_cost_sats", 0),
            channel_count=tlv_data.get("channel_count", 0)
        )

        plugin.log(
            f"Financial snapshot recorded: TLV={tlv_data.get('tlv_sats', 0)} sats, "
            f"channels={tlv_data.get('channel_count', 0)}"
        )

    # =========================================================================
    # STARTUP HYGIENE: Clean up orphan jobs from previous runs
    # =========================================================================
    if rebalancer and config.sling_available:
        try:
            rebalancer.job_manager.cleanup_orphans()
        except Exception as e:
            plugin.log(f"Warning: Could not clean up orphan jobs: {e}", level='warn')

        # M-11: Clean up stale budget reservations from crashed jobs
        try:
            database.cleanup_stale_reservations()
        except Exception as e:
            plugin.log(f"Warning: Could not clean up stale reservations: {e}", level='warn')

        # PHASE 6: Sync peer exclusions with sling on startup
        try:
            rebalancer.job_manager.sync_peer_exclusions(policy_manager)
        except Exception as e:
            plugin.log(f"Warning: Could not sync peer exclusions: {e}", level='warn')

        # Sling hygiene: stats retention settings.
        # NOTE: setconfig on plugin-owned options (sling-*) triggers a segfault
        # in CLN v25.12.1 (configvar_finalize_overrides). These must be set in
        # the CLN config file at startup instead:
        #   sling-stats-delete-failures-age=30
        #   sling-stats-delete-successes-age=30
        #   sling-candidates-min-age=144
        plugin.log("Sling hygiene: configure stats retention in CLN config (setconfig unsafe on v25.12.1)", level='debug')
    
    # Start background threads (daemon=True so they don't block shutdown)
    threading.Thread(target=flow_analysis_loop, daemon=True, name="flow-analysis").start()
    threading.Thread(target=fee_adjustment_loop, daemon=True, name="fee-adjustment").start()
    threading.Thread(target=rebalance_check_loop, daemon=True, name="rebalance-check").start()
    threading.Thread(target=snapshot_peers_delayed, daemon=True, name="startup-snapshot").start()
    threading.Thread(target=financial_snapshot_loop, daemon=True, name="financial-snapshot").start()
    threading.Thread(target=boltz_auto_cycle_loop, daemon=True, name="boltz-auto-cycle").start()

    # Signal that init is complete — hive bridge can now make plugin-to-plugin
    # RPCs safely (CLN releases its lock after init returns).
    if hive_bridge:
        hive_bridge.mark_init_complete()

    plugin.log("cl-revenue-ops plugin initialized successfully!")
    return None


# =============================================================================
# CORE LOGIC FUNCTIONS
# =============================================================================

def run_flow_analysis():
    """
    Module 1: Flow Analysis & Sink/Source Detection
    
    Query bookkeeper to calculate the "Net Flow" of every channel over 
    the last N days. Calculate FlowRatio and mark channels as Source/Sink/Balanced.
    
    Also applies reputation decay to ensure recent peer behavior matters more
    than ancient history.
    """
    if flow_analyzer is None:
        plugin.log("Flow analyzer not initialized", level='error')
        return
    
    try:
        results = flow_analyzer.analyze_all_channels()
        plugin.log(f"Flow analysis complete: {len(results)} channels analyzed")
        
        # Log summary
        sources = sum(1 for r in results.values() if r.state == ChannelState.SOURCE)
        sinks = sum(1 for r in results.values() if r.state == ChannelState.SINK)
        balanced = sum(1 for r in results.values() if r.state == ChannelState.BALANCED)
        plugin.log(f"Channel states: {sources} sources, {sinks} sinks, {balanced} balanced")
        
        # Apply reputation decay (Phase 3: Time-windowing)
        # This ensures recent peer behavior matters more than ancient history
        if database and config and config.enable_reputation:
            database.decay_reputation(config.reputation_decay)
            plugin.log(f"Applied reputation decay (factor={config.reputation_decay})")

        # Report flow observations to cl-hive for temporal pattern detection
        if hive_bridge and hive_bridge.is_available():
            reported = 0
            for channel_id, metrics in results.items():
                try:
                    hive_bridge.report_flow_observation(
                        channel_id=channel_id,
                        inbound_sats=metrics.sats_in,
                        outbound_sats=metrics.sats_out
                    )
                    reported += 1
                except Exception as e:
                    plugin.log(f"Flow observation report failed for {channel_id[:12]}...: {e}", level='debug')
            if reported > 0:
                plugin.log(f"Reported {reported} flow observations to cl-hive")

    except Exception as e:
        plugin.log(f"Flow analysis failed: {e}", level='error')
        raise


def run_fee_adjustment():
    """
    Module 2: Hill Climbing Fee Controller (Dynamic Pricing)
    
    Adjust channel fees using Perturb & Observe optimization.
    Before setting fees, unmanage from clboss to prevent conflicts.
    """
    if fee_controller is None:
        plugin.log("Fee controller not initialized", level='error')
        return
    
    try:
        adjustments = fee_controller.adjust_all_fees()
        plugin.log(f"Fee adjustment complete: {len(adjustments)} channels adjusted")
        
    except Exception as e:
        plugin.log(f"Fee adjustment failed: {e}", level='error')
        raise


def run_rebalance_check():
    """
    Module 3: EV-Based Rebalancing (Profit-Aware)
    
    Identify rebalance candidates based on expected value calculation.
    Only trigger rebalances when the EV is positive and significant.
    """
    if rebalancer is None:
        plugin.log("Rebalancer not initialized", level='error')
        return
    
    try:
        candidates = rebalancer.find_rebalance_candidates()
        plugin.log(f"Rebalance check complete: {len(candidates)} profitable candidates found")
        
        for candidate in candidates:
            rebalancer.execute_rebalance(candidate)
            
    except Exception as e:
        plugin.log(f"Rebalance check failed: {e}", level='error')
        raise


# =============================================================================
# RPC METHODS - Exposed to lightning-cli
# =============================================================================

@plugin.method("revenue-status")
def revenue_status(plugin: Plugin) -> Dict[str, Any]:
    """
    Get the current status of the revenue operations plugin.
    
    Usage: lightning-cli revenue-status
    """
    if database is None:
        return {"error": "Plugin not fully initialized"}
    
    channel_states = database.get_all_channel_states()
    fee_history = database.get_recent_fee_changes(limit=10)
    rebalance_history = database.get_recent_rebalances(limit=10)
    
    return {
        "status": "running",
        "version": PLUGIN_VERSION,
        "config": {
            "target_flow_sats": config.target_flow,
            "fee_range_ppm": [config.min_fee_ppm, config.max_fee_ppm],
            "rebalance_min_profit_sats": config.rebalance_min_profit,
            "dry_run": config.dry_run
        },
        "channel_states": channel_states,
        "recent_fee_changes": fee_history,
        "recent_rebalances": rebalance_history
    }


@plugin.method("revenue-hive-status")
def revenue_hive_status(plugin: Plugin) -> Dict[str, Any]:
    """
    Get the current hive integration status.

    Shows whether hive mode is enabled, active, and available features.

    Usage: lightning-cli revenue-hive-status
    """
    result = {
        "hive_enabled_setting": config.hive_enabled if config else "unknown",
        "mode": "unknown",
        "hive_bridge_initialized": hive_bridge is not None,
        "cl_hive_available": False,
        "features": {
            "coordinated_fees": False,
            "fleet_intelligence": False,
            "rebalance_coordination": False,
            "collective_defense": False,
            "anticipatory_liquidity": False,
            "time_based_fees": False
        },
        "bridge_status": None,
        "recommendations": []
    }

    if config is None:
        result["error"] = "Plugin not fully initialized"
        return result

    # Determine mode and availability
    if config.hive_enabled == 'false':
        result["mode"] = "standalone"
        result["recommendations"].append(
            "Hive integration is disabled. To enable, set revenue-ops-hive-enabled=auto or true"
        )
    elif hive_bridge is None:
        result["mode"] = "standalone"
        result["recommendations"].append(
            "Hive bridge not initialized. Check plugin startup logs."
        )
    else:
        # Check if cl-hive is available
        result["cl_hive_available"] = hive_bridge.is_available()

        if result["cl_hive_available"]:
            result["mode"] = "hive"
            result["features"] = {
                "coordinated_fees": True,
                "fleet_intelligence": True,
                "rebalance_coordination": True,
                "collective_defense": True,
                "anticipatory_liquidity": True,
                "time_based_fees": True
            }
        else:
            result["mode"] = "standalone_degraded" if config.hive_enabled == 'true' else "standalone"
            if config.hive_enabled == 'true':
                result["recommendations"].append(
                    "hive-enabled=true but hive mode not active. Check if cl-hive is loaded and you are a member."
                )
                result["recommendations"].append(
                    "To join a hive: open a channel to any hive member (permissionless join)"
                )
            else:
                result["recommendations"].append(
                    "Not a hive member. Operating in standalone mode."
                )
                result["recommendations"].append(
                    "To join a hive: install cl-hive and open a channel to any hive member"
                )

        # Get bridge status for diagnostics
        result["bridge_status"] = hive_bridge.get_status()

    # Add hive-specific config
    result["hive_config"] = {
        "hive_fee_ppm": config.hive_fee_ppm,
        "hive_rebalance_tolerance": config.hive_rebalance_tolerance
    }

    return result


@plugin.method("revenue-rebalance-debug")
def revenue_rebalance_debug(plugin: Plugin) -> Dict[str, Any]:
    """
    Diagnostic command to understand why rebalancing may not be happening.

    Shows:
    - Capital control status (budget/reserve)
    - Depleted channels (potential destinations)
    - Source channels (potential sources)
    - Why candidates are rejected

    Usage: lightning-cli revenue-rebalance-debug
    """
    if rebalancer is None:
        return {"error": "Rebalancer not initialized"}

    result = {
        "sling_available": config.sling_available if config else False,
        "dry_run": config.dry_run if config else False,
        "capital_controls": {},
        "thresholds": {},
        "channels": {
            "depleted": [],
            "source": [],
            "active_jobs": []
        },
        "rejection_reasons": []
    }

    if not config.sling_available:
        result["rejection_reasons"].append("Sling plugin not available - rebalancing disabled")
        return result

    # Get thresholds
    cfg = config.snapshot()
    result["thresholds"] = {
        "low_liquidity_threshold": cfg.low_liquidity_threshold,
        "high_liquidity_threshold": cfg.high_liquidity_threshold,
        "rebalance_min_profit_sats": cfg.rebalance_min_profit
    }

    # Check capital controls
    try:
        listfunds = safe_plugin.rpc.listfunds()
        onchain_sats = sum(
            (_parse_msat(o.get("amount_msat", 0)) // 1000)
            for o in listfunds.get("outputs", [])
            if o.get("status") == "confirmed"
        )
        channel_sats = sum(
            (_parse_msat(c.get("our_amount_msat", 0)) // 1000)
            for c in listfunds.get("channels", [])
        )
        total_liquid = onchain_sats + channel_sats

        # Get detailed spending info (Issue #23 + #24)
        spend_info = database.get_daily_rebalance_spend() if database else {}
        daily_spent = spend_info.get('total_spent_sats', 0)
        daily_reserved = spend_info.get('total_reserved_sats', 0)
        stale_count = spend_info.get('stale_reservations', 0)
        total_budget = _total_cost_budget_status()
        daily_budget = int(total_budget.get("effective_budget_sats", cfg.daily_budget_sats) or cfg.daily_budget_sats)
        boltz_costs = _boltz_liquidity_cost_components()
        boltz_spent = int(boltz_costs.get("spent_24h_sats", 0) or 0)
        boltz_reserved = int(boltz_costs.get("reserved_24h_sats", 0) or 0)
        total_liquidity_spent = int(total_budget.get("actual_spent_sats", int(daily_spent) + boltz_spent) or 0)
        total_liquidity_reserved = int(total_budget.get("reserved_sats", int(daily_reserved) + boltz_reserved) or 0)
        budget_remaining = int(total_budget.get("remaining_sats", daily_budget - total_liquidity_spent - total_liquidity_reserved) or 0)

        result["capital_controls"] = {
            "onchain_sats": onchain_sats,
            "channel_sats": channel_sats,
            "total_liquid_sats": total_liquid,
            "wallet_reserve_sats": cfg.min_wallet_reserve,
            "reserve_ok": total_liquid >= cfg.min_wallet_reserve,
            "daily_budget_sats": daily_budget,
            "budget_mode": total_budget.get("mode", "fixed"),
            "budget_window_hours": total_budget.get("window_hours", 24),
            "budget_floor_sats": total_budget.get("daily_budget_floor_sats", cfg.daily_budget_sats),
            "profit_based_budget_sats": total_budget.get("profit_based_budget_sats"),
            "profit_pct_effective": total_budget.get("profit_pct_effective"),
            "daily_spent_sats": daily_spent,          # rebalance-only (legacy field)
            "daily_reserved_sats": daily_reserved,    # rebalance-only (legacy field)
            "boltz_spent_sats": boltz_spent,
            "boltz_reserved_sats": boltz_reserved,
            "total_liquidity_spent_sats": total_liquidity_spent,
            "total_liquidity_reserved_sats": total_liquidity_reserved,
            "total_liquidity_breakdown": {
                "actual_spent_by_category": total_budget.get("actual_spent_by_category", {}),
                "reserved_by_category": total_budget.get("reserved_by_category", {}),
            },
            "stale_reservations": stale_count,
            "budget_remaining_sats": budget_remaining,
            "budget_ok": budget_remaining > 0,
            "job_count": spend_info.get('job_count', 0),
            "success_count": spend_info.get('success_count', 0),
            "success_rate": spend_info.get('success_rate', 0.0)
        }

        if total_liquid < cfg.min_wallet_reserve:
            result["rejection_reasons"].append(
                f"Wallet reserve violated: {total_liquid} < {cfg.min_wallet_reserve}"
            )
        if budget_remaining <= 0:
            result["rejection_reasons"].append(
                f"Unified liquidity budget exhausted: rebalance {daily_spent}+{daily_reserved}, "
                f"boltz {boltz_spent}+{boltz_reserved}, budget {daily_budget}"
            )
        if stale_count > 0:
            result["rejection_reasons"].append(
                f"Warning: {stale_count} stale budget reservations detected (will auto-cleanup)"
            )
    except Exception as e:
        result["capital_controls"]["error"] = str(e)

    # Get channel analysis
    try:
        channels = rebalancer._get_channels_with_balances()
        active_channels = set(rebalancer.job_manager.active_channels)

        for cid, info in channels.items():
            capacity = info.get("capacity", 0)
            if capacity == 0:
                continue

            spendable = info.get("spendable_sats", 0)
            ratio = spendable / capacity
            fee_ppm = info.get("fee_ppm", 0)
            peer_id = info.get("peer_id", "")[:16]

            state = database.get_channel_state(cid) if database else {}
            flow_state = state.get("state", "unknown") if state else "unknown"

            channel_info = {
                "scid": cid[:20],
                "peer": peer_id,
                "local_pct": round(ratio * 100, 1),
                "fee_ppm": fee_ppm,
                "flow_state": flow_state
            }

            if cid in active_channels:
                result["channels"]["active_jobs"].append(channel_info)
            elif ratio < cfg.low_liquidity_threshold:
                channel_info["reason"] = "low local balance"
                if flow_state == "sink":
                    channel_info["skip_reason"] = "SINK - filling naturally"
                result["channels"]["depleted"].append(channel_info)
            elif ratio > cfg.high_liquidity_threshold:
                channel_info["reason"] = "high local balance"
                result["channels"]["source"].append(channel_info)

        if not result["channels"]["depleted"]:
            result["rejection_reasons"].append(
                f"No depleted channels (none below {cfg.low_liquidity_threshold*100}% local balance)"
            )
        if not result["channels"]["source"]:
            result["rejection_reasons"].append(
                f"No source channels (none above {cfg.high_liquidity_threshold*100}% local balance)"
            )

    except Exception as e:
        result["channels"]["error"] = str(e)

    return result


@plugin.method("revenue-fee-debug")
def revenue_fee_debug(plugin: Plugin) -> Dict[str, Any]:
    """
    Diagnostic command to understand why fee adjustments may not be happening.

    Shows:
    - Hill Climb state for each channel (sleeping, last_update, forward count)
    - Why each channel was skipped in the last cycle
    - Dynamic window status
    - Hysteresis/sleep status

    Usage: lightning-cli revenue-fee-debug
    """
    if database is None or fee_controller is None:
        return {"error": "Plugin not fully initialized"}

    # Import fee controller constants for accurate debug output
    from modules.fee_controller import HillClimbingFeeController
    min_obs_hours = HillClimbingFeeController.MIN_OBSERVATION_HOURS
    min_forwards = HillClimbingFeeController.MIN_FORWARDS_FOR_SIGNAL
    max_obs_hours = HillClimbingFeeController.MAX_OBSERVATION_HOURS
    enable_dyn_windows = HillClimbingFeeController.ENABLE_DYNAMIC_WINDOWS

    now = int(time.time())
    result = {
        "timestamp": now,
        "config": {
            "fee_interval_seconds": config.fee_interval if config else 1800,
            "min_observation_hours": min_obs_hours,
            "min_forwards_for_signal": min_forwards,
            "max_observation_hours": max_obs_hours,
            "enable_dynamic_windows": enable_dyn_windows
        },
        "channels": [],
        "summary": {
            "total": 0,
            "sleeping": 0,
            "waiting_time": 0,
            "waiting_forwards": 0,
            "ready": 0
        }
    }

    # Get all fee strategy states
    fee_states = database.get_all_fee_strategy_states()
    channel_states = database.get_all_channel_states()

    # Create lookup for channel states
    state_lookup = {s.get("channel_id"): s for s in channel_states}

    for fs in fee_states:
        channel_id = fs.get("channel_id", "unknown")
        is_sleeping = fs.get("is_sleeping", 0)
        sleep_until = fs.get("sleep_until", 0)
        last_update = fs.get("last_update", 0)
        forward_count = fs.get("forward_count_since_update", 0)
        last_broadcast_fee = fs.get("last_broadcast_fee_ppm", 0)
        last_revenue_rate = fs.get("last_revenue_rate", 0.0)

        hours_since_update = (now - last_update) / 3600.0 if last_update > 0 else 0.0

        # Determine skip reason
        # With ENABLE_DYNAMIC_WINDOWS=True and OR logic:
        # Channel is ready if EITHER time >= min_obs_hours OR forwards >= min_forwards
        skip_reason = None
        status = "ready"

        time_ok = hours_since_update >= min_obs_hours
        forwards_ok = forward_count >= min_forwards

        if is_sleeping:
            mins_until_wake = (sleep_until - now) // 60
            skip_reason = f"SLEEPING (wake in {mins_until_wake} min)"
            status = "sleeping"
            result["summary"]["sleeping"] += 1
        elif enable_dyn_windows and (time_ok or forwards_ok):
            # OR logic: either condition met = ready
            status = "ready"
            result["summary"]["ready"] += 1
        elif not enable_dyn_windows and time_ok:
            # Legacy: time-only check
            status = "ready"
            result["summary"]["ready"] += 1
        elif not time_ok and not forwards_ok:
            # Neither condition met - waiting for either
            skip_reason = f"WAITING ({forward_count}/{min_forwards} fwds, {hours_since_update:.1f}/{min_obs_hours}h)"
            status = "waiting"
            result["summary"]["waiting_time"] += 1
        else:
            status = "ready"
            result["summary"]["ready"] += 1

        chan_state = state_lookup.get(channel_id, {})

        result["channels"].append({
            "channel_id": channel_id[:12] + "..." if len(channel_id) > 12 else channel_id,
            "status": status,
            "skip_reason": skip_reason,
            "is_sleeping": bool(is_sleeping),
            "hours_since_update": round(hours_since_update, 2),
            "forwards_since_update": forward_count,
            "last_broadcast_fee_ppm": last_broadcast_fee,
            "last_revenue_rate": round(last_revenue_rate, 2),
            "flow_state": chan_state.get("state", "unknown")
        })
        result["summary"]["total"] += 1

    return result


@plugin.method("revenue-analyze")
def revenue_analyze(plugin: Plugin, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run flow analysis on demand (optionally for a specific channel).

    Usage: lightning-cli revenue-analyze [channel_id]
    """
    if flow_analyzer is None:
        return {"error": "Plugin not fully initialized"}

    # L-22: Validate SCID format if provided
    if channel_id and not re.match(r'^\d+[x:]\d+[x:]\d+$', channel_id):
        return {"error": f"Invalid channel format: {channel_id}. Use SCID format (e.g., 123x456x789)."}

    if channel_id:
        result = flow_analyzer.analyze_channel(channel_id)
        return {"channel": channel_id, "analysis": result.to_dict() if result else None}
    else:
        run_flow_analysis()
        return {"status": "Flow analysis triggered"}


@plugin.method("revenue-wake-all")
def revenue_wake_all(plugin: Plugin) -> Dict[str, Any]:
    """
    Wake all sleeping channels immediately.

    Use this after changing fee_interval or when you need to force
    all channels to re-evaluate their fees on the next cycle.

    Usage: lightning-cli revenue-wake-all
    """
    if fee_controller is None:
        return {"error": "Plugin not fully initialized"}

    woken = fee_controller.wake_all_sleeping_channels()
    return {
        "status": "ok",
        "channels_woken": woken,
        "message": f"Woke {woken} sleeping channel(s). They will be evaluated on the next fee cycle."
    }


@plugin.method("revenue-capacity-report")
def revenue_capacity_report(plugin: Plugin, **kwargs):
    """
    Generate a strategic capital redeployment report.
    
    Identifies "Winner" channels for capital injection (Splice-In)
    and "Loser" channels for capital extraction (Splice-Out/Close).
    """
    if capacity_planner is None:
        raise RpcError("revenue-capacity-report", {}, "Capacity planner not initialized")
        
    return capacity_planner.generate_report()


@plugin.method("revenue-set-fee")
def revenue_set_fee(plugin: Plugin, channel_id: str, fee_ppm: int, force: bool = False) -> Dict[str, Any]:
    """
    Manually set fee for a channel (with clboss unmanage).

    Usage: lightning-cli revenue-set-fee channel_id fee_ppm [force=false]
    """
    if fee_controller is None or config is None:
        return {"error": "Plugin not fully initialized"}

    # MAJOR-09 FIX: Rate limit force operations
    if force:
        allowed, msg = force_rate_limiter.check_rate_limit("revenue-set-fee")
        if not allowed:
            return {"status": "error", "error": msg}

    # 1. Validation
    try:
        fee_ppm = int(fee_ppm)
        if fee_ppm < 0:
            return {"status": "error", "error": "fee_ppm must be non-negative"}
    except ValueError:
        return {"status": "error", "error": "fee_ppm must be an integer"}

    # SCID or PeerID format check
    if not (re.match(r'^\d+[x:]\d+[x:]\d+$', channel_id) or len(channel_id) == 66):
        return {"status": "error", "error": "Invalid channel_id or node_id format"}

    # 2. Force Gates
    if not force:
        if fee_ppm < config.min_fee_ppm or fee_ppm > config.max_fee_ppm:
            return {
                "status": "error", 
                "error": f"Fee {fee_ppm} is outside configured range [{config.min_fee_ppm}, {config.max_fee_ppm}]. Use force=true to override."
            }
    
    try:
        result = fee_controller.set_channel_fee(channel_id, fee_ppm, manual=True, enforce_limits=(not force))
        applied_fee = result.get("fee_ppm", fee_ppm)
        return {"status": "success", "channel": channel_id, "new_fee_ppm": applied_fee, **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-fee-anchor")
def revenue_fee_anchor(plugin: Plugin,
                       action: str,
                       channel_id: str = "",
                       target_fee_ppm: int = 0,
                       confidence: float = 1.0,
                       base_weight: float = 0.7,
                       ttl_hours: int = 24,
                       reason: str = "") -> Dict[str, Any]:
    """
    Manage advisor fee anchors (soft fee targets with decaying weight).

    Usage:
      lightning-cli revenue-fee-anchor action=set channel_id=X target_fee_ppm=N [confidence=0.8] [base_weight=0.7] [ttl_hours=24] [reason="..."]
      lightning-cli revenue-fee-anchor action=list
      lightning-cli revenue-fee-anchor action=get channel_id=X
      lightning-cli revenue-fee-anchor action=clear channel_id=X
      lightning-cli revenue-fee-anchor action=clear-all
    """
    if fee_controller is None:
        return {"error": "Plugin not fully initialized"}

    if action == "set":
        if not channel_id:
            return {"status": "error", "error": "channel_id is required for set"}
        try:
            target_fee_ppm = int(target_fee_ppm)
        except (ValueError, TypeError):
            return {"status": "error", "error": "target_fee_ppm must be an integer"}
        if target_fee_ppm < 0:
            return {"status": "error", "error": "target_fee_ppm must be non-negative"}

        # SCID or PeerID format check
        if not (re.match(r'^\d+[x:]\d+[x:]\d+$', channel_id) or len(channel_id) == 66):
            return {"status": "error", "error": "Invalid channel_id format"}

        try:
            ttl_seconds = int(ttl_hours) * 3600
        except (ValueError, TypeError):
            return {"status": "error", "error": "ttl_hours must be an integer"}
        if ttl_seconds < 1 or ttl_seconds > 604800:
            return {"status": "error", "error": "ttl must be between 1 second and 7 days"}

        try:
            base_weight = float(base_weight)
        except (ValueError, TypeError):
            return {"status": "error", "error": "base_weight must be a number"}
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            return {"status": "error", "error": "confidence must be a number"}

        return fee_controller.set_fee_anchor(
            channel_id=channel_id,
            target_fee_ppm=target_fee_ppm,
            base_weight=base_weight,
            confidence=confidence,
            ttl_seconds=ttl_seconds,
            reason=reason,
        )

    elif action == "list":
        anchors = fee_controller.list_fee_anchors()
        return {"status": "success", "anchors": anchors, "count": len(anchors)}

    elif action == "get":
        if not channel_id:
            return {"status": "error", "error": "channel_id is required for get"}
        anchor = fee_controller.get_fee_anchor(channel_id)
        if anchor is None:
            return {"status": "success", "anchor": None, "message": "No active anchor"}
        return {"status": "success", "anchor": anchor}

    elif action == "clear":
        if not channel_id:
            return {"status": "error", "error": "channel_id is required for clear"}
        return fee_controller.clear_fee_anchor(channel_id)

    elif action == "clear-all":
        return fee_controller.clear_all_fee_anchors()

    else:
        return {"status": "error", "error": f"Unknown action: {action}. Use set/list/get/clear/clear-all"}


@plugin.method("revenue-rebalance")
def revenue_rebalance(plugin: Plugin,
                      from_channel: str,
                      to_channel: str,
                      amount_sats: int,
                      max_fee_sats: Optional[int] = None,
                      force: bool = False) -> Dict[str, Any]:
    """
    Manually trigger a rebalance with profit/budget constraints.

    Usage: lightning-cli revenue-rebalance from_channel to_channel amount_sats [max_fee_sats] [force=false]
    """
    if rebalancer is None:
        return {"error": "Plugin not fully initialized"}

    # MAJOR-09 FIX: Rate limit force operations
    if force:
        allowed, msg = force_rate_limiter.check_rate_limit("revenue-rebalance")
        if not allowed:
            return {"status": "error", "error": msg}

    if config and not config.sling_available:
        return {"error": "Rebalancing disabled: sling plugin not found. Install cln-sling to enable."}

    # L-21: Validate SCID format
    for cid in (from_channel, to_channel):
        if not re.match(r'^\d+[x:]\d+[x:]\d+$', cid):
            return {"status": "error", "error": f"Invalid channel format for {cid}. Use SCID format (e.g., 123x456x789)."}

    # 1. Validation
    try:
        amount_sats = int(amount_sats)
        if amount_sats < 1:
            return {"status": "error", "error": "amount_sats must be at least 1"}
    except ValueError:
        return {"status": "error", "error": "amount_sats must be an integer"}
        
    if max_fee_sats is not None:
        try:
            max_fee_sats = int(max_fee_sats)
            if max_fee_sats < 0:
                return {"status": "error", "error": "max_fee_sats must be non-negative"}
        except ValueError:
            return {"status": "error", "error": "max_fee_sats must be an integer or null"}

    # Basic SCID format check
    for cid in (from_channel, to_channel):
        if not (":" in cid or "x" in cid):
            return {"status": "error", "error": f"Invalid channel format for {cid}. Use SCID format."}

    try:
        result = rebalancer.manual_rebalance(from_channel, to_channel, amount_sats, max_fee_sats, force=force)
        # Check if manual_rebalance returned an error dict
        if "error" in result:
            result_copy = {k: v for k, v in result.items() if k != "status"}
            return {"status": "error", **result_copy}
        # Check the success field from execute_rebalance
        if result.get("success") is False:
            result_copy = {k: v for k, v in result.items() if k != "status"}
            return {"status": "error", **result_copy}
        result_copy = {k: v for k, v in result.items() if k != "status"}
        return {"status": "success", **result_copy}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-clboss-status")
def revenue_clboss_status(plugin: Plugin) -> Dict[str, Any]:
    """
    Check which channels are currently unmanaged from clboss.
    
    Usage: lightning-cli revenue-clboss-status
    """
    if clboss_manager is None:
        return {"error": "Plugin not fully initialized"}
    
    return clboss_manager.get_unmanaged_status()


@plugin.method("revenue-profitability")
def revenue_profitability(plugin: Plugin, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get channel profitability analysis.
    
    Shows each channel's:
    - Total costs (opening + rebalancing)
    - Total revenue (routing fees)
    - Net profit/loss
    - ROI percentage
    - Profitability classification (profitable, break_even, underwater, zombie)
    
    Usage: lightning-cli revenue-profitability [channel_id]
    """
    if profitability_analyzer is None:
        return {"error": "Plugin not fully initialized"}
    
    try:
        if channel_id:
            # Analyze single channel
            result = profitability_analyzer.analyze_channel(channel_id)
            if result:
                # Calculate flow profile
                outbound_count = result.revenue.forward_count
                inbound_count = result.revenue.sourced_forward_count
                total_count = outbound_count + inbound_count

                if total_count == 0:
                    flow_profile = "inactive"
                    inbound_outbound_ratio = 0.0
                elif outbound_count == 0:
                    flow_profile = "inbound_only"
                    inbound_outbound_ratio = float('inf')
                elif inbound_count == 0:
                    flow_profile = "outbound_only"
                    inbound_outbound_ratio = 0.0
                else:
                    inbound_outbound_ratio = round(inbound_count / outbound_count, 2)
                    if inbound_outbound_ratio > 3.0:
                        flow_profile = "inbound_dominant"
                    elif inbound_outbound_ratio < 0.33:
                        flow_profile = "outbound_dominant"
                    else:
                        flow_profile = "balanced"

                return {
                    "channel_id": channel_id,
                    "profitability": {
                        "total_costs_sats": result.costs.total_cost_sats,
                        "total_contribution_sats": result.revenue.total_contribution_sats,
                        "net_profit_sats": result.net_profit_sats,
                        "roi_percentage": round(result.roi_percent, 2),
                        "profitability_class": result.classification.value,
                        "days_active": result.days_open,
                        "fee_multiplier": profitability_analyzer.get_fee_multiplier(channel_id),
                        # Outbound flow (channel as exit - we earn fees)
                        "outbound_flow": {
                            "payment_count": outbound_count,
                            "volume_sats": result.revenue.volume_routed_sats,
                            "revenue_earned_sats": result.revenue.fees_earned_sats
                        },
                        # Inbound flow (channel as entry - generates revenue elsewhere)
                        "inbound_flow": {
                            "payment_count": inbound_count,
                            "volume_sats": result.revenue.sourced_volume_sats,
                            "contribution_to_other_channels_sats": result.revenue.sourced_fee_contribution_sats
                        },
                        # Flow profile summary
                        "flow_profile": flow_profile,
                        "inbound_outbound_ratio": inbound_outbound_ratio if inbound_outbound_ratio != float('inf') else "infinite",
                        # Legacy fields for backward compatibility
                        "total_revenue_sats": result.revenue.fees_earned_sats,
                        "volume_routed_sats": result.revenue.volume_routed_sats,
                        "forward_count": result.revenue.forward_count
                    }
                }
            else:
                return {"channel_id": channel_id, "error": "No data available"}
        else:
            # Analyze all channels
            all_results = profitability_analyzer.analyze_all_channels(force=True)

            # Group by profitability class
            summary = {
                "profitable": [],
                "break_even": [],
                "underwater": [],
                "stagnant_candidate": [],
                "zombie": []
            }
            # Track flow profiles
            flow_profiles = {
                "inbound_dominant": [],
                "outbound_dominant": [],
                "balanced": [],
                "inbound_only": [],
                "outbound_only": [],
                "inactive": []
            }
            total_profit = 0
            total_revenue = 0
            total_contribution = 0
            total_costs = 0

            for ch_id, result in all_results.items():
                # Calculate flow profile
                outbound_count = result.revenue.forward_count
                inbound_count = result.revenue.sourced_forward_count

                if outbound_count + inbound_count == 0:
                    flow_profile = "inactive"
                elif outbound_count == 0:
                    flow_profile = "inbound_only"
                elif inbound_count == 0:
                    flow_profile = "outbound_only"
                else:
                    ratio = inbound_count / outbound_count
                    if ratio > 3.0:
                        flow_profile = "inbound_dominant"
                    elif ratio < 0.33:
                        flow_profile = "outbound_dominant"
                    else:
                        flow_profile = "balanced"

                channel_summary = {
                    "channel_id": ch_id,
                    "net_profit_sats": result.net_profit_sats,
                    "roi_percentage": round(result.roi_percent, 2),
                    "days_active": result.days_open,
                    "flow_profile": flow_profile,
                    "forward_count": outbound_count,
                    "sourced_forward_count": inbound_count,
                    "fees_earned_sats": result.revenue.fees_earned_sats,
                    "volume_routed_sats": result.revenue.volume_routed_sats,
                }
                summary[result.classification.value].append(channel_summary)
                flow_profiles[flow_profile].append(ch_id)
                total_profit += result.net_profit_sats
                total_revenue += result.revenue.fees_earned_sats
                total_contribution += result.revenue.total_contribution_sats
                total_costs += result.costs.total_cost_sats

            return {
                "summary": {
                    "total_channels": len(all_results),
                    "profitable_count": len(summary["profitable"]),
                    "break_even_count": len(summary["break_even"]),
                    "underwater_count": len(summary["underwater"]),
                    "stagnant_candidate_count": len(summary["stagnant_candidate"]),
                    "zombie_count": len(summary["zombie"]),
                    "total_profit_sats": total_profit,
                    "total_revenue_sats": total_revenue,
                    "total_contribution_sats": total_contribution,
                    "total_costs_sats": total_costs,
                    "overall_roi_pct": round((total_profit / total_costs * 100) if total_costs > 0 else 0, 2),
                    # Flow profile distribution
                    "flow_profiles": {
                        "inbound_dominant_count": len(flow_profiles["inbound_dominant"]),
                        "outbound_dominant_count": len(flow_profiles["outbound_dominant"]),
                        "balanced_count": len(flow_profiles["balanced"]),
                        "inactive_count": len(flow_profiles["inactive"])
                    }
                },
                "channels_by_class": summary
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-history")
def revenue_history(plugin: Plugin) -> Dict[str, Any]:
    """
    Get lifetime financial history including closed channels.
    
    Reports aggregate financial performance since the plugin was installed,
    including data from channels that have since been closed. This provides
    a true "Lifetime P&L" view.
    
    Returns:
        - Lifetime Revenue (total routing fees earned)
        - Lifetime Costs (opening fees + rebalancing fees)
        - Lifetime Net Profit (revenue - costs)
        - Lifetime ROI percentage
        - Total number of forwards processed
    
    Usage: lightning-cli revenue-history
    """
    if profitability_analyzer is None:
        return {"error": "Plugin not initialized"}
    
    try:
        return profitability_analyzer.get_lifetime_report()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-remanage")
def revenue_remanage(plugin: Plugin, peer_id: str, tag: Optional[str] = None) -> Dict[str, Any]:
    """
    Re-enable clboss management for a peer (release our override).
    
    Usage: lightning-cli revenue-remanage peer_id [tag]
    """
    if clboss_manager is None:
        return {"error": "Plugin not fully initialized"}

    # L-22: Validate peer_id format (66-char hex pubkey)
    if not re.match(r'^[0-9a-fA-F]{66}$', peer_id):
        return {"error": f"Invalid peer_id format: expected 66-character hex pubkey"}

    try:
        result = clboss_manager.remanage(peer_id, tag)
        return {"status": "success", "peer_id": peer_id, **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-ignore")
def revenue_ignore(plugin: Plugin, peer_id: str, reason: str = "manual") -> Dict[str, Any]:
    """
    DEPRECATED: Use 'revenue-policy set <peer_id> strategy=passive rebalance=disabled' instead.
    
    Stop cl-revenue-ops from managing this peer (fees or rebalancing).
    
    Usage: lightning-cli revenue-ignore peer_id [reason]
    """
    if policy_manager is None:
        return {"error": "Plugin not initialized"}
    
    plugin.log(
        f"DEPRECATED: revenue-ignore is deprecated. Use 'revenue-policy set {peer_id} "
        f"strategy=passive rebalance=disabled' instead.",
        level='warn'
    )
    
    # Map to new policy system: passive strategy + disabled rebalancing
    try:
        policy = policy_manager.set_policy(
            peer_id=peer_id,
            strategy="passive",
            rebalance_mode="disabled",
            tags=["ignored", reason] if reason != "ignored" else ["ignored"]
        )
        return {
            "status": "success",
            "action": "ignore",
            "peer_id": peer_id,
            "reason": reason,
            "message": f"Peer {peer_id} set to passive strategy with rebalancing disabled.",
            "warning": "DEPRECATED: Use 'revenue-policy set' instead."
        }
    except ValueError as e:
        return {"status": "error", "error": str(e)}


@plugin.method("revenue-unignore")
def revenue_unignore(plugin: Plugin, peer_id: str) -> Dict[str, Any]:
    """
    DEPRECATED: Use 'revenue-policy delete <peer_id>' instead.
    
    Resume cl-revenue-ops management for this peer.
    
    Usage: lightning-cli revenue-unignore peer_id
    """
    if policy_manager is None:
        return {"error": "Plugin not initialized"}
    
    plugin.log(
        f"DEPRECATED: revenue-unignore is deprecated. Use 'revenue-policy delete {peer_id}' instead.",
        level='warn'
    )
    
    # Map to new policy system: delete policy (reverts to defaults)
    deleted = policy_manager.delete_policy(peer_id)
    return {
        "status": "success",
        "action": "unignore",
        "peer_id": peer_id,
        "message": f"Peer {peer_id} reverted to default policy (dynamic strategy, rebalancing enabled).",
        "warning": "DEPRECATED: Use 'revenue-policy delete' instead."
    }


@plugin.method("revenue-list-ignored")
def revenue_list_ignored(plugin: Plugin) -> Dict[str, Any]:
    """
    DEPRECATED: Use 'revenue-policy list' or 'revenue-report policies' instead.
    
    List all peers currently ignored by cl-revenue-ops.
    """
    if policy_manager is None:
        return {"error": "Plugin not initialized"}
    
    plugin.log(
        "DEPRECATED: revenue-list-ignored is deprecated. Use 'revenue-policy list' instead.",
        level='warn'
    )
    
    # Find all peers with passive strategy and disabled rebalancing (equivalent to "ignored")
    all_policies = policy_manager.get_all_policies()
    ignored = []
    for p in all_policies:
        if p.strategy == FeeStrategy.PASSIVE and p.rebalance_mode == RebalanceMode.DISABLED:
            ignored.append({
                "peer_id": p.peer_id,
                "reason": next((t for t in p.tags if t != "ignored"), "manual"),
                "ignored_at": p.updated_at
            })
    
    return {
        "ignored_peers": ignored,
        "count": len(ignored),
        "warning": "DEPRECATED: Use 'revenue-policy list' instead."
    }



# =============================================================================
# POLICY MANAGEMENT (v1.4: Policy-Driven Architecture)
# =============================================================================

@plugin.method("revenue-policy")
def revenue_policy(plugin: Plugin, action: str, peer_id: str = None,
                   strategy: str = None, rebalance: str = None,
                   fee_ppm: int = None, tag: str = None,
                   fee_multiplier_min: float = None,
                   fee_multiplier_max: float = None,
                   expires_in_hours: int = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Manage peer-level fee and rebalance policies (v1.4 API).

    Usage:
      lightning-cli revenue-policy list                           # List all policies
      lightning-cli revenue-policy get <peer_id>                  # Get policy for peer
      lightning-cli revenue-policy set <peer_id> [options]        # Set/update policy
      lightning-cli revenue-policy delete <peer_id>               # Delete policy (revert to defaults)
      lightning-cli revenue-policy tag <peer_id> <tag>            # Add tag to peer
      lightning-cli revenue-policy untag <peer_id> <tag>          # Remove tag from peer
      lightning-cli revenue-policy find <tag>                     # Find peers by tag
      lightning-cli revenue-policy changes [since=<timestamp>]    # Get policy changes (cl-hive)

    Options for 'set':
      strategy=dynamic|static|hive|passive   Fee control strategy
      rebalance=enabled|disabled|source_only|sink_only   Rebalance mode
      fee_ppm=N   Target fee for static strategy (required if strategy=static)
      fee_multiplier_min=X.Y   Dynamic fee autoband floor multiplier (uses fee_ppm_target as anchor)
      fee_multiplier_max=X.Y   Dynamic fee autoband ceiling multiplier (uses fee_ppm_target as anchor)
      expires_in_hours=N       Optional auto-expiry for policy (revert to defaults)

    Strategies:
      dynamic  - Hill Climbing + Scarcity Pricing (default)
      static   - Fixed fee (requires fee_ppm)
      hive     - Zero/low fee for fleet members (cl-hive integration)
      passive  - Do not manage (CLBOSS/manual control)

    Rebalance Modes:
      enabled     - Full rebalancing allowed (default)
      disabled    - No rebalancing (equivalent to old 'ignore')
      source_only - Can drain from, cannot fill
      sink_only   - Can fill, cannot drain from

    Options for 'changes' (cl-hive integration):
      since=<timestamp>   Unix timestamp. Returns policies changed after this time.
                          If omitted, returns all policies.

    Options for 'batch' (cl-hive integration):
      updates='[...]'     JSON array of policy updates. Each entry has:
                          peer_id, strategy, rebalance_mode, fee_ppm_target, tags
                          Bypasses rate limiting for bulk hive fleet updates.

    Examples:
      lightning-cli revenue-policy set 02abc... strategy=static fee_ppm=500
      lightning-cli revenue-policy set 02abc... strategy=passive rebalance=disabled
      lightning-cli revenue-policy tag 02abc... whale
      lightning-cli -k revenue-policy action=changes since=1704067200
    """
    if policy_manager is None:
        return {"error": "Plugin not initialized"}
    
    try:
        if action == "list":
            policies = policy_manager.get_all_policies()
            return {
                "policies": [p.to_dict() for p in policies],
                "count": len(policies)
            }
        
        elif action == "get":
            if not peer_id:
                return {"error": "Usage: revenue-policy get <peer_id>"}
            # L-22: Validate peer_id format
            if not re.match(r'^[0-9a-fA-F]{66}$', peer_id):
                return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}
            policy = policy_manager.get_policy(peer_id)
            return {"policy": policy.to_dict()}
        
        elif action == "set":
            if not peer_id:
                return {"error": "Usage: revenue-policy set <peer_id> [strategy=X] [rebalance=X] [fee_ppm=N]"}
            if not re.match(r'^[0-9a-fA-F]{66}$', peer_id):
                return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}

            def _parse_optional_float(value, field_name: str) -> Optional[float]:
                if value is None:
                    return None
                if isinstance(value, str):
                    s = value.strip().lower()
                    if s in ('', 'null', 'none'):
                        return None
                    value = s
                try:
                    return float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid {field_name}: expected float")

            def _parse_optional_int(value, field_name: str) -> Optional[int]:
                if value is None:
                    return None
                if isinstance(value, str):
                    s = value.strip().lower()
                    if s in ('', 'null', 'none'):
                        return None
                    value = s
                try:
                    return int(value)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid {field_name}: expected integer")

            mult_min_arg = fee_multiplier_min if fee_multiplier_min is not None else kwargs.get('fee_multiplier_min')
            mult_max_arg = fee_multiplier_max if fee_multiplier_max is not None else kwargs.get('fee_multiplier_max')
            expires_arg = expires_in_hours if expires_in_hours is not None else kwargs.get('expires_in_hours')

            # Set policy with provided options
            policy = policy_manager.set_policy(
                peer_id=peer_id,
                strategy=strategy,
                rebalance_mode=rebalance,
                fee_ppm_target=fee_ppm,
                fee_multiplier_min=_parse_optional_float(mult_min_arg, 'fee_multiplier_min'),
                fee_multiplier_max=_parse_optional_float(mult_max_arg, 'fee_multiplier_max'),
                expires_in_hours=_parse_optional_int(expires_arg, 'expires_in_hours')
            )
            
            return {
                "status": "success",
                "policy": policy.to_dict(),
                "message": f"Policy updated for peer {peer_id[:12]}..."
            }
        
        elif action == "delete":
            if not peer_id:
                return {"error": "Usage: revenue-policy delete <peer_id>"}
            if not re.match(r'^[0-9a-fA-F]{66}$', peer_id):
                return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}
            deleted = policy_manager.delete_policy(peer_id)
            if deleted:
                return {
                    "status": "success",
                    "peer_id": peer_id,
                    "message": "Policy deleted, peer reverted to defaults (dynamic strategy, rebalancing enabled)"
                }
            return {"status": "noop", "message": "No policy existed for this peer"}
        
        elif action == "tag":
            if not peer_id or not tag:
                return {"error": "Usage: revenue-policy tag <peer_id> <tag>"}
            if not re.match(r'^[0-9a-fA-F]{66}$', peer_id):
                return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}
            policy = policy_manager.add_tag(peer_id, tag)
            return {
                "status": "success",
                "peer_id": peer_id,
                "tags": policy.tags
            }
        
        elif action == "untag":
            if not peer_id or not tag:
                return {"error": "Usage: revenue-policy untag <peer_id> <tag>"}
            if not re.match(r'^[0-9a-fA-F]{66}$', peer_id):
                return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}
            policy = policy_manager.remove_tag(peer_id, tag)
            return {
                "status": "success",
                "peer_id": peer_id,
                "tags": policy.tags
            }
        
        elif action == "find":
            if not tag:
                return {"error": "Usage: revenue-policy find <tag>"}
            policies = policy_manager.get_peers_by_tag(tag)
            return {
                "peers": [p.to_dict() for p in policies],
                "count": len(policies),
                "tag": tag
            }

        elif action == "changes":
            # cl-hive integration: Get policy changes since timestamp
            # Usage: revenue-policy changes [since=<timestamp>]
            since = kwargs.get('since', 0)
            try:
                since = int(since) if since else 0
            except (ValueError, TypeError):
                return {"error": "Invalid 'since' timestamp. Must be a Unix timestamp."}

            changes = policy_manager.get_policy_changes_since(since)
            last_change = policy_manager.get_last_policy_change_timestamp()
            return {
                "changes": changes,
                "count": len(changes),
                "since": since,
                "last_change_timestamp": last_change
            }

        elif action == "batch":
            # cl-hive integration: Bulk policy updates (bypasses rate limiting)
            # Usage: revenue-policy batch updates='[{"peer_id": "...", "strategy": "hive"}, ...]'
            updates_json = kwargs.get('updates', '[]')
            try:
                import json
                if isinstance(updates_json, str):
                    updates = json.loads(updates_json)
                else:
                    updates = updates_json
                if not isinstance(updates, list):
                    return {"error": "updates must be a JSON array"}
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON in updates: {e}"}

            try:
                policies = policy_manager.set_policies_batch(updates)
                plugin.log(f"Batch policy update: {len(policies)} policies changed (rate limiting bypassed)", level='info')
                return {
                    "status": "success",
                    "updated": len(policies),
                    "policies": [p.to_dict() for p in policies],
                    "message": f"Batch updated {len(policies)} policies"
                }
            except ValueError as e:
                return {"status": "error", "error": str(e)}

        else:
            return {"error": f"Unknown action: {action}. Use 'list', 'get', 'set', 'delete', 'tag', 'untag', 'find', 'changes', or 'batch'"}
    
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error: {e}"}


@plugin.method("revenue-report")
def revenue_report(plugin: Plugin, report_type: str = "summary",
                   peer_id: str = None) -> Dict[str, Any]:
    """
    Generate reports for node financial health and peer status (v1.4 API).

    Usage:
      lightning-cli revenue-report                    # Summary report
      lightning-cli revenue-report summary            # Same as above
      lightning-cli revenue-report peer <peer_id>    # Detailed peer report
      lightning-cli revenue-report hive              # List hive fleet members
      lightning-cli revenue-report policies          # Policy distribution stats
      lightning-cli revenue-report costs             # Closure/splice cost history (cl-hive)

    Report Types:
      summary   - Overall node P&L, active channels, warnings
      peer      - Specific peer metrics (profitability, flow, policy)
      hive      - List of peers with HIVE strategy (for cl-hive)
      policies  - Statistics on policy distribution
      costs     - Closure/splice costs for capacity planning (cl-hive)
    """
    if database is None or policy_manager is None:
        return {"error": "Plugin not initialized"}
    
    try:
        if report_type == "summary":
            # Basic summary - expand with Phase 8 P&L when available
            all_policies = policy_manager.get_all_policies()
            
            strategy_counts = {}
            rebalance_counts = {}
            for p in all_policies:
                s = p.strategy.value
                r = p.rebalance_mode.value
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
                rebalance_counts[r] = rebalance_counts.get(r, 0) + 1
            
            return {
                "type": "summary",
                "policies": {
                    "total": len(all_policies),
                    "by_strategy": strategy_counts,
                    "by_rebalance_mode": rebalance_counts
                },
                "generated_at": int(time.time())
            }
        
        elif report_type == "peer":
            if not peer_id:
                return {"error": "Usage: revenue-report peer <peer_id>"}
            
            # Get policy
            policy = policy_manager.get_policy(peer_id)
            
            # Get profitability if available
            prof_data = None
            if profitability_analyzer:
                prof_data = profitability_analyzer.get_profitability_by_peer(peer_id)
            
            # Get flow state
            flow_state = None
            if database:
                states = database.get_all_channel_states()
                for s in states:
                    if s.get("peer_id") == peer_id:
                        flow_state = s
                        break
            
            return {
                "type": "peer",
                "peer_id": peer_id,
                "policy": policy.to_dict(),
                "profitability": prof_data.to_dict() if prof_data else None,
                "flow_state": flow_state
            }
        
        elif report_type == "hive":
            # List all hive members (for cl-hive integration)
            hive_peers = policy_manager.get_peers_by_strategy(FeeStrategy.HIVE)
            return {
                "type": "hive",
                "peers": [p.to_dict() for p in hive_peers],
                "count": len(hive_peers)
            }
        
        elif report_type == "policies":
            all_policies = policy_manager.get_all_policies()

            by_strategy = {}
            by_mode = {}
            by_tag = {}

            for p in all_policies:
                # Count by strategy
                s = p.strategy.value
                by_strategy[s] = by_strategy.get(s, 0) + 1

                # Count by mode
                m = p.rebalance_mode.value
                by_mode[m] = by_mode.get(m, 0) + 1

                # Count by tag
                for t in p.tags:
                    by_tag[t] = by_tag.get(t, 0) + 1

            return {
                "type": "policies",
                "total": len(all_policies),
                "by_strategy": by_strategy,
                "by_rebalance_mode": by_mode,
                "by_tag": by_tag
            }

        elif report_type == "costs":
            # cl-hive integration: Expose closure/splice/swap costs for capacity planning
            now = int(time.time())
            day_ago = now - 86400
            week_ago = now - (7 * 86400)
            month_ago = now - (30 * 86400)

            # Get historical costs
            closure_costs_day = database.get_closure_costs_since(day_ago)
            closure_costs_week = database.get_closure_costs_since(week_ago)
            closure_costs_month = database.get_closure_costs_since(month_ago)
            closure_costs_total = database.get_total_closure_costs()

            splice_costs_day = database.get_splice_costs_since(day_ago)
            splice_costs_week = database.get_splice_costs_since(week_ago)
            splice_costs_month = database.get_splice_costs_since(month_ago)
            splice_costs_total = database.get_total_splice_costs()

            # Get splice summary for detailed breakdown
            splice_summary = database.get_splice_summary()

            # Include default chain cost estimates
            from modules.config import ChainCostDefaults
            estimated_costs = {
                "channel_open_sats": ChainCostDefaults.CHANNEL_OPEN_COST_SATS,
                "channel_close_sats": ChainCostDefaults.CHANNEL_CLOSE_COST_SATS,
                "splice_sats": ChainCostDefaults.SPLICE_COST_SATS,
            }

            return {
                "type": "costs",
                "closure_costs": {
                    "last_24h_sats": closure_costs_day,
                    "last_7d_sats": closure_costs_week,
                    "last_30d_sats": closure_costs_month,
                    "total_sats": closure_costs_total
                },
                "splice_costs": {
                    "last_24h_sats": splice_costs_day,
                    "last_7d_sats": splice_costs_week,
                    "last_30d_sats": splice_costs_month,
                    "total_sats": splice_costs_total,
                    "summary": splice_summary
                },
                "estimated_defaults": estimated_costs,
                "generated_at": now
            }

        else:
            return {"error": f"Unknown report type: {report_type}. Use 'summary', 'peer', 'hive', 'policies', or 'costs'"}
    
    except Exception as e:
        return {"status": "error", "error": f"Report generation failed: {e}"}


@plugin.method("revenue-config")
def revenue_config(plugin: Plugin, action: str, key: str = None, value: str = None) -> Dict[str, Any]:
    """
    Get or set runtime configuration (Phase 7: Dynamic Runtime Configuration).
    
    Usage:
      lightning-cli revenue-config get           # Get all config
      lightning-cli revenue-config get <key>     # Get specific key
      lightning-cli revenue-config set <key> <value>  # Set key
      lightning-cli revenue-config reset <key>   # Reset to default
      lightning-cli revenue-config list-mutable  # List changeable keys
    
    Examples:
      lightning-cli revenue-config get daily_budget_sats
      lightning-cli revenue-config set daily_budget_sats 10000
      lightning-cli revenue-config set enable_vegas_reflex false
    """
    if config is None or database is None:
        return {"error": "Plugin not initialized"}
    
    if action == "get":
        if key:
            if not hasattr(config, key) or key.startswith('_'):
                return {"error": f"Unknown config key: {key}"}
            return {
                "key": key,
                "value": getattr(config, key),
                "version": config._version
            }
        else:
            # Return all config as dict (exclude private fields)
            snapshot = config.snapshot()
            from dataclasses import asdict
            config_dict = asdict(snapshot)
            return {
                "config": config_dict,
                "version": config._version
            }
    
    elif action == "set":
        if not key or value is None:
            return {"error": "Usage: revenue-config set <key> <value>"}
        
        result = config.update_runtime(database, key, str(value))
        
        if result.get("status") == "success":
            plugin.log(
                f"CONFIG UPDATE: {key} changed from {result['old_value']} "
                f"to {result['new_value']} (v{result['version']})",
                level='info'
            )
        
        return result
    
    elif action == "reset":
        if not key:
            return {"error": "Usage: revenue-config reset <key>"}
        
        if database.delete_config_override(key):
            return {
                "status": "success",
                "message": f"Override for '{key}' removed. Restart plugin to apply default."
            }
        return {"error": f"No override found for '{key}'"}
    
    elif action == "list-mutable":
        from modules.config import CONFIG_FIELD_TYPES, IMMUTABLE_CONFIG_KEYS
        mutable = [k for k in CONFIG_FIELD_TYPES.keys() if k not in IMMUTABLE_CONFIG_KEYS]
        return {"mutable_keys": sorted(mutable), "count": len(mutable)}
    
    else:
        return {"error": f"Unknown action: {action}. Use 'get', 'set', 'reset', or 'list-mutable'"}


@plugin.method("revenue-dashboard")
def revenue_dashboard(plugin: Plugin, window_days: int = 30) -> Dict[str, Any]:
    """
    Phase 8: The Sovereign Dashboard - P&L Engine

    Returns financial health metrics and warnings about underperforming channels.

    Args:
        window_days: Number of days for P&L calculation (default: 30)

    Returns:
        {
            "financial_health": {
                "tlv_sats": int,           # Total Liquidating Value
                "net_profit_sats": int,    # Net profit for window
                "operating_margin_pct": float,  # (Net/Gross)*100
                "annualized_roc_pct": float     # Return on Capacity annualized
            },
            "period": {
                "window_days": int,
                "gross_revenue_sats": int,
                "opex_sats": int
            },
            "warnings": [str]  # Bleeder channel warnings
        }
    """
    if profitability_analyzer is None:
        return {"error": "Profitability analyzer not initialized"}

    if database is None:
        return {"error": "Database not initialized"}

    # L-23: Clamp window_days to sane range
    window_days = max(1, min(int(window_days), 365))

    try:
        # Get TLV (Total Liquidating Value)
        tlv_data = profitability_analyzer.get_tlv()
        tlv_sats = tlv_data.get("tlv_sats", 0)

        # Get P&L summary for the window
        pnl = profitability_analyzer.get_pnl_summary(window_days)

        # Get annualized ROC
        roc_data = profitability_analyzer.calculate_roc(window_days)
        annualized_roc_pct = roc_data.get("annualized_roc_pct", 0.0)

        # Identify bleeder channels
        bleeders = profitability_analyzer.identify_bleeders(window_days)

        # Build warnings list
        warnings = []
        for bleeder in bleeders:
            scid = bleeder.get("short_channel_id", "unknown")
            spent = bleeder.get("rebalance_cost_sats", 0)
            earned = bleeder.get("revenue_sats", 0)
            alias = bleeder.get("alias", "")
            if alias:
                warnings.append(
                    f"Channel {scid} ({alias}) is bleeding: "
                    f"Spent {spent} sats rebalancing, earned {earned} sats."
                )
            else:
                warnings.append(
                    f"Channel {scid} is bleeding: "
                    f"Spent {spent} sats rebalancing, earned {earned} sats."
                )

        return {
            "financial_health": {
                "tlv_sats": tlv_sats,
                "net_profit_sats": pnl.get("net_profit_sats", 0),
                "operating_margin_pct": pnl.get("operating_margin_pct", 0.0),
                "annualized_roc_pct": annualized_roc_pct
            },
            "period": {
                "window_days": window_days,
                "gross_revenue_sats": pnl.get("gross_revenue_sats", 0),
                "opex_sats": pnl.get("opex_sats", 0),
                "rebalance_cost_sats": pnl.get("rebalance_cost_sats", 0),
                "closure_cost_sats": pnl.get("closure_cost_sats", 0),
                "splice_cost_sats": pnl.get("splice_cost_sats", 0),
                "volume_sats": pnl.get("volume_sats", 0),
                "forward_count": pnl.get("forward_count", 0),
            },
            "warnings": warnings,
            "bleeder_count": len(bleeders)
        }
    except Exception as e:
        plugin.log(f"Error generating revenue dashboard: {e}", level='error')
        return {"error": str(e)}


# =============================================================================
# PORTFOLIO OPTIMIZATION (Mean-Variance)
# =============================================================================

@plugin.method("revenue-portfolio")
def revenue_portfolio(
    plugin: Plugin,
    risk_aversion: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze channel portfolio using Mean-Variance optimization.

    Treats channels as assets in a portfolio, optimizing liquidity allocation
    to maximize risk-adjusted returns (Sharpe ratio).

    Args:
        risk_aversion: Higher values penalize variance more (default: 1.0)
                       0.5 = aggressive, 1.0 = balanced, 2.0 = conservative

    Returns:
        Complete portfolio analysis including:
        - summary: Portfolio-level metrics (Sharpe, diversification, etc.)
        - channel_statistics: Per-channel return/variance stats
        - optimal_allocations: Recommended liquidity distribution
        - recommendations: Prioritized rebalance actions
        - correlations: Notable channel correlations
        - hedging_opportunities: Negatively correlated pairs
        - concentration_risks: Highly correlated pairs
    """
    # SEC-1: Validate and clamp parameters
    risk_aversion = max(0.0, min(10.0, float(risk_aversion)))

    global database, safe_plugin

    if database is None:
        return {"error": "Database not initialized"}

    if safe_plugin is None:
        return {"error": "Plugin not initialized"}

    try:
        from modules.portfolio_optimizer import PortfolioOptimizer

        # Get channel data
        channels = safe_plugin.rpc.listpeerchannels().get("channels", [])

        # Get forwards from local SQLite table (bucketed).
        # This avoids calling listforwards, which can be expensive on large nodes.
        import time
        now = int(time.time())
        cutoff = now - (14 * 86400)
        out_scids = [
            (ch.get("short_channel_id") or ch.get("channel_id"))
            for ch in channels
            if (ch.get("short_channel_id") or ch.get("channel_id"))
        ]
        forwards = database.get_portfolio_forward_buckets(
            since_timestamp=cutoff,
            interval_hours=4,
            out_channels=out_scids
        )

        # Get Kalman flow states from the dedicated kalman_state table
        flow_states = {}
        try:
            kalman_rows = database.get_all_kalman_states()
            for ks in kalman_rows:
                cid = ks.get("channel_id")
                if cid:
                    flow_states[cid] = ks
        except Exception:
            pass

        # Initialize optimizer
        optimizer = PortfolioOptimizer(
            database=database,
            plugin=plugin,
            hive_bridge=None  # Can integrate later
        )

        # Run analysis
        analysis = optimizer.analyze_portfolio(
            channels=channels,
            forwards=forwards,
            flow_states=flow_states,
            risk_aversion=risk_aversion
        )

        return {
            "status": "ok",
            **analysis
        }

    except Exception as e:
        # SEC-2: Log exception message at error level, full trace at debug
        plugin.log(f"Error in portfolio analysis: {e}", level='error')
        plugin.log(f"Portfolio traceback: {traceback.format_exc()}", level='debug')
        return {"error": str(e)}


@plugin.method("revenue-portfolio-summary")
def revenue_portfolio_summary(plugin: Plugin) -> Dict[str, Any]:
    """
    Get portfolio summary metrics only (lightweight).

    Returns:
        Portfolio-level metrics without full channel details.
    """
    result = revenue_portfolio(plugin, risk_aversion=1.0)

    if "error" in result:
        return result

    return {
        "status": "ok",
        "summary": result.get("summary", {}),
        "improvement_potential_pct": result.get("summary", {}).get("improvement_potential_pct", 0),
        "recommendation_count": len(result.get("recommendations", [])),
        "hedging_opportunities": len(result.get("hedging_opportunities", [])),
        "concentration_risks": len(result.get("concentration_risks", []))
    }


@plugin.method("revenue-portfolio-rebalance")
def revenue_portfolio_rebalance(
    plugin: Plugin,
    max_recommendations: int = 5
) -> Dict[str, Any]:
    """
    Get portfolio-optimized rebalance recommendations.

    Prioritizes rebalances that improve portfolio efficiency
    (Sharpe ratio) rather than just individual channel balance.

    Args:
        max_recommendations: Maximum number of recommendations (default: 5)

    Returns:
        List of rebalance recommendations with priority and amounts.
    """
    # SEC-1: Validate and clamp parameters
    max_recommendations = max(1, min(100, int(max_recommendations)))

    global database, safe_plugin

    if database is None:
        return {"error": "Database not initialized"}

    if safe_plugin is None:
        return {"error": "Plugin not initialized"}

    try:
        from modules.portfolio_optimizer import PortfolioOptimizer

        # Get channel data
        channels = safe_plugin.rpc.listpeerchannels().get("channels", [])

        # Get forwards from local SQLite table (bucketed).
        import time
        now = int(time.time())
        cutoff = now - (14 * 86400)
        out_scids = [
            (ch.get("short_channel_id") or ch.get("channel_id"))
            for ch in channels
            if (ch.get("short_channel_id") or ch.get("channel_id"))
        ]
        forwards = database.get_portfolio_forward_buckets(
            since_timestamp=cutoff,
            interval_hours=4,
            out_channels=out_scids
        )

        optimizer = PortfolioOptimizer(
            database=database,
            plugin=plugin
        )

        recommendations = optimizer.get_rebalance_priorities(
            channels=channels,
            forwards=forwards,
            max_recommendations=max_recommendations
        )

        return {
            "status": "ok",
            "recommendation_count": len(recommendations),
            "recommendations": recommendations
        }

    except Exception as e:
        plugin.log(f"Error in portfolio rebalance: {e}", level='error')
        return {"error": str(e)}


@plugin.method("revenue-portfolio-correlations")
def revenue_portfolio_correlations(
    plugin: Plugin,
    min_correlation: float = 0.3
) -> Dict[str, Any]:
    """
    Get channel correlation analysis.

    Identifies:
    - Hedging opportunities (negatively correlated channels)
    - Concentration risks (highly correlated channels)

    Args:
        min_correlation: Minimum |correlation| to include (default: 0.3)

    Returns:
        Correlation pairs with relationship classification.
    """
    # SEC-1: Validate and clamp parameters
    min_correlation = max(-1.0, min(1.0, float(min_correlation)))

    result = revenue_portfolio(plugin, risk_aversion=1.0)

    if "error" in result:
        return result

    correlations = result.get("correlations", [])

    # Filter by minimum correlation
    filtered = [c for c in correlations if abs(c.get("correlation", 0)) >= min_correlation]

    return {
        "status": "ok",
        "total_pairs": len(filtered),
        "hedging_opportunities": [c for c in filtered if c.get("relationship") == "hedging"],
        "concentration_risks": [c for c in filtered if c.get("relationship") == "correlated"],
        "all_correlations": filtered
    }


@plugin.method("revenue-cleanup-closed")
def revenue_cleanup_closed(plugin: Plugin) -> Dict[str, Any]:
    """
    Detect and clean up closed channels from active tracking tables.

    This is a backfill operation that finds channels in the tracking database
    that no longer exist (have been closed) and:
    1. Archives them to closed_channels table with P&L data
    2. Removes them from active tracking tables

    Use this to clean up stale data from channels that closed before the
    cleanup feature was implemented.

    Returns:
        {
            "archived": int,      # Number of channels archived
            "cleaned": int,       # Number of tracking records removed
            "channels": [str],    # List of cleaned channel IDs
            "errors": [str]       # Any errors encountered
        }
    """
    global database, safe_plugin

    if database is None:
        return {"error": "Database not initialized"}

    if safe_plugin is None:
        return {"error": "Plugin not initialized"}

    result = {
        "archived": 0,
        "cleaned": 0,
        "channels": [],
        "errors": []
    }

    try:
        import time

        # Get all channels currently tracked in channel_states
        tracked_channels = database.get_all_channel_states()
        tracked_ids = {ch['channel_id'] for ch in tracked_channels}

        if not tracked_ids:
            return {"message": "No tracked channels found", **result}

        # Get all currently open channels
        open_ids = set()
        try:
            channels = safe_plugin.rpc.call("listpeerchannels")
            for ch in channels.get('channels', []):
                scid = normalize_scid(ch.get('short_channel_id', ''))
                if scid:
                    open_ids.add(scid)
        except Exception as e:
            result["errors"].append(f"Failed to get open channels: {e}")
            return result

        # Find closed channels (in tracking but not open)
        closed_ids = tracked_ids - open_ids

        if not closed_ids:
            return {"message": "No closed channels found to clean up", **result}

        plugin.log(
            f"Found {len(closed_ids)} closed channels to clean up: {closed_ids}",
            level='info'
        )

        # Get closure info from listclosedchannels
        closed_info = {}
        try:
            closed_list = safe_plugin.rpc.call("listclosedchannels")
            for ch in closed_list.get('closedchannels', []):
                scid = normalize_scid(ch.get('short_channel_id', ''))
                if scid:
                    closed_info[scid] = ch
        except Exception as e:
            plugin.log(f"listclosedchannels not available: {e}", level='debug')

        # Process each closed channel
        for channel_id in closed_ids:
            try:
                # Get tracked state for peer_id
                tracked_state = next(
                    (ch for ch in tracked_channels if ch['channel_id'] == channel_id),
                    None
                )
                peer_id = tracked_state.get('peer_id') if tracked_state else None

                # Get info from listclosedchannels
                ch_info = closed_info.get(channel_id, {})

                # Determine close type and closer
                close_type = 'unknown'
                closer = ch_info.get('closer', 'unknown')

                if ch_info:
                    # Map CLN close cause to our close_type
                    cause = ch_info.get('close_cause', '')
                    if 'mutual' in cause.lower():
                        close_type = 'mutual'
                    elif closer == 'local':
                        close_type = 'local_unilateral'
                    elif closer == 'remote':
                        close_type = 'remote_unilateral'

                # Archive the channel
                _archive_closed_channel(
                    channel_id=channel_id,
                    peer_id=peer_id or ch_info.get('peer_id'),
                    close_type=close_type,
                    closing_txid=ch_info.get('closing_txid')
                )

                result["archived"] += 1
                result["cleaned"] += 1
                result["channels"].append(channel_id)

            except Exception as e:
                result["errors"].append(f"Error processing {channel_id}: {e}")
                plugin.log(f"Error cleaning up {channel_id}: {e}", level='error')

        plugin.log(
            f"Cleaned up {result['archived']} closed channels",
            level='info'
        )

        return result

    except Exception as e:
        plugin.log(f"Error in cleanup-closed: {e}", level='error')
        result["errors"].append(str(e))
        return result


@plugin.method("revenue-clear-reservations")
def revenue_clear_reservations(plugin: Plugin) -> Dict[str, Any]:
    """
    Clear all active budget reservations (Issue #33).

    Use this command after manually stopping sling jobs to release their
    budget reservations. This resets the reservation system so new
    rebalances can use the daily budget.

    Typical workflow:
    1. lightning-cli sling-deletejob all   # Stop all sling jobs
    2. lightning-cli revenue-clear-reservations  # Release budget

    Returns:
        {
            "status": "success",
            "cleared_count": int,    # Number of reservations cleared
            "released_sats": int,    # Total sats released back to budget
            "budget_available": int  # New available budget after clearing
        }
    """
    global database, config

    if database is None:
        return {"error": "Database not initialized"}

    try:
        # Clear all active reservations
        result = database.clear_all_reservations()

        # Get updated budget status
        cfg = config.snapshot() if hasattr(config, 'snapshot') else config
        spend_info = database.get_daily_rebalance_spend()
        daily_spent = spend_info.get('total_spent_sats', 0)
        budget_available = max(0, cfg.daily_budget_sats - daily_spent)

        return {
            "status": "success",
            "cleared_count": result["cleared_count"],
            "released_sats": result["released_sats"],
            "budget_available": budget_available
        }

    except Exception as e:
        plugin.log(f"Error clearing reservations: {e}", level='error')
        return {"error": str(e)}


# =============================================================================
# HOOKS - React to Lightning events
# =============================================================================

@plugin.hook("htlc_accepted")
def on_htlc_accepted(onion: Dict, htlc: Dict, plugin: Plugin, **kwargs) -> Dict[str, str]:
    """
    Hook called when an HTLC is accepted.
    
    We can use this to track live routing activity and update our flow metrics
    in real-time rather than waiting for periodic analysis.
    
    For now, we just let it pass through. Periodic flow analysis is computed from
    the local forwards table (and hydrated on startup if needed).
    """
    # Just continue - we don't want to interfere with routing
    return {"result": "continue"}


def _resolve_scid_to_peer(scid: str) -> Optional[str]:
    """
    Resolve a short_channel_id to its peer_id.

    Uses a cache to avoid repeated RPC calls. Cache is refreshed if the
    SCID is not found (channel might be new).

    Args:
        scid: Short channel ID (e.g., "123x456x0")

    Returns:
        peer_id (node pubkey) or None if not found
    """
    global _scid_to_peer_cache, _scid_cache_last_cleared

    scid_norm = normalize_scid(scid)

    with _scid_cache_lock:
        # Expire cache periodically to prevent stale mappings
        now = time.time()
        if now - _scid_cache_last_cleared > _SCID_CACHE_TTL_SECONDS:
            _scid_to_peer_cache.clear()
            _scid_cache_last_cleared = now

        # Check cache first
        if scid_norm in _scid_to_peer_cache:
            return _scid_to_peer_cache[scid_norm]

    # Cache miss - RPC call outside lock
    try:
        result = safe_plugin.rpc.listpeerchannels()
        new_cache = {}
        for channel in result.get("channels", []):
            channel_scid = channel.get("short_channel_id") or channel.get("channel_id")
            peer_id = channel.get("peer_id")
            if channel_scid and peer_id:
                new_cache[normalize_scid(channel_scid)] = peer_id

        with _scid_cache_lock:
            _scid_to_peer_cache.update(new_cache)

        return new_cache.get(scid_norm)
    except Exception as e:
        plugin.log(f"Error resolving SCID {scid} to peer: {e}", level='warn')
        return None


def _looks_like_scid(value: Any) -> bool:
    """Return True if a value looks like a short_channel_id."""
    return isinstance(value, str) and bool(re.match(r'^\d+[x:]\d+[x:]\d+$', value))


def _resolve_event_channel_scid(event: Dict[str, Any]) -> Optional[str]:
    """
    Resolve a channel_state_changed event to a short_channel_id (SCID).

    CLN may provide `channel_id` as a funding txid hex string while downstream
    accounting code requires a SCID. We try lightweight RPC resolution when the
    event lacks `short_channel_id`.
    """
    short_scid = event.get('short_channel_id')
    if _looks_like_scid(short_scid):
        return normalize_scid(short_scid)

    raw_channel_id = event.get('channel_id')
    if _looks_like_scid(raw_channel_id):
        return normalize_scid(raw_channel_id)

    if not isinstance(raw_channel_id, str) or safe_plugin is None:
        return None

    raw_channel_id_lc = raw_channel_id.lower()
    peer_id = event.get('peer_id')

    def _match_scid_from_channels(channels: List[Dict[str, Any]]) -> Optional[str]:
        for ch in channels:
            if not isinstance(ch, dict):
                continue
            scid = ch.get('short_channel_id')
            if not _looks_like_scid(scid):
                continue
            candidates = [
                ch.get('channel_id'),
                ch.get('funding_txid'),
                ch.get('txid'),
            ]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.lower() == raw_channel_id_lc:
                    return normalize_scid(scid)
        return None

    # Try open channels first (likely for ONCHAIN/FUNDING_SPEND_SEEN transitions)
    peer_payloads = []
    if isinstance(peer_id, str) and len(peer_id) == 66:
        peer_payloads.append({"id": peer_id})
    peer_payloads.append({})

    for payload in peer_payloads:
        try:
            channels = safe_plugin.rpc.call("listpeerchannels", payload).get("channels", [])
            scid = _match_scid_from_channels(channels)
            if scid:
                return scid
        except Exception:
            continue

    # Fallback for CLOSED events where the channel may already be gone
    try:
        closed = safe_plugin.rpc.call("listclosedchannels").get("closedchannels", [])
        for ch in closed:
            if not isinstance(ch, dict):
                continue
            scid = ch.get('short_channel_id')
            if not _looks_like_scid(scid):
                continue
            candidates = [
                ch.get('channel_id'),
                ch.get('funding_txid'),
                ch.get('txid'),
            ]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.lower() == raw_channel_id_lc:
                    return normalize_scid(scid)
    except Exception:
        pass

    return None


def _parse_msat(msat_val: Any) -> int:
    """
    Safely convert msat values to integers.
    Handles '1000msat' strings, raw integers, Millisatoshi objects, and plain numeric strings.
    """
    return parse_msat(msat_val)


@plugin.subscribe("forward_event")
def on_forward_event(forward_event: Dict, plugin: Plugin, **kwargs):
    """
    Notification when a forward completes (success or failure).
    
    We use this for:
    1. Real-time flow tracking (settled forwards)
    2. Peer reputation tracking (success/failure rates)
    
    Reputation tracking helps identify unreliable peers for traffic intelligence.
    """
    if database is None:
        return
    
    status = forward_event.get("status")
    in_channel = normalize_scid(forward_event.get("in_channel")) if forward_event.get("in_channel") else None

    # Track peer reputation for all forward outcomes
    if in_channel:
        peer_id = _resolve_scid_to_peer(in_channel)
        if peer_id:
            if status == "settled":
                database.update_peer_reputation(peer_id, is_success=True)
            elif status == "failed":
                # Only penalize in_channel peer on downstream failure, NOT
                # local_failed (which means OUR node rejected the forward,
                # e.g. insufficient outbound balance — the sender did nothing wrong).
                database.update_peer_reputation(peer_id, is_success=False)

                # Report failure to cl-hive for pheromone evaporation (Yield Optimization Phase 2)
                if hive_bridge:
                    try:
                        out_channel = forward_event.get("out_channel")
                        out_channel = normalize_scid(out_channel) if out_channel else None
                        if out_channel:
                            out_peer_id = _resolve_scid_to_peer(out_channel)
                            if out_peer_id:
                                hive_bridge.report_routing_outcome(
                                    channel_id=out_channel,
                                    peer_id=out_peer_id,
                                    fee_ppm=0,  # Unknown for failures
                                    success=False,
                                    amount_sats=0,
                                    source=peer_id,
                                    destination=out_peer_id,
                                )
                    except Exception as e:
                        plugin.log(f"FORWARD_EVENT: Hive failure report failed: {e}", level="debug")

    # Record successful forwards for flow metrics
    if status == "settled":
        out_channel = forward_event.get("out_channel")
        out_channel = normalize_scid(out_channel) if out_channel else None

        # CLN v23.05+ uses in_msat/out_msat/fee_msat; older versions used *_msatoshi
        in_msat = _parse_msat(forward_event.get("in_msat", forward_event.get("in_msatoshi", 0)))
        out_msat = _parse_msat(forward_event.get("out_msat", forward_event.get("out_msatoshi", 0)))
        fee_msat = _parse_msat(forward_event.get("fee_msat", forward_event.get("fee_msatoshi", 0)))

        # Calculate resolution duration (Risk Premium tracking)
        received_time = int(forward_event.get("received_time", 0) or 0)
        resolved_time = int(forward_event.get("resolved_time", 0) or 0)
        resolution_duration = max(0, resolved_time - received_time) if resolved_time > 0 else 0

        database.record_forward(
            in_channel or "",
            out_channel or "",
            in_msat,
            out_msat,
            fee_msat,
            received_time,
            resolved_time,
            resolution_duration,
        )

        # Report routing outcome to cl-hive for stigmergic learning (Yield Optimization Phase 2)
        if hive_bridge and out_channel:
            try:
                out_peer_id = _resolve_scid_to_peer(out_channel)
                in_peer_id = _resolve_scid_to_peer(in_channel) if in_channel else None
                amount_sats = out_msat // 1000 if out_msat else 0
                fee_ppm = (fee_msat * 1_000_000 // out_msat) if out_msat > 0 else 0

                if out_peer_id:
                    hive_bridge.report_routing_outcome(
                        channel_id=out_channel,
                        peer_id=out_peer_id,
                        fee_ppm=fee_ppm,
                        success=True,
                        amount_sats=amount_sats,
                        source=in_peer_id,
                        destination=out_peer_id,
                    )
            except Exception as e:
                plugin.log(f"FORWARD_EVENT: Hive routing outcome report failed: {e}", level="debug")


@plugin.subscribe("connect")
def on_peer_connect(plugin: Plugin, **kwargs):
    """
    Notification when a peer connects.
    
    Records the connection event for uptime tracking.
    """
    if database is None:
        return
    
    # Log full structure for debugging
    plugin.log(f"Connect notification: {kwargs}", level='debug')
    
    # Try multiple extraction methods for compatibility
    peer_id = None
    
    # Method 1: Nested under 'connect' key
    if 'connect' in kwargs and isinstance(kwargs['connect'], dict):
        peer_id = kwargs['connect'].get('id')
    
    # Method 2: Direct 'id' key
    if not peer_id and 'id' in kwargs:
        peer_id = kwargs['id']
    
    # Method 3: Check for nested peer_id
    if not peer_id and 'connect' in kwargs and isinstance(kwargs['connect'], dict):
        peer_id = kwargs['connect'].get('peer_id')
    
    if peer_id:
        database.record_connection_event(peer_id, "connected")
        plugin.log(f"Peer connected: {peer_id[:12]}...", level='debug')
    else:
        plugin.log(f"Connect event - could not extract peer_id from: {kwargs}", level='warn')


@plugin.subscribe("disconnect")
def on_peer_disconnect(plugin: Plugin, **kwargs):
    """
    Notification when a peer disconnects.

    Records the disconnection event for uptime tracking.
    """
    if database is None:
        return

    # Log full structure for debugging
    plugin.log(f"Disconnect notification: {kwargs}", level='debug')

    # Try multiple extraction methods for compatibility
    peer_id = None

    # Method 1: Nested under 'disconnect' key
    if 'disconnect' in kwargs and isinstance(kwargs['disconnect'], dict):
        peer_id = kwargs['disconnect'].get('id')

    # Method 2: Direct 'id' key
    if not peer_id and 'id' in kwargs:
        peer_id = kwargs['id']

    # Method 3: Check for nested peer_id
    if not peer_id and 'disconnect' in kwargs and isinstance(kwargs['disconnect'], dict):
        peer_id = kwargs['disconnect'].get('peer_id')

    if peer_id:
        database.record_connection_event(peer_id, "disconnected")
        plugin.log(f"Peer disconnected: {peer_id[:12]}...", level='debug')
    else:
        plugin.log(f"Disconnect event - could not extract peer_id from: {kwargs}", level='warn')


@plugin.subscribe("channel_state_changed")
def on_channel_state_changed(plugin: Plugin, **kwargs):
    """
    Notification when a channel changes state (Accounting v2.0).

    This handler tracks channel closures to record on-chain costs for accurate P&L.
    When a channel transitions to ONCHAIN or CLOSED state, we:
    1. Query bookkeeper for actual on-chain fees
    2. Record closure costs in the database
    3. Archive the complete channel P&L history

    States that indicate closure:
    - ONCHAIN: Channel has gone to chain (unilateral close in progress)
    - CLOSED: Channel is fully closed and resolved
    - FUNDING_SPEND_SEEN: Funding output has been spent (close initiated)
    """
    if database is None:
        return

    # Extract event data - may be nested under 'channel_state_changed' key
    event = kwargs.get('channel_state_changed', kwargs)

    plugin.log(f"Channel state changed: {event}", level='debug')

    # Extract channel information
    # CLN's channel_state_changed provides `channel_id` as a hex funding txid
    # and `short_channel_id` as the SCID (e.g., "123x456x0"). We need the SCID
    # for all downstream operations (DB lookups, fee setting, archiving).
    peer_id = event.get('peer_id')
    raw_channel_id = event.get('short_channel_id') or event.get('channel_id')
    new_state = event.get('new_state', '')
    old_state = event.get('old_state', '')
    cause = event.get('cause', 'unknown')

    if not raw_channel_id:
        plugin.log(f"Channel state change - no channel_id in event: {event}", level='warn')
        return

    # Resolve to SCID (events may provide funding txid in `channel_id`)
    channel_id = _resolve_event_channel_scid(event)
    if not channel_id:
        plugin.log(
            f"Channel state change - could not resolve SCID from event channel_id={raw_channel_id!r}; "
            f"skipping accounting for state={new_state}",
            level='warn'
        )
        return

    # =========================================================================
    # Channel Open Detection (Hive Integration)
    # =========================================================================
    # Channel is opened when it transitions TO CHANNELD_NORMAL from opening states
    opening_states = {
        'DUALOPEND_AWAITING_LOCKIN',
        'DUALOPEND_OPEN_INIT',
        'CHANNELD_AWAITING_LOCKIN',
        'OPENINGD'
    }
    if new_state == 'CHANNELD_NORMAL' and old_state in opening_states:
        plugin.log(
            f"Channel opened: {channel_id} peer={peer_id[:16] if peer_id else 'unknown'}... "
            f"(from {old_state})",
            level='info'
        )
        _handle_channel_open(channel_id, peer_id, old_state, cause)
        # Don't return - continue to allow normal channel handling

    # =========================================================================
    # Splice Detection (Accounting v2.0)
    # =========================================================================
    # Splice is complete when channel transitions FROM CHANNELD_AWAITING_SPLICE
    # back TO CHANNELD_NORMAL (after splice tx confirms)
    if old_state == 'CHANNELD_AWAITING_SPLICE' and new_state == 'CHANNELD_NORMAL':
        plugin.log(
            f"Splice completed: {channel_id} (was awaiting splice, now normal)",
            level='info'
        )
        _handle_splice_completion(channel_id, peer_id)
        return

    # =========================================================================
    # Closure Detection
    # =========================================================================
    # States indicating the channel is closing or closed
    closure_states = {'ONCHAIN', 'CLOSED', 'FUNDING_SPEND_SEEN', 'CLOSINGD_COMPLETE'}

    if new_state not in closure_states:
        # Not a closure event, ignore
        return

    plugin.log(
        f"Channel closure detected: {channel_id} state={new_state} cause={cause}",
        level='info'
    )

    # Determine close type from state and cause
    close_type = _determine_close_type(new_state, old_state, cause)

    # Query bookkeeper for on-chain fees (if available)
    closure_fee_sats = 0
    htlc_sweep_fee_sats = 0
    funding_txid = None
    closing_txid = None

    try:
        # Try to get on-chain fee data from bookkeeper
        closure_data = _get_closure_costs_from_bookkeeper(channel_id)
        if closure_data:
            closure_fee_sats = closure_data.get('closure_fee_sats', 0)
            htlc_sweep_fee_sats = closure_data.get('htlc_sweep_fee_sats', 0)
            funding_txid = closure_data.get('funding_txid')
            closing_txid = closure_data.get('closing_txid')
    except Exception as e:
        plugin.log(f"Error querying bookkeeper for closure costs: {e}", level='warn')
        # Fall back to estimated costs from config
        from modules.config import ChainCostDefaults
        closure_fee_sats = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS

    # Record the closure cost
    database.record_channel_closure(
        channel_id=channel_id,
        peer_id=peer_id or 'unknown',
        close_type=close_type,
        closure_fee_sats=closure_fee_sats,
        htlc_sweep_fee_sats=htlc_sweep_fee_sats,
        funding_txid=funding_txid,
        closing_txid=closing_txid
    )

    # If the channel is fully closed, archive its P&L history
    if new_state == 'CLOSED':
        _archive_closed_channel(channel_id, peer_id, close_type, closing_txid)


def _determine_close_type(new_state: str, old_state: str, cause: str) -> str:
    """
    Determine the type of channel closure from state transition.

    Args:
        new_state: The new channel state
        old_state: The previous channel state
        cause: The cause of the state change

    Returns:
        Close type: 'mutual', 'local_unilateral', 'remote_unilateral', or 'unknown'
    """
    cause_lower = cause.lower() if cause else ''

    # Mutual close - both parties agreed
    if 'mutual' in cause_lower or old_state == 'CLOSINGD_SIGEXCHANGE':
        return 'mutual'

    # Local initiated unilateral
    if cause_lower in ('local', 'user'):
        return 'local_unilateral'

    # Remote initiated unilateral
    if cause_lower in ('remote', 'protocol', 'onchain'):
        return 'remote_unilateral'

    # Check state transitions
    if 'CLOSINGD' in old_state:
        return 'mutual'

    if new_state == 'ONCHAIN':
        # Unilateral close - determine who initiated
        if cause_lower == 'local':
            return 'local_unilateral'
        elif cause_lower == 'remote':
            return 'remote_unilateral'

    return 'unknown'


def _determine_closer(close_type: str) -> str:
    """
    Determine who initiated the closure from the close type.

    Args:
        close_type: Type of closure from _determine_close_type

    Returns:
        Who initiated: 'local', 'remote', 'mutual', or 'unknown'
    """
    if close_type == 'mutual':
        return 'mutual'
    elif close_type == 'local_unilateral':
        return 'local'
    elif close_type == 'remote_unilateral':
        return 'remote'
    return 'unknown'


def _notify_hive_of_closure(channel_id: str, peer_id: str, closer: str,
                             close_type: str, capacity_sats: int = 0,
                             duration_days: int = 0, total_revenue_sats: int = 0,
                             total_rebalance_cost_sats: int = 0, net_pnl_sats: int = 0,
                             forward_count: int = 0) -> bool:
    """
    Notify cl-hive plugin of a channel closure if it's available.

    ALL closures are sent to cl-hive for topology awareness.
    Includes full profitability data to help hive members make decisions.

    Args:
        channel_id: The closed channel ID
        peer_id: The peer whose channel closed
        closer: Who initiated: 'local', 'remote', 'mutual', or 'unknown'
        close_type: Type of closure
        capacity_sats: Channel capacity that was closed
        duration_days: How long channel was open
        total_revenue_sats: Total routing fees earned
        total_rebalance_cost_sats: Total rebalancing costs
        net_pnl_sats: Net profit/loss
        forward_count: Number of forwards routed

    Returns:
        True if notification was sent successfully
    """
    global safe_plugin

    if safe_plugin is None:
        return False

    try:
        # Check if cl-hive plugin is available (cached for performance)
        if not hive_availability_cache.is_available(safe_plugin.rpc):
            return False

        # Calculate routing score from forward count
        routing_score = 0.5  # Default mid-range
        if forward_count > 100:
            routing_score = 0.9
        elif forward_count > 50:
            routing_score = 0.7
        elif forward_count > 10:
            routing_score = 0.5
        elif forward_count > 0:
            routing_score = 0.3
        else:
            routing_score = 0.1

        # Calculate profitability score
        profitability_score = 0.5
        if duration_days > 0 and capacity_sats > 0:
            # Annualized ROC
            annual_pnl = (net_pnl_sats / duration_days) * 365 if duration_days > 0 else 0
            roc_pct = (annual_pnl / capacity_sats) * 100 if capacity_sats > 0 else 0
            if roc_pct > 10:
                profitability_score = 0.9
            elif roc_pct > 5:
                profitability_score = 0.7
            elif roc_pct > 0:
                profitability_score = 0.5
            elif roc_pct > -5:
                profitability_score = 0.3
            else:
                profitability_score = 0.1

        # Get fee rates if available
        our_fee_ppm = 0
        their_fee_ppm = 0
        forward_volume_sats = 0
        if database:
            try:
                # Get our fee rate from strategy state
                state = database.get_fee_strategy_state(channel_id)
                if state:
                    our_fee_ppm = state.get('current_fee_ppm', 0)

                # Estimate volume from revenue
                if our_fee_ppm > 0 and total_revenue_sats > 0:
                    forward_volume_sats = (total_revenue_sats * 1_000_000) // our_fee_ppm
            except Exception:
                pass

        # M-9: Use fire_and_forget since return value is only for logging
        safe_plugin.rpc.fire_and_forget("hive-channel-closed", {
            "peer_id": peer_id,
            "channel_id": channel_id,
            "closer": closer,
            "close_type": close_type,
            "capacity_sats": capacity_sats,
            "duration_days": duration_days,
            "total_revenue_sats": total_revenue_sats,
            "total_rebalance_cost_sats": total_rebalance_cost_sats,
            "net_pnl_sats": net_pnl_sats,
            "forward_count": forward_count,
            "forward_volume_sats": forward_volume_sats,
            "our_fee_ppm": our_fee_ppm,
            "their_fee_ppm": their_fee_ppm,
            "routing_score": routing_score,
            "profitability_score": profitability_score
        })

        plugin.log(
            f"Notified cl-hive of closure: {channel_id} by {closer} "
            f"(pnl={net_pnl_sats}, forwards={forward_count})",
            level='info'
        )
        return True

    except Exception as e:
        # Log at warn level for visibility; include channel ID for debugging
        plugin.log(
            f"Failed to notify cl-hive of channel closure {channel_id}: {e}",
            level='warn'
        )
        return False


def _notify_hive_of_open(channel_id: str, peer_id: str, opener: str,
                          capacity_sats: int = 0, our_funding_sats: int = 0,
                          their_funding_sats: int = 0) -> bool:
    """
    Notify cl-hive plugin of a channel opening if it's available.

    ALL opens are sent to cl-hive for topology awareness.

    Args:
        channel_id: The new channel ID
        peer_id: The peer the channel was opened with
        opener: Who initiated: 'local' or 'remote'
        capacity_sats: Total channel capacity
        our_funding_sats: Amount we funded
        their_funding_sats: Amount they funded

    Returns:
        True if notification was sent successfully
    """
    global safe_plugin

    if safe_plugin is None:
        return False

    try:
        # Check if cl-hive plugin is available (cached for performance)
        if not hive_availability_cache.is_available(safe_plugin.rpc):
            return False

        # M-9: Use fire_and_forget since return value is only for logging
        safe_plugin.rpc.fire_and_forget("hive-channel-opened", {
            "peer_id": peer_id,
            "channel_id": channel_id,
            "opener": opener,
            "capacity_sats": capacity_sats,
            "our_funding_sats": our_funding_sats,
            "their_funding_sats": their_funding_sats
        })

        plugin.log(
            f"Notified cl-hive of channel open: {channel_id} with {peer_id[:16]}... ({opener})",
            level='info'
        )
        return True

    except Exception as e:
        # Log at warn level for visibility; include channel ID for debugging
        plugin.log(
            f"Failed to notify cl-hive of channel open {channel_id}: {e}",
            level='warn'
        )
        return False


def _handle_channel_open(channel_id: str, peer_id: Optional[str],
                          old_state: str, cause: str) -> None:
    """
    Handle a channel open event.

    Called when a channel transitions to CHANNELD_NORMAL from an opening state.
    Notifies cl-hive for topology awareness.

    Args:
        channel_id: The new channel ID
        peer_id: The peer the channel was opened with
        old_state: The previous state (indicates open type)
        cause: The cause of the state change
    """
    global safe_plugin

    if safe_plugin is None or not peer_id:
        return

    try:
        # Determine opener from old_state and cause
        # DUALOPEND states typically mean we initiated (dual-funded)
        # CHANNELD_AWAITING_LOCKIN typically means remote initiated
        # M-8: Single RPC call for both opener detection and channel details
        opener = 'unknown'
        if cause == 'remote':
            opener = 'remote'
        elif cause == 'user':
            opener = 'local'

        capacity_sats = 0
        our_funding_sats = 0
        their_funding_sats = 0

        try:
            channels = safe_plugin.rpc.call("listpeerchannels", {"id": peer_id})
            for ch in channels.get('channels', []):
                scid = normalize_scid(ch.get('short_channel_id', ''))
                if scid == channel_id:
                    if opener == 'unknown':
                        opener = ch.get('opener', 'unknown')
                    capacity_sats = _parse_msat(ch.get('total_msat', 0)) // 1000
                    our_funding_sats = _parse_msat(ch.get('funding', {}).get('local_funds_msat', 0)) // 1000
                    their_funding_sats = _parse_msat(ch.get('funding', {}).get('remote_funds_msat', 0)) // 1000
                    break
        except Exception as e:
            plugin.log(f"Failed to get channel details for {channel_id}: {e}", level='debug')

        # Notify cl-hive
        _notify_hive_of_open(
            channel_id=channel_id,
            peer_id=peer_id,
            opener=opener,
            capacity_sats=capacity_sats,
            our_funding_sats=our_funding_sats,
            their_funding_sats=their_funding_sats
        )

        # Set initial fee immediately so the channel doesn't sit with CLN
        # defaults until the next periodic fee adjustment cycle.
        if fee_controller is not None:
            try:
                fee_controller.set_initial_fee(channel_id, peer_id)
            except Exception as fee_err:
                plugin.log(
                    f"Failed to set initial fee for {channel_id}: {fee_err}",
                    level='warn'
                )

    except Exception as e:
        plugin.log(f"Error handling channel open {channel_id}: {e}", level='debug')


def _get_closure_costs_from_bookkeeper(channel_id: str) -> Optional[Dict[str, Any]]:
    """
    Query bookkeeper for on-chain fees related to channel closure.

    Uses bkpr-listaccountevents to find onchain_fee events for the channel.

    Args:
        channel_id: The channel short ID

    Returns:
        Dict with closure_fee_sats, htlc_sweep_fee_sats, funding_txid, closing_txid
        or None if bookkeeper unavailable
    """
    global safe_plugin

    if safe_plugin is None:
        return None

    try:
        # Query bookkeeper for account events
        # The account name for a channel is typically the channel_id
        events = safe_plugin.rpc.call("bkpr-listaccountevents", {"account": channel_id})

        if not events or 'events' not in events:
            return None

        closure_fee_sats = 0
        htlc_sweep_fee_sats = 0
        funding_txid = None
        closing_txid = None

        # Security: Validate events structure
        event_list = events.get('events', [])
        if not isinstance(event_list, list):
            plugin.log(f"Security: Invalid events structure from bookkeeper for {channel_id}", level='warn')
            return None

        for event in event_list:
            # Security: Type check each event is a dict
            if not isinstance(event, dict):
                continue

            event_type = event.get('type', '')
            tag = event.get('tag', '')

            # Track funding transaction
            if tag == 'channel_open':
                funding_txid = event.get('txid')

            # Track closing transaction and fees
            if tag in ('channel_close', 'mutual_close', 'unilateral_close'):
                closing_txid = event.get('txid')

            # Accumulate on-chain fees
            if event_type == 'onchain_fee':
                # Security: Type check fee values before arithmetic
                credit_msat = event.get('credit_msat', 0)
                debit_msat = event.get('debit_msat', 0)

                # Parse msat values safely (handles Millisatoshi objects, strings, ints)
                credit_msat = parse_msat(credit_msat)
                debit_msat = parse_msat(debit_msat)

                fee_msat = max(abs(credit_msat), abs(debit_msat))
                fee_sats = fee_msat // 1000

                # Security: Bounds check (max 50,000 sats per fee event)
                fee_sats = min(fee_sats, 50000)

                # Categorize the fee
                if 'htlc' in tag.lower() or 'sweep' in tag.lower():
                    htlc_sweep_fee_sats += fee_sats
                else:
                    closure_fee_sats += fee_sats

        return {
            'closure_fee_sats': closure_fee_sats,
            'htlc_sweep_fee_sats': htlc_sweep_fee_sats,
            'funding_txid': funding_txid,
            'closing_txid': closing_txid
        }

    except Exception as e:
        # Bookkeeper might not be available or channel not found
        plugin.log(f"Bookkeeper query failed for {channel_id}: {e}", level='debug')
        return None


def _archive_closed_channel(channel_id: str, peer_id: Optional[str], close_type: str,
                            closing_txid: Optional[str]) -> None:
    """
    Archive the complete P&L history for a closed channel.

    This preserves all accounting data before the channel is forgotten,
    ensuring accurate lifetime P&L calculations.

    Args:
        channel_id: The channel short ID
        peer_id: The peer node ID
        close_type: Type of closure
        closing_txid: The closing transaction ID
    """
    global database, safe_plugin

    if database is None:
        return

    try:
        import time

        # Get channel cost data (opening cost)
        channel_cost = database.get_channel_cost(channel_id)
        open_cost_sats = channel_cost.get('open_cost_sats', 0) if channel_cost else 0
        opened_at = channel_cost.get('opened_at') if channel_cost else None
        funding_txid = channel_cost.get('funding_txid') if channel_cost else None

        # Get closure cost data
        closure_cost = database.get_channel_closure_cost(channel_id)
        closure_cost_sats = closure_cost.get('total_closure_cost_sats', 0) if closure_cost else 0

        # Get channel P&L from current data
        pnl = database.get_channel_pnl(channel_id, window_days=3650)  # 10 years = all time
        total_revenue_sats = pnl.get('revenue_sats', 0)
        total_rebalance_cost_sats = pnl.get('rebalance_cost_sats', 0)
        forward_count = pnl.get('forward_count', 0)

        # Determine closer from close_type
        closer = _determine_closer(close_type)

        # Try to get capacity and additional info from listclosedchannels (CLN v23.11+)
        capacity_sats = 0
        if safe_plugin:
            try:
                closed = safe_plugin.rpc.call("listclosedchannels")
                for ch in closed.get('closedchannels', []):
                    if normalize_scid(ch.get('short_channel_id', '')) == channel_id:
                        capacity_sats = _parse_msat(ch.get('total_msat', 0)) // 1000
                        if not peer_id:
                            peer_id = ch.get('peer_id')
                        # CLN provides 'closer' field in listclosedchannels (v24.02+)
                        if closer == 'unknown' and ch.get('closer'):
                            closer = ch.get('closer')  # 'local' or 'remote'
                        break
            except Exception:
                pass

        now = int(time.time())

        # Record the closed channel history
        database.record_closed_channel_history(
            channel_id=channel_id,
            peer_id=peer_id or 'unknown',
            capacity_sats=capacity_sats,
            opened_at=opened_at,
            closed_at=now,
            close_type=close_type,
            open_cost_sats=open_cost_sats,
            closure_cost_sats=closure_cost_sats,
            total_revenue_sats=total_revenue_sats,
            total_rebalance_cost_sats=total_rebalance_cost_sats,
            forward_count=forward_count,
            funding_txid=funding_txid,
            closing_txid=closing_txid,
            closer=closer
        )

        plugin.log(
            f"Archived closed channel {channel_id}: "
            f"revenue={total_revenue_sats}, costs={open_cost_sats + closure_cost_sats + total_rebalance_cost_sats}, "
            f"closer={closer}",
            level='info'
        )

        # Clean up active tracking tables now that channel is archived
        database.remove_closed_channel_data(channel_id, peer_id)

        # Notify cl-hive of ALL closures for topology awareness
        # Includes full profitability data to help hive members make decisions
        if peer_id:
            days_open = ((now - opened_at) // 86400) if opened_at else 0
            net_pnl = total_revenue_sats - (open_cost_sats + closure_cost_sats + total_rebalance_cost_sats)
            _notify_hive_of_closure(
                channel_id=channel_id,
                peer_id=peer_id,
                closer=closer,
                close_type=close_type,
                capacity_sats=capacity_sats,
                duration_days=days_open,
                total_revenue_sats=total_revenue_sats,
                total_rebalance_cost_sats=total_rebalance_cost_sats,
                net_pnl_sats=net_pnl,
                forward_count=forward_count
            )

    except Exception as e:
        plugin.log(f"Error archiving closed channel {channel_id}: {e}", level='error')
        plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')


def _handle_splice_completion(channel_id: str, peer_id: Optional[str]) -> None:
    """
    Handle a completed splice operation (Accounting v2.0).

    Called when a channel transitions from CHANNELD_AWAITING_SPLICE to CHANNELD_NORMAL,
    indicating the splice transaction has confirmed.

    Args:
        channel_id: The channel short ID
        peer_id: The peer node ID
    """
    global database, safe_plugin

    if database is None:
        return

    try:
        # Get splice data from bookkeeper
        splice_data = _get_splice_costs_from_bookkeeper(channel_id)

        if splice_data:
            splice_type = splice_data.get('splice_type', 'splice_in')
            amount_sats = splice_data.get('amount_sats', 0)
            fee_sats = splice_data.get('fee_sats', 0)
            old_capacity = splice_data.get('old_capacity_sats')
            new_capacity = splice_data.get('new_capacity_sats')
            txid = splice_data.get('txid')
        else:
            # Fallback: try to determine from channel info
            splice_type = 'splice_in'  # Assume splice_in if we can't determine
            amount_sats = 0
            fee_sats = 0
            old_capacity = None
            new_capacity = None
            txid = None

            # Try to get current capacity from listpeerchannels
            if safe_plugin:
                try:
                    peers = safe_plugin.rpc.listpeerchannels()
                    for ch in peers.get('channels', []):
                        scid = normalize_scid(ch.get('short_channel_id', ''))
                        if scid == channel_id:
                            new_capacity = _parse_msat(ch.get('total_msat', 0)) // 1000
                            if not peer_id:
                                peer_id = ch.get('peer_id')
                            break
                except Exception as e:
                    plugin.log(f"Error getting channel info for splice: {e}", level='debug')

            # Estimate splice fee from config if bookkeeper unavailable
            from modules.config import ChainCostDefaults
            fee_sats = ChainCostDefaults.SPLICE_COST_SATS

        # Record the splice
        database.record_splice(
            channel_id=channel_id,
            peer_id=peer_id or 'unknown',
            splice_type=splice_type,
            amount_sats=amount_sats,
            fee_sats=fee_sats,
            old_capacity_sats=old_capacity,
            new_capacity_sats=new_capacity,
            txid=txid
        )

    except Exception as e:
        plugin.log(f"Error handling splice completion for {channel_id}: {e}", level='error')
        plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')


def _get_splice_costs_from_bookkeeper(channel_id: str) -> Optional[Dict[str, Any]]:
    """
    Query bookkeeper for on-chain fees related to a splice operation.

    Uses bkpr-listaccountevents to find splice-related events for the channel.

    Args:
        channel_id: The channel short ID

    Returns:
        Dict with splice_type, amount_sats, fee_sats, old_capacity_sats, new_capacity_sats, txid
        or None if bookkeeper unavailable or no splice data found
    """
    global safe_plugin

    if safe_plugin is None:
        return None

    try:
        # Query bookkeeper for account events
        events = safe_plugin.rpc.call("bkpr-listaccountevents", {"account": channel_id})

        if not events or 'events' not in events:
            return None

        # Look for recent splice-related events
        splice_fee_sats = 0
        splice_txid = None
        splice_amount = 0

        # Security: Validate events structure
        all_events = events.get('events', [])
        if not isinstance(all_events, list):
            plugin.log(f"Security: Invalid events structure from bookkeeper for splice {channel_id}", level='warn')
            return None

        for event in all_events:  # Process oldest to newest
            # Security: Type check each event is a dict
            if not isinstance(event, dict):
                continue

            event_type = event.get('type', '')
            tag = str(event.get('tag', '')).lower()  # Ensure tag is string

            # Look for splice-related tags
            # Note: CLN bookkeeper may use tags like 'splice', 'splice_in', 'splice_out'
            if 'splice' in tag:
                splice_txid = event.get('txid')

                # Parse msat values safely (handles Millisatoshi objects, strings, ints)
                credit = parse_msat(event.get('credit_msat', 0))
                debit = parse_msat(event.get('debit_msat', 0))

                splice_amount = (credit - debit) // 1000  # Convert to sats

            # Accumulate on-chain fees for splice
            if event_type == 'onchain_fee' and 'splice' in tag:
                # Security: Type check fee values
                credit_msat = event.get('credit_msat', 0)
                debit_msat = event.get('debit_msat', 0)

                # Parse msat values safely (handles Millisatoshi objects, strings, ints)
                credit_msat = parse_msat(credit_msat)
                debit_msat = parse_msat(debit_msat)

                fee_msat = max(abs(credit_msat), abs(debit_msat))
                fee_sats = fee_msat // 1000

                # Security: Bounds check (max 50,000 sats per fee event)
                fee_sats = min(fee_sats, 50000)
                splice_fee_sats += fee_sats

        # If we found splice data, return it
        if splice_txid or splice_fee_sats > 0:
            # Determine splice type from amount
            splice_type = 'splice_in' if splice_amount >= 0 else 'splice_out'

            return {
                'splice_type': splice_type,
                'amount_sats': abs(splice_amount),
                'fee_sats': splice_fee_sats,
                'old_capacity_sats': None,  # Would need to track this separately
                'new_capacity_sats': None,
                'txid': splice_txid
            }

        return None

    except Exception as e:
        plugin.log(f"Bookkeeper query failed for splice on {channel_id}: {e}", level='debug')
        return None


# =============================================================================
# BOLTZ CLI INTEGRATION (optional)
# =============================================================================

def _require_boltz_manager() -> BoltzCliManager:
    if boltz_manager is None:
        raise BoltzCliError("Boltz CLI integration not initialized")
    return boltz_manager


@plugin.method("revenue-total-cost-budget")
def revenue_total_cost_budget(plugin: Plugin, window_hours: int = None) -> Dict[str, Any]:
    """Unified budget status across rebalances, Boltz, and on-chain liquidity costs."""
    try:
        return _total_cost_budget_status(window_hours=window_hours)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-spend-ledger")
def revenue_spend_ledger(plugin: Plugin, window_hours: int = 24) -> Dict[str, Any]:
    """Summary of generic spend ledger events/reservations (for opens/closes/splices/etc.)."""
    if database is None:
        return {"error": "Database not initialized"}
    try:
        return database.get_spend_ledger_summary(window_hours=int(window_hours))
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-spend-reserve")
def revenue_spend_reserve(
    plugin: Plugin,
    reservation_id: str,
    category: str,
    amount_sats: int,
    subcategory: str = None,
    reference_id: str = None,
    channel_id: str = None,
    metadata_json: str = None,
) -> Dict[str, Any]:
    """Reserve spend in the generic ledger, enforcing the unified total-cost budget first."""
    if database is None:
        return {"error": "Database not initialized"}
    try:
        amount_sats = int(amount_sats)
        if amount_sats <= 0:
            return {"error": "amount_sats must be > 0"}
        budget = _total_cost_budget_status()
        if "error" in budget:
            return budget
        remaining = int(budget.get("remaining_sats", 0) or 0)
        if amount_sats > remaining:
            return {
                "status": "rejected",
                "reason": "insufficient_unified_budget",
                "requested_sats": amount_sats,
                "remaining_sats": remaining,
                "budget": budget,
            }
        metadata = None
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except Exception:
                metadata = {"raw": metadata_json}
        ok = database.reserve_spend(
            reservation_id=str(reservation_id),
            amount_sats=amount_sats,
            category=str(category),
            subcategory=subcategory,
            reference_id=reference_id,
            channel_id=channel_id,
            metadata=metadata,
        )
        if not ok:
            return {"status": "error", "error": "Failed to reserve spend"}
        return {
            "status": "success",
            "reservation_id": str(reservation_id),
            "category": str(category),
            "amount_sats": amount_sats,
            "budget_before": budget,
            "budget_after_estimate": _total_cost_budget_status(),
        }
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-spend-release")
def revenue_spend_release(plugin: Plugin, reservation_id: str) -> Dict[str, Any]:
    if database is None:
        return {"error": "Database not initialized"}
    try:
        ok = database.release_spend_reservation(str(reservation_id))
        return {"status": "success" if ok else "not_found", "reservation_id": str(reservation_id)}
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-spend-settle")
def revenue_spend_settle(
    plugin: Plugin,
    reservation_id: str,
    actual_spent_sats: int = None,
    source: str = None,
    record_event: bool = False,
) -> Dict[str, Any]:
    """Mark a reservation spent and optionally record a generic spend event."""
    if database is None:
        return {"error": "Database not initialized"}
    try:
        ok = database.mark_spend_reservation_spent(
            reservation_id=str(reservation_id),
            actual_spent_sats=(None if actual_spent_sats is None else int(actual_spent_sats)),
            source=source,
            record_event=bool(record_event),
        )
        return {"status": "success" if ok else "not_found", "reservation_id": str(reservation_id)}
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-quote")
def revenue_boltz_quote(plugin: Plugin, amount_sats: int, swap_type: str = "reverse", currency: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().quote(amount_sats=amount_sats, swap_type=swap_type, currency=currency)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-loop-out")
def revenue_boltz_loop_out(plugin: Plugin, amount_sats: int, address: str = None, channel_id: str = None,
                           peer_id: str = None, currency: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().loop_out(
            amount_sats=amount_sats, address=address, channel_id=channel_id, peer_id=peer_id, currency=currency
        )
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-loop-in")
def revenue_boltz_loop_in(plugin: Plugin, amount_sats: int, channel_id: str = None,
                          peer_id: str = None, currency: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().loop_in(
            amount_sats=amount_sats, channel_id=channel_id, peer_id=peer_id, currency=currency
        )
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-status")
def revenue_boltz_status(plugin: Plugin, swap_id: str) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().swap_status(swap_id)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-history")
def revenue_boltz_history(plugin: Plugin, limit: int = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().swap_history(limit=limit)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-budget")
def revenue_boltz_budget(plugin: Plugin) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().budget()
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-wallet")
def revenue_boltz_wallet(plugin: Plugin) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().wallet_balances()
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-refund")
def revenue_boltz_refund(plugin: Plugin, swap_id: str, destination: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().refund(swap_id=swap_id, destination=destination)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-claim")
def revenue_boltz_claim(plugin: Plugin, swap_ids: List[str], destination: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().claim(swap_ids=swap_ids, destination=destination)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-chainswap")
def revenue_boltz_chainswap(plugin: Plugin, amount_sats: int, from_currency: str = None,
                            to_currency: str = None, to_address: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().chainswap(
            amount_sats=amount_sats, from_currency=from_currency, to_currency=to_currency, to_address=to_address
        )
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-withdraw")
def revenue_boltz_withdraw(plugin: Plugin, amount_sats: int = None, destination: str = None, currency: str = None,
                           sat_per_vbyte: int = None, sweep: bool = False) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().withdraw(
            amount_sats=amount_sats, destination=destination, currency=currency,
            sat_per_vbyte=sat_per_vbyte, sweep=sweep
        )
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-deposit")
def revenue_boltz_deposit(plugin: Plugin, currency: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().deposit_address(currency=currency)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-backup")
def revenue_boltz_backup(plugin: Plugin) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().backup()
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-backup-verify")
def revenue_boltz_backup_verify(plugin: Plugin, swap_mnemonic: str) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().backup_verify(swap_mnemonic=swap_mnemonic)
    except Exception as e:
        return {"error": str(e)}



def _boltz_balance_pct_from_channel_info(channel_info: Dict[str, Any]) -> float:
    capacity = int(channel_info.get("capacity", 0) or 0)
    spendable_msat = parse_msat(channel_info.get("spendable_msat", 0))
    local_sats = spendable_msat // 1000
    if capacity <= 0:
        return 50.0
    return max(0.0, min(100.0, (100.0 * local_sats) / capacity))


def _boltz_pending_swap_count() -> int:
    """Best-effort count of active/pending manual swaps to avoid overlap."""
    try:
        history = _require_boltz_manager().swap_history(limit=200)
        swaps = history.get("swaps", []) if isinstance(history, dict) else []
        pending = 0
        for sw in swaps:
            if not isinstance(sw, dict):
                continue
            state = str(sw.get("state") or "").lower()
            status = str(sw.get("status") or "").lower()
            txt = f"{state} {status}"
            done = any(x in txt for x in ("success", "completed", "claimed", "failed", "refunded", "expired", "cancel"))
            active = any(x in txt for x in ("pending", "created", "mempool", "transaction", "lockup", "invoice", "claim"))
            if active and not done:
                pending += 1
        return pending
    except Exception:
        return 0


def _rebalance_liquidity_cost_components(window_hours: int = 24) -> Dict[str, Any]:
    """Rebalance spend/reservations for unified liquidity-cost accounting."""
    if database is None:
        return {"source": "rebalance", "spent_24h_sats": 0, "reserved_24h_sats": 0, "available": False}
    try:
        spend = database.get_daily_rebalance_spend(window_hours=window_hours)
        return {
            "source": "rebalance",
            "available": True,
            "window_hours": window_hours,
            "spent_24h_sats": int(spend.get("total_spent_sats", 0) or 0),
            "reserved_24h_sats": int(spend.get("total_reserved_sats", 0) or 0),
            "job_count": int(spend.get("job_count", 0) or 0),
            "success_count": int(spend.get("success_count", 0) or 0),
        }
    except Exception as e:
        return {
            "source": "rebalance",
            "available": False,
            "error": str(e),
            "spent_24h_sats": 0,
            "reserved_24h_sats": 0,
        }


def _boltz_liquidity_cost_components(window_hours: int = 24) -> Dict[str, Any]:
    """Boltz swap spend component only (no external costs, no unified budget recursion)."""
    if boltz_manager is None:
        return {"source": "boltz", "spent_24h_sats": 0, "reserved_24h_sats": 0, "available": False}
    try:
        if hasattr(boltz_manager, "get_boltz_cost_components"):
            comps = boltz_manager.get_boltz_cost_components(window_hours=window_hours)
        else:
            budget = boltz_manager.budget()
            comps = {
                "spent_24h_sats": int(budget.get("boltz_spent_24h_sats_estimate", budget.get("spent_24h_sats_estimate", 0)) or 0),
                "reserved_24h_sats": 0,
                "counted_swaps": int(budget.get("counted_swaps", 0) or 0),
            }
        return {
            "source": "boltz",
            "available": True,
            "spent_24h_sats": int(comps.get("spent_24h_sats", 0) or 0),
            "reserved_24h_sats": int(comps.get("reserved_24h_sats", 0) or 0),
            "counted_swaps": int(comps.get("counted_swaps", 0) or 0),
        }
    except Exception as e:
        return {
            "source": "boltz",
            "available": False,
            "error": str(e),
            "spent_24h_sats": 0,
            "reserved_24h_sats": 0,
        }


def _normalize_total_cost_budget_mode(mode: Optional[str]) -> str:
    m = str(mode or "fixed").strip().lower()
    return "profit_pct" if m in ("profit", "profit_pct", "profit-percent", "percentage") else "fixed"


def _total_cost_budget_status(window_hours: Optional[int] = None) -> Dict[str, Any]:
    """Unified budget status across rebalances, Boltz swaps, and on-chain liquidity ops."""
    if config is None or database is None:
        return {"error": "Plugin not initialized"}

    cfg = config.snapshot() if hasattr(config, "snapshot") else config
    wh = int(window_hours or getattr(cfg, "total_cost_budget_window_hours", 24) or 24)
    wh = max(1, min(168, wh))
    now = int(time.time())
    since = now - (wh * 3600)

    # Best-effort cleanup for generic spend reservations (e.g. channel open/splice
    # reservations) so accepted actions that aren't explicitly settled do not block
    # budget forever. Uses max(reservation_timeout_hours, window_hours).
    try:
        stale_hours = max(int(getattr(cfg, "reservation_timeout_hours", 4) or 4), wh)
        database.cleanup_stale_spend_reservations(max_age_seconds=stale_hours * 3600)
    except Exception:
        pass

    # Actual cost components (canonical data sources)
    rebalance = _rebalance_liquidity_cost_components(window_hours=wh)
    boltz = _boltz_liquidity_cost_components(window_hours=wh)
    generic_ledger = database.get_spend_ledger_summary(window_hours=wh) if database else {
        "spent_24h_sats": 0, "reserved_24h_sats": 0, "spent_by_category": {}, "reserved_by_category": {}
    }
    revenue_sats = int(database.get_total_routing_revenue(since)) if database else 0
    open_cost_sats = int(database.get_opening_costs_since(since)) if database else 0
    closure_cost_sats = int(database.get_closure_costs_since(since)) if database else 0
    splice_cost_sats = int(database.get_splice_costs_since(since)) if database else 0

    actual_by_category = {
        "rebalance": int(rebalance.get("spent_24h_sats", 0) or 0),
        "boltz": int(boltz.get("spent_24h_sats", 0) or 0),
        "open": open_cost_sats,
        "close": closure_cost_sats,
        "splice": splice_cost_sats,
        "ledger": int(generic_ledger.get("spent_24h_sats", 0) or 0),
    }
    reserved_by_category = {
        "rebalance": int(rebalance.get("reserved_24h_sats", 0) or 0),
        "boltz": int(boltz.get("reserved_24h_sats", 0) or 0),
        "ledger": int(generic_ledger.get("reserved_24h_sats", 0) or 0),
    }

    actual_total = sum(max(0, int(v or 0)) for v in actual_by_category.values())
    reserved_total = sum(max(0, int(v or 0)) for v in reserved_by_category.values())

    mode = _normalize_total_cost_budget_mode(getattr(cfg, "total_cost_budget_mode", "fixed"))
    fixed_floor = max(0, int(getattr(cfg, "daily_budget_sats", 0) or 0))
    pct_cfg = float(getattr(cfg, "total_cost_budget_profit_pct", 0.30) or 0.30)
    pct_cap = float(getattr(cfg, "total_cost_budget_profit_pct_cap", 0.75) or 0.75)
    pct_effective = max(0.0, min(pct_cfg, pct_cap, 1.0))
    net_profit_sats = int(revenue_sats - actual_total)
    profit_based_budget_sats = int(max(0, net_profit_sats) * pct_effective)

    if mode == "profit_pct":
        # Keep the fixed budget as a floor for resilience/recovery.
        effective_budget_sats = max(fixed_floor, profit_based_budget_sats)
    else:
        effective_budget_sats = fixed_floor

    remaining_sats = max(0, int(effective_budget_sats) - actual_total - reserved_total)

    return {
        "source": "total_cost_budget",
        "window_hours": wh,
        "since_timestamp": since,
        "mode": mode,
        "daily_budget_floor_sats": fixed_floor,
        "profit_pct_requested": pct_cfg,
        "profit_pct_cap": pct_cap,
        "profit_pct_effective": pct_effective,
        "profit_based_budget_sats": profit_based_budget_sats,
        "effective_budget_sats": int(effective_budget_sats),
        "revenue_sats": revenue_sats,
        "actual_spent_sats": actual_total,
        "reserved_sats": reserved_total,
        "remaining_sats": remaining_sats,
        "net_profit_sats_after_costs": net_profit_sats,
        "actual_spent_by_category": actual_by_category,
        "reserved_by_category": reserved_by_category,
        "components": {
            "rebalance": rebalance,
            "boltz": boltz,
            "generic_ledger": generic_ledger,
            "open_cost_sats": open_cost_sats,
            "closure_cost_sats": closure_cost_sats,
            "splice_cost_sats": splice_cost_sats,
        },
    }


def _total_cost_budget_limit_provider() -> Dict[str, Any]:
    status = _total_cost_budget_status()
    if "error" in status:
        # Fall back to fixed budget floor if unavailable.
        floor = int(getattr(config, "daily_budget_sats", 0) or 0) if config is not None else 0
        return {"source": "fallback", "effective_budget_sats": max(0, floor)}
    return {
        "source": "total_cost_budget",
        "effective_budget_sats": int(status.get("effective_budget_sats", 0) or 0),
        "mode": status.get("mode"),
        "window_hours": status.get("window_hours"),
        "remaining_sats": status.get("remaining_sats"),
    }


def _non_rebalance_liquidity_cost_components(window_hours: Optional[int] = None) -> Dict[str, Any]:
    status = _total_cost_budget_status(window_hours=window_hours)
    if "error" in status:
        return {"source": "non_rebalance_total_costs", "spent_24h_sats": 0, "reserved_24h_sats": 0, "available": False}
    actual = status.get("actual_spent_by_category", {}) if isinstance(status.get("actual_spent_by_category"), dict) else {}
    reserved = status.get("reserved_by_category", {}) if isinstance(status.get("reserved_by_category"), dict) else {}
    return {
        "source": "non_rebalance_total_costs",
        "available": True,
        "spent_24h_sats": max(0, int(status.get("actual_spent_sats", 0) or 0) - int(actual.get("rebalance", 0) or 0)),
        "reserved_24h_sats": max(0, int(status.get("reserved_sats", 0) or 0) - int(reserved.get("rebalance", 0) or 0)),
        "window_hours": int(status.get("window_hours", 24) or 24),
    }


def _non_boltz_liquidity_cost_components(window_hours: Optional[int] = None) -> Dict[str, Any]:
    status = _total_cost_budget_status(window_hours=window_hours)
    if "error" in status:
        return {"source": "non_boltz_total_costs", "spent_24h_sats": 0, "reserved_24h_sats": 0, "available": False}
    actual = status.get("actual_spent_by_category", {}) if isinstance(status.get("actual_spent_by_category"), dict) else {}
    reserved = status.get("reserved_by_category", {}) if isinstance(status.get("reserved_by_category"), dict) else {}
    return {
        "source": "non_boltz_total_costs",
        "available": True,
        "spent_24h_sats": max(0, int(status.get("actual_spent_sats", 0) or 0) - int(actual.get("boltz", 0) or 0)),
        "reserved_24h_sats": max(0, int(status.get("reserved_sats", 0) or 0) - int(reserved.get("boltz", 0) or 0)),
        "window_hours": int(status.get("window_hours", 24) or 24),
    }


def _boltz_direction_allowed_by_policy(peer_id: str, direction: str) -> Tuple[bool, str]:
    """direction: loop_in (fill local) or loop_out (drain local)."""
    if policy_manager is None:
        return True, "no_policy_manager"
    try:
        pol = policy_manager.get_policy(peer_id)
    except Exception as e:
        return False, f"policy_lookup_failed: {e}"

    if pol.strategy == FeeStrategy.PASSIVE:
        return False, "policy_passive"

    mode = pol.rebalance_mode
    if direction == "loop_in":
        if mode in (RebalanceMode.ENABLED, RebalanceMode.SINK_ONLY):
            return True, mode.value
        return False, f"policy_rebalance_mode={mode.value}"
    if direction == "loop_out":
        if mode in (RebalanceMode.ENABLED, RebalanceMode.SOURCE_ONLY):
            return True, mode.value
        return False, f"policy_rebalance_mode={mode.value}"
    return False, "unknown_direction"


def _boltz_channel_daily_contribution_estimate_sats(prof) -> float:
    if not prof:
        return 0.0
    try:
        total = float(getattr(prof.revenue, "total_contribution_sats", 0) or 0)
        days_open = int(getattr(prof, "days_open", 0) or 0)
        # Conservative normalization: use up to 30 days if channel is older.
        days = max(1, min(days_open, 30))
        return total / days
    except Exception:
        return 0.0


def _boltz_dynamic_channel_tuning(*,
    local_pct: float,
    low_trigger_pct: float,
    low_target_pct: float,
    high_trigger_pct: float,
    high_target_pct: float,
    flow_state: str,
    daily_contrib_est: float,
    marginal_roi: Optional[float],
    state_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive per-channel dynamic thresholds/sizing hints for Boltz balancing.

    Purpose: protect hot, fast-draining channels (e.g. LOOP) by refilling earlier and
    more deeply while preserving profit/budget guards.
    """
    state_row = state_row or {}

    # Normalize profitability and velocity signals into 0..1 scores.
    roi = float(marginal_roi or 0.0)
    # roi is typically fraction (e.g. 0.1 = 10%); clamp aggressively for safety.
    roi_score = max(0.0, min(1.0, roi / 0.25))  # full score at ~25% marginal ROI

    daily_contrib = max(0.0, float(daily_contrib_est or 0.0))
    contrib_score = max(0.0, min(1.0, daily_contrib / 5000.0))  # full score at 5k sats/day

    kalman_ratio = float(state_row.get('kalman_flow_ratio', state_row.get('flow_ratio', 0.0)) or 0.0)
    kalman_velocity = float(state_row.get('kalman_velocity', state_row.get('velocity', 0.0)) or 0.0)

    # SOURCE channels (draining local) are the key loop-in protection case.
    source_signal = 1.0 if str(flow_state).lower() == 'source' else 0.0
    source_signal = max(source_signal, max(0.0, min(1.0, kalman_ratio)))

    # Positive velocity means source-ness increasing (more urgently draining) in this model.
    drain_accel_score = max(0.0, min(1.0, kalman_velocity / 0.05))  # saturate at ~0.05/day

    # Low local balance also contributes to urgency before the static threshold is crossed.
    depletion_score = max(0.0, min(1.0, (60.0 - float(local_pct)) / 40.0))

    hotness_score = max(0.0, min(1.0, 0.55 * contrib_score + 0.45 * roi_score))
    drain_score = max(0.0, min(1.0, 0.50 * source_signal + 0.30 * drain_accel_score + 0.20 * depletion_score))
    protection_score = max(0.0, min(1.0, 0.60 * hotness_score + 0.40 * drain_score))

    # Dynamic loop-in behavior: trigger earlier and refill deeper for high-score channels.
    trigger_boost = 20.0 * protection_score      # up to +20pp (40 -> 60)
    target_boost = 15.0 * protection_score       # up to +15pp (55 -> 70)
    amount_multiplier = 1.0 + (2.0 * protection_score)  # up to 3x amount cap
    cooldown_multiplier = 1.0 - (0.75 * protection_score)  # down to 25% of base cooldown

    eff_low_trigger = min(70.0, max(float(low_trigger_pct), float(low_trigger_pct) + trigger_boost))
    # keep target at least 10pp above trigger, bounded for safety
    eff_low_target = min(85.0, max(float(low_target_pct), eff_low_trigger + 10.0, float(low_target_pct) + target_boost))

    # Loop-out can also become mildly more assertive for hot profitable channels (harvest excess),
    # but keep it conservative relative to loop-in protection.
    out_adjust = 5.0 * max(0.0, min(1.0, hotness_score))
    eff_high_trigger = max(60.0, min(float(high_trigger_pct), float(high_trigger_pct) - out_adjust))
    eff_high_target = min(float(high_target_pct) + out_adjust, float(high_target_pct) + 5.0)

    return {
        'hotness_score': round(hotness_score, 4),
        'drain_score': round(drain_score, 4),
        'protection_score': round(protection_score, 4),
        'dynamic_thresholds': {
            'low_trigger_pct': round(eff_low_trigger, 2),
            'low_target_pct': round(eff_low_target, 2),
            'high_trigger_pct': round(eff_high_trigger, 2),
            'high_target_pct': round(eff_high_target, 2),
        },
        'execution_hints': {
            'amount_cap_multiplier': round(amount_multiplier, 3),
            'cooldown_multiplier': round(max(0.25, cooldown_multiplier), 3),
            'prioritize_channel_protection': protection_score >= 0.6,
        },
        'signals': {
            'contrib_score': round(contrib_score, 4),
            'roi_score': round(roi_score, 4),
            'source_signal': round(source_signal, 4),
            'drain_accel_score': round(drain_accel_score, 4),
            'depletion_score': round(depletion_score, 4),
            'kalman_flow_ratio': round(kalman_ratio, 4),
            'kalman_velocity': round(kalman_velocity, 6),
        },
    }


def _build_boltz_balance_plan(
    *,
    low_trigger_pct: float = 40.0,
    low_target_pct: float = 55.0,
    high_trigger_pct: float = 80.0,
    high_target_pct: float = 60.0,
    min_amount_sats: int = 100_000,
    max_amount_sats: int = 1_000_000,
    max_candidates: int = 20,
    only_peer_id: Optional[str] = None,
    only_channel_id: Optional[str] = None,
    require_profitable: bool = True,
    min_marginal_roi: float = 0.0,
    profit_margin_factor: float = 1.2,
    expected_horizon_days: float = 3.0,
    loop_out_currency: str = "LBTC",
    loop_in_currency: str = "LBTC",
) -> Dict[str, Any]:
    if fee_controller is None or database is None:
        return {"error": "Plugin not initialized"}

    bm = _require_boltz_manager()

    channels = fee_controller._get_channels_info()
    if not channels:
        return {"error": "No normal channels available"}

    # Refresh profitability cache on demand.
    if profitability_analyzer is not None:
        try:
            profitability_analyzer.analyze_all_channels()
        except Exception as e:
            plugin.log(f"BOLTZ_BALANCE: profitability refresh failed: {e}", level='warn')

    state_rows = {str(r.get("channel_id")): r for r in database.get_all_channel_states()}

    budget_status = bm.budget()
    pending_swaps = _boltz_pending_swap_count()

    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for channel_id, ch in channels.items():
        peer_id = str(ch.get("peer_id") or "")
        if only_channel_id and str(channel_id).replace(':', 'x') != str(only_channel_id).replace(':', 'x'):
            continue
        if only_peer_id and peer_id != only_peer_id:
            continue

        capacity_sats = int(ch.get("capacity", 0) or 0)
        if capacity_sats <= 0:
            skipped.append({"channel_id": channel_id, "peer_id": peer_id, "reason": "zero_capacity"})
            continue

        local_pct = _boltz_balance_pct_from_channel_info(ch)
        local_sats = parse_msat(ch.get("spendable_msat", 0)) // 1000
        receivable_sats = parse_msat(ch.get("receivable_msat", 0)) // 1000
        state_row = state_rows.get(channel_id, {})
        flow_state = str(state_row.get("state") or "unknown")

        prof = profitability_analyzer.get_profitability(channel_id) if profitability_analyzer is not None else None
        marginal_roi = float(getattr(prof, "marginal_roi", 0.0) or 0.0) if prof else None
        prof_class = getattr(getattr(prof, "classification", None), "value", None)
        daily_contrib_est = _boltz_channel_daily_contribution_estimate_sats(prof)

        if require_profitable:
            if prof is None:
                skipped.append({"channel_id": channel_id, "peer_id": peer_id, "reason": "no_profitability_data"})
                continue
            if marginal_roi is None or marginal_roi < float(min_marginal_roi):
                skipped.append({
                    "channel_id": channel_id,
                    "peer_id": peer_id,
                    "reason": "below_marginal_roi_threshold",
                    "marginal_roi": marginal_roi,
                    "min_marginal_roi": min_marginal_roi,
                })
                continue

        tuning = _boltz_dynamic_channel_tuning(
            local_pct=local_pct,
            low_trigger_pct=low_trigger_pct,
            low_target_pct=low_target_pct,
            high_trigger_pct=high_trigger_pct,
            high_target_pct=high_target_pct,
            flow_state=flow_state,
            daily_contrib_est=daily_contrib_est,
            marginal_roi=marginal_roi,
            state_row=state_row,
        )
        dyn = tuning.get("dynamic_thresholds", {}) if isinstance(tuning, dict) else {}
        hints = tuning.get("execution_hints", {}) if isinstance(tuning, dict) else {}
        eff_low_trigger_pct = float(dyn.get("low_trigger_pct", low_trigger_pct) or low_trigger_pct)
        eff_low_target_pct = float(dyn.get("low_target_pct", low_target_pct) or low_target_pct)
        eff_high_trigger_pct = float(dyn.get("high_trigger_pct", high_trigger_pct) or high_trigger_pct)
        eff_high_target_pct = float(dyn.get("high_target_pct", high_target_pct) or high_target_pct)

        direction = None
        target_pct = None
        target_currency = None
        severity = 0.0

        if local_pct < eff_low_trigger_pct:
            direction = "loop_in"
            target_pct = eff_low_target_pct
            target_currency = loop_in_currency
            severity = max(0.0, (eff_low_trigger_pct - local_pct) / max(eff_low_trigger_pct, 1.0))
        elif local_pct > eff_high_trigger_pct:
            direction = "loop_out"
            target_pct = eff_high_target_pct
            target_currency = loop_out_currency
            severity = max(0.0, (local_pct - eff_high_trigger_pct) / max(100.0 - eff_high_trigger_pct, 1.0))
        else:
            continue

        allowed, policy_reason = _boltz_direction_allowed_by_policy(peer_id, direction)
        if not allowed:
            skipped.append({"channel_id": channel_id, "peer_id": peer_id, "reason": policy_reason, "direction": direction})
            continue

        target_local_sats = int(capacity_sats * (float(target_pct) / 100.0))
        if direction == "loop_in":
            raw_amount = max(0, target_local_sats - local_sats)
        else:
            raw_amount = max(0, local_sats - target_local_sats)

        dynamic_amount_cap = int(max_amount_sats)
        try:
            dynamic_amount_cap = int(max(int(max_amount_sats), int(int(max_amount_sats) * float(hints.get("amount_cap_multiplier", 1.0) or 1.0))))
        except Exception:
            dynamic_amount_cap = int(max_amount_sats)
        # Safety caps: never exceed 25% of channel capacity or 5M sats in one Boltz action.
        dynamic_amount_cap = min(dynamic_amount_cap, max(1, int(capacity_sats * 0.25)), 5_000_000)
        amount_sats = max(int(min_amount_sats), min(int(dynamic_amount_cap), int(raw_amount))) if raw_amount > 0 else 0
        if amount_sats < int(min_amount_sats):
            skipped.append({
                "channel_id": channel_id,
                "peer_id": peer_id,
                "reason": "below_min_amount",
                "raw_amount_sats": raw_amount,
                "min_amount_sats": int(min_amount_sats),
                "direction": direction,
            })
            continue

        try:
            quote_resp = bm.quote(
                amount_sats=amount_sats,
                swap_type=("submarine" if direction == "loop_in" else "reverse"),
                currency=target_currency,
            )
            estimated_fee_sats = int(quote_resp.get("estimated_total_fee_sats", 0) or 0)
        except Exception as e:
            skipped.append({
                "channel_id": channel_id,
                "peer_id": peer_id,
                "direction": direction,
                "reason": f"quote_failed: {e}",
            })
            continue

        # Conservative expected uplift model (heuristic): daily contribution * imbalance severity * horizon.
        # This is intentionally cautious and is only used as a profit guard/ranking signal.
        severity_factor = max(0.1, min(1.0, severity))
        expected_gross_uplift_sats = int(max(0.0, daily_contrib_est) * float(expected_horizon_days) * severity_factor)
        required_profit_threshold_sats = int(round(estimated_fee_sats * float(profit_margin_factor)))
        passes_profit_guard = (expected_gross_uplift_sats >= required_profit_threshold_sats) if require_profitable else True
        expected_net_sats = expected_gross_uplift_sats - estimated_fee_sats

        # Additional guard for loop-in with no channel pinning support.
        non_pinned_penalty = 0.7 if direction == "loop_in" else 1.0
        risk_adjusted_net_sats = int(expected_net_sats * non_pinned_penalty)
        if require_profitable and direction == "loop_in":
            # Make loop-in more conservative because it cannot be channel-pinned with current boltzcli.
            passes_profit_guard = passes_profit_guard and (risk_adjusted_net_sats > 0)

        candidate = {
            "channel_id": channel_id,
            "peer_id": peer_id,
            "direction": direction,
            "trigger_threshold_pct": eff_low_trigger_pct if direction == "loop_in" else eff_high_trigger_pct,
            "target_pct": target_pct,
            "dynamic_tuning": tuning,
            "execution_hints": {
                **(hints if isinstance(hints, dict) else {}),
                "dynamic_amount_cap_sats": int(dynamic_amount_cap),
                "recommended_cooldown_hours": round(max(0.5, 4.0 * float((hints or {}).get("cooldown_multiplier", 1.0) or 1.0)), 2),
            },
            "local_balance_pct": round(local_pct, 2),
            "capacity_sats": capacity_sats,
            "local_sats": local_sats,
            "remote_sats": receivable_sats,
            "amount_sats": amount_sats,
            "raw_amount_sats": raw_amount,
            "flow_state": flow_state,
            "policy_gate": policy_reason,
            "profitability": None if prof is None else {
                "classification": prof_class,
                "net_profit_sats": getattr(prof, "net_profit_sats", None),
                "roi_percent": getattr(prof, "roi_percent", None),
                "marginal_roi_percent": round(getattr(prof, "marginal_roi_percent", 0.0), 2),
                "is_operationally_profitable": bool(getattr(prof, "is_operationally_profitable", False)),
                "daily_contribution_estimate_sats": int(daily_contrib_est),
            },
            "economics": {
                "estimated_swap_fee_sats": estimated_fee_sats,
                "expected_gross_uplift_sats": expected_gross_uplift_sats,
                "expected_net_sats": expected_net_sats,
                "risk_adjusted_net_sats": risk_adjusted_net_sats,
                "profit_margin_factor": profit_margin_factor,
                "passes_profit_guard": bool(passes_profit_guard),
                "loop_in_non_pinnable": direction == "loop_in",
            },
            "quote": quote_resp,
            "score": {
                "severity": round(severity, 4),
                "daily_contribution_estimate_sats": int(daily_contrib_est),
                "risk_adjusted_net_sats": risk_adjusted_net_sats,
            }
        }
        candidates.append(candidate)

    # Sort by profit-safe first, then best estimated net, then severity.
    candidates.sort(
        key=lambda c: (
            1 if c.get("economics", {}).get("passes_profit_guard") else 0,
            int(c.get("economics", {}).get("risk_adjusted_net_sats", 0) or 0),
            float(c.get("dynamic_tuning", {}).get("protection_score", 0.0) or 0.0),
            float(c.get("score", {}).get("severity", 0.0) or 0.0),
        ),
        reverse=True,
    )

    return {
        "generated_at": int(time.time()),
        "budget": budget_status,
        "pending_swap_count": pending_swaps,
        "thresholds": {
            "low_trigger_pct": low_trigger_pct,
            "low_target_pct": low_target_pct,
            "high_trigger_pct": high_trigger_pct,
            "high_target_pct": high_target_pct,
            "min_amount_sats": int(min_amount_sats),
            "max_amount_sats": int(max_amount_sats),
            "profit_margin_factor": float(profit_margin_factor),
            "expected_horizon_days": float(expected_horizon_days),
            "require_profitable": bool(require_profitable),
            "min_marginal_roi": float(min_marginal_roi),
            "loop_in_currency": str(loop_in_currency).upper(),
            "loop_out_currency": str(loop_out_currency).upper(),
        },
        "recommendations": candidates[: max(0, int(max_candidates))],
        "total_candidates": len(candidates),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
    }


@plugin.method("revenue-boltz-balance-recommendations")
def revenue_boltz_balance_recommendations(
    plugin: Plugin,
    low_trigger_pct: float = 40.0,
    low_target_pct: float = 55.0,
    high_trigger_pct: float = 80.0,
    high_target_pct: float = 60.0,
    min_amount_sats: int = 100_000,
    max_amount_sats: int = 1_000_000,
    max_candidates: int = 20,
    only_peer_id: str = None,
    only_channel_id: str = None,
    require_profitable: bool = True,
    min_marginal_roi: float = 0.0,
    profit_margin_factor: float = 1.2,
    expected_horizon_days: float = 3.0,
    loop_in_currency: str = "LBTC",
    loop_out_currency: str = "LBTC",
) -> Dict[str, Any]:
    """Recommend profit-constrained Boltz loop-in/out actions by channel balance."""
    try:
        return _build_boltz_balance_plan(
            low_trigger_pct=low_trigger_pct,
            low_target_pct=low_target_pct,
            high_trigger_pct=high_trigger_pct,
            high_target_pct=high_target_pct,
            min_amount_sats=min_amount_sats,
            max_amount_sats=max_amount_sats,
            max_candidates=max_candidates,
            only_peer_id=only_peer_id,
            only_channel_id=only_channel_id,
            require_profitable=require_profitable,
            min_marginal_roi=min_marginal_roi,
            profit_margin_factor=profit_margin_factor,
            expected_horizon_days=expected_horizon_days,
            loop_in_currency=loop_in_currency,
            loop_out_currency=loop_out_currency,
        )
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-auto-cycle-status")
def revenue_boltz_auto_cycle_status(plugin: Plugin) -> Dict[str, Any]:
    """Return scheduler status for the in-plugin Boltz auto-cycle."""
    with _boltz_auto_cycle_state_lock:
        state = dict(_boltz_auto_cycle_state)
    state.update({
        "boltz_enabled": bool(boltz_manager and getattr(boltz_manager, 'enabled', False)),
        "config": {
            "boltz_auto_cycle_enabled": bool(getattr(config, 'boltz_auto_cycle_enabled', True)) if config else None,
            "boltz_auto_cycle_interval_minutes": int(getattr(config, 'boltz_auto_cycle_interval_minutes', 15)) if config else None,
            "boltz_auto_cycle_max_actions": int(getattr(config, 'boltz_auto_cycle_max_actions', 1)) if config else None,
            "boltz_auto_cycle_startup_delay_seconds": int(getattr(config, 'boltz_auto_cycle_startup_delay_seconds', 120)) if config else None,
        },
    })
    return state


@plugin.method("revenue-boltz-auto-cycle-run-now")
def revenue_boltz_auto_cycle_run_now(plugin: Plugin, force: bool = False) -> Dict[str, Any]:
    """Trigger one immediate Boltz auto-cycle run using scheduler settings."""
    try:
        result = _run_boltz_auto_cycle_once(trigger="manual", force=bool(force))
        _boltz_auto_cycle_mark_state(last_result=result)
        return result
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-boltz-balance-cycle")
def revenue_boltz_balance_cycle(
    plugin: Plugin,
    dry_run: bool = True,
    max_actions: int = 1,
    low_trigger_pct: float = 40.0,
    low_target_pct: float = 55.0,
    high_trigger_pct: float = 80.0,
    high_target_pct: float = 60.0,
    min_amount_sats: int = 100_000,
    max_amount_sats: int = 1_000_000,
    only_peer_id: str = None,
    only_channel_id: str = None,
    require_profitable: bool = True,
    min_marginal_roi: float = 0.0,
    profit_margin_factor: float = 1.2,
    expected_horizon_days: float = 3.0,
    cooldown_hours: float = 4.0,
    allow_concurrent_swaps: bool = False,
    loop_in_currency: str = "LBTC",
    loop_out_currency: str = "LBTC",
) -> Dict[str, Any]:
    """Execute a profit-constrained Boltz balance cycle (loop-in/loop-out) with budget + cooldown guards."""
    try:
        plan = _build_boltz_balance_plan(
            low_trigger_pct=low_trigger_pct,
            low_target_pct=low_target_pct,
            high_trigger_pct=high_trigger_pct,
            high_target_pct=high_target_pct,
            min_amount_sats=min_amount_sats,
            max_amount_sats=max_amount_sats,
            max_candidates=max(max(5, int(max_actions) * 5), 20),
            only_peer_id=only_peer_id,
            only_channel_id=only_channel_id,
            require_profitable=require_profitable,
            min_marginal_roi=min_marginal_roi,
            profit_margin_factor=profit_margin_factor,
            expected_horizon_days=expected_horizon_days,
            loop_in_currency=loop_in_currency,
            loop_out_currency=loop_out_currency,
        )
    except Exception as e:
        return {"error": str(e)}

    if "error" in plan:
        return plan

    pending_swaps = int(plan.get("pending_swap_count", 0) or 0)
    if pending_swaps > 0 and not allow_concurrent_swaps:
        return {
            "status": "blocked",
            "reason": f"{pending_swaps} pending Boltz swap(s) detected",
            "plan": plan,
            "executed": [],
            "skipped": [],
        }

    recommendations = list(plan.get("recommendations", []))
    budget = plan.get("budget", {}) if isinstance(plan.get("budget"), dict) else {}
    remaining_budget = int(budget.get("remaining_24h_sats_estimate", 0) or 0)
    cooldown_seconds = max(0, int(float(cooldown_hours) * 3600))

    executed: List[Dict[str, Any]] = []
    skipped_exec: List[Dict[str, Any]] = []
    now = int(time.time())

    for rec in recommendations:
        if len(executed) >= max(0, int(max_actions)):
            break

        ch_id = str(rec.get("channel_id") or "")
        peer_id = str(rec.get("peer_id") or "")
        direction = str(rec.get("direction") or "")
        amount_sats = int(rec.get("amount_sats", 0) or 0)
        econ = rec.get("economics", {}) if isinstance(rec.get("economics"), dict) else {}
        est_fee = int(econ.get("estimated_swap_fee_sats", 0) or 0)

        if not econ.get("passes_profit_guard", False):
            skipped_exec.append({"channel_id": ch_id, "peer_id": peer_id, "reason": "profit_guard_failed", "recommendation": rec})
            continue
        if est_fee > remaining_budget:
            skipped_exec.append({
                "channel_id": ch_id,
                "peer_id": peer_id,
                "reason": "insufficient_remaining_budget",
                "estimated_fee_sats": est_fee,
                "remaining_budget_sats": remaining_budget,
                "recommendation": rec,
            })
            continue

        rec_hints = rec.get("execution_hints", {}) if isinstance(rec.get("execution_hints"), dict) else {}
        rec_cooldown_hours = rec_hints.get("recommended_cooldown_hours")
        rec_cooldown_seconds = cooldown_seconds
        try:
            if rec_cooldown_hours is not None:
                rec_cooldown_seconds = max(0, int(float(rec_cooldown_hours) * 3600))
        except Exception:
            rec_cooldown_seconds = cooldown_seconds

        with _boltz_balance_lock:
            last_ts = int(_boltz_balance_last_action.get(ch_id, 0) or 0)
        if rec_cooldown_seconds > 0 and last_ts > 0 and (now - last_ts) < rec_cooldown_seconds:
            skipped_exec.append({
                "channel_id": ch_id,
                "peer_id": peer_id,
                "reason": "cooldown_active",
                "cooldown_remaining_sec": rec_cooldown_seconds - (now - last_ts),
                "recommendation": rec,
            })
            continue

        if dry_run:
            executed.append({
                "status": "would_execute",
                "direction": direction,
                "channel_id": ch_id,
                "peer_id": peer_id,
                "amount_sats": amount_sats,
                "estimated_fee_sats": est_fee,
                "recommendation": rec,
            })
            remaining_budget = max(0, remaining_budget - est_fee)
            continue

        try:
            bm = _require_boltz_manager()
            if direction == "loop_in":
                res = bm.loop_in(amount_sats=amount_sats, channel_id=ch_id, peer_id=peer_id, currency=loop_in_currency)
            elif direction == "loop_out":
                res = bm.loop_out(amount_sats=amount_sats, channel_id=ch_id, peer_id=peer_id, currency=loop_out_currency)
            else:
                raise BoltzCliError(f"Unknown direction: {direction}")

            # Treat accepted/rejected separately.
            status = str(res.get("status") or "")
            if status in ("accepted", "rejected"):
                executed.append({
                    "status": status,
                    "direction": direction,
                    "channel_id": ch_id,
                    "peer_id": peer_id,
                    "amount_sats": amount_sats,
                    "estimated_fee_sats": est_fee,
                    "result": res,
                    "recommendation": rec,
                })
                if status == "accepted":
                    with _boltz_balance_lock:
                        _boltz_balance_last_action[ch_id] = int(time.time())
                    remaining_budget = max(0, remaining_budget - est_fee)
                else:
                    skipped_exec.append({"channel_id": ch_id, "peer_id": peer_id, "reason": "execution_rejected", "result": res})
            else:
                executed.append({
                    "status": "unknown",
                    "direction": direction,
                    "channel_id": ch_id,
                    "peer_id": peer_id,
                    "amount_sats": amount_sats,
                    "estimated_fee_sats": est_fee,
                    "result": res,
                    "recommendation": rec,
                })
        except Exception as e:
            skipped_exec.append({
                "channel_id": ch_id,
                "peer_id": peer_id,
                "reason": f"execution_failed: {e}",
                "recommendation": rec,
            })

    return {
        "status": "dry_run" if dry_run else "executed",
        "executed_count": len(executed),
        "skipped_count": len(skipped_exec),
        "remaining_budget_sats_estimate_after_cycle": remaining_budget,
        "executed": executed,
        "skipped": skipped_exec,
        "plan": plan,
        "notes": [
            "loop-out is channel-pinnable via boltzcli --chan-id",
            "loop-in (submarine) is not channel-pinnable on boltzcli v2.11.0; target channel/peer is a planning hint and should be verified post-swap",
            "profit guard uses a conservative heuristic expected-uplift estimate based on recent channel contribution and imbalance severity",
            "dynamic channel protection tuning raises loop-in trigger/target, amount cap, and cadence for fast-draining high-profit channels",
        ],
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    plugin.run()
