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
- bookkeeper plugin (built-in): Historical routing data
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import multiprocessing
import queue
import uuid
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


# =============================================================================
# PLUGIN VERSION
# =============================================================================
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
PLUGIN_VERSION = "2.1.0"


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
# pyln-client's RPC is not inherently thread-safe for concurrent calls.
# This lock serializes all RPC calls to prevent race conditions when
# multiple background loops (Fee, Flow, Rebalance) fire simultaneously.

RPC_LOCK = threading.Lock()


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


class RPCBreakerOpen(RpcError):
    """Exception raised when the circuit breaker is open for a method group."""
    def __init__(self, group, until_ts):
        self.group = group
        self.until_ts = until_ts
        until_str = datetime.fromtimestamp(until_ts).strftime('%H:%M:%S')
        # Initialize RpcError with compatible fields
        super().__init__(group, {}, f"RPC circuit breaker open for group '{group}' until {until_str}")


class RpcBroker:
    """
    A hardened RPC broker that executes lightningd RPC calls in a separate process.

    Why:
    - pyln-client RPC can hang indefinitely on certain transport / plugin interactions.
    - A thread timeout (ThreadPoolExecutor) does not stop a hung RPC call.
    - By isolating RPC calls in a subprocess, we can terminate and restart the broker
      on timeout, guaranteeing bounded waiting for callers.

    Design:
    - One broker process + one request queue + one response queue
    - Calls are serialized via an internal call lock (matches prior max_workers=1)
    - On timeout: terminate broker, recreate queues, restart broker, raise TimeoutError
    """

    def __init__(self, socket_path: str, plugin_instance: Plugin):
        self.socket_path = socket_path
        self._plugin = plugin_instance

        # Use spawn for safety (avoid forking a process after threads have started).
        self._ctx = multiprocessing.get_context("spawn")

        self._proc: Optional[multiprocessing.Process] = None
        self._req_q: Any = None
        self._resp_q: Any = None

        # Serialize calls (behaviorally equivalent to the old single-worker executor).
        self._call_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

        self.start()

    @staticmethod
    def _broker_main(socket_path: str, req_q, resp_q):
        # NOTE: Runs in a separate process.
        from pyln.client import LightningRpc, RpcError as _RpcError
        import traceback as _traceback

        rpc = LightningRpc(socket_path)

        while True:
            req = req_q.get()
            if not req:
                continue
            if req.get("op") == "stop":
                break

            req_id = req.get("id")
            kind = req.get("kind", "call")
            method = req.get("method")
            payload = req.get("payload")
            args = req.get("args") or []
            kwargs = req.get("kwargs") or {}

            try:
                if kind == "attr":
                    # E.g. listpeers(), plugin("list"), listforwards(status="settled")
                    result = getattr(rpc, method)(*args, **kwargs)
                else:
                    # Generic rpc.call(method, payload)
                    result = rpc.call(method, {} if payload is None else payload)

                resp_q.put({"id": req_id, "ok": True, "result": result})
            except _RpcError as e:
                # Serialize error details; caller reconstructs a compatible RpcError.
                resp_q.put({
                    "id": req_id,
                    "ok": False,
                    "error_type": "RpcError",
                    "error": getattr(e, "error", None),
                    "message": str(e),
                })
            except Exception as e:
                resp_q.put({
                    "id": req_id,
                    "ok": False,
                    "error_type": "Exception",
                    "message": str(e),
                    "traceback": _traceback.format_exc(),
                })

    def start(self):
        with self._lifecycle_lock:
            # Fresh queues each start to avoid stale messages after restarts.
            self._req_q = self._ctx.Queue()
            self._resp_q = self._ctx.Queue()

            self._proc = self._ctx.Process(
                target=RpcBroker._broker_main,
                args=(self.socket_path, self._req_q, self._resp_q),
                daemon=True,
                name="rpc_broker",
            )
            self._proc.start()

    def stop(self):
        with self._lifecycle_lock:
            if self._proc is None:
                return
            try:
                if self._req_q:
                    self._req_q.put_nowait({"op": "stop"})
            except Exception:
                pass

            try:
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
            except Exception:
                pass

            self._proc = None
            self._req_q = None
            self._resp_q = None

    def restart(self, reason: str):
        # Keep logs rate-limited in caller layer; here we log once per restart.
        self._plugin.log(f"RPC broker restart: {reason}", level="warn")
        self.stop()
        self.start()

    def request(self, *, kind: str, method: str, payload: Any = None,
                args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None,
                timeout: int = 15):
        """
        Perform a single RPC request through the broker.

        Raises:
            TimeoutError: if the broker does not return within timeout.
            RpcError: reconstructed from broker error payload.
        """
        if not method:
            raise RpcError("request", {}, "Empty RPC method")

        with self._call_lock:
            # Broker may have died; restart defensively.
            if self._proc is None or (hasattr(self._proc, "is_alive") and not self._proc.is_alive()):
                self.restart("broker not running")

            req_id = uuid.uuid4().hex
            req = {
                "id": req_id,
                "kind": kind,
                "method": method,
                "payload": payload,
                "args": args or [],
                "kwargs": kwargs or {},
            }

            assert self._req_q is not None and self._resp_q is not None

            self._req_q.put(req)

            try:
                resp = self._resp_q.get(timeout=timeout)
                # In normal operation (serialized), the first response is ours.
                # If we ever see mismatch (stale message), drain until match.
                while resp and resp.get("id") != req_id:
                    resp = self._resp_q.get(timeout=timeout)
            except queue.Empty:
                # Hard guarantee: kill the hung broker, restart, and surface timeout.
                self.restart(f"timeout waiting for RPC response ({timeout}s) on {method}")
                raise TimeoutError(f"RPC broker timeout on {method}")

            if resp.get("ok"):
                return resp.get("result")

            # Reconstruct a compatible RpcError in the main process.
            if resp.get("traceback"):
                self._plugin.log(
                    f"RPC broker exception in {method}: {resp.get('message')}\n{resp.get('traceback')}",
                    level="error"
                )

            err = resp.get("error")
            msg = resp.get("message") or "RPC error"
            raise RpcError(method, {} if payload is None else payload, err if err is not None else msg)


class ThreadSafeRpcProxy:
    """
    A thread-safe proxy for the plugin's RPC interface with timeouts and circuit breakers.

    Phase 1 Hardening (revised):
    - Bounded execution (RPC broker subprocess with hard timeouts)
    - Circuit Breaker (group-based cooldowns)
    - Broker restart on timeout (guarantees forward progress)
    """

    def __init__(self, broker: RpcBroker, plugin_instance: Plugin):
        self._broker = broker
        self._plugin = plugin_instance
        self._breakers: Dict[str, float] = {}
        self._log_history: Dict[Tuple[str, str], float] = {}

    def _get_group(self, method_name: str) -> str:
        """Determine method group for circuit breaking."""
        if method_name.startswith("sling-"):
            return "sling"
        if method_name.startswith("bkpr-"):
            return "bkpr"
        if method_name == "listforwards":
            return "listforwards"
        return "general"

    def _should_log(self, group: str, msg_type: str, cooldown: int = 60) -> bool:
        """Rate-limit logs to once per cooldown window."""
        now = time.time()
        key = (group, msg_type)
        if now - self._log_history.get(key, 0) > cooldown:
            self._log_history[key] = now
            return True
        return False

    def __getattr__(self, name):
        # Internal attribute access
        if name in ("_broker", "_plugin", "_breakers", "_log_history",
                    "call", "_get_group", "_should_log"):
            return super().__getattribute__(name)

        # Expose a callable wrapper matching pyln-client's LightningRpc style.
        def wrapper(*args, **kwargs):
            # For normal methods (listpeers, listchannels, plugin, etc), we treat this
            # as an attribute call and let the broker execute getattr(rpc, name)(*args, **kwargs).
            return self.call(name, list(args) if args else None, **kwargs)

        return wrapper

    def call(self, method_name: str, payload: Any = None, **kwargs):
        """
        Thread-safe wrapper for RPC calls with timeout and circuit breaker.
        """
        group = self._get_group(method_name)
        now = time.time()

        # 1. Circuit Breaker
        until = self._breakers.get(group, 0)
        if until > now:
            if self._should_log(group, "breaker_open"):
                self._plugin.log(
                    f"RPC Circuit Breaker OPEN for group '{group}' until "
                    f"{datetime.fromtimestamp(until).strftime('%H:%M:%S')}. Skipping call.",
                    level="warn",
                )
            raise RPCBreakerOpen(group, until)

        # 2. Timeouts from config
        timeout = 15
        breaker_window = 60
        if config:
            timeout = config.rpc_timeout_seconds
            breaker_window = config.rpc_circuit_breaker_seconds

        try:
            # If payload is a list, this came from an attribute-style call like
            # rpc.plugin("list") or rpc.listforwards(status="settled").
            if isinstance(payload, list) or payload is None and kwargs:
                args = payload if isinstance(payload, list) else []
                return self._broker.request(
                    kind="attr",
                    method=method_name,
                    args=args,
                    kwargs=kwargs,
                    timeout=timeout,
                )

            # Otherwise treat it as generic rpc.call(method, payload_dict).
            return self._broker.request(
                kind="call",
                method=method_name,
                payload={} if payload is None else payload,
                timeout=timeout,
            )

        except TimeoutError:
            # Trip breaker on timeout and surface RPCTimeoutError
            self._breakers[group] = time.time() + breaker_window
            self._plugin.log(
                f"RPC TIMEOUT after {timeout}s on {method_name}. "
                f"Group '{group}' breaker tripped for {breaker_window}s.",
                level="warn",
            )
            raise RPCTimeoutError(method_name)
        except RpcError:
            raise
        except Exception as e:
            self._plugin.log(f"RPC ERROR on {method_name}: {e}", level="error")
            raise

class ThreadSafePluginProxy:
    """
    A proxy for the Plugin object that provides thread-safe resilient RPC access.
    """

    def __init__(self, plugin_instance: Plugin, rpc_broker: RpcBroker):
        """Wrap the original plugin with a resilient RPC proxy."""
        self._plugin = plugin_instance
        self._rpc_broker = rpc_broker
        self.rpc = ThreadSafeRpcProxy(rpc_broker, plugin_instance)

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
rpc_broker: Optional['RpcBroker'] = None  # RPC broker subprocess
safe_plugin: Optional['ThreadSafePluginProxy'] = None  # Thread-safe plugin wrapper
policy_manager: Optional[PolicyManager] = None  # v1.4: Peer policy management
hive_bridge: Optional[HiveFeeIntelligenceBridge] = None  # v1.6: Hive intelligence

# SCID to Peer ID cache for reputation tracking
# Maps short_channel_id -> peer_id for quick lookups
_scid_to_peer_cache: Dict[str, str] = {}


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
    description='Max sats allowed to lose when rebalancing TO a Hive member (Strategic CapEx)',
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
    global flow_analyzer, fee_controller, rebalancer, clboss_manager, database, config, profitability_analyzer, capacity_planner, safe_plugin, policy_manager, hive_bridge
    
    plugin.log("Initializing cl-revenue-ops plugin...")
    
    # Build configuration from options
    config = Config(
        db_path=os.path.expanduser(options['revenue-ops-db-path']),
        flow_interval=int(options['revenue-ops-flow-interval']),
        fee_interval=int(options['revenue-ops-fee-interval']),
        rebalance_interval=int(options['revenue-ops-rebalance-interval']),
        target_flow=int(options['revenue-ops-target-flow']),
        min_fee_ppm=int(options['revenue-ops-min-fee-ppm']),
        max_fee_ppm=int(options['revenue-ops-max-fee-ppm']),
        rebalance_min_profit=int(options['revenue-ops-rebalance-min-profit']),
        flow_window_days=int(options['revenue-ops-flow-window-days']),
        clboss_enabled=options['revenue-ops-clboss-enabled'].lower() == 'true',
        rebalancer_plugin=options['revenue-ops-rebalancer'],
        daily_budget_sats=int(options['revenue-ops-daily-budget-sats']),
        min_wallet_reserve=int(options['revenue-ops-min-wallet-reserve']),
        enable_proportional_budget=options['revenue-ops-proportional-budget'].lower() == 'true',
        proportional_budget_pct=float(options['revenue-ops-proportional-budget-pct']),
        dry_run=options['revenue-ops-dry-run'].lower() == 'true',
        htlc_congestion_threshold=float(options['revenue-ops-htlc-congestion-threshold']),
        enable_reputation=options['revenue-ops-enable-reputation'].lower() == 'true',
        reputation_decay=float(options['revenue-ops-reputation-decay']),
        enable_kelly=options['revenue-ops-enable-kelly'].lower() == 'true',
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
    
    plugin.log(f"Configuration loaded: target_flow={config.target_flow}, "
               f"fee_range=[{config.min_fee_ppm}, {config.max_fee_ppm}], "
               f"dry_run={config.dry_run}")
    
    # Create thread-safe RPC proxy (Phase 5.5: High-Uptime Stability)
    # All background threads share a single RPC connection - serialize access
    # to prevent corruption from concurrent calls to lightningd

    # Phase 1: RPC Broker (subprocess) + thread-safe proxy
    rpc_socket_path = getattr(plugin.rpc, "socket_path", None)
    if not rpc_socket_path:
        # Best-effort derive from CLN init configuration if available.
        ldir = configuration.get("lightning-dir") or configuration.get("lightning_dir")
        rpcfile = configuration.get("rpc-file") or configuration.get("rpc_file")
        if ldir and rpcfile:
            rpc_socket_path = rpcfile if os.path.isabs(rpcfile) else os.path.join(ldir, rpcfile)

    if not rpc_socket_path:
        # Last-resort fallback (common default)
        ldir = configuration.get("lightning-dir") or "~/.lightning"
        rpc_socket_path = os.path.expanduser(os.path.join(ldir, "lightning-rpc"))

    rpc_broker = RpcBroker(str(rpc_socket_path), plugin)
    safe_plugin = ThreadSafePluginProxy(plugin, rpc_broker)
    plugin.log(f"RPC broker initialized (socket={rpc_socket_path})", level="info")

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
                "Dependency 'bookkeeper' not found. Using 'listforwards' fallback for flow analysis. "
                "Enable bookkeeper for accurate cost tracking.",
                level='info'
            )
        else:
            plugin.log("Dependency check: bookkeeper plugin detected")
            
    except Exception as e:
        plugin.log(f"Error checking plugin dependencies: {e}", level='warn')
        # Assume plugins are available if check fails
        config.sling_available = True
    
    
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
        
        if last_forward_ts is None:
            # Empty database - hydrate from flow_window_days ago (or 14 days default)
            hydrate_days = max(config.flow_window_days, 14)
            start_time = int(time.time()) - (hydrate_days * 86400)
            plugin.log(f"Forwards table empty. Hydrating last {hydrate_days} days of forwards...")
        else:
            # Have data - only fetch what we missed while offline
            start_time = max(0, last_forward_ts - 3600)
            plugin.log(f"Hydrating forwards since {time.strftime('%Y-%m-%d %H:%M', time.localtime(start_time))}...")
        
        # Fetch from RPC - this is the ONLY listforwards call we make
        # CLN's listforwards doesn't support 'since' natively, so we filter client-side
        result = safe_plugin.rpc.listforwards(status="settled")
        forwards_to_insert = []
        
        for fwd in result.get("forwards", []):
            received_time = fwd.get("received_time", 0)
            if received_time > start_time:
                forwards_to_insert.append({
                    'in_channel': fwd.get("in_channel", ""),
                    'out_channel': fwd.get("out_channel", ""),
                    'in_msat': fwd.get("in_msat", fwd.get("in_msatoshi", 0)),
                    'out_msat': fwd.get("out_msat", fwd.get("out_msatoshi", 0)),
                    'fee_msat': fwd.get("fee_msat", fwd.get("fee_msatoshi", 0)),
                    'resolution_time': (fwd.get("resolved_time", 0) - received_time) if fwd.get("resolved_time") else 0,
                    'received_time': received_time,
                    'resolved_time': int(fwd.get("resolved_time", 0) or 0)
                })
        
        if forwards_to_insert:
            inserted = database.bulk_insert_forwards(forwards_to_insert)
            plugin.log(f"Hydration complete: inserted {inserted} forwards into local database")
        else:
            plugin.log("Hydration complete: no new forwards to insert")
            
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
        import traceback
        plugin.log(f"Traceback: {traceback.format_exc()}", level='warn')
    
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
        hive_bridge = HiveFeeIntelligenceBridge(safe_plugin, database)
        hive_available = hive_bridge.is_available()

        if config.hive_enabled == 'true' and not hive_available:
            # Required mode but hive not available - warn but continue
            plugin.log("=" * 60, level='warn')
            plugin.log("WARNING: hive-enabled=true but hive mode not active!", level='warn')
            plugin.log("Possible reasons:", level='warn')
            plugin.log("  - cl-hive plugin not loaded", level='warn')
            plugin.log("  - Node not yet a hive member (open channel to a member)", level='warn')
            plugin.log("Hive features will be unavailable until membership established", level='warn')
            plugin.log("Plugin will continue in standalone mode", level='warn')
            plugin.log("=" * 60, level='warn')
        elif hive_available:
            plugin.log("=" * 60)
            plugin.log("HIVE MODE ACTIVE: Authenticated hive member")
            plugin.log("Hive features enabled:")
            plugin.log("  - Coordinated fee recommendations")
            plugin.log("  - Fleet-wide fee intelligence")
            plugin.log("  - Rebalancing conflict detection")
            plugin.log("  - Collective defense against drain attacks")
            plugin.log("  - Anticipatory liquidity predictions")
            plugin.log("=" * 60)
        else:
            plugin.log("=" * 60)
            plugin.log("STANDALONE MODE: Not a hive member (hive-enabled=auto)")
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
            
            # Calculate +/- 20% jitter
            jitter_seconds = int(config.flow_interval * 0.2)
            sleep_time = config.flow_interval + random.randint(-jitter_seconds, jitter_seconds)
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
            except (RPCTimeoutError, RPCBreakerOpen) as e:
                plugin.log(f"RPC degraded in fee adjustment: {e}. Skipping this cycle.", level='warn')
            except Exception as e:
                plugin.log(f"Error in fee adjustment: {e}", level='error')
            
            # Calculate +/- 20% jitter
            jitter_seconds = int(config.fee_interval * 0.2)
            sleep_time = config.fee_interval + random.randint(-jitter_seconds, jitter_seconds)
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
            
            # Calculate +/- 20% jitter
            jitter_seconds = int(config.rebalance_interval * 0.2)
            sleep_time = config.rebalance_interval + random.randint(-jitter_seconds, jitter_seconds)
            plugin.log(f"Rebalance check sleeping for {sleep_time}s")
            
            # Interruptible sleep: wait for timeout OR shutdown signal
            if shutdown_event.wait(sleep_time):
                plugin.log("Rebalance check loop stopping due to shutdown signal")
                break
    
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
            import traceback
            plugin.log(f"Traceback: {traceback.format_exc()}", level='warn')

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
        database.record_financial_snapshot(
            local_balance_sats=tlv_data.get("total_local_sats", 0),
            remote_balance_sats=tlv_data.get("total_remote_sats", 0),
            onchain_sats=tlv_data.get("onchain_sats", 0),
            capacity_sats=tlv_data.get("total_capacity_sats", 0),
            revenue_accumulated_sats=revenue_sats,
            rebalance_cost_accumulated_sats=lifetime_stats.get("total_rebalance_cost_sats", 0),
            channel_count=tlv_data.get("channel_count", 0)
        )

        plugin.log(
            f"Financial snapshot recorded: TLV={tlv_data.get('tlv_sats', 0)} sats, "
            f"channels={tlv_data.get('channel_count', 0)}"
        )

    # =========================================================================
    # SIGNAL HANDLER: Clean Shutdown on `lightning-cli plugin stop`
    # =========================================================================
    def handle_shutdown_signal(signum, frame):
        """
        Handle SIGTERM for graceful shutdown.
        
        CLN sends SIGTERM when `lightning-cli plugin stop cl-revenue-ops` is called.
        This handler sets the shutdown_event, causing all background loops to exit
        immediately instead of waiting for their sleep timers.
        """
        plugin.log("Received SIGTERM, initiating clean shutdown...", level='info')
        shutdown_event.set()
        
        # Stop active rebalance jobs to prevent phantom spending
        if rebalancer and rebalancer.job_manager:
            try:
                stopped = rebalancer.job_manager.stop_all_jobs(reason="plugin_shutdown")
                if stopped > 0:
                    plugin.log(f"Stopped {stopped} active rebalance jobs", level='info')
            except Exception as e:
                plugin.log(f"Error stopping rebalance jobs: {e}", level='warn')
        
        # Stop RPC broker subprocess
        if rpc_broker:
            try:
                rpc_broker.stop()
            except Exception:
                pass

        # MAJOR-11 FIX: Clean up database connections on shutdown
        if database:
            try:
                database.close_all_connections()
            except Exception as e:
                plugin.log(f"Error closing database: {e}", level='warn')

    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    # =========================================================================
    # STARTUP HYGIENE: Clean up orphan jobs from previous runs
    # =========================================================================
    if rebalancer and config.sling_available:
        try:
            rebalancer.job_manager.cleanup_orphans()
        except Exception as e:
            plugin.log(f"Warning: Could not clean up orphan jobs: {e}", level='warn')

        # PHASE 6: Sync peer exclusions with sling on startup
        try:
            rebalancer.job_manager.sync_peer_exclusions(policy_manager)
        except Exception as e:
            plugin.log(f"Warning: Could not sync peer exclusions: {e}", level='warn')
    
    # Start background threads (daemon=True so they don't block shutdown)
    threading.Thread(target=flow_analysis_loop, daemon=True, name="flow-analysis").start()
    threading.Thread(target=fee_adjustment_loop, daemon=True, name="fee-adjustment").start()
    threading.Thread(target=rebalance_check_loop, daemon=True, name="rebalance-check").start()
    threading.Thread(target=snapshot_peers_delayed, daemon=True, name="startup-snapshot").start()
    threading.Thread(target=financial_snapshot_loop, daemon=True, name="financial-snapshot").start()

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
        listfunds = plugin.rpc.listfunds()
        onchain_sats = sum(
            (int(str(o.get("amount_msat", "0")).replace("msat", "")) // 1000)
            for o in listfunds.get("outputs", [])
            if o.get("status") == "confirmed"
        )
        channel_sats = sum(
            (int(str(c.get("our_amount_msat", "0")).replace("msat", "")) // 1000)
            for c in listfunds.get("channels", [])
        )
        total_liquid = onchain_sats + channel_sats

        # Get detailed spending info (Issue #23 + #24)
        spend_info = database.get_daily_rebalance_spend() if database else {}
        daily_spent = spend_info.get('total_spent_sats', 0)
        daily_reserved = spend_info.get('total_reserved_sats', 0)
        stale_count = spend_info.get('stale_reservations', 0)
        daily_budget = cfg.daily_budget_sats
        budget_remaining = daily_budget - daily_spent - daily_reserved

        result["capital_controls"] = {
            "onchain_sats": onchain_sats,
            "channel_sats": channel_sats,
            "total_liquid_sats": total_liquid,
            "wallet_reserve_sats": cfg.min_wallet_reserve,
            "reserve_ok": total_liquid >= cfg.min_wallet_reserve,
            "daily_budget_sats": daily_budget,
            "daily_spent_sats": daily_spent,
            "daily_reserved_sats": daily_reserved,
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
                f"Daily budget exhausted: spent {daily_spent} + reserved {daily_reserved} of {daily_budget}"
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
    from .modules.fee_controller import HillClimbingFeeController
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
        skip_reason = None
        status = "ready"

        if is_sleeping:
            mins_until_wake = (sleep_until - now) // 60
            skip_reason = f"SLEEPING (wake in {mins_until_wake} min)"
            status = "sleeping"
            result["summary"]["sleeping"] += 1
        elif hours_since_update < 1.0:
            skip_reason = f"WAITING_TIME ({hours_since_update:.2f}h < 1h min)"
            status = "waiting_time"
            result["summary"]["waiting_time"] += 1
        elif forward_count < min_forwards:
            skip_reason = f"WAITING_FORWARDS ({forward_count}/{min_forwards} forwards)"
            status = "waiting_forwards"
            result["summary"]["waiting_forwards"] += 1
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

    # Basic SCID or PeerID format check (simple regex-less check)
    if not (":" in channel_id or "x" in channel_id or len(channel_id) == 66):
        return {"status": "error", "error": "Invalid channel_id or node_id format"}

    # 2. Force Gates
    if not force:
        if fee_ppm < config.min_fee_ppm or fee_ppm > config.max_fee_ppm:
            return {
                "status": "error", 
                "error": f"Fee {fee_ppm} is outside configured range [{config.min_fee_ppm}, {config.max_fee_ppm}]. Use force=true to override."
            }
    
    try:
        result = fee_controller.set_channel_fee(channel_id, fee_ppm, manual=True)
        return {"status": "success", "channel": channel_id, "new_fee_ppm": fee_ppm, **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


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
            return {"status": "error", **result}
        # Check the success field from execute_rebalance
        if result.get("success") is False:
            return {"status": "error", **result}
        return {"status": "success", **result}
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
            all_results = profitability_analyzer.analyze_all_channels()

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
                    "flow_profile": flow_profile
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
                   fee_ppm: int = None, tag: str = None, **kwargs) -> Dict[str, Any]:
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
            policy = policy_manager.get_policy(peer_id)
            return {"policy": policy.to_dict()}
        
        elif action == "set":
            if not peer_id:
                return {"error": "Usage: revenue-policy set <peer_id> [strategy=X] [rebalance=X] [fee_ppm=N]"}
            
            # Set policy with provided options
            policy = policy_manager.set_policy(
                peer_id=peer_id,
                strategy=strategy,
                rebalance_mode=rebalance,
                fee_ppm_target=fee_ppm
            )
            
            return {
                "status": "success",
                "policy": policy.to_dict(),
                "message": f"Policy updated for peer {peer_id[:12]}..."
            }
        
        elif action == "delete":
            if not peer_id:
                return {"error": "Usage: revenue-policy delete <peer_id>"}
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
            policy = policy_manager.add_tag(peer_id, tag)
            return {
                "status": "success",
                "peer_id": peer_id,
                "tags": policy.tags
            }
        
        elif action == "untag":
            if not peer_id or not tag:
                return {"error": "Usage: revenue-policy untag <peer_id> <tag>"}
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
            # cl-hive integration: Expose closure/splice costs for capacity planning
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
                "opex_sats": pnl.get("opex_sats", 0)
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
    global database, safe_plugin

    if database is None:
        return {"error": "Database not initialized"}

    if safe_plugin is None:
        return {"error": "Plugin not initialized"}

    try:
        from modules.portfolio_optimizer import PortfolioOptimizer

        # Get channel data
        channels = safe_plugin.rpc.listpeerchannels().get("channels", [])

        # Get forwards from last 14 days
        import time
        now = int(time.time())
        cutoff = now - (14 * 86400)

        # Try bookkeeper first, fall back to listforwards
        try:
            income = safe_plugin.rpc.call("bkpr-listincome", {"consolidate_fees": False})
            forwards = []
            for event in income.get("income_events", []):
                if event.get("tag") == "routed":
                    forwards.append({
                        "out_channel": event.get("outpoint", "").split(":")[0] if ":" in event.get("outpoint", "") else event.get("account", ""),
                        "received_time": event.get("timestamp", 0),
                        "fee_msat": event.get("credit_msat", 0),
                        "out_msat": event.get("debit_msat", 0)
                    })
        except Exception:
            # Fall back to listforwards
            fwd_result = safe_plugin.rpc.listforwards(status="settled")
            forwards = fwd_result.get("forwards", [])

        # Get Kalman flow states if available
        flow_states = {}
        try:
            states = database.get_all_channel_states()
            for state in states:
                if state.get("kalman_state"):
                    import json
                    ks = json.loads(state["kalman_state"])
                    flow_states[state["channel_id"]] = ks
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
        plugin.log(f"Error in portfolio analysis: {e}", level='error')
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


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
    global database, safe_plugin

    if database is None:
        return {"error": "Database not initialized"}

    if safe_plugin is None:
        return {"error": "Plugin not initialized"}

    try:
        from modules.portfolio_optimizer import PortfolioOptimizer

        # Get channel data
        channels = safe_plugin.rpc.listpeerchannels().get("channels", [])

        # Get forwards
        import time
        now = int(time.time())

        try:
            income = safe_plugin.rpc.call("bkpr-listincome", {"consolidate_fees": False})
            forwards = []
            for event in income.get("income_events", []):
                if event.get("tag") == "routed":
                    forwards.append({
                        "out_channel": event.get("outpoint", "").split(":")[0] if ":" in event.get("outpoint", "") else event.get("account", ""),
                        "received_time": event.get("timestamp", 0),
                        "fee_msat": event.get("credit_msat", 0),
                        "out_msat": event.get("debit_msat", 0)
                    })
        except Exception:
            fwd_result = safe_plugin.rpc.listforwards(status="settled")
            forwards = fwd_result.get("forwards", [])

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
                scid = ch.get('short_channel_id', '').replace(':', 'x')
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
                scid = ch.get('short_channel_id', '').replace(':', 'x')
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
    
    For now, we just let it pass through - the periodic analysis from bookkeeper
    is sufficient for initial implementation.
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
    global _scid_to_peer_cache
    
    # Check cache first
    if scid in _scid_to_peer_cache:
        return _scid_to_peer_cache[scid]
    
    # Cache miss - refresh cache from listpeerchannels
    # Use safe_plugin for thread-safe RPC access
    try:
        result = safe_plugin.rpc.listpeerchannels()
        for channel in result.get("channels", []):
            channel_scid = channel.get("short_channel_id") or channel.get("channel_id")
            peer_id = channel.get("peer_id")
            if channel_scid and peer_id:
                _scid_to_peer_cache[channel_scid] = peer_id
        
        # Try again after refresh
        return _scid_to_peer_cache.get(scid)
    except RpcError as e:
        plugin.log(f"Error resolving SCID {scid} to peer: {e}", level='warn')
        return None
        

def _parse_msat(msat_val: Any) -> int:
    """
    Safely convert msat values to integers.
    Handles '1000msat' strings, raw integers, Millisatoshi objects, and plain numeric strings.
    """
    if msat_val is None:
        return 0
    if hasattr(msat_val, 'millisatoshis'):
        return int(msat_val.millisatoshis)
    if isinstance(msat_val, int):
        return msat_val
    if isinstance(msat_val, str):
        # Strip suffix if present
        if msat_val.endswith('msat'):
            clean_val = msat_val[:-4]
        else:
            clean_val = msat_val
            
        try:
            return int(clean_val)
        except ValueError:
            return 0
    return 0


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
    in_channel = forward_event.get("in_channel")
    
    # Normalize SCID: replace colons with 'x' for consistency
    if in_channel:
        in_channel = in_channel.replace(':', 'x')
    
    # Track peer reputation for all forward outcomes
    if in_channel:
        peer_id = _resolve_scid_to_peer(in_channel)
        if peer_id:
            if status == "settled":
                # Success - increment success count
                database.update_peer_reputation(peer_id, is_success=True)
            elif status in ("failed", "local_failed"):
                # Failure - increment failure count
                database.update_peer_reputation(peer_id, is_success=False)

                # Report failure to cl-hive for pheromone evaporation (Yield Optimization Phase 2)
                # Failed forwards cause pheromone to evaporate, triggering fee exploration
                if hive_bridge:
                    try:
                        out_channel = forward_event.get("out_channel")
                        if out_channel:
                            out_channel = out_channel.replace(':', 'x')
                            out_peer_id = _resolve_scid_to_peer(out_channel)
                            if out_peer_id:
                                # Report failure with 0 amount to trigger evaporation
                                hive_bridge.report_routing_outcome(
                                    channel_id=out_channel,
                                    peer_id=out_peer_id,
                                    fee_ppm=0,  # Unknown for failures
                                    success=False,
                                    amount_sats=0,
                                    source=peer_id,
                                    destination=out_peer_id
                                )
                    except Exception as e:
                        plugin.log(f"FORWARD_EVENT: Hive failure report failed: {e}", level="debug")

    # Record successful forwards for flow metrics
    if status == "settled":
        out_channel = forward_event.get("out_channel")
        if out_channel:
            out_channel = out_channel.replace(':', 'x')
            
        # CLN v23.05+ uses in_msat/out_msat/fee_msat; older versions used *_msatoshi
        in_msat = _parse_msat(forward_event.get("in_msat", forward_event.get("in_msatoshi", 0)))
        out_msat = _parse_msat(forward_event.get("out_msat", forward_event.get("out_msatoshi", 0)))
        fee_msat = _parse_msat(forward_event.get("fee_msat", forward_event.get("fee_msatoshi", 0)))
        
        # Calculate resolution duration (Risk Premium tracking)
        # durations in CLN are usually in seconds (float)
        received_time = forward_event.get("received_time", 0)
        resolved_time = forward_event.get("resolved_time", 0)
        resolution_duration = resolved_time - received_time if resolved_time > 0 else 0
        
        database.record_forward(in_channel, out_channel, in_msat, out_msat, fee_msat, int(received_time or 0), int(resolved_time or 0), resolution_duration)

        # Report routing outcome to cl-hive for stigmergic learning (Yield Optimization Phase 2)
        # This enables pheromone-based fee learning and fleet coordination
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
                        source=in_peer_id,  # Where payment came from
                        destination=out_peer_id  # Where it went
                    )
            except Exception as e:
                # Don't let hive reporting failures affect core functionality
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
    peer_id = event.get('peer_id')
    channel_id = event.get('channel_id')
    new_state = event.get('new_state', '')
    old_state = event.get('old_state', '')
    cause = event.get('cause', 'unknown')

    if not channel_id:
        plugin.log(f"Channel state change - no channel_id in event: {event}", level='warn')
        return

    # Normalize channel_id format
    channel_id = channel_id.replace(':', 'x')

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

        # Call cl-hive's channel-closed notification with full data
        result = safe_plugin.rpc.call("hive-channel-closed", {
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

        if result.get("action") == "notified_hive":
            plugin.log(
                f"Notified cl-hive of closure: {channel_id} by {closer} "
                f"(pnl={net_pnl_sats}, forwards={forward_count})",
                level='info'
            )
            return True

        return False

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

        # Call cl-hive's channel-opened notification
        result = safe_plugin.rpc.call("hive-channel-opened", {
            "peer_id": peer_id,
            "channel_id": channel_id,
            "opener": opener,
            "capacity_sats": capacity_sats,
            "our_funding_sats": our_funding_sats,
            "their_funding_sats": their_funding_sats
        })

        if result.get("action") == "notified_hive":
            plugin.log(
                f"Notified cl-hive of channel open: {channel_id} with {peer_id[:16]}... ({opener})",
                level='info'
            )
            return True

        return False

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
        opener = 'unknown'
        if 'DUALOPEND' in old_state:
            opener = 'local'  # We typically initiate dual-funded opens
        elif cause == 'remote':
            opener = 'remote'
        elif cause == 'user':
            opener = 'local'
        else:
            # Try to determine from channel info
            try:
                channels = safe_plugin.rpc.call("listpeerchannels", {"id": peer_id})
                for ch in channels.get('channels', []):
                    scid = ch.get('short_channel_id', '').replace(':', 'x')
                    if scid == channel_id:
                        opener = ch.get('opener', 'unknown')
                        break
            except Exception:
                pass

        # Get channel details
        capacity_sats = 0
        our_funding_sats = 0
        their_funding_sats = 0

        try:
            channels = safe_plugin.rpc.call("listpeerchannels", {"id": peer_id})
            for ch in channels.get('channels', []):
                scid = ch.get('short_channel_id', '').replace(':', 'x')
                if scid == channel_id:
                    capacity_sats = ch.get('total_msat', 0) // 1000
                    our_funding_sats = ch.get('funding', {}).get('local_funds_msat', 0) // 1000
                    their_funding_sats = ch.get('funding', {}).get('remote_funds_msat', 0) // 1000

                    # Also get opener if we didn't determine it yet
                    if opener == 'unknown':
                        opener = ch.get('opener', 'unknown')
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

                # Ensure values are numeric
                if not isinstance(credit_msat, (int, float)):
                    credit_msat = 0
                if not isinstance(debit_msat, (int, float)):
                    debit_msat = 0

                fee_msat = abs(int(credit_msat) or int(debit_msat))
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
                    if ch.get('short_channel_id', '').replace(':', 'x') == channel_id:
                        capacity_sats = ch.get('total_msat', 0) // 1000
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
        import traceback
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
                        scid = ch.get('short_channel_id', '').replace(':', 'x')
                        if scid == channel_id:
                            new_capacity = ch.get('total_msat', 0) // 1000
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
        import traceback
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

        for event in reversed(all_events):  # Process oldest to newest
            # Security: Type check each event is a dict
            if not isinstance(event, dict):
                continue

            event_type = event.get('type', '')
            tag = str(event.get('tag', '')).lower()  # Ensure tag is string

            # Look for splice-related tags
            # Note: CLN bookkeeper may use tags like 'splice', 'splice_in', 'splice_out'
            if 'splice' in tag:
                splice_txid = event.get('txid')

                # Security: Type check credit/debit values before arithmetic
                credit = event.get('credit_msat', 0)
                debit = event.get('debit_msat', 0)

                # Ensure values are numeric
                if not isinstance(credit, (int, float)):
                    credit = 0
                if not isinstance(debit, (int, float)):
                    debit = 0

                splice_amount = (int(credit) - int(debit)) // 1000  # Convert to sats

            # Accumulate on-chain fees for splice
            if event_type == 'onchain_fee' and 'splice' in tag:
                # Security: Type check fee values
                credit_msat = event.get('credit_msat', 0)
                debit_msat = event.get('debit_msat', 0)

                if not isinstance(credit_msat, (int, float)):
                    credit_msat = 0
                if not isinstance(debit_msat, (int, float)):
                    debit_msat = 0

                fee_msat = abs(int(credit_msat) or int(debit_msat))
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
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    plugin.run()