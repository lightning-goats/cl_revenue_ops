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
import sys
import time
import json
import random
import sqlite3
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path
import concurrent.futures

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
from modules.metrics import PrometheusExporter, MetricNames, METRIC_HELP
from modules.policy_manager import PolicyManager, FeeStrategy, RebalanceMode, PeerPolicy


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
metrics_exporter: Optional[PrometheusExporter] = None
rpc_broker: Optional['RpcBroker'] = None  # RPC broker subprocess
safe_plugin: Optional['ThreadSafePluginProxy'] = None  # Thread-safe plugin wrapper
policy_manager: Optional[PolicyManager] = None  # v1.4: Peer policy management

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
    default='false',
    description='If true, scale daily budget based on 24h revenue (default: false)'
)

plugin.add_option(
    name='revenue-ops-proportional-budget-pct',
    default='0.05',
    description='Percentage of 24h revenue to use as budget when proportional budget enabled (default: 0.05 = 5%)'
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
    name='revenue-ops-enable-prometheus',
    default='false',
    description='If true, start Prometheus metrics exporter HTTP server (default: false)'
)

plugin.add_option(
    name='revenue-ops-prometheus-port',
    default='9800',
    description='Port for Prometheus HTTP metrics server (default: 9800)'
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
    5. Start Prometheus metrics exporter (if enabled)
    """
    global flow_analyzer, fee_controller, rebalancer, clboss_manager, database, config, profitability_analyzer, capacity_planner, metrics_exporter, safe_plugin, policy_manager
    
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
        enable_prometheus=options['revenue-ops-enable-prometheus'].lower() == 'true',
        prometheus_port=int(options['revenue-ops-prometheus-port']),
        enable_kelly=options['revenue-ops-enable-kelly'].lower() == 'true',
        kelly_fraction=float(options['revenue-ops-kelly-fraction']),
        # Phase 7 options (v1.3.0)
        enable_vegas_reflex=options['revenue-ops-vegas-reflex'].lower() == 'true',
        vegas_decay_rate=float(options['revenue-ops-vegas-decay']),
        enable_scarcity_pricing=options['revenue-ops-scarcity-pricing'].lower() == 'true',
        scarcity_threshold=float(options['revenue-ops-scarcity-threshold']),
        rpc_timeout_seconds=int(options['revenue-ops-rpc-timeout-seconds']),
        rpc_circuit_breaker_seconds=int(options['revenue-ops-rpc-circuit-breaker-seconds'])
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
    
    # Initialize Prometheus metrics exporter (Phase 2: Observability)
    if config.enable_prometheus:
        metrics_exporter = PrometheusExporter(port=config.prometheus_port, plugin=safe_plugin)
        if not metrics_exporter.start_server():
            plugin.log("Prometheus metrics disabled due to server startup failure", level='warn')
            metrics_exporter = None
    else:
        metrics_exporter = None
        plugin.log("Prometheus metrics exporter disabled by configuration")
    
    # Initialize clboss manager (handles unmanage commands)
    clboss_manager = ClbossManager(safe_plugin, config)
    
    # Initialize policy manager (v1.4: Policy-Driven Architecture)
    policy_manager = PolicyManager(database, safe_plugin)
    plugin.log("PolicyManager initialized for peer-level fee/rebalance policies")
    
    # Initialize profitability analyzer (with metrics exporter)
    profitability_analyzer = ChannelProfitabilityAnalyzer(safe_plugin, config, database, metrics_exporter)
    
    # Initialize analysis modules with profitability analyzer and metrics exporter
    flow_analyzer = FlowAnalyzer(safe_plugin, config, database)
    capacity_planner = CapacityPlanner(safe_plugin, config, profitability_analyzer, flow_analyzer)
    fee_controller = PIDFeeController(safe_plugin, config, database, clboss_manager, policy_manager, profitability_analyzer, metrics_exporter)
    rebalancer = EVRebalancer(safe_plugin, config, database, clboss_manager, policy_manager, metrics_exporter)
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
                
                # Export peer reputation metrics (Phase 2: Observability)
                if metrics_exporter and database:
                    update_peer_reputation_metrics()
                
                # Update last run timestamp for health monitoring
                if metrics_exporter:
                    metrics_exporter.set_gauge(
                        MetricNames.SYSTEM_LAST_RUN_TIMESTAMP,
                        int(time.time()),
                        {"task": "flow"},
                        METRIC_HELP.get(MetricNames.SYSTEM_LAST_RUN_TIMESTAMP, "")
                    )
                    
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
                
                # Update last run timestamp for health monitoring
                if metrics_exporter:
                    metrics_exporter.set_gauge(
                        MetricNames.SYSTEM_LAST_RUN_TIMESTAMP,
                        int(time.time()),
                        {"task": "fee"},
                        METRIC_HELP.get(MetricNames.SYSTEM_LAST_RUN_TIMESTAMP, "")
                    )
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
                
                # Update last run timestamp for health monitoring
                if metrics_exporter:
                    metrics_exporter.set_gauge(
                        MetricNames.SYSTEM_LAST_RUN_TIMESTAMP,
                        int(time.time()),
                        {"task": "rebalance"},
                        METRIC_HELP.get(MetricNames.SYSTEM_LAST_RUN_TIMESTAMP, "")
                    )
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
        
        # Stop Prometheus server if running
        if metrics_exporter:
            try:
                metrics_exporter.stop_server()
            except Exception as e:
                plugin.log(f"Error stopping metrics server: {e}", level='warn')

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


def update_peer_reputation_metrics():
    """
    Export peer reputation data to Prometheus metrics.
    
    Phase 2: Observability - Track peer reliability scores.
    
    Called periodically from flow_analysis_loop to update:
    - cl_revenue_peer_reputation_score: Success rate (0.0 to 1.0)
    - cl_revenue_peer_success_count: Total successful forwards
    - cl_revenue_peer_failure_count: Total failed forwards
    """
    if database is None or metrics_exporter is None:
        return
    
    try:
        reputations = database.get_all_peer_reputations()
        
        for rep in reputations:
            peer_id = rep.get('peer_id', '')
            if not peer_id:
                continue
            
            labels = {"peer_id": peer_id}
            
            # Gauge: Reputation score (success rate 0.0 to 1.0)
            metrics_exporter.set_gauge(
                MetricNames.PEER_REPUTATION_SCORE,
                rep.get('score', 1.0),
                labels,
                METRIC_HELP.get(MetricNames.PEER_REPUTATION_SCORE, "")
            )
            
            # Gauge: Success count (using gauge so we can see current state)
            metrics_exporter.set_gauge(
                MetricNames.PEER_SUCCESS_COUNT,
                rep.get('successes', 0),
                labels,
                METRIC_HELP.get(MetricNames.PEER_SUCCESS_COUNT, "")
            )
            
            # Gauge: Failure count
            metrics_exporter.set_gauge(
                MetricNames.PEER_FAILURE_COUNT,
                rep.get('failures', 0),
                labels,
                METRIC_HELP.get(MetricNames.PEER_FAILURE_COUNT, "")
            )
        
        plugin.log(f"Updated Prometheus metrics for {len(reputations)} peer reputations", level='debug')
        
    except Exception as e:
        plugin.log(f"Error updating peer reputation metrics: {e}", level='warn')


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

        daily_spent = database.get_daily_rebalance_spend() if database else 0
        daily_budget = cfg.rebalance_budget_sats
        budget_remaining = daily_budget - daily_spent

        result["capital_controls"] = {
            "onchain_sats": onchain_sats,
            "channel_sats": channel_sats,
            "total_liquid_sats": total_liquid,
            "wallet_reserve_sats": cfg.wallet_reserve_sats,
            "reserve_ok": total_liquid >= cfg.wallet_reserve_sats,
            "daily_budget_sats": daily_budget,
            "daily_spent_sats": daily_spent,
            "budget_remaining_sats": budget_remaining,
            "budget_ok": budget_remaining > 0
        }

        if total_liquid < cfg.wallet_reserve_sats:
            result["rejection_reasons"].append(
                f"Wallet reserve violated: {total_liquid} < {cfg.wallet_reserve_sats}"
            )
        if budget_remaining <= 0:
            result["rejection_reasons"].append(
                f"Daily budget exhausted: spent {daily_spent} of {daily_budget}"
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
                return {
                    "channel_id": channel_id,
                    "profitability": {
                        "total_costs_sats": result.costs.total_cost_sats,
                        "total_revenue_sats": result.revenue.fees_earned_sats,
                        "net_profit_sats": result.net_profit_sats,
                        "roi_percentage": round(result.roi_percent, 2),
                        "profitability_class": result.classification.value,
                        "days_active": result.days_open,
                        "volume_routed_sats": result.revenue.volume_routed_sats,
                        "forward_count": result.revenue.forward_count,
                        "fee_multiplier": profitability_analyzer.get_fee_multiplier(channel_id)
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
                "zombie": []
            }
            total_profit = 0
            total_revenue = 0
            total_costs = 0
            
            for ch_id, result in all_results.items():
                channel_summary = {
                    "channel_id": ch_id,
                    "net_profit_sats": result.net_profit_sats,
                    "roi_percentage": round(result.roi_percent, 2),
                    "days_active": result.days_open
                }
                summary[result.classification.value].append(channel_summary)
                total_profit += result.net_profit_sats
                total_revenue += result.revenue.fees_earned_sats
                total_costs += result.costs.total_cost_sats
            
            return {
                "summary": {
                    "total_channels": len(all_results),
                    "profitable_count": len(summary["profitable"]),
                    "break_even_count": len(summary["break_even"]),
                    "underwater_count": len(summary["underwater"]),
                    "zombie_count": len(summary["zombie"]),
                    "total_profit_sats": total_profit,
                    "total_revenue_sats": total_revenue,
                    "total_costs_sats": total_costs,
                    "overall_roi_pct": round((total_profit / total_costs * 100) if total_costs > 0 else 0, 2)
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
                   fee_ppm: int = None, tag: str = None) -> Dict[str, Any]:
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
    
    Examples:
      lightning-cli revenue-policy set 02abc... strategy=static fee_ppm=500
      lightning-cli revenue-policy set 02abc... strategy=passive rebalance=disabled
      lightning-cli revenue-policy tag 02abc... whale
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
        
        else:
            return {"error": f"Unknown action: {action}. Use 'list', 'get', 'set', 'delete', 'tag', 'untag', or 'find'"}
    
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
    
    Report Types:
      summary   - Overall node P&L, active channels, warnings
      peer      - Specific peer metrics (profitability, flow, policy)
      hive      - List of peers with HIVE strategy (for cl-hive)
      policies  - Statistics on policy distribution
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
        
        else:
            return {"error": f"Unknown report type: {report_type}. Use 'summary', 'peer', 'hive', or 'policies'"}
    
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
            
            # Real-time metrics update (Phase 2: Observability)
            # Update Prometheus metrics immediately after DB update
            if metrics_exporter:
                rep = database.get_peer_reputation(peer_id)
                labels = {"peer_id": peer_id}
                
                metrics_exporter.set_gauge(
                    MetricNames.PEER_REPUTATION_SCORE,
                    rep.get('score', 1.0),
                    labels
                )
                metrics_exporter.set_gauge(
                    MetricNames.PEER_SUCCESS_COUNT,
                    rep.get('successes', 0),
                    labels
                )
                metrics_exporter.set_gauge(
                    MetricNames.PEER_FAILURE_COUNT,
                    rep.get('failures', 0),
                    labels
                )
    
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    plugin.run()