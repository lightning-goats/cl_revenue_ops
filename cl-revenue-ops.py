#!/usr/bin/env python3
"""
cl-revenue-ops: A Revenue Operations Plugin for Core Lightning

This plugin acts as a "Revenue Operations" layer for Lightning nodes,
making data-driven decisions to maximize profitability based on economic
principles rather than heuristics.

Dependencies:
- pyln-client: Core Lightning plugin framework
- bookkeeper plugin (built-in): On-chain cost attribution (opens/closes) and accounting-grade events
- Local forwards table (SQLite): Routing history for flow analysis (hydrated once on startup)
- Native rebalancer: prices pairs with askrene getroutes and executes explicit sendpay routes

Author: Lightning Goats Team
License: BSD 3-Clause
"""

import copy
import os
import time
import json
import random
import threading
import signal
import atexit
import re
import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import traceback
from pyln.client import Plugin, RpcError

# Import our modules
from modules.flow_analysis import FlowAnalyzer, ChannelState
from modules import flow_analysis as flow_analysis_mod
from modules.fee_controller import FeeController
from modules.rebalancer import EVRebalancer
from modules.config import Config
from modules.database import Database
from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
from modules.capacity_planner import CapacityPlanner
from modules.hive_hints import HiveHintAdapter
from modules.hive_router import HiveRouter
from modules.hive_runtime import refresh_hive_runtime
from modules.policy_manager import (
    PolicyManager,
    FeeStrategy,
    RebalanceMode,
    PeerPolicy,
    READ_ONLY_POLICY_ACTIONS,
    TACTICAL_POLICY_ACTIONS,
)
from modules.boltz_manager import BoltzCliManager, BoltzCliConfig, BoltzCliError
from modules.capex_budget import CapexBudgetEngine
from modules.capital_efficiency import CapitalEfficiencyAnalyzer
from modules.segment_observations import SegmentObservationStore
from modules.utils import normalize_scid, parse_msat


# =============================================================================
# PLUGIN VERSION
# =============================================================================
# v2.10.0: Hive member zero-fee policy restoration
#   - Restores confirmed hive-member channels to 0 ppm and 0 msat base fee
#   - Keeps stale/malformed/missing hive hints neutral unless recent membership grace applies
#   - Preserves cl_revenue_ops executor authority while honoring fleet membership policy
# v2.9.0: Zero-flow fee ratchet guard, intent contracts, and audit visibility
#   - Prevents DTS+PID upward fee moves during persistent zero-flow stalls
#   - Adds adversarial intent contracts and Hermes data-quality audit tooling
#   - Improves fee-debug visibility for stalled/high-posterior channels
# v2.8.0: Structural liquidity, fee optimizer honesty, and hot-path hardening
#   - Expands hive hint freshness diagnostics and standalone safety coverage
#   - Documents cross-plugin datastore contracts for cl-mycelium/revenue artifacts
#   - Hardens budget/execution boundaries and residual action RPC safety checks
# v2.5.1: Fee market boundary deprecation
#   - Makes fee market boundary settings no-op compatibility controls
#   - Prevents remote peer fee policies from anchoring local fee floors/caps
#   - Keeps fee-debug output explicit about configured vs effective state
# v2.5.0: Fee Controller, Native Rebalance, and Safety Controls
#   - Patch release so the published tag contains the declared plugin version
#   - Keeps repo version metadata aligned with GitHub release state
# v2.4.0: Native Route Rebalancing + Fleet Equalization
#   - Native route-aware rebalancer stack ported to main
#   - Promoted segment hints consumed in the v3 rebalancer
#   - Hive equalization restored and expanded with hint-driven 0ppm paths
#   - Pair cooldown persistence across restarts
#   - Capex bootstrap and budget accounting hardening
#   - Askrene layer refresh and stale-hint recovery improvements
# v2.2.4: Stability + correctness fixes (DB rollups, policy precedence, rebalancer reliability)
# v2.1.0: Kalman Filter for Flow State Estimation
# v2.0.0: DTS+PID Fee Controller
PLUGIN_VERSION = "2.10.0"
HIVE_HINTS_DIAGNOSTICS_VERSION = "standalone-hints-v1"

# Supply-chain / runtime version floors (Phase 3C).
# These drive NON-FATAL startup probes: a version below floor logs a warning but
# never blocks init, so a running production node is not bricked by a skew.
CLN_VERSION_FLOOR = "24.11.1"   # CORE_LIGHTNING_COMPATIBILITY.md minimum
BOLTZCLI_MIN_VERSION = "2.11.0"  # documented in loop-out/loop-in help text


def _parse_version_tuple(raw: Optional[str]):
    """Extract a numeric (major, minor, patch, ...) tuple from a version string.

    Tolerant of prefixes ("v24.11.1"), suffixes ("2.11.0-beta", "24.11.1gl"),
    and build metadata. Returns () when no leading numeric component is found,
    which callers treat as "unknown / skip the comparison".
    """
    if not raw:
        return ()
    text = str(raw).strip()
    if text and text[0] in ('v', 'V'):
        text = text[1:]
    parts = []
    for token in re.split(r'[.\-+_ ]', text):
        m = re.match(r'^(\d+)', token)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def _version_below_floor(observed: Optional[str], floor: str) -> Optional[bool]:
    """Return True if `observed` < `floor`, False if >=, None if undetermined."""
    obs = _parse_version_tuple(observed)
    flr = _parse_version_tuple(floor)
    if not obs or not flr:
        return None
    return obs < flr


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
FORWARD_HYDRATION_EVENT_JITTER_SECONDS = 300


# =============================================================================
# DD5 / P1-010: daemon per-thread heartbeat + canonical loop guard
# =============================================================================
# Each daemon loop records a last-iteration timestamp so a stalled or dead loop
# becomes operator-detectable on revenue-health. The canonical guard wraps the
# ENTIRE per-iteration body (work AND the interval/jitter/sleep tail) so a tail
# exception can no longer silently kill the thread; on failure it logs and does
# a bounded, interruptible backoff before the next iteration.
_loop_heartbeats: Dict[str, Dict[str, Any]] = {}
_loop_heartbeats_lock = threading.Lock()
# Bounded backoff after an unhandled per-iteration failure so a hot failure loop
# cannot spin the CPU. The wait is interruptible (exits on shutdown).
_LOOP_BACKOFF_SECONDS = 30
# A loop is considered "stalled" if its last tick is older than this many
# multiples of its own interval is not tracked here (intervals vary); instead
# revenue-health reports the raw age and a coarse threshold flag.
_LOOP_STALL_SECONDS = 3600


def _record_loop_heartbeat(name: str) -> None:
    """Record a per-thread heartbeat for daemon loop ``name`` (best-effort)."""
    try:
        with _loop_heartbeats_lock:
            _loop_heartbeats[name] = {
                "last_tick_monotonic": time.monotonic(),
                "last_tick_ts": int(time.time()),
            }
    except Exception:
        pass


def _loop_liveness_snapshot() -> Dict[str, Any]:
    """Snapshot of daemon-loop liveness for revenue-health (thread -> age/state)."""
    now_mono = time.monotonic()
    out: Dict[str, Any] = {}
    with _loop_heartbeats_lock:
        items = {k: dict(v) for k, v in _loop_heartbeats.items()}
    for name, hb in items.items():
        age = max(0.0, now_mono - float(hb.get("last_tick_monotonic", now_mono)))
        out[name] = {
            "last_tick_ts": int(hb.get("last_tick_ts", 0) or 0),
            "last_tick_age_seconds": int(age),
            "state": "stalled" if age > _LOOP_STALL_SECONDS else "alive",
        }
    return out


# =============================================================================
# OPERATOR-PARAM VALIDATION HELPERS
# =============================================================================
# RPC handlers accept raw operator-supplied params (via lightning-cli or the
# JSON-RPC surface). These helpers coerce/clamp them BEFORE they reach SQL or
# regex so a hostile/typo'd value cannot crash a handler or run unbounded.

_QUERY_LIMIT_MAX = 1000


class _ParamError(ValueError):
    """Raised when an operator param cannot be coerced to the expected type."""


def _clamp_query_limit(value, default=20, lo=1, hi=_QUERY_LIMIT_MAX):
    """Coerce ``value`` to an int and clamp it into [lo, hi].

    Raises ``_ParamError`` for non-integer input so the caller can return a
    clean error dict rather than leaking a ValueError/TypeError. A negative
    limit is clamped to ``lo`` (never passed to SQLite, where LIMIT -1 means
    "unbounded" and would return the whole table).
    """
    try:
        ivalue = int(value)
    except (ValueError, TypeError) as e:
        raise _ParamError(f"limit must be an integer, got {value!r}") from e
    return max(lo, min(ivalue, hi))


def _snapshot_peers_once():
    """Record a one-shot connection snapshot for currently connected peers.

    Extracted from ``snapshot_peers_delayed`` so it can be unit-tested and so
    the thread-local SQLite connection it opens is always released.

    P1-021: this runs on a one-shot startup thread. Without an explicit
    ``close_connection()`` the thread-local SQLite connection stays referenced
    (via the DB's thread-connection tracking) until process shutdown. Close it
    in a ``finally`` so the one-shot thread does not retain a connection.
    """
    try:
        peers = data_service.get_peers() if data_service else safe_plugin.rpc.listpeers()
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
    finally:
        # P1-021: release this one-shot thread's thread-local DB connection.
        if database is not None:
            try:
                database.close_connection()
            except Exception:
                pass


# =============================================================================
# INIT-TIME CONFIG OPTION VALIDATION (P1-008 / P1-009 / P1-026)
# =============================================================================
# Config(**kwargs) is constructed once at init from operator-supplied options.
# The upstream _safe_int/_safe_float parsers validate type but not range; a
# 0/negative/out-of-band numeric or a typo'd enum would otherwise be accepted
# and fail open (e.g. rpc-timeout-seconds=0 breaks every RPC). These helpers
# clamp/correct such values with a loud warning, matching the warn+repair
# style used by Config._apply_override / load_overrides for the runtime path.

# Numeric options range-checked at init. Ranges mirror CONFIG_FIELD_RANGES so
# init and the runtime-override path agree on what is in-band.
_INIT_NUMERIC_RANGES = {
    'rpc_timeout_seconds': (1, 300),
    'daily_budget_sats': (0, 10_000_000),
    'weekly_budget_sats': (0, 70_000_000),
    'reputation_decay': (0.0, 1.0),
    'htlc_congestion_threshold': (0.0, 1.0),
}


def _init_warn(log, msg):
    if log is not None:
        try:
            log(msg, level='warn')
        except Exception:
            pass


def _validate_numeric_config_options(kwargs, log=None):
    """P1-008: clamp init numeric options into safe ranges (warn on clamp)."""
    for key, (lo, hi) in _INIT_NUMERIC_RANGES.items():
        if key not in kwargs:
            continue
        val = kwargs[key]
        try:
            num = float(val) if isinstance(lo, float) or isinstance(hi, float) else int(val)
        except (ValueError, TypeError):
            # Type is enforced upstream by _safe_int/_safe_float; leave as-is.
            continue
        clamped = max(lo, min(num, hi))
        if clamped != num:
            _init_warn(log, f"Config option {key}={num} out of range [{lo}, {hi}]; clamped to {clamped}")
            kwargs[key] = clamped
    return kwargs


def _enforce_fee_bound_invariant(kwargs, log=None):
    """P1-009: enforce min_fee_ppm <= max_fee_ppm (swap with warning).

    Inverted bounds would otherwise silently pin fees to a low ceiling and
    suppress revenue with no crash or warning.
    """
    if 'min_fee_ppm' not in kwargs or 'max_fee_ppm' not in kwargs:
        return kwargs
    try:
        mn = int(kwargs['min_fee_ppm'])
        mx = int(kwargs['max_fee_ppm'])
    except (ValueError, TypeError):
        return kwargs
    if mn > mx:
        _init_warn(log, f"Config min_fee_ppm ({mn}) > max_fee_ppm ({mx}); swapping to keep min <= max")
        kwargs['min_fee_ppm'], kwargs['max_fee_ppm'] = mx, mn
    return kwargs


# Enum-style options validated at init and their documented defaults. Valid
# value sets come from Config.STRING_ENUM_VALID_VALUES where present;
# base_fee_policy is not registered there and only means off/adaptive.
_INIT_ENUM_DEFAULTS = {
    'market_fee_mode': 'undercut',
    'base_fee_policy': 'off',
    'fee_profile': 'active',
    'expansion_treasury_preferred_currency': 'BTC',
}
_BASE_FEE_POLICY_VALID = ('off', 'adaptive')


def _valid_enum_values(key):
    """Return the valid value tuple for an enum option, or None if unknown."""
    try:
        from modules.config import STRING_ENUM_VALID_VALUES
        if key in STRING_ENUM_VALID_VALUES:
            return STRING_ENUM_VALID_VALUES[key]
    except Exception:
        pass
    if key == 'base_fee_policy':
        return _BASE_FEE_POLICY_VALID
    return None


def _validate_enum_config_options(kwargs, log=None):
    """P1-026: validate enum-style string options; unknown -> warn + default."""
    for key, default in _INIT_ENUM_DEFAULTS.items():
        if key not in kwargs:
            continue
        valid = _valid_enum_values(key)
        if not valid:
            continue
        val = kwargs[key]
        if val not in valid:
            _init_warn(log, f"Config option {key}={val!r} not one of {tuple(valid)}; using default {default!r}")
            kwargs[key] = default
    return kwargs


_ATEXIT_SHUTDOWN_TIMEOUT = 10.0


def _bounded_executor_shutdown(executor, timeout=_ATEXIT_SHUTDOWN_TIMEOUT):
    """Shut down a ThreadPoolExecutor without blocking process exit forever.

    P1-029: ``executor.shutdown(wait=True, cancel_futures=True)`` can only
    cancel QUEUED futures; an in-flight worker blocked on a wedged lightningd
    ``recv()`` cannot be cancelled, so a plain ``wait=True`` call blocks atexit
    indefinitely (relies on external SIGKILL). Run the draining shutdown on a
    daemon thread and join with a timeout: on a healthy node the join returns
    immediately (unchanged behavior); on a wedged node atexit proceeds after
    ``timeout`` and lets the process exit.

    Returns True if the executor drained within the timeout, else False.
    """
    if executor is None:
        return True
    done = threading.Event()

    def _drain():
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_drain, name="rpc-shutdown", daemon=True).start()
    return done.wait(timeout)


def _compute_forward_hydration_start(
    last_forward_ts: Optional[int],
    flow_window_days: int,
    now: Optional[int] = None,
) -> Optional[int]:
    """Compute a bounded startup backfill start time for the forwards table.

    Empty tables get a full warm start so flow analysis has enough history.
    Non-empty tables only get a bounded overlap backfill when the last stored
    forward is stale enough to justify one.
    """
    current_time = int(time.time()) if now is None else int(now)

    if last_forward_ts is None:
        return current_time - (max(flow_window_days, 14) * 86400)

    last_forward_ts = int(last_forward_ts)
    gap_seconds = max(0, current_time - last_forward_ts)
    if gap_seconds <= FORWARD_HYDRATION_EVENT_JITTER_SECONDS:
        return None

    hydration_floor = current_time - (max(flow_window_days + 1, 15) * 86400)
    overlap_start = last_forward_ts - 86400
    return max(overlap_start, hydration_floor)


_HYDRATION_PAGE_LIMIT = 1000
_HYDRATION_MAX_PAGES = 10_000  # Hard stop: 10M forwards, prevents runaway paging


def _hydration_fetch_settled_forwards(start_time: int) -> List[Dict[str, Any]]:
    """Fetch settled forwards newer than start_time for startup hydration.

    Prefers a paged listforwards(index="created", start=..., limit=N) loop so
    the node's entire settled-forward history is never materialized in a
    single RPC response. Any error (older CLN without index paging, mocked or
    unexpected response shapes, RPC failures) falls back to the legacy full
    settled fetch. The unique-index dedup in bulk_insert_forwards remains the
    correctness backstop on both paths.
    """
    start_time = int(start_time)
    try:
        collected: List[Dict[str, Any]] = []
        next_start = 0
        for _ in range(_HYDRATION_MAX_PAGES):
            page = safe_plugin.rpc.listforwards(
                status="settled",
                index="created",
                start=next_start,
                limit=_HYDRATION_PAGE_LIMIT,
            )
            forwards = page["forwards"]
            if not isinstance(forwards, list):
                raise TypeError("listforwards returned non-list forwards")
            for fwd in forwards:
                if int(fwd.get("received_time", 0) or 0) > start_time:
                    collected.append(fwd)
            if len(forwards) < _HYDRATION_PAGE_LIMIT:
                return collected
            # Advance past the last created_index. Missing field means this
            # CLN doesn't support index paging — KeyError triggers fallback.
            advanced = int(forwards[-1]["created_index"]) + 1
            if advanced <= next_start:
                raise ValueError("listforwards paging did not advance")
            next_start = advanced
        raise RuntimeError("listforwards paging exceeded max page count")
    except Exception as e:
        plugin.log(
            f"Paged listforwards hydration unavailable ({e}); "
            f"falling back to full settled fetch",
            level='debug'
        )

    try:
        result = data_service.get_forwards(status="settled")
    except Exception as e:
        plugin.log(f"Warning: listforwards RPC failed during hydration: {e}. "
                   f"Flow analysis will use existing database data only.", level='warn')
        return []
    forwards = result.get("forwards", []) or []
    return [f for f in forwards if int(f.get("received_time", 0) or 0) > start_time]


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
# THREAD-SAFE RPC WRAPPER (High-Uptime Stability)
# =============================================================================

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


# =============================================================================
# RPC SOCKET-LEVEL TIMEOUT (P1-007)
# =============================================================================
# pyln opens a fresh Unix socket per call and blocks in recv() with no timeout.
# When lightningd wedges, every worker thread blocks on recv() forever and the
# 16-thread pool is permanently consumed with no self-recovery. The proxy's
# future.result(timeout) frees the *caller* but not the *worker*.
#
# Fix: apply a socket-level recv timeout to the per-call socket. pyln creates
# the socket internally, so we patch UnixSocket.connect once to read a
# per-thread desired timeout (set by the worker before it runs the RPC) and
# apply it via socket.settimeout(). Long-poll methods (wait*) intentionally
# block indefinitely and are exempted so healthy-node behavior is unchanged.

# Per-worker-thread desired socket timeout (seconds) or None to disable.
_rpc_socket_timeout = threading.local()

# Backstop buffer added on top of the proxy timeout: the caller-side
# future.result(proxy_timeout) fires first (preserving the RPCTimeoutError the
# caller already expects); the socket timeout is the slightly-later backstop
# that frees the wedged worker.
_RPC_SOCKET_TIMEOUT_BUFFER = 5.0

# Methods that legitimately block for a long time; never given a socket
# timeout so their behavior is unchanged.
_LONG_POLL_RPC_METHODS = frozenset({
    "wait", "waitanyinvoice", "waitinvoice", "waitsendpay",
    "waitblockheight", "waitrune",
})


def _socket_timeout_for(method_name, proxy_timeout):
    """Return the socket recv timeout for a method, or None to leave unbounded.

    Long-poll (wait*) methods return None. All others return
    ``proxy_timeout + _RPC_SOCKET_TIMEOUT_BUFFER`` so the caller-side timeout
    fires first and the socket timeout is a backstop that frees the worker.
    """
    name = str(method_name or "")
    if name in _LONG_POLL_RPC_METHODS or name.startswith("wait"):
        return None
    try:
        base = float(proxy_timeout)
    except (ValueError, TypeError):
        base = 30.0
    if base <= 0:
        return None
    return base + _RPC_SOCKET_TIMEOUT_BUFFER


def _install_rpc_socket_timeout(log=None):
    """Patch pyln's UnixSocket.connect to honor the per-thread socket timeout.

    Best-effort and idempotent. Returns True if the patch is installed (or was
    already), False if pyln's internals are not the expected shape in this
    version — in which case the caller-side future timeout remains the only
    guard (documented residual for P1-007).
    """
    try:
        from pyln.client import lightning as _pyln_lightning
    except Exception:
        return False
    unix_socket_cls = getattr(_pyln_lightning, "UnixSocket", None)
    if unix_socket_cls is None or not hasattr(unix_socket_cls, "connect"):
        return False
    if getattr(unix_socket_cls, "_revops_timeout_patched", False):
        return True
    try:
        _orig_connect = unix_socket_cls.connect

        def _connect_with_timeout(self, *args, **kwargs):
            _orig_connect(self, *args, **kwargs)
            desired = getattr(_rpc_socket_timeout, "value", None)
            sock = getattr(self, "sock", None)
            if desired and sock is not None:
                try:
                    sock.settimeout(float(desired))
                except Exception:
                    pass

        unix_socket_cls.connect = _connect_with_timeout
        unix_socket_cls._revops_timeout_patched = True
        if log is not None:
            try:
                log("RPC socket-level timeout guard installed", level="debug")
            except Exception:
                pass
        return True
    except Exception:
        return False


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
    background pushes can't starve the main pool.
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
        # P1-007: install the pyln socket-level timeout guard so a wedged
        # lightningd cannot permanently consume all worker threads.
        self._socket_timeout_installed = _install_rpc_socket_timeout(
            getattr(plugin_instance, "log", None)
        )

    def __getattr__(self, name):
        if name in ("_plugin", "_rpc", "_executor", "_async_executor",
                     "_main_submit_slots", "_async_submit_slots",
                     "_async_fail_count", "_async_lock",
                     "_submit_main", "_submit_async",
                     "call", "fire_and_forget"):
            return super().__getattribute__(name)

        fn = getattr(self._rpc, name)

        def wrapper(*args, **kwargs):
            proxy_timeout = 30
            if config:
                proxy_timeout = config.rpc_timeout_seconds
            # If the caller supplies its own ``timeout`` kwarg, extend the
            # proxy's effective wait so we never cut a legitimately long
            # call short. For methods that natively accept ``timeout`` (e.g.
            # ``waitsendpay``) we leave it in kwargs; for callers that pass
            # ``_proxy_timeout`` we strip it so it never reaches pyln.
            proxy_only = kwargs.pop("_proxy_timeout", None)
            user_timeout = proxy_only if proxy_only is not None else kwargs.get("timeout")
            if isinstance(user_timeout, (int, float)) and user_timeout > 0:
                proxy_timeout = max(proxy_timeout, int(user_timeout) + 5)
            future = self._submit_main(fn, name, proxy_timeout, *args, **kwargs)
            try:
                return future.result(timeout=proxy_timeout)
            except (TimeoutError, self._FuturesTimeoutError):
                self._plugin.log(f"RPC timeout after {proxy_timeout}s on {name}", level="warn")
                raise RPCTimeoutError(name)

        return wrapper

    def _submit_main(self, fn, method_name: str, proxy_timeout: int, *args, **kwargs):
        # NOTE: this parameter MUST NOT be named 'timeout'. Several CLN RPC
        # methods (notably waitsendpay) accept a 'timeout' keyword argument
        # of their own, and the proxy wrapper forwards **kwargs unchanged.
        # A parameter named 'timeout' here collides with the user's kwarg
        # and raises "TypeError: got multiple values for argument 'timeout'",
        # silently breaking every rebalance execution attempt.
        acquire_timeout = min(max(float(proxy_timeout), 0.1), 1.0)
        if not self._main_submit_slots.acquire(timeout=acquire_timeout):
            self._plugin.log(
                f"RPC pool saturated on {method_name} (submission queue full)",
                level="warn"
            )
            raise RPCOverloadedError(method_name)

        # P1-007: run the RPC in a wrapper that sets a per-thread desired
        # socket timeout before the call opens its pyln socket, so a wedged
        # lightningd surfaces socket.timeout in the worker and frees it rather
        # than blocking recv() forever. Long-poll methods get None (unbounded).
        sock_timeout = _socket_timeout_for(method_name, proxy_timeout)

        def _worker():
            prev = getattr(_rpc_socket_timeout, "value", None)
            _rpc_socket_timeout.value = sock_timeout
            try:
                return fn(*args, **kwargs)
            finally:
                _rpc_socket_timeout.value = prev

        try:
            future = self._executor.submit(_worker)
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

    def call(self, method_name: str, payload: Any = None, *, timeout: Optional[float] = None, **kwargs):
        proxy_timeout = 30
        if config:
            proxy_timeout = config.rpc_timeout_seconds
        # Callers (e.g. askrene-getroutes on a loaded node) can extend the
        # proxy ceiling past ``config.rpc_timeout_seconds`` without changing
        # it globally. The kwarg is consumed here and never reaches pyln,
        # so CLN's JSON-RPC schema never sees an unknown ``timeout`` field.
        if isinstance(timeout, (int, float)) and timeout > 0:
            proxy_timeout = max(proxy_timeout, int(timeout) + 5)
        future = self._submit_main(
            self._rpc.call,
            method_name,
            proxy_timeout,
            method_name,
            payload if payload is not None else {},
        )
        try:
            return future.result(timeout=proxy_timeout)
        except (TimeoutError, self._FuturesTimeoutError):
            self._plugin.log(f"RPC timeout after {proxy_timeout}s on {method_name}", level="warn")
            raise RPCTimeoutError(method_name)

    def fire_and_forget(self, method_name: str, payload: Any = None):
        """Submit an RPC call to the async pool without waiting.

        Uses a separate 4-thread pool so fire-and-forget calls can't
        starve synchronous RPCs.
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
fee_controller: Optional[FeeController] = None
rebalancer: Optional[EVRebalancer] = None
database: Optional[Database] = None
config: Optional[Config] = None
profitability_analyzer: Optional[ChannelProfitabilityAnalyzer] = None
capacity_planner: Optional[CapacityPlanner] = None
safe_plugin: Optional['ThreadSafePluginProxy'] = None  # Thread-safe plugin wrapper
data_service = None  # Unified data service (DataService instance)
policy_manager: Optional[PolicyManager] = None  # v1.4: Peer policy management
boltz_manager: Optional[BoltzCliManager] = None  # Boltz CLI integration (optional)
hive_hints: Optional[HiveHintAdapter] = None  # cl_hive fleet hint adapter
capex_engine: Optional[CapexBudgetEngine] = None  # Unified capex budget engine
hive_router = None  # HiveRouter: shared askrene fleet route discovery
_boltz_balance_last_action: Dict[str, int] = {}  # channel_id -> unix ts of last Boltz balance action
_boltz_balance_lock = threading.Lock()
# Resource-growth bound: _boltz_balance_last_action only stores per-channel
# cooldown timestamps. Nothing evicted closed-channel entries, so the dict grew
# one entry per channel ever swapped for the process lifetime. Prune entries far
# older than any realistic cooldown (default 4h; hint overrides are hours). A
# 30-day TTL dwarfs every cooldown window, so eviction is provably
# non-behavioral: any pruned entry is already treated as expired by the cooldown
# check (now - last_ts >> any cooldown_seconds).
_BOLTZ_BALANCE_ACTION_TTL_SECONDS: int = 30 * 86400


def _prune_boltz_balance_actions(now: int) -> int:
    """Evict cooldown entries older than the TTL. Returns entries removed.

    Acquires _boltz_balance_lock internally; callers must NOT already hold it.
    """
    cutoff = int(now) - _BOLTZ_BALANCE_ACTION_TTL_SECONDS
    with _boltz_balance_lock:
        stale = [k for k, ts in _boltz_balance_last_action.items()
                 if int(ts or 0) < cutoff]
        for k in stale:
            _boltz_balance_last_action.pop(k, None)
    return len(stale)
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
_scid_to_peer_cache: Dict[str, Optional[str]] = {}  # None = negatively-cached unknown SCID
_scid_cache_last_cleared: float = 0.0
_SCID_CACHE_TTL_SECONDS: int = 3600  # Clear cache every hour
_SCID_NEGATIVE_CACHE_MAX_ENTRIES: int = 512  # Bound on negatively-cached unknown SCIDs
_scid_cache_lock = threading.Lock()
_scid_cache_fetch_lock = threading.Lock()  # M-2: Serializes cache-miss RPC calls


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
    default='2000',
    description='Maximum fee ceiling in PPM (default: 2000)'
)

plugin.add_option(
    name='revenue-ops-fee-profile',
    default='active',
    description="Fee controller profile: 'active' for normal routing nodes, 'conservative' for low-volume nodes"
)

plugin.add_option(
    name='revenue-ops-fee-market-boundary-enabled',
    default='false',
    description='Deprecated no-op compatibility flag; fee market boundary logic is ignored (default: false)'
)

plugin.add_option(
    name='revenue-ops-fee-market-boundary-min-competitors',
    default='3',
    description='Deprecated no-op compatibility setting for fee market boundary logic (default: 3)'
)

plugin.add_option(
    name='revenue-ops-fee-market-boundary-margin-ppm',
    default='5',
    description='Deprecated no-op compatibility setting for fee market boundary logic (default: 5)'
)

plugin.add_option(
    name='revenue-ops-fee-market-boundary-margin-ratio',
    default='0.05',
    description='Deprecated no-op compatibility setting for fee market boundary logic (default: 0.05)'
)

plugin.add_option(
    name='revenue-ops-fee-market-boundary-max-downshift-ratio',
    default='0.35',
    description='Deprecated no-op compatibility setting for fee market boundary logic (default: 0.35)'
)

plugin.add_option(
    name='revenue-ops-fee-market-boundary-cache-seconds',
    default='60',
    description='Deprecated no-op compatibility setting for fee market boundary logic (default: 60)'
)

plugin.add_option(
    name='revenue-ops-market-fee-mode',
    default='undercut',
    description=(
        'How to price relative to neighbor-median fee. '
        '"undercut" (default): price below the median to win volume. '
        '"match": target the median, no undercut. '
        '"premium": price above the median using the same per-corridor '
        'weight that would otherwise undercut. Use "premium" in '
        'inelastic-demand markets (e.g., hive-coordinated with reliable '
        'routing) to maximize revenue per forward. '
        '"competition_aware": apply the median-undercut ONLY when a '
        "competitor is priced at or below DTS's target; preserve DTS when "
        "we're already below every competitor (undercut would otherwise "
        'drag fees down against an inelastic market).'
    )
)


plugin.add_option(
    name='revenue-ops-base-fee-policy',
    default='off',
    description=(
        'Base-fee policy (Upgrade A, 2026-04-22). '
        '"off" (default): use revenue-ops-base-fee-msat for all channels. '
        '"adaptive": apply revenue-ops-base-fee-intra-fleet to hive fleet '
        'members and revenue-ops-base-fee-non-hive to everyone else. '
        'Motivated by the 168x per-forward fee gap observed vs clboss.'
    )
)

plugin.add_option(
    name='revenue-ops-base-fee-intra-fleet',
    default='0',
    description='Base fee in msat for channels to hive fleet members (default: 0).'
)

plugin.add_option(
    name='revenue-ops-base-fee-non-hive',
    default='1000',
    description=(
        'Base fee in msat for channels to non-hive peers when '
        'revenue-ops-base-fee-policy=adaptive (default: 1000 msat). '
        'Conservative starting point; clboss observed at 15,307 msat.'
    )
)

plugin.add_option(
    name='revenue-ops-fee-ppm-intra-fleet',
    default='1',
    description=(
        'Proportional fee in ppm for channels to hive fleet members '
        '(default: 1). The legacy 0-PPM policy leaked revenue on external '
        'traffic transiting the hive mesh. 1 ppm keeps members "cheapest '
        'path" (50-500x below typical competitors) while recapturing '
        'that leak. Set to 0 to restore legacy 0-PPM.'
    )
)

plugin.add_option(
    name='revenue-ops-neighbor-median-min-competitors',
    default='2',
    description=(
        'Minimum competitor count for neighbor_median computation '
        '(default: 2). The median is used by market-fee-mode '
        '(undercut/match/premium/competition_aware); below this threshold '
        'the helper returns None and the market-fee-mode branch is '
        'skipped. Prod nodes with dense gossip may prefer 3.'
    )
)

plugin.add_option(
    name='revenue-ops-rebalance-min-profit',
    default='10',
    description='Minimum profit in sats to trigger rebalance (default: 10)'
)

plugin.add_option(
    name='revenue-ops-futility-cooldown-hours',
    default='48',
    description='Hours before retrying a channel after 10+ consecutive rebalance failures (default: 48)'
)

plugin.add_option(
    name='revenue-ops-rebalance-emergency-local-ratio',
    default='0.10',
    description='Local ratio below which a destination bypasses the channel-level rebalance cooldown (Phase 3, default: 0.10; 0 disables)'
)

plugin.add_option(
    name='revenue-ops-rebalance-drift-override-ratio',
    default='0.30',
    description='Drift since last successful rebalance that bypasses the cooldown (Phase 3, default: 0.30; 0 disables)'
)


def _on_rebalance_tuning_change(plugin_: Plugin, option_name: str, new_value: Any) -> None:
    """Apply rebalance tuning changes from lightning-cli setconfig at runtime."""
    tuning_map = {
        # final_score is in SATS of expected net value; match CONFIG_FIELD_RANGES.
        'revenue-ops-rebalance-hold-margin': ('rebalance_hold_margin', float, 0.0, 1000.0),
        'revenue-ops-pair-fee-cap-ppm': ('pair_fee_cap_ppm', int, 0, None),
        'revenue-ops-hive-rebalance-bootstrap-budget-sats': (
            'hive_rebalance_bootstrap_budget_sats',
            int,
            0,
            None,
        ),
    }
    if option_name not in tuning_map:
        return
    attr, parser, minimum, maximum = tuning_map[option_name]
    try:
        parsed_value = parser(new_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{option_name} must be a {parser.__name__}") from exc
    if minimum is not None and parsed_value < minimum:
        raise ValueError(f"{option_name} must be >= {minimum}")
    if maximum is not None and parsed_value > maximum:
        raise ValueError(f"{option_name} must be <= {maximum}")

    cfg = globals().get('config')
    if cfg is None:
        return
    if not hasattr(cfg, attr):
        raise ValueError(f"active Config does not support {attr}")
    old_value = getattr(cfg, attr)
    setattr(cfg, attr, parsed_value)
    plugin_.log(
        f"REBALANCE TUNING: {option_name} changed from {old_value} to {parsed_value}",
        level='info',
    )


plugin.add_option(
    name='revenue-ops-rebalance-hold-margin',
    default='0.0',
    description='Minimum final_score in SATS of expected net value a priced pair must clear or it is rejected as below_hold_margin (Phase 4, default: 0.0)',
    dynamic=True,
    on_change=_on_rebalance_tuning_change,
)

plugin.add_option(
    name='revenue-ops-pair-fee-cap-ppm',
    default='1000',
    description='Per-pair fee budget = max(dest capex, ceil(amount * ppm / 1M)). Decouples per-rebalance fee from capex bootstrap (Iter1, default: 1000 = 0.1% of amount; 0 disables)',
    dynamic=True,
    on_change=_on_rebalance_tuning_change,
)

plugin.add_option(
    name='revenue-ops-hive-rebalance-bootstrap-budget-sats',
    default='300',
    description='Conservative per-pair fee budget for active hive-member channels before capex/profitability history appears (default: 300 sats; 0 disables)',
    dynamic=True,
    on_change=_on_rebalance_tuning_change,
)

plugin.add_option(
    name='revenue-ops-rebalance-coordination-reserved-slots',
    default='2',
    description=(
        "Reserved slots for coordination pairs (from cl-hive's rebalance_"
        "recommendations / rebalance_campaigns hints) on top of the "
        "planner's max_pairs cap. Default 2 lets a small number of "
        "hive-blessed pairs bypass the cap without letting coordination "
        "dominate arbitrarily. Set to 0 to restore strict-cap behavior "
        "(coordination competes inside max_pairs)."
    )
)


plugin.add_option(
    name='revenue-ops-flow-window-days',
    default='7',
    description='Number of days to analyze for flow calculation (default: 7)'
)

plugin.add_option(
    name='revenue-ops-daily-budget-sats',
    default='5000',
    description='Max rebalancing fees to spend in 24 hours (default: 5000)'
)

plugin.add_option(
    name='revenue-ops-diagnostic-rebalance-max-fee-sats',
    default='400',
    description=(
        'Fee cap in sats for the diagnostic (defibrillator) shock rebalance; '
        'the shock ppm ceiling is derived from this cap. Clamped to '
        '[1, min(daily budget, 10000)] (default: 400)'
    )
)

plugin.add_option(
    name='revenue-ops-allow-zero-cost-auto-rebalance-when-budget-zero',
    default='false',
    description='Allow automatic zero-fee rebalances when daily_budget_sats is zero (default: false)'
)

plugin.add_option(
    name='revenue-ops-weekly-budget-sats',
    default='35000',
    description='Max rebalancing fees to spend in 7 days - hard ceiling over daily burst limit (default: 35000)'
)

plugin.add_option(
    name='revenue-ops-min-wallet-reserve',
    default='1000000',
    description='Minimum total funds (on-chain + off-chain) to keep in reserve (default: 1,000,000)'
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

# Vegas Reflex options
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
    description='Enable aggressive rebalance protection for fast-draining high-profit channels (default: true)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-override-peers',
    default='',
    description='CSV peer pubkeys to force hot-channel protection (default: empty)'
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
    description='Max multiplier on rebalance chunk size for hot-channel protection (default: 4.0)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-min-cooldown-hours',
    default='1.0',
    description='Minimum cooldown hours for protected hot channels (default: 1.0)'
)

plugin.add_option(
    name='revenue-ops-hot-channel-protection-max-rebalance-fee-ppm',
    default='2000',
    description='Hard max routing fee ppm for protected hot-channel rebalances (default: 2000)',
    opt_type='int'
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
    description='Daily Boltz swap fee budget used by revenue-boltz loop methods (default: 3000)',
    dynamic=True
)

plugin.add_option(
    name='revenue-ops-boltz-enforce-budget',
    default='true',
    description='If true, reject Boltz swaps when estimated fee exceeds remaining daily budget (default: true)',
    dynamic=True
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
    name='revenue-ops-boltz-routing-fee-limit-ppm',
    default='0',
    description='Max routing fee in PPM for reverse swaps (0 = no limit, boltzcli default)'
)

plugin.add_option(
    name='revenue-ops-boltz-max-withdraw-sats',
    default='10000000',
    description='Hard cap on a single Boltz on-chain withdraw in sats (default: 10000000; 0 disables). A sweep withdraw bypasses this cap and requires confirm_sweep=true.'
)

plugin.add_option(
    name='revenue-ops-boltz-auto-cycle-enabled',
    default='false',
    description='Enable in-plugin periodic profit-gated Boltz balance cycles (default: false; automated spending must be opted into explicitly)'
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

plugin.add_option(
    name='revenue-ops-receivable-ratio-target',
    default='0.30',
    description='Node receivable/capacity ratio at which structural drain pressure reaches zero'
)

plugin.add_option(
    name='revenue-ops-receivable-ratio-floor',
    default='0.20',
    description='Receivable/capacity ratio below which the node is considered inbound-starved'
)

plugin.add_option(
    name='revenue-ops-boltz-structural-budget-sats',
    default='0',
    description='Daily swap-fee budget (sats) for scarcity-credited Boltz loop-outs (0 = disabled)'
)

plugin.add_option(
    name='revenue-ops-drain-fee-discount-max',
    default='0.0',
    description='Max bounded fee discount on stagnant over-local channels (0.0 = disabled)'
)

plugin.add_option(
    name='revenue-ops-expansion-treasury-enabled',
    default='false',
    description='Enable expansion treasury mode (reverse-swap excess LN to on-chain for opens) (default: false)'
)

plugin.add_option(
    name='revenue-ops-expansion-treasury-onchain-target-sats',
    default='5000000',
    description='Target confirmed on-chain reserve in sats for expansion treasury mode (default: 5000000)'
)

plugin.add_option(
    name='revenue-ops-expansion-treasury-min-deficit-sats',
    default='250000',
    description='Minimum on-chain reserve deficit before treasury swaps are attempted (default: 250000)'
)

plugin.add_option(
    name='revenue-ops-expansion-treasury-preferred-currency',
    default='BTC',
    description='Preferred Boltz reverse-swap output currency for expansion treasury (BTC or LBTC; default: BTC)'
)

plugin.add_option(
    name='revenue-ops-expansion-treasury-max-actions',
    default='1',
    description='Max treasury reverse swaps per treasury cycle (default: 1)'
)

plugin.add_option(
    name='revenue-ops-expansion-treasury-min-source-local-pct',
    default='80.0',
    description='Minimum local balance percent to consider a channel as treasury harvest source (default: 80.0)'
)

plugin.add_option(
    name='revenue-ops-expansion-treasury-exclude-protected',
    default='true',
    description='Exclude hot/protected channels from treasury harvesting (default: true)'
)

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
    default='false',
    description='Log planner decisions without executing (default: false)'
)
plugin.add_option(
    name='revenue-ops-planner-max-opens-per-cycle',
    default='1',
    description='Maximum automated channel opens per planner cycle (default: 1)'
)
plugin.add_option(
    name='revenue-ops-planner-max-closes-per-cycle',
    default='0',
    description='Maximum planner close executions per cycle when close execution is enabled (default: 0)'
)
plugin.add_option(
    name='revenue-ops-planner-close-fee-reserve-multiplier',
    default='2.0',
    description='Multiplier applied to estimated close fee for planner close budget reservation (default: 2.0)'
)
plugin.add_option(
    name='revenue-ops-planner-close-fee-cap-sats',
    default='0',
    description='Fixed planner close fee cap/reservation in sats; 0 uses reserve multiplier (default: 0)'
)
plugin.add_option(
    name='revenue-ops-planner-close-feerange-enabled',
    default='false',
    description='Pass a CLN close feerange cap derived from the planner close fee reservation (default: false)'
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
plugin.add_option(
    name='revenue-ops-planner-min-annual-roi-pct',
    default='1.0',
    description='Minimum annualized return hurdle for automated channel opens (default: 1.0%)'
)
plugin.add_option(
    name='revenue-ops-planner-execute-closes',
    default='false',
    description='Allow the capacity planner to execute close RPCs (default: false)',
    dynamic=True
)
plugin.add_option(
    name='revenue-ops-hive-hints-enabled',
    default='true',
    description='Enable bounded fee/rebalance bias from cl_hive fleet hints (default: true)'
)
plugin.add_option(
    name='revenue-ops-hive-hints-ttl',
    default='0',
    description='Override hint snapshot TTL in seconds; 0 = use snapshot value (default: 0)'
)
plugin.add_option(
    name='revenue-ops-hive-hints-allow-all-hints-m2-scope',
    default='false',
    description='Allow lab-only all_hints M2 scope from hive hints (default: false)'
)

def _on_rebalance_router_change(plugin_: Plugin, option_name: str, new_value: Any) -> None:
    """Validate + log a runtime rebalance-router flip triggered by setconfig.

    Only the askrene-backed v3 router is supported. The option remains as a
    compatibility shim so stale operator config can be surfaced cleanly.
    """
    if new_value != "v3":
        raise ValueError(
            f"rebalance-router only supports 'v3'; legacy 'v2' routing was removed (got {new_value!r})"
        )
    r = globals().get("rebalancer")
    eng = getattr(r, "rebalance_engine_v2", None) if r is not None else None
    if eng is None:
        raise ValueError(
            "rebalance engine not initialized; cannot change router"
        )
    if getattr(eng, "router_v3", None) is None:
        raise ValueError(
            "askrene unavailable on this node; v3 router cannot be enabled"
        )
    cfg = globals().get("config")
    if cfg is not None:
        cfg.rebalance_router = "v3"
    plugin_.log(
        "rebalance-router pinned to v3 "
        f"(takes effect at next cycle boundary)",
        level="info",
    )


plugin.add_option(
    name='revenue-ops-rebalance-router',
    default='v3',
    description="Rebalance route discovery: 'v3' (askrene getroutes, required). "
                "Legacy 'v2' getroute routing has been removed. "
                "This option remains only as a compatibility shim and only accepts 'v3'.",
    opt_type='string',
    dynamic=True,
    on_change=_on_rebalance_router_change,
)

plugin.add_option(
    name='revenue-ops-askrene-layers',
    default='hive-fleet',
    description="CSV of askrene layer names passed to v3 router getroutes calls. "
                "Missing layers are silently dropped by askrene. Blank values use the default "
                "'hive-fleet' cl-hive bias; set to 'none' or 'standalone' for no configured layers."
)


def _boltz_auto_cycle_mark_state(**updates):
    with _boltz_auto_cycle_state_lock:
        _boltz_auto_cycle_state.update(updates)


def _treasury_recommendation_executable(rec: Any) -> bool:
    """True when the treasury executor would actually RUN this rec.

    The treasury executor skips recs that fail the profit guard
    (profit_guard_failed) and recs whose pass depends on the structural
    inbound-scarcity credit (structural_credit_requires_balance_cycle).
    Counting those toward treasury-mode selection deadlocks the auto-cycle:
    treasury wins the mode pick, executes 0 actions, and the balance cycle
    (where the structural envelope lives) never gets a turn.
    """
    if not isinstance(rec, dict):
        return False
    econ = rec.get("economics")
    econ = econ if isinstance(econ, dict) else {}
    return bool(econ.get("passes_profit_guard")) and not bool(econ.get("structural"))


def _select_boltz_auto_cycle_mode(
    *,
    treasury_plan: Optional[Dict[str, Any]],
    balance_plan: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pick the next Boltz mode without side effects."""
    treasury_plan = treasury_plan if isinstance(treasury_plan, dict) else {}
    balance_plan = balance_plan if isinstance(balance_plan, dict) else {}

    treasury = treasury_plan.get("treasury")
    treasury = treasury if isinstance(treasury, dict) else {}
    treasury_recommendations = (
        list(treasury_plan.get("recommendations", []))
        if isinstance(treasury_plan.get("recommendations", []), list)
        else []
    )
    balance_recommendations = (
        list(balance_plan.get("recommendations", []))
        if isinstance(balance_plan.get("recommendations", []), list)
        else []
    )
    # Only EXECUTABLE treasury recs count toward picking treasury mode.
    treasury_executable = [
        r for r in treasury_recommendations if _treasury_recommendation_executable(r)
    ]
    reserve_deficit_sats = int(treasury.get("deficit_sats", 0) or 0)

    if str(treasury_plan.get("status") or "") == "ok" and treasury_executable:
        return {
            "mode": "treasury",
            "reason": "standing_onchain_reserve_below_target",
            "reserve_deficit_sats": reserve_deficit_sats,
            "treasury_candidate_count": len(treasury_recommendations),
            "treasury_executable_count": len(treasury_executable),
            "balance_candidate_count": len(balance_recommendations),
        }

    if balance_recommendations:
        return {
            "mode": "balance",
            "reason": "onchain_reserve_healthy_use_balance_mode",
            "reserve_deficit_sats": reserve_deficit_sats,
            "treasury_candidate_count": len(treasury_recommendations),
            "treasury_executable_count": len(treasury_executable),
            "balance_candidate_count": len(balance_recommendations),
        }

    return {
        "mode": "idle",
        "reason": "no_eligible_boltz_actions",
        "reserve_deficit_sats": reserve_deficit_sats,
        "treasury_candidate_count": len(treasury_recommendations),
        "treasury_executable_count": len(treasury_executable),
        "balance_candidate_count": len(balance_recommendations),
    }


def _select_boltz_currency(direction: str, amount_sats: int) -> str:
    """Compare BTC vs LBTC quote for a swap and return the cheaper currency.

    Prefers LBTC when costs are equal (faster settlement).
    Falls back to LBTC if quoting fails.
    """
    bm = _require_boltz_manager()
    swap_type = "submarine" if direction == "loop_in" else "reverse"

    try:
        btc_quote = bm.quote(amount_sats, swap_type=swap_type, currency="BTC")
        btc_fee = int(btc_quote.get("estimated_total_fee_sats", 0) or 0)
    except Exception:
        btc_fee = float('inf')

    try:
        lbtc_quote = bm.quote(amount_sats, swap_type=swap_type, currency="LBTC")
        lbtc_fee = int(lbtc_quote.get("estimated_total_fee_sats", 0) or 0)
    except Exception:
        lbtc_fee = float('inf')

    if btc_fee == float('inf') and lbtc_fee == float('inf'):
        return "LBTC"  # fallback

    # Prefer LBTC when equal (faster settlement)
    if lbtc_fee <= btc_fee:
        chosen, reason = "LBTC", f"LBTC={lbtc_fee} <= BTC={btc_fee}"
    else:
        chosen, reason = "BTC", f"BTC={btc_fee} < LBTC={lbtc_fee}"

    if plugin:
        plugin.log(f"Boltz currency selection ({direction}, {amount_sats}sats): {chosen} ({reason})", level='debug')

    return chosen


def _run_boltz_auto_cycle_once(trigger: str = "manual", force: bool = False,
                               dry_run: bool = False) -> Dict[str, Any]:
    """Run one in-plugin Boltz auto-cycle using existing RPC logic (single-flight).

    DD4 / P1-018 (operator ruling 2026-07-02): dry_run defaults False so both
    the scheduled daemon path AND the manual RPC (revenue-boltz-auto-cycle-run-now)
    run live by default — force=true alone runs one live cycle. dry_run=true is
    an opt-in preview (builds the plan, computes profitability) without creating
    any swap.
    """
    if boltz_manager is None or not getattr(boltz_manager, 'enabled', False):
        result = {
            'status': 'disabled',
            'reason': 'boltz integration disabled',
            'trigger': trigger,
        }
        _boltz_auto_cycle_mark_state(last_result=result, last_error=None, enabled=False)
        return result

    cfg = config.snapshot() if config else None
    enabled = bool(getattr(cfg, 'boltz_auto_cycle_enabled', False)) if cfg else False
    _boltz_auto_cycle_mark_state(enabled=enabled)
    if not enabled and not force:
        result = {
            'status': 'disabled',
            'reason': 'boltz auto-cycle disabled by config',
            'trigger': trigger,
        }
        _boltz_auto_cycle_mark_state(last_result=result, last_error=None)
        return result

    # L-1 FIX: Guard lock release with acquired flag to prevent RuntimeError
    acquired = _boltz_auto_cycle_run_lock.acquire(blocking=False)
    if not acquired:
        result = {'status': 'skipped', 'reason': 'auto-cycle already running', 'trigger': trigger}
        _boltz_auto_cycle_mark_state(last_result=result)
        return result

    try:
        started = int(time.time())
        start_monotonic = time.monotonic()
        _boltz_auto_cycle_mark_state(
            running=True,
            last_trigger=trigger,
            last_started_ts=started,
            last_error=None,
        )
        max_actions = max(1, int(getattr(cfg, 'boltz_auto_cycle_max_actions', 1) if cfg else 1))
        treasury_max_actions = max(1, int(getattr(cfg, 'expansion_treasury_max_actions', max_actions) if cfg else max_actions))
        treasury_plan = None
        selection = {
            "mode": "idle",
            "reason": "no_eligible_boltz_actions",
            "reserve_deficit_sats": 0,
        }

        # Fetch coordination signals from capacity planner and rebalancer
        _planner_coord = None
        if capacity_planner is not None:
            try:
                _planner_coord = capacity_planner.get_boltz_coordination()
            except Exception:
                pass
        _rebalancer_coord = None
        if rebalancer is not None:
            try:
                _rebalancer_coord = rebalancer.get_boltz_coordination()
            except Exception:
                pass
        # Merge rebalancer exhaustion into planner coord for single-param passing
        if _planner_coord is None:
            _planner_coord = {}
        if _rebalancer_coord:
            _planner_coord["rebalancer_exhausted"] = _rebalancer_coord.get("rebalancer_exhausted", False)
            _planner_coord["rebalancer_depleted_count"] = _rebalancer_coord.get("depleted_count", 0)

        # One pending-swap listswaps subprocess for the whole cycle: every
        # plan build below (treasury, balance, fall-through) reuses this
        # count instead of paying its own boltzcli call. None = no treasury
        # involvement; the balance build then resolves it itself.
        pending_swaps: Optional[int] = None

        if bool(getattr(cfg, 'expansion_treasury_enabled', False)) if cfg else False:
            pending_swaps = _boltz_pending_swap_count()
            treasury_plan = _build_boltz_expansion_treasury_plan(
                onchain_target_sats=int(getattr(cfg, 'expansion_treasury_onchain_target_sats', 5_000_000) if cfg else 5_000_000),
                min_deficit_sats=int(getattr(cfg, 'expansion_treasury_min_deficit_sats', 250_000) if cfg else 250_000),
                preferred_currency=str(getattr(cfg, 'expansion_treasury_preferred_currency', 'BTC') if cfg else 'BTC').upper(),
                max_actions=treasury_max_actions,
                min_source_local_pct=float(getattr(cfg, 'expansion_treasury_min_source_local_pct', 80.0) if cfg else 80.0),
                exclude_protected=bool(getattr(cfg, 'expansion_treasury_exclude_protected', True)) if cfg else True,
                planner_coordination=_planner_coord,
                pending_swap_count=pending_swaps,
            )
            if isinstance(treasury_plan, dict) and 'error' in treasury_plan:
                result = dict(treasury_plan)
                result['trigger'] = trigger
                with _boltz_auto_cycle_state_lock:
                    _boltz_auto_cycle_state['consecutive_errors'] = int(_boltz_auto_cycle_state.get('consecutive_errors', 0) or 0) + 1
                _boltz_auto_cycle_mark_state(last_error=str(result.get('error')))
                return result
            selection = _select_boltz_auto_cycle_mode(treasury_plan=treasury_plan, balance_plan=None)

        balance_plan = None
        if selection.get("mode") != "treasury":
            balance_plan = _build_boltz_balance_plan(
                max_candidates=max(max(5, int(max_actions) * 5), 20),
                require_profitable=True,
                min_marginal_roi=0.0,
                profit_margin_factor=1.2,
                expected_horizon_days=3.0,
                loop_in_currency='auto',
                loop_out_currency='auto',
                planner_coordination=_planner_coord,
                pending_swap_count=pending_swaps,
            )
            if isinstance(balance_plan, dict) and 'error' in balance_plan:
                result = balance_plan
            else:
                # Detect when profitability data is completely missing —
                # all channels skipped with no_profitability_data means the
                # profitability analyzer hasn't produced results yet.
                if isinstance(balance_plan, dict):
                    skipped_examples = balance_plan.get("skipped_examples", [])
                    total_candidates = balance_plan.get("total_candidates", 0)
                    no_prof_count = sum(
                        1 for s in skipped_examples
                        if s.get("reason") == "no_profitability_data"
                    )
                    if total_candidates == 0 and no_prof_count > 0 and no_prof_count == len(skipped_examples):
                        plugin.log(
                            f"cl-revenue-ops: Boltz auto-cycle: all {no_prof_count} channels "
                            "lack profitability data — profitability analyzer may not have run yet",
                            level='warn',
                        )

                selection = _select_boltz_auto_cycle_mode(treasury_plan=treasury_plan, balance_plan=balance_plan)
                if selection.get("mode") == "balance":
                    # Reuse the plan built above for mode selection — rebuilding
                    # it would repeat every per-candidate boltzcli quote call.
                    result = _execute_boltz_balance_cycle(
                        dry_run=dry_run,
                        max_actions=max_actions,
                        allow_concurrent_swaps=False,
                        loop_in_currency='auto',
                        loop_out_currency='auto',
                        precomputed_plan=balance_plan,
                    )
                else:
                    reason = selection.get("reason", "no_eligible_boltz_actions")
                    # Surface profitability gap in the reason
                    if isinstance(balance_plan, dict):
                        skipped_examples = balance_plan.get("skipped_examples", [])
                        total_candidates = balance_plan.get("total_candidates", 0)
                        no_prof_count = sum(
                            1 for s in skipped_examples
                            if s.get("reason") == "no_profitability_data"
                        )
                        if total_candidates == 0 and no_prof_count > 0:
                            reason = f"no_profitability_data ({no_prof_count} channels)"
                    result = {
                        'status': 'idle',
                        'executed_count': 0,
                        'skipped_count': 0,
                        'reason': reason,
                    }
        else:
            # Reuse the selection-time treasury plan — rebuilding it inside
            # the cycle would repeat the nested balance-plan build (quotes,
            # budget, listswaps subprocesses) a second time per cycle.
            result = _execute_boltz_expansion_treasury_cycle(
                dry_run=dry_run,
                max_actions=treasury_max_actions,
                allow_concurrent_swaps=False,
                precomputed_plan=treasury_plan,
            )
            # F1 fall-through: when treasury mode executed nothing (every
            # candidate was skipped at execution time), run the balance
            # cycle within the same auto-cycle run so the structural drain
            # envelope still gets a turn. Blocked cycles (pending swaps)
            # would block the balance cycle too, so don't bother.
            _treasury_status = str(result.get('status') or '') if isinstance(result, dict) else ''
            _treasury_executed = int(result.get('executed_count', 0) or 0) if isinstance(result, dict) else 0
            if _treasury_status == 'executed' and _treasury_executed == 0:
                # Reuse the selection-time balance plan when one exists;
                # otherwise build it once (and only once) for the fall-through.
                if isinstance(balance_plan, dict) and 'error' not in balance_plan:
                    fallback_plan = balance_plan
                else:
                    fallback_plan = _build_boltz_balance_plan(
                        max_candidates=max(max(5, int(max_actions) * 5), 20),
                        require_profitable=True,
                        min_marginal_roi=0.0,
                        profit_margin_factor=1.2,
                        expected_horizon_days=3.0,
                        loop_in_currency='auto',
                        loop_out_currency='auto',
                        planner_coordination=_planner_coord,
                        pending_swap_count=pending_swaps,
                    )
                if isinstance(fallback_plan, dict) and 'error' not in fallback_plan:
                    fallback_selection = _select_boltz_auto_cycle_mode(
                        treasury_plan=None, balance_plan=fallback_plan,
                    )
                    if fallback_selection.get("mode") == "balance":
                        selection = dict(fallback_selection)
                        selection["reason"] = "treasury_executed_zero_fallback_to_balance"
                        result = _execute_boltz_balance_cycle(
                            dry_run=dry_run,
                            max_actions=max_actions,
                            allow_concurrent_swaps=False,
                            loop_in_currency='auto',
                            loop_out_currency='auto',
                            precomputed_plan=fallback_plan,
                        )
        status = str(result.get('status') or 'unknown') if isinstance(result, dict) else 'unknown'
        if isinstance(result, dict):
            result.update({
                'mode': str(selection.get("mode") or 'idle'),
                'selection_reason': str(selection.get("reason") or 'no_eligible_boltz_actions'),
                'reserve_deficit_sats': int(selection.get("reserve_deficit_sats", 0) or 0),
                'trigger': trigger,
            })
        if isinstance(result, dict) and 'error' in result:
            with _boltz_auto_cycle_state_lock:
                _boltz_auto_cycle_state['consecutive_errors'] = int(_boltz_auto_cycle_state.get('consecutive_errors', 0) or 0) + 1
            _boltz_auto_cycle_mark_state(last_error=str(result.get('error')))
        else:
            # C3 FIX: Only reset error counter on actual success, not on blocked/other states
            status = str(result.get('status') or 'unknown') if isinstance(result, dict) else 'unknown'
            if status in ('executed', 'dry_run'):
                with _boltz_auto_cycle_state_lock:
                    _boltz_auto_cycle_state['consecutive_errors'] = 0
            _boltz_auto_cycle_mark_state(last_error=None)
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
        if acquired:
            _boltz_auto_cycle_run_lock.release()


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
    global flow_analyzer, fee_controller, rebalancer, database, config, profitability_analyzer, capacity_planner, safe_plugin, policy_manager, boltz_manager, capex_engine, data_service
    
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
        # The plugin can also exit via stdin EOF (pyln run() returning), in
        # which case the SIGTERM handler never ran: signal the daemon loops
        # before tearing down their resources.
        try:
            shutdown_event.set()
        except Exception:
            pass
        if safe_plugin and hasattr(safe_plugin, 'rpc'):
            try:
                # P1-029: cancel_futures drops queued-but-unstarted RPC work,
                # but an in-flight worker blocked on a wedged lightningd cannot
                # be cancelled. Bound the drain so atexit can never block the
                # process exit forever waiting on such a worker.
                if not _bounded_executor_shutdown(safe_plugin.rpc._executor):
                    plugin.log(
                        "RPC executor did not drain within shutdown timeout; proceeding with exit",
                        level='warn'
                    )
                _bounded_executor_shutdown(safe_plugin.rpc._async_executor)
            except Exception:
                pass

        # Shutdown the active rebalance engine thread pool, if it was created.
        rebalance_engine = getattr(rebalancer, "rebalance_engine_v2", None) if rebalancer else None
        if rebalance_engine is not None and hasattr(rebalance_engine, "shutdown"):
            try:
                rebalance_engine.shutdown()
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
    def _safe_int(key):
        """Parse option as int with descriptive error on failure."""
        try:
            return int(options[key])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid integer for config '{key}': {options.get(key)!r}") from e

    def _safe_float(key, default=0.0):
        """Parse option as float with descriptive error on failure."""
        try:
            return float(options[key]) if key in options else float(options.get(key, default))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid float for config '{key}': {options.get(key)!r}") from e

    def _safe_int_opt(key, default=0):
        """Parse optional option as int with descriptive error on failure."""
        try:
            return int(options.get(key, default))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid integer for config '{key}': {options.get(key)!r}") from e

    def _safe_float_opt(key, default=0.0):
        """Parse optional option as float with descriptive error on failure."""
        try:
            return float(options.get(key, default))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid float for config '{key}': {options.get(key)!r}") from e

    config_kwargs = dict(
        db_path=os.path.expanduser(options['revenue-ops-db-path']),
        flow_interval=_safe_int('revenue-ops-flow-interval'),
        fee_interval=_safe_int('revenue-ops-fee-interval'),
        rebalance_interval=_safe_int('revenue-ops-rebalance-interval'),
        hot_channel_protection_enabled=options.get('revenue-ops-hot-channel-protection-enabled', 'true').lower() == 'true',
        hot_channel_protection_override_peers=str(options.get('revenue-ops-hot-channel-protection-override-peers', '') or ''),
        hot_channel_protection_min_velocity=_safe_float_opt('revenue-ops-hot-channel-protection-min-velocity', '0.20'),
        hot_channel_protection_min_marginal_roi=_safe_float_opt('revenue-ops-hot-channel-protection-min-marginal-roi', '0.20'),
        hot_channel_protection_profit_budget_pct=_safe_float_opt('revenue-ops-hot-channel-protection-profit-budget-pct', '0.75'),
        hot_channel_protection_max_chunk_multiplier=_safe_float_opt('revenue-ops-hot-channel-protection-max-chunk-multiplier', '4.0'),
        hot_channel_protection_min_cooldown_hours=_safe_float_opt('revenue-ops-hot-channel-protection-min-cooldown-hours', '1.0'),
        hot_channel_protection_max_rebalance_fee_ppm=_safe_int_opt('revenue-ops-hot-channel-protection-max-rebalance-fee-ppm', '2000'),
        boltz_auto_cycle_enabled=options.get('revenue-ops-boltz-auto-cycle-enabled', 'false').lower() == 'true',
        boltz_auto_cycle_interval_minutes=_safe_int_opt('revenue-ops-boltz-auto-cycle-interval-minutes', '15'),
        boltz_auto_cycle_max_actions=_safe_int_opt('revenue-ops-boltz-auto-cycle-max-actions', '1'),
        boltz_auto_cycle_startup_delay_seconds=_safe_int_opt('revenue-ops-boltz-auto-cycle-startup-delay-seconds', '120'),
        receivable_ratio_target=_safe_float_opt('revenue-ops-receivable-ratio-target', '0.30'),
        receivable_ratio_floor=_safe_float_opt('revenue-ops-receivable-ratio-floor', '0.20'),
        boltz_structural_budget_sats_per_day=_safe_int_opt('revenue-ops-boltz-structural-budget-sats', '0'),
        drain_fee_discount_max=_safe_float_opt('revenue-ops-drain-fee-discount-max', '0.0'),
        expansion_treasury_enabled=options.get('revenue-ops-expansion-treasury-enabled', 'false').lower() == 'true',
        expansion_treasury_onchain_target_sats=_safe_int_opt('revenue-ops-expansion-treasury-onchain-target-sats', '5000000'),
        expansion_treasury_min_deficit_sats=_safe_int_opt('revenue-ops-expansion-treasury-min-deficit-sats', '250000'),
        expansion_treasury_preferred_currency=str(options.get('revenue-ops-expansion-treasury-preferred-currency', 'BTC') or 'BTC').upper(),
        expansion_treasury_max_actions=_safe_int_opt('revenue-ops-expansion-treasury-max-actions', '1'),
        expansion_treasury_min_source_local_pct=_safe_float_opt('revenue-ops-expansion-treasury-min-source-local-pct', '80.0'),
        expansion_treasury_exclude_protected=options.get('revenue-ops-expansion-treasury-exclude-protected', 'true').lower() == 'true',
        target_flow=_safe_int('revenue-ops-target-flow'),
        min_fee_ppm=_safe_int('revenue-ops-min-fee-ppm'),
        max_fee_ppm=_safe_int('revenue-ops-max-fee-ppm'),
        market_fee_mode=options.get('revenue-ops-market-fee-mode', 'undercut').lower(),
        base_fee_policy=options.get('revenue-ops-base-fee-policy', 'off').lower(),
        base_fee_msat_intra_fleet=_safe_int_opt('revenue-ops-base-fee-intra-fleet', '0'),
        base_fee_msat_non_hive=_safe_int_opt('revenue-ops-base-fee-non-hive', '1000'),
        fee_ppm_intra_fleet=_safe_int_opt('revenue-ops-fee-ppm-intra-fleet', '1'),
        neighbor_median_min_competitors=_safe_int_opt('revenue-ops-neighbor-median-min-competitors', '2'),
        fee_profile=str(options.get('revenue-ops-fee-profile', 'active') or 'active').lower(),
        fee_market_boundary_enabled=options.get(
            'revenue-ops-fee-market-boundary-enabled', 'false'
        ).lower() == 'true',
        fee_market_boundary_min_competitors=_safe_int_opt(
            'revenue-ops-fee-market-boundary-min-competitors', '3'
        ),
        fee_market_boundary_margin_ppm=_safe_int_opt(
            'revenue-ops-fee-market-boundary-margin-ppm', '5'
        ),
        fee_market_boundary_margin_ratio=_safe_float_opt(
            'revenue-ops-fee-market-boundary-margin-ratio', '0.05'
        ),
        fee_market_boundary_max_downshift_ratio=_safe_float_opt(
            'revenue-ops-fee-market-boundary-max-downshift-ratio', '0.35'
        ),
        fee_market_boundary_cache_seconds=_safe_int_opt(
            'revenue-ops-fee-market-boundary-cache-seconds', '60'
        ),
        rebalance_min_profit=_safe_int('revenue-ops-rebalance-min-profit'),
        rebalance_emergency_local_ratio=_safe_float_opt(
            'revenue-ops-rebalance-emergency-local-ratio', '0.10'
        ),
        rebalance_drift_override_ratio=_safe_float_opt(
            'revenue-ops-rebalance-drift-override-ratio', '0.30'
        ),
        rebalance_hold_margin=_safe_float_opt(
            'revenue-ops-rebalance-hold-margin', '0.0'
        ),
        pair_fee_cap_ppm=_safe_int_opt(
            'revenue-ops-pair-fee-cap-ppm', '1000'
        ),
        hive_rebalance_bootstrap_budget_sats=_safe_int_opt(
            'revenue-ops-hive-rebalance-bootstrap-budget-sats', '300'
        ),
        rebalance_coordination_reserved_slots=_safe_int_opt(
            'revenue-ops-rebalance-coordination-reserved-slots', '2'
        ),
        futility_cooldown_hours=_safe_int('revenue-ops-futility-cooldown-hours'),
        flow_window_days=_safe_int('revenue-ops-flow-window-days'),
        daily_budget_sats=_safe_int('revenue-ops-daily-budget-sats'),
        diagnostic_rebalance_max_fee_sats=_safe_int_opt(
            'revenue-ops-diagnostic-rebalance-max-fee-sats', '400'
        ),
        allow_zero_cost_auto_rebalance_when_budget_zero=options.get(
            'revenue-ops-allow-zero-cost-auto-rebalance-when-budget-zero', 'false'
        ).lower() in ('true', '1', 'yes'),
        weekly_budget_sats=_safe_int('revenue-ops-weekly-budget-sats'),
        min_wallet_reserve=_safe_int('revenue-ops-min-wallet-reserve'),
        dry_run=options['revenue-ops-dry-run'].lower() == 'true',
        htlc_congestion_threshold=_safe_float('revenue-ops-htlc-congestion-threshold'),
        enable_reputation=options['revenue-ops-enable-reputation'].lower() == 'true',
        reputation_decay=_safe_float('revenue-ops-reputation-decay'),
        enable_kelly=options['revenue-ops-enable-kelly'].lower() == 'true',
        # Vegas Reflex options
        enable_vegas_reflex=options['revenue-ops-vegas-reflex'].lower() == 'true',
        vegas_decay_rate=_safe_float('revenue-ops-vegas-decay'),
        rpc_timeout_seconds=_safe_int('revenue-ops-rpc-timeout-seconds'),
        rpc_circuit_breaker_seconds=_safe_int('revenue-ops-rpc-circuit-breaker-seconds'),
        reservation_timeout_hours=_safe_int('revenue-ops-reservation-timeout-hours'),
        planner_enabled=options.get('revenue-ops-planner-enabled', 'false').lower() in ('true', '1', 'yes'),
        planner_interval=_safe_int('revenue-ops-planner-interval'),
        planner_dry_run=options.get('revenue-ops-planner-dry-run', 'false').lower() in ('true', '1', 'yes'),
        planner_execute_closes=options.get('revenue-ops-planner-execute-closes', 'false').lower() in ('true', '1', 'yes'),
        planner_max_opens_per_cycle=_safe_int('revenue-ops-planner-max-opens-per-cycle'),
        planner_max_closes_per_cycle=_safe_int('revenue-ops-planner-max-closes-per-cycle'),
        planner_close_fee_reserve_multiplier=_safe_float('revenue-ops-planner-close-fee-reserve-multiplier'),
        planner_close_fee_cap_sats=_safe_int('revenue-ops-planner-close-fee-cap-sats'),
        planner_close_feerange_enabled=options.get('revenue-ops-planner-close-feerange-enabled', 'false').lower() in ('true', '1', 'yes'),
        planner_min_channel_sats=_safe_int('revenue-ops-planner-min-channel-sats'),
        planner_max_channel_sats=_safe_int('revenue-ops-planner-max-channel-sats'),
        planner_max_fee_rate_sat_vb=_safe_float('revenue-ops-planner-max-fee-rate'),
        planner_min_annual_roi_pct=_safe_float_opt(
            'revenue-ops-planner-min-annual-roi-pct',
            '1.0',
        ),
        hive_hints_enabled=options.get('revenue-ops-hive-hints-enabled', 'true').lower() in ('true', '1', 'yes'),
        hive_hints_ttl_seconds=_safe_int('revenue-ops-hive-hints-ttl'),
        hive_hints_allow_all_hints_m2_scope=options.get(
            'revenue-ops-hive-hints-allow-all-hints-m2-scope', 'false'
        ).lower() in ('true', '1', 'yes'),
        rebalance_router='v3',
        askrene_layers=str(options.get('revenue-ops-askrene-layers', '') or '').strip() or 'hive-fleet',
    )
    # P1-008: range-validate numeric options before Config construction so a
    # 0/negative/out-of-band value can never fail open.
    _validate_numeric_config_options(config_kwargs, log=plugin.log)
    # P1-009: enforce min_fee_ppm <= max_fee_ppm before construction.
    _enforce_fee_bound_invariant(config_kwargs, log=plugin.log)
    # P1-026: validate enum-style options before construction.
    _validate_enum_config_options(config_kwargs, log=plugin.log)

    configured_router = str(options.get('revenue-ops-rebalance-router', 'v3') or 'v3').lower()
    if configured_router != 'v3':
        plugin.log(
            f"Configuration requested rebalance-router={configured_router!r}; forcing 'v3' because legacy routing was removed",
            level='warn',
        )
    try:
        config_fields = {f.name for f in dataclasses.fields(Config)}
    except Exception as e:
        plugin.log(f"Config field introspection failed, using kwargs: {e}", level='debug')
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
               f"fee_profile={config.fee_profile}, "
               f"rebalance_executor=native, "
               f"dry_run={config.dry_run}")
    
    # Create thread-safe RPC proxy (High-Uptime Stability)
    # pyln-client opens a new Unix socket per call — thread-safe by design.
    # ThreadSafeRpcProxy adds timeout protection via ThreadPoolExecutor.
    safe_plugin = ThreadSafePluginProxy(plugin)
    plugin.log("Thread-safe RPC proxy initialized", level="info")

    from modules.data_service import DataService
    data_service = DataService(safe_plugin)

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
            routing_fee_limit_ppm=int(options.get('revenue-ops-boltz-routing-fee-limit-ppm', '0')),
            max_withdraw_sats=int(options.get('revenue-ops-boltz-max-withdraw-sats', '10000000')),
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
    # STARTUP VERSION PROBES (Phase 3C supply-chain)
    # NON-FATAL: a version below floor logs a warning but never aborts init.
    # Hard-failing here would brick a running node on a benign version skew.
    # =========================================================================
    try:
        cln_version = None
        try:
            cln_version = data_service._ensure_getinfo().get("version")
        except Exception:
            cln_version = None
        below = _version_below_floor(cln_version, CLN_VERSION_FLOOR)
        if below is True:
            plugin.log(
                f"Core Lightning version {cln_version} is BELOW the supported "
                f"floor v{CLN_VERSION_FLOOR}. The askrene layer lifecycle this "
                f"plugin depends on may be missing or partial; behavior is "
                f"unsupported. Upgrade Core Lightning. Continuing anyway.",
                level='warn'
            )
        elif below is None:
            plugin.log(
                f"Could not determine Core Lightning version (got {cln_version!r}); "
                f"skipping the v{CLN_VERSION_FLOOR} floor check.",
                level='debug'
            )
        else:
            plugin.log(f"Core Lightning version check OK ({cln_version} >= v{CLN_VERSION_FLOOR})")
    except Exception as e:
        plugin.log(f"Core Lightning version probe failed (non-fatal): {e}", level='warn')

    # boltzcli --version probe (only when the integration is enabled and a
    # binary path is configured). Warning-only: a skew must not block init.
    try:
        if boltz_manager is not None and getattr(boltz_manager, "enabled", False):
            import subprocess
            cli_path = boltz_cfg.cli_path
            try:
                proc = subprocess.run(
                    [cli_path, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    text=True,
                )
                boltz_version_raw = (proc.stdout or "").strip()
                below = _version_below_floor(boltz_version_raw, BOLTZCLI_MIN_VERSION)
                if below is True:
                    plugin.log(
                        f"boltzcli reports version '{boltz_version_raw}', below the "
                        f"documented minimum v{BOLTZCLI_MIN_VERSION}. Loop-in/loop-out "
                        f"channel-pinning and quote fields may differ. Continuing anyway.",
                        level='warn'
                    )
                elif below is None:
                    plugin.log(
                        f"Could not parse boltzcli version from '{boltz_version_raw}'; "
                        f"skipping the v{BOLTZCLI_MIN_VERSION} minimum check.",
                        level='warn'
                    )
                else:
                    plugin.log(f"boltzcli version check OK ({boltz_version_raw} >= v{BOLTZCLI_MIN_VERSION})")
            except FileNotFoundError:
                plugin.log(
                    f"boltzcli not found at '{cli_path}' during version probe; "
                    f"Boltz integration is enabled but the binary is missing.",
                    level='warn'
                )
            except subprocess.TimeoutExpired:
                plugin.log("boltzcli --version timed out (non-fatal); skipping version check", level='warn')
    except Exception as e:
        plugin.log(f"boltzcli version probe failed (non-fatal): {e}", level='warn')

    # =========================================================================
    # STARTUP DEPENDENCY CHECKS (Stability)
    # Verify external plugins are available before initializing dependent modules
    # =========================================================================
    try:
        # Try modern 'plugin list' command first, fallback to 'listplugins' for older nodes
        plugins_result = data_service.list_plugins()
            
        active_plugins = [p.get("name", "").lower() for p in plugins_result.get("plugins", [])]
        
        plugin.log("Rebalancing uses RebalanceEngineV2 with native route execution")

        # Check for bookkeeper plugin
        bookkeeper_found = any("bookkeeper" in name for name in active_plugins)
        if not bookkeeper_found:
            plugin.log(
                "Dependency 'bookkeeper' not found. Flow analysis uses the local forwards table (hydrated once at startup). "
                "Bookkeeper is still recommended for accurate on-chain cost tracking (opens/closes).",
                level='info'
            )
        else:
            plugin.log("Dependency check: bookkeeper plugin detected")
            
    except Exception as e:
        plugin.log(f"Error checking plugin dependencies: {e}", level='warn')


    # Initialize database
    database = Database(config.db_path, safe_plugin)
    database.initialize()

    # Issue #24: Clean up stale budget reservations on startup
    # Reservations from crashed jobs should be released immediately
    timeout_seconds = config.reservation_timeout_hours * 3600
    cleaned = database.cleanup_stale_reservations(timeout_seconds)
    if cleaned > 0:
        plugin.log(f"Startup cleanup: Released {cleaned} stale budget reservations")

    # Load config overrides from database (persisted runtime changes)
    try:
        override_warnings = config.load_overrides(database)
        if config._version > 0:
            plugin.log(f"Loaded config overrides from database (version {config._version})")
        for w in override_warnings:
            plugin.log(f"Config override: {w}", level='warn')
    except Exception as e:
        plugin.log(f"Warning: Could not load config overrides: {e}", level='warn')

    # =========================================================================
    # FORWARDS TABLE HYDRATION (#19: double-dip prevented)
    # =========================================================================
    # The forwards table is populated in real-time by forward_event hook.
    # Startup hydration backfills empty tables and bounded overlap gaps.
    # Hook and hydration overlap is safe: both insert paths use
    # INSERT OR IGNORE under idx_forwards_unique with the same SCID and
    # timestamp normalization (see TestForwardDoubleDipPrevention).
    # The RPC fetch prefers a paged listforwards(index="created") loop with a
    # full-fetch fallback; the local insert window below bounds what gets written.
    # =========================================================================
    try:
        last_forward_ts = database.get_latest_forward_timestamp()
        now = int(time.time())

        start_time = _compute_forward_hydration_start(
            last_forward_ts,
            config.flow_window_days,
            now,
        )

        if start_time is None:
            # Table is current enough — forward_event hook keeps it current.
            gap_hours = (now - last_forward_ts) / 3600
            if gap_hours > 1:
                plugin.log(
                    f"Forwards table has {gap_hours:.1f}h gap — "
                    f"forward_event hook will catch up naturally",
                    level='debug'
                )
        elif last_forward_ts is None:
            hydrate_days = max(config.flow_window_days, 14)
            plugin.log(f"Forwards table empty. Hydrating last {hydrate_days} days of forwards...")
        else:
            gap_hours = (now - last_forward_ts) / 3600
            overlap_days = max(config.flow_window_days + 1, 15)
            plugin.log(
                f"Forwards table has {gap_hours:.1f}h gap — "
                f"hydrating bounded overlap window capped at {overlap_days} days",
                level='debug'
            )

        if start_time is not None:
            # Fetch from RPC via _hydration_fetch_settled_forwards: paged
            # listforwards(index="created") when supported, full settled fetch
            # otherwise. The received_time filter bounds the insert window in
            # both cases (`start` pages by created_index, not by timestamp).
            #
            # Empty-table warm starts already use the helper's exact window.
            # Apply the extra-day overlap floor only when we have a non-empty
            # table and want to backfill a stale gap.
            if last_forward_ts is not None:
                max_hydration_days = max(config.flow_window_days + 1, 15)
                hydration_floor = now - (max_hydration_days * 86400)
                start_time = max(start_time, hydration_floor)
            forwards_to_insert = []
            for fwd in _hydration_fetch_settled_forwards(start_time):
                received_time = int(fwd.get("received_time", 0) or 0)
                resolved_time = int(fwd.get("resolved_time", 0) or 0)
                forwards_to_insert.append({
                    'in_channel': fwd.get("in_channel", ""),
                    'out_channel': fwd.get("out_channel", ""),
                    'in_msat': _parse_msat(fwd.get("in_msat", fwd.get("in_msatoshi", 0))),
                    'out_msat': _parse_msat(fwd.get("out_msat", fwd.get("out_msatoshi", 0))),
                    'fee_msat': _parse_msat(fwd.get("fee_msat", fwd.get("fee_msatoshi", 0))),
                    'resolution_time': max(0, resolved_time - received_time) if resolved_time > 0 else 0,
                    'received_time': received_time,
                    'resolved_time': resolved_time
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
        peers = data_service.get_peers()
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
    
    # Initialize policy manager (v1.4: Policy-Driven Architecture)
    policy_manager = PolicyManager(database, safe_plugin)
    plugin.log("PolicyManager initialized for peer-level fee/rebalance policies")

    # Initialize profitability analyzer
    profitability_analyzer = ChannelProfitabilityAnalyzer(
        safe_plugin, config, database
    )

    # Initialize analysis modules
    flow_analyzer = FlowAnalyzer(safe_plugin, config, database)
    capacity_planner = CapacityPlanner(safe_plugin, profitability_analyzer, flow_analyzer, policy_manager=policy_manager, config=config)
    fee_controller = FeeController(
        safe_plugin,
        config,
        database,
        policy_manager,
        profitability_analyzer,
    )
    rebalancer = EVRebalancer(
        safe_plugin, config, database, policy_manager,
    )
    rebalancer.set_profitability_analyzer(profitability_analyzer)
    rebalancer.set_capacity_planner(capacity_planner)
    capacity_planner.rebalancer = rebalancer
    if hasattr(capacity_planner, "set_rebalancer"):
        capacity_planner.set_rebalancer(rebalancer)
    # Unified liquidity-cost accounting:
    # - Rebalancer sees Boltz spend as external liquidity cost
    # - Boltz manager sees rebalance spend/reservations as external liquidity cost
    if rebalancer is not None:
        rebalancer.external_liquidity_cost_provider = _non_rebalance_liquidity_cost_components
        rebalancer.global_budget_limit_provider = _total_cost_budget_limit_provider
    if boltz_manager is not None:
        boltz_manager.external_liquidity_cost_provider = _non_boltz_liquidity_cost_components
        boltz_manager.global_budget_limit_provider = _total_cost_budget_limit_provider
        # Live envelope read (runtime-mutable config): with a configured
        # structural envelope, structural loop-outs skip the per-channel
        # capex gate inside the manager — the envelope + unified budget gate
        # them instead.
        boltz_manager.structural_envelope_provider = _structural_envelope_sats_provider

    # Hive Hints adapter (sole integration boundary with cl_hive)
    global hive_hints
    if config.hive_hints_enabled:
        hive_hints = HiveHintAdapter(
            safe_plugin,
            ttl_override=config.hive_hints_ttl_seconds,
            allow_all_hints_m2_scope=config.hive_hints_allow_all_hints_m2_scope,
        )
        plugin.log("HiveHintAdapter initialized - fleet hint bias enabled")
    else:
        hive_hints = None

    if fee_controller is not None:
        fee_controller.hive_hints = hive_hints
    if rebalancer is not None:
        rebalancer.hive_hints = hive_hints
    if profitability_analyzer is not None:
        profitability_analyzer.hive_hints = hive_hints
    if policy_manager is not None:
        policy_manager.hive_hints = hive_hints
    if capacity_planner is not None:
        capacity_planner.hive_hints = hive_hints
        capacity_planner.global_budget_limit_provider = _total_cost_budget_limit_provider
        capacity_planner.external_liquidity_cost_provider = _non_boltz_liquidity_cost_components

    capital_efficiency = CapitalEfficiencyAnalyzer(
        profitability_analyzer=profitability_analyzer,
        flow_analyzer=flow_analyzer,
        database=database,
        hive_hints=hive_hints,
        config=config,
    )
    plugin.log("CapitalEfficiencyAnalyzer initialized")

    # Construct unified capex budget engine (after hive_hints are available)
    capex_engine = CapexBudgetEngine(
        profitability_analyzer=profitability_analyzer,
        database=database,
        config=config,
        hive_hints=hive_hints,
        capital_efficiency=capital_efficiency,
        hive_member_check=rebalancer._is_hive_member if rebalancer is not None else None,
    )
    plugin.log("CapexBudgetEngine initialized")

    # Wire capex engine to all consumers
    if rebalancer is not None:
        rebalancer.set_capex_engine(capex_engine)
    if capacity_planner is not None:
        capacity_planner.set_capital_efficiency(capital_efficiency)
        capacity_planner.set_capex_engine(capex_engine)
        capacity_planner.global_budget_limit_provider = _total_cost_budget_limit_provider
        capacity_planner.external_liquidity_cost_provider = _non_boltz_liquidity_cost_components
    if boltz_manager is not None:
        boltz_manager.set_capex_engine(capex_engine)

    # Hive Router (shared askrene fleet route discovery)
    global hive_router
    hive_router = None
    if hive_hints is not None:
        hive_router = HiveRouter(safe_plugin, hive_hints)
        plugin.log("HiveRouter initialized - fleet route discovery enabled")

    if rebalancer is not None and hive_router is not None:
        rebalancer.hive_router = hive_router

    if hive_router is not None:
        hive_router.data_service = data_service
        if profitability_analyzer is not None:
            hive_router.profitability_analyzer = profitability_analyzer

    # Rebalance engine: unified actual-fee pipeline
    from modules.rebalance_engine_v2 import RebalanceEngine
    segment_observation_store = SegmentObservationStore()
    if rebalancer is not None:
        rebalancer.rebalance_engine_v2 = RebalanceEngine(
            plugin=safe_plugin,
            config=config,
            database=database,
            capex_engine=capex_engine,
            profitability=profitability_analyzer,
            hive_hints=hive_hints,
            data_service=data_service,
            hive_router=hive_router,
            segment_observation_store=segment_observation_store,
            global_budget_limit_provider=_total_cost_budget_limit_provider,
            external_liquidity_cost_provider=_non_rebalance_liquidity_cost_components,
        )
        rebalancer.data_service = data_service
        plugin.log("RebalanceEngine initialized")

    if fee_controller is not None:
        fee_controller.data_service = data_service
    if profitability_analyzer is not None:
        profitability_analyzer.data_service = data_service
    if policy_manager is not None:
        policy_manager.data_service = data_service
    if flow_analyzer is not None:
        flow_analyzer.data_service = data_service
    if capacity_planner is not None:
        capacity_planner.data_service = data_service
    if boltz_manager is not None:
        boltz_manager.data_service = data_service
    if hive_hints is not None:
        hive_hints.data_service = data_service

    # Set up periodic background tasks using threading
    # Note: plugin.log() is safe to call from threads in pyln-client
    # We use daemon threads so they don't block shutdown
    
    def flow_analysis_loop():
        """Background loop for flow analysis."""
        # Staggered startup: flow at 30s (was 10s) to avoid thundering herd
        if shutdown_event.wait(30):
            plugin.log("Flow analysis loop cancelled during startup delay")
            return
        
        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard wraps the ENTIRE iteration (work AND
            # the interval/sleep tail) so no exception can kill the thread.
            try:
                _record_loop_heartbeat("flow-analysis")
                try:
                    plugin.log("Running scheduled flow analysis...")
                    run_flow_analysis()

                    # Run cleanup on each iteration (it's a fast DELETE query)
                    # Keeps history tables from growing unbounded over months
                    # Use flow_window_days + 1 day buffer, minimum 8 days
                    if database:
                        cleanup_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                        days_to_keep = max(8, cleanup_snap.flow_window_days + 1)
                        database.cleanup_old_data(days_to_keep=days_to_keep)

                    # AUDIT FIX PM-5: Clean up expired time-limited policies
                    if policy_manager:
                        try:
                            policy_manager.cleanup_expired_policies()
                        except Exception as e:
                            plugin.log(f"Error cleaning expired policies: {e}", level='debug')

                except (RPCTimeoutError, RPCBreakerOpen) as e:
                    plugin.log(f"RPC degraded in flow analysis: {e}. Skipping this cycle.", level='warn')
                except Exception as e:
                    plugin.log(f"Error in flow analysis: {e}", level='error')

                # M-3 FIX: Use config snapshot for interval to avoid mid-loop mutation
                cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                interval = max(60, cfg_snap.flow_interval)
                jitter_seconds = int(interval * 0.2)
                sleep_time = interval + random.randint(-jitter_seconds, jitter_seconds)
                plugin.log(f"Flow analysis sleeping for {sleep_time}s")

                # Interruptible sleep: wait for timeout OR shutdown signal
                if shutdown_event.wait(sleep_time):
                    plugin.log("Flow analysis loop stopping due to shutdown signal")
                    break
            except Exception as e:
                plugin.log(f"Unhandled error in flow-analysis loop iteration: {e}", level='error')
                try:
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
                    break
    
    def fee_adjustment_loop():
        """Background loop for fee adjustment."""
        # Staggered startup: fees at 90s (was 60s) to avoid thundering herd
        if shutdown_event.wait(90):
            plugin.log("Fee adjustment loop cancelled during startup delay")
            return

        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard over the ENTIRE iteration incl. tail.
            try:
                _record_loop_heartbeat("fee-adjustment")
                _refresh_fee_cycle_hive_inputs()

                try:
                    plugin.log("Running scheduled fee adjustment...")
                    run_fee_adjustment()
                except (RPCTimeoutError, RPCBreakerOpen) as e:
                    plugin.log(f"RPC degraded in fee adjustment: {e}. Skipping this cycle.", level='warn')
                except Exception as e:
                    plugin.log(f"Error in fee adjustment: {e}", level='error')

                # M-3 FIX: Use config snapshot for interval to avoid mid-loop mutation
                cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                interval = max(60, cfg_snap.fee_interval)
                jitter_seconds = int(interval * 0.2)
                sleep_time = interval + random.randint(-jitter_seconds, jitter_seconds)
                plugin.log(f"Fee adjustment sleeping for {sleep_time}s")

                # Interruptible sleep: wait for timeout OR shutdown signal
                if shutdown_event.wait(sleep_time):
                    plugin.log("Fee adjustment loop stopping due to shutdown signal")
                    break
            except Exception as e:
                plugin.log(f"Unhandled error in fee-adjustment loop iteration: {e}", level='error')
                try:
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
                    break
    
    def rebalance_check_loop():
        """Background loop for rebalance checks."""
        # Staggered startup: rebalance at 180s (was 120s) to avoid thundering herd
        if shutdown_event.wait(180):
            plugin.log("Rebalance check loop cancelled during startup delay")
            return
        
        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard over the ENTIRE iteration incl. tail.
            try:
                _record_loop_heartbeat("rebalance-check")
                try:
                    refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router, log=plugin.log)
                except Exception:
                    pass  # fail-open
                try:
                    plugin.log("Running scheduled rebalance check...")
                    run_rebalance_check()
                except (RPCTimeoutError, RPCBreakerOpen) as e:
                    plugin.log(f"RPC degraded in rebalance check: {e}. Skipping this cycle.", level='warn')
                except Exception as e:
                    plugin.log(f"Error in rebalance check: {e}", level='error')

                # M-3 FIX: Use config snapshot for interval to avoid mid-loop mutation
                cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                interval = max(60, cfg_snap.rebalance_interval)
                jitter_seconds = int(interval * 0.2)
                sleep_time = interval + random.randint(-jitter_seconds, jitter_seconds)
                plugin.log(f"Rebalance check sleeping for {sleep_time}s")

                # Interruptible sleep: wait for timeout OR shutdown signal
                if shutdown_event.wait(sleep_time):
                    plugin.log("Rebalance check loop stopping due to shutdown signal")
                    break
            except Exception as e:
                plugin.log(f"Unhandled error in rebalance-check loop iteration: {e}", level='error')
                try:
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
                    break
    
    def boltz_auto_cycle_loop():
        """Background loop for profit-gated Boltz auto-balance cycles."""
        if boltz_manager is None or not getattr(boltz_manager, 'enabled', False):
            _boltz_auto_cycle_mark_state(enabled=False, thread_started=False, next_run_ts=None)
            plugin.log("Boltz auto-cycle loop disabled: Boltz CLI integration not enabled", level='debug')
            return

        _boltz_auto_cycle_mark_state(enabled=bool(getattr(config, 'boltz_auto_cycle_enabled', False)), thread_started=True)

        startup_delay = max(0, int(getattr(config, 'boltz_auto_cycle_startup_delay_seconds', 120) or 0))
        if startup_delay > 0:
            _boltz_auto_cycle_mark_state(next_run_ts=int(time.time()) + startup_delay)
            if shutdown_event.wait(startup_delay):
                plugin.log("Boltz auto-cycle loop cancelled during startup delay")
                _boltz_auto_cycle_mark_state(next_run_ts=None)
                return

        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard over the ENTIRE iteration incl. tail.
            try:
                _record_loop_heartbeat("boltz-auto-cycle")
                enabled = bool(getattr(config, 'boltz_auto_cycle_enabled', False))
                _boltz_auto_cycle_mark_state(enabled=enabled)

                if enabled:
                    try:
                        _refresh_dynamic_config()
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
            except Exception as e:
                plugin.log(f"Unhandled error in boltz-auto-cycle loop iteration: {e}", level='error')
                try:
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
                    break

        _boltz_auto_cycle_mark_state(next_run_ts=None, running=False)

    def capacity_planner_loop():
        """Background loop for automated capacity planning."""
        if not config.planner_enabled:
            plugin.log("Capacity planner disabled, loop not started", level='debug')
            return

        # Respect the interval from the last cycle (survives restarts).
        # Fall back to a warmup delay if no prior cycle exists.
        warmup_delay = 300
        startup_delay = warmup_delay
        try:
            if database:
                recent = database.get_planner_actions(limit=1)
                if recent:
                    last_ts = recent[0].get("created_at", 0)
                    elapsed = int(time.time()) - last_ts
                    interval = max(600, config.planner_interval if hasattr(config, 'planner_interval') else 21600)
                    remaining = interval - elapsed
                    if remaining > 0:
                        startup_delay = remaining
                        plugin.log(f"Planner: last cycle {elapsed}s ago, waiting {remaining}s", level='info')
                    else:
                        plugin.log(f"Planner: last cycle {elapsed}s ago, starting after warmup", level='info')
        except Exception:
            pass  # Fall back to default warmup delay

        if shutdown_event.wait(startup_delay):
            return

        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard over the ENTIRE iteration incl. tail.
            try:
                _record_loop_heartbeat("capacity-planner")
                try:
                    _refresh_dynamic_config()
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
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')

                cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                interval = max(600, cfg_snap.planner_interval)
                jitter = int(interval * 0.2)
                sleep_time = interval + random.randint(-jitter, jitter)
                if shutdown_event.wait(sleep_time):
                    break
            except Exception as e:
                plugin.log(f"Unhandled error in capacity-planner loop iteration: {e}", level='error')
                try:
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
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

        # P1-021: delegate to the extracted helper, which closes the
        # thread-local DB connection before this one-shot thread exits.
        # DD5 / P1-010: guard the single run so an exception cannot kill the
        # thread silently; record a heartbeat so the one-shot is observable.
        _record_loop_heartbeat("startup-snapshot")
        try:
            _snapshot_peers_once()
        except Exception as e:
            plugin.log(f"Startup snapshot failed: {e}", level='error')
            try:
                plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
            except Exception:
                pass

    def financial_snapshot_loop():
        """
        Background loop for daily financial snapshots (Dashboard).

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
            # DD5 / P1-010: canonical guard over the ENTIRE iteration. This loop
            # is inverted (sleep-tail at the TOP), so the guard wraps it too.
            try:
                _record_loop_heartbeat("financial-snapshot")
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
            except Exception as e:
                plugin.log(f"Unhandled error in financial-snapshot loop iteration: {e}", level='error')
                try:
                    plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
                    break

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

    # Start background threads (daemon=True so they don't block shutdown)
    threading.Thread(target=flow_analysis_loop, daemon=True, name="flow-analysis").start()
    threading.Thread(target=fee_adjustment_loop, daemon=True, name="fee-adjustment").start()
    threading.Thread(target=rebalance_check_loop, daemon=True, name="rebalance-check").start()
    threading.Thread(target=snapshot_peers_delayed, daemon=True, name="startup-snapshot").start()
    threading.Thread(target=financial_snapshot_loop, daemon=True, name="financial-snapshot").start()
    threading.Thread(target=boltz_auto_cycle_loop, daemon=True, name="boltz-auto-cycle").start()
    threading.Thread(target=capacity_planner_loop, daemon=True, name="capacity-planner").start()

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
        balanced = sum(1 for r in results.values() if r.state.is_balanced)
        plugin.log(f"Channel states: {sources} sources, {sinks} sinks, {balanced} balanced")
        
        # Apply reputation decay (Time-windowing)
        # This ensures recent peer behavior matters more than ancient history
        if database and config and config.enable_reputation:
            database.decay_reputation(config.reputation_decay)
            plugin.log(f"Applied reputation decay (factor={config.reputation_decay})")

    except Exception as e:
        plugin.log(f"Flow analysis failed: {e}", level='error')
        raise


def _refresh_fee_cycle_hive_inputs():
    """Refresh advisory hive inputs before any fee adjustment path."""
    try:
        refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router, log=plugin.log)
    except Exception:
        pass  # fail-open

    if policy_manager is not None and hive_hints is not None:
        try:
            policy_manager.apply_corridor_policies()
        except Exception:
            pass  # fail-open



def run_fee_adjustment():
    """
    Module 2: DTS+PID Fee Controller (Dynamic Pricing)

    Adjust channel fees using DTS+PID optimization.
    """
    if fee_controller is None:
        plugin.log("Fee controller not initialized", level='error')
        return []
    
    try:
        adjustments = fee_controller.adjust_all_fees()
        plugin.log(f"Fee adjustment complete: {len(adjustments)} channels adjusted")

        # Push status + profitability to datastore for external consumers.
        # This keeps local reporting cheap and consistent.
        try:
            import json as _json
            status_data = {
                "operator_controls": {
                    "values": config.public_runtime_dict() if config else {},
                },
                "fee_decision": (
                    fee_controller.get_last_decision_summary()
                    if hasattr(fee_controller, "get_last_decision_summary")
                    else {}
                ),
            }
            if data_service:
                data_service.datastore_push(["revenue", "status"], status_data)

            # Push fee bounds for external consumers that read the datastore snapshot.
            cfg_snap = config.snapshot() if config else None
            if cfg_snap and data_service:
                data_service.datastore_push(["revenue", "fee-bounds"], {
                    "min_fee_ppm": cfg_snap.min_fee_ppm,
                    "max_fee_ppm": cfg_snap.max_fee_ppm,
                    "mid_fee_ppm": (cfg_snap.min_fee_ppm + cfg_snap.max_fee_ppm) // 2,
                })

            # NOTE: revenue-profitability is too large for datastore (47 channels
            # of detailed data).  The Bridge's get_profitability() call is
            # infrequent and can stay as cross-plugin RPC fallback.
        except Exception:
            pass  # Datastore push is best-effort

        # Push dashboard snapshot (cheap, idempotent, runs each fee cycle)
        _push_dashboard_to_datastore()
        return adjustments

    except Exception as e:
        plugin.log(f"Fee adjustment failed: {e}", level='error')
        raise


def _push_dashboard_to_datastore() -> None:
    """Push 30-day dashboard snapshot to datastore."""
    global safe_plugin, profitability_analyzer, database, data_service
    if safe_plugin is None or profitability_analyzer is None or database is None:
        return
    try:
        dashboard = revenue_dashboard(plugin, window_days=30)
        if isinstance(dashboard, dict) and "error" not in dashboard:
            if data_service:
                data_service.datastore_push(["revenue", "dashboard"], dashboard)
    except Exception:
        pass  # datastore_push handles its own error logging


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
        plugin.log(f"Rebalance check complete: {len(candidates)} candidates found")
        
        for candidate in candidates:
            rebalancer.execute_rebalance(candidate)
            
    except Exception as e:
        plugin.log(f"Rebalance check failed: {e}", level='error')
        raise


@plugin.method("revenue-rebalance-cycle")
def revenue_rebalance_cycle(plugin: Plugin, max_candidates: int = 20) -> Dict[str, Any]:
    """Run one automatic rebalance cycle immediately and return debug state."""
    if rebalancer is None:
        return {"error": "Rebalancer not initialized"}
    try:
        refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router, log=plugin.log)
    except Exception:
        pass  # fail-open; missing hive must not block standalone rebalancing
    try:
        run_rebalance_check()
        engine = getattr(rebalancer, "rebalance_engine_v2", None)
        cycle_debug = (
            engine.get_last_cycle_debug(max_candidates=max_candidates)
            if engine is not None and hasattr(engine, "get_last_cycle_debug")
            else {}
        )
        return {
            "status": "success",
            "rebalance_decision": rebalancer.get_last_decision_summary(),
            "last_cycle": cycle_debug,
        }
    except Exception as e:
        return {"error": str(e)}


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

    # AUDIT FIX: Guard against database exceptions in status RPC
    try:
        channel_states = database.get_all_channel_states()
        fee_history = database.get_recent_fee_changes(limit=10)
        rebalance_history = database.get_recent_rebalances(limit=10)
    except Exception as e:
        return {"error": f"Database query failed: {e}"}

    return {
        "status": "running",
        "version": PLUGIN_VERSION,
        "operator_controls": {
            "public_keys": config.public_runtime_keys() if config else [],
            "values": config.public_runtime_dict() if config else {},
        },
        "fee_decision": (
            fee_controller.get_last_decision_summary()
            if fee_controller and hasattr(fee_controller, "get_last_decision_summary")
            else {
                "action": "hold",
                "reason": "unavailable",
                "dominant_input": "fee_controller",
                "safety_block": False,
            }
        ),
        "rebalance_decision": (
            rebalancer.get_last_decision_summary()
            if rebalancer and hasattr(rebalancer, "get_last_decision_summary")
            else {
                "action": "hold",
                "reason": "unavailable",
                "dominant_input": "rebalancer",
                "safety_block": False,
                "budget_blocked": False,
            }
        ),
        "receivable": _node_receivable_status(),
        "channel_states": channel_states,
        "recent_fee_changes": fee_history,
        "recent_rebalances": rebalance_history,
    }


def _hive_hints_status_for_debug(plugin: Plugin, max_segment_scores: int = 20) -> Dict[str, Any]:
    """Return read-only hive hint freshness diagnostics for operator surfaces."""
    hive_refresh_error = ""
    hive_refresh_attempted = False
    hive_refresh_diagnostics = None

    if hive_hints:
        debug_refresher = getattr(hive_hints, "refresh_status_for_debug", None)
        if callable(debug_refresher):
            try:
                candidate_diagnostics = debug_refresher()
                if isinstance(candidate_diagnostics, dict):
                    hive_refresh_diagnostics = candidate_diagnostics
            except Exception as e:
                hive_refresh_error = str(e)
                plugin.log(f"Hive hints debug refresh failed: {e}", level='debug')

        if hive_refresh_diagnostics is None:
            poller = getattr(hive_hints, "poll", None)
            if callable(poller):
                hive_refresh_attempted = True
                try:
                    poller()
                except Exception as e:
                    hive_refresh_error = str(e)
                    plugin.log(f"Hive hints debug refresh failed: {e}", level='debug')

        try:
            hive_status = hive_hints.get_status(live_refresh=False)
        except Exception as e:
            hive_status = {
                "snapshot_fresh": False,
                "snapshot_usable": False,
                "hints_count": 0,
                "status_error": str(e),
            }
    else:
        hive_status = {"snapshot_fresh": False, "snapshot_usable": False, "hints_count": 0}

    if not isinstance(hive_status, dict):
        hive_status = {"snapshot_fresh": False, "snapshot_usable": False, "hints_count": 0}

    if isinstance(hive_refresh_diagnostics, dict):
        hive_refresh_attempted = bool(hive_refresh_diagnostics.get("refresh_attempted", False))
        hive_status["cache"] = dict(hive_refresh_diagnostics.get("cache", {}) or {})
        hive_status["cache_after_refresh"] = dict(hive_refresh_diagnostics.get("cache_after_refresh", {}) or {})
        hive_status["live_datastore"] = dict(hive_refresh_diagnostics.get("live_datastore", {}) or {})
        hive_status["live_hive_export"] = dict(hive_refresh_diagnostics.get("live_hive_export", {}) or {})
        hive_status["fallback"] = dict(hive_refresh_diagnostics.get("fallback", {}) or {})
        hive_status["status_refresh_needed"] = bool(hive_refresh_diagnostics.get("refresh_needed", False))
        hive_status["status_refresh_result"] = str(hive_refresh_diagnostics.get("refresh_result") or "")

    hive_status["status_refresh_attempted"] = hive_refresh_attempted
    hive_status["status_refresh_ok"] = (not hive_refresh_error) and (
        bool(hive_refresh_diagnostics is not None) or hive_refresh_attempted
    )
    if hive_refresh_error:
        hive_status["status_refresh_error"] = hive_refresh_error
    hive_status["diagnostics_version"] = HIVE_HINTS_DIAGNOSTICS_VERSION

    segment_scores: List[Dict[str, Any]] = []
    if hive_hints is not None:
        getter = getattr(hive_hints, "get_segment_scores", None)
        if callable(getter):
            try:
                raw_scores = getter() or []
            except Exception:
                raw_scores = []
            for raw in raw_scores:
                if isinstance(raw, dict):
                    segment_scores.append(dict(raw))
    segment_scores.sort(
        key=lambda entry: (
            -abs(float(entry.get("net_utility", 0.0) or 0.0))
            * float(entry.get("confidence", 0.0) or 0.0),
            -float(entry.get("confidence", 0.0) or 0.0),
            str(entry.get("short_channel_id") or ""),
            int(entry.get("direction", 0) or 0),
        )
    )
    # P1-012: coerce operator-supplied max_segment_scores; a non-int must not
    # raise ValueError out of a diagnostic handler.
    try:
        max_segment_scores = max(0, int(max_segment_scores or 0))
    except (ValueError, TypeError):
        max_segment_scores = 0
    if max_segment_scores:
        segment_scores = segment_scores[:max_segment_scores]
    hive_status["segment_scores_count"] = len(segment_scores)
    hive_status["segment_scores"] = segment_scores
    return hive_status


@plugin.method("revenue-hive-hints-status")
def revenue_hive_hints_status(plugin: Plugin, max_segment_scores: int = 20) -> Dict[str, Any]:
    """Diagnostic: cl-mycelium hint freshness, fallback, and signal coverage."""
    return _hive_hints_status_for_debug(plugin, max_segment_scores=max_segment_scores)


@plugin.method("revenue-rebalance-debug")
def revenue_rebalance_debug(
    plugin: Plugin,
    channel_id: str = None,
    peer_id: str = None,
    summary_only: bool = False,
    include_hot_markers: bool = True,
    max_candidates: int = 0,
) -> Dict[str, Any]:
    """
    Diagnostic command to understand why rebalancing may not be happening.

    Shows:
    - Capital control status (budget/reserve)
    - Depleted channels (potential destinations)
    - Source channels (potential sources)
    - Why candidates are rejected

    Optional filters for lighter responses:
    - channel_id=<scid>
    - peer_id=<pubkey>
    - summary_only=true
    - include_hot_markers=false
    - max_candidates=<n>

    Usage: lightning-cli revenue-rebalance-debug
    """
    if rebalancer is None:
        return {"error": "Rebalancer not initialized"}

    def _filtered_segment_scores() -> List[Dict[str, Any]]:
        if hive_hints is None:
            return []
        getter = getattr(hive_hints, "get_segment_scores", None)
        if not callable(getter):
            return []
        try:
            raw_scores = getter() or []
        except Exception:
            return []

        scores: List[Dict[str, Any]] = []
        for raw in raw_scores:
            if not isinstance(raw, dict):
                continue
            short_channel_id = str(raw.get("short_channel_id") or "").strip()
            if filter_channel_id and short_channel_id != filter_channel_id:
                continue
            scores.append(dict(raw))
        scores.sort(
            key=lambda entry: (
                -abs(float(entry.get("net_utility", 0.0) or 0.0))
                * float(entry.get("confidence", 0.0) or 0.0),
                -float(entry.get("confidence", 0.0) or 0.0),
                str(entry.get("short_channel_id") or ""),
                int(entry.get("direction", 0) or 0),
            )
        )
        if max_candidates > 0:
            return scores[:max_candidates]
        return scores[:20]

    filter_channel_id = str(channel_id or "").strip()
    filter_peer_id = str(peer_id or "").strip().lower()
    summary_only = bool(summary_only)
    include_hot_markers = bool(include_hot_markers) and not summary_only
    # P1-012 class: coerce operator-supplied max_candidates; a non-int must not
    # raise ValueError/TypeError out of a diagnostic handler.
    try:
        max_candidates = max(0, int(max_candidates or 0))
    except (ValueError, TypeError):
        max_candidates = 0

    result = {
        "executor_available": True,
        "dry_run": config.dry_run if config else False,
        "filters": {
            "channel_id": filter_channel_id or None,
            "peer_id": filter_peer_id or None,
            "summary_only": summary_only,
            "include_hot_markers": include_hot_markers,
            "max_candidates": max_candidates or None,
        },
        "capital_controls": {},
        "thresholds": {},
        "channels": {
            "depleted": [],
            "source": [],
            "active_jobs": [],
            "counts": {
                "considered": 0,
                "depleted": 0,
                "source": 0,
                "active_jobs": 0,
            },
            "truncated": {
                "depleted": 0,
                "source": 0,
                "active_jobs": 0,
            }
        },
        "rejection_reasons": [],
        "last_decision": (
            rebalancer.get_last_decision_summary()
            if hasattr(rebalancer, "get_last_decision_summary")
            else {}
        ),
        "last_cycle": {},
    }

    cfg = config.snapshot()
    result["thresholds"] = {
        "low_liquidity_threshold": cfg.low_liquidity_threshold,
        "high_liquidity_threshold": cfg.high_liquidity_threshold,
        "rebalance_min_profit_sats": cfg.rebalance_min_profit
    }

    # Check capital controls
    try:
        listfunds = data_service.get_funds() if data_service else safe_plugin.rpc.listfunds()
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
            "budget_floor_sats": total_budget.get("daily_budget_sats", cfg.daily_budget_sats),
            "daily_spent_sats": daily_spent,
            "daily_reserved_sats": daily_reserved,
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

    # Get channel analysis (request-local caching + optional filtering for performance)
    try:
        channels = rebalancer._get_channels_with_balances()
        active_channels = set(rebalancer.job_manager.active_channels)

        state_lookup = {}
        if database is not None:
            try:
                state_lookup = {
                    (s.get("channel_id") or ""): s
                    for s in (database.get_all_channel_states() or [])
                    if (s.get("channel_id") or "")
                }
            except Exception as e:
                plugin.log(f"Channel state lookup failed: {e}", level='debug')
                state_lookup = {}

        # AUDIT FIX Issue-11: Renamed to avoid shadowing the global profitability_analyzer
        prof_analyzer = getattr(rebalancer, "_profitability_analyzer", None) if include_hot_markers else None
        compute_hot = getattr(rebalancer, "_compute_hot_channel_protection", None) if include_hot_markers else None
        hot_profile_cache: Dict[str, Dict[str, Any]] = {}
        prof_cache: Dict[str, Any] = {}

        hot_override_depletion_thresholds: Dict[str, float] = {}
        if database is not None:
            try:
                for r in (database.list_hot_channel_protection_override_peers() or []):
                    pid = str(r.get("peer_id") or "").strip()
                    pct = r.get("min_depletion_trigger_pct")
                    try:
                        pct_f = float(pct) if pct is not None else None
                    except Exception as e:
                        plugin.log(f"Hot channel override pct parse failed for {pid}: {e}", level='debug')
                        pct_f = None
                    if pid and pct_f is not None and 0.0 < pct_f <= 100.0:
                        hot_override_depletion_thresholds[pid] = pct_f / 100.0
            except Exception as e:
                plugin.log(f"Hot channel override list failed: {e}", level='debug')
                hot_override_depletion_thresholds = {}

        effective_low_thresholds_seen_pct: set[float] = set()

        def _append_channel(bucket: str, item: Dict[str, Any]) -> None:
            counts = result["channels"]["counts"]
            trunc = result["channels"]["truncated"]
            counts[bucket] = int(counts.get(bucket, 0) or 0) + 1
            if summary_only:
                return
            if max_candidates > 0 and len(result["channels"][bucket]) >= max_candidates:
                trunc[bucket] = int(trunc.get(bucket, 0) or 0) + 1
                return
            result["channels"][bucket].append(item)

        for cid, info in channels.items():
            capacity = info.get("capacity", 0)
            if capacity == 0:
                continue

            peer_id_full = str(info.get("peer_id", "") or "")
            if filter_channel_id and cid != filter_channel_id:
                continue
            if filter_peer_id and peer_id_full.lower() != filter_peer_id:
                continue

            spendable = info.get("spendable_sats", 0)
            ratio = spendable / capacity if capacity else 0.0
            fee_ppm = info.get("fee_ppm", 0)
            peer_id_short = peer_id_full[:16]

            state = state_lookup.get(cid) or (database.get_channel_state(cid) if database else {}) or {}
            flow_state = state.get("state", "unknown") if state else "unknown"

            result["channels"]["counts"]["considered"] = int(result["channels"]["counts"].get("considered", 0) or 0) + 1

            effective_low_threshold = float(hot_override_depletion_thresholds.get(peer_id_full, cfg.low_liquidity_threshold))
            effective_low_thresholds_seen_pct.add(round(effective_low_threshold * 100, 1))

            channel_info = {
                "scid": cid[:20],
                "peer": peer_id_short,
                "peer_id": peer_id_full if (filter_peer_id or filter_channel_id) else None,
                "local_pct": round(ratio * 100, 1),
                "fee_ppm": fee_ppm,
                "flow_state": flow_state,
                "effective_low_liquidity_threshold_pct": round(effective_low_threshold * 100, 1),
            }
            if channel_info.get("peer_id") is None:
                channel_info.pop("peer_id", None)

            if include_hot_markers:
                hot_profile = hot_profile_cache.get(cid)
                if hot_profile is None:
                    try:
                        velocity = 0.0
                        if capacity > 0 and state:
                            sats_in = float(state.get("sats_in", 0) or 0)
                            sats_out = float(state.get("sats_out", 0) or 0)
                            velocity = (sats_in + sats_out) / max(float(capacity), 1.0) / max(float(getattr(cfg, "flow_window_days", 7) or 7), 1.0)

                        prof = prof_cache.get(cid, None)
                        if cid not in prof_cache and prof_analyzer is not None:
                            try:
                                prof = prof_analyzer.analyze_channel(cid)
                            except Exception as e:
                                plugin.log(f"Profitability analysis failed for {cid[:12]}...: {e}", level='debug')
                                prof = None
                            prof_cache[cid] = prof

                        if callable(compute_hot):
                            hot_profile = compute_hot(
                                dest_channel=cid,
                                dest_peer_id=peer_id_full,
                                dest_flow_state=flow_state,
                                dest_ratio=ratio,
                                velocity=velocity,
                                prof=prof,
                                cfg=cfg,
                            ) or {}
                        else:
                            hot_profile = {}
                    except Exception as e:
                        hot_profile = {"enabled": False, "eligible": False, "reason": f"debug_hot_profile_error:{e}"}
                    hot_profile_cache[cid] = hot_profile

                channel_info.update({
                    "hot_channel_protection": bool(hot_profile.get("eligible", False)),
                    "hot_channel_protection_enabled": bool(hot_profile.get("enabled", False)),
                    "hot_channel_protection_reason": hot_profile.get("reason"),
                    "hot_channel_protection_score": round(float(hot_profile.get("score", 0.0) or 0.0), 4),
                    "hot_channel_protection_peer_override": bool(hot_profile.get("peer_forced", False)),
                    "hot_channel_protection_peer_depletion_trigger_pct": hot_profile.get("peer_override_min_depletion_trigger_pct"),
                    "profit_budget_override_sats": int(hot_profile.get("channel_profit_budget_sats", 0) or 0),
                    "hot_recommended_cooldown_hours": hot_profile.get("recommended_cooldown_hours"),
                    "hot_chunk_multiplier": hot_profile.get("chunk_multiplier"),
                })

            if cid in active_channels:
                _append_channel("active_jobs", channel_info)
            elif ratio < effective_low_threshold:
                channel_info["reason"] = "low local balance"
                if flow_state == "sink":
                    channel_info["skip_reason"] = "SINK - filling naturally"
                _append_channel("depleted", channel_info)
            elif ratio > cfg.high_liquidity_threshold:
                channel_info["reason"] = "high local balance"
                _append_channel("source", channel_info)

        if result["channels"]["counts"]["depleted"] == 0:
            if (filter_channel_id or filter_peer_id) and effective_low_thresholds_seen_pct:
                thresholds_sorted = sorted(effective_low_thresholds_seen_pct)
                if len(thresholds_sorted) == 1:
                    threshold_txt = f"{thresholds_sorted[0]}%"
                else:
                    threshold_txt = f"{thresholds_sorted[0]}%-{thresholds_sorted[-1]}%"
                result["rejection_reasons"].append(
                    f"No depleted channels (none below effective filtered threshold {threshold_txt} local balance)"
                )
            else:
                result["rejection_reasons"].append(
                    f"No depleted channels (none below {cfg.low_liquidity_threshold*100}% local balance)"
                )
        if result["channels"]["counts"]["source"] == 0:
            result["rejection_reasons"].append(
                f"No source channels (none above {cfg.high_liquidity_threshold*100}% local balance)"
            )

    except Exception as e:
        result["channels"]["error"] = str(e)

    hive_refresh_error = ""
    hive_refresh_attempted = False
    hive_refresh_diagnostics = None
    if hive_hints:
        debug_refresher = getattr(hive_hints, "refresh_status_for_debug", None)
        if callable(debug_refresher):
            try:
                candidate_diagnostics = debug_refresher()
                if isinstance(candidate_diagnostics, dict):
                    hive_refresh_diagnostics = candidate_diagnostics
            except Exception as e:
                hive_refresh_error = str(e)
                plugin.log(f"Hive hints debug refresh failed: {e}", level='debug')
        if hive_refresh_diagnostics is None:
            poller = getattr(hive_hints, "poll", None)
            if callable(poller):
                hive_refresh_attempted = True
                try:
                    poller()
                except Exception as e:
                    hive_refresh_error = str(e)
                    plugin.log(f"Hive hints debug refresh failed: {e}", level='debug')
        try:
            hive_status = hive_hints.get_status(live_refresh=False)
        except Exception as e:
            hive_status = {
                "snapshot_fresh": False,
                "snapshot_usable": False,
                "hints_count": 0,
                "status_error": str(e),
            }
    else:
        hive_status = {"snapshot_fresh": False, "snapshot_usable": False, "hints_count": 0}
    if not isinstance(hive_status, dict):
        hive_status = {"snapshot_fresh": False, "hints_count": 0}
    if isinstance(hive_refresh_diagnostics, dict):
        hive_refresh_attempted = bool(hive_refresh_diagnostics.get("refresh_attempted", False))
        hive_status["cache"] = dict(hive_refresh_diagnostics.get("cache", {}) or {})
        hive_status["cache_after_refresh"] = dict(hive_refresh_diagnostics.get("cache_after_refresh", {}) or {})
        hive_status["live_datastore"] = dict(hive_refresh_diagnostics.get("live_datastore", {}) or {})
        hive_status["live_hive_export"] = dict(hive_refresh_diagnostics.get("live_hive_export", {}) or {})
        hive_status["fallback"] = dict(hive_refresh_diagnostics.get("fallback", {}) or {})
        hive_status["status_refresh_needed"] = bool(hive_refresh_diagnostics.get("refresh_needed", False))
        hive_status["status_refresh_result"] = str(hive_refresh_diagnostics.get("refresh_result") or "")
    hive_status["status_refresh_attempted"] = hive_refresh_attempted
    hive_status["status_refresh_ok"] = (not hive_refresh_error) and (
        bool(hive_refresh_diagnostics is not None) or hive_refresh_attempted
    )
    if hive_refresh_error:
        hive_status["status_refresh_error"] = hive_refresh_error
    hive_status["diagnostics_version"] = HIVE_HINTS_DIAGNOSTICS_VERSION
    segment_scores = _filtered_segment_scores()
    hive_status["segment_scores_count"] = len(segment_scores)
    if not summary_only:
        hive_status["segment_scores"] = segment_scores
    result["hive_hints"] = hive_status

    engine = getattr(rebalancer, "rebalance_engine_v2", None)
    if engine is not None and hasattr(engine, "get_last_cycle_debug"):
        try:
            cycle_limit = max_candidates if max_candidates > 0 else 10
            result["last_cycle"] = engine.get_last_cycle_debug(
                max_candidates=cycle_limit,
            )
        except Exception as e:
            result["last_cycle"] = {"error": str(e)}

    try:
        drain = None
        demand = engine.get_drain_demand() if engine is not None and hasattr(engine, "get_drain_demand") else None
        if demand is not None:
            drain = {
                "total_excess_sats": demand.total_excess_sats,
                "over_local_count": demand.over_local_count,
                "paired_count": demand.paired_count,
                "top_entries": [
                    {"channel_id": entry.channel_id, "excess_sats": entry.excess_sats,
                     "drain_score": entry.drain_score, "value_class": entry.value_class}
                    for entry in demand.entries[:10]
                ],
            }
        result["drain_demand"] = drain
    except Exception as e:
        result["drain_demand"] = {"error": str(e)}

    return result


def _supported_fee_ceiling_from_state(ts_state: Dict[str, Any]) -> Optional[float]:
    """Compute the supported-fee ceiling from a persisted thompson payload.

    Diagnostic-only rehydration for revenue-fee-debug: the 2026-06-12 LOOP
    incident took an hour of forensics because the debug surface didn't show
    what the earning evidence supported vs what the posterior believed.
    """
    try:
        from modules.fee_controller import GaussianThompsonState
        st = GaussianThompsonState.from_dict(ts_state or {})
        cap = st.supported_fee_ceiling()
        return round(cap, 1) if cap is not None else None
    except Exception:
        return None


def _zero_flow_guard_state(
    last_revenue_rate: Any,
    forwards_since_update: Any,
    zero_revenue_streak: Any,
) -> Optional[str]:
    """Return the active DTS+PID zero-flow guard state for diagnostics."""
    try:
        rate = float(last_revenue_rate)
        forwards = int(forwards_since_update)
        streak = int(zero_revenue_streak)
    except (TypeError, ValueError, OverflowError):
        return None
    if rate != 0.0 or forwards != 0:
        return None
    if streak >= FeeController.ZERO_FLOW_DOWNSHIFT_STREAK:
        return "zero_flow_downshift"
    if streak >= FeeController.ZERO_FLOW_GUARD_STREAK:
        return "zero_flow_ratchet_guard"
    return None


@plugin.method("revenue-fee-debug")
def revenue_fee_debug(plugin: Plugin) -> Dict[str, Any]:
    """
    Diagnostic command to understand why fee adjustments may not be happening.

    Shows:
    - Cycle state for each channel (sleeping, last_update, forward count)
    - Why each channel was skipped in the last cycle
    - Dynamic window status
    - Hysteresis/sleep status

    Usage: lightning-cli revenue-fee-debug
    """
    if database is None or fee_controller is None:
        return {"error": "Plugin not fully initialized"}

    cfg_snap = config.snapshot() if hasattr(config, "snapshot") else config
    profile = fee_controller.get_fee_profile_settings(cfg_snap)
    min_obs_hours = profile["min_observation_hours"]
    min_forwards = profile["min_forwards_for_signal"]
    market_boundary_configured = bool(getattr(cfg_snap, "fee_market_boundary_enabled", False))

    hive_refresh_debug = {
        "attempted": False,
        "ok": False,
    }
    if hive_hints is not None or hive_router is not None:
        hive_refresh_debug["attempted"] = True
        try:
            refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router, log=plugin.log)
            hive_refresh_debug["ok"] = True
        except Exception as e:
            hive_refresh_debug["error"] = str(e)

    now = int(time.time())
    result = {
        "timestamp": now,
        "hive_refresh": hive_refresh_debug,
        "config": {
            "fee_interval_seconds": config.fee_interval if config else 1800,
            "fee_profile": profile["name"],
            "market_boundary_enabled": False,
            "market_boundary_configured": market_boundary_configured,
            "market_boundary_effective": False,
            "market_boundary_deprecated": True,
            "market_boundary_min_competitors": getattr(cfg_snap, "fee_market_boundary_min_competitors", 3),
            "market_boundary_margin_ppm": getattr(cfg_snap, "fee_market_boundary_margin_ppm", 5),
            "market_boundary_margin_ratio": getattr(cfg_snap, "fee_market_boundary_margin_ratio", 0.05),
            "market_boundary_max_downshift_ratio": getattr(cfg_snap, "fee_market_boundary_max_downshift_ratio", 0.35),
            "market_boundary_cache_seconds": getattr(cfg_snap, "fee_market_boundary_cache_seconds", 60),
            "min_observation_hours": min_obs_hours,
            "min_forwards_for_signal": min_forwards,
            "profile_settings": profile,
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
        v2_state = {}
        try:
            v2_state = json.loads(fs.get("v2_state_json") or "{}")
        except Exception:
            v2_state = {}
        # Nested-first: the fee controller no longer writes the flat
        # v2_state["thompson_state"] mirror; new rows carry it only under
        # fee_state. Old rows keep the flat copy, so fall back for them.
        _fee_state = v2_state.get("fee_state")
        _fee_state = _fee_state if isinstance(_fee_state, dict) else {}
        ts_state = _fee_state.get("thompson_state") or v2_state.get("thompson_state") or {}

        hours_since_update = (now - last_update) / 3600.0 if last_update > 0 else 0.0

        # Determine skip reason
        # Dynamic windows: channel is ready if EITHER time >= min_obs_hours OR forwards >= min_forwards
        skip_reason = None
        status = "ready"

        time_ok = hours_since_update >= min_obs_hours
        forwards_ok = forward_count >= min_forwards

        if is_sleeping:
            # E1 FIX: Clamp to 0 to avoid displaying negative minutes when
            # sleep_until has passed but is_sleeping flag hasn't been cleared yet.
            mins_until_wake = max(0, (sleep_until - now) // 60)
            skip_reason = f"SLEEPING (wake in {mins_until_wake} min)"
            status = "sleeping"
            result["summary"]["sleeping"] += 1
        elif time_ok or forwards_ok:
            status = "ready"
            result["summary"]["ready"] += 1
        else:
            skip_reason = f"WAITING ({forward_count}/{min_forwards} fwds, {hours_since_update:.1f}/{min_obs_hours}h)"
            status = "waiting"
            result["summary"]["waiting_time"] += 1

        chan_state = state_lookup.get(channel_id, {})
        peer_id = str(chan_state.get("peer_id") or "")
        hive_fee_debug = {}
        if peer_id and hasattr(fee_controller, "get_hive_fee_hint_debug"):
            try:
                hive_fee_debug = fee_controller.get_hive_fee_hint_debug(peer_id)
            except Exception as e:
                hive_fee_debug = {"error": str(e)}
        zero_revenue_streak = ts_state.get("zero_revenue_streak", 0)
        zero_flow_guard = _zero_flow_guard_state(
            last_revenue_rate,
            forward_count,
            zero_revenue_streak,
        )
        result["channels"].append({
            "channel_id": channel_id[:12] + "..." if len(channel_id) > 12 else channel_id,
            "peer_id": peer_id,
            "status": status,
            "skip_reason": skip_reason,
            "is_sleeping": bool(is_sleeping),
            "hours_since_update": round(hours_since_update, 2),
            "forwards_since_update": forward_count,
            "last_broadcast_fee_ppm": last_broadcast_fee,
            "last_revenue_rate": round(last_revenue_rate, 2),
            "zero_flow_guard": zero_flow_guard,
            "flow_state": chan_state.get("state", "unknown"),
            "fee_profile": v2_state.get("last_fee_profile", profile["name"]),
            "dts": {
                "posterior_mean": ts_state.get("posterior_mean"),
                "posterior_std": ts_state.get("posterior_std"),
                "observations": len(ts_state.get("observations") or []),
                "last_sampled_fee": ts_state.get("last_sampled_fee"),
                "zero_revenue_streak": zero_revenue_streak,
                "positive_rate_ref": ts_state.get("positive_rate_ref", 0.0),
                "supported_fee_ceiling": _supported_fee_ceiling_from_state(ts_state),
            },
            "context": {
                "key": v2_state.get("last_context_key", ""),
                "time_bucket": v2_state.get("last_time_bucket", "normal"),
                "corridor_role": v2_state.get("last_corridor_role", "P"),
                "contextual_sample_used": bool(v2_state.get("last_contextual_sample_used", False)),
                "contexts_tracked": len(ts_state.get("contextual_posteriors") or {}),
            },
            "hive": hive_fee_debug,
        })
        result["summary"]["total"] += 1

    return result


@plugin.method("revenue-fee-cycle")
def revenue_fee_cycle(plugin: Plugin) -> Dict[str, Any]:
    """Run one fee adjustment cycle immediately."""
    _refresh_fee_cycle_hive_inputs()
    adjustments = run_fee_adjustment() or []
    return {
        "ok": True,
        "adjusted_channels": len(adjustments),
        "fee_debug": revenue_fee_debug(plugin),
    }


@plugin.method("revenue-analyze")
def revenue_analyze(plugin: Plugin, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run flow analysis on demand (optionally for a specific channel).

    Usage: lightning-cli revenue-analyze [channel_id]
    """
    if flow_analyzer is None:
        return {"error": "Plugin not fully initialized"}

    # L-22: Validate SCID format if provided
    # P1-012: guard against non-str channel_id (re.match on a non-str raises
    # TypeError and leaks a traceback instead of a clean error dict).
    if channel_id is not None and not isinstance(channel_id, str):
        return {"error": "channel_id must be a string SCID (e.g., 123x456x789)."}
    if channel_id and not re.match(r'^\d+[x:]\d+[x:]\d+$', channel_id):
        return {"error": f"Invalid channel format: {channel_id}. Use SCID format (e.g., 123x456x789)."}

    if channel_id:
        channel_id = normalize_scid(channel_id)
        result = flow_analyzer.analyze_channel(channel_id)
        return {"channel": channel_id, "analysis": result.to_dict() if result else None}
    else:
        # AUDIT FIX Issue-4: Catch exceptions from run_flow_analysis (it re-raises)
        try:
            run_flow_analysis()
        except Exception as e:
            return {"error": f"Flow analysis failed: {e}"}
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

    Identifies "Winner" channels for capital injection
    and "Loser" channels for capital extraction or closure.
    """
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}

    try:
        return capacity_planner.generate_report()
    except Exception as e:
        plugin.log(f"Error generating capacity report: {e}", level='error')
        return {"error": f"Report generation failed: {e}"}


@plugin.method("revenue-planner-status")
def revenue_planner_status(plugin: Plugin) -> Dict[str, Any]:
    """Get capacity planner status — pending actions, last cycle result, config."""
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}
    return capacity_planner.get_status()


@plugin.method("revenue-planner-candidate-sources")
def planner_candidate_sources(plugin: Plugin):
    """Show strategy distribution of the current candidate pool."""
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}
    return capacity_planner.get_candidate_sources()


@plugin.method("revenue-planner-candidates")
def revenue_planner_candidates(plugin: Plugin, limit: int = 20) -> Dict[str, Any]:
    """List scored peer candidates for channel opens."""
    if capacity_planner is None:
        return {"error": "Capacity planner not initialized"}
    try:
        limit = _clamp_query_limit(limit)
    except _ParamError as e:
        return {"error": str(e)}
    candidates = database.get_planner_candidates(limit=limit)
    metabolic_planner_influence = {}
    try:
        getter = getattr(capacity_planner, "get_metabolic_planner_influence_debug", None)
        if callable(getter):
            metabolic_planner_influence = getter()
    except Exception:
        metabolic_planner_influence = {"seen": False, "usable": False, "reason": "debug_error"}
    return {
        "candidates": candidates,
        "count": len(candidates),
        "metabolic_planner_influence": metabolic_planner_influence,
    }


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
    try:
        limit = _clamp_query_limit(limit)
    except _ParamError as e:
        return {"error": str(e)}
    actions = database.get_planner_actions(limit=limit)
    return {"actions": actions, "count": len(actions)}


@plugin.method("revenue-set-fee")
def revenue_set_fee(plugin: Plugin, channel_id: str, fee_ppm: int, force: bool = False) -> Dict[str, Any]:
    """
    Manually set fee for a channel.

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
    except (ValueError, TypeError):
        return {"status": "error", "error": "fee_ppm must be an integer"}

    # SCID, full channel_id, or peer ID format check
    # P1-012: guard against non-str channel_id before regex matching.
    if not isinstance(channel_id, str):
        return {"status": "error", "error": "channel_id must be a string"}
    if not (
        re.match(r'^\d+[x:]\d+[x:]\d+$', channel_id)
        or re.match(r'^[0-9a-fA-F]{64}$', channel_id)
        or re.match(r'^[0-9a-fA-F]{66}$', channel_id)
    ):
        return {"status": "error", "error": "Invalid channel_id or node_id format"}

    # 2. Force Gates
    if not force:
        if fee_ppm < config.min_fee_ppm or fee_ppm > config.max_fee_ppm:
            return {
                "status": "error",
                "error": f"Fee {fee_ppm} is outside configured range [{config.min_fee_ppm}, {config.max_fee_ppm}]. Use force=true to override."
            }

    # DD2 / P1-001, P1-004: the [min_fee_ppm, max_fee_ppm] rail and the absolute
    # ceiling bind even under force. force may bypass soft gates (deadband,
    # cooldown, sleep) but NEVER the hard fee rails — an above-max force set is
    # clamped to max, a below-min force set is raised to min, and fee_ppm can
    # never be set to 0 below min_fee_ppm.
    hard_min = int(config.min_fee_ppm)
    hard_max = min(int(config.max_fee_ppm), int(FeeController.ABS_MAX_FEE_PPM))
    requested_fee_ppm = fee_ppm
    fee_ppm = max(hard_min, min(hard_max, fee_ppm))
    fee_rail_clamped = fee_ppm != requested_fee_ppm
    if fee_rail_clamped:
        plugin.log(
            f"revenue-set-fee: clamped requested {requested_fee_ppm} to hard rail "
            f"[{hard_min}, {hard_max}] ppm for {channel_id[:16]}",
            level='warn',
        )

    try:
        result = fee_controller.set_channel_fee(channel_id, fee_ppm, manual=True, enforce_limits=(not force), force=force)
        if not result.get("success"):
            return {
                "status": "error",
                "error": result.get("message") or "Fee update failed",
                **result,
            }
        applied_fee = result.get("fee_ppm", fee_ppm)
        resolved_channel = result.get("channel_id", channel_id)
        out = {"status": "success", "channel": resolved_channel, "new_fee_ppm": applied_fee, **result}
        if fee_rail_clamped:
            out["requested_fee_ppm"] = requested_fee_ppm
            out["clamped_to_rail"] = [hard_min, hard_max]
        return out
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

    # DD2 / P1-001: rate-limit BOTH force and non-force manual rebalances (the
    # prior force-only asymmetry let force=false spam the money path un-gated).
    allowed, msg = force_rate_limiter.check_rate_limit("revenue-rebalance")
    if not allowed:
        return {"status": "error", "error": msg}

    # Native route execution is handled by RebalanceEngineV2.

    # L-21: Validate SCID format
    # P1-012: guard against non-str channel args before regex matching.
    for cid in (from_channel, to_channel):
        if not isinstance(cid, str) or not re.match(r'^\d+[x:]\d+[x:]\d+$', cid):
            return {"status": "error", "error": f"Invalid channel format for {cid!r}. Use SCID format (e.g., 123x456x789)."}

    # 1. Validation
    try:
        amount_sats = int(amount_sats)
        if amount_sats < 1:
            return {"status": "error", "error": "amount_sats must be at least 1"}
    except (ValueError, TypeError):
        return {"status": "error", "error": "amount_sats must be an integer"}

    # DD2 / P1-004: a hard maximum rebalance amount binds regardless of force.
    # force may bypass soft budget/cooldown gates but never the absolute amount
    # rail — an over-cap amount is rejected under both force values.
    try:
        hard_max_amount = int(getattr(config, "rebalance_max_amount", 0) or 0) if config is not None else 0
    except (ValueError, TypeError):
        hard_max_amount = 0
    if hard_max_amount > 0 and amount_sats > hard_max_amount:
        return {
            "status": "error",
            "error": f"amount_sats {amount_sats} exceeds hard rebalance cap "
                     f"{hard_max_amount} (rebalance_max_amount); rejected even under force.",
            "requested_sats": amount_sats,
            "max_amount_sats": hard_max_amount,
        }


    if max_fee_sats is not None:
        try:
            max_fee_sats = int(max_fee_sats)
            if max_fee_sats < 0:
                return {"status": "error", "error": "max_fee_sats must be non-negative"}
        except (ValueError, TypeError):
            return {"status": "error", "error": "max_fee_sats must be an integer or null"}

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


@plugin.method("revenue-profitability")
def revenue_profitability(plugin: Plugin, channel_id: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
    """
    Get channel profitability analysis.

    Shows each channel's:
    - Total costs (opening + rebalancing)
    - Total revenue (routing fees)
    - Net profit/loss
    - ROI percentage
    - Profitability classification (profitable, break_even, underwater, zombie)

    Usage: lightning-cli revenue-profitability [channel_id] [refresh]

    refresh=true forces a full re-analysis on the dispatch thread; the default
    serves the analyzer's cached results (refreshed periodically in background).
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
            # Analyze all channels. Default uses cached results; refresh=true
            # forces a full re-analysis (expensive — blocks the dispatch thread).
            all_results = profitability_analyzer.analyze_all_channels(force=bool(refresh))

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
            total_revenue_msat = 0
            total_contribution_msat = 0
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
                    "total_forward_count": result.revenue.total_forward_count,
                    "fees_earned_sats": result.revenue.fees_earned_sats,
                    "volume_routed_sats": result.revenue.volume_routed_sats,
                    "sourced_fee_contribution_sats": result.revenue.sourced_fee_contribution_sats,
                    "sourced_volume_sats": result.revenue.sourced_volume_sats,
                    "total_contribution_sats": result.revenue.total_contribution_sats,
                }
                summary[result.classification.value].append(channel_summary)
                flow_profiles[flow_profile].append(ch_id)
                total_revenue_msat += result.revenue.fees_earned_msat
                total_contribution_msat += result.revenue.total_contribution_msat
                total_costs += result.costs.total_cost_sats

            total_revenue = -(-total_revenue_msat // 1000)
            total_contribution = -(-total_contribution_msat // 1000)
            total_profit = total_revenue - total_costs

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


def _policy_write_override(kwargs: Dict[str, Any]) -> bool:
    """True when an internal/admin caller explicitly unlocks policy writes.

    The deprecated ignore/unignore aliases must honor the same gate as
    'revenue-policy set/delete' — otherwise the operator-write lockdown is
    trivially circumvented through the aliases.
    """
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    return _truthy(kwargs.get("internal")) or _truthy(kwargs.get("admin"))


@plugin.method("revenue-ignore")
def revenue_ignore(plugin: Plugin, peer_id: str, reason: str = "manual", **kwargs) -> Dict[str, Any]:
    """
    DEPRECATED: Use 'revenue-policy set <peer_id> strategy=passive rebalance=disabled' instead.

    Stop cl-revenue-ops from managing this peer (fees or rebalancing).

    Usage: lightning-cli revenue-ignore peer_id [reason]
    """
    if policy_manager is None:
        return {"error": "Plugin not initialized"}

    if not _policy_write_override(kwargs):
        return {
            "error": (
                "revenue-ignore is deprecated for normal operator use. "
                "Use revenue-policy list/get/find/changes for diagnostics."
            )
        }
    if not re.match(r'^[0-9a-fA-F]{66}$', str(peer_id or "")):
        return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}

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
def revenue_unignore(plugin: Plugin, peer_id: str, **kwargs) -> Dict[str, Any]:
    """
    DEPRECATED: Use 'revenue-policy delete <peer_id>' instead.

    Resume cl-revenue-ops management for this peer.

    Usage: lightning-cli revenue-unignore peer_id
    """
    if policy_manager is None:
        return {"error": "Plugin not initialized"}

    if not _policy_write_override(kwargs):
        return {
            "error": (
                "revenue-unignore is deprecated for normal operator use. "
                "Use revenue-policy list/get/find/changes for diagnostics."
            )
        }
    if not re.match(r'^[0-9a-fA-F]{66}$', str(peer_id or "")):
        return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}

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
def revenue_policy(plugin: Plugin, action: str = "list", peer_id: str = None,
                   strategy: str = None, rebalance: str = None,
                   fee_ppm: int = None, tag: str = None,
                   fee_multiplier_min: float = None,
                   fee_multiplier_max: float = None,
                   expires_in_hours: int = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Manage peer-level fee and rebalance policies (v1.4 API).

    Normal operator use is now read-only. Tactical policy writes remain
    available only through explicit internal/debug override flags.

    Usage:
      lightning-cli revenue-policy list                           # List all policies
      lightning-cli revenue-policy get <peer_id>                  # Get policy for peer
      lightning-cli revenue-policy find <tag>                     # Find peers by tag
      lightning-cli revenue-policy changes [since=<timestamp>]    # Get policy changes
      lightning-cli revenue-policy set <peer_id> [options]        # Deprecated for normal operator use
      lightning-cli revenue-policy delete <peer_id>               # Deprecated for normal operator use
      lightning-cli revenue-policy tag <peer_id> <tag>            # Deprecated for normal operator use
      lightning-cli revenue-policy untag <peer_id> <tag>          # Deprecated for normal operator use

    Options for 'set':
      strategy=dynamic|static|passive   Fee control strategy
      rebalance=enabled|disabled|source_only|sink_only   Rebalance mode
      fee_ppm=N   Target fee for static strategy (required if strategy=static)
      fee_multiplier_min=X.Y   Dynamic fee floor multiplier (uses fee_ppm_target as anchor)
      fee_multiplier_max=X.Y   Dynamic fee ceiling multiplier (uses fee_ppm_target as anchor)
      expires_in_hours=N       Optional auto-expiry for policy (revert to defaults)

    Strategies:
      dynamic  - DTS+PID fee optimization (default)
      static   - Fixed fee (requires fee_ppm)
      passive  - Do not manage (manual control)

    Rebalance Modes:
      enabled     - Full rebalancing allowed (default)
      disabled    - No rebalancing (equivalent to old 'ignore')
      source_only - Can drain from, cannot fill
      sink_only   - Can fill, cannot drain from

    Options for 'changes':
      since=<timestamp>   Unix timestamp. Returns policies changed after this time.
                          If omitted, returns all policies.

    Options for 'batch':
      updates='[...]'     JSON array of policy updates. Each entry has:
                          peer_id, strategy, rebalance_mode, fee_ppm_target, tags
                          Bypasses rate limiting for bulk batch updates.

    Examples:
      lightning-cli revenue-policy set 02abc... strategy=static fee_ppm=500
      lightning-cli revenue-policy set 02abc... strategy=passive rebalance=disabled
      lightning-cli revenue-policy tag 02abc... whale
      lightning-cli -k revenue-policy action=changes since=1704067200
    """
    if policy_manager is None:
        return {"error": "Plugin not initialized"}

    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    action = str(action or "").strip().lower()
    internal_override = _truthy(kwargs.get("internal")) or _truthy(kwargs.get("admin"))

    if action in TACTICAL_POLICY_ACTIONS and not internal_override:
        return {
            "error": (
                f"revenue-policy {action} is deprecated for normal operator use. "
                "Use revenue-policy list/get/find/changes for diagnostics."
            )
        }
    
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
            # Get policy changes since timestamp
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
            # Bulk policy updates (bypasses rate limiting)
            # Usage: revenue-policy batch updates='[{"peer_id": "...", "strategy": "..."}, ...]'
            updates_json = kwargs.get('updates', '[]')
            try:
                import json
                if isinstance(updates_json, str):
                    updates = json.loads(updates_json)
                else:
                    updates = updates_json
                if not isinstance(updates, list):
                    return {"error": "updates must be a JSON array"}
                # AUDIT FIX Issue-18: Cap batch size to prevent unbounded processing
                if len(updates) > 100:
                    return {"error": f"Batch too large: {len(updates)} entries, max 100"}
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
            allowed = sorted(READ_ONLY_POLICY_ACTIONS | TACTICAL_POLICY_ACTIONS)
            allowed_text = ", ".join(f"'{name}'" for name in allowed)
            return {"error": f"Unknown action: {action}. Use {allowed_text}"}
    
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
      lightning-cli revenue-report policies          # Policy distribution stats
      lightning-cli revenue-report costs             # Closure cost history

    Report Types:
      summary   - Overall node P&L, active channels, warnings
      peer      - Specific peer metrics (profitability, flow, policy)
      policies  - Statistics on policy distribution
      costs     - Closure costs for capacity planning
    """
    if database is None or policy_manager is None:
        return {"error": "Plugin not initialized"}
    
    try:
        if report_type == "summary":
            # Basic summary - expand with P&L when available
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
            
            # Get per-channel flow states for this peer.
            flow_states = []
            if database:
                states = database.get_all_channel_states()
                flow_states = sorted(
                    [s for s in states if s.get("peer_id") == peer_id],
                    key=lambda state: state.get("channel_id") or state.get("short_channel_id") or "",
                )
            
            return {
                "type": "peer",
                "peer_id": peer_id,
                "policy": policy.to_dict(),
                "profitability": prof_data,
                "flow_state": flow_states[0] if len(flow_states) == 1 else None,
                "flow_states": flow_states,
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
            # Expose closure/swap costs for capacity planning
            now = int(time.time())
            day_ago = now - 86400
            week_ago = now - (7 * 86400)
            month_ago = now - (30 * 86400)

            # Get historical costs
            closure_costs_day = database.get_closure_costs_since(day_ago)
            closure_costs_week = database.get_closure_costs_since(week_ago)
            closure_costs_month = database.get_closure_costs_since(month_ago)
            closure_costs_total = database.get_total_closure_costs()

            # Include default chain cost estimates
            from modules.config import ChainCostDefaults
            estimated_costs = {
                "channel_open_sats": ChainCostDefaults.CHANNEL_OPEN_COST_SATS,
                "channel_close_sats": ChainCostDefaults.CHANNEL_CLOSE_COST_SATS,
            }

            return {
                "type": "costs",
                "closure_costs": {
                    "last_24h_sats": closure_costs_day,
                    "last_7d_sats": closure_costs_week,
                    "last_30d_sats": closure_costs_month,
                    "total_sats": closure_costs_total
                },
                "estimated_defaults": estimated_costs,
                "generated_at": now
            }

        else:
            return {"error": f"Unknown report type: {report_type}. Use 'summary', 'peer', 'policies', or 'costs'"}
    
    except Exception as e:
        return {"status": "error", "error": f"Report generation failed: {e}"}


@plugin.method("revenue-hot-channel-protection-peers")
def revenue_hot_channel_protection_peers(plugin: Plugin, action: str = "list", peer_id: str = None, note: str = None, min_depletion_trigger_pct: float = None) -> Dict[str, Any]:
    """Manage persistent peer overrides for hot-channel protection.

    Actions:
      list
      add <peer_id> [note] [min_depletion_trigger_pct]
      remove <peer_id>
      clear
    """
    if database is None:
        return {"error": "Plugin not initialized"}

    action = str(action or "list").lower()
    try:
        if action == "list":
            rows = database.list_hot_channel_protection_override_peers()
            return {"status": "success", "count": len(rows), "peers": rows}

        if action == "add":
            if not peer_id:
                return {"error": "Usage: revenue-hot-channel-protection-peers add <peer_id> [note] [min_depletion_trigger_pct]"}
            if not re.match(r'^[0-9a-fA-F]{66}$', peer_id):
                return {"error": "Invalid peer_id format: expected 66-character hex pubkey"}
            if min_depletion_trigger_pct is not None:
                try:
                    pct_val = float(min_depletion_trigger_pct)
                except (ValueError, TypeError):
                    return {"error": "min_depletion_trigger_pct must be a number"}
                if not (0.0 < pct_val <= 100.0):
                    return {"error": "min_depletion_trigger_pct must be between 0 (exclusive) and 100"}
            database.add_hot_channel_protection_override_peer(str(peer_id), note or "", min_depletion_trigger_pct=min_depletion_trigger_pct)
            plugin.log(f"HOT CHANNEL OVERRIDE: added peer {peer_id}" + (f" depletion_trigger={float(min_depletion_trigger_pct):.1f}%" if min_depletion_trigger_pct is not None else ""), level='info')
            rows = database.list_hot_channel_protection_override_peers()
            return {"status": "success", "action": "add", "peer_id": str(peer_id), "count": len(rows), "peers": rows}

        if action == "remove":
            if not peer_id:
                return {"error": "Usage: revenue-hot-channel-protection-peers remove <peer_id>"}
            removed = database.remove_hot_channel_protection_override_peer(str(peer_id))
            rows = database.list_hot_channel_protection_override_peers()
            return {"status": "success", "action": "remove", "peer_id": str(peer_id), "removed": bool(removed), "count": len(rows), "peers": rows}

        if action == "clear":
            rows = database.list_hot_channel_protection_override_peers()
            removed = 0
            for r in rows:
                if database.remove_hot_channel_protection_override_peer(str(r.get('peer_id') or '')):
                    removed += 1
            return {"status": "success", "action": "clear", "removed": removed, "count": 0, "peers": []}

        return {"error": f"Unknown action: {action}. Use list|add|remove|clear"}
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-config")
def revenue_config(
    plugin: Plugin,
    action: str = "get",
    key: str = None,
    value: str = None,
    **_unused: Any,
) -> Dict[str, Any]:
    """
    Get or set runtime configuration (Dynamic Runtime Configuration).
    
    Usage:
      lightning-cli revenue-config get           # Get public operator controls
      lightning-cli revenue-config get <key>     # Get specific key
      lightning-cli revenue-config set <key> <value>  # Set key
      lightning-cli revenue-config reset <key>   # Reset to default
      lightning-cli revenue-config list-mutable  # List changeable keys
    
    Examples:
      lightning-cli revenue-config get daily_budget_sats
      lightning-cli revenue-config set daily_budget_sats 10000
      lightning-cli revenue-config set paused true
    """
    if config is None or database is None:
        return {"error": "Plugin not initialized"}

    # P1-012 class: guard non-str key before hasattr/getattr(config, key),
    # which raise TypeError for a non-string attribute name.
    if key is not None and not isinstance(key, str):
        return {"error": "config key must be a string"}

    def _not_public_error(runtime_key: str) -> Dict[str, Any]:
        return {"error": f"Key '{runtime_key}' is not a public runtime control"}
    
    if action == "get":
        if key:
            if not hasattr(config, key) or key.startswith('_'):
                return {"error": f"Unknown config key: {key}"}
            result = {
                "key": key,
                "value": getattr(config, key),
                "version": config._version,
                "classification": config.classify_runtime_key(key),
            }
            if not config.is_public_runtime_key(key):
                result["warning"] = _not_public_error(key)["error"]
            return result
        else:
            return {
                "config": config.public_runtime_dict(),
                "version": config._version
            }
    
    elif action == "set":
        if not key or value is None:
            return {"error": "Usage: revenue-config set <key> <value>"}

        if not config.is_public_runtime_key(key):
            return _not_public_error(key)
        
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

        if not config.is_public_runtime_key(key):
            return _not_public_error(key)
        
        if database.delete_config_override(key):
            return {
                "status": "success",
                "message": f"Override for '{key}' removed. Restart plugin to apply default."
            }
        return {"error": f"No override found for '{key}'"}
    
    elif action == "list-mutable":
        mutable = sorted(config.public_runtime_keys())
        return {"mutable_keys": sorted(mutable), "count": len(mutable)}
    
    else:
        return {"error": f"Unknown action: {action}. Use 'get', 'set', 'reset', or 'list-mutable'"}


@plugin.method("revenue-dashboard")
def revenue_dashboard(plugin: Plugin, window_days: int = 30) -> Dict[str, Any]:
    """
    P&L Engine

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
    # P1-012: coerce operator-supplied window_days; a non-int must return a
    # clean error dict, not leak a ValueError traceback.
    try:
        window_days = max(1, min(int(window_days), 365))
    except (ValueError, TypeError):
        return {"error": "window_days must be an integer"}

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
            scid = (
                bleeder.get("channel_id")
                or bleeder.get("short_channel_id")
                or "unknown"
            )
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
                "volume_sats": pnl.get("volume_sats", 0),
                "forward_count": pnl.get("forward_count", 0),
            },
            "warnings": warnings,
            "bleeder_count": len(bleeders)
        }
    except Exception as e:
        plugin.log(f"Error generating revenue dashboard: {e}", level='error')
        return {"error": str(e)}


@plugin.method("revenue-health")
def revenue_health(plugin: Plugin) -> Dict[str, Any]:
    """
    Consolidated health check -- single command for full operational picture.

    Usage: lightning-cli revenue-health

    Returns P&L, fee convergence, rebalance state, budget health,
    channel classification trends, and coordination status.
    """
    now = int(time.time())
    result = {"generated_at": now}

    # --- 1. Financial health (1-day and 7-day) ---
    if profitability_analyzer and database:
        try:
            pnl_1d = profitability_analyzer.get_pnl_summary(1)
            pnl_7d = profitability_analyzer.get_pnl_summary(7)
            roc_7d = profitability_analyzer.calculate_roc(7)
            result["financials"] = {
                "today": {
                    "revenue_sats": pnl_1d.get("gross_revenue_sats", 0),
                    "costs_sats": pnl_1d.get("opex_sats", 0),
                    "net_profit_sats": pnl_1d.get("net_profit_sats", 0),
                    "forward_count": pnl_1d.get("forward_count", 0),
                    "volume_sats": pnl_1d.get("volume_sats", 0),
                },
                "week": {
                    "revenue_sats": pnl_7d.get("gross_revenue_sats", 0),
                    "costs_sats": pnl_7d.get("opex_sats", 0),
                    "net_profit_sats": pnl_7d.get("net_profit_sats", 0),
                    "forward_count": pnl_7d.get("forward_count", 0),
                    "operating_margin_pct": round(pnl_7d.get("operating_margin_pct", 0.0), 1),
                    "annualized_roc_pct": round(roc_7d.get("annualized_roc_pct", 0.0), 2),
                },
            }
        except Exception as e:
            result["financials"] = {"error": str(e)}

    # --- 2. Channel classifications ---
    if profitability_analyzer:
        try:
            all_prof = profitability_analyzer.analyze_all_channels()
            classifications = {}
            for prof in all_prof.values():
                cls = getattr(getattr(prof, 'classification', None), 'value', 'unknown')
                classifications[cls] = classifications.get(cls, 0) + 1
            result["channels"] = {
                "total": len(all_prof),
                "classifications": classifications,
            }
        except Exception as e:
            result["channels"] = {"error": str(e)}

    # --- 3. Fee convergence ---
    if fee_controller:
        try:
            sparse_count = 0
            converged_count = 0
            sleeping_count = 0
            total_managed = 0
            for cid, fs in list(fee_controller._channel_fee_states.items()):
                total_managed += 1
                ts = fs.thompson
                if ts.posterior_std < 50:
                    converged_count += 1
                if fs.is_sleeping:
                    sleeping_count += 1
            for cid, cs in list(fee_controller._cycle_states.items()):
                if cid not in fee_controller._channel_fee_states:
                    total_managed += 1
                    if cs.is_sleeping:
                        sleeping_count += 1
            # Sparse detection from last cycle log isn't stored; approximate
            sparse_count = max(0, total_managed - converged_count)
            result["fees"] = {
                "managed_channels": total_managed,
                "converged": converged_count,
                "still_learning": sparse_count,
                "sleeping": sleeping_count,
            }
        except Exception as e:
            result["fees"] = {"error": str(e)}

    # --- 4. Rebalance state ---
    if rebalancer:
        try:
            coord = rebalancer.get_boltz_coordination()
            decision = rebalancer.get_last_decision_summary()
            active_jobs = rebalancer.job_manager.active_job_count if hasattr(rebalancer, 'job_manager') else 0
            result["rebalancer"] = {
                "last_action": decision.get("action"),
                "last_reason": decision.get("reason"),
                "active_jobs": active_jobs,
                "depleted_channels": coord.get("depleted_count", 0),
                "profitable_candidates": coord.get("profitable_count", 0),
                "rebalancer_exhausted": coord.get("rebalancer_exhausted", False),
            }
        except Exception as e:
            result["rebalancer"] = {"error": str(e)}

    # --- 5. Budget health ---
    try:
        budget = _total_cost_budget_status()
        if isinstance(budget, dict):
            actual_spent = int(budget.get("actual_spent_sats", 0) or 0)
            result["budget"] = {
                "effective_budget_sats": budget.get("effective_budget_sats", 0),
                "total_spent_sats": actual_spent,
                "remaining_sats": budget.get("remaining_sats", 0),
                "spent_by_category": budget.get("actual_spent_by_category", {}),
                "utilization_pct": round(
                    100.0 * actual_spent / max(1, budget.get("effective_budget_sats", 1)), 1
                ),
            }
    except Exception as e:
        result["budget"] = {"error": str(e)}

    # --- 6. Boltz state ---
    if boltz_manager and getattr(boltz_manager, 'enabled', False):
        try:
            with _boltz_auto_cycle_state_lock:
                cycle_state = dict(_boltz_auto_cycle_state)
            result["boltz"] = {
                "last_mode": cycle_state.get("last_result", {}).get("mode") if isinstance(cycle_state.get("last_result"), dict) else None,
                "last_status": cycle_state.get("last_result", {}).get("status") if isinstance(cycle_state.get("last_result"), dict) else None,
                "consecutive_errors": cycle_state.get("consecutive_errors", 0),
                "running": cycle_state.get("running", False),
            }
        except Exception as e:
            result["boltz"] = {"error": str(e)}
    else:
        result["boltz"] = {"enabled": False}

    # --- 7. Planner state ---
    if capacity_planner:
        try:
            coord = capacity_planner.get_boltz_coordination()
            status = capacity_planner.get_status()
            result["planner"] = {
                "enabled": status.get("enabled", False),
                "loser_close_candidates": len(coord.get("loser_scids", set())),
                "funding_deficit_sats": coord.get("funding_deficit_sats", 0),
                "candidate_pool_size": status.get("candidate_pool_size", 0),
            }
        except Exception as e:
            result["planner"] = {"error": str(e)}

    # --- 8. Route pairs (top 5 revenue routes) ---
    if database:
        try:
            pairs = database.get_top_route_pairs(days=7, min_forwards=2, limit=5)
            result["top_routes"] = [
                {
                    "in_channel": str(p.get("in_channel", "")).replace(":", "x"),
                    "out_channel": str(p.get("out_channel", "")).replace(":", "x"),
                    "fee_sats_7d": int(p.get("total_fee_msat", 0)) // 1000,
                    "forward_count": int(p.get("forward_count", 0)),
                }
                for p in pairs
            ]
        except Exception:
            result["top_routes"] = []

    # --- 9. Daemon-loop liveness (DD5 / P1-010 heartbeat surface) ---
    # Per-thread last-iteration age + alive/stalled so a dead or stalled daemon
    # loop is operator-detectable instead of failing silently.
    try:
        loops = _loop_liveness_snapshot()
        stalled = sorted(n for n, v in loops.items() if v.get("state") == "stalled")
        result["loops"] = {
            "threads": loops,
            "stalled": stalled,
            "all_alive": len(stalled) == 0,
            "stall_threshold_seconds": _LOOP_STALL_SECONDS,
        }
    except Exception as e:
        result["loops"] = {"error": str(e)}

    return result


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
        # Get all channels currently tracked in channel_states
        tracked_channels = database.get_all_channel_states()
        tracked_ids = {ch['channel_id'] for ch in tracked_channels}

        if not tracked_ids:
            return {"message": "No tracked channels found", **result}

        # Get all currently open channels
        open_ids = set()
        try:
            channels = data_service.get_peer_channels() if data_service else safe_plugin.rpc.call("listpeerchannels")
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
            closed_list = data_service.get_closed_channels() if data_service else safe_plugin.rpc.call("listclosedchannels")
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

    Use this command to release stale budget reservations. This resets
    the reservation system so new rebalances can use the daily budget.

    Typical workflow:
    1. lightning-cli revenue-clear-reservations  # Release budget

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



def _resolve_scid_to_peer(scid: str) -> Optional[str]:
    """
    Resolve a short_channel_id to its peer_id.

    Uses a cache to avoid repeated RPC calls. Cache is refreshed if the
    SCID is not found (channel might be new).

    Unknown SCIDs are negatively cached (value None) so a burst of forwards
    referencing a closed/foreign channel doesn't rebuild the SCID map on every
    event. Negative entries are bounded and flushed by the hourly cache clear.

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

    # Cache miss - RPC call outside cache lock
    # M-2 FIX: Serialize cache-miss fetches so only one thread makes the RPC call
    with _scid_cache_fetch_lock:
        # Re-check cache: another thread may have populated it while we waited
        with _scid_cache_lock:
            if scid_norm in _scid_to_peer_cache:
                return _scid_to_peer_cache[scid_norm]

        try:
            result = data_service.get_peer_channels() if data_service else safe_plugin.rpc.listpeerchannels()
            new_cache = {}
            for channel in result.get("channels", []):
                channel_scid = channel.get("short_channel_id") or channel.get("channel_id")
                peer_id = channel.get("peer_id")
                if channel_scid and peer_id:
                    new_cache[normalize_scid(channel_scid)] = peer_id

            resolved = new_cache.get(scid_norm)
            with _scid_cache_lock:
                _scid_to_peer_cache.update(new_cache)
                if resolved is None:
                    # Negative cache: remember the miss so the next forward on
                    # this unknown SCID doesn't rebuild the map again. Bounded
                    # to keep hostile/garbage SCIDs from growing the dict; the
                    # hourly clear above flushes entries (e.g. newly confirmed
                    # channels) naturally.
                    negative_count = sum(1 for v in _scid_to_peer_cache.values() if v is None)
                    if negative_count < _SCID_NEGATIVE_CACHE_MAX_ENTRIES:
                        _scid_to_peer_cache[scid_norm] = None

            return resolved
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
    # Try per-peer first (filtered), then all channels as fallback
    peer_queries = []
    if isinstance(peer_id, str) and len(peer_id) == 66:
        peer_queries.append(peer_id)
    peer_queries.append(None)  # None = all channels

    for query_peer_id in peer_queries:
        try:
            if data_service:
                channels = data_service.get_peer_channels(query_peer_id).get("channels", [])
            else:
                payload = {"id": query_peer_id} if query_peer_id else {}
                channels = safe_plugin.rpc.call("listpeerchannels", payload).get("channels", [])
            scid = _match_scid_from_channels(channels)
            if scid:
                return scid
        except Exception as e:
            plugin.log(f"SCID resolution attempt failed for {raw_channel_id_lc[:16]}: {e}", level='debug')
            continue

    # Fallback for CLOSED events where the channel may already be gone
    try:
        closed_result = data_service.get_closed_channels() if data_service else safe_plugin.rpc.call("listclosedchannels")
        closed = closed_result.get("closedchannels", [])
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
    except Exception as e:
        plugin.log(f"Closed channel SCID fallback failed for {raw_channel_id_lc[:16]}: {e}", level='debug')

    return None


def _parse_msat(msat_val: Any) -> int:
    """
    Safely convert msat values to integers.
    Handles '1000msat' strings, raw integers, Millisatoshi objects, and plain numeric strings.
    """
    return parse_msat(msat_val)


def _refresh_dynamic_config():
    """Read dynamic option values from CLN and update in-memory objects.

    Called at the top of each boltz/planner cycle. CLN stores setconfig
    values persistently but pyln-client's plugin.options dict is only
    populated at init. This function bridges the gap.
    """
    try:
        # Use the timeout-protected proxy: this runs on the boltz/planner
        # background threads, where a hung lightningd response would
        # otherwise stall the loop indefinitely.
        rpc = safe_plugin.rpc if safe_plugin is not None else plugin.rpc
        all_configs = rpc.listconfigs()
        configs = all_configs.get("configs", {})
    except Exception:
        return

    if boltz_manager and boltz_manager.cfg:
        eb = configs.get("revenue-ops-boltz-enforce-budget", {})
        val = eb.get("value_str", "")
        if val:
            new_val = val.lower() in ("true", "1", "yes")
            if new_val != boltz_manager.cfg.enforce_budget:
                boltz_manager.cfg.enforce_budget = new_val
                plugin.log(f"Dynamic config refresh: enforce_budget = {new_val}")

        db_cfg = configs.get("revenue-ops-boltz-daily-budget-sats", {})
        val = db_cfg.get("value_str", "")
        if val:
            try:
                new_val = int(val)
                if new_val != boltz_manager.cfg.daily_budget_sats:
                    boltz_manager.cfg.daily_budget_sats = new_val
                    plugin.log(f"Dynamic config refresh: daily_budget_sats = {new_val}")
            except ValueError:
                pass

    if config:
        ec = configs.get("revenue-ops-planner-execute-closes", {})
        val = ec.get("value_str", "")
        if val:
            new_val = val.lower() in ("true", "1", "yes")
            if new_val != config.planner_execute_closes:
                config.planner_execute_closes = new_val
                plugin.log(f"Dynamic config refresh: planner_execute_closes = {new_val}")


# Bounded wait for fee_controller._state_lock in the forward_event handler.
# adjust_all_fees holds the lock for the whole fee cycle; pyln dispatches all
# notifications on one thread, so we must never block here indefinitely.
FORWARD_EVENT_LOCK_TIMEOUT_SECS = 2.0


@plugin.subscribe("forward_event")
def on_forward_event(forward_event: Dict, plugin: Plugin, **kwargs):
    """
    Notification when a forward completes (success or failure).

    We use this for:
    1. Real-time flow tracking (settled forwards)
    2. Peer reputation tracking (success/failure rates)

    Reputation tracking helps identify unreliable peers for traffic intelligence.
    """
    # AUDIT FIX: Top-level guard prevents unhandled exceptions from crashing CLN event processing
    try:
        _on_forward_event_impl(forward_event, plugin, **kwargs)
    except Exception as e:
        plugin.log(f"Error in forward_event handler: {e}", level='error')


def _on_forward_event_impl(forward_event: Dict, plugin: Plugin, **kwargs):
    if database is None:
        return

    status = forward_event.get("status")
    in_channel = normalize_scid(forward_event.get("in_channel")) if forward_event.get("in_channel") else None

    # Per-forward write coalescing: when the database exposes the combined
    # single-transaction method, settled forwards record the forward row AND
    # the reputation upsert in one transaction instead of two autocommit writes.
    record_combined = getattr(database, "record_forward_and_reputation", None)

    # Track peer reputation for all forward outcomes
    peer_id = _resolve_scid_to_peer(in_channel) if in_channel else None
    if peer_id:
        if status == "settled":
            if not callable(record_combined):
                database.update_peer_reputation(peer_id, is_success=True)
            # else: deferred to the combined settled-forward write below
        elif status == "failed":
            # Only penalize in_channel peer on downstream failure, NOT
            # local_failed (which means OUR node rejected the forward,
            # e.g. insufficient outbound balance — the sender did nothing wrong).
            database.update_peer_reputation(peer_id, is_success=False)

    # Record fee-relevant failed forwards as a weak negative DTS signal
    # (amount-weighted), keyed to the OUT channel.
    #
    # Audit DTS-4:
    # (a) Per BOLT 7 the fee a sender pays for traversing our node is OUR
    #     advertised policy on the OUTGOING channel — the in_channel's fee
    #     is set by our peer, so the old in_channel nudge was systematically
    #     training the wrong channel's posterior.
    # (b) CLN's forward_event carries failcode/failreason ONLY when status
    #     is "local_failed" (our node rejected the HTLC, e.g.
    #     WIRE_FEE_INSUFFICIENT when the offered fee no longer covers our
    #     out-channel policy after a fee raise). A plain "failed" is a
    #     downstream error inside an onion we cannot decrypt: no usable
    #     failure reason exists, and most such failures are liquidity
    #     failures that say nothing about our fee, so the nudge is DROPPED
    #     entirely — a misdirected systematic signal is worse than none.
    if status in ("failed", "local_failed") and fee_controller is not None:
        out_scid = forward_event.get("out_channel")
        out_scid = normalize_scid(out_scid) if out_scid else None
        failcode = forward_event.get("failcode")
        failreason = forward_event.get("failreason")
        if out_scid and fee_controller.is_fee_relevant_failure(failcode, failreason):
            try:
                # _channel_fee_states is mutated by the fee loop under _state_lock,
                # which adjust_all_fees holds across the ENTIRE fee cycle (including
                # dozens of setchannel RPCs). pyln dispatches all notifications and
                # RPC handlers on a single thread, so a blocking acquire here would
                # freeze the whole plugin until the cycle ends. Use a bounded
                # acquire and drop the nudge under contention — it is advisory
                # negative feedback, so losing one is harmless.
                if fee_controller._state_lock.acquire(timeout=FORWARD_EVENT_LOCK_TIMEOUT_SECS):
                    try:
                        cfs = fee_controller._channel_fee_states.get(out_scid)
                        current_fee = cfs.last_fee_ppm if cfs else 0
                        if current_fee > 0:
                            failed_in_msat = _parse_msat(forward_event.get("in_msat", forward_event.get("in_msatoshi", 0)))
                            fee_controller.record_failed_forward(
                                out_scid, current_fee,
                                amount_msat=failed_in_msat,
                                failcode=failcode,
                                failreason=failreason,
                            )
                    finally:
                        fee_controller._state_lock.release()
                else:
                    plugin.log(
                        "forward_event: fee state lock busy (fee cycle in progress); "
                        "skipping failed-forward nudge",
                        level='debug'
                    )
            except Exception:
                pass

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

        if callable(record_combined) and peer_id:
            record_combined(
                {
                    "in_channel": in_channel or "",
                    "out_channel": out_channel or "",
                    "in_msat": in_msat,
                    "out_msat": out_msat,
                    "fee_msat": fee_msat,
                    "received_time": received_time,
                    "resolved_time": resolved_time,
                    "resolution_time": resolution_duration,
                },
                peer_id,
                True,
            )
        else:
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


@plugin.subscribe("connect")
def on_peer_connect(plugin: Plugin, **kwargs):
    """
    Notification when a peer connects.

    Records the connection event for uptime tracking.
    """
    # AUDIT FIX: Top-level guard prevents unhandled exceptions from crashing CLN event processing
    try:
        _on_peer_connect_impl(plugin, **kwargs)
    except Exception as e:
        plugin.log(f"Error in peer_connect handler: {e}", level='error')


def _on_peer_connect_impl(plugin: Plugin, **kwargs):
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
    # AUDIT FIX: Top-level guard prevents unhandled exceptions from crashing CLN event processing
    try:
        _on_peer_disconnect_impl(plugin, **kwargs)
    except Exception as e:
        plugin.log(f"Error in peer_disconnect handler: {e}", level='error')


def _on_peer_disconnect_impl(plugin: Plugin, **kwargs):
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
    # AUDIT FIX: Top-level guard prevents unhandled exceptions from crashing CLN event processing
    try:
        _on_channel_state_changed_impl(plugin, **kwargs)
    except Exception as e:
        plugin.log(f"Error in channel_state_changed handler: {e}", level='error')


def _on_channel_state_changed_impl(plugin: Plugin, **kwargs):
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

    # Pre-confirmation states never have an SCID — skip silently
    pre_confirmation_states = {
        'CHANNELD_AWAITING_LOCKIN', 'DUALOPEND_AWAITING_LOCKIN',
        'DUALOPEND_OPEN_INIT', 'OPENINGD',
    }
    if new_state in pre_confirmation_states:
        plugin.log(
            f"Channel {raw_channel_id[:16]}... entered {new_state} (awaiting confirmation)",
            level='debug'
        )
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
    # Channel Open Detection
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

    # Bookkeeper accounts use the hex channel_id, not the SCID.
    # The event's 'channel_id' field is the hex format bookkeeper expects.
    bkpr_account = event.get('channel_id') or channel_id

    try:
        # Try to get on-chain fee data from bookkeeper
        closure_data = _get_closure_costs_from_bookkeeper(bkpr_account)
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


def _handle_channel_open(channel_id: str, peer_id: Optional[str],
                          old_state: str, cause: str) -> None:
    """
    Handle a channel open event.

    Called when a channel transitions to CHANNELD_NORMAL from an opening state.
    Sets initial fee for the new channel.

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


def _get_closure_costs_from_bookkeeper(bkpr_account: str) -> Optional[Dict[str, Any]]:
    """
    Query bookkeeper for on-chain fees related to channel closure.

    Uses bkpr-inspect to get fees_paid_msat per transaction directly,
    avoiding raw event scanning.

    Args:
        bkpr_account: Bookkeeper account identifier (hex channel_id, not SCID)

    Returns:
        Dict with closure_fee_sats, htlc_sweep_fee_sats, funding_txid, closing_txid
        or None if bookkeeper unavailable
    """
    global safe_plugin, data_service

    if safe_plugin is None:
        return None

    try:
        result = data_service.bkpr_inspect(bkpr_account) if data_service else safe_plugin.rpc.call("bkpr-inspect", {"account": bkpr_account})

        if not result or "txs" not in result:
            return None

        txs = result.get("txs", [])
        if not isinstance(txs, list):
            plugin.log(f"Security: Invalid txs structure from bkpr-inspect for {bkpr_account}", level='warn')
            return None

        closure_fee_sats = 0
        htlc_sweep_fee_sats = 0
        funding_txid = None
        closing_txid = None

        for tx in txs:
            if not isinstance(tx, dict):
                continue

            txid = tx.get("txid")
            fees_msat = parse_msat(tx.get("fees_paid_msat", 0))
            fee_sats = min(fees_msat // 1000, 50000)  # Bounds check

            outputs = tx.get("outputs", [])
            if not isinstance(outputs, list):
                continue

            tags = {o.get("output_tag", "") for o in outputs if isinstance(o, dict)}
            spend_tags = {o.get("spend_tag", "") for o in outputs if isinstance(o, dict)}
            all_tags = tags | spend_tags

            if "channel_open" in all_tags:
                funding_txid = txid

            is_close = any(
                t for t in all_tags
                if t in ("channel_close", "mutual_close", "unilateral_close")
            )
            is_sweep = any(
                t for t in all_tags
                if "htlc" in t.lower() or "sweep" in t.lower()
            )

            if is_sweep:
                htlc_sweep_fee_sats += fee_sats
            elif is_close:
                closing_txid = txid
                closure_fee_sats += fee_sats

        return {
            'closure_fee_sats': closure_fee_sats,
            'htlc_sweep_fee_sats': htlc_sweep_fee_sats,
            'funding_txid': funding_txid,
            'closing_txid': closing_txid
        }

    except Exception as e:
        plugin.log(f"Bookkeeper query failed for {bkpr_account}: {e}", level='debug')
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
        total_revenue_sats = pnl.get('revenue_msat', 0) // 1000
        total_rebalance_cost_sats = pnl.get('rebalance_cost_sats', 0)
        forward_count = pnl.get('forward_count', 0)

        # Determine closer from close_type
        closer = _determine_closer(close_type)

        # Try to get capacity and additional info from listclosedchannels (CLN v23.11+)
        capacity_sats = 0
        if safe_plugin:
            try:
                closed_result = data_service.get_closed_channels() if data_service else safe_plugin.rpc.call("listclosedchannels")
                for ch in closed_result.get('closedchannels', []):
                    if normalize_scid(ch.get('short_channel_id', '')) == channel_id:
                        capacity_sats = _parse_msat(ch.get('total_msat', 0)) // 1000
                        if not peer_id:
                            peer_id = ch.get('peer_id')
                        # CLN provides 'closer' field in listclosedchannels (v24.02+)
                        if closer == 'unknown' and ch.get('closer'):
                            closer = ch.get('closer')  # 'local' or 'remote'
                        break
            except Exception as e:
                plugin.log(f"Closed channel lookup failed for {channel_id[:12]}...: {e}", level='debug')

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

    except Exception as e:
        plugin.log(f"Error archiving closed channel {channel_id}: {e}", level='error')
        plugin.log(f"Traceback: {traceback.format_exc()}", level='debug')



# =============================================================================
# BOLTZ CLI INTEGRATION (optional)
# =============================================================================

def _require_boltz_manager() -> BoltzCliManager:
    if boltz_manager is None:
        raise BoltzCliError("Boltz CLI integration not initialized")
    return boltz_manager


def _structural_envelope_sats_provider() -> int:
    """Current daily structural envelope (sats) for the Boltz manager.

    Reads the live config at call time so revenue-config runtime updates to
    boltz_structural_budget_sats_per_day take effect without a restart.
    """
    return int(getattr(config, "boltz_structural_budget_sats_per_day", 0) or 0)


@plugin.method("revenue-total-cost-budget")
def revenue_total_cost_budget(plugin: Plugin, window_hours: int = None) -> Dict[str, Any]:
    """Unified budget status across rebalances, Boltz, and on-chain liquidity costs."""
    try:
        return _total_cost_budget_status(window_hours=window_hours)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-capex-status")
def revenue_capex_status(plugin, **kwargs):
    """Return unified capex budget allocations.

    Shows per-channel budgets, fleet exploration budget, tactical budget,
    priority class, and global envelope. Pushes summary to datastore.
    """
    global capex_engine
    if capex_engine is None:
        return {"error": "Capex engine not initialized"}

    try:
        alloc = capex_engine.compute_allocations()
    except Exception as e:
        return {"error": f"Allocation failed: {e}"}

    # Format channel budgets
    channels = {}
    for ch_id, b in alloc.channel_budgets.items():
        channels[ch_id] = {
            "budget_sats": b.budget_sats,
            "tier": b.tier,
            "tier_ppm": b.tier_ppm,
            "priority_class": b.priority_class,
            "hive_multiplier": b.hive_multiplier,
        }

    now = int(time.time())
    result = {
        "timestamp": now,
        "generated_at": now,
        "ttl_seconds": 1800,
        "status": "ok",
        "priority_class": alloc.priority_class,
        "global_envelope_sats": alloc.global_envelope_sats,
        "fleet_exploration_budget_sats": alloc.fleet_exploration_budget_sats,
        "tactical_budget_sats": alloc.tactical_budget_sats,
        "total_fleet_contribution_sats": alloc.total_fleet_contribution_sats,
        "allocated_by_priority_sats": alloc.allocated_by_priority_sats,
        "channel_count": len(channels),
        "channels": channels,
    }

    # Push to datastore for MCP consumption
    try:
        import json as _json
        summary = {
            "timestamp": now,
            "generated_at": now,
            "ttl_seconds": 1800,
            "status": "ok",
            "priority_class": alloc.priority_class,
            "global_envelope_sats": alloc.global_envelope_sats,
            "fleet_exploration_budget_sats": alloc.fleet_exploration_budget_sats,
            "tactical_budget_sats": alloc.tactical_budget_sats,
            "total_fleet_contribution_sats": alloc.total_fleet_contribution_sats,
            "allocated_by_priority_sats": alloc.allocated_by_priority_sats,
            "channel_count": len(channels),
        }
        if data_service:
            data_service.datastore_push(["revenue", "capex-summary"], summary)
    except Exception:
        pass  # datastore_push handles its own error logging

    return result


@plugin.method("revenue-spend-ledger")
def revenue_spend_ledger(
    plugin: Plugin,
    window_hours: int = 24,
    include_reservations: bool = False,
    reservation_limit: int = 50,
) -> Dict[str, Any]:
    """Summary of generic spend ledger events/reservations (for opens/closes/etc.)."""
    if database is None:
        return {"error": "Database not initialized"}
    try:
        return database.get_spend_ledger_summary(
            window_hours=int(window_hours),
            include_reservations=bool(include_reservations),
            reservation_limit=int(reservation_limit),
        )
    except Exception as e:
        return {"error": str(e)}


# Serializes the unified-budget check with the reservation insert below.
# Without it two concurrent reserve calls can both pass the remaining-budget
# check and jointly exceed it (TOCTOU).
_spend_reserve_lock = threading.Lock()


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
        with _spend_reserve_lock:
            # DD1 / P2-011: gate against a LIVE unified budget (force_fresh
            # bypasses the 30s telemetry memo) so N reservations in one window
            # cannot pass against the same stale snapshot. The AUTHORITATIVE
            # cross-category rail is enforced inside reserve_spend's
            # BEGIN IMMEDIATE (below) — this pre-check only yields a friendly
            # early rejection.
            budget = _total_cost_budget_status(force_fresh=True)
            if "error" in budget:
                return budget
            remaining = int(budget.get("remaining_sats", 0) or 0)
            effective_budget_sats = int(budget.get("effective_budget_sats", 0) or 0)
            budget_window_hours = int(budget.get("window_hours", 24) or 24)
            budget_since = int(time.time()) - (budget_window_hours * 3600)
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
                effective_budget_sats=effective_budget_sats,
                since_timestamp=budget_since,
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


@plugin.method("revenue-spend-release-stale")
def revenue_spend_release_stale(
    plugin: Plugin,
    max_age_seconds: int = 3600,
    category: str = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Release stale generic spend reservations (safe recovery path for orphaned reservations)."""
    if database is None:
        return {"error": "Database not initialized"}
    try:
        result = database.release_spend_reservations(
            category=(None if not category else str(category).strip().lower()),
            older_than_seconds=max(1, int(max_age_seconds)),
            limit=max(1, int(limit)),
        )
        return {
            "status": "success",
            "released_count": int(result.get("released_count", 0) or 0),
            "released_sats": int(result.get("released_sats", 0) or 0),
            "reservation_ids": result.get("reservation_ids", []),
            "budget_after": _total_cost_budget_status(),
        }
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
                           peer_id: str = None, currency: str = None, routing_fee_limit_ppm: int = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().loop_out(
            amount_sats=amount_sats, address=address, channel_id=channel_id, peer_id=peer_id, currency=currency,
            routing_fee_limit_ppm=routing_fee_limit_ppm
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


@plugin.method("revenue-boltz-external-pay-ignores")
def revenue_boltz_external_pay_ignores(plugin: Plugin, action: str = "list", swap_id: str = None, note: str = None) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().manage_external_pay_ignores(action=action, swap_id=swap_id, note=note)
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
                           sat_per_vbyte: int = None, sweep: bool = False,
                           confirm_sweep: bool = False) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().withdraw(
            amount_sats=amount_sats, destination=destination, currency=currency,
            sat_per_vbyte=sat_per_vbyte, sweep=sweep, confirm_sweep=bool(confirm_sweep)
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
def revenue_boltz_backup(plugin: Plugin, include_mnemonic: bool = False) -> Dict[str, Any]:
    try:
        return _require_boltz_manager().backup(include_mnemonic=bool(include_mnemonic))
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
            if bool(sw.get("ignored_external_swap")):
                continue
            state = str(sw.get("state") or "").lower()
            status = str(sw.get("status") or "").lower()
            txt = f"{state} {status}"
            # Treat explicit error states as terminal even if status text remains "swap.created".
            done = any(x in txt for x in ("success", "completed", "claimed", "failed", "refunded", "expired", "cancel", "error"))
            active = any(x in txt for x in ("pending", "created", "mempool", "transaction", "lockup", "invoice", "claim"))
            if active and not done:
                pending += 1
        return pending
    except Exception as exc:
        plugin.log(f"boltz: pending swap count check failed, assuming 1 pending: {exc}", level="warn")
        return 1  # Fail closed: assume a swap is pending to prevent overlap


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
        if not hasattr(boltz_manager, "get_boltz_cost_components"):
            return {"source": "boltz", "spent_24h_sats": 0, "reserved_24h_sats": 0, "available": False}
        # Pass the unified budget cap explicitly instead of letting the Boltz
        # manager call back into the global budget provider (which evaluates
        # this very function — mutual recursion). In fixed mode the effective
        # unified budget equals the configured daily budget.
        global_cap_sats = None
        try:
            if config is not None:
                cfg = config.snapshot() if hasattr(config, "snapshot") else config
                global_cap_sats = max(0, int(getattr(cfg, "daily_budget_sats", 0) or 0))
        except Exception:
            global_cap_sats = None
        try:
            comps = boltz_manager.get_boltz_cost_components(
                window_hours=window_hours,
                global_budget_cap_sats=global_cap_sats,
            )
        except TypeError:
            # Older/stubbed managers without the explicit-cap parameter.
            comps = boltz_manager.get_boltz_cost_components(window_hours=window_hours)
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


_CANONICAL_TOTAL_COST_LEDGER_CATEGORIES = frozenset({"channel_open", "channel_close"})


def _normalize_generic_ledger_for_total_cost_budget(generic_ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Exclude canonical open/close spend events from the generic ledger budget bucket."""
    normalized = dict(generic_ledger or {})
    spent_by_category = normalized.get("spent_by_category")
    if not isinstance(spent_by_category, dict):
        spent_by_category = {}

    counted_spent_categories = {}
    excluded_spent_categories = {}
    for category, amount in spent_by_category.items():
        amount_int = int(amount or 0)
        if str(category) in _CANONICAL_TOTAL_COST_LEDGER_CATEGORIES:
            excluded_spent_categories[str(category)] = amount_int
        else:
            counted_spent_categories[str(category)] = amount_int

    normalized["raw_spent_24h_sats"] = int(normalized.get("spent_24h_sats", 0) or 0)
    normalized["spent_24h_sats"] = sum(counted_spent_categories.values())
    normalized["counted_spent_categories"] = counted_spent_categories
    normalized["excluded_spent_categories"] = excluded_spent_categories
    normalized.setdefault("event_count_by_category", {})
    normalized.setdefault("active_reservation_count_by_category", {})
    return normalized


def _open_close_cost_visibility(generic_ledger: Dict[str, Any], open_cost_sats: int, closure_cost_sats: int) -> Dict[str, Any]:
    excluded = generic_ledger.get("excluded_spent_categories", {}) if isinstance(generic_ledger, dict) else {}
    reserved = generic_ledger.get("reserved_by_category", {}) if isinstance(generic_ledger, dict) else {}
    event_counts = generic_ledger.get("event_count_by_category", {}) if isinstance(generic_ledger, dict) else {}
    reservation_counts = generic_ledger.get("active_reservation_count_by_category", {}) if isinstance(generic_ledger, dict) else {}
    if not isinstance(excluded, dict):
        excluded = {}
    if not isinstance(reserved, dict):
        reserved = {}
    if not isinstance(event_counts, dict):
        event_counts = {}
    if not isinstance(reservation_counts, dict):
        reservation_counts = {}

    pending_events = 0
    if int(open_cost_sats or 0) <= 0 and int(excluded.get("channel_open", 0) or 0) > 0:
        pending_events += max(1, int(event_counts.get("channel_open", 0) or 0))
    if int(closure_cost_sats or 0) <= 0 and int(excluded.get("channel_close", 0) or 0) > 0:
        pending_events += max(1, int(event_counts.get("channel_close", 0) or 0))
    pending_events += max(0, int(reservation_counts.get("channel_open", 0) or 0))
    pending_events += max(0, int(reservation_counts.get("channel_close", 0) or 0))

    return {
        "canonical_open_cost_available": int(open_cost_sats or 0) > 0,
        "canonical_close_cost_available": int(closure_cost_sats or 0) > 0,
        "pending_open_close_spend_events": pending_events,
        "excluded_from_generic_totals_to_avoid_double_count": True,
        "excluded_open_close_spend_sats": int(excluded.get("channel_open", 0) or 0) + int(excluded.get("channel_close", 0) or 0),
        "reserved_open_close_sats": int(reserved.get("channel_open", 0) or 0) + int(reserved.get("channel_close", 0) or 0),
    }


def _node_receivable_status() -> Dict[str, Any]:
    """Node-level inbound-capacity objective.

    scarcity is 0.0 at/above receivable_ratio_target, 1.0 at/below
    receivable_ratio_floor, linear in between. Errors neutralize to
    scarcity 0.0 so a telemetry failure can never enable spending.
    """
    safe = {"receivable_ratio": None, "scarcity": 0.0,
            "total_capacity_sats": 0, "total_receivable_sats": 0}
    if fee_controller is None or config is None:
        return safe
    try:
        channels = fee_controller._get_channels_info()
        total_cap = 0
        total_recv = 0
        for ch in channels.values():
            spend = max(0, parse_msat(ch.get("spendable_msat", 0)) // 1000)
            recv = max(0, parse_msat(ch.get("receivable_msat", 0)) // 1000)
            total_cap += spend + recv
            total_recv += recv
        if total_cap <= 0:
            return safe
        ratio = total_recv / total_cap
        floor = float(getattr(config, "receivable_ratio_floor", 0.20))
        target = float(getattr(config, "receivable_ratio_target", 0.30))
        if ratio >= target:
            scarcity = 0.0
        elif ratio <= floor:
            scarcity = 1.0
        else:
            scarcity = (target - ratio) / max(1e-9, target - floor)
        return {
            "receivable_ratio": round(ratio, 4),
            "scarcity": round(scarcity, 4),
            "total_capacity_sats": total_cap,
            "total_receivable_sats": total_recv,
        }
    except Exception:
        return safe


# Memoization for _total_cost_budget_status: the status is advisory (budgets are
# enforced atomically at reservation time) but expensive to compute (boltzcli
# subprocess + ~6 aggregate queries). One rebalance/boltz cycle evaluates it
# 4-8 times through the provider wiring, so a short TTL collapses those calls.
_TOTAL_COST_BUDGET_MEMO_TTL_SECONDS = 30.0
_total_cost_budget_memo: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_total_cost_budget_memo_lock = threading.Lock()
# Reentrancy guard (defense in depth): the unified status aggregates component
# providers that may, through misconfigured wiring, call back into the status.
_total_cost_budget_reentry = threading.local()


def _total_cost_budget_status(window_hours: Optional[int] = None,
                              force_fresh: bool = False) -> Dict[str, Any]:
    """Unified budget status across rebalances, Boltz swaps, and on-chain liquidity ops.

    Memoized per window_hours with a short TTL; guarded against re-entrant
    evaluation from cost-component providers (returns the last cached value or
    a minimal safe result instead of recursing).

    DD1 / P2-011: gating callers (the reservation path) MUST pass
    ``force_fresh=True`` so they never gate against the 30s-stale memo. The memo
    remains for non-gating telemetry reads (dashboards, status), which the fresh
    recompute still refreshes.
    """
    if config is None or database is None:
        return {"error": "Plugin not initialized"}

    wh = int(window_hours or 24)
    wh = max(1, min(168, wh))

    # Fast path: fresh memoized value for this window. deepcopy, not
    # dict(): a shallow copy shares the nested components/category dicts
    # with the memo, so a caller mutating its copy would poison every
    # cached read for the TTL window. The dict is small and the 30s TTL
    # makes the copy cost irrelevant. Skipped entirely when force_fresh.
    if not force_fresh:
        with _total_cost_budget_memo_lock:
            entry = _total_cost_budget_memo.get(wh)
            if entry is not None and (time.monotonic() - entry[0]) < _TOTAL_COST_BUDGET_MEMO_TTL_SECONDS:
                return copy.deepcopy(entry[1])

    # Re-entrancy guard: never recurse into a full recomputation. Prefer the
    # last known value for this window (even if stale); otherwise return a
    # minimal error result that all consumers already handle conservatively.
    if getattr(_total_cost_budget_reentry, "active", False):
        with _total_cost_budget_memo_lock:
            entry = _total_cost_budget_memo.get(wh)
            if entry is not None:
                return copy.deepcopy(entry[1])
        return {"error": "reentrant total-cost budget evaluation", "window_hours": wh}

    _total_cost_budget_reentry.active = True
    try:
        result = _compute_total_cost_budget_status(wh)
    finally:
        _total_cost_budget_reentry.active = False

    if isinstance(result, dict) and "error" not in result:
        with _total_cost_budget_memo_lock:
            # deepcopy: the freshly computed result is returned to the
            # caller, which must not share nested dicts with the memo.
            _total_cost_budget_memo[wh] = (time.monotonic(), copy.deepcopy(result))
    return result


def _compute_total_cost_budget_status(wh: int) -> Dict[str, Any]:
    cfg = config.snapshot() if hasattr(config, "snapshot") else config
    now = int(time.time())
    since = now - (wh * 3600)

    # Best-effort cleanup for generic spend reservations (e.g. channel open
    # reservations) so accepted actions that aren't explicitly settled do not block
    # budget for an entire window. Keep timeout bounded and independent from window size.
    try:
        stale_hours = max(1, int(getattr(cfg, "reservation_timeout_hours", 4) or 4))
        database.cleanup_stale_spend_reservations(max_age_seconds=stale_hours * 3600)
    except Exception as exc:
        plugin.log(f"cleanup_stale_spend_reservations failed: {exc}", level="debug")

    # Actual cost components (canonical data sources)
    rebalance = _rebalance_liquidity_cost_components(window_hours=wh)
    boltz = _boltz_liquidity_cost_components(window_hours=wh)
    if database:
        try:
            generic_ledger = database.get_spend_ledger_summary(window_hours=wh, include_reservations=True)
        except TypeError:
            generic_ledger = database.get_spend_ledger_summary(window_hours=wh)
    else:
        generic_ledger = {
            "spent_24h_sats": 0, "reserved_24h_sats": 0, "spent_by_category": {}, "reserved_by_category": {}
        }
    generic_ledger = _normalize_generic_ledger_for_total_cost_budget(generic_ledger)
    revenue_msat = int(database.get_total_routing_revenue(since)) if database else 0
    revenue_sats = revenue_msat // 1000
    open_cost_sats = int(database.get_opening_costs_since(since)) if database else 0
    closure_cost_sats = int(database.get_closure_costs_since(since)) if database else 0
    open_close_cost_visibility = _open_close_cost_visibility(
        generic_ledger,
        open_cost_sats,
        closure_cost_sats,
    )

    actual_by_category = {
        "rebalance": int(rebalance.get("spent_24h_sats", 0) or 0),
        "boltz": int(boltz.get("spent_24h_sats", 0) or 0),
        "open": open_cost_sats,
        "close": closure_cost_sats,
        "ledger": int(generic_ledger.get("spent_24h_sats", 0) or 0),
    }
    reserved_by_category = {
        "rebalance": int(rebalance.get("reserved_24h_sats", 0) or 0),
        "boltz": int(boltz.get("reserved_24h_sats", 0) or 0),
        "ledger": int(generic_ledger.get("reserved_24h_sats", 0) or 0),
    }

    actual_total = sum(max(0, int(v or 0)) for v in actual_by_category.values())
    reserved_total = sum(max(0, int(v or 0)) for v in reserved_by_category.values())

    daily_budget_sats = max(0, int(getattr(cfg, "daily_budget_sats", 0) or 0))
    effective_budget_sats = daily_budget_sats
    net_profit_sats = int(revenue_sats - actual_total)

    remaining_sats = max(0, int(effective_budget_sats) - actual_total - reserved_total)

    # Measured window coverage (honest, never an echo of the request):
    # hours between the oldest cost-evidence row and now, capped at the
    # window. When the measurement basis is absent or malformed, report an
    # honest unknown (covered_hours=null) rather than a fabricated
    # "complete". cl-hive's _ledger_window_coverage reads covered_hours/
    # coverage_hours numerically and treats null as unknown.
    covered_hours: Optional[float] = None
    coverage_status = "unknown"
    try:
        coverage = database.get_cost_evidence_coverage(window_hours=wh) if database else None
    except Exception as exc:
        coverage = None
        plugin.log(f"get_cost_evidence_coverage failed: {exc}", level="debug")
    if isinstance(coverage, dict):
        raw_covered = coverage.get("covered_hours")
        if isinstance(raw_covered, (int, float)) and not isinstance(raw_covered, bool):
            covered_hours = min(float(raw_covered), float(wh))
            if covered_hours == int(covered_hours):
                covered_hours = int(covered_hours)
            raw_status = coverage.get("coverage_status")
            if isinstance(raw_status, str) and raw_status:
                coverage_status = raw_status
            else:
                coverage_status = "complete" if covered_hours >= wh else "partial"

    return {
        "source": "total_cost_budget",
        "timestamp": now,
        "generated_at": now,
        "ttl_seconds": 1800,
        "window_hours": wh,
        "coverage_hours": covered_hours,
        "covered_hours": covered_hours,
        "coverage_status": coverage_status,
        "since_timestamp": since,
        "mode": "fixed",
        "daily_budget_sats": daily_budget_sats,
        "effective_budget_sats": int(effective_budget_sats),
        "revenue_sats": revenue_sats,
        "actual_spent_sats": actual_total,
        "reserved_sats": reserved_total,
        "remaining_sats": remaining_sats,
        "net_profit_sats_after_costs": net_profit_sats,
        "actual_spent_by_category": actual_by_category,
        "reserved_by_category": reserved_by_category,
        "open_close_cost_visibility": open_close_cost_visibility,
        "components": {
            "rebalance": rebalance,
            "boltz": boltz,
            "generic_ledger": generic_ledger,
            "open_cost_sats": open_cost_sats,
            "closure_cost_sats": closure_cost_sats,
        },
    }


def _total_cost_budget_limit_provider() -> Dict[str, Any]:
    # DD1 / P4-018: this provider feeds the GATING path of every autonomous
    # spender (rebalance capital controls, boltz gate, and the capacity-planner
    # open/close + defibrillation gates via _check_unified_budget /
    # _check_capital_controls). Gating callers must read the LIVE unified total,
    # not the 30s telemetry memo — otherwise N gate checks in one window admit
    # against the same stale snapshot (the P2-011 mechanism, uncovered on the
    # planner path). force_fresh; the authoritative rail remains the atomic
    # reserve, but the gate must not admit against a stale budget.
    status = _total_cost_budget_status(force_fresh=True)
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
    # DD1 / P4-014: this feeds the Boltz budget GATE (get_budget_status ->
    # _get_external_liquidity_costs). The P2-011 force_fresh fix only touched the
    # generic spend-reserve gate (cro:6719) and left this boltz-side component on
    # the 30s memo, so N boltz gate checks in one window read the same stale
    # non-boltz total → cross-category overspend. Read the LIVE total here; the
    # authoritative rail is the atomic reserve (reserve_boltz_swap_budget), but
    # the gate must not admit against a stale snapshot.
    status = _total_cost_budget_status(window_hours=window_hours, force_fresh=True)
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
    except Exception as e:
        plugin.log(f"Daily contribution estimate failed: {e}", level='debug')
        return 0.0


# F2: minimum gap (percentage points) between the tuned loop-in target and
# the tuned loop-out trigger. Without it, hot channels tuned to overlapping
# bands (eff_low_target above eff_high_trigger) and ping-ponged loop-in ->
# loop-out, burning swap fees in circles.
BAND_SEPARATION_MARGIN_PP = 5.0
# Maximum tuning movement of the band edges (must match the boost/cut math
# inside _boltz_dynamic_channel_tuning): trigger_boost <= 20pp on the loop-in
# trigger, out_adjust <= 15pp (10 hotness + 5 source) on the loop-out trigger.
MAX_LOW_TRIGGER_BOOST_PP = 20.0
MAX_HIGH_TRIGGER_CUT_PP = 15.0


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
    predicted_depletion_hours: Optional[float] = None,
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
    drain_accel_score = max(0.0, min(1.0, kalman_velocity / (0.05 / 24.0)))  # saturate at ~0.05/day (velocity is per-hour)

    # Anticipatory liquidity signal -- predicted depletion boosts urgency
    anticipatory_urgency = 0.0
    if predicted_depletion_hours is not None and predicted_depletion_hours > 0:
        # Saturate at 6h: anything <6h gets max urgency
        anticipatory_urgency = max(0.0, min(1.0, (6.0 - predicted_depletion_hours) / 6.0))

    # Low local balance also contributes to urgency before the static threshold is crossed.
    depletion_score = max(0.0, min(1.0, (60.0 - float(local_pct)) / 40.0))

    hotness_score = max(0.0, min(1.0, 0.55 * contrib_score + 0.45 * roi_score))
    drain_score = max(0.0, min(1.0,
        0.40 * source_signal +
        0.25 * drain_accel_score +
        0.20 * depletion_score +
        0.15 * anticipatory_urgency
    ))
    protection_score = max(0.0, min(1.0, 0.60 * hotness_score + 0.40 * drain_score))

    # Dynamic loop-in behavior: trigger earlier and refill deeper for high-score channels.
    trigger_boost = 20.0 * protection_score      # up to +20pp (40 -> 60)
    target_boost = 15.0 * protection_score       # up to +15pp (55 -> 70)
    amount_multiplier = 1.0 + (2.0 * protection_score)  # up to 3x amount cap
    cooldown_multiplier = 1.0 - (0.75 * protection_score)  # down to 25% of base cooldown

    eff_low_trigger = min(70.0, max(float(low_trigger_pct), float(low_trigger_pct) + trigger_boost))
    # keep target at least 10pp above trigger, bounded for safety
    eff_low_target = min(85.0, max(float(low_target_pct), eff_low_trigger + 10.0, float(low_target_pct) + target_boost))

    # Loop-out: primary Boltz action (generates on-chain funds + rebalances).
    # Profitable source channels get a lower trigger to enable more loop-out opportunities.
    # source_signal adds up to 5pp extra for channels with natural local refill.
    out_adjust = 10.0 * max(0.0, min(1.0, hotness_score)) + 5.0 * source_signal
    eff_high_trigger = max(55.0, min(float(high_trigger_pct), float(high_trigger_pct) - out_adjust))
    eff_high_target = min(float(high_target_pct) + out_adjust * 0.5, float(high_target_pct) + 10.0)

    # F2: enforce band ordering AFTER tuning. The loop-in target must sit at
    # least BAND_SEPARATION_MARGIN_PP below the loop-out trigger, and the
    # targets must stay strictly inside their triggers, or a hot channel
    # ping-pongs loop-in <-> loop-out and burns swap fees in circles.
    band_clamped = False
    if eff_low_target > eff_high_trigger - BAND_SEPARATION_MARGIN_PP:
        band_clamped = True
        # Reduce the loop-in target first, keeping it strictly above the
        # loop-in trigger; if that cannot create the gap, raise the loop-out
        # trigger instead.
        low_target_floor = eff_low_trigger + 1.0
        eff_low_target = max(low_target_floor, eff_high_trigger - BAND_SEPARATION_MARGIN_PP)
        if eff_low_target > eff_high_trigger - BAND_SEPARATION_MARGIN_PP:
            eff_high_trigger = eff_low_target + BAND_SEPARATION_MARGIN_PP
    if eff_high_target >= eff_high_trigger:
        band_clamped = True
        eff_high_target = eff_high_trigger - 1.0
    if eff_low_trigger >= eff_low_target:
        band_clamped = True
        eff_low_trigger = max(0.0, eff_low_target - 1.0)

    return {
        'band_clamped': band_clamped,
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
            'anticipatory_urgency': round(anticipatory_urgency, 4),
            'predicted_depletion_hours': predicted_depletion_hours,
            'kalman_flow_ratio': round(kalman_ratio, 4),
            'kalman_velocity': round(kalman_velocity, 6),
        },
    }


def _get_confirmed_onchain_sats() -> Optional[int]:
    """Return confirmed on-chain wallet outputs in sats from CLN listfunds.

    F9 fail-safe: returns None on RPC failure. Returning 0 made a telemetry
    failure look like a full reserve deficit and engaged treasury harvesting
    — callers must treat None as 'unknown, skip treasury this cycle'
    (mirroring the scarcity-neutralizes-on-error convention).
    """
    try:
        lf = data_service.get_funds() if data_service else safe_plugin.rpc.listfunds()
    except Exception as e:
        plugin.log(f"listfunds RPC failed for onchain balance: {e}", level='warn')
        return None
    outputs = lf.get("outputs", []) if isinstance(lf, dict) else []
    total = 0
    for o in outputs:
        if str(o.get("status") or "") != "confirmed":
            continue
        total += _parse_msat(o.get("amount_msat", 0)) // 1000
    return int(total)


def _filter_boltz_treasury_recommendations(
    plan: Dict[str, Any],
    *,
    deficit_sats: int,
    exclude_protected: bool = True,
    min_source_local_pct: Optional[float] = None,
    min_amount_sats: int = 100_000,
) -> Dict[str, Any]:
    """Filter a balance plan down to treasury-appropriate reverse swaps."""
    recs = list(plan.get("recommendations", [])) if isinstance(plan, dict) else []
    filtered = []
    skipped = []
    remaining_target = max(0, int(deficit_sats))
    for rec in recs:
        direction = str(rec.get("direction") or "")
        if direction != "loop_out":
            skipped.append({"channel_id": rec.get("channel_id"), "reason": "not_loop_out"})
            continue
        # F6: hard floor on source local balance. The treasury plan passes
        # min_source_local_pct as the balance plan's high_trigger_pct, but
        # dynamic tuning may lower the effective trigger by up to 15pp —
        # letting under-floor channels into the plan. Enforce the floor here
        # regardless of tuning.
        if min_source_local_pct is not None:
            local_pct = float(rec.get("local_balance_pct", 0.0) or 0.0)
            if local_pct < float(min_source_local_pct):
                skipped.append({
                    "channel_id": rec.get("channel_id"),
                    "peer_id": rec.get("peer_id"),
                    "reason": "below_min_source_local_pct",
                    "local_balance_pct": local_pct,
                    "min_source_local_pct": float(min_source_local_pct),
                })
                continue
        tuning = rec.get("dynamic_tuning", {}) if isinstance(rec.get("dynamic_tuning"), dict) else {}
        hints = rec.get("execution_hints", {}) if isinstance(rec.get("execution_hints"), dict) else {}
        if exclude_protected and (bool(hints.get("prioritize_channel_protection")) or float(tuning.get("protection_score", 0.0) or 0.0) >= 0.6):
            skipped.append({
                "channel_id": rec.get("channel_id"),
                "peer_id": rec.get("peer_id"),
                "reason": "protected_or_hot_channel",
                "protection_score": tuning.get("protection_score"),
            })
            continue
        amt = int(rec.get("amount_sats", 0) or 0)
        if remaining_target > 0 and amt > remaining_target:
            # F10: enforce the target cap at plan time. The annotation alone
            # was never honored by the executor, which swapped the full
            # amount and overshot the treasury target.
            if remaining_target < int(min_amount_sats):
                skipped.append({
                    "channel_id": rec.get("channel_id"),
                    "peer_id": rec.get("peer_id"),
                    "reason": "remaining_target_below_min_amount",
                    "remaining_target_sats": int(remaining_target),
                    "min_amount_sats": int(min_amount_sats),
                })
                continue
            rec = dict(rec)
            rec["amount_sats"] = int(remaining_target)
            rec["treasury_target_cap_sats"] = int(remaining_target)
            rec["treasury_amount_exceeds_deficit"] = True
            rec["treasury_amount_clamped"] = True
        filtered.append(rec)
    out = dict(plan)
    out["recommendations"] = filtered
    out["total_candidates"] = len(filtered)
    examples = list(plan.get("skipped_examples", [])) if isinstance(plan.get("skipped_examples"), list) else []
    out["skipped_examples"] = (examples + skipped)[:20]
    out["skipped_count"] = int(plan.get("skipped_count", 0) or 0) + len(skipped)
    return out


def _build_boltz_expansion_treasury_plan(
    *,
    onchain_target_sats: int,
    min_deficit_sats: int = 250_000,
    preferred_currency: str = "BTC",
    max_actions: int = 1,
    min_source_local_pct: float = 80.0,
    exclude_protected: bool = True,
    require_profitable: bool = True,
    min_marginal_roi: float = 0.0,
    profit_margin_factor: float = 1.2,
    expected_horizon_days: float = 3.0,
    max_amount_sats: int = 1_500_000,
    min_amount_sats: int = 100_000,
    planner_coordination: Optional[Dict[str, Any]] = None,
    pending_swap_count: Optional[int] = None,
) -> Dict[str, Any]:
    # Fail fast when Boltz is unavailable; the budget/pending reads moved to
    # the post-deficit-check path (F4), so only the guard remains here.
    _require_boltz_manager()
    onchain_confirmed_sats = _get_confirmed_onchain_sats()
    target = max(0, int(onchain_target_sats))
    if onchain_confirmed_sats is None:
        # F9: unknown on-chain balance (listfunds failed) must not read as a
        # full deficit. Report the plan unavailable so the auto-cycle skips
        # treasury mode this cycle instead of harvesting on bad telemetry.
        return {
            "generated_at": int(time.time()),
            "treasury": {
                "enabled": bool(getattr(config, "expansion_treasury_enabled", False)) if config else False,
                "onchain_confirmed_sats": None,
                "onchain_target_sats": target,
                "deficit_sats": 0,
                "min_deficit_sats": int(min_deficit_sats),
                "preferred_currency": str(preferred_currency).upper(),
                "exclude_protected": bool(exclude_protected),
                "max_actions": int(max_actions),
                "min_source_local_pct": float(min_source_local_pct),
            },
            "status": "unavailable",
            "reason": "onchain_balance_unavailable_rpc_error",
            "recommendations": [],
            "total_candidates": 0,
            "skipped_count": 0,
            "skipped_examples": [],
        }
    deficit = max(0, target - onchain_confirmed_sats)

    treasury = {
        "enabled": bool(getattr(config, "expansion_treasury_enabled", False)) if config else False,
        "onchain_confirmed_sats": onchain_confirmed_sats,
        "onchain_target_sats": target,
        "deficit_sats": deficit,
        "min_deficit_sats": int(min_deficit_sats),
        "preferred_currency": str(preferred_currency).upper(),
        "exclude_protected": bool(exclude_protected),
        "max_actions": int(max_actions),
        "min_source_local_pct": float(min_source_local_pct),
    }

    if deficit < int(min_deficit_sats):
        # Deliberately NO bm.budget() / pending-swap lookup here: both are
        # boltzcli subprocess calls and nothing consumes them on an
        # at_target plan — the treasury cycle and the auto-cycle mode
        # selector early-return on status alone. The caller's count is
        # echoed when provided; otherwise None means "not evaluated".
        return {
            "generated_at": int(time.time()),
            "treasury": treasury,
            "status": "at_target",
            "reason": "onchain_reserve_deficit_below_minimum",
            "recommendations": [],
            "total_candidates": 0,
            "skipped_count": 0,
            "skipped_examples": [],
            "budget": None,
            "pending_swap_count": (
                int(pending_swap_count) if pending_swap_count is not None else None
            ),
        }

    max_amt = max(int(min_amount_sats), min(int(max_amount_sats), int(max(deficit, min_amount_sats))))
    base_plan = _build_boltz_balance_plan(
        low_trigger_pct=0.0,
        low_target_pct=50.0,
        high_trigger_pct=float(min_source_local_pct),
        high_target_pct=max(50.0, float(min_source_local_pct) - 20.0),
        min_amount_sats=int(min_amount_sats),
        max_amount_sats=max_amt,
        max_candidates=max(10, int(max_actions) * 8),
        require_profitable=bool(require_profitable),
        min_marginal_roi=float(min_marginal_roi),
        profit_margin_factor=float(profit_margin_factor),
        expected_horizon_days=float(expected_horizon_days),
        loop_out_currency=str(preferred_currency).upper(),
        loop_in_currency="LBTC",
        planner_coordination=planner_coordination,
        pending_swap_count=pending_swap_count,
    )
    if "error" in base_plan:
        base_plan["treasury"] = treasury
        return base_plan
    filtered = _filter_boltz_treasury_recommendations(
        base_plan,
        deficit_sats=deficit,
        exclude_protected=bool(exclude_protected),
        min_source_local_pct=float(min_source_local_pct),
        min_amount_sats=int(min_amount_sats),
    )
    filtered["treasury"] = treasury
    filtered["status"] = "ok"
    return filtered


# Minimum settled outbound volume (sats) in the 7d window before the node's
# realized fee ppm is trusted to price the structural inbound credit. Below
# this floor a handful of lucky high-fee forwards would set a wildly
# unrepresentative rate, so the credit is disabled instead.
STRUCTURAL_MIN_VOLUME_SATS = 1_000_000

# Final-stage discount applied when the structural credit is priced from our
# OWN current out-fee policies instead of realized forwards: our asks are
# optimistic — they have never cleared at volume (if they had, a realized
# window would have priced the credit instead), so take them at half value.
OWN_POLICY_HAIRCUT = 0.5

# How many ~3-day turnover horizons of value the structural inbound credit
# amortizes over (10 ≈ 30 days of the structural asset working). One horizon
# priced only ~0.5x amount of freed-inbound turnover, while real reverse-swap
# costs run ~1000-2500 ppm — the profit guard could essentially never flip.
# The freed inbound keeps earning beyond one horizon, and actual spend is
# bounded by the daily structural envelope, not by this estimate; the credit
# prices a month of the structural asset.
STRUCTURAL_AMORTIZATION_HORIZONS = 10

# Quote-subprocess budget for _build_boltz_balance_plan: only the top
# QUOTE_CANDIDATE_MULTIPLIER x max_candidates trigger-passing candidates
# (ranked by severity, then estimated daily contribution) pay a boltzcli
# quote; the rest are skipped as quote_budget_exceeded. The margin above
# max_candidates is deliberate — the profit guard needs the quote, so some
# quoted candidates fail post-quote and a 1x budget could starve the final
# slice; 3x bounds the worst case while leaving room for guard failures and
# the post-quote multi-goal re-ranking.
QUOTE_CANDIDATE_MULTIPLIER = 3


def _policy_derived_fee_ppm(channels: Dict[str, Dict[str, Any]]) -> int:
    """Capacity-weighted mean of our own out-fee policies, haircut applied.

    Last stage of the structural-credit pricing cascade: an inbound-starved
    node may have NO realized volume in any window (it cannot route — that
    is exactly why it needs the drain), leaving only our advertised fee
    policies as a price signal. Channels with a zero policy still count in
    the denominator: a zero ask IS the policy.
    """
    total_cap = 0
    weighted = 0.0
    for ch in channels.values():
        cap = int(ch.get("capacity", 0) or 0)
        if cap <= 0:
            continue
        ppm = int(ch.get("fee_proportional_millionths", 0) or 0)
        total_cap += cap
        weighted += cap * max(0, ppm)
    if total_cap <= 0:
        return 0
    return int((weighted / total_cap) * OWN_POLICY_HAIRCUT)


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
    planner_coordination: Optional[Dict[str, Any]] = None,
    pending_swap_count: Optional[int] = None,
) -> Dict[str, Any]:
    if fee_controller is None or database is None:
        return {"error": "Plugin not initialized"}

    # F2: reject caller bases that dynamic tuning could push into overlap.
    # Tuning may raise the loop-in trigger by up to MAX_LOW_TRIGGER_BOOST_PP
    # and lower the loop-out trigger by up to MAX_HIGH_TRIGGER_CUT_PP; bases
    # that collide under maximum tuning would ping-pong loop-in/loop-out.
    if float(low_trigger_pct) + MAX_LOW_TRIGGER_BOOST_PP >= float(high_trigger_pct) - MAX_HIGH_TRIGGER_CUT_PP:
        return {
            "error": (
                f"invalid trigger bands: low_trigger_pct={float(low_trigger_pct)} + "
                f"max tuning boost ({MAX_LOW_TRIGGER_BOOST_PP}pp) overlaps "
                f"high_trigger_pct={float(high_trigger_pct)} - max tuning cut "
                f"({MAX_HIGH_TRIGGER_CUT_PP}pp); widen the trigger band"
            )
        }

    bm = _require_boltz_manager()

    channels = fee_controller._get_channels_info()
    if not channels:
        return {"error": "No normal channels available"}

    # Planner + rebalancer coordination: avoid loser channels, factor in funding needs
    planner_coord = planner_coordination if isinstance(planner_coordination, dict) else {}
    planner_loser_scids = set(planner_coord.get("loser_scids", set()))
    planner_funding_deficit = int(planner_coord.get("funding_deficit_sats", 0) or 0)
    rebalancer_exhausted = bool(planner_coord.get("rebalancer_exhausted", False))

    # Channels the circular rebalancer reported as unplaceable this cycle.
    # Snapshot from the last rebalance cycle: a channel drained since then
    # simply won't pass the loop-out trigger below (live balances re-read),
    # so staleness is self-correcting.
    drain_demand_scids: set = set()
    try:
        engine = getattr(rebalancer, "rebalance_engine_v2", None) if rebalancer else None
        demand = engine.get_drain_demand() if engine is not None and hasattr(engine, "get_drain_demand") else None
        if demand is not None:
            drain_demand_scids = {
                str(e.channel_id).replace(":", "x") for e in demand.entries
            }
    except Exception:
        drain_demand_scids = set()

    # Route-pair awareness: protect inbound legs of top revenue routes from loop-out drain
    route_pair_in_channels = set()
    try:
        pairs = database.get_top_route_pairs(days=30, min_forwards=3, limit=10)
        for p in pairs:
            in_ch = str(p.get("in_channel", "")).replace(":", "x")
            if in_ch:
                route_pair_in_channels.add(in_ch)
    except Exception:
        pass

    # Refresh profitability cache on demand.
    if profitability_analyzer is not None:
        try:
            profitability_analyzer.analyze_all_channels()
        except Exception as e:
            plugin.log(f"BOLTZ_BALANCE: profitability refresh failed: {e}", level='warn')

    state_rows = {str(r.get("channel_id")): r for r in database.get_all_channel_states()}

    budget_status = bm.budget()
    # Reuse the caller's pending-swap count when provided (it is a boltzcli
    # subprocess call) — e.g. the balance cycle checks it before plan build.
    pending_swaps = int(pending_swap_count) if pending_swap_count is not None else _boltz_pending_swap_count()

    # Structural inbound-credit inputs, hoisted to ONE evaluation per plan
    # build (the candidate loop runs for up to ~50-100 channels and must not
    # re-query node receivable status or the realized fee rate per channel).
    # Both default to neutral (0) on error or when the envelope is disabled.
    structural_budget_sats = int(getattr(config, "boltz_structural_budget_sats_per_day", 0) or 0)
    structural_scarcity = 0.0
    structural_realized_ppm = 0
    structural_credit_source: Optional[str] = None
    if structural_budget_sats > 0:
        try:
            structural_scarcity = float(_node_receivable_status().get("scarcity", 0.0) or 0.0)
        except Exception:
            structural_scarcity = 0.0
        if structural_scarcity > 0.0:
            # Pricing cascade — each stage only runs when the prior returned
            # 0. The volume floor means an inbound-starved node frequently
            # has NO qualifying realized window (it cannot route — that is
            # exactly why it needs the drain), so the credit must not die
            # with the 7d query (chicken-and-egg).
            #   (a) 7d realized rate, 1M-sat floor (raw forwards);
            #   (b) 30d realized rate, 1M-sat floor (daily rollups + raw);
            #   (c) capacity-weighted mean of our OWN out-fee policies,
            #       discounted by OWN_POLICY_HAIRCUT (asks that never
            #       cleared at volume are optimistic).
            try:
                structural_realized_ppm = int(database.get_node_realized_fee_ppm(
                    window_days=7, min_volume_sats=STRUCTURAL_MIN_VOLUME_SATS) or 0)
            except Exception:
                structural_realized_ppm = 0
            if structural_realized_ppm > 0:
                structural_credit_source = "realized_7d"
            else:
                try:
                    structural_realized_ppm = int(database.get_node_realized_fee_ppm_30d(
                        min_volume_sats=STRUCTURAL_MIN_VOLUME_SATS) or 0)
                except Exception:
                    structural_realized_ppm = 0
                if structural_realized_ppm > 0:
                    structural_credit_source = "realized_30d"
                else:
                    try:
                        structural_realized_ppm = _policy_derived_fee_ppm(channels)
                    except Exception:
                        structural_realized_ppm = 0
                    if structural_realized_ppm > 0:
                        structural_credit_source = "policy_derived"

    # 'auto' currency must be resolved at PLAN time: _norm_currency passes
    # unknown strings through uppercased, so an unresolved 'auto' would reach
    # the boltzcli quote subprocess as a literal "AUTO" currency argument.
    # Resolution (a BTC-vs-LBTC quote comparison) runs lazily, at most once
    # per direction per plan build, sized by the first triggered candidate's
    # amount; resolution failure falls back to BTC.
    _resolved_auto_currency: Dict[str, Optional[str]] = {}

    def _plan_currency(direction: str, configured: str, amount_sats: int) -> str:
        if str(configured or "").strip().lower() != "auto":
            return configured
        if direction not in _resolved_auto_currency:
            try:
                _resolved_auto_currency[direction] = _select_boltz_currency(direction, int(amount_sats))
            except Exception:
                _resolved_auto_currency[direction] = "BTC"
        return _resolved_auto_currency[direction] or "BTC"

    def _annotated_currency(direction: str, configured: str) -> Optional[str]:
        """Concrete currency this plan's quotes used for `direction`.

        None when 'auto' was configured but no candidate in that direction
        ever triggered (resolution never ran).
        """
        if str(configured or "").strip().lower() == "auto":
            return _resolved_auto_currency.get(direction)
        return str(configured).upper()

    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    # Phase 1 output: quote-ready candidates (passed triggers, policy, and
    # amount gates) that have NOT paid a quote subprocess yet.
    pending_quotes: List[Dict[str, Any]] = []

    for channel_id, ch in channels.items():
        peer_id = str(ch.get("peer_id") or "")
        if only_channel_id and str(channel_id).replace(':', 'x') != str(only_channel_id).replace(':', 'x'):
            continue
        if only_peer_id and peer_id != only_peer_id:
            continue

        # Skip channels marked for closure by capacity planner (avoid wasting swap fees)
        scid_display = str(channel_id).replace(':', 'x')
        if planner_loser_scids and scid_display in planner_loser_scids:
            skipped.append({"channel_id": channel_id, "peer_id": peer_id, "reason": "planner_loser_close_pending"})
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

        # --- Enrichment: DTS posterior, rebalance feasibility, hive hints ---
        dts_summary = None
        if fee_controller is not None:
            try:
                dts_summary = fee_controller.get_dts_summary(channel_id)
            except Exception:
                pass
        broadcast_fee_ppm = int((dts_summary or {}).get("broadcast_fee_ppm", 0) or 0)
        posterior_mean = float((dts_summary or {}).get("posterior_mean", 0) or 0)
        posterior_std = float((dts_summary or {}).get("posterior_std", 200) or 200)

        # Predicted depletion hours from Kalman flow state.
        predicted_depletion_hours = None
        kalman_velocity = float(state_row.get('kalman_velocity', 0.0) or 0.0)
        kalman_ratio = float(state_row.get('kalman_flow_ratio', 0.0) or 0.0)
        # Source channels (positive ratio = draining local): estimate hours
        # until depleted. Gate lowered from 0.1 to 0.02 so mild but
        # persistent drains are anticipated too.
        if kalman_ratio > 0.02 and local_sats > 0 and capacity_sats > 0:
            # Preferred: flow_analysis.estimate_depletion_hours, the pure
            # helper with correct day-fraction units (kalman_ratio is net
            # flow per day as a fraction of capacity). The old in-plan
            # contrib/fee proxy below was ~7-8x off.
            _estimate_depletion = getattr(flow_analysis_mod, "estimate_depletion_hours", None)
            if callable(_estimate_depletion):
                try:
                    predicted_depletion_hours = _estimate_depletion(
                        local_sats, capacity_sats, kalman_ratio, kalman_velocity
                    )
                except Exception:
                    predicted_depletion_hours = None
            elif daily_contrib_est > 0:
                # Legacy proxy (helper absent): back volume out of revenue.
                # revenue = volume * fee_ppm / 1e6 => volume ≈ revenue * 1e6 / fee_ppm
                est_fee = max(broadcast_fee_ppm, 50)  # floor at 50 to avoid division issues
                daily_volume_est = daily_contrib_est * 1_000_000 / est_fee
                net_drain_sats_per_day = daily_volume_est * min(1.0, kalman_ratio)
                if net_drain_sats_per_day > 0:
                    predicted_depletion_hours = (local_sats / net_drain_sats_per_day) * 24.0

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
            predicted_depletion_hours=predicted_depletion_hours,
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

        # --- Post-trigger enrichment (deliberately AFTER direction selection so
        # balanced channels never pay these per-channel DB/hint lookups) ---

        # Rebalance feasibility: if native rebalancer can't rebalance this channel, Boltz is the only option
        rebal_success = None
        rebalance_impossible = False
        if database is not None:
            try:
                rebal_success = database.get_channel_rebalance_success_rate(channel_id, window_days=7)
            except Exception:
                pass
        if rebal_success is not None:
            # 3+ attempts with 0% success in last 7 days = rebalancer can't do it
            if rebal_success.get("total", 0) >= 3 and rebal_success.get("success_rate", 1.0) == 0.0:
                rebalance_impossible = True

        # Hive hints: rebalance preference and peer quality (via bounded bias)
        hive_rebal_bias = 1.0
        if hive_hints is not None:
            try:
                hive_rebal_bias = hive_hints.get_rebalance_bias(peer_id)
            except Exception:
                pass

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
        amount_sats = 0
        if raw_amount > 0:
            amount_sats = min(int(dynamic_amount_cap), int(raw_amount))
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

        # Quote deferred to phase 2: only the best-ranked candidates pay the
        # boltzcli quote subprocess. Capture everything phase 2 needs.
        pending_quotes.append({
            "channel_id": channel_id,
            "peer_id": peer_id,
            "direction": direction,
            "target_currency": target_currency,
            "target_pct": target_pct,
            "severity": severity,
            "amount_sats": amount_sats,
            "raw_amount": raw_amount,
            "dynamic_amount_cap": dynamic_amount_cap,
            "daily_contrib_est": daily_contrib_est,
            "marginal_roi": marginal_roi,
            "prof": prof,
            "prof_class": prof_class,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "broadcast_fee_ppm": broadcast_fee_ppm,
            "rebalance_impossible": rebalance_impossible,
            "hive_rebal_bias": hive_rebal_bias,
            "flow_state": flow_state,
            "local_pct": local_pct,
            "scid_display": scid_display,
            "tuning": tuning,
            "hints": hints,
            "eff_low_trigger_pct": eff_low_trigger_pct,
            "eff_high_trigger_pct": eff_high_trigger_pct,
            "capacity_sats": capacity_sats,
            "local_sats": local_sats,
            "receivable_sats": receivable_sats,
            "policy_reason": policy_reason,
            "predicted_depletion_hours": predicted_depletion_hours,
        })

    # --- Phase 2: bounded quoting. Every quote is a boltzcli subprocess, but
    # only max_candidates recommendations survive the final slice — so rank
    # quote-ready candidates by (severity, daily_contrib_est) and quote only
    # the top QUOTE_CANDIDATE_MULTIPLIER x max_candidates. The multiplier
    # margin matters: the profit guard needs the quote, so some quoted
    # candidates fail post-quote and a 1x budget could starve the slice.
    quote_budget = max(0, int(max_candidates)) * QUOTE_CANDIDATE_MULTIPLIER
    pending_quotes.sort(
        key=lambda p: (float(p["severity"]), int(p["daily_contrib_est"])),
        reverse=True,
    )
    for p in pending_quotes[quote_budget:]:
        skipped.append({
            "channel_id": p["channel_id"],
            "peer_id": p["peer_id"],
            "direction": p["direction"],
            "reason": "quote_budget_exceeded",
            "severity": round(float(p["severity"]), 4),
            "quote_budget": quote_budget,
        })

    for p in pending_quotes[:quote_budget]:
        channel_id = p["channel_id"]
        peer_id = p["peer_id"]
        direction = p["direction"]
        target_currency = p["target_currency"]
        target_pct = p["target_pct"]
        severity = p["severity"]
        amount_sats = p["amount_sats"]
        raw_amount = p["raw_amount"]
        dynamic_amount_cap = p["dynamic_amount_cap"]
        daily_contrib_est = p["daily_contrib_est"]
        marginal_roi = p["marginal_roi"]
        prof = p["prof"]
        prof_class = p["prof_class"]
        posterior_mean = p["posterior_mean"]
        posterior_std = p["posterior_std"]
        broadcast_fee_ppm = p["broadcast_fee_ppm"]
        rebalance_impossible = p["rebalance_impossible"]
        hive_rebal_bias = p["hive_rebal_bias"]
        flow_state = p["flow_state"]
        local_pct = p["local_pct"]
        scid_display = p["scid_display"]
        tuning = p["tuning"]
        hints = p["hints"]
        eff_low_trigger_pct = p["eff_low_trigger_pct"]
        eff_high_trigger_pct = p["eff_high_trigger_pct"]
        capacity_sats = p["capacity_sats"]
        local_sats = p["local_sats"]
        receivable_sats = p["receivable_sats"]
        policy_reason = p["policy_reason"]
        predicted_depletion_hours = p["predicted_depletion_hours"]

        target_currency = _plan_currency(direction, target_currency, amount_sats)
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
        # F4 amount-proportionality: the heuristic estimates the value of
        # closing the FULL gap (raw_amount); a cap-bound partial swap only
        # earns the pro-rata share of it.
        amount_fraction = min(1.0, amount_sats / max(1, raw_amount))
        expected_gross_uplift_sats = int(
            max(0.0, daily_contrib_est) * float(expected_horizon_days) * severity_factor * amount_fraction
        )

        # DTS posterior uplift: if the channel has a proven high fee (tight posterior),
        # use the posterior mean to estimate potential revenue even if recent history is thin.
        dts_uplift_sats = 0
        if posterior_mean > 0 and posterior_std < 100:
            # Tight posterior = confident fee estimate; project revenue from rebalanced capacity
            # Conservative: use 50% of the amount as volume estimate over the horizon.
            # F3b horizon honesty: the 0.5 turnover IS the per-horizon volume
            # assumption, so there is no extra xhorizon_days multiplier (it
            # used to inflate the projection into 150% turnover at 3 days).
            projected_volume = amount_sats * 0.5
            dts_uplift_sats = int(projected_volume * posterior_mean / 1_000_000)
        # Use the better of historical and DTS-projected uplift
        expected_gross_uplift_sats = max(expected_gross_uplift_sats, dts_uplift_sats)

        required_profit_threshold_sats = int(round(estimated_fee_sats * float(profit_margin_factor)))

        # Rebalance-impossible channels get a relaxed profit threshold: when native rebalancing
        # can't work (all sources have negative spread), Boltz is the only option.
        # rebalancer_exhausted: system-wide signal (depleted channels exist, 0 profitable rebalances)
        effective_profit_margin = float(profit_margin_factor)
        if rebalance_impossible:
            effective_profit_margin = max(0.5, effective_profit_margin * 0.6)
        elif rebalancer_exhausted and direction == "loop_in":
            # Rebalancer is stuck globally — Boltz loop-ins should be more willing
            effective_profit_margin = max(0.8, effective_profit_margin * 0.8)
        if effective_profit_margin != float(profit_margin_factor):
            required_profit_threshold_sats = int(round(estimated_fee_sats * effective_profit_margin))

        # Structural inbound credit: a receivable-starved node values the
        # inbound capacity a loop-out creates, priced at the node's realized
        # earn rate. Zero when the node is healthy or the structural
        # envelope is disabled — the per-channel guard is then unchanged.
        # NOTE: this is deliberately a Boltz-only concept. The circular
        # rebalancer must never use it: rebalancing conserves the aggregate
        # ratio, so inbound scarcity is not actionable there.
        structural_uplift_sats = 0
        if (
            direction == "loop_out"
            and structural_budget_sats > 0
            and structural_scarcity > 0.0
            and structural_realized_ppm > 0
        ):
            # F3 marginal credit: when DTS already prices this channel's own
            # volume (tight posterior), expected_gross already contains that
            # revenue via dts_uplift — crediting the full realized rate again
            # would double-pay for the same freed volume. Structural therefore
            # only credits the NODE-LEVEL PREMIUM over the channel's own rate.
            # With a loose posterior (std >= 100) the channel rate is unknown
            # and the full realized rate applies.
            channel_posterior_ppm = (
                float(posterior_mean) if (posterior_mean > 0 and posterior_std < 100) else 0.0
            )
            premium_ppm = max(0.0, float(structural_realized_ppm) - channel_posterior_ppm)
            # Mirror the conservative DTS-uplift volume model: half the freed
            # inbound turns over per horizon — then amortize over
            # STRUCTURAL_AMORTIZATION_HORIZONS, because the freed inbound
            # keeps working long past one ~3-day horizon. Spend is bounded
            # by the daily structural envelope, not this estimate; the
            # credit prices a month of the structural asset.
            projected_volume = (
                amount_sats * 0.5 * structural_scarcity
                * STRUCTURAL_AMORTIZATION_HORIZONS
            )
            structural_uplift_sats = int(projected_volume * premium_ppm / 1_000_000)
        passes_without_credit = expected_gross_uplift_sats >= required_profit_threshold_sats
        expected_gross_uplift_sats += structural_uplift_sats

        passes_profit_guard = (expected_gross_uplift_sats >= required_profit_threshold_sats) if require_profitable else True
        expected_net_sats = expected_gross_uplift_sats - estimated_fee_sats

        # Apply hive rebalance bias to the net estimate (±15% bounded)
        if hive_rebal_bias != 1.0:
            expected_net_sats = int(expected_net_sats * hive_rebal_bias)

        # Additional guard for loop-in with no channel pinning support.
        non_pinned_penalty = 0.7 if direction == "loop_in" else 1.0
        risk_adjusted_net_sats = int(expected_net_sats * non_pinned_penalty)
        if require_profitable and direction == "loop_in":
            # Make loop-in more conservative because it cannot be channel-pinned with current boltzcli.
            passes_profit_guard = passes_profit_guard and (risk_adjusted_net_sats > 0)

        # Multi-goal value for loop-outs: one swap achieves both on-chain fund generation
        # AND channel rebalancing.  Higher score = more beneficial to drain this channel.
        #   - excess_ratio: how much local balance exceeds 50% (0-1)
        #   - profitability: marginal ROI signal (0-1)
        #   - fee_value: broadcast fee indicating revenue potential from freed remote cap
        #   - flow_bonus: source channels naturally refill local, so draining is safe
        #   - rebalance_bonus: if native rebalancer can't rebalance, Boltz is the only option
        #   - planner_bonus: capacity planner needs on-chain funds for a channel open
        #   - route_bonus: loop-out through inbound revenue legs creates headroom for more inbound traffic
        #   - hive_bonus: hive rebalance bias compounds with route-pair signal
        #   - drain_bonus: channels the circular rebalancer reported unplaceable drain first
        multi_goal_value = 0.0
        if direction == "loop_out":
            excess_ratio = max(0.0, min(1.0, (local_pct - 50.0) / 50.0))
            # F7: marginal_roi is a FRACTION (0.25 = 25%), matching the
            # dynamic-tuning roi_score above; full signal at 25% marginal
            # ROI. The old /25.0 treated it as a percent, capping the signal
            # at <=0.08 and deadening 35% of the loop-out score.
            roi_signal = max(0.0, min(1.0, (float(marginal_roi or 0.0)) / 0.25))
            fee_signal = min(1.0, broadcast_fee_ppm / 500.0) if broadcast_fee_ppm > 0 else 0.0
            flow_bonus = 1.3 if flow_state in ('source',) else 1.0
            rebalance_bonus = 1.2 if rebalance_impossible else 1.0
            planner_bonus = 1.25 if planner_funding_deficit > 0 else 1.0
            route_bonus = 1.3 if scid_display in route_pair_in_channels else 1.0
            hive_bonus = hive_rebal_bias  # ±15% from fleet hints, compounds with route signal
            # Hive topology: prefer swaps that benefit fleet structure
            hive_topo = 1.0
            if hive_router and hive_router.available:
                hive_topo = hive_router.score_channel_for_hive(
                    peer_id, direction, liquidity_ratio=local_pct / 100.0
                )
            drain_bonus = 1.5 if scid_display in drain_demand_scids else 1.0
            multi_goal_value = excess_ratio * (0.35 * roi_signal + 0.35 * fee_signal + 0.30) * flow_bonus * rebalance_bonus * planner_bonus * route_bonus * hive_bonus * hive_topo * drain_bonus

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
                "dts_uplift_sats": dts_uplift_sats,
                "expected_net_sats": expected_net_sats,
                "risk_adjusted_net_sats": risk_adjusted_net_sats,
                "profit_margin_factor": effective_profit_margin,
                "passes_profit_guard": bool(passes_profit_guard),
                "structural_uplift_sats": structural_uplift_sats,
                # True only when the pass actually DEPENDED on the credit:
                # with require_profitable=False every candidate passes anyway,
                # so nothing is structural (the envelope must not be charged).
                "structural": bool(
                    require_profitable
                    and passes_profit_guard
                    and structural_uplift_sats > 0
                    and not passes_without_credit
                ),
                "rebalance_impossible": rebalance_impossible,
                "loop_in_non_pinnable": direction == "loop_in",
            },
            "dts": {
                "broadcast_fee_ppm": broadcast_fee_ppm,
                "posterior_mean": round(posterior_mean, 1) if posterior_mean else None,
                "posterior_std": round(posterior_std, 1) if posterior_std else None,
            },
            "hive": {
                "rebalance_bias": round(hive_rebal_bias, 3),
            },
            "quote": quote_resp,
            "score": {
                "multi_goal_value": round(multi_goal_value, 4),
                "severity": round(severity, 4),
                "daily_contribution_estimate_sats": int(daily_contrib_est),
                "risk_adjusted_net_sats": risk_adjusted_net_sats,
                "broadcast_fee_ppm": broadcast_fee_ppm,
                "rebalance_impossible": rebalance_impossible,
                "predicted_depletion_hours": round(predicted_depletion_hours, 1) if predicted_depletion_hours is not None else None,
                "in_drain_demand": scid_display in drain_demand_scids,
            }
        }
        candidates.append(candidate)

    # Sort: loop-outs first (generate on-chain funds + rebalance), then by multi-goal
    # value, profit safety, rebalance impossibility, estimated net, and severity.
    candidates.sort(
        key=lambda c: (
            1 if c.get("direction") == "loop_out" else 0,
            1 if c.get("economics", {}).get("passes_profit_guard") else 0,
            float(c.get("score", {}).get("multi_goal_value", 0.0) or 0.0),
            1 if c.get("economics", {}).get("rebalance_impossible") else 0,
            int(c.get("economics", {}).get("risk_adjusted_net_sats", 0) or 0),
            int(c.get("score", {}).get("broadcast_fee_ppm", 0) or 0),
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
            # Concrete currency the quotes used ('auto' resolves at plan
            # time; None = 'auto' configured but direction never triggered).
            "resolved_loop_in_currency": _annotated_currency("loop_in", loop_in_currency),
            "resolved_loop_out_currency": _annotated_currency("loop_out", loop_out_currency),
        },
        "recommendations": candidates[: max(0, int(max_candidates))],
        "total_candidates": len(candidates),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
        # Which cascade stage priced the structural inbound credit this plan
        # build: "realized_7d" / "realized_30d" / "policy_derived", or None
        # when no stage produced a positive rate (credit disabled).
        "structural_credit": {
            "scarcity": round(structural_scarcity, 4),
            "realized_ppm": int(structural_realized_ppm),
            "source": structural_credit_source,
        },
        "planner_coordination": {
            "loser_scids_excluded": len(planner_loser_scids),
            "funding_deficit_sats": planner_funding_deficit,
            "best_open_candidate": planner_coord.get("best_candidate"),
        } if planner_coord else None,
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
            "boltz_auto_cycle_enabled": bool(getattr(config, 'boltz_auto_cycle_enabled', False)) if config else None,
            "boltz_auto_cycle_interval_minutes": int(getattr(config, 'boltz_auto_cycle_interval_minutes', 15)) if config else None,
            "boltz_auto_cycle_max_actions": int(getattr(config, 'boltz_auto_cycle_max_actions', 1)) if config else None,
            "boltz_auto_cycle_startup_delay_seconds": int(getattr(config, 'boltz_auto_cycle_startup_delay_seconds', 120)) if config else None,
            "expansion_treasury_enabled": bool(getattr(config, 'expansion_treasury_enabled', False)) if config else None,
            "expansion_treasury_onchain_target_sats": int(getattr(config, 'expansion_treasury_onchain_target_sats', 5_000_000)) if config else None,
            "expansion_treasury_min_deficit_sats": int(getattr(config, 'expansion_treasury_min_deficit_sats', 250_000)) if config else None,
        },
    })
    return state


@plugin.method("revenue-boltz-auto-cycle-run-now")
def revenue_boltz_auto_cycle_run_now(plugin: Plugin, force: bool = False,
                                     dry_run: bool = False) -> Dict[str, Any]:
    """Trigger one immediate Boltz auto-cycle run using scheduler settings.

    DD4 / P1-018 (operator ruling 2026-07-02): dry_run defaults False —
    `force=true` alone runs a LIVE cycle (preserves the prior "run one now"
    semantics and bypasses the boltz_auto_cycle_enabled toggle). The dry_run
    option is exposed so an operator CAN preview with dry_run=true, but the
    default is not a preview. All economic gates (budget/cooldown/profit/
    pending-swap) still apply to the live run.
    """
    try:
        result = _run_boltz_auto_cycle_once(
            trigger="manual", force=bool(force), dry_run=bool(dry_run)
        )
        _boltz_auto_cycle_mark_state(last_result=result)
        return result
    except Exception as e:
        return {"error": str(e)}


def _execute_boltz_balance_cycle(
    *,
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
    precomputed_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Internal Boltz balance cycle executor.

    precomputed_plan lets internal callers (the auto-cycle scheduler) reuse a
    balance plan they already built instead of rebuilding the identical plan
    (each build issues per-candidate boltzcli quote subprocesses). It is NOT an
    RPC argument — the revenue-boltz-balance-cycle method never exposes it.
    """
    # Pending-swap short-circuit BEFORE building any plan: a pending swap blocks
    # the whole cycle anyway, so don't pay for quotes/profitability first.
    pending_swaps: Optional[int] = None
    if isinstance(precomputed_plan, dict) and "pending_swap_count" in precomputed_plan:
        pending_swaps = int(precomputed_plan.get("pending_swap_count", 0) or 0)
    if not allow_concurrent_swaps:
        if pending_swaps is None:
            pending_swaps = _boltz_pending_swap_count()
        if pending_swaps > 0:
            return {
                "status": "blocked",
                "reason": f"{pending_swaps} pending Boltz swap(s) detected",
                "plan": precomputed_plan,
                "executed": [],
                "skipped": [],
            }

    if isinstance(precomputed_plan, dict):
        plan = precomputed_plan
    else:
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
                pending_swap_count=pending_swaps,
            )
        except Exception as e:
            return {"error": str(e)}

    if "error" in plan:
        return plan

    recommendations = list(plan.get("recommendations", []))
    budget = plan.get("budget", {}) if isinstance(plan.get("budget"), dict) else {}
    remaining_budget = int(budget.get("remaining_24h_sats_estimate", 0) or 0)
    # P1-012 class: coerce operator-supplied cooldown_hours; a non-numeric must
    # not raise out of the balance-cycle path. Fall back to the 4h default.
    try:
        cooldown_seconds = max(0, int(float(cooldown_hours) * 3600))
    except (ValueError, TypeError):
        cooldown_seconds = 4 * 3600

    executed: List[Dict[str, Any]] = []
    skipped_exec: List[Dict[str, Any]] = []
    now = int(time.time())
    # Bound _boltz_balance_last_action: drop cooldown entries for channels not
    # touched within the TTL (e.g. closed channels) so it can't grow unbounded.
    _prune_boltz_balance_actions(now)

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

        # Structural envelope gate: candidates that only pass the profit guard
        # because of the inbound-scarcity credit spend from a dedicated daily
        # envelope (boltz_structural_budget_sats_per_day) on top of the
        # tactical budget gate above. The 24h sum is fed by structural swap
        # fees recorded under category="boltz"/subcategory="structural".
        if econ.get("structural"):
            envelope = int(getattr(config, "boltz_structural_budget_sats_per_day", 0) or 0)
            spent_24h = None
            if envelope > 0:
                try:
                    spent_24h = int(database.get_category_spend_sats(
                        "boltz", subcategory="structural",
                        since_timestamp=now - 86400,
                    ) or 0)
                except Exception:
                    # Fail closed: unknown spend means no structural execution.
                    spent_24h = None
            if envelope <= 0 or spent_24h is None or spent_24h + est_fee > envelope:
                skipped_exec.append({
                    "channel_id": ch_id,
                    "peer_id": peer_id,
                    "reason": "structural_envelope_exhausted",
                    "envelope_sats": envelope,
                    "spent_24h_sats": spent_24h,
                    "estimated_fee_sats": est_fee,
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

        # H-5 FIX: Keep lock held during cooldown decision to prevent TOCTOU race
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
            # C1 FIX: Pre-claim cooldown slot to prevent TOCTOU double-execution
            _boltz_balance_last_action[ch_id] = now

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

            # Resolve currency: "auto" compares BTC vs LBTC and picks cheaper
            if direction == "loop_in":
                currency = loop_in_currency
                if currency == "auto":
                    try:
                        currency = _select_boltz_currency("loop_in", amount_sats)
                    except Exception:
                        currency = "LBTC"
                res = bm.loop_in(amount_sats=amount_sats, channel_id=ch_id, peer_id=peer_id, currency=currency)
            elif direction == "loop_out":
                currency = loop_out_currency
                if currency == "auto":
                    try:
                        currency = _select_boltz_currency("loop_out", amount_sats)
                    except Exception:
                        currency = "LBTC"
                # Hive route discovery: find cheaper first-hop through fleet
                exec_ch_id = ch_id
                exec_peer_id = peer_id
                if hive_router and hive_router.available and peer_id:
                    try:
                        hr = hive_router.discover_route(peer_id, amount_sats)
                        if hr and hr.source_scid and hr.fee_ppm < 200:
                            exec_ch_id = hr.source_scid
                            plugin.log(
                                f"BOLTZ HIVE ROUTE: Using fleet path for loop-out "
                                f"({hr.hops} hops, {hr.fee_ppm} ppm, via {hr.source_scid})",
                            )
                    except Exception:
                        pass  # Fall back to original channel selection
                # structural=True routes the swap fee into the dedicated
                # "structural" spend subcategory so the envelope gate's 24h
                # sum advances with each structural execution.
                res = bm.loop_out(
                    amount_sats=amount_sats,
                    channel_id=exec_ch_id,
                    peer_id=exec_peer_id,
                    currency=currency,
                    structural=bool(econ.get("structural")),
                )
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
                    # C1: Pre-claim already set; just update budget
                    remaining_budget = max(0, remaining_budget - est_fee)
                else:
                    # C1: Rejected - restore original cooldown timestamp
                    with _boltz_balance_lock:
                        if _boltz_balance_last_action.get(ch_id) == now:
                            _boltz_balance_last_action[ch_id] = last_ts
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
            # C1: Exception - restore original cooldown timestamp
            with _boltz_balance_lock:
                if _boltz_balance_last_action.get(ch_id) == now:
                    _boltz_balance_last_action[ch_id] = last_ts
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
    return _execute_boltz_balance_cycle(
        dry_run=dry_run,
        max_actions=max_actions,
        low_trigger_pct=low_trigger_pct,
        low_target_pct=low_target_pct,
        high_trigger_pct=high_trigger_pct,
        high_target_pct=high_target_pct,
        min_amount_sats=min_amount_sats,
        max_amount_sats=max_amount_sats,
        only_peer_id=only_peer_id,
        only_channel_id=only_channel_id,
        require_profitable=require_profitable,
        min_marginal_roi=min_marginal_roi,
        profit_margin_factor=profit_margin_factor,
        expected_horizon_days=expected_horizon_days,
        cooldown_hours=cooldown_hours,
        allow_concurrent_swaps=allow_concurrent_swaps,
        loop_in_currency=loop_in_currency,
        loop_out_currency=loop_out_currency,
    )


@plugin.method("revenue-boltz-expansion-treasury-status")
def revenue_boltz_expansion_treasury_status(plugin: Plugin) -> Dict[str, Any]:
    """Show expansion treasury reserve target status and current on-chain reserve."""
    try:
        bm = _require_boltz_manager()
        cfg = config.snapshot() if config else None
        preferred = str(getattr(cfg, 'expansion_treasury_preferred_currency', 'BTC') if cfg else 'BTC').upper()
        target = int(getattr(cfg, 'expansion_treasury_onchain_target_sats', 5_000_000) if cfg else 5_000_000)
        min_deficit = int(getattr(cfg, 'expansion_treasury_min_deficit_sats', 250_000) if cfg else 250_000)
        onchain = _get_confirmed_onchain_sats()
        # F9: None means listfunds failed — report unavailable, never a
        # fabricated full deficit.
        deficit = max(0, target - onchain) if onchain is not None else None
        return {
            'enabled': bool(getattr(cfg, 'expansion_treasury_enabled', False)) if cfg else False,
            'status': 'ok' if onchain is not None else 'unavailable',
            'onchain_confirmed_sats': onchain,
            'onchain_target_sats': target,
            'deficit_sats': deficit,
            'min_deficit_sats': min_deficit,
            'needs_harvest': bool(deficit is not None and deficit >= min_deficit),
            'preferred_currency': preferred,
            'budget': bm.budget(),
            'pending_swap_count': _boltz_pending_swap_count(),
        }
    except Exception as e:
        return {'error': str(e)}


@plugin.method("revenue-boltz-expansion-treasury-recommendations")
def revenue_boltz_expansion_treasury_recommendations(
    plugin: Plugin,
    onchain_target_sats: int = None,
    min_deficit_sats: int = None,
    preferred_currency: str = None,
    max_actions: int = None,
    min_source_local_pct: float = None,
    exclude_protected: bool = None,
    require_profitable: bool = True,
    min_marginal_roi: float = 0.0,
    profit_margin_factor: float = 1.2,
    expected_horizon_days: float = 3.0,
    min_amount_sats: int = 100_000,
    max_amount_sats: int = 1_500_000,
) -> Dict[str, Any]:
    """Recommend reverse swaps to build on-chain expansion treasury funds."""
    try:
        cfg = config.snapshot() if config else None
        return _build_boltz_expansion_treasury_plan(
            onchain_target_sats=int(onchain_target_sats if onchain_target_sats is not None else (getattr(cfg, 'expansion_treasury_onchain_target_sats', 5_000_000) if cfg else 5_000_000)),
            min_deficit_sats=int(min_deficit_sats if min_deficit_sats is not None else (getattr(cfg, 'expansion_treasury_min_deficit_sats', 250_000) if cfg else 250_000)),
            preferred_currency=str(preferred_currency if preferred_currency is not None else (getattr(cfg, 'expansion_treasury_preferred_currency', 'BTC') if cfg else 'BTC')).upper(),
            max_actions=int(max_actions if max_actions is not None else (getattr(cfg, 'expansion_treasury_max_actions', 1) if cfg else 1)),
            min_source_local_pct=float(min_source_local_pct if min_source_local_pct is not None else (getattr(cfg, 'expansion_treasury_min_source_local_pct', 80.0) if cfg else 80.0)),
            exclude_protected=bool(exclude_protected if exclude_protected is not None else (getattr(cfg, 'expansion_treasury_exclude_protected', True) if cfg else True)),
            require_profitable=bool(require_profitable),
            min_marginal_roi=float(min_marginal_roi),
            profit_margin_factor=float(profit_margin_factor),
            expected_horizon_days=float(expected_horizon_days),
            min_amount_sats=int(min_amount_sats),
            max_amount_sats=int(max_amount_sats),
        )
    except Exception as e:
        return {'error': str(e)}


def _execute_boltz_expansion_treasury_cycle(
    *,
    dry_run: bool = True,
    max_actions: int = None,
    onchain_target_sats: int = None,
    min_deficit_sats: int = None,
    preferred_currency: str = None,
    min_source_local_pct: float = None,
    exclude_protected: bool = None,
    require_profitable: bool = True,
    min_marginal_roi: float = 0.0,
    profit_margin_factor: float = 1.2,
    expected_horizon_days: float = 3.0,
    min_amount_sats: int = 100_000,
    max_amount_sats: int = 1_500_000,
    cooldown_hours: float = 4.0,
    allow_concurrent_swaps: bool = False,
    precomputed_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Internal treasury-funding cycle executor.

    precomputed_plan lets internal callers (the auto-cycle scheduler) reuse the
    treasury plan they already built for mode selection instead of rebuilding it
    (each build nests a full balance-plan build: per-candidate boltzcli quotes
    plus budget/listswaps subprocesses). It is NOT an RPC argument — the
    revenue-boltz-expansion-treasury-cycle method never exposes it.
    """
    cfg = config.snapshot() if config else None
    if isinstance(precomputed_plan, dict):
        plan = precomputed_plan
    else:
        try:
            plan = _build_boltz_expansion_treasury_plan(
                onchain_target_sats=int(onchain_target_sats if onchain_target_sats is not None else (getattr(cfg, 'expansion_treasury_onchain_target_sats', 5_000_000) if cfg else 5_000_000)),
                min_deficit_sats=int(min_deficit_sats if min_deficit_sats is not None else (getattr(cfg, 'expansion_treasury_min_deficit_sats', 250_000) if cfg else 250_000)),
                preferred_currency=str(preferred_currency if preferred_currency is not None else (getattr(cfg, 'expansion_treasury_preferred_currency', 'BTC') if cfg else 'BTC')).upper(),
                max_actions=int(max_actions if max_actions is not None else (getattr(cfg, 'expansion_treasury_max_actions', 1) if cfg else 1)),
                min_source_local_pct=float(min_source_local_pct if min_source_local_pct is not None else (getattr(cfg, 'expansion_treasury_min_source_local_pct', 80.0) if cfg else 80.0)),
                exclude_protected=bool(exclude_protected if exclude_protected is not None else (getattr(cfg, 'expansion_treasury_exclude_protected', True) if cfg else True)),
                require_profitable=bool(require_profitable),
                min_marginal_roi=float(min_marginal_roi),
                profit_margin_factor=float(profit_margin_factor),
                expected_horizon_days=float(expected_horizon_days),
                min_amount_sats=int(min_amount_sats),
                max_amount_sats=int(max_amount_sats),
            )
        except Exception as e:
            return {'error': str(e)}

    if 'error' in plan:
        return plan
    if str(plan.get('status') or '') in ('at_target', 'unavailable'):
        return {'status': str(plan.get('status')), 'plan': plan, 'executed': [], 'skipped': []}

    pending_swaps = int(plan.get('pending_swap_count', 0) or 0)
    if pending_swaps > 0 and not allow_concurrent_swaps:
        return {
            'status': 'blocked',
            'reason': f'{pending_swaps} pending Boltz swap(s) detected',
            'plan': plan,
            'executed': [],
            'skipped': [],
        }

    recs = list(plan.get('recommendations', []))
    budget = plan.get('budget', {}) if isinstance(plan.get('budget'), dict) else {}
    remaining_budget = int(budget.get('remaining_24h_sats_estimate', 0) or 0)
    cooldown_seconds = max(0, int(float(cooldown_hours) * 3600))
    target_deficit_remaining = int(((plan.get('treasury') or {}).get('deficit_sats', 0) if isinstance(plan.get('treasury'), dict) else 0) or 0)
    max_exec = max(1, int(max_actions if max_actions is not None else (getattr(cfg, 'expansion_treasury_max_actions', 1) if cfg else 1)))

    executed = []
    skipped_exec = []
    now = int(time.time())
    # Bound _boltz_balance_last_action (see _prune_boltz_balance_actions).
    _prune_boltz_balance_actions(now)

    for rec in recs:
        if len(executed) >= max_exec:
            break
        if target_deficit_remaining <= 0:
            break

        ch_id = str(rec.get('channel_id') or '')
        peer_id = str(rec.get('peer_id') or '')
        direction = str(rec.get('direction') or '')
        amount_sats = int(rec.get('amount_sats', 0) or 0)
        econ = rec.get('economics', {}) if isinstance(rec.get('economics'), dict) else {}
        est_fee = int(econ.get('estimated_swap_fee_sats', 0) or 0)
        quote = rec.get('quote', {}) if isinstance(rec.get('quote'), dict) else {}
        est_receive = int(quote.get('receiveAmount') or quote.get('receive_amount_sats') or amount_sats or 0)

        if direction != 'loop_out':
            skipped_exec.append({'channel_id': ch_id, 'peer_id': peer_id, 'reason': 'not_loop_out'})
            continue
        if not econ.get('passes_profit_guard', False):
            skipped_exec.append({'channel_id': ch_id, 'peer_id': peer_id, 'reason': 'profit_guard_failed', 'recommendation': rec})
            continue
        if econ.get('structural'):
            # Credit-dependent (structural) passes are balance-cycle only: the
            # treasury executor has no structural-envelope accounting, so a
            # swap justified solely by the inbound-scarcity credit must not
            # fund the treasury. Candidates that clear the guard on their own
            # economics have structural=False and proceed unchanged.
            skipped_exec.append({'channel_id': ch_id, 'peer_id': peer_id, 'reason': 'structural_credit_requires_balance_cycle', 'recommendation': rec})
            continue
        if est_fee > remaining_budget:
            skipped_exec.append({'channel_id': ch_id, 'peer_id': peer_id, 'reason': 'insufficient_remaining_budget', 'estimated_fee_sats': est_fee, 'remaining_budget_sats': remaining_budget, 'recommendation': rec})
            continue

        rec_hints = rec.get('execution_hints', {}) if isinstance(rec.get('execution_hints'), dict) else {}
        try:
            rec_cd = int(float(rec_hints.get('recommended_cooldown_hours', cooldown_hours) or cooldown_hours) * 3600)
        except Exception:
            rec_cd = cooldown_seconds
        with _boltz_balance_lock:
            last_ts = int(_boltz_balance_last_action.get(ch_id, 0) or 0)
            if rec_cd > 0 and last_ts > 0 and (now - last_ts) < rec_cd:
                skipped_exec.append({'channel_id': ch_id, 'peer_id': peer_id, 'reason': 'cooldown_active', 'cooldown_remaining_sec': rec_cd - (now - last_ts), 'recommendation': rec})
                continue
            # C1 FIX: Pre-claim cooldown slot to prevent TOCTOU double-execution
            _boltz_balance_last_action[ch_id] = now

        if dry_run:
            executed.append({'status': 'would_execute', 'direction': direction, 'channel_id': ch_id, 'peer_id': peer_id, 'amount_sats': amount_sats, 'estimated_fee_sats': est_fee, 'estimated_receive_onchain_sats': est_receive, 'recommendation': rec})
            remaining_budget = max(0, remaining_budget - est_fee)
            target_deficit_remaining = max(0, target_deficit_remaining - est_receive)
            continue

        try:
            bm = _require_boltz_manager()
            res = bm.loop_out(amount_sats=amount_sats, channel_id=ch_id, peer_id=peer_id, currency=str(((plan.get('treasury') or {}).get('preferred_currency', 'BTC'))).upper())
            status = str(res.get('status') or '')
            payload = {'status': status or 'unknown', 'direction': direction, 'channel_id': ch_id, 'peer_id': peer_id, 'amount_sats': amount_sats, 'estimated_fee_sats': est_fee, 'estimated_receive_onchain_sats': est_receive, 'result': res, 'recommendation': rec}
            executed.append(payload)
            if status == 'accepted':
                # C1: Pre-claim already set; just update budget
                remaining_budget = max(0, remaining_budget - est_fee)
                target_deficit_remaining = max(0, target_deficit_remaining - est_receive)
            elif status == 'rejected':
                # C1: Rejected - restore original cooldown timestamp
                with _boltz_balance_lock:
                    if _boltz_balance_last_action.get(ch_id) == now:
                        _boltz_balance_last_action[ch_id] = last_ts
                skipped_exec.append({'channel_id': ch_id, 'peer_id': peer_id, 'reason': 'execution_rejected', 'result': res})
        except Exception as e:
            # C1: Exception - restore original cooldown timestamp
            with _boltz_balance_lock:
                if _boltz_balance_last_action.get(ch_id) == now:
                    _boltz_balance_last_action[ch_id] = last_ts
            skipped_exec.append({'channel_id': ch_id, 'peer_id': peer_id, 'reason': f'execution_failed: {e}', 'recommendation': rec})

    return {
        'status': 'dry_run' if dry_run else 'executed',
        'mode': 'expansion_treasury',
        'executed_count': len(executed),
        'skipped_count': len(skipped_exec),
        'remaining_budget_sats_estimate_after_cycle': remaining_budget,
        'remaining_treasury_deficit_sats_estimate_after_cycle': target_deficit_remaining,
        'executed': executed,
        'skipped': skipped_exec,
        'plan': plan,
        'notes': [
            'Treasury mode only executes reverse swaps (LN -> BTC/LBTC) to build on-chain expansion reserve',
            'Protected hot channels can be excluded to avoid harvesting liquidity from critical routing channels',
            'Reverse swaps on CLN are not channel-pinnable via chanIds; exact path control uses external-pay + first-hop constrained CLN pay when available',
        ],
    }


@plugin.method("revenue-boltz-expansion-treasury-cycle")
def revenue_boltz_expansion_treasury_cycle(
    plugin: Plugin,
    dry_run: bool = True,
    max_actions: int = None,
    onchain_target_sats: int = None,
    min_deficit_sats: int = None,
    preferred_currency: str = None,
    min_source_local_pct: float = None,
    exclude_protected: bool = None,
    require_profitable: bool = True,
    min_marginal_roi: float = 0.0,
    profit_margin_factor: float = 1.2,
    expected_horizon_days: float = 3.0,
    min_amount_sats: int = 100_000,
    max_amount_sats: int = 1_500_000,
    cooldown_hours: float = 4.0,
    allow_concurrent_swaps: bool = False,
) -> Dict[str, Any]:
    """Run a treasury-funding reverse-swap cycle (LN -> on-chain) for expansion reserve."""
    return _execute_boltz_expansion_treasury_cycle(
        dry_run=dry_run,
        max_actions=max_actions,
        onchain_target_sats=onchain_target_sats,
        min_deficit_sats=min_deficit_sats,
        preferred_currency=preferred_currency,
        min_source_local_pct=min_source_local_pct,
        exclude_protected=exclude_protected,
        require_profitable=require_profitable,
        min_marginal_roi=min_marginal_roi,
        profit_margin_factor=profit_margin_factor,
        expected_horizon_days=expected_horizon_days,
        min_amount_sats=min_amount_sats,
        max_amount_sats=max_amount_sats,
        cooldown_hours=cooldown_hours,
        allow_concurrent_swaps=allow_concurrent_swaps,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    plugin.run()
