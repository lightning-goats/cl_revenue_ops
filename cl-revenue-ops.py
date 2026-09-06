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
import queue
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
from modules.fee_authority import FeeAuthorityGate
from modules.rebalancer import EVRebalancer
from modules.config import CONFIG_FIELD_RANGES, Config
from modules.growth_budget import compute_growth_budget_status
from modules.database import Database
from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
from modules.policy_manager import (
    PolicyManager,
    FeeStrategy,
    RebalanceMode,
    PeerPolicy,
    READ_ONLY_POLICY_ACTIONS,
    TACTICAL_POLICY_ACTIONS,
)
from modules.capex_budget import CapexBudgetEngine
from modules.capital_efficiency import CapitalEfficiencyAnalyzer
from modules.segment_observations import SegmentObservationStore
from modules.utils import normalize_scid, parse_msat, channel_local_balance_msat
from modules.econ_shadow import EconShadow, fail_open_fee_evidence_guard
from modules.forward_archive import ForwardArchiveError, parse_cln_time_ns
from modules.forward_archive_sync import ForwardArchiveSynchronizer
from modules.forward_identity import ForwardSource, observe_settled_identity
from modules.forward_precision import ForwardPrecisionPluginMixin, configure_forward_precision


# =============================================================================
# PLUGIN VERSION
# =============================================================================
# v3.0.0: Retire unused liquidity executors (2026-08-03)
#   - Removed the capacity lifecycle planner, automatic channel open/close, planner defibrillation,
#     Boltz, and LN+ execution surfaces. Historical accounting rows remain readable.
#   - Retained fee control, revenue reporting, profitability analysis, and
#     budget-constrained automatic/manual circular rebalancing.
# v2.17.1: Deep-audit fix release (2026-07-10)
#   - Six-agent top-to-bottom audit (module clusters + live RPC/DB
#     cross-validation) followed by two fix iterations and an adversarial
#     verification pass. 1 critical (days_open frozen at 30 for every
#     channel opened in the last ~9 months — SCID timestamp estimate now
#     tip-anchored, poisoned opened_at rows repaired), 13 major (dead
#     cooldown burn on dry-run/policy-block, RPC-handler tracebacks,
#     poisoning futility/success-rates, execute_candidate inflight-guard
#     bypass, PnL 30d-revenue-vs-8d-volume window mismatch), and ~20 minor
#     fixes. Orphan hive tables dropped by migration. fee_controller dead
#     code removed (-207 lines). Full suite 3091 green.
# v2.17.0: Standalone Phases 4-5 — complete the de-hive (2026-07-09/10)
#   - Phase 5: removed every remaining inert hive branch — fee_controller
#     (temporal adjustment, fleet fee prior, skewed-prior reseed, fleet-sibling
#     exclusion), the orchestrator's hive globals/RPC (revenue-hive-hints-status
#     removed) and 9 dead option registrations, config.py dead keys, module
#     bias plumbing (PairCandidate hive/metabolic/immune fields), the
#     hive-observed-liquidity layer, and the tools/audit hive sweeps.
#   - Tests scrubbed; architecture guard now pins the de-hive (deleted modules
#     stay deleted, no non-comment hive/mycelium references in runtime source).
#   - Docs: hive contracts/audits deleted, README/AGENTS rewritten standalone.
#   - Grep gate: modules/ + cl-revenue-ops.py contain only historical comments.
# v2.16.0: Standalone Phases 2-3 — de-hive the revenue core (2026-07-09)#   - Removed all remaining cl-hive/fleet/coordination code from the revenue
#     engine: profitability_analyzer, capital_efficiency, policy_manager,
#     capex_budget, database (hive tables + corridor instrumentation),
#     and rebalancer (coordination cluster, hive discovery +
#     close-protection + value weights), the atomic routing cluster (deleted
#     rebalance_coordination_overlay + rebalance_hive_router; collapsed
#     rebalance_route_policy to market-only; de-hived rebalance_engine_v2 /
#     _state_v2 / _planner_v2), and fee_controller (zero-fee corridor).
#   - Behavior-neutral except two intended standalone changes: fleet peers are
#     no longer close-protected, and "fleet member" channels are no longer
#     force-priced to 0 ppm (they price normally via DTS/PID). Neither has any
#     effect today — there are no fleet members without cl-hive.
#   - Preserved all general pricing/routing math (DTS/PID, drain bias, the
# v2.15.0: Standalone Phase 1 — delete the dedicated hive modules (2026-07-09)
#   - Removed modules/hive_hints.py, hive_router.py, hive_runtime.py (~3,368
#     lines) and every orchestrator reference to them: the imports, the
#     HiveRouter construction, all refresh_hive_runtime call sites, and the
#     hive_refresh debug block. hive_hints/hive_router stay as permanently-None
#     globals (neutral seams); the guarded consumer branches no-op.
#   - Deleted 18 dedicated hive test modules; surgically pruned 5 mixed test
#     files (kept their non-hive coverage) and 1 daemon-survival helper list.
#   - Deferred to later phases: rebalance_hive_router.py (module-level dep of
#     rebalance_engine_v2 → Phase 3) and the tools/audit/* hive sweeps (Phase 5).
# v2.14.0: Standalone Phase 0 — cut the cl-mycelium live wires (2026-07-09)
#   - cl-mycelium retired: hive_hints is now permanently None (HiveHintAdapter
#     is never constructed); every consumer neutralizes through its existing
#     getattr defaults. Injection into fee_controller/rebalancer/etc. is kept.
#   - The three cl-hive coordination RPCs are gone: hive-report-rebalance-intent
#     and both hive-report-rebalance-outcome sites are no-ops (coordinated
#     candidates cannot arise without hive hints, so these paths are inert).
#   - askrene default layer flipped from "hive-fleet" to the "standalone"
#     sentinel, so blank config resolves to no askrene layers (plain CLN
#     routing). Fixed _configured_layer_names ordering so the default itself
#     resolves through the standalone sentinel to [].
#   - Config options remain registered-but-unused (removing an option a node's
#     config still references is restart-fatal); their excision is deferred.
# v2.13.2: Policy lazy-evaluation audit fixes (2026-07-09)
#   - Native rebalance engine now enforces peer policy (rebalance_mode /
#     passive) — eager candidate filter + lazy re-check at execution; the
#     highest-frequency spend path previously never read policy at all
#   - Recycle protection set actually populates (list/.items() crash was
#     swallowed); close/open/swap/defib gates fail CLOSED on policy errors
#     hive-route override's first-hop peer passes the same gate
#   - Close gate lazily re-reads hive membership; opens blocked for passive
#     peers; defib respects rebalance_mode (no_close never blocks defib)
#     never claimed under policy-lookup uncertainty
# v2.13.1: RPC surface-quality fixes from the read-only sweep (2026-07-08)
#     swap_id and points at the global-state RPCs
#   - expansion-treasury-recommendations returns one stable key set on every
#     branch; auto-cycle-status embeds compact recommendation summaries
# v2.13.0: Econ audit wave (2026-07-08)
#   - Class-aware saturated/source min-fee floor (min_fee_ppm_saturated) so saturated
#     and pure-source channels can price below the global fee floor
#   - Dynamic-htlcmax valve gains live-outbound-depletion keying so a near-drained
#     channel never advertises an htlc_max sized off total capacity
#   - Node-liquidity-aware drain-bias, weekly-budget live-raise fix, and
#     rebalance_min_profit deprecated to a no-op (superseded by the sats-EV gate)
# v2.12.0: True zero-fee hive corridor (2026-07-08)
#   - All hive-internal channels default to 0 ppm / 0 base msat, public and announced
#   - Durable through hint staleness via a DB-confirmed membership grace window
#     (hive_zero_fee_stale_grace_seconds, default 7 days)
#   - Adds mycelial corridor-flow utilization instrumentation (internal/edge/external)
#   - Autonomous join-only liquidity-swap ring participation, scored against the
#     capacity planner's own open-EV machinery with an inbound-liquidity credit
#   - One swap in flight per node; one-strike circuit breaker; both-side no_close
#     contract protection; intent-first ledger writes on every irreversible step
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
PLUGIN_VERSION = "3.0.0"

# Supply-chain / runtime version floor.  v26.06.7 is an embargoed security
# release, so an older or unidentified runtime must not load this plugin.
# Failing plugin init leaves lightningd itself running and makes the required
# operator upgrade explicit instead of silently operating on a vulnerable CLN.
CLN_VERSION_FLOOR = "26.06.7"


def _parse_version_tuple(raw: Optional[str]):
    """Extract a numeric (major, minor, patch, ...) tuple from a version string.

    Tolerant of command prefixes,
    leading "v", suffixes ("2.11.0-beta", "24.11.1gl"), and build
    metadata. Returns () when no version-shaped numeric component is found,
    which callers treat as "unknown / skip the comparison".
    """
    if not raw:
        return ()
    text = str(raw).strip()
    match = re.search(r'(?i)(?<![A-Za-z0-9])v?(\d+(?:\.\d+)*)', text)
    if not match:
        return ()
    return tuple(int(token) for token in match.group(1).split('.'))


def _version_below_floor(observed: Optional[str], floor: str) -> Optional[bool]:
    """Return True if `observed` < `floor`, False if >=, None if undetermined."""
    obs = _parse_version_tuple(observed)
    flr = _parse_version_tuple(floor)
    if not obs or not flr:
        return None
    return obs < flr


def _require_cln_version(observed: Optional[str], floor: str = CLN_VERSION_FLOOR):
    """Return the parsed version or reject an old/unknown CLN runtime.

    Unknown input fails closed because a version string is the only portable
    startup evidence available to a Python plugin.  This gate does not attest
    binary provenance; operators must still use a maintainer-verified release
    artifact, especially where a container tag has been republished.
    """
    parsed = _parse_version_tuple(observed)
    below = _version_below_floor(observed, floor)
    if below is None:
        raise RuntimeError(
            "cl-revenue-ops requires a verifiable Core Lightning version "
            f">= v{floor}; getinfo returned {observed!r}"
        )
    if below:
        raise RuntimeError(
            f"cl-revenue-ops requires Core Lightning >= v{floor}; "
            f"observed {observed!r}"
        )
    return parsed


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
# A periodic loop is "stalled" only once its tick age exceeds a threshold
# derived from its OWN interval: max(_LOOP_STALL_FLOOR_SECONDS,
# _LOOP_STALL_INTERVAL_MULTIPLE * its_own_interval). This avoids false
# positives for loops that legitimately run less than hourly (e.g. the
# daily financial-snapshot loop).
# A one-shot loop (ticks exactly once, e.g. startup-snapshot) is never
# considered stalled. _LOOP_STALL_SECONDS remains the fallback used to
# derive a threshold (~3600s, matching the multiple below) for any loop
# that reports no interval.
_LOOP_STALL_SECONDS = 3600
_LOOP_STALL_INTERVAL_MULTIPLE = 3     # stalled only after 3x its own interval
_LOOP_STALL_FLOOR_SECONDS = 900       # never flag a loop stalled under 15 min


def _record_loop_heartbeat(name: str, interval_seconds: Optional[float] = None, one_shot: bool = False) -> None:
    """Record a per-thread heartbeat for daemon loop ``name`` (best-effort).

    ``interval_seconds`` is the loop's own base (pre-jitter) interval, used
    by ``_loop_liveness_snapshot`` to derive a per-loop stall threshold.
    ``one_shot`` marks a loop that ticks exactly once by design (e.g. the
    startup-snapshot thread), which is never reported as stalled.
    """
    try:
        with _loop_heartbeats_lock:
            _loop_heartbeats[name] = {
                "last_tick_monotonic": time.monotonic(),
                "last_tick_ts": int(time.time()),
                "interval_seconds": interval_seconds,
                "one_shot": bool(one_shot),
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
        if hb.get("one_shot"):
            out[name] = {
                "last_tick_ts": int(hb.get("last_tick_ts", 0) or 0),
                "last_tick_age_seconds": int(age),
                "state": "complete",
                "one_shot": True,
            }
            continue
        interval = hb.get("interval_seconds")
        if not interval or interval <= 0:
            # No reported interval: fall back to the flat threshold so the
            # resulting threshold matches the historical ~3600s behavior.
            interval = _LOOP_STALL_SECONDS / _LOOP_STALL_INTERVAL_MULTIPLE
        threshold = max(_LOOP_STALL_FLOOR_SECONDS, _LOOP_STALL_INTERVAL_MULTIPLE * float(interval))
        out[name] = {
            "last_tick_ts": int(hb.get("last_tick_ts", 0) or 0),
            "last_tick_age_seconds": int(age),
            "state": "stalled" if age > threshold else "alive",
            "stall_threshold_seconds": int(threshold),
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

# Numeric options range-checked at init. Use the same authoritative table as
# runtime overrides so startup cannot silently omit a newly governed field.
_INIT_NUMERIC_RANGES = dict(CONFIG_FIELD_RANGES)


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
    """P1-009: enforce min_fee_ppm <= max_fee_ppm without lowering the floor.

    Inverted bounds would otherwise silently pin fees to a low ceiling and
    suppress revenue. Raise the ceiling because lowering min_fee_ppm can
    violate the CRITICAL-02 economic floor.
    """
    if 'min_fee_ppm' not in kwargs or 'max_fee_ppm' not in kwargs:
        return kwargs
    try:
        mn = int(kwargs['min_fee_ppm'])
        mx = int(kwargs['max_fee_ppm'])
    except (ValueError, TypeError):
        return kwargs
    if mn > mx:
        _init_warn(log, f"Config min_fee_ppm ({mn}) > max_fee_ppm ({mx}); raising max_fee_ppm to {mn}")
        kwargs["max_fee_ppm"] = mn
    return kwargs


# Enum-style options validated at init and their documented defaults. Valid
# value sets come from Config.STRING_ENUM_VALID_VALUES where present;
# base_fee_policy is not registered there and only means off/adaptive.
_INIT_ENUM_DEFAULTS = {
    'market_fee_mode': 'undercut',
    'base_fee_policy': 'off',
    'fee_profile': 'active',
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


def _validate_startup_config_options(kwargs, log=None):
    """Apply startup repairs in an order that leaves every invariant true."""
    _validate_numeric_config_options(kwargs, log=log)
    _enforce_fee_bound_invariant(kwargs, log=log)
    _validate_enum_config_options(kwargs, log=log)
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


def _native_forward_ingestion_source() -> Optional[ForwardSource]:
    # Older Database implementations do not have this explicit admission API.
    # Require an implemented method, not a synthesized dynamic attribute.
    if not callable(getattr(type(database), "get_native_forward_source", None)):
        return None
    return database.get_native_forward_source()


def _hydration_received_after(forward, start_time: int) -> bool:
    if not isinstance(forward, dict):
        return False
    try:
        received_ns = parse_cln_time_ns(forward.get("received_time"))
        return received_ns is not None and received_ns > start_time * 1_000_000_000
    except (ValueError, TypeError, OverflowError):
        return False


def _hydrate_settled_forward_rows(start_time: int) -> Tuple[int, int]:
    """Actual startup adapter; retain native payload before any lossy parsing.

    A new process must obtain explicit source admission before a native-mode
    DB can ingest. This function does not verify a wallet, bind a source,
    activate a schema or read the archive for decisions.
    """
    source = _native_forward_ingestion_source()
    forwards = _hydration_fetch_settled_forwards(start_time)
    if source is not None:
        return database.bulk_insert_forwards(forwards, native_source=source), len(forwards)
    rows = []
    for fwd in forwards:
        received_time = int(fwd.get("received_time", 0) or 0)
        resolved_time = int(fwd.get("resolved_time", 0) or 0)
        rows.append({
            'in_channel': fwd.get("in_channel", ""),
            'out_channel': fwd.get("out_channel", ""),
            'in_msat': _parse_msat(fwd.get("in_msat", fwd.get("in_msatoshi", 0))),
            'out_msat': _parse_msat(fwd.get("out_msat", fwd.get("out_msatoshi", 0))),
            'fee_msat': _parse_msat(fwd.get("fee_msat", fwd.get("fee_msatoshi", 0))),
            'resolution_time': max(0, resolved_time - received_time) if resolved_time > 0 else 0,
            'received_time': received_time, 'resolved_time': resolved_time,
        })
    return (database.bulk_insert_forwards(rows) if rows else 0), len(rows)


def _hydration_fetch_settled_forwards(start_time: int) -> List[Dict[str, Any]]:
    """Fetch settled forwards newer than start_time for startup hydration.

    Prefers a paged listforwards(index="created", start=..., limit=N) loop so
    the node's entire settled-forward history is never materialized in a
    single RPC response. Any error (older CLN without index paging, mocked or
    unexpected response shapes, RPC failures) falls back to the legacy full
    settled fetch. Native mode uses durable receipts in bulk_insert_forwards;
    the legacy coarse uniqueness path remains unqualified for historical replay.
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
                if _hydration_received_after(fwd, start_time):
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
    return [f for f in forwards if _hydration_received_after(f, start_time)]


# Initialize the plugin. Default dispatch remains pyln's original behavior.
class RevenuePlugin(ForwardPrecisionPluginMixin, Plugin):
    pass


plugin = RevenuePlugin()

plugin.add_option(
    name='revenue-ops-exact-forward-times', default=False, opt_type='bool',
    description='Startup-only exact forward JSON timestamps; disabled until precision repair and compatibility qualification.',
)

# =============================================================================
# GRACEFUL SHUTDOWN SUPPORT (Plugin Lifecycle Management)
# =============================================================================
# This event is used to signal all background threads to exit cleanly.
# When `lightning-cli plugin stop cl-revenue-ops` is called, CLN sends SIGTERM.
# We catch this signal and set the event, causing all loops to exit immediately
# instead of waiting for their sleep timers (which could be 30+ minutes).

shutdown_event = threading.Event()
# Maxsize one deliberately coalesces burst evidence.  One pending wake is
# enough because the next fee cycle reads the complete durable forward history.
# This leaves the steady-state fee cadence unchanged.
fee_adjustment_wake_queue = queue.Queue(maxsize=1)


def _request_fee_adjustment_wake() -> None:
    """Wake the governed fee loop without blocking a notification thread."""
    try:
        fee_adjustment_wake_queue.put_nowait(None)
    except queue.Full:
        pass


def _wait_for_fee_adjustment_wake(timeout: float) -> bool:
    """Wait for schedule/wake/shutdown; return true only for shutdown."""
    try:
        fee_adjustment_wake_queue.get(timeout=max(0.0, float(timeout)))
    except queue.Empty:
        pass
    return shutdown_event.is_set()

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
fee_authority_gate = FeeAuthorityGate()
_fee_authority_transition_lock = threading.Lock()
rebalancer: Optional[EVRebalancer] = None
database: Optional[Database] = None
config: Optional[Config] = None
profitability_analyzer: Optional[ChannelProfitabilityAnalyzer] = None
safe_plugin: Optional['ThreadSafePluginProxy'] = None  # Thread-safe plugin wrapper
data_service = None  # Unified data service (DataService instance)
policy_manager: Optional[PolicyManager] = None  # v1.4: Peer policy management
capex_engine: Optional[CapexBudgetEngine] = None  # Unified capex budget engine
econ_shadow: Optional[EconShadow] = None  # Phase 1 shadow: observe-mode intents + snapshot preview (fail-open)
forward_archive_sync: Optional[ForwardArchiveSynchronizer] = None

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


def _parse_dynamic_bool(option_name: str, value: Any) -> bool:
    """Parse a dynamic boolean option without accepting ambiguous values."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{option_name} must be a boolean")


def _on_fee_authority_change(
    plugin_: Plugin,
    option_name: str,
    new_value: Any,
) -> str:
    """Atomically apply the dynamic Python fee-authority setting."""
    enabled = _parse_dynamic_bool(option_name, new_value)
    cfg = globals().get("config")
    if cfg is None:
        raise ValueError("fee authority configuration is not initialized")
    # Do not hold Config._lock while draining fee work: an accepted cycle
    # may need Config.snapshot() before releasing its authority lease.
    with _fee_authority_transition_lock:
        status = fee_authority_gate.set_enabled(enabled, reason="setconfig")
        with cfg._lock:
            cfg.fee_authority_enabled = status.enabled
    return (
        f"{option_name}={str(status.enabled).lower()} "
        f"generation={status.generation}"
    )


def _fee_authority_denial(operation: str) -> dict[str, object] | None:
    return fee_authority_gate.deny_reason(operation)


def _on_fee_replay_capture_change(
    plugin_: Plugin,
    option_name: str,
    new_value: Any,
) -> None:
    """Apply fee replay capture lifecycle changes from setconfig."""
    enabled = _parse_dynamic_bool(option_name, new_value)
    cfg = globals().get("config")
    controller = globals().get("fee_controller")
    manager = (
        getattr(controller, "_fee_capture", None)
        if controller is not None
        else None
    )
    if manager is not None:
        manager_ready = manager.set_enabled(enabled, timeout_seconds=5.0)
        if enabled and manager_ready is not True:
            raise ValueError(f"{option_name} could not be enabled")
        if not enabled and manager_ready is not True:
            if cfg is not None:
                cfg.fee_replay_capture_enabled = False
            plugin_.log(
                "FEE REPLAY CAPTURE: disabled; writer is still draining",
                level="warn",
            )
            return
    if cfg is not None:
        cfg.fee_replay_capture_enabled = enabled
    plugin_.log(
        f"FEE REPLAY CAPTURE: {'enabled' if enabled else 'disabled'}",
        level="info",
    )


def _on_rebalance_replay_capture_change(
    plugin_: Plugin, option_name: str, new_value: Any,
) -> None:
    """Apply rebalance capture lifecycle changes without running a cycle."""
    enabled = _parse_dynamic_bool(option_name, new_value)
    cfg = globals().get("config")
    rebalancer_ = globals().get("rebalancer")
    engine = getattr(rebalancer_, "rebalance_engine_v2", None) if rebalancer_ is not None else None
    manager = getattr(engine, "rebalance_capture_manager", None)
    if manager is None and enabled:
        raise ValueError(f"{option_name} manager is unavailable")
    if manager is not None:
        manager_ready = manager.set_enabled(enabled, timeout_seconds=5.0)
        if enabled and manager_ready is not True:
            raise ValueError(f"{option_name} could not be enabled")
        if not enabled and manager_ready is not True:
            raise ValueError(f"{option_name} could not be disabled")
    if cfg is not None:
        cfg.rebalance_replay_capture_enabled = enabled
    plugin_.log(f"REBALANCE REPLAY CAPTURE: {'enabled' if enabled else 'disabled'}", level="info")

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
    name="revenue-ops-fee-authority-enabled",
    default=True,
    description="Permit Python fee evaluation and setchannel authority",
    opt_type="bool",
    dynamic=True,
    on_change=_on_fee_authority_change,
)

plugin.add_option(
    name='revenue-ops-fee-replay-capture-enabled',
    default='false',
    description=(
        'Internal observational fee-cycle replay capture. Disabled by '
        'default; enabling observes the next naturally scheduled cycle '
        'without starting a cycle.'
    ),
    dynamic=True,
    on_change=_on_fee_replay_capture_change,
)

plugin.add_option(
    name="revenue-ops-rebalance-replay-capture-enabled",
    default="false",
    description="Internal observational rebalance-cycle replay capture; disabled by default.",
    dynamic=True,
    on_change=_on_rebalance_replay_capture_change,
)

plugin.add_option(
    name='revenue-ops-rebalance-interval',
    default='900',
    description='Interval in seconds for rebalance checks (default: 15 min)'
)


plugin.add_option(
    name='revenue-ops-min-fee-ppm',
    default='50',
    description='Minimum fee floor in PPM (default: 50)'
)

plugin.add_option(
    name='revenue-ops-min-fee-ppm-saturated',
    default='0',
    description=(
        'Class-aware min-fee floor (PPM) for channels classified saturated '
        '(outbound >= 85% of capacity) or source. Applied only when set '
        'BELOW revenue-ops-min-fee-ppm; 0 (default) allows true cheap '
        'egress on saturated edges. Cost-recovery floors still apply. '
        'Flow-aware exemption: high-local channels whose 7d flow is '
        'balanced at healthy turnover (self-refilling routers the discount '
        'cannot drain) keep the normal min-fee-ppm floor.'
    ),
    dynamic=True
)

plugin.add_option(
    name='revenue-ops-max-fee-ppm',
    default='2000',
    description='Maximum fee ceiling in PPM (default: 2000)'
)

plugin.add_option(
    name='revenue-ops-acquisition-experiment-enabled',
    default=True,
    description=(
        'Enable one bounded zero-fee acquisition experiment on a qualifying '
        'cold saturated/source channel (default: true)'
    ),
    opt_type='bool',
    dynamic=True,
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
        'inelastic-demand markets to maximize revenue per forward. '
        '"yield_aware": ratchet upward only after paid demand, combining '
        'capacity-weighted market power with inventory scarcity while '
        'retaining configured fee rails and a bounded yield-discovery step. '
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
        '"off" (default) and "adaptive" are equivalent: every channel gets '
        'the internal base fee default of 0 msat (the per-role split was '
        'retired; there is no revenue-ops-base-fee-msat option).'
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
    description=(
        'Deprecated no-op compatibility setting; minimum-profit gating is '
        'enforced by the sats-EV gate and revenue-ops-rebalance-hold-margin '
        '(default: 10)'
    )
)

plugin.add_option(
    name='revenue-ops-rebalance-emergency-local-ratio',
    default='0.20',
    description='Local ratio below which a destination bypasses the channel-level rebalance cooldown (Phase 3, default: 0.20; 0 disables)'
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
    name='revenue-ops-rebalance-value-model',
    default='legacy_sum',
    description='Startup-only rebalance valuation: legacy_sum (default) or experimental joint_lower_bound. The latter is not production-qualified.',
)

plugin.add_option(
    name='revenue-ops-pair-fee-cap-ppm',
    default='1000',
    description='Per-pair fee budget = max(dest capex, ceil(amount * ppm / 1M)). Decouples per-rebalance fee from capex bootstrap (Iter1, default: 1000 = 0.1% of amount; 0 disables)',
    dynamic=True,
    on_change=_on_rebalance_tuning_change,
)

# Rebalancer flow-fact and EV tuning used by ChannelFlowFacts and the
# retained rebalance engine.
plugin.add_option(
    name='revenue-ops-rebalance-activity-window-seconds',
    default='3600',
    description='Window in seconds over which recent forwarding activity is measured for the activity-recency penalty (default: 3600 = 1 hour)'
)

plugin.add_option(
    name='revenue-ops-rebalance-activity-penalty-coeff',
    default='0.5',
    description='Coefficient applied to the activity-recency penalty when scoring a rebalance candidate (default: 0.5; 0 disables)'
)

plugin.add_option(
    name='revenue-ops-rebalance-activity-penalty-cap-frac',
    default='0.5',
    description='Maximum fraction of score the activity-recency penalty may remove (default: 0.5)'
)

plugin.add_option(
    name='revenue-ops-rebalance-utilization-window-days',
    default='7',
    description='Trailing window in days over which forwarding utilization is measured for utilization-based sizing (default: 7)'
)

plugin.add_option(
    name='revenue-ops-rebalance-utilization-floor',
    default='0.05',
    description='Minimum utilization ratio floor used when normalizing observed flow (default: 0.05)'
)

plugin.add_option(
    name='revenue-ops-rebalance-utilization-ceiling',
    default='1.0',
    description='Maximum utilization ratio ceiling used when normalizing observed flow (default: 1.0)'
)

plugin.add_option(
    name='revenue-ops-rebalance-utilization-min-forwards',
    default='5',
    description='Minimum number of forwards in the utilization window required before utilization-based sizing is trusted (default: 5)'
)

plugin.add_option(
    name='revenue-ops-rebalance-size-tiered-targets',
    default='true',
    description='Enable size-tiered rebalance targets (bucket target amounts by channel capacity percentile instead of a single flat target) (default: true)'
)

plugin.add_option(
    name='revenue-ops-rebalance-size-reference-percentile',
    default='0.5',
    description='Percentile (0.0-1.0) of the channel-capacity distribution used as the reference point for size-tiered target amounts (default: 0.5)'
)

plugin.add_option(
    name='revenue-ops-rebalance-small-channel-band-half-width',
    default='0.15',
    description='Half-width (fraction) of the "small channel" band around the reference percentile (default: 0.15)'
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
    name='revenue-ops-growth-budget-enabled',
    default='false',
    description='Enable dynamic growth budget: base daily budget plus bounded earned/growth credit (default: false)'
)

plugin.add_option(
    name='revenue-ops-growth-budget-earned-fraction',
    default='0.25',
    description='Fraction of trailing net profit that can raise the effective daily budget when growth budget is enabled (default: 0.25)'
)

plugin.add_option(
    name='revenue-ops-growth-budget-experiment-fraction',
    default='0.10',
    description='INERT since v2.17.0: the growth-experiment credit required the retired fleet prior producer; this option is kept only so existing configs load (default: 0.10)'
)

plugin.add_option(
    name='revenue-ops-growth-budget-max-extra-sats',
    default='2000',
    description='INERT since v2.17.0: caps the growth-experiment credit, which can no longer be granted (fleet prior producer retired); kept only so existing configs load (default: 2000)'
)

plugin.add_option(
    name='revenue-ops-growth-budget-hard-ceiling-sats',
    default='10000',
    description='Local hard ceiling for dynamic effective daily budget; fleet hints cannot exceed this (default: 10000)'
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
    name='revenue-ops-drain-fee-discount-max',
    default='0.0',
    description='Max bounded fee discount on stagnant over-local channels (0.0 = disabled)'
)

plugin.add_option(
    name='revenue-ops-node-drain-bias-enabled',
    default='false',
    description='Enable node-liquidity-aware auto-drain-bias on over-local channel fees when the node is source-heavy (default: false)'
)

plugin.add_option(
    name='revenue-ops-node-drain-bias-max',
    default='0.3',
    description='Max node-scaled fee discount applied to over-local channels when node-drain-bias is enabled (0.0-0.5)'
)

plugin.add_option(
    name='revenue-ops-enable-dynamic-htlcmax',
    default='true',
    description='Enable dynamic htlc_max: advertise no more than 85% of live spendable outbound by default, with optional flow-class caps. Default: true'
)

plugin.add_option(
    name='revenue-ops-htlcmax-source-pct',
    default='0.85',
    description='Max HTLC as a fraction of capacity for SOURCE channels when dynamic htlcmax is enabled (0.01-1.0)'
)

plugin.add_option(
    name='revenue-ops-htlcmax-sink-pct',
    default='0.85',
    description='Max HTLC as a fraction of capacity for SINK channels when dynamic htlcmax is enabled (0.01-1.0)'
)

plugin.add_option(
    name='revenue-ops-htlcmax-balanced-pct',
    default='0.85',
    description='Max HTLC as a fraction of capacity for balanced channels when dynamic htlcmax is enabled (0.01-1.0)'
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
    default='standalone',
    description="CSV of askrene layer names passed to v3 router getroutes calls. "
                "Missing layers are silently dropped by askrene. Blank or 'standalone'/'none' "
                "values configure no layers (askrene's own gossip view only)."
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
    global flow_analyzer, fee_controller, rebalancer, database, config, profitability_analyzer, safe_plugin, policy_manager, capex_engine, data_service, econ_shadow, forward_archive_sync

    plugin.log("Initializing cl-revenue-ops plugin...")
    configure_forward_precision(plugin, options.get('revenue-ops-exact-forward-times', False))

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
        _request_fee_adjustment_wake()
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
            _request_fee_adjustment_wake()
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
        fee_authority_enabled=_parse_dynamic_bool(
            "revenue-ops-fee-authority-enabled",
            options.get("revenue-ops-fee-authority-enabled", True),
        ),
        fee_replay_capture_enabled=(
            str(options.get(
                'revenue-ops-fee-replay-capture-enabled',
                'false',
            )).strip().lower() == 'true'
        ),
        rebalance_replay_capture_enabled=(
            str(options.get("revenue-ops-rebalance-replay-capture-enabled", "false")).strip().lower() == "true"
        ),
        hot_channel_protection_enabled=options.get('revenue-ops-hot-channel-protection-enabled', 'true').lower() == 'true',
        hot_channel_protection_override_peers=str(options.get('revenue-ops-hot-channel-protection-override-peers', '') or ''),
        hot_channel_protection_min_velocity=_safe_float_opt('revenue-ops-hot-channel-protection-min-velocity', '0.20'),
        hot_channel_protection_min_marginal_roi=_safe_float_opt('revenue-ops-hot-channel-protection-min-marginal-roi', '0.20'),
        hot_channel_protection_profit_budget_pct=_safe_float_opt('revenue-ops-hot-channel-protection-profit-budget-pct', '0.75'),
        hot_channel_protection_max_chunk_multiplier=_safe_float_opt('revenue-ops-hot-channel-protection-max-chunk-multiplier', '4.0'),
        hot_channel_protection_min_cooldown_hours=_safe_float_opt('revenue-ops-hot-channel-protection-min-cooldown-hours', '1.0'),
        receivable_ratio_target=_safe_float_opt('revenue-ops-receivable-ratio-target', '0.30'),
        receivable_ratio_floor=_safe_float_opt('revenue-ops-receivable-ratio-floor', '0.20'),
        drain_fee_discount_max=_safe_float_opt('revenue-ops-drain-fee-discount-max', '0.0'),
        node_drain_bias_enabled=options.get('revenue-ops-node-drain-bias-enabled', 'false').lower() == 'true',
        node_drain_bias_max=_safe_float_opt('revenue-ops-node-drain-bias-max', '0.3'),
        enable_dynamic_htlcmax=options.get('revenue-ops-enable-dynamic-htlcmax', 'true').lower() == 'true',
        htlcmax_source_pct=_safe_float_opt('revenue-ops-htlcmax-source-pct', '0.85'),
        htlcmax_sink_pct=_safe_float_opt('revenue-ops-htlcmax-sink-pct', '0.85'),
        htlcmax_balanced_pct=_safe_float_opt('revenue-ops-htlcmax-balanced-pct', '0.85'),
        min_fee_ppm=_safe_int('revenue-ops-min-fee-ppm'),
        min_fee_ppm_saturated=_safe_int('revenue-ops-min-fee-ppm-saturated'),
        acquisition_experiment_enabled=_parse_dynamic_bool(
            'revenue-ops-acquisition-experiment-enabled',
            options.get('revenue-ops-acquisition-experiment-enabled', True),
        ),
        max_fee_ppm=_safe_int('revenue-ops-max-fee-ppm'),
        market_fee_mode=options.get('revenue-ops-market-fee-mode', 'undercut').lower(),
        base_fee_policy=options.get('revenue-ops-base-fee-policy', 'off').lower(),
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
            'revenue-ops-rebalance-emergency-local-ratio', '0.20'
        ),
        rebalance_drift_override_ratio=_safe_float_opt(
            'revenue-ops-rebalance-drift-override-ratio', '0.30'
        ),
        rebalance_hold_margin=_safe_float_opt(
            'revenue-ops-rebalance-hold-margin', '0.0'
        ),
        rebalance_value_model=options.get('revenue-ops-rebalance-value-model', 'legacy_sum'),
        pair_fee_cap_ppm=_safe_int_opt(
            'revenue-ops-pair-fee-cap-ppm', '1000'
        ),
        rebalance_activity_window_seconds=_safe_int_opt(
            'revenue-ops-rebalance-activity-window-seconds', '3600'
        ),
        rebalance_activity_penalty_coeff=_safe_float_opt(
            'revenue-ops-rebalance-activity-penalty-coeff', '0.5'
        ),
        rebalance_activity_penalty_cap_frac=_safe_float_opt(
            'revenue-ops-rebalance-activity-penalty-cap-frac', '0.5'
        ),
        rebalance_utilization_window_days=_safe_int_opt(
            'revenue-ops-rebalance-utilization-window-days', '7'
        ),
        rebalance_utilization_floor=_safe_float_opt(
            'revenue-ops-rebalance-utilization-floor', '0.05'
        ),
        rebalance_utilization_ceiling=_safe_float_opt(
            'revenue-ops-rebalance-utilization-ceiling', '1.0'
        ),
        rebalance_utilization_min_forwards=_safe_int_opt(
            'revenue-ops-rebalance-utilization-min-forwards', '5'
        ),
        rebalance_size_tiered_targets=options.get(
            'revenue-ops-rebalance-size-tiered-targets', 'true'
        ).lower() in ('true', '1', 'yes'),
        rebalance_size_reference_percentile=_safe_float_opt(
            'revenue-ops-rebalance-size-reference-percentile', '0.5'
        ),
        rebalance_small_channel_band_half_width=_safe_float_opt(
            'revenue-ops-rebalance-small-channel-band-half-width', '0.15'
        ),
        flow_window_days=_safe_int('revenue-ops-flow-window-days'),
        daily_budget_sats=_safe_int('revenue-ops-daily-budget-sats'),
        growth_budget_enabled=options.get(
            'revenue-ops-growth-budget-enabled', 'false'
        ).lower() in ('true', '1', 'yes'),
        growth_budget_earned_fraction=_safe_float_opt(
            'revenue-ops-growth-budget-earned-fraction', '0.25'
        ),
        growth_budget_experiment_fraction=_safe_float_opt(
            'revenue-ops-growth-budget-experiment-fraction', '0.10'
        ),
        growth_budget_max_extra_sats=_safe_int_opt(
            'revenue-ops-growth-budget-max-extra-sats', '2000'
        ),
        growth_budget_hard_ceiling_sats=_safe_int_opt(
            'revenue-ops-growth-budget-hard-ceiling-sats', '10000'
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
        reservation_timeout_hours=_safe_int('revenue-ops-reservation-timeout-hours'),
        rebalance_router='v3',
        askrene_layers=str(options.get('revenue-ops-askrene-layers', '') or '').strip() or 'standalone',
    )
    # P1-008/P1-009/P1-026: apply the authoritative numeric ranges, then
    # raise any crossed ceiling to the validated floor. Swapping can lower
    # min_fee_ppm below its CRITICAL-02 range when the bounds have different
    # lower limits.
    _validate_startup_config_options(config_kwargs, log=plugin.log)

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
    with config._lock:
        fee_authority_gate.set_enabled(
            config.fee_authority_enabled,
            reason="init",
        )
    
    plugin.log(f"Configuration loaded: "
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

    # =========================================================================
    # STARTUP VERSION GATE
    # A plugin-init failure does not stop lightningd; it refuses only this
    # plugin until the security runtime requirement is met.
    # =========================================================================
    try:
        cln_version = data_service._ensure_getinfo().get("version")
    except Exception as exc:
        raise RuntimeError(
            "cl-revenue-ops could not determine the Core Lightning version; "
            f"v{CLN_VERSION_FLOOR}+ is required"
        ) from exc
    _require_cln_version(cln_version)
    plugin.log(
        f"Core Lightning version check OK "
        f"({cln_version} >= v{CLN_VERSION_FLOOR})"
    )

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
    # A persisted native scope is not proof that this process is attached to
    # the same wallet. Do not downgrade this admission failure to a hydration
    # warning and then run decision loops on silently stale evidence. A live
    # continuity verifier/migration is still required before native activation.
    _native_forward_ingestion_source()
    # Keep the advisory total-cost memo coherent with committed spend-ledger
    # mutations.  Database callbacks are best-effort and run only after a
    # successful commit, so cache maintenance can never affect authorization.
    database.cost_budget_invalidator = _invalidate_total_cost_budget_memo

    # Canonical forward evidence is synchronized independently from the legacy
    # operational forwards table. Construction is fail-isolated so an archive
    # schema problem cannot prevent the revenue plugin from starting.
    try:
        forward_archive_sync = ForwardArchiveSynchronizer(
            safe_plugin.rpc,
            database.forward_archive,
            safe_plugin.log,
        )
    except Exception as e:
        forward_archive_sync = None
        plugin.log(
            f"Forward archive synchronization unavailable: {e}",
            level="warn",
        )

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
    # Explicitly admitted native mode preserves identities across both paths.
    # The unchanged default legacy mode still has coarse-collision/prune-replay
    # limitations; do not use it as proof of safe historical model bootstrap.
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
            inserted, observed = _hydrate_settled_forward_rows(start_time)
            if observed:
                if inserted > 0:
                    plugin.log(f"Hydration complete: inserted {inserted} forwards into local database")
                else:
                    plugin.log(f"Hydration: {observed} observations, no new admitted settlements "
                               "(replayed or unusable evidence)", level='debug')
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
    fee_controller = FeeController(
        safe_plugin,
        config,
        database,
        policy_manager,
        profitability_analyzer,
        fee_authority_gate=fee_authority_gate,
    )
    # Phase 1 shadow (docs/planning/2026-07-12-refactor-phase1-wiring.md):
    # observe-mode intent recording + snapshot preview. Fail-open by
    # contract; inert unless econ_shadow_enabled is set.
    try:
        econ_shadow = EconShadow(safe_plugin, config)
        # PR 3a: policies obtain TTL-cached canonical-snapshot refs from
        # the hub; the provider shares the RPC's assembly path.
        econ_shadow.snapshot_provider = _assemble_econ_snapshot
        # Phase 2 pilot: journal the generic spend lifecycle (all callers
        # of Database.reserve_spend/settle/release) into the econ ledger.
        database.spend_journal = econ_shadow
        # Phase 2H: governor/ledger plumbing for fee broadcasts.
        fee_controller.econ_shadow = econ_shadow
    except Exception as e:
        econ_shadow = None
        plugin.log(f"EconShadow unavailable: {e}", level='warn')
    rebalancer = EVRebalancer(
        safe_plugin, config, database, policy_manager,
    )
    rebalancer.set_profitability_analyzer(profitability_analyzer)
    # Unified liquidity-cost accounting for the retained rebalance path.
    if rebalancer is not None:
        rebalancer.external_liquidity_cost_provider = _non_rebalance_liquidity_cost_components
        rebalancer.global_budget_limit_provider = _total_cost_budget_limit_provider


    capital_efficiency = CapitalEfficiencyAnalyzer(
        profitability_analyzer=profitability_analyzer,
        flow_analyzer=flow_analyzer,
        database=database,
        config=config,
    )
    plugin.log("CapitalEfficiencyAnalyzer initialized")

    # Construct unified capex budget engine
    capex_engine = CapexBudgetEngine(
        profitability_analyzer=profitability_analyzer,
        database=database,
        config=config,
        capital_efficiency=capital_efficiency,
    )
    plugin.log("CapexBudgetEngine initialized")

    # Wire capex engine to all consumers
    if rebalancer is not None:
        rebalancer.set_capex_engine(capex_engine)

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
            data_service=data_service,
            segment_observation_store=segment_observation_store,
            global_budget_limit_provider=_total_cost_budget_limit_provider,
            external_liquidity_cost_provider=_non_rebalance_liquidity_cost_components,
            policy_manager=policy_manager,
        )
        rebalancer.data_service = data_service
        # Phase 2C: shadow-governor on real rebalance reservations.
        try:
            rebalancer.rebalance_engine_v2.econ_shadow = econ_shadow
        except Exception:
            pass
        plugin.log("RebalanceEngine initialized")

    if fee_controller is not None:
        fee_controller.data_service = data_service
    if profitability_analyzer is not None:
        profitability_analyzer.data_service = data_service
    if policy_manager is not None:
        policy_manager.data_service = data_service
    if flow_analyzer is not None:
        flow_analyzer.data_service = data_service

    # Set up periodic background tasks using threading
    # Note: plugin.log() is safe to call from threads in pyln-client
    # We use daemon threads so they don't block shutdown
    
    def flow_analysis_loop():
        """Background loop for flow analysis."""
        # Staggered startup: flow at 30s (was 10s) to avoid thundering herd
        _startup_cfg = config.snapshot() if hasattr(config, 'snapshot') else config
        if shutdown_event.wait(min(30, max(15, _startup_cfg.flow_interval))):
            plugin.log("Flow analysis loop cancelled during startup delay")
            return
        
        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard wraps the ENTIRE iteration (work AND
            # the interval/sleep tail) so no exception can kill the thread.
            try:
                _hb_cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                _record_loop_heartbeat("flow-analysis", interval_seconds=max(15, _hb_cfg_snap.flow_interval))
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
                interval = max(15, cfg_snap.flow_interval)
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
    
    def fee_adjustment_loop(scheduled_work=None):
        """Background loop for fee adjustment."""
        # Staggered startup: fees at 90s (was 60s) to avoid thundering herd
        _startup_cfg = config.snapshot() if hasattr(config, 'snapshot') else config
        if _wait_for_fee_adjustment_wake(
            min(90, max(15, _startup_cfg.fee_interval))
        ):
            plugin.log("Fee adjustment loop cancelled during startup delay")
            return

        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard over the ENTIRE iteration incl. tail.
            try:
                _hb_cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                _record_loop_heartbeat("fee-adjustment", interval_seconds=max(15, _hb_cfg_snap.fee_interval))

                try:
                    plugin.log("Running scheduled fee adjustment...")
                    (scheduled_work or run_fee_adjustment)()
                except (RPCTimeoutError, RPCBreakerOpen) as e:
                    plugin.log(f"RPC degraded in fee adjustment: {e}. Skipping this cycle.", level='warn')
                except Exception as e:
                    plugin.log(f"Error in fee adjustment: {e}", level='error')

                # M-3 FIX: Use config snapshot for interval to avoid mid-loop mutation
                cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                interval = max(15, cfg_snap.fee_interval)
                jitter_seconds = int(interval * 0.2)
                sleep_time = interval + random.randint(-jitter_seconds, jitter_seconds)
                plugin.log(f"Fee adjustment sleeping for {sleep_time}s")

                # Interruptible sleep: timeout, shutdown, or enough material
                # acquisition evidence for a governed lifecycle check.
                if _wait_for_fee_adjustment_wake(sleep_time):
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

    def forward_archive_loop():
        # Synchronize canonical CLN forward evidence without taking action.
        interval_seconds = 15 * 60
        if shutdown_event.wait(60):
            plugin.log(
                "Forward archive loop cancelled during startup delay"
            )
            return

        while not shutdown_event.is_set():
            try:
                _record_loop_heartbeat(
                    "forward-archive",
                    interval_seconds=interval_seconds,
                )
                try:
                    result = forward_archive_sync.sync_once()
                    if result.caught_up:
                        plugin.log(
                            "Forward archive synchronized: "
                            f"created_pages={result.created_pages} "
                            f"updated_pages={result.updated_pages}",
                            level="debug",
                        )
                    else:
                        plugin.log(
                            "Forward archive backlog checkpointed: "
                            f"family={result.backlog_family} "
                            f"created_pages={result.created_pages} "
                            f"updated_pages={result.updated_pages}",
                            level="info",
                        )
                except (RPCTimeoutError, RPCBreakerOpen) as e:
                    plugin.log(
                        f"RPC degraded in forward archive sync: {e}. "
                        "Skipping this cycle.",
                        level="warn",
                    )
                except Exception as e:
                    plugin.log(
                        f"Error in forward archive sync: {e}",
                        level="error",
                    )

                if shutdown_event.wait(interval_seconds):
                    plugin.log(
                        "Forward archive loop stopping due to shutdown signal"
                    )
                    break
            except Exception as e:
                plugin.log(
                    f"Unhandled error in forward-archive loop iteration: {e}",
                    level="error",
                )
                try:
                    plugin.log(
                        f"Traceback: {traceback.format_exc()}",
                        level="debug",
                    )
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
                    break

    def reconciliation_loop():
        """Run reconciliation once per UTC hour, independent of fee authority."""
        while not shutdown_event.is_set():
            try:
                _record_loop_heartbeat(
                    "econ-reconciliation", interval_seconds=3600)
                current = int(time.time())
                slot_started_at = current - (current % 3600)
                try:
                    _run_scheduled_reconciliation(now=current)
                except Exception as e:
                    plugin.log(
                        f"Error in scheduled reconciliation: {e}",
                        level="error",
                    )

                # Anchor sleep to the slot just attempted. If the sweep
                # crosses the next boundary, iterate immediately so that new
                # UTC-hour slot is not silently skipped.
                sleep_time = max(
                    0, slot_started_at + 3600 - int(time.time()))
                if sleep_time == 0:
                    continue
                if shutdown_event.wait(sleep_time):
                    plugin.log(
                        "Reconciliation loop stopping due to shutdown signal")
                    break
            except Exception as e:
                plugin.log(
                    f"Unhandled error in reconciliation loop iteration: {e}",
                    level="error",
                )
                try:
                    plugin.log(
                        f"Traceback: {traceback.format_exc()}", level="debug")
                except Exception:
                    pass
                if shutdown_event.wait(_LOOP_BACKOFF_SECONDS):
                    break
    
    def rebalance_check_loop():
        """Background loop for rebalance checks."""
        # Staggered startup: rebalance at 180s (was 120s) to avoid thundering herd
        _startup_cfg = config.snapshot() if hasattr(config, 'snapshot') else config
        if shutdown_event.wait(min(180, max(15, _startup_cfg.rebalance_interval))):
            plugin.log("Rebalance check loop cancelled during startup delay")
            return
        
        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard over the ENTIRE iteration incl. tail.
            try:
                _hb_cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                _record_loop_heartbeat("rebalance-check", interval_seconds=max(15, _hb_cfg_snap.rebalance_interval))
                try:
                    plugin.log("Running scheduled rebalance check...")
                    run_rebalance_check()
                except (RPCTimeoutError, RPCBreakerOpen) as e:
                    plugin.log(f"RPC degraded in rebalance check: {e}. Skipping this cycle.", level='warn')
                except Exception as e:
                    plugin.log(f"Error in rebalance check: {e}", level='error')

                # M-3 FIX: Use config snapshot for interval to avoid mid-loop mutation
                cfg_snap = config.snapshot() if hasattr(config, 'snapshot') else config
                interval = max(15, cfg_snap.rebalance_interval)
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

    def snapshot_peers_delayed():
        """Record one delayed startup snapshot of connected peers."""
        delay_seconds = 60
        plugin.log(
            f"Startup snapshot: waiting {delay_seconds}s for network connections..."
        )
        if shutdown_event.wait(delay_seconds):
            plugin.log("Startup snapshot cancelled due to shutdown signal")
            return

        _record_loop_heartbeat("startup-snapshot", one_shot=True)
        try:
            _snapshot_peers_once()
        except Exception as e:
            plugin.log(f"Startup snapshot failed: {e}", level="error")
            try:
                plugin.log(f"Traceback: {traceback.format_exc()}", level="debug")
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
        try:
            _reconcile_closure_resolutions()
        except Exception as e:
            plugin.log(f"Error reconciling closure resolutions: {e}", level='warn')

        while not shutdown_event.is_set():
            # DD5 / P1-010: canonical guard over the ENTIRE iteration. This loop
            # is inverted (sleep-tail at the TOP), so the guard wraps it too.
            try:
                _record_loop_heartbeat("financial-snapshot", interval_seconds=SNAPSHOT_INTERVAL)
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

                # Daily closure-resolution sweep: accumulate post-close HTLC
                # sweep fees for closures whose outputs were still unresolved
                # at close-detection time.
                try:
                    _reconcile_closure_resolutions()
                except Exception as e:
                    plugin.log(f"Error reconciling closure resolutions: {e}", level='warn')
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
    threading.Thread(
        target=fee_adjustment_loop,
        args=(_run_scheduled_fee_adjustment,),
        daemon=True,
        name="fee-adjustment",
    ).start()
    threading.Thread(
        target=reconciliation_loop,
        daemon=True,
        name="econ-reconciliation",
    ).start()
    if forward_archive_sync is not None:
        threading.Thread(
            target=forward_archive_loop,
            daemon=True,
            name="forward-archive",
        ).start()
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



def _run_scheduled_fee_adjustment():
    denial = _fee_authority_denial("scheduled_fee_cycle")
    if denial is not None:
        return denial
    return run_fee_adjustment()


def _run_scheduled_reconciliation(now=None):
    """Run the read/ledger-only sweep without fee authority coupling."""
    if econ_shadow is None or not econ_shadow.enabled():
        return None
    current = int(time.time()) if now is None else int(now)
    try:
        return econ_shadow.maybe_run_reconciliation(database, current)
    except Exception as e:
        plugin.log(f"reconciliation sweep skipped: {e}", level="debug")
        return None


def _fee_evidence_guard_or_noop():
    """Use optional process-local evidence synchronization fail-open."""
    return fail_open_fee_evidence_guard(
        lambda: econ_shadow.fee_evidence_guard())


def run_fee_adjustment():
    """Run one complete Python fee cycle under the shared authority lease."""
    with fee_authority_gate.execution_lease(
        "fee_adjustment"
    ) as denial:
        if denial is not None:
            return denial
        return _run_fee_adjustment_authorized()


def _run_fee_adjustment_authorized():
    return _run_fee_adjustment_evidence_locked()


def _run_fee_adjustment_evidence_locked():
    """
    Module 2: DTS+PID Fee Controller (Dynamic Pricing)

    Adjust channel fees using DTS+PID optimization.
    """
    if fee_controller is None:
        plugin.log("Fee controller not initialized", level='error')
        return []
    
    try:
        with _fee_evidence_guard_or_noop():
            adjustments = fee_controller.adjust_all_fees()
            plugin.log(
                f"Fee adjustment complete: "
                f"{len(adjustments)} channels adjusted")

            # Phase 1 shadow: record this cycle's decisions as typed intents.
            # Fail-open by contract — a shadow failure must never affect fees.
            # Phase 2H: skipped when fee broadcasts are GOVERNED — each
            # broadcast then records its own pre-authorization trail, and
            # double-recording would skew the completeness detector.
            try:
                _cfg_gov = config.snapshot() if config else None
                _fees_governed = getattr(
                    _cfg_gov, "econ_governor_fees_enabled", False) is True
                if econ_shadow is not None and econ_shadow.enabled() \
                        and not _fees_governed:
                    econ_shadow.record_fee_intents(
                        adjustments, int(time.time()))
            except Exception as _shadow_err:
                plugin.log(
                    f"econ shadow skipped: {_shadow_err}", level='debug')

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


_dashboard_push_state: Dict[str, Any] = {"ts": 0}
_DASHBOARD_PUSH_MIN_INTERVAL = 1500  # seconds; ~one rebuild per fee cycle


def _push_dashboard_to_datastore() -> None:
    """Push 30-day dashboard snapshot to datastore.

    Throttled: revenue_dashboard re-runs uncached 30d aggregates (pnl, roc,
    bleeder scan), and this push used to rebuild it from scratch on every
    fee cycle purely to refresh a telemetry snapshot.
    """
    global safe_plugin, profitability_analyzer, database, data_service
    if safe_plugin is None or profitability_analyzer is None or database is None:
        return
    now = int(time.time())
    if now - int(_dashboard_push_state.get("ts", 0) or 0) < _DASHBOARD_PUSH_MIN_INTERVAL:
        return
    try:
        dashboard = revenue_dashboard(plugin, window_days=30)
        if isinstance(dashboard, dict) and "error" not in dashboard:
            if data_service:
                data_service.datastore_push(["revenue", "dashboard"], dashboard)
            _dashboard_push_state["ts"] = now
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
        # The v2 engine selects AND executes inside find_rebalance_candidates
        # (engine.run_cycle); its return is always [] by design. E-4.5: the
        # old `for candidate: execute_rebalance(candidate)` loop here was
        # dead code, and the "N candidates found" log always reported 0.
        rebalancer.find_rebalance_candidates()
        decision = (
            rebalancer.get_last_decision_summary()
            if hasattr(rebalancer, "get_last_decision_summary") else {}
        )
        plugin.log(
            "Rebalance check complete: "
            f"action={decision.get('action', 'unknown')} "
            f"reason={decision.get('reason', 'unknown')}"
        )

    except Exception as e:
        plugin.log(f"Rebalance check failed: {e}", level='error')
        raise


# =============================================================================
# Phase C (operator-surface reduction 2026-08-01): deprecated-alias plumbing.
#
# The retained dispatcher names are the primary operator names. Every
# pre-existing name keeps working as a thin forwarding
# alias whose dict response gains ONLY an additive `deprecation` field; the
# alias window ends 2026-09-05. See
# docs/audits/OPERATOR_SURFACE_REDUCTION_2026-08-01.md (§1, §4) and the
# 2026-08-01 announcement in
# docs/refactor/phase0/contract-compatibility-policy.md.
# =============================================================================

_SURFACE_REDUCTION_DOC = "docs/audits/OPERATOR_SURFACE_REDUCTION_2026-08-01.md"
_RPC_ALIAS_REMOVAL_DATE = "2026-09-05"


def _alias_deprecation_notice(new_name: str) -> str:
    """Deprecation text for an old RPC name renamed to a dispatcher form."""
    return (
        f"renamed to '{new_name}' — this name is scheduled for removal "
        f"{_RPC_ALIAS_REMOVAL_DATE}; see {_SURFACE_REDUCTION_DOC}"
    )


def _removal_deprecation_notice() -> str:
    """Deprecation text for the ignore trio (removed without a rename)."""
    return (
        f"scheduled for removal {_RPC_ALIAS_REMOVAL_DATE} with no replacement "
        f"— revenue-policy actions cover it; see {_SURFACE_REDUCTION_DOC}"
    )


def _deprecated_alias(result: Any, new_name: str) -> Any:
    """ADD the `deprecation` field to an old-name RPC response.

    Additive only: never changes the response shape otherwise, and never
    mutates the helper's dict in place (helpers may return shared state).
    Dispatcher responses never pass through here, so the new names stay
    deprecation-free.
    """
    if isinstance(result, dict) and "deprecation" not in result:
        out = dict(result)
        out["deprecation"] = _alias_deprecation_notice(new_name)
        return out
    return result


def _deprecated_removal(result: Any) -> Any:
    """ADD the `deprecation` field to a response of an RPC slated for
    removal with no replacement (the ignore trio)."""
    if isinstance(result, dict) and "deprecation" not in result:
        out = dict(result)
        out["deprecation"] = _removal_deprecation_notice()
        return out
    return result


def _dispatch_subcommand(plugin: Plugin, rpc_name: str, arg_label: str,
                         table: Dict[str, Any], subcommand: Any,
                         kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Shared dispatcher core: route `<rpc_name> <subcommand>` onto the same
    helper the old standalone method calls. Unknown subcommands return the
    list of valid ones instead of raising."""
    key = str(subcommand or "").strip().lower()
    handler = table.get(key)
    if handler is None:
        valid = sorted(table)
        return {
            "error": (
                f"Unknown {arg_label} '{subcommand}' for {rpc_name}. "
                f"Valid {arg_label}s: {', '.join(valid)}"
            ),
            f"valid_{arg_label}s": valid,
        }
    try:
        return handler(plugin, **kwargs)
    except TypeError as e:
        # Bad passthrough kwargs bind-fail here; the helpers themselves
        # catch their own runtime errors and return error dicts.
        return {"error": f"Invalid arguments for '{rpc_name} {key}': {e}"}


def _rpc_rebalance_cycle(plugin: Plugin, max_candidates: int = 20) -> Dict[str, Any]:
    """Run one automatic rebalance cycle immediately and return debug state."""
    if rebalancer is None:
        return {"error": "Rebalancer not initialized"}
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


@plugin.method("revenue-rebalance-cycle")
def revenue_rebalance_cycle(plugin: Plugin, max_candidates: int = 20) -> Dict[str, Any]:
    """Deprecated alias for 'revenue-cycle rebalance' (removal 2026-09-05)."""
    return _deprecated_alias(
        _rpc_rebalance_cycle(plugin, max_candidates=max_candidates),
        "revenue-cycle rebalance",
    )


# =============================================================================
# RPC METHODS - Exposed to lightning-cli
# =============================================================================


@plugin.method("revenue-fee-authority-status")
def revenue_fee_authority_status(plugin: Plugin) -> Dict[str, Any]:
    """Return the current in-process Python fee-authority state."""
    status = fee_authority_gate.snapshot()
    return {
        "schema": "revenue_ops_fee_authority/v1",
        "enabled": status.enabled,
        "generation": status.generation,
        "transitioned_at": status.transitioned_at,
        "observed_at": int(time.time()),
        "reason": status.reason,
    }


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
        acquisition_history_raw = database.get_recent_acquisition_experiments(limit=10)
        acquisition_history = (
            acquisition_history_raw
            if isinstance(acquisition_history_raw, list)
            else []
        )
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
        "acquisition_experiments": acquisition_history,
    }


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
        # E-4.5: report the profit gate that is actually ENFORCED (the
        # sats-EV hold margin). rebalance_min_profit was echoed here while
        # being enforced nowhere — a misleading operator surface.
        "rebalance_hold_margin_sats": cfg.rebalance_hold_margin,
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
        total_liquidity_spent = int(total_budget.get("actual_spent_sats", int(daily_spent)) or 0)
        total_liquidity_reserved = int(total_budget.get("reserved_sats", int(daily_reserved)) or 0)
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
                f"Unified liquidity budget exhausted: spent {total_liquidity_spent}, "
                f"reserved {total_liquidity_reserved}, budget {daily_budget}"
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

    engine = getattr(rebalancer, "rebalance_engine_v2", None)
    if engine is not None and hasattr(engine, "get_last_cycle_debug"):
        try:
            cycle_limit = max_candidates if max_candidates > 0 else 10
            result["last_cycle"] = engine.get_last_cycle_debug(
                max_candidates=cycle_limit,
            )
        except Exception as e:
            result["last_cycle"] = {"error": str(e)}

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


def _fee_debug_controller_block(v2_state: Dict[str, Any],
                                fee_state: Dict[str, Any],
                                is_sleeping: Any,
                                sleep_until: Any) -> Dict[str, Any]:
    """ADR-001: per-channel controller components for revenue-fee-debug.
    Reads only persisted state (v2_state_json); purely additive output.
    PID terms reported pre-capacity-scale."""
    pid = fee_state.get("pid_state") or v2_state.get("pid_state") or {}

    def _f(key, default=0.0):
        try:
            return float(pid.get(key, default))
        except (TypeError, ValueError):
            return default

    kp, ki, kd = _f("kp", 2.0), _f("ki", 0.1), _f("kd", 0.0)
    ewma_error = _f("ewma_error")
    integral_error = _f("integral_error")
    return {
        "algorithm_version": v2_state.get("algorithm_version",
                                          "dts_pid_v1"),
        "pid": {
            "kp": kp, "ki": ki, "kd": kd,
            "ewma_error": ewma_error,
            "integral_error": integral_error,
            "integral_clamp": _f("integral_clamp", 3.0),
            "p_term_unscaled": kp * ewma_error,
            "i_term_unscaled": ki * integral_error,
            "d_term": 0.0,
        },
        "stages": {
            "cooldown": {
                "sleeping": bool(is_sleeping),
                "sleep_until": int(sleep_until or 0),
            },
        },
    }


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

    now = int(time.time())
    # ADR-001 (PR 4): the debug surface names the REAL controller and
    # its staged-constraint contract — never multiplicative factors.
    try:
        last_cycle_decision = fee_controller.get_last_decision_summary()
    except Exception:
        last_cycle_decision = None
    result = {
        "timestamp": now,
        "controller_contract": {
            "algorithm": "dts_pid_v1",
            "stage_order": ["rails", "rate_limit", "deadband", "cooldown"],
            "adr": "docs/refactor/adr/ADR-001-dts-pid-fee-controller.md",
        },
        "last_cycle_decision": last_cycle_decision,
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
            # ADR-001 (PR 4): real controller components. PID terms are
            # pre-capacity-scale (runtime scales gains by
            # 1/log2(capacity/1e6+2); multiplier = 1.5**(p+i), clamped
            # to [0.5, 2.0]). Cooldown/deadband stage state is the
            # sleep/hysteresis surface above (is_sleeping/skip_reason).
            "controller": _fee_debug_controller_block(
                v2_state, _fee_state, is_sleeping, sleep_until),
        })
        result["summary"]["total"] += 1

    return result


def _rpc_fee_cycle(plugin: Plugin) -> Dict[str, Any]:
    """Run one fee adjustment cycle immediately."""
    with fee_authority_gate.execution_lease(
        "revenue-fee-cycle"
    ) as denial:
        if denial is not None:
            return {
                "ok": False,
                "adjusted_channels": 0,
                "fee_debug": {},
                **denial,
            }
        adjustments = run_fee_adjustment() or []
        return {
            "ok": True,
            "adjusted_channels": len(adjustments),
            "fee_debug": revenue_fee_debug(plugin),
        }


@plugin.method("revenue-fee-cycle")
def revenue_fee_cycle(plugin: Plugin) -> Dict[str, Any]:
    """Deprecated alias for 'revenue-cycle fees' (removal 2026-09-05)."""
    return _deprecated_alias(_rpc_fee_cycle(plugin), "revenue-cycle fees")


def _rpc_analyze(plugin: Plugin, channel_id: Optional[str] = None) -> Dict[str, Any]:
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


@plugin.method("revenue-analyze")
def revenue_analyze(plugin: Plugin, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """Deprecated alias for 'revenue-cycle flow' (removal 2026-09-05)."""
    return _deprecated_alias(
        _rpc_analyze(plugin, channel_id=channel_id), "revenue-cycle flow"
    )


def _rpc_wake_all(plugin: Plugin) -> Dict[str, Any]:
    """
    Wake all sleeping channels immediately.

    Use this after changing fee_interval or when you need to force
    all channels to re-evaluate their fees on the next cycle.

    Usage: lightning-cli revenue-wake-all
    """
    with fee_authority_gate.execution_lease(
        "revenue-wake-all"
    ) as denial:
        if denial is not None:
            return {
                "channels_woken": 0,
                "message": "Fee authority disabled",
                **denial,
            }
        if fee_controller is None:
            return {"error": "Plugin not fully initialized"}

        woken = fee_controller.wake_all_sleeping_channels()
        return {
            "status": "ok",
            "channels_woken": woken,
            "message": f"Woke {woken} sleeping channel(s). They will be evaluated on the next fee cycle."
        }


@plugin.method("revenue-wake-all")
def revenue_wake_all(plugin: Plugin) -> Dict[str, Any]:
    """Deprecated alias for 'revenue-cycle all' (removal 2026-09-05)."""
    return _deprecated_alias(_rpc_wake_all(plugin), "revenue-cycle all")


@plugin.method("revenue-set-fee")
def revenue_set_fee(plugin: Plugin, channel_id: str = None, fee_ppm: int = None, force: bool = False) -> Dict[str, Any]:
    """Manually set a fee while holding the shared authority lease."""
    with fee_authority_gate.execution_lease(
        "revenue-set-fee"
    ) as denial:
        if denial is not None:
            return {
                "error": "Fee authority disabled",
                **denial,
            }
        return _revenue_set_fee_authorized(
            plugin,
            channel_id=channel_id,
            fee_ppm=fee_ppm,
            force=force,
        )


def _revenue_set_fee_authorized(
    plugin: Plugin,
    channel_id: str = None,
    fee_ppm: int = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Manually set fee for a channel.

    Usage: lightning-cli revenue-set-fee channel_id fee_ppm [force=false]
    """
    if channel_id is None or fee_ppm is None:
        return {"error": "usage: revenue-set-fee channel_id fee_ppm [force=false]"}
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
        result = fee_controller.set_channel_fee(channel_id, fee_ppm, manual=True, enforce_limits=(not force))
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
                      from_channel: str = None,
                      to_channel: str = None,
                      amount_sats: int = None,
                      max_fee_sats: Optional[int] = None,
                      force: bool = False) -> Dict[str, Any]:
    """
    Manually trigger a rebalance with profit/budget constraints.

    Usage: lightning-cli revenue-rebalance from_channel to_channel amount_sats [max_fee_sats] [force=false]
    """
    if not from_channel or not to_channel or amount_sats is None:
        return {"error": "usage: revenue-rebalance from_channel to_channel amount_sats [max_fee_sats] [force=false]"}
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


@plugin.method("revenue-forward-history")
def revenue_forward_history(
    plugin: Plugin,
    history_since: int,
    history_until: int,
    channel_id: Optional[str] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """Return bounded canonical forward evidence for UTC-midnight bounds."""
    if database is None or not hasattr(database, "forward_archive"):
        return {"error": "Forward archive not initialized"}
    try:
        if any(isinstance(value, bool) for value in (
            history_since, history_until, limit
        )):
            raise ValueError("history bounds and limit must be integers")
        if channel_id is not None and not isinstance(channel_id, str):
            raise ValueError("channel_id must be a string")
        start = int(history_since)
        end = int(history_until)
        bounded_limit = int(limit)
        if start % 86400 or end % 86400:
            raise ValueError(
                "history bounds must be UTC-midnight aligned"
            )
        if end <= start:
            raise ValueError(
                "history_until must be greater than history_since"
            )
        if end - start > 400 * 86400:
            raise ValueError("history window exceeds 400 days")
        if not 1 <= bounded_limit <= 5000:
            raise ValueError("limit must be between 1 and 5000")
        normalized_channel = (
            normalize_scid(channel_id) if channel_id else None
        )
        return database.forward_archive.history(
            start,
            end,
            normalized_channel,
            bounded_limit,
        )
    except (TypeError, ValueError, ForwardArchiveError) as exc:
        return {"error": str(exc)}


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


def _rpc_ignore(plugin: Plugin, peer_id: str, reason: str = "manual", **kwargs) -> Dict[str, Any]:
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


@plugin.method("revenue-ignore")
def revenue_ignore(plugin: Plugin, peer_id: str, reason: str = "manual", **kwargs) -> Dict[str, Any]:
    """DEPRECATED (removal 2026-09-05, no replacement — revenue-policy covers it)."""
    return _deprecated_removal(_rpc_ignore(plugin, peer_id, reason=reason, **kwargs))


def _rpc_unignore(plugin: Plugin, peer_id: str, **kwargs) -> Dict[str, Any]:
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


@plugin.method("revenue-unignore")
def revenue_unignore(plugin: Plugin, peer_id: str, **kwargs) -> Dict[str, Any]:
    """DEPRECATED (removal 2026-09-05, no replacement — revenue-policy covers it)."""
    return _deprecated_removal(_rpc_unignore(plugin, peer_id, **kwargs))


def _rpc_list_ignored(plugin: Plugin) -> Dict[str, Any]:
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


@plugin.method("revenue-list-ignored")
def revenue_list_ignored(plugin: Plugin) -> Dict[str, Any]:
    """DEPRECATED (removal 2026-09-05, no replacement — revenue-policy covers it)."""
    return _deprecated_removal(_rpc_list_ignored(plugin))



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
            # Preserve the policy summary for compatibility, but fulfil the
            # documented summary contract with the canonical dashboard P&L
            # and live channel-state coverage as well.
            all_policies = policy_manager.get_all_policies()
            
            strategy_counts = {}
            rebalance_counts = {}
            for p in all_policies:
                s = p.strategy.value
                r = p.rebalance_mode.value
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
                rebalance_counts[r] = rebalance_counts.get(r, 0) + 1
            
            dashboard = revenue_dashboard(plugin, window_days=30)
            channel_states = database.get_all_channel_states() or []
            channel_ids = set()
            state_counts = {}
            for state in channel_states:
                channel_id = (
                    state.get("channel_id")
                    or state.get("short_channel_id")
                )
                if channel_id:
                    channel_ids.add(str(channel_id))
                state_name = str(state.get("state") or "unknown")
                state_counts[state_name] = state_counts.get(state_name, 0) + 1

            result = {
                "type": "summary",
                "policies": {
                    "total": len(all_policies),
                    "by_strategy": strategy_counts,
                    "by_rebalance_mode": rebalance_counts
                },
                "channels": {
                    "total": len(channel_ids),
                    "by_state": state_counts,
                },
                "generated_at": int(time.time())
            }
            if "error" in dashboard:
                result["financial_health"] = {"error": dashboard["error"]}
                result["period"] = {}
                result["warnings"] = []
                result["bleeder_count"] = 0
            else:
                result["financial_health"] = dashboard.get(
                    "financial_health", {})
                result["period"] = dashboard.get("period", {})
                result["warnings"] = dashboard.get("warnings", [])
                result["bleeder_count"] = dashboard.get(
                    "bleeder_count", 0)
            return result
        
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


@plugin.method("revenue-profile-preview")
def revenue_profile_preview(plugin: Plugin,
                            profile: Optional[str] = None) -> Dict[str, Any]:
    """READ-ONLY risk-profile preview/diff (gap-closure Phase D, PR 8).

    With `profile`: what selecting it would change at the next restart —
    per-key current vs profile value, explicit-override precedence
    blocks, contradiction pre-check. Without: the observe-only
    comparison of every profile against current effective config.
    Mutates nothing; activation is always the separate
    `revenue-config set risk_profile <name>` + restart. Internal
    diagnostic — no compatibility promise yet.
    """
    try:
        if config is None or database is None:
            return {"error": "Plugin not fully initialized"}
        from modules.risk_profiles import (
            PROFILE_BUNDLES,
            preview_all,
            preview_profile,
        )
        overrides = database.get_all_config_overrides() or {}
        explicit = set(overrides) - {"risk_profile"}
        bundle_keys = set()
        for bundle in PROFILE_BUNDLES.values():
            bundle_keys |= set(bundle)
        current = {key: getattr(config, key, None) for key in bundle_keys}
        active = str(getattr(config, "risk_profile", "custom") or "custom")
        persisted = str(overrides.get("risk_profile", active)
                        or active).strip().lower()
        header = {
            "active_profile": active,
            "persisted_profile": persisted,
            "pending_restart": persisted != active,
            "explicit_override_keys": sorted(explicit & bundle_keys),
        }
        if profile is not None:
            header["preview"] = preview_profile(current, profile, explicit)
        else:
            header["comparison"] = preview_all(current, explicit)
        return header
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
            # Public runtime controls include the daily/weekly envelopes and
            # growth-budget inputs consumed by the unified budget report.
            # Clearing for every successful public update is cheap and avoids
            # a fragile second list of budget-affecting keys.
            _invalidate_total_cost_budget_memo()
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

        result = {
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

        return result
    except Exception as e:
        plugin.log(f"Error generating revenue dashboard: {e}", level='error')
        return {"error": str(e)}


def _assemble_econ_snapshot() -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Assemble the canonical EconomicSnapshot wire dict from the live
    caches. Shared by the revenue-econ-snapshot RPC and the shadow hub's
    snapshot_provider (PR 3a policy adoption). Raises only if the
    channel read fails — every other input degrades with a declared
    approximation."""
    channels = []
    if data_service is not None:
        channels = (data_service.get_peer_channels() or {}).get(
            "channels", []) or []
    profitability = {}
    try:
        if profitability_analyzer is not None:
            profitability = profitability_analyzer.analyze_all_channels(
                force=False) or {}
    except Exception:
        profitability = {}
    budget = {}
    try:
        cfg_snap = config.snapshot() if config else None
        if database is not None and cfg_snap is not None:
            status = database.get_budget_status(
                int(time.time()) - 24 * 3600) or {}
            budget = {
                "cap_sats": int(getattr(cfg_snap, "daily_budget_sats", 0)
                                or 0),
                "reserved_sats": int(status.get("reserved", 0) or 0),
                "spent_sats": int(status.get("spent", 0) or 0),
            }
    except Exception:
        budget = {}
    return econ_shadow.build_snapshot_preview(
        channels=channels,
        profitability=profitability,
        budget=budget,
        now=int(time.time()),
        receivable_ratio_target=float(getattr(
            config.snapshot() if config else object(),
            "receivable_ratio_target", 0.0) or 0.0),
    )


@plugin.method("revenue-econ-snapshot")
def revenue_econ_snapshot(plugin: Plugin) -> Dict[str, Any]:
    """READ-ONLY Phase 1 preview of the canonical EconomicSnapshot
    (docs/planning/2026-07-12-refactor-phase1-wiring.md).

    Assembled on demand from existing caches; placeholder fields are
    declared in `approximations`. Requires econ_shadow_enabled. Internal
    diagnostic — no compatibility promise yet.
    """
    try:
        if econ_shadow is None or not econ_shadow.enabled():
            return {"enabled": False,
                    "hint": "revenue-config set econ_shadow_enabled true"}
        try:
            wire, approximations = _assemble_econ_snapshot()
        except Exception as e:
            return {"enabled": True, "snapshot": None,
                    "approximations": [f"channel read failed: {e}"]}
        intents_durable = None
        try:
            _ledger = econ_shadow.ledger_for_reconciliation()
            if _ledger is not None:
                intents_durable = _ledger.count_events("intent_proposed")
        except Exception:
            pass
        return {
            "enabled": True,
            "snapshot": wire,
            "approximations": approximations,
            # Session counter (resets on reload) + durable ledger count.
            "intents_recorded_total": econ_shadow.intents_recorded_total,
            "intents_ledger_total": intents_durable,
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@plugin.method("revenue-econ-reconcile")
def revenue_econ_reconcile(plugin: Plugin, apply: bool = False,
                           stale_after_seconds: int = 3600,
                           history_since: int = 0,
                           history_until: int = 0,
                           history_limit: int = 1000) -> Dict[str, Any]:
    """Reconcile the econ ledger against spend_reservations truth
    (Phase 2 pilot B). Dry-run by default; apply=true appends
    reconciliation_completed events to econ_ledger.db. NEVER writes
    revenue_ops.db. Explicit history bounds add durable scheduled-run
    evidence for the half-open UTC epoch range. Requires
    econ_shadow_enabled."""
    try:
        from modules import econ_reconcile
        if econ_shadow is None or not econ_shadow.enabled():
            return {"enabled": False,
                    "hint": "revenue-config set econ_shadow_enabled true"}
        ledger = econ_shadow.ledger_for_reconciliation()
        if ledger is None:
            return {"enabled": True, "error": "ledger unavailable"}
        since_value = int(history_since)
        until_value = int(history_until)
        limit_value = int(history_limit)
        history_requested = bool(since_value or until_value)
        if history_requested and not (since_value and until_value):
            return {
                "enabled": True,
                "error": "history_since and history_until are both required",
            }
        if history_requested and until_value <= since_value:
            return {
                "enabled": True,
                "error": "history_until must be greater than history_since",
            }
        if not (1 <= limit_value <= 10_000):
            return {
                "enabled": True,
                "error": "history_limit must be between 1 and 10000",
            }
        if history_requested and (
                since_value % 3600 or until_value % 3600):
            return {
                "enabled": True,
                "error": "history bounds must align to UTC-hour epochs",
            }
        if history_requested \
                and (until_value - since_value) // 3600 > 10_000:
            return {
                "enabled": True,
                "error": (
                    "history range cannot exceed 10000 UTC-hour slots"),
            }
        result = {"enabled": True}
        if database is None:
            result["error"] = "database unavailable"
        else:
            observed_now = int(time.time())
            db_states = database.get_spend_reservation_states()
            report = econ_reconcile.reconcile(
                ledger, db_states, now=observed_now,
                stale_after_seconds=max(60, int(stale_after_seconds)))
            result.update({
                "checked": report.checked,
                "matched": report.matched,
                "divergences": [
                    {
                        "kind": d.kind,
                        "key": d.key,
                        "ledger_reserved_msat": d.ledger_reserved_msat,
                        "db_status": d.db_status,
                        "db_reserved_sats": d.db_reserved_sats,
                        "quarantined": d.resolution is None,
                        "details": d.details,
                    }
                    for d in report.divergences
                ],
            })
            try:
                with econ_shadow.fee_evidence_guard():
                    fee_since, fee_until = (
                        econ_reconcile.fee_change_query_bounds(observed_now)
                    )
                    recent_changes = database.get_fee_changes_between(
                        fee_since, fee_until)
                    result["fee_intent_completeness"] = (
                        econ_reconcile.fee_intent_completeness(
                            ledger, recent_changes, now=observed_now))
            except Exception as completeness_err:
                result["fee_intent_completeness"] = {
                    "status": "error", "error": str(completeness_err)}
            if apply:
                result["applied"] = econ_reconcile.apply(
                    ledger, report, now=observed_now)
        if history_requested:
            history_data = ledger.reconciliation_runs(
                since_at=since_value,
                until_at=until_value,
                limit=limit_value,
            )
            runs = history_data["runs"]
            result_counts = {
                name: sum(1 for run in runs if run["result"] == name)
                for name in (
                    "clean", "divergence_found", "failed", "skipped",
                    "incomplete",
                )
            }
            expected_slots = list(range(
                since_value, until_value, 3600))
            expected_runs = len(expected_slots)
            started = len(runs)
            completed = sum(
                1 for run in runs if run["completed_at"] is not None)
            slot_counts = {}
            for run in runs:
                slot = int(run["slot_started_at"])
                slot_counts[slot] = slot_counts.get(slot, 0) + 1
            covered_slots = sorted(
                slot for slot in slot_counts if slot in expected_slots)
            missing_slots = [
                slot for slot in expected_slots if slot not in slot_counts]
            duplicate_slots = sorted(
                slot for slot, count in slot_counts.items() if count > 1)
            unexplained_values = [
                run.get("unexplained_divergence_count") for run in runs]
            unexplained_unknown = any(
                value is None for value in unexplained_values)
            unexplained = (
                None if unexplained_unknown
                else sum(int(value) for value in unexplained_values)
            )
            fee_intent_complete = (
                started == expected_runs
                and all(run.get("fee_intent_completeness") == "ok"
                        for run in runs)
            )
            truncated = bool(history_data["truncated"])
            complete = (
                not truncated
                and not missing_slots
                and not duplicate_slots
                and started == expected_runs
                and completed == expected_runs
                and result_counts["failed"] == 0
                and result_counts["skipped"] == 0
                and result_counts["incomplete"] == 0
            )
            all_clean = (
                complete
                and result_counts["clean"] == expected_runs
                and result_counts["divergence_found"] == 0
                and fee_intent_complete
                and unexplained_unknown is False
                and unexplained == 0
            )
            result["history"] = {
                "runs": runs,
                "truncated": truncated,
                "summary": {
                    "since": since_value,
                    "until": until_value,
                    "expected_runs": expected_runs,
                    "started": started,
                    "completed": completed,
                    "covered_slots": len(covered_slots),
                    "missing_slots": missing_slots,
                    "duplicate_slots": duplicate_slots,
                    **result_counts,
                    "unexplained_divergence_count": unexplained,
                    "unexplained_divergence_count_unknown": (
                        unexplained_unknown),
                    "fee_intent_complete": fee_intent_complete,
                    "truncated": truncated,
                    "complete": complete,
                    "all_clean": all_clean,
                },
            }
        return result
    except Exception as e:
        return {"enabled": True, "error": str(e)}


_econ_cycle_seq = 0


@plugin.method("revenue-econ-cycle")
def revenue_econ_cycle(plugin: Plugin) -> Dict[str, Any]:
    """READ-ONLY Workstream H shadow cycle: one collection pass, pure
    intent generation from retained fee and rebalance evidence, BATCH arbitration under
    the J3 ladder. No execution authority — publishes and ledgers only.
    Requires econ_shadow_enabled. Internal diagnostic."""
    global _econ_cycle_seq
    try:
        if econ_shadow is None or not econ_shadow.enabled():
            return {"enabled": False,
                    "hint": "revenue-config set econ_shadow_enabled true"}
        engine = getattr(rebalancer, "rebalance_engine_v2", None) \
            if rebalancer is not None else None
        if engine is None:
            return {"enabled": True, "error": "rebalance engine unavailable"}
        from modules.econ_cycle import run_shadow_cycle
        _econ_cycle_seq += 1
        result = run_shadow_cycle(
            rebalance_engine=engine, econ_shadow=econ_shadow,
            now=int(time.time()), cycle_seq=_econ_cycle_seq)
        if result is None:
            return {"enabled": True, "error": "shadow cycle failed open"}
        return {"enabled": True, "cycle": result}
    except Exception as e:
        return {"enabled": True, "error": str(e)}


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
            decision = rebalancer.get_last_decision_summary()
            active_jobs = rebalancer.job_manager.active_job_count if hasattr(rebalancer, 'job_manager') else 0
            result["rebalancer"] = {
                "last_action": decision.get("action"),
                "last_reason": decision.get("reason"),
                "active_jobs": active_jobs,
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

    Called by retained runtime cycles before reading dynamic options. CLN stores setconfig
    values persistently but pyln-client's plugin.options dict is only
    populated at init. This function bridges the gap.
    """
    try:
        # Use the timeout-protected proxy so a hung lightningd response cannot
        # otherwise stall the loop indefinitely.
        rpc = safe_plugin.rpc if safe_plugin is not None else plugin.rpc
        all_configs = rpc.listconfigs()
        configs = all_configs.get("configs", {})
    except Exception:
        return

    if config:
        for _opt, _field, _cast in (
            # E-2 (2026-07 econ audit): class-aware saturated/source min-fee
            # floor, dynamic so fee-band decompression is tunable live.
            ("revenue-ops-min-fee-ppm-saturated", "min_fee_ppm_saturated", "int"),
            (
                "revenue-ops-acquisition-experiment-enabled",
                "acquisition_experiment_enabled",
                "bool",
            ),
        ):
            _val = configs.get(_opt, {}).get("value_str", "")
            # C-1(b) (2026-07-08 audit): the blanket "skip if empty" guard
            # made str-cast fields unclearable via setconfig — an operator
            # emptying the value would never see it take effect. bool/int/
            # float casts still skip on empty (an unset numeric/bool option
            # has no meaningful "empty" value to apply).
            if not _val and _cast != "str":
                continue
            # Runtime precedence: an active revenue-config DB override wins
            # over the listconfigs view. Without this guard the refresh loop
            # stomps a `revenue-config set` value with the (possibly empty)
            # setconfig/file value every cycle — observed live on nexus-01
            # 2026-07-08: a str option cleared 1 minute after being set.
            # Operators drop the override with `revenue-config reset` if
            # they want the setconfig/file value to govern again.
            try:
                if database is not None and database.get_config_override(_field) is not None:
                    continue
            except Exception:
                pass
            try:
                if _cast == "bool":
                    _new = _val.lower() in ("true", "1", "yes")
                elif _cast == "int":
                    _new = int(_val)
                elif _cast == "str":
                    _new = _val
                else:
                    _new = float(_val)
            except ValueError:
                continue
            if getattr(config, _field, None) != _new:
                setattr(config, _field, _new)
                plugin.log(f"Dynamic config refresh: {_field} = {_new}")


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
    if not isinstance(forward_event, dict):
        plugin.log("forward_event: malformed event ignored", level="debug")
        return

    status = forward_event.get("status")
    native_source = _native_forward_ingestion_source() if status == "settled" else None
    native_observation = None
    if native_source is not None:
        native_observation = observe_settled_identity(forward_event, native_source)
        if native_observation.status != "usable":
            plugin.log("forward_event: native evidence " + native_observation.status,
                       level="debug")
            return
    in_channel = normalize_scid(forward_event.get("in_channel")) if forward_event.get("in_channel") else None
    if native_observation is not None:
        in_channel = native_observation.record.in_channel

    # Per-forward write coalescing: when the database exposes the combined
    # single-transaction method, settled forwards record the forward row AND
    # the reputation upsert in one transaction instead of two autocommit writes.
    record_combined = getattr(database, "record_forward_and_reputation", None)

    # Track peer reputation for all forward outcomes
    peer_id = _resolve_scid_to_peer(in_channel) if in_channel else None
    if native_source is not None and peer_id and not callable(record_combined):
        raise ValueError("native ingestion requires atomic forward/reputation writer")
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
    if (
        status in ("failed", "local_failed")
        and fee_controller is not None
        and _fee_authority_denial("failed_forward_trigger") is None
    ):
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
        if native_observation is not None:
            record = native_observation.record
            out_channel = record.out_channel
            in_msat, out_msat, fee_msat = record.in_msat, record.out_msat, record.fee_msat
            # Keep the actual transport values; integer-second projection is
            # the Database's job, after it has claimed the native identity.
            received_time = forward_event["received_time"]
            resolved_time = forward_event["resolved_time"]
            resolution_duration = (record.resolved_time_ns - record.received_time_ns) / 1e9
        else:
            in_msat = _parse_msat(forward_event.get("in_msat", forward_event.get("in_msatoshi", 0)))
            out_msat = _parse_msat(forward_event.get("out_msat", forward_event.get("out_msatoshi", 0)))
            fee_msat = _parse_msat(forward_event.get("fee_msat", forward_event.get("fee_msatoshi", 0)))
            received_time = int(forward_event.get("received_time", 0) or 0)
            resolved_time = int(forward_event.get("resolved_time", 0) or 0)
            resolution_duration = max(0, resolved_time - received_time) if resolved_time > 0 else 0

        if native_source is not None and peer_id:
            inserted = record_combined(forward_event, peer_id, True, native_source=native_source)
        elif callable(record_combined) and peer_id:
            inserted = record_combined(
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
            native_kwargs = {} if native_source is None else {
                "native_source": native_source,
                "in_htlc_id": forward_event.get("in_htlc_id"),
                "created_index": forward_event.get("created_index"),
                "updated_index": forward_event.get("updated_index"),
            }
            inserted = database.record_forward(
                in_channel or "",
                out_channel or "",
                in_msat,
                out_msat,
                fee_msat,
                received_time,
                resolved_time,
                resolution_duration,
                **native_kwargs,
            )

        # Only newly committed native evidence can move the wake monitors.
        # Legacy writers historically return None, which remains compatible;
        # their existing coarse deduplication is not upgraded by this gate.
        if inserted is False or (native_source is not None and inserted is not True):
            return

        # Persist evidence before requesting evaluation.  The notification
        # handler never mutates fees; it only wakes the existing governed loop
        # after a fixed active-experiment or yield-inventory volume step.
        if fee_controller is not None:
            try:
                wake_requested = False
                if _fee_authority_denial("acquisition_monitor_trigger") is None:
                    wake_requested = fee_controller.should_wake_acquisition_cycle(
                        out_channel, out_msat
                    )
                    # A settled forward drains the outgoing lane AND refills
                    # the incoming lane. Register both before coalescing the
                    # loop wake; an acquisition wake must not hide either
                    # inventory marker from the next governed fee cycle.
                    for changed_channel, changed_msat in (
                        (out_channel, out_msat), (in_channel, in_msat),
                    ):
                        if fee_controller.should_wake_yield_inventory_cycle(
                            changed_channel, changed_msat
                        ):
                            wake_requested = True
                if wake_requested:
                    _request_fee_adjustment_wake()
            except Exception as exc:
                plugin.log(
                    f"FEE_WAKE_MONITOR: settled evidence ignored: {exc}",
                    level="debug",
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
        closing_txid=closing_txid,
        bkpr_account=bkpr_account
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


# Give up polling a closure this long after close: whatever the bookkeeper
# reports by then is final for our accounting (timelocks and sweeps resolve
# in days, not months).
CLOSURE_RESOLUTION_TIMEOUT_DAYS = 90
# Write-ahead crash marker for the closure sweep. Set before the first
# bookkeeper RPC and cleared after a clean finish. If a sweep run dies
# mid-flight (2026-07-10 lnnode: bkpr-listbalances walked an account with a
# corrupted ledger — "Account balance underflow" — bookkeeper fatally
# exited, and being an important plugin it took lightningd down with it),
# the marker survives and permanently disables the sweep on this node until
# the operator repairs the bookkeeper ledger and clears the key. This
# bounds the blast radius to ONE crash per node, ever.
_CLOSURE_SWEEP_TRIP_KEY = "_closure_sweep_tripped"


def _reconcile_closure_resolutions() -> Dict[str, Any]:
    """Accumulate post-close resolution fees (HTLC sweeps) for closures
    recorded before their outputs were fully swept.

    record_channel_closure fires once at close detection; unilateral closes
    keep paying sweep fees for hours or days afterwards. This sweep
    re-queries the bookkeeper for every unresolved closure row, adds any new
    fees via update_closure_resolution, and marks the row complete once the
    bookkeeper account balance reaches zero (or after
    CLOSURE_RESOLUTION_TIMEOUT_DAYS).
    """
    summary = {"checked": 0, "updated": 0, "completed": 0, "added_fee_sats": 0}
    if database is None or safe_plugin is None:
        return summary

    # Crash-once guard: a marker left by a previous run means that run died
    # mid-sweep (bookkeeper/lightningd crash while we were querying it).
    # Never touch the bookkeeper again on this node until the operator
    # repairs the underlying ledger and clears the marker.
    tripped = database.get_config_override(_CLOSURE_SWEEP_TRIP_KEY)
    if tripped and not str(tripped).startswith("inflight:"):
        summary["tripped"] = str(tripped)
        return summary
    if tripped:
        # Leftover "inflight:" marker == the previous sweep never finished.
        database.set_config_override(
            _CLOSURE_SWEEP_TRIP_KEY,
            f"{int(time.time())}: previous sweep died mid-run (probable "
            "bookkeeper crash, e.g. account balance underflow) — closure "
            "sweep disabled on this node. Repair the bookkeeper ledger, "
            "then clear with: sqlite3 <revenue_ops.db> \"DELETE FROM "
            f"config_overrides WHERE key='{_CLOSURE_SWEEP_TRIP_KEY}'\"",
        )
        plugin.log(
            "CLOSURE_SWEEP: previous run died mid-sweep — the bookkeeper on "
            "this node likely has a corrupted account ledger (see 'Account "
            "balance underflow' in the lightningd log). Sweep permanently "
            "disabled until the marker is cleared.",
            level='error'
        )
        summary["tripped"] = "detected_crashed_run"
        return summary

    rows = database.get_unresolved_closures()
    if not rows:
        return summary

    # Arm the write-ahead marker before the first bookkeeper RPC.
    database.set_config_override(
        _CLOSURE_SWEEP_TRIP_KEY, f"inflight:{int(time.time())}"
    )

    # One balances call for every account: zero balance == fully swept.
    balances_by_account: Dict[str, int] = {}
    balances_available = False
    try:
        balances = safe_plugin.rpc.call("bkpr-listbalances", {})
        for acct in balances.get("accounts", []) or []:
            name = str(acct.get("account") or "")
            total = 0
            for bal in acct.get("balances", []) or []:
                total += parse_msat(bal.get("balance_msat", 0) or 0)
            balances_by_account[name] = total
        balances_available = True
    except Exception as e:
        plugin.log(f"CLOSURE_SWEEP: bkpr-listbalances unavailable: {e}", level='debug')

    # Legacy rows (recorded before bkpr_account was stored) resolve their
    # account via listclosedchannels' SCID -> hex channel_id mapping.
    scid_to_account: Dict[str, str] = {}

    def _account_for(row: Dict[str, Any]) -> Optional[str]:
        account = str(row.get("bkpr_account") or "")
        if account:
            return account
        nonlocal scid_to_account
        if not scid_to_account:
            try:
                closed = (data_service.get_closed_channels() if data_service
                          else safe_plugin.rpc.call("listclosedchannels"))
                for ch in closed.get("closedchannels", []) or []:
                    scid = normalize_scid(ch.get("short_channel_id", "") or "")
                    acct = str(ch.get("channel_id") or "")
                    if scid and acct:
                        scid_to_account[scid] = acct
            except Exception as e:
                plugin.log(f"CLOSURE_SWEEP: listclosedchannels unavailable: {e}", level='debug')
                scid_to_account = {"": ""}  # sentinel: don't retry this pass
        account = scid_to_account.get(normalize_scid(row.get("channel_id") or ""), "")
        if account:
            database.set_closure_bkpr_account(row["channel_id"], account)
        return account or None

    now = int(time.time())
    for row in rows:
        summary["checked"] += 1
        channel_id = row.get("channel_id")
        closed_at = int(row.get("closed_at") or 0)
        aged_out = closed_at > 0 and (now - closed_at) > CLOSURE_RESOLUTION_TIMEOUT_DAYS * 86400

        account = _account_for(row)
        if not account:
            # Without an account we cannot re-query; once aged out, stop
            # revisiting the row every pass.
            if aged_out:
                database.mark_closure_complete(channel_id)
                summary["completed"] += 1
            continue

        fresh = None
        try:
            fresh = _get_closure_costs_from_bookkeeper(account)
        except Exception as e:
            plugin.log(f"CLOSURE_SWEEP: bkpr query failed for {channel_id}: {e}", level='debug')
        if fresh:
            stored_total = (int(row.get("closure_fee_sats") or 0)
                            + int(row.get("htlc_sweep_fee_sats") or 0))
            fresh_total = (int(fresh.get("closure_fee_sats") or 0)
                           + int(fresh.get("htlc_sweep_fee_sats") or 0))
            delta = fresh_total - stored_total
            if delta > 0:
                database.update_closure_resolution(channel_id, delta)
                summary["updated"] += 1
                summary["added_fee_sats"] += delta
                plugin.log(
                    f"CLOSURE_SWEEP: {channel_id} +{delta} sats post-close "
                    f"resolution fees (total now {fresh_total})",
                    level='info'
                )

        resolved = balances_available and balances_by_account.get(account, 0) == 0
        if resolved or aged_out:
            database.mark_closure_complete(channel_id)
            summary["completed"] += 1

    # Clean finish: disarm the crash marker.
    database.delete_config_override(_CLOSURE_SWEEP_TRIP_KEY)

    if summary["updated"] or summary["completed"]:
        plugin.log(
            f"CLOSURE_SWEEP: checked={summary['checked']} "
            f"updated={summary['updated']} (+{summary['added_fee_sats']} sats) "
            f"completed={summary['completed']}",
            level='info'
        )
    return summary


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

        # Plausibility repair for opened_at, mirrored from the profitability
        # analyzer: rows poisoned by the pre-fix rolling now-30d estimate
        # otherwise escape into the PERMANENT closed_channels ledger here
        # (days_open snapshotted from the raw value). Anchor on the live tip:
        # a stored opened_at deviating from the SCID estimate by more than
        # max(7d, 15% of estimated age) is repaired with the estimate.
        try:
            if channel_id and 'x' in channel_id:
                _block_height = int(channel_id.split('x')[0])
                _tip = 0
                try:
                    _tip = int(data_service.get_block_height() or 0) if data_service \
                        else int(safe_plugin.rpc.getinfo().get("blockheight", 0) or 0)
                except Exception:
                    _tip = 0
                _now = int(time.time())
                if _tip >= _block_height > 0:
                    _estimate = _now - (_tip - _block_height) * 600
                    _slack = max(7 * 86400, int(0.15 * max(0, _now - _estimate)))
                    if not opened_at or abs(int(opened_at) - _estimate) > _slack:
                        plugin.log(
                            f"CLOSE_ARCHIVE: repairing implausible opened_at for "
                            f"{channel_id} ({opened_at} -> {_estimate})",
                            level='info'
                        )
                        opened_at = _estimate
        except (ValueError, IndexError):
            pass

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
# UNIFIED COST REPORTING
# =============================================================================

def _rpc_total_cost_budget(plugin: Plugin, window_hours: int = None) -> Dict[str, Any]:
    """Unified budget status across rebalances and historical liquidity costs."""
    try:
        # An explicit operator read is a synchronization boundary: never show
        # a known-stale pre-settlement/config snapshot merely to save a handful
        # of local aggregate queries.  The fresh result repopulates the memo for
        # lower-priority telemetry callers.
        return _total_cost_budget_status(window_hours=window_hours, force_fresh=True)
    except Exception as e:
        return {"error": str(e)}


@plugin.method("revenue-total-cost-budget")
def revenue_total_cost_budget(plugin: Plugin, window_hours: int = None) -> Dict[str, Any]:
    """Deprecated alias for 'revenue-budget' (removal 2026-09-05)."""
    return _deprecated_alias(
        _rpc_total_cost_budget(plugin, window_hours=window_hours), "revenue-budget"
    )


def _rpc_capex_status(plugin, **kwargs):
    """Return unified capex budget allocations.

    Shows per-channel rebalance budgets,
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
        }

    now = int(time.time())
    result = {
        "timestamp": now,
        "generated_at": now,
        "ttl_seconds": 1800,
        "status": "ok",
        "priority_class": alloc.priority_class,
        "global_envelope_sats": alloc.global_envelope_sats,
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
            "total_fleet_contribution_sats": alloc.total_fleet_contribution_sats,
            "allocated_by_priority_sats": alloc.allocated_by_priority_sats,
            "channel_count": len(channels),
        }
        if data_service:
            data_service.datastore_push(["revenue", "capex-summary"], summary)
    except Exception:
        pass  # datastore_push handles its own error logging

    return result


@plugin.method("revenue-capex-status")
def revenue_capex_status(plugin, **kwargs):
    """Deprecated alias for 'revenue-budget' (removal 2026-09-05)."""
    return _deprecated_alias(_rpc_capex_status(plugin, **kwargs), "revenue-budget")


def _rpc_spend_ledger(
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


@plugin.method("revenue-spend-ledger")
def revenue_spend_ledger(
    plugin: Plugin,
    window_hours: int = 24,
    include_reservations: bool = False,
    reservation_limit: int = 50,
) -> Dict[str, Any]:
    """Deprecated alias for 'revenue-budget ledger' (removal 2026-09-05)."""
    return _deprecated_alias(
        _rpc_spend_ledger(
            plugin,
            window_hours=window_hours,
            include_reservations=include_reservations,
            reservation_limit=reservation_limit,
        ),
        "revenue-budget ledger",
    )


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
    # Audit 2026-08-01 wave2 FIX 4: validate at the RPC boundary — a negative
    # actual_spent_sats would record a negative spend event and silently
    # inflate the remaining daily budget for every autonomous spender.
    if actual_spent_sats is not None:
        try:
            actual_spent_sats = int(actual_spent_sats)
        except (ValueError, TypeError):
            return {"error": f"invalid actual_spent_sats: {actual_spent_sats!r} (must be a non-negative integer)"}
        if actual_spent_sats < 0:
            return {"error": f"invalid actual_spent_sats: {actual_spent_sats} (must be >= 0)"}
    try:
        ok = database.mark_spend_reservation_spent(
            reservation_id=str(reservation_id),
            actual_spent_sats=actual_spent_sats,
            source=source,
            record_event=bool(record_event),
        )
        return {"status": "success" if ok else "not_found", "reservation_id": str(reservation_id)}
    except Exception as e:
        return {"error": str(e)}


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


_CANONICAL_TOTAL_COST_LEDGER_CATEGORIES = frozenset({"channel_open", "channel_close", "rebalance", "boltz"})


def _safe_cost_int(value: Any) -> int:
    """Return a neutral non-negative cost for malformed historical values."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _normalize_generic_ledger_for_total_cost_budget(generic_ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Exclude canonical open/close spend events from the generic ledger budget bucket."""
    normalized = dict(generic_ledger or {})
    spent_by_category = normalized.get("spent_by_category")
    if not isinstance(spent_by_category, dict):
        spent_by_category = {}

    counted_spent_categories = {}
    excluded_spent_categories = {}
    for category, amount in spent_by_category.items():
        amount_int = _safe_cost_int(amount)
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
            capacity = max(0, int(ch.get("capacity") or 0))
            local = max(0, channel_local_balance_msat(ch) // 1000)
            local = min(local, capacity)
            total_cap += capacity
            total_recv += max(0, capacity - local)
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
# enforced atomically at reservation time) but expensive to compute (~6 aggregate queries). One retained decision cycle evaluates it
# 4-8 times through the provider wiring, so a short TTL collapses those calls.
_TOTAL_COST_BUDGET_MEMO_TTL_SECONDS = 30.0
_total_cost_budget_memo: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_total_cost_budget_memo_lock = threading.Lock()
# Reentrancy guard (defense in depth): the unified status aggregates component
# providers that may, through misconfigured wiring, call back into the status.
_total_cost_budget_reentry = threading.local()


def _invalidate_total_cost_budget_memo() -> None:
    """Discard advisory budget snapshots after committed input changes."""
    with _total_cost_budget_memo_lock:
        _total_cost_budget_memo.clear()


def _total_cost_budget_status(window_hours: Optional[int] = None,
                              force_fresh: bool = False) -> Dict[str, Any]:
    """Unified budget status across rebalances and historical liquidity costs.

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

    # Actual cost components (canonical data sources)
    rebalance = _rebalance_liquidity_cost_components(window_hours=wh)
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

    _historical_spent = generic_ledger.get("excluded_spent_categories", {})
    actual_by_category = {
        "rebalance": _safe_cost_int(rebalance.get("spent_24h_sats", 0)),
        "boltz": _safe_cost_int(_historical_spent.get("boltz", 0)),
        "open": _safe_cost_int(open_cost_sats),
        "close": _safe_cost_int(closure_cost_sats),
        "ledger": _safe_cost_int(generic_ledger.get("spent_24h_sats", 0)),
    }
    # Phase 2J: unified rebalance reservations live in the generic ledger
    # under category='rebalance' AND are already counted in the
    # "rebalance" bucket (get_daily_rebalance_spend sums both halves) —
    # exclude them from the "ledger" bucket so each hold counts once.
    _ledger_reserved = int(generic_ledger.get("reserved_24h_sats", 0) or 0)
    _reserved_categories = generic_ledger.get("reserved_by_category", {}) or {}
    _ledger_rebalance_reserved = _safe_cost_int(_reserved_categories.get("rebalance", 0))
    _ledger_boltz_reserved = _safe_cost_int(_reserved_categories.get("boltz", 0))
    reserved_by_category = {
        "rebalance": _safe_cost_int(rebalance.get("reserved_24h_sats", 0)),
        "boltz": _ledger_boltz_reserved,
        "ledger": max(0, _ledger_reserved - _ledger_rebalance_reserved - _ledger_boltz_reserved),
    }

    actual_total = sum(max(0, int(v or 0)) for v in actual_by_category.values())
    reserved_total = sum(max(0, int(v or 0)) for v in reserved_by_category.values())

    daily_budget_sats = max(0, int(getattr(cfg, "daily_budget_sats", 0) or 0))
    net_profit_sats = int(revenue_sats - actual_total)
    growth_budget = compute_growth_budget_status(
        base_budget_sats=daily_budget_sats,
        net_profit_sats=net_profit_sats,
        actual_spent_sats=actual_total,
        reserved_sats=reserved_total,
        enabled=bool(getattr(cfg, "growth_budget_enabled", False)),
        earned_fraction=float(getattr(cfg, "growth_budget_earned_fraction", 0.25) or 0.0),
        growth_fraction=float(getattr(cfg, "growth_budget_experiment_fraction", 0.10) or 0.0),
        growth_max_extra_sats=int(getattr(cfg, "growth_budget_max_extra_sats", 0) or 0),
        hard_ceiling_sats=int(getattr(cfg, "growth_budget_hard_ceiling_sats", daily_budget_sats) or daily_budget_sats),
    )
    effective_budget_sats = int(growth_budget.get("effective_budget_sats", daily_budget_sats) or 0)

    remaining_sats = int(growth_budget.get("remaining_sats", max(0, int(effective_budget_sats) - actual_total - reserved_total)) or 0)

    # Measured window coverage (honest, never an echo of the request):
    # hours between the oldest cost-evidence row and now, capped at the
    # window. When the measurement basis is absent or malformed, report an
    # honest unknown (covered_hours=null) rather than a fabricated
    # "complete". External consumers read covered_hours/
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
        "mode": growth_budget.get("mode", "fixed"),
        "daily_budget_sats": daily_budget_sats,
        "effective_budget_sats": int(effective_budget_sats),
        "revenue_sats": revenue_sats,
        "actual_spent_sats": actual_total,
        "reserved_sats": reserved_total,
        "remaining_sats": remaining_sats,
        "net_profit_sats_after_costs": net_profit_sats,
        "growth_budget": growth_budget,
        "actual_spent_by_category": actual_by_category,
        "reserved_by_category": reserved_by_category,
        "open_close_cost_visibility": open_close_cost_visibility,
        "components": {
            "rebalance": rebalance,
            "generic_ledger": generic_ledger,
            "open_cost_sats": open_cost_sats,
            "closure_cost_sats": closure_cost_sats,
        },
    }


def _total_cost_budget_limit_provider() -> Dict[str, Any]:
    # This provider feeds the retained rebalance gating path. Gating callers must
    # read the live unified total, not the telemetry memo, so concurrent checks
    # do not admit against the same stale snapshot. force_fresh; the authoritative rail remains the atomic
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


@plugin.method("revenue-cycle")
def revenue_cycle(plugin: Plugin, subsystem: str = None, **kwargs) -> Dict[str, Any]:
    """Run one manual cycle of a subsystem (primary name since 2026-08-01).

    Usage: lightning-cli -k revenue-cycle subsystem=<subsystem> [key=value ...]

    Subsystems: fees (was revenue-fee-cycle), rebalance
    (revenue-rebalance-cycle), flow (revenue-analyze), all (revenue-wake-all). An unknown subsystem returns the valid list.
    """
    table = {
        "fees": _rpc_fee_cycle,
        "rebalance": _rpc_rebalance_cycle,
        "flow": _rpc_analyze,
        "all": _rpc_wake_all,
    }
    return _dispatch_subcommand(
        plugin, "revenue-cycle", "subsystem", table, subsystem, kwargs
    )


@plugin.method("revenue-budget")
def revenue_budget(plugin: Plugin, section: str = None, **kwargs) -> Dict[str, Any]:
    """Unified budget view (primary name since 2026-08-01).

    Usage:
      lightning-cli revenue-budget
          One response with sections: total_cost (was
          revenue-total-cost-budget) and capex (revenue-capex-status).
      lightning-cli -k revenue-budget section=total_cost [window_hours=N]
          Read-only total-cost budget status (was revenue-total-cost-budget).
      lightning-cli -k revenue-budget section=ledger [window_hours=N]
          [include_reservations=true] [reservation_limit=N]
          Forwards to the spend ledger (was revenue-spend-ledger).
    """
    key = str(section or "").strip().lower()
    if key in ("", "all"):
        def _section(fn, **kw):
            try:
                return fn(plugin, **kw)
            except Exception as e:
                return {"error": str(e)}

        total_cost_kwargs = {}
        if kwargs.get("window_hours") is not None:
            total_cost_kwargs["window_hours"] = kwargs["window_hours"]
        return {
            "total_cost": _section(_rpc_total_cost_budget, **total_cost_kwargs),
            "capex": _section(_rpc_capex_status),
        }
    table = {"total_cost": _rpc_total_cost_budget, "ledger": _rpc_spend_ledger}
    return _dispatch_subcommand(
        plugin, "revenue-budget", "section", table, section, kwargs
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    plugin.run()
