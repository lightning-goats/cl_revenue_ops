"""
Bridge for querying fee intelligence from cl-hive.

This module provides cl-revenue-ops with access to the collective fee
intelligence gathered by the hive fleet. The bridge implements:

1. Circuit Breaker Pattern: Prevents cascading failures if cl-hive is down
2. In-Memory Cache: Reduces RPC calls with 30-minute TTL
3. Graceful Degradation: Falls back to local-only mode when hive unavailable
4. Stale Cache Usage: Uses cached data with reduced confidence when fresh unavailable

Phase 1: Query Integration
- query_fee_intelligence(): Get competitor fee data for a peer
- is_available(): Check if cl-hive plugin is active

Phase 2: Bidirectional Sharing
- report_observation(): Report fee observations back to cl-hive
- report_health_update(): Report health status to fleet
- query_member_health(): Query member NNLB health
- report_liquidity_state(): Report our liquidity state
- check_rebalance_conflict(): Check for rebalancing conflicts

Yield Optimization Phase 2 - Fee Coordination:
- query_coordinated_fee_recommendation(): Get coordinated fee (corridors, pheromones, defense)
- report_routing_outcome(): Report routing for stigmergic learning
- query_defense_status(): Query threat peer defense status

Yield Optimization Phase 3 - Cost Reduction:
- query_fleet_rebalance_path(): Check if fleet path is cheaper
- report_rebalance_outcome(): Report rebalance for coordination

Yield Metrics:
- report_yield_metrics(): Report TLV, costs, revenue to fleet

Author: Lightning Goats Team
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# CONSTANTS
# =============================================================================

# Cache TTLs
CACHE_TTL_SECONDS = 1800          # 30 minutes - fresh cache
STALE_CACHE_TTL_SECONDS = 86400   # 24 hours - reduced confidence

# Cache size limits (prevent unbounded memory growth)
MAX_CACHE_ENTRIES = 500           # Maximum peers to cache
CACHE_CLEANUP_INTERVAL = 3600     # Cleanup stale entries every hour

# Circuit breaker settings
CIRCUIT_FAILURES_THRESHOLD = 3    # Failures before opening circuit
CIRCUIT_RESET_TIMEOUT = 60        # Seconds before trying again

# Hive intelligence settings
MIN_CONFIDENCE_THRESHOLD = 0.3    # Ignore data below this confidence


# =============================================================================
# COORDINATION INPUTS
# =============================================================================

@dataclass
class CoordinationInputs:
    mode: str
    priors: Dict[str, Any] = field(default_factory=dict)
    corridor_hint: Optional[str] = None
    peer_quality: Optional[Dict[str, Any]] = None


# =============================================================================
# CIRCUIT BREAKER STATE
# =============================================================================

@dataclass
class CircuitBreakerState:
    """
    Tracks circuit breaker state for cl-hive RPC calls.

    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Fail-fast mode, calls immediately return cached data
    - HALF_OPEN: Testing recovery, limited calls allowed
    """
    failures: int = 0
    last_failure: float = 0
    is_open: bool = False


# =============================================================================
# CACHED PROFILE
# =============================================================================

@dataclass
class CachedProfile:
    """Cached fee intelligence profile with timestamp."""
    data: Dict[str, Any]
    timestamp: float


@dataclass(frozen=True)
class RpcMethodPolicy:
    """Reliability policy for a class of Hive bridge RPC calls."""
    name: str
    retries: int = 0
    backoff_seconds: float = 0.0
    use_breaker: bool = True
    async_preferred: bool = False
    count_failures: bool = True


@dataclass
class RpcPolicyStats:
    """Operational counters for one policy class."""
    calls: int = 0
    success: int = 0
    failures: int = 0
    retries: int = 0
    short_circuit: int = 0
    unavailable: int = 0
    async_queued: int = 0
    async_dropped: int = 0
    last_error: Optional[str] = None
    last_error_at: int = 0


# =============================================================================
# HIVE FEE INTELLIGENCE BRIDGE
# =============================================================================

class HiveFeeIntelligenceBridge:
    """
    Bridge for querying fee intelligence from cl-hive.

    Provides cl-revenue-ops with collective fee intelligence from the hive
    while handling failures gracefully through circuit breaker pattern
    and caching.

    Usage:
        bridge = HiveFeeIntelligenceBridge(plugin, database, config)

        # Check if cl-hive is available
        if bridge.is_available():
            intel = bridge.query_fee_intelligence(peer_id)
            if intel:
                # Use competitor data for fee decisions
                their_avg_fee = intel.get("avg_fee_charged", 0)
    """

    def __init__(self, plugin, database, config=None):
        """
        Initialize the HiveFeeIntelligenceBridge.

        Args:
            plugin: Reference to the pyln Plugin (or ThreadSafePluginProxy)
            database: Database instance for state persistence (future use)
            config: Config instance for fee bounds (optional)
        """
        self.plugin = plugin
        self.database = database
        self.config = config
        self._circuit_breaker_enabled = bool(getattr(config, 'hive_bridge_circuit_breaker_enabled', False))
        self._circuit_reset_timeout = int(getattr(config, 'rpc_circuit_breaker_seconds', CIRCUIT_RESET_TIMEOUT) or CIRCUIT_RESET_TIMEOUT)
        self._rpc_policies: Dict[str, RpcMethodPolicy] = {
            # Optional reads should fail fast and allow cache fallback.
            "optional_read": RpcMethodPolicy(name="optional_read", retries=0, backoff_seconds=0.0, use_breaker=True, async_preferred=False, count_failures=True),
            # Critical writes can retry briefly; breaker is per-policy when enabled.
            "critical_write": RpcMethodPolicy(name="critical_write", retries=1, backoff_seconds=0.25, use_breaker=True, async_preferred=False, count_failures=True),
            # Telemetry should be async/droppable and never poison breaker state.
            "telemetry": RpcMethodPolicy(name="telemetry", retries=0, backoff_seconds=0.0, use_breaker=False, async_preferred=True, count_failures=False),
        }

        # Cache: peer_id -> CachedProfile
        self._cache: Dict[str, CachedProfile] = {}
        self._cache_lock = threading.RLock()  # RLock: _evict_oldest_cache_entries is called under lock from _set_cached

        # Circuit breaker state
        self._circuit = CircuitBreakerState()
        self._circuit_lock = threading.Lock()
        self._policy_circuits: Dict[str, CircuitBreakerState] = {
            name: CircuitBreakerState() for name in self._rpc_policies.keys()
        }
        self._policy_circuit_lock = threading.Lock()

        # Availability cache: None = unknown, True/False = known
        self._hive_available: Optional[bool] = None
        self._availability_check_time: float = 0
        self._availability_ttl: float = 60.0  # Re-check every 60 seconds
        self._availability_lock = threading.Lock()  # Prevent stampede on cache refresh
        self._init_complete = True  # Gate: False during plugin init, set True after

        # M-2: Initialize integration cache in __init__
        self._integration_cache: Dict[str, Any] = {}
        self._intel_cache_lock = threading.Lock()
        self._rpc_policy_stats: Dict[str, RpcPolicyStats] = {
            name: RpcPolicyStats() for name in self._rpc_policies.keys()
        }
        self._rpc_policy_stats_lock = threading.Lock()

        # Coalesce non-critical settlement cost telemetry to avoid blocking on
        # cl-hive fee gossip lock contention and prevent noisy bridge breaker trips.
        self._period_costs_report_lock = threading.Lock()
        self._last_reported_period_costs_sats: Optional[int] = None
        self._last_period_costs_report_time: float = 0.0
        self._period_costs_report_min_interval: float = 30.0

    def _log(self, message: str, level: str = "debug") -> None:
        """Log a message if plugin is available."""
        if self.plugin:
            self.plugin.log(f"HIVE_BRIDGE: {message}", level=level)

    def mark_init_complete(self) -> None:
        """Signal that plugin init has finished and hive calls are safe."""
        self._init_complete = True

    # =========================================================================
    # AVAILABILITY CHECK
    # =========================================================================

    def is_available(self) -> bool:
        """
        Check if cl-hive plugin is available AND we are a hive member (cached).

        Returns cached result if within TTL to avoid expensive RPC calls.
        Membership is verified by checking our tier via hive-status RPC.

        Uses a non-blocking lock to prevent multiple threads from refreshing
        the cache simultaneously (stampede prevention).

        Returns:
            True if cl-hive is active AND we are a member/neophyte, False otherwise
        """
        now = time.time()

        # Block all hive calls until plugin init completes — CLN holds a
        # lock during plugin start that blocks plugin-to-plugin RPCs.
        if not self._init_complete:
            return False

        # Short-circuit if our own bridge circuit breaker is open
        if self._is_circuit_open():
            return self._hive_available if self._hive_available is not None else False

        # Return cached result if within TTL
        if (self._hive_available is not None and
                (now - self._availability_check_time) < self._availability_ttl):
            return self._hive_available

        # Cache expired — try to acquire the refresh lock (non-blocking)
        if not self._availability_lock.acquire(blocking=False):
            # Another thread is already refreshing; return stale cached value
            return self._hive_available if self._hive_available is not None else False

        try:
            # Double-check cache inside lock (another thread may have just refreshed)
            now = time.time()
            if (self._hive_available is not None and
                    (now - self._availability_check_time) < self._availability_ttl):
                return self._hive_available

            # Check plugin list (via pool — works during and after init)
            try:
                plugins = self.plugin.rpc.plugin("list")
                hive_loaded = False
                for p in plugins.get("plugins", []):
                    if "cl-hive" in p.get("name", "") and p.get("active", False):
                        hive_loaded = True
                        break

                if not hive_loaded:
                    self._hive_available = False
                    self._availability_check_time = now
                    return False

                # cl-hive is loaded and active — assume available.
                # We intentionally do NOT call hive-status here because:
                # 1. During init: CLN blocks plugin-to-plugin RPCs
                # 2. After init: queued hive-status requests can exhaust
                #    cl-hive's RPC_LOCK, deadlocking it
                # Instead, we trust the plugin list. If cl-hive is active
                # but we're not a member, hive RPC calls will fail
                # gracefully via the circuit breaker in production code.
                self._hive_available = True
                self._availability_check_time = now
                self._log("Hive available: cl-hive plugin active")
                return True

            except Exception as e:
                self._log(f"Error checking cl-hive availability: {e}", level="warn")
                self._hive_available = False
                self._availability_check_time = now - (self._availability_ttl - 5)
                return False
        finally:
            self._availability_lock.release()

    def invalidate_availability(self) -> None:
        """Force a fresh availability check on next is_available() call.

        Only called when we *want* a fresh probe (startup retries, explicit
        reset) — not on every call.
        """
        # TS-6: Protect shared state with availability lock
        with self._availability_lock:
            self._hive_available = None
            self._availability_check_time = 0

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _is_circuit_open(self) -> bool:
        """
        Check if circuit breaker is open.

        Returns:
            True if circuit is open (should fail fast)
        """
        if not self._circuit_breaker_enabled:
            return False

        with self._circuit_lock:
            if not self._circuit.is_open:
                return False

            # Check if reset timeout has passed
            if time.time() - self._circuit.last_failure > self._circuit_reset_timeout:
                self._circuit.is_open = False
                self._circuit.failures = 0
                self._log("Circuit breaker reset to CLOSED")
                return False

            return True

    def _record_success(self) -> None:
        """Record a successful RPC call."""
        if not self._circuit_breaker_enabled:
            return
        with self._circuit_lock:
            self._circuit.failures = 0
            self._circuit.is_open = False

    def _record_failure(self) -> None:
        """Record a failed RPC call."""
        if not self._circuit_breaker_enabled:
            return
        with self._circuit_lock:
            self._circuit.failures += 1
            self._circuit.last_failure = time.time()
            if self._circuit.failures >= CIRCUIT_FAILURES_THRESHOLD:
                self._circuit.is_open = True
                self._log(
                    f"Circuit breaker OPEN after {self._circuit.failures} failures",
                    level="warn"
                )

    def _is_policy_circuit_open(self, policy_name: str) -> bool:
        """Per-policy breaker check (disabled when global bridge breaker is disabled)."""
        if not self._circuit_breaker_enabled:
            return False
        with self._policy_circuit_lock:
            state = self._policy_circuits.get(policy_name)
            if state is None or not state.is_open:
                return False
            if time.time() - state.last_failure > self._circuit_reset_timeout:
                state.is_open = False
                state.failures = 0
                self._log(f"Policy circuit breaker reset to CLOSED ({policy_name})", level="debug")
                return False
            return True

    def _record_policy_success(self, policy_name: str) -> None:
        if not self._circuit_breaker_enabled:
            return
        with self._policy_circuit_lock:
            state = self._policy_circuits.get(policy_name)
            if state is None:
                return
            state.failures = 0
            state.is_open = False

    def _record_policy_failure(self, policy_name: str) -> None:
        if not self._circuit_breaker_enabled:
            return
        with self._policy_circuit_lock:
            state = self._policy_circuits.get(policy_name)
            if state is None:
                return
            state.failures += 1
            state.last_failure = time.time()
            if state.failures >= CIRCUIT_FAILURES_THRESHOLD:
                state.is_open = True
                self._log(
                    f"Policy circuit breaker OPEN after {state.failures} failures ({policy_name})",
                    level="warn",
                )

    def _policy_stats_inc(self, policy_name: str, field: str, value: int = 1) -> None:
        with self._rpc_policy_stats_lock:
            stats = self._rpc_policy_stats.get(policy_name)
            if not stats:
                return
            setattr(stats, field, int(getattr(stats, field, 0) or 0) + value)

    def _policy_stats_error(self, policy_name: str, err: Any) -> None:
        with self._rpc_policy_stats_lock:
            stats = self._rpc_policy_stats.get(policy_name)
            if not stats:
                return
            stats.failures = int(stats.failures or 0) + 1
            stats.last_error = str(err)
            stats.last_error_at = int(time.time())

    def _rpc_call_with_policy(
        self,
        method_name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        policy_key: str,
        require_available: bool = True,
        count_error_response_failure: bool = True,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a Hive RPC using a method-class policy.

        Returns:
            (ok, result, error_reason)
        """
        policy = self._rpc_policies.get(policy_key) or self._rpc_policies["optional_read"]
        self._policy_stats_inc(policy.name, "calls")

        if require_available and not self.is_available():
            self._policy_stats_inc(policy.name, "unavailable")
            return False, None, "unavailable"

        if policy.use_breaker and self._is_policy_circuit_open(policy.name):
            self._policy_stats_inc(policy.name, "short_circuit")
            return False, None, "policy_circuit_open"

        rpc = self.plugin.rpc
        if policy.async_preferred:
            fire_and_forget = getattr(rpc, "fire_and_forget", None)
            if callable(fire_and_forget):
                queued = bool(fire_and_forget(method_name, payload if payload is not None else {}))
                if queued:
                    self._policy_stats_inc(policy.name, "async_queued")
                    # Async queue success should not manipulate breaker state.
                    return True, {"queued": True}, None
                self._policy_stats_inc(policy.name, "async_dropped")
                return False, None, "async_queue_full"

        attempts = max(1, policy.retries + 1)
        last_err: Optional[str] = None
        for attempt in range(1, attempts + 1):
            try:
                result = rpc.call(method_name, payload if payload is not None else {})
                if isinstance(result, dict) and result.get("error"):
                    last_err = str(result.get("error"))
                    if count_error_response_failure and policy.count_failures:
                        self._policy_stats_error(policy.name, last_err)
                        if policy.use_breaker:
                            self._record_policy_failure(policy.name)
                    if attempt < attempts:
                        self._policy_stats_inc(policy.name, "retries")
                        if policy.backoff_seconds > 0:
                            time.sleep(policy.backoff_seconds * attempt)
                        continue
                    return False, result, f"rpc_error:{last_err}"

                self._policy_stats_inc(policy.name, "success")
                if policy.use_breaker:
                    self._record_policy_success(policy.name)
                return True, result if isinstance(result, dict) else {"result": result}, None
            except Exception as e:
                last_err = str(e)
                if policy.count_failures:
                    self._policy_stats_error(policy.name, e)
                    if policy.use_breaker:
                        self._record_policy_failure(policy.name)
                if attempt < attempts:
                    self._policy_stats_inc(policy.name, "retries")
                    if policy.backoff_seconds > 0:
                        time.sleep(policy.backoff_seconds * attempt)
                    continue
                return False, None, f"exception:{last_err}"

        return False, None, f"unknown:{last_err or 'unknown'}"

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def _get_cached(self, peer_id: str) -> Tuple[Optional[Dict], bool]:
        """
        Get cached profile for a peer.

        Args:
            peer_id: Peer public key

        Returns:
            (data, is_fresh) tuple where data is the profile or None,
            and is_fresh indicates if data is within fresh TTL
        """
        with self._cache_lock:
            if peer_id not in self._cache:
                return None, False

            cached = self._cache[peer_id]
            age = time.time() - cached.timestamp

            if age < CACHE_TTL_SECONDS:
                return cached.data, True  # Fresh cache
            elif age < STALE_CACHE_TTL_SECONDS:
                return cached.data, False  # Stale but usable
            else:
                # Too old, remove from cache
                del self._cache[peer_id]
                return None, False

    def _set_cached(self, peer_id: str, data: Dict) -> None:
        """
        Cache a profile with LRU eviction if at capacity.

        Enforces MAX_CACHE_ENTRIES limit by evicting oldest entries
        when cache is full.
        """
        with self._cache_lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= MAX_CACHE_ENTRIES and peer_id not in self._cache:
                self._evict_oldest_cache_entries(count=10)  # Evict 10 at a time

            self._cache[peer_id] = CachedProfile(
                data=data,
                timestamp=time.time()
            )

    def _evict_oldest_cache_entries(self, count: int = 10) -> None:
        """
        Evict the oldest cache entries.

        Args:
            count: Number of entries to evict
        """
        with self._cache_lock:
            if not self._cache:
                return

            # Sort by timestamp (oldest first)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].timestamp
            )

            # Remove oldest entries
            for peer_id, _ in sorted_entries[:count]:
                del self._cache[peer_id]

            self._log(f"Evicted {min(count, len(sorted_entries))} old cache entries", level="debug")


    def _stale_with_reduced_confidence(
        self,
        data: Dict,
        age_seconds: float
    ) -> Dict:
        """
        Return stale data with reduced confidence.

        Confidence decays linearly, reaching 50% at 12 hours and floored at 10%.

        Args:
            data: Original cached data
            age_seconds: Age of data in seconds

        Returns:
            Copy of data with reduced confidence and staleness markers
        """
        result = dict(data)
        age_hours = age_seconds / 3600

        # Linear decay: confidence drops to 50% at 12 hours, 0% at 24 hours (floored at 10%)
        decay_factor = max(0.1, 1.0 - (age_hours / 24))
        result["confidence"] = result.get("confidence", 0.5) * decay_factor
        result["stale"] = True
        result["age_hours"] = round(age_hours, 1)

        return result

    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================

    def query_fee_intelligence(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Query cl-hive for peer fee intelligence.

        This is the primary interface for cl-revenue-ops to get competitor
        fee data. Returns cached data when:
        - Fresh cache hit (within TTL)
        - Circuit breaker is open
        - RPC query fails

        Args:
            peer_id: External peer to query

        Returns:
            Fee intelligence dict or None if no data available:
            {
                "peer_id": "02abc...",
                "avg_fee_charged": 250,
                "min_fee": 100,
                "max_fee": 500,
                "fee_volatility": 0.15,
                "estimated_elasticity": -0.8,
                "optimal_fee_estimate": 180,
                "confidence": 0.75,
                "market_share": 0.0,
                "hive_capacity_sats": 6000000,
                "hive_reporters": 3,
                "last_updated": 1705000000,
                "stale": False  # True if using stale cache
            }
        """
        # Check cache first
        cached_data, is_fresh = self._get_cached(peer_id)

        if is_fresh:
            return cached_data

        # If circuit is open, return stale cache or None
        # Get cache entry under lock to avoid race with eviction
        with self._cache_lock:
            cache_entry = self._cache.get(peer_id)
            cache_timestamp = cache_entry.timestamp if cache_entry else 0

        if self._is_circuit_open():
            if cached_data and cache_entry:
                age = time.time() - cache_timestamp
                return self._stale_with_reduced_confidence(cached_data, age)
            return None

        # Check if cl-hive is available
        if not self.is_available():
            if cached_data and cache_entry:
                age = time.time() - cache_timestamp
                return self._stale_with_reduced_confidence(cached_data, age)
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-fee-intel-query",
            {"peer_id": peer_id, "action": "query"},
            policy_key="optional_read",
            require_available=False,  # already checked above
            count_error_response_failure=False,  # "no_data" is expected and benign
        )
        if not ok:
            if err and not err.startswith("rpc_error:no_data"):
                self._log(f"Failed to query fee intelligence: {err}", level="debug")
            if cached_data and cache_entry:
                age = time.time() - cache_timestamp
                return self._stale_with_reduced_confidence(cached_data, age)
            return None

        # AUDIT FIX HB-1: Replace assert with explicit check (assert is stripped with -O flag)
        if result is None:
            if cached_data and cache_entry:
                age = time.time() - cache_timestamp
                return self._stale_with_reduced_confidence(cached_data, age)
            return None
        if result.get("error"):
            if result.get("error") == "no_data":
                return None
            self._log(f"Query error: {result.get('error')}", level="debug")
            if cached_data and cache_entry:
                age = time.time() - cache_timestamp
                return self._stale_with_reduced_confidence(cached_data, age)
            return None

        self._set_cached(peer_id, result)
        return result


    # =========================================================================
    # PHASE 2: OBSERVATION REPORTING (Bidirectional Integration)
    # =========================================================================

    def report_observation(
        self,
        peer_id: str,
        our_fee_ppm: int,
        their_fee_ppm: Optional[int] = None,
        volume_sats: int = 0,
        forward_count: int = 0,
        period_hours: float = 1.0
    ) -> bool:
        """
        Report fee observation to cl-hive.

        Called after each fee optimization cycle to share observations
        with the hive fleet. Fire-and-forget pattern - doesn't block.

        Args:
            peer_id: External peer being observed
            our_fee_ppm: Our current fee toward this peer
            their_fee_ppm: Their fee toward us (if known)
            volume_sats: Volume routed in observation period
            forward_count: Number of forwards
            period_hours: Observation window length

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.is_available():
            return False

        # Calculate revenue rate
        revenue_sats = (volume_sats * our_fee_ppm) // 1_000_000
        revenue_rate = revenue_sats / period_hours if period_hours > 0 else 0

        ok, result, err = self._rpc_call_with_policy(
            "hive-report-fee-observation",
            {
                "peer_id": peer_id,
                "our_fee_ppm": our_fee_ppm,
                "their_fee_ppm": their_fee_ppm,
                "volume_sats": volume_sats,
                "forward_count": forward_count,
                "period_hours": period_hours,
                "revenue_rate": revenue_rate,
            },
            policy_key="telemetry",
        )
        if not ok:
            if err not in ("async_queue_full",):
                self._log(f"Failed to report observation: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Observation report error: {result.get('error')}", level="debug")
            return False
        return True

    # =========================================================================
    # NNLB HEALTH QUERIES (Phase 1 - NNLB-Aware Rebalancing)
    # =========================================================================

    def query_member_health(self, member_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Query NNLB health score for a member.

        Information sharing only - used to adjust OWN rebalancing priorities.
        No fund transfers between nodes.

        Args:
            member_id: Member to query (None for self)

        Returns:
            Health data dict or None if unavailable:
            {
                "member_id": "02abc...",
                "health_score": 65,
                "health_tier": "stable",
                "budget_multiplier": 1.0,
                "capacity_score": 70,
                "revenue_score": 60,
                ...
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        params = {"action": "query"}
        if member_id:
            params["member_id"] = member_id

        ok, result, err = self._rpc_call_with_policy(
            "hive-member-health", params, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query member health: {err}", level="debug")
            return None
        if result.get("error"):
            self._log(f"Health query error: {result.get('error')}", level="debug")
            return None
        return result

    def query_fleet_health(self) -> Optional[Dict[str, Any]]:
        """
        Query aggregated fleet health for situational awareness.

        Returns:
            Fleet health summary or None if unavailable:
            {
                "fleet_health": 58,
                "member_count": 5,
                "struggling_count": 1,
                "vulnerable_count": 2,
                "stable_count": 2,
                "thriving_count": 0,
                "members": [...]
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-member-health",
            {"member_id": "all", "action": "aggregate"},
            policy_key="optional_read",
            require_available=False,
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query fleet health: {err}", level="debug")
            return None
        if result.get("error"):
            self._log(f"Fleet health query error: {result.get('error')}", level="debug")
            return None
        return result

    def report_health_update(
        self,
        profitable_channels: int,
        underwater_channels: int,
        stagnant_channels: int,
        total_channels: int = None,
        revenue_trend: str = "stable",
        liquidity_score: int = 50
    ) -> bool:
        """
        Report our health status to cl-hive.

        Shares information so fleet knows our state.
        No sats move - purely informational.

        Args:
            profitable_channels: Number of profitable channels
            underwater_channels: Number of underwater channels
            stagnant_channels: Number of stagnant channels
            total_channels: Total channel count (optional)
            revenue_trend: "improving", "stable", or "declining"
            liquidity_score: Balance distribution score (0-100)

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        params = {
            "profitable_channels": profitable_channels,
            "underwater_channels": underwater_channels,
            "stagnant_channels": stagnant_channels,
            "revenue_trend": revenue_trend,
            "liquidity_score": liquidity_score,
        }
        if total_channels is not None:
            params["total_channels"] = total_channels

        ok, result, err = self._rpc_call_with_policy(
            "hive-report-health", params, policy_key="telemetry"
        )
        if not ok:
            if err not in ("async_queue_full",):
                self._log(f"Failed to report health: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Health report error: {result.get('error')}", level="debug")
            return False

        self._log(
            f"Health reported: profitable={profitable_channels}, "
            f"underwater={underwater_channels}, stagnant={stagnant_channels}, "
            f"liquidity_score={liquidity_score}"
        )
        return True

    # =========================================================================
    # PHASE 2: LIQUIDITY INTELLIGENCE SHARING
    # =========================================================================
    # These methods share INFORMATION about liquidity state.
    # No fund transfers between nodes - purely informational coordination.


    def query_fleet_liquidity_needs(self) -> List[Dict[str, Any]]:
        """
        Get fleet liquidity needs for coordination.

        Knowing what others need helps us:
        - Adjust our fees to direct flow helpfully
        - Avoid rebalancing through congested routes

        Returns:
            List of fleet liquidity needs with relevance scores
        """
        if self._is_circuit_open() or not self.is_available():
            return []

        ok, result, err = self._rpc_call_with_policy(
            "hive-liquidity-state", {"action": "needs"}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query fleet needs: {err}", level="debug")
            return []
        if result.get("error"):
            self._log(f"Fleet needs query error: {result.get('error')}", level="debug")
            return []
        return result.get("fleet_needs", [])

    def report_liquidity_state(
        self,
        depleted_channels: List[Dict[str, Any]],
        saturated_channels: List[Dict[str, Any]],
        rebalancing_active: bool = False,
        rebalancing_peers: List[str] = None,
        liquidity_needs: List[Dict[str, Any]] = None
    ) -> bool:
        """
        Report our liquidity state to the fleet.

        Sharing this information helps the fleet make better
        coordinated decisions. No sats transfer.

        Args:
            depleted_channels: List of {peer_id, local_pct, capacity_sats}
            saturated_channels: List of {peer_id, local_pct, capacity_sats}
            rebalancing_active: Whether we're currently rebalancing
            rebalancing_peers: Which peers we're rebalancing through
            liquidity_needs: Flow-aware enriched needs (top 10)

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        payload = {
            "depleted_channels": depleted_channels,
            "saturated_channels": saturated_channels,
            "rebalancing_active": rebalancing_active,
            "rebalancing_peers": rebalancing_peers or [],
        }
        if liquidity_needs:
            payload["liquidity_needs"] = liquidity_needs[:10]

        ok, result, err = self._rpc_call_with_policy(
            "hive-report-liquidity-state", payload, policy_key="telemetry"
        )
        if not ok:
            if err not in ("async_queue_full",):
                self._log(f"Failed to report liquidity state: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Liquidity state report error: {result.get('error')}", level="debug")
            return False

        self._log(
            f"Liquidity state reported: depleted={len(depleted_channels or [])}, "
            f"saturated={len(saturated_channels or [])}"
        )
        return True

    def update_rebalancing_activity(
        self,
        rebalancing_active: bool,
        rebalancing_peers: List[str]
    ) -> bool:
        """
        Push rebalancing activity to cl-hive (targeted update).

        Only updates rebalancing_active and rebalancing_peers fields,
        preserving existing depleted/saturated channel data.

        Args:
            rebalancing_active: Whether we're currently rebalancing
            rebalancing_peers: Peer IDs involved in active rebalances

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        ok, result, err = self._rpc_call_with_policy(
            "hive-update-rebalancing-activity",
            {
                "rebalancing_active": rebalancing_active,
                "rebalancing_peers": rebalancing_peers or [],
            },
            policy_key="telemetry",
        )
        if not ok:
            if err not in ("async_queue_full",):
                self._log(f"Failed to update rebalancing activity: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Rebalancing activity update error: {result.get('error')}", level="debug")
            return False
        return True

    def check_rebalance_conflict(self, peer_id: str) -> Dict[str, Any]:
        """
        Check if another fleet member is rebalancing through a peer.

        Avoids competing for the same routes, which wastes fees.
        Information-based coordination - no fund transfer.

        Args:
            peer_id: The peer to check

        Returns:
            Conflict info dict:
            {
                "conflict": True/False,
                "member_id": "...",  # If conflict
                "recommendation": "delay_rebalance"  # If conflict
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return {"conflict": False, "reason": "hive_unavailable"}

        ok, result, err = self._rpc_call_with_policy(
            "hive-check-rebalance-conflict", {"peer_id": peer_id}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to check rebalance conflict: {err}", level="debug")
            return {"conflict": False, "reason": "exception" if err and err.startswith('exception:') else "check_failed"}
        if result.get("error"):
            self._log(f"Conflict check error: {result.get('error')}", level="debug")
            return {"conflict": False, "reason": "check_failed"}
        return result


    def check_circular_flow_risk(
        self,
        source_peer_id: str,
        dest_peer_id: str
    ) -> Dict[str, Any]:
        """
        Check if source or dest peer is in a known circular flow pattern.

        Fails open (returns risk=False on any error).

        Args:
            source_peer_id: Source peer of the rebalance
            dest_peer_id: Destination peer of the rebalance

        Returns:
            {"risk": True/False, "flow": {...}} if risk detected
        """
        try:
            status = self.query_circular_flow_status()
            flows = status.get("circular_flows", [])

            for flow in flows:
                members = flow.get("members", [])
                if source_peer_id in members or dest_peer_id in members:
                    return {
                        "risk": True,
                        "source_in_flow": source_peer_id in members,
                        "dest_in_flow": dest_peer_id in members,
                        "flow_members": members,
                        "total_cost_sats": flow.get("total_cost_sats", 0)
                    }

            return {"risk": False}
        except Exception as e:
            self._log(f"Circular flow risk check failed: {e}", level="debug")
            return {"risk": False, "reason": "exception"}


    # =========================================================================
    # YIELD OPTIMIZATION PHASE 2: FEE COORDINATION
    # =========================================================================
    # These methods integrate with cl-hive's Phase 2 fee coordination features:
    # - Coordinated fee recommendations (corridor ownership, pheromones, defense)
    # - Stigmergic learning via routing outcome reporting
    # - Collective defense against drain attacks

    def query_coordinated_fee_recommendation(
        self,
        channel_id: str,
        current_fee: int = 500,
        local_balance_pct: float = 0.5,
        source: str = None,
        destination: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query cl-hive for coordinated fee recommendation.

        Uses corridor ownership, pheromone signals, stigmergic markers,
        and defense status to recommend optimal fee that avoids internal
        competition while maximizing fleet-wide yield.

        Args:
            channel_id: Channel to get recommendation for
            current_fee: Current fee in ppm
            local_balance_pct: Current local balance percentage (0.0-1.0)
            source: Source peer hint for corridor lookup (optional)
            destination: Destination peer hint for corridor lookup (optional)

        Returns:
            Fee recommendation dict or None if unavailable:
            {
                "recommended_fee_ppm": 350,
                "is_primary": True,
                "corridor_role": "primary",
                "adjustment_reason": "Corridor primary, competitive rate",
                "pheromone_level": 0.75,
                "defense_multiplier": 1.0,
                "confidence": 0.85,
                "floor_applied": False,
                "ceiling_applied": False
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        params = {
            "channel_id": channel_id,
            "current_fee": current_fee,
            "local_balance_pct": local_balance_pct
        }
        if source:
            params["source"] = source
        if destination:
            params["destination"] = destination

        ok, result, err = self._rpc_call_with_policy(
            "hive-coord-fee-recommendation", params, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query coordinated fee recommendation: {err}", level="debug")
            return None
        if result.get("error"):
            self._log(f"Coordinated fee recommendation error: {result.get('error')}", level="debug")
            return None

        # M-18: Clamp recommended fee to configured bounds
        if 'recommended_fee_ppm' in result:
            # AUDIT FIX HB-3: Type-validate recommended_fee_ppm from hive RPC
            try:
                result['recommended_fee_ppm'] = int(result['recommended_fee_ppm'])
            except (TypeError, ValueError):
                self._log(f"Invalid recommended_fee_ppm type: {type(result['recommended_fee_ppm'])}", level="warn")
                del result['recommended_fee_ppm']
                return result
            min_fee = self.config.min_fee_ppm if self.config else 0
            max_fee = self.config.max_fee_ppm if self.config else 100000
            result['recommended_fee_ppm'] = max(min_fee, min(result['recommended_fee_ppm'], max_fee))
        return result

    def report_routing_outcome(
        self,
        channel_id: str,
        peer_id: str,
        fee_ppm: int,
        success: bool,
        amount_sats: int,
        source: str = None,
        destination: str = None
    ) -> bool:
        """
        Report routing outcome to cl-hive for stigmergic learning.

        This enables pheromone-based fee learning and stigmergic coordination:
        - Success deposits pheromone (reinforces the fee)
        - Failure lets pheromone evaporate (explores new fees)
        - Route markers help fleet coordinate without direct messaging

        Args:
            channel_id: Channel that routed the payment
            peer_id: Peer on this channel
            fee_ppm: Fee charged for this routing
            success: Whether routing succeeded
            amount_sats: Amount routed in satoshis
            source: Source peer (where payment came from)
            destination: Destination peer (where payment went)

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        params = {
            "channel_id": channel_id,
            "peer_id": peer_id,
            "fee_ppm": fee_ppm,
            "success": success,
            "amount_sats": amount_sats,
        }
        if source:
            params["source"] = source
        if destination:
            params["destination"] = destination

        # Use deposit-marker RPC if source/destination provided,
        # otherwise use record-routing-outcome which handles pheromone updates.
        method = "hive-deposit-marker" if (source and destination) else "hive-record-routing-outcome"
        payload = params if (source and destination) else {
            "channel_id": channel_id,
            "peer_id": peer_id,
            "fee_ppm": fee_ppm,
            "success": success,
            "amount_sats": amount_sats,
        }
        ok, result, err = self._rpc_call_with_policy(method, payload, policy_key="telemetry")
        if not ok:
            if err not in ("async_queue_full",):
                self._log(f"Failed to report routing outcome: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Routing outcome report error: {result.get('error')}", level="debug")
            return False
        return True

    def query_defense_status(self, peer_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Query defense status for potential threat peers.

        The mycelium defense system detects drain attacks and unreliable
        peers across the fleet. Use this to apply defensive fee multipliers.

        Args:
            peer_id: Specific peer to check (optional, None for all threats)

        Returns:
            Defense status dict or None:
            {
                "active_warnings": [
                    {
                        "peer_id": "02abc...",
                        "threat_type": "drain",
                        "severity": 0.8,
                        "expires_at": 1705100000,
                        "defensive_multiplier": 2.6
                    }
                ],
                "warning_count": 1,
                "peer_threat": {  # Only if peer_id specified
                    "is_threat": True,
                    "threat_type": "drain",
                    "severity": 0.8,
                    "defensive_multiplier": 2.6
                }
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        params = {}
        if peer_id:
            params["peer_id"] = peer_id

        ok, result, err = self._rpc_call_with_policy(
            "hive-defense-status", params, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query defense status: {err}", level="debug")
            return None
        if result.get("error"):
            self._log(f"Defense status query error: {result.get('error')}", level="debug")
            return None

        if 'defensive_multiplier' in result:
            result['defensive_multiplier'] = max(0.1, min(5.0, result['defensive_multiplier']))
        return result


    def broadcast_fee_observation(
        self,
        peer_id: str,
        fee_ppm: int,
        revenue_rate: float,
        confidence: float,
        discovery_type: str = "observation",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Broadcast a fee observation/discovery to the fleet.

        When Thompson Sampling discovers a particularly successful fee point,
        share this knowledge with the fleet so other members can benefit.
        This is like pheromone trail reinforcement - successful paths get marked.

        Args:
            peer_id: Peer this observation is for
            fee_ppm: Fee that was charged (ppm)
            revenue_rate: Observed revenue rate (sats/hour)
            confidence: Confidence in this observation (0.0-1.0)
            discovery_type: Type of discovery:
                - "observation": Regular observation
                - "high_revenue": Unusually high revenue at this fee
                - "optimal_fee": Confirmed good fee near posterior mean
            metadata: Optional additional data (posterior_mean, etc.)

        Returns:
            True if observation broadcasted successfully
        """
        if not self.is_available():
            return False

        # Rate limit: don't flood hive with observations
        # Only broadcast discoveries, not every observation
        if discovery_type == "observation" and confidence < 0.7:
            return False

        params = {
            "peer_id": peer_id,
            "fee_ppm": fee_ppm,
            "revenue_rate": revenue_rate,
            "confidence": confidence,
            "discovery_type": discovery_type,
            "timestamp": int(time.time())
        }
        if metadata:
            params["metadata"] = metadata

        ok, result, err = self._rpc_call_with_policy(
            "hive-broadcast-fee-observation", params, policy_key="telemetry"
        )
        if not ok:
            if err and "unknown" not in err.lower() and err not in ("async_queue_full",):
                self._log(f"Failed to broadcast fee observation: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Fee observation broadcast error: {result.get('error')}", level="debug")
            return False

        self._log(
            f"Fee observation broadcasted: peer={peer_id[:12]}... fee={fee_ppm}ppm "
            f"revenue={revenue_rate:.1f}sats/hr type={discovery_type} conf={confidence:.2f}",
            level="info"
        )
        return True

    def query_fee_coordination_status(self) -> Optional[Dict[str, Any]]:
        """
        Query overall fee coordination status from cl-hive.

        Provides visibility into corridor assignments, active markers,
        pheromone levels, and defense state.

        Returns:
            Coordination status dict or None:
            {
                "corridor_assignments": [...],
                "active_markers": 15,
                "defense_status": {...},
                "fleet_fee_floor": 50,
                "fleet_fee_ceiling": 2500,
                "our_corridors": {
                    "primary": 5,
                    "secondary": 3
                }
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-fee-coordination-status", {}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query fee coordination status: {err}", level="debug")
            return None
        if result.get("error"):
            self._log(f"Fee coordination status error: {result.get('error')}", level="debug")
            return None
        return result


    def get_channel_routing_intelligence(self, scid: str) -> Optional[Dict[str, Any]]:
        """
        Get routing intelligence for a specific channel.

        Queries cl-hive directly for a single channel's pheromone and
        corridor data.

        Args:
            scid: Channel short_channel_id

        Returns:
            Channel intelligence dict or None:
            {
                "pheromone_level": 3.98,
                "pheromone_trend": "stable",
                "last_forward_age_hours": 2.5,
                "marker_count": 3,
                "on_active_corridor": true
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-get-routing-intelligence",
            {"scid": scid},
            policy_key="optional_read",
            require_available=False,
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query routing intelligence: {err}", level="debug")
            return None

        if result.get("error"):
            self._log(f"Routing intelligence query error: {result.get('error')}", level="debug")
            return None

        if "channels" in result:
            return result["channels"].get(scid)
        return None

    def set_routing_intelligence_cache_ttl(self, ttl_seconds: int) -> None:
        """
        Set the cache TTL for routing intelligence queries.

        Args:
            ttl_seconds: Cache time-to-live in seconds (default 300)
        """
        self._routing_intel_cache_ttl = max(60, min(3600, ttl_seconds))

    # =========================================================================
    # P2 Integration: Elasticity Sharing
    # =========================================================================

    def broadcast_elasticity_observation(
        self,
        peer_id: str,
        elasticity: float,
        confidence: float,
        sample_count: int = 0
    ) -> bool:
        """
        Broadcast elasticity observation to the fleet.

        Shares demand elasticity data so fleet members can learn from each
        other's price sensitivity observations.

        Args:
            peer_id: Peer this elasticity is for
            elasticity: Estimated elasticity (negative = elastic)
            confidence: Confidence in estimate (0.0-1.0)
            sample_count: Number of samples used

        Returns:
            True if broadcasted successfully
        """
        if not self.is_available():
            return False

        # Only broadcast high-confidence elasticity
        if confidence < 0.5:
            return False

        params = {
            "peer_id": peer_id,
            "elasticity": elasticity,
            "confidence": confidence,
            "sample_count": sample_count,
            "timestamp": int(time.time())
        }

        ok, result, err = self._rpc_call_with_policy("hive-broadcast-elasticity", params, policy_key="telemetry")
        if not ok:
            if err and "unknown" not in err.lower() and err not in ("async_queue_full",):
                self._log(f"Failed to broadcast elasticity: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Elasticity broadcast error: {result.get('error')}", level="debug")
            return False

        self._log(
            f"Elasticity broadcasted: peer={peer_id[:12]}... elasticity={elasticity:.2f} conf={confidence:.2f}",
            level="debug"
        )
        return True

    def query_fleet_elasticity(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Query fleet-aggregated elasticity for a peer.

        Gets elasticity data aggregated from multiple fleet members for
        better demand sensitivity estimation.

        Args:
            peer_id: Peer to query elasticity for

        Returns:
            Aggregated elasticity dict or None:
            {
                "peer_id": "02abc...",
                "fleet_elasticity": -1.2,
                "fleet_confidence": 0.75,
                "reporter_count": 3,
                "min_elasticity": -1.8,
                "max_elasticity": -0.6
            }
        """
        if not self.is_available() or self._is_circuit_open():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-query-elasticity", {"peer_id": peer_id}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err and "unknown" not in err.lower():
                self._log(f"Failed to query fleet elasticity: {err}", level="debug")
            return None
        if result.get("error"):
            return None
        return result

    # =========================================================================
    # P2 Integration: Historical Response Curve Aggregation
    # =========================================================================

    def broadcast_curve_observation(
        self,
        peer_id: str,
        fee_ppm: int,
        revenue_rate: float,
        forward_count: int
    ) -> bool:
        """
        Broadcast response curve observation to fleet.

        Shares fee→revenue data points for fleet-wide curve aggregation,
        enabling better market understanding across all members.

        Args:
            peer_id: Peer this observation is for
            fee_ppm: Fee that was charged
            revenue_rate: Observed revenue rate
            forward_count: Number of forwards

        Returns:
            True if broadcasted successfully
        """
        if not self.is_available():
            return False

        # Only broadcast meaningful observations
        if forward_count < 1 or revenue_rate < 1.0:
            return False

        params = {
            "peer_id": peer_id,
            "fee_ppm": fee_ppm,
            "revenue_rate": revenue_rate,
            "forward_count": forward_count,
            "timestamp": int(time.time())
        }

        ok, result, err = self._rpc_call_with_policy("hive-broadcast-curve-observation", params, policy_key="telemetry")
        if not ok:
            if err and "unknown" not in err.lower() and err not in ("async_queue_full",):
                self._log(f"Failed to broadcast curve observation: {err}", level="debug")
            return False
        if result and result.get("error"):
            return False
        return True

    def query_aggregated_curve(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Query fleet-aggregated response curve for a peer.

        Gets fee→revenue curve data aggregated from multiple fleet members.

        Args:
            peer_id: Peer to query curve for

        Returns:
            Aggregated curve dict or None:
            {
                "peer_id": "02abc...",
                "observations": [
                    {"fee_ppm": 100, "avg_revenue": 50.0, "sample_count": 5},
                    {"fee_ppm": 200, "avg_revenue": 75.0, "sample_count": 8},
                    ...
                ],
                "optimal_fee_estimate": 180,
                "confidence": 0.7,
                "reporter_count": 3
            }
        """
        if not self.is_available() or self._is_circuit_open():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-query-aggregated-curve", {"peer_id": peer_id}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err and "unknown" not in err.lower():
                self._log(f"Failed to query aggregated curve: {err}", level="debug")
            return None
        if result.get("error"):
            return None
        return result

    # =========================================================================
    # P2 Integration: Regime Change Coordination
    # =========================================================================

    def broadcast_regime_change(
        self,
        peer_id: str,
        change_type: str,
        old_regime: str,
        new_regime: str,
        evidence: Dict[str, Any] = None
    ) -> bool:
        """
        Broadcast regime change detection to fleet.

        When local analysis detects a market regime change, share with fleet
        so members can coordinate their response.

        Args:
            peer_id: Peer where regime change detected
            change_type: Type of change ("demand_shift", "competition", "seasonal")
            old_regime: Previous regime description
            new_regime: New regime description
            evidence: Supporting data for the detection

        Returns:
            True if broadcasted successfully
        """
        if not self.is_available():
            return False

        params = {
            "peer_id": peer_id,
            "change_type": change_type,
            "old_regime": old_regime,
            "new_regime": new_regime,
            "timestamp": int(time.time())
        }
        if evidence:
            params["evidence"] = evidence

        ok, result, err = self._rpc_call_with_policy("hive-broadcast-regime-change", params, policy_key="telemetry")
        if not ok:
            if err and "unknown" not in err.lower() and err not in ("async_queue_full",):
                self._log(f"Failed to broadcast regime change: {err}", level="debug")
            return False
        if result and result.get("error"):
            return False

        self._log(
            f"Regime change broadcasted: peer={peer_id[:12]}... type={change_type} {old_regime}->{new_regime}",
            level="info"
        )
        return True

    def query_fleet_regime_status(self, peer_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Query fleet-wide regime status.

        Check if other fleet members have detected regime changes for a peer,
        which helps validate local detections and reduce false positives.

        Args:
            peer_id: Specific peer to check (None for all recent changes)

        Returns:
            Fleet regime status dict or None:
            {
                "recent_changes": [
                    {
                        "peer_id": "02abc...",
                        "change_type": "demand_shift",
                        "reporters": 2,
                        "first_detected": 1705000000,
                        "confidence": 0.8
                    }
                ],
                "peer_status": {  # Only if peer_id specified
                    "regime_stable": False,
                    "recent_change": True,
                    "change_type": "demand_shift",
                    "fleet_consensus": 0.7
                }
            }
        """
        if not self.is_available() or self._is_circuit_open():
            return None

        params = {}
        if peer_id:
            params["peer_id"] = peer_id

        ok, result, err = self._rpc_call_with_policy(
            "hive-query-regime-status", params, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err and "unknown" not in err.lower():
                self._log(f"Failed to query fleet regime status: {err}", level="debug")
            return None
        if result.get("error"):
            return None
        return result

    # =========================================================================
    # P2 Integration: Thompson Posterior Sharing
    # =========================================================================

    def share_posterior_summary(
        self,
        peer_id: str,
        posterior_mean: float,
        posterior_std: float,
        observation_count: int,
        corridor_role: str = "P"
    ) -> bool:
        """
        Share Thompson posterior summary with fleet.

        Enables coordination by sharing what fee range we think is optimal.
        Primary corridors share to help secondaries avoid undercutting.

        Args:
            peer_id: Peer this posterior is for
            posterior_mean: Current posterior mean fee
            posterior_std: Current posterior std
            observation_count: Number of observations
            corridor_role: Our role ("P" primary, "S" secondary)

        Returns:
            True if shared successfully
        """
        if not self.is_available():
            return False

        # Only share if we have meaningful data
        if observation_count < 5:
            return False

        params = {
            "peer_id": peer_id,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "observation_count": observation_count,
            "corridor_role": corridor_role,
            "timestamp": int(time.time())
        }

        ok, result, err = self._rpc_call_with_policy("hive-share-posterior", params, policy_key="telemetry")
        if not ok:
            if err and "unknown" not in err.lower() and err not in ("async_queue_full",):
                self._log(f"Failed to share posterior: {err}", level="debug")
            return False
        if result and result.get("error"):
            return False
        return True

    def query_fleet_posteriors(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Query fleet posterior summaries for a peer.

        Gets Thompson posterior data from other fleet members for coordination.
        Useful for secondary corridors to avoid undercutting primaries.

        Args:
            peer_id: Peer to query posteriors for

        Returns:
            Fleet posteriors dict or None:
            {
                "peer_id": "02abc...",
                "posteriors": [
                    {
                        "member_id": "03def...",
                        "corridor_role": "P",
                        "posterior_mean": 200.0,
                        "posterior_std": 30.0,
                        "observation_count": 50
                    },
                    ...
                ],
                "primary_mean": 200.0,  # Weighted mean of primaries
                "secondary_mean": 180.0,  # Weighted mean of secondaries
                "fleet_consensus_fee": 195.0  # Overall fleet estimate
            }
        """
        if not self.is_available() or self._is_circuit_open():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-query-posteriors", {"peer_id": peer_id}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err and "unknown" not in err.lower():
                self._log(f"Failed to query fleet posteriors: {err}", level="debug")
            return None
        if result.get("error"):
            return None
        return result


    def query_fleet_rebalance_path(
        self,
        from_channel: str,
        to_channel: str,
        amount_sats: int
    ) -> Optional[Dict[str, Any]]:
        """
        Check if rebalancing through fleet members is cheaper.

        Fleet members have coordinated fees, so internal routes
        may be cheaper than external paths.

        Args:
            from_channel: Source channel SCID
            to_channel: Destination channel SCID
            amount_sats: Amount to rebalance

        Returns:
            Fleet path recommendation or None:
            {
                "fleet_path_available": True,
                "fleet_path": ["node1", "node2"],
                "estimated_fleet_cost_sats": 150,
                "estimated_external_cost_sats": 500,
                "savings_pct": 70,
                "recommendation": "use_fleet_path"
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-fleet-rebalance-path",
            {"from_channel": from_channel, "to_channel": to_channel, "amount_sats": amount_sats},
            policy_key="optional_read",
            require_available=False,
        )
        if not ok or result is None:
            if err and "unknown" not in err.lower():
                self._log(f"Failed to query fleet rebalance path: {err}", level="debug")
            return None
        if result.get("error"):
            if "unknown" in str(result.get("error")).lower():
                return None
            self._log(f"Fleet path query error: {result.get('error')}", level="debug")
            return None
        return result

    def execute_circular_rebalance(
        self,
        from_channel: str,
        to_channel: str,
        amount_sats: int,
        via_members: list = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a circular rebalance through hive fleet members.

        Attempts a zero-fee circular rebalance where liquidity flows through
        fleet members who all benefit from the movement. Falls back gracefully
        if the RPC is not available or the operation fails.

        Args:
            from_channel: Source channel SCID
            to_channel: Destination channel SCID
            amount_sats: Amount to rebalance
            via_members: Optional list of member pubkeys to route through

        Returns:
            Result dict or None:
            {
                "success": True,
                "cost_sats": 0,
                "path": ["node1", "node2"],
                "amount_sats": 50000
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        # HB-6: Validate amount_sats to prevent nonsensical RPC calls
        try:
            amount_sats = int(amount_sats)
        except (TypeError, ValueError):
            self.plugin.log(f"Invalid amount_sats type: {type(amount_sats)}", level='debug')
            return None
        if amount_sats <= 0 or amount_sats > 100_000_000:  # Cap at 1 BTC
            self.plugin.log(f"amount_sats out of range: {amount_sats}", level='debug')
            return None

        params = {
            "from_channel": from_channel,
            "to_channel": to_channel,
            "amount_sats": amount_sats,
            "dry_run": False,
        }
        if via_members:
            params["via_members"] = via_members

        ok, result, err = self._rpc_call_with_policy(
            "hive-execute-circular-rebalance", params, policy_key="critical_write", require_available=False
        )
        if not ok or result is None:
            if err and "unknown" not in err.lower():
                self._log(f"Failed to execute circular rebalance: {err}", level="debug")
            return None
        if result.get("error"):
            if "unknown" in str(result.get("error")).lower():
                return None
            self._log(f"Circular rebalance error: {result.get('error')}", level="debug")
            return None
        return result


    def report_rebalance_outcome(
        self,
        from_channel: str,
        to_channel: str,
        amount_sats: int,
        cost_sats: int,
        success: bool,
        via_fleet: bool = False,
        failure_reason: str = ""
    ) -> bool:
        """
        Report rebalance outcome for fleet coordination.

        Helps detect circular flows (A→B→C→A) that waste fees
        and enables better rebalance coordination across fleet.

        Args:
            from_channel: Source channel SCID
            to_channel: Destination channel SCID
            amount_sats: Amount rebalanced
            cost_sats: Cost of rebalancing
            success: Whether rebalance succeeded
            via_fleet: Whether routed through fleet members

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        payload = {
            "from_channel": from_channel,
            "to_channel": to_channel,
            "amount_sats": amount_sats,
            "cost_sats": cost_sats,
            "success": success,
            "via_fleet": via_fleet,
        }
        if failure_reason:
            payload["failure_reason"] = failure_reason

        ok, result, err = self._rpc_call_with_policy(
            "hive-report-rebalance-outcome", payload, policy_key="telemetry"
        )
        if not ok:
            # Don't block on this telemetry path, but don't lie about success.
            if err not in ("async_queue_full",) and err and "unknown" not in err.lower():
                self._log(f"Failed to report rebalance outcome: {err}", level="debug")
            return False
        if result and result.get("error"):
            if "unknown" in str(result.get("error")).lower():
                return False
            self._log(f"Rebalance outcome report error: {result.get('error')}", level="debug")
            return False
        return True


    def query_pheromone_level(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Query pheromone level for a specific channel.

        Pheromones represent the "memory" of successful fee levels - higher
        levels indicate more routing success at certain fees. Use this to
        inform fee starting points.

        Args:
            channel_id: Channel SCID to query

        Returns:
            Pheromone data or None:
            {
                "channel_id": "123x456x0",
                "level": 0.75,
                "successful_fee_ppm": 350,  # Fee that worked well
                "above_threshold": True      # High confidence signal
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-pheromone-levels", {"channel_id": channel_id}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to query pheromone level: {err}", level="debug")
            return None
        if result.get("error"):
            return None

        # Extract relevant data for this channel
        # Prefer list format under "pheromone_levels" key
        levels = result.get("pheromone_levels", [])
        for level_data in levels:
            if level_data.get("channel_id") == channel_id:
                self._record_success()
                return {
                    "channel_id": channel_id,
                    "level": level_data.get("level", 0),
                    "above_threshold": level_data.get("above_threshold", False)
                }

        # Fallback: handle flat dict response (legacy cl-hive format)
        if result.get("channel_id") == channel_id:
            self._record_success()
            return {
                "channel_id": channel_id,
                "level": result.get("pheromone_level", result.get("level", 0)),
                "above_threshold": result.get("above_exploit_threshold",
                                              result.get("above_threshold", False))
            }

        # Channel exists but no pheromone data yet
        self._record_success()
        return {"channel_id": channel_id, "level": 0, "above_threshold": False}

    def check_internal_competition_for_peer(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if we're competing with other fleet members for routes to/from this peer.

        Use this before adjusting fees to avoid undercutting fleet members.

        Args:
            peer_id: Peer pubkey to check

        Returns:
            Competition info or None:
            {
                "is_competing": True,
                "competing_members": ["hive-nexus-01", "hive-nexus-02"],
                "our_role": "secondary",  # "primary", "secondary", or "none"
                "recommended_action": "defer_to_primary",
                "primary_fee_ppm": 300  # What the primary is charging
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        ok, result, err = self._rpc_call_with_policy(
            "hive-internal-competition", {}, policy_key="optional_read", require_available=False
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to check internal competition: {err}", level="debug")
            return None
        if result.get("error"):
            return None

        # Check if this peer is in any competing routes
        competing_routes = result.get("competing_routes", [])
        for route in competing_routes:
            if route.get("destination") == peer_id or route.get("source") == peer_id:
                self._record_success()
                return {
                    "is_competing": True,
                    "competing_members": route.get("competing_members", []),
                    "member_count": route.get("member_count", 0),
                    "recommendation": route.get("recommendation", "coordinate_fees")
                }

        self._record_success()
        return {"is_competing": False}

    # =========================================================================
    # YIELD OPTIMIZATION: YIELD METRICS REPORTING
    # =========================================================================
    # Report yield metrics to cl-hive for fleet-wide tracking

    def report_yield_metrics(
        self,
        tlv_sats: int,
        operating_costs_sats: int,
        routing_revenue_sats: int,
        period_days: int = 30
    ) -> bool:
        """
        Report yield metrics to cl-hive for fleet aggregation.

        Args:
            tlv_sats: Total Lightning Value (capacity under management)
            operating_costs_sats: Operating costs in period (rebalancing, opens, etc.)
            routing_revenue_sats: Routing revenue in period
            period_days: Period length in days

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        if self._is_circuit_open():
            return False

        ok, result, err = self._rpc_call_with_policy(
            "hive-report-yield-metrics",
            {
                "tlv_sats": tlv_sats,
                "operating_costs_sats": operating_costs_sats,
                "routing_revenue_sats": routing_revenue_sats,
                "period_days": period_days
            },
            policy_key="critical_write",
            require_available=False,  # checked above
        )
        if not ok or result is None:
            if err:
                self._log(f"Failed to report yield metrics: {err}", level="debug")
            return False
        if result.get("error"):
            self._log(f"Yield metrics report error: {result.get('error')}", level="debug")
            return False
        return True


    def report_flow_observation(
        self,
        channel_id: str,
        inbound_sats: int,
        outbound_sats: int,
        timestamp: int = None
    ) -> bool:
        """
        Report a flow observation to cl-hive for pattern building.

        Should be called periodically (e.g., hourly) to build the historical
        data needed for temporal pattern detection.

        Args:
            channel_id: Channel SCID
            inbound_sats: Satoshis received in this period
            outbound_sats: Satoshis sent in this period
            timestamp: Observation timestamp (defaults to now)

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        params = {
            "channel_id": channel_id,
            "inbound_sats": inbound_sats,
            "outbound_sats": outbound_sats
        }
        if timestamp:
            params["timestamp"] = timestamp

        ok, result, err = self._rpc_call_with_policy("hive-record-flow", params, policy_key="telemetry")
        if not ok:
            if err not in ("async_queue_full",):
                self._log(f"Failed to report flow observation: {err}", level="debug")
            return False
        if result and result.get("error"):
            self._log(f"Flow report error: {result.get('error')}", level="debug")
            return False
        return True


    # =========================================================================
    # TIME-BASED FEE QUERIES (Phase 7.4)
    # =========================================================================

    def query_time_fee_adjustment(
        self,
        channel_id: str,
        base_fee: int = 250
    ) -> Optional[Dict[str, Any]]:
        """
        Query time-based fee adjustment for a channel from cl-hive.

        Uses temporal patterns to determine if current time is peak or low activity,
        returning adjusted fee recommendation.

        Args:
            channel_id: Channel SCID
            base_fee: Current/base fee in ppm (default: 250)

        Returns:
            Adjustment dict or None:
            {
                "channel_id": "123x1x0",
                "base_fee_ppm": 250,
                "adjusted_fee_ppm": 275,
                "adjustment_pct": 10.0,
                "adjustment_type": "peak_increase",
                "current_hour": 14,
                "current_day": 2,
                "pattern_intensity": 0.85,
                "confidence": 0.75,
                "reason": "Peak outbound hour (85% intensity, +10%)"
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            ok, result, err = self._rpc_call_with_policy(
                "hive-time-fee-adjustment",
                {"channel_id": channel_id, "base_fee": base_fee},
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query time fee adjustment: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"Time fee query error: {result.get('error')}", level="debug")
                return None

            return result

        except Exception as e:
            self._log(f"Failed to query time fee adjustment (unexpected): {e}", level="debug")
            return None


    # =========================================================================
    # MCF (MIN-COST MAX-FLOW) OPTIMIZATION METHODS
    # =========================================================================


    def claim_mcf_assignment(
        self,
        assignment_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Claim an MCF assignment for execution.

        Marks the assignment as "executing" to prevent double execution
        by multiple rebalancing attempts.

        Args:
            assignment_id: Specific assignment to claim, or None for next pending

        Returns:
            Claimed assignment dict or None if claim failed
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            payload = {"assignment_id": assignment_id} if assignment_id else {}
            ok, result, err = self._rpc_call_with_policy(
                "hive-claim-mcf-assignment",
                payload,
                policy_key="critical_write",
                require_available=False,
            )
            if not ok or result is None:
                self._log(f"Error claiming MCF assignment: {err}", level="debug")
                return None

            if result.get("success"):
                return result.get("assignment")
            return None

        except Exception as e:
            self._log(f"Error claiming MCF assignment (unexpected): {e}", level="debug")
            return None

    def should_use_mcf_path(
        self,
        from_channel: str,
        to_channel: str,
        amount_sats: int,
        max_fee_ppm: int
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if MCF-optimized path should be used for rebalancing.

        Queries MCF for optimal path and validates it meets fee constraints.

        Args:
            from_channel: Source channel SCID
            to_channel: Destination channel SCID
            amount_sats: Amount to rebalance
            max_fee_ppm: Maximum acceptable fee in ppm

        Returns:
            Tuple of (should_use, path_info)
        """
        path_info = self.query_mcf_optimized_path(
            from_channel, to_channel, amount_sats
        )

        if not path_info:
            return False, None

        if not path_info.get("fleet_path_available"):
            return False, path_info

        # Check if cost is within budget
        estimated_cost = path_info.get("estimated_fleet_cost_sats", 0)
        estimated_ppm = (estimated_cost * 1_000_000) // amount_sats if amount_sats > 0 else 999999

        if estimated_ppm > max_fee_ppm:
            self._log(
                f"MCF path too expensive: {estimated_ppm}ppm > {max_fee_ppm}ppm",
                level="debug"
            )
            return False, path_info

        return True, path_info

    # =========================================================================
    # DIAGNOSTIC METHODS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get bridge status for diagnostics.

        Returns:
            Dict with bridge status information including membership details
        """
        now = time.time()

        # Count fresh vs stale cache entries (under lock to avoid RuntimeError
        # from concurrent dict mutation)
        fresh_count = 0
        stale_count = 0
        with self._cache_lock:
            cache_size = len(self._cache)
            for cached in list(self._cache.values()):
                age = now - cached.timestamp
                if age < CACHE_TTL_SECONDS:
                    fresh_count += 1
                elif age < STALE_CACHE_TTL_SECONDS:
                    stale_count += 1

        # Read circuit breaker state under its lock for consistency
        with self._circuit_lock:
            circuit_open = self._circuit.is_open
            circuit_failures = self._circuit.failures

        status = {
            "hive_available": self._hive_available,
            "circuit_breaker_enabled": self._circuit_breaker_enabled,
            "circuit_breaker_open": circuit_open if self._circuit_breaker_enabled else False,
            "circuit_failures": circuit_failures,
            "cache_entries": cache_size,
            "cache_fresh": fresh_count,
            "cache_stale": stale_count,
            "last_availability_check": int(self._availability_check_time),
            "membership": None
        }
        with self._rpc_policy_stats_lock:
            status["rpc_policy_stats"] = {
                name: {
                    "calls": stats.calls,
                    "success": stats.success,
                    "failures": stats.failures,
                    "retries": stats.retries,
                    "short_circuit": stats.short_circuit,
                    "unavailable": stats.unavailable,
                    "async_queued": stats.async_queued,
                    "async_dropped": stats.async_dropped,
                    "last_error": stats.last_error,
                    "last_error_at": stats.last_error_at,
                }
                for name, stats in self._rpc_policy_stats.items()
            }
        with self._policy_circuit_lock:
            status["rpc_policy_breakers"] = {
                name: {
                    "enabled": bool(
                        self._circuit_breaker_enabled and
                        (self._rpc_policies.get(name).use_breaker if self._rpc_policies.get(name) else False)
                    ),
                    "open": state.is_open if self._circuit_breaker_enabled else False,
                    "failures": state.failures,
                    "last_failure": int(state.last_failure or 0),
                }
                for name, state in self._policy_circuits.items()
            }

        # If hive is available, get membership details
        if self._hive_available:
            try:
                ok, hive_status, err = self._rpc_call_with_policy(
                    "hive-status",
                    {},
                    policy_key="optional_read",
                    require_available=False,
                    count_error_response_failure=False,
                )
                if not ok or hive_status is None:
                    raise RuntimeError(err or "hive-status unavailable")
                membership = hive_status.get("membership", {})
                status["membership"] = {
                    "tier": membership.get("tier"),
                    "hive_id": membership.get("hive_id"),
                    "joined_at": membership.get("joined_at"),
                    "uptime_pct": membership.get("uptime_pct", 0.0),
                    "contribution_ratio": membership.get("contribution_ratio", 0.0)
                }
            except Exception as e:
                self._log(f"Error fetching membership details: {e}", level="debug")

        return status


    # =========================================================================
    # SETTLEMENT INTEGRATION (Issue #42: Net Profit Settlement)
    # =========================================================================

    def report_period_costs(self, rebalance_costs_sats: int, boltz_costs_sats: int = 0) -> bool:
        """
        Report rebalancing and Boltz costs to cl-hive for net profit settlement.

        This is non-critical telemetry. We intentionally avoid a blocking RPC
        round-trip so lock contention in cl-hive fee gossip paths cannot stall
        cl-revenue-ops or trip the hive bridge circuit breaker.

        Args:
            rebalance_costs_sats: Total rebalancing costs in sats for the period
            boltz_costs_sats: Boltz swap costs in sats for the period

        Returns:
            True if the report was queued (or recently coalesced), False if not.
        """
        if not self.is_available():
            self._log("Cannot report period costs - cl-hive not available", level="debug")
            return False

        if rebalance_costs_sats < 0:
            rebalance_costs_sats = 0

        now = time.time()
        with self._period_costs_report_lock:
            # Coalesce duplicate values on a short interval; this call may be
            # made frequently from fee/yield loops and does not need exact ack.
            total_costs = rebalance_costs_sats + max(0, int(boltz_costs_sats or 0))
            if (
                self._last_reported_period_costs_sats == total_costs
                and (now - self._last_period_costs_report_time) < self._period_costs_report_min_interval
            ):
                return True
            self._last_reported_period_costs_sats = total_costs
            self._last_period_costs_report_time = now

        params = {"rebalance_costs_sats": rebalance_costs_sats}
        if boltz_costs_sats > 0:
            params["boltz_costs_sats"] = int(boltz_costs_sats)
        ok, result, err = self._rpc_call_with_policy(
            "hive-report-period-costs",
            params,
            policy_key="telemetry",
            require_available=False,  # checked above
            count_error_response_failure=False,
        )
        if not ok:
            if err == "async_queue_full":
                self._log("Period cost report dropped (async RPC queue full)", level="debug")
            elif err:
                self._log(f"Failed to report period costs (non-critical): {err}", level="debug")
            return False
        if isinstance(result, dict) and result.get("error"):
            self._log(f"Error reporting period costs: {result.get('error')}", level="debug")
            return False
        return True

    def report_yield_and_costs(
        self,
        tlv_sats: int,
        operating_costs_sats: int,
        routing_revenue_sats: int,
        rebalance_costs_sats: int,
        period_days: int = 30,
        boltz_costs_sats: int = 0,
    ) -> bool:
        """
        Report both yield metrics and settlement period costs to cl-hive.

        Convenience method that combines report_yield_metrics() and
        report_period_costs() for use in periodic reporting loops.

        Args:
            tlv_sats: Total Lightning Value (capacity under management)
            operating_costs_sats: Total operating costs in period
            routing_revenue_sats: Routing revenue in period
            rebalance_costs_sats: Rebalance costs for settlement (may overlap with operating_costs)
            period_days: Period length in days
            boltz_costs_sats: Boltz swap costs in sats for the period

        Returns:
            True if both reports succeeded
        """
        yield_ok = self.report_yield_metrics(
            tlv_sats, operating_costs_sats, routing_revenue_sats, period_days
        )
        costs_ok = self.report_period_costs(rebalance_costs_sats, boltz_costs_sats=boltz_costs_sats)
        return yield_ok and costs_ok

    # =========================================================================
    # COMPREHENSIVE DATA INTEGRATION (Revenue Ops Enhancement)
    # =========================================================================
    # These methods query cl-hive for additional data to improve fee
    # optimization and rebalancing decisions.

    # Cache TTLs for different data types (seconds)
    _DEFENSE_STATUS_CACHE_TTL = 60       # 1 minute - attacks are time-sensitive
    _PEER_QUALITY_CACHE_TTL = 300        # 5 minutes
    _FEE_HISTORY_CACHE_TTL = 300         # 5 minutes
    _CHANNEL_FLAGS_CACHE_TTL = 300       # 5 minutes
    _MCF_TARGETS_CACHE_TTL = 300         # 5 minutes
    _NNLB_CACHE_TTL = 120                # 2 minutes
    _CHANNEL_AGES_CACHE_TTL = 3600       # 1 hour - ages change slowly

    def _get_from_cache(self, cache_key: str, ttl: float) -> Optional[Dict[str, Any]]:
        """Get data from integration cache if fresh."""
        with self._intel_cache_lock:
            cached = self._integration_cache.get(cache_key)
        if cached:
            age = time.time() - cached.get('_cache_time', 0)
            if age < ttl:
                return cached.get('data')
        return None

    def _set_in_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store data in integration cache."""
        with self._intel_cache_lock:
            self._integration_cache[cache_key] = {
                'data': data,
                '_cache_time': time.time()
            }

            # Limit cache size
            if len(self._integration_cache) > 200:
                oldest_key = min(
                    self._integration_cache.keys(),
                    key=lambda k: self._integration_cache[k].get('_cache_time', 0)
                )
                del self._integration_cache[oldest_key]

    def get_defense_status(self, scid: str = None) -> Optional[Dict[str, Any]]:
        """
        Get defense status for channel(s) from cl-hive.

        Returns whether channels are under defensive fee protection due to
        drain attacks, spam, or fee wars. Used to avoid overriding defensive
        fees during optimization.

        Args:
            scid: Optional specific channel SCID. If None, returns all channels.

        Returns:
            Dict with defense status or None if unavailable:
            {
                "channels": {
                    "932263x1883x0": {
                        "under_defense": false,
                        "defense_type": null,
                        "defensive_fee_ppm": null,
                        "defense_started_at": null,
                        "defense_reason": null
                    }
                }
            }
        """
        cache_key = f"defense_status_{scid or 'all'}"
        cached = self._get_from_cache(cache_key, self._DEFENSE_STATUS_CACHE_TTL)
        if cached:
            return cached

        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            params = {}
            if scid:
                params["scid"] = scid

            ok, result, err = self._rpc_call_with_policy(
                "hive-get-defense-status",
                params,
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query defense status: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"Defense status query error: {result.get('error')}", level="debug")
                return None

            self._set_in_cache(cache_key, result)
            return result

        except Exception as e:
            self._log(f"Failed to query defense status (unexpected): {e}", level="debug")
            return None

    def get_channel_defense_status(self, scid: str) -> Optional[Dict[str, Any]]:
        """
        Get defense status for a specific channel.

        Convenience method that extracts single channel data.

        Args:
            scid: Channel short_channel_id

        Returns:
            Channel defense dict or None:
            {
                "under_defense": false,
                "defense_type": null,
                "defensive_fee_ppm": null,
                "defense_started_at": null,
                "defense_reason": null
            }
        """
        result = self.get_defense_status(scid=scid)
        if result and "channels" in result:
            return result["channels"].get(scid)
        return None

    def get_peer_quality(self, peer_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get peer quality assessments from cl-hive.

        Returns quality ratings based on uptime, routing success, fee stability,
        and fleet-wide reputation. Used to adjust optimization intensity.

        Args:
            peer_id: Optional specific peer ID. If None, returns all peers.

        Returns:
            Dict with peer quality or None if unavailable:
            {
                "peers": {
                    "03abc...": {
                        "quality": "good",
                        "quality_score": 0.85,
                        "reasons": ["high_uptime", "good_routing_partner"],
                        "recommendation": "expand",
                        "last_assessed": 1707600000
                    }
                }
            }
        """
        cache_key = f"peer_quality_{peer_id or 'all'}"
        cached = self._get_from_cache(cache_key, self._PEER_QUALITY_CACHE_TTL)
        if cached:
            return cached

        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            params = {}
            if peer_id:
                params["peer_id"] = peer_id

            ok, result, err = self._rpc_call_with_policy(
                "hive-get-peer-quality",
                params,
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query peer quality: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"Peer quality query error: {result.get('error')}", level="debug")
                return None

            self._set_in_cache(cache_key, result)
            return result

        except Exception as e:
            self._log(f"Failed to query peer quality (unexpected): {e}", level="debug")
            return None

    def get_single_peer_quality(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get quality assessment for a specific peer.

        Convenience method that extracts single peer data.

        Args:
            peer_id: Peer public key

        Returns:
            Peer quality dict or None:
            {
                "quality": "good",
                "quality_score": 0.85,
                "reasons": ["high_uptime", "good_routing_partner"],
                "recommendation": "expand",
                "last_assessed": 1707600000
            }
        """
        result = self.get_peer_quality(peer_id=peer_id)
        if result and "peers" in result:
            return result["peers"].get(peer_id)
        return None

    def get_fee_change_outcomes(
        self,
        scid: str = None,
        days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get outcomes of past fee changes from cl-hive for learning.

        Returns historical fee changes with before/after metrics to help
        learn from past decisions and adjust Thompson priors.

        Args:
            scid: Optional specific channel SCID. If None, returns all.
            days: Number of days of history (default: 30, max: 90)

        Returns:
            Dict with fee change outcomes or None:
            {
                "changes": [
                    {
                        "scid": "932263x1883x0",
                        "timestamp": 1707500000,
                        "old_fee_ppm": 200,
                        "new_fee_ppm": 300,
                        "source": "advisor",
                        "outcome": {
                            "forwards_before_24h": 5,
                            "forwards_after_24h": 3,
                            "revenue_before_24h": 500,
                            "revenue_after_24h": 600,
                            "verdict": "positive"
                        }
                    }
                ]
            }
        """
        cache_key = f"fee_outcomes_{scid or 'all'}_{days}"
        cached = self._get_from_cache(cache_key, self._FEE_HISTORY_CACHE_TTL)
        if cached:
            return cached

        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            params = {"days": days}
            if scid:
                params["scid"] = scid

            ok, result, err = self._rpc_call_with_policy(
                "hive-get-fee-change-outcomes",
                params,
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query fee change outcomes: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"Fee outcomes query error: {result.get('error')}", level="debug")
                return None

            self._set_in_cache(cache_key, result)
            return result

        except Exception as e:
            self._log(f"Failed to query fee change outcomes (unexpected): {e}", level="debug")
            return None

    def get_channel_flags(self, scid: str = None) -> Optional[Dict[str, Any]]:
        """
        Get special flags for channels from cl-hive.

        Returns flags identifying hive-internal channels that should be excluded
        from optimization (always 0 fee) or have other special treatment.

        Args:
            scid: Optional specific channel SCID. If None, returns all.

        Returns:
            Dict with channel flags or None:
            {
                "channels": {
                    "932263x1883x0": {
                        "is_hive_internal": false,
                        "is_hive_member": false,
                        "fixed_fee": null,
                        "exclude_from_optimization": false
                    }
                }
            }
        """
        cache_key = f"channel_flags_{scid or 'all'}"
        cached = self._get_from_cache(cache_key, self._CHANNEL_FLAGS_CACHE_TTL)
        if cached:
            return cached

        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            params = {}
            if scid:
                params["scid"] = scid

            ok, result, err = self._rpc_call_with_policy(
                "hive-get-channel-flags",
                params,
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query channel flags: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"Channel flags query error: {result.get('error')}", level="debug")
                return None

            self._set_in_cache(cache_key, result)
            return result

        except Exception as e:
            self._log(f"Failed to query channel flags (unexpected): {e}", level="debug")
            return None

    def get_single_channel_flags(self, scid: str) -> Optional[Dict[str, Any]]:
        """
        Get flags for a specific channel.

        Convenience method that extracts single channel data.

        Args:
            scid: Channel short_channel_id

        Returns:
            Channel flags dict or None
        """
        result = self.get_channel_flags(scid=scid)
        if result and "channels" in result:
            return result["channels"].get(scid)
        return None

    def get_mcf_targets(self) -> Optional[Dict[str, Any]]:
        """
        Get MCF-computed optimal balance targets from cl-hive.

        Returns the Multi-Commodity Flow computed optimal local balance
        percentages for each channel. Used to guide rebalancing toward
        globally optimal distribution.

        Returns:
            Dict with MCF targets or None:
            {
                "targets": {
                    "932263x1883x0": {
                        "optimal_local_pct": 45,
                        "current_local_pct": 30,
                        "delta_sats": 150000,
                        "priority": "high"
                    }
                },
                "computed_at": 1707600000
            }
        """
        cache_key = "mcf_targets"
        cached = self._get_from_cache(cache_key, self._MCF_TARGETS_CACHE_TTL)
        if cached:
            return cached

        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            ok, result, err = self._rpc_call_with_policy(
                "hive-get-mcf-targets",
                {},
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query MCF targets: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"MCF targets query error: {result.get('error')}", level="debug")
                return None

            self._set_in_cache(cache_key, result)
            return result

        except Exception as e:
            self._log(f"Failed to query MCF targets (unexpected): {e}", level="debug")
            return None

    def get_nnlb_opportunities(self, min_amount: int = 50000) -> Optional[Dict[str, Any]]:
        """
        Get Nearest-Neighbor Load Balancing opportunities from cl-hive.

        Returns low-cost rebalance opportunities between fleet members where
        the rebalance can be done at zero or minimal fee.

        Args:
            min_amount: Minimum amount in sats to consider (default: 50000)

        Returns:
            Dict with NNLB opportunities or None:
            {
                "opportunities": [
                    {
                        "source_scid": "932263x1883x0",
                        "sink_scid": "931308x1256x0",
                        "amount_sats": 200000,
                        "estimated_cost_sats": 0,
                        "path_hops": 1,
                        "is_hive_internal": true
                    }
                ]
            }
        """
        cache_key = f"nnlb_opps_{min_amount}"
        cached = self._get_from_cache(cache_key, self._NNLB_CACHE_TTL)
        if cached:
            return cached

        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            ok, result, err = self._rpc_call_with_policy(
                "hive-get-nnlb-opportunities",
                {"min_amount": min_amount},
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query NNLB opportunities: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"NNLB opportunities query error: {result.get('error')}", level="debug")
                return None

            self._set_in_cache(cache_key, result)
            return result

        except Exception as e:
            self._log(f"Failed to query NNLB opportunities (unexpected): {e}", level="debug")
            return None

    def get_channel_ages(self, scid: str = None) -> Optional[Dict[str, Any]]:
        """
        Get channel age information from cl-hive.

        Returns age and maturity classification for channels. Used to adjust
        exploration vs exploitation in Thompson sampling - new channels need
        more exploration, mature channels should exploit known-good fees.

        Args:
            scid: Optional specific channel SCID. If None, returns all.

        Returns:
            Dict with channel ages or None:
            {
                "channels": {
                    "932263x1883x0": {
                        "age_days": 45,
                        "maturity": "mature",
                        "first_forward_days_ago": 40,
                        "total_forwards": 250
                    }
                }
            }
        """
        cache_key = f"channel_ages_{scid or 'all'}"
        cached = self._get_from_cache(cache_key, self._CHANNEL_AGES_CACHE_TTL)
        if cached:
            return cached

        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            params = {}
            if scid:
                params["scid"] = scid

            ok, result, err = self._rpc_call_with_policy(
                "hive-get-channel-ages",
                params,
                policy_key="optional_read",
                require_available=False,
            )
            if not ok or result is None:
                if err:
                    self._log(f"Failed to query channel ages: {err}", level="debug")
                return None

            if result.get("error"):
                self._log(f"Channel ages query error: {result.get('error')}", level="debug")
                return None

            self._set_in_cache(cache_key, result)
            return result

        except Exception as e:
            self._log(f"Failed to query channel ages (unexpected): {e}", level="debug")
            return None

    def get_single_channel_age(self, scid: str) -> Optional[Dict[str, Any]]:
        """
        Get age info for a specific channel.

        Convenience method that extracts single channel data.

        Args:
            scid: Channel short_channel_id

        Returns:
            Channel age dict or None:
            {
                "age_days": 45,
                "maturity": "mature",
                "first_forward_days_ago": 40,
                "total_forwards": 250
            }
        """
        result = self.get_channel_ages(scid=scid)
        if result and "channels" in result:
            return result["channels"].get(scid)
        return None

    def is_channel_excluded_from_optimization(self, scid: str) -> bool:
        """
        Check if a channel should be excluded from fee optimization.

        Convenience method to quickly check if a channel is hive-internal
        or otherwise flagged for exclusion.

        Args:
            scid: Channel short_channel_id

        Returns:
            True if channel should be excluded, False otherwise
        """
        flags = self.get_single_channel_flags(scid)
        if flags:
            return flags.get('exclude_from_optimization', False)
        return False

    def is_channel_under_defense(self, scid: str) -> bool:
        """
        Check if a channel is currently under defensive fee protection.

        Convenience method to quickly check defense status.

        Args:
            scid: Channel short_channel_id

        Returns:
            True if channel is under defense, False otherwise
        """
        defense = self.get_channel_defense_status(scid)
        if defense:
            return defense.get('under_defense', False)
        return False

    def get_defensive_fee_floor(self, scid: str) -> Optional[int]:
        """
        Get the defensive fee floor for a channel under attack.

        Returns the minimum fee that should be charged to protect against
        drain attacks or other threats.

        Args:
            scid: Channel short_channel_id

        Returns:
            Minimum fee in ppm, or None if not under defense
        """
        defense = self.get_channel_defense_status(scid)
        if defense and defense.get('under_defense'):
            return defense.get('defensive_fee_ppm')
        return None

    def should_rebalance_into_peer(self, peer_id: str) -> bool:
        """
        Check if we should rebalance liquidity toward a peer.

        Uses peer quality assessment to avoid investing in bad peers.

        Args:
            peer_id: Peer public key

        Returns:
            True if okay to rebalance into, False if should avoid
        """
        quality = self.get_single_peer_quality(peer_id)
        if quality:
            return quality.get('quality') != 'avoid'
        return True  # Default to allowing if no data

    def get_exploration_rate_for_channel(
        self,
        scid: str,
        default_rate: float = 0.15
    ) -> float:
        """
        Get recommended exploration rate based on channel maturity.

        New channels need more exploration, mature channels should mostly
        exploit known-good fees.

        Args:
            scid: Channel short_channel_id
            default_rate: Default rate if no age data available

        Returns:
            Exploration rate (0.0-1.0)
        """
        age_info = self.get_single_channel_age(scid)
        if not age_info:
            return default_rate

        maturity = age_info.get('maturity', 'developing')

        if maturity == 'new':
            return 0.30   # 30% exploration for new channels
        elif maturity == 'mature':
            return 0.05   # 5% exploration for mature channels
        else:
            return 0.15   # 15% default for developing channels

    def clear_integration_cache(self) -> int:
        """
        Clear the integration data cache.

        Returns:
            Number of entries cleared
        """
        with self._intel_cache_lock:
            count = len(self._integration_cache)
            self._integration_cache.clear()
        return count
