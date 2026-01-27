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
- query_fleet_liquidity_state(): Query fleet liquidity for coordination
- report_liquidity_state(): Report our liquidity state
- check_rebalance_conflict(): Check for rebalancing conflicts

Phase 3: Splice Coordination
- check_splice_safety(): Advisory check for splice operations
- get_splice_recommendations(): Get splice recommendations

Yield Optimization Phase 2 - Fee Coordination:
- query_coordinated_fee_recommendation(): Get coordinated fee (corridors, pheromones, defense)
- report_routing_outcome(): Report routing for stigmergic learning
- query_defense_status(): Query threat peer defense status
- broadcast_peer_warning(): Broadcast threat warning to fleet
- query_fee_coordination_status(): Get overall coordination status

Yield Optimization Phase 3 - Cost Reduction:
- query_velocity_prediction(): Get channel velocity prediction
- query_critical_velocity_channels(): Get channels needing attention
- query_fleet_rebalance_path(): Check if fleet path is cheaper
- report_rebalance_outcome(): Report rebalance for coordination

Yield Optimization Phase 5 - Strategic Positioning:
- query_flow_recommendations(): Get Physarum-inspired channel recommendations
- report_flow_intensity(): Report flow metrics for optimization
- query_internal_competition(): Detect internal competition

Yield Metrics:
- report_yield_metrics(): Report TLV, costs, revenue to fleet
- query_yield_summary(): Get fleet yield summary

Author: Lightning Goats Team
"""

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
        bridge = HiveFeeIntelligenceBridge(plugin, database)

        # Check if cl-hive is available
        if bridge.is_available():
            intel = bridge.query_fee_intelligence(peer_id)
            if intel:
                # Use competitor data for fee decisions
                their_avg_fee = intel.get("avg_fee_charged", 0)
    """

    def __init__(self, plugin, database):
        """
        Initialize the HiveFeeIntelligenceBridge.

        Args:
            plugin: Reference to the pyln Plugin (or ThreadSafePluginProxy)
            database: Database instance for state persistence (future use)
        """
        self.plugin = plugin
        self.database = database

        # Cache: peer_id -> CachedProfile
        self._cache: Dict[str, CachedProfile] = {}

        # Circuit breaker state
        self._circuit = CircuitBreakerState()

        # Availability cache: None = unknown, True/False = known
        self._hive_available: Optional[bool] = None
        self._availability_check_time: float = 0
        self._availability_ttl: float = 60.0  # Re-check every 60 seconds

    def _log(self, message: str, level: str = "debug") -> None:
        """Log a message if plugin is available."""
        if self.plugin:
            self.plugin.log(f"HIVE_BRIDGE: {message}", level=level)

    # =========================================================================
    # AVAILABILITY CHECK
    # =========================================================================

    def is_available(self) -> bool:
        """
        Check if cl-hive plugin is available AND we are a hive member (cached).

        Returns cached result if within TTL to avoid expensive RPC calls.
        Membership is verified by checking our tier via hive-status RPC.

        Returns:
            True if cl-hive is active AND we are a member/neophyte, False otherwise
        """
        now = time.time()

        # Return cached result if within TTL
        if (self._hive_available is not None and
                (now - self._availability_check_time) < self._availability_ttl):
            return self._hive_available

        # Check plugin list first
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

            # Plugin is loaded - now check if we're actually a hive member
            # This enables hive mode only when we have membership status
            try:
                status = self.plugin.rpc.call("hive-status")
                tier = status.get("membership", {}).get("tier")

                # Only activate hive mode if we're a member or neophyte
                # Note: Admin tier was removed in permissionless join update
                # but we still accept it for backward compatibility with existing DBs
                if tier in ["member", "neophyte", "admin"]:
                    self._hive_available = True
                    self._availability_check_time = now
                    self._log(f"Hive mode active: tier={tier}")
                    return True
                else:
                    # cl-hive is loaded but we're not a member yet
                    self._hive_available = False
                    self._availability_check_time = now
                    self._log(f"cl-hive loaded but not a member (tier={tier})")
                    return False

            except Exception as e:
                # hive-status RPC failed - cl-hive might be starting up
                self._log(f"hive-status check failed: {e}", level="debug")
                self._hive_available = False
                self._availability_check_time = now - (self._availability_ttl - 10)
                return False

        except Exception as e:
            self._log(f"Error checking cl-hive availability: {e}", level="warn")
            # Cache negative result with shorter TTL
            self._hive_available = False
            self._availability_check_time = now - (self._availability_ttl - 5)
            return False

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _is_circuit_open(self) -> bool:
        """
        Check if circuit breaker is open.

        Returns:
            True if circuit is open (should fail fast)
        """
        if not self._circuit.is_open:
            return False

        # Check if reset timeout has passed
        if time.time() - self._circuit.last_failure > CIRCUIT_RESET_TIMEOUT:
            self._circuit.is_open = False
            self._circuit.failures = 0
            self._log("Circuit breaker reset to CLOSED")
            return False

        return True

    def _record_success(self) -> None:
        """Record a successful RPC call."""
        self._circuit.failures = 0
        self._circuit.is_open = False

    def _record_failure(self) -> None:
        """Record a failed RPC call."""
        self._circuit.failures += 1
        self._circuit.last_failure = time.time()
        if self._circuit.failures >= CIRCUIT_FAILURES_THRESHOLD:
            self._circuit.is_open = True
            self._log(
                f"Circuit breaker OPEN after {self._circuit.failures} failures",
                level="warn"
            )

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

    def cleanup_stale_cache(self) -> int:
        """
        Remove stale cache entries older than STALE_CACHE_TTL_SECONDS.

        Call this periodically (e.g., hourly) to prevent memory bloat
        from accumulated stale entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        stale_peers = [
            peer_id for peer_id, cached in self._cache.items()
            if (now - cached.timestamp) > STALE_CACHE_TTL_SECONDS
        ]

        for peer_id in stale_peers:
            del self._cache[peer_id]

        if stale_peers:
            self._log(f"Cleaned up {len(stale_peers)} stale cache entries", level="debug")

        return len(stale_peers)

    def _stale_with_reduced_confidence(
        self,
        data: Dict,
        age_seconds: float
    ) -> Dict:
        """
        Return stale data with reduced confidence.

        Confidence decays by 50% per 12 hours of staleness.

        Args:
            data: Original cached data
            age_seconds: Age of data in seconds

        Returns:
            Copy of data with reduced confidence and staleness markers
        """
        result = dict(data)
        age_hours = age_seconds / 3600

        # Reduce confidence by 50% per 12 hours of staleness
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
        # Get cache entry once to avoid race conditions
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

        # Query cl-hive
        try:
            result = self.plugin.rpc.call("hive-fee-intel-query", {
                "peer_id": peer_id,
                "action": "query"
            })

            # Check for error response
            if result.get("error"):
                if result.get("error") == "no_data":
                    # No data for this peer - not a failure
                    return None
                self._log(f"Query error: {result.get('error')}", level="debug")
                self._record_failure()
                if cached_data and cache_entry:
                    age = time.time() - cache_timestamp
                    return self._stale_with_reduced_confidence(cached_data, age)
                return None

            # Success - cache and return
            self._set_cached(peer_id, result)
            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query fee intelligence: {e}", level="debug")
            self._record_failure()

            if cached_data and cache_entry:
                age = time.time() - cache_timestamp
                return self._stale_with_reduced_confidence(cached_data, age)
            return None

    def query_all_profiles(self) -> List[Dict[str, Any]]:
        """
        Query all known peer profiles from cl-hive.

        Useful for batch operations or competitor analysis.

        Returns:
            List of fee intelligence profiles
        """
        if self._is_circuit_open() or not self.is_available():
            # Return cached profiles
            return [
                cached.data for cached in self._cache.values()
                if (time.time() - cached.timestamp) < STALE_CACHE_TTL_SECONDS
            ]

        try:
            result = self.plugin.rpc.call("hive-fee-intel-query", {
                "action": "list"
            })

            if result.get("error"):
                self._record_failure()
                return []

            profiles = result.get("peers", [])

            # Cache all profiles
            for profile in profiles:
                peer_id = profile.get("peer_id")
                if peer_id:
                    self._set_cached(peer_id, profile)

            self._record_success()
            return profiles

        except Exception as e:
            self._log(f"Failed to query all profiles: {e}", level="debug")
            self._record_failure()
            return []

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

        if self._is_circuit_open():
            return False

        try:
            # Calculate revenue rate
            revenue_sats = (volume_sats * our_fee_ppm) // 1_000_000
            revenue_rate = revenue_sats / period_hours if period_hours > 0 else 0

            result = self.plugin.rpc.call("hive-report-fee-observation", {
                "peer_id": peer_id,
                "our_fee_ppm": our_fee_ppm,
                "their_fee_ppm": their_fee_ppm,
                "volume_sats": volume_sats,
                "forward_count": forward_count,
                "period_hours": period_hours,
                "revenue_rate": revenue_rate
            })

            if result.get("error"):
                self._log(
                    f"Observation report error: {result.get('error')}",
                    level="debug"
                )
                return False

            return True

        except Exception as e:
            self._log(f"Failed to report observation: {e}", level="debug")
            return False

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

        try:
            params = {"action": "query"}
            if member_id:
                params["member_id"] = member_id

            result = self.plugin.rpc.call("hive-member-health", params)

            if result.get("error"):
                self._log(f"Health query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query member health: {e}", level="debug")
            self._record_failure()
            return None

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

        try:
            result = self.plugin.rpc.call("hive-member-health", {
                "member_id": "all",
                "action": "aggregate"
            })

            if result.get("error"):
                self._log(f"Fleet health query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query fleet health: {e}", level="debug")
            self._record_failure()
            return None

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

        try:
            params = {
                "profitable_channels": profitable_channels,
                "underwater_channels": underwater_channels,
                "stagnant_channels": stagnant_channels,
                "revenue_trend": revenue_trend,
                "liquidity_score": liquidity_score
            }
            if total_channels is not None:
                params["total_channels"] = total_channels

            result = self.plugin.rpc.call("hive-report-health", params)

            if result.get("error"):
                self._log(f"Health report error: {result.get('error')}", level="debug")
                return False

            self._log(
                f"Health reported: score={result.get('health_score')}, "
                f"tier={result.get('health_tier')}, "
                f"multiplier={result.get('budget_multiplier')}"
            )
            return True

        except Exception as e:
            self._log(f"Failed to report health: {e}", level="debug")
            return False

    # =========================================================================
    # PHASE 2: LIQUIDITY INTELLIGENCE SHARING
    # =========================================================================
    # These methods share INFORMATION about liquidity state.
    # No fund transfers between nodes - purely informational coordination.

    def query_fleet_liquidity_state(self) -> Optional[Dict[str, Any]]:
        """
        Query fleet liquidity state for coordinated decision-making.

        Information only - helps us make better decisions about
        our own rebalancing and fee adjustments.

        Returns:
            Fleet liquidity state or None if unavailable:
            {
                "active": True,
                "fleet_summary": {
                    "members_with_depleted_channels": 2,
                    "members_with_saturated_channels": 3,
                    "common_bottleneck_peers": ["02abc...", "03xyz..."]
                },
                "our_state": {...}
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            result = self.plugin.rpc.call("hive-liquidity-state", {
                "action": "status"
            })

            if result.get("error"):
                self._log(f"Liquidity state query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query liquidity state: {e}", level="debug")
            self._record_failure()
            return None

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

        try:
            result = self.plugin.rpc.call("hive-liquidity-state", {
                "action": "needs"
            })

            if result.get("error"):
                self._log(f"Fleet needs query error: {result.get('error')}", level="debug")
                return []

            self._record_success()
            return result.get("fleet_needs", [])

        except Exception as e:
            self._log(f"Failed to query fleet needs: {e}", level="debug")
            self._record_failure()
            return []

    def report_liquidity_state(
        self,
        depleted_channels: List[Dict[str, Any]],
        saturated_channels: List[Dict[str, Any]],
        rebalancing_active: bool = False,
        rebalancing_peers: List[str] = None
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

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        try:
            result = self.plugin.rpc.call("hive-report-liquidity-state", {
                "depleted_channels": depleted_channels,
                "saturated_channels": saturated_channels,
                "rebalancing_active": rebalancing_active,
                "rebalancing_peers": rebalancing_peers or []
            })

            if result.get("error"):
                self._log(f"Liquidity state report error: {result.get('error')}", level="debug")
                return False

            self._log(
                f"Liquidity state reported: depleted={result.get('depleted_count')}, "
                f"saturated={result.get('saturated_count')}"
            )
            return True

        except Exception as e:
            self._log(f"Failed to report liquidity state: {e}", level="debug")
            return False

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

        try:
            result = self.plugin.rpc.call("hive-check-rebalance-conflict", {
                "peer_id": peer_id
            })

            if result.get("error"):
                self._log(f"Conflict check error: {result.get('error')}", level="debug")
                return {"conflict": False, "reason": "check_failed"}

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to check rebalance conflict: {e}", level="debug")
            self._record_failure()
            return {"conflict": False, "reason": "exception"}

    # =========================================================================
    # PHASE 3: SPLICE COORDINATION
    # =========================================================================
    # Safety checks for splice operations to maintain fleet connectivity.
    # ADVISORY ONLY - each node manages its own funds.

    def check_splice_safety(
        self,
        peer_id: str,
        splice_type: str,
        amount_sats: int,
        channel_id: str = None
    ) -> Dict[str, Any]:
        """
        Check if a splice operation is safe for fleet connectivity.

        SAFETY CHECK ONLY - no fund movement.
        We manage our own splice, just checking if timing is safe.

        Args:
            peer_id: External peer we're splicing from/to
            splice_type: "splice_in" or "splice_out"
            amount_sats: Amount to splice in/out
            channel_id: Optional specific channel ID

        Returns:
            Safety assessment:
            {
                "safe": bool,
                "safety_level": "safe" | "coordinate" | "blocked",
                "reason": str,
                "can_proceed": bool,
                "recommendation": str (if not safe),
                "fleet_share": float,
                "new_share": float
            }
        """
        if not self.is_available():
            # Default to safe if hive unavailable (fail open)
            return {
                "safe": True,
                "safety_level": "safe",
                "reason": "Hive unavailable, local decision",
                "can_proceed": True
            }

        if self._is_circuit_open():
            return {
                "safe": True,
                "safety_level": "safe",
                "reason": "Circuit breaker open, local decision",
                "can_proceed": True
            }

        try:
            params = {
                "peer_id": peer_id,
                "splice_type": splice_type,
                "amount_sats": amount_sats
            }
            if channel_id:
                params["channel_id"] = channel_id

            result = self.plugin.rpc.call("hive-splice-check", params)

            if result.get("error"):
                self._log(f"Splice check error: {result.get('error')}", level="debug")
                # Fail open - allow local decision
                return {
                    "safe": True,
                    "safety_level": "safe",
                    "reason": f"Check error: {result.get('error')}",
                    "can_proceed": True
                }

            safety = result.get("safety", "safe")
            self._record_success()

            return {
                "safe": safety == "safe",
                "safety_level": safety,
                "reason": result.get("reason", ""),
                "can_proceed": safety != "blocked",
                "recommendation": result.get("recommendation"),
                "fleet_capacity": result.get("fleet_capacity"),
                "new_fleet_capacity": result.get("new_fleet_capacity"),
                "fleet_share": result.get("fleet_share"),
                "new_share": result.get("new_share")
            }

        except Exception as e:
            self._log(f"Splice safety check failed: {e}", level="debug")
            self._record_failure()
            # Fail open - allow local decision
            return {
                "safe": True,
                "safety_level": "safe",
                "reason": f"Check failed: {e}",
                "can_proceed": True
            }

    def get_splice_recommendations(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get splice recommendations for a specific peer.

        Returns info about fleet connectivity and safe splice amounts.
        INFORMATION ONLY - helps make informed splice decisions.

        Args:
            peer_id: External peer to analyze

        Returns:
            Recommendations or None:
            {
                "peer_id": str,
                "fleet_capacity": int,
                "our_capacity": int,
                "other_member_capacity": int,
                "safe_splice_out_amount": int,
                "has_fleet_coverage": bool,
                "recommendations": [str]
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            result = self.plugin.rpc.call("hive-splice-recommendations", {
                "peer_id": peer_id
            })

            if result.get("error"):
                self._log(
                    f"Splice recommendations error: {result.get('error')}",
                    level="debug"
                )
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to get splice recommendations: {e}", level="debug")
            self._record_failure()
            return None

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

        try:
            params = {
                "channel_id": channel_id,
                "current_fee": current_fee,
                "local_balance_pct": local_balance_pct
            }
            if source:
                params["source"] = source
            if destination:
                params["destination"] = destination

            result = self.plugin.rpc.call("hive-coord-fee-recommendation", params)

            if result.get("error"):
                self._log(
                    f"Coordinated fee recommendation error: {result.get('error')}",
                    level="debug"
                )
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query coordinated fee recommendation: {e}", level="debug")
            self._record_failure()
            return None

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

        if self._is_circuit_open():
            return False

        try:
            params = {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "fee_ppm": fee_ppm,
                "success": success,
                "amount_sats": amount_sats
            }
            if source:
                params["source"] = source
            if destination:
                params["destination"] = destination

            # Use deposit-marker RPC if source/destination provided
            if source and destination:
                result = self.plugin.rpc.call("hive-deposit-marker", params)
            else:
                # Fall back to generic pheromone update
                result = self.plugin.rpc.call("hive-pheromone-levels", {
                    "channel_id": channel_id,
                    "action": "update",
                    "fee_ppm": fee_ppm,
                    "success": success,
                    "amount_sats": amount_sats
                })

            if result.get("error"):
                self._log(f"Routing outcome report error: {result.get('error')}", level="debug")
                return False

            return True

        except Exception as e:
            self._log(f"Failed to report routing outcome: {e}", level="debug")
            return False

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

        try:
            params = {}
            if peer_id:
                params["peer_id"] = peer_id

            result = self.plugin.rpc.call("hive-defense-status", params)

            if result.get("error"):
                self._log(f"Defense status query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query defense status: {e}", level="debug")
            self._record_failure()
            return None

    def broadcast_peer_warning(
        self,
        peer_id: str,
        threat_type: str,
        severity: float,
        evidence: Dict[str, Any] = None
    ) -> bool:
        """
        Broadcast a threat warning about a peer to the fleet.

        Like mycelium warning signals - when one tree is attacked,
        neighbors activate defenses. The fleet collectively raises
        fees to threatening peers.

        Args:
            peer_id: Peer to warn about
            threat_type: "drain" (outflow imbalance) or "unreliable" (high failures)
            severity: Severity score 0.0-1.0 (higher = more severe)
            evidence: Optional evidence dict (drain_rate, failure_rate, etc.)

        Returns:
            True if warning broadcasted successfully
        """
        if not self.is_available():
            return False

        if self._is_circuit_open():
            return False

        try:
            params = {
                "peer_id": peer_id,
                "threat_type": threat_type,
                "severity": severity
            }
            if evidence:
                params["evidence"] = evidence

            result = self.plugin.rpc.call("hive-broadcast-warning", params)

            if result.get("error"):
                self._log(f"Warning broadcast error: {result.get('error')}", level="debug")
                return False

            self._log(
                f"Warning broadcasted: peer={peer_id[:12]}... type={threat_type} "
                f"severity={severity:.2f}",
                level="info"
            )
            return True

        except Exception as e:
            self._log(f"Failed to broadcast warning: {e}", level="debug")
            return False

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

        if self._is_circuit_open():
            return False

        # Rate limit: don't flood hive with observations
        # Only broadcast discoveries, not every observation
        if discovery_type == "observation" and confidence < 0.7:
            return False

        try:
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

            result = self.plugin.rpc.call("hive-broadcast-fee-observation", params)

            if result.get("error"):
                self._log(
                    f"Fee observation broadcast error: {result.get('error')}",
                    level="debug"
                )
                return False

            self._log(
                f"Fee observation broadcasted: peer={peer_id[:12]}... "
                f"fee={fee_ppm}ppm revenue={revenue_rate:.1f}sats/hr "
                f"type={discovery_type} conf={confidence:.2f}",
                level="info"
            )
            return True

        except Exception as e:
            # Don't record failure for non-existent RPC (graceful degradation)
            if "Unknown command" not in str(e):
                self._log(f"Failed to broadcast fee observation: {e}", level="debug")
            return False

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

        try:
            result = self.plugin.rpc.call("hive-fee-coordination-status", {})

            if result.get("error"):
                self._log(f"Fee coordination status error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query fee coordination status: {e}", level="debug")
            self._record_failure()
            return None

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
        if not self.is_available() or self._is_circuit_open():
            return False

        # Only broadcast high-confidence elasticity
        if confidence < 0.5:
            return False

        try:
            params = {
                "peer_id": peer_id,
                "elasticity": elasticity,
                "confidence": confidence,
                "sample_count": sample_count,
                "timestamp": int(time.time())
            }

            result = self.plugin.rpc.call("hive-broadcast-elasticity", params)

            if result.get("error"):
                self._log(f"Elasticity broadcast error: {result.get('error')}", level="debug")
                return False

            self._log(
                f"Elasticity broadcasted: peer={peer_id[:12]}... "
                f"elasticity={elasticity:.2f} conf={confidence:.2f}",
                level="debug"
            )
            return True

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to broadcast elasticity: {e}", level="debug")
            return False

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

        try:
            result = self.plugin.rpc.call("hive-query-elasticity", {"peer_id": peer_id})

            if result.get("error"):
                return None

            self._record_success()
            return result

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to query fleet elasticity: {e}", level="debug")
            return None

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
        if not self.is_available() or self._is_circuit_open():
            return False

        # Only broadcast meaningful observations
        if forward_count < 1 or revenue_rate < 1.0:
            return False

        try:
            params = {
                "peer_id": peer_id,
                "fee_ppm": fee_ppm,
                "revenue_rate": revenue_rate,
                "forward_count": forward_count,
                "timestamp": int(time.time())
            }

            result = self.plugin.rpc.call("hive-broadcast-curve-observation", params)

            if result.get("error"):
                return False

            return True

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to broadcast curve observation: {e}", level="debug")
            return False

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

        try:
            result = self.plugin.rpc.call("hive-query-aggregated-curve", {"peer_id": peer_id})

            if result.get("error"):
                return None

            self._record_success()
            return result

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to query aggregated curve: {e}", level="debug")
            return None

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
        if not self.is_available() or self._is_circuit_open():
            return False

        try:
            params = {
                "peer_id": peer_id,
                "change_type": change_type,
                "old_regime": old_regime,
                "new_regime": new_regime,
                "timestamp": int(time.time())
            }
            if evidence:
                params["evidence"] = evidence

            result = self.plugin.rpc.call("hive-broadcast-regime-change", params)

            if result.get("error"):
                return False

            self._log(
                f"Regime change broadcasted: peer={peer_id[:12]}... "
                f"type={change_type} {old_regime}->{new_regime}",
                level="info"
            )
            return True

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to broadcast regime change: {e}", level="debug")
            return False

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

        try:
            params = {}
            if peer_id:
                params["peer_id"] = peer_id

            result = self.plugin.rpc.call("hive-query-regime-status", params)

            if result.get("error"):
                return None

            self._record_success()
            return result

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to query fleet regime status: {e}", level="debug")
            return None

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
        if not self.is_available() or self._is_circuit_open():
            return False

        # Only share if we have meaningful data
        if observation_count < 5:
            return False

        try:
            params = {
                "peer_id": peer_id,
                "posterior_mean": posterior_mean,
                "posterior_std": posterior_std,
                "observation_count": observation_count,
                "corridor_role": corridor_role,
                "timestamp": int(time.time())
            }

            result = self.plugin.rpc.call("hive-share-posterior", params)

            if result.get("error"):
                return False

            return True

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to share posterior: {e}", level="debug")
            return False

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

        try:
            result = self.plugin.rpc.call("hive-query-posteriors", {"peer_id": peer_id})

            if result.get("error"):
                return None

            self._record_success()
            return result

        except Exception as e:
            if "Unknown command" not in str(e):
                self._log(f"Failed to query fleet posteriors: {e}", level="debug")
            return None

    # =========================================================================
    # YIELD OPTIMIZATION PHASE 3: COST REDUCTION (PREDICTIVE REBALANCING)
    # =========================================================================
    # These methods support predictive rebalancing and fleet path optimization
    # to reduce rebalancing costs by up to 50%.

    def query_velocity_prediction(
        self,
        channel_id: str,
        hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Query velocity prediction for a channel.

        Used for predictive rebalancing - rebalance BEFORE depletion
        when urgency is low and fees are cheaper.

        Args:
            channel_id: Channel to predict
            hours: Prediction window in hours (default: 24)

        Returns:
            Velocity prediction dict or None:
            {
                "channel_id": "123x1x0",
                "current_local_pct": 0.35,
                "velocity_pct_per_hour": -0.02,
                "predicted_local_pct": 0.11,
                "hours_to_depletion": 17.5,
                "hours_to_saturation": null,
                "depletion_risk": 0.75,
                "saturation_risk": 0.0,
                "recommended_action": "preemptive_rebalance",
                "urgency": "low"
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            result = self.plugin.rpc.call("hive-velocity-prediction", {
                "channel_id": channel_id,
                "hours": hours
            })

            if result.get("error"):
                self._log(f"Velocity prediction error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query velocity prediction: {e}", level="debug")
            self._record_failure()
            return None

    def query_critical_velocity_channels(
        self,
        hours_threshold: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get channels with critical velocity (depleting/filling rapidly).

        These channels need urgent attention - either rebalancing or
        fee changes to prevent depletion/saturation.

        Args:
            hours_threshold: Alert threshold in hours

        Returns:
            List of channels with critical velocity
        """
        if self._is_circuit_open() or not self.is_available():
            return []

        try:
            result = self.plugin.rpc.call("hive-critical-velocity", {
                "hours_threshold": hours_threshold
            })

            if result.get("error"):
                self._log(f"Critical velocity query error: {result.get('error')}", level="debug")
                return []

            self._record_success()
            return result.get("channels", [])

        except Exception as e:
            self._log(f"Failed to query critical velocity: {e}", level="debug")
            self._record_failure()
            return []

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

        try:
            result = self.plugin.rpc.call("hive-fleet-rebalance-path", {
                "from_channel": from_channel,
                "to_channel": to_channel,
                "amount_sats": amount_sats
            })

            if result.get("error"):
                # This might not be implemented yet - that's OK
                if "unknown" in str(result.get("error")).lower():
                    return None
                self._log(f"Fleet path query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            # Method might not exist yet - fail gracefully
            self._log(f"Failed to query fleet rebalance path: {e}", level="debug")
            return None

    def report_kalman_velocity(
        self,
        channel_id: str,
        peer_id: str,
        velocity_pct_per_hour: float,
        uncertainty: float,
        flow_ratio: float,
        confidence: float,
        is_regime_change: bool = False
    ) -> bool:
        """
        Report Kalman-estimated velocity to cl-hive for coordinated predictions.

        Shares our Kalman filter's velocity estimate with the hive so that
        anticipatory_liquidity can use superior state estimation instead of
        simple net flow calculations.

        Args:
            channel_id: Channel SCID
            peer_id: Peer pubkey
            velocity_pct_per_hour: Kalman velocity estimate (% balance change per hour)
            uncertainty: Standard deviation of velocity estimate
            flow_ratio: Current Kalman-estimated flow ratio (-1 to 1)
            confidence: Observation confidence (0.0-1.0)
            is_regime_change: True if regime change detected

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        if self._is_circuit_open():
            return False

        # Validate parameters to prevent invalid data propagation
        if not (0.0 <= confidence <= 1.0):
            self._log(f"Invalid confidence value {confidence}, clamping to [0,1]", level="debug")
            confidence = max(0.0, min(1.0, confidence))

        if not (-1.0 <= flow_ratio <= 1.0):
            self._log(f"Invalid flow_ratio value {flow_ratio}, clamping to [-1,1]", level="debug")
            flow_ratio = max(-1.0, min(1.0, flow_ratio))

        if uncertainty < 0:
            self._log(f"Invalid uncertainty value {uncertainty}, using abs", level="debug")
            uncertainty = abs(uncertainty)

        try:
            result = self.plugin.rpc.call("hive-report-kalman-velocity", {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "velocity_pct_per_hour": velocity_pct_per_hour,
                "uncertainty": uncertainty,
                "flow_ratio": flow_ratio,
                "confidence": confidence,
                "is_regime_change": is_regime_change
            })

            if result.get("error"):
                # Method might not be implemented yet - that's OK
                if "unknown" in str(result.get("error")).lower():
                    return True  # Silently succeed if not implemented
                self._log(f"Kalman velocity report error: {result.get('error')}", level="debug")
                return False

            self._record_success()
            return True

        except Exception as e:
            # Method might not exist yet - fail gracefully
            self._log(f"Failed to report Kalman velocity: {e}", level="debug")
            return True  # Don't block on this

    def query_kalman_velocity(
        self,
        channel_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Query Kalman velocity data from cl-hive for a channel.

        This returns the velocity estimate that was reported by any fleet
        member who has a channel to the same peer.

        Args:
            channel_id: Channel SCID

        Returns:
            Kalman velocity data dict or None:
            {
                "velocity_pct_per_hour": -0.02,
                "uncertainty": 0.005,
                "flow_ratio": -0.3,
                "confidence": 0.85,
                "reporters": 2,
                "is_consensus": True,
                "last_update": 1706000000
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            result = self.plugin.rpc.call("hive-query-kalman-velocity", {
                "channel_id": channel_id
            })

            if result.get("error"):
                if "unknown" in str(result.get("error")).lower():
                    return None
                self._log(f"Kalman velocity query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query Kalman velocity: {e}", level="debug")
            return None

    def report_rebalance_outcome(
        self,
        from_channel: str,
        to_channel: str,
        amount_sats: int,
        cost_sats: int,
        success: bool,
        via_fleet: bool = False
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

        if self._is_circuit_open():
            return False

        try:
            result = self.plugin.rpc.call("hive-report-rebalance-outcome", {
                "from_channel": from_channel,
                "to_channel": to_channel,
                "amount_sats": amount_sats,
                "cost_sats": cost_sats,
                "success": success,
                "via_fleet": via_fleet
            })

            if result.get("error"):
                # This might not be implemented yet - that's OK
                if "unknown" in str(result.get("error")).lower():
                    return True  # Silently succeed if not implemented
                self._log(f"Rebalance outcome report error: {result.get('error')}", level="debug")
                return False

            return True

        except Exception as e:
            # Method might not exist yet - fail gracefully
            self._log(f"Failed to report rebalance outcome: {e}", level="debug")
            return True  # Don't block on this

    # =========================================================================
    # YIELD OPTIMIZATION PHASE 5: STRATEGIC POSITIONING (PHYSARUM)
    # =========================================================================
    # These methods support Physarum-inspired channel lifecycle management:
    # - High flow channels → strengthen (splice in)
    # - Low flow channels → atrophy (close)

    def query_flow_recommendations(
        self,
        channel_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get Physarum-inspired flow recommendations for channels.

        Like slime mold tubes that strengthen with flow and atrophy without,
        channels should grow or shrink based on their flow intensity.

        Args:
            channel_id: Specific channel (optional, None for all)

        Returns:
            Flow recommendations dict or None:
            {
                "recommendations": [
                    {
                        "channel_id": "123x1x0",
                        "peer_id": "02abc...",
                        "flow_intensity": 0.035,
                        "action": "strengthen",
                        "method": "splice_in",
                        "recommended_amount_sats": 2000000,
                        "reason": "Flow intensity 3.5% exceeds 2% threshold",
                        "expected_yield_improvement": 0.015
                    },
                    {
                        "channel_id": "456x2x1",
                        "peer_id": "03xyz...",
                        "flow_intensity": 0.0005,
                        "action": "atrophy",
                        "method": "cooperative_close",
                        "reason": "Mature channel with flow 0.05% below 0.1% threshold",
                        "capital_to_redeploy": 5000000
                    }
                ],
                "summary": {
                    "strengthen_count": 3,
                    "maintain_count": 18,
                    "stimulate_count": 2,
                    "atrophy_count": 2
                }
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            params = {}
            if channel_id:
                params["channel_id"] = channel_id

            result = self.plugin.rpc.call("hive-flow-recommendations", params)

            if result.get("error"):
                # This might not be implemented yet
                if "unknown" in str(result.get("error")).lower():
                    return None
                self._log(f"Flow recommendations error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query flow recommendations: {e}", level="debug")
            return None

    def report_flow_intensity(
        self,
        channel_id: str,
        peer_id: str,
        capacity_sats: int,
        volume_7d_sats: int,
        revenue_7d_sats: int,
        forward_count_7d: int
    ) -> bool:
        """
        Report flow intensity metrics for a channel.

        Contributes to fleet-wide flow analysis for Physarum optimization.

        Args:
            channel_id: Channel SCID
            peer_id: Peer pubkey
            capacity_sats: Channel capacity
            volume_7d_sats: 7-day volume
            revenue_7d_sats: 7-day revenue
            forward_count_7d: 7-day forward count

        Returns:
            True if reported successfully
        """
        if not self.is_available():
            return False

        if self._is_circuit_open():
            return False

        try:
            result = self.plugin.rpc.call("hive-report-flow-intensity", {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "capacity_sats": capacity_sats,
                "volume_7d_sats": volume_7d_sats,
                "revenue_7d_sats": revenue_7d_sats,
                "forward_count_7d": forward_count_7d
            })

            if result.get("error"):
                # This might not be implemented yet
                if "unknown" in str(result.get("error")).lower():
                    return True  # Silently succeed
                self._log(f"Flow intensity report error: {result.get('error')}", level="debug")
                return False

            return True

        except Exception as e:
            self._log(f"Failed to report flow intensity: {e}", level="debug")
            return True  # Don't block on this

    def query_internal_competition(self) -> Optional[Dict[str, Any]]:
        """
        Query internal competition detection from cl-hive.

        Identifies routes where multiple fleet members compete,
        enabling coordination to avoid undercutting.

        Returns:
            Internal competition analysis or None:
            {
                "competing_routes": [
                    {
                        "source": "02abc...",
                        "destination": "03xyz...",
                        "competing_members": ["node1", "node2", "node3"],
                        "member_count": 3,
                        "recommendation": "coordinate_fees",
                        "estimated_revenue_loss_pct": 15
                    }
                ],
                "competition_index": 0.25,
                "recommendation": "High internal competition - enable fee coordination"
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            result = self.plugin.rpc.call("hive-internal-competition", {})

            if result.get("error"):
                self._log(f"Internal competition query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query internal competition: {e}", level="debug")
            self._record_failure()
            return None

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

        try:
            result = self.plugin.rpc.call("hive-pheromone-levels", {
                "channel_id": channel_id
            })

            if result.get("error"):
                return None

            # Extract relevant data for this channel
            levels = result.get("pheromone_levels", [])
            for level_data in levels:
                if level_data.get("channel_id") == channel_id:
                    self._record_success()
                    return {
                        "channel_id": channel_id,
                        "level": level_data.get("level", 0),
                        "above_threshold": level_data.get("above_threshold", False)
                    }

            # Channel exists but no pheromone data yet
            self._record_success()
            return {"channel_id": channel_id, "level": 0, "above_threshold": False}

        except Exception as e:
            self._log(f"Failed to query pheromone level: {e}", level="debug")
            self._record_failure()
            return None

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

        try:
            result = self.plugin.rpc.call("hive-internal-competition", {})

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

        except Exception as e:
            self._log(f"Failed to check internal competition: {e}", level="debug")
            self._record_failure()
            return None

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

        try:
            result = self.plugin.rpc.call("hive-report-yield-metrics", {
                "tlv_sats": tlv_sats,
                "operating_costs_sats": operating_costs_sats,
                "routing_revenue_sats": routing_revenue_sats,
                "period_days": period_days
            })

            if result.get("error"):
                # This might not be implemented yet
                if "unknown" in str(result.get("error")).lower():
                    return True
                self._log(f"Yield metrics report error: {result.get('error')}", level="debug")
                return False

            return True

        except Exception as e:
            self._log(f"Failed to report yield metrics: {e}", level="debug")
            return True  # Don't block

    def query_yield_summary(self) -> Optional[Dict[str, Any]]:
        """
        Query yield summary from cl-hive.

        Returns:
            Yield summary or None:
            {
                "fleet_tlv_sats": 1650000000,
                "fleet_revenue_30d_sats": 150000,
                "fleet_costs_30d_sats": 50000,
                "fleet_net_yield_30d_sats": 100000,
                "annualized_roc_pct": 7.3,
                "our_contribution_pct": 18.5
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            result = self.plugin.rpc.call("hive-yield-summary", {})

            if result.get("error"):
                self._log(f"Yield summary query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query yield summary: {e}", level="debug")
            self._record_failure()
            return None

    # =========================================================================
    # ANTICIPATORY LIQUIDITY (Phase 7.1)
    # =========================================================================
    # These methods enable predictive rebalancing by leveraging temporal
    # patterns detected by cl-hive. Rebalance BEFORE depletion when
    # urgency is low and fees are cheaper.

    def query_anticipatory_prediction(
        self,
        channel_id: str,
        hours_ahead: int = 12
    ) -> Optional[Dict[str, Any]]:
        """
        Query liquidity prediction for a channel from cl-hive.

        Uses temporal patterns to predict future balance state,
        enabling preemptive rebalancing before depletion/saturation.

        Args:
            channel_id: Channel SCID to predict
            hours_ahead: Prediction horizon in hours (default: 12)

        Returns:
            Prediction dict or None:
            {
                "channel_id": "123x1x0",
                "current_local_pct": 0.35,
                "predicted_local_pct": 0.11,
                "velocity_pct_per_hour": -0.02,
                "depletion_risk": 0.75,
                "saturation_risk": 0.0,
                "hours_to_critical": 17.5,
                "recommended_action": "preemptive_rebalance",
                "urgency": "preemptive",
                "confidence": 0.8,
                "pattern_match": "weekday_afternoon_drain"
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            result = self.plugin.rpc.call("hive-predict-liquidity", {
                "channel_id": channel_id,
                "hours_ahead": hours_ahead
            })

            if result.get("error"):
                # No data is not a failure
                if result.get("error") == "no_data":
                    return None
                self._log(f"Prediction query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query anticipatory prediction: {e}", level="debug")
            self._record_failure()
            return None

    def query_all_anticipatory_predictions(
        self,
        hours_ahead: int = 12,
        min_risk: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Query predictions for all at-risk channels from cl-hive.

        Returns channels with significant depletion or saturation risk,
        enabling proactive rebalancing prioritization.

        Args:
            hours_ahead: Prediction horizon in hours (default: 12)
            min_risk: Minimum risk threshold to include (default: 0.3)

        Returns:
            List of prediction dicts for at-risk channels
        """
        if self._is_circuit_open() or not self.is_available():
            return []

        try:
            result = self.plugin.rpc.call("hive-anticipatory-predictions", {
                "hours_ahead": hours_ahead,
                "min_risk": min_risk
            })

            if result.get("error"):
                self._log(f"All predictions query error: {result.get('error')}", level="debug")
                return []

            self._record_success()
            return result.get("predictions", [])

        except Exception as e:
            self._log(f"Failed to query all predictions: {e}", level="debug")
            self._record_failure()
            return []

    def query_temporal_patterns(
        self,
        channel_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query detected temporal patterns from cl-hive.

        Patterns indicate when channels typically experience high flow
        in either direction, enabling time-based fee and rebalancing optimization.

        Args:
            channel_id: Specific channel (None for summary of all)

        Returns:
            Pattern data dict or None:
            {
                "channel_id": "123x1x0",
                "pattern_count": 3,
                "patterns": [
                    {
                        "hour_of_day": 14,
                        "day_of_week": null,
                        "direction": "outbound",
                        "intensity": 1.8,
                        "confidence": 0.75,
                        "avg_flow_sats": 500000
                    }
                ]
            }
        """
        if self._is_circuit_open() or not self.is_available():
            return None

        try:
            params = {}
            if channel_id:
                params["channel_id"] = channel_id

            result = self.plugin.rpc.call("hive-detect-patterns", params)

            if result.get("error"):
                self._log(f"Patterns query error: {result.get('error')}", level="debug")
                return None

            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query temporal patterns: {e}", level="debug")
            self._record_failure()
            return None

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

        if self._is_circuit_open():
            return False

        try:
            params = {
                "channel_id": channel_id,
                "inbound_sats": inbound_sats,
                "outbound_sats": outbound_sats
            }
            if timestamp:
                params["timestamp"] = timestamp

            result = self.plugin.rpc.call("hive-record-flow", params)

            if result.get("error"):
                self._log(f"Flow report error: {result.get('error')}", level="debug")
                return False

            return True

        except Exception as e:
            self._log(f"Failed to report flow observation: {e}", level="debug")
            return False

    def should_preemptive_rebalance(
        self,
        channel_id: str,
        current_local_pct: float
    ) -> Dict[str, Any]:
        """
        Check if a channel should be rebalanced preemptively.

        Combines current state with prediction to recommend
        whether to rebalance now (preemptively) or wait.

        Args:
            channel_id: Channel SCID
            current_local_pct: Current local balance percentage (0.0-1.0)

        Returns:
            Recommendation dict:
            {
                "should_rebalance": True,
                "reason": "Predicted depletion in 8 hours",
                "urgency": "preemptive",
                "recommended_amount_pct": 0.25,
                "cost_advantage": "Rebalancing now vs urgently saves ~20-40% in fees"
            }
        """
        result = {
            "should_rebalance": False,
            "reason": "No prediction available",
            "urgency": "none",
            "recommended_amount_pct": 0.0,
            "cost_advantage": None
        }

        prediction = self.query_anticipatory_prediction(channel_id, hours_ahead=24)
        if not prediction:
            return result

        # Check urgency levels
        urgency = prediction.get("urgency", "none")
        depletion_risk = prediction.get("depletion_risk", 0)
        saturation_risk = prediction.get("saturation_risk", 0)
        hours_to_critical = prediction.get("hours_to_critical")

        if urgency in ["critical", "urgent"]:
            # Need to rebalance now regardless
            result["should_rebalance"] = True
            result["reason"] = f"Critical: {urgency} urgency, {hours_to_critical:.0f}h to critical"
            result["urgency"] = urgency
            result["recommended_amount_pct"] = 0.3 if depletion_risk > saturation_risk else -0.3

        elif urgency == "preemptive":
            # Ideal window for preemptive rebalancing
            result["should_rebalance"] = True
            result["reason"] = f"Preemptive window: {hours_to_critical:.0f}h to critical"
            result["urgency"] = "preemptive"
            result["recommended_amount_pct"] = 0.25 if depletion_risk > saturation_risk else -0.25
            result["cost_advantage"] = "Rebalancing now vs urgently saves ~20-40% in fees"

        elif urgency == "low":
            # Could rebalance, but not urgent
            if depletion_risk > 0.3 or saturation_risk > 0.3:
                result["should_rebalance"] = True
                result["reason"] = f"Optional: moderate risk ({max(depletion_risk, saturation_risk):.0%})"
                result["urgency"] = "low"
                result["recommended_amount_pct"] = 0.15 if depletion_risk > saturation_risk else -0.15
            else:
                result["reason"] = "Channel stable, no rebalancing needed"

        return result

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
        if not self._hive_available or self._circuit.is_open:
            return None

        try:
            result = self.plugin.rpc.call("hive-time-fee-adjustment", {
                "channel_id": channel_id,
                "base_fee": base_fee
            })

            if result.get("error"):
                self._log(f"Time fee query error: {result.get('error')}", level="debug")
                return None

            return result

        except Exception as e:
            self._log(f"Failed to query time fee adjustment: {e}", level="debug")
            self._circuit.record_failure()
            return None

    def query_time_fee_status(self) -> Optional[Dict[str, Any]]:
        """
        Query time-based fee system status from cl-hive.

        Returns overview of active time-based adjustments and configuration.

        Returns:
            Status dict or None:
            {
                "enabled": True,
                "current_hour": 14,
                "current_day": 2,
                "current_day_name": "Wed",
                "active_adjustments": 5,
                "adjustments": [...],
                "config": {
                    "max_increase_pct": 25,
                    "max_decrease_pct": 15,
                    "peak_threshold": 0.7,
                    "low_threshold": 0.3,
                    "min_confidence": 0.5
                }
            }
        """
        if not self._hive_available or self._circuit.is_open:
            return None

        try:
            result = self.plugin.rpc.call("hive-time-fee-status", {})

            if result.get("error"):
                self._log(f"Time fee status error: {result.get('error')}", level="debug")
                return None

            return result

        except Exception as e:
            self._log(f"Failed to query time fee status: {e}", level="debug")
            self._circuit.record_failure()
            return None

    def query_channel_peak_hours(
        self,
        channel_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Query detected peak hours for a channel from cl-hive.

        Returns hours with above-average routing volume.

        Args:
            channel_id: Channel SCID

        Returns:
            List of peak hour dicts or None:
            [
                {
                    "hour": 14,
                    "day": -1,
                    "day_name": "Any",
                    "intensity": 0.85,
                    "direction": "outbound",
                    "confidence": 0.8,
                    "samples": 45
                }
            ]
        """
        if not self._hive_available or self._circuit.is_open:
            return None

        try:
            result = self.plugin.rpc.call("hive-time-peak-hours", {
                "channel_id": channel_id
            })

            if result.get("error"):
                self._log(f"Peak hours query error: {result.get('error')}", level="debug")
                return None

            return result.get("peak_hours", [])

        except Exception as e:
            self._log(f"Failed to query peak hours: {e}", level="debug")
            self._circuit.record_failure()
            return None

    def should_use_time_adjusted_fee(
        self,
        channel_id: str,
        current_fee: int
    ) -> Dict[str, Any]:
        """
        Check if a time-adjusted fee should be used for a channel.

        Combines current fee with time-based adjustment to recommend
        whether to change the fee.

        Args:
            channel_id: Channel SCID
            current_fee: Current fee in ppm

        Returns:
            Recommendation dict:
            {
                "should_adjust": True,
                "recommended_fee": 275,
                "adjustment_pct": 10.0,
                "adjustment_type": "peak_increase",
                "reason": "Peak hour detected - increase fee to capture premium",
                "reverts_in_hours": 2
            }
        """
        result = {
            "should_adjust": False,
            "recommended_fee": current_fee,
            "adjustment_pct": 0.0,
            "adjustment_type": "none",
            "reason": "No time adjustment available",
            "reverts_in_hours": None
        }

        adjustment = self.query_time_fee_adjustment(channel_id, current_fee)
        if not adjustment:
            return result

        adj_type = adjustment.get("adjustment_type", "none")
        if adj_type == "none":
            result["reason"] = "Current time is normal activity period"
            return result

        adjusted_fee = adjustment.get("adjusted_fee_ppm", current_fee)
        adj_pct = adjustment.get("adjustment_pct", 0)

        # Only recommend if change is meaningful (>5%)
        fee_diff_pct = abs(adjusted_fee - current_fee) / max(current_fee, 1) * 100
        if fee_diff_pct < 5:
            result["reason"] = f"Adjustment too small ({fee_diff_pct:.1f}%)"
            return result

        result["should_adjust"] = True
        result["recommended_fee"] = adjusted_fee
        result["adjustment_pct"] = adj_pct
        result["adjustment_type"] = adj_type

        if adj_type == "peak_increase":
            result["reason"] = (
                f"Peak hour detected ({adjustment.get('pattern_intensity', 0):.0%} intensity) - "
                "increase fee to capture premium"
            )
        elif adj_type == "low_decrease":
            result["reason"] = (
                f"Low activity period ({adjustment.get('pattern_intensity', 0):.0%} intensity) - "
                "decrease fee to attract flow"
            )

        # Estimate when adjustment reverts (rough: 1-2 hours typically)
        result["reverts_in_hours"] = 1

        return result

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

        # Count fresh vs stale cache entries
        fresh_count = 0
        stale_count = 0
        for cached in self._cache.values():
            age = now - cached.timestamp
            if age < CACHE_TTL_SECONDS:
                fresh_count += 1
            elif age < STALE_CACHE_TTL_SECONDS:
                stale_count += 1

        status = {
            "hive_available": self._hive_available,
            "circuit_breaker_open": self._circuit.is_open,
            "circuit_failures": self._circuit.failures,
            "cache_entries": len(self._cache),
            "cache_fresh": fresh_count,
            "cache_stale": stale_count,
            "last_availability_check": int(self._availability_check_time),
            "membership": None
        }

        # If hive is available, get membership details
        if self._hive_available:
            try:
                hive_status = self.plugin.rpc.call("hive-status")
                membership = hive_status.get("membership", {})
                status["membership"] = {
                    "tier": membership.get("tier"),
                    "hive_id": membership.get("hive_id"),
                    "joined_at": membership.get("joined_at"),
                    "uptime_pct": membership.get("uptime_pct"),
                    "contribution_ratio": membership.get("contribution_ratio")
                }
            except Exception as e:
                self._log(f"Error fetching membership details: {e}", level="debug")

        return status

    def clear_cache(self) -> int:
        """
        Clear all cached profiles.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count
