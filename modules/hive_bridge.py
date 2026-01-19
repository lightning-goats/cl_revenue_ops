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

Phase 2: Bidirectional Sharing (added later)
- report_observation(): Report fee observations back to cl-hive

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
        Check if cl-hive plugin is available (cached).

        Returns cached result if within TTL to avoid expensive RPC calls.

        Returns:
            True if cl-hive is active, False otherwise
        """
        now = time.time()

        # Return cached result if within TTL
        if (self._hive_available is not None and
                (now - self._availability_check_time) < self._availability_ttl):
            return self._hive_available

        # Check plugin list
        try:
            plugins = self.plugin.rpc.plugin("list")
            available = False
            for p in plugins.get("plugins", []):
                if "cl-hive" in p.get("name", "") and p.get("active", False):
                    available = True
                    break

            self._hive_available = available
            self._availability_check_time = now

            if available:
                self._log("cl-hive plugin detected as active")
            return available

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
        """Cache a profile."""
        self._cache[peer_id] = CachedProfile(
            data=data,
            timestamp=time.time()
        )

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
        if self._is_circuit_open():
            if cached_data:
                age = time.time() - self._cache[peer_id].timestamp
                return self._stale_with_reduced_confidence(cached_data, age)
            return None

        # Check if cl-hive is available
        if not self.is_available():
            if cached_data:
                age = time.time() - self._cache[peer_id].timestamp
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
                if cached_data:
                    age = time.time() - self._cache[peer_id].timestamp
                    return self._stale_with_reduced_confidence(cached_data, age)
                return None

            # Success - cache and return
            self._set_cached(peer_id, result)
            self._record_success()
            return result

        except Exception as e:
            self._log(f"Failed to query fee intelligence: {e}", level="debug")
            self._record_failure()

            if cached_data:
                age = time.time() - self._cache[peer_id].timestamp
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
    # DIAGNOSTIC METHODS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get bridge status for diagnostics.

        Returns:
            Dict with bridge status information
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

        return {
            "hive_available": self._hive_available,
            "circuit_breaker_open": self._circuit.is_open,
            "circuit_failures": self._circuit.failures,
            "cache_entries": len(self._cache),
            "cache_fresh": fresh_count,
            "cache_stale": stale_count,
            "last_availability_check": int(self._availability_check_time)
        }

    def clear_cache(self) -> int:
        """
        Clear all cached profiles.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count
