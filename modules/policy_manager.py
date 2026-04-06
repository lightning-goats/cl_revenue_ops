"""
Policy Manager module for cl-revenue-ops

Implements the Policy-Driven architecture for managing peer behavior.
Replaces the legacy ignored_peers system with a centralized, declarative
policy system that supports:
- Fee strategies: dynamic, static, passive
- Rebalance modes: enabled, disabled, source_only, sink_only
- Tags for grouping and filtering peers

v2.0 Improvements:
- Granular cache invalidation (write-through pattern)
- Per-policy fee multiplier bounds
- Time-limited policy overrides with auto-expiry
- Policy change event callbacks
- Auto-policy suggestions from profitability data
- Batch policy operations

Provides API hooks for external integration.
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database

# =============================================================================
# v2.0 CONFIGURATION CONSTANTS
# =============================================================================

# Per-policy fee multiplier bounds (security limits)
GLOBAL_MIN_FEE_MULTIPLIER = 0.1  # Absolute minimum allowed
GLOBAL_MAX_FEE_MULTIPLIER = 5.0  # Absolute maximum allowed

# Time-limited policy settings
MAX_POLICY_EXPIRY_DAYS = 30  # Maximum expiry duration
ENABLE_AUTO_EXPIRY = True  # Enable time-limited policies

# Auto-suggestion settings
ENABLE_AUTO_SUGGESTIONS = True
MIN_OBSERVATION_DAYS = 7  # Minimum data before suggesting
BLEEDER_THRESHOLD_PERIODS = 3  # Consecutive loss periods to suggest disable
ZOMBIE_FORWARD_THRESHOLD = 0  # Forwards threshold for zombie detection

# Policy change rate limiting
MAX_POLICY_CHANGES_PER_MINUTE = 10  # Rate limit per peer

READ_ONLY_POLICY_ACTIONS = frozenset({"list", "get", "find", "changes"})
TACTICAL_POLICY_ACTIONS = frozenset({"set", "delete", "tag", "untag", "batch"})


class FeeStrategy(Enum):
    """Fee control strategy for a peer."""
    DYNAMIC = "dynamic"   # DTS+PID fee optimization (Default)
    STATIC = "static"     # Fixed fee (User Override)
    PASSIVE = "passive"   # Do nothing (manual control)


class RebalanceMode(Enum):
    """Rebalancing behavior for a peer."""
    ENABLED = "enabled"       # Full rebalancing allowed
    DISABLED = "disabled"     # No rebalancing
    SOURCE_ONLY = "source_only"  # Can drain, cannot fill
    SINK_ONLY = "sink_only"      # Can fill, cannot drain


@dataclass
class PeerPolicy:
    """
    Immutable policy snapshot for a peer.

    Attributes:
        peer_id: 66-character hex public key
        strategy: Fee control strategy
        rebalance_mode: Rebalancing behavior
        fee_ppm_target: Target fee for static strategy
        tags: List of string tags for grouping
        updated_at: Unix timestamp of last update

        v2.0 Fields:
        fee_multiplier_min: Override minimum flow multiplier for this peer
        fee_multiplier_max: Override maximum flow multiplier for this peer
        expires_at: Unix timestamp when policy auto-reverts (None = permanent)
    """
    peer_id: str
    strategy: FeeStrategy = FeeStrategy.DYNAMIC
    rebalance_mode: RebalanceMode = RebalanceMode.ENABLED
    fee_ppm_target: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    updated_at: int = 0
    # v2.0 fields
    fee_multiplier_min: Optional[float] = None
    fee_multiplier_max: Optional[float] = None
    expires_at: Optional[int] = None  # Unix timestamp; None = permanent

    def has_tag(self, tag: str) -> bool:
        """Check if this policy has a specific tag."""
        return tag in self.tags

    def is_expired(self) -> bool:
        """Check if this policy has expired."""
        if not ENABLE_AUTO_EXPIRY or self.expires_at is None:
            return False
        return int(time.time()) > self.expires_at

    def get_fee_multiplier_bounds(self) -> tuple:
        """
        Get effective fee multiplier bounds for this peer.

        Returns (min, max) with global limits enforced.
        """
        min_mult = self.fee_multiplier_min if self.fee_multiplier_min is not None else GLOBAL_MIN_FEE_MULTIPLIER
        max_mult = self.fee_multiplier_max if self.fee_multiplier_max is not None else GLOBAL_MAX_FEE_MULTIPLIER

        # Enforce global security bounds
        min_mult = max(GLOBAL_MIN_FEE_MULTIPLIER, min(min_mult, GLOBAL_MAX_FEE_MULTIPLIER))
        max_mult = max(GLOBAL_MIN_FEE_MULTIPLIER, min(max_mult, GLOBAL_MAX_FEE_MULTIPLIER))

        # Ensure min <= max
        if min_mult > max_mult:
            min_mult, max_mult = max_mult, min_mult

        return (min_mult, max_mult)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "peer_id": self.peer_id,
            "strategy": self.strategy.value,
            "rebalance_mode": self.rebalance_mode.value,
            "fee_ppm_target": self.fee_ppm_target,
            "tags": self.tags,
            "updated_at": self.updated_at,
            # v2.0 fields
            "fee_multiplier_min": self.fee_multiplier_min,
            "fee_multiplier_max": self.fee_multiplier_max,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired()
        }


# Regex for validating 66-character hex peer IDs
PEER_ID_PATTERN = re.compile(r'^[0-9a-fA-F]{66}$')


class PolicyManager:
    """
    Centralized policy management for peer behavior.

    Replaces the legacy ignored_peers system with a declarative policy
    architecture. All peer-specific behavior decisions flow through this
    manager to ensure consistency.

    v2.0 Improvements:
    - Write-through cache with granular invalidation
    - Policy change callbacks for immediate response
    - Time-limited policies with auto-expiry
    - Per-policy fee multiplier bounds
    - Auto-suggestions from profitability data
    - Batch operations for efficiency

    Thread Safety:
        Uses the Database's thread-local connection pattern.
        Policy reads are idempotent; writes are atomic via SQLite.

    Known Limitations:
    - PM-1: Write-through cache update can overwrite concurrent DB writes in a narrow
      window. DB is authoritative; cache is rebuilt on next full load. Non-critical.
    - PM-4: Batch rate-limit checks use arrival-order, not policy-priority order. This means
      high-priority policy changes could be rate-limited by earlier low-priority ones in the
      same batch. Non-critical timing issue.
    """

    # Default policy for peers without explicit configuration
    # L11 FIX: Use default_factory (no explicit tags=[]) to avoid shared mutable list
    DEFAULT_POLICY = PeerPolicy(
        peer_id="",
        strategy=FeeStrategy.DYNAMIC,
        rebalance_mode=RebalanceMode.ENABLED,
        fee_ppm_target=None,
        updated_at=0
    )

    def __init__(self, database: 'Database', plugin):
        """
        Initialize the PolicyManager.

        Args:
            database: Database instance for persistence
            plugin: Reference to the pyln Plugin for logging
        """
        self.database = database
        self.plugin = plugin
        self.hive_hints = None  # Injected by main plugin for corridor-aware policies
        self.rpc_cache = None
        self.data_service = None  # Unified data service (injected by main plugin)

        # In-memory cache with write-through pattern (v2.0)
        self._cache: Dict[str, PeerPolicy] = {}
        self._cache_valid = False
        self._cache_lock = threading.Lock()

        # v2.0: Policy change callbacks
        self._on_change_callbacks: List[Callable[[str, PeerPolicy], None]] = []
        self._callback_lock = threading.Lock()  # TS-8: Protect callback list

        # v2.0: Rate limiting for policy changes (peer_id -> list of timestamps)
        self._change_timestamps: Dict[str, List[int]] = {}

    # =========================================================================
    # v2.0 Callback Registration
    # =========================================================================

    def register_on_change(self, callback: Callable[[str, PeerPolicy], None]) -> None:
        """
        Register a callback to be invoked when policies change.

        Callbacks receive (peer_id, new_policy) and are called after
        the policy is persisted but before the method returns.

        Args:
            callback: Function taking (peer_id: str, policy: PeerPolicy)
        """
        # TS-8: Protect callback list modifications
        with self._callback_lock:
            if callback not in self._on_change_callbacks:
                self._on_change_callbacks.append(callback)
                self.plugin.log(f"PolicyManager: Registered change callback", level='debug')

    def unregister_on_change(self, callback: Callable[[str, PeerPolicy], None]) -> None:
        """Remove a previously registered callback."""
        with self._callback_lock:
            if callback in self._on_change_callbacks:
                self._on_change_callbacks.remove(callback)

    def _notify_change(self, peer_id: str, policy: PeerPolicy) -> None:
        """Invoke all registered callbacks for a policy change."""
        # TS-8: Snapshot under lock, invoke outside lock to avoid deadlock
        with self._callback_lock:
            callbacks = list(self._on_change_callbacks)
        for cb in callbacks:
            try:
                cb(peer_id, policy)
            except Exception as e:
                self.plugin.log(
                    f"PolicyManager: Callback error for {peer_id[:12]}...: {e}",
                    level='warn'
                )

    # =========================================================================
    # v2.0 Rate Limiting
    # =========================================================================

    def _check_rate_limit(self, peer_id: str) -> bool:
        """
        Check if policy change is within rate limit.

        Returns True if change is allowed, False if rate limited.
        """
        now = int(time.time())
        window_start = now - 60  # 1 minute window

        with self._cache_lock:
            # SEC-4: Periodically clean up stale entries to bound memory growth
            if len(self._change_timestamps) > 500:
                self._change_timestamps = {
                    pid: [ts for ts in tss if ts > window_start]
                    for pid, tss in self._change_timestamps.items()
                    if any(ts > window_start for ts in tss)
                }

            # Get timestamps for this peer
            timestamps = self._change_timestamps.get(peer_id, [])

            # Prune old timestamps
            timestamps = [ts for ts in timestamps if ts > window_start]

            if len(timestamps) >= MAX_POLICY_CHANGES_PER_MINUTE:
                return False

            # Check passes — record only after caller confirms the DB write succeeded
            # (previously recorded here, but that polluted the counter on DB write failure)
            return True

    def _record_rate_limit_change(self, peer_id: str) -> None:
        """Record a successful policy change for rate limiting."""
        now = int(time.time())
        window_start = now - 60
        with self._cache_lock:
            timestamps = self._change_timestamps.get(peer_id, [])
            timestamps = [ts for ts in timestamps if ts > window_start]
            timestamps.append(now)
            self._change_timestamps[peer_id] = timestamps
    
    def _validate_peer_id(self, peer_id: str) -> None:
        """
        Validate peer ID format.
        
        Raises:
            ValueError: If peer_id is not a valid 66-char hex string
        """
        # L-R5-2 FIX: Handle None/non-string inputs without TypeError
        if not peer_id or not isinstance(peer_id, str) or not PEER_ID_PATTERN.match(peer_id):
            peer_str = str(peer_id or '')
            raise ValueError(
                f"Invalid peer_id: must be 66-character hex string, got '{peer_str[:20]}...'"
                if len(peer_str) > 20 else f"Invalid peer_id: '{peer_str}'"
            )
    
    def _invalidate_cache(self) -> None:
        """Invalidate the in-memory policy cache (full reload on next access)."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_valid = False

    def _update_cache(self, peer_id: str, policy: PeerPolicy) -> None:
        """
        v2.0: Write-through cache update for single peer.

        More efficient than full invalidation for single-peer operations.
        """
        with self._cache_lock:
            self._cache[peer_id] = policy
            # Cache remains valid since we updated it directly

    def _remove_from_cache(self, peer_id: str) -> None:
        """v2.0: Remove single peer from cache."""
        with self._cache_lock:
            if peer_id in self._cache:
                del self._cache[peer_id]

    def _load_cache(self) -> None:
        """Load all policies into cache from database."""
        # M6 FIX: Check validity under lock, but read DB outside lock
        # to avoid blocking all policy-reading threads during DB I/O.
        with self._cache_lock:
            if self._cache_valid:
                return

        # Read DB outside the lock (thread-local connections are safe)
        rows = self.database.get_all_policies()

        new_cache = {}
        for row in rows:
            policy = self._row_to_policy(row)
            # v2.0: Skip expired policies during cache load
            if not policy.is_expired():
                new_cache[policy.peer_id] = policy

        # Update cache atomically under lock
        with self._cache_lock:
            # Re-check: another thread may have loaded while we were reading
            if not self._cache_valid:
                self._cache = new_cache
                self._cache_valid = True

    def _row_to_policy(self, row) -> PeerPolicy:
        """Convert a database row to a PeerPolicy object."""
        tags_json = row['tags'] or '[]'
        try:
            tags = json.loads(tags_json)
        except (json.JSONDecodeError, TypeError) as e:
            self.plugin.log(
                f"PolicyManager: Corrupted tags JSON for peer {row['peer_id'][:12]}...: {e}. "
                f"Defaulting to empty tags.",
                level='unusual'
            )
            tags = []

        # v2.0: Handle new fields with backwards compatibility
        fee_multiplier_min = None
        fee_multiplier_max = None
        expires_at = None

        try:
            fee_multiplier_min = row['fee_multiplier_min']
            fee_multiplier_max = row['fee_multiplier_max']
            expires_at = row['expires_at']
        except (KeyError, IndexError):
            # Old schema without v2.0 columns - use defaults
            pass

        try:
            strategy = FeeStrategy(row['strategy'])
        except ValueError:
            self.plugin.log(
                f"PolicyManager: Invalid strategy '{row['strategy']}' for peer "
                f"{row['peer_id'][:12]}..., defaulting to dynamic",
                level='unusual'
            )
            strategy = FeeStrategy.DYNAMIC

        try:
            rebalance_mode = RebalanceMode(row['rebalance_mode'])
        except ValueError:
            self.plugin.log(
                f"PolicyManager: Invalid rebalance_mode '{row['rebalance_mode']}' for peer "
                f"{row['peer_id'][:12]}..., defaulting to enabled",
                level='unusual'
            )
            rebalance_mode = RebalanceMode.ENABLED

        return PeerPolicy(
            peer_id=row['peer_id'],
            strategy=strategy,
            rebalance_mode=rebalance_mode,
            fee_ppm_target=row['fee_ppm_target'],
            tags=tags if isinstance(tags, list) else [],
            updated_at=row['updated_at'],
            fee_multiplier_min=fee_multiplier_min,
            fee_multiplier_max=fee_multiplier_max,
            expires_at=expires_at
        )
    
    # =========================================================================
    # Core Policy Methods
    # =========================================================================
    
    def get_policy(self, peer_id: str) -> PeerPolicy:
        """
        Get the policy for a specific peer.

        If no explicit policy exists, returns the default policy
        (dynamic strategy, rebalancing enabled).

        v2.0: Handles expired policies by returning defaults.

        Args:
            peer_id: 66-character hex public key

        Returns:
            PeerPolicy for the peer
        """
        # Check cache first
        self._load_cache()
        expired = False
        with self._cache_lock:
            if peer_id in self._cache:
                policy = self._cache[peer_id]
                # v2.0: Check expiry
                if policy.is_expired():
                    self.plugin.log(
                        f"PolicyManager: Policy for {peer_id[:12]}... expired, reverting to defaults",
                        level='info'
                    )
                    # Remove expired policy from cache
                    del self._cache[peer_id]
                    expired = True
                else:
                    return policy
        # Handle expired policy DB cleanup outside the lock
        if expired:
            self._delete_expired_policy(peer_id)

        # Return default policy with this peer_id
        return PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
            rebalance_mode=RebalanceMode.ENABLED,
            fee_ppm_target=None,
            tags=[],
            updated_at=0
        )

    def _delete_expired_policy(self, peer_id: str) -> None:
        """Delete an expired policy from the database."""
        try:
            now = int(time.time())
            self.database.delete_expired_policies(now)
        except Exception as e:
            self.plugin.log(f"PolicyManager: Error deleting expired policy: {e}", level='warn')

    def get_policy_changes_since(self, since_timestamp: int) -> List[Dict[str, Any]]:
        """
        Get all policy changes since a given timestamp.

        Get all policy changes since a given timestamp, allowing efficient
        polling for policy updates without fetching all policies.

        Args:
            since_timestamp: Unix timestamp. Returns policies updated after this time.

        Returns:
            List of policy dicts with peer_id, strategy, rebalance_mode, and updated_at.
            Returns empty list if no changes since timestamp.

        Example:
            # Get policies changed in last 5 minutes
            changes = policy_manager.get_policy_changes_since(int(time.time()) - 300)
        """
        try:
            rows = self.database.get_policy_changes_since(since_timestamp)

            changes = []
            for row in rows:
                policy = self._row_to_policy(row)
                # Skip expired policies
                if policy.is_expired():
                    continue
                changes.append(policy.to_dict())

            return changes

        except Exception as e:
            self.plugin.log(
                f"PolicyManager: Error getting policy changes: {e}",
                level='warn'
            )
            return []

    def get_last_policy_change_timestamp(self) -> int:
        """
        Get the timestamp of the most recent policy change.

        Useful for checking if there are any new changes
        without fetching all changes.

        Returns:
            Unix timestamp of most recent change, or 0 if no policies exist.
        """
        try:
            return self.database.get_last_policy_change_timestamp()
        except Exception as e:
            self.plugin.log(
                f"PolicyManager: Error getting last change timestamp: {e}",
                level='warn'
            )
            return 0

    def set_policy(
        self,
        peer_id: str,
        strategy: Optional[str] = None,
        rebalance_mode: Optional[str] = None,
        fee_ppm_target: Optional[int] = None,
        tags: Optional[List[str]] = None,
        # v2.0 parameters
        fee_multiplier_min: Optional[float] = None,
        fee_multiplier_max: Optional[float] = None,
        expires_in_hours: Optional[int] = None
    ) -> PeerPolicy:
        """
        Set or update the policy for a peer.

        Only provided fields are updated; others retain existing values
        or defaults for new peers.

        v2.0: Adds fee multiplier bounds, expiry, rate limiting, and callbacks.

        Args:
            peer_id: 66-character hex public key
            strategy: Fee strategy (dynamic, static, passive)
            rebalance_mode: Rebalance mode (enabled, disabled, source_only, sink_only)
            fee_ppm_target: Target fee for static strategy
            tags: List of string tags (replaces existing tags)
            fee_multiplier_min: v2.0 - Minimum fee multiplier for this peer
            fee_multiplier_max: v2.0 - Maximum fee multiplier for this peer
            expires_in_hours: v2.0 - Hours until policy auto-reverts (None = permanent)

        Returns:
            Updated PeerPolicy

        Raises:
            ValueError: If peer_id or enum values are invalid
            RuntimeError: If rate limited
        """
        self._validate_peer_id(peer_id)

        # B2 FIX: Rate limit check moved after validation but before DB write.
        # Previously it was here, which incremented the counter even when
        # validation below would reject the request.

        # Get existing policy or default
        existing = self.get_policy(peer_id)

        # Validate and convert strategy
        new_strategy = existing.strategy
        if strategy is not None:
            try:
                new_strategy = FeeStrategy(strategy.lower())
            except ValueError:
                valid = [s.value for s in FeeStrategy]
                raise ValueError(f"Invalid strategy '{strategy}'. Valid: {valid}")

        # Validate and convert rebalance_mode
        new_rebalance_mode = existing.rebalance_mode
        if rebalance_mode is not None:
            try:
                new_rebalance_mode = RebalanceMode(rebalance_mode.lower())
            except ValueError:
                valid = [m.value for m in RebalanceMode]
                raise ValueError(f"Invalid rebalance_mode '{rebalance_mode}'. Valid: {valid}")

        # Validate fee_ppm_target
        new_fee_ppm = fee_ppm_target if fee_ppm_target is not None else existing.fee_ppm_target
        if new_fee_ppm is not None:
            if not isinstance(new_fee_ppm, int) or new_fee_ppm < 0:
                raise ValueError(f"fee_ppm_target must be a non-negative integer, got {new_fee_ppm}")
            if new_fee_ppm > 100000:
                raise ValueError(f"fee_ppm_target cannot exceed 100000 PPM")

        # Validate tags
        new_tags = tags if tags is not None else existing.tags
        if not isinstance(new_tags, list):
            raise ValueError("tags must be a list of strings")
        new_tags = [str(t) for t in new_tags]

        # v2.0: Validate fee multiplier bounds
        new_mult_min = fee_multiplier_min if fee_multiplier_min is not None else existing.fee_multiplier_min
        new_mult_max = fee_multiplier_max if fee_multiplier_max is not None else existing.fee_multiplier_max

        if new_mult_min is not None:
            if not isinstance(new_mult_min, (int, float)) or new_mult_min < GLOBAL_MIN_FEE_MULTIPLIER:
                raise ValueError(f"fee_multiplier_min must be >= {GLOBAL_MIN_FEE_MULTIPLIER}")
            if new_mult_min > GLOBAL_MAX_FEE_MULTIPLIER:
                raise ValueError(f"fee_multiplier_min must be <= {GLOBAL_MAX_FEE_MULTIPLIER}")

        if new_mult_max is not None:
            if not isinstance(new_mult_max, (int, float)) or new_mult_max < GLOBAL_MIN_FEE_MULTIPLIER:
                raise ValueError(f"fee_multiplier_max must be >= {GLOBAL_MIN_FEE_MULTIPLIER}")
            if new_mult_max > GLOBAL_MAX_FEE_MULTIPLIER:
                raise ValueError(f"fee_multiplier_max must be <= {GLOBAL_MAX_FEE_MULTIPLIER}")

        # L-R5-8 FIX: Cross-validate multiplier min/max instead of silently storing
        # inverted bounds that get swapped at read time by get_fee_multiplier_bounds().
        if new_mult_min is not None and new_mult_max is not None:
            if new_mult_min > new_mult_max:
                raise ValueError(
                    f"fee_multiplier_min ({new_mult_min}) cannot exceed "
                    f"fee_multiplier_max ({new_mult_max})"
                )

        # v2.0: Calculate expiry timestamp
        new_expires_at = existing.expires_at
        if expires_in_hours is not None:
            if expires_in_hours <= 0:
                new_expires_at = None  # Clear expiry
            else:
                max_hours = MAX_POLICY_EXPIRY_DAYS * 24
                if expires_in_hours > max_hours:
                    raise ValueError(f"expires_in_hours cannot exceed {max_hours} ({MAX_POLICY_EXPIRY_DAYS} days)")
                new_expires_at = int(time.time()) + (expires_in_hours * 3600)

        now = int(time.time())

        # B2 FIX: Rate limit check after validation, before DB write.
        # This ensures the counter only increments for valid, committed changes.
        if not self._check_rate_limit(peer_id):
            raise RuntimeError(
                f"Rate limited: max {MAX_POLICY_CHANGES_PER_MINUTE} changes/minute for {peer_id[:12]}..."
            )

        # Persist to database (v2.0: includes new columns)
        self.database.upsert_policy(
            peer_id, new_strategy.value, new_rebalance_mode.value,
            new_fee_ppm, json.dumps(new_tags), now,
            new_mult_min, new_mult_max, new_expires_at
        )

        # Record rate-limit change AFTER successful DB write (not before)
        self._record_rate_limit_change(peer_id)

        # v2.0: Write-through cache update (instead of full invalidation)
        new_policy = PeerPolicy(
            peer_id=peer_id,
            strategy=new_strategy,
            rebalance_mode=new_rebalance_mode,
            fee_ppm_target=new_fee_ppm,
            tags=new_tags,
            updated_at=now,
            fee_multiplier_min=new_mult_min,
            fee_multiplier_max=new_mult_max,
            expires_at=new_expires_at
        )

        # v2.0: Update cache directly (write-through)
        self._update_cache(peer_id, new_policy)

        self.plugin.log(
            f"PolicyManager: Set policy for {peer_id[:12]}... -> "
            f"strategy={new_strategy.value}, rebalance={new_rebalance_mode.value}"
            + (f", expires_at={new_expires_at}" if new_expires_at else ""),
            level='info'
        )

        # v2.0: Notify callbacks
        self._notify_change(peer_id, new_policy)

        return new_policy

    def delete_policy(self, peer_id: str) -> bool:
        """
        Delete the policy for a peer, reverting to defaults.

        v2.0: Uses granular cache removal and notifies callbacks.

        Args:
            peer_id: 66-character hex public key

        Returns:
            True if a policy was deleted, False if none existed
        """
        self._validate_peer_id(peer_id)

        deleted = self.database.delete_policy(peer_id)

        # v2.0: Granular cache removal
        self._remove_from_cache(peer_id)
        if deleted:
            self.plugin.log(
                f"PolicyManager: Deleted policy for {peer_id[:12]}..., reverting to defaults",
                level='info'
            )
            # v2.0: Notify with default policy
            default_policy = PeerPolicy(
                peer_id=peer_id,
                strategy=FeeStrategy.DYNAMIC,
                rebalance_mode=RebalanceMode.ENABLED
            )
            self._notify_change(peer_id, default_policy)

        return deleted
    
    def get_all_policies(self) -> List[PeerPolicy]:
        """
        Get all explicitly configured policies.

        Returns:
            List of non-expired PeerPolicy objects
        """
        self._load_cache()
        with self._cache_lock:
            return [p for p in self._cache.values() if not p.is_expired()]
    
    # =========================================================================
    # Tag Management
    # =========================================================================
    
    def add_tag(self, peer_id: str, tag: str) -> PeerPolicy:
        """
        Add a tag to a peer's policy.
        
        Creates a policy with defaults if none exists.
        
        Args:
            peer_id: 66-character hex public key
            tag: Tag string to add
            
        Returns:
            Updated PeerPolicy
        """
        self._validate_peer_id(peer_id)
        
        existing = self.get_policy(peer_id)
        new_tags = list(existing.tags)
        
        if tag not in new_tags:
            new_tags.append(tag)
            return self.set_policy(peer_id, tags=new_tags)
        
        return existing
    
    def remove_tag(self, peer_id: str, tag: str) -> PeerPolicy:
        """
        Remove a tag from a peer's policy.
        
        Args:
            peer_id: 66-character hex public key
            tag: Tag string to remove
            
        Returns:
            Updated PeerPolicy
        """
        self._validate_peer_id(peer_id)
        
        existing = self.get_policy(peer_id)
        new_tags = [t for t in existing.tags if t != tag]
        
        if len(new_tags) != len(existing.tags):
            return self.set_policy(peer_id, tags=new_tags)
        
        return existing
    
    def get_peers_by_tag(self, tag: str) -> List[PeerPolicy]:
        """
        Get all peers with a specific tag.

        Args:
            tag: Tag string to search for

        Returns:
            List of non-expired PeerPolicy objects with the tag
        """
        self._load_cache()
        with self._cache_lock:
            return [p for p in self._cache.values() if p.has_tag(tag) and not p.is_expired()]

    def get_peers_by_strategy(self, strategy: FeeStrategy) -> List[PeerPolicy]:
        """
        Get all peers with a specific fee strategy.

        Args:
            strategy: FeeStrategy enum value

        Returns:
            List of non-expired PeerPolicy objects with that strategy
        """
        self._load_cache()
        with self._cache_lock:
            return [p for p in self._cache.values() if p.strategy == strategy and not p.is_expired()]
    
    # =========================================================================
    # Convenience Methods for Logic Cores
    # =========================================================================
    
    def should_manage_fees(self, peer_id: str) -> bool:
        """
        Check if cl-revenue-ops should manage fees for this peer.
        
        Returns False for PASSIVE strategy (equivalent to old is_peer_ignored).
        
        Args:
            peer_id: 66-character hex public key
            
        Returns:
            True if fees should be managed, False otherwise
        """
        policy = self.get_policy(peer_id)
        return policy.strategy != FeeStrategy.PASSIVE
    
    def should_rebalance(self, peer_id: str, as_destination: bool = False) -> bool:
        """
        Check if rebalancing is allowed for this peer.
        
        Args:
            peer_id: 66-character hex public key
            as_destination: True if checking for filling, False for draining
            
        Returns:
            True if rebalancing is allowed, False otherwise
        """
        policy = self.get_policy(peer_id)
        mode = policy.rebalance_mode
        
        if mode == RebalanceMode.DISABLED:
            return False
        if mode == RebalanceMode.SOURCE_ONLY and as_destination:
            return False
        if mode == RebalanceMode.SINK_ONLY and not as_destination:
            return False
        
        return True
    
    def get_static_fee(self, peer_id: str) -> Optional[int]:
        """
        Get the static fee target for a peer, if applicable.
        
        Args:
            peer_id: 66-character hex public key
            
        Returns:
            Fee in PPM if strategy is STATIC and fee_ppm_target is set,
            None otherwise
        """
        policy = self.get_policy(peer_id)
        if policy.strategy == FeeStrategy.STATIC and policy.fee_ppm_target is not None:
            return policy.fee_ppm_target
        return None
    
    def get_fee_multiplier_bounds(self, peer_id: str) -> tuple:
        """
        v2.0: Get effective fee multiplier bounds for a peer.

        Returns (min, max) with per-policy overrides applied and global limits enforced.

        Args:
            peer_id: 66-character hex public key

        Returns:
            Tuple of (min_multiplier, max_multiplier)
        """
        policy = self.get_policy(peer_id)
        return policy.get_fee_multiplier_bounds()

    # =========================================================================
    # v2.0 Auto-Policy Suggestions
    # =========================================================================

    def get_policy_suggestions(self, profitability_analyzer=None) -> List[Dict[str, Any]]:
        """
        v2.0: Generate policy suggestions based on profitability data.

        Analyzes channel performance and suggests policy changes to improve alpha:
        - Consistent bleeders: Suggest disabling rebalance
        - Zombies (0 activity + underwater): Suggest passive + close
        - High-velocity sources: Suggest source_only rebalance mode

        Args:
            profitability_analyzer: Optional ProfitabilityAnalyzer instance

        Returns:
            List of suggestion dicts with peer_id, current_policy, suggested_changes, reason
        """
        if not ENABLE_AUTO_SUGGESTIONS:
            return []

        suggestions = []

        try:
            if profitability_analyzer is None:
                self.plugin.log("PolicyManager: No profitability_analyzer for suggestions", level='debug')
                return []

            # Get bleeders (channels where rebalance cost > revenue)
            bleeders = profitability_analyzer.identify_bleeders(window_days=MIN_OBSERVATION_DAYS)

            for bleeder in bleeders:
                peer_id = bleeder.get('peer_id', '')
                channel_id = bleeder.get('channel_id', '')
                if not peer_id:
                    continue

                current_policy = self.get_policy(peer_id)

                # Skip if already set to passive/disabled
                if current_policy.strategy == FeeStrategy.PASSIVE:
                    continue
                if current_policy.rebalance_mode == RebalanceMode.DISABLED:
                    continue

                net_pnl = bleeder.get('net_pnl_sats', 0)
                forward_count = bleeder.get('forward_count', 0)
                sourced_forward_count = bleeder.get('sourced_forward_count', 0)
                total_activity = forward_count + sourced_forward_count
                rebalance_cost = bleeder.get('rebalance_cost_sats', 0)

                # Query success rate trend: compare recent (7d) vs historical (30d)
                trend = "stable"
                if channel_id and self.database:
                    try:
                        recent_sr = self.database.get_channel_rebalance_success_rate(channel_id, 7)
                        historical_sr = self.database.get_channel_rebalance_success_rate(channel_id, 30)
                        if (recent_sr and historical_sr and
                                recent_sr['total'] >= 2 and historical_sr['total'] >= 3):
                            delta = recent_sr['success_rate'] - historical_sr['success_rate']
                            if delta < -0.15:
                                trend = "declining"
                            elif delta > 0.15:
                                trend = "improving"
                    except Exception as e:
                        self.plugin.log(f"Bleeder trend query failed for {channel_id[:12]}...: {e}", level='debug')

                # Determine suggestion type
                if total_activity == ZOMBIE_FORWARD_THRESHOLD and net_pnl < 0:
                    # Zombie: No activity but costs money
                    suggestions.append({
                        'peer_id': peer_id,
                        'peer_id_short': peer_id[:12] + '...',
                        'current_strategy': current_policy.strategy.value,
                        'current_rebalance_mode': current_policy.rebalance_mode.value,
                        'suggested_strategy': 'passive',
                        'suggested_rebalance_mode': 'disabled',
                        'reason': f"Zombie channel: 0 forwards, {abs(net_pnl)} sats loss",
                        'severity': 'high',
                        'action': 'consider_close',
                        'trend': trend
                    })
                elif net_pnl < 0 and rebalance_cost > 0:
                    # Bleeder: Active but rebalance costs exceed revenue
                    # Declining trend escalates severity; improving trend softens it
                    if trend == "declining":
                        severity = 'high'
                        action = 'disable_rebalance'
                    elif trend == "improving":
                        severity = 'medium'
                        action = 'monitor'
                    else:
                        severity = 'medium'
                        action = 'disable_rebalance'

                    suggestions.append({
                        'peer_id': peer_id,
                        'peer_id_short': peer_id[:12] + '...',
                        'current_strategy': current_policy.strategy.value,
                        'current_rebalance_mode': current_policy.rebalance_mode.value,
                        'suggested_strategy': None,  # Keep current
                        'suggested_rebalance_mode': 'disabled' if action == 'disable_rebalance' else None,
                        'reason': f"Bleeder: rebalance cost {rebalance_cost} > revenue, net loss {abs(net_pnl)} sats",
                        'severity': severity,
                        'action': action,
                        'trend': trend
                    })

            # Check for high-velocity sources that should be protected
            try:
                channel_states = self.database.get_all_channel_states()
                for state in channel_states:
                    peer_id = state.get('peer_id', '')
                    if not peer_id:
                        continue

                    flow_state = state.get('state', '')
                    flow_ratio = state.get('flow_ratio', 0)

                    # High-velocity source: strong outflow ratio
                    if flow_state == 'source' and flow_ratio > 0.7:
                        current_policy = self.get_policy(peer_id)

                        # Skip if already source_only
                        if current_policy.rebalance_mode == RebalanceMode.SOURCE_ONLY:
                            continue
                        if current_policy.rebalance_mode == RebalanceMode.DISABLED:
                            continue

                        suggestions.append({
                            'peer_id': peer_id,
                            'peer_id_short': peer_id[:12] + '...',
                            'current_strategy': current_policy.strategy.value,
                            'current_rebalance_mode': current_policy.rebalance_mode.value,
                            'suggested_strategy': None,
                            'suggested_rebalance_mode': 'source_only',
                            'reason': f"High-velocity source: flow_ratio {flow_ratio:.2f}, protect from draining",
                            'severity': 'low',
                            'action': 'protect_source'
                        })
            except Exception as e:
                self.plugin.log(f"PolicyManager: Error checking sources: {e}", level='debug')

        except Exception as e:
            self.plugin.log(f"PolicyManager: Error generating suggestions: {e}", level='warn')

        return suggestions

    # =========================================================================
    # Corridor-Aware Auto-Policies
    # =========================================================================

    def apply_corridor_policies(self) -> int:
        """Auto-set protective policies for corridor owner channels.

        Corridor owners get:
        - strategy=DYNAMIC (active fee optimization)
        - rebalance_mode=ENABLED (allow rebalancing to maintain liquidity)
        - tag='corridor_owner' for identification

        Fleet members get:
        - strategy=HIVE (0-fee fleet policy)
        - rebalance_mode=ENABLED

        Only applies to peers without explicit manual policies.
        Returns count of policies applied.
        """
        if not self.hive_hints:
            return 0

        applied = 0
        try:
            # Get all channels to find peers
            channels = self.rpc_cache.listpeerchannels() if self.rpc_cache else self.plugin.rpc.listpeerchannels()
            for ch in channels.get("channels", []):
                if ch.get("state") != "CHANNELD_NORMAL":
                    continue
                peer_id = ch.get("peer_id", "")
                if not peer_id:
                    continue

                # Check current policy — skip if manually set
                current = self.get_policy(peer_id)
                if current.peer_id and current.tags and "manual" in current.tags:
                    continue

                is_member = self.hive_hints.is_hive_member(peer_id)
                corridor_role = self.hive_hints.get_corridor_role(peer_id)

                if is_member:
                    # Fleet members: STATIC 0-fee policy
                    if not (current.tags and "auto_fleet" in current.tags):
                        self.set_policy(
                            peer_id,
                            strategy=FeeStrategy.STATIC.value,
                            fee_ppm_target=0,
                            rebalance_mode=RebalanceMode.ENABLED.value,
                            tags=["corridor_owner", "auto_fleet"],
                        )
                        applied += 1
                elif corridor_role == "owner":
                    # Corridor owners: DYNAMIC + protected
                    if not (current.tags and "corridor_owner" in current.tags):
                        self.set_policy(
                            peer_id,
                            strategy=FeeStrategy.DYNAMIC.value,
                            rebalance_mode=RebalanceMode.ENABLED.value,
                            tags=["corridor_owner", "auto_corridor"],
                        )
                        applied += 1

        except Exception as e:
            self.plugin.log(f"PolicyManager: Corridor policy error: {e}", level='warn')

        if applied > 0:
            self.plugin.log(
                f"PolicyManager: Applied {applied} corridor-aware policies",
                level='info'
            )
        return applied

    # =========================================================================
    # v2.0 Batch Operations
    # =========================================================================

    def set_policies_batch(self, updates: List[Dict[str, Any]]) -> List[PeerPolicy]:
        """
        v2.0: Batch update multiple peer policies in a single transaction.

        More efficient than calling set_policy() multiple times for bulk updates.

        Args:
            updates: List of dicts with keys: peer_id, strategy, rebalance_mode,
                    fee_ppm_target, tags, fee_multiplier_min, fee_multiplier_max,
                    expires_in_hours

        Returns:
            List of updated PeerPolicy objects

        Raises:
            ValueError: If any update is invalid (entire batch fails)
        """
        MAX_BATCH_SIZE = 100

        if len(updates) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(updates)} exceeds maximum {MAX_BATCH_SIZE}")

        if not updates:
            return []

        results = []
        now = int(time.time())

        # M5 FIX: Validate all updates first (fail fast).
        # Rate limit check is deferred until after full validation to avoid
        # incrementing counters for batches that fail on a later entry.
        validated = []
        for update in updates:
            peer_id = update.get('peer_id', '')
            self._validate_peer_id(peer_id)

            existing = self.get_policy(peer_id)

            # Process strategy
            strategy = update.get('strategy')
            new_strategy = existing.strategy
            if strategy is not None:
                try:
                    new_strategy = FeeStrategy(strategy.lower())
                except ValueError:
                    raise ValueError(f"Invalid strategy '{strategy}' for peer {peer_id[:12]}...")

            # Process rebalance_mode
            rebalance_mode = update.get('rebalance_mode')
            new_rebalance_mode = existing.rebalance_mode
            if rebalance_mode is not None:
                try:
                    new_rebalance_mode = RebalanceMode(rebalance_mode.lower())
                except ValueError:
                    raise ValueError(f"Invalid rebalance_mode '{rebalance_mode}' for peer {peer_id[:12]}...")

            # Process and validate other fields (matching set_policy validation)
            fee_ppm = update.get('fee_ppm_target', existing.fee_ppm_target)
            if fee_ppm is not None:
                if not isinstance(fee_ppm, int) or fee_ppm < 0:
                    raise ValueError(f"fee_ppm_target must be a non-negative integer for peer {peer_id[:12]}...")
                if fee_ppm > 100000:
                    raise ValueError(f"fee_ppm_target cannot exceed 100000 PPM for peer {peer_id[:12]}...")

            tags = update.get('tags', existing.tags)
            if not isinstance(tags, list):
                raise ValueError(f"tags must be a list of strings for peer {peer_id[:12]}...")
            tags = [str(t) for t in tags]

            mult_min = update.get('fee_multiplier_min', existing.fee_multiplier_min)
            if mult_min is not None:
                if not isinstance(mult_min, (int, float)) or mult_min < GLOBAL_MIN_FEE_MULTIPLIER:
                    raise ValueError(f"fee_multiplier_min must be >= {GLOBAL_MIN_FEE_MULTIPLIER} for peer {peer_id[:12]}...")
                if mult_min > GLOBAL_MAX_FEE_MULTIPLIER:
                    raise ValueError(f"fee_multiplier_min must be <= {GLOBAL_MAX_FEE_MULTIPLIER} for peer {peer_id[:12]}...")

            mult_max = update.get('fee_multiplier_max', existing.fee_multiplier_max)
            if mult_max is not None:
                if not isinstance(mult_max, (int, float)) or mult_max < GLOBAL_MIN_FEE_MULTIPLIER:
                    raise ValueError(f"fee_multiplier_max must be >= {GLOBAL_MIN_FEE_MULTIPLIER} for peer {peer_id[:12]}...")
                if mult_max > GLOBAL_MAX_FEE_MULTIPLIER:
                    raise ValueError(f"fee_multiplier_max must be <= {GLOBAL_MAX_FEE_MULTIPLIER} for peer {peer_id[:12]}...")

            # L-R5-8 FIX: Cross-validate multiplier min/max in batch path too
            if mult_min is not None and mult_max is not None:
                if mult_min > mult_max:
                    raise ValueError(
                        f"fee_multiplier_min ({mult_min}) cannot exceed "
                        f"fee_multiplier_max ({mult_max}) for peer {peer_id[:12]}..."
                    )

            # Process expiry
            expires_at = existing.expires_at
            expires_in_hours = update.get('expires_in_hours')
            if expires_in_hours is not None:
                if expires_in_hours <= 0:
                    expires_at = None
                else:
                    max_hours = MAX_POLICY_EXPIRY_DAYS * 24
                    if expires_in_hours > max_hours:
                        raise ValueError(f"expires_in_hours exceeds {max_hours} for peer {peer_id[:12]}...")
                    expires_at = now + (expires_in_hours * 3600)

            validated.append((
                peer_id, new_strategy, new_rebalance_mode, fee_ppm,
                tags, mult_min, mult_max, expires_at
            ))

        # M5 FIX + M-R6-2 FIX: Check rate limits after full validation passes.
        # First do a read-only check for ALL peers (no counter mutation).
        # Only record timestamps after all peers pass, preventing counter
        # pollution when a later peer in the batch fails the rate limit.
        now = int(time.time())
        window_start = now - 60
        with self._cache_lock:
            for peer_id, *_ in validated:
                timestamps = self._change_timestamps.get(peer_id, [])
                active = [ts for ts in timestamps if ts > window_start]
                if len(active) >= MAX_POLICY_CHANGES_PER_MINUTE:
                    raise RuntimeError(
                        f"Rate limited: max {MAX_POLICY_CHANGES_PER_MINUTE} changes/minute for {peer_id[:12]}..."
                    )
            # All passed - now record timestamps atomically
            for peer_id, *_ in validated:
                timestamps = self._change_timestamps.get(peer_id, [])
                timestamps = [ts for ts in timestamps if ts > window_start]
                timestamps.append(now)
                self._change_timestamps[peer_id] = timestamps

        # Execute batch insert atomically
        self.database.upsert_policies_batch([
            (peer_id, strategy.value, mode.value, fee_ppm, json.dumps(tags), now,
             mult_min, mult_max, expires_at)
            for peer_id, strategy, mode, fee_ppm, tags, mult_min, mult_max, expires_at in validated
        ])

        # Build results and update cache
        for peer_id, strategy, mode, fee_ppm, tags, mult_min, mult_max, expires_at in validated:
            policy = PeerPolicy(
                peer_id=peer_id,
                strategy=strategy,
                rebalance_mode=mode,
                fee_ppm_target=fee_ppm,
                tags=tags,
                updated_at=now,
                fee_multiplier_min=mult_min,
                fee_multiplier_max=mult_max,
                expires_at=expires_at
            )
            self._update_cache(peer_id, policy)
            results.append(policy)

            # Notify callbacks (batch mode - consider rate limiting)
            self._notify_change(peer_id, policy)

        self.plugin.log(f"PolicyManager: Batch updated {len(results)} policies", level='info')
        return results

    def cleanup_expired_policies(self) -> int:
        """
        v2.0: Remove all expired policies from the database.

        Called periodically to clean up time-limited policies that have expired.

        Returns:
            Number of expired policies removed
        """
        if not ENABLE_AUTO_EXPIRY:
            return 0

        now = int(time.time())
        expired_peer_ids = self.database.delete_expired_policies(now)

        if not expired_peer_ids:
            return 0

        deleted_count = len(expired_peer_ids)

        # Update cache and notify subscribers so they switch to default strategy.
        # Only evict from cache if the policy is still expired (guards against
        # a concurrent set_policy() that inserted a fresh policy between the
        # DB DELETE and this cache update).
        for peer_id in expired_peer_ids:
            with self._cache_lock:
                cached = self._cache.get(peer_id)
                if cached and cached.is_expired():
                    del self._cache[peer_id]
            self._notify_change(peer_id, self.get_policy(peer_id))

        if deleted_count > 0:
            self.plugin.log(
                f"PolicyManager: Cleaned up {deleted_count} expired policies",
                level='info'
            )

        return deleted_count

    # =========================================================================
    # Legacy Compatibility (Deprecated)
    # =========================================================================

    def is_peer_ignored(self, peer_id: str) -> bool:
        """
        Legacy compatibility method.

        DEPRECATED: Use should_manage_fees() and should_rebalance() instead.

        Returns True if peer has PASSIVE strategy and DISABLED rebalancing.
        """
        policy = self.get_policy(peer_id)
        return (
            policy.strategy == FeeStrategy.PASSIVE and
            policy.rebalance_mode == RebalanceMode.DISABLED
        )
