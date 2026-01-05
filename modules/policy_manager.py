"""
Policy Manager module for cl-revenue-ops

Implements the Policy-Driven architecture for managing peer behavior.
Replaces the legacy ignored_peers system with a centralized, declarative
policy system that supports:
- Fee strategies: dynamic, static, hive, passive
- Rebalance modes: enabled, disabled, source_only, sink_only
- Tags for grouping and filtering peers

Phase 9 Preparation: Provides API hooks for cl-hive integration.
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database


class FeeStrategy(Enum):
    """Fee control strategy for a peer."""
    DYNAMIC = "dynamic"   # Hill Climbing + Scarcity (Default)
    STATIC = "static"     # Fixed fee (User Override)
    HIVE = "hive"         # 0-Fee / Low Fee (Fleet Member)
    PASSIVE = "passive"   # Do nothing (allow CLBOSS/Manual control)


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
    """
    peer_id: str
    strategy: FeeStrategy = FeeStrategy.DYNAMIC
    rebalance_mode: RebalanceMode = RebalanceMode.ENABLED
    fee_ppm_target: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    updated_at: int = 0
    
    def has_tag(self, tag: str) -> bool:
        """Check if this policy has a specific tag."""
        return tag in self.tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "peer_id": self.peer_id,
            "strategy": self.strategy.value,
            "rebalance_mode": self.rebalance_mode.value,
            "fee_ppm_target": self.fee_ppm_target,
            "tags": self.tags,
            "updated_at": self.updated_at
        }


# Regex for validating 66-character hex peer IDs
PEER_ID_PATTERN = re.compile(r'^[0-9a-fA-F]{66}$')


class PolicyManager:
    """
    Centralized policy management for peer behavior.
    
    Replaces the legacy ignored_peers system with a declarative policy
    architecture. All peer-specific behavior decisions flow through this
    manager to ensure consistency.
    
    Thread Safety:
        Uses the Database's thread-local connection pattern.
        Policy reads are idempotent; writes are atomic via SQLite.
    """
    
    # Default policy for peers without explicit configuration
    DEFAULT_POLICY = PeerPolicy(
        peer_id="",
        strategy=FeeStrategy.DYNAMIC,
        rebalance_mode=RebalanceMode.ENABLED,
        fee_ppm_target=None,
        tags=[],
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
        
        # In-memory cache for hot-path performance
        # Invalidated on any policy write
        self._cache: Dict[str, PeerPolicy] = {}
        self._cache_valid = False
    
    def _validate_peer_id(self, peer_id: str) -> None:
        """
        Validate peer ID format.
        
        Raises:
            ValueError: If peer_id is not a valid 66-char hex string
        """
        if not peer_id or not PEER_ID_PATTERN.match(peer_id):
            raise ValueError(
                f"Invalid peer_id: must be 66-character hex string, got '{peer_id[:20]}...'"
                if len(peer_id) > 20 else f"Invalid peer_id: '{peer_id}'"
            )
    
    def _invalidate_cache(self) -> None:
        """Invalidate the in-memory policy cache."""
        self._cache.clear()
        self._cache_valid = False
    
    def _load_cache(self) -> None:
        """Load all policies into cache from database."""
        if self._cache_valid:
            return
            
        conn = self.database._get_connection()
        rows = conn.execute(
            "SELECT * FROM peer_policies ORDER BY updated_at DESC"
        ).fetchall()
        
        self._cache.clear()
        for row in rows:
            policy = self._row_to_policy(row)
            self._cache[policy.peer_id] = policy
        
        self._cache_valid = True
    
    def _row_to_policy(self, row) -> PeerPolicy:
        """Convert a database row to a PeerPolicy object."""
        tags_json = row['tags'] or '[]'
        try:
            tags = json.loads(tags_json)
        except (json.JSONDecodeError, TypeError):
            tags = []
        
        return PeerPolicy(
            peer_id=row['peer_id'],
            strategy=FeeStrategy(row['strategy']),
            rebalance_mode=RebalanceMode(row['rebalance_mode']),
            fee_ppm_target=row['fee_ppm_target'],
            tags=tags if isinstance(tags, list) else [],
            updated_at=row['updated_at']
        )
    
    # =========================================================================
    # Core Policy Methods
    # =========================================================================
    
    def get_policy(self, peer_id: str) -> PeerPolicy:
        """
        Get the policy for a specific peer.
        
        If no explicit policy exists, returns the default policy
        (dynamic strategy, rebalancing enabled).
        
        Args:
            peer_id: 66-character hex public key
            
        Returns:
            PeerPolicy for the peer
        """
        # Check cache first
        self._load_cache()
        if peer_id in self._cache:
            return self._cache[peer_id]
        
        # Return default policy with this peer_id
        return PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
            rebalance_mode=RebalanceMode.ENABLED,
            fee_ppm_target=None,
            tags=[],
            updated_at=0
        )
    
    def set_policy(
        self,
        peer_id: str,
        strategy: Optional[str] = None,
        rebalance_mode: Optional[str] = None,
        fee_ppm_target: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> PeerPolicy:
        """
        Set or update the policy for a peer.
        
        Only provided fields are updated; others retain existing values
        or defaults for new peers.
        
        Args:
            peer_id: 66-character hex public key
            strategy: Fee strategy (dynamic, static, hive, passive)
            rebalance_mode: Rebalance mode (enabled, disabled, source_only, sink_only)
            fee_ppm_target: Target fee for static strategy
            tags: List of string tags (replaces existing tags)
            
        Returns:
            Updated PeerPolicy
            
        Raises:
            ValueError: If peer_id or enum values are invalid
        """
        self._validate_peer_id(peer_id)
        
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
        new_tags = [str(t) for t in new_tags]  # Ensure all strings
        
        now = int(time.time())
        
        # Persist to database
        conn = self.database._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO peer_policies 
                (peer_id, strategy, rebalance_mode, fee_ppm_target, tags, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            peer_id,
            new_strategy.value,
            new_rebalance_mode.value,
            new_fee_ppm,
            json.dumps(new_tags),
            now
        ))
        
        # Invalidate cache
        self._invalidate_cache()
        
        # Return new policy
        new_policy = PeerPolicy(
            peer_id=peer_id,
            strategy=new_strategy,
            rebalance_mode=new_rebalance_mode,
            fee_ppm_target=new_fee_ppm,
            tags=new_tags,
            updated_at=now
        )
        
        self.plugin.log(
            f"PolicyManager: Set policy for {peer_id[:12]}... -> "
            f"strategy={new_strategy.value}, rebalance={new_rebalance_mode.value}",
            level='info'
        )
        
        return new_policy
    
    def delete_policy(self, peer_id: str) -> bool:
        """
        Delete the policy for a peer, reverting to defaults.
        
        Args:
            peer_id: 66-character hex public key
            
        Returns:
            True if a policy was deleted, False if none existed
        """
        self._validate_peer_id(peer_id)
        
        conn = self.database._get_connection()
        cursor = conn.execute(
            "DELETE FROM peer_policies WHERE peer_id = ?",
            (peer_id,)
        )
        
        self._invalidate_cache()
        
        deleted = cursor.rowcount > 0
        if deleted:
            self.plugin.log(
                f"PolicyManager: Deleted policy for {peer_id[:12]}..., reverting to defaults",
                level='info'
            )
        
        return deleted
    
    def get_all_policies(self) -> List[PeerPolicy]:
        """
        Get all explicitly configured policies.
        
        Returns:
            List of PeerPolicy objects
        """
        self._load_cache()
        return list(self._cache.values())
    
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
            List of PeerPolicy objects with the tag
        """
        self._load_cache()
        return [p for p in self._cache.values() if p.has_tag(tag)]
    
    def get_peers_by_strategy(self, strategy: FeeStrategy) -> List[PeerPolicy]:
        """
        Get all peers with a specific fee strategy.
        
        Args:
            strategy: FeeStrategy enum value
            
        Returns:
            List of PeerPolicy objects with that strategy
        """
        self._load_cache()
        return [p for p in self._cache.values() if p.strategy == strategy]
    
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
    
    def is_hive_peer(self, peer_id: str) -> bool:
        """
        Check if this peer is a Hive fleet member.
        
        Args:
            peer_id: 66-character hex public key
            
        Returns:
            True if peer has HIVE strategy
        """
        policy = self.get_policy(peer_id)
        return policy.strategy == FeeStrategy.HIVE
    
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
