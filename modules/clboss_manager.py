"""
Clboss Manager module for cl-revenue-ops

Handles interaction with clboss for the Manager-Override pattern.
Before making fee or liquidity changes, we must unmanage the peer
from clboss to prevent conflicts.

MANAGER-OVERRIDE PATTERN:
-------------------------
clboss is excellent at:
- Channel creation and peer selection
- Monitoring node reliability
- Basic fee adjustments

But we want to override it for:
- Fee setting (we use Hill Climbing with revenue maximization)
- Rebalancing decisions (we use EV-based profit analysis)

The pattern:
1. Check if clboss is managing the peer
2. Call clboss-unmanage for the specific tag (lnfee, rebalance)
3. Make our changes
4. Track what we've unmanaged so we can re-enable if needed
"""

import time
import threading
from typing import Dict, Any, Optional, List
from pyln.client import Plugin, RpcError


# Tags used by clboss that we may want to override
class ClbossTags:
    """
    Clboss management tags.
    
    These correspond to different aspects clboss manages:
    - lnfee: Fee management
    - balance: Liquidity rebalancing (NOT "rebalance"!)
    - open: Channel opening
    - close: Channel closing
    
    Note: clboss-unmanage expects tags as comma-separated string: "lnfee,balance"
    """
    FEE = "lnfee"
    BALANCE = "balance"  # This is the correct tag for rebalancing!
    OPEN = "open"
    CLOSE = "close"
    
    # For fee + rebalance control (most common use case)
    # Pass as list or comma-separated string to manager methods
    FEE_AND_BALANCE = ["lnfee", "balance"]
    ALL = ["lnfee", "balance", "open", "close"]


class ClbossManager:
    """
    Manager for clboss interaction.
    
    Provides methods to safely unmanage peers from clboss before
    making changes, and to re-enable management when desired.
    """
    
    def __init__(self, plugin: Plugin, config):
        """
        Initialize the clboss manager.
        
        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
        """
        self.plugin = plugin
        self.config = config
        self._clboss_available: Optional[bool] = None
        self._clboss_check_time: float = 0
        # CB-1: Lock protects _clboss_available and _clboss_check_time from
        # concurrent reads/writes across timer threads and RPC handlers
        self._clboss_lock = threading.Lock()
        # Retry checking for clboss every 5 minutes if previously unavailable
        self._clboss_retry_interval: float = 300

    def is_clboss_available(self) -> bool:
        """
        Check if clboss is available and running.

        Returns:
            True if clboss commands are available

        Note: If clboss was not available on first check, will retry
        after _clboss_retry_interval seconds to handle startup race conditions.
        """
        if not self.config.clboss_enabled:
            return False

        with self._clboss_lock:
            # If we have a cached result and it's True, use it
            # If cached result is False, retry after the interval
            if self._clboss_available is not None:
                if self._clboss_available:
                    return True
                # Previously unavailable - check if we should retry
                if time.time() - self._clboss_check_time < self._clboss_retry_interval:
                    return False
                # Time to retry
                self.plugin.log("Retrying clboss availability check...")

        # RPC call outside lock to avoid holding lock during I/O
        try:
            result = self.plugin.rpc.call("clboss-status")
            with self._clboss_lock:
                self._clboss_available = True
                self._clboss_check_time = time.time()
            self.plugin.log("clboss detected and available")
            return True
        except RpcError as e:
            if "Unknown command" in str(e) or "not found" in str(e).lower():
                with self._clboss_lock:
                    self._clboss_available = False
                    self._clboss_check_time = time.time()
                self.plugin.log("clboss not available - commands will be skipped (will retry in 5 min)")
                return False
            # Other RPC errors - clboss may be present but had a transient issue.
            # Don't cache True; leave uncached so we retry next time.
            self.plugin.log(f"clboss RPC error (will retry): {e}", level='debug')
            with self._clboss_lock:
                return self._clboss_available if self._clboss_available is not None else False
        except Exception as e:
            self.plugin.log(f"Error checking clboss availability: {e}", level='warn')
            with self._clboss_lock:
                self._clboss_available = False
                self._clboss_check_time = time.time()
            return False
    
    def reset_availability_cache(self) -> bool:
        """
        Reset the clboss availability cache to force a fresh check.

        Returns:
            The result of the fresh availability check
        """
        self._clboss_available = None
        self._clboss_check_time = 0
        return self.is_clboss_available()

    def unmanage_for_fee(self, peer_id: str) -> Dict[str, Any]:
        """
        Unmanage a peer from clboss fee management.
        
        This MUST be called before setting fees on a channel to prevent
        clboss from reverting our changes.
        
        Args:
            peer_id: The node ID of the peer
            
        Returns:
            Result dict with status and details
        """
        return self.unmanage(peer_id, ClbossTags.FEE_AND_BALANCE)
    
    def unmanage_for_rebalance(self, peer_id: str) -> Dict[str, Any]:
        """
        Unmanage a peer from clboss rebalancing.
        
        Args:
            peer_id: The node ID of the peer
            
        Returns:
            Result dict with status and details
        """
        return self.unmanage(peer_id, ClbossTags.FEE_AND_BALANCE)
    
    def unmanage(self, peer_id: str, tag: str) -> Dict[str, Any]:
        """
        Unmanage a peer from clboss for a specific tag.
        
        This is the core override method. It tells clboss to stop
        managing this peer for the specified aspect (fee/rebalance).
        
        Args:
            peer_id: The node ID of the peer
            tag: The management tag to disable (e.g., 'lnfee')
            
        Returns:
            Result dict with status and details
        """
        result = {
            "peer_id": peer_id,
            "tag": tag,
            "action": "unmanage",
            "success": False,
            "skipped": False,
            "message": ""
        }
        
        # Check if clboss integration is enabled
        if not self.config.clboss_enabled:
            result["skipped"] = True
            result["message"] = "clboss integration disabled in config"
            return result
        
        # Check if clboss is available
        if not self.is_clboss_available():
            result["skipped"] = True
            result["message"] = "clboss not available"
            return result
        
        # Check if we've already unmanaged this peer/tag
        try:
            # Check if already unmanaged (via plugin's database reference)
            # Note: We'll need to access the database through plugin context
            
            if self.config.dry_run:
                result["success"] = True
                result["message"] = f"[DRY RUN] Would unmanage {peer_id} for {tag}"
                self.plugin.log(result["message"])
                return result
            
            # Call clboss-unmanage with positional args: nodeid tags
            # Normalize tag: if list, join with commas for clboss-unmanage
            tags_str = ",".join(tag) if isinstance(tag, list) else tag
            
            try:
                unmanage_result = self.plugin.rpc.call(
                    "clboss-unmanage",
                    [peer_id, tags_str]  # positional: nodeid, tags
                )
                
                result["success"] = True
                result["message"] = f"Successfully unmanaged {peer_id} for {tag}"
                result["clboss_response"] = unmanage_result
                
                self.plugin.log(f"Unmanaged peer {peer_id[:16]}... from clboss {tag} management")
                
            except RpcError as e:
                error_str = str(e)
                error_lower = error_str.lower()
                
                # Handle clboss versions without clboss-unmanage
                if "unknown command" in error_lower and "clboss-unmanage" in error_lower:
                    result["success"] = False
                    result["skipped"] = True
                    result["message"] = "clboss-unmanage not available"
                    self.plugin.log(
                        "clboss-unmanage not available - skipping override",
                        level='debug'
                    )
                    return result
                
                # Handle case where peer is not managed by clboss
                if "not managed" in error_lower or "already unmanaged" in error_lower:
                    result["success"] = True
                    result["message"] = f"Peer {peer_id} already not managed by clboss for {tag}"
                else:
                    result["success"] = False
                    result["message"] = f"clboss-unmanage failed: {error_str}"
                    self.plugin.log(f"Failed to unmanage {peer_id}: {error_str}", level='warn')
                    
        except Exception as e:
            result["success"] = False
            result["message"] = f"Unexpected error: {str(e)}"
            self.plugin.log(f"Error in unmanage: {e}", level='error')
        
        return result
    
    def remanage(self, peer_id: str, tag=None, database=None) -> Dict[str, Any]:
        """
        Re-enable clboss management for a peer.

        Use this to hand control back to clboss for a peer we previously
        unmanaged. If no tag is specified, re-enable all tags.

        Args:
            peer_id: The node ID of the peer
            tag: Optional specific tag to re-enable (None = all tags)
            database: Optional database instance for cleanup of unmanage records

        Returns:
            Result dict with status and details
        """
        result = {
            "peer_id": peer_id,
            "tag": tag or "all",
            "action": "remanage",
            "success": False,
            "message": ""
        }

        if not self.config.clboss_enabled or not self.is_clboss_available():
            result["message"] = "clboss not available"
            return result

        if self.config.dry_run:
            result["success"] = True
            result["message"] = f"[DRY RUN] Would remanage {peer_id} for {tag or 'all tags'}"
            self.plugin.log(result["message"])
            return result
        try:
            # P0-2 FIX: Use clboss-manage (not clboss-unmanage with empty string)
            # for all code paths, including tag=None (re-enable all).
            if tag is None:
                tags_to_process = list(ClbossTags.ALL)
            elif isinstance(tag, list):
                tags_to_process = tag
            else:
                tags_to_process = [tag]

            succeeded_tags = []
            failed_tags = []
            for t in tags_to_process:
                try:
                    self.plugin.rpc.call(
                        "clboss-manage",
                        [peer_id, t]  # positional: nodeid, tags
                    )
                    succeeded_tags.append(t)
                except RpcError as e:
                    failed_tags.append(t)
                    self.plugin.log(f"Could not remanage {t} for {peer_id}: {e}", level='debug')

            result["success"] = len(succeeded_tags) > 0
            if failed_tags and succeeded_tags:
                result["message"] = f"Partially remanaged {peer_id}: {len(succeeded_tags)} ok, {len(failed_tags)} failed"
            elif failed_tags:
                result["message"] = f"Failed to remanage any tags for {peer_id}"
            else:
                result["message"] = f"Re-enabled clboss management for {peer_id}"
            self.plugin.log(f"Remanaged peer {peer_id[:16]}...: {len(succeeded_tags)} ok, {len(failed_tags)} failed")

            # P0-1 FIX: Clean up DB unmanage records on success
            if result["success"] and database:
                try:
                    if tag is None:
                        database.remove_unmanage(peer_id)  # remove all tags
                    else:
                        database.remove_unmanage(peer_id, tag)
                except Exception as db_err:
                    self.plugin.log(f"Failed to clean up unmanage DB record for {peer_id}: {db_err}", level='warn')

        except Exception as e:
            result["success"] = False
            result["message"] = f"Error: {str(e)}"
            self.plugin.log(f"Error in remanage: {e}", level='error')

        return result

    def reconcile_unmanaged(self, database) -> Dict[str, Any]:
        """
        Startup reconciliation: re-manage any peers left orphaned by a crash.

        Reads all DB unmanage records and calls clboss-manage for each,
        then cleans up the DB records. This ensures peers don't stay
        permanently unmanaged after a plugin crash/restart.

        Args:
            database: Database instance with unmanage tracking methods

        Returns:
            Summary of reconciliation actions
        """
        if not self.config.clboss_enabled or not self.is_clboss_available():
            return {"skipped": True, "reason": "clboss not available"}

        try:
            orphaned = database.get_all_unmanaged()
        except Exception as e:
            return {"skipped": True, "reason": f"DB error: {e}"}

        if not orphaned:
            return {"skipped": False, "orphaned_count": 0, "remanaged": 0}

        remanaged = 0
        failed = 0
        for record in orphaned:
            peer_id = record.get("peer_id", "")
            tag = record.get("tag", "")
            if not peer_id:
                continue
            try:
                tags = tag.split(",") if tag else list(ClbossTags.ALL)
                for t in tags:
                    try:
                        self.plugin.rpc.call("clboss-manage", [peer_id, t])
                    except RpcError:
                        pass  # best-effort
                database.remove_unmanage(peer_id, tag if tag else None)
                remanaged += 1
            except Exception as e:
                failed += 1
                self.plugin.log(f"Failed to reconcile unmanage for {peer_id}: {e}", level='warn')

        self.plugin.log(f"Clboss reconciliation: {remanaged} remanaged, {failed} failed out of {len(orphaned)} orphaned")
        return {"skipped": False, "orphaned_count": len(orphaned), "remanaged": remanaged, "failed": failed}
    
    def get_unmanaged_status(self) -> Dict[str, Any]:
        """
        Get the current status of clboss management overrides.
        
        Returns:
            Dict with clboss status and list of unmanaged peers
        """
        status = {
            "clboss_enabled": self.config.clboss_enabled,
            "clboss_available": self.is_clboss_available(),
            "unmanaged_peers": []
        }
        
        if not self.config.clboss_enabled:
            status["message"] = "clboss integration is disabled"
            return status
        
        if not self.is_clboss_available():
            status["message"] = "clboss is not available"
            return status
        
        try:
            # Try to get clboss status
            clboss_status = self.plugin.rpc.call("clboss-status")
            status["clboss_status"] = clboss_status
            
            # Get unmanaged list if available
            try:
                unmanaged = self.plugin.rpc.call("clboss-unmanaged-list")
                status["unmanaged_peers"] = unmanaged.get("unmanaged", [])
            except RpcError:
                # Command might not exist in older versions
                pass
                
        except Exception as e:
            status["message"] = f"Could not get clboss status: {e}"
        
        return status
    
    def is_peer_managed(self, peer_id: str, tag: str) -> bool:
        """
        Check if a peer is currently managed by clboss for a specific tag.

        Args:
            peer_id: The node ID of the peer
            tag: The management tag to check

        Returns:
            True if clboss is managing this peer/tag, False otherwise
        """
        if not self.is_clboss_available():
            return False

        try:
            # Query clboss for unmanaged peers
            unmanaged = self.plugin.rpc.call("clboss-unmanaged-list")
            for entry in unmanaged.get("unmanaged", []):
                if entry.get("nodeid") == peer_id:
                    entry_tags = entry.get("tags", "")
                    if tag in (entry_tags.split(",") if entry_tags else []):
                        return False  # Peer is unmanaged for this tag
            return True  # Not in unmanaged list = managed
        except RpcError:
            # clboss-unmanaged-list not available, assume managed
            return True
        except Exception:
            return False
    
    def ensure_unmanaged_for_channel(self, channel_id: str, peer_id: str, 
                                     tag: str, database) -> bool:
        """
        Ensure a channel's peer is unmanaged before making changes.
        
        This is the main entry point for the Manager-Override pattern.
        It checks if we need to unmanage, and does so if necessary.
        
        Args:
            channel_id: The channel ID
            peer_id: The peer's node ID
            tag: The management tag (fee/rebalance)
            database: Database instance for tracking
            
        Returns:
            True if we can proceed with changes, False if blocked
        """
        # Always try to unmanage with clboss (it handles "already unmanaged" gracefully)
        # Don't rely solely on our DB - clboss state may have changed (restart, etc.)
        result = self.unmanage(peer_id, tag)
        
        if result["success"] or result["skipped"]:
            # Record that we unmanaged (if not skipped and not already in DB)
            if result["success"] and not result.get("skipped"):
                if not database.is_unmanaged(peer_id, tag):
                    database.record_unmanage(peer_id, tag)
            return True
        
        # Failed to unmanage - log and return False
        self.plugin.log(
            f"Could not unmanage {peer_id} for {tag}: {result['message']}", 
            level='warn'
        )
        return False
