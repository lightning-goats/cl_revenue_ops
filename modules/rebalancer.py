"""
EV-Based Rebalancer module for cl-revenue-ops

MODULE 3: EV-Based Rebalancing (Profit-Aware with Opportunity Cost)

This module implements Expected Value (EV) based rebalancing decisions.
Unlike clboss which often makes negative EV rebalances, this module only
triggers rebalances when the math shows positive expected profit.

Architecture Pattern: "Strategist, Manager, and Driver"
- STRATEGIST (EVRebalancer): Calculates EV, determines IF and HOW MUCH to rebalance
- MANAGER (JobManager): Manages lifecycle of background sling jobs
- DRIVER (Sling plugin): Actually executes the payments in the background

Phase 4: Async Job Queue
- Decouples decision-making from execution
- Allows concurrent rebalancing attempts
- Uses sling-job (background) instead of sling-once (blocking)
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from enum import Enum

from pyln.client import Plugin, RpcError

from .config import Config, ConfigSnapshot
from .database import Database
from .clboss_manager import ClbossManager, ClbossTags
from .policy_manager import PolicyManager, RebalanceMode, FeeStrategy

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer
    from .hive_bridge import HiveFeeIntelligenceBridge


class JobStatus(Enum):
    """Status of a sling background job."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


@dataclass
class RebalanceCandidate:
    """A candidate for rebalancing with multi-source support."""
    source_candidates: List[str]  # List of source SCIDs, sorted by score (best first)
    to_channel: str
    primary_source_peer_id: str  # Peer ID of the best (first) source candidate
    to_peer_id: str
    amount_sats: int
    amount_msat: int
    outbound_fee_ppm: int
    inbound_fee_ppm: int
    source_fee_ppm: int  # Fee PPM of the primary (best) source
    weighted_opp_cost_ppm: int  # Weighted opportunity cost of the primary source
    spread_ppm: int  # Spread based on primary source
    max_budget_sats: int
    max_budget_msat: int
    max_fee_ppm: int
    expected_profit_sats: int
    liquidity_ratio: float
    dest_flow_state: str
    dest_turnover_rate: float
    source_turnover_rate: float  # Turnover rate of the primary source
    
    # Backwards compatibility property
    @property
    def from_channel(self) -> str:
        """Returns the primary (best) source channel for backwards compatibility."""
        return self.source_candidates[0] if self.source_candidates else ""
    
    @property
    def from_peer_id(self) -> str:
        """Returns the primary source peer ID for backwards compatibility."""
        return self.primary_source_peer_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_candidates": self.source_candidates,
            "from_channel": self.from_channel,  # Primary source for backwards compat
            "to_channel": self.to_channel,
            "from_peer_id": self.primary_source_peer_id,
            "to_peer_id": self.to_peer_id,
            "amount_sats": self.amount_sats,
            "amount_msat": self.amount_msat,
            "outbound_fee_ppm": self.outbound_fee_ppm,
            "inbound_fee_ppm": self.inbound_fee_ppm,
            "source_fee_ppm": self.source_fee_ppm,
            "weighted_opp_cost_ppm": self.weighted_opp_cost_ppm,
            "spread_ppm": self.spread_ppm,
            "max_budget_sats": self.max_budget_sats,
            "max_budget_msat": self.max_budget_msat,
            "max_fee_ppm": self.max_fee_ppm,
            "expected_profit_sats": self.expected_profit_sats,
            "liquidity_ratio": round(self.liquidity_ratio, 4),
            "dest_flow_state": self.dest_flow_state,
            "dest_turnover_rate": round(self.dest_turnover_rate, 4),
            "source_turnover_rate": round(self.source_turnover_rate, 4),
            "num_source_candidates": len(self.source_candidates)
        }


@dataclass
class ActiveJob:
    """Tracks an active sling background job."""
    scid: str                      # Target channel SCID (colon format for sling)
    scid_normalized: str           # Original SCID format (for our tracking)
    source_candidates: List[str]   # List of source channel SCIDs (colon format)
    start_time: int                # Unix timestamp when job started
    candidate: RebalanceCandidate  # Original candidate data
    rebalance_id: int              # Database record ID
    target_amount_sats: int        # Total amount we want to rebalance
    initial_local_sats: int        # Local balance when job started
    max_fee_ppm: int               # Max fee rate for this job
    status: JobStatus = JobStatus.PENDING
    
    # Backwards compatibility property
    @property
    def from_scid(self) -> str:
        """Returns the primary (best) source SCID for backwards compatibility."""
        return self.source_candidates[0] if self.source_candidates else ""


class JobManager:
    """
    Manages the lifecycle of Sling background rebalancing jobs.

    Responsibilities:
    - Start new sling-job workers
    - Monitor job progress via sling-stats
    - Stop jobs on success, timeout, or error
    - Record results to database
    - Report outcomes to hive for fleet coordination (Phase 7)

    Key Design Decision:
    We use sling-job for TACTICAL rebalancing (one-off moves), not permanent
    pegging. As soon as any successful payment is detected or timeout is reached,
    we DELETE the job to prevent infinite spending.
    """

    # Default timeout: 2 hours (configurable)
    DEFAULT_JOB_TIMEOUT_SECONDS = 7200

    def __init__(self, plugin: Plugin, config: Config, database: Database,
                 hive_bridge: Optional["HiveFeeIntelligenceBridge"] = None):
        self.plugin = plugin
        self.config = config
        self.database = database
        self.hive_bridge = hive_bridge

        # Active jobs indexed by target channel SCID (normalized format)
        self._active_jobs: Dict[str, ActiveJob] = {}

        # Configurable settings
        self.job_timeout_seconds = getattr(config, 'sling_job_timeout_seconds',
                                           self.DEFAULT_JOB_TIMEOUT_SECONDS)
        self.max_concurrent_jobs = getattr(config, 'max_concurrent_jobs', 5)

        # Chunk size for sling rebalances (sats per attempt)
        self.chunk_size_sats = getattr(config, 'sling_chunk_size_sats', 500000)

        # Source reliability tracking
        self.source_failure_counts: Dict[str, float] = {}
        self.last_decay_time = time.time()

    def _report_outcome_to_hive(self, job: ActiveJob, success: bool, cost_sats: int,
                                 amount_transferred: int = 0) -> None:
        """
        Report rebalance outcome to hive for fleet coordination.

        This enables:
        - Circular flow detection (A→B→C→A wastes fees)
        - Better rebalance coordination across fleet members
        - Learning from successful/failed routes

        Args:
            job: The completed job
            success: Whether rebalance succeeded
            cost_sats: Fee cost of the rebalance
            amount_transferred: Amount successfully moved (0 if failed)
        """
        if not self.hive_bridge:
            return

        try:
            # Determine if this was routed via fleet (check candidate metadata)
            via_fleet = getattr(job.candidate, 'via_fleet', False) if job.candidate else False

            self.hive_bridge.report_rebalance_outcome(
                from_channel=job.from_scid,
                to_channel=job.scid,
                amount_sats=amount_transferred if success else job.target_amount_sats,
                cost_sats=cost_sats,
                success=success,
                via_fleet=via_fleet
            )

            self.plugin.log(
                f"Reported rebalance outcome to hive: {job.scid} "
                f"success={success} cost={cost_sats}sats",
                level='debug'
            )
        except Exception as e:
            # Non-fatal - don't fail the job handling for hive reporting
            self.plugin.log(
                f"Failed to report rebalance outcome to hive: {e}",
                level='debug'
            )

    def prune_stale_source_failures(self, active_channel_ids: set) -> int:
        """
        Remove in-memory failure counts for channels that no longer exist.
        
        This prevents memory bloat from closed channels over time.
        
        Args:
            active_channel_ids: Set of currently active channel IDs
            
        Returns:
            Number of stale entries pruned
        """
        pruned = 0
        stale_keys = [k for k in self.source_failure_counts.keys() if k not in active_channel_ids]
        for key in stale_keys:
            del self.source_failure_counts[key]
            pruned += 1
        
        if pruned > 0:
            self.plugin.log(
                f"GC: Pruned {pruned} stale source failure counts from closed channels",
                level='debug'
            )
        
        return pruned
    
    @property
    def active_job_count(self) -> int:
        """Returns the number of currently active jobs."""
        return len(self._active_jobs)
    
    @property
    def active_channels(self) -> List[str]:
        """Returns list of channel SCIDs with active jobs."""
        return list(self._active_jobs.keys())
    
    def has_active_job(self, channel_id: str) -> bool:
        """Check if a channel has an active rebalance job."""
        normalized = self._normalize_scid(channel_id)
        return normalized in self._active_jobs
    
    def slots_available(self) -> int:
        """Returns number of available job slots."""
        return max(0, self.max_concurrent_jobs - self.active_job_count)
    
    def _normalize_scid(self, scid: str) -> str:
        """Normalize SCID to consistent format (with 'x' separators)."""
        return scid.replace(':', 'x')
    
    def _to_sling_scid(self, scid: str) -> str:
        """Normalize SCID to sling's expected 'x' separator format."""
        # Sling expects format like 930866x2599x2 (with 'x' separators)
        return scid.replace(':', 'x')
    
    def _get_channel_local_balance(self, channel_id: str) -> int:
        """Get current local balance of a channel in sats."""
        try:
            listfunds = self.plugin.rpc.listfunds()
            normalized = self._normalize_scid(channel_id)
            
            for channel in listfunds.get("channels", []):
                scid = channel.get("short_channel_id", "")
                if self._normalize_scid(scid) == normalized:
                    our_amount_msat = channel.get("our_amount_msat", 0)
                    if isinstance(our_amount_msat, str):
                        our_amount_msat = int(our_amount_msat.replace("msat", ""))
                    return our_amount_msat // 1000
        except Exception as e:
            self.plugin.log(f"Error getting channel balance: {e}", level='debug')
        return 0

    # NOTE: _get_channel_age_days removed - duplicate of EVRebalancer method and was never called

    def start_job(self, candidate: RebalanceCandidate, rebalance_id: int) -> Dict[str, Any]:
        """
        Start a new sling-job for the given candidate with multi-source support.
        
        sling-job creates a persistent background worker that will keep
        attempting to rebalance until stopped or target is reached.
        Passes ALL profitable source candidates to Sling so it can handle
        pathfinding failover automatically.
        
        Args:
            candidate: The rebalance candidate with all parameters
            rebalance_id: Database record ID for this rebalance attempt
            
        Returns:
            Dict with 'success' bool and 'message' or 'error'
        """
        normalized_scid = self._normalize_scid(candidate.to_channel)
        
        # Check if job already exists
        if normalized_scid in self._active_jobs:
            return {"success": False, "error": "Job already exists for this channel"}
        
        # Check slot availability
        if self.active_job_count >= self.max_concurrent_jobs:
            return {"success": False, "error": "No job slots available"}
        
        # Convert SCIDs to sling format (x-separated, e.g., 930866x2599x2)
        to_scid = self._to_sling_scid(candidate.to_channel)
        
        # Convert all source candidates to sling format
        source_scids_sling = [self._to_sling_scid(scid) for scid in candidate.source_candidates]
        
        # Get initial balance for progress tracking
        initial_balance = self._get_channel_local_balance(candidate.to_channel)
        
        # Calculate chunk size (amount per rebalance attempt)
        chunk_size = min(candidate.amount_sats, self.chunk_size_sats)

        # ZERO-TOLERANCE: Enforce sats-budget-derived fee cap on the execution unit (chunk).
        budget_ppm = (candidate.max_budget_msat * 1_000_000) // (chunk_size * 1000) if chunk_size > 0 else 0
        maxppm = max(1, min(candidate.max_fee_ppm, budget_ppm)) if budget_ppm > 0 else 0
        if maxppm <= 0:
            return {"success": False, "error": "Budget too small to allow any routing fee (maxppm=0)"}
        
        try:
            primary_source = source_scids_sling[0] if source_scids_sling else "none"

            # =================================================================
            # PHASE 6: Flow-Aware Target Selection
            # =================================================================
            # Different channel types want different balance targets:
            # - SINK channels: Want more inbound capacity (lower target)
            # - SOURCE channels: Want more outbound capacity (higher target)
            # - BALANCED: Neutral 50/50
            # =================================================================
            flow_state = candidate.dest_flow_state if hasattr(candidate, 'dest_flow_state') else "balanced"
            if flow_state == "sink":
                target = self.config.sling_target_sink
            elif flow_state == "source":
                target = self.config.sling_target_source
            else:
                target = self.config.sling_target_balanced

            self.plugin.log(
                f"Starting sling-job: {to_scid} <- [{len(source_scids_sling)} candidates], "
                f"primary={primary_source}, amount={chunk_size}, "
                f"maxppm={maxppm}, maxhops={self.config.sling_max_hops}, "
                f"target={target} (flow={flow_state}), budget_sats={candidate.max_budget_sats}"
            )

            # =================================================================
            # PHASE 6: Enhanced Sling Parameters
            # =================================================================
            # - maxhops: Shorter routes are faster and more reliable
            # - target: Flow-aware balance target
            # - outppm: Fallback for source discovery when candidates list fails
            # =================================================================
            job_params = {
                "scid": to_scid,
                "direction": "pull",
                "amount": chunk_size,
                "maxppm": maxppm,
                "maxhops": self.config.sling_max_hops,
                "target": target,
            }

            # Add candidates if we have them
            if source_scids_sling:
                job_params["candidates"] = source_scids_sling

            # Add outppm as fallback source discovery (if configured and no candidates)
            if self.config.sling_outppm_fallback > 0:
                if not source_scids_sling:
                    # No candidates - use outppm for discovery
                    job_params["outppm"] = self.config.sling_outppm_fallback
                    self.plugin.log(
                        f"No candidates for {to_scid}, using outppm={self.config.sling_outppm_fallback} fallback",
                        level='info'
                    )
                else:
                    # Have candidates but also add outppm as backup
                    job_params["outppm"] = self.config.sling_outppm_fallback

            self.plugin.rpc.call("sling-job", job_params)
            
            # Start the job (sling-job only creates it, sling-go starts it)
            try:
                self.plugin.rpc.call("sling-go", {"scid": to_scid})
            except RpcError as e:
                # sling-go might fail if job auto-started, that's OK
                if "already running" not in str(e).lower():
                    self.plugin.log(f"sling-go warning: {e}", level='debug')
            
            # Track the job with all source candidates
            job = ActiveJob(
                scid=to_scid,
                scid_normalized=normalized_scid,
                source_candidates=source_scids_sling,
                start_time=int(time.time()),
                candidate=candidate,
                rebalance_id=rebalance_id,
                target_amount_sats=candidate.amount_sats,
                initial_local_sats=initial_balance,
                max_fee_ppm=maxppm,  # Use enforced budget-derived maxppm
                status=JobStatus.RUNNING
            )
            self._active_jobs[normalized_scid] = job
            
            self.plugin.log(
                f"Sling job started for {to_scid}, tracking as {normalized_scid} "
                f"with {len(source_scids_sling)} source candidates"
            )
            
            return {
                "success": True, 
                "message": f"Job started for {to_scid} with {len(source_scids_sling)} source candidates"
            }
            
        except RpcError as e:
            error_msg = str(e)
            self.plugin.log(f"Failed to start sling-job: {error_msg}", level='warn')
            return {"success": False, "error": f"Sling RPC error: {error_msg}"}
        except Exception as e:
            self.plugin.log(f"Error starting sling-job: {e}", level='error')
            return {"success": False, "error": str(e)}
    
    def stop_job(self, channel_id: str, reason: str = "manual") -> bool:
        """
        Stop and delete a sling job.
        
        Args:
            channel_id: Channel SCID (any format)
            reason: Why the job is being stopped (for logging)
            
        Returns:
            True if job was stopped, False if not found or error
        """
        normalized = self._normalize_scid(channel_id)
        job = self._active_jobs.get(normalized)
        
        if not job:
            return False
        
        try:
            # First stop the job gracefully
            try:
                self.plugin.rpc.call("sling-stop", {"scid": job.scid})
            except RpcError:
                pass  # May already be stopped
            
            # Then delete it to prevent restart
            try:
                self.plugin.rpc.call("sling-deletejob", {
                    "job": job.scid,
                    "delete_stats": False  # Keep stats for analysis
                })
            except RpcError as e:
                self.plugin.log(f"sling-deletejob warning: {e}", level='debug')
            
            self.plugin.log(f"Stopped sling job {job.scid} (reason: {reason})")
            
        except Exception as e:
            self.plugin.log(f"Error stopping job {job.scid}: {e}", level='warn')
        
        # Remove from tracking regardless
        del self._active_jobs[normalized]
        return True
    
    def monitor_jobs(self) -> Dict[str, Any]:
        """
        Monitor all active jobs and handle completed/failed/timed-out ones.
        
        This should be called periodically (e.g., every rebalance interval).
        
        Returns:
            Summary dict with counts of various outcomes
        """
        summary = {
            "checked": 0,
            "completed": 0,
            "failed": 0,
            "timed_out": 0,
            "still_running": 0
        }
        
        # Get current time
        now = int(time.time())
        
        # Periodic decay of failure counts (every hour)
        if now - self.last_decay_time > 3600:
            for scid in list(self.source_failure_counts.keys()):
                self.source_failure_counts[scid] *= 0.5
                if self.source_failure_counts[scid] < 0.1:
                    del self.source_failure_counts[scid]
            self.last_decay_time = now
            
        if not self._active_jobs:
            return summary
        
        # Get sling stats for all jobs
        sling_stats = self._get_sling_stats()
        
        # Copy keys to avoid modifying dict during iteration
        job_scids = list(self._active_jobs.keys())
        
        for normalized_scid in job_scids:
            job = self._active_jobs.get(normalized_scid)
            if not job:
                continue
                
            summary["checked"] += 1
            
            # Check timeout first
            elapsed = now - job.start_time
            if elapsed > self.job_timeout_seconds:
                self._handle_job_timeout(job)
                summary["timed_out"] += 1
                continue
            
            # Check current channel balance for progress
            current_balance = self._get_channel_local_balance(job.scid_normalized)
            amount_transferred = current_balance - job.initial_local_sats
            
            # Get job-specific stats from sling
            job_stats = sling_stats.get(job.scid, {})

            # ZERO-TOLERANCE: Abort if the job is spending at/above its msat budget.
            fee_msat = job_stats.get("fee_total_msat", 0) or 0
            if not fee_msat:
                fee_sats = job_stats.get("fee_total_sats", 0) or 0
                fee_msat = fee_sats * 1000 if fee_sats else 0

            if fee_msat and job.candidate and fee_msat > job.candidate.max_budget_msat:
                self._handle_job_budget_exceeded(job, fee_msat, job_stats)
                summary["failed"] += 1
                continue
            
            # Check for success: any positive transfer means we've achieved something
            if amount_transferred > 0:
                self._handle_job_success(job, amount_transferred, job_stats)
                summary["completed"] += 1
                continue
            
            # Check for sling-reported errors
            if self._check_job_error(job, job_stats):
                self._handle_job_failure(job, job_stats)
                summary["failed"] += 1
                continue
            
            # Job still running
            summary["still_running"] += 1
            self.plugin.log(
                f"Job {job.scid} running: {elapsed}s elapsed, "
                f"transferred={amount_transferred} sats",
                level='debug'
            )
        
        return summary
    
    def _get_sling_stats(self) -> Dict[str, Dict[str, Any]]:
        """Query sling-stats for all jobs and return as dict keyed by SCID."""
        stats = {}
        try:
            # sling-stats with json=true returns structured data
            result = self.plugin.rpc.call("sling-stats", {"json": True})
            
            if isinstance(result, dict):
                # Result might be keyed by SCID or be a list
                if "jobs" in result:
                    for job in result["jobs"]:
                        scid = job.get("scid", "")
                        if scid:
                            stats[scid] = job
                else:
                    # Assume dict is already keyed by SCID
                    stats = result
            elif isinstance(result, list):
                for job in result:
                    scid = job.get("scid", "")
                    if scid:
                        stats[scid] = job
                        
        except RpcError as e:
            self.plugin.log(f"sling-stats error: {e}", level='debug')
        except Exception as e:
            self.plugin.log(f"Error getting sling stats: {e}", level='debug')
        
        return stats
    
    def _check_job_error(self, job: ActiveJob, stats: Dict[str, Any]) -> bool:
        """Check if sling reports an error state for this job."""
        # Check for explicit error status
        # Handle case where status might be a list (sling plugin inconsistency)
        status = stats.get("status", "")
        if isinstance(status, list):
            status = status[0] if status else ""
        status = str(status).lower()
        if status in ("error", "failed", "stopped"):
            return True
        
        # Check for high consecutive failure count
        consecutive_failures = stats.get("consecutive_failures", 0)
        if consecutive_failures >= 10:
            return True
        
        return False
    
    def _handle_job_success(self, job: ActiveJob, amount_transferred: int, 
                            stats: Dict[str, Any]) -> None:
        """Handle a successfully completed job."""
        # Calculate actual fee paid (from sling stats if available)
        fee_sats = stats.get("fee_total_sats", 0)
        if not fee_sats:
            fee_msat = stats.get("fee_total_msat", 0)
            fee_sats = fee_msat // 1000 if fee_msat else 0
        
        # Estimate fee from amount if sling doesn't report it
        if fee_sats == 0 and amount_transferred > 0:
            # Use a conservative estimate based on max_fee_ppm
            fee_sats = (amount_transferred * job.max_fee_ppm) // 1_000_000
        
        # Calculate actual profit
        # The expected_profit was calculated with max_budget_sats as the assumed cost.
        # Actual profit = expected_profit + (budgeted_cost - actual_cost)
        # If we paid less than budgeted, profit increases; if more, it decreases.
        expected_profit = job.candidate.expected_profit_sats
        budgeted_fee = job.candidate.max_budget_sats
        actual_profit = expected_profit + (budgeted_fee - fee_sats)
        
        self.plugin.log(
            f"Rebalance SUCCESS: {job.scid} filled with {amount_transferred} sats. "
            f"Fee: {fee_sats} sats, Profit: {actual_profit} sats"
        )
        
        # Update database
        self.database.update_rebalance_result(
            job.rebalance_id, 
            'success', 
            fee_sats, 
            actual_profit
        )
        self.database.reset_failure_count(job.scid_normalized)
        
        # Record cost in rebalance_costs for lifetime accounting (revenue-history)
        # Uses rebalance_id as part of idempotency: each job has a unique rebalance_id,
        # and _handle_job_success is only called once per job lifecycle.
        if fee_sats > 0:
            self.database.record_rebalance_cost(
                channel_id=job.scid_normalized,
                peer_id=job.candidate.to_peer_id,
                cost_sats=fee_sats,
                amount_sats=amount_transferred,
                timestamp=int(time.time())
            )
        
        # RELIABILITY: Reset failure count for the source channel since it delivered
        if job.candidate and job.candidate.source_candidates:
            primary_source = job.candidate.source_candidates[0]
            if primary_source in self.source_failure_counts:
                # Significant reduction (rewarding success)
                self.source_failure_counts[primary_source] = 0.0

        # Mark budget reservation as spent (CRITICAL-01 fix)
        self.database.mark_budget_spent(job.rebalance_id, fee_sats)

        # Report outcome to hive for fleet coordination (Phase 7)
        self._report_outcome_to_hive(job, success=True, cost_sats=fee_sats,
                                     amount_transferred=amount_transferred)

        # Stop the job
        self.stop_job(job.scid_normalized, reason="success")

    def _handle_job_failure(self, job: ActiveJob, stats: Dict[str, Any]) -> None:
        """Handle a failed job."""
        error_msg = stats.get("last_error", "Unknown error from sling")
        # sling is the only supported backend; hide legacy wording if it appears
        if isinstance(error_msg, str) and "method: circular" in error_msg:
            error_msg = error_msg.replace("method: circular", "method: sling")
        
        self.plugin.log(
            f"Rebalance FAILED: {job.scid} - {error_msg}",
            level='warn'
        )
        
        # Update database
        self.database.update_rebalance_result(
            job.rebalance_id,
            'failed',
            error_message=error_msg
        )
        self.database.increment_failure_count(job.scid_normalized)
        
        # Track source failure for reliability scoring
        if job.candidate and job.candidate.source_candidates:
            # Penalize the primary source
            primary_source = job.candidate.source_candidates[0]
            self.source_failure_counts[primary_source] = self.source_failure_counts.get(primary_source, 0.0) + 1.0

        # Release budget reservation (CRITICAL-01 fix)
        self.database.release_budget_reservation(job.rebalance_id)

        # Report outcome to hive for fleet coordination (Phase 7)
        self._report_outcome_to_hive(job, success=False, cost_sats=0, amount_transferred=0)

        # Stop the job
        self.stop_job(job.scid_normalized, reason="failure")

    def _handle_job_budget_exceeded(self, job: ActiveJob, fee_msat: int,
                                    stats: Dict[str, Any]) -> None:
        """Handle a job that exceeded its configured sats budget."""
        error_msg = stats.get("last_error", "")
        budget_msat = job.candidate.max_budget_msat if job.candidate else 0
        msg = f"Exceeded msat budget: fee_msat={fee_msat} > budget_msat={budget_msat}"
        if error_msg:
            msg = f"{msg}; last_error={error_msg}"

        self.plugin.log(
            f"Rebalance FAILED (budget): {job.scid} - {msg}",
            level='warn'
        )

        # Update database (treat as failure with explicit error message)
        self.database.update_rebalance_result(
            job.rebalance_id,
            'failed',
            actual_fee_sats=(fee_msat + 999) // 1000,
            error_message=f"exceeded_budget: {msg}"
        )
        self.database.increment_failure_count(job.scid_normalized)

        # Penalize primary source reliability (it led us into an overspend scenario)
        if job.candidate and job.candidate.source_candidates:
            primary_source = job.candidate.source_candidates[0]
            self.source_failure_counts[primary_source] = self.source_failure_counts.get(primary_source, 0.0) + 1.0

        # Release budget reservation - job failed (CRITICAL-01 fix)
        self.database.release_budget_reservation(job.rebalance_id)

        # Report outcome to hive for fleet coordination (Phase 7)
        # Report the actual cost incurred even though job failed
        actual_cost_sats = (fee_msat + 999) // 1000
        self._report_outcome_to_hive(job, success=False, cost_sats=actual_cost_sats,
                                     amount_transferred=0)

        # Stop the job
        self.stop_job(job.scid_normalized, reason="exceeded_budget")

    def _handle_job_timeout(self, job: ActiveJob) -> None:
        """Handle a timed-out job."""
        elapsed_hours = (int(time.time()) - job.start_time) / 3600
        
        # Check if any progress was made
        current_balance = self._get_channel_local_balance(job.scid_normalized)
        amount_transferred = current_balance - job.initial_local_sats
        
        if amount_transferred > 0:
            # Partial success - still record the progress
            self.plugin.log(
                f"Rebalance TIMEOUT (partial): {job.scid} after {elapsed_hours:.1f}h. "
                f"Transferred {amount_transferred} sats before timeout."
            )
            self.database.update_rebalance_result(
                job.rebalance_id,
                'partial',
                fee_paid_sats=0,  # Unknown actual fee
                actual_profit_sats=0
            )
        else:
            self.plugin.log(
                f"Rebalance TIMEOUT: {job.scid} after {elapsed_hours:.1f}h with no progress",
                level='warn'
            )
            self.database.update_rebalance_result(
                job.rebalance_id,
                'timeout',
                error_message=f"Timeout after {elapsed_hours:.1f} hours"
            )
            self.database.increment_failure_count(job.scid_normalized)

        # Release budget reservation - job timed out (CRITICAL-01 fix)
        self.database.release_budget_reservation(job.rebalance_id)

        # Report outcome to hive for fleet coordination (Phase 7)
        # Partial success is still reported as success to help fleet learning
        self._report_outcome_to_hive(
            job,
            success=(amount_transferred > 0),
            cost_sats=0,  # Unknown actual fee on timeout
            amount_transferred=amount_transferred
        )

        # Stop the job
        self.stop_job(job.scid_normalized, reason="timeout")
    
    def stop_all_jobs(self, reason: str = "shutdown") -> int:
        """Stop all active jobs. Returns count of jobs stopped."""
        count = 0
        for scid in list(self._active_jobs.keys()):
            if self.stop_job(scid, reason=reason):
                count += 1
        return count
    
    def cleanup_orphans(self) -> int:
        """
        Clean up orphan sling jobs on startup.
        
        If the plugin crashes or restarts, sling jobs continue running in the
        background. This method queries sling for all active jobs and terminates
        them to prevent "Phantom Spending" where old logic fights new logic.
        
        Called during plugin init() to ensure clean state.
        
        Returns:
            Number of orphan jobs terminated
        """
        try:
            # Get list of all sling jobs
            result = self.plugin.rpc.call("sling-job", {})
            jobs = result.get("jobs", [])
            
            if not jobs:
                self.plugin.log("Startup Hygiene: No orphan sling jobs found", level='debug')
                return 0
            
            orphan_count = 0
            for job in jobs:
                scid = job.get("scid", "")
                if not scid:
                    continue
                
                try:
                    # Delete the orphan job
                    # BUG FIX: Use "job" key to match stop_job() method
                    self.plugin.rpc.call("sling-deletejob", {"job": scid})
                    orphan_count += 1
                    self.plugin.log(f"Startup Hygiene: Terminated orphan job for {scid}", level='debug')
                except RpcError as e:
                    self.plugin.log(f"Failed to delete orphan job {scid}: {e}", level='warn')
            
            if orphan_count > 0:
                self.plugin.log(
                    f"Startup Hygiene: Terminated {orphan_count} orphan sling jobs",
                    level='info'
                )
            
            return orphan_count
            
        except RpcError as e:
            # sling-job might not be available or no jobs exist
            self.plugin.log(f"Startup Hygiene: Could not query sling jobs: {e}", level='debug')
            return 0
        except Exception as e:
            self.plugin.log(f"Startup Hygiene: Unexpected error: {e}", level='warn')
            return 0

    def sync_peer_exclusions(self, policy_manager=None) -> int:
        """
        Sync peer exclusions with sling's global exclusion list.

        PHASE 6: Global Exclusion Sync
        When peers are disabled for rebalancing in our policy system,
        tell sling to globally exclude them. This prevents sling from
        considering them as sources or routing through them.

        Args:
            policy_manager: Optional PolicyManager to get disabled peers

        Returns:
            Number of peers added to sling exclusion list
        """
        excluded_count = 0

        try:
            # Get current sling exclusions
            try:
                result = self.plugin.rpc.call("sling-except-peer", {})
                current_exclusions = set(result.get("peers", []))
            except (RpcError, KeyError):
                current_exclusions = set()

            # Collect peers that should be excluded
            peers_to_exclude = set()

            # From policy manager (disabled rebalance mode)
            if policy_manager:
                try:
                    from .policy_manager import RebalanceMode
                    for peer_id, policy in policy_manager.get_all_policies().items():
                        if policy.rebalance_mode == RebalanceMode.DISABLED:
                            peers_to_exclude.add(peer_id)
                except Exception as e:
                    self.plugin.log(f"Could not get policies for exclusion sync: {e}", level='debug')

            # Add new exclusions to sling
            for peer_id in peers_to_exclude:
                if peer_id not in current_exclusions:
                    try:
                        self.plugin.rpc.call("sling-except-peer", {
                            "peer": peer_id,
                            "add": True
                        })
                        excluded_count += 1
                        self.plugin.log(
                            f"Sling Exclusion: Added {peer_id[:16]}... to global exclusion list",
                            level='debug'
                        )
                    except RpcError as e:
                        self.plugin.log(f"Failed to add peer exclusion: {e}", level='warn')

            if excluded_count > 0:
                self.plugin.log(
                    f"Sling Exclusion Sync: Added {excluded_count} peers to global exclusion list",
                    level='info'
                )

        except Exception as e:
            self.plugin.log(f"Peer exclusion sync error: {e}", level='warn')

        return excluded_count

    def add_peer_exclusion(self, peer_id: str) -> bool:
        """
        Add a single peer to sling's global exclusion list.

        Called when a peer is dynamically disabled for rebalancing.

        Args:
            peer_id: The peer node ID to exclude

        Returns:
            True if successfully added, False otherwise
        """
        try:
            self.plugin.rpc.call("sling-except-peer", {
                "peer": peer_id,
                "add": True
            })
            self.plugin.log(
                f"Sling Exclusion: Added {peer_id[:16]}... to exclusion list",
                level='info'
            )
            return True
        except RpcError as e:
            self.plugin.log(f"Failed to add sling peer exclusion: {e}", level='warn')
            return False

    def remove_peer_exclusion(self, peer_id: str) -> bool:
        """
        Remove a peer from sling's global exclusion list.

        Called when a peer is re-enabled for rebalancing.

        Args:
            peer_id: The peer node ID to un-exclude

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            self.plugin.rpc.call("sling-except-peer", {
                "peer": peer_id,
                "remove": True
            })
            self.plugin.log(
                f"Sling Exclusion: Removed {peer_id[:16]}... from exclusion list",
                level='info'
            )
            return True
        except RpcError as e:
            self.plugin.log(f"Failed to remove sling peer exclusion: {e}", level='warn')
            return False

    def get_job_status(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get status info for a specific job."""
        normalized = self._normalize_scid(channel_id)
        job = self._active_jobs.get(normalized)
        
        if not job:
            return None
        
        elapsed = int(time.time()) - job.start_time
        current_balance = self._get_channel_local_balance(normalized)
        transferred = current_balance - job.initial_local_sats
        
        return {
            "scid": job.scid,
            "source_candidates": job.source_candidates,
            "from_scid": job.from_scid,  # Primary source for backwards compat
            "num_sources": len(job.source_candidates),
            "status": job.status.value,
            "elapsed_seconds": elapsed,
            "target_amount_sats": job.target_amount_sats,
            "transferred_sats": transferred,
            "progress_pct": round(transferred / job.target_amount_sats * 100, 1) if job.target_amount_sats > 0 else 0,
            "max_fee_ppm": job.max_fee_ppm
        }
    
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """Get status info for all active jobs."""
        # BUG FIX: Avoid calling get_job_status twice per channel
        result = []
        for scid in self._active_jobs.keys():
            status = self.get_job_status(scid)
            if status:
                result.append(status)
        return result

    def get_source_failure_count(self, channel_id: str) -> float:
        """Get the recent failure count for a source channel."""
        return self.source_failure_counts.get(channel_id, 0.0)


# =============================================================================
# NNLB Health-Aware Rebalancing Constants
# =============================================================================
# Each node adjusts its OWN rebalancing based on its health tier.
# No sats transfer between nodes - purely local optimization.
ENABLE_NNLB_BUDGET_SCALING = True
DEFAULT_BUDGET_MULTIPLIER = 1.0

# Tier multipliers for OWN operations
NNLB_BUDGET_MULTIPLIERS = {
    "struggling": 2.0,    # Accept higher costs to recover own channels
    "vulnerable": 1.5,    # Elevated priority for own recovery
    "stable": 1.0,        # Normal operation
    "thriving": 0.75      # Be selective, save on routing fees
}

MIN_BUDGET_MULTIPLIER = 0.5
MAX_BUDGET_MULTIPLIER = 2.5
HEALTH_CACHE_TTL_SECONDS = 300  # 5 minutes


class EVRebalancer:
    """
    Expected Value based rebalancer with async job queue support.

    This class acts as the "Strategist" - it calculates EV and determines
    IF and HOW MUCH to rebalance. The actual execution is delegated to
    the JobManager which manages sling background jobs.

    NNLB Integration:
    When cl-hive is available, the rebalancer adjusts its EV threshold
    based on our health tier. Struggling nodes accept lower EV to recover
    faster; thriving nodes are more selective to conserve routing fees.
    """

    def __init__(self, plugin: Plugin, config: Config, database: Database,
                 clboss_manager: ClbossManager,
                 policy_manager: Optional[PolicyManager] = None,
                 hive_bridge: Optional["HiveFeeIntelligenceBridge"] = None):
        self.plugin = plugin
        self.config = config
        self.database = database
        self.clboss = clboss_manager
        self.policy_manager = policy_manager
        self.hive_bridge = hive_bridge
        self._pending: Dict[str, int] = {}
        self._our_node_id: Optional[str] = None
        self._profitability_analyzer: Optional['ChannelProfitabilityAnalyzer'] = None

        # NNLB health caching
        self._cached_health: Optional[Dict] = None
        self._health_cache_time: float = 0

        # Initialize job manager for async execution (pass hive_bridge for outcome reporting)
        self.job_manager = JobManager(plugin, config, database, hive_bridge=hive_bridge)
    
    def _get_our_node_id(self) -> str:
        if self._our_node_id is None:
            try:
                info = self.plugin.rpc.getinfo()
                self._our_node_id = info.get("id", "")
            except Exception as e:
                self.plugin.log(f"Error getting our node ID: {e}", level='error')
                self._our_node_id = ""
        return self._our_node_id

    def _get_channel_age_days(self, channel_id: str, channel_info: Dict = None) -> int:
        """
        Get the age of a channel in days (Issue #30: Velocity Gate).

        Uses SCID block height to estimate channel age. SCID format is
        "blockheight x txindex x output".

        Args:
            channel_id: Short channel ID
            channel_info: Optional channel info dict (for future use)

        Returns:
            Estimated channel age in days (0 if unknown)
        """
        try:
            # Parse block height from SCID
            if 'x' in channel_id:
                block_height = int(channel_id.split('x')[0])
            elif ':' in channel_id:
                block_height = int(channel_id.split(':')[0])
            else:
                return 0

            # Get current block height
            getinfo = self.plugin.rpc.getinfo()
            current_height = getinfo.get("blockheight", 0)

            if current_height <= 0 or block_height <= 0:
                return 0

            # Blocks since channel opened
            blocks_since_open = current_height - block_height

            # ~10 minutes per block = 144 blocks per day
            days_open = blocks_since_open // 144

            return max(0, days_open)

        except Exception as e:
            self.plugin.log(f"Error getting channel age: {e}", level='debug')
            return 0

    def set_profitability_analyzer(self, analyzer: 'ChannelProfitabilityAnalyzer') -> None:
        self._profitability_analyzer = analyzer

    def _calculate_nnlb_budget_multiplier(self) -> float:
        """
        Calculate OUR rebalance budget multiplier based on OUR health.

        This adjusts how aggressively WE rebalance OUR OWN channels.
        No sats transfer to other nodes - purely local optimization.

        When struggling: accept lower EV (more willing to pay fees to recover)
        When thriving: require higher EV (be selective, save on fees)

        Returns:
            Budget multiplier (0.5 - 2.5)
        """
        if not ENABLE_NNLB_BUDGET_SCALING or not self.hive_bridge:
            return DEFAULT_BUDGET_MULTIPLIER

        # Check cache
        now = time.time()
        if (self._cached_health is not None and
                now - self._health_cache_time < HEALTH_CACHE_TTL_SECONDS):
            return self._cached_health.get("budget_multiplier", DEFAULT_BUDGET_MULTIPLIER)

        # Query hive for OUR health (None = self)
        health = self.hive_bridge.query_member_health()
        if not health:
            return DEFAULT_BUDGET_MULTIPLIER

        # Cache result
        self._cached_health = health
        self._health_cache_time = now

        tier = health.get("health_tier", "stable")
        multiplier = NNLB_BUDGET_MULTIPLIERS.get(tier, DEFAULT_BUDGET_MULTIPLIER)

        # Clamp to bounds
        multiplier = max(MIN_BUDGET_MULTIPLIER, min(MAX_BUDGET_MULTIPLIER, multiplier))

        self.plugin.log(
            f"NNLB: Our health tier={tier}, budget_multiplier={multiplier:.2f}",
            level='debug'
        )

        return multiplier

    def find_rebalance_candidates(self) -> List[RebalanceCandidate]:
        """
        Find channels that would benefit from rebalancing.
        
        This method:
        1. First monitors existing jobs to clean up finished ones
        2. Filters out channels with active jobs
        3. Respects max concurrent job limit
        4. Returns prioritized list of candidates
        
        Performance optimizations:
        - Hoists listpeers RPC call to avoid N+1 queries
        - Uses ephemeral fee cache for listchannels calls
        """
        candidates = []

        # Initialize ephemeral fee cache for this run (cleared at end)
        self._fee_cache: Dict[str, Optional[int]] = {}

        # Thread-safe config snapshot for this rebalance cycle
        cfg = self.config.snapshot()

        # Issue #24: Clean up stale reservations before each rebalance cycle
        # This prevents budget leakage from crashed jobs
        timeout_seconds = cfg.reservation_timeout_hours * 3600
        cleaned = self.database.cleanup_stale_reservations(timeout_seconds)
        if cleaned > 0:
            self.plugin.log(f"Cleaned {cleaned} stale budget reservations before rebalance cycle")

        try:
            # First, monitor existing jobs and clean up finished ones
            if self.job_manager.active_job_count > 0:
                monitor_result = self.job_manager.monitor_jobs()
                self.plugin.log(
                    f"Job monitor: {monitor_result['checked']} checked, "
                    f"{monitor_result['completed']} completed, "
                    f"{monitor_result['failed']} failed, "
                    f"{monitor_result['timed_out']} timed out, "
                    f"{monitor_result['still_running']} running"
                )
            
            # Check if we have slots available
            available_slots = self.job_manager.slots_available()
            if available_slots <= 0:
                self.plugin.log(
                    f"No rebalance slots available ({self.job_manager.active_job_count}/"
                    f"{self.job_manager.max_concurrent_jobs} jobs active)"
                )
                return candidates
            
            # Check capital controls (pass cfg for thread-safe config access)
            if not self._check_capital_controls(cfg):
                return candidates
            
            channels = self._get_channels_with_balances()
            if not channels:
                return candidates
            
            # Hoist peer connection status call - do it once instead of per-candidate
            peer_status = self._get_peer_connection_status()
            
            # Get set of channels with active jobs
            active_channels = set(self.job_manager.active_channels)
            
            depleted_channels = []
            source_channels = []
            
            for raw_channel_id, info in channels.items():
                channel_id = raw_channel_id.replace(':', 'x')
                capacity = info.get("capacity", 0)
                spendable = info.get("spendable_sats", 0)
                if capacity == 0: 
                    continue
                
                # Check policy for this peer (v1.4: Policy-Driven Architecture)
                peer_id = info.get("peer_id")
                if peer_id and self.policy_manager:
                    # Cannot fill if rebalance_mode is DISABLED or SOURCE_ONLY
                    if not self.policy_manager.should_rebalance(peer_id, as_destination=True):
                        continue
                
                outbound_ratio = spendable / capacity
                
                # Skip channels with active jobs
                if channel_id in active_channels:
                    continue
                
                # STAGNANT INVENTORY DETECTION
                # Check if a channel is "Stagnant" (Balanced but not moving for ~1 week)
                # Threshold: turnover < 0.0015 per day (~1% per week)
                turnover = self._calculate_turnover_rate(channel_id, capacity)
                is_stagnant = (0.4 <= outbound_ratio <= 0.6) and (turnover < 0.0015)

                if is_stagnant:
                    # Treat stagnant balanced channels as source candidates to redeploy capital
                    source_channels.append((channel_id, info, outbound_ratio))
                    self.plugin.log(f"STAGNANT AWAKENING: {channel_id[:12]}... is idle (turnover {turnover:.4f}). Adding to source pool.", level='debug')
                
                elif outbound_ratio < cfg.low_liquidity_threshold:
                    depleted_channels.append((channel_id, info, outbound_ratio))
                elif outbound_ratio > cfg.high_liquidity_threshold:
                    source_channels.append((channel_id, info, outbound_ratio))
            
            if not depleted_channels or not source_channels:
                return candidates
                
            self.plugin.log(
                f"Found {len(depleted_channels)} depleted and {len(source_channels)} source channels "
                f"(excluding {len(active_channels)} with active jobs)"
            )
            
            for dest_id, dest_info, dest_ratio in depleted_channels:
                if self._is_pending_with_backoff(dest_id): 
                    continue
                
                # =====================================================================
                # FUTILITY CIRCUIT BREAKER (TODO #15)
                # =====================================================================
                # Some channels have positive EV spreads but broken routing paths.
                # Exponential backoff slows down retries, but doesn't stop them.
                # After 10+ failures, the channel is likely a "Dead End" and further
                # attempts waste gossip bandwidth and lock HTLCs.
                #
                # Hard Cap: If failed > 10 times, require 48h cooldown before retry
                # =====================================================================
                fail_count, last_fail = self.database.get_failure_count(dest_id)
                if fail_count > 10:
                    now = int(time.time())
                    futility_cooldown = 172800  # 48 hours in seconds
                    if (now - last_fail) < futility_cooldown:
                        self.plugin.log(
                            f"FUTILITY BREAKER: Skipping {dest_id[:12]}... - {fail_count} consecutive failures, "
                            f"cooldown {(futility_cooldown - (now - last_fail)) // 3600}h remaining",
                            level='debug'
                        )
                        continue
                    else:
                        # Cooldown expired - allow retry but log it
                        self.plugin.log(
                            f"FUTILITY BREAKER: {dest_id[:12]}... cooldown expired after {fail_count} failures, allowing retry",
                            level='info'
                        )
                
                # CONGESTION PROTECTION: Skip congested channels as rebalance destinations
                # Rebalancing into a slot-congested channel can worsen HTLC contention
                dest_state = self.database.get_channel_state(dest_id)
                if dest_state and dest_state.get("state") == "congested":
                    self.plugin.log(
                        f"CONGESTION GUARD: Skipping {dest_id[:12]}... as rebalance target (HTLC slots stressed)",
                        level='info'
                    )
                    continue
                
                last_rebalance = self.database.get_last_rebalance_time(dest_id)
                if last_rebalance:
                    cooldown = self.config.rebalance_cooldown_hours * 3600
                    if int(time.time()) - last_rebalance < cooldown: 
                        continue
                
                candidate = self._analyze_rebalance_ev(
                    dest_id, dest_info, dest_ratio, source_channels, peer_status
                )
                if candidate:
                    candidates.append(candidate)
                    
                    # Stop if we have enough candidates to fill available slots
                    if len(candidates) >= available_slots:
                        break
            
            # Sort by priority
            def sort_key(c):
                dest_state = self.database.get_channel_state(c.to_channel)
                flow_state = dest_state.get("state", "balanced") if dest_state else "balanced"
                priority = 2 if flow_state == "source" else 1
                return (priority, c.expected_profit_sats)
            
            candidates.sort(key=sort_key, reverse=True)
            
            # Limit to available slots
            return candidates[:available_slots]
        
        finally:
            # Clear ephemeral fee cache at end of run
            self._fee_cache = {}
            
            # Garbage Collection: Prune stale source failure counts (TODO #18)
            # BUG FIX: Use try/except to check if 'channels' exists (may not if early exception)
            try:
                active_channel_ids = set(channels.keys())
                if active_channel_ids:
                    self.job_manager.prune_stale_source_failures(active_channel_ids)
            except (NameError, Exception):
                pass  # Don't fail the main method for GC errors

    def _analyze_rebalance_ev(self, dest_channel: str, dest_info: Dict[str, Any],
                              dest_ratio: float,
                              sources: List[Tuple[str, Dict[str, Any], float]],
                              peer_status: Optional[Dict] = None) -> Optional[RebalanceCandidate]:
        """
        Analyze expected value of rebalancing a channel with multi-source support.
        
        This method now identifies ALL profitable source channels and includes them
        in the candidate. EV calculations are based on the primary (best) source,
        but additional sources serve as fallbacks for Sling's pathfinding.
        
        Args:
            dest_channel: Destination channel SCID
            dest_info: Channel info dict
            dest_ratio: Current outbound liquidity ratio
            sources: List of potential source channels
            peer_status: Pre-fetched peer connection status (optimization)
        """
        dest_state = self.database.get_channel_state(dest_channel)
        dest_flow_state = dest_state.get("state", "unknown") if dest_state else "unknown"
        
        if dest_flow_state == "sink": 
            return None
        
        # FLAP PROTECTION: Skip unstable destination peers
        # Peers with low uptime (high disconnect rate) are unreliable rebalance targets
        dest_peer_id = dest_info.get("peer_id", "")
        if dest_peer_id:
            uptime_pct = self.database.get_peer_uptime_percent(dest_peer_id, 86400)  # 24h window
            if uptime_pct < 90.0:
                self.plugin.log(
                    f"Skipping rebalance candidate {dest_peer_id}: unstable connection "
                    f"({uptime_pct:.1f}% uptime in 24h).",
                    level='info'
                )
                return None
        
        # Check profitability logic
        if self._profitability_analyzer:
            try:
                prof = self._profitability_analyzer.analyze_channel(dest_channel)
                if prof and prof.classification.value == "zombie": 
                    return None
                if prof and prof.classification.value == "underwater" and prof.marginal_roi <= 0:
                    return None
            except Exception: 
                pass

        capacity = dest_info.get("capacity", 0)
        spendable = dest_info.get("spendable_sats", 0)

        # ZERO-TOLERANCE: Never attempt to rebalance channels with non-positive capacity.
        # These can appear transiently (closing/failed states) or via incomplete channel info.
        if capacity <= 0:
            return None
        
        # Dynamic targeting based on flow state
        if dest_flow_state == "source": 
            target_ratio = 0.85
        elif dest_flow_state == "sink": 
            target_ratio = 0.15
        else: 
            target_ratio = 0.50
        
        # =====================================================================
        # VOLUME-WEIGHTED LIQUIDITY TARGETS (TODO #14 - Smart Allocation)
        # =====================================================================
        # Instead of blindly targeting fixed ratios (e.g., 50% for Balanced),
        # we calculate a volume-aware target:
        # 1. Velocity: Average daily volume over the last 7 days
        # 2. Inventory Goal: Enough liquidity for 3 days of flow
        # 3. Cap: Never exceed capacity * target_ratio (don't overfill)
        # 4. Floor: Never drop below rebalance_min_amount (burst buffer)
        #
        # Benefit: Frees idle Bitcoin from slow-moving large channels to be
        # deployed to high-velocity channels, improving Return on Capital.
        # =====================================================================
        
        # Get flow stats for volume calculation
        if dest_state:
            sats_in = dest_state.get("sats_in", 0)
            sats_out = dest_state.get("sats_out", 0)
            # Daily volume is the average of 7-day totals
            daily_volume = (sats_in + sats_out) / max(self.config.flow_window_days, 1)
        else:
            daily_volume = 0

        # =====================================================================
        # Issue #30: VELOCITY GATE - Prevent overfilling low-velocity channels
        # =====================================================================
        # Channels with little to no routing history shouldn't get aggressively
        # rebalanced. We calculate velocity (daily turnover as fraction of
        # capacity) and use conservative targets for low-velocity channels.
        # =====================================================================
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        velocity = daily_volume / capacity if capacity > 0 else 0.0

        # Get channel age for grace period
        channel_age_days = self._get_channel_age_days(dest_channel, dest_info)

        # Apply velocity gate
        velocity_adjusted_target_ratio = target_ratio
        velocity_gate_reason = None

        if cfg.enable_velocity_gate:
            # Grace period for new channels - they get normal targeting
            if channel_age_days < cfg.new_channel_grace_days:
                velocity_gate_reason = f"new_channel_grace (age={channel_age_days}d)"
            elif velocity < cfg.min_velocity_threshold:
                # Low velocity - use conservative target (15% of capacity)
                # This is enough to test routing without wasting budget
                velocity_adjusted_target_ratio = 0.15
                velocity_gate_reason = f"low_velocity ({velocity:.4f} < {cfg.min_velocity_threshold})"
                self.plugin.log(
                    f"VELOCITY GATE: {dest_channel[:12]}... conservative target "
                    f"(velocity={velocity:.4f}, age={channel_age_days}d, "
                    f"target={velocity_adjusted_target_ratio:.0%} vs original {target_ratio:.0%})",
                    level='debug'
                )
            else:
                velocity_gate_reason = f"velocity_ok ({velocity:.4f})"

        # Use velocity-adjusted target ratio
        target_ratio = velocity_adjusted_target_ratio

        # =====================================================================
        # HOTFIX 0.1: Destination Sizing Guard
        # =====================================================================
        # Problem: target_spendable = max(min_amount, target_spendable) could force
        # target above capacity, causing repeated failures and pathological candidates.
        # Solution: Clamp to capacity and skip tiny channels that can't meet min_amount.
        # =====================================================================
        
        # Guard: Skip zero-capacity channels entirely
        if capacity <= 0:
            return None
        
        # Calculate volume-based target (3 days of buffer)
        vol_target = int(daily_volume * 3)
        
        # Calculate capacity-based target (original logic)
        cap_target = int(capacity * target_ratio)
        
        # Smart Allocation: Use the LOWER of volume target or capacity target
        # This prevents overfilling slow channels while still allowing fast channels
        # to be fully stocked
        if vol_target > 0:
            raw_target = min(cap_target, vol_target)
        else:
            # No volume data yet - fall back to capacity-based target
            raw_target = cap_target
        
        # CRITICAL: Clamp raw_target to capacity (never exceed what's possible)
        raw_target = min(raw_target, capacity)
        
        # Skip tiny channels that can't meet the minimum rebalance amount
        # Instead of force-filling them (which caused target > capacity), we skip them
        if raw_target < self.config.rebalance_min_amount:
            self.plugin.log(
                f"SIZING GUARD: Skipping {dest_channel[:12]}... - raw_target {raw_target:,} < "
                f"min_amount {self.config.rebalance_min_amount:,} (capacity: {capacity:,})",
                level='debug'
            )
            return None
        
        target_spendable = raw_target
        
        # Log when volume-weighting reduces target significantly
        if vol_target > 0 and vol_target < cap_target * 0.8:
            self.plugin.log(
                f"SMART ALLOCATION: {dest_channel[:12]}... volume-weighted target "
                f"{target_spendable:,} sats (vol: {vol_target:,}, cap: {cap_target:,}, "
                f"daily_vol: {daily_volume:,.0f})",
                level='debug'
            )
        
        amount_needed = target_spendable - spendable
        if amount_needed <= 0: 
            return None
        
        # ZERO-TOLERANCE: Never attempt to push more sats than the channel can accept.
        headroom = max(0, capacity - spendable)
        if headroom <= 0:
            return None

        # Compute desired amount, then clamp to headroom and execute per-chunk.
        desired_amount = min(
            self.config.rebalance_max_amount,
            max(self.config.rebalance_min_amount, amount_needed)
        )
        desired_amount = min(desired_amount, headroom)
        if desired_amount <= 0:
            return None

        # ZERO-TOLERANCE: Evaluate EV on the actual execution unit (one chunk).
        # This matches the "stop after first success" execution model.
        rebalance_amount = min(desired_amount, self.config.sling_chunk_size_sats)
        amount_msat = rebalance_amount * 1000
        
        # BROADCAST FEE ALIGNMENT (Phase 5.5): Use confirmed broadcast fee for EV
        # This prevents "Self-Arbitrage" where we pay for a rebalance expecting to
        # earn at the internal target fee, but Hysteresis blocked the update so we're
        # actually still selling liquidity at a lower broadcast fee.
        fee_state = self.database.get_fee_strategy_state(dest_channel)
        broadcast_fee_ppm = fee_state.get("last_broadcast_fee_ppm", 0)
        
        # Fallback to listpeerchannels fee if no broadcast fee recorded
        if broadcast_fee_ppm <= 0:
            broadcast_fee_ppm = dest_info.get("fee_ppm", 0)
        
        outbound_fee_ppm = broadcast_fee_ppm
        inbound_fee_ppm = self._estimate_inbound_fee(dest_info.get("peer_id", ""))

        # Check if destination is a hive peer (relax profitability requirements)
        is_hive_destination = False
        if self.policy_manager:
            dest_peer_id = dest_info.get("peer_id", "")
            if dest_peer_id:
                policy = self.policy_manager.get_policy(dest_peer_id)
                if policy.strategy == FeeStrategy.HIVE:
                    is_hive_destination = True

        # Get ALL profitable source candidates (sorted by score, best first)
        source_candidates = self._select_source_candidates(
            sources, rebalance_amount, dest_channel, outbound_fee_ppm, inbound_fee_ppm,
            peer_status=peer_status, is_hive_destination=is_hive_destination
        )
        
        if not source_candidates: 
            return None
        
        # Extract just the SCIDs for the candidate list
        source_scids = [cid for cid, _, _, _ in source_candidates]
        
        # Use the PRIMARY (best) source for EV calculations
        primary_source_id, primary_source_info, primary_score, primary_opp_cost = source_candidates[0]
        
        source_fee_ppm = primary_source_info.get("fee_ppm", 0)
        source_capacity = primary_source_info.get("capacity", 1)
        source_turnover_rate = self._calculate_turnover_rate(primary_source_id, source_capacity)
        
        # Use the primary source's opportunity cost for spread calculation
        weighted_opp_cost = primary_opp_cost
        spread_ppm = outbound_fee_ppm - inbound_fee_ppm - weighted_opp_cost
        
        # This should always be positive since _select_source_candidates filters for it,
        # but check anyway for safety
        if spread_ppm <= 0: 
            return None
        
        raw_budget_msat = (spread_ppm * amount_msat) // 1_000_000
        # ZERO-TOLERANCE: Avoid a zero-sats budget due to integer truncation.
        # We clamp to at least 1 sat (1000 msat). This is conservative: it makes EV slightly worse,
        # and ensures execution can enforce a non-zero fee cap.
        max_budget_msat = max(1000, raw_budget_msat)
        # Use ceiling sats for conservative accounting.
        max_budget_sats = (max_budget_msat + 999) // 1000
        
        # MAJOR-13 DOCUMENTATION: Modified Kelly Criterion for Per-Trade Sizing
        #
        # This is NOT classical Kelly for portfolio allocation. Instead, we apply
        # Kelly scaling to each individual rebalance "bet" to size the maximum fee
        # we're willing to pay based on:
        #   - p: Success probability (peer reputation score, 0-1)
        #   - b: Odds offered (outbound_fee_ppm / cost_ppm ratio)
        #
        # Kelly formula: f* = p - (1-p)/b
        #   - If f* > 0: We have positive EV, scale budget by f* * kelly_fraction
        #   - If f* <= 0: Negative EV, reject this rebalance
        #
        # The kelly_fraction (default 0.5 = "Half Kelly") reduces volatility drag.
        # Full Kelly (1.0) maximizes theoretical growth but suffers from high variance.
        #
        # Note: This applies to max_budget_sats (the fee cap for this trade), not
        # total routing capital. For daily budget management, see reserve_budget().
        if self.config.enable_kelly:
            reputation = self.database.get_peer_reputation(dest_info.get("peer_id", ""))
            p = reputation.get('score', 0.5)  # Success probability
            cost_ppm = inbound_fee_ppm + weighted_opp_cost
            b = outbound_fee_ppm / cost_ppm if cost_ppm > 0 else float('inf')  # Odds
            kelly_f = p - (1 - p) / b if b > 0 else -1.0  # Raw Kelly fraction
            kelly_safe = min(kelly_f * self.config.kelly_fraction, 1.0)

            if kelly_safe <= 0:
                return None  # Negative EV, reject
            max_budget_sats = int(max_budget_sats * kelly_safe)
            max_budget_msat = max_budget_sats * 1000

        if amount_msat > 0:
            # ZERO-TOLERANCE: Derive max routing fee from the sats budget for this chunk.
            # Our EV math subtracts max_budget_sats as a worst-case routing cost, so we must
            # ensure execution cannot exceed that budget.
            budget_ppm = (max_budget_msat * 1_000_000) // amount_msat if amount_msat > 0 else 0

            # Optional heuristic upper bound, but ALWAYS clamp to the sats-budget-derived ppm.
            heuristic_ppm = inbound_fee_ppm + (spread_ppm // 2)
            max_fee_ppm = max(1, min(heuristic_ppm, budget_ppm)) if budget_ppm > 0 else 0
        else:
            max_fee_ppm = 0
            
        if max_fee_ppm <= 0: 
            return None
        
        dest_turnover_rate = self._calculate_turnover_rate(dest_channel, capacity)
        cooldown_days = self.config.rebalance_cooldown_hours / 24.0
        expected_utilization = max(min(dest_turnover_rate * cooldown_days, 1.0), 0.05)
        
        expected_income = (rebalance_amount * expected_utilization * outbound_fee_ppm) // 1_000_000
        turnover_weight = min(1.0, source_turnover_rate * 7)
        expected_source_loss = (rebalance_amount * expected_utilization * source_fee_ppm * turnover_weight) // 1_000_000
        expected_profit = expected_income - max_budget_sats - expected_source_loss
        
        # Strategic Rebalance Exemption: Dynamic threshold based on destination policy
        # PPM-BASED PROFIT GATE: When rebalance_min_profit_ppm > 0, the threshold
        # scales linearly with rebalance_amount, decoupling acceptance from chunk size.
        if self.config.rebalance_min_profit_ppm > 0:
            profit_threshold = (rebalance_amount * self.config.rebalance_min_profit_ppm) // 1_000_000
        else:
            profit_threshold = self.config.rebalance_min_profit

        # NNLB Health-Aware Threshold Adjustment:
        # When struggling: accept lower profit (threshold / multiplier)
        # When thriving: require higher profit (threshold / multiplier)
        # This adjusts OUR OWN rebalancing aggression - no fund transfers.
        nnlb_multiplier = self._calculate_nnlb_budget_multiplier()
        if nnlb_multiplier != 1.0 and profit_threshold > 0:
            # Divide threshold by multiplier:
            # - Struggling (2.0x): threshold becomes 50% -> accept lower profit
            # - Thriving (0.75x): threshold becomes 133% -> require higher profit
            profit_threshold = int(profit_threshold / nnlb_multiplier)

        is_hive_transfer = False
        
        if self.policy_manager:
            dest_peer_id = dest_info.get("peer_id", "")
            if dest_peer_id:
                policy = self.policy_manager.get_policy(dest_peer_id)
                if policy.strategy == FeeStrategy.HIVE:
                    is_hive_transfer = True
                    # Allow negative profit (cost) up to tolerance
                    profit_threshold = -(self.config.hive_rebalance_tolerance)
        
        # Check Profit against Dynamic Threshold
        if expected_profit < profit_threshold:
            # Add debug logging to explain rejection
            if is_hive_transfer:
                self.plugin.log(
                    f"HIVE REBALANCE SKIPPED: Cost too high. Profit {expected_profit} < Tolerance {-self.config.hive_rebalance_tolerance}",
                    level='debug'
                )
            return None
        
        # Log Success (Strategic override)
        if is_hive_transfer and expected_profit < 0:
            self.plugin.log(
                f"STRATEGIC EXEMPTION: Allowing negative EV rebalance to Hive Peer {dest_channel}. "
                f"Cost: {abs(expected_profit)} sats (Tolerance: {self.config.hive_rebalance_tolerance})",
                level='info'
            )
        
        return RebalanceCandidate(
            source_candidates=source_scids,
            to_channel=dest_channel,
            primary_source_peer_id=primary_source_info.get("peer_id", ""),
            to_peer_id=dest_info.get("peer_id", ""),
            amount_sats=rebalance_amount,
            amount_msat=amount_msat,
            outbound_fee_ppm=outbound_fee_ppm,
            inbound_fee_ppm=inbound_fee_ppm,
            source_fee_ppm=source_fee_ppm,
            weighted_opp_cost_ppm=weighted_opp_cost,
            spread_ppm=spread_ppm,
            max_budget_sats=max_budget_sats,
            max_budget_msat=max_budget_msat,
            max_fee_ppm=max_fee_ppm,
            expected_profit_sats=expected_profit,
            liquidity_ratio=dest_ratio,
            dest_flow_state=dest_flow_state,
            dest_turnover_rate=dest_turnover_rate,
            source_turnover_rate=source_turnover_rate
        )

    def _calculate_turnover_rate(self, channel_id: str, capacity: int) -> float:
        if capacity <= 0: 
            return 0.0
        try:
            state = self.database.get_channel_state(channel_id)
            if not state: 
                return 0.05
            volume = (state.get("sats_in", 0) + state.get("sats_out", 0)) / max(self.config.flow_window_days, 1)
            return max(0.0001, min(1.0, volume / capacity))
        except Exception: 
            return 0.05

    def _estimate_inbound_fee(self, peer_id: str, amount_msat: int = 100000000) -> int:
        """
        Estimate the inbound routing fee to reach a peer.

        ENHANCED (Phase 6): Prioritizes historical actual costs over heuristics.
        ENHANCED (Phase 7): Zero fee for hive fleet members.

        Priority order:
        0. HIVE peer - Zero fee (fleet members have 0 fee channels)
        1. Historical data (high confidence) - Use median, most accurate
        2. Historical data (medium) - Blend with last-hop fee
        3. Historical data (low) - Use with buffer
        4. Last hop fee + buffer - Gossip-based estimate
        5. Route estimation - Ask CLN for a route
        6. Default fallback - 1000 PPM

        Returns:
            Estimated inbound fee in PPM
        """
        # =====================================================================
        # PHASE 7: HIVE Fleet Zero-Fee Priority
        # =====================================================================
        # Hive fleet members have 0 fee channels between them. When routing
        # through a hive peer, the cost is zero. This is the highest priority
        # check to ensure we utilize fleet connectivity efficiently.
        # =====================================================================

        if self.policy_manager and self.policy_manager.is_hive_peer(peer_id):
            self.plugin.log(
                f"INBOUND FEE EST [{peer_id[:12]}...]: HIVE peer - 0 PPM (fleet zero-fee)",
                level='debug'
            )
            return 0

        # =====================================================================
        # PHASE 6: Historical-First Fee Estimation
        # =====================================================================
        # Real rebalance costs are the ground truth. Use them when available.
        # Historical data accounts for actual multi-hop routes, not just last hop.
        # =====================================================================

        hist_data = self.database.get_historical_inbound_fee_ppm(peer_id)
        last_hop = self._get_last_hop_fee(peer_id)

        if hist_data:
            confidence = hist_data['confidence']
            median_ppm = hist_data['median_fee_ppm']
            avg_ppm = hist_data['avg_fee_ppm']
            samples = hist_data['sample_count']

            if confidence == 'high':
                # 10+ samples: trust the data, use median (robust to outliers)
                estimate = median_ppm
                self.plugin.log(
                    f"INBOUND FEE EST [{peer_id[:12]}...]: Using historical median "
                    f"{estimate} PPM (n={samples}, conf=high)",
                    level='debug'
                )
                return estimate

            elif confidence == 'medium':
                # 5-9 samples: blend historical with last-hop if available
                if last_hop is not None:
                    # Weighted average: 70% historical, 30% last-hop based
                    last_hop_estimate = last_hop + self.config.inbound_fee_estimate_ppm
                    estimate = int(median_ppm * 0.7 + last_hop_estimate * 0.3)
                else:
                    estimate = median_ppm
                self.plugin.log(
                    f"INBOUND FEE EST [{peer_id[:12]}...]: Blended estimate "
                    f"{estimate} PPM (hist={median_ppm}, n={samples}, conf=medium)",
                    level='debug'
                )
                return estimate

            else:
                # 3-4 samples: use with 10% buffer for uncertainty
                estimate = int(avg_ppm * 1.1)
                self.plugin.log(
                    f"INBOUND FEE EST [{peer_id[:12]}...]: Historical with buffer "
                    f"{estimate} PPM (avg={avg_ppm}, n={samples}, conf=low)",
                    level='debug'
                )
                return estimate

        # No historical data - fall back to heuristics
        if last_hop is not None:
            estimate = last_hop + self.config.inbound_fee_estimate_ppm
            self.plugin.log(
                f"INBOUND FEE EST [{peer_id[:12]}...]: Last-hop based "
                f"{estimate} PPM (last_hop={last_hop})",
                level='debug'
            )
            return estimate

        route_fee = self._get_route_fee_estimate(peer_id, amount_msat)
        if route_fee:
            self.plugin.log(
                f"INBOUND FEE EST [{peer_id[:12]}...]: Route-based "
                f"{route_fee} PPM",
                level='debug'
            )
            return route_fee

        # Ultimate fallback
        self.plugin.log(
            f"INBOUND FEE EST [{peer_id[:12]}...]: Default fallback 1000 PPM",
            level='debug'
        )
        return 1000

    def _get_last_hop_fee(self, peer_id: str) -> Optional[int]:
        """
        Get the fee for the last hop from a peer to us.
        
        Uses memoization via self._fee_cache to avoid repeated listchannels
        RPC calls within a single find_rebalance_candidates run.
        """
        # Check cache first (memoization for this run)
        if hasattr(self, '_fee_cache') and peer_id in self._fee_cache:
            return self._fee_cache[peer_id]
        
        result = None
        try:
            our_id = self._get_our_node_id()
            if not our_id: 
                return None
            channels = self.plugin.rpc.listchannels(source=peer_id)
            for ch in channels.get("channels", []):
                if ch.get("destination") == our_id:
                    result = ch.get("fee_per_millionth", 0) + (ch.get("base_fee_millisatoshi", 0) // 1000)
                    break
        except Exception: 
            pass
        
        # Cache the result (even if None, to avoid re-querying)
        if hasattr(self, '_fee_cache'):
            self._fee_cache[peer_id] = result
        
        return result

    def _get_route_fee_estimate(self, peer_id: str, amount_msat: int) -> Optional[int]:
        try:
            route = self.plugin.rpc.getroute(id=peer_id, amount_msat=amount_msat, riskfactor=10, maxhops=6)
            if route.get("route"):
                first_hop = route["route"][0].get("amount_msat", amount_msat)
                if isinstance(first_hop, str): 
                    first_hop = int(first_hop.replace("msat", ""))
                return int(((first_hop - amount_msat) / amount_msat) * 1_000_000)
        except Exception: 
            pass
        return None

    def _get_historical_inbound_fee(self, peer_id: str) -> Optional[int]:
        try:
            hist = self.database.get_rebalance_history_by_peer(peer_id)
            if not hist: 
                return None
            total_ppm, count = 0, 0
            for r in hist:
                if r.get("status") == "success" and r.get("amount_msat", 0) > 0:
                    total_ppm += int((r["fee_paid_msat"] / r["amount_msat"]) * 1_000_000)
                    count += 1
            if count > 0: 
                return total_ppm // count
        except Exception: 
            pass
        return None

    def _select_source_candidates(
        self,
        sources: List[Tuple[str, Dict[str, Any], float]],
        amount_needed: int,
        dest_channel: str,
        dest_outbound_fee_ppm: int,
        dest_inbound_fee_ppm: int,
        peer_status: Optional[Dict] = None,
        is_hive_destination: bool = False
    ) -> List[Tuple[str, Dict[str, Any], int, float]]:
        """
        Select all profitable source channels for a rebalance.

        Instead of returning a single "best" source, this returns ALL sources
        that have a positive spread (EV > 0), sorted by score (highest first).
        For hive destinations, allows negative spread up to hive_rebalance_tolerance.
        This allows Sling to handle pathfinding failover automatically.
        
        Args:
            sources: List of (channel_id, info, outbound_ratio) tuples
            amount_needed: Amount to rebalance in sats
            dest_channel: Destination channel SCID
            dest_outbound_fee_ppm: Outbound fee of destination channel
            dest_inbound_fee_ppm: Estimated inbound fee to destination
            peer_status: Pre-fetched peer connection status (optimization)
            
        Returns:
            List of (channel_id, info, score, weighted_opp_cost) tuples,
            sorted by score (highest first). Empty list if no profitable sources.
        """
        candidates = []
        # Use provided peer_status or fetch if not provided (fallback for direct calls)
        peers = peer_status if peer_status is not None else self._get_peer_connection_status()

        # Exclude sources with active jobs
        active_channels = set(self.job_manager.active_channels)

        # =================================================================
        # PHASE 6: Rejection Diagnostics
        # =================================================================
        # Track why sources are rejected to help diagnose "0 candidates" cases
        rejections = {
            'active_job': 0,
            'policy_blocked': 0,
            'insufficient_balance': 0,
            'disconnected': 0,
            'unstable_uptime': 0,
            'source_protected': 0,
            'negative_spread': 0,
            'below_profit_threshold': 0
        }
        best_rejected_spread = None  # Track closest-to-profitable rejection

        for cid, info, ratio in sources:
            # Skip if this source has an active job
            normalized = cid.replace(':', 'x')
            if normalized in active_channels:
                rejections['active_job'] += 1
                continue

            # Check policy for draining this source (v1.4: Policy-Driven Architecture)
            pid = info.get("peer_id", "")
            if pid and self.policy_manager:
                # Cannot drain if rebalance_mode is DISABLED or SINK_ONLY
                if not self.policy_manager.should_rebalance(pid, as_destination=False):
                    rejections['policy_blocked'] += 1
                    continue

            # Skip if insufficient balance
            if info.get("spendable_sats", 0) < amount_needed:
                rejections['insufficient_balance'] += 1
                continue

            # Skip disconnected peers
            if pid and pid in peers and not peers[pid].get("connected"):
                rejections['disconnected'] += 1
                continue

            # FLAP PROTECTION: Skip unstable source peers
            # Peers with low uptime (high disconnect rate) are unreliable rebalance sources
            if pid:
                uptime_pct = self.database.get_peer_uptime_percent(pid, 86400)  # 24h window
                if uptime_pct < 90.0:
                    rejections['unstable_uptime'] += 1
                    self.plugin.log(
                        f"Skipping source candidate {pid}: unstable connection "
                        f"({uptime_pct:.1f}% uptime in 24h).",
                        level='info'
                    )
                    continue

            # SOURCE PROTECTION (Anti-Cannibalization)
            # Prevent draining our best source channels unless they are overflowing.
            # A "Source" is meant to sell INBOUND liquidity. Rebalancing OUT destroys that value.
            #
            # RELAXED MODE: Only allow if local balance > 80% (outbound_ratio > 0.8)
            state = self.database.get_channel_state(cid)
            if state and state.get("state") == "source":
                if ratio < 0.80:
                    rejections['source_protected'] += 1
                    self.plugin.log(
                        f"Skipping source candidate {cid}: Protected Source "
                        f"(ratio={ratio:.2f} < 0.80)",
                        level='debug'
                    )
                    continue
            
            # Calculate opportunity cost for this source
            source_fee_ppm = info.get("fee_ppm", 1000)
            source_capacity = info.get("capacity", 1)
            source_turnover_rate = self._calculate_turnover_rate(cid, source_capacity)

            # Get flow state FIRST - needed for flow-aware opportunity cost
            state = self.database.get_channel_state(cid)
            flow_state = state.get("state", "balanced") if state else "balanced"

            # =================================================================
            # FLOW-AWARE OPPORTUNITY COST (Phase 6 Enhancement)
            # =================================================================
            # The cost of using liquidity from a channel depends on its flow:
            #
            # SINK channels: Naturally receiving inbound liquidity, so draining
            #   them has LOWER opportunity cost - they will replenish passively.
            #   Factor: 0.3x (70% discount)
            #
            # SOURCE channels: Actively forwarding outbound. Draining them
            #   destroys revenue-generating capacity. HIGHER opportunity cost.
            #   Factor: 1.5x (50% premium) - but already filtered by SOURCE PROTECTION
            #
            # BALANCED channels: Neutral flow. Standard calculation applies.
            #   Factor: 1.0x
            #
            # Combined with BUFFER-AWARE logic (idle channels cheaper to use)
            # =================================================================

            # Base turnover weight (buffer-aware)
            if source_turnover_rate < 0.10:
                # Channel is mostly idle. Effective weight should be very low.
                base_turnover_weight = max(0.01, source_turnover_rate)
            else:
                # Channel is active. Standard penalty applies.
                base_turnover_weight = min(1.0, source_turnover_rate * 7)

            # Apply flow-aware multiplier
            if flow_state == "sink":
                # Sink channel: receiving liquidity naturally, lower opp cost
                flow_multiplier = 0.3
            elif flow_state == "source":
                # Source channel: losing liquidity, higher opp cost
                # Note: SOURCE PROTECTION already filters ratio < 0.80
                flow_multiplier = 1.5
            else:
                # Balanced: neutral
                flow_multiplier = 1.0

            turnover_weight = base_turnover_weight * flow_multiplier
            weighted_opp_cost = int(source_fee_ppm * turnover_weight)

            # Calculate spread: what we earn minus what it costs
            spread_ppm = dest_outbound_fee_ppm - dest_inbound_fee_ppm - weighted_opp_cost

            # For hive destinations, allow negative spread up to tolerance
            # (hive channels are 0-fee, so rebalancing is much cheaper)
            if is_hive_destination:
                # Convert tolerance from sats to approximate ppm for comparison
                tolerance_ppm = int((self.config.hive_rebalance_tolerance * 1_000_000) / max(amount_needed, 1))
                min_spread = -tolerance_ppm
            else:
                min_spread = 0

            # Only include sources meeting spread threshold
            if spread_ppm < min_spread:
                rejections['negative_spread'] += 1
                # Track the best rejected spread for diagnostics
                if best_rejected_spread is None or spread_ppm > best_rejected_spread['spread']:
                    best_rejected_spread = {
                        'channel': cid,
                        'spread': spread_ppm,
                        'dest_fee': dest_outbound_fee_ppm,
                        'inbound_fee': dest_inbound_fee_ppm,
                        'opp_cost': weighted_opp_cost,
                        'flow_state': flow_state,
                        'is_hive': is_hive_destination
                    }
                continue

            # Check minimum profit threshold
            # PPM-BASED PROFIT GATE: Scale threshold with amount to decouple from chunk size
            expected_profit_estimate = (spread_ppm * amount_needed) // 1_000_000
            if self.config.rebalance_min_profit_ppm > 0:
                min_profit_threshold = (amount_needed * self.config.rebalance_min_profit_ppm) // 1_000_000
            else:
                min_profit_threshold = self.config.rebalance_min_profit
            if expected_profit_estimate < min_profit_threshold:
                rejections['below_profit_threshold'] += 1
                continue

            # Calculate score for sorting (higher is better)
            score = (ratio * 50) - (source_fee_ppm / 10)

            # Bonus for sink/balanced channels (they have excess outbound we want to use)
            if flow_state == "sink":
                score += 100
            elif flow_state == "balanced":
                # Apply Stagnant Inventory Bonus
                if source_turnover_rate < 0.0015:
                    score += 10 # Awakening Bonus
                    self.plugin.log(f"STAGNANT BONUS: Applying +10 priority to stagnant channel {cid[:12]}...", level='info')
                
                score += 20
            
            # RELIABILITY PENALTY: Penalize sources with recent failures
            fails = self.job_manager.get_source_failure_count(cid)
            if fails > 0:
                penalty = fails * 50
                score -= penalty
                self.plugin.log(
                    f"Applying reliability penalty to {cid}: -{penalty:.1f} (fails: {fails:.1f})",
                    level='debug'
                )
            
            candidates.append((cid, info, score, weighted_opp_cost))

        # Sort by score (highest first) so Sling tries most profitable sources first
        candidates.sort(key=lambda x: x[2], reverse=True)

        # =================================================================
        # PHASE 6: Log Rejection Summary for Diagnostics
        # =================================================================
        total_rejected = sum(rejections.values())
        if total_rejected > 0 and not candidates:
            # No candidates found - log detailed breakdown
            non_zero = {k: v for k, v in rejections.items() if v > 0}
            self.plugin.log(
                f"SOURCE REJECTION BREAKDOWN for {dest_channel[:12]}...: "
                f"Evaluated {len(sources)} sources, {total_rejected} rejected: {non_zero}",
                level='info'
            )

            # Log the "near miss" - closest to profitable
            if best_rejected_spread:
                b = best_rejected_spread
                self.plugin.log(
                    f"NEAR MISS: {b['channel'][:12]}... had spread={b['spread']} PPM "
                    f"(need >0). Components: dest_fee={b['dest_fee']}, "
                    f"inbound_cost={b['inbound_fee']}, opp_cost={b['opp_cost']} "
                    f"(flow={b['flow_state']})",
                    level='info'
                )

        return candidates

    def _get_peer_connection_status(self) -> Dict:
        status = {}
        try:
            for p in self.plugin.rpc.listpeers().get("peers", []):
                status[p.get("id")] = {"connected": p.get("connected", False)}
        except Exception: 
            pass
        return status

    def _get_channels_with_balances(self) -> Dict[str, Dict[str, Any]]:
        """Get all channels with their current balances and fee info."""
        channels = {}
        try:
            listfunds = self.plugin.rpc.listfunds()
            listpeers = self.plugin.rpc.listpeers()
            
            # Build peer info map
            peer_info = {}
            for peer in listpeers.get("peers", []):
                peer_id = peer.get("id")
                for ch in peer.get("channels", []):
                    scid = ch.get("short_channel_id")
                    if scid and ch.get("state") == "CHANNELD_NORMAL":
                        peer_info[scid] = {
                            "peer_id": peer_id,
                            "fee_ppm": ch.get("fee_proportional_millionths", 0),
                            "base_fee_msat": ch.get("fee_base_msat", 0),
                            "htlcs": len(ch.get("htlcs", []))
                        }
            
            # Get balances from listfunds
            for channel in listfunds.get("channels", []):
                if channel.get("state") != "CHANNELD_NORMAL":
                    continue
                    
                scid = channel.get("short_channel_id", "")
                if not scid:
                    continue
                
                our_amount_msat = channel.get("our_amount_msat", 0)
                if isinstance(our_amount_msat, str):
                    our_amount_msat = int(our_amount_msat.replace("msat", ""))
                
                amount_msat = channel.get("amount_msat", 0)
                if isinstance(amount_msat, str):
                    amount_msat = int(amount_msat.replace("msat", ""))
                
                info = peer_info.get(scid, {})
                channels[scid] = {
                    "capacity": amount_msat // 1000,
                    "spendable_sats": our_amount_msat // 1000,
                    "peer_id": info.get("peer_id", channel.get("peer_id", "")),
                    "fee_ppm": info.get("fee_ppm", 0),
                    "base_fee_msat": info.get("base_fee_msat", 0),
                    "htlcs": info.get("htlcs", 0)
                }
                
        except Exception as e:
            self.plugin.log(f"Error getting channel balances: {e}", level='error')
        
        return channels

    def execute_rebalance(self, candidate: RebalanceCandidate, **kwargs) -> Dict[str, Any]:
        """
        Execute a rebalance for the given candidate.

        Uses the async JobManager to spawn sling background jobs.
        This plugin acts as the "Strategist" while sling workers handle execution.
        """
        result = {"success": False, "candidate": candidate.to_dict(), "message": ""}
        self._pending[candidate.to_channel] = int(time.time())

        # Thread-safe config snapshot for this execution
        cfg = self.config.snapshot()

        # =====================================================================
        # PHASE 2: Check for Fleet Rebalancing Conflict
        # Avoid competing for same routes as other hive members.
        # INFORMATION ONLY - no fund transfers between nodes.
        # =====================================================================
        fleet_path_info = None
        if self.hive_bridge:
            conflict = self.hive_bridge.check_rebalance_conflict(candidate.to_peer_id)
            if conflict.get("conflict"):
                reason = conflict.get("reason", "Fleet member rebalancing through same peer")
                self.plugin.log(
                    f"FLEET_CONFLICT: Skipping rebalance to {candidate.to_channel[:12]}... "
                    f"({reason})",
                    level='info'
                )
                result["message"] = f"Skipped due to fleet conflict: {reason}"
                result["fleet_conflict"] = True
                del self._pending[candidate.to_channel]
                return result

            # =====================================================================
            # PHASE 7: Query Fleet Rebalance Path
            # Check if routing through fleet members is cheaper.
            # Fleet channels have 0 fees, so internal paths may save significantly.
            # =====================================================================
            fleet_path_info = self.hive_bridge.query_fleet_rebalance_path(
                from_channel=candidate.from_channel,
                to_channel=candidate.to_channel,
                amount_sats=candidate.amount_sats
            )

            if fleet_path_info and fleet_path_info.get("fleet_path_available"):
                savings_pct = fleet_path_info.get("savings_pct", 0)
                fleet_cost = fleet_path_info.get("estimated_fleet_cost_sats", 0)
                external_cost = fleet_path_info.get("estimated_external_cost_sats", 0)

                self.plugin.log(
                    f"FLEET_PATH: Internal route available for {candidate.to_channel[:12]}... "
                    f"Fleet cost: {fleet_cost} sats vs External: {external_cost} sats "
                    f"(savings: {savings_pct:.0f}%)",
                    level='info'
                )

                # Store fleet path info for outcome reporting
                result["fleet_path_available"] = True
                result["fleet_savings_pct"] = savings_pct

        try:
            # Ensure channels are unmanaged from clboss
            # Unmanage ALL source candidates since Sling may use any of them
            for source_scid in candidate.source_candidates:
                # We only have peer_id for primary source, but clboss can work with just SCID
                self.clboss.ensure_unmanaged_for_channel(
                    str(source_scid), str(candidate.primary_source_peer_id), 
                    ClbossTags.FEE_AND_BALANCE, self.database
                )
            self.clboss.ensure_unmanaged_for_channel(
                str(candidate.to_channel), str(candidate.to_peer_id), 
                ClbossTags.FEE_AND_BALANCE, self.database
            )
            
            # --- CRITICAL BUG FIX: Ensure all values are simple types for SQLite ---
            # Assertion guards: Fail-fast on empty/None channel IDs (HO-01)
            assert candidate.from_channel, "from_channel cannot be empty"
            assert candidate.to_channel, "to_channel cannot be empty"
            
            db_from_channel = str(candidate.from_channel)
            db_to_channel = str(candidate.to_channel)
            db_amount = int(candidate.amount_sats)
            db_max_fee = int(candidate.max_budget_sats)
            db_profit = int(candidate.expected_profit_sats)
            
            # Record rebalance attempt in database using SAFE primitives
            rebalance_id = self.database.record_rebalance(
                db_from_channel,
                db_to_channel,
                db_amount,
                db_max_fee,
                db_profit,
                'pending',
                rebalance_type=kwargs.get('rebalance_type', 'normal')
            )

            # CRITICAL-01 FIX: Atomic budget reservation
            # Reserve budget BEFORE starting the job to prevent concurrent overspend
            now = int(time.time())
            since_24h = now - 86400

            # Calculate effective budget (same logic as _check_capital_controls)
            effective_budget = cfg.daily_budget_sats
            if cfg.enable_proportional_budget:
                revenue_24h = self.database.get_total_routing_revenue(since_24h)
                proportional_budget = int(revenue_24h * cfg.proportional_budget_pct)
                effective_budget = max(cfg.daily_budget_sats, proportional_budget)

            reserved, remaining = self.database.reserve_budget(
                reservation_id=rebalance_id,
                amount_sats=db_max_fee,
                channel_id=db_to_channel,
                budget_limit=effective_budget,
                since_timestamp=since_24h
            )

            if not reserved:
                self.database.update_rebalance_result(
                    rebalance_id, 'failed',
                    error_message=f"Budget exhausted: {remaining} sats remaining of {effective_budget}"
                )
                result["message"] = f"Budget exhausted: only {remaining} sats remaining"
                self.plugin.log(
                    f"CAPITAL CONTROL: Budget reservation failed for {db_to_channel}. "
                    f"Remaining: {remaining} sats",
                    level='warn'
                )
                return result

            if cfg.dry_run:
                self.plugin.log(f"[DRY RUN] Would rebalance {candidate.amount_sats} sats "
                              f"from {candidate.from_channel} to {candidate.to_channel}")
                self.database.update_rebalance_result(
                    rebalance_id, 'success', 0, candidate.expected_profit_sats
                )
                return {"success": True, "message": "Dry run", "rebalance_id": rebalance_id}

            # Async execution via JobManager (sling background jobs)
            res = self.job_manager.start_job(candidate, rebalance_id)
            
            if res.get("success"):
                # Update DB status to pending_async
                self.database.update_rebalance_result(rebalance_id, 'pending_async')
                result.update({
                    "success": True, 
                    "message": "Async job started",
                    "rebalance_id": rebalance_id
                })
                self.plugin.log(
                    f"Rebalance job queued: {candidate.to_channel} "
                    f"(job #{self.job_manager.active_job_count})"
                )
            else:
                error = res.get("error", "Failed to start job")
                self.database.update_rebalance_result(
                    rebalance_id, 'failed', error_message=error
                )
                result["message"] = f"Failed: {error}"
                self.plugin.log(f"Failed to start rebalance job: {error}", level='warn')

        except Exception as e:
            result["message"] = str(e)
            self.plugin.log(f"Execution error: {e}", level='error')
        
        return result

    def diagnostic_rebalance(self, channel_id: str) -> Dict[str, Any]:
        """
        Trigger a "Channel Defibrillator" sequence:
        1. Set Fee to 0 (Passive Lure).
        2. Execute small active rebalance (Active Shock).
        
        This is a diagnostic operation to verify channel liveness before
        confirming a channel as a "Zombie" for closure. The small rebalance
        (50k sats) forces liquidity into the channel immediately rather than
        waiting for organic routing traffic.
        """
        self.plugin.log(f"Defibrillator: Triggering Zero-Fee Probe for channel {channel_id}")
        
        # 1. Set the probe flag in the database (Fee Controller will see this and set 0 PPM)
        self.database.set_channel_probe(channel_id, probe_type='zero_fee')
        
        # 2. THE ACTIVE SHOCK: Attempt a small rebalance immediately
        try:
            # Find a healthy source channel
            channels = self._get_channels_with_balances()
            if channel_id not in channels:
                return {"success": False, "message": "Channel not found locally"}

            dest_info = channels[channel_id]
            
            # Find best source (highest spendable sats, excluding target)
            valid_sources = [
                (cid, info) for cid, info in channels.items() 
                if cid != channel_id and info.get('spendable_sats', 0) > 100_000
            ]
            
            if not valid_sources:
                return {
                    "success": True, 
                    "message": "Zero-Fee flag set, but no sources available for active shock."
                }
            
            # Sort by spendable capacity desc, pick the best
            best_source_id, best_source_info = sorted(
                valid_sources, 
                key=lambda x: x[1].get('spendable_sats', 0), 
                reverse=True
            )[0]
            
            # Construct a diagnostic candidate (50k sats - small enough to be OpEx)
            shock_amount = 50_000
            
            # Estimate inbound fee (we accept a loss here, it's a diagnostic cost)
            # Note: outbound_fee is 0 because we set the probe flag above
            inbound_fee = self._estimate_inbound_fee(dest_info.get('peer_id'))
            
            candidate = RebalanceCandidate(
                source_candidates=[best_source_id],
                to_channel=channel_id,
                primary_source_peer_id=best_source_info.get('peer_id', ''),
                to_peer_id=dest_info.get('peer_id', ''),
                amount_sats=shock_amount,
                amount_msat=shock_amount * 1000,
                outbound_fee_ppm=0,
                inbound_fee_ppm=inbound_fee,
                source_fee_ppm=best_source_info.get('fee_ppm', 0),
                weighted_opp_cost_ppm=0,
                spread_ppm=0,  # Likely negative, we don't care for diagnostic
                max_budget_sats=100,  # Cap the diagnostic cost at 100 sats
                max_budget_msat=100_000,
                max_fee_ppm=2000,  # Allow up to 2000ppm for the shock packet
                expected_profit_sats=-50,  # Expect a small loss (diagnostic cost)
                liquidity_ratio=0.5,
                dest_flow_state="diagnostic",
                dest_turnover_rate=0.0,
                source_turnover_rate=0.0
            )
            
            # Capital Controls Check - diagnostic rebalances count against daily budget
            if not self._check_capital_controls():
                self.plugin.log("Defibrillator Active Shock blocked by capital controls", level='warn')
                return {
                    "success": True,
                    "message": "Zero-Fee flag set, but Active Shock blocked: daily budget exhausted or reserve too low"
                }
            
            # Execute Active Shock
            result = self.execute_rebalance(candidate, rebalance_type='diagnostic')
            
            return {
                "success": True, 
                "message": f"Defibrillator active: Zero-Fee flag set + Shock job queued ({result.get('message', 'pending')})"
            }

        except Exception as e:
            self.plugin.log(f"Defibrillator shock failed: {e}", level='error')
            return {
                "success": True, 
                "message": f"Zero-Fee flag set, but active shock failed: {e}"
            }

    def manual_rebalance(self, from_channel: str, to_channel: str,
                         amount_sats: int, max_fee_sats: Optional[int] = None,
                         force: bool = False) -> Dict[str, Any]:
        """Execute a manual rebalance between two channels.

        Note: Manual rebalances bypass capital controls by design (user override),
        but fees ARE recorded and count toward the daily budget for automated rebalances.

        Args:
            from_channel: Source channel ID (where liquidity comes from)
            to_channel: Destination channel ID (where liquidity goes)
            amount_sats: Amount to rebalance in satoshis
            max_fee_sats: Maximum fee willing to pay (optional)
            force: If True, suppress capital control warnings
        """
        # Warn if capital controls would block this (but don't enforce for manual)
        capital_ok = self._check_capital_controls()
        if not capital_ok and not force:
            self.plugin.log(
                "WARNING: Manual rebalance executing despite capital controls. "
                "Budget may be exhausted or reserve low.",
                level='warn'
            )
        
        channels = self._get_channels_with_balances()
        if from_channel not in channels or to_channel not in channels:
            return {"error": "Channels not found"}
            
        f_info = channels[from_channel]
        t_info = channels[to_channel]
        
        fee_ppm = t_info.get("fee_ppm", 0)
        src_ppm = f_info.get("fee_ppm", 0)
        est_in = self._estimate_inbound_fee(t_info.get("peer_id"))
        
        if max_fee_sats is None:
            # Calculate a budget for a manual push based on estimated spread
            max_fee_sats = int(amount_sats * (fee_ppm - est_in - src_ppm) / 1e6)
            if max_fee_sats < 0: 
                max_fee_sats = 100
        
        max_fee_ppm = int(max_fee_sats * 1e6 / amount_sats) if amount_sats > 0 else 0
            
        cand = RebalanceCandidate(
            source_candidates=[from_channel],
            to_channel=to_channel,
            primary_source_peer_id=f_info.get("peer_id", ""),
            to_peer_id=t_info.get("peer_id", ""),
            amount_sats=amount_sats,
            amount_msat=amount_sats * 1000,
            outbound_fee_ppm=fee_ppm,
            inbound_fee_ppm=est_in,
            source_fee_ppm=src_ppm,
            weighted_opp_cost_ppm=0,
            spread_ppm=fee_ppm - est_in - src_ppm,
            max_budget_sats=max_fee_sats,
            max_budget_msat=max_fee_sats * 1000,
            max_fee_ppm=max_fee_ppm,
            expected_profit_sats=0,
            liquidity_ratio=0.5,
            dest_flow_state="manual",
            dest_turnover_rate=0.0,
            source_turnover_rate=0.0
        )
        result = self.execute_rebalance(cand, rebalance_type='manual')

        # Include capital controls warning in result (unless force=True)
        if not capital_ok and not force:
            result['capital_controls_warning'] = "Budget exhausted or reserve low (manual override)"

        return result

    def _check_capital_controls(self, cfg: Optional[ConfigSnapshot] = None) -> bool:
        """Check if capital controls allow rebalancing."""
        if cfg is None:
            cfg = self.config.snapshot()
        try:
            listfunds = self.plugin.rpc.listfunds()
            onchain_sats = 0
            for output in listfunds.get("outputs", []):
                if output.get("status") == "confirmed":
                    amount_msat = output.get("amount_msat", 0)
                    if isinstance(amount_msat, str): 
                        amount_msat = int(amount_msat.replace("msat", ""))
                    onchain_sats += amount_msat // 1000
            
            channel_spendable_sats = 0
            for channel in listfunds.get("channels", []):
                if channel.get("state") != "CHANNELD_NORMAL": 
                    continue
                our_amount_msat = channel.get("our_amount_msat", 0)
                if isinstance(our_amount_msat, str): 
                    our_amount_msat = int(our_amount_msat.replace("msat", ""))
                spendable = our_amount_msat // 1000
                if spendable > 0: 
                    channel_spendable_sats += spendable
            
            total_reserve = onchain_sats + channel_spendable_sats
            if total_reserve < cfg.min_wallet_reserve:
                self.plugin.log(
                    f"CAPITAL CONTROL: Wallet reserve (confirmed on-chain + channel spendable) {total_reserve} < "
                    f"{cfg.min_wallet_reserve}",
                    level='warn'
                )
                return False

            # Calculate effective daily budget
            # If proportional budget enabled: max(fixed_floor, revenue_24h * percentage)
            effective_budget = cfg.daily_budget_sats

            if cfg.enable_proportional_budget:
                now = int(time.time())
                revenue_24h = self.database.get_total_routing_revenue(now - 86400)
                proportional_budget = int(revenue_24h * cfg.proportional_budget_pct)
                effective_budget = max(cfg.daily_budget_sats, proportional_budget)

                self.plugin.log(
                    f"CAPITAL CONTROL: Revenue-proportional budget active. "
                    f"Revenue 24h: {revenue_24h} sats, {cfg.proportional_budget_pct*100:.1f}% = {proportional_budget} sats, "
                    f"Effective budget: {effective_budget} sats (floor: {cfg.daily_budget_sats})",
                    level='debug'
                )
                
            fees_spent_24h = self.database.get_total_rebalance_fees(int(time.time()) - 86400)
            if fees_spent_24h >= effective_budget:
                self.plugin.log(
                    f"CAPITAL CONTROL: Daily budget exceeded "
                    f"({fees_spent_24h} >= {effective_budget})", 
                    level='warn'
                )
                return False
                
        except Exception as e:
            self.plugin.log(f"Error checking capital controls: {e}", level='error')
        return True 
    
    def _is_pending_with_backoff(self, channel_id: str) -> bool:
        """Check if channel has a pending operation with exponential backoff."""
        # Also check job manager for active jobs
        if self.job_manager.has_active_job(channel_id):
            return True
            
        pending_time = self._pending.get(channel_id, 0)
        if pending_time == 0: 
            return False
        
        failure_count, _ = self.database.get_failure_count(channel_id)
        base_cooldown = 600
        cooldown = base_cooldown * (2 ** min(failure_count, 4))
        
        if int(time.time()) - pending_time > cooldown:
            del self._pending[channel_id]
            return False
        return True
    
    # =========================================================================
    # Job Management API (exposed for RPC commands)
    # =========================================================================
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get status of all active rebalance jobs."""
        return self.job_manager.get_all_jobs_status()
    
    def stop_rebalance_job(self, channel_id: str) -> Dict[str, Any]:
        """Manually stop a rebalance job."""
        if self.job_manager.stop_job(channel_id, reason="manual"):
            return {"success": True, "message": f"Stopped job for {channel_id}"}
        return {"success": False, "error": f"No active job for {channel_id}"}
    
    def stop_all_rebalance_jobs(self) -> Dict[str, Any]:
        """Stop all active rebalance jobs."""
        count = self.job_manager.stop_all_jobs(reason="manual_stop_all")
        return {"success": True, "stopped": count}