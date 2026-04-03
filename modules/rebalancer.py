"""
EV-Based Rebalancer module for cl-revenue-ops

MODULE 3: EV-Based Rebalancing (Profit-Aware with Opportunity Cost)

This module implements Expected Value (EV) based rebalancing decisions.
This module only triggers rebalances when the math shows positive expected profit.

Architecture Pattern: "Strategist, Manager, and Driver"
- STRATEGIST (EVRebalancer): Calculates EV, determines IF and HOW MUCH to rebalance
- MANAGER (JobManager): Manages lifecycle of background sling jobs
- DRIVER (Sling plugin): Actually executes the payments in the background

Async Job Queue
- Decouples decision-making from execution
- Allows concurrent rebalancing attempts
- Uses sling-job (background) instead of sling-once (blocking)
"""

import math
import time
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from enum import Enum

from pyln.client import Plugin, RpcError

from .config import Config, ConfigSnapshot
from .database import Database
from .policy_manager import PolicyManager, RebalanceMode, FeeStrategy
from .utils import parse_msat as _shared_parse_msat

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer


class JobStatus(Enum):
    """Status of a sling background job."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


# =============================================================================
# REASON CODES FOR EXPLAINABILITY
# =============================================================================
# Structured reason codes for rebalance decisions. These codes enable
# debugging and auditing of rebalancer behavior.
# =============================================================================

class RebalanceReasonCode(Enum):
    """
    Structured reason codes for rebalance decisions.

    Categories:
    - Success codes: Why a rebalance was attempted
    - Skip codes: Why a channel was skipped for rebalancing
    """
    # Success codes (rebalance was attempted)
    EV_POSITIVE = "ev_positive"                   # Normal EV-positive rebalance

    # Skip codes (rebalance was not attempted)
    SKIP_HARD_BLEEDER = "skip_hard_bleeder"       # Channel is a hard bleeder (rebal_cost > 2x revenue)
    SKIP_SOFT_BLEEDER = "skip_soft_bleeder"       # Channel is a soft bleeder (7d negative, 30d positive)
    SKIP_NO_SOURCE = "skip_no_source"             # No profitable source channel found
    SKIP_EV_NEGATIVE = "skip_ev_negative"         # Expected value is negative
    SKIP_COOLDOWN = "skip_cooldown"               # Channel is in rebalance cooldown period
    SKIP_POLICY_DISABLED = "skip_policy_disabled" # Rebalancing disabled by policy
    SKIP_FUTILITY_BREAKER = "skip_futility_breaker"  # Too many consecutive failures
    SKIP_ZOMBIE = "skip_zombie"                   # Channel classified as zombie
    SKIP_UNDERWATER = "skip_underwater"           # Channel is underwater with negative marginal ROI
    SKIP_SINK = "skip_sink"                       # Channel is a sink (filling for free)
    SKIP_CONGESTED = "skip_congested"             # Channel is HTLC congested
    SKIP_UNSTABLE_PEER = "skip_unstable_peer"     # Peer has low uptime


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

    # EV v2.0: The cost assumption used to compute expected_profit_sats.
    # Needed by _handle_job_success to reconcile expected vs actual profit correctly.
    # Defaults to 0, meaning reconciliation falls back to max_budget_sats (old behavior).
    expected_fee_sats: int = 0

    # EV-derived max fee PPM before graduated escalation.
    # Used by adaptive chunk sizing to scale chunk inversely with fee escalation.
    ev_base_fee_ppm: int = 0

    # Dynamic hot-channel protection metadata (optional)
    hot_channel_protection: bool = False
    hot_channel_protection_score: float = 0.0
    dynamic_budget_override_sats: int = 0  # Candidate-specific rebalance budget cap (24h)
    dynamic_channel_profit_budget_sats: int = 0  # Derived from channel contribution * pct
    recommended_cooldown_hours: float = 0.0

    # Explainability fields
    reason_code: str = RebalanceReasonCode.EV_POSITIVE.value  # Why this rebalance was approved
    bleeder_status: str = "none"  # 'hard', 'soft', or 'none'

    # Direction: "pull" fills to_channel from sources; "push" drains to_channel to destinations
    direction: str = "pull"

    # Multi-source peer IDs aligned with source_candidates (best-first).
    # Optional for backward compatibility; when absent, callers may fall back to primary_source_peer_id.
    source_candidate_peer_ids: List[str] = field(default_factory=list)

    # Hive route discovery: if askrene found a cheap fleet route, store hop count
    # so start_job can tighten sling's maxhops for better route matching.
    hive_route_hops: int = 0

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
            "source_candidate_peer_ids": self.source_candidate_peer_ids,
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
            "num_source_candidates": len(self.source_candidates),
            "reason_code": self.reason_code,
            "bleeder_status": self.bleeder_status,
            "direction": self.direction,
            "hot_channel_protection": self.hot_channel_protection,
            "hot_channel_protection_score": round(self.hot_channel_protection_score, 4),
            "dynamic_budget_override_sats": self.dynamic_budget_override_sats,
            "dynamic_channel_profit_budget_sats": self.dynamic_channel_profit_budget_sats,
            "recommended_cooldown_hours": round(self.recommended_cooldown_hours, 2) if self.recommended_cooldown_hours else 0.0,
            "expected_fee_sats": self.expected_fee_sats,
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
    direction: str = "pull"

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

    Key Design Decision:
    We use sling-job for TACTICAL rebalancing (one-off moves), not permanent
    pegging. As soon as any successful payment is detected or timeout is reached,
    we DELETE the job to prevent infinite spending.
    """

    # Default timeout: 2 hours (configurable)
    DEFAULT_JOB_TIMEOUT_SECONDS = 7200

    def __init__(self, plugin: Plugin, config: Config, database: Database):
        self.plugin = plugin
        self.config = config
        self.database = database

        # Active jobs indexed by target channel SCID (normalized format)
        self._active_jobs: Dict[str, ActiveJob] = {}
        self._jobs_lock = threading.Lock()

        # Configurable settings
        self.job_timeout_seconds = getattr(config, 'sling_job_timeout_seconds',
                                           self.DEFAULT_JOB_TIMEOUT_SECONDS)
        self.max_concurrent_jobs = getattr(config, 'max_concurrent_jobs', 5)

        # Chunk size for sling rebalances (sats per attempt)
        self.chunk_size_sats = getattr(config, 'sling_chunk_size_sats', 500000)

        # AskRene integration (read-only): use constraints for preflight sizing + intelligence.
        self._askrene_cache_ts = 0
        self._askrene_cache: Dict[str, int] = {}  # short_channel_id_dir -> maximum_msat
        self._askrene_lock = threading.Lock()  # TS-5: Protect cache access
        self.askrene_layer = getattr(config, 'askrene_layer', 'xpay')
        self.askrene_max_age_sec = getattr(config, 'askrene_max_age_sec', 900)

        # Source reliability tracking
        self.source_failure_counts: Dict[str, float] = {}
        self._source_failures_lock = threading.Lock()
        self.last_decay_time = time.time()

        # Periodic exclusion sync tracking
        self._last_exclusion_sync: float = 0
        self._policy_manager_ref = None

        # HiveRouter for askrene job reservations (injected by EVRebalancer)
        self.hive_router = None

    @staticmethod
    def _classify_sling_error(error_msg: str) -> str:
        """Classify a sling error message for failure-informed routing."""
        msg = error_msg.lower()
        if any(s in msg for s in ("no route", "unknown_next_peer", "no path", "no channels")):
            return "no_route"
        if any(s in msg for s in ("timeout", "timed out", "deadline")):
            return "timeout"
        if any(s in msg for s in ("exceeded", "budget", "overpaid")):
            return "budget_exceeded"
        return "other"

    @staticmethod
    def _scale_chunk_for_escalation(base_chunk: int, base_ppm: int, actual_ppm: int, min_amount: int) -> int:
        """
        Scale chunk size inversely with fee escalation to keep per-attempt cost constant.

        If we're paying 3x the base fee rate, use 1/3 the chunk size.
        """
        if base_ppm <= 0 or actual_ppm <= base_ppm:
            return base_chunk
        scaled = int(base_chunk * base_ppm / actual_ppm)
        return max(min_amount, scaled)

    def get_active_rebalancing_peers(self) -> List[str]:
        """Get deduplicated peer IDs from all active jobs (source + dest peers)."""
        peers = set()
        with self._jobs_lock:
            for job in self._active_jobs.values():
                if job is not None and job.candidate:
                    peers.add(job.candidate.to_peer_id)
                    peers.add(job.candidate.primary_source_peer_id)
        return list(peers)

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
        with self._source_failures_lock:
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
        with self._jobs_lock:
            return len(self._active_jobs)

    @property
    def active_channels(self) -> List[str]:
        """Returns list of channel SCIDs with active jobs."""
        with self._jobs_lock:
            return list(self._active_jobs.keys())

    def has_active_job(self, channel_id: str) -> bool:
        """Check if a channel has an active rebalance job."""
        normalized = self._normalize_scid(channel_id)
        with self._jobs_lock:
            return normalized in self._active_jobs

    def slots_available(self) -> int:
        """Returns number of available job slots."""
        with self._jobs_lock:
            return max(0, self.max_concurrent_jobs - len(self._active_jobs))
    
    def _normalize_scid(self, scid: str) -> str:
        """Normalize SCID to consistent format (with 'x' separators)."""
        return scid.replace(':', 'x')
    
    def _to_sling_scid(self, scid: str) -> str:
        """Normalize SCID to sling's expected 'x' separator format."""
        return self._normalize_scid(scid)
    
    def _get_channel_local_balance(self, channel_id: str) -> int:
        """Get current local balance of a channel in sats.

        Only returns balance for channels in CHANNELD_NORMAL state (or if state
        is not reported). Returns 0 for channels in closing/non-normal states.
        """
        try:
            listfunds = self.plugin.rpc.listfunds()
            normalized = self._normalize_scid(channel_id)

            for channel in listfunds.get("channels", []):
                scid = channel.get("short_channel_id", "")
                if self._normalize_scid(scid) == normalized:
                    # I-8 FIX: Filter out non-normal channels to prevent false
                    # positive success detection during channel closure
                    ch_state = channel.get("state")
                    if ch_state and ch_state != "CHANNELD_NORMAL":
                        return 0
                    # M-22: Use _parse_msat for consistent Millisatoshi handling
                    our_amount_msat = self._parse_msat(channel.get("our_amount_msat", 0))
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

        # H-1: Atomic check-and-reserve under lock to prevent TOCTOU race
        with self._jobs_lock:
            if normalized_scid in self._active_jobs:
                return {"success": False, "error": "Job already exists for this channel"}
            if len(self._active_jobs) >= self.max_concurrent_jobs:
                return {"success": False, "error": "No job slots available"}
            # Reserve slot with sentinel timestamp immediately
            self._active_jobs[normalized_scid] = time.time()

        # Convert SCIDs to sling format (x-separated, e.g., 930866x2599x2)
        to_scid = self._to_sling_scid(candidate.to_channel)
        
        # Convert all source candidates to sling format
        source_scids_sling = [self._to_sling_scid(scid) for scid in candidate.source_candidates]
        
        # Get initial balance for progress tracking
        initial_balance = self._get_channel_local_balance(candidate.to_channel)
        
        # Calculate chunk size (amount per rebalance attempt)
        chunk_size = min(candidate.amount_sats, self.chunk_size_sats)

        # Adaptive chunk sizing: reduce chunk when fee has been escalated
        # to keep per-attempt fee risk constant.
        ev_base_ppm = getattr(candidate, 'ev_base_fee_ppm', 0)
        if ev_base_ppm > 0 and candidate.max_fee_ppm > ev_base_ppm:
            chunk_size = self._scale_chunk_for_escalation(
                base_chunk=chunk_size,
                base_ppm=ev_base_ppm,
                actual_ppm=candidate.max_fee_ppm,
                min_amount=getattr(self.config, 'rebalance_min_amount', 50000),
            )

        # I-4 FIX: Scale budget proportionally to chunk size. The EV analysis computed
        # max_budget_sats for the full candidate.amount_sats. If we chunk into smaller
        # pieces, the per-chunk budget must be proportionally smaller to maintain the
        # same effective PPM cap. Without this, the PPM cap is inflated by the ratio
        # of full_amount / chunk_size, allowing sling to overpay on routes.
        if chunk_size > 0 and candidate.amount_sats > 0:
            chunk_budget_msat = (candidate.max_budget_msat * chunk_size) // candidate.amount_sats
            budget_ppm = (chunk_budget_msat * 1_000_000) // (chunk_size * 1000) if chunk_size > 0 else 0
        else:
            budget_ppm = 0
        maxppm = int(max(1, min(candidate.max_fee_ppm, budget_ppm))) if budget_ppm > 0 else 0
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
            flow_state = candidate.dest_flow_state
            direction = candidate.direction

            if direction == "push":
                # Push: target is the local balance ratio to drain DOWN to
                if flow_state == "source":
                    target = 1.0 - self.config.sling_target_source
                elif flow_state == "sink":
                    target = 1.0 - self.config.sling_target_sink
                else:
                    target = 1.0 - self.config.sling_target_balanced
            else:
                # Pull: existing logic unchanged
                if flow_state == "sink":
                    target = self.config.sling_target_sink
                elif flow_state == "source":
                    target = self.config.sling_target_source
                else:
                    target = self.config.sling_target_balanced

            self.plugin.log(
                f"Starting sling-job: {to_scid} {'<-' if direction == 'pull' else '->'} "
                f"[{len(source_scids_sling)} candidates], "
                f"primary={primary_source}, dir={direction}, amount={chunk_size}, "
                f"maxppm={maxppm}, maxhops={self.config.sling_max_hops}, "
                f"target={target} (flow={flow_state}), budget_sats={candidate.max_budget_sats}"
            )

            # =================================================================
            # PHASE 6: Enhanced Sling Parameters
            # =================================================================
            # - maxhops: Shorter routes are faster and more reliable
            # - target: Flow-aware balance target
            # - outppm: Cap per-hop fee; also used as sole source signal when no candidates
            # =================================================================
            # If askrene found a hive route, tighten maxhops so sling
            # discovers the same short path through fleet peers.
            maxhops = self.config.sling_max_hops
            if candidate.hive_route_hops > 0:
                maxhops = min(maxhops, candidate.hive_route_hops + 1)
                self.plugin.log(
                    f"HIVE ROUTE: Tightening maxhops to {maxhops} "
                    f"(askrene found {candidate.hive_route_hops}-hop route)",
                    level='debug'
                )

            job_params = {
                "scid": to_scid,
                "direction": direction,
                "amount": chunk_size,
                "maxppm": maxppm,
                "maxhops": maxhops,
                "target": target,
                "paralleljobs": self.config.sling_parallel_jobs,
            }

            # Flow-aware depletion: how aggressively to drain source candidates
            if flow_state == "sink":
                deplete_pct = self.config.sling_deplete_pct_sink
            elif flow_state == "source":
                deplete_pct = self.config.sling_deplete_pct_source
            else:
                deplete_pct = self.config.sling_deplete_pct_balanced

            job_params["depleteuptopercent"] = deplete_pct

            # EV v2.0: Asymmetric depletion tied to actual source excess capital.
            # Instead of a fixed chunk_size * 2 limit, compute how much the primary
            # source is overweight and allow Sling to drain exactly that amount.
            try:
                primary_source_scid = candidate.source_candidates[0] if candidate.source_candidates else None
                source_state = self.database.get_channel_state(primary_source_scid) if primary_source_scid else None
                if source_state and source_state.get("capacity", 0) > 0:
                    source_cap = source_state["capacity"]
                    source_bal = self._get_channel_local_balance(primary_source_scid)
                    # Use the SOURCE channel's own flow state (not the destination's)
                    source_flow = source_state.get("state", "balanced")
                    if source_flow == "source":
                        source_target_ratio = self.config.sling_target_source
                    elif source_flow == "sink":
                        source_target_ratio = self.config.sling_target_sink
                    else:
                        source_target_ratio = self.config.sling_target_balanced
                    source_target_sats = int(source_cap * source_target_ratio)
                    actual_excess = max(0, source_bal - source_target_sats)
                    job_params["depleteuptoamount"] = max(100_000, actual_excess)
                else:
                    job_params["depleteuptoamount"] = max(100_000, chunk_size * 2)
            except Exception:
                job_params["depleteuptoamount"] = max(100_000, chunk_size * 2)

            # Add candidates if we have them
            if source_scids_sling:
                job_params["candidates"] = source_scids_sling

            # Add outppm to cap source-side fee (if configured)
            if self.config.sling_outppm_fallback > 0:
                if not source_scids_sling:
                    # No candidates - outppm is the only source-selection signal
                    job_params["outppm"] = self.config.sling_outppm_fallback
                    self.plugin.log(
                        f"No candidates for {to_scid}, using outppm={self.config.sling_outppm_fallback} fallback",
                        level='info'
                    )
                else:
                    # Have candidates; outppm still caps per-hop fee for all routes
                    job_params["outppm"] = self.config.sling_outppm_fallback

            self.plugin.rpc.call("sling-job", job_params)
            
            # Start the job (sling-job only creates it, sling-go starts it)
            try:
                self.plugin.rpc.call("sling-go", {"scid": to_scid})
            except RpcError as e:
                # sling-go might fail if job auto-started, that's OK
                if "already running" not in str(e).lower():
                    self.plugin.log(f"sling-go warning: {e}", level='debug')
            
            # Track the job with all source candidates (replace sentinel with real job)
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
                status=JobStatus.RUNNING,
                direction=direction,
            )
            with self._jobs_lock:
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
            # H-1: Remove sentinel on failure
            with self._jobs_lock:
                self._active_jobs.pop(normalized_scid, None)
            error_msg = str(e)
            self.plugin.log(f"Failed to start sling-job: {error_msg}", level='warn')
            return {"success": False, "error": f"Sling RPC error: {error_msg}"}
        except Exception as e:
            # H-1: Remove sentinel on failure
            with self._jobs_lock:
                self._active_jobs.pop(normalized_scid, None)
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

        # L-3: Atomic pop under lock first, then do RPC calls with the removed job
        with self._jobs_lock:
            job = self._active_jobs.pop(normalized, None)

        if not job or job is None:  # None = sentinel from start_job
            return False

        sling_stopped = False
        try:
            # First stop the job gracefully
            try:
                self.plugin.rpc.call("sling-stop", {"scid": job.scid})
                sling_stopped = True
            except RpcError:
                sling_stopped = True  # RpcError means sling responded (job may already be stopped)

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
            # C-3 FIX: If sling is unreachable (timeout, connection error), the job
            # may still be running and spending fees. Re-add to _active_jobs so
            # monitor_jobs continues tracking it. RpcError is fine (sling responded),
            # but other exceptions (timeout, connection) mean we couldn't confirm the stop.
            if not sling_stopped:
                with self._jobs_lock:
                    if normalized not in self._active_jobs:
                        self._active_jobs[normalized] = job
                        self.plugin.log(
                            f"Re-added job {job.scid} to tracking (sling unreachable, "
                            f"job may still be running)",
                            level='warn'
                        )
                return False

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
            with self._source_failures_lock:
                for scid in list(self.source_failure_counts.keys()):
                    self.source_failure_counts[scid] *= 0.5
                    if self.source_failure_counts[scid] < 0.1:
                        del self.source_failure_counts[scid]
            self.last_decay_time = now

        # Periodic exclusion sync (every 30 minutes)
        if now - self._last_exclusion_sync > 1800:
            try:
                self.sync_peer_exclusions(self._policy_manager_ref)
                self.sync_channel_exclusions()
            except Exception as e:
                self.plugin.log(f"Periodic exclusion sync error: {e}", level='debug')
            self._last_exclusion_sync = now

        # M-1: Snapshot active jobs under lock for thread-safe iteration
        # C-2 FIX: Also clean up stale sentinel entries (None values) from stuck start_job calls.
        # If a sling RPC hangs during start_job, the sentinel blocks the slot indefinitely.
        sentinel_timeout = 300  # 5 minutes
        now_sentinel = time.time()
        with self._jobs_lock:
            stale_sentinels = [
                k for k, v in self._active_jobs.items()
                if not isinstance(v, ActiveJob) and (
                    v is None or  # Legacy None sentinels (always stale)
                    (isinstance(v, (int, float)) and now_sentinel - v > sentinel_timeout)
                )
            ]
            for k in stale_sentinels:
                del self._active_jobs[k]
                self.plugin.log(
                    f"Cleaned up stale sentinel for {k} (stuck start_job?)",
                    level='info'
                )
            jobs_snapshot = {k: v for k, v in self._active_jobs.items() if isinstance(v, ActiveJob)}

        if not jobs_snapshot:
            return summary

        # Get sling stats for all jobs (using snapshot)
        sling_stats = self._get_sling_stats(jobs_snapshot)

        # Hoist listfunds to avoid N+1 RPC calls (per-job balance checks).
        local_balances = self._get_local_balances_map()

        for normalized_scid, job in jobs_snapshot.items():
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
            current_balance = local_balances.get(job.scid_normalized)

            # C-4 FIX: Detect channel closure (not in CHANNELD_NORMAL state).
            # _get_local_balances_map only includes CHANNELD_NORMAL channels,
            # so a missing key means the channel is closing/closed.
            if current_balance is None:
                self.plugin.log(
                    f"Job {job.scid}: channel no longer in CHANNELD_NORMAL state, "
                    f"terminating job immediately",
                    level='info'
                )
                self._handle_job_failure(job, {})
                summary["failed"] += 1
                continue

            # For pull: positive delta = liquidity gained (success)
            # For push: negative delta = liquidity drained (success), so flip sign
            raw_delta = current_balance - job.initial_local_sats
            balance_delta = -raw_delta if job.direction == "push" else raw_delta
            
            # Get job-specific stats from sling
            job_stats = sling_stats.get(job.scid, {})

            # ZERO-TOLERANCE: Abort if the job is spending at/above its msat budget.
            # H-1: Use _parse_msat/_parse_sats to handle string "Nmsat" values
            fee_msat = self._parse_msat(job_stats.get("fee_total_msat"))
            if not fee_msat:
                fee_sats = self._parse_sats(job_stats.get("fee_total_sats"))
                fee_msat = fee_sats * 1000 if fee_sats else 0

            if fee_msat and job.candidate and fee_msat > job.candidate.max_budget_msat:
                self._handle_job_budget_exceeded(job, fee_msat, job_stats)
                summary["failed"] += 1
                continue
            
            # Check for success:
            # - Prefer sling stats when they provide explicit success signals (less fragile than balance deltas,
            #   which can be masked by concurrent forwarding).
            # - Fallback to positive balance delta if stats don't expose success.
            stats_success_amount = self._extract_success_amount_sats(job_stats)
            stats_success_count = self._extract_success_count(job_stats)
            if stats_success_amount is not None and stats_success_amount > 0:
                amount_transferred = max(balance_delta, stats_success_amount)
                self._handle_job_success(job, amount_transferred, job_stats)
                summary["completed"] += 1
                continue
            if stats_success_count > 0:
                amount_transferred = balance_delta
                if amount_transferred <= 0:
                    # We know at least one payment succeeded, but local balance may not reflect it due to
                    # concurrent routing. Use the job's intended chunk as a conservative proxy.
                    amount_transferred = max(1, min(int(job.target_amount_sats or 0), self.chunk_size_sats))
                self._handle_job_success(job, amount_transferred, job_stats)
                summary["completed"] += 1
                continue
            if balance_delta > 0:
                self._handle_job_success(job, balance_delta, job_stats)
                summary["completed"] += 1
                continue
            
            # Check for sling-reported errors
            if self._check_job_error(job, job_stats):
                self._handle_job_failure(job, job_stats)
                summary["failed"] += 1
                continue
            
            # Job still running
            summary["still_running"] += 1
            failure_count = self._extract_failure_count(job_stats)
            fee_ppm = self._extract_fee_ppm(job_stats)
            self.plugin.log(
                f"Job {job.scid} running: {elapsed}s elapsed, "
                f"transferred={balance_delta} sats, failures={failure_count}"
                + (f", avg_ppm={fee_ppm}" if fee_ppm else ""),
                level='debug'
            )
        
        return summary

    def _get_local_balances_map(self) -> Dict[str, int]:
        """Return a map of normalized scid -> local balance sats (single listfunds call).

        Only includes channels in CHANNELD_NORMAL state. Channels that are closing
        or in other non-normal states are excluded so that monitor_jobs can detect
        channel closures (missing key = channel no longer normal).
        """
        balances: Dict[str, int] = {}
        try:
            listfunds = self.plugin.rpc.listfunds()
            for channel in listfunds.get("channels", []):
                scid = channel.get("short_channel_id", "")
                if not scid:
                    continue
                # C-4 FIX: Exclude channels in known non-normal states.
                # This allows monitor_jobs to detect channel closures via missing keys.
                # If state is absent (old CLN or test mock), include the channel.
                ch_state = channel.get("state")
                if ch_state and ch_state != "CHANNELD_NORMAL":
                    continue
                # M-22b: Use _parse_msat for consistent Millisatoshi handling
                our_amount_msat = self._parse_msat(channel.get("our_amount_msat", 0))
                balances[self._normalize_scid(scid)] = our_amount_msat // 1000
        except Exception as e:
            self.plugin.log(f"Error preloading channel balances: {e}", level='debug')
        return balances

    def _parse_msat(self, v: Any) -> int:
        """L-18: Delegate to shared parse_msat in utils.py."""
        return _shared_parse_msat(v)

    def _parse_sats(self, v: Any) -> int:
        if v is None:
            return 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s.endswith("sat"):
                s = s[:-3]
            try:
                return int(s)
            except Exception:
                return 0
        try:
            return int(v)
        except Exception:
            return 0

    def _extract_success_amount_sats(self, stats: Dict[str, Any]) -> Optional[int]:
        """
        Extract how much has been successfully transferred from sling stats.

        Preferred source: per-scid ``successes_in_time_window.total_amount_sats``
        (returned by ``sling-stats scid=X json=true``).  Falls back to legacy
        flat keys for older sling versions or bulk-query results.
        """
        if not stats:
            return None

        # Per-scid detailed stats (from per-job query)
        successes = stats.get("successes_in_time_window")
        if isinstance(successes, dict):
            sats = self._parse_sats(successes.get("total_amount_sats"))
            if sats > 0:
                return sats

        # Fallback: bulk query fields (legacy/version compat)
        for k in ("success_total_msat", "successful_total_msat"):
            if k in stats:
                msat = self._parse_msat(stats[k])
                if msat > 0:
                    return msat // 1000

        for k in ("success_total_sats", "successful_total_sats"):
            if k in stats:
                sats = self._parse_sats(stats[k])
                if sats > 0:
                    return sats

        return None

    def _extract_success_count(self, stats: Dict[str, Any]) -> int:
        """Extract successful rebalance count from sling stats."""
        if not stats:
            return 0

        # Per-scid detailed stats
        successes = stats.get("successes_in_time_window")
        if isinstance(successes, dict):
            n = self._parse_sats(successes.get("total_rebalances"))
            if n > 0:
                return n

        # Fallback
        for k in ("success_count", "successful_payments"):
            if k in stats:
                n = self._parse_sats(stats[k])
                if n > 0:
                    return n

        return 0

    def _extract_failure_count(self, stats: Dict[str, Any]) -> int:
        """Extract failure count from sling per-scid stats."""
        if not stats:
            return 0

        failures = stats.get("failures_in_time_window")
        if isinstance(failures, dict):
            n = self._parse_sats(failures.get("total_rebalances"))
            if n > 0:
                return n

        # Fallback
        for k in ("consecutive_failures", "failure_count"):
            if k in stats:
                n = self._parse_sats(stats[k])
                if n > 0:
                    return n

        return 0

    def _extract_fee_ppm(self, stats: Dict[str, Any]) -> Optional[int]:
        """Extract weighted average fee PPM from sling per-scid stats."""
        if not stats:
            return None

        successes = stats.get("successes_in_time_window")
        if isinstance(successes, dict):
            ppm = self._parse_sats(successes.get("feeppm_weighted_avg"))
            if ppm > 0:
                return ppm

        return None

    def _get_sling_stats(self, jobs_snapshot: Optional[Dict[str, 'ActiveJob']] = None) -> Dict[str, Dict[str, Any]]:
        """Query sling-stats for all active jobs, returning dict keyed by SCID.

        Preferred approach: per-job ``sling-stats scid=<scid> json=true``
        which returns a known schema with ``successes_in_time_window`` /
        ``failures_in_time_window`` nested dicts.  Falls back to the bulk
        ``sling-stats json=true`` call if per-scid queries fail.

        Args:
            jobs_snapshot: Optional pre-taken snapshot of active jobs. If None,
                          takes a snapshot under lock.
        """
        stats: Dict[str, Dict[str, Any]] = {}

        # M-1: Use provided snapshot or take one under lock
        if jobs_snapshot is None:
            with self._jobs_lock:
                jobs_snapshot = {k: v for k, v in self._active_jobs.items() if isinstance(v, ActiveJob)}

        # Per-job detailed stats (preferred — returns known schema)
        for normalized_scid, job in jobs_snapshot.items():
            try:
                result = self.plugin.rpc.call("sling-stats", {
                    "scid": job.scid,
                    "json": True
                })
                if isinstance(result, dict):
                    stats[job.scid] = result
            except Exception:
                pass  # Fall through to bulk query

        # Bulk fallback if per-job queries failed
        if not stats:
            try:
                result = self.plugin.rpc.call("sling-stats", {"json": True})
                if isinstance(result, dict):
                    if "jobs" in result:
                        for job_data in result["jobs"]:
                            scid = job_data.get("scid", "")
                            if scid:
                                stats[scid] = job_data
                    else:
                        stats = result
                elif isinstance(result, list):
                    for job_data in result:
                        scid = job_data.get("scid", "")
                        if scid:
                            stats[scid] = job_data
            except Exception as e:
                self.plugin.log(f"sling-stats error: {e}", level='debug')

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
        consecutive_failures = self._parse_sats(stats.get("consecutive_failures", 0))
        if consecutive_failures >= 10:
            return True
        
        return False
    
    def _handle_job_success(self, job: ActiveJob, amount_transferred: int, 
                            stats: Dict[str, Any]) -> None:
        """Handle a successfully completed job."""
        # Calculate actual fee paid (from sling stats if available)
        # H-1b: Use _parse_sats/_parse_msat to handle string "Nmsat" values
        fee_sats = self._parse_sats(stats.get("fee_total_sats"))
        if not fee_sats:
            fee_msat = self._parse_msat(stats.get("fee_total_msat"))
            fee_sats = fee_msat // 1000 if fee_msat else 0
        if not fee_sats:
            # B5 FIX: total_spent_sats includes principal + fee.
            # Derive fee as total_spent - amount_transferred.
            successes = stats.get("successes_in_time_window")
            if isinstance(successes, dict):
                total_spent = self._parse_sats(successes.get("total_spent_sats"))
                if total_spent and total_spent > amount_transferred:
                    fee_sats = total_spent - amount_transferred
        
        # Estimate fee from amount if sling doesn't report it
        if fee_sats == 0 and amount_transferred > 0:
            # Conservative estimate: half of max_fee_ppm (actual is usually well below max)
            fee_sats = (amount_transferred * job.max_fee_ppm) // 2_000_000
        
        # Calculate actual profit
        # expected_profit was computed as: income - expected_fee - source_loss
        # Actual profit replaces expected_fee with the real fee paid.
        # B6 FIX: When expected_fee_sats==0 (legacy candidates), use fee_sats
        # to avoid inflation from max_budget_sats fallback.
        expected_profit = job.candidate.expected_profit_sats
        assumed_fee = job.candidate.expected_fee_sats or fee_sats
        actual_profit = expected_profit + (assumed_fee - fee_sats)
        
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
            with self._source_failures_lock:
                if primary_source in self.source_failure_counts:
                    # Significant reduction (rewarding success)
                    self.source_failure_counts[primary_source] = 0.0

        # Mark budget reservation as spent (CRITICAL-01 fix)
        # H-4: Ensure reservation_id is str to match DB column type
        self.database.mark_budget_spent(str(job.rebalance_id), fee_sats)

        # Release askrene reservation before stopping
        if self.hive_router:
            job_ref = self._active_jobs.get(job.scid_normalized)
            if job_ref and hasattr(job_ref, 'candidate') and job_ref.candidate:
                self.hive_router.unreserve_for_job(
                    job_ref.scid, job_ref.candidate.amount_msat,
                )

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
        error_type = self._classify_sling_error(error_msg)
        self.database.increment_failure_count(
            job.scid_normalized,
            attempted_ppm=job.max_fee_ppm,
            attempted_amount=job.candidate.amount_sats if job.candidate else 0,
            error_type=error_type,
        )

        # Track source failure for reliability scoring
        if job.candidate and job.candidate.source_candidates:
            # Penalize the primary source
            primary_source = job.candidate.source_candidates[0]
            with self._source_failures_lock:
                self.source_failure_counts[primary_source] = self.source_failure_counts.get(primary_source, 0.0) + 1.0

        # B7 FIX: Check for partial fee spend before releasing budget.
        partial_fee_sats = 0
        fee_msat = self._parse_msat(stats.get("fee_total_msat"))
        if fee_msat:
            partial_fee_sats = fee_msat // 1000
        if not partial_fee_sats:
            partial_fee_sats = self._parse_sats(stats.get("fee_total_sats"))

        if partial_fee_sats > 0:
            self.database.mark_budget_spent(str(job.rebalance_id), partial_fee_sats)
        else:
            self.database.release_budget_reservation(str(job.rebalance_id))

        # Release askrene reservation before stopping
        if self.hive_router:
            job_ref = self._active_jobs.get(job.scid_normalized)
            if job_ref and hasattr(job_ref, 'candidate') and job_ref.candidate:
                self.hive_router.unreserve_for_job(
                    job_ref.scid, job_ref.candidate.amount_msat,
                )

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
            with self._source_failures_lock:
                self.source_failure_counts[primary_source] = self.source_failure_counts.get(primary_source, 0.0) + 1.0

        # Record actual rebalance cost even on budget failure.
        # Compute actual amount transferred from balance delta (not 0)
        # to avoid distorting cost-per-sat metrics.
        actual_cost_sats = (fee_msat + 999) // 1000
        current_balance = self._get_channel_local_balance(job.scid_normalized)
        raw_delta = current_balance - job.initial_local_sats
        amount_transferred = max(0, -raw_delta if job.direction == "push" else raw_delta)
        if actual_cost_sats > 0 and job.candidate:
            try:
                dest_peer_id = job.candidate.to_peer_id
                self.database.record_rebalance_cost(
                    job.scid_normalized, dest_peer_id,
                    actual_cost_sats, amount_transferred)
            except Exception as e:
                self.plugin.log(f"Failed to record rebalance cost: {e}", level='debug')

        # L-17: Budget was actually spent (overspent), mark as spent not released
        # H-4: Ensure reservation_id is str to match DB column type
        self.database.mark_budget_spent(str(job.rebalance_id), actual_cost_sats)

        # Release askrene reservation before stopping
        if self.hive_router:
            job_ref = self._active_jobs.get(job.scid_normalized)
            if job_ref and hasattr(job_ref, 'candidate') and job_ref.candidate:
                self.hive_router.unreserve_for_job(
                    job_ref.scid, job_ref.candidate.amount_msat,
                )

        # Stop the job
        self.stop_job(job.scid_normalized, reason="exceeded_budget")

    def _handle_job_timeout(self, job: ActiveJob) -> None:
        """Handle a timed-out job."""
        elapsed_hours = (int(time.time()) - job.start_time) / 3600

        # Check if any progress was made
        # For push: negative delta = liquidity drained (success), so flip sign
        current_balance = self._get_channel_local_balance(job.scid_normalized)
        raw_delta = current_balance - job.initial_local_sats
        amount_transferred = -raw_delta if job.direction == "push" else raw_delta
        fee_sats = 0  # MA-9: Initialize before branching so it's always defined

        if amount_transferred > 0:
            # Partial success - try to get actual fee from sling stats
            job_stats = self._get_sling_stats()
            stats = job_stats.get(job.scid, {})
            fee_sats = 0
            successes = stats.get("successes_in_time_window")
            if isinstance(successes, dict):
                fee_sats = self._parse_sats(successes.get("total_spent_sats"))
            if fee_sats == 0:
                # H-1c: Use _parse_msat to handle string "Nmsat" values
                fee_msat = self._parse_msat(stats.get("fee_total_msat"))
                fee_sats = fee_msat // 1000 if fee_msat else 0
            if fee_sats == 0 and amount_transferred > 0:
                # Conservative estimate: half of max fee
                fee_sats = (amount_transferred * job.max_fee_ppm) // 2_000_000

            self.plugin.log(
                f"Rebalance TIMEOUT (partial): {job.scid} after {elapsed_hours:.1f}h. "
                f"Transferred {amount_transferred} sats, fee ~{fee_sats} sats."
            )
            self.database.update_rebalance_result(
                job.rebalance_id,
                'partial',
                actual_fee_sats=fee_sats,
                actual_profit_sats=0
            )
            # E3 FIX: Record cost in rebalance_costs for lifetime accounting,
            # matching _handle_job_success (line 1001) and _handle_job_budget_exceeded (line 1100).
            # Without this, partial timeout fees are lost from budget tracking.
            if fee_sats > 0:
                self.database.record_rebalance_cost(
                    channel_id=job.scid_normalized,
                    peer_id=job.candidate.to_peer_id,
                    cost_sats=fee_sats,
                    amount_sats=amount_transferred,
                    timestamp=int(time.time())
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

        # M-15: Partial success spent real fees; only release if no progress
        # H-4: Ensure reservation_id is str to match DB column type
        if amount_transferred > 0:
            self.database.mark_budget_spent(str(job.rebalance_id), fee_sats)
        else:
            self.database.release_budget_reservation(str(job.rebalance_id))

        # Release askrene reservation before stopping
        if self.hive_router:
            job_ref = self._active_jobs.get(job.scid_normalized)
            if job_ref and hasattr(job_ref, 'candidate') and job_ref.candidate:
                self.hive_router.unreserve_for_job(
                    job_ref.scid, job_ref.candidate.amount_msat,
                )

        # Stop the job
        self.stop_job(job.scid_normalized, reason="timeout")
    
    def stop_all_jobs(self, reason: str = "shutdown") -> int:
        """Stop all active jobs and release their budget reservations. Returns count of jobs stopped."""
        count = 0
        with self._jobs_lock:
            jobs_snapshot = [(k, v) for k, v in self._active_jobs.items() if isinstance(v, ActiveJob)]
        for scid, job in jobs_snapshot:
            # I-10 FIX: Stop the job FIRST, then release budget. Releasing before
            # stop means if sling-stop fails, the budget is freed but the job
            # continues spending. By stopping first, we at least attempt to halt
            # the spend before releasing the reservation.
            if self.stop_job(scid, reason=reason):
                count += 1
            # B4 FIX: Only release budget for jobs still in PENDING/RUNNING status.
            # Jobs in SUCCESS/FAILED/TIMEOUT already had their budget handled by
            # monitor_jobs (via mark_budget_spent or release_budget_reservation).
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                try:
                    self.database.release_budget_reservation(str(job.rebalance_id))
                except Exception as e:
                    self.plugin.log(f"Failed to release budget reservation during stop_all: {e}", level='debug')
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
            # Get list of all sling jobs (sling-jobsettings returns dict keyed by SCID)
            result = self.plugin.rpc.call("sling-jobsettings", {})
            jobs = result if isinstance(result, dict) else {}

            if not jobs:
                self.plugin.log("Startup Hygiene: No orphan sling jobs found", level='debug')
                return 0

            orphan_count = 0
            for scid in jobs:
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

            # M-14: Mark orphaned pending/pending_async DB records as failed.
            # Records older than job timeout that are still pending are from crashed jobs.
            try:
                cutoff = int(time.time()) - self.job_timeout_seconds
                conn = self.database._get_connection()
                # I-7 FIX: Wrap orphan cleanup in transaction for atomicity
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # First, get the IDs of orphaned records so we can release their budget reservations
                    orphaned_rows = conn.execute("""
                        SELECT id FROM rebalance_history
                        WHERE status IN ('pending', 'pending_async')
                          AND timestamp < ?
                    """, (cutoff,)).fetchall()
                    orphaned_ids = [row['id'] for row in orphaned_rows]

                    cursor = conn.execute("""
                        UPDATE rebalance_history
                        SET status = 'failed', error_message = 'orphaned_on_restart'
                        WHERE status IN ('pending', 'pending_async')
                          AND timestamp < ?
                    """, (cutoff,))
                    orphaned_db = cursor.rowcount
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise

                # Release budget reservations for orphaned jobs (outside transaction —
                # release_budget_reservation manages its own transactions)
                released_count = 0
                for rid in orphaned_ids:
                    try:
                        self.database.release_budget_reservation(str(rid))
                        released_count += 1
                    except Exception:
                        pass

                if orphaned_db > 0:
                    self.plugin.log(
                        f"Startup Hygiene: Marked {orphaned_db} orphaned DB rebalance records as failed, "
                        f"released {released_count} budget reservations",
                        level='info'
                    )
            except Exception as e:
                self.plugin.log(f"Startup Hygiene: DB orphan cleanup error: {e}", level='debug')

            return orphan_count

        except RpcError as e:
            # sling-jobsettings might not be available or no jobs exist
            self.plugin.log(f"Startup Hygiene: Could not query sling jobs: {e}", level='debug')
            return 0
        except Exception as e:
            self.plugin.log(f"Startup Hygiene: Unexpected error: {e}", level='warn')
            return 0

    def execute_once(self, scid: str, direction: str, amount: int,
                     maxppm: int, onceamount: Optional[int] = None,
                     candidates: Optional[List[str]] = None,
                     outppm: Optional[int] = None,
                     maxhops: Optional[int] = None,
                     depleteuptopercent: Optional[float] = None,
                     depleteuptoamount: Optional[int] = None,
                     paralleljobs: Optional[int] = None) -> Dict[str, Any]:
        """
        One-shot rebalance via sling-once.  Blocks until complete.

        Unlike sling-job, no persistent job is created — nothing to
        track, monitor, or delete afterward.

        Used for: Defibrillator shocks, manual rebalances.

        sling-once params: scid, direction, amount, maxppm, onceamount
        (must be multiple of amount).  Target is forbidden for sling-once.
        """
        sling_scid = self._to_sling_scid(scid)
        if onceamount is None:
            onceamount = amount
        # Ensure multiple of amount
        if amount > 0 and onceamount % amount != 0:
            onceamount = ((onceamount // amount) + 1) * amount

        params: Dict[str, Any] = {
            "scid": sling_scid,
            "direction": direction,
            "amount": amount,
            "maxppm": maxppm,
            "onceamount": onceamount,
        }
        if candidates:
            params["candidates"] = [self._to_sling_scid(c) for c in candidates]
        if outppm and outppm > 0:
            params["outppm"] = outppm

        # Apply config defaults for params not explicitly provided
        if maxhops is None:
            maxhops = self.config.sling_max_hops
        params["maxhops"] = maxhops

        if depleteuptopercent is not None:
            params["depleteuptopercent"] = depleteuptopercent

        if depleteuptoamount is not None:
            params["depleteuptoamount"] = depleteuptoamount

        if paralleljobs is None:
            paralleljobs = self.config.sling_parallel_jobs
        if paralleljobs > 1:
            params["paralleljobs"] = paralleljobs

        self.plugin.log(
            f"sling-once: {sling_scid} dir={direction} amt={amount} "
            f"maxppm={maxppm} total={onceamount}",
            level='info'
        )

        # AskRene preflight sizing: if AskRene says the channel/direction has a lower maximum,
        # shrink the attempt so we fail less and learn faster.
        try:
            max_sats = self._askrene_max_sats_for_scid_dir(sling_scid)
            if max_sats is not None and max_sats > 0 and amount > max_sats:
                self.plugin.log(
                    f"AskRene preflight: capping sling-once amount {amount} -> {max_sats} sats for {sling_scid}",
                    level='info'
                )
                amount = max_sats
                params["amount"] = amount
                params["onceamount"] = amount
        except Exception as e:
            # Never block execution on AskRene parsing issues
            self.plugin.log(f"AskRene preflight failed (ignored): {e}", level='debug')

        try:
            result = self.plugin.rpc.call("sling-once", params)

            # Try to derive actual fees from sling stats (best-effort)
            fee_sats = 0
            try:
                st = self.plugin.rpc.call("sling-stats", {"scid": sling_scid, "json": True})
                # sling-stats may return a dict or a list of dicts
                if isinstance(st, list) and st:
                    st = st[0]
                if isinstance(st, dict):
                    # Preferred: explicit totals (use safe parsers for sling string values)
                    fee_sats = self._parse_sats(st.get("fee_total_sats"))
                    if not fee_sats:
                        fee_msat = self._parse_msat(st.get("fee_total_msat"))
                        fee_sats = fee_msat // 1000 if fee_msat else 0
                    # Fallback: weighted avg fee ppm
                    if fee_sats == 0:
                        w_feeppm = st.get("w_feeppm")
                        if w_feeppm is not None and amount > 0:
                            fee_sats = int((amount * self._parse_sats(w_feeppm) + 999_999) // 1_000_000)
            except Exception:
                pass

            return {"success": True, "message": "sling-once completed", "raw": result, "actual_fee_sats": fee_sats}
        except RpcError as e:
            err = str(e)
            # Auto-heal: stale job locks. If sling says a job is already running for this scid,
            # clear job registry and retry once.
            if "already a job for that scid running" in err.lower():
                try:
                    self.plugin.log(f"Sling job lock detected for {sling_scid}. Clearing that job and retrying once.", level='warn')
                    self.plugin.rpc.call("sling-deletejob", {"job": sling_scid})
                    result = self.plugin.rpc.call("sling-once", params)
                    # best-effort fee calc on retry as well
                    fee_sats = 0
                    try:
                        st = self.plugin.rpc.call("sling-stats", {"scid": sling_scid, "json": True})
                        if isinstance(st, list) and st:
                            st = st[0]
                        if isinstance(st, dict):
                            fee_sats = self._parse_sats(st.get("fee_total_sats"))
                            if not fee_sats:
                                fee_msat = self._parse_msat(st.get("fee_total_msat"))
                                fee_sats = fee_msat // 1000 if fee_msat else 0
                            if fee_sats == 0:
                                w_feeppm = st.get("w_feeppm")
                                if w_feeppm is not None and amount > 0:
                                    fee_sats = int((amount * self._parse_sats(w_feeppm) + 999_999) // 1_000_000)
                    except Exception:
                        pass
                    return {"success": True, "message": "sling-once completed (after deletejob)", "raw": result, "actual_fee_sats": fee_sats}
                except Exception as e2:
                    return {"success": False, "error": f"sling-once RPC error (job lock retry failed): {e2}"}

            return {"success": False, "error": f"sling-once RPC error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _askrene_refresh_cache(self) -> None:
        """Refresh AskRene constraints cache (best-effort).

        Stores short_channel_id_dir -> maximum_msat for the configured layer.
        Uses a time-based cache to avoid hammering RPC.
        """
        now = int(time.time())
        # TS-5: Protect cache access with lock
        with self._askrene_lock:
            if self._askrene_cache_ts and (now - self._askrene_cache_ts) < 30:
                return
        try:
            res = self.plugin.rpc.call("askrene-listlayers", {"layer": self.askrene_layer})
            layers = res.get("layers", [])
            cache: Dict[str, int] = {}
            for layer in layers:
                if layer.get("layer") != self.askrene_layer:
                    continue
                for c in layer.get("constraints", []) or []:
                    scid_dir = c.get("short_channel_id_dir")
                    try:
                        ts = int(c.get("timestamp") or 0)
                        # FIX: Use class method, 'parse_msat' is not defined in global scope
                        max_msat = self._parse_msat(c.get("maximum_msat", 0))
                    except (TypeError, ValueError):
                        continue  # Skip malformed entry, keep rest of cache
                    if not scid_dir or max_msat <= 0:
                        continue
                    # Age filter
                    if ts and (now - ts) > int(self.askrene_max_age_sec):
                        continue
                    # Keep the tightest constraint if multiple
                    if scid_dir not in cache or max_msat < cache[scid_dir]:
                        cache[scid_dir] = max_msat
            with self._askrene_lock:
                self._askrene_cache = cache
                self._askrene_cache_ts = now
        except Exception:
            # Silent: AskRene is optional; sling will still function.
            return

    def _askrene_max_sats_for_scid_dir(self, sling_scid: str) -> Optional[int]:
        """Return the tightest AskRene constraint (in sats) for a given scid (either dir).

        We don't always know the correct /0 vs /1 mapping for pull/push here,
        so we take the minimum across both directions when present.
        """
        self._askrene_refresh_cache()
        # TS-5: Read cache under lock
        with self._askrene_lock:
            cache_snapshot = dict(self._askrene_cache)
        best_msat = None
        for suffix in ("/0", "/1"):
            key = f"{sling_scid}{suffix}"
            v = cache_snapshot.get(key)
            if v is None:
                continue
            best_msat = v if best_msat is None else min(best_msat, v)
        if best_msat is None:
            return None
        return max(0, best_msat // 1000)

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
                result = self.plugin.rpc.call("sling-except-peer", ["list"])
                current_exclusions = set(result) if isinstance(result, list) else set(result.get("peers", []))
            except (RpcError, KeyError):
                current_exclusions = set()

            # Collect peers that should be excluded
            peers_to_exclude = set()

            # From policy manager (disabled rebalance mode)
            if policy_manager:
                self._policy_manager_ref = policy_manager
                try:
                    from .policy_manager import RebalanceMode
                    for policy in policy_manager.get_all_policies():
                        if policy.rebalance_mode == RebalanceMode.DISABLED:
                            peers_to_exclude.add(policy.peer_id)
                except Exception as e:
                    self.plugin.log(f"Could not get policies for exclusion sync: {e}", level='debug')

            # Add new exclusions to sling
            for peer_id in peers_to_exclude:
                if peer_id not in current_exclusions:
                    try:
                        self.plugin.rpc.call("sling-except-peer", ["add", peer_id])
                        excluded_count += 1
                        self.plugin.log(
                            f"Sling Exclusion: Added {peer_id[:16]}... to global exclusion list",
                            level='debug'
                        )
                    except RpcError as e:
                        self.plugin.log(f"Failed to add peer exclusion: {e}", level='warn')

            # B9 FIX: Remove stale exclusions for re-enabled peers
            for peer_id in current_exclusions:
                if peer_id not in peers_to_exclude:
                    try:
                        self.plugin.rpc.call("sling-except-peer", ["remove", peer_id])
                        self.plugin.log(
                            f"Sling Exclusion: Removed {peer_id[:16]}... (peer re-enabled)",
                            level='debug'
                        )
                    except RpcError as e:
                        self.plugin.log(f"Failed to remove peer exclusion: {e}", level='warn')

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
            self.plugin.rpc.call("sling-except-peer", ["add", peer_id])
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
            self.plugin.rpc.call("sling-except-peer", ["remove", peer_id])
            self.plugin.log(
                f"Sling Exclusion: Removed {peer_id[:16]}... from exclusion list",
                level='info'
            )
            return True
        except RpcError as e:
            self.plugin.log(f"Failed to remove sling peer exclusion: {e}", level='warn')
            return False

    def sync_channel_exclusions(self) -> int:
        """
        Sync channel exclusions with sling's channel exclusion list.

        Excludes channels with high failure counts from sling routing.
        Removes stale exclusions for channels whose failure counts have decayed.

        Returns:
            Number of exclusion changes made
        """
        changes = 0

        try:
            # Get current sling channel exclusions
            try:
                result = self.plugin.rpc.call("sling-except-chan", ["list"])
                current_exclusions = set(result) if isinstance(result, list) else set(result.get("channels", []))
            except (RpcError, KeyError):
                current_exclusions = set()

            # Channels that should be excluded (high failure count)
            channels_to_exclude = set()
            channels_to_remove = set()
            with self._source_failures_lock:
                for scid, count in self.source_failure_counts.items():
                    if count >= 5.0:
                        channels_to_exclude.add(self._to_sling_scid(scid))

                # Channels to un-exclude (failure count decayed)
                for scid in current_exclusions:
                    normalized = self._normalize_scid(scid)
                    count = self.source_failure_counts.get(normalized, 0)
                    if count < 2.0:
                        channels_to_remove.add(scid)

            # Add new exclusions
            for scid in channels_to_exclude:
                if scid not in current_exclusions:
                    if self.add_channel_exclusion(scid):
                        changes += 1

            # Remove stale exclusions
            for scid in channels_to_remove:
                if self.remove_channel_exclusion(scid):
                    changes += 1

            if changes > 0:
                self.plugin.log(
                    f"Sling Channel Exclusion Sync: {changes} changes",
                    level='info'
                )

        except Exception as e:
            self.plugin.log(f"Channel exclusion sync error: {e}", level='warn')

        return changes

    def add_channel_exclusion(self, scid: str) -> bool:
        """
        Add a single channel to sling's channel exclusion list.

        Args:
            scid: The short channel ID to exclude

        Returns:
            True if successfully added, False otherwise
        """
        try:
            sling_scid = self._to_sling_scid(scid)
            self.plugin.rpc.call("sling-except-chan", ["add", sling_scid])
            self.plugin.log(
                f"Sling Channel Exclusion: Added {sling_scid} to exclusion list",
                level='debug'
            )
            return True
        except RpcError as e:
            self.plugin.log(f"Failed to add sling channel exclusion: {e}", level='warn')
            return False

    def remove_channel_exclusion(self, scid: str) -> bool:
        """
        Remove a channel from sling's channel exclusion list.

        Args:
            scid: The short channel ID to un-exclude

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            sling_scid = self._to_sling_scid(scid)
            self.plugin.rpc.call("sling-except-chan", ["remove", sling_scid])
            self.plugin.log(
                f"Sling Channel Exclusion: Removed {sling_scid} from exclusion list",
                level='debug'
            )
            return True
        except RpcError as e:
            self.plugin.log(f"Failed to remove sling channel exclusion: {e}", level='warn')
            return False

    def get_job_status(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get status info for a specific job."""
        normalized = self._normalize_scid(channel_id)
        # TS-4: Protect _active_jobs read with _jobs_lock
        with self._jobs_lock:
            job = self._active_jobs.get(normalized)
        
        if not job:
            return None
        
        elapsed = int(time.time()) - job.start_time
        current_balance = self._get_channel_local_balance(normalized)
        raw_delta = current_balance - job.initial_local_sats
        # For push: negative delta = liquidity drained (success), so flip sign
        transferred = -raw_delta if job.direction == "push" else raw_delta

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
        result = []
        with self._jobs_lock:
            scids = list(self._active_jobs.keys())
        for scid in scids:
            status = self.get_job_status(scid)
            if status:
                result.append(status)
        return result

    def get_source_failure_count(self, channel_id: str) -> float:
        """Get the recent failure count for a source channel."""
        with self._source_failures_lock:
            return self.source_failure_counts.get(channel_id, 0.0)


class EVRebalancer:
    """
    Expected Value based rebalancer with async job queue support.

    This class acts as the "Strategist" - it calculates EV and determines
    IF and HOW MUCH to rebalance. The actual execution is delegated to
    the JobManager which manages sling background jobs.

    Thread Safety (I-13, I-14, S-9):
    The rebalance cycle runs single-threaded on a timer. Candidate evaluation,
    EV calculation, and sling job dispatch all happen within one timer callback.
    No locking is needed for cycle-local state because only one cycle runs at
    a time. The _pending dict is the only shared state (accessed by async job
    callbacks) and is protected by _pending_lock.

    Known Limitations (documented, not bugs):
    - I-5: Balance delta can false-positive under concurrent forwarding — a forward
      completing during the measurement window inflates/deflates the delta. Fixing
      this requires async forwarding awareness (architectural change).
    - I-16: SCID-keyed failure counts are invalidated when the SCID changes.
      A future migration to peer_id-keyed tracking would fix this.
    - I-18: Predictive rebalancing (pre-position liquidity before demand spikes) is a
      future feature requiring demand forecasting integration.
    """

    def __init__(self, plugin: Plugin, config: Config, database: Database,
                 policy_manager: Optional[PolicyManager] = None):
        self.plugin = plugin
        self.config = config
        self.database = database
        self.policy_manager = policy_manager
        self._pending: Dict[str, int] = {}
        self._pending_lock = threading.Lock()  # L-14: Protect _pending dict
        self._our_node_id: Optional[str] = None
        self._fee_cache: Dict[Tuple[str, int], Optional[int]] = {}  # F11 FIX: Initialize in __init__
        self._profitability_analyzer: Optional['ChannelProfitabilityAnalyzer'] = None

        # Optional callback injected by cl-revenue-ops to report external liquidity
        # costs (e.g. Boltz swap fees) for unified budget gating.
        self.external_liquidity_cost_provider = None
        # Optional callback injected by cl-revenue-ops to provide unified total-cost budget limit.
        self.global_budget_limit_provider = None

        self._capacity_planner = None  # Set via set_capacity_planner()
        # Exhaustion tracking: cached after each cycle for Boltz coordination
        self._last_depleted_count: int = 0
        self._last_profitable_count: int = 0
        self._last_cycle_ts: int = 0

        self._last_decision_summary: Dict[str, Any] = {
            "action": "hold",
            "reason": "not_run",
            "dominant_input": "startup",
            "safety_block": False,
            "budget_blocked": False,
        }

        # Initialize job manager for async execution
        self.job_manager = JobManager(plugin, config, database)

        # Hive hints adapter (injected by main plugin; None = disabled)
        self.hive_hints = None
        self._hive_router = None  # HiveRouter for fleet route discovery

    @property
    def hive_router(self):
        return self._hive_router

    @hive_router.setter
    def hive_router(self, value):
        self._hive_router = value
        # Propagate to job_manager so _handle_job_* methods can call unreserve
        self.job_manager.hive_router = value

    def _set_last_decision_summary(
        self,
        *,
        action: str,
        reason: str,
        dominant_input: Optional[str],
        safety_block: bool,
        budget_blocked: bool,
        error_detail: Optional[str] = None,
    ) -> None:
        self._last_decision_summary = {
            "action": action,
            "reason": reason,
            "dominant_input": dominant_input,
            "safety_block": bool(safety_block),
            "budget_blocked": bool(budget_blocked),
        }
        if error_detail:
            self._last_decision_summary["error_detail"] = error_detail

    def get_last_decision_summary(self) -> Dict[str, Any]:
        return dict(self._last_decision_summary)

    def get_boltz_coordination(self) -> Dict[str, Any]:
        """Return rebalancer exhaustion state for Boltz integration.

        When depleted channels exist but 0 profitable candidates were found,
        the rebalancer is exhausted and Boltz should be more aggressive.
        """
        exhausted = (self._last_depleted_count > 0 and self._last_profitable_count == 0)
        return {
            "sling_exhausted": exhausted,
            "depleted_count": self._last_depleted_count,
            "profitable_count": self._last_profitable_count,
            "cycle_ts": self._last_cycle_ts,
        }

    def _build_hive_liquidity_state_payload(
        self,
        depleted_channels: List[Tuple[str, Dict[str, Any], float]],
        source_channels: List[Tuple[str, Dict[str, Any], float]],
        candidates: List[RebalanceCandidate],
    ) -> Dict[str, Any]:
        """Build a local cl-hive liquidity-state payload from the current cycle."""
        depleted_payload = []
        for _channel_id, info, local_pct in depleted_channels:
            peer_id = str(info.get("peer_id") or "").strip()
            capacity_sats = int(info.get("capacity", 0) or 0)
            if not peer_id:
                continue
            depleted_payload.append({
                "peer_id": peer_id,
                "local_pct": float(local_pct),
                "capacity_sats": capacity_sats,
            })

        saturated_payload = []
        for _channel_id, info, local_pct in source_channels:
            peer_id = str(info.get("peer_id") or "").strip()
            capacity_sats = int(info.get("capacity", 0) or 0)
            if not peer_id:
                continue
            saturated_payload.append({
                "peer_id": peer_id,
                "local_pct": float(local_pct),
                "capacity_sats": capacity_sats,
            })

        liquidity_needs = []
        seen_pairs = set()
        for candidate in candidates[:10]:
            source_peer_id = str(candidate.primary_source_peer_id or "").strip()
            destination_peer_id = str(candidate.to_peer_id or "").strip()
            if (
                not source_peer_id
                or not destination_peer_id
                or source_peer_id == destination_peer_id
            ):
                continue
            pair = (source_peer_id, destination_peer_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            liquidity_needs.append({
                "source_peer_id": source_peer_id,
                "destination_peer_id": destination_peer_id,
                "capacity_sats": int(candidate.amount_sats),
                "priority_tier": "high" if candidate.expected_profit_sats > 0 else "medium",
                "flow_state": candidate.dest_flow_state,
                "expected_profit_sats": int(candidate.expected_profit_sats),
            })

        return {
            "depleted_channels": depleted_payload,
            "saturated_channels": saturated_payload,
            "rebalancing_active": self.job_manager.active_job_count > 0,
            "rebalancing_peers": self.job_manager.get_active_rebalancing_peers(),
            "liquidity_needs": liquidity_needs,
        }

    def _report_hive_liquidity_state(
        self,
        depleted_channels: List[Tuple[str, Dict[str, Any], float]],
        source_channels: List[Tuple[str, Dict[str, Any], float]],
        candidates: List[RebalanceCandidate],
    ) -> None:
        """Fail-open local report of current liquidity state into cl-hive."""
        payload = self._build_hive_liquidity_state_payload(
            depleted_channels,
            source_channels,
            candidates,
        )
        try:
            self.plugin.rpc.call("hive-report-liquidity-state", payload)
        except Exception as e:
            self.plugin.log(f"HIVE: liquidity state report failed: {e}", level='debug')

    def _get_hive_rebalance_bias(self, peer_id: str) -> float:
        """Return bounded multiplicative rebalance score bias from hive hints. 1.0 if unavailable."""
        if self.hive_hints is None:
            return 1.0
        try:
            bias = self.hive_hints.get_rebalance_bias(peer_id)
            return max(0.85, min(1.15, bias))
        except Exception:
            return 1.0

    def _get_hive_corridor_utilization_bias(self, peer_id: str) -> float:
        """Return bounded multiplicative utilization bias from hive corridor hints. 1.0 if unavailable."""
        if self.hive_hints is None:
            return 1.0
        try:
            bias = self.hive_hints.get_corridor_utilization_bias(peer_id)
            return max(0.90, min(1.10, bias))
        except Exception:
            return 1.0

    @staticmethod
    def _should_skip_futility(fail_count: int, last_error_type: str) -> bool:
        """
        Check if a channel should be skipped by the futility breaker.

        No-route failures trigger at 4 attempts (path likely doesn't exist).
        Other failures trigger at 10 (existing threshold).
        """
        if last_error_type == "no_route" and fail_count >= 4:
            return True
        if fail_count >= 10:
            return True
        return False

    @staticmethod
    def _apply_fee_escalation(ev_max_fee_ppm: int, fail_count: int, last_attempted_ppm: int) -> int:
        """
        Escalate fee budget based on failure history.

        If previous attempts failed at a lower fee, start above that fee
        (1.5x multiplier). Capped at the EV-derived maximum.
        """
        if fail_count == 0 or last_attempted_ppm <= 0:
            return ev_max_fee_ppm
        if last_attempted_ppm >= ev_max_fee_ppm:
            return ev_max_fee_ppm
        return min(int(last_attempted_ppm * 1.5), ev_max_fee_ppm)

    def _parse_msat(self, v: Any) -> int:
        """Delegate to shared parse_msat in utils.py."""
        return _shared_parse_msat(v)

    def _get_external_liquidity_costs(self) -> Dict[str, int]:
        provider = getattr(self, "external_liquidity_cost_provider", None)
        if not callable(provider):
            return {"spent_24h_sats": 0, "reserved_24h_sats": 0}
        try:
            data = provider()
            if not isinstance(data, dict):
                return {"spent_24h_sats": 0, "reserved_24h_sats": 0}
            return {
                "spent_24h_sats": max(0, int(data.get("spent_24h_sats", 0) or 0)),
                "reserved_24h_sats": max(0, int(data.get("reserved_24h_sats", 0) or 0)),
            }
        except Exception as e:
            self.plugin.log(f"External liquidity cost provider failed: {e}", level='warn')
            return {"spent_24h_sats": 0, "reserved_24h_sats": 0}

    def _get_global_budget_limit(self, cfg: Optional[ConfigSnapshot] = None) -> int:
        provider = getattr(self, "global_budget_limit_provider", None)
        if callable(provider):
            try:
                data = provider()
                if isinstance(data, dict):
                    if "effective_budget_sats" in data:
                        return max(0, int(data.get("effective_budget_sats", 0) or 0))
                    if "budget_sats" in data:
                        return max(0, int(data.get("budget_sats", 0) or 0))
                if isinstance(data, (int, float, str)):
                    return max(0, int(float(data)))
            except Exception as e:
                self.plugin.log(f"Global budget limit provider failed: {e}", level='warn')
        return max(0, int((cfg or self.config.snapshot()).daily_budget_sats))

    def _get_our_node_id(self) -> str:
        if self._our_node_id:
            return self._our_node_id
        try:
            info = self.plugin.rpc.getinfo()
            node_id = info.get("id", "")
            if node_id:
                self._our_node_id = node_id
            return node_id or None
        except Exception as e:
            self.plugin.log(f"Error getting our node ID: {e}", level='error')
            return None  # F10 FIX: Don't cache failure, retry next call

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

    def set_capacity_planner(self, planner) -> None:
        """Set reference to capacity planner for coordination."""
        self._capacity_planner = planner

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
        channels: Dict[str, Dict[str, Any]] = {}
        depleted_channels: List[Tuple[str, Dict[str, Any], float]] = []
        source_channels: List[Tuple[str, Dict[str, Any], float]] = []

        # Initialize ephemeral fee cache for this run (cleared at end)
        self._fee_cache: Dict[Tuple[str, int], Optional[int]] = {}

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
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="no_rebalance_slots",
                    dominant_input="concurrency_limit",
                    safety_block=True,
                    budget_blocked=False,
                )
                self.plugin.log(
                    f"No rebalance slots available ({self.job_manager.active_job_count}/"
                    f"{self.job_manager.max_concurrent_jobs} jobs active)"
                )
                self._report_hive_liquidity_state(depleted_channels, source_channels, candidates)
                return candidates
            
            # Check capital controls (pass cfg for thread-safe config access)
            if not self._check_capital_controls(cfg):
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="capital_controls_blocked",
                    dominant_input=getattr(self, '_capital_control_blocker', 'daily_budget_sats'),
                    safety_block=True,
                    budget_blocked=True,
                )
                self._report_hive_liquidity_state(depleted_channels, source_channels, candidates)
                return candidates
            
            channels = self._get_channels_with_balances()
            if not channels:
                reason = (
                    "rpc_error_channel_balances"
                    if self._last_balance_error
                    else "no_channel_balance_data"
                )
                self._set_last_decision_summary(
                    action="hold",
                    reason=reason,
                    dominant_input="channel_balances",
                    safety_block=False,
                    budget_blocked=False,
                    error_detail=self._last_balance_error,
                )
                self._report_hive_liquidity_state(depleted_channels, source_channels, candidates)
                return candidates
            
            # Note: _peer_inbound_fees cache is now populated by _get_channels_with_balances()
            # This provides actual peer fees from listpeerchannels.updates.remote

            # Refresh askrene hive-fleet layer for route discovery (best effort)
            if self.hive_router:
                self.hive_router.refresh_layer()
            
            # Hoist peer connection status call - do it once instead of per-candidate
            peer_status = self._get_peer_connection_status()

            # Get set of channels with active jobs
            active_channels = set(self.job_manager.active_channels)
            
            # Hoist planner loser lookup once (avoid per-channel call)
            planner_loser_scids: set = set()
            if self._capacity_planner is not None:
                try:
                    planner_loser_scids = self._capacity_planner.get_boltz_coordination().get("loser_scids", set())
                except Exception:
                    pass

            hot_override_depletion_thresholds = {}  # peer_id -> ratio threshold override
            try:
                if self.database is not None:
                    for r in (self.database.list_hot_channel_protection_override_peers() or []):
                        pid = str(r.get('peer_id') or '').strip()
                        pct = r.get('min_depletion_trigger_pct')
                        try:
                            pct_f = float(pct) if pct is not None else None
                        except Exception:
                            pct_f = None
                        if pid and pct_f is not None and 0.0 < pct_f <= 100.0:
                            hot_override_depletion_thresholds[pid] = pct_f / 100.0
            except Exception as e:
                self.plugin.log(f"Hot-channel override depletion threshold lookup failed: {e}", level='debug')

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

                # Skip channels the capacity planner has marked for closure
                if channel_id in planner_loser_scids:
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
                
                else:
                    effective_low_threshold = float(hot_override_depletion_thresholds.get(peer_id, cfg.low_liquidity_threshold))
                    if outbound_ratio < effective_low_threshold:
                        depleted_channels.append((channel_id, info, outbound_ratio))
                    elif outbound_ratio > cfg.high_liquidity_threshold:
                        source_channels.append((channel_id, info, outbound_ratio))
            
            if not depleted_channels or not source_channels:
                total = len(channels) - len(active_channels)
                if not depleted_channels and not source_channels:
                    self.plugin.log(
                        f"No rebalance candidates: all {total} channels are in balanced range "
                        f"({cfg.low_liquidity_threshold:.0%}-{cfg.high_liquidity_threshold:.0%} outbound). "
                        f"None are depleted (<{cfg.low_liquidity_threshold:.0%}) or overfull (>{cfg.high_liquidity_threshold:.0%}).",
                        level='info'
                    )
                elif not depleted_channels:
                    self.plugin.log(
                        f"No rebalance candidates: {len(source_channels)} overfull channels but no depleted "
                        f"channels (<{cfg.low_liquidity_threshold:.0%} outbound) to fill.",
                        level='info'
                    )
                else:
                    self.plugin.log(
                        f"No rebalance candidates: {len(depleted_channels)} depleted channels but no source "
                        f"channels (>{cfg.high_liquidity_threshold:.0%} outbound) to drain.",
                        level='info'
                    )
                self._set_last_decision_summary(
                    action="hold",
                    reason="no_rebalance_candidates",
                    dominant_input="liquidity_balance",
                    safety_block=False,
                    budget_blocked=False,
                )
                self._report_hive_liquidity_state(depleted_channels, source_channels, candidates)
                return candidates

            self.plugin.log(
                f"Found {len(depleted_channels)} depleted and {len(source_channels)} source channels "
                f"(excluding {len(active_channels)} with active jobs)"
            )
            
            for dest_id, dest_info, dest_ratio in depleted_channels:
                # =====================================================================
                # FUTILITY CIRCUIT BREAKER (TODO #15)
                # =====================================================================
                # Some channels have positive EV spreads but broken routing paths.
                # Exponential backoff slows down retries, but doesn't stop them.
                # After repeated failures, the channel is likely a "Dead End" and
                # further attempts waste gossip bandwidth and lock HTLCs.
                #
                # Error-aware thresholds:
                #   - no_route failures: 4 attempts (path likely doesn't exist)
                #   - other failures:   10 attempts (existing threshold)
                # After threshold hit, require 48h cooldown before retry
                # =====================================================================
                fail_count, last_fail = self.database.get_failure_count(dest_id)
                fail_meta = self.database.get_failure_metadata(dest_id)
                if self._should_skip_futility(fail_count, fail_meta["last_error_type"]):
                    now = int(time.time())
                    futility_cooldown = getattr(cfg, 'futility_cooldown_hours', 48) * 3600
                    if (now - last_fail) < futility_cooldown:
                        threshold = "4 no_route" if fail_meta["last_error_type"] == "no_route" else "10"
                        self.plugin.log(
                            f"FUTILITY BREAKER: Skipping {dest_id[:12]}... - {fail_count} failures "
                            f"(threshold: {threshold}), "
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

                candidate = self._analyze_rebalance_ev(
                    dest_id, dest_info, dest_ratio, source_channels, peer_status, cfg=cfg
                )
                if candidate:
                    if last_rebalance:
                        cd_hours = float(getattr(candidate, 'recommended_cooldown_hours', 0.0) or 0.0)
                        if cd_hours <= 0:
                            cd_hours = float(self.config.rebalance_cooldown_hours)
                        cooldown = int(max(0.0, cd_hours) * 3600)
                        if cooldown > 0 and int(time.time()) - last_rebalance < cooldown:
                            self.plugin.log(
                                f"REBAL_SKIP: {dest_id[:12]}... [skip_cooldown] hot_cd={cd_hours:.2f}h",
                                level='debug'
                            )
                            continue
                    candidates.append(candidate)
                    
                    # Stop if we have enough candidates to fill available slots
                    if len(candidates) >= available_slots:
                        break
            
            # === PUSH CANDIDATE DETECTION ===
            # Overfull channels that have high source failure rates benefit from
            # PUSH direction (drain out to destinations) instead of being used
            # as pull sources (which keeps failing).
            remaining_slots = available_slots - len(candidates)
            if remaining_slots > 0 and depleted_channels:
                push_candidates = []
                # Channels already selected as pull sources — skip them for push
                used_sources = set()
                for c in candidates:
                    used_sources.update(c.source_candidates)

                for src_id, src_info, src_ratio in source_channels:
                    if src_id in active_channels or src_id in used_sources:
                        continue
                    # Only extremely overfull channels (>85% local)
                    if src_ratio < 0.85:
                        continue
                    # Must have source failure history (>3 failures) — this is the signal
                    # that pull-from-this-channel doesn't work well
                    src_fail_count = self.job_manager.get_source_failure_count(src_id)
                    if src_fail_count < 3:
                        continue

                    # Profitability guard: don't drain highly profitable channels.
                    # A channel with high ROI is earning well even while overfull —
                    # draining it costs rebalance fees and may reduce future revenue.
                    # Prefer draining low-ROI overfull channels instead.
                    if self._profitability_analyzer is not None:
                        try:
                            src_prof = self._profitability_analyzer.get_profitability(src_id)
                            if src_prof is not None:
                                src_roi = getattr(src_prof, 'marginal_roi_percent', 0.0) or 0.0
                                if src_roi > 20.0:
                                    # >20% marginal ROI: skip push — channel is earning well
                                    continue
                        except Exception:
                            pass

                    # Build push candidate: drain src_id, liquidity flows to depleted channels
                    dest_scids = [d[0] for d in depleted_channels[:5]]
                    dest_peer_ids = [d[1].get("peer_id", "") for d in depleted_channels[:5]]
                    push_ev = self._estimate_push_ev(src_id, src_info, src_ratio, dest_scids, dest_peer_ids)
                    if push_ev and push_ev.expected_profit_sats >= 0:
                        push_candidates.append(push_ev)

                # Sort by profit, take remaining slots
                push_candidates.sort(key=lambda c: c.expected_profit_sats, reverse=True)
                candidates.extend(push_candidates[:remaining_slots])

            # Route-pair awareness: identify channels on proven revenue routes
            route_pair_out_channels = set()
            route_pair_in_channels = set()
            try:
                pairs = self.database.get_top_route_pairs(days=30, min_forwards=3, limit=10)
                for p in pairs:
                    out_ch = str(p.get("out_channel", "")).replace(":", "x")
                    in_ch = str(p.get("in_channel", "")).replace(":", "x")
                    if out_ch:
                        route_pair_out_channels.add(out_ch)
                    if in_ch:
                        route_pair_in_channels.add(in_ch)
            except Exception:
                pass
            # Store for use in _select_source_candidates
            self._cycle_route_pair_in_channels = route_pair_in_channels

            # Sort by priority
            def sort_key(c):
                dest_state = self.database.get_channel_state(c.to_channel)
                flow_state = dest_state.get("state", "balanced") if dest_state else "balanced"
                priority = 2 if flow_state == "source" else 1
                # Boost channels on proven revenue routes
                route_bonus = 1.3 if c.to_channel in route_pair_out_channels else 1.0
                hive_bias = self._get_hive_rebalance_bias(c.to_peer_id)
                biased_profit = c.expected_profit_sats * hive_bias * route_bonus
                return (priority, biased_profit)

            candidates.sort(key=sort_key, reverse=True)

            # Limit to available slots
            selected = candidates[:available_slots]
            if selected:
                self._set_last_decision_summary(
                    action="rebalance",
                    reason="profitable_candidates_found",
                    dominant_input=selected[0].reason_code,
                    safety_block=False,
                    budget_blocked=False,
                )
            else:
                self._set_last_decision_summary(
                    action="hold",
                    reason="no_profitable_candidates",
                    dominant_input="ev_filter",
                    safety_block=False,
                    budget_blocked=False,
                )
            self._report_hive_liquidity_state(depleted_channels, source_channels, selected)
            return selected
        
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
                              peer_status: Optional[Dict] = None,
                              cfg=None) -> Optional[RebalanceCandidate]:
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
        
        # Track bleeder status for explainability (default to "none")
        dest_bleeder_status = "none"

        # F6 FIX: Initialize prof before try block so we can reference it
        # directly instead of the fragile locals().get('prof') pattern.
        prof = None

        # Check profitability logic
        if self._profitability_analyzer:
            try:
                prof = self._profitability_analyzer.analyze_channel(dest_channel)
                if prof and prof.classification.value == "zombie":
                    self.plugin.log(
                        f"REBAL_SKIP: {dest_channel[:12]}... [{RebalanceReasonCode.SKIP_ZOMBIE.value}]",
                        level='debug'
                    )
                    return None
                if prof and prof.classification.value == "underwater" and prof.marginal_roi <= 0:
                    self.plugin.log(
                        f"REBAL_SKIP: {dest_channel[:12]}... [{RebalanceReasonCode.SKIP_UNDERWATER.value}] "
                        f"marginal_roi={prof.marginal_roi:.2f}",
                        level='debug'
                    )
                    return None

                # =============================================================
                # BLEEDER DETECTION CHECK (Enhanced)
                # =============================================================
                # Check for bleeder status using the v2 detection
                # - Hard bleeders: Skip entirely (rebalancing disabled)
                # - Soft bleeders: Log but continue with reduced priority
                # =============================================================
                bleeder_check = self._profitability_analyzer.get_bleeder_status(dest_channel)
                if bleeder_check:
                    dest_bleeder_status = bleeder_check.classification

                    if bleeder_check.is_hard_bleeder:
                        self.plugin.log(
                            f"REBAL_SKIP: {dest_channel[:12]}... [{RebalanceReasonCode.SKIP_HARD_BLEEDER.value}] "
                            f"rebal_cost={bleeder_check.rebalance_cost_30d} > 2x revenue={bleeder_check.revenue_30d}, "
                            f"net={bleeder_check.net_profit_30d}",
                            level='info'
                        )
                        return None

                    # Soft bleeders get logged but continue with normal flow
                    # (amount reduction happens later in the flow)
                    if bleeder_check.is_soft_bleeder:
                        self.plugin.log(
                            f"REBAL_WARN: {dest_channel[:12]}... soft bleeder detected - "
                            f"7d={bleeder_check.net_profit_7d}, 30d={bleeder_check.net_profit_30d}",
                            level='debug'
                        )

            except Exception as e:
                self.plugin.log(f"Error checking profitability for {dest_channel}: {e}", level='debug')

        capacity = dest_info.get("capacity", 0)
        spendable = dest_info.get("spendable_sats", 0)

        # ZERO-TOLERANCE: Never attempt to rebalance channels with non-positive capacity.
        # These can appear transiently (closing/failed states) or via incomplete channel info.
        if capacity <= 0:
            return None
        
        # Dynamic targeting based on flow state
        # Note: sink channels are filtered out at method entry (return None)
        if dest_flow_state == "source":
            target_ratio = 0.85
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
        if cfg is None:
            cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        velocity = daily_volume / capacity if capacity > 0 else 0.0

        # Get channel age for grace period
        channel_age_days = self._get_channel_age_days(dest_channel, dest_info)

        hot_profile = self._compute_hot_channel_protection(
            dest_channel=dest_channel,
            dest_peer_id=dest_info.get('peer_id', ''),
            dest_flow_state=dest_flow_state,
            dest_ratio=dest_ratio,
            velocity=velocity,
            prof=prof,
            cfg=cfg,
        )

        # Apply velocity gate
        velocity_adjusted_target_ratio = target_ratio

        if cfg.enable_velocity_gate:
            # Grace period for new channels - they get normal targeting
            if channel_age_days >= cfg.new_channel_grace_days and velocity < cfg.min_velocity_threshold:
                # Low velocity - use conservative target (15% of capacity)
                # This is enough to test routing without wasting budget
                velocity_adjusted_target_ratio = 0.15
                self.plugin.log(
                    f"VELOCITY GATE: {dest_channel[:12]}... conservative target "
                    f"(velocity={velocity:.4f}, age={channel_age_days}d, "
                    f"target={velocity_adjusted_target_ratio:.0%} vs original {target_ratio:.0%})",
                    level='debug'
                )

        # Use velocity-adjusted target ratio
        target_ratio = velocity_adjusted_target_ratio

        # HOT CHANNEL PROTECTION: For fast-draining high-profit source channels,
        # refill earlier/deeper to reduce remote-close risk from depletion.
        if hot_profile.get('eligible') and dest_flow_state == 'source':
            boost = float(hot_profile.get('target_ratio_boost', 0.0) or 0.0)
            target_ratio = min(0.95, max(target_ratio, target_ratio + boost))
            self.plugin.log(
                f"HOT CHANNEL PROTECTION: {dest_channel[:12]}... score={hot_profile.get('score', 0.0):.2f} \
target_ratio={target_ratio:.0%} vel={velocity:.3f} roi={float(hot_profile.get('marginal_roi', 0.0) or 0.0):.2f}",
                level='info'
            )

        # =====================================================================
        # HOTFIX 0.1: Destination Sizing Guard
        # =====================================================================
        # Problem: target_spendable = max(min_amount, target_spendable) could force
        # target above capacity, causing repeated failures and pathological candidates.
        # Solution: Clamp to capacity and skip tiny channels that can't meet min_amount.
        # =====================================================================
        
        # Calculate volume-based target (3 days of buffer)
        vol_target = int(daily_volume * 3)
        
        # Calculate capacity-based target (original logic)
        cap_target = int(capacity * target_ratio)
        
        # Smart Allocation: Use the LOWER of volume target or capacity target
        # This prevents overfilling slow channels while still allowing fast channels
        # to be fully stocked.
        # FIX: Only constrain by volume when vol_target is meaningful (above min_amount).
        # Otherwise a tiny trickle of volume (e.g. 10k sats) would reduce the target
        # below min_amount and get killed by the sizing guard — worse than zero volume.
        if vol_target >= self.config.rebalance_min_amount:
            raw_target = min(cap_target, vol_target)
        else:
            # No meaningful volume data - fall back to capacity-based target
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
        #
        dynamic_chunk_cap = int(self.config.sling_chunk_size_sats)
        if hot_profile.get('eligible'):
            try:
                dynamic_chunk_cap = int(dynamic_chunk_cap * float(hot_profile.get('chunk_multiplier', 1.0) or 1.0))
            except Exception:
                pass
        # Safety: keep per-action size bounded.
        dynamic_chunk_cap = max(self.config.rebalance_min_amount, min(dynamic_chunk_cap, self.config.rebalance_max_amount, max(1, capacity // 4)))
        rebalance_amount = min(desired_amount, dynamic_chunk_cap)

        # AskRene constraint: don't exceed 75% of max believed routable amount.
        # This improves success rates by staying within CLN's liquidity beliefs.
        try:
            dest_sling_scid = self.job_manager._to_sling_scid(dest_channel)
            askrene_max = self.job_manager._askrene_max_sats_for_scid_dir(dest_sling_scid)
            if askrene_max is not None and askrene_max > 0:
                askrene_cap = int(askrene_max * 0.75)
                if askrene_cap < rebalance_amount and askrene_cap >= self.config.rebalance_min_amount:
                    self.plugin.log(
                        f"ASKRENE_CAP: {dest_channel[:12]}... clamped from {rebalance_amount:,} to "
                        f"{askrene_cap:,} sats (AskRene max: {askrene_max:,})",
                        level='debug'
                    )
                    rebalance_amount = askrene_cap
        except Exception:
            pass  # AskRene is optional

        amount_msat = rebalance_amount * 1000

        # FEE FOR EV: Use the LOWER of broadcast fee and DTS posterior mean.
        # Broadcast fee = what the network actually sees (conservative baseline).
        # DTS posterior mean = what we believe the market fee is (may be higher
        # if hysteresis blocked an update, or lower if we're about to drop).
        # Using the lower prevents overestimating income.
        fee_state = self.database.get_fee_strategy_state(dest_channel)
        broadcast_fee_ppm = fee_state.get("last_broadcast_fee_ppm", 0)

        # Fallback to listpeerchannels fee if no broadcast fee recorded
        if broadcast_fee_ppm <= 0:
            broadcast_fee_ppm = dest_info.get("fee_ppm", 0)

        # Use DTS posterior mean if available (more accurate than broadcast)
        dts_fee_ppm = broadcast_fee_ppm
        try:
            v2_json = fee_state.get("v2_state_json") if fee_state else None
            if v2_json:
                import json
                dts_data = json.loads(v2_json) if isinstance(v2_json, str) else v2_json
                posterior_mean = dts_data.get("posterior_mean")
                if isinstance(posterior_mean, (int, float)) and posterior_mean > 0:
                    dts_fee_ppm = int(posterior_mean)
        except Exception:
            pass

        # Conservative: use the lower of broadcast and DTS estimate
        outbound_fee_ppm = min(broadcast_fee_ppm, dts_fee_ppm) if dts_fee_ppm > 0 else broadcast_fee_ppm
        inbound_fee_ppm = self._estimate_inbound_fee(dest_info.get("peer_id", ""))

        dest_peer_id = dest_info.get("peer_id", "")

        # HIVE ROUTE DISCOVERY: Try askrene to find cheap routes through fleet
        # before falling back to generic source selection.
        hive_route = None
        if self.hive_router and self.hive_router.available and dest_peer_id:
            hr = self.hive_router.discover_route(dest_peer_id, rebalance_amount)
            if hr:
                hive_route = {
                    "fee_ppm": hr.fee_ppm,
                    "hops": hr.hops,
                    "source_scid": hr.source_scid,
                }
            if hive_route and hive_route.get("fee_ppm", 9999) < inbound_fee_ppm:
                # Hive route is cheaper than estimated inbound — use its fee
                inbound_fee_ppm = hive_route["fee_ppm"]
                self.plugin.log(
                    f"HIVE ROUTE: Using askrene-discovered fee {inbound_fee_ppm} ppm "
                    f"for {dest_peer_id[:12]}... ({hive_route.get('hops', '?')} hops)",
                    level='info'
                )

        # Get ALL profitable source candidates (sorted by score, best first)
        source_candidates = self._select_source_candidates(
            sources, rebalance_amount, dest_channel, outbound_fee_ppm, inbound_fee_ppm,
            peer_status=peer_status,
        )

        if not source_candidates:
            return None

        # If askrene found a hive route, boost its source channel to the top
        if hive_route and hive_route.get("source_scid"):
            hive_scid = hive_route["source_scid"].replace(":", "x")
            # Move the hive route's source to position 0 if it's in the list
            for i, (cid, info, score, opp) in enumerate(source_candidates):
                if cid.replace(":", "x") == hive_scid:
                    if i > 0:
                        source_candidates.insert(0, source_candidates.pop(i))
                        self.plugin.log(
                            f"HIVE ROUTE: Promoted {cid} to primary source "
                            f"(askrene route via fleet, {hive_route.get('fee_ppm', '?')} ppm)",
                            level='info'
                        )
                    break
        
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
        
        # Require non-negative spread to avoid consistent leakage.
        if spread_ppm < 0:
            return None

        effective_spread_ppm = max(1, spread_ppm)
        raw_budget_msat = (effective_spread_ppm * amount_msat) // 1_000_000
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
            historical_p = reputation.get('score', 0.5)  # Historical success probability

            # EV v2.0: Blend AskRene real-time liquidity belief with historical reputation.
            # AskRene tracks maximum believed capacity per channel direction from
            # payment successes/failures. If AskRene says max capacity is X and we
            # want to route Y, probability drops as Y approaches X.
            p = historical_p
            try:
                dest_sling_scid = self.job_manager._to_sling_scid(dest_channel)
                askrene_max_sats = self.job_manager._askrene_max_sats_for_scid_dir(dest_sling_scid)
                if askrene_max_sats is not None and askrene_max_sats > 0:
                    # Clamp ratio to 0.99 so probability never goes negative
                    capacity_ratio = min(0.99, rebalance_amount / askrene_max_sats)
                    askrene_p = 1.0 - capacity_ratio
                    # Blend: 30% historical, 70% AskRene (real-time is more informative)
                    p = (historical_p * 0.3) + (askrene_p * 0.7)
            except Exception:
                pass  # AskRene is optional; fall back to pure historical

            cost_ppm = inbound_fee_ppm + weighted_opp_cost
            # MA-4: Avoid b=inf when cost_ppm=0 (free routing); use large finite value
            b = outbound_fee_ppm / cost_ppm if cost_ppm > 0 else 100.0  # Odds
            kelly_f = p - (1 - p) / b if b > 0 else -1.0  # Raw Kelly fraction
            kelly_safe = min(kelly_f * self.config.kelly_fraction, 1.0)

            if kelly_safe <= 0:
                return None  # Negative EV, reject
            # MA-5: Ensure at least 1 sat budget when Kelly fraction is positive but small
            max_budget_sats = max(1, int(max_budget_sats * kelly_safe))
            max_budget_msat = max_budget_sats * 1000

        # HOT CHANNEL PROTECTION: budget override tied to channel profitability.
        hot_budget_override_sats = 0
        hot_channel_profit_budget_sats = 0
        hot_score = 0.0
        hot_cooldown_hours = 0.0
        if hot_profile.get('eligible'):
            hot_channel_profit_budget_sats = int(hot_profile.get('channel_profit_budget_sats', 0) or 0)
            hot_score = float(hot_profile.get('score', 0.0) or 0.0)
            hot_cooldown_hours = float(hot_profile.get('recommended_cooldown_hours', 0.0) or 0.0)
            # Lift per-trade fee cap up to the per-channel daily profit budget, with a hard routing-fee sanity cap.
            per_trade_hot_cap = min(hot_channel_profit_budget_sats, max(1, int((rebalance_amount * 5000) / 1_000_000)))
            if per_trade_hot_cap > max_budget_sats:
                self.plugin.log(
                    f"HOT CHANNEL PROTECTION: Raising fee budget for {dest_channel[:12]}... {max_budget_sats} -> {per_trade_hot_cap} sats",
                    level='info'
                )
                max_budget_sats = per_trade_hot_cap
                max_budget_msat = max_budget_sats * 1000
            hot_budget_override_sats = max(0, hot_channel_profit_budget_sats)

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

        # Hot-channel protection safety guard: do not allow protected emergency fills
        # to pay unbounded route fees. This is a hard execution ceiling even if the
        # profit-budget override would otherwise permit a larger sats budget.
        if max_fee_ppm > 0 and hot_profile.get('eligible'):
            protected_fee_cap_ppm = int(getattr(cfg, 'hot_channel_protection_max_rebalance_fee_ppm', 2000) or 0)
            if protected_fee_cap_ppm > 0 and max_fee_ppm > protected_fee_cap_ppm:
                self.plugin.log(
                    f"HOT CHANNEL PROTECTION: Capping max_fee_ppm for {dest_channel[:12]}... "
                    f"{max_fee_ppm} -> {protected_fee_cap_ppm}",
                    level='info'
                )
                max_fee_ppm = protected_fee_cap_ppm
                # Keep the sats-budget consistent with the capped ppm so reservations and EV use
                # the true executable ceiling (conservative rounding up to sats).
                if amount_msat > 0:
                    capped_budget_msat = max(1, (amount_msat * max_fee_ppm) // 1_000_000)
                    if capped_budget_msat < max_budget_msat:
                        max_budget_msat = capped_budget_msat
                        max_budget_sats = max(1, (max_budget_msat + 999) // 1000)
            
        if max_fee_ppm <= 0:
            return None

        # Snapshot EV-derived fee before escalation (for adaptive chunk sizing)
        ev_base_fee_ppm = max_fee_ppm

        # Graduated fee escalation: if previous attempts failed at lower fees,
        # start above the last failure point (capped at EV-derived max).
        fail_count, _ = self.database.get_failure_count(dest_channel)
        if fail_count > 0:
            meta = self.database.get_failure_metadata(dest_channel)
            max_fee_ppm = self._apply_fee_escalation(
                ev_max_fee_ppm=max_fee_ppm,
                fail_count=fail_count,
                last_attempted_ppm=meta["last_attempted_ppm"],
            )

        dest_turnover_rate = self._calculate_turnover_rate(dest_channel, capacity)
        cooldown_days = self.config.rebalance_cooldown_hours / 24.0

        # =================================================================
        # EV v2.0: PROBABILISTIC UTILIZATION (Normal CDF)
        # =================================================================
        # Instead of linear expected_utilization = turnover * cooldown (capped at 1.0),
        # model demand as a Gaussian process using Kalman filter predictions.
        # P(utilized) = P(demand > rebalance_amount) = 1 - Phi(z)
        # where z = (rebalance_amount - predicted_volume) / std_dev
        #
        # Fallback: When Kalman data is unavailable, use the original linear heuristic.
        # =================================================================
        kalman = self._get_kalman_metrics(dest_channel)
        kalman_velocity = kalman["velocity"]
        kalman_uncertainty = kalman["uncertainty"]

        if abs(kalman_velocity) > 1e-6 and kalman_uncertainty > 1e-6:
            # Kalman-based probabilistic utilization
            # kalman_velocity is per-hour; multiply by hours for volume prediction
            cooldown_hours = cooldown_days * 24.0
            predicted_volume = abs(kalman_velocity) * capacity * cooldown_hours
            # Uncertainty is dimensionless (on flow_ratio); diffusion scales with sqrt(days)
            std_dev = kalman_uncertainty * capacity * math.sqrt(max(cooldown_days, 0.01))

            # P(demand >= rebalance_amount) using complementary error function
            z = (rebalance_amount - predicted_volume) / (std_dev * math.sqrt(2) + 1e-8)
            prob_utilized = 0.5 * (1.0 - math.erf(z))

            # Guard: NaN propagation through erf() would produce 1.0 (most optimistic)
            # due to Python's min/max argument-order semantics. Treat NaN as fallback.
            if not math.isfinite(prob_utilized):
                prob_utilized = dest_turnover_rate * cooldown_days

            # Clamp to [0.05, 1.0] — never assume zero utilization for a live channel
            expected_utilization = max(0.05, min(1.0, prob_utilized))
        else:
            # Fallback: linear heuristic (original logic)
            expected_utilization = max(min(dest_turnover_rate * cooldown_days, 1.0), 0.05)

        expected_income = int((rebalance_amount * expected_utilization * outbound_fee_ppm) // 1_000_000)

        # Source reliability: discount income by probability the rebalance succeeds.
        # Uses primary source's failure history — high failure count = lower success prob.
        source_fail_count = self.job_manager.get_source_failure_count(primary_source_id)
        if source_fail_count > 0:
            # Exponential decay: 0 fails=100%, 1 fail=80%, 2 fails=64%, 4 fails=41%
            source_success_prob = max(0.1, 0.8 ** source_fail_count)
            expected_income = int(expected_income * source_success_prob)

        # Corridor ownership is a bounded, confidence-weighted utilization prior.
        expected_income = int(
            expected_income * self._get_hive_corridor_utilization_bias(dest_peer_id)
        )

        turnover_weight = min(1.0, source_turnover_rate * 7)
        # I-2 FIX: Use source channel's own utilization probability for opportunity cost,
        # not the destination's expected_utilization. These are independent probabilistic events.
        source_utilization = max(0.05, min(1.0, source_turnover_rate * cooldown_days))
        expected_source_loss = int(
            (rebalance_amount * source_utilization * source_fee_ppm * turnover_weight) // 1_000_000
        )

        # =================================================================
        # EV v2.0: EXPECTED COST vs MAX BUDGET
        # =================================================================
        # Use the expected routing fee (historical median) instead of max_budget_sats
        # (the worst-case ceiling). max_budget_sats is still passed to Sling as the
        # hard cap, but EV should reflect the likely cost, not the worst case.
        # =================================================================
        # I-3 FIX: Cap budget ceiling at utilization-adjusted income.
        # Without this, Sling can spend up to the full-spread budget even though
        # utilization < 1.0 means we won't earn the full spread before next rebalance.
        if expected_income > 0:
            max_budget_sats = min(max_budget_sats, max(1, expected_income))
            max_budget_msat = max_budget_sats * 1000
            # B1 FIX: Re-derive max_fee_ppm from the capped budget.
            # Without this, the stale pre-cap max_fee_ppm is recorded as
            # attempted_ppm on failure, poisoning fee escalation feedback.
            if amount_msat > 0:
                capped_budget_ppm = (max_budget_msat * 1_000_000) // amount_msat
                max_fee_ppm = min(max_fee_ppm, max(1, capped_budget_ppm)) if capped_budget_ppm > 0 else max_fee_ppm

        expected_fee_sats = self._estimate_expected_fee_sats(dest_peer_id, rebalance_amount)
        # Never let expected fee exceed the max budget (it's a ceiling)
        expected_fee_sats = min(expected_fee_sats, max_budget_sats)

        expected_profit = expected_income - expected_fee_sats - expected_source_loss
        
        # PPM-BASED PROFIT GATE: When rebalance_min_profit_ppm > 0, the threshold
        # scales linearly with rebalance_amount, decoupling acceptance from chunk size.
        if self.config.rebalance_min_profit_ppm > 0:
            profit_threshold = (rebalance_amount * self.config.rebalance_min_profit_ppm) // 1_000_000
        else:
            profit_threshold = self.config.rebalance_min_profit

        # Check Profit against Dynamic Threshold
        if expected_profit < profit_threshold:
            self.plugin.log(
                f"REBALANCE SKIPPED: Profit {expected_profit} < Threshold {profit_threshold}",
                level='debug'
            )
            return None

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
            source_turnover_rate=source_turnover_rate,
            expected_fee_sats=expected_fee_sats,
            ev_base_fee_ppm=ev_base_fee_ppm,
            reason_code=RebalanceReasonCode.EV_POSITIVE.value,
            bleeder_status=dest_bleeder_status,
            source_candidate_peer_ids=[info.get("peer_id", "") for _, info, _, _ in source_candidates],
            hot_channel_protection=bool(hot_profile.get('eligible')),
            hot_channel_protection_score=float(hot_score),
            dynamic_budget_override_sats=int(hot_budget_override_sats),
            dynamic_channel_profit_budget_sats=int(hot_channel_profit_budget_sats),
            recommended_cooldown_hours=float(hot_cooldown_hours),
            hive_route_hops=hive_route.get("hops", 0) if hive_route else 0,
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

    def _get_kalman_metrics(self, channel_id: str) -> Dict[str, float]:
        """Retrieve Kalman filter state for a channel from the DB.

        Returns dict with flow_ratio, flow_velocity, uncertainty (sqrt of variance_ratio).
        Falls back to neutral defaults if unavailable.
        """
        try:
            ks = self.database.get_kalman_state(channel_id)
            if ks:
                return {
                    "flow_ratio": float(ks.get("flow_ratio", 0.0)),
                    "velocity": float(ks.get("flow_velocity", 0.0)),
                    "uncertainty": math.sqrt(max(0.0, float(ks.get("variance_ratio", 0.1)))),
                }
        except Exception:
            pass
        return {"flow_ratio": 0.0, "velocity": 0.0, "uncertainty": 0.316}

    def _estimate_expected_fee_sats(self, dest_peer_id: str, rebalance_amount: int) -> int:
        """Estimate the expected routing fee (not the max budget) for a rebalance.

        Uses historical median when available, falls back to estimated inbound fee.
        This is used for EV calculation (expected cost), while max_budget_sats
        remains the hard execution cap passed to Sling.
        """
        try:
            hist_data = self.database.get_historical_inbound_fee_ppm(dest_peer_id)
            if hist_data and hist_data.get('sample_count', 0) >= 3:
                median_ppm = hist_data.get('median_fee_ppm', 0)
                return max(1, (rebalance_amount * median_ppm) // 1_000_000)
        except Exception:
            pass
        # Fallback: use the inbound fee estimate (already accounts for historical data)
        inbound_ppm = self._estimate_inbound_fee(dest_peer_id)
        return max(1, (rebalance_amount * inbound_ppm) // 1_000_000)

    def _estimate_daily_channel_contribution(self, prof: Optional[Any]) -> float:
        if not prof:
            return 0.0
        try:
            total = float(getattr(getattr(prof, 'revenue', None), 'total_contribution_sats', 0) or 0)
            days_open = int(getattr(prof, 'days_open', 0) or 0)
            days = max(1, min(days_open, 30))
            return max(0.0, total / days)
        except Exception:
            return 0.0

    def _compute_hot_channel_protection(self, *, dest_channel: str, dest_peer_id: str, dest_flow_state: str, dest_ratio: float,
                                        velocity: float, prof: Optional[Any], cfg: Optional[ConfigSnapshot] = None) -> Dict[str, Any]:
        cfg = cfg or self.config.snapshot()
        if not getattr(cfg, 'hot_channel_protection_enabled', False):
            return {'enabled': False, 'eligible': False, 'reason': 'disabled'}
        override_peers_raw = str(getattr(cfg, 'hot_channel_protection_override_peers', '') or '')
        override_peers = {p.strip() for p in override_peers_raw.split(',') if p.strip()}
        override_min_depletion_trigger_pct = None
        try:
            if self.database is not None:
                db_override_rows = self.database.list_hot_channel_protection_override_peers()
                for r in (db_override_rows or []):
                    pid = str(r.get('peer_id') or '').strip()
                    if pid:
                        override_peers.add(pid)
                    if dest_peer_id and pid == dest_peer_id:
                        try:
                            pct = r.get('min_depletion_trigger_pct')
                            pctf = float(pct) if pct is not None else None
                            if pctf is not None and 0.0 < pctf <= 100.0:
                                override_min_depletion_trigger_pct = pctf
                        except Exception:
                            pass
        except Exception as e:
            self.plugin.log(f"Hot-channel override peer lookup failed: {e}", level='debug')
        peer_forced = bool(dest_peer_id and dest_peer_id in override_peers)
        if str(dest_flow_state) != 'source' and not peer_forced:
            return {'enabled': True, 'eligible': False, 'reason': 'not_source'}
        marginal_roi = float(getattr(prof, 'marginal_roi', 0.0) or 0.0)
        daily_contrib_est = self._estimate_daily_channel_contribution(prof)

        # Peer-history hot/profit inheritance: use recently closed channels to the same peer
        # so replacement channels are protected without waiting for local history to build.
        inherited = None
        inherited_active = False
        try:
            if self.database is not None and dest_peer_id:
                inherited = self.database.get_peer_closed_channel_profit_summary(
                    dest_peer_id, lookback_days=30, limit=5
                )
                recency_sec = int(time.time()) - int((inherited or {}).get('most_recent_closed_at') or 0)
                channel_is_new = (prof is None) or (int(getattr(prof, 'days_open', 0) or 0) <= 1)
                inherited_daily = int((inherited or {}).get('daily_revenue_est_sats', 0) or 0)
                inherited_roi = float((inherited or {}).get('marginal_roi_proxy', 0.0) or 0.0)
                inherited_remote_closes = int((inherited or {}).get('remote_close_count', 0) or 0)
                inherited_active = bool(
                    channel_is_new and
                    int((inherited or {}).get('count', 0) or 0) > 0 and
                    0 <= recency_sec <= (14 * 86400) and
                    (inherited_daily > 0 or inherited_roi > 0 or inherited_remote_closes > 0)
                )
                if inherited_active:
                    # Use inherited profitability signal when current channel is too new.
                    daily_contrib_est = max(daily_contrib_est, inherited_daily)
                    marginal_roi = max(marginal_roi, inherited_roi)
        except Exception as e:
            self.plugin.log(f"Hot-channel inheritance lookup failed for {dest_peer_id[:16]}...: {e}", level='debug')

        if not prof and not peer_forced and not inherited_active:
            return {'enabled': True, 'eligible': False, 'reason': 'no_profitability'}

        if inherited_active:
            # New replacement channels may not have flow-window state yet; seed velocity to threshold floor.
            velocity = max(velocity, float(getattr(cfg, 'hot_channel_protection_min_velocity', 0.20) or 0.20))

        if velocity < float(getattr(cfg, 'hot_channel_protection_min_velocity', 0.20) or 0.20) and not (peer_forced or inherited_active):
            return {'enabled': True, 'eligible': False, 'reason': 'velocity_below_threshold', 'velocity': velocity, 'marginal_roi': marginal_roi}
        if marginal_roi < float(getattr(cfg, 'hot_channel_protection_min_marginal_roi', 0.20) or 0.20) and not (peer_forced or inherited_active):
            return {'enabled': True, 'eligible': False, 'reason': 'roi_below_threshold', 'velocity': velocity, 'marginal_roi': marginal_roi}

        if daily_contrib_est <= 0 and not (peer_forced or inherited_active):
            return {'enabled': True, 'eligible': False, 'reason': 'no_daily_contribution', 'velocity': velocity, 'marginal_roi': marginal_roi}
        if daily_contrib_est <= 0 and (peer_forced or inherited_active):
            daily_contrib_est = 1000.0  # conservative floor for explicit operator override

        vel_thr = max(0.0001, float(getattr(cfg, 'hot_channel_protection_min_velocity', 0.20) or 0.20))
        roi_thr = max(0.0001, float(getattr(cfg, 'hot_channel_protection_min_marginal_roi', 0.20) or 0.20))
        velocity_score = max(0.0, min(1.0, (velocity - vel_thr) / max(vel_thr, 0.05)))
        roi_score = max(0.0, min(1.0, (marginal_roi - roi_thr) / max(roi_thr, 0.20)))
        contrib_score = max(0.0, min(1.0, daily_contrib_est / 5000.0))
        depletion_score = max(0.0, min(1.0, (0.50 - float(dest_ratio)) / 0.50))
        score = max(0.0, min(1.0, 0.35*velocity_score + 0.30*roi_score + 0.20*contrib_score + 0.15*depletion_score))
        if peer_forced:
            score = max(score, 0.70)
        elif inherited_active:
            score = max(score, 0.60)

        profit_budget_pct = float(getattr(cfg, 'hot_channel_protection_profit_budget_pct', 0.75) or 0.75)
        # I-12 FIX: Use ceiling instead of floor to prevent truncation to 0 for small
        # daily_contrib_est values (e.g., 0.5 sats/day * 0.75 = 0.375 -> was 0, now 1)
        channel_profit_budget_sats = max(1, int(math.ceil(max(0.0, daily_contrib_est) * max(0.0, min(1.0, profit_budget_pct)))))

        max_chunk_mult = max(1.0, float(getattr(cfg, 'hot_channel_protection_max_chunk_multiplier', 4.0) or 4.0))
        chunk_multiplier = 1.0 + ((max_chunk_mult - 1.0) * score)

        base_cd = float(getattr(cfg, 'rebalance_cooldown_hours', 24) or 24)
        min_cd = max(0.0, float(getattr(cfg, 'hot_channel_protection_min_cooldown_hours', 1.0) or 1.0))
        recommended_cooldown_hours = max(min_cd, base_cd * (1.0 - 0.85 * score))

        target_ratio_boost = 0.10 * score  # can raise source target from 85% to ~95%

        return {
            'enabled': True,
            'eligible': True,
            'reason': (
                'hot_source_profitable_forced'
                if peer_forced else
                ('hot_source_profitable_inherited' if inherited_active else 'hot_source_profitable')
            ),
            'score': round(score, 4),
            'peer_forced': bool(peer_forced),
            'peer_override_min_depletion_trigger_pct': override_min_depletion_trigger_pct,
            'peer_history_inherited': bool(inherited_active),
            'peer_history_summary': inherited if isinstance(inherited, dict) and inherited_active else None,
            'dest_peer_id': dest_peer_id,
            'velocity': velocity,
            'marginal_roi': marginal_roi,
            'daily_contribution_est_sats': int(daily_contrib_est),
            'channel_profit_budget_sats': max(0, channel_profit_budget_sats),
            'chunk_multiplier': round(chunk_multiplier, 3),
            'recommended_cooldown_hours': round(recommended_cooldown_hours, 2),
            'target_ratio_boost': round(target_ratio_boost, 4),
        }

    def _estimate_push_ev(self, src_channel: str, src_info: Dict,
                          src_ratio: float, dest_scids: List[str],
                          dest_peer_ids: Optional[List[str]] = None) -> Optional[RebalanceCandidate]:
        """Estimate EV for a push rebalance (draining an overfull channel)."""
        cfg = self.config.snapshot()
        capacity = src_info.get("capacity", 0)
        if capacity == 0:
            return None

        # Push amount: drain to 50% outbound ratio
        target_ratio = 0.50
        excess_sats = int((src_ratio - target_ratio) * capacity)
        # M-15: Skip if excess is below minimum (avoid inflating small amounts)
        if excess_sats < cfg.rebalance_min_amount:
            return None
        amount = min(excess_sats, cfg.rebalance_max_amount)

        src_peer_id = src_info.get("peer_id", "")
        src_fee = src_info.get("fee_ppm", 0)
        inbound_fee = self._estimate_inbound_fee(src_peer_id)

        # For push, the "outbound fee" is what we earn when traffic flows in the
        # direction we're creating capacity for (the reverse direction)
        spread = src_fee - inbound_fee
        # M-17: Negative spread means we'd lose money on this direction
        if spread <= 0:
            return None
        # B10 FIX: Guard kelly_fraction behind enable_kelly, matching pull path.
        kelly = cfg.kelly_fraction if cfg.enable_kelly else 1.0
        max_fee_ppm = max(1, int(spread * kelly))
        max_budget = max(1, (amount * max_fee_ppm + 999_999) // 1_000_000)

        # B3 FIX: Use primary destination peer for fee estimation, not the source.
        # In push rebalancing, routing fees go through destination peers.
        primary_dest_peer = dest_peer_ids[0] if dest_peer_ids else src_peer_id
        expected_fee_sats = self._estimate_expected_fee_sats(primary_dest_peer, amount)
        expected_fee_sats = min(expected_fee_sats, max_budget)

        # I-1 FIX: Apply utilization discount to push EV (same as pull EV).
        # Push targets are channels with 3+ source failures, which may have low demand.
        src_turnover = self._calculate_turnover_rate(src_channel, capacity)
        cooldown_days = max(0.01, getattr(cfg, 'rebalance_cooldown_hours', 24) / 24.0)
        expected_utilization = max(0.05, min(1.0, src_turnover * cooldown_days))

        # Calculate expected profit with utilization discount
        expected_income = int(spread * amount * expected_utilization / 1_000_000)
        expected_profit = expected_income - expected_fee_sats

        # Resolve peer IDs for the destination (source_candidates in push semantics)
        resolved_peer_ids = dest_peer_ids or []

        return RebalanceCandidate(
            source_candidates=dest_scids,
            to_channel=src_channel,
            primary_source_peer_id=resolved_peer_ids[0] if resolved_peer_ids else "",
            to_peer_id=src_peer_id,
            amount_sats=amount,
            amount_msat=amount * 1000,
            outbound_fee_ppm=src_fee,
            inbound_fee_ppm=inbound_fee,
            source_fee_ppm=0,
            weighted_opp_cost_ppm=0,
            spread_ppm=spread,
            max_budget_sats=max_budget,
            max_budget_msat=max_budget * 1000,
            max_fee_ppm=max_fee_ppm,
            expected_profit_sats=expected_profit,
            liquidity_ratio=src_ratio,
            dest_flow_state="push_drain",
            dest_turnover_rate=0.0,
            source_turnover_rate=0.0,
            expected_fee_sats=expected_fee_sats,
            direction="push",
            source_candidate_peer_ids=resolved_peer_ids,
        )

    def _estimate_inbound_fee(self, peer_id: str, amount_msat: int = 100000000) -> int:
        """
        Estimate the inbound routing fee to reach a peer.

        Prioritizes historical actual costs over heuristics.

        Priority order:
        0. Hive member - direct channel, 0 fee
        1. Historical data (high confidence) - Use median, most accurate
        2. Historical data (medium) - Blend with last-hop fee
        3. Historical data (low) - Use with buffer
        4. Last hop fee + buffer - Gossip-based estimate
        5. Route estimation - Ask CLN for a route
        6. Default fallback - configured inbound_fee_estimate_ppm

        Returns:
            Estimated inbound fee in PPM
        """
        # Priority 0: Hive fleet members have direct channels with 0 fee
        if self.hive_hints and peer_id and self.hive_hints.is_hive_member(peer_id):
            self.plugin.log(
                f"INBOUND FEE EST [{peer_id[:12]}...]: Hive fleet member, 0 PPM",
                level='debug'
            )
            return 0

        # =====================================================================
        # Historical-First Fee Estimation
        # =====================================================================
        # Real rebalance costs are the ground truth. Use them when available.
        # Historical data accounts for actual multi-hop routes, not just last hop.
        # =====================================================================

        hist_data = self.database.get_historical_inbound_fee_ppm(peer_id)
        last_hop = self._get_last_hop_fee(peer_id, amount_msat=amount_msat)

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

        # --- Cost-Curve from Failures ---
        # If we have recent failures, the true cost floor is higher than the failed attempts
        failed_floor = 0
        try:
            # Look at recent failures to this peer (joins through channel_states for SCID→peer mapping)
            recent_peer_rebalances = self.database.get_rebalance_history_by_peer(peer_id, limit=20)
            peer_fails = [
                f for f in recent_peer_rebalances
                if f.get('status') == 'failed'
            ]
            if peer_fails:
                # Find the highest PPM that resulted in a routing failure.
                # Use each record's own amount_sats for accurate PPM conversion.
                failed_ppms = []
                for f in peer_fails:
                    f_fee = f.get('max_fee_sats', 0) or 0
                    f_amt = f.get('amount_sats', 0) or 0
                    if f_fee > 0 and f_amt > 0:
                        failed_ppms.append((f_fee * 1_000_000) // f_amt)
                if failed_ppms:
                    failed_floor = max(failed_ppms)
        except Exception:
            pass

        # No historical data - fall back to heuristics
        if last_hop is not None:
            estimate = last_hop + self.config.inbound_fee_estimate_ppm

            # Ensure our estimate is strictly above the proven failure floor
            if failed_floor > 0 and estimate <= failed_floor:
                estimate = failed_floor + 25  # Small buffer above the known failure point

            self.plugin.log(
                f"INBOUND FEE EST [{peer_id[:12]}...]: Last-hop based "
                f"{estimate} PPM (last_hop={last_hop}, fail_floor={failed_floor})",
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

        # Ultimate fallback: use configured estimate (not hardcoded 1000 PPM)
        fallback = self.config.inbound_fee_estimate_ppm
        self.plugin.log(
            f"INBOUND FEE EST [{peer_id[:12]}...]: Default fallback {fallback} PPM",
            level='debug'
        )
        return fallback

    def _get_last_hop_fee(self, peer_id: str, amount_msat: int = 100000000) -> Optional[int]:
        """
        Get the fee for the last hop from a peer to us.
        
        ENHANCED: Now prefers actual peer fees from listpeerchannels.updates.remote
        over gossip-based listchannels data. This is more accurate and avoids
        stale gossip issues.
        
        Uses memoization via self._fee_cache to avoid repeated lookups
        within a single find_rebalance_candidates run.
        """
        # Check cache first (memoization for this run)
        cache_key = (peer_id, int(amount_msat or 0))
        if cache_key in self._fee_cache:
            return self._fee_cache[cache_key]
        
        result = None
        
        # PRIORITY 1: Use actual peer inbound fee from listpeerchannels.updates.remote
        # This is the most accurate source - directly from our channel state, not gossip
        if hasattr(self, '_peer_inbound_fees') and peer_id in self._peer_inbound_fees:
            peer_fee_info = self._peer_inbound_fees[peer_id]
            ppm = int(peer_fee_info.get("fee_ppm", 0) or 0)
            base_msat = int(peer_fee_info.get("base_msat", 0) or 0)
            # Convert base fee to ppm-equivalent at amount_msat
            base_ppm = int((base_msat * 1_000_000) // max(int(amount_msat or 0), 1))
            base_ppm = min(base_ppm, 1_000_000)  # Cap base fee PPM-equivalent to 100%
            result = ppm + base_ppm
            self.plugin.log(
                f"LAST_HOP_FEE [{peer_id[:12]}...]: Using actual peer fee {result} PPM "
                f"(ppm={ppm}, base_ppm={base_ppm}) from listpeerchannels",
                level='debug'
            )
        else:
            # PRIORITY 2: Fall back to gossip-based listchannels lookup
            try:
                our_id = self._get_our_node_id()
                if our_id:
                    channels = self.plugin.rpc.listchannels(source=peer_id)
                    for ch in channels.get("channels", []):
                        if ch.get("destination") == our_id:
                            ppm = int(ch.get("fee_per_millionth", 0) or 0)
                            base_fee_msat = int(ch.get("base_fee_millisatoshi", 0) or 0)
                            # Convert the base fee (msat) into a ppm-equivalent at amount_msat.
                            base_ppm = int((base_fee_msat * 1_000_000) // max(int(amount_msat or 0), 1))
                            result = ppm + base_ppm
                            self.plugin.log(
                                f"LAST_HOP_FEE [{peer_id[:12]}...]: Using gossip fee {result} PPM "
                                f"(fallback, peer not in channel cache)",
                                level='debug'
                            )
                            break
            except Exception as e:
                self.plugin.log(f"Failed to query gossip for last-hop fee: {e}", level='debug')

        # Cache the result (even if None, to avoid re-querying)
        self._fee_cache[cache_key] = result
        
        return result

    def _get_route_fee_estimate(self, peer_id: str, amount_msat: int) -> Optional[int]:
        if amount_msat <= 0:
            return None
        try:
            route = self.plugin.rpc.getroute(id=peer_id, amount_msat=amount_msat, riskfactor=10, maxhops=6)
            if route.get("route"):
                first_hop = route["route"][0].get("amount_msat", amount_msat)
                first_hop = self._parse_msat(first_hop) if not isinstance(first_hop, int) else first_hop
                return max(0, int(((first_hop - amount_msat) / amount_msat) * 1_000_000))
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
    ) -> List[Tuple[str, Dict[str, Any], int, float]]:
        """
        Select all profitable source channels for a rebalance.

        Instead of returning a single "best" source, this returns ALL sources
        that have a positive spread (EV > 0), sorted by score (highest first).
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
                        level='debug'
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

            # HIVE FLEET: If source peer is a hive member, they charge us
            # 0 fee — the first hop is free.  This dramatically improves the
            # spread for circular routes through the fleet.
            is_hive_source = False
            if pid:
                if self.hive_router:
                    is_hive_source = self.hive_router.is_hive_member(pid)
                elif self.hive_hints:
                    is_hive_source = self.hive_hints.is_hive_member(pid)
                if is_hive_source:
                    source_fee_ppm = 0

            # E5 FIX: Reuse state from source protection check above (line 3244)
            # instead of making a duplicate DB query for the same channel.
            flow_state = state.get("state", "balanced") if state else "balanced"

            # =================================================================
            # FLOW-AWARE OPPORTUNITY COST
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

            # Require non-negative spread to avoid consistent leakage.
            if spread_ppm < 0:
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
            elif flow_state in ("balanced", "balanced_active"):
                # Apply Stagnant Inventory Bonus (only for truly dormant channels)
                if flow_state == "balanced" and source_turnover_rate < 0.0015:
                    score += 10 # Awakening Bonus
                    self.plugin.log(f"STAGNANT BONUS: Applying +10 priority to stagnant channel {cid[:12]}...", level='info')

                score += 20

            # Fleet rebalance bias is the public, bounded integration surface.
            # Convert the multiplicative bias into a moderate additive score delta.
            source_peer = info.get("peer_id", "")
            if source_peer:
                score += int(round((self._get_hive_rebalance_bias(source_peer) - 1.0) * 200))

            # HIVE FLEET BONUS: Strongly prefer hive member sources.
            # Routes through fleet peers have a free first hop, making
            # circular rebalances much cheaper.
            if is_hive_source:
                score += 200
                self.plugin.log(
                    f"HIVE SOURCE BONUS: +200 priority for {cid[:12]}... "
                    f"(peer {pid[:12]}... is fleet member, fee=0)",
                    level='debug'
                )

            # Route-pair bonus: inbound revenue legs are ideal sources.
            # Draining local balance creates headroom for the inbound traffic
            # that generates fees on this route.
            route_pair_ins = getattr(self, '_cycle_route_pair_in_channels', set())
            if normalized in route_pair_ins:
                score += 40

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

        # Track exhaustion for Boltz coordination: depleted channels exist
        # but no profitable rebalance is possible → Boltz should step up
        try:
            self._last_depleted_count = len(depleted_channels)
        except Exception:
            pass
        self._last_profitable_count = len(candidates)
        self._last_cycle_ts = int(time.time())

        return candidates

    def _get_peer_connection_status(self) -> Dict:
        status = {}
        try:
            for p in self.plugin.rpc.listpeers().get("peers", []):
                status[p.get("id")] = {"connected": p.get("connected", False)}
        except Exception as e:
            self.plugin.log(f"Failed to get peer connection status: {e}", level='debug')
        return status

    def _get_channels_with_balances(self) -> Dict[str, Dict[str, Any]]:
        """Get all channels with their current balances and fee info.

        Retries once on RPC failure.  Stores the last error in
        ``_last_balance_error`` so callers can include it in diagnostics.
        """
        self._last_balance_error: Optional[str] = None
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            channels: Dict[str, Dict[str, Any]] = {}
            try:
                listfunds = self.plugin.rpc.listfunds()
                # Use listpeerchannels instead of deprecated listpeers (CLN 23.08+)
                listpeerchannels = self.plugin.rpc.listpeerchannels()

                # Build peer info map from listpeerchannels
                peer_info = {}
                for ch in listpeerchannels.get("channels", []):
                    scid = ch.get("short_channel_id")
                    if scid and ch.get("state") == "CHANNELD_NORMAL":
                        # Extract peer's inbound fee from updates.remote (what they charge us)
                        updates = ch.get("updates", {})
                        remote_updates = updates.get("remote", {})
                        peer_inbound_fee_ppm = remote_updates.get("fee_proportional_millionths")
                        peer_inbound_base_msat = remote_updates.get("fee_base_msat")

                        peer_info[scid] = {
                            "peer_id": ch.get("peer_id"),
                            "fee_ppm": ch.get("fee_proportional_millionths", 0),
                            "base_fee_msat": ch.get("fee_base_msat", 0),
                            "htlcs": len(ch.get("htlcs", [])),
                            # Peer's inbound fee - what they charge for last hop to us
                            "peer_inbound_fee_ppm": peer_inbound_fee_ppm,
                            "peer_inbound_base_msat": peer_inbound_base_msat
                        }

                # Get balances from listfunds
                for channel in listfunds.get("channels", []):
                    if channel.get("state") != "CHANNELD_NORMAL":
                        continue

                    scid = channel.get("short_channel_id", "")
                    if not scid:
                        continue

                    our_amount_msat = self._parse_msat(channel.get("our_amount_msat", 0))
                    amount_msat = self._parse_msat(channel.get("amount_msat", 0))

                    info = peer_info.get(scid, {})
                    channels[scid] = {
                        "capacity": amount_msat // 1000,
                        "spendable_sats": our_amount_msat // 1000,
                        "peer_id": info.get("peer_id", channel.get("peer_id", "")),
                        "fee_ppm": info.get("fee_ppm", 0),
                        "base_fee_msat": info.get("base_fee_msat", 0),
                        "htlcs": info.get("htlcs", 0),
                        # Peer's actual inbound fee from updates.remote (None if unavailable)
                        "peer_inbound_fee_ppm": info.get("peer_inbound_fee_ppm"),
                        "peer_inbound_base_msat": info.get("peer_inbound_base_msat")
                    }

                # Populate peer_id -> peer inbound fee cache for _get_last_hop_fee()
                # This allows _estimate_inbound_fee() to use actual fees instead of gossip
                self._peer_inbound_fees = {}
                for scid, info in channels.items():
                    peer_id = info.get("peer_id")
                    if peer_id and info.get("peer_inbound_fee_ppm") is not None:
                        self._peer_inbound_fees[peer_id] = {
                            "fee_ppm": info["peer_inbound_fee_ppm"],
                            "base_msat": info.get("peer_inbound_base_msat", 0) or 0
                        }

                return channels

            except Exception as e:
                self._last_balance_error = str(e)
                if attempt < max_attempts:
                    self.plugin.log(
                        f"RPC error getting channel balances (attempt {attempt}/{max_attempts}), retrying: {e}",
                        level='warn',
                    )
                    time.sleep(1)
                else:
                    self.plugin.log(
                        f"RPC error getting channel balances after {max_attempts} attempts: {e}",
                        level='error',
                    )

        return {}

    def execute_rebalance(self, candidate: RebalanceCandidate, enforce_budget: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute a rebalance for the given candidate.

        Uses the async JobManager to spawn sling background jobs.
        This plugin acts as the "Strategist" while sling workers handle execution.
        """
        result = {"success": False, "candidate": candidate.to_dict(), "message": ""}
        with self._pending_lock:
            self._pending[candidate.to_channel] = int(time.time())

        # Thread-safe config snapshot for this execution
        cfg = self.config.snapshot()

        rebalance_id: Optional[int] = None
        reserved_budget = False
        job_started = False
        try:
            # Validation: Return error on empty/None channel IDs (HO-01)
            if not candidate.from_channel or not candidate.to_channel:
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="invalid_channel_ids",
                    dominant_input="validation",
                    safety_block=True,
                    budget_blocked=False,
                )
                self.plugin.log(
                    f"Invalid channel IDs: from={candidate.from_channel}, to={candidate.to_channel}",
                    level='error'
                )
                with self._pending_lock:
                    self._pending.pop(candidate.to_channel, None)
                return {
                    "success": False,
                    "error": "Invalid channel IDs - from_channel or to_channel is empty"
                }

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
                rebalance_type=kwargs.get('rebalance_type', 'normal'),
                reason_code=candidate.reason_code,
                bleeder_status=candidate.bleeder_status
            )

            if cfg.dry_run:
                self._set_last_decision_summary(
                    action="rebalance",
                    reason="dry_run",
                    dominant_input=candidate.reason_code,
                    safety_block=False,
                    budget_blocked=False,
                )
                self.plugin.log(
                    f"[DRY RUN] Would rebalance {candidate.amount_sats} sats "
                    f"from {candidate.from_channel} to {candidate.to_channel}"
                )
                self.database.update_rebalance_result(
                    rebalance_id, 'success', 0, candidate.expected_profit_sats
                )
                with self._pending_lock:
                    self._pending.pop(candidate.to_channel, None)
                return {"success": True, "message": "Dry run", "rebalance_id": rebalance_id}

            if enforce_budget:
                # CRITICAL-01 FIX: Atomic budget reservation
                # Reserve budget BEFORE starting the job to prevent concurrent overspend.
                now = int(time.time())
                budget_window_hours = max(1, int(getattr(cfg, "total_cost_budget_window_hours", 24) or 24))
                since_24h = now - (budget_window_hours * 3600)

                # Calculate effective budget (same logic as _check_capital_controls)
                effective_budget = cfg.daily_budget_sats
                if cfg.enable_proportional_budget:
                    revenue_24h = self.database.get_total_routing_revenue(since_24h)
                    proportional_budget = int(revenue_24h * cfg.proportional_budget_pct)
                    effective_budget = max(cfg.daily_budget_sats, proportional_budget)
                # Only override with global provider if one is configured;
                # otherwise preserve the proportional budget calculated above.
                if getattr(self, "global_budget_limit_provider", None) is not None:
                    effective_budget = self._get_global_budget_limit(cfg)

                ext_costs = self._get_external_liquidity_costs()
                ext_spent = int(ext_costs.get("spent_24h_sats", 0) or 0)
                ext_reserved = int(ext_costs.get("reserved_24h_sats", 0) or 0)
                rebalance_budget_limit = max(0, effective_budget - ext_spent - ext_reserved)
                hot_override_limit = int(getattr(candidate, 'dynamic_budget_override_sats', 0) or 0)
                if hot_override_limit > 0:
                    # Candidate-specific protection budget can exceed the standard daily cap.
                    protected_limit = max(0, hot_override_limit - ext_spent - ext_reserved)
                    # Cap aggregate hot channel spend at the configured daily budget.
                    # Hot channel protection provides per-channel priority within the budget,
                    # but never allows total spend to exceed the effective daily budget.
                    max_hot_budget = effective_budget
                    protected_limit = min(protected_limit, max(0, max_hot_budget - ext_spent - ext_reserved))
                    if protected_limit > rebalance_budget_limit:
                        self.plugin.log(
                            f"HOT CHANNEL PROTECTION: Using protected rebalance budget limit {protected_limit} sats "
                            f"(global remaining {rebalance_budget_limit}, channel_profit_budget={hot_override_limit}, "
                            f"aggregate cap={max_hot_budget}) for {db_to_channel}",
                            level='info'
                        )
                        rebalance_budget_limit = protected_limit

                # Compute effective weekly budget for atomic reservation
                _effective_weekly = cfg.weekly_budget_sats
                if cfg.enable_proportional_budget:
                    _weekly_rev = self.database.get_total_routing_revenue(now - 7 * 86400)
                    _prop_weekly = int(_weekly_rev * cfg.proportional_budget_pct)
                    _effective_weekly = max(cfg.weekly_budget_sats, _prop_weekly)

                reserved, remaining = self.database.reserve_budget(
                    reservation_id=str(rebalance_id),
                    amount_sats=db_max_fee,
                    channel_id=db_to_channel,
                    budget_limit=rebalance_budget_limit,
                    since_timestamp=since_24h,
                    weekly_budget_limit=_effective_weekly,
                    weekly_since_timestamp=now - 7 * 86400,
                )
                reserved_budget = bool(reserved)

                if not reserved_budget:
                    # Determine which budget limit blocked the reservation:
                    # remaining reflects the tighter of daily/weekly headroom.
                    # If remaining > 0 but less than what daily alone would allow,
                    # weekly was the binding constraint.
                    _budget_blocker = "daily_budget_sats"
                    if remaining > 0 and remaining < rebalance_budget_limit:
                        _budget_blocker = "weekly_budget_sats"
                    self._set_last_decision_summary(
                        action="suppressed",
                        reason="budget_exhausted",
                        dominant_input=_budget_blocker,
                        safety_block=True,
                        budget_blocked=True,
                    )
                    self.database.update_rebalance_result(
                        rebalance_id, 'failed',
                        error_message=(
                            f"Unified liquidity budget exhausted: {remaining} sats remaining for rebalances "
                            f"after external costs ({ext_spent} spent + {ext_reserved} reserved) "
                            f"of total {effective_budget}"
                        )
                    )
                    result["message"] = (
                        f"Unified liquidity budget exhausted: only {remaining} sats available "
                        f"for rebalances after external costs"
                    )
                    self.plugin.log(
                        f"CAPITAL CONTROL: Budget reservation failed for {db_to_channel}. "
                        f"Remaining for rebalances: {remaining} sats "
                        f"(external costs: spent={ext_spent}, reserved={ext_reserved}, total_budget={effective_budget})",
                        level='warn'
                    )
                    # Budget exhaustion is global; don't backoff a specific channel.
                    with self._pending_lock:
                        self._pending.pop(candidate.to_channel, None)
                    return result

            # Async execution via JobManager (sling background jobs)
            res = self.job_manager.start_job(candidate, rebalance_id)

            if res.get("success"):
                self._set_last_decision_summary(
                    action="rebalance",
                    reason="async_job_started",
                    dominant_input=candidate.reason_code,
                    safety_block=False,
                    budget_blocked=False,
                )
                job_started = True
                if self.hive_router:
                    self.hive_router.reserve_for_job(
                        candidate.to_channel.replace(":", "x"),
                        candidate.amount_msat,
                    )
                # Clean up pending entry - job is now tracked by JobManager
                with self._pending_lock:
                    self._pending.pop(candidate.to_channel, None)
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
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="start_job_failed",
                    dominant_input=candidate.reason_code,
                    safety_block=False,
                    budget_blocked=False,
                )
                self.database.update_rebalance_result(
                    rebalance_id, 'failed', error_message=error
                )
                result["message"] = f"Failed: {error}"
                self.plugin.log(f"Failed to start rebalance job: {error}", level='warn')
                if reserved_budget:
                    self.database.release_budget_reservation(str(rebalance_id))
                with self._pending_lock:
                    self._pending.pop(candidate.to_channel, None)

        except Exception as e:
            self._set_last_decision_summary(
                action="suppressed",
                reason="execution_error",
                dominant_input="execution_error",
                safety_block=False,
                budget_blocked=False,
            )
            result["message"] = str(e)
            self.plugin.log(f"Execution error: {e}", level='error')
            if rebalance_id is not None:
                try:
                    self.database.update_rebalance_result(
                        rebalance_id, 'failed', error_message=str(e)
                    )
                except Exception as db_err:
                    self.plugin.log(f"Failed to record rebalance failure: {db_err}", level='debug')
            if reserved_budget and rebalance_id is not None and not job_started:
                try:
                    self.database.release_budget_reservation(str(rebalance_id))
                except Exception as db_err:
                    self.plugin.log(f"Failed to release budget reservation: {db_err}", level='debug')
            with self._pending_lock:
                self._pending.pop(candidate.to_channel, None)

        return result

    def diagnostic_rebalance(self, channel_id: str) -> Dict[str, Any]:
        """
        Trigger a "Channel Defibrillator" sequence:
        1. Enable bounded low-fee exploration (Passive Lure).
        2. Execute small active rebalance (Active Shock).
        
        This is a diagnostic operation to verify channel liveness before
        confirming a channel as a "Zombie" for closure. The small rebalance
        (50k sats) forces liquidity into the channel immediately rather than
        waiting for organic routing traffic.
        """
        self.plugin.log(f"Defibrillator: Triggering bounded exploration for channel {channel_id}")
        
        # 1. Set the exploration flag in the database (Fee Controller will
        # map it to bounded low-fee exploration above the configured floor)
        self.database.set_channel_probe(channel_id, probe_type='bounded_low_fee')
        
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
                    "message": "Exploration flag set, but no sources available for active shock."
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
            # Note: outbound_fee is 0 because the active shock itself is not the
            # fee-controller path; the controller-side exploration remains non-zero.
            inbound_fee = self._estimate_inbound_fee(dest_info.get('peer_id', ''))
            
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

            # Record in database (execute_once bypasses normal job flow)
            rebalance_id = self.database.record_rebalance(
                from_channel=best_source_id,
                to_channel=channel_id,
                amount_sats=shock_amount,
                max_fee_sats=100,
                expected_profit_sats=-50,
                rebalance_type='diagnostic',
                reason_code='defibrillator'
            )

            result = self.job_manager.execute_once(
                scid=channel_id,
                direction="pull",
                amount=shock_amount,
                maxppm=2000,
                onceamount=shock_amount,
                candidates=[best_source_id]
            )

            # Update database with outcome
            if result.get("success"):
                fee_sats = result.get("actual_fee_sats")
                self.database.update_rebalance_result(rebalance_id, 'success', actual_fee_sats=fee_sats)
            else:
                self.database.update_rebalance_result(
                    rebalance_id, 'failed',
                    error_message=result.get("error", "sling-once failed")
                )

            return {
                "success": True,
                "message": f"Defibrillator active: Zero-Fee flag set + Shock {'completed' if result.get('success') else 'failed'}"
            }

        except Exception as e:
            self.plugin.log(f"Defibrillator shock failed: {e}", level='error')
            return {
                "success": False,
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
        # Normalize SCIDs to 'x' format for consistent DB storage and queries
        from_channel = from_channel.replace(':', 'x')
        to_channel = to_channel.replace(':', 'x')
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
        est_in = self._estimate_inbound_fee(t_info.get("peer_id", ""))
        
        if max_fee_sats is None:
            # Calculate a budget for a manual push based on estimated spread
            max_fee_sats = int(amount_sats * (fee_ppm - est_in - src_ppm) / 1e6)
            if max_fee_sats <= 0:
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
        # Manual rebalances use execute_once (blocking sling-once) and bypass budget reservations.
        # Fees are still recorded in history and will reduce budget available for automated runs.
        rebalance_id = self.database.record_rebalance(
            from_channel=from_channel,
            to_channel=to_channel,
            amount_sats=amount_sats,
            max_fee_sats=max_fee_sats,
            expected_profit_sats=0,
            rebalance_type='manual',
            reason_code='manual'
        )

        once_result = self.job_manager.execute_once(
            scid=to_channel,
            direction="pull",
            amount=amount_sats,
            maxppm=max_fee_ppm,
            onceamount=amount_sats,
            candidates=[from_channel]
        )

        if once_result.get("success"):
            fee_sats = once_result.get("actual_fee_sats")
            self.database.update_rebalance_result(rebalance_id, 'success', actual_fee_sats=fee_sats)
            if fee_sats and fee_sats > 0:
                try:
                    self.database.record_rebalance_cost(
                        channel_id=to_channel,
                        peer_id=t_info.get("peer_id", ""),
                        cost_sats=int(fee_sats),
                        amount_sats=amount_sats,
                        timestamp=int(time.time())
                    )
                except Exception as e:
                    self.plugin.log(f"Failed to record rebalance cost for {to_channel}: {e}", level='debug')
            result = {"success": True, "message": once_result.get("message", "completed"), "actual_fee_sats": fee_sats}
        else:
            self.database.update_rebalance_result(
                rebalance_id, 'failed',
                error_message=once_result.get("error", "")
            )
            result = {"success": False, "error": once_result.get("error", "sling-once failed")}

        # Include capital controls warning in result (unless force=True)
        if not capital_ok and not force:
            result['capital_controls_warning'] = "Budget exhausted or reserve low (manual override)"

        return result

    def _check_capital_controls(self, cfg: Optional[ConfigSnapshot] = None) -> bool:
        """Check if capital controls allow rebalancing.

        Reserve check (RPC-dependent) fails open on timeout — a transient
        RPC issue should not block all rebalancing.  Budget check (DB-only)
        always runs and fails closed.
        """
        if cfg is None:
            cfg = self.config.snapshot()
        try:
            # --- Reserve check (needs listfunds RPC) ---
            try:
                listfunds = self.plugin.rpc.listfunds()
                onchain_sats = 0
                for output in listfunds.get("outputs", []):
                    if output.get("status") == "confirmed":
                        amount_msat = output.get("amount_msat", 0)
                        amount_msat = self._parse_msat(amount_msat)
                        onchain_sats += amount_msat // 1000

                channel_spendable_sats = 0
                for channel in listfunds.get("channels", []):
                    if channel.get("state") != "CHANNELD_NORMAL":
                        continue
                    our_amount_msat = self._parse_msat(channel.get("our_amount_msat", 0))
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
            except RpcError:
                # RPC timeout / failure — skip reserve check, still enforce budget
                self.plugin.log(
                    "Capital controls: listfunds RPC failed, skipping reserve check (budget still enforced)",
                    level='warn'
                )

            # --- Budget check (DB-only, no RPC needed) ---
            # Calculate effective daily budget
            # If proportional budget enabled: max(fixed_floor, revenue_24h * percentage)
            effective_budget = cfg.daily_budget_sats

            # R-R6-1 FIX: Compute budget_window_hours once, used by both
            # proportional budget and fee-spent check below.
            now = int(time.time())
            budget_window_hours = max(1, int(getattr(cfg, "total_cost_budget_window_hours", 24) or 24))

            if cfg.enable_proportional_budget:
                # E4 FIX: Use configurable budget window, matching execute_rebalance (line 3760)
                # instead of hardcoded 86400 which ignores total_cost_budget_window_hours.
                revenue_window = self.database.get_total_routing_revenue(now - (budget_window_hours * 3600))
                proportional_budget = int(revenue_window * cfg.proportional_budget_pct)
                effective_budget = max(cfg.daily_budget_sats, proportional_budget)

                self.plugin.log(
                    f"CAPITAL CONTROL: Revenue-proportional budget active. "
                    f"Revenue {budget_window_hours}h: {revenue_window} sats, {cfg.proportional_budget_pct*100:.1f}% = {proportional_budget} sats, "
                    f"Effective budget: {effective_budget} sats (floor: {cfg.daily_budget_sats})",
                    level='debug'
                )
            # Only override with global provider if one is configured;
            # otherwise preserve the proportional budget calculated above.
            if getattr(self, "global_budget_limit_provider", None) is not None:
                effective_budget = self._get_global_budget_limit(cfg)

            fees_spent_24h = self.database.get_total_rebalance_fees(now - (budget_window_hours * 3600))
            ext_costs = self._get_external_liquidity_costs()
            ext_spent = int(ext_costs.get("spent_24h_sats", 0) or 0)
            ext_reserved = int(ext_costs.get("reserved_24h_sats", 0) or 0)
            # Gate on actual spending only — each subsystem (Boltz, rebalancer)
            # enforces its own reservation limits independently.
            total_actual_spent = fees_spent_24h + ext_spent
            if total_actual_spent >= effective_budget:
                self.plugin.log(
                    f"CAPITAL CONTROL: Unified liquidity budget exceeded "
                    f"(rebalance_fees={fees_spent_24h} + external_spent={ext_spent} "
                    f"= {total_actual_spent} >= {effective_budget})",
                    level='warn'
                )
                self._capital_control_blocker = "daily_budget_sats"
                return False
            if ext_reserved > 0:
                self.plugin.log(
                    f"CAPITAL CONTROL: External reservations={ext_reserved} sats pending "
                    f"(actual_spent={total_actual_spent}/{effective_budget})",
                    level='debug'
                )

            # --- Weekly budget check ---
            effective_weekly = cfg.weekly_budget_sats

            if cfg.enable_proportional_budget:
                weekly_revenue = self.database.get_total_routing_revenue(now - 7 * 86400)
                proportional_weekly = int(weekly_revenue * cfg.proportional_budget_pct)
                effective_weekly = max(cfg.weekly_budget_sats, proportional_weekly)
                self.plugin.log(
                    f"CAPITAL CONTROL: Proportional weekly budget. "
                    f"Revenue 7d: {weekly_revenue} sats, {cfg.proportional_budget_pct*100:.1f}% = {proportional_weekly} sats, "
                    f"Effective weekly: {effective_weekly} sats (floor: {cfg.weekly_budget_sats})",
                    level='debug'
                )

            weekly_fees_spent = self.database.get_total_rebalance_fees(now - 7 * 86400)
            # B2 FIX: ext_spent is a 24h figure. Scale to 7d for weekly comparison.
            weekly_total_spent = weekly_fees_spent + (ext_spent * 7)
            if weekly_total_spent >= effective_weekly:
                self.plugin.log(
                    f"CAPITAL CONTROL: Weekly budget exceeded "
                    f"(rebalance_fees_7d={weekly_fees_spent} + external_spent_7d_est={ext_spent * 7} "
                    f"= {weekly_total_spent} >= {effective_weekly})",
                    level='warn'
                )
                self._capital_control_blocker = "weekly_budget_sats"
                return False

            return True
        except Exception as e:
            self.plugin.log(f"Error checking capital controls: {e}", level='error')
            return False
    
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
