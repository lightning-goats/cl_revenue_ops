"""
EV-Based Rebalancer module for cl-revenue-ops

Automatic rebalance cycles delegate to RebalanceEngineV2 via
find_rebalance_candidates(). Manual rebalance RPCs (manual_rebalance,
execute_rebalance) run through the shared RebalanceEngineV2 native-route
execution path.

JobManager is a stripped stub that retains only source-failure tracking
(used by the v1 manual paths) and no-op properties referenced by
diagnostic RPCs and cl-revenue-ops.py.
"""

import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from enum import Enum

from pyln.client import Plugin, RpcError

from .config import Config, ConfigSnapshot
from .database import Database
from .rebalance_execution import stable_failure_reason
from .rebalance_state_v2 import build_state_snapshot as build_state_snapshot_v2
from .policy_manager import PolicyManager
from .utils import parse_msat as _shared_parse_msat, base_to_sats_floor, base_to_sats_ceil, sats_to_base

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer
    from .capex_budget import CapexBudgetEngine


# Operator ruling D4 (2026-07-01): static ceiling on the diagnostic
# ("defibrillator") shock fee cap. Whatever diagnostic_rebalance_max_fee_sats
# is configured to, the effective envelope never exceeds this (nor the daily
# budget) — a typo cannot authorize huge diagnostic spend.
DIAGNOSTIC_FEE_CAP_CEILING_SATS = 10_000


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
    CAPEX_FALLBACK = "capex_fallback"             # Capex-aware fallback rebalance
    HIVE_EQUALIZATION = "hive_equalization"       # Fallback pure-hive inventory equalization
    HIVE_PUSH = "hive_push"                      # Deploy capital to fleet member channels
    COORDINATED_REBALANCE = "coordinated_rebalance"  # Matched assigned fleet coordination hint

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
    coordination_hint_type: str = ""
    coordination_hint_id: str = ""
    coordination_rank_bonus: float = 0.0

    # Direction: "pull" fills to_channel from sources; "push" drains to_channel to destinations
    direction: str = "pull"

    # Multi-source peer IDs aligned with source_candidates (best-first).
    # Optional for backward compatibility; when absent, callers may fall back to primary_source_peer_id.
    source_candidate_peer_ids: List[str] = field(default_factory=list)

    # Hive route discovery: if askrene found a cheap fleet route, store hop count
    # for RebalanceEngineV2 fleet-aware routing.
    hive_route_hops: int = 0
    # True if destination peer is a hive member — enables zero-fee return hop
    dest_is_hive_member: bool = False

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
            "coordination_hint_type": self.coordination_hint_type,
            "coordination_hint_id": self.coordination_hint_id,
            "coordination_rank_bonus": round(self.coordination_rank_bonus, 4),
            "direction": self.direction,
            "hot_channel_protection": self.hot_channel_protection,
            "hot_channel_protection_score": round(self.hot_channel_protection_score, 4),
            "dynamic_budget_override_sats": self.dynamic_budget_override_sats,
            "dynamic_channel_profit_budget_sats": self.dynamic_channel_profit_budget_sats,
            "recommended_cooldown_hours": round(self.recommended_cooldown_hours, 2) if self.recommended_cooldown_hours else 0.0,
            "expected_fee_sats": self.expected_fee_sats,
        }


class JobManager:
    """Stripped stub retaining only live surface.

    - Source failure tracking used by the v1 manual rebalance paths.
    - No-op properties referenced by diagnostic RPCs and cl-revenue-ops.py.
    """

    def __init__(self, plugin: Plugin, config: Config, database: Database):
        self.plugin = plugin
        self.config = config
        self.database = database

        self.source_failure_counts: Dict[str, float] = {}
        self._source_failures_lock = threading.Lock()

        self.hive_router = None

    @property
    def active_job_count(self) -> int:
        return 0

    @property
    def active_channels(self) -> list:
        return []

    def has_active_job(self, channel_id: str) -> bool:
        return False

    def slots_available(self) -> int:
        return 999

    def get_active_rebalancing_peers(self) -> List[str]:
        return []

    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        return []

    def stop_job(self, channel_id: str, reason: str = "manual") -> bool:
        return False

    def stop_all_jobs(self, reason: str = "shutdown") -> int:
        return 0

    # ---- Source failure tracking (still used by EVRebalancer) ----

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

    def get_source_failure_count(self, channel_id: str) -> float:
        """Get the recent failure count for a source channel."""
        with self._source_failures_lock:
            return self.source_failure_counts.get(channel_id, 0.0)

    # ---- AskRene constraint cache (still used by EVRebalancer) ----





class EVRebalancer:
    """
    Expected Value based rebalancer with async job queue support.

    This class acts as the "Strategist" - it calculates EV and determines
    IF and HOW MUCH to rebalance. The actual execution is delegated to
    RebalanceEngineV2, which prices pairs locally and executes explicit routes.

    Thread Safety (I-13, I-14, S-9):
    The rebalance cycle runs single-threaded on a timer. Candidate evaluation,
    EV calculation, and executor dispatch all happen within one timer callback.
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
        # P2-005: Guards _fee_cache and _peer_inbound_fees. The rebalance daemon
        # (T3) and a manual force=false rebalance (T0, which bypasses the engine
        # cycle single-flight) both read/write these instance caches. Without a
        # lock a `= {}` reset can clear the dict mid-read on the other thread,
        # raising KeyError or returning a torn/partial fee. The lock is only ever
        # held for O(1) dict ops — never across an RPC — so it cannot stall the
        # other thread.
        self._cache_lock = threading.Lock()
        self._peer_inbound_fees: Dict[str, Dict[str, int]] = {}
        self._profitability_analyzer: Optional['ChannelProfitabilityAnalyzer'] = None
        self._capex_engine: Optional['CapexBudgetEngine'] = None

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
        self.rebalance_engine_v2 = None  # Injected by plugin init

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
        self.data_service = None  # Unified data service (injected by main plugin)


    @property
    def hive_router(self):
        return self._hive_router

    @hive_router.setter
    def hive_router(self, value):
        self._hive_router = value
        # Propagate to job_manager so _handle_job_* methods can call unreserve
        self.job_manager.hive_router = value

    def _execute_candidate_v2(
        self,
        candidate: RebalanceCandidate,
        rebalance_id: Optional[int] = None,
    ):
        """Execute one candidate through the shared v2 engine.

        ``rebalance_id``: the caller's existing rebalance_history row. The
        engine updates that row in place instead of inserting its own
        'pending' row, so each rebalance produces exactly one history row.
        """
        if self.rebalance_engine_v2 is None:
            raise RuntimeError("no rebalance engine available")
        return self.rebalance_engine_v2.execute_candidate(
            candidate, rebalance_id=rebalance_id
        )

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

    def _derive_hold_reason(self, engine: Any) -> str:
        """Map the engine's last-cycle debug into a specific hold reason.

        Priority order:
        1. Pair-level rejection on a considered candidate (route_over_budget,
           below_hold_margin, no_route, pair_cooldown, native_unavailable,
           pair_futility) -- a pair did form but failed downstream.
        2. Channel-level hold from hold_diagnostics -- no pair could form.
        3. Fallback: no_rebalance_candidates.
        """
        if engine is None or not hasattr(engine, "get_last_cycle_debug"):
            return "no_rebalance_candidates"
        try:
            debug = engine.get_last_cycle_debug()
        except Exception:
            return "no_rebalance_candidates"

        for candidate in debug.get("considered_candidates", []) or []:
            decomp = candidate.get("score_decomposition") or {}
            reason = decomp.get("rejection_reason")
            if reason:
                return str(reason)

        diagnostics = debug.get("hold_diagnostics") or {}
        priority = (
            "dest_blocked_by_cooldown",
            "dest_not_funded",
            "source_rejected_neutral",
            "source_protected",
            "source_inside_band",
        )
        for bucket in priority:
            if int(diagnostics.get(bucket, 0) or 0) > 0:
                return bucket
        return "no_rebalance_candidates"

    def get_boltz_coordination(self) -> Dict[str, Any]:
        """Return rebalancer exhaustion state for Boltz integration.

        When depleted channels exist but 0 profitable candidates were found,
        the rebalancer is exhausted and Boltz should be more aggressive.
        """
        exhausted = (self._last_depleted_count > 0 and self._last_profitable_count == 0)
        return {
            "rebalancer_exhausted": exhausted,
            "depleted_count": self._last_depleted_count,
            "profitable_count": self._last_profitable_count,
            "cycle_ts": self._last_cycle_ts,
        }

    def _build_hive_liquidity_state_payload(
        self,
        depleted_channels: List[Tuple[str, Dict[str, Any], float]],
        source_channels: List[Tuple[str, Dict[str, Any], float]],
        candidates: List[Any],
    ) -> Dict[str, Any]:
        """Build a local cl-hive liquidity-state payload from the current cycle.

        Channel entries carry capacity in BOTH keys: capacity_msat (primary —
        cl-hive's liquidity_coordinator reads capacity_msat) and capacity_sats
        (compat). ``candidates`` accepts both the legacy RebalanceCandidate
        and the v2 engine's PairCandidate shapes.
        """
        def _channel_entries(
            entries: List[Tuple[str, Dict[str, Any], float]]
        ) -> List[Dict[str, Any]]:
            result = []
            for _channel_id, info, local_pct in entries:
                peer_id = str(info.get("peer_id") or "").strip()
                capacity_sats = int(info.get("capacity", 0) or 0)
                if not peer_id:
                    continue
                result.append({
                    "peer_id": peer_id,
                    "local_pct": float(local_pct),
                    "capacity_msat": capacity_sats * 1000,
                    "capacity_sats": capacity_sats,
                })
            return result

        depleted_payload = _channel_entries(depleted_channels)
        saturated_payload = _channel_entries(source_channels)

        liquidity_needs = []
        seen_pairs = set()
        for candidate in candidates[:10]:
            # Legacy RebalanceCandidate uses primary_source_peer_id/to_peer_id;
            # v2 PairCandidate uses source_peer_id/dest_peer_id.
            source_peer_id = str(
                getattr(candidate, "primary_source_peer_id", None)
                or getattr(candidate, "source_peer_id", None)
                or ""
            ).strip()
            destination_peer_id = str(
                getattr(candidate, "to_peer_id", None)
                or getattr(candidate, "dest_peer_id", None)
                or ""
            ).strip()
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
            amount_sats = int(getattr(candidate, "amount_sats", 0) or 0)
            expected_profit_sats = int(
                getattr(candidate, "expected_profit_sats", 0) or 0
            )
            liquidity_needs.append({
                "source_peer_id": source_peer_id,
                "destination_peer_id": destination_peer_id,
                "capacity_msat": amount_sats * 1000,
                "capacity_sats": amount_sats,
                "priority_tier": "high" if expected_profit_sats > 0 else "medium",
                "flow_state": str(getattr(candidate, "dest_flow_state", "") or ""),
                "expected_profit_sats": expected_profit_sats,
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
        candidates: List[Any],
    ) -> None:
        """Push liquidity state to CLN datastore for cl-hive to read.

        Uses datastore (fast local write) instead of cross-plugin RPC
        that was timing out every cycle.
        """
        payload = self._build_hive_liquidity_state_payload(
            depleted_channels,
            source_channels,
            candidates,
        )

        if self.data_service:
            self.data_service.datastore_push(["revenue", "liquidity-state"], payload)

    def _report_liquidity_state_from_cycle(self, cycle_result: Any, cfg: Any) -> None:
        """Report REAL liquidity state from a completed v2 engine cycle.

        Audit fix (fleet liquidity pipeline): the liquidity-state payload was
        previously written only on early-suppression paths with hardcoded
        empty channel lists, so cl-hive never saw real member state.

        depleted = channels below the planner band-low,
        source   = channels above the planner band-high,
        both derived from the engine's state snapshot;
        candidates = the selected pair summaries from this cycle.
        """
        if self.data_service is None:
            return
        snapshot = getattr(cycle_result, "snapshot", None)
        if snapshot is None:
            return

        def _band(name: str, default: float) -> float:
            value = getattr(cfg, name, default)
            return float(value) if isinstance(value, (int, float)) and value else default

        band_low = _band("low_liquidity_threshold", 0.35)
        band_high = _band("high_liquidity_threshold", 0.65)

        depleted: List[Tuple[str, Dict[str, Any], float]] = []
        source: List[Tuple[str, Dict[str, Any], float]] = []
        for channel in getattr(snapshot, "channels", ()) or ():
            local_ratio = float(getattr(channel, "local_ratio", 0.0) or 0.0)
            info = {
                "peer_id": getattr(channel, "peer_id", ""),
                "capacity": int(getattr(channel, "capacity_sats", 0) or 0),
            }
            entry = (str(getattr(channel, "channel_id", "")), info, local_ratio)
            if local_ratio < band_low:
                depleted.append(entry)
            elif local_ratio > band_high:
                source.append(entry)

        candidates = list(getattr(cycle_result, "candidates", []) or [])
        self._report_hive_liquidity_state(depleted, source, candidates)

    def _is_hive_member(self, peer_id: str) -> bool:
        """Check hive membership from the live router first, then cached hints."""
        if not peer_id:
            return False
        if self.hive_router is not None:
            try:
                return bool(self.hive_router.is_hive_member(peer_id))
            except Exception:
                pass
        if self.hive_hints is not None:
            try:
                return bool(self.hive_hints.is_hive_member(peer_id))
            except Exception:
                pass
        return False

    def _get_hive_rebalance_bias(self, peer_id: str) -> float:
        """Return bounded multiplicative rebalance score bias from hive hints. 1.0 if unavailable."""
        if self.hive_hints is None:
            return 1.0
        try:
            bias = self.hive_hints.get_rebalance_bias(peer_id)
            return max(0.85, min(1.15, bias))
        except Exception:
            return 1.0

    def _fresh_hive_entries(self, getter_name: str) -> List[Dict[str, Any]]:
        """Return action-grade hive entries only when the snapshot is fresh."""
        if self.hive_hints is None:
            return []
        freshness = getattr(self.hive_hints, "is_fresh", None)
        if callable(freshness):
            try:
                if not freshness():
                    return []
            except Exception:
                return []
        fresh_getter = getattr(self.hive_hints, f"{getter_name}_fresh", None)
        getter = fresh_getter if callable(fresh_getter) else getattr(self.hive_hints, getter_name, None)
        if not callable(getter):
            return []
        try:
            return [entry for entry in (getter() or []) if isinstance(entry, dict)]
        except Exception:
            return []

    @staticmethod
    def _normalize_coordination_value(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text.replace(":", "x")

    @classmethod
    def _normalize_route_segment(cls, segment: Any) -> Optional[Tuple[str, str]]:
        if isinstance(segment, dict):
            source = cls._normalize_coordination_value(segment.get("source"))
            destination = cls._normalize_coordination_value(segment.get("destination"))
            if source and destination:
                return (source, destination)
            return None
        if isinstance(segment, str):
            text = segment.strip()
            if not text:
                return None
            separator = "->" if "->" in text else ">" if ">" in text else None
            if separator is None:
                return None
            source, destination = text.split(separator, 1)
            source = cls._normalize_coordination_value(source)
            destination = cls._normalize_coordination_value(destination)
            if source and destination:
                return (source, destination)
        return None

    def _candidate_route_segments(self, candidate: RebalanceCandidate) -> set[Tuple[str, str]]:
        segments: set[Tuple[str, str]] = set()
        sink_scid = self._normalize_coordination_value(candidate.to_channel)
        for source_scid in candidate.source_candidates:
            normalized_source = self._normalize_coordination_value(source_scid)
            if normalized_source and sink_scid:
                segments.add((normalized_source, sink_scid))

        sink_peer = self._normalize_coordination_value(candidate.to_peer_id)
        source_peers = list(candidate.source_candidate_peer_ids or [])
        if not source_peers and candidate.primary_source_peer_id:
            source_peers = [candidate.primary_source_peer_id]
        for source_peer in source_peers:
            normalized_source = self._normalize_coordination_value(source_peer)
            if normalized_source and sink_peer:
                segments.add((normalized_source, sink_peer))
        return segments

    def _coordination_entry_view(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        view: Dict[str, Any] = dict(entry or {})
        for nested_key in ("active_chunk_recommendation", "active_chunk_lease"):
            nested = entry.get(nested_key) if isinstance(entry, dict) else None
            if not isinstance(nested, dict):
                continue
            for key, value in nested.items():
                if key not in view or not view.get(key):
                    view[key] = value
        return view

    def _coordination_entry_segments(self, entry: Dict[str, Any]) -> set[Tuple[str, str]]:
        view = self._coordination_entry_view(entry)
        segments: set[Tuple[str, str]] = set()

        for raw_segment in view.get("route_segments", []) or []:
            normalized = self._normalize_route_segment(raw_segment)
            if normalized:
                segments.add(normalized)

        source_scid = self._normalize_coordination_value(view.get("source_scid"))
        sink_scid = self._normalize_coordination_value(view.get("sink_scid"))
        if source_scid and sink_scid:
            segments.add((source_scid, sink_scid))

        source_peer = self._normalize_coordination_value(
            view.get("source_peer_id") or view.get("from_peer_id")
        )
        dest_peer = self._normalize_coordination_value(
            view.get("destination_peer_id")
            or view.get("dest_peer_id")
            or view.get("to_peer_id")
            or view.get("target_peer_id")
        )
        if source_peer and dest_peer:
            segments.add((source_peer, dest_peer))

        return segments

    def _candidate_matches_coordination_entry(
        self,
        candidate: RebalanceCandidate,
        entry: Dict[str, Any],
    ) -> bool:
        view = self._coordination_entry_view(entry)
        for amount_key in ("amount_sats", "chunk_size_sats", "active_chunk_amount_sats"):
            if view.get(amount_key) is None:
                continue
            try:
                expected_amount = int(view.get(amount_key) or 0)
            except Exception:
                expected_amount = 0
            if expected_amount > 0 and int(candidate.amount_sats or 0) != expected_amount:
                return False
        source_scid = self._normalize_coordination_value(view.get("source_scid"))
        sink_scid = self._normalize_coordination_value(view.get("sink_scid"))
        candidate_sink = self._normalize_coordination_value(candidate.to_channel)
        candidate_sources = {
            self._normalize_coordination_value(source_scid)
            for source_scid in candidate.source_candidates
            if source_scid
        }
        if source_scid and sink_scid:
            return sink_scid == candidate_sink and source_scid in candidate_sources

        candidate_segments = self._candidate_route_segments(candidate)
        if not candidate_segments:
            return False
        return bool(candidate_segments & self._coordination_entry_segments(entry))


    def _coordination_priority_score(self, entry: Dict[str, Any]) -> float:
        view = self._coordination_entry_view(entry)
        try:
            return max(0.0, float(view.get("priority_score", 0.0) or 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _serialize_route_segments(segments: set[Tuple[str, str]]) -> List[str]:
        serialized = []
        for source, destination in sorted(segments):
            if source and destination:
                serialized.append(f"{source}>{destination}")
        return serialized

    def _is_coordinated_candidate(self, candidate: RebalanceCandidate) -> bool:
        if not candidate:
            return False
        if getattr(candidate, "reason_code", "") == RebalanceReasonCode.COORDINATED_REBALANCE.value:
            return True
        return bool(
            getattr(candidate, "coordination_hint_type", "") or
            getattr(candidate, "coordination_hint_id", "")
        )

    def _get_coordination_execution_context(
        self,
        candidate: RebalanceCandidate,
    ) -> Optional[Dict[str, Any]]:
        if not self._is_coordinated_candidate(candidate):
            return None

        route_segments = self._serialize_route_segments(
            self._candidate_route_segments(candidate)
        )
        our_node_id = self._get_our_node_id() or ""
        context: Dict[str, Any] = {
            "coordination_hint_type": str(getattr(candidate, "coordination_hint_type", "") or "").strip(),
            "coordination_hint_id": str(getattr(candidate, "coordination_hint_id", "") or "").strip(),
            "recommendation_id": "",
            "campaign_id": "",
            "route_segments": route_segments,
            "primary_executor_member_id": our_node_id,
            "fallback_executor_member_ids": [],
            "priority_score": float(getattr(candidate, "coordination_rank_bonus", 0.0) or 0.0),
            "source_scid": str(candidate.from_channel or ""),
            "sink_scid": str(candidate.to_channel or ""),
            "amount_sats": int(candidate.amount_sats or 0),
            "campaign_goal_type": "",
            "campaign_target_peer_or_corridor": "",
            "campaign_target_total_amount_sats": 0,
            "campaign_remaining_amount_sats": None,
            "campaign_chunk_size_sats": 0,
            "campaign_chunk_index": 1,
            "lease_id": None,
        }

        if context["coordination_hint_type"] == "campaign":
            context["campaign_id"] = context["coordination_hint_id"]
        elif context["coordination_hint_id"]:
            context["recommendation_id"] = context["coordination_hint_id"]

        entry: Dict[str, Any] = {}
        if self.hive_hints is not None:
            try:
                if context["coordination_hint_type"] == "campaign":
                    campaigns = self._fresh_hive_entries("get_rebalance_campaigns")
                    entry = next(
                        (
                            campaign for campaign in campaigns
                            if str(campaign.get("campaign_id") or "").strip() == context["campaign_id"]
                        ),
                        {},
                    )
                    if not entry:
                        entry = next(
                            (
                                campaign for campaign in campaigns
                                if self._candidate_matches_coordination_entry(candidate, campaign)
                            ),
                            {},
                        )
                else:
                    recommendations = self._fresh_hive_entries("get_rebalance_recommendations")
                    entry = next(
                        (
                            recommendation for recommendation in recommendations
                            if str(recommendation.get("recommendation_id") or "").strip()
                            == context["recommendation_id"]
                        ),
                        {},
                    )
                    if not entry:
                        entry = next(
                            (
                                recommendation for recommendation in recommendations
                                if self._candidate_matches_coordination_entry(candidate, recommendation)
                            ),
                            {},
                        )
            except Exception as e:
                self.plugin.log(
                    f"HIVE_COORDINATION: failed to refresh execution context: {e}",
                    level='debug',
                )
                entry = {}

        view = self._coordination_entry_view(entry)
        if view:
            recommendation_id = str(view.get("recommendation_id") or "").strip()
            campaign_id = str(view.get("campaign_id") or "").strip()
            if recommendation_id:
                context["recommendation_id"] = recommendation_id
            if campaign_id:
                context["campaign_id"] = campaign_id
            primary = str(view.get("primary_executor_member_id") or "").strip()
            if primary:
                context["primary_executor_member_id"] = primary
            fallbacks = view.get("fallback_executor_member_ids")
            if isinstance(fallbacks, list):
                context["fallback_executor_member_ids"] = [
                    str(member_id).strip()
                    for member_id in fallbacks
                    if str(member_id).strip()
                ]
            context["priority_score"] = self._coordination_priority_score(view)

            entry_segments = self._serialize_route_segments(
                self._coordination_entry_segments(view)
            )
            if entry_segments:
                context["route_segments"] = entry_segments

            context["source_scid"] = str(view.get("source_scid") or context["source_scid"] or "")
            context["sink_scid"] = str(view.get("sink_scid") or context["sink_scid"] or "")
            context["campaign_goal_type"] = str(view.get("goal_type") or "").strip()
            context["campaign_target_peer_or_corridor"] = str(
                view.get("target_peer_or_corridor") or ""
            ).strip()
            try:
                total_amount = int(view.get("target_total_amount_sats") or 0)
            except Exception:
                total_amount = 0
            context["campaign_target_total_amount_sats"] = total_amount
            remaining_amount = view.get("remaining_amount_sats")
            if remaining_amount is not None:
                try:
                    context["campaign_remaining_amount_sats"] = int(remaining_amount or 0)
                except Exception:
                    context["campaign_remaining_amount_sats"] = None
            try:
                context["campaign_chunk_size_sats"] = int(
                    view.get("chunk_size_sats")
                    or view.get("active_chunk_amount_sats")
                    or 0
                )
            except Exception:
                context["campaign_chunk_size_sats"] = 0
            try:
                context["campaign_chunk_index"] = max(1, int(view.get("chunk_index") or 1))
            except Exception:
                context["campaign_chunk_index"] = 1

        return context

    def _report_coordination_intent(
        self,
        candidate: RebalanceCandidate,
    ) -> Optional[Dict[str, Any]]:
        context = self._get_coordination_execution_context(candidate)
        if not context:
            return None

        payload = {
            "recommendation_id": context.get("recommendation_id", ""),
            "route_segments": list(context.get("route_segments") or []),
            "primary_executor_member_id": context.get("primary_executor_member_id", ""),
            "priority_score": float(context.get("priority_score", 0.0) or 0.0),
            "source_scid": context.get("source_scid"),
            "sink_scid": context.get("sink_scid"),
            "amount_sats": context.get("amount_sats"),
            "fallback_executor_member_ids": list(context.get("fallback_executor_member_ids") or []),
        }

        if context.get("campaign_id"):
            payload.update({
                "campaign_goal_type": context.get("campaign_goal_type", ""),
                "campaign_target_peer_or_corridor": context.get(
                    "campaign_target_peer_or_corridor", ""
                ),
                "campaign_target_total_amount_sats": int(
                    context.get("campaign_target_total_amount_sats", 0) or 0
                ),
                "campaign_chunk_size_sats": int(
                    context.get("campaign_chunk_size_sats", 0) or 0
                ),
            })

        try:
            response = self.plugin.rpc.call("hive-report-rebalance-intent", payload)
        except Exception as e:
            context["intent_status"] = "report_failed"
            self.plugin.log(
                f"HIVE_COORDINATION: intent report failed for {candidate.to_channel}: {e}",
                level='debug',
            )
            return context

        if isinstance(response, dict):
            status = str(response.get("status") or "").strip().lower()
            context["intent_status"] = status or "invalid_response"
            recommendation_id = str(response.get("recommendation_id") or "").strip()
            if recommendation_id:
                context["recommendation_id"] = recommendation_id
            response_segments = response.get("route_segments")
            if isinstance(response_segments, list) and response_segments:
                context["route_segments"] = [
                    str(segment).strip()
                    for segment in response_segments
                    if str(segment).strip()
                ]
            lease = response.get("lease")
            if isinstance(lease, dict):
                context["lease_id"] = lease.get("lease_id")
            campaign = response.get("campaign")
            if isinstance(campaign, dict):
                context["campaign_id"] = (
                    campaign.get("campaign_id") or context.get("campaign_id")
                )
                context["campaign_goal_type"] = str(
                    campaign.get("goal_type") or context.get("campaign_goal_type") or ""
                ).strip()
                context["campaign_target_peer_or_corridor"] = str(
                    campaign.get("target_peer_or_corridor")
                    or context.get("campaign_target_peer_or_corridor")
                    or ""
                ).strip()
                if campaign.get("target_total_amount_sats") is not None:
                    try:
                        context["campaign_target_total_amount_sats"] = int(
                            campaign.get("target_total_amount_sats") or 0
                        )
                    except Exception:
                        pass
                if campaign.get("remaining_amount_sats") is not None:
                    try:
                        context["campaign_remaining_amount_sats"] = int(
                            campaign.get("remaining_amount_sats") or 0
                        )
                    except Exception:
                        pass
                if campaign.get("chunk_size_sats") is not None:
                    try:
                        context["campaign_chunk_size_sats"] = int(
                            campaign.get("chunk_size_sats") or 0
                        )
                    except Exception:
                        pass
                if campaign.get("chunk_index") is not None:
                    try:
                        context["campaign_chunk_index"] = max(
                            1, int(campaign.get("chunk_index") or 1)
                        )
                    except Exception:
                        pass
        else:
            context["intent_status"] = "invalid_response"
        return context

    def _report_coordination_outcome(
        self,
        candidate: RebalanceCandidate,
        context: Optional[Dict[str, Any]],
        *,
        status: str,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if context is None:
            context = self._get_coordination_execution_context(candidate)
        if not context:
            return

        payload: Dict[str, Any] = {
            "status": status,
            "reason": reason,
            "lease_id": context.get("lease_id"),
            "campaign_id": context.get("campaign_id"),
            "recommendation_id": context.get("recommendation_id", ""),
            "amount_sats": context.get("amount_sats"),
            "details": details or {},
            "primary_executor_member_id": context.get("primary_executor_member_id", ""),
            "fallback_executor_member_ids": list(context.get("fallback_executor_member_ids") or []),
            "route_segments": list(context.get("route_segments") or []),
            "priority_score": float(context.get("priority_score", 0.0) or 0.0),
            "source_scid": context.get("source_scid"),
            "sink_scid": context.get("sink_scid"),
            "campaign_goal_type": context.get("campaign_goal_type", ""),
            "campaign_target_peer_or_corridor": context.get(
                "campaign_target_peer_or_corridor", ""
            ),
            "campaign_target_total_amount_sats": int(
                context.get("campaign_target_total_amount_sats", 0) or 0
            ),
            "campaign_remaining_amount_sats": context.get("campaign_remaining_amount_sats"),
            "campaign_chunk_size_sats": int(
                context.get("campaign_chunk_size_sats", 0) or 0
            ),
            "campaign_chunk_index": int(context.get("campaign_chunk_index", 1) or 1),
        }

        try:
            self.plugin.rpc.call("hive-report-rebalance-outcome", payload)
        except Exception as e:
            self.plugin.log(
                f"HIVE_COORDINATION: outcome report failed for {candidate.to_channel}: {e}",
                level='debug',
            )



    @staticmethod
    def _should_skip_futility(fail_count: int, last_error_type: str) -> bool:
        """
        Check if a channel should be skipped by the futility breaker.

        No-route and budget-exceeded failures trigger at 4 attempts
        (structural problems unlikely to resolve within a cycle).
        Other failures trigger at 10 (existing threshold).
        """
        if last_error_type in ("no_route", "budget_exceeded") and fail_count >= 4:
            return True
        if fail_count >= 10:
            return True
        return False

    @staticmethod
    def _classify_error(error_msg: str) -> str:
        """Classify a rebalance error message for failure-informed routing."""
        msg = error_msg.lower()
        if any(s in msg for s in ("no route", "no_route", "unknown_next_peer", "no path", "no channels")):
            return "no_route"
        if any(s in msg for s in ("timeout", "timed out", "deadline")):
            return "timeout"
        if any(s in msg for s in ("route_over_budget", "budget", "exceeded")):
            return "budget_exceeded"
        return "other"

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

    @staticmethod
    def _normalize_rebalance_success_signal(
        data: Optional[Dict[str, Any]],
        *,
        min_samples: int = 3,
    ) -> Optional[Dict[str, float]]:
        """Normalize rebalance success-rate history into a bounded weighted signal."""
        if not data:
            return None
        total = int(data.get("total", 0) or 0)
        if total < min_samples:
            return None
        rate = max(0.10, min(0.95, float(data.get("success_rate", 0.0) or 0.0)))
        confidence = max(0.0, min(1.0, total / 10.0))
        return {"rate": rate, "confidence": confidence, "total": total}


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

    def _get_our_node_id(self) -> Optional[str]:
        if self._our_node_id:
            return self._our_node_id
        try:
            node_id = self.data_service.get_node_id()
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
            current_height = self.data_service.get_block_height()

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

    def set_capex_engine(self, engine: 'CapexBudgetEngine'):
        """Inject the unified capex budget engine."""
        self._capex_engine = engine

    def build_state_v2(self, channels, capex_allocations):
        """Build the normalized v2 state snapshot from normalized inputs."""
        return build_state_snapshot_v2(channels, capex_allocations)

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
        # Early-suppression paths (no slots / capital controls) deliberately
        # do NOT report liquidity state: no engine snapshot exists on those
        # paths, and the consumer (cl-hive's record_member_liquidity_report)
        # overwrites depleted/saturated wholesale — partial updates are not
        # supported — so writing hardcoded-empty lists with a fresh timestamp
        # would clobber the last REAL state under sustained suppression.
        # Skipping lets the previous real payload stand until the
        # coordinator's TTL ages it out, which is honest. The NORMAL path
        # reports REAL state derived from the engine cycle snapshot via
        # _report_liquidity_state_from_cycle.

        # Initialize ephemeral fee cache for this run (cleared at end)
        with self._cache_lock:  # P2-005
            self._fee_cache = {}

        # Thread-safe config snapshot for this rebalance cycle
        cfg = self.config.snapshot()

        # Capex allocations are computed once per cycle by the v2 engine's
        # _build_snapshot (engine.find_candidates/run_cycle below). Calling
        # compute_allocations here as well doubled the full flow-analysis
        # pass (~100 DB writes) every cycle for a result that was discarded.

        # Issue #24: Clean up stale reservations before each rebalance cycle
        # This prevents budget leakage from crashed jobs
        timeout_seconds = cfg.reservation_timeout_hours * 3600
        cleaned = self.database.cleanup_stale_reservations(timeout_seconds)
        if cleaned > 0:
            self.plugin.log(f"Cleaned {cleaned} stale budget reservations before rebalance cycle")

        try:
            # Slot check (legacy async job monitoring removed; stubs always allow)
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
                    f"No rebalance slots available ({self.job_manager.active_job_count} jobs active)"
                )
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
                return candidates

            # V2 engine handles candidate selection AND execution.
            engine = getattr(self, "rebalance_engine_v2", None)
            if engine is None:
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="rebalance_engine_unavailable",
                    dominant_input="rebalance_engine",
                    safety_block=True,
                    budget_blocked=False,
                )
                self.plugin.log(
                    "Rebalance engine not initialized; suppressing",
                    level='warn',
                )
                return candidates

            if cfg.dry_run:
                dry_run_candidates = engine.find_candidates()
                self.plugin.log(
                    f"[DRY RUN] Rebalance engine identified "
                    f"{len(dry_run_candidates)} candidate(s); execution suppressed",
                    level='debug'
                )
                if dry_run_candidates:
                    self._set_last_decision_summary(
                        action="rebalance",
                        reason="dry_run",
                        dominant_input="rebalance_engine",
                        safety_block=False,
                        budget_blocked=False,
                    )
                else:
                    hold_reason = self._derive_hold_reason(engine)
                    self._set_last_decision_summary(
                        action="hold",
                        reason=hold_reason,
                        dominant_input="rebalance_engine",
                        safety_block=False,
                        budget_blocked=False,
                    )
                return []

            cycle_result = engine.run_cycle()

            # Report REAL liquidity state to cl-hive each normal cycle
            # (depleted/saturated from the engine snapshot, selected pairs as
            # enriched needs). Reporting must never break the cycle itself.
            try:
                self._report_liquidity_state_from_cycle(cycle_result, cfg)
            except Exception as e:
                self.plugin.log(
                    f"hive liquidity-state report failed: {e}", level='debug'
                )

            executed = len(cycle_result.executions)
            succeeded = sum(1 for e in cycle_result.executions if e.success)
            if executed > 0:
                self._set_last_decision_summary(
                    action="rebalance",
                    reason=f"{succeeded}/{executed} rebalances succeeded",
                    dominant_input="rebalance_engine",
                    safety_block=False,
                    budget_blocked=False,
                )
            elif cycle_result.candidates:
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="candidates found but all executions failed",
                    dominant_input="rebalance_engine",
                    safety_block=False,
                    budget_blocked=False,
                )
            else:
                # Phase 1 deferred completion: surface the most specific
                # blocker from the engine's last_cycle debug so operators
                # can act on the hold instead of seeing only the coarse
                # no_rebalance_candidates.
                hold_reason = self._derive_hold_reason(engine)
                self._set_last_decision_summary(
                    action="hold",
                    reason=hold_reason,
                    dominant_input="rebalance_engine",
                    safety_block=False,
                    budget_blocked=False,
                )
            return []

        finally:
            with self._cache_lock:  # P2-005
                self._fee_cache = {}

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
        # Priority 0: Hive fleet members — forward hops are free but the
        # return hop (dest_peer → us) still costs dest_peer's published fee.
        # Use the last-hop fee as the floor: fleet route cost ≈ return hop fee.
        if self.hive_hints and peer_id and self.hive_hints.is_hive_member(peer_id):
            return_hop_fee = self._get_last_hop_fee(peer_id, amount_msat) or 0
            self.plugin.log(
                f"INBOUND FEE EST [{peer_id[:12]}...]: Hive fleet member, "
                f"return hop fee {return_hop_fee} PPM",
                level='debug'
            )
            return return_hop_fee

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

        # Priority 4: getroutes with fleet layers (replaces last-hop + route + fallback)
        # All fleet intelligence (corridors, traffic, reputation, profitability)
        # is encoded in askrene layers — a single getroutes call captures it all.
        if self.hive_router and self.hive_router.available:
            route = self.hive_router.discover_route(peer_id, base_to_sats_floor(amount_msat))
            if route and route.fee_ppm >= 0:
                estimate = route.fee_ppm
                # Ensure estimate respects failure floor
                if failed_floor > 0 and estimate <= failed_floor:
                    estimate = failed_floor + 25
                self.plugin.log(
                    f"INBOUND FEE EST [{peer_id[:12]}...]: Fleet-aware getroutes "
                    f"{estimate} PPM ({route.hops} hops, fail_floor={failed_floor})",
                    level='debug'
                )
                return estimate

        # Priority 5: Last-hop fee + buffer (legacy fallback when no fleet layers)
        if last_hop is not None:
            estimate = last_hop + self.config.inbound_fee_estimate_ppm
            if failed_floor > 0 and estimate <= failed_floor:
                estimate = failed_floor + 25
            self.plugin.log(
                f"INBOUND FEE EST [{peer_id[:12]}...]: Last-hop fallback "
                f"{estimate} PPM (last_hop={last_hop}, fail_floor={failed_floor})",
                level='debug'
            )
            return estimate

        # Priority 6: Configured default
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
        # Always use actual peer channel fees, even for fleet members.
        # cl-hive intends 0 fee but gossip propagation is asynchronous —
        # the real fee may still be non-zero. Using the actual fee gives
        # accurate budget estimates and avoids WIRE_FEE_INSUFFICIENT.

        # Check cache first (memoization for this run)
        cache_key = (peer_id, int(amount_msat or 0))
        # P2-005: Read the memo hit and the peer-inbound-fee snapshot atomically,
        # then compute (possibly with an RPC) OUTSIDE the lock, then store the
        # memo atomically. The lock is never held across the PRIORITY-2 RPC.
        with self._cache_lock:
            if cache_key in self._fee_cache:
                return self._fee_cache[cache_key]
            peer_fee_info = self._peer_inbound_fees.get(peer_id)
            if peer_fee_info is not None:
                peer_fee_info = dict(peer_fee_info)  # detached copy

        result = None

        # PRIORITY 1: Use actual peer inbound fee from listpeerchannels.updates.remote
        # This is the most accurate source - directly from our channel state, not gossip
        if peer_fee_info is not None:
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
                    channels = self.data_service.get_channels(source=peer_id)
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
        with self._cache_lock:  # P2-005
            self._fee_cache[cache_key] = result

        return result


    def _get_peer_connection_status(self) -> Dict:
        status = {}
        try:
            for p in self.data_service.get_peers().get("peers", []):
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
                listfunds = self.data_service.get_funds()
                listpeerchannels = self.data_service.get_peer_channels()

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
                        "capacity": base_to_sats_floor(amount_msat),
                        "spendable_sats": base_to_sats_floor(our_amount_msat),
                        "peer_id": info.get("peer_id", channel.get("peer_id", "")),
                        "fee_ppm": info.get("fee_ppm", 0),
                        "base_fee_msat": info.get("base_fee_msat", 0),
                        "htlcs": info.get("htlcs", 0),
                        # Peer's actual inbound fee from updates.remote (None if unavailable)
                        "peer_inbound_fee_ppm": info.get("peer_inbound_fee_ppm"),
                        "peer_inbound_base_msat": info.get("peer_inbound_base_msat")
                    }

                # Populate peer_id -> peer inbound fee cache for _get_last_hop_fee()
                # This allows _estimate_inbound_fee() to use actual fees instead of gossip.
                # P2-005: Build the new map locally, then swap it in under the lock
                # so a concurrent _get_last_hop_fee never sees a half-populated map
                # (no RPC inside the lock — channels is already materialised).
                rebuilt: Dict[str, Dict[str, int]] = {}
                for scid, info in channels.items():
                    peer_id = info.get("peer_id")
                    if peer_id and info.get("peer_inbound_fee_ppm") is not None:
                        rebuilt[peer_id] = {
                            "fee_ppm": info["peer_inbound_fee_ppm"],
                            "base_msat": info.get("peer_inbound_base_msat", 0) or 0
                        }
                with self._cache_lock:
                    self._peer_inbound_fees = rebuilt

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

    def _record_successful_rebalance_fee(
        self,
        rebalance_id: int,
        *,
        status: str,
        channel_id: str,
        peer_id: str,
        amount_sats: int,
        fee_msat: int,
    ) -> int:
        """Persist successful rebalance fees in both history and cost ledgers."""
        persisted_fee_msat = max(0, int(fee_msat or 0))
        persisted_fee_sats = base_to_sats_ceil(persisted_fee_msat)
        self.database.update_rebalance_result(
            rebalance_id,
            status,
            actual_fee_sats=persisted_fee_sats,
            actual_fee_msat=persisted_fee_msat,
        )
        if persisted_fee_msat > 0:
            self.database.record_rebalance_cost(
                channel_id=channel_id,
                peer_id=peer_id,
                cost_sats=persisted_fee_sats,
                cost_msat=persisted_fee_msat,
                amount_sats=amount_sats,
                timestamp=int(time.time()),
            )
        return persisted_fee_sats

    def execute_rebalance(self, candidate: RebalanceCandidate, enforce_budget: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute a rebalance for the given candidate.

        Uses RebalanceEngineV2 for all live rebalances.
        Fleet intelligence still influences planning, but execution flows
        through the router-v3 plus native execution path for both fleet-planned and
        network-planned jobs.
        """
        result = {"success": False, "candidate": candidate.to_dict(), "message": ""}
        with self._pending_lock:
            self._pending[candidate.to_channel] = int(time.time())

        # Thread-safe config snapshot for this execution
        cfg = self.config.snapshot()

        rebalance_id: Optional[int] = None
        reserved_budget = False
        job_started = False
        coordination_context: Optional[Dict[str, Any]] = None
        coordination_started = False
        try:
            # Validation: Return error on empty/None channel IDs (HO-01)
            if not candidate.from_channel or not candidate.to_channel:
                if self._is_coordinated_candidate(candidate):
                    self._report_coordination_outcome(
                        candidate,
                        None,
                        status="declined",
                        reason="local_policy_block",
                        details={"error": "invalid_channel_ids"},
                    )
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
                    f"from {candidate.from_channel} to {candidate.to_channel}",
                    level='debug'
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

                effective_budget = cfg.daily_budget_sats
                # Only override with global provider if one is configured.
                if getattr(self, "global_budget_limit_provider", None) is not None:
                    effective_budget = self._get_global_budget_limit(cfg)

                ext_costs = self._get_external_liquidity_costs()
                ext_spent = int(ext_costs.get("spent_24h_sats", 0) or 0)
                ext_reserved = int(ext_costs.get("reserved_24h_sats", 0) or 0)
                rebalance_budget_limit = max(0, effective_budget - ext_spent - ext_reserved)
                # Capex candidates use per-channel budget as their limit.
                # Global daily/weekly caps only apply as emergency overrides when > 0.
                is_capex = getattr(candidate, 'reason_code', '') == RebalanceReasonCode.CAPEX_FALLBACK.value
                if is_capex:
                    capex_limit = candidate.max_budget_sats
                    if cfg.daily_budget_sats > 0:
                        rebalance_budget_limit = min(capex_limit, rebalance_budget_limit)
                    else:
                        rebalance_budget_limit = capex_limit
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
                    result["error"] = "local_budget_block"
                    self.plugin.log(
                        f"CAPITAL CONTROL: Budget reservation failed for {db_to_channel}. "
                        f"Remaining for rebalances: {remaining} sats "
                        f"(external costs: spent={ext_spent}, reserved={ext_reserved}, total_budget={effective_budget})",
                        level='warn'
                    )
                    if self._is_coordinated_candidate(candidate):
                        self._report_coordination_outcome(
                            candidate,
                            None,
                            status="declined",
                            reason="local_budget_block",
                            details={
                                "remaining_budget_sats": remaining,
                                "effective_budget_sats": effective_budget,
                            },
                        )
                    # Budget exhaustion is global; don't backoff a specific channel.
                    with self._pending_lock:
                        self._pending.pop(candidate.to_channel, None)
                    return result

            if self.rebalance_engine_v2:
                if self._is_coordinated_candidate(candidate):
                    coordination_context = self._report_coordination_intent(candidate)
                    intent_status = str(
                        (coordination_context or {}).get("intent_status") or ""
                    ).strip().lower()
                    if intent_status != "accepted" and intent_status not in {"report_failed", "invalid_response"}:
                        intent_failure_reason = "shared_conflict_changed"
                        self._report_coordination_outcome(
                            candidate,
                            coordination_context,
                            status="declined",
                            reason=intent_failure_reason,
                            details={"intent_status": intent_status},
                        )
                        res = {
                            "success": False,
                            "error": intent_failure_reason,
                            "message": f"Coordination intent rejected: {intent_status}",
                        }
                    else:
                        if intent_status == "accepted":
                            self._report_coordination_outcome(
                                candidate,
                                coordination_context,
                                status="started",
                            )
                            coordination_started = True
                        try:
                            exec_result = self._execute_candidate_v2(
                                candidate, rebalance_id=rebalance_id
                            )
                            if exec_result.success:
                                actual_fee_sats = self._record_successful_rebalance_fee(
                                    rebalance_id,
                                    status='success',
                                    channel_id=candidate.to_channel,
                                    peer_id=candidate.to_peer_id,
                                    amount_sats=candidate.amount_sats,
                                    fee_msat=exec_result.fee_msat,
                                )
                                res = {
                                    "success": True,
                                    "actual_fee_sats": actual_fee_sats,
                                    "message": (
                                        f"Rebalance completed via {exec_result.route_type} engine "
                                        f"({exec_result.fee_ppm}ppm, {exec_result.hops} hops, "
                                        f"{exec_result.parts} parts, {exec_result.attempts} attempts)"
                                    ),
                                }
                                self.database.reset_failure_count(candidate.to_channel)
                                self._report_coordination_outcome(
                                    candidate,
                                    coordination_context,
                                    status="succeeded",
                                    details={
                                        "intent_status": intent_status,
                                        "route_type": exec_result.route_type,
                                        "attempts": exec_result.attempts,
                                        "parts": exec_result.parts,
                                        "fee_ppm": exec_result.fee_ppm,
                                    },
                                )
                            else:
                                error_str = exec_result.error or "no_routes"
                                stable_error = stable_failure_reason(error_str)
                                res = {
                                    "success": False,
                                    "error": stable_error,
                                    "message": (
                                        f"RebalanceEngineV2: {error_str} "
                                        f"({exec_result.attempts} attempts, "
                                        f"type={exec_result.route_type})"
                                    ),
                                    "payment_pending": bool(
                                        getattr(exec_result, "payment_pending", False)
                                    ),
                                }
                                # The engine shares our history row; when the
                                # payment is parked as 'pending_settlement' for
                                # the reconciliation sweep, do not clobber it
                                # with 'failed'.
                                if rebalance_id and not res["payment_pending"]:
                                    self.database.update_rebalance_result(
                                        rebalance_id, 'failed',
                                        error_message=error_str,
                                    )
                                error_type = self._classify_error(error_str)
                                self.database.increment_failure_count(
                                    candidate.to_channel,
                                    attempted_ppm=candidate.max_fee_ppm,
                                    attempted_amount=candidate.amount_sats,
                                    error_type=error_type,
                                )
                                self._report_coordination_outcome(
                                    candidate,
                                    coordination_context,
                                    status="failed",
                                    reason=stable_error,
                                    details={
                                        "intent_status": intent_status,
                                        "executor_error": error_str,
                                        "route_type": exec_result.route_type,
                                        "attempts": exec_result.attempts,
                                    },
                                )
                        except Exception as e:
                            stable_error = "local_execution_failed"
                            res = {"success": False, "error": stable_error}
                            if rebalance_id:
                                self.database.update_rebalance_result(
                                    rebalance_id, 'failed', error_message=str(e),
                                )
                            error_type = self._classify_error(str(e))
                            try:
                                self.database.increment_failure_count(
                                    candidate.to_channel,
                                    attempted_ppm=candidate.max_fee_ppm,
                                    attempted_amount=candidate.amount_sats,
                                    error_type=error_type,
                                )
                            except Exception:
                                pass
                            self._report_coordination_outcome(
                                candidate,
                                coordination_context,
                                status="failed",
                                reason=stable_error,
                                details={"exception": str(e)},
                            )
                else:
                    try:
                        exec_result = self._execute_candidate_v2(
                            candidate, rebalance_id=rebalance_id
                        )
                        if exec_result.success:
                            actual_fee_sats = self._record_successful_rebalance_fee(
                                rebalance_id,
                                status='success',
                                channel_id=candidate.to_channel,
                                peer_id=candidate.to_peer_id,
                                amount_sats=candidate.amount_sats,
                                fee_msat=exec_result.fee_msat,
                            )
                            res = {
                                "success": True,
                                "actual_fee_sats": actual_fee_sats,
                                "message": (
                                    f"Rebalance completed via {exec_result.route_type} engine "
                                    f"({exec_result.fee_ppm}ppm, {exec_result.hops} hops, "
                                    f"{exec_result.parts} parts, {exec_result.attempts} attempts)"
                                ),
                            }
                            # Success resets failure count so channel re-enters rotation
                            self.database.reset_failure_count(candidate.to_channel)
                        else:
                            error_str = exec_result.error or "no_routes"
                            res = {
                                "success": False,
                                "error": error_str,
                                "message": (
                                    f"RebalanceEngineV2: {error_str} "
                                    f"({exec_result.attempts} attempts, "
                                    f"type={exec_result.route_type})"
                                ),
                                "payment_pending": bool(
                                    getattr(exec_result, "payment_pending", False)
                                ),
                            }
                            # Engine shares our history row; keep its
                            # 'pending_settlement' status intact for the
                            # reconciliation sweep.
                            if rebalance_id and not res["payment_pending"]:
                                self.database.update_rebalance_result(
                                    rebalance_id, 'failed',
                                    error_message=error_str,
                                )
                            # Record failure for futility breaker
                            error_type = self._classify_error(error_str)
                            self.database.increment_failure_count(
                                candidate.to_channel,
                                attempted_ppm=candidate.max_fee_ppm,
                                attempted_amount=candidate.amount_sats,
                                error_type=error_type,
                            )
                    except Exception as e:
                        res = {"success": False, "error": str(e)}
                        if rebalance_id:
                            self.database.update_rebalance_result(
                                rebalance_id, 'failed', error_message=str(e),
                            )
                        # Record failure for futility breaker
                        error_type = self._classify_error(str(e))
                        try:
                            self.database.increment_failure_count(
                                candidate.to_channel,
                                attempted_ppm=candidate.max_fee_ppm,
                                attempted_amount=candidate.amount_sats,
                                error_type=error_type,
                            )
                        except Exception:
                            pass
            else:
                stable_error = (
                    "local_policy_block"
                    if self._is_coordinated_candidate(candidate)
                    else "no_rebalance_engine"
                )
                res = {"success": False, "error": stable_error}
                if self._is_coordinated_candidate(candidate):
                    self._report_coordination_outcome(
                        candidate,
                        None,
                        status="declined",
                        reason=stable_error,
                        details={"error": "no_rebalance_engine"},
                    )

            if res.get("success"):
                self._set_last_decision_summary(
                    action="rebalance",
                    reason="rebalance_completed",
                    dominant_input=candidate.reason_code,
                    safety_block=False,
                    budget_blocked=False,
                )
                job_started = True
                if reserved_budget and rebalance_id is not None:
                    try:
                        self.database.mark_budget_spent(
                            str(rebalance_id),
                            res.get("actual_fee_sats", 0) or 0,
                        )
                    except Exception:
                        pass
                with self._pending_lock:
                    self._pending.pop(candidate.to_channel, None)
                result.update({
                    "success": True,
                    "message": res.get("message", "Rebalance completed"),
                    "rebalance_id": rebalance_id
                })
                self.plugin.log(
                    f"Rebalance completed: {candidate.to_channel} — "
                    f"{res.get('message', '')}"
                )
            else:
                error = res.get("error", "Rebalance failed")
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="start_job_failed",
                    dominant_input=candidate.reason_code,
                    safety_block=False,
                    budget_blocked=False,
                )
                if rebalance_id and not res.get("payment_pending"):
                    self.database.update_rebalance_result(
                        rebalance_id, 'failed', error_message=error
                    )
                result["error"] = error
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
            if self._is_coordinated_candidate(candidate):
                reason = "local_execution_failed" if coordination_started else "local_policy_block"
                self._report_coordination_outcome(
                    candidate,
                    coordination_context,
                    status="failed" if coordination_started else "declined",
                    reason=reason,
                    details={"exception": str(e)},
                )
                result["error"] = reason
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
                return {
                    "success": False,
                    "shock_status": "failed",
                    "message": "Channel not found locally",
                }

            dest_info = channels[channel_id]
            
            # Find best source (highest spendable sats, excluding target)
            valid_sources = [
                (cid, info) for cid, info in channels.items() 
                if cid != channel_id and info.get('spendable_sats', 0) > 100_000
            ]
            
            if not valid_sources:
                return {
                    "success": True,
                    "shock_status": "failed",
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

            # Operator ruling D4 (2026-07-01): the shock fee envelope is the
            # configured diagnostic_rebalance_max_fee_sats (default 400),
            # clamped to [1, min(daily_budget_sats, 10_000)] so a typo can't
            # authorize huge diagnostic spend. The ppm ceiling is DERIVED from
            # the sat cap (ceil(cap/amount*1e6)) so the sat cap is the single
            # binding knob — under the old hardcoded pair (100 sats, 2000 ppm)
            # both bounds bound at exactly 100 sats on the 50k shock and every
            # observed market route (118-363 sats) was rejected
            # route_over_budget, so the diagnostic could never fire.
            max_fee_sats = max(1, min(
                int(getattr(self.config, 'diagnostic_rebalance_max_fee_sats', 400)),
                int(getattr(self.config, 'daily_budget_sats', 0) or 0),
                DIAGNOSTIC_FEE_CAP_CEILING_SATS,
            ))
            max_fee_ppm = math.ceil(max_fee_sats / shock_amount * 1_000_000)


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
                amount_msat=sats_to_base(shock_amount),
                outbound_fee_ppm=0,
                inbound_fee_ppm=inbound_fee,
                source_fee_ppm=best_source_info.get('fee_ppm', 0),
                weighted_opp_cost_ppm=0,
                spread_ppm=0,  # Likely negative, we don't care for diagnostic
                max_budget_sats=max_fee_sats,  # Configured diagnostic fee cap (D4)
                max_budget_msat=sats_to_base(max_fee_sats),
                max_fee_ppm=max_fee_ppm,  # Derived from the sat cap (D4)
                expected_profit_sats=-50,  # Expect a small loss (diagnostic cost)
                liquidity_ratio=0.5,
                dest_flow_state="diagnostic",
                dest_turnover_rate=0.0,
                source_turnover_rate=0.0
            )
            
            # Capital Controls Check - diagnostic rebalances count against daily budget
            if not self._check_capital_controls():
                self.plugin.log("Defibrillator Active Shock blocked by capital controls", level='warn')
                # Audit (defibrillation honesty): a blocked shock delivered
                # no liquidity — report it as blocked, never completed.
                return {
                    "success": True,
                    "shock_status": "blocked",
                    "message": "Zero-Fee flag set, but Active Shock blocked: daily budget exhausted or reserve too low"
                }

            # Record in database (direct diagnostic execution bypasses normal job flow)
            rebalance_id = self.database.record_rebalance(
                from_channel=best_source_id,
                to_channel=channel_id,
                amount_sats=shock_amount,
                max_fee_sats=max_fee_sats,
                expected_profit_sats=-50,
                rebalance_type='diagnostic',
                reason_code='defibrillator'
            )

            actual_fee_sats = None
            if self.rebalance_engine_v2:
                exec_result = self._execute_candidate_v2(
                    candidate, rebalance_id=rebalance_id
                )
                if exec_result.success:
                    actual_fee_sats = self._record_successful_rebalance_fee(
                        rebalance_id,
                        status='success',
                        channel_id=channel_id,
                        peer_id=dest_info.get('peer_id', ''),
                        amount_sats=shock_amount,
                        fee_msat=exec_result.fee_msat,
                    )
                elif not getattr(exec_result, "payment_pending", False):
                    # Engine shares this history row; leave its
                    # 'pending_settlement' status for the reconcile sweep.
                    self.database.update_rebalance_result(
                        rebalance_id, 'failed',
                        error_message=exec_result.error or "rebalance failed"
                    )
                shock_ok = exec_result.success
                shock_pending = bool(getattr(exec_result, "payment_pending", False)) and not shock_ok
            else:
                self.database.update_rebalance_result(rebalance_id, 'failed', error_message="no rebalance engine available")
                shock_ok = False
                shock_pending = False

            # Audit (defibrillation honesty): report the ACTUAL shock
            # outcome. A failed or still-pending shock delivered no
            # confirmed liquidity and must not read as completed.
            if shock_ok:
                shock_status = "completed"
            elif shock_pending:
                shock_status = "pending"
            else:
                shock_status = "failed"
            result = {
                "success": True,
                "shock_status": shock_status,
                "message": f"Defibrillator active: Zero-Fee flag set + Shock {shock_status}"
            }
            if actual_fee_sats is not None:
                result["actual_fee_sats"] = actual_fee_sats
            return result

        except Exception as e:
            self.plugin.log(f"Defibrillator shock failed: {e}", level='error')
            return {
                "success": False,
                "shock_status": "failed",
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
            amount_msat=sats_to_base(amount_sats),
            outbound_fee_ppm=fee_ppm,
            inbound_fee_ppm=est_in,
            source_fee_ppm=src_ppm,
            weighted_opp_cost_ppm=0,
            spread_ppm=fee_ppm - est_in - src_ppm,
            max_budget_sats=max_fee_sats,
            max_budget_msat=sats_to_base(max_fee_sats),
            max_fee_ppm=max_fee_ppm,
            expected_profit_sats=0,
            liquidity_ratio=0.5,
            dest_flow_state="manual",
            dest_turnover_rate=0.0,
            source_turnover_rate=0.0
        )
        # Manual rebalances bypass budget reservations.
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

        if not self.rebalance_engine_v2:
            self.database.update_rebalance_result(rebalance_id, 'failed', error_message="no rebalance engine available")
            return {"success": False, "error": "no rebalance engine available"}

        exec_result = self._execute_candidate_v2(cand, rebalance_id=rebalance_id)

        if exec_result.success:
            fee_sats = self._record_successful_rebalance_fee(
                rebalance_id,
                status='success',
                channel_id=to_channel,
                peer_id=t_info.get("peer_id", ""),
                amount_sats=amount_sats,
                fee_msat=exec_result.fee_msat,
            )
            result = {"success": True, "message": "completed", "actual_fee_sats": fee_sats}
        else:
            if not getattr(exec_result, "payment_pending", False):
                # Engine shares this history row; keep 'pending_settlement'
                # intact for the reconciliation sweep.
                self.database.update_rebalance_result(
                    rebalance_id, 'failed',
                    error_message=exec_result.error or ""
                )
            result = {"success": False, "error": exec_result.error or "rebalance failed"}

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
                listfunds = self.data_service.get_funds()
                onchain_sats = 0
                for output in listfunds.get("outputs", []):
                    if output.get("status") == "confirmed":
                        amount_msat = output.get("amount_msat", 0)
                        amount_msat = self._parse_msat(amount_msat)
                        onchain_sats += base_to_sats_floor(amount_msat)

                channel_spendable_sats = 0
                for channel in listfunds.get("channels", []):
                    if channel.get("state") != "CHANNELD_NORMAL":
                        continue
                    our_amount_msat = self._parse_msat(channel.get("our_amount_msat", 0))
                    spendable = base_to_sats_floor(our_amount_msat)
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
            now = int(time.time())
            budget_window_hours = max(1, int(getattr(cfg, "total_cost_budget_window_hours", 24) or 24))

            effective_budget = cfg.daily_budget_sats
            # Only override with global provider if one is configured.
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

            weekly_fees_spent = self.database.get_total_rebalance_fees(now - 7 * 86400)
            # Use daily external spend as-is for the weekly check rather than
            # multiplying by 7 (which grossly overestimates after a single
            # large Boltz swap).  The weekly rebalance fees already cover 7
            # days; adding one day of external costs is conservative enough.
            weekly_total_spent = weekly_fees_spent + ext_spent
            if weekly_total_spent >= effective_weekly:
                self.plugin.log(
                    f"CAPITAL CONTROL: Weekly budget exceeded "
                    f"(rebalance_fees_7d={weekly_fees_spent} + external_spent_24h={ext_spent} "
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
