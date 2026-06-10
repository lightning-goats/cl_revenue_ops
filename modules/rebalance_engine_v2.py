"""Rebalance engine v2 orchestrator.

Wires: state snapshot → planner → router → executor → audit.
Single entry point: find_candidates() for dry-run, run_cycle() for live.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .rebalance_audit_v2 import RebalanceAudit
from .rebalance_coordination_overlay import (
    build_coordination_overlay,
    merge_coordination_pairs,
    pair_segment_bias_multiplier,
)
from .rebalance_execution import ExecutionResult
from .rebalance_native_executor_v2 import NativeRouteExecutor
from .rebalance_hive_router import RebalanceHiveRouter
from .rebalance_planner_v2 import RebalancePlanner
from .rebalance_route_policy import (
    RouteDecision,
    RoutePolicy,
    RoutePriority,
    decide_route_policy,
)
from .rebalance_router_v2 import RouteResult
from .rebalance_router_v3 import RebalanceRouterV3, _configured_layer_names
from .rebalance_state_v2 import StateSnapshot, build_state_snapshot
from .segment_observations import SegmentObservationStore
from .rebalance_types_v2 import PairCandidate, PlanResult, SkipRecord
from .utils import base_to_sats_ceil, parse_msat


@dataclass
class CycleResult:
    """Full result of a v2 rebalance cycle."""

    considered_candidates: List[PairCandidate] = field(default_factory=list)
    candidates: List[PairCandidate] = field(default_factory=list)
    executions: List[ExecutionResult] = field(default_factory=list)
    audit_records: List[SkipRecord] = field(default_factory=list)
    snapshot: Optional[StateSnapshot] = None
    plan: Optional[PlanResult] = None


class RebalanceEngine:
    """V2 rebalance engine — unified, actual-fee-based."""

    def __init__(
        self,
        plugin: Any,
        config: Any,
        database: Any,
        capex_engine: Any = None,
        profitability: Any = None,
        hive_hints: Any = None,
        data_service: Any = None,
        hive_router: Any = None,
        segment_observation_store: Any = None,
        global_budget_limit_provider: Any = None,
        external_liquidity_cost_provider: Any = None,
    ):
        self.plugin = plugin
        self.config = config
        self.database = database
        self._capex_engine = capex_engine
        self._profitability = profitability
        self._hive_hints = hive_hints
        self._data_service = data_service
        self._membership_router = (
            hive_router if hasattr(hive_router, "is_hive_member") else None
        )
        self._hive_router = (
            hive_router if hasattr(hive_router, "price_pair") else None
        )
        self._segment_observation_store = segment_observation_store
        self.global_budget_limit_provider = global_budget_limit_provider
        self.external_liquidity_cost_provider = external_liquidity_cost_provider

        self._our_id: Optional[str] = None
        self._audit = RebalanceAudit(plugin)
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_concurrent_jobs(),
            thread_name_prefix="rebal-v2",
        )
        # Pair-level futility tracker: {(src_scid, dst_scid): [failure_ts, ...]}.
        # A pair that fails _futility_threshold times within _futility_window_sec
        # is skipped from subsequent cycles until the stale entries decay out
        # of the window. Prevents re-picking the same unroutable pair every
        # cycle (the old JobManager futility breaker was stubbed during the
        # cleanup refactor; this replaces it at the engine level so it covers
        # both v2 and v3 routers).
        self._pair_failures: Dict[Tuple[str, str], List[float]] = {}
        self._futility_threshold: int = 3
        self._futility_window_sec: float = 1800.0  # 30 minutes
        # Iter2: lower base cooldown for the most common transient failure
        # (no_route / temporary_channel_failure). Backoff multiplier in
        # record_pair_rebalance_failure already escalates repeated failures
        # (x failure_count up to x6).
        self._pair_failure_cooldowns: Dict[str, int] = {
            "temporary_channel_failure": 300,    # 5 min on first failure
            "fee_insufficient": 1800,            # 30 min (rare; usually persistent)
            "incorrect_cltv_expiry": 3600,       # 1 hour (CLN config alignment)
            "permanent_failure": 21600,          # 6 hours
            "payment_pending_timeout": 3600,     # 1 hour: HTLC may pend until CLTV expiry
            "local_execution_failed": 600,       # 10 min
            "other_retriable": 600,              # 10 min
        }

        our_id = self._get_our_id() or ""

        # Build v3 router iff askrene is available. Missing askrene now means
        # fail-closed: legacy getroute routing has been removed.
        self.router_v3: Optional[RebalanceRouterV3] = None
        if self._probe_askrene():
            raw_layer_config = getattr(config, "askrene_layers", None)
            layer_names = _configured_layer_names(raw_layer_config)
            if raw_layer_config is None or not str(raw_layer_config).strip():
                self._log(
                    "[router-v3] empty askrene_layers config; using default "
                    f"layers={layer_names}",
                    "warn",
                )
            self.router_v3 = RebalanceRouterV3(
                plugin=plugin,
                our_node_id=our_id,
                layer_names=layer_names,
                log=self._log,
                data_service=self._data_service,
            )
            # Clean any orphan exclude layers from a previous crashed cycle.
            self._sweep_orphan_exclude_layers()
            if self._hive_router is None and self._hive_hints is not None:
                self._hive_router = RebalanceHiveRouter(
                    plugin=plugin,
                    our_node_id=our_id,
                    hive_hints=self._hive_hints,
                    data_service=self._data_service,
                    log=self._log,
                )

        self._cycle_router: Optional[Any] = self._active_router()
        self._last_cycle_result: CycleResult = CycleResult()
        # Single-flight execution guard. run_cycle() (background loop) and
        # execute_candidate() (manual RPCs) mutate shared cycle state
        # (_cycle_router, _last_cycle_result, _pair_failures, hive_router
        # begin_cycle/end_cycle) and can otherwise pay for the same
        # source->dest pair twice when they overlap. Contenders never block:
        # run_cycle() skips with a 'cycle_already_running' audit marker and
        # execute_candidate() fails fast with error='engine_busy' (same
        # non-blocking pattern as the fee controller's state lock).
        self._cycle_lock = threading.Lock()

    def _probe_askrene(self) -> bool:
        """One-shot probe: does this CLN instance have askrene loaded?"""
        try:
            if self._data_service is not None:
                self._data_service.get_askrene_layers()
            else:
                self.plugin.rpc.call("askrene-listlayers", {})
            return True
        except Exception:
            return False

    def _active_router(self) -> Optional[Any]:
        """Return the active router currently configured for dispatch."""
        return self.router_v3

    def _max_concurrent_jobs(self, cfg: Optional[Any] = None) -> int:
        """Return the configured automatic rebalance execution cap."""
        if cfg is None:
            cfg = self.config if not hasattr(self.config, "snapshot") else self.config.snapshot()
        value = getattr(cfg, "max_concurrent_jobs", 5)
        if not isinstance(value, (int, float, str)):
            value = 5
        try:
            max_jobs = int(float(value))
        except (TypeError, ValueError):
            max_jobs = 5
        return max(1, min(20, max_jobs))

    def _sweep_orphan_exclude_layers(self) -> int:
        """Remove leftover rebalance-exclude-* layers from previous crashed cycles.

        Called once at init after the askrene probe succeeds. Iterates every
        live layer and removes any whose name matches the prefix. Returns
        the number removed, for logging only.
        """
        try:
            if self._data_service is not None:
                result = self._data_service.get_askrene_layers()
            else:
                result = self.plugin.rpc.call("askrene-listlayers", {})
        except Exception as e:
            self._log(f"orphan sweep failed to list layers: {e}", level="warn")
            return 0

        orphans = [
            l.get("layer", "")
            for l in result.get("layers", [])
            if l.get("layer", "").startswith("rebalance-exclude-")
        ]
        for name in orphans:
            try:
                if self._data_service is not None:
                    self._data_service.askrene_remove_layer(name)
                else:
                    self.plugin.rpc.call("askrene-remove-layer", {"layer": name})
            except Exception as e:
                self._log(
                    f"failed to remove orphan layer {name}: {e}",
                    level="warn",
                )
        if orphans:
            self._log(
                f"swept {len(orphans)} orphan rebalance-exclude-* layer(s)",
                level="info",
            )
        return len(orphans)

    def shutdown(self) -> None:
        """Release thread pool resources."""
        self._pool.shutdown(wait=False)

    def _cache_cycle_result(self, result: CycleResult) -> None:
        self._last_cycle_result = result

    @staticmethod
    def _pair_key(pair: PairCandidate) -> Tuple[str, str]:
        return (str(pair.source_channel_id), str(pair.dest_channel_id))

    def _failure_penalty(self, pair: PairCandidate) -> float:
        fresh = self._prune_pair_failures(self._pair_key(pair))
        return round(min(1.0, len(fresh) * 0.25), 6)

    def _build_score_decomposition(
        self,
        pair: PairCandidate,
        *,
        probability_ppm: int = 0,
        route_cost_sats: Optional[int] = None,
        effective_budget_sats: Optional[int] = None,
        rejection_reason: str = "",
        route_status: str = "unpriced",
    ) -> Dict[str, Any]:
        """Build an explicit, operator-visible score breakdown.

        This is the first refactor step from the brief: expose the current
        decision model clearly before replacing it with a stronger empirical EV
        model. The values are normalized planner/debug units, not a final
        sat-denominated economics model.
        """
        cfg = self.config if not hasattr(self.config, "snapshot") else self.config.snapshot()
        high_threshold = float(getattr(cfg, "high_liquidity_threshold", 0.65) or 0.65)
        low_threshold = float(getattr(cfg, "low_liquidity_threshold", 0.35) or 0.35)

        source_capacity = int(getattr(pair, "source_capacity_sats", 0) or 0)
        dest_capacity = int(getattr(pair, "dest_capacity_sats", 0) or 0)
        amount_sats = int(getattr(pair, "amount_sats", 0) or 0)
        source_post_ratio = float(getattr(pair, "source_local_ratio", 0.0) or 0.0)
        dest_post_ratio = float(getattr(pair, "dest_local_ratio", 0.0) or 0.0)
        if source_capacity > 0:
            source_post_ratio -= amount_sats / max(1, source_capacity)
        if dest_capacity > 0:
            dest_post_ratio += amount_sats / max(1, dest_capacity)

        source_excess_before = max(0.0, float(getattr(pair, "source_local_ratio", 0.0) or 0.0) - high_threshold)
        source_excess_after = max(0.0, source_post_ratio - high_threshold)
        source_opportunity_cost = 0.0
        if source_excess_before > 0.0:
            source_opportunity_cost = min(
                0.25,
                max(0.0, source_excess_before - source_excess_after) / source_excess_before * 0.25,
            )

        if probability_ppm > 0:
            p_success = min(0.99, max(0.05, probability_ppm / 1_000_000.0))
        elif rejection_reason == "no_route":
            p_success = 0.05
        else:
            p_success = 0.5

        expected_fee_sats = int(route_cost_sats or 0)
        raw_budget_sats = int(getattr(pair, "pair_budget_sats", 0) or 0)
        effective_budget = int(
            effective_budget_sats
            if effective_budget_sats is not None
            else (raw_budget_sats or 0)
        )
        expected_fee = 0.0
        capital_risk_penalty = 0.0
        if effective_budget > 0 and expected_fee_sats > 0:
            expected_fee = min(1.0, expected_fee_sats / effective_budget)
        if raw_budget_sats > 0 and expected_fee_sats > 0:
            capital_risk_penalty = min(1.0, expected_fee_sats / raw_budget_sats)

        failure_penalty = self._failure_penalty(pair)
        expected_future_value = round(float(getattr(pair, "score", 0.0) or 0.0), 6)
        final_score = round(
            p_success * expected_future_value
            - expected_fee
            - source_opportunity_cost
            - failure_penalty
            - capital_risk_penalty,
            6,
        )
        beats_do_nothing = bool(not rejection_reason and final_score > 0.0)

        return {
            "model_version": "v2-bootstrap-explainability",
            "score_units": "planner_score_minus_budget_share",
            "stage": route_status,
            "p_success": round(p_success, 6),
            "expected_future_value": expected_future_value,
            "expected_fee": round(expected_fee, 6),
            "source_opportunity_cost": round(source_opportunity_cost, 6),
            "failure_penalty": round(failure_penalty, 6),
            "capital_risk_penalty": round(capital_risk_penalty, 6),
            "do_nothing_score": 0.0,
            "final_score": final_score,
            "beats_do_nothing": beats_do_nothing,
            "rejection_reason": rejection_reason,
            "inputs": {
                "reason_code": str(getattr(pair, "reason_code", "") or ""),
                "route_policy": getattr(
                    getattr(getattr(pair, "route_decision", None), "policy", None),
                    "value",
                    None,
                ),
                "probability_ppm": int(probability_ppm or 0),
                "expected_fee_sats": expected_fee_sats,
                "pair_budget_sats": raw_budget_sats,
                "effective_budget_sats": effective_budget,
                "source_local_ratio": round(float(getattr(pair, "source_local_ratio", 0.0) or 0.0), 6),
                "dest_local_ratio": round(float(getattr(pair, "dest_local_ratio", 0.0) or 0.0), 6),
                "source_post_ratio": round(source_post_ratio, 6),
                "dest_post_ratio": round(dest_post_ratio, 6),
                "target_band_low": round(low_threshold, 6),
                "target_band_high": round(high_threshold, 6),
                "source_value_class": str(getattr(pair, "source_value_class", "") or ""),
                "dest_value_class": str(getattr(pair, "dest_value_class", "") or ""),
                "source_budget_source": str(getattr(pair, "source_budget_source", "") or ""),
                "dest_budget_source": str(getattr(pair, "dest_budget_source", "") or ""),
                "hive_source_rebalance_bias": round(
                    float(getattr(pair, "hive_source_rebalance_bias", 1.0) or 1.0),
                    6,
                ),
                "hive_dest_rebalance_bias": round(
                    float(getattr(pair, "hive_dest_rebalance_bias", 1.0) or 1.0),
                    6,
                ),
                "hive_hint_score_multiplier": round(
                    float(getattr(pair, "hive_hint_score_multiplier", 1.0) or 1.0),
                    6,
                ),
                "metabolic_rebalance_bias": round(
                    float(getattr(pair, "metabolic_rebalance_bias", 1.0) or 1.0),
                    6,
                ),
                "immune_rebalance_bias": round(
                    float(getattr(pair, "immune_rebalance_bias", 1.0) or 1.0),
                    6,
                ),
            },
        }

    def _update_pair_score_decomposition(
        self,
        pair: PairCandidate,
        *,
        probability_ppm: int = 0,
        route_cost_sats: Optional[int] = None,
        effective_budget_sats: Optional[int] = None,
        rejection_reason: str = "",
        route_status: str = "unpriced",
    ) -> None:
        pair.rejection_reason = rejection_reason
        pair.score_decomposition = self._build_score_decomposition(
            pair,
            probability_ppm=probability_ppm,
            route_cost_sats=route_cost_sats,
            effective_budget_sats=effective_budget_sats,
            rejection_reason=rejection_reason,
            route_status=route_status,
        )

    def _serialize_pair_candidate(self, pair: PairCandidate) -> Dict[str, Any]:
        return {
            "source_channel_id": pair.source_channel_id,
            "dest_channel_id": pair.dest_channel_id,
            "source_peer_id": pair.source_peer_id,
            "dest_peer_id": pair.dest_peer_id,
            "amount_sats": int(pair.amount_sats or 0),
            "pair_budget_sats": int(pair.pair_budget_sats or 0),
            "route_cost_sats": (
                int(pair.route_cost_sats) if pair.route_cost_sats is not None else None
            ),
            "score": round(float(pair.score or 0.0), 6),
            "source_local_ratio": round(float(pair.source_local_ratio or 0.0), 6),
            "dest_local_ratio": round(float(pair.dest_local_ratio or 0.0), 6),
            "source_value_class": str(getattr(pair, "source_value_class", "") or ""),
            "dest_value_class": str(getattr(pair, "dest_value_class", "") or ""),
            "source_budget_source": str(getattr(pair, "source_budget_source", "") or ""),
            "dest_budget_source": str(getattr(pair, "dest_budget_source", "") or ""),
            "hive_source_rebalance_bias": round(
                float(getattr(pair, "hive_source_rebalance_bias", 1.0) or 1.0),
                6,
            ),
            "hive_dest_rebalance_bias": round(
                float(getattr(pair, "hive_dest_rebalance_bias", 1.0) or 1.0),
                6,
            ),
            "hive_hint_score_multiplier": round(
                float(getattr(pair, "hive_hint_score_multiplier", 1.0) or 1.0),
                6,
            ),
            "metabolic_rebalance_bias": round(
                float(getattr(pair, "metabolic_rebalance_bias", 1.0) or 1.0),
                6,
            ),
            "metabolic_rebalance_influence": copy.deepcopy(
                getattr(pair, "metabolic_rebalance_influence", {}) or {}
            ),
            "immune_rebalance_bias": round(
                float(getattr(pair, "immune_rebalance_bias", 1.0) or 1.0),
                6,
            ),
            "immune_rebalance_influence": copy.deepcopy(
                getattr(pair, "immune_rebalance_influence", {}) or {}
            ),
            "reason_code": pair.reason_code,
            "route_policy": getattr(
                getattr(getattr(pair, "route_decision", None), "policy", None),
                "value",
                None,
            ),
            "route_summary": self._route_summary(pair.route or []),
            "rejection_reason": pair.rejection_reason or None,
            "score_decomposition": copy.deepcopy(pair.score_decomposition or {}),
        }

    @staticmethod
    def _route_summary(route: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for index, hop in enumerate(route or []):
            if not isinstance(hop, dict):
                continue
            summary.append(
                {
                    "index": index,
                    "channel": str(hop.get("channel", "") or ""),
                    "direction": hop.get("direction"),
                    "id": str(hop.get("id", "") or ""),
                    "amount_msat": hop.get("amount_msat"),
                    "delay": hop.get("delay"),
                }
            )
        return summary

    def _hold_diagnostics(self, snapshot: Optional[Any]) -> Dict[str, int]:
        """Phase 1.2: bucket per-channel state so an operator can tell why
        zero pairs formed without re-deriving it from raw skip rows."""
        buckets = {
            "dest_blocked_by_cooldown": 0,
            "dest_not_funded": 0,
            "source_rejected_neutral": 0,
            "source_protected": 0,
            "source_inside_band": 0,
        }
        channels = getattr(snapshot, "channels", None) or ()
        if not channels:
            return buckets

        cfg = self.config if not hasattr(self.config, "snapshot") else self.config.snapshot()
        low = float(getattr(cfg, "low_liquidity_threshold", 0.35) or 0.35)
        high = float(getattr(cfg, "high_liquidity_threshold", 0.65) or 0.65)

        for ch in channels:
            local_ratio = float(getattr(ch, "local_ratio", 0.0) or 0.0)
            source_reason = str(getattr(ch, "source_reason", "") or "")
            dest_reason = str(getattr(ch, "dest_reason", "") or "")
            if local_ratio < low:
                if dest_reason == "cooldown":
                    buckets["dest_blocked_by_cooldown"] += 1
                elif dest_reason == "no_budget":
                    buckets["dest_not_funded"] += 1
            elif local_ratio > high:
                if source_reason == "not_valuable":
                    buckets["source_rejected_neutral"] += 1
                elif source_reason == "cooldown":
                    buckets["source_protected"] += 1
            else:
                buckets["source_inside_band"] += 1
        return buckets

    def get_drain_demand(self):
        """Residual over-local demand from the last planning pass, or None.

        Consumed by the Boltz structural loop-out path. Read-only snapshot;
        the planner regenerates it every cycle.
        """
        last = self._last_cycle_result
        plan = getattr(last, "plan", None) if last is not None else None
        return getattr(plan, "drain_demand", None)

    def get_last_cycle_debug(self, max_candidates: int = 10) -> Dict[str, Any]:
        result = self._last_cycle_result or CycleResult()
        limit = max(0, int(max_candidates or 0))

        def _limit(items: List[Any]) -> List[Any]:
            if limit <= 0:
                return items
            return items[:limit]

        executions: List[Dict[str, Any]] = []
        for item in _limit(list(result.executions)):
            executions.append({
                "success": bool(getattr(item, "success", False)),
                "error": getattr(item, "error", "") or "",
                "fee_sats": int(getattr(item, "fee_sats", 0) or 0),
                "fee_msat": int(getattr(item, "fee_msat", 0) or 0),
                "attempts": int(getattr(item, "attempts", 0) or 0),
                "route_type": getattr(item, "route_type", "") or "",
                "excluded_channels": list(getattr(item, "excluded_channels", []) or []),
                "failure_data": copy.deepcopy(getattr(item, "failure_data", {}) or {}),
            })

        skipped: List[Dict[str, Any]] = []
        for item in _limit(list(getattr(result.plan, "skipped", []) or [])):
            skipped.append({
                "channel_id": item.channel_id,
                "reason": item.reason,
                "value_class": item.value_class,
                "remaining_budget_sats": int(item.remaining_budget_sats or 0),
                "detail": item.detail or "",
            })

        considered = list(getattr(result, "considered_candidates", []) or [])
        selected = list(getattr(result, "candidates", []) or [])
        snapshot = getattr(result, "snapshot", None)
        return {
            "summary": {
                "considered_pairs": len(considered),
                "selected_pairs": len(selected),
                "skipped_pairs": len(getattr(result.plan, "skipped", []) or []),
                "execution_count": len(result.executions),
                "execution_success_count": sum(1 for item in result.executions if getattr(item, "success", False)),
                "valuable_channel_count": int(getattr(snapshot, "valuable_channel_count", 0) or 0),
                "total_remaining_budget_sats": int(getattr(snapshot, "total_remaining_budget_sats", 0) or 0),
            },
            "hold_diagnostics": self._hold_diagnostics(snapshot),
            "considered_candidates": [
                self._serialize_pair_candidate(pair) for pair in _limit(considered)
            ],
            "selected_candidates": [
                self._serialize_pair_candidate(pair) for pair in _limit(selected)
            ],
            "skipped": skipped,
            "executions": executions,
            "metabolic_rebalance_influence": self._metabolic_rebalance_debug_for_candidates(considered, selected),
            "immune_rebalance_influence": self._immune_rebalance_debug_for_candidates(considered, selected),
        }


    def _metabolic_rebalance_debug_for_candidates(self, considered: List[PairCandidate], selected: List[PairCandidate]) -> Dict[str, Any]:
        influences = []
        for pair in list(considered or []) + list(selected or []):
            influence = getattr(pair, "metabolic_rebalance_influence", {}) or {}
            if isinstance(influence, dict) and influence:
                influences.append(influence)
        reason_codes = []
        for influence in influences:
            for code in influence.get("reason_codes", []) or []:
                code = str(code or "")
                if code and code not in reason_codes:
                    reason_codes.append(code)
        constraints = {}
        try:
            getter = getattr(self._hive_hints, "get_metabolic_action_constraints", None)
            if callable(getter):
                constraints = getter()
        except Exception:
            constraints = {}
        return {
            "seen": any(bool(item.get("seen", False)) for item in influences),
            "usable": any(bool(item.get("usable", False)) for item in influences),
            "candidate_bias_applied": any(float(item.get("bias", 1.0) or 1.0) != 1.0 for item in influences),
            "bias_capped": any(bool(item.get("bias_capped", False)) for item in influences),
            "constraints": constraints if isinstance(constraints, dict) else {},
            "reason_codes": reason_codes,
        }

    def _immune_rebalance_debug_for_candidates(self, considered: List[PairCandidate], selected: List[PairCandidate]) -> Dict[str, Any]:
        influences = []
        for pair in list(considered or []) + list(selected or []):
            influence = getattr(pair, "immune_rebalance_influence", {}) or {}
            if isinstance(influence, dict) and influence:
                influences.append(influence)
        reason_codes = []
        for influence in influences:
            for code in influence.get("reason_codes", []) or []:
                code = str(code or "")
                if code and code not in reason_codes:
                    reason_codes.append(code)
        constraints = {}
        try:
            getter = getattr(self._hive_hints, "get_immune_action_constraints", None)
            if callable(getter):
                constraints = getter()
        except Exception:
            constraints = {}
        return {
            "seen": any(bool(item.get("seen", False)) for item in influences),
            "usable": any(bool(item.get("usable", False)) for item in influences),
            "candidate_bias_applied": any(float(item.get("bias", 1.0) or 1.0) != 1.0 for item in influences),
            "bias_capped": any(bool(item.get("bias_capped", False)) for item in influences),
            "constraints": constraints if isinstance(constraints, dict) else {},
            "reason_codes": reason_codes,
        }

    def _log(self, msg: str, level: str = "info") -> None:
        if self.plugin:
            self.plugin.log(f"[EngineV2] {msg}", level=level)

    def _get_our_id(self) -> Optional[str]:
        if self._our_id:
            return self._our_id
        try:
            if self._data_service is not None:
                self._our_id = self._data_service.get_node_id()
            else:
                self._our_id = self.plugin.rpc.getinfo()["id"]
        except Exception:
            pass
        return self._our_id

    def _build_snapshot(self) -> Optional[StateSnapshot]:
        """Build a normalized state snapshot from live data."""
        cfg = self.config if not hasattr(self.config, 'snapshot') else self.config.snapshot()

        try:
            if self._data_service is not None:
                channels_raw = self._data_service.get_peer_channels().get("channels", [])
            else:
                channels_raw = self.plugin.rpc.listpeerchannels().get("channels", [])
        except Exception as e:
            self._log(f"Failed to get channels: {e}", level="warn")
            return None

        # Build capex allocations
        capex_allocations = None
        if self._capex_engine:
            try:
                capex_allocations = self._capex_engine.compute_allocations()
            except Exception as e:
                self._log(f"Capex allocation failed: {e}", level="warn")

        # Cooldown inputs are loop-invariant: resolve them once. Prefer the
        # batched last-rebalance-time query (one GROUP BY statement for all
        # channels) over a point query per channel when the database exposes
        # it; fall back to per-channel lookups otherwise.
        cooldown_hours = getattr(cfg, 'rebalance_cooldown_hours', 24)
        cooldown_secs = int(cooldown_hours * 3600)
        last_rebalance_times: Optional[Dict[str, int]] = None
        if self.database is not None and cooldown_secs > 0:
            batch_getter = getattr(self.database, "get_last_rebalance_times", None)
            if callable(batch_getter):
                try:
                    fetched = batch_getter()
                    if isinstance(fetched, dict):
                        last_rebalance_times = fetched
                except Exception as e:
                    self._log(
                        f"batch last-rebalance-time lookup failed: {e}",
                        level="debug",
                    )

        # Normalize channel inputs
        normalized = []
        for ch in channels_raw:
            state = ch.get("state", "")
            if "NORMAL" not in state.upper():
                continue

            peer_id = ch.get("peer_id", "")
            scid = ch.get("short_channel_id", "")
            if not scid:
                continue

            capacity_msat = ch.get("total_msat", 0)
            if isinstance(capacity_msat, str):
                capacity_msat = int(capacity_msat.rstrip("msat"))
            capacity_sats = capacity_msat // 1000

            # Use our_amount_msat (true local balance) not spendable_msat
            # (which subtracts channel reserves and in-flight HTLCs).
            local_msat = ch.get("our_amount_msat", ch.get("to_us_msat", 0))
            if isinstance(local_msat, str):
                local_msat = int(local_msat.rstrip("msat"))
            local_sats = local_msat // 1000

            # Actual inbound fee from peer's remote updates
            inbound_fee_ppm = 0
            updates = ch.get("updates", {})
            if updates:
                remote = updates.get("remote", {})
                if remote:
                    inbound_fee_ppm = remote.get("fee_proportional_millionths", 0)

            # Hive membership check
            is_hive = False
            if peer_id:
                is_hive = self._is_hive_member(peer_id)

            # Profitability check
            is_profitable = False
            is_active = False
            if self._profitability:
                try:
                    prof = self._profitability.get_profitability(scid)
                    if prof:
                        revenue = getattr(prof, 'revenue', None)
                        if revenue:
                            is_profitable = getattr(revenue, 'total_contribution_msat', 0) > 0
                            is_active = getattr(revenue, 'total_forward_count', 0) > 5
                except Exception:
                    pass

            # Cooldown: skip if rebalanced recently (default 1 hour)
            cooldown = False
            if self.database and cooldown_secs > 0:
                try:
                    if last_rebalance_times is not None:
                        last_ts = last_rebalance_times.get(scid)
                    else:
                        last_ts = self.database.get_last_rebalance_time(scid)
                    if last_ts and (int(time.time()) - last_ts) < cooldown_secs:
                        cooldown = True
                except Exception:
                    pass

            # Phase 3.3: drift override using anchor state from the last
            # successful rebalance. If the channel has dropped materially
            # below the post-rebalance local ratio, treat it as drifted and
            # let the destination role bypass the cooldown gate.
            cooldown_override = False
            if cooldown and self.database is not None:
                drift_threshold_raw = getattr(
                    cfg, "rebalance_drift_override_ratio", 0.30
                )
                if isinstance(drift_threshold_raw, (int, float)):
                    drift_threshold = float(drift_threshold_raw)
                else:
                    drift_threshold = 0.0
                if drift_threshold > 0.0 and capacity_sats > 0:
                    try:
                        anchor = self.database.get_last_post_rebalance_state(scid)
                    except Exception:
                        anchor = None
                    if anchor and anchor.get("post_local_ratio") is not None:
                        current_ratio = local_sats / capacity_sats
                        anchor_ratio = float(anchor["post_local_ratio"])
                        if anchor_ratio - current_ratio >= drift_threshold:
                            cooldown_override = True

            normalized.append({
                "channel_id": scid,
                "peer_id": peer_id,
                "capacity_sats": capacity_sats,
                "local_sats": local_sats,
                "actual_inbound_fee_ppm": inbound_fee_ppm,
                "is_hive_member": is_hive,
                "is_profitable": is_profitable,
                "is_active": is_active,
                "rebalance_bias": self._get_hive_rebalance_bias(peer_id),
                "cooldown_active": cooldown,
                "cooldown_override": cooldown_override,
            })

        def _coerce_float(name: str, default: float) -> float:
            value = getattr(cfg, name, default)
            if isinstance(value, (int, float)):
                return float(value) if value else float(default if default else 0.0)
            return float(default)

        target_band_low = _coerce_float("low_liquidity_threshold", 0.35)
        target_band_high = _coerce_float("high_liquidity_threshold", 0.65)
        target_emergency_low = _coerce_float("rebalance_emergency_local_ratio", 0.10)
        return build_state_snapshot(
            normalized,
            capex_allocations,
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            target_emergency_low=target_emergency_low,
            hive_bootstrap_budget_sats=int(
                getattr(cfg, "hive_rebalance_bootstrap_budget_sats", 0) or 0
            ),
        )

    def _is_hive_member(self, peer_id: str) -> bool:
        if not peer_id:
            return False
        if self._membership_router is not None:
            try:
                return bool(self._membership_router.is_hive_member(peer_id))
            except Exception:
                pass
        if self._hive_hints is not None:
            try:
                return bool(self._hive_hints.is_hive_member(peer_id))
            except Exception:
                pass
        return False

    def _get_hive_rebalance_bias(self, peer_id: str) -> float:
        if not peer_id or self._hive_hints is None:
            return 1.0
        try:
            bias = float(self._hive_hints.get_rebalance_bias(peer_id))
        except Exception:
            return 1.0
        return max(0.85, min(1.15, bias))

    def _hive_equalization_overlay(
        self,
        snapshot: StateSnapshot,
        cfg: Any,
        *,
        max_chunk_sats: int,
    ) -> PlanResult:
        result = PlanResult()
        if not bool(getattr(cfg, "hive_equalization_enabled", True)):
            return result

        max_candidates = int(
            getattr(cfg, "hive_equalization_max_candidates_per_cycle", 1) or 0
        )
        if max_candidates <= 0:
            return result

        low_pct = float(
            getattr(cfg, "hive_equalization_low_pct", 0.35) or 0.35
        )
        high_pct = float(
            getattr(cfg, "hive_equalization_high_pct", 0.65) or 0.65
        )
        cooldown_hours = int(
            getattr(cfg, "hive_equalization_cooldown_hours", 48) or 0
        )
        cooldown_secs = max(0, cooldown_hours * 3600)
        now = int(time.time())

        hive_high = []
        hive_low = []
        for channel in snapshot.channels:
            if channel.value_class != "hive":
                continue
            if channel.local_ratio > high_pct:
                hive_high.append(channel)
            elif channel.local_ratio < low_pct:
                if cooldown_secs > 0 and self.database is not None:
                    try:
                        last_ts = self.database.get_last_rebalance_time(
                            channel.channel_id,
                            reason_code="hive_equalization",
                        )
                    except Exception:
                        last_ts = None
                    if last_ts and (now - int(last_ts)) < cooldown_secs:
                        result.skipped.append(
                            SkipRecord(
                                channel_id=channel.channel_id,
                                reason="hive_equalization_cooldown",
                                value_class="hive",
                                detail=f"cooldown_until={int(last_ts) + cooldown_secs}",
                            )
                        )
                        continue
                hive_low.append(channel)

        candidate_pairs: List[PairCandidate] = []
        for source in hive_high:
            source_excess = max(
                0,
                int((source.local_ratio - high_pct) * source.capacity_sats),
            )
            if source_excess <= 0:
                continue
            for dest in hive_low:
                if source.channel_id == dest.channel_id:
                    continue
                dest_need = max(
                    0,
                    int((low_pct - dest.local_ratio) * dest.capacity_sats),
                )
                amount_sats = min(source_excess, dest_need, max_chunk_sats)
                if amount_sats <= 0:
                    continue
                candidate_pairs.append(
                    PairCandidate(
                        source_channel_id=source.channel_id,
                        dest_channel_id=dest.channel_id,
                        source_peer_id=source.peer_id,
                        dest_peer_id=dest.peer_id,
                        amount_sats=amount_sats,
                        pair_budget_sats=0,
                        source_capacity_sats=source.capacity_sats,
                        dest_capacity_sats=dest.capacity_sats,
                        source_value_class=source.value_class,
                        dest_value_class=dest.value_class,
                        source_budget_source=source.budget_source,
                        dest_budget_source=dest.budget_source,
                        score=float(source_excess + dest_need),
                        source_local_ratio=source.local_ratio,
                        dest_local_ratio=dest.local_ratio,
                        reason_code="hive_equalization",
                        route_decision=RouteDecision(
                            policy=RoutePolicy.HIVE_ONLY,
                            priority=RoutePriority.HIVE_EQUALIZATION,
                            reason="hive_equalization",
                            allow_market_fallback=False,
                        ),
                    )
                )

        candidate_pairs.sort(
            key=lambda pair: (
                -float(pair.score or 0.0),
                0 if pair.source_peer_id == pair.dest_peer_id else 1,
                -int(pair.amount_sats or 0),
            )
        )

        used_sources: set[str] = set()
        used_dests: set[str] = set()
        for pair in candidate_pairs:
            if len(result.selected) >= max_candidates:
                break
            if pair.source_channel_id in used_sources or pair.dest_channel_id in used_dests:
                continue
            used_sources.add(pair.source_channel_id)
            used_dests.add(pair.dest_channel_id)
            result.selected.append(pair)

        return result

    def find_candidates(self) -> List[PairCandidate]:
        """Dry-run: build snapshot, plan, and return candidates without executing.

        This is the entry point called by EVRebalancer when rebalance_engine=v2.
        Captures the active router at the start of the cycle so mid-cycle
        config flips do not split a cycle across two routers.
        """
        self._cycle_router = self._active_router()
        if self._cycle_router is None:
            self._log(
                "askrene unavailable; v3 router is required for rebalancing",
                level="warn",
            )
            self._cache_cycle_result(CycleResult())
            return []
        router_tag = "v3"

        snapshot = self._build_snapshot()
        if not snapshot or not snapshot.channels:
            self._log("No channels in snapshot")
            self._cache_cycle_result(CycleResult(snapshot=snapshot))
            return []

        cfg = self.config if not hasattr(self.config, 'snapshot') else self.config.snapshot()
        target_band_low = getattr(cfg, 'low_liquidity_threshold', 0.35)
        target_band_high = getattr(cfg, 'high_liquidity_threshold', 0.65)
        max_chunk_sats = getattr(cfg, 'rebalance_max_amount', 2_000_000)
        pair_fee_cap_ppm_raw = getattr(cfg, 'pair_fee_cap_ppm', 0)
        pair_fee_cap_ppm = (
            int(pair_fee_cap_ppm_raw)
            if isinstance(pair_fee_cap_ppm_raw, (int, float))
            else 0
        )

        planner = RebalancePlanner(
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            max_chunk_sats=max_chunk_sats,
            max_pairs=self._max_concurrent_jobs(cfg),
            pair_fee_cap_ppm=pair_fee_cap_ppm,
        )

        plan = planner.plan(snapshot)
        overlay = build_coordination_overlay(
            snapshot,
            hive_hints=self._hive_hints,
            our_node_id=self._get_our_id() or "",
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            max_chunk_sats=max_chunk_sats,
        )
        plan.skipped.extend(overlay.skipped)
        planner_max_pairs = getattr(planner, "max_pairs", 10)
        if not isinstance(planner_max_pairs, int) or planner_max_pairs <= 0:
            planner_max_pairs = 10
        reserved = int(getattr(cfg, "rebalance_coordination_reserved_slots", 0) or 0)
        plan.selected = merge_coordination_pairs(
            plan,
            overlay.selected,
            max_pairs=planner_max_pairs,
            coordination_reserved_slots=reserved,
        )
        if not plan.selected:
            hive_equalization = self._hive_equalization_overlay(
                snapshot,
                cfg,
                max_chunk_sats=max_chunk_sats,
            )
            if hive_equalization.selected:
                selected_channels = {
                    channel_id
                    for pair in hive_equalization.selected
                    for channel_id in (pair.source_channel_id, pair.dest_channel_id)
                }
                plan.skipped = [
                    skip
                    for skip in plan.skipped
                    if skip.channel_id not in selected_channels
                ]
            plan.selected = hive_equalization.selected
            plan.skipped.extend(hive_equalization.skipped)
        for pair in plan.selected:
            self._route_decision_for_pair(pair)
            self._apply_segment_score_bias(pair)
            self._apply_metabolic_rebalance_bias(pair)
            self._apply_immune_rebalance_bias(pair)
            self._update_pair_score_decomposition(pair, route_status="planned")
        priority_rank = {
            RoutePriority.COORDINATED: 0,
            RoutePriority.HIVE_EQUALIZATION: 1,
            RoutePriority.EV_POSITIVE: 2,
            RoutePriority.BACKGROUND: 3,
        }
        plan.selected.sort(
            key=lambda pair: (
                priority_rank.get(getattr(getattr(pair, "route_decision", None), "priority", RoutePriority.EV_POSITIVE), 99),
                -float(getattr(getattr(pair, "route_decision", None), "priority_score", 0.0) or 0.0),
                -float(getattr(pair, "score", 0.0) or 0.0),
            )
        )
        considered_candidates = [copy.deepcopy(pair) for pair in plan.selected]
        considered_lookup = {
            self._pair_key(pair): pair for pair in considered_candidates
        }

        # Route-price selected pairs using the cycle's captured router
        if plan.selected:
            router = self._cycle_router
            priced = []
            hive_router = self._hive_router
            router_begin = getattr(router, "begin_cycle", None)
            if callable(router_begin):
                router_begin()
            if hive_router is not None:
                hive_router.begin_cycle()
            try:
                for pair in plan.selected:
                    pair_key = self._pair_key(pair)
                    debug_pair = considered_lookup.get(pair_key)
                    cooldown = self._get_persisted_pair_cooldown(
                        pair.source_channel_id, pair.dest_channel_id
                    )
                    if cooldown is not None:
                        self._update_pair_score_decomposition(
                            pair,
                            rejection_reason="pair_cooldown",
                            route_status="pair_cooldown",
                        )
                        if debug_pair is not None:
                            self._update_pair_score_decomposition(
                                debug_pair,
                                rejection_reason="pair_cooldown",
                                route_status="pair_cooldown",
                            )
                        plan.skipped.append(SkipRecord(
                            channel_id=pair.dest_channel_id,
                            reason="pair_cooldown",
                            value_class="valuable",
                            detail=(
                                f"src={pair.source_channel_id} "
                                f"kind={cooldown['failure_kind']} "
                                f"count={cooldown['failure_count']} "
                                f"cooldown_until={cooldown['cooldown_until']}"
                            ),
                        ))
                        continue
                    # In-memory futility filter, applied BEFORE route pricing.
                    # run_cycle keeps the same check as a post-pricing backstop,
                    # but pricing a pair that is guaranteed to be discarded
                    # wastes the full router RPC budget for that pair.
                    if self._is_pair_in_futility(
                        pair.source_channel_id, pair.dest_channel_id
                    ):
                        fresh = self._pair_failures.get(
                            (pair.source_channel_id, pair.dest_channel_id), []
                        )
                        self._update_pair_score_decomposition(
                            pair,
                            rejection_reason="pair_futility",
                            route_status="pair_futility",
                        )
                        if debug_pair is not None:
                            self._update_pair_score_decomposition(
                                debug_pair,
                                rejection_reason="pair_futility",
                                route_status="pair_futility",
                            )
                        plan.skipped.append(SkipRecord(
                            channel_id=pair.dest_channel_id,
                            reason="pair_futility",
                            value_class="valuable",
                            remaining_budget_sats=int(
                                getattr(pair, "pair_budget_sats", 0) or 0
                            ),
                            detail=(
                                f"src={pair.source_channel_id} "
                                f"failures={len(fresh)} in window "
                                f"{int(self._futility_window_sec)}s"
                            ),
                        ))
                        continue
                    route_result, route_label = self._route_pair(
                        pair=pair,
                        router=router,
                        exclude=None,
                    )
                    if route_result.success:
                        pair.route_cost_sats = route_result.route_cost_sats
                        pair.route = route_result.route
                        effective_budget = self._probability_adjusted_budget(
                            pair.pair_budget_sats,
                            getattr(route_result, "probability_ppm", 0),
                        )
                        # Persist the planner's effective ceiling so execution
                        # and budget reservation honor the same budget that
                        # accepted this route (no-op when the bonus is 0).
                        pair.effective_budget_sats = int(effective_budget)
                        self._update_pair_score_decomposition(
                            pair,
                            probability_ppm=int(getattr(route_result, "probability_ppm", 0) or 0),
                            route_cost_sats=route_result.route_cost_sats,
                            effective_budget_sats=effective_budget,
                            route_status="priced",
                        )
                        if debug_pair is not None:
                            debug_pair.route_cost_sats = route_result.route_cost_sats
                            debug_pair.route = route_result.route
                            self._update_pair_score_decomposition(
                                debug_pair,
                                probability_ppm=int(getattr(route_result, "probability_ppm", 0) or 0),
                                route_cost_sats=route_result.route_cost_sats,
                                effective_budget_sats=effective_budget,
                                route_status="priced",
                            )
                        if route_result.route_cost_sats <= effective_budget:
                            # Phase 4.3: do_nothing hard gate. A priced pair
                            # whose final_score does not clear the configured
                            # hold margin is rejected with an explicit reason
                            # rather than silently picked.
                            hold_margin_raw = getattr(cfg, "rebalance_hold_margin", 0.0)
                            if isinstance(hold_margin_raw, (int, float)):
                                hold_margin = float(hold_margin_raw)
                            else:
                                hold_margin = 0.0
                            decomp = pair.score_decomposition or {}
                            final_score_present = "final_score" in decomp
                            final_score = float(decomp.get("final_score", 0.0) or 0.0)
                            if final_score_present and final_score <= hold_margin:
                                self._update_pair_score_decomposition(
                                    pair,
                                    probability_ppm=int(getattr(route_result, "probability_ppm", 0) or 0),
                                    route_cost_sats=route_result.route_cost_sats,
                                    effective_budget_sats=effective_budget,
                                    rejection_reason="below_hold_margin",
                                    route_status="below_hold_margin",
                                )
                                if debug_pair is not None:
                                    self._update_pair_score_decomposition(
                                        debug_pair,
                                        probability_ppm=int(getattr(route_result, "probability_ppm", 0) or 0),
                                        route_cost_sats=route_result.route_cost_sats,
                                        effective_budget_sats=effective_budget,
                                        rejection_reason="below_hold_margin",
                                        route_status="below_hold_margin",
                                    )
                                self._audit.log_skip(
                                    channel_id=pair.dest_channel_id,
                                    reason="below_hold_margin",
                                    value_class="valuable",
                                    remaining_budget_sats=pair.pair_budget_sats,
                                    detail=(
                                        f"score={final_score:.4f} margin={hold_margin:.4f}"
                                    ),
                                    router=router_tag,
                                )
                                plan.skipped.append(SkipRecord(
                                    channel_id=pair.dest_channel_id,
                                    reason="below_hold_margin",
                                    value_class="valuable",
                                    detail=(
                                        f"src={pair.source_channel_id} "
                                        f"score={final_score:.4f} "
                                        f"margin={hold_margin:.4f}"
                                    ),
                                ))
                                continue
                            priced.append(pair)
                            self._audit.log_pick(
                                pair.source_channel_id,
                                pair.dest_channel_id,
                                pair.amount_sats,
                                route_result.route_cost_sats,
                                pair.score,
                                router=route_label,
                            )
                        else:
                            self._update_pair_score_decomposition(
                                pair,
                                probability_ppm=int(getattr(route_result, "probability_ppm", 0) or 0),
                                route_cost_sats=route_result.route_cost_sats,
                                effective_budget_sats=effective_budget,
                                rejection_reason="route_over_budget",
                                route_status="route_over_budget",
                            )
                            if debug_pair is not None:
                                self._update_pair_score_decomposition(
                                    debug_pair,
                                    probability_ppm=int(getattr(route_result, "probability_ppm", 0) or 0),
                                    route_cost_sats=route_result.route_cost_sats,
                                    effective_budget_sats=effective_budget,
                                    rejection_reason="route_over_budget",
                                    route_status="route_over_budget",
                                )
                            plan.skipped.append(SkipRecord(
                                channel_id=pair.dest_channel_id,
                                reason="route_over_budget",
                                value_class="valuable",
                                remaining_budget_sats=pair.pair_budget_sats,
                                detail=(
                                    f"route_cost={route_result.route_cost_sats} "
                                    f"effective_budget={effective_budget} "
                                    f"probability_ppm={getattr(route_result, 'probability_ppm', 0)}"
                                ),
                            ))
                    else:
                        # Native execution requires a priced explicit route.
                        # Skip with reason='no_route' instead of submitting an
                        # unpriced attempt.
                        self._update_pair_score_decomposition(
                            pair,
                            rejection_reason="no_route",
                            route_status="no_route",
                        )
                        if debug_pair is not None:
                            self._update_pair_score_decomposition(
                                debug_pair,
                                rejection_reason="no_route",
                                route_status="no_route",
                            )
                        plan.skipped.append(SkipRecord(
                            channel_id=pair.dest_channel_id,
                            reason="no_route",
                            value_class="valuable",
                            remaining_budget_sats=pair.pair_budget_sats,
                            detail=str(route_result.error or "router_no_route"),
                        ))
                        continue

            finally:
                if hive_router is not None:
                    hive_router.end_cycle()
                router_end = getattr(router, "end_cycle", None)
                if callable(router_end):
                    router_end()

            plan.selected = priced

        # Audit all skips
        for skip in plan.skipped:
            self._audit.log_skip(
                channel_id=skip.channel_id,
                reason=skip.reason,
                value_class=skip.value_class,
                remaining_budget_sats=skip.remaining_budget_sats,
                detail=skip.detail or "",
                router=router_tag,
            )

        self._audit.log_cycle_summary(
            selected_count=len(plan.selected),
            skipped_count=len(plan.skipped),
            total_valuable=snapshot.valuable_channel_count,
            total_channels=len(snapshot.channels),
            total_budget_sats=snapshot.total_remaining_budget_sats,
        )
        self._cache_cycle_result(
            CycleResult(
                considered_candidates=considered_candidates,
                candidates=list(plan.selected),
                audit_records=list(plan.skipped),
                snapshot=snapshot,
                plan=plan,
            )
        )

        return plan.selected

    def _route_decision_for_pair(self, pair: PairCandidate):
        decision = getattr(pair, "route_decision", None)
        if decision is not None:
            return decision
        decision = decide_route_policy(
            pair,
            reason_code=getattr(pair, "reason_code", "") or "ev_positive",
            hive_hints=self._hive_hints,
        )
        pair.route_decision = decision
        return decision

    def _apply_segment_score_bias(self, pair: PairCandidate) -> None:
        """Apply bounded fleet segment utility bias to pair scoring."""
        hive_hints = self._hive_hints
        if hive_hints is None:
            return
        multiplier = pair_segment_bias_multiplier(
            pair,
            hive_hints,
            self._get_our_id() or "",
        )
        pair.score = float(getattr(pair, "score", 0.0) or 0.0) * multiplier

    def _apply_metabolic_rebalance_bias(self, pair: PairCandidate) -> None:
        """Apply fresh, scoped metabolic influence as a bounded score modifier."""
        hive_hints = self._hive_hints
        if hive_hints is None:
            return
        try:
            getter = getattr(hive_hints, "get_metabolic_rebalance_bias", None)
            if not callable(getter):
                return
            raw_bias = float(getter(pair.source_peer_id, pair.dest_peer_id))
            bias = max(0.85, min(1.15, raw_bias))
            pair.metabolic_rebalance_bias = bias
            influence = {
                "seen": False,
                "usable": bias != 1.0,
                "bias": bias,
                "bias_capped": abs(raw_bias - bias) > 1e-9,
                "reason_codes": [],
            }
            peer_getter = getattr(hive_hints, "get_metabolic_peer_effect", None)
            if callable(peer_getter):
                reason_codes = []
                capped = influence["bias_capped"]
                seen = False
                usable = False
                for peer_id in (pair.source_peer_id, pair.dest_peer_id):
                    effect = peer_getter(peer_id)
                    if not isinstance(effect, dict):
                        continue
                    seen = seen or bool(effect)
                    usable = usable or bool(effect.get("usable", False))
                    capped = capped or bool(effect.get("bias_capped", False))
                    for code in effect.get("reason_codes", []) or []:
                        code = str(code or "")
                        if code and code not in reason_codes:
                            reason_codes.append(code)
                influence.update({
                    "seen": seen,
                    "usable": usable and bias != 1.0,
                    "bias_capped": capped,
                    "reason_codes": reason_codes,
                })
            pair.metabolic_rebalance_influence = influence
            pair.score = float(getattr(pair, "score", 0.0) or 0.0) * bias
        except Exception:
            pair.metabolic_rebalance_bias = 1.0
            pair.metabolic_rebalance_influence = {
                "seen": False,
                "usable": False,
                "bias": 1.0,
                "bias_capped": False,
                "reason_codes": [],
                "reason": "error",
            }

    def _apply_immune_rebalance_bias(self, pair: PairCandidate) -> None:
        """Apply fresh, scoped immune influence as a bounded score modifier."""
        hive_hints = self._hive_hints
        if hive_hints is None:
            return
        try:
            getter = getattr(hive_hints, "get_immune_rebalance_bias", None)
            if not callable(getter):
                return
            raw_bias = float(getter(pair.source_peer_id, pair.dest_peer_id))
            bias = max(0.85, min(1.15, raw_bias))
            pair.immune_rebalance_bias = bias
            influence = {
                "seen": False,
                "usable": bias != 1.0,
                "bias": bias,
                "bias_capped": abs(raw_bias - bias) > 1e-9,
                "reason_codes": [],
            }
            peer_getter = getattr(hive_hints, "get_immune_peer_effect", None)
            if callable(peer_getter):
                reason_codes = []
                capped = influence["bias_capped"]
                seen = False
                usable = False
                for peer_id in (pair.source_peer_id, pair.dest_peer_id):
                    effect = peer_getter(peer_id)
                    if not isinstance(effect, dict):
                        continue
                    seen = seen or bool(effect)
                    usable = usable or bool(effect.get("usable", False))
                    capped = capped or bool(effect.get("bias_capped", False))
                    for code in effect.get("reason_codes", []) or []:
                        code = str(code or "")
                        if code and code not in reason_codes:
                            reason_codes.append(code)
                influence.update({
                    "seen": seen,
                    "usable": usable and bias != 1.0,
                    "bias_capped": capped,
                    "reason_codes": reason_codes,
                })
            pair.immune_rebalance_influence = influence
            pair.score = float(getattr(pair, "score", 0.0) or 0.0) * bias
        except Exception:
            pair.immune_rebalance_bias = 1.0
            pair.immune_rebalance_influence = {
                "seen": False,
                "usable": False,
                "bias": 1.0,
                "bias_capped": False,
                "reason_codes": [],
                "reason": "error",
            }

    @staticmethod
    def _market_price_pair(
        router: Any,
        pair: PairCandidate,
        exclude: Optional[List[str]],
        *,
        market_only_layers: bool = False,
    ):
        kwargs = {
            "source_channel_id": pair.source_channel_id,
            "dest_channel_id": pair.dest_channel_id,
            "source_peer_id": pair.source_peer_id,
            "dest_peer_id": pair.dest_peer_id,
            "amount_sats": pair.amount_sats,
            "exclude": exclude,
        }
        if market_only_layers:
            kwargs.update(
                {
                    "layer_names_override": [],
                    "include_observed_liquidity": False,
                }
            )
        try:
            return router.price_pair(**kwargs)
        except TypeError as exc:
            if not market_only_layers:
                raise
            message = str(exc)
            if (
                "layer_names_override" not in message
                and "include_observed_liquidity" not in message
                and "unexpected keyword" not in message
            ):
                raise
            kwargs.pop("layer_names_override", None)
            kwargs.pop("include_observed_liquidity", None)
            return router.price_pair(**kwargs)

    @staticmethod
    def _route_error(error: str) -> RouteResult:
        return RouteResult(success=False, error=error)

    def _hybrid_choice(self, hive_result: Any, market_result: Any, decision: Any):
        if getattr(hive_result, "success", False) and not getattr(market_result, "success", False):
            return hive_result, "hive"
        if getattr(market_result, "success", False) and not getattr(hive_result, "success", False):
            return market_result, "market"
        if not getattr(hive_result, "success", False) and not getattr(market_result, "success", False):
            return market_result, "market"
        hive_cost = int(getattr(hive_result, "route_cost_sats", 0) or 0)
        market_cost = int(getattr(market_result, "route_cost_sats", 0) or 0)
        if hive_cost < market_cost:
            return hive_result, "hive"
        if market_cost < hive_cost:
            return market_result, "market"
        if bool(getattr(decision, "prefer_hive_on_tie", True)):
            return hive_result, "hive"
        return market_result, "market"

    @staticmethod
    def _fail_closed_on_route_failure(decision: Any) -> bool:
        policy = getattr(decision, "policy", None)
        if policy not in (RoutePolicy.HIVE_ONLY, RoutePolicy.HYBRID):
            return False
        return not bool(getattr(decision, "allow_market_fallback", True))

    def _route_pair(
        self,
        *,
        pair: PairCandidate,
        router: Any,
        exclude: Optional[List[str]],
    ):
        decision = self._route_decision_for_pair(pair)
        if decision.policy is RoutePolicy.MARKET_ONLY:
            return self._market_price_pair(
                router,
                pair,
                exclude,
                market_only_layers=True,
            ), "market"

        hive_router = self._hive_router
        if hive_router is None:
            if self._fail_closed_on_route_failure(decision):
                return self._route_error("hive_router_unavailable"), "hive"
            return self._market_price_pair(
                router,
                pair,
                exclude,
                market_only_layers=True,
            ), "market"

        try:
            hive_result = hive_router.price_pair(
                pair,
                decision,
                exclude=exclude,
            )
        except Exception as e:
            hive_result = self._route_error(f"hive_route_error: {e}")

        if decision.policy is RoutePolicy.HIVE_ONLY:
            if getattr(hive_result, "success", False) or self._fail_closed_on_route_failure(decision):
                return hive_result, "hive"
            return self._market_price_pair(
                router,
                pair,
                exclude,
                market_only_layers=True,
            ), "market"

        if (
            decision.policy is RoutePolicy.HYBRID
            and getattr(hive_result, "success", False)
            and int(getattr(hive_result, "route_cost_sats", 0) or 0) == 0
            and bool(getattr(decision, "prefer_hive_on_tie", True))
        ):
            # Free hive route (typical intra-fleet). A market quote cannot
            # beat zero cost, and prefer_hive_on_tie would pick the hive
            # route even on a 0-cost tie, so the market pricing pass is
            # provably redundant — skip its RPCs entirely.
            return hive_result, "hive"

        market_result = self._market_price_pair(
            router,
            pair,
            exclude,
            market_only_layers=True,
        )
        if decision.policy is RoutePolicy.HYBRID:
            if not getattr(hive_result, "success", False) and self._fail_closed_on_route_failure(decision):
                return hive_result, "hive"
            return self._hybrid_choice(hive_result, market_result, decision)

        return market_result, "market"

    def _prune_pair_failures(self, key: Tuple[str, str]) -> List[float]:
        """Drop failure timestamps older than the futility window and return survivors."""
        timestamps = self._pair_failures.get(key, [])
        if not timestamps:
            return timestamps
        cutoff = time.time() - self._futility_window_sec
        fresh = [t for t in timestamps if t >= cutoff]
        if len(fresh) != len(timestamps):
            if fresh:
                self._pair_failures[key] = fresh
            else:
                self._pair_failures.pop(key, None)
        return fresh

    def _get_persisted_pair_cooldown(
        self, source_channel_id: str, dest_channel_id: str
    ) -> Optional[Dict[str, Any]]:
        if self.database is None:
            return None
        getter = getattr(self.database, "get_pair_rebalance_cooldown", None)
        if getter is None:
            return None
        try:
            cooldown = getter(source_channel_id, dest_channel_id)
            return cooldown if isinstance(cooldown, dict) else None
        except Exception as e:
            self._log(
                f"Pair cooldown lookup failed for {source_channel_id}->{dest_channel_id}: {e}",
                level="warn",
            )
            return None

    def _is_pair_in_futility(self, source_channel_id: str, dest_channel_id: str) -> bool:
        """Return True if this pair has hit the failure threshold in the window."""
        key = (source_channel_id, dest_channel_id)
        fresh = self._prune_pair_failures(key)
        return len(fresh) >= self._futility_threshold

    def _record_pair_failure(self, source_channel_id: str, dest_channel_id: str) -> None:
        """Record an execution failure for this pair at the current time."""
        key = (source_channel_id, dest_channel_id)
        self._pair_failures.setdefault(key, []).append(time.time())

    def _record_pair_success(self, source_channel_id: str, dest_channel_id: str) -> None:
        """Clear any failure history for this pair (success resets the counter)."""
        self._pair_failures.pop((source_channel_id, dest_channel_id), None)

    def _classify_failure_kind(self, exec_result: ExecutionResult) -> str:
        error = str(getattr(exec_result, "error", "") or "").lower()
        if "temporary_channel_failure" in error:
            return "temporary_channel_failure"
        # Treat NoRoutes as transient: gossip may not be converged, peer-side
        # liquidity may not be visible, or capacity may have briefly dipped.
        if "noroutes" in error or "no_routes" in error or "no route" in error:
            return "temporary_channel_failure"
        if "fee_insufficient" in error:
            return "fee_insufficient"
        if "incorrect_cltv_expiry" in error:
            return "incorrect_cltv_expiry"
        if "permanent_failure" in error:
            return "permanent_failure"
        if "payment_pending_timeout" in error:
            return "payment_pending_timeout"
        if "local_execution_failed" in error:
            return "local_execution_failed"
        return "other_retriable"

    def _persist_pair_failure(self, pair: PairCandidate, exec_result: ExecutionResult) -> None:
        if self.database is None:
            return
        recorder = getattr(self.database, "record_pair_rebalance_failure", None)
        if recorder is None:
            return
        failure_kind = self._classify_failure_kind(exec_result)
        cooldown_seconds = self._pair_failure_cooldowns.get(
            failure_kind,
            self._pair_failure_cooldowns["other_retriable"],
        )
        try:
            recorder(
                pair.source_channel_id,
                pair.dest_channel_id,
                failure_kind,
                cooldown_seconds=cooldown_seconds,
            )
        except Exception as e:
            self._log(
                f"Persist pair failure failed for {pair.source_channel_id}->{pair.dest_channel_id}: {e}",
                level="warn",
            )

    def _clear_persisted_pair_failure(self, pair: PairCandidate) -> None:
        if self.database is None:
            return
        clearer = getattr(self.database, "clear_pair_rebalance_failure", None)
        if clearer is None:
            return
        try:
            clearer(pair.source_channel_id, pair.dest_channel_id)
        except Exception as e:
            self._log(
                f"Clear pair failure failed for {pair.source_channel_id}->{pair.dest_channel_id}: {e}",
                level="warn",
            )

    def _probability_adjusted_budget(
        self, pair_budget_sats: int, probability_ppm: int
    ) -> int:
        """Relax the raw pair budget by a probability-weighted bonus.

        Returns the base pair_budget_sats unchanged when either the config
        bonus rate is 0 (default) or the router reported no probability.
        With a positive bonus rate and non-zero probability, the effective
        budget is:

            pair_budget * (1 + clamp(probability_ppm, 0, 1_000_000) / 1_000_000 * bonus)

        Example: bonus=0.25, probability_ppm=982_339 →
            effective = pair_budget * (1 + 0.982339 * 0.25)
                      = pair_budget * 1.2456

        The intent is to unlock higher-probability-but-pricier routes when
        the configured router can score route probability.
        """
        bonus_rate = getattr(self.config, "capex_probability_budget_bonus", 0.0)
        if not isinstance(bonus_rate, (int, float)):
            bonus_rate = 0.0
        if bonus_rate <= 0.0 or probability_ppm <= 0:
            return pair_budget_sats
        clamped = min(1.0, max(0.0, probability_ppm / 1_000_000.0))
        bonus_fraction = clamped * bonus_rate
        return int(pair_budget_sats * (1.0 + bonus_fraction))

    @staticmethod
    def _pair_max_fee_sats(pair: PairCandidate) -> int:
        """Return the fee ceiling execution must honor for this pair.

        When the probability budget bonus accepted a route above the raw
        pair budget, find_candidates stores the planner's effective budget
        on the pair (``effective_budget_sats``). Execution and budget
        reservation must use that same ceiling or _validate_route would
        deterministically reject every bonus-band route. With the default
        bonus of 0.0 the effective budget equals the raw pair budget and
        behavior is unchanged.
        """
        base = int(getattr(pair, "pair_budget_sats", 0) or 0)
        effective = getattr(pair, "effective_budget_sats", None)
        if effective is None:
            return base
        try:
            effective = int(effective)
        except (TypeError, ValueError):
            return base
        return max(base, effective)

    def _config_snapshot(self) -> Any:
        return self.config.snapshot() if hasattr(self.config, "snapshot") else self.config

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
        except Exception as exc:
            self._log(f"external liquidity cost provider failed: {exc}", level="warn")
            return {"spent_24h_sats": 0, "reserved_24h_sats": 0}

    def _get_global_budget_limit(self, cfg: Any) -> int:
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
            except Exception as exc:
                self._log(f"global budget limit provider failed: {exc}", level="warn")
        return max(0, int(getattr(cfg, "daily_budget_sats", 0) or 0))

    def _reserve_execution_budget(
        self,
        pair: PairCandidate,
        *,
        reservation_id: str,
    ) -> Tuple[bool, Optional[ExecutionResult]]:
        """Reserve the maximum fee budget for one automatic execution.

        Manual/explicit execute_candidate callers deliberately skip this path;
        their caller owns override semantics and accounting.
        """
        max_fee_sats = max(0, self._pair_max_fee_sats(pair))
        cfg = self._config_snapshot()
        effective_budget = self._get_global_budget_limit(cfg)
        allow_zero_cost_with_zero_budget = bool(
            getattr(cfg, "allow_zero_cost_auto_rebalance_when_budget_zero", False)
        )
        if effective_budget <= 0:
            if max_fee_sats <= 0 and allow_zero_cost_with_zero_budget:
                return False, None
            return False, ExecutionResult(
                success=False,
                amount_sats=int(getattr(pair, "amount_sats", 0) or 0),
                error="zero_budget_blocks_auto_rebalance",
                route_type=self._executor_mode(),
            )
        if max_fee_sats <= 0:
            return False, None
        if self.database is None or not callable(getattr(self.database, "reserve_budget", None)):
            return False, ExecutionResult(
                success=False,
                amount_sats=int(getattr(pair, "amount_sats", 0) or 0),
                error="local_budget_block: reserve_budget_unavailable",
                route_type=self._executor_mode(),
            )

        now = int(time.time())
        window_hours = max(
            1,
            int(getattr(cfg, "total_cost_budget_window_hours", 24) or 24),
        )
        since_ts = now - (window_hours * 3600)
        external = self._get_external_liquidity_costs()
        ext_spent = int(external.get("spent_24h_sats", 0) or 0)
        ext_reserved = int(external.get("reserved_24h_sats", 0) or 0)
        budget_limit = max(0, effective_budget - ext_spent - ext_reserved)

        try:
            reserved, remaining = self.database.reserve_budget(
                reservation_id=reservation_id,
                amount_sats=max_fee_sats,
                channel_id=pair.dest_channel_id,
                budget_limit=budget_limit,
                since_timestamp=since_ts,
                weekly_budget_limit=getattr(cfg, "weekly_budget_sats", None),
                weekly_since_timestamp=now - 7 * 86400,
            )
        except Exception as exc:
            self._log(
                f"budget reservation failed for {pair.source_channel_id}->{pair.dest_channel_id}: {exc}",
                level="warn",
            )
            return False, ExecutionResult(
                success=False,
                amount_sats=int(getattr(pair, "amount_sats", 0) or 0),
                error=f"local_budget_block: {exc}",
                route_type=self._executor_mode(),
            )

        if reserved:
            return True, None

        return False, ExecutionResult(
            success=False,
            amount_sats=int(getattr(pair, "amount_sats", 0) or 0),
            error=(
                f"local_budget_block: {remaining} sats remaining "
                f"of {budget_limit} after external costs"
            ),
            route_type=self._executor_mode(),
        )

    def _finish_execution_budget(
        self,
        *,
        reservation_id: str,
        reserved_budget: bool,
        result: Optional[ExecutionResult],
    ) -> None:
        if not reserved_budget or self.database is None:
            return
        try:
            if result is not None and getattr(result, "payment_pending", False):
                # Payment unresolved: keep the reservation active so the
                # budget stays held until the reconciliation sweep confirms
                # settlement or failure. Releasing here would let the next
                # cycle spend the same budget while the HTLC can still settle.
                return
            if result is not None and result.success:
                self.database.mark_budget_spent(
                    reservation_id,
                    max(0, int(getattr(result, "fee_sats", 0) or 0)),
                )
            else:
                self.database.release_budget_reservation(reservation_id)
        except Exception as exc:
            self._log(
                f"budget reservation cleanup failed for {reservation_id}: {exc}",
                level="warn",
            )

    def _execution_kwargs(self, pair: PairCandidate) -> Dict[str, Any]:
        router_kind = "v3" if self._cycle_router is self.router_v3 else "v2"
        decision = self._route_decision_for_pair(pair)
        return {
            "route": pair.route or [],
            "amount_sats": pair.amount_sats,
            "source_channel_id": pair.source_channel_id,
            "dest_channel_id": pair.dest_channel_id,
            "max_fee_sats": self._pair_max_fee_sats(pair),
            "observation_store": self._segment_observation_store,
            "observation_context": {
                "short_channel_id": pair.dest_channel_id,
                "direction": self._segment_observation_direction(pair.dest_peer_id),
                "source_channel_id": pair.source_channel_id,
                "dest_channel_id": pair.dest_channel_id,
                "route_policy": getattr(
                    decision.policy, "value", str(decision.policy)
                ),
                "router_kind": router_kind,
                "correlation_id": (
                    f"{pair.source_channel_id}->{pair.dest_channel_id}:{int(time.time())}"
                ),
            },
        }

    def _executor_mode(self) -> str:
        return "native"

    def _make_executor(self) -> NativeRouteExecutor:
        return NativeRouteExecutor(
            self.plugin,
            self.database,
            observation_store=self._segment_observation_store,
            # Inject the engine's cached node id so executor.is_available()
            # does not pay a getinfo RPC every cycle. Falls back to the
            # executor's own lazy getinfo when the engine has no id yet.
            our_id=self._get_our_id(),
        )

    def _executor_unavailable_reason(self) -> tuple[str, str]:
        return "native_unavailable", "native route executor RPC surface unavailable"

    def _retry_native_pair_with_exclusions(
        self,
        pair: PairCandidate,
        executor: Any,
        first_result: ExecutionResult,
    ) -> ExecutionResult:
        """Retry a native sendpay once after excluding failed route segments."""
        if self._executor_mode() != "native":
            return first_result
        if getattr(first_result, "success", False):
            return first_result
        if getattr(first_result, "payment_pending", False):
            # The first payment may still settle — never pay again on top.
            return first_result

        exclusions: List[str] = []
        seen: set[str] = set()
        for entry in getattr(first_result, "excluded_channels", []) or []:
            value = str(entry or "").strip()
            if value and value not in seen:
                seen.add(value)
                exclusions.append(value)
        if not exclusions:
            return first_result

        router = self._cycle_router or self._active_router()
        if router is None:
            return first_result

        # Keep the retry conservative: when the hive router owns route choice,
        # do not call it from an execution worker outside its cycle window.
        decision = self._route_decision_for_pair(pair)
        if self._hive_router is not None and decision.policy is not RoutePolicy.MARKET_ONLY:
            return first_result

        retry_data = dict(getattr(first_result, "failure_data", {}) or {})
        retry_data["retry_excluded_channels"] = list(exclusions)
        try:
            route_result, route_label = self._route_pair(
                pair=pair,
                router=router,
                exclude=exclusions,
            )
        except Exception as exc:
            first_result.error = f"{first_result.error}; retry_pricing_failed: {exc}"
            retry_data["retry_error"] = str(exc)
            first_result.failure_data = retry_data
            first_result.attempts = max(1, int(first_result.attempts or 1)) + 1
            return first_result

        if not getattr(route_result, "success", False):
            detail = str(getattr(route_result, "error", "") or "no_route")
            first_result.error = f"{first_result.error}; retry_no_route: {detail}"
            retry_data["retry_error"] = detail
            first_result.failure_data = retry_data
            first_result.attempts = max(1, int(first_result.attempts or 1)) + 1
            return first_result

        effective_budget = self._probability_adjusted_budget(
            int(getattr(pair, "pair_budget_sats", 0) or 0),
            int(getattr(route_result, "probability_ppm", 0) or 0),
        )
        if int(getattr(route_result, "route_cost_sats", 0) or 0) > effective_budget:
            first_result.error = (
                f"{first_result.error}; retry_route_over_budget: "
                f"{int(getattr(route_result, 'route_cost_sats', 0) or 0)} > {effective_budget}"
            )
            retry_data["retry_error"] = "route_over_budget"
            first_result.failure_data = retry_data
            first_result.attempts = max(1, int(first_result.attempts or 1)) + 1
            return first_result

        pair.route = list(getattr(route_result, "route", []) or [])
        pair.route_cost_sats = int(getattr(route_result, "route_cost_sats", 0) or 0)
        self._log(
            f"Retrying native rebalance {pair.source_channel_id}->{pair.dest_channel_id} "
            f"with exclusions={exclusions} via {route_label}",
            level="debug",
        )
        try:
            retry_result = executor.execute(**self._execution_kwargs(pair))
        except Exception as exc:
            retry_result = ExecutionResult(
                success=False,
                amount_sats=int(getattr(pair, "amount_sats", 0) or 0),
                error=f"executor_error: {exc}",
                route_type="native",
            )

        retry_attempts = max(1, int(getattr(retry_result, "attempts", 0) or 1))
        first_attempts = max(1, int(getattr(first_result, "attempts", 0) or 1))
        retry_result.attempts = first_attempts + retry_attempts
        merged_data = dict(getattr(retry_result, "failure_data", {}) or {})
        merged_data.setdefault("previous_failure", first_result.error)
        merged_data.setdefault("retry_excluded_channels", list(exclusions))
        merged_data.setdefault("retry_route_cost_sats", pair.route_cost_sats)
        retry_result.failure_data = merged_data
        if not retry_result.success:
            retry_result.excluded_channels = list(
                dict.fromkeys(
                    list(getattr(retry_result, "excluded_channels", []) or [])
                    + exclusions
                )
            )
        return retry_result

    @staticmethod
    def _native_partial_amounts(amount_sats: int) -> List[int]:
        """Return bounded descending partial-fill retry amounts."""
        try:
            original = int(amount_sats)
        except (TypeError, ValueError):
            return []
        if original <= 0:
            return []

        min_amount = min(original - 1, max(1_000, min(5_000, original // 2)))
        if min_amount <= 0:
            return []

        amounts: List[int] = []
        current = original // 2
        while current >= min_amount and len(amounts) < 6:
            if current < original and current not in amounts:
                amounts.append(current)
            next_amount = current // 2
            if next_amount == current:
                break
            current = next_amount

        if min_amount < original and min_amount not in amounts:
            amounts.append(min_amount)
        return amounts[:7]

    @staticmethod
    def _native_failure_allows_partial(result: ExecutionResult) -> bool:
        if getattr(result, "success", False):
            return False
        if str(getattr(result, "route_type", "") or "") != "native":
            return False
        data = getattr(result, "failure_data", {}) or {}
        if str(data.get("failure_class") or "").lower() == "liquidity":
            return True
        error = str(getattr(result, "error", "") or "").lower()
        return "temporary_channel_failure" in error

    def _retry_native_pair_with_partial_amounts(
        self,
        pair: PairCandidate,
        executor: Any,
        prior_result: ExecutionResult,
    ) -> ExecutionResult:
        """Retry native execution at smaller amounts after a liquidity failure."""
        if self._executor_mode() != "native":
            return prior_result
        if getattr(prior_result, "payment_pending", False):
            # The first payment may still settle — never pay again on top.
            return prior_result
        if not self._native_failure_allows_partial(prior_result):
            return prior_result

        router = self._cycle_router or self._active_router()
        if router is None:
            return prior_result

        decision = self._route_decision_for_pair(pair)
        if self._hive_router is not None and decision.policy is not RoutePolicy.MARKET_ONLY:
            return prior_result

        original_amount = int(getattr(pair, "amount_sats", 0) or 0)
        original_route = list(getattr(pair, "route", []) or [])
        original_route_cost = getattr(pair, "route_cost_sats", None)
        original_budget = int(getattr(pair, "pair_budget_sats", 0) or 0)
        original_effective_budget = getattr(pair, "effective_budget_sats", None)
        partial_attempts: List[Dict[str, Any]] = []
        total_attempts = max(1, int(getattr(prior_result, "attempts", 0) or 1))

        for amount_sats in self._native_partial_amounts(original_amount):
            pair.amount_sats = amount_sats
            # Scale the fee budget proportionally to the retry amount so the
            # partial fill keeps the original plan's fee-rate ceiling. Without
            # this a 5k-sat fill could legally pay the full pair budget
            # (e.g. 5,000 sats fee on 5,000 sats = 1,000,000 ppm).
            scaled_budget = original_budget
            if original_amount > 0 and original_budget > 0:
                scaled_budget = max(
                    1,
                    -(-original_budget * amount_sats // original_amount),  # ceil
                )
            pair.pair_budget_sats = scaled_budget
            try:
                route_result, route_label = self._route_pair(
                    pair=pair,
                    router=router,
                    exclude=None,
                )
            except Exception as exc:
                partial_attempts.append(
                    {
                        "amount_sats": amount_sats,
                        "status": "pricing_error",
                        "error": str(exc),
                    }
                )
                continue

            if not getattr(route_result, "success", False):
                partial_attempts.append(
                    {
                        "amount_sats": amount_sats,
                        "status": "no_route",
                        "error": str(getattr(route_result, "error", "") or "no_route"),
                    }
                )
                continue

            route_cost = int(getattr(route_result, "route_cost_sats", 0) or 0)
            effective_budget = self._probability_adjusted_budget(
                scaled_budget,
                int(getattr(route_result, "probability_ppm", 0) or 0),
            )
            pair.effective_budget_sats = int(effective_budget)
            if route_cost > effective_budget:
                partial_attempts.append(
                    {
                        "amount_sats": amount_sats,
                        "status": "route_over_budget",
                        "route_cost_sats": route_cost,
                        "effective_budget_sats": effective_budget,
                    }
                )
                continue

            pair.route = list(getattr(route_result, "route", []) or [])
            pair.route_cost_sats = route_cost
            self._log(
                f"Retrying native rebalance {pair.source_channel_id}->{pair.dest_channel_id} "
                f"as partial amount={amount_sats} sats via {route_label}",
                level="debug",
            )
            try:
                retry_result = executor.execute(**self._execution_kwargs(pair))
            except Exception as exc:
                retry_result = ExecutionResult(
                    success=False,
                    amount_sats=amount_sats,
                    error=f"executor_error: {exc}",
                    route_type="native",
                )

            total_attempts += max(1, int(getattr(retry_result, "attempts", 0) or 1))
            if getattr(retry_result, "success", False):
                retry_result.attempts = total_attempts
                retry_result.amount_sats = int(getattr(retry_result, "amount_sats", 0) or amount_sats)
                self._update_pair_score_decomposition(
                    pair,
                    probability_ppm=int(getattr(route_result, "probability_ppm", 0) or 0),
                    route_cost_sats=route_cost,
                    effective_budget_sats=effective_budget,
                    route_status="partial_priced",
                )
                merged_data = dict(getattr(retry_result, "failure_data", {}) or {})
                merged_data.setdefault("previous_failure", prior_result.error)
                merged_data["partial_fill"] = {
                    "planned_amount_sats": original_amount,
                    "executed_amount_sats": retry_result.amount_sats,
                    "route_cost_sats": route_cost,
                    "attempts": partial_attempts
                    + [{"amount_sats": amount_sats, "status": "success"}],
                }
                pair.score_decomposition["partial_fill"] = dict(
                    merged_data["partial_fill"]
                )
                retry_result.failure_data = merged_data
                return retry_result

            partial_attempts.append(
                {
                    "amount_sats": amount_sats,
                    "status": "execution_failed",
                    "error": str(getattr(retry_result, "error", "") or ""),
                    "excluded_channels": list(
                        getattr(retry_result, "excluded_channels", []) or []
                    ),
                }
            )

        pair.amount_sats = original_amount
        pair.route = original_route
        pair.route_cost_sats = original_route_cost
        pair.pair_budget_sats = original_budget
        if original_effective_budget is None:
            if hasattr(pair, "effective_budget_sats"):
                del pair.effective_budget_sats
        else:
            pair.effective_budget_sats = original_effective_budget
        data = dict(getattr(prior_result, "failure_data", {}) or {})
        data["partial_fill"] = {
            "planned_amount_sats": original_amount,
            "executed_amount_sats": 0,
            "attempts": partial_attempts,
        }
        prior_result.failure_data = data
        prior_result.attempts = total_attempts
        if partial_attempts:
            prior_result.error = (
                f"{prior_result.error}; partial_retry_failed: "
                f"{partial_attempts[-1].get('status')}"
            )
        return prior_result

    def _execute_pair(
        self,
        pair: PairCandidate,
        executor: NativeRouteExecutor,
        *,
        reserve_budget: bool = False,
        account_costs: bool = False,
        rebalance_id: Optional[int] = None,
    ) -> Optional[ExecutionResult]:
        """Execute a single pair in a worker thread.

        Native execution requires the route priced by the active router and
        executes that exact route.

        ``rebalance_id``: when the caller (manual/diagnostic/execute_rebalance
        paths in the rebalancer) already inserted its own rebalance_history
        row, it passes that row id here and the engine updates it in place
        instead of inserting a second 'pending' row. This keeps one history
        row per rebalance so success-rate/fee stats are not double-counted.
        """
        # Stage 2D Defect 3 fix: the v2 engine used to leave ``rebalance_history``
        # empty on automatic cycles, which made ``revenue-status.rebalance_decision.
        # action == 'rebalance'`` reachable alongside ``recent_rebalances == []``.
        # Legacy ``execute_rebalance`` already records to the DB; we mirror that
        # behavior here so the summary and the history agree.
        if rebalance_id is None:
            rebalance_id = self._record_rebalance_pending(pair)
        else:
            rebalance_id = int(rebalance_id)
        reservation_id = (
            str(rebalance_id)
            if rebalance_id is not None
            else f"v2-{time.time_ns()}-{pair.source_channel_id}-{pair.dest_channel_id}"
        )
        reserved_budget = False
        result: Optional[ExecutionResult] = None

        if reserve_budget:
            reserved_budget, budget_result = self._reserve_execution_budget(
                pair,
                reservation_id=reservation_id,
            )
            if budget_result is not None:
                self._record_rebalance_result(
                    rebalance_id,
                    budget_result,
                    pair=pair,
                    account_costs=False,
                )
                return budget_result

        try:
            result = executor.execute(**self._execution_kwargs(pair))
        except Exception as exc:
            result = ExecutionResult(
                success=False,
                amount_sats=int(getattr(pair, "amount_sats", 0) or 0),
                error=f"executor_error: {exc}",
                route_type=self._executor_mode(),
            )

        if result is not None:
            result = self._retry_native_pair_with_exclusions(pair, executor, result)
        if result is not None:
            result = self._retry_native_pair_with_partial_amounts(pair, executor, result)

        self._record_rebalance_result(
            rebalance_id,
            result,
            pair=pair,
            account_costs=account_costs,
        )
        self._finish_execution_budget(
            reservation_id=reservation_id,
            reserved_budget=reserved_budget,
            result=result,
        )
        if result is not None and not result.success:
            self._push_segment_observation_snapshot()
        return result

    def _record_rebalance_pending(self, pair: PairCandidate) -> Optional[int]:
        """Insert a 'pending' row into rebalance_history for this pair.

        Returns the row id, or ``None`` if the DB is unavailable or the insert
        fails. Failure must not prevent execution — the engine still runs even
        if bookkeeping breaks.
        """
        if self.database is None:
            return None
        try:
            return int(
                self.database.record_rebalance(
                    from_channel=pair.source_channel_id,
                    to_channel=pair.dest_channel_id,
                    amount_sats=int(pair.amount_sats),
                    max_fee_sats=int(pair.pair_budget_sats or 0),
                    expected_profit_sats=0,
                    status="pending",
                    rebalance_type="normal",
                    reason_code=pair.reason_code or "ev_positive",
                )
            )
        except Exception as exc:
            self._log(
                f"record_rebalance_pending failed for "
                f"{pair.source_channel_id}->{pair.dest_channel_id}: {exc}",
                level="debug",
            )
            return None

    def _record_rebalance_result(
        self,
        rebalance_id: Optional[int],
        result: Optional[ExecutionResult],
        *,
        pair: Optional[PairCandidate] = None,
        account_costs: bool = False,
    ) -> None:
        """Update the rebalance_history row with the terminal status."""
        if rebalance_id is None or self.database is None:
            return
        try:
            if result is None:
                self.database.update_rebalance_result(
                    rebalance_id,
                    "failed",
                    error_message="executor_returned_none",
                )
                return
            if result.success:
                # Phase 3.3: persist the destination's post-rebalance local
                # ratio anchor. After a successful refill the dest gains the
                # rebalance amount, so we project from the pre-rebalance
                # local_ratio and amount_sats. Capacity is taken from the pair.
                post_local_ratio = None
                if pair is not None:
                    capacity = int(getattr(pair, "dest_capacity_sats", 0) or 0)
                    amount = int(getattr(pair, "amount_sats", 0) or 0)
                    pre_ratio = float(getattr(pair, "dest_local_ratio", 0.0) or 0.0)
                    if capacity > 0:
                        post_local_ratio = max(
                            0.0, min(1.0, pre_ratio + amount / capacity)
                        )
                self.database.update_rebalance_result(
                    rebalance_id,
                    "success",
                    actual_fee_sats=int(getattr(result, "fee_sats", 0) or 0),
                    actual_fee_msat=int(getattr(result, "fee_msat", 0) or 0),
                    post_local_ratio=post_local_ratio,
                    amount_sats=int(
                        getattr(result, "amount_sats", 0)
                        or getattr(pair, "amount_sats", 0)
                        or 0
                    ),
                )
                if account_costs and pair is not None:
                    fee_msat = int(getattr(result, "fee_msat", 0) or 0)
                    if fee_msat <= 0:
                        fee_msat = int(getattr(result, "fee_sats", 0) or 0) * 1000
                    if fee_msat > 0:
                        self.database.record_rebalance_cost(
                            channel_id=pair.dest_channel_id,
                            peer_id=pair.dest_peer_id,
                            cost_sats=base_to_sats_ceil(fee_msat),
                            cost_msat=fee_msat,
                            amount_sats=int(
                                getattr(result, "amount_sats", 0)
                                or getattr(pair, "amount_sats", 0)
                                or 0
                            ),
                            timestamp=int(time.time()),
                        )
            elif getattr(result, "payment_pending", False):
                # Park the row for the reconciliation sweep. No cost is
                # recorded yet — that happens once listsendpays reports a
                # terminal state for this payment_hash.
                failure_data = getattr(result, "failure_data", {}) or {}
                self.database.update_rebalance_result(
                    rebalance_id,
                    "pending_settlement",
                    error_message=str(getattr(result, "error", "") or ""),
                    payment_hash=str(failure_data.get("payment_hash", "") or ""),
                )
            else:
                self.database.update_rebalance_result(
                    rebalance_id,
                    "failed",
                    actual_fee_sats=int(getattr(result, "fee_sats", 0) or 0),
                    actual_fee_msat=int(getattr(result, "fee_msat", 0) or 0),
                    error_message=str(getattr(result, "error", "") or ""),
                )
        except Exception as exc:
            self._log(
                f"update_rebalance_result failed for id={rebalance_id}: {exc}",
                level="debug",
            )

    def reconcile_pending_settlements(self) -> int:
        """Resolve rebalance payments that previously timed out unresolved.

        Sweeps rebalance_history rows parked as 'pending_settlement' against
        listsendpays. Settled payments get their fee recorded into
        rebalance_costs (the budget source of truth) and the reservation
        marked spent; failed payments release the reservation. Still-pending
        payments are left for the next cycle. Returns the number of rows
        resolved either way.
        """
        if self.database is None:
            return 0
        getter = getattr(self.database, "get_pending_settlement_rebalances", None)
        if getter is None:
            return 0
        try:
            rows = getter() or []
        except Exception as exc:
            self._log(f"pending settlement query failed: {exc}", level="warn")
            return 0
        resolved = 0
        # Build the SCID -> peer map at most once per sweep (lazily, only
        # when a settled row actually needs cost attribution) instead of one
        # listpeerchannels RPC per settled row.
        peer_map_cache: Dict[str, Dict[str, str]] = {}

        def scid_peer_map() -> Dict[str, str]:
            if "map" not in peer_map_cache:
                peer_map_cache["map"] = self._build_scid_peer_map()
            return peer_map_cache["map"]

        for row in rows:
            try:
                if self._reconcile_pending_row(
                    row, scid_peer_map_provider=scid_peer_map
                ):
                    resolved += 1
            except Exception as exc:
                self._log(
                    f"pending settlement reconcile failed for id={row.get('id')}: {exc}",
                    level="warn",
                )
        return resolved

    def _reconcile_pending_row(
        self,
        row: Dict[str, Any],
        scid_peer_map_provider: Optional[Any] = None,
    ) -> bool:
        rebalance_id = int(row.get("id") or 0)
        payment_hash = str(row.get("payment_hash") or "")
        if rebalance_id <= 0 or not payment_hash:
            return False
        response = self.plugin.rpc.call("listsendpays", {"payment_hash": payment_hash})
        payments = response.get("payments", []) if isinstance(response, dict) else []
        payments = [p for p in payments if isinstance(p, dict)]
        statuses = {str(p.get("status") or "") for p in payments}

        if "pending" in statuses:
            return False

        reservation_id = str(rebalance_id)
        if "complete" in statuses:
            settled = next(p for p in payments if str(p.get("status")) == "complete")
            amount_msat = parse_msat(settled.get("amount_msat"))
            sent_msat = parse_msat(settled.get("amount_sent_msat"))
            fee_msat = max(0, sent_msat - amount_msat)
            fee_sats = base_to_sats_ceil(fee_msat)
            dest_channel = str(row.get("to_channel") or "")
            self.database.update_rebalance_result(
                rebalance_id,
                "success",
                actual_fee_sats=fee_sats,
                actual_fee_msat=fee_msat,
            )
            if fee_msat > 0:
                self.database.record_rebalance_cost(
                    channel_id=dest_channel,
                    peer_id=self._peer_id_for_channel(
                        dest_channel, scid_peer_map_provider
                    ),
                    cost_sats=fee_sats,
                    cost_msat=fee_msat,
                    amount_sats=int(row.get("amount_sats") or 0),
                    timestamp=int(time.time()),
                )
            self.database.mark_budget_spent(reservation_id, fee_sats)
            self._record_pair_success(
                str(row.get("from_channel") or ""), dest_channel
            )
            self._log(
                f"late settlement confirmed for rebalance {rebalance_id}: "
                f"fee {fee_sats} sats recorded",
            )
            return True

        # No pending and no complete: the payment failed or never existed.
        self.database.update_rebalance_result(
            rebalance_id,
            "failed",
            error_message="payment_pending_resolved_failed",
        )
        self.database.release_budget_reservation(reservation_id)
        try:
            self.plugin.rpc.call(
                "delpay", {"payment_hash": payment_hash, "status": "failed"}
            )
        except Exception:
            pass
        return True

    def _build_scid_peer_map(self) -> Dict[str, str]:
        """Build a SCID -> peer id map from a single listpeerchannels call."""
        mapping: Dict[str, str] = {}
        try:
            if self._data_service is not None:
                channels = self._data_service.get_peer_channels().get("channels", [])
            else:
                channels = self.plugin.rpc.listpeerchannels().get("channels", [])
            for channel in channels:
                if not isinstance(channel, dict):
                    continue
                scid = str(
                    channel.get("short_channel_id")
                    or (channel.get("alias") or {}).get("local", "")
                    or ""
                )
                if scid and scid not in mapping:
                    mapping[scid] = str(channel.get("peer_id") or "")
        except Exception:
            pass
        return mapping

    def _peer_id_for_channel(
        self,
        channel_id: str,
        scid_peer_map_provider: Optional[Any] = None,
    ) -> str:
        """Best-effort SCID -> peer id lookup for cost attribution.

        ``scid_peer_map_provider``: optional zero-arg callable returning a
        prebuilt SCID -> peer map (the reconcile sweep shares one map across
        all rows). When absent, a fresh single-call map is built — same RPC
        cost as the old per-call scan.
        """
        try:
            if callable(scid_peer_map_provider):
                mapping = scid_peer_map_provider()
            else:
                mapping = self._build_scid_peer_map()
            return str(mapping.get(channel_id, "") or "")
        except Exception:
            return ""

    def _segment_observation_direction(self, peer_id: str) -> int:
        """Return the directional edge for inbound-to-local liquidity on dest channel."""
        our_id = self._get_our_id() or ""
        if not our_id or not peer_id:
            return 0
        local_direction = 0 if our_id < peer_id else 1
        return 1 - local_direction

    def _push_segment_observation_snapshot(self) -> bool:
        store = self._segment_observation_store
        observer_member_id = self._get_our_id() or ""
        if store is None or not observer_member_id:
            return False

        snapshot = store.export_snapshot(observer_member_id=observer_member_id)
        if not snapshot.get("segment_observations"):
            return False

        if self._data_service is not None and hasattr(self._data_service, "datastore_push"):
            return bool(
                self._data_service.datastore_push(
                    SegmentObservationStore.DATASTORE_KEY,
                    snapshot,
                )
            )

        try:
            self.plugin.rpc.datastore(
                key=SegmentObservationStore.DATASTORE_KEY,
                string=json.dumps(snapshot),
                mode="create-or-replace",
            )
            return True
        except Exception as exc:
            self._log(f"segment observation export failed: {exc}", level="debug")
            return False

    def execute_candidate(
        self, candidate: Any, rebalance_id: Optional[int] = None
    ) -> ExecutionResult:
        """Price and execute one explicit candidate on the v2 stack.

        Single-flight: shares the engine cycle lock with run_cycle(). The
        acquire is non-blocking by design — if a cycle (or another explicit
        execution) is in flight, this returns a failed ExecutionResult with
        error='engine_busy' instead of waiting, so manual RPC callers get an
        immediate, retriable answer rather than blocking behind a full
        background cycle.

        ``rebalance_id``: optional existing rebalance_history row id owned by
        the caller; see _execute_pair.
        """
        if not self._cycle_lock.acquire(blocking=False):
            self._log(
                "engine_busy: another rebalance cycle/execution holds the "
                "engine lock; rejecting explicit candidate",
                level="info",
            )
            return ExecutionResult(
                success=False,
                error="engine_busy",
                amount_sats=int(getattr(candidate, "amount_sats", 0) or 0),
                route_type=self._executor_mode(),
            )
        try:
            return self._execute_candidate_locked(
                candidate, rebalance_id=rebalance_id
            )
        finally:
            self._cycle_lock.release()

    def _execute_candidate_locked(
        self, candidate: Any, rebalance_id: Optional[int] = None
    ) -> ExecutionResult:
        source_channel_id = str(getattr(candidate, "from_channel", "") or "")
        dest_channel_id = str(getattr(candidate, "to_channel", "") or "")
        source_peer_id = str(getattr(candidate, "from_peer_id", "") or "")
        dest_peer_id = str(getattr(candidate, "to_peer_id", "") or "")
        amount_sats = int(getattr(candidate, "amount_sats", 0) or 0)
        max_fee_sats = int(getattr(candidate, "max_budget_sats", 0) or 0)

        if not source_channel_id or not dest_channel_id:
            return ExecutionResult(success=False, error="invalid_channel_ids")
        if not source_peer_id or not dest_peer_id:
            return ExecutionResult(success=False, error="missing_peer_ids")
        if amount_sats <= 0:
            return ExecutionResult(success=False, error="invalid_amount")

        self._cycle_router = self._active_router()
        if self._cycle_router is None:
            return ExecutionResult(
                success=False,
                error="router_unavailable",
                amount_sats=amount_sats,
                route_type="native",
            )
        pair = PairCandidate(
            source_channel_id=source_channel_id,
            dest_channel_id=dest_channel_id,
            source_peer_id=source_peer_id,
            dest_peer_id=dest_peer_id,
            amount_sats=amount_sats,
            pair_budget_sats=max_fee_sats,
            route_cost_sats=max_fee_sats,
            route=None,
            reason_code=str(getattr(candidate, "reason_code", "") or "manual"),
        )
        pair.route_decision = getattr(candidate, "route_decision", None)

        hive_router = self._hive_router
        router_begin = getattr(self._cycle_router, "begin_cycle", None)
        if callable(router_begin):
            router_begin()
        if hive_router is not None:
            hive_router.begin_cycle()
        try:
            try:
                route_result, route_label = self._route_pair(
                    pair=pair,
                    router=self._cycle_router,
                    exclude=None,
                )
            except Exception as e:
                self._log(
                    f"Manual route pricing failed for {source_channel_id}->{dest_channel_id}: {e}",
                    level="info",
                )
                return ExecutionResult(
                    success=False,
                    error=f"route_pricing_failed: {e}",
                    amount_sats=amount_sats,
                    route_type="native",
                )
            else:
                if route_result.success:
                    pair.route_cost_sats = route_result.route_cost_sats
                    pair.route = route_result.route
                else:
                    self._log(
                        f"Manual route pricing failed for {source_channel_id}->{dest_channel_id}: "
                        f"{route_result.error or 'no_route'} ({route_label})",
                        level="info",
                    )
        finally:
            if hive_router is not None:
                hive_router.end_cycle()
            router_end = getattr(self._cycle_router, "end_cycle", None)
            if callable(router_end):
                router_end()

        decision = self._route_decision_for_pair(pair)
        if route_result is not None and not route_result.success and self._fail_closed_on_route_failure(decision):
            return ExecutionResult(
                success=False,
                error=route_result.error or "hive_route_unavailable",
                amount_sats=amount_sats,
                route_type="native",
            )

        executor = self._make_executor()
        return self._execute_pair(
            pair,
            executor,
            reserve_budget=False,
            account_costs=False,
            rebalance_id=rebalance_id,
        )

    def run_cycle(self) -> CycleResult:
        """Live execution: find candidates (already priced), execute concurrently.

        Single-flight: if another cycle (or an explicit execute_candidate)
        already holds the engine lock, returns immediately with an empty
        CycleResult carrying a 'cycle_already_running' audit marker instead
        of mutating shared cycle state or double-paying the same pairs.

        Filters out pairs in futility state before submitting to the executor,
        and records success/failure in the pair tracker for the next cycle.
        """
        if not self._cycle_lock.acquire(blocking=False):
            self._log(
                "cycle_already_running: another rebalance cycle holds the "
                "engine lock; skipping this cycle",
                level="info",
            )
            return CycleResult(
                audit_records=[
                    SkipRecord(
                        channel_id="",
                        reason="cycle_already_running",
                        value_class="none",
                        detail="engine cycle lock held by another caller",
                    )
                ]
            )
        try:
            return self._run_cycle_locked()
        finally:
            self._cycle_lock.release()

    def _run_cycle_locked(self) -> CycleResult:
        try:
            self.reconcile_pending_settlements()
        except Exception as exc:
            self._log(f"pending settlement sweep failed: {exc}", level="warn")
        candidates = self.find_candidates()
        planned = self._last_cycle_result or CycleResult()
        result = CycleResult(
            considered_candidates=list(getattr(planned, "considered_candidates", []) or []),
            candidates=list(candidates or []),
            audit_records=list(getattr(planned, "audit_records", []) or []),
            snapshot=getattr(planned, "snapshot", None),
            plan=getattr(planned, "plan", None),
        )
        considered_lookup = {
            self._pair_key(pair): pair for pair in result.considered_candidates
        }

        if not candidates:
            self._cache_cycle_result(result)
            return result

        # Pair-level futility filter: skip pairs that failed too many times in
        # the recent window. Emits a pair_futility audit record so the skip is
        # visible in the REBAL_SKIP stream.
        live_candidates: List[PairCandidate] = []
        for pair in candidates:
            if self._is_pair_in_futility(pair.source_channel_id, pair.dest_channel_id):
                debug_pair = considered_lookup.get(self._pair_key(pair))
                if debug_pair is not None:
                    self._update_pair_score_decomposition(
                        debug_pair,
                        route_cost_sats=debug_pair.route_cost_sats,
                        effective_budget_sats=int(getattr(debug_pair, "pair_budget_sats", 0) or 0),
                        rejection_reason="pair_futility",
                        route_status="pair_futility",
                    )
                fresh = self._pair_failures.get(
                    (pair.source_channel_id, pair.dest_channel_id), []
                )
                self._audit.log_skip(
                    channel_id=pair.dest_channel_id,
                    reason="pair_futility",
                    value_class="valuable",
                    remaining_budget_sats=int(getattr(pair, "pair_budget_sats", 0) or 0),
                    detail=(
                        f"src={pair.source_channel_id} "
                        f"failures={len(fresh)} in window {int(self._futility_window_sec)}s"
                    ),
                    router="v3",
                )
                continue
            live_candidates.append(pair)

        result.candidates = list(live_candidates)
        if not live_candidates:
            self._cache_cycle_result(result)
            return result

        executor = self._make_executor()
        if not executor.is_available():
            unavailable_reason, unavailable_detail = self._executor_unavailable_reason()
            for pair in live_candidates:
                debug_pair = considered_lookup.get(self._pair_key(pair))
                if debug_pair is not None:
                    self._update_pair_score_decomposition(
                        debug_pair,
                        route_cost_sats=debug_pair.route_cost_sats,
                        effective_budget_sats=int(getattr(debug_pair, "pair_budget_sats", 0) or 0),
                        rejection_reason=unavailable_reason,
                        route_status=unavailable_reason,
                    )
                self._audit.log_skip(
                    channel_id=pair.dest_channel_id,
                    reason=unavailable_reason,
                    value_class="valuable",
                    remaining_budget_sats=int(getattr(pair, "pair_budget_sats", 0) or 0),
                    detail=unavailable_detail,
                    router="v3",
                )
            result.candidates = []
            self._cache_cycle_result(result)
            return result

        execution_limit = self._max_concurrent_jobs()
        if len(live_candidates) > execution_limit:
            overflow = live_candidates[execution_limit:]
            live_candidates = live_candidates[:execution_limit]
            result.candidates = list(live_candidates)
            for pair in overflow:
                debug_pair = considered_lookup.get(self._pair_key(pair))
                if debug_pair is not None:
                    self._update_pair_score_decomposition(
                        debug_pair,
                        route_cost_sats=debug_pair.route_cost_sats,
                        effective_budget_sats=int(getattr(debug_pair, "pair_budget_sats", 0) or 0),
                        rejection_reason="max_pairs_reached",
                        route_status="max_concurrent_jobs",
                    )
                skip = SkipRecord(
                    channel_id=pair.dest_channel_id,
                    reason="max_pairs_reached",
                    value_class="valuable",
                    remaining_budget_sats=int(getattr(pair, "pair_budget_sats", 0) or 0),
                    detail=(
                        f"src={pair.source_channel_id} "
                        f"max_concurrent_jobs={execution_limit}"
                    ),
                )
                result.audit_records.append(skip)
                self._audit.log_skip(
                    channel_id=skip.channel_id,
                    reason=skip.reason,
                    value_class=skip.value_class,
                    remaining_budget_sats=skip.remaining_budget_sats,
                    detail=skip.detail,
                    router="v3",
                )

        # Submit the surviving candidates to the thread pool
        futures: Dict[Future, PairCandidate] = {}
        for pair in live_candidates:
            future = self._pool.submit(
                self._execute_pair,
                pair,
                executor,
                reserve_budget=True,
                account_costs=True,
            )
            futures[future] = pair

        # Collect results as they complete (main thread only — no lock needed)
        completed_futures: set[Future] = set()

        def consume_future_result(future: Future, pair: PairCandidate) -> None:
            try:
                exec_result = future.result()
                completed_futures.add(future)
                if exec_result is not None:
                    result.executions.append(exec_result)
                    if exec_result.success:
                        self._record_pair_success(
                            pair.source_channel_id, pair.dest_channel_id
                        )
                        self._clear_persisted_pair_failure(pair)
                    else:
                        self._record_pair_failure(
                            pair.source_channel_id, pair.dest_channel_id
                        )
                        self._persist_pair_failure(pair, exec_result)
                else:
                    # _execute_pair returned None (no route stored on the pair) — count as failure
                    self._record_pair_failure(
                        pair.source_channel_id, pair.dest_channel_id
                    )
            except Exception as e:
                completed_futures.add(future)
                self._record_pair_failure(
                    pair.source_channel_id, pair.dest_channel_id
                )
                self._log(
                    f"Execution thread failed for "
                    f"{pair.source_channel_id}->{pair.dest_channel_id}: {e}",
                    level="warn",
                )

        try:
            for future in as_completed(futures, timeout=120):
                pair = futures[future]
                consume_future_result(future, pair)
        except TimeoutError:
            self._log(
                f"Execution timed out after 120s, "
                f"{len(result.executions)}/{len(futures)} completed",
                level="warn",
            )
            for future, pair in futures.items():
                if future in completed_futures:
                    continue
                if future.done():
                    consume_future_result(future, pair)
                    continue
                if future.cancel():
                    timeout_result = ExecutionResult(
                        success=False,
                        error="executor_timeout_cancelled",
                        amount_sats=int(getattr(pair, "amount_sats", 0) or 0),
                        fee_sats=0,
                        fee_msat=0,
                        route_type=self._executor_mode(),
                    )
                    result.executions.append(timeout_result)
                    continue
                self._log(
                    f"Execution still running after cycle timeout for "
                    f"{pair.source_channel_id}->{pair.dest_channel_id}; "
                    "worker will finish bookkeeping asynchronously",
                    level="warn",
                )

        self._cache_cycle_result(result)
        return result
