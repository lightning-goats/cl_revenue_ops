"""Rebalance engine v2 orchestrator.

Wires: state snapshot → planner → router → executor → audit.
Single entry point: find_candidates() for dry-run, run_cycle() for live.
"""

from __future__ import annotations

import copy
import json
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
from .rebalance_executor_v2 import RebalanceExecutor, ExecutionResult
from .rebalance_hive_router import RebalanceHiveRouter
from .rebalance_planner_v2 import RebalancePlanner
from .rebalance_route_policy import (
    RouteDecision,
    RoutePolicy,
    RoutePriority,
    decide_route_policy,
)
from .rebalance_router_v2 import RouteResult
from .rebalance_router_v3 import RebalanceRouterV3, _parse_layer_names
from .rebalance_state_v2 import StateSnapshot, build_state_snapshot
from .sling_segment_observations import SlingSegmentObservationStore
from .rebalance_types_v2 import PairCandidate, PlanResult, SkipRecord


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

        self._our_id: Optional[str] = None
        self._audit = RebalanceAudit(plugin)
        self._pool = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="rebal-v2"
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
        self._pair_failure_cooldowns: Dict[str, int] = {
            "temporary_channel_failure": 1800,
            "fee_insufficient": 7200,
            "incorrect_cltv_expiry": 7200,
            "permanent_failure": 21600,
            "payment_pending_timeout": 900,
            "local_execution_failed": 1800,
            "other_retriable": 1800,
        }

        our_id = self._get_our_id() or ""

        # Build v3 router iff askrene is available. Missing askrene now means
        # fail-closed: legacy getroute routing has been removed.
        self.router_v3: Optional[RebalanceRouterV3] = None
        if self._probe_askrene():
            layer_names = _parse_layer_names(
                getattr(config, "askrene_layers", "")
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
        elif route_status == "fallback_unpriced":
            p_success = 0.25
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
            "reason_code": pair.reason_code,
            "route_policy": getattr(
                getattr(getattr(pair, "route_decision", None), "policy", None),
                "value",
                None,
            ),
            "rejection_reason": pair.rejection_reason or None,
            "score_decomposition": copy.deepcopy(pair.score_decomposition or {}),
        }

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
            cooldown_hours = getattr(cfg, 'rebalance_cooldown_hours', 24)
            cooldown_secs = int(cooldown_hours * 3600)
            if self.database and cooldown_secs > 0:
                try:
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
                drift_threshold = float(
                    getattr(cfg, "rebalance_drift_override_ratio", 0.30) or 0.0
                )
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
                "cooldown_active": cooldown,
                "cooldown_override": cooldown_override,
            })

        target_band_low = float(getattr(cfg, "low_liquidity_threshold", 0.35) or 0.35)
        target_band_high = float(getattr(cfg, "high_liquidity_threshold", 0.65) or 0.65)
        target_emergency_low = float(
            getattr(cfg, "rebalance_emergency_local_ratio", 0.10) or 0.0
        )
        return build_state_snapshot(
            normalized,
            capex_allocations,
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            target_emergency_low=target_emergency_low,
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

        planner = RebalancePlanner(
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            max_chunk_sats=max_chunk_sats,
            max_pairs=10,
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
        plan.selected = merge_coordination_pairs(
            plan,
            overlay.selected,
            max_pairs=planner_max_pairs,
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
                    self._audit.log_skip(
                        pair.dest_channel_id,
                        reason="no_route",
                        value_class="valuable",
                        remaining_budget_sats=pair.pair_budget_sats,
                        detail=route_result.error,
                        router=route_label,
                    )
                    decision = self._route_decision_for_pair(pair)
                    if self._fail_closed_on_route_failure(decision):
                        continue
                    pair.route = None
                    pair.route_cost_sats = pair.pair_budget_sats
                    self._update_pair_score_decomposition(
                        pair,
                        route_cost_sats=pair.pair_budget_sats,
                        effective_budget_sats=pair.pair_budget_sats,
                        route_status="fallback_unpriced",
                    )
                    if debug_pair is not None:
                        debug_pair.route = None
                        debug_pair.route_cost_sats = pair.pair_budget_sats
                        self._update_pair_score_decomposition(
                            debug_pair,
                            route_cost_sats=pair.pair_budget_sats,
                            effective_budget_sats=pair.pair_budget_sats,
                            route_status="fallback_unpriced",
                        )
                    priced.append(pair)

            plan.selected = priced

        # Audit all skips
        for skip in plan.skipped:
            self._audit.log_skip(
                skip.channel_id,
                skip.reason,
                skip.value_class,
                skip.remaining_budget_sats,
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

    @staticmethod
    def _market_price_pair(router: Any, pair: PairCandidate, exclude: Optional[List[str]]):
        return router.price_pair(
            source_channel_id=pair.source_channel_id,
            dest_channel_id=pair.dest_channel_id,
            source_peer_id=pair.source_peer_id,
            dest_peer_id=pair.dest_peer_id,
            amount_sats=pair.amount_sats,
            exclude=exclude,
        )

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
            return self._market_price_pair(router, pair, exclude), "market"

        hive_router = self._hive_router
        if hive_router is None:
            if self._fail_closed_on_route_failure(decision):
                return self._route_error("hive_router_unavailable"), "hive"
            return self._market_price_pair(router, pair, exclude), "market"

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
            return self._market_price_pair(router, pair, exclude), "market"

        market_result = self._market_price_pair(router, pair, exclude)
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

    def _execution_kwargs(self, pair: PairCandidate) -> Dict[str, Any]:
        router_kind = "v3" if self._cycle_router is self.router_v3 else "v2"
        decision = self._route_decision_for_pair(pair)
        return {
            "route": pair.route or [],
            "amount_sats": pair.amount_sats,
            "source_channel_id": pair.source_channel_id,
            "dest_channel_id": pair.dest_channel_id,
            "max_fee_sats": pair.pair_budget_sats,
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

    def _execute_pair(
        self,
        pair: PairCandidate,
        executor: RebalanceExecutor,
    ) -> Optional[ExecutionResult]:
        """Execute a single pair in a worker thread.

        Sling-backed execution does not require a stored local route snapshot.
        If one is present it is ignored by the executor; if it is absent we
        still execute the selected pair directly.
        """
        # Stage 2D Defect 3 fix: the v2 engine used to leave ``rebalance_history``
        # empty on automatic cycles, which made ``revenue-status.rebalance_decision.
        # action == 'rebalance'`` reachable alongside ``recent_rebalances == []``.
        # Legacy ``execute_rebalance`` already records to the DB; we mirror that
        # behavior here so the summary and the history agree.
        rebalance_id: Optional[int] = self._record_rebalance_pending(pair)
        result = executor.execute(**self._execution_kwargs(pair))
        self._record_rebalance_result(rebalance_id, result, pair=pair)
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
                    SlingSegmentObservationStore.DATASTORE_KEY,
                    snapshot,
                )
            )

        try:
            self.plugin.rpc.datastore(
                key=SlingSegmentObservationStore.DATASTORE_KEY,
                string=json.dumps(snapshot),
                mode="create-or-replace",
            )
            return True
        except Exception as exc:
            self._log(f"segment observation export failed: {exc}", level="debug")
            return False

    def execute_candidate(self, candidate: Any) -> ExecutionResult:
        """Price and execute one explicit candidate on the v2 stack."""
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
                route_type="sling",
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
                route_type="sling",
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

        decision = self._route_decision_for_pair(pair)
        if route_result is not None and not route_result.success and self._fail_closed_on_route_failure(decision):
            return ExecutionResult(
                success=False,
                error=route_result.error or "hive_route_unavailable",
                amount_sats=amount_sats,
                route_type="sling",
            )

        executor = RebalanceExecutor(
            self.plugin,
            self.database,
            observation_store=self._segment_observation_store,
        )
        return self._execute_pair(pair, executor)

    def run_cycle(self) -> CycleResult:
        """Live execution: find candidates (already priced), execute concurrently.

        Filters out pairs in futility state before submitting to the executor,
        and records success/failure in the pair tracker for the next cycle.
        """
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
                    detail=(
                        f"src={pair.source_channel_id} "
                        f"failures={len(fresh)} in window {int(self._futility_window_sec)}s"
                    ),
                )
                continue
            live_candidates.append(pair)

        result.candidates = list(live_candidates)
        if not live_candidates:
            self._cache_cycle_result(result)
            return result

        executor = RebalanceExecutor(
            self.plugin,
            self.database,
            observation_store=self._segment_observation_store,
        )
        if not executor.is_available():
            for pair in live_candidates:
                debug_pair = considered_lookup.get(self._pair_key(pair))
                if debug_pair is not None:
                    self._update_pair_score_decomposition(
                        debug_pair,
                        route_cost_sats=debug_pair.route_cost_sats,
                        effective_budget_sats=int(getattr(debug_pair, "pair_budget_sats", 0) or 0),
                        rejection_reason="sling_unavailable",
                        route_status="sling_unavailable",
                    )
                self._audit.log_skip(
                    channel_id=pair.dest_channel_id,
                    reason="sling_unavailable",
                    value_class="valuable",
                    detail="sling-once RPC not loaded",
                )
            result.candidates = []
            self._cache_cycle_result(result)
            return result

        # Submit the surviving candidates to the thread pool
        futures: Dict[Future, PairCandidate] = {}
        for pair in live_candidates:
            future = self._pool.submit(self._execute_pair, pair, executor)
            futures[future] = pair

        # Collect results as they complete (main thread only — no lock needed)
        try:
            for future in as_completed(futures, timeout=120):
                pair = futures[future]
                try:
                    exec_result = future.result()
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
                    self._record_pair_failure(
                        pair.source_channel_id, pair.dest_channel_id
                    )
                    self._log(
                        f"Execution thread failed for "
                        f"{pair.source_channel_id}->{pair.dest_channel_id}: {e}",
                        level="warn",
                    )
        except TimeoutError:
            self._log(
                f"Execution timed out after 120s, "
                f"{len(result.executions)}/{len(futures)} completed",
                level="warn",
            )

        self._cache_cycle_result(result)
        return result
