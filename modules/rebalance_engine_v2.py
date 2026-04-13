"""Rebalance engine v2 orchestrator.

Wires: state snapshot → planner → router → executor → audit.
Single entry point: find_candidates() for dry-run, run_cycle() for live.
"""

from __future__ import annotations

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
from .rebalance_route_policy import RoutePolicy, RoutePriority, decide_route_policy
from .rebalance_router_v2 import RebalanceRouter, RouteResult
from .rebalance_router_v3 import RebalanceRouterV3, _parse_layer_names
from .rebalance_state_v2 import StateSnapshot, build_state_snapshot
from .sling_segment_observations import SlingSegmentObservationStore
from .rebalance_types_v2 import PairCandidate, PlanResult, SkipRecord


@dataclass
class CycleResult:
    """Full result of a v2 rebalance cycle."""

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
        self._legacy_hive_router = hive_router
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

        # Build v2 router unconditionally — no RPC dependency at construction.
        our_id = self._get_our_id() or ""
        self.router_v2 = RebalanceRouter(
            plugin, our_id, data_service=self._data_service
        )
        if self._hive_router is None and self._hive_hints is not None:
            self._hive_router = RebalanceHiveRouter(
                plugin=plugin,
                our_node_id=our_id,
                hive_hints=self._hive_hints,
                data_service=self._data_service,
                log=self._log,
            )

        # Build v3 router iff askrene is available. Missing askrene means
        # standalone fallback: the active router stays on v2 regardless of
        # config.rebalance_router.
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

        self._cycle_router: Any = self._active_router()

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

    def _active_router(self) -> Any:
        """Return the router currently configured for dispatch.

        Reads config.rebalance_router each call so setconfig can hot-switch
        between cycles. Falls back to v2 if v3 is requested but unavailable.
        """
        want = getattr(self.config, "rebalance_router", "v2")
        if want == "v3" and self.router_v3 is not None:
            return self.router_v3
        return self.router_v2

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
            if self._hive_hints:
                try:
                    is_hive = self._hive_hints.is_hive_member(peer_id)
                except Exception:
                    pass

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
            })

        return build_state_snapshot(normalized, capex_allocations)

    def find_candidates(self) -> List[PairCandidate]:
        """Dry-run: build snapshot, plan, and return candidates without executing.

        This is the entry point called by EVRebalancer when rebalance_engine=v2.
        Captures the active router at the start of the cycle so mid-cycle
        config flips do not split a cycle across two routers.
        """
        self._cycle_router = self._active_router()
        router_tag = "v3" if self._cycle_router is self.router_v3 else "v2"

        snapshot = self._build_snapshot()
        if not snapshot or not snapshot.channels:
            self._log("No channels in snapshot")
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
        for pair in plan.selected:
            self._route_decision_for_pair(pair)
            self._apply_segment_score_bias(pair)
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

        # Route-price selected pairs using the cycle's captured router
        if plan.selected:
            router = self._cycle_router
            priced = []
            for pair in plan.selected:
                cooldown = self._get_persisted_pair_cooldown(
                    pair.source_channel_id, pair.dest_channel_id
                )
                if cooldown is not None:
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
                    self._audit.log_skip(
                        pair.dest_channel_id,
                        reason="no_route",
                        value_class="valuable",
                        remaining_budget_sats=pair.pair_budget_sats,
                        detail=route_result.error,
                        router=route_label,
                    )
                    decision = self._route_decision_for_pair(pair)
                    strict_hive_only = (
                        decision.policy is RoutePolicy.HIVE_ONLY
                        and not decision.allow_market_fallback
                    )
                    if strict_hive_only:
                        continue
                    pair.route = None
                    pair.route_cost_sats = pair.pair_budget_sats
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

    def _route_pair(
        self,
        *,
        pair: PairCandidate,
        router: Any,
        exclude: Optional[List[str]],
    ):
        decision = self._route_decision_for_pair(pair)
        hive_router = self._hive_router if hasattr(self._hive_router, "price_pair") else None
        if decision.policy is RoutePolicy.HIVE_ONLY:
            if hive_router is None:
                if decision.allow_market_fallback:
                    return self._market_price_pair(router, pair, exclude), (
                        "v3" if router is self.router_v3 else "v2"
                    )
                return self._route_error("hive_router_unavailable"), "hive"
            hive_result = hive_router.price_pair(pair, decision, exclude=exclude)
            if getattr(hive_result, "success", False) or not decision.allow_market_fallback:
                return hive_result, "hive"
            market_result = self._market_price_pair(router, pair, exclude)
            if getattr(market_result, "success", False):
                return market_result, "market"
            return hive_result, "hive"
        if decision.policy is RoutePolicy.HYBRID and hive_router is not None:
            hive_result = hive_router.price_pair(pair, decision, exclude=exclude)
            market_result = self._market_price_pair(router, pair, exclude)
            return self._hybrid_choice(hive_result, market_result, decision)
        return self._market_price_pair(router, pair, exclude), (
            "v3" if router is self.router_v3 else "v2"
        )

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
        router_kind = "v3" if self._cycle_router is self.router_v3 else "v2"
        decision = self._route_decision_for_pair(pair)
        result = executor.execute(
            route=pair.route or [],
            amount_sats=pair.amount_sats,
            source_channel_id=pair.source_channel_id,
            dest_channel_id=pair.dest_channel_id,
            max_fee_sats=pair.pair_budget_sats,
            observation_store=self._segment_observation_store,
            observation_context={
                "short_channel_id": pair.dest_channel_id,
                "direction": self._segment_observation_direction(pair.dest_peer_id),
                "source_channel_id": pair.source_channel_id,
                "dest_channel_id": pair.dest_channel_id,
                "route_policy": getattr(decision.policy, "value", str(decision.policy)),
                "router_kind": router_kind,
                "correlation_id": (
                    f"{pair.source_channel_id}->{pair.dest_channel_id}:{int(time.time())}"
                ),
            },
        )
        if result is not None and not result.success:
            self._push_segment_observation_snapshot()
        return result

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
            route_result = None
            route_label = "v3" if self._cycle_router is self.router_v3 else "v2"
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
        strict_hive_only = (
            decision.policy is RoutePolicy.HIVE_ONLY and not decision.allow_market_fallback
        )
        if route_result is not None and not route_result.success and strict_hive_only:
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
        result = CycleResult()

        candidates = self.find_candidates()
        result.candidates = candidates

        if not candidates:
            return result

        # Pair-level futility filter: skip pairs that failed too many times in
        # the recent window. Emits a pair_futility audit record so the skip is
        # visible in the REBAL_SKIP stream.
        live_candidates: List[PairCandidate] = []
        for pair in candidates:
            if self._is_pair_in_futility(pair.source_channel_id, pair.dest_channel_id):
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

        if not live_candidates:
            return result

        executor = RebalanceExecutor(
            self.plugin,
            self.database,
            observation_store=self._segment_observation_store,
        )
        if not executor.is_available():
            for pair in live_candidates:
                self._audit.log_skip(
                    channel_id=pair.dest_channel_id,
                    reason="sling_unavailable",
                    value_class="valuable",
                    detail="sling-once RPC not loaded",
                )
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

        return result
