"""Rebalance engine v2 orchestrator.

Wires: state snapshot → planner → router → executor → audit.
Single entry point: find_candidates() for dry-run, run_cycle() for live.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .rebalance_audit_v2 import RebalanceAudit
from .rebalance_executor_v2 import RebalanceExecutor, ExecutionResult
from .rebalance_planner_v2 import RebalancePlanner
from .rebalance_router_v2 import RebalanceRouter
from .rebalance_state_v2 import StateSnapshot, build_state_snapshot
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
    ):
        self.plugin = plugin
        self.config = config
        self.database = database
        self._capex_engine = capex_engine
        self._profitability = profitability
        self._hive_hints = hive_hints

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
            self._our_id = self.plugin.rpc.getinfo()["id"]
        except Exception:
            pass
        return self._our_id

    def _build_snapshot(self) -> Optional[StateSnapshot]:
        """Build a normalized state snapshot from live data."""
        cfg = self.config if not hasattr(self.config, 'snapshot') else self.config.snapshot()

        try:
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
        """
        snapshot = self._build_snapshot()
        if not snapshot or not snapshot.channels:
            self._log("No channels in snapshot")
            return []

        cfg = self.config if not hasattr(self.config, 'snapshot') else self.config.snapshot()

        planner = RebalancePlanner(
            target_band_low=getattr(cfg, 'low_liquidity_threshold', 0.35),
            target_band_high=getattr(cfg, 'high_liquidity_threshold', 0.65),
            max_chunk_sats=getattr(cfg, 'rebalance_max_amount', 2_000_000),
            max_pairs=10,
        )

        plan = planner.plan(snapshot)

        # Route-price selected pairs
        our_id = self._get_our_id()
        if our_id and plan.selected:
            router = RebalanceRouter(self.plugin, our_id)
            priced = []
            for pair in plan.selected:
                route_result = router.price_pair(
                    source_channel_id=pair.source_channel_id,
                    dest_channel_id=pair.dest_channel_id,
                    source_peer_id=pair.source_peer_id,
                    dest_peer_id=pair.dest_peer_id,
                    amount_sats=pair.amount_sats,
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
                    plan.skipped.append(SkipRecord(
                        channel_id=pair.dest_channel_id,
                        reason="no_route",
                        value_class="valuable",
                        detail=route_result.error,
                    ))

            plan.selected = priced

        # Audit all skips
        for skip in plan.skipped:
            self._audit.log_skip(
                skip.channel_id,
                skip.reason,
                skip.value_class,
                skip.remaining_budget_sats,
                detail=skip.detail or "",
            )

        self._audit.log_cycle_summary(
            selected_count=len(plan.selected),
            skipped_count=len(plan.skipped),
            total_valuable=snapshot.valuable_channel_count,
            total_channels=len(snapshot.channels),
            total_budget_sats=snapshot.total_remaining_budget_sats,
        )

        return plan.selected

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

    def _probability_adjusted_budget(
        self, pair_budget_sats: int, probability_ppm: int
    ) -> int:
        """Relax the raw pair budget by a probability-weighted bonus.

        Returns the base pair_budget_sats unchanged when either the config
        bonus rate is 0 (default) or the router reported no probability
        (v2/getroute does this). With a positive bonus rate and non-zero
        probability, the effective budget is:

            pair_budget * (1 + clamp(probability_ppm, 0, 1_000_000) / 1_000_000 * bonus)

        Example: bonus=0.25, probability_ppm=982_339 →
            effective = pair_budget * (1 + 0.982339 * 0.25)
                      = pair_budget * 1.2456

        The intent is to unlock v3/askrene's high-probability-but-pricier
        routes on topologies where v2/getroute's cheap paths are actually
        unroutable (see Phase B test on nexus-01, 2026-04-10).
        """
        bonus_rate = getattr(self.config, "capex_probability_budget_bonus", 0.0)
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

        On retriable failure with an erring channel, re-prices the pair
        with the failing channel excluded and attempts one retry. Covers
        the common stale-gossip WIRE_FEE_INSUFFICIENT case where the
        first attempt picks an intermediate whose real fee is higher
        than gossip shows — the retry picks a different path that
        bypasses the failing hop.

        Only one retry per call, by design: unbounded retry would cascade
        exclude layers and waste budget on repeatedly-failing topology.
        """
        if not pair.route:
            self._log(
                f"No route stored for {pair.source_channel_id}->"
                f"{pair.dest_channel_id}, skipping",
                level="info",
            )
            return None

        exec_result = executor.execute(
            route=pair.route,
            amount_sats=pair.amount_sats,
            source_channel_id=pair.source_channel_id,
            dest_channel_id=pair.dest_channel_id,
            max_fee_sats=pair.pair_budget_sats,
        )

        if exec_result.success:
            self._log(
                f"Rebalanced {pair.amount_sats} sats "
                f"{pair.source_channel_id}->{pair.dest_channel_id} "
                f"fee={exec_result.fee_sats} sats"
            )
            return exec_result

        # Retry on retriable failures that identified a specific bad hop.
        # Permanent failures (channel disabled, unknown peer) can't be
        # usefully retried with an exclude — the pair is dead for this cycle.
        if not self._should_retry_with_exclude(exec_result):
            return exec_result

        retry_result = self._attempt_retry_with_exclude(pair, executor, exec_result)
        if retry_result is not None:
            return retry_result
        return exec_result

    @staticmethod
    def _should_retry_with_exclude(exec_result: "ExecutionResult") -> bool:
        """Return True iff the executor failure is retriable AND it identified
        at least one excluded channel we can try to route around."""
        error = exec_result.error or ""
        if "retriable_failure" not in error:
            return False
        return bool(exec_result.excluded_channels)

    def _attempt_retry_with_exclude(
        self,
        pair: PairCandidate,
        executor: RebalanceExecutor,
        original_failure: "ExecutionResult",
    ) -> Optional["ExecutionResult"]:
        """Re-price the pair with the failing channel excluded, execute once.

        Returns the retry ExecutionResult on actual retry (success or fail),
        or None if the retry was abandoned (no route / over budget / no
        router available). Caller treats None as "stick with original_failure".
        """
        router = getattr(self, "_cycle_router", None)
        if router is None:
            return None

        self._log(
            f"Retrying {pair.source_channel_id}->{pair.dest_channel_id} with "
            f"exclude={original_failure.excluded_channels}",
            level="info",
        )
        try:
            new_route = router.price_pair(
                source_channel_id=pair.source_channel_id,
                dest_channel_id=pair.dest_channel_id,
                source_peer_id=pair.source_peer_id,
                dest_peer_id=pair.dest_peer_id,
                amount_sats=pair.amount_sats,
                exclude=list(original_failure.excluded_channels),
            )
        except Exception as e:
            self._log(
                f"Retry re-price raised {type(e).__name__}: {e}", level="warn"
            )
            return None

        if not new_route.success:
            self._log(
                f"Retry re-price returned no route: {new_route.error}",
                level="info",
            )
            return None

        effective_budget = self._probability_adjusted_budget(
            pair.pair_budget_sats,
            getattr(new_route, "probability_ppm", 0),
        )
        if new_route.route_cost_sats > effective_budget:
            self._log(
                f"Retry re-price over budget: route_cost={new_route.route_cost_sats} "
                f"effective_budget={effective_budget}",
                level="info",
            )
            return None

        pair.route = new_route.route
        pair.route_cost_sats = new_route.route_cost_sats

        retry_result = executor.execute(
            route=pair.route,
            amount_sats=pair.amount_sats,
            source_channel_id=pair.source_channel_id,
            dest_channel_id=pair.dest_channel_id,
            max_fee_sats=pair.pair_budget_sats,
        )
        if retry_result.success:
            self._log(
                f"Retry succeeded: {pair.source_channel_id}->"
                f"{pair.dest_channel_id} fee={retry_result.fee_sats} sats"
            )
        return retry_result

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

        executor = RebalanceExecutor(self.plugin, self.database)

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
                        else:
                            self._record_pair_failure(
                                pair.source_channel_id, pair.dest_channel_id
                            )
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
