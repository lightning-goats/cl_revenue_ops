"""Rebalance engine v2 orchestrator.

Wires: state snapshot → planner → router → executor → audit.
Single entry point: find_candidates() for dry-run, run_cycle() for live.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
        self._lock = threading.Lock()

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
                    prof = self._profitability.get_channel_profitability(scid)
                    if prof:
                        contribution = getattr(prof, 'revenue', None)
                        if contribution:
                            is_profitable = getattr(contribution, 'total_contribution_msat', 0) > 0
                        is_active = getattr(prof, 'total_forward_count', 0) > 5
                except Exception:
                    pass

            # Cooldown: skip if rebalanced recently (default 1 hour)
            cooldown = False
            cooldown_secs = getattr(cfg, 'rebalance_cooldown_secs', 3600)
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
                    if route_result.route_cost_sats <= pair.pair_budget_sats:
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
                            detail=f"route_cost={route_result.route_cost_sats}",
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

    def _execute_pair(
        self,
        pair: PairCandidate,
        executor: RebalanceExecutor,
    ) -> Optional[ExecutionResult]:
        """Execute a single pair in a worker thread."""
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

    def run_cycle(self) -> CycleResult:
        """Live execution: find candidates (already priced), execute concurrently."""
        result = CycleResult()

        candidates = self.find_candidates()
        result.candidates = candidates

        if not candidates:
            return result

        executor = RebalanceExecutor(self.plugin, self.database)

        # Submit all candidates to the thread pool
        futures: Dict[Future, PairCandidate] = {}
        for pair in candidates:
            future = self._pool.submit(self._execute_pair, pair, executor)
            futures[future] = pair

        # Collect results as they complete
        for future in as_completed(futures, timeout=120):
            pair = futures[future]
            try:
                exec_result = future.result()
                if exec_result is not None:
                    with self._lock:
                        result.executions.append(exec_result)
            except Exception as e:
                self._log(
                    f"Execution thread failed for "
                    f"{pair.source_channel_id}->{pair.dest_channel_id}: {e}",
                    level="warn",
                )

        return result
