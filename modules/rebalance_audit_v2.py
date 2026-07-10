"""Structured audit logging for the v2 rebalance engine.

Every cycle must explain itself. For every valuable channel the engine
evaluates, one of the ``log_*`` helpers emits a deterministic, grep-friendly
record that replaces the vague ad-hoc messages of the v1 engine.

Canonical skip reasons live in ``VALID_SKIP_REASONS``. Reasons produced by
the v3 askrene-based router are included so audit consumers grouped by
``reason=`` can bucket both router versions identically.
"""

from __future__ import annotations

from typing import Any, Optional


# Skip reasons that describe channels with nothing to do (healthy or not
# worth refilling). They dominate the per-cycle skip volume (~96 log lines
# per cycle, ~9.2k/day at scale) while carrying no operator action, so the
# log emission is aggregated to one summary line per reason. The full
# per-channel SkipRecords are NOT affected: the cycle result / debug
# surface keeps every record (the audit contract is unchanged).
NON_ACTIONABLE_SKIP_REASONS: frozenset[str] = frozenset({
    "inside_band",
    "not_valuable",
})


VALID_SKIP_REASONS: frozenset[str] = frozenset({
    # Produced by the planner
    "inside_band",
    "not_valuable",
    "no_partner",
    "cooldown",
    "no_budget",
    "max_pairs_reached",
    "outcompeted",
    # Planner eligibility fallbacks (RA2-1) when the state layer supplies
    # no specific source/dest reason
    "source_ineligible",
    "dest_ineligible",
    # Produced by the router / engine (both v2 and v3)
    "no_route",
    "route_over_budget",
    # Produced by the engine's pair-level futility tracker
    "pair_futility",
    # Produced by the engine when the native route executor's RPC surface is
    # unavailable (rebalance_engine_v2._executor_unavailable_reason). DEF-077.
    "native_unavailable",
    # Produced by the engine's gates and cycle lock (RA2-1)
    "pair_cooldown",
    "below_hold_margin",
    "cycle_already_running",
    # P4-008 in-flight-destination guard: a prior cycle's still-running
    # (possibly orphaned) worker holds an unresolved payment to this dest.
    "dest_inflight",
    # Produced by the coordination overlay (RA2-1)
    "not_designated_executor",
    "coordination_unavailable",
    "coordination_unresolvable_endpoint",
    "coordination_preempted",
    "lease_conflict",
    # Produced by the v3 askrene router specifically
    "unknown_source_node",
    "unknown_dest_node",
    "unknown_layer",
    "askrene_child_died",
    "path_loops_through_us",
})


class RebalanceAudit:
    """Structured audit logger for the v2 rebalance engine."""

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin

    # ------------------------------------------------------------------
    # Pure formatters (no side-effects, easy to unit-test)
    # ------------------------------------------------------------------

    @staticmethod
    def format_pick(
        source_channel_id: str,
        dest_channel_id: str,
        amount_sats: int,
        route_cost_sats: int,
        value_score: float,
        router: str = "v2",
    ) -> str:
        return (
            f"REBAL_PICK source={source_channel_id} dest={dest_channel_id} "
            f"amount={amount_sats} route_cost_sats={route_cost_sats} "
            f"value_score={value_score} router={router}"
        )

    @staticmethod
    def format_skip(
        channel_id: str,
        reason: str,
        value_class: str,
        remaining_budget_sats: int = 0,
        route_cost_sats: int = 0,
        detail: str = "",
        router: str = "v2",
    ) -> str:
        parts = [
            f"REBAL_SKIP channel={channel_id}",
            f"reason={reason}",
            f"value_class={value_class}",
            f"budget={remaining_budget_sats}",
            f"router={router}",
        ]
        if route_cost_sats:
            parts.append(f"route_cost={route_cost_sats}")
        if detail:
            parts.append(f"detail={detail}")
        return " ".join(parts)

    @staticmethod
    def format_skip_summary(reason: str, count: int, router: str = "v2") -> str:
        return f"REBAL_SKIP reason={reason} count={count} router={router}"

    @staticmethod
    def format_cycle_summary(
        selected_count: int,
        skipped_count: int,
        total_valuable: int,
        total_channels: int,
        total_budget_sats: int,
    ) -> str:
        return (
            f"REBAL_CYCLE selected={selected_count} skipped={skipped_count} "
            f"valuable={total_valuable}/{total_channels} "
            f"budget_remaining={total_budget_sats}"
        )

    # ------------------------------------------------------------------
    # Logging wrappers (delegate to plugin.log)
    # ------------------------------------------------------------------

    def log_pick(
        self,
        source_channel_id: str,
        dest_channel_id: str,
        amount_sats: int,
        route_cost_sats: int,
        value_score: float,
        router: str = "v2",
    ) -> None:
        self._plugin.log(
            self.format_pick(
                source_channel_id,
                dest_channel_id,
                amount_sats,
                route_cost_sats,
                value_score,
                router=router,
            ),
            level="debug",
        )

    def log_skip(
        self,
        channel_id: str,
        reason: str,
        value_class: str,
        remaining_budget_sats: int = 0,
        route_cost_sats: int = 0,
        detail: str = "",
        router: str = "v2",
    ) -> None:
        self._plugin.log(
            self.format_skip(
                channel_id,
                reason,
                value_class,
                remaining_budget_sats,
                route_cost_sats,
                detail,
                router=router,
            ),
            level="debug",
        )

    def log_skips(self, skips: Any, router: str = "v2") -> None:
        """Emit the cycle's skip records with aggregated log volume.

        Actionable reasons (cooldowns, budget, futility, below_hold_margin,
        lease, ...) keep one grep-friendly line per channel. Non-actionable
        reasons (NON_ACTIONABLE_SKIP_REASONS) are rolled up into one
        summary line per reason — e.g. ``REBAL_SKIP reason=inside_band
        count=61`` — since per-channel lines for healthy channels are pure
        IPC volume. Callers keep the full per-channel records in the cycle
        result; only the log emission is aggregated.
        """
        aggregated: dict[str, int] = {}
        for skip in skips:
            reason = str(getattr(skip, "reason", "") or "")
            if reason in NON_ACTIONABLE_SKIP_REASONS:
                aggregated[reason] = aggregated.get(reason, 0) + 1
                continue
            self.log_skip(
                channel_id=getattr(skip, "channel_id", ""),
                reason=reason,
                value_class=getattr(skip, "value_class", ""),
                remaining_budget_sats=int(
                    getattr(skip, "remaining_budget_sats", 0) or 0
                ),
                detail=getattr(skip, "detail", "") or "",
                router=router,
            )
        for reason, count in aggregated.items():
            self._plugin.log(
                self.format_skip_summary(reason, count, router=router),
                level="debug",
            )

    def log_cycle_summary(
        self,
        selected_count: int,
        skipped_count: int,
        total_valuable: int,
        total_channels: int,
        total_budget_sats: int,
    ) -> None:
        self._plugin.log(
            self.format_cycle_summary(
                selected_count,
                skipped_count,
                total_valuable,
                total_channels,
                total_budget_sats,
            ),
            level="debug",
        )
