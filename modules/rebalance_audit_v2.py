"""Structured audit logging for the v2 rebalance engine.

Every cycle must explain itself. For every valuable channel the engine
evaluates, one of the ``log_*`` helpers emits a deterministic, grep-friendly
record that replaces the vague ad-hoc messages of the v1 engine.

Tag vocabulary (``reason`` field in REBAL_SKIP):
    selected, skipped_inside_band, skipped_not_valuable, skipped_no_partner,
    skipped_no_budget, skipped_no_route, skipped_route_over_budget,
    skipped_cooldown, skipped_policy
"""

from __future__ import annotations

from typing import Any, Optional


class RebalanceAuditV2:
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
    ) -> str:
        return (
            f"REBAL_PICK source={source_channel_id} dest={dest_channel_id} "
            f"amount={amount_sats} route_cost_sats={route_cost_sats} "
            f"value_score={value_score}"
        )

    @staticmethod
    def format_skip(
        channel_id: str,
        reason: str,
        value_class: str,
        remaining_budget_sats: int = 0,
        route_cost_sats: int = 0,
        detail: str = "",
    ) -> str:
        parts = [
            f"REBAL_SKIP channel={channel_id}",
            f"reason={reason}",
            f"value_class={value_class}",
            f"budget={remaining_budget_sats}",
        ]
        if route_cost_sats:
            parts.append(f"route_cost={route_cost_sats}")
        if detail:
            parts.append(f"detail={detail}")
        return " ".join(parts)

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
    ) -> None:
        self._plugin.log(
            self.format_pick(
                source_channel_id,
                dest_channel_id,
                amount_sats,
                route_cost_sats,
                value_score,
            ),
            level="info",
        )

    def log_skip(
        self,
        channel_id: str,
        reason: str,
        value_class: str,
        remaining_budget_sats: int = 0,
        route_cost_sats: int = 0,
        detail: str = "",
    ) -> None:
        self._plugin.log(
            self.format_skip(
                channel_id,
                reason,
                value_class,
                remaining_budget_sats,
                route_cost_sats,
                detail,
            ),
            level="info",
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
            level="info",
        )
