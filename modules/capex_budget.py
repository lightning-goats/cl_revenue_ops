"""Unified Capex Budget Engine.

Computes per-channel, fleet exploration, and tactical budgets from
profitability and spend data. Pure calculation layer — no CLN RPC calls.

Inputs:
- Profitability analyzer cache (contribution, classification, bleeder status)
- Database spend history (rebalance_costs + spend_events tables)
- Config (reinvestment rates, caps, thresholds)
- Hive hints (optional: member/corridor multipliers)

Outputs:
- Per-channel budgets with tier classification
- Fleet exploration budget for opens and recycling
- Tactical budget for Boltz treasury
- Priority class based on fleet state
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ChannelCapexBudget:
    """Per-channel capex budget computed by the engine."""
    channel_id: str
    budget_sats: int = 0           # Remaining 30d budget
    tier: str = "blocked"          # proven / active / bootstrap / blocked
    tier_ppm: int = 0              # Max PPM per rebalance attempt
    priority_class: str = "growth" # defensive / preservation / operational / growth
    hive_multiplier: float = 1.0   # 1.0 / 1.5 / 2.0


@dataclass
class CapexAllocations:
    """Complete budget allocation snapshot for one cycle."""
    priority_class: str = "growth"
    global_envelope_sats: int = 0
    channel_budgets: Dict[str, ChannelCapexBudget] = field(default_factory=dict)
    fleet_exploration_budget_sats: int = 0
    tactical_budget_sats: int = 0
    allocated_by_priority: Dict[str, int] = field(default_factory=dict)
    total_fleet_contribution_sats: int = 0


class CapexBudgetEngine:
    """Unified capital expenditure budget engine.

    Computes budgets from existing profitability and spend data.
    No CLN RPC calls — reads from analyzer cache and local database.
    """

    def __init__(
        self,
        profitability_analyzer,
        database,
        config,
        hive_hints=None,
    ):
        self._profitability = profitability_analyzer
        self._database = database
        self._config = config
        self._hive_hints = hive_hints
        self._last_allocations: Optional[CapexAllocations] = None

    def compute_allocations(self) -> CapexAllocations:
        """Compute all budgets for the current cycle.

        Reads profitability cache + spend history, computes per-channel
        budgets, fleet exploration, tactical, and priority class.
        """
        cfg = self._config.snapshot() if hasattr(self._config, 'snapshot') else self._config

        # Get profitability data for all channels
        all_prof = {}
        try:
            all_prof = self._profitability.analyze_all_channels()
        except Exception:
            pass

        # Get total capex per channel (rebalance_costs + spend_events)
        capex_by_channel = self._get_total_capex_by_channel(window_days=30)

        # Compute per-channel budgets
        channel_budgets: Dict[str, ChannelCapexBudget] = {}
        total_fleet_contribution = 0
        has_hard_bleeders = False
        has_depleted_earners = False

        for ch_id, prof in all_prof.items():
            contribution = prof.revenue.total_contribution_sats
            total_fleet_contribution += prof.revenue.fees_earned_sats  # Fleet revenue = exit only
            total_capex = capex_by_channel.get(ch_id, 0)

            # Bleeder status
            bleeder_status = "none"
            try:
                bleeder = self._profitability.get_bleeder_status(ch_id)
                if bleeder:
                    bleeder_status = bleeder.classification
                    if bleeder_status == "hard":
                        has_hard_bleeders = True
            except Exception:
                pass

            # Depleted earner detection
            if contribution > 100:
                classification = prof.classification.value if hasattr(prof.classification, 'value') else str(prof.classification)
                if classification in ("underwater", "stagnant_candidate"):
                    has_depleted_earners = True

            budget = self._compute_channel_budget(
                ch_id=ch_id,
                prof=prof,
                total_capex_30d=total_capex,
                bleeder_status=bleeder_status,
                cfg=cfg,
            )
            channel_budgets[ch_id] = budget

        # Detect priority class
        priority_class = self._detect_priority_class(
            has_hard_bleeders=has_hard_bleeders,
            has_depleted_earners=has_depleted_earners,
            reserve_deficit=self._get_reserve_deficit(cfg),
        )

        # Compute fleet exploration and tactical budgets
        exploration = int(total_fleet_contribution * cfg.capex_exploration_rate)
        reserve_deficit = self._get_reserve_deficit(cfg)
        tactical = min(reserve_deficit, int(total_fleet_contribution * cfg.capex_tactical_rate))
        tactical = max(0, tactical)

        # Global envelope enforcement
        total_channel_budgets = sum(b.budget_sats for b in channel_budgets.values())
        raw_total = total_channel_budgets + exploration + tactical

        if cfg.capex_global_envelope_sats > 0:
            envelope = cfg.capex_global_envelope_sats
        else:
            envelope = raw_total

        # Emergency overrides
        if cfg.daily_budget_sats > 0:
            daily_30d = cfg.daily_budget_sats * 30
            envelope = min(envelope, daily_30d)
        if cfg.weekly_budget_sats > 0:
            weekly_30d = int(cfg.weekly_budget_sats * (30 / 7))
            envelope = min(envelope, weekly_30d)

        # Scale down if over envelope
        if raw_total > envelope and raw_total > 0:
            scale = envelope / raw_total
            exploration = int(exploration * scale)
            tactical = int(tactical * scale)
            for b in channel_budgets.values():
                b.budget_sats = int(b.budget_sats * scale)

        # Priority allocation tracking
        defensive_total = sum(
            b.budget_sats for b in channel_budgets.values()
            if b.priority_class == "defensive"
        )
        preservation_total = sum(
            b.budget_sats for b in channel_budgets.values()
            if b.priority_class == "preservation"
        )

        alloc = CapexAllocations(
            priority_class=priority_class,
            global_envelope_sats=envelope,
            channel_budgets=channel_budgets,
            fleet_exploration_budget_sats=exploration,
            tactical_budget_sats=tactical,
            total_fleet_contribution_sats=total_fleet_contribution,
            allocated_by_priority={
                "defensive": defensive_total,
                "preservation": preservation_total,
                "operational": tactical,
                "growth": exploration,
            },
        )
        self._last_allocations = alloc
        return alloc

    def get_channel_budget(self, channel_id: str) -> ChannelCapexBudget:
        """Get per-channel budget from last compute_allocations."""
        if self._last_allocations and channel_id in self._last_allocations.channel_budgets:
            return self._last_allocations.channel_budgets[channel_id]
        return ChannelCapexBudget(channel_id=channel_id)

    def get_fleet_exploration_budget(self) -> int:
        """Remaining fleet exploration budget for opens and recycling."""
        if self._last_allocations:
            return self._last_allocations.fleet_exploration_budget_sats
        return 0

    def get_tactical_budget(self) -> int:
        """Remaining tactical budget for Boltz treasury."""
        if self._last_allocations:
            return self._last_allocations.tactical_budget_sats
        return 0

    def get_priority_class(self) -> str:
        """Current fleet state priority class."""
        if self._last_allocations:
            return self._last_allocations.priority_class
        return "growth"

    def attribute_boltz_cost(
        self, cost_sats: int, channel_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Split a Boltz cost between channel and tactical budgets.

        Returns:
            {"channel": amount, "tactical": amount}
        """
        if channel_id is None:
            # Pure treasury swap — all tactical
            return {"channel": 0, "tactical": cost_sats}
        # Channel-targeted swap — 50/50 split
        channel_share = cost_sats // 2
        tactical_share = cost_sats - channel_share
        return {"channel": channel_share, "tactical": tactical_share}

    # --- Internal methods ---

    def _compute_channel_budget(
        self,
        ch_id: str,
        prof,
        total_capex_30d: int,
        bleeder_status: str,
        cfg,
    ) -> ChannelCapexBudget:
        """Compute budget for a single channel."""
        classification = prof.classification.value if hasattr(prof.classification, 'value') else str(prof.classification)
        contribution = prof.revenue.total_contribution_sats
        total_fwd = prof.revenue.total_forward_count
        days_open = prof.days_open
        marginal_roi = getattr(prof, 'marginal_roi', 0.0)

        # Hive multiplier
        hive_mult = 1.0
        if self._hive_hints:
            peer_id = getattr(prof, 'peer_id', '')
            try:
                if self._hive_hints.is_hive_member(peer_id):
                    hive_mult = 1.5
                corridor = self._hive_hints.get_corridor_role(peer_id)
                if corridor == "owner":
                    hive_mult = 2.0
            except Exception:
                pass

        # Blocked channels
        if classification == "zombie":
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked", priority_class="defensive",
                hive_multiplier=hive_mult,
            )
        if bleeder_status == "hard":
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked", priority_class="defensive",
                hive_multiplier=hive_mult,
            )
        if days_open < cfg.capex_grace_days and contribution == 0:
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked", hive_multiplier=hive_mult,
            )
        if marginal_roi < 0 and contribution == 0:
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked", priority_class="defensive",
                hive_multiplier=hive_mult,
            )

        # Success rate discount
        sr_data = None
        try:
            sr_data = self._database.get_channel_rebalance_success_rate(ch_id, 30)
        except Exception:
            pass
        if sr_data and sr_data.get('total', 0) >= 3:
            discount = max(0.1, sr_data['success_rate'])
        else:
            discount = 1.0

        # Proven budget: contribution x rate - total capex spent
        reinvestment = cfg.capex_reinvestment_rate
        proven_budget = max(0, int(contribution * reinvestment) - total_capex_30d)

        # Bootstrap budget: basis points of capacity
        capacity = getattr(prof, 'capacity_sats', 0) or 0
        bootstrap_budget = min(
            int(capacity * cfg.capex_bootstrap_bps / 10000),
            cfg.capex_bootstrap_max_sats,
        )

        # Tier classification
        if contribution > 100:
            tier = "proven"
            tier_ppm = 2000
            priority = "preservation"
            raw_budget = proven_budget
        elif total_fwd > 5:
            tier = "active"
            tier_ppm = 500
            priority = "preservation"
            # Active: pick higher of proven (already capex-adjusted) or bootstrap - capex
            if proven_budget > 0:
                raw_budget = max(proven_budget, max(0, bootstrap_budget - total_capex_30d))
            else:
                raw_budget = max(0, bootstrap_budget - total_capex_30d)
        elif days_open >= cfg.capex_grace_days:
            tier = "bootstrap"
            tier_ppm = 250
            priority = "growth"
            raw_budget = max(0, bootstrap_budget - total_capex_30d)
        else:
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked", hive_multiplier=hive_mult,
            )

        budget = int(raw_budget * discount * hive_mult)

        return ChannelCapexBudget(
            channel_id=ch_id,
            budget_sats=budget,
            tier=tier,
            tier_ppm=tier_ppm,
            priority_class=priority,
            hive_multiplier=hive_mult,
        )

    def _detect_priority_class(
        self,
        has_hard_bleeders: bool,
        has_depleted_earners: bool,
        reserve_deficit: int,
    ) -> str:
        """Detect fleet state priority class."""
        if has_hard_bleeders:
            return "defensive"
        if has_depleted_earners:
            return "preservation"
        if reserve_deficit > 0:
            return "operational"
        return "growth"

    def _get_reserve_deficit(self, cfg) -> int:
        """Get on-chain reserve deficit in sats."""
        try:
            onchain = self._database.get_confirmed_onchain_sats()
            deficit = max(0, cfg.min_wallet_reserve - onchain)
            return deficit
        except Exception:
            return 0

    def _get_total_capex_by_channel(self, window_days: int = 30) -> Dict[str, int]:
        """Get total capex per channel from rebalance_costs + spend_events.

        Sums both tables to get the complete picture of what's been spent
        per channel across all expense types.
        """
        since = int(time.time()) - (window_days * 86400)
        result: Dict[str, int] = {}

        try:
            conn = self._database._get_connection()

            # Rebalance costs (canonical table for rebalancing spend)
            rows = conn.execute("""
                SELECT channel_id, COALESCE(SUM(cost_sats), 0) as total
                FROM rebalance_costs
                WHERE timestamp >= ?
                GROUP BY channel_id
            """, (since,)).fetchall()
            for r in rows:
                cid = r["channel_id"]
                if cid:
                    result[cid] = result.get(cid, 0) + int(r["total"] or 0)

            # Spend events (opens, boltz, closures, etc.)
            rows = conn.execute("""
                SELECT channel_id, COALESCE(SUM(amount_sats), 0) as total
                FROM spend_events
                WHERE timestamp >= ? AND channel_id IS NOT NULL
                GROUP BY channel_id
            """, (since,)).fetchall()
            for r in rows:
                cid = r["channel_id"]
                if cid:
                    result[cid] = result.get(cid, 0) + int(r["total"] or 0)

        except Exception:
            pass

        return result
