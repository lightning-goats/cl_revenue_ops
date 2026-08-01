"""Unified Capex Budget Engine.

Computes per-channel, fleet exploration, and tactical budgets from
profitability and spend data. Pure calculation layer — no CLN RPC calls.

Inputs:
- Profitability analyzer cache (contribution, classification, bleeder status)
- Database spend history (rebalance_costs + spend_events tables)
- Config (reinvestment rates, caps, thresholds)

Outputs:
- Per-channel budgets with tier classification
- Fleet exploration budget for opens and recycling
- Tactical budget for Boltz treasury
- Priority class based on fleet state
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .utils import MSAT_PER_SAT, base_to_sats_ceil, base_to_sats_floor, parse_msat

_log = logging.getLogger("cl-revenue-ops.capex_budget")


def _classify(obj) -> str:
    """Extract classification string from enum or string."""
    if hasattr(obj, 'value'):
        return str(obj.value)
    return str(obj).rsplit('.', 1)[-1].lower()


def _has_30d_window(prof) -> bool:
    """Whether the prof object carries trailing-30d windowed fields (audit F1).

    Producers that prefetch the 30d P&L set window_30d_available=True; older
    objects (or mocks without the flag) fall back to lifetime aggregates.
    """
    return getattr(prof, 'window_30d_available', False) is True


def _windowed_msat(prof, attr: str, fallback_msat: int) -> int:
    """Read a windowed msat field from prof, falling back to a lifetime value.

    Falls back when the 30d window is unavailable or the field is not a
    plain number (e.g. partially-mocked objects).
    """
    if _has_30d_window(prof):
        val = getattr(prof, attr, None)
        if isinstance(val, (int, float)):
            return int(val)
    try:
        return int(fallback_msat)
    except (TypeError, ValueError):
        return 0


@dataclass
class ChannelCapexBudget:
    """Per-channel capex budget computed by the engine."""
    channel_id: str
    budget_msat: int = 0           # Remaining 30d budget (millisatoshis)
    tier: str = "blocked"          # proven / active / bootstrap / fleet / blocked
    tier_ppm: int = 0              # Max PPM per rebalance attempt
    priority_class: str = "growth" # defensive / preservation / operational / growth
    # Diagnostics (audit F6): sr no longer discounts the budget (v2 bleeder
    # detection already divides effective cost by sr); kept for visibility.
    success_rate_30d: Optional[float] = None
    roi_multiplier: float = 1.0    # clamp(1 + marginal_roi, 0.25, 1.5)

    @property
    def budget_sats(self) -> int:
        """Budget in sats, ceiling-rounded. Zero msat yields zero sats (no false floor)."""
        return base_to_sats_ceil(self.budget_msat)


@dataclass
class CapexAllocations:
    """Complete budget allocation snapshot for one cycle."""
    priority_class: str = "growth"
    global_envelope_msat: int = 0
    channel_budgets: Dict[str, ChannelCapexBudget] = field(default_factory=dict)
    fleet_exploration_budget_msat: int = 0
    tactical_budget_msat: int = 0
    allocated_by_priority_msat: Dict[str, int] = field(default_factory=dict)
    total_fleet_contribution_msat: int = 0
    # CB-4 fail-closed: True when spend history could not be read from the
    # database this cycle, so all budgets were zeroed (spend denied) rather
    # than re-granted as if nothing had been spent.
    db_degraded: bool = False
    # Wave2 F2 fail-closed: True when the profitability analyzer had no real
    # snapshot this cycle (RPC outage), so exploration budgeting was zeroed
    # instead of falling into the zero-revenue bootstrap path that grants
    # the entire wallet excess.
    profitability_unavailable: bool = False

    @property
    def global_envelope_sats(self) -> int:
        """Global envelope in sats, ceiling-rounded."""
        return base_to_sats_ceil(self.global_envelope_msat)

    @property
    def fleet_exploration_budget_sats(self) -> int:
        """Fleet exploration budget in sats, ceiling-rounded."""
        return base_to_sats_ceil(self.fleet_exploration_budget_msat)

    @property
    def tactical_budget_sats(self) -> int:
        """Tactical budget in sats, ceiling-rounded."""
        return base_to_sats_ceil(self.tactical_budget_msat)

    @property
    def total_fleet_contribution_sats(self) -> int:
        """Total fleet contribution in sats, ceiling-rounded."""
        return base_to_sats_ceil(self.total_fleet_contribution_msat)

    @property
    def allocated_by_priority_sats(self) -> Dict[str, int]:
        """Allocated by priority in sats, ceiling-rounded."""
        return {k: base_to_sats_ceil(v) for k, v in self.allocated_by_priority_msat.items()}


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
        capital_efficiency=None,
    ):
        self._profitability = profitability_analyzer
        self._database = database
        self._config = config
        self._capital_efficiency = capital_efficiency
        self._last_allocations: Optional[CapexAllocations] = None

    def compute_allocations(self) -> CapexAllocations:
        """Compute all budgets for the current cycle.

        Reads profitability cache + spend history, computes per-channel
        budgets, fleet exploration, tactical, and priority class.
        """
        cfg = self._config.snapshot() if hasattr(self._config, 'snapshot') else self._config

        # Get profitability data for all channels.
        # Wave2 F2: distinguish "RPC failed" from "node genuinely has no
        # channels/revenue". An outage must never route budgeting into the
        # zero-revenue bootstrap path (which grants the whole wallet excess).
        all_prof = {}
        prof_unavailable = False
        try:
            all_prof = self._profitability.analyze_all_channels()
        except Exception:
            all_prof = {}
            prof_unavailable = True
        if not prof_unavailable:
            avail_fn = getattr(self._profitability, 'data_available', None)
            if callable(avail_fn):
                try:
                    prof_unavailable = not bool(avail_fn())
                except Exception:
                    prof_unavailable = True

        # Get total capex per channel (rebalance_costs + spend_events) — in sats.
        # CB-4 fail-closed: None means the DB read failed; treat spend history
        # as unknown and deny all spend this cycle (db_degraded below).
        capex_by_channel = self._get_total_capex_by_channel(window_days=30)
        db_degraded = capex_by_channel is None
        if capex_by_channel is None:
            capex_by_channel = {}
        fleet_efficiency = None
        if self._capital_efficiency is not None:
            try:
                fleet_efficiency = self._capital_efficiency.analyze()
            except Exception:
                fleet_efficiency = None

        # Compute per-channel budgets (all arithmetic in msat)
        channel_budgets: Dict[str, ChannelCapexBudget] = {}
        total_fleet_contribution_msat = 0
        hard_bleeder_count = 0
        hard_bleeder_capacity_sats = 0
        total_capacity_sats = 0
        has_depleted_earners = False

        for ch_id, prof in all_prof.items():
            try:
                capacity_sats = int(getattr(prof, 'capacity_sats', 0) or 0)
            except (TypeError, ValueError):
                capacity_sats = 0
            total_capacity_sats += capacity_sats
            contribution_msat = prof.revenue.total_contribution_msat
            # Audit F1(d): fleet budgets are debited by 30d spend, so they must
            # be FUNDED by 30d revenue (exit fees only), not lifetime totals.
            total_fleet_contribution_msat += _windowed_msat(
                prof, 'fees_earned_30d_msat', prof.revenue.fees_earned_msat
            )  # Fleet revenue = exit only
            total_capex_msat = capex_by_channel.get(ch_id, 0) * MSAT_PER_SAT

            # Bleeder status
            bleeder_status = "none"
            try:
                bleeder = self._profitability.get_bleeder_status(ch_id)
                if bleeder:
                    bleeder_status = bleeder.classification
                    if bleeder_status == "hard":
                        hard_bleeder_count += 1
                        hard_bleeder_capacity_sats += capacity_sats
            except Exception:
                pass

            # Depleted earner detection
            if contribution_msat > 100 * MSAT_PER_SAT:
                classification = _classify(prof.classification)
                if classification in ("underwater", "stagnant_candidate"):
                    has_depleted_earners = True

            budget = self._compute_channel_budget(
                ch_id=ch_id,
                prof=prof,
                total_capex_30d_msat=total_capex_msat,
                bleeder_status=bleeder_status,
                cfg=cfg,
                fleet_efficiency=fleet_efficiency,
            )
            channel_budgets[ch_id] = budget

        # Audit F4b: a single small bleeder must not flip the WHOLE fleet
        # into defensive mode. Fleet-significant bleeding means more than one
        # hard bleeder, or hard-bleeder capacity above 10% of fleet capacity.
        # (The bleeder channel itself is still individually blocked.)
        has_hard_bleeders = hard_bleeder_count > 1 or (
            total_capacity_sats > 0
            and hard_bleeder_capacity_sats / total_capacity_sats > 0.10
        )

        # Detect priority class
        priority_class = self._detect_priority_class(
            has_hard_bleeders=has_hard_bleeders,
            has_depleted_earners=has_depleted_earners,
            reserve_deficit=self._get_reserve_deficit(cfg),
        )

        # Compute fleet exploration and tactical budgets (msat)
        reserve_deficit_msat = self._get_reserve_deficit(cfg) * MSAT_PER_SAT
        tactical_msat = min(reserve_deficit_msat, int(total_fleet_contribution_msat * cfg.capex_tactical_rate))
        tactical_msat = max(0, tactical_msat)

        spend_summary = self._get_spend_ledger_summary(window_days=30)
        if spend_summary is None:
            db_degraded = True
            spend_summary = {"spent_by_category": {}, "reserved_by_category": {}}
        if prof_unavailable:
            # Wave2 F2 fail-closed: bootstrap mode requires AFFIRMATIVE
            # evidence of a zero-revenue node (a real, non-error snapshot).
            # With no snapshot at all, a mature revenue node would otherwise
            # flip into bootstrap and get the entire wallet excess as
            # exploration budget. Grant nothing this cycle.
            _log.warning(
                "capex_budget: profitability data unavailable this cycle — "
                "granting NO exploration budget (bootstrap requires a real "
                "zero-revenue snapshot, not a data outage)"
            )
            exploration_msat = 0
        elif total_fleet_contribution_msat > 0:
            revenue_exploration_msat = int(total_fleet_contribution_msat * cfg.capex_exploration_rate)
            wallet_floor_msat = self._get_wallet_exploration_floor_msat(cfg)
            exploration_msat = max(revenue_exploration_msat, wallet_floor_msat)
            exploration_msat = self._apply_category_spend_remaining(
                budget_msat=exploration_msat,
                summary=spend_summary,
                category="channel_open",
            )
        else:
            exploration_msat = self._get_bootstrap_wallet_exploration_budget_msat(
                cfg=cfg,
                summary=spend_summary,
            )
        tactical_msat = self._apply_category_spend_remaining(
            budget_msat=tactical_msat,
            summary=spend_summary,
            category="boltz",
        )

        # CB-4 fail-closed: with spend history unreadable we cannot know what
        # has already been spent or reserved, so deny ALL spend this cycle
        # (zero every budget) rather than re-grant full budgets fleet-wide.
        if db_degraded:
            _log.warning(
                "capex_budget: DB degraded this cycle — failing closed: "
                "all channel, exploration and tactical budgets zeroed"
            )
            exploration_msat = 0
            tactical_msat = 0
            for b in channel_budgets.values():
                b.budget_msat = 0

        # Global envelope enforcement (msat)
        total_channel_budgets_msat = sum(b.budget_msat for b in channel_budgets.values())
        raw_total_msat = total_channel_budgets_msat + exploration_msat + tactical_msat

        if cfg.capex_global_envelope_sats > 0:
            envelope_msat = cfg.capex_global_envelope_sats * MSAT_PER_SAT
        else:
            envelope_msat = raw_total_msat

        # Emergency overrides
        if cfg.daily_budget_sats > 0:
            daily_30d_msat = cfg.daily_budget_sats * 30 * MSAT_PER_SAT
            envelope_msat = min(envelope_msat, daily_30d_msat)
        if cfg.weekly_budget_sats > 0:
            weekly_30d_msat = int(cfg.weekly_budget_sats * MSAT_PER_SAT * (30 / 7))
            envelope_msat = min(envelope_msat, weekly_30d_msat)

        # Scale down if over envelope
        if raw_total_msat > envelope_msat and raw_total_msat > 0:
            scale = envelope_msat / raw_total_msat
            exploration_msat = int(exploration_msat * scale)
            tactical_msat = int(tactical_msat * scale)
            for b in channel_budgets.values():
                b.budget_msat = int(b.budget_msat * scale)

        # Priority allocation tracking (msat)
        defensive_total_msat = sum(
            b.budget_msat for b in channel_budgets.values()
            if b.priority_class == "defensive"
        )
        preservation_total_msat = sum(
            b.budget_msat for b in channel_budgets.values()
            if b.priority_class == "preservation"
        )

        alloc = CapexAllocations(
            priority_class=priority_class,
            global_envelope_msat=envelope_msat,
            channel_budgets=channel_budgets,
            fleet_exploration_budget_msat=exploration_msat,
            tactical_budget_msat=tactical_msat,
            total_fleet_contribution_msat=total_fleet_contribution_msat,
            db_degraded=db_degraded,
            profitability_unavailable=prof_unavailable,
            allocated_by_priority_msat={
                "defensive": defensive_total_msat,
                "preservation": preservation_total_msat,
                "operational": tactical_msat,
                "growth": exploration_msat,
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
        """Remaining fleet exploration budget in sats (ceiling)."""
        if self._last_allocations:
            return self._last_allocations.fleet_exploration_budget_sats
        return 0

    def get_tactical_budget(self) -> int:
        """Remaining tactical budget in sats (ceiling)."""
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

        Operates in sats (consumer-facing boundary method).
        Callers provide cost in sats; returns split in sats.

        Returns:
            {"channel": amount_sats, "tactical": amount_sats}
        """
        if channel_id is None:
            # Pure treasury swap — all tactical
            return {"channel": 0, "tactical": cost_sats}
        # Channel-targeted swap — 50/50 split
        channel_share = cost_sats // 2
        tactical_share = cost_sats - channel_share
        return {"channel": channel_share, "tactical": tactical_share}

    def record_boltz_spend(
        self,
        swap_id: str,
        fee_sats: int,
        channel_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None,
        subcategory: str = "swap_fee",
    ) -> bool:
        """Record an executed Boltz swap fee in the generic spend ledger.

        Writes a category="boltz" spend event so that
        _apply_category_spend_remaining depletes the tactical budget (and,
        when channel_id is set, per-channel capex via spend_events).
        Keyed by swap id, so repeated journal updates for the same swap are
        idempotent (INSERT OR REPLACE in the database layer).

        subcategory partitions the unified boltz category: "structural"
        swaps (executed only because of the inbound-scarcity credit) are
        summed by the daily structural envelope gate, while the tactical
        budget keeps counting the whole category exactly once.
        """
        sid = str(swap_id or "").strip()
        if not sid:
            return False
        try:
            fee = int(fee_sats)
        except (TypeError, ValueError):
            return False
        if fee <= 0:
            return False
        try:
            return bool(self._database.record_spend_event(
                event_id=f"boltz:{sid}",
                category="boltz",
                amount_sats=fee,
                subcategory=str(subcategory or "swap_fee"),
                reference_id=sid,
                channel_id=str(channel_id).replace(':', 'x') if channel_id else None,
                source=source or "boltz_manager",
                metadata=metadata,
            ))
        except Exception:
            return False

    # --- P4-014: atomic unified-budget reservation for Boltz swaps ---------
    #
    # DD1: a Boltz swap must reserve its committed cost against the SAME unified
    # cross-category budget the rebalance/generic paths use, BEFORE the swap is
    # created, so a concurrent rebalance reserve both sees and is counted by the
    # boltz commitment. reserve_spend enforces the full cross-category total
    # inside its BEGIN IMMEDIATE (serialized by the WAL single-writer lock), so
    # no interleaving can jointly overshoot the budget.

    def reserve_boltz_swap_budget(
        self,
        reservation_id: str,
        estimated_fee_sats: int,
        channel_id: Optional[str] = None,
        effective_budget_sats: int = 0,
        subcategory: str = "swap_fee",
    ) -> bool:
        """Reserve a swap's estimated committed cost. False = budget would be
        exceeded (caller must not create the swap)."""
        try:
            fee = int(estimated_fee_sats)
        except (TypeError, ValueError):
            return False
        if fee <= 0:
            return False
        since = int(time.time()) - 24 * 3600
        try:
            return bool(self._database.reserve_spend(
                reservation_id=str(reservation_id),
                amount_sats=fee,
                category="boltz",
                subcategory=str(subcategory or "swap_fee"),
                channel_id=str(channel_id).replace(':', 'x') if channel_id else None,
                effective_budget_sats=int(effective_budget_sats),
                since_timestamp=since,
            ))
        except Exception:
            return False

    def settle_boltz_swap_reservation(
        self,
        reservation_id: str,
        swap_id: str,
        estimated_fee_sats: int,
        channel_id: Optional[str] = None,
        subcategory: str = "swap_fee",
    ) -> bool:
        """Swap created: persist the estimated committed cost as a spend event
        keyed by swap id (later REPLACED by the actual fee when the swap settles
        via record_boltz_spend — same boltz:{sid} event id) and release the
        pre-create reservation so the cost is not double-counted.

        P4-019 (mirrors the P2-008 loud-write requirement): the reservation must
        NOT be released unconditionally. If the spend-event write FAILS, releasing
        the reservation would leave the created swap with NO reservation AND NO
        spend event — its committed cost would vanish from the unified budget in
        the overspend-permitting direction. So we release ONLY once the event has
        persisted; on write failure the reservation stays active (the cost keeps
        being counted) and we return False so the caller logs/retries. The event
        write is idempotent (INSERT OR REPLACE on boltz:{sid}), so a later journal
        update for the same swap re-settles and then releases.
        """
        try:
            fee = int(estimated_fee_sats)
        except (TypeError, ValueError):
            fee = 0
        if fee <= 0:
            # Nothing to commit — release the (zero-cost) reservation.
            self.release_boltz_swap_reservation(reservation_id)
            return True
        recorded = self.record_boltz_spend(
            swap_id=swap_id,
            fee_sats=fee,
            channel_id=channel_id,
            source="boltz_swap_estimate",
            subcategory=subcategory,
        )
        if not recorded:
            # Loud write failed: keep the reservation so the committed cost stays
            # visible to the unified budget. Do NOT release.
            return False
        self.release_boltz_swap_reservation(reservation_id)
        return True

    def release_boltz_swap_reservation(self, reservation_id: str) -> bool:
        """Swap failed/never created (or superseded by its spend event): release
        the reservation so the budget is not held."""
        try:
            return bool(self._database.release_spend_reservation(str(reservation_id)))
        except Exception:
            return False

    # --- Internal methods ---

    def _compute_channel_budget(
        self,
        ch_id: str,
        prof,
        total_capex_30d_msat: int,
        bleeder_status: str,
        cfg,
        fleet_efficiency=None,
    ) -> ChannelCapexBudget:
        """Compute budget for a single channel (all arithmetic in msat)."""
        classification = _classify(prof.classification)
        contribution_msat = prof.revenue.total_contribution_msat
        # Audit F1: budgets are debited by 30d spend, so the funding side must
        # also be the trailing-30d contribution. Funding from LIFETIME
        # contribution let decayed channels (50k sats lifetime, ~100 sats/30d)
        # claim 20k+/month budgets, manufacturing hard bleeders and a
        # block -> age-out -> "proven" again oscillation.
        contribution_30d_msat = _windowed_msat(
            prof, 'contribution_30d_msat', contribution_msat
        )
        total_fwd = prof.revenue.total_forward_count
        days_open = prof.days_open
        marginal_roi = getattr(prof, 'marginal_roi', 0.0)

        # Blocked channels
        if classification == "zombie":
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked", priority_class="defensive",
            )
        if bleeder_status == "hard":
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked", priority_class="defensive",
            )
        if days_open < cfg.capex_grace_days and contribution_msat == 0:
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked",
            )
        # NOTE: Removed the marginal_roi < 0 + zero contribution block.
        # New channels have negative ROI (open cost > 0, revenue = 0) and
        # zero contribution. Blocking them prevents the capital deployment
        # that would START generating returns. The BOOTSTRAP tier handles
        # these channels with conservative budgets.

        # Audit F6: marginal-ROI multiplier replaces the success-rate
        # discount. Dividing by sr double-penalized (identify_bleeders_v2
        # already inflates effective cost by sr) and mispredicted: failed
        # rebalance attempts cost ~0 sats, so a 50% sr does not double the
        # realized cost. Scale instead by realized 30d marginal return,
        # clamped to [0.25, 1.5]; unreliable marginal ROI (< 100 sats of 30d
        # spend evidence, audit F8) is treated as neutral. The success rate
        # is still fetched and recorded for diagnostics.
        sr_data = None
        try:
            sr_data = self._database.get_channel_rebalance_success_rate(ch_id, 30)
        except Exception:
            pass
        success_rate = None
        if sr_data and sr_data.get('total', 0) >= 3:
            try:
                success_rate = float(sr_data['success_rate'])
            except (TypeError, ValueError):
                success_rate = None

        roi_reliable = getattr(prof, 'marginal_roi_reliable', True)
        if isinstance(marginal_roi, (int, float)) and roi_reliable is not False:
            roi_mult = min(1.5, max(0.25, 1.0 + float(marginal_roi)))
        else:
            roi_mult = 1.0

        # Proven budget (msat): 30d contribution x rate - 30d capex spent.
        # Same window on both sides (audit F1b).
        reinvestment = cfg.capex_reinvestment_rate
        proven_budget_msat = max(0, int(contribution_30d_msat * reinvestment) - total_capex_30d_msat)

        # Bootstrap budget (msat): basis points of capacity
        capacity_msat = (getattr(prof, 'capacity_sats', 0) or 0) * MSAT_PER_SAT
        bootstrap_budget_msat = min(
            int(capacity_msat * cfg.capex_bootstrap_bps / 10000),
            cfg.capex_bootstrap_max_sats * MSAT_PER_SAT,
        )

        # Tier classification
        # Audit F1c: the proven gate is windowed — a channel must have EARNED
        # >100 sats in the last 30 days, not at any point in its lifetime.
        if contribution_30d_msat > 100 * MSAT_PER_SAT:
            tier = "proven"
            tier_ppm = 2000
            priority = "preservation"
            raw_budget_msat = proven_budget_msat
        elif total_fwd > 5:
            tier = "active"
            tier_ppm = 500
            priority = "preservation"
            # Active: pick higher of proven (already capex-adjusted) or bootstrap - capex
            if proven_budget_msat > 0:
                raw_budget_msat = max(proven_budget_msat, max(0, bootstrap_budget_msat - total_capex_30d_msat))
            else:
                raw_budget_msat = max(0, bootstrap_budget_msat - total_capex_30d_msat)
        elif days_open >= cfg.capex_grace_days:
            tier = "bootstrap"
            tier_ppm = 250
            priority = "growth"
            raw_budget_msat = max(0, bootstrap_budget_msat - total_capex_30d_msat)
        else:
            return ChannelCapexBudget(
                channel_id=ch_id, tier="blocked",
            )

        efficiency_mult = self._get_efficiency_multiplier(
            ch_id, fleet_efficiency, prof=prof)
        budget_msat = int(raw_budget_msat * roi_mult * efficiency_mult)

        return ChannelCapexBudget(
            channel_id=ch_id,
            budget_msat=budget_msat,
            tier=tier,
            tier_ppm=tier_ppm,
            priority_class=priority,
            success_rate_30d=success_rate,
            roi_multiplier=roi_mult,
        )

    def _get_efficiency_multiplier(self, channel_id: str, fleet_efficiency,
                                   prof=None) -> float:
        """Return the capital-efficiency multiplier for a channel budget."""
        if fleet_efficiency is None:
            return 1.0

        channel_efficiencies = getattr(fleet_efficiency, "channel_efficiencies", {}) or {}
        channel_eff = channel_efficiencies.get(channel_id)
        if channel_eff is None:
            return 1.0

        if getattr(channel_eff, "is_dead_capital", False):
            # Audit F5: dead-capital zeroing ignored gateway protections.
            # Quiet inbound gateways earn ~0 direct fees (RPSD sees them as
            # dead) but source the volume other channels monetize. Floor at
            # 0.25 when the prof shows gateway/sourced value.
            if self._has_gateway_value(prof):
                return 0.25
            return 0.0

        median_rpsd = float(getattr(fleet_efficiency, "median_rpsd", 0.0) or 0.0)
        rpsd = float(getattr(channel_eff, "rpsd", 0.0) or 0.0)
        if median_rpsd <= 0:
            return 1.0

        if rpsd >= median_rpsd:
            return 1.0 + min(0.5, (rpsd / median_rpsd - 1.0) * 0.25)

        return max(0.5, rpsd / median_rpsd)

    @staticmethod
    def _has_gateway_value(prof) -> bool:
        """Whether a prof object shows inbound-gateway / sourced-fee value.

        Predicate (audit F5): lifetime role == INBOUND_GATEWAY or lifetime
        sourced_fee_contribution_msat > 100_000 (100 sats).
        """
        if prof is None:
            return False
        role = getattr(prof, 'channel_role', None)
        role_val = getattr(role, 'value', role)
        if role_val == "inbound_gateway":
            return True
        revenue = getattr(prof, 'revenue', None)
        sourced = getattr(revenue, 'sourced_fee_contribution_msat', 0)
        return isinstance(sourced, (int, float)) and sourced > 100_000

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
        """Get on-chain reserve deficit in sats.

        Wave2 F3 fail-closed: an UNREADABLE wallet must not read as an EMPTY
        wallet. `deficit = min_wallet_reserve - 0` flipped the priority class
        to "operational" and granted a tactical Boltz budget to refill a
        possibly-full wallet. When the wallet cannot be read, report NO
        deficit (grant nothing) this cycle.
        """
        onchain = self._get_confirmed_onchain_sats()
        if onchain is None:
            _log.warning(
                "capex_budget: wallet unreadable (listfunds failed) — "
                "assuming NO reserve deficit this cycle (no tactical refill "
                "budget granted on unknown balance)"
            )
            return 0
        deficit = max(0, cfg.min_wallet_reserve - onchain)
        return deficit

    def _get_confirmed_onchain_sats(self) -> Optional[int]:
        """Get confirmed on-chain balance in sats; None when unreadable.

        Wave2 F3: Database.get_confirmed_onchain_sats swallows listfunds
        failures to 0, which is indistinguishable from a genuinely empty
        wallet. Read listfunds through the database's RPC handle here so an
        RPC failure surfaces as None (callers treat unknown conservatively).
        When no usable RPC surface exists (e.g. fully-mocked database in
        tests), fall back to the DB helper's value.
        """
        rpc = getattr(getattr(self._database, 'plugin', None), 'rpc', None)
        if rpc is not None:
            try:
                lf = rpc.listfunds()
            except Exception as exc:
                _log.warning(
                    "capex_budget: listfunds failed (%s) — confirmed on-chain "
                    "balance unknown", exc
                )
                return None
            if isinstance(lf, dict):
                total_sats = 0
                for output in lf.get("outputs", []) or []:
                    if not isinstance(output, dict):
                        continue
                    if str(output.get("status") or "") != "confirmed":
                        continue
                    total_sats += base_to_sats_floor(
                        parse_msat(output.get("amount_msat", 0))
                    )
                return int(total_sats)
        try:
            return int(self._database.get_confirmed_onchain_sats() or 0)
        except Exception:
            return None

    def _get_bootstrap_wallet_exploration_budget_msat(self, cfg, summary) -> int:
        """Use wallet excess for bootstrap opens when the fleet has no fee revenue yet.

        This path is intentionally wallet-backed rather than spend-ledger-backed:
        confirmed on-chain balance already reflects prior open fees, so subtracting
        historical spend again would double-count. We only remove active reservations.
        """
        onchain_sats = self._get_confirmed_onchain_sats()
        if onchain_sats is None:
            # Wave2 F3: unknown wallet -> no bootstrap exploration budget.
            return 0
        wallet_excess_sats = max(0, onchain_sats - cfg.min_wallet_reserve)
        reserved_by_category = summary.get("reserved_by_category", {}) or {}
        reserved_open_sats = int(reserved_by_category.get("channel_open", 0) or 0)
        return max(0, wallet_excess_sats - reserved_open_sats) * MSAT_PER_SAT

    def _get_wallet_exploration_floor_msat(self, cfg) -> int:
        """Allow one open-fee worth of exploration when wallet excess exists.

        Revenue-funded fleets can otherwise get stuck below the current open
        fee after earning small amounts. This floor covers only the configured
        fee estimate, not the full wallet excess used during zero-revenue
        bootstrap.
        """
        onchain_sats = self._get_confirmed_onchain_sats()
        if onchain_sats is None:
            # Wave2 F3: unknown wallet -> no wallet-backed exploration floor.
            return 0
        wallet_excess_sats = max(0, onchain_sats - cfg.min_wallet_reserve)
        if wallet_excess_sats <= 0:
            return 0
        try:
            open_fee_sats = int(getattr(cfg, "estimated_open_cost_sats", 0) or 0)
        except (TypeError, ValueError):
            open_fee_sats = 0
        open_fee_sats = max(0, open_fee_sats)
        return min(wallet_excess_sats, open_fee_sats) * MSAT_PER_SAT

    def _get_total_capex_by_channel(self, window_days: int = 30) -> Optional[Dict[str, int]]:
        """Get total capex per channel from rebalance_costs + spend_events.

        CB-4 fail-closed: returns None on DB error (never an empty dict, which
        downstream arithmetic would treat as "nothing spent" and re-grant full
        budgets). Callers must treat None as degraded and deny spend.
        """
        try:
            return self._database.get_total_capex_by_channel(window_days)
        except Exception as exc:
            _log.warning(
                "capex_budget: DB error reading 30d capex by channel (%s); "
                "failing closed — denying capex spend this cycle", exc
            )
            return None

    def _get_spend_ledger_summary(self, window_days: int = 30) -> Optional[Dict[str, Dict[str, int]]]:
        """Get generic spend ledger totals for the requested rolling window.

        CB-4 fail-closed: returns None on DB error (never empty category dicts,
        which downstream depletion would treat as fully un-depleted budgets).
        Callers must treat None as degraded and deny spend.
        """
        try:
            return self._database.get_spend_ledger_summary(window_hours=window_days * 24)
        except Exception as exc:
            _log.warning(
                "capex_budget: DB error reading spend ledger summary (%s); "
                "failing closed — denying capex spend this cycle", exc
            )
            return None

    def _apply_category_spend_remaining(
        self,
        *,
        budget_msat: int,
        summary: Dict[str, Dict[str, int]],
        category: str,
    ) -> int:
        """Subtract spent and reserved sats for one category from a nominal msat budget."""
        spent_by_category = summary.get("spent_by_category", {}) or {}
        reserved_by_category = summary.get("reserved_by_category", {}) or {}
        consumed_sats = int(spent_by_category.get(category, 0) or 0) + int(
            reserved_by_category.get(category, 0) or 0
        )
        remaining_msat = budget_msat - (consumed_sats * MSAT_PER_SAT)
        return max(0, remaining_msat)
