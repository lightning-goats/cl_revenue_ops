"""Unified Capex Budget Engine.

Computes per-channel rebalance budgets from
profitability and spend data. Pure calculation layer — no CLN RPC calls.

Inputs:
- Profitability analyzer cache (contribution, classification, bleeder status)
- Database spend history (rebalance_costs + spend_events tables)
- Config (reinvestment rates, caps, thresholds)

Outputs:
- Per-channel budgets with tier classification
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


def _safe_int(value, default: int = 0) -> int:
    """Normalize untrusted analyzer fields without crashing a budget cycle."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


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
        budgets and priority class.
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
            contribution_msat = _safe_int(
                getattr(prof.revenue, 'total_contribution_msat', 0)
            )
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

        # CB-4 fail-closed: with spend history unreadable we cannot know what
        # has already been spent or reserved, so deny ALL spend this cycle
        # (zero every budget) rather than re-grant full budgets fleet-wide.
        if db_degraded:
            _log.warning(
                "capex_budget: DB degraded this cycle — failing closed: "
                "all channel budgets zeroed"
            )
            for b in channel_budgets.values():
                b.budget_msat = 0

        # Global envelope enforcement (msat)
        total_channel_budgets_msat = sum(b.budget_msat for b in channel_budgets.values())
        raw_total_msat = total_channel_budgets_msat

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
        operational_total_msat = sum(
            b.budget_msat for b in channel_budgets.values()
            if b.priority_class == "operational"
        )
        growth_total_msat = sum(
            b.budget_msat for b in channel_budgets.values()
            if b.priority_class == "growth"
        )

        alloc = CapexAllocations(
            priority_class=priority_class,
            global_envelope_msat=envelope_msat,
            channel_budgets=channel_budgets,
            total_fleet_contribution_msat=total_fleet_contribution_msat,
            db_degraded=db_degraded,
            profitability_unavailable=prof_unavailable,
            allocated_by_priority_msat={
                "defensive": defensive_total_msat,
                "preservation": preservation_total_msat,
                "operational": operational_total_msat,
                "growth": growth_total_msat,
            },
        )
        self._last_allocations = alloc
        return alloc

    def get_channel_budget(self, channel_id: str) -> ChannelCapexBudget:
        """Get per-channel budget from last compute_allocations."""
        if self._last_allocations and channel_id in self._last_allocations.channel_budgets:
            return self._last_allocations.channel_budgets[channel_id]
        return ChannelCapexBudget(channel_id=channel_id)

    def get_priority_class(self) -> str:
        """Current fleet state priority class."""
        if self._last_allocations:
            return self._last_allocations.priority_class
        return "growth"

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
        contribution_msat = _safe_int(
            getattr(prof.revenue, 'total_contribution_msat', 0)
        )
        # Audit F1: budgets are debited by 30d spend, so the funding side must
        # also be the trailing-30d contribution. Funding from LIFETIME
        # contribution let decayed channels (50k sats lifetime, ~100 sats/30d)
        # claim 20k+/month budgets, manufacturing hard bleeders and a
        # block -> age-out -> "proven" again oscillation.
        contribution_30d_msat = _windowed_msat(
            prof, 'contribution_30d_msat', contribution_msat
        )
        total_fwd = max(0, _safe_int(getattr(prof.revenue, 'total_forward_count', 0)))
        days_open = max(0, _safe_int(getattr(prof, 'days_open', 0)))
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
        elif total_fwd > 0 and contribution_30d_msat > 0:
            # A young channel with a small number of real forwards used to
            # fall through to ``blocked`` until it crossed either the >5
            # forward gate or the >100-sat proven gate. That cliff can deny a
            # profitable, already-depleted channel any refill budget even
            # though canonical contribution evidence exists. Admit an early
            # active tier, but fund it only from the normal reinvestment share
            # of realized 30d contribution and cap it by the bootstrap rail.
            # There is no speculative floor: malformed, absent, zero, or
            # negative contribution still grants nothing.
            tier = "active"
            tier_ppm = 250
            priority = "growth"
            raw_budget_msat = min(
                proven_budget_msat,
                max(0, bootstrap_budget_msat - total_capex_30d_msat),
            )
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
        to "operational" by incorrectly treating a possibly-full wallet as empty. When the wallet cannot be read, report NO
        deficit (grant nothing) this cycle.
        """
        onchain = self._get_confirmed_onchain_sats()
        if onchain is None:
            _log.warning(
                "capex_budget: wallet unreadable (listfunds failed) — "
                "assuming NO reserve deficit this cycle"
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
