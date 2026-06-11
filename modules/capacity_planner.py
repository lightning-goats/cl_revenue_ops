"""
Capacity Planner Module for cl-revenue-ops

This module identifies "Winner" channels for capital injection
and "Loser" channels for capital redeployment (Close).
"""

from __future__ import annotations

import json
import math
import time
from statistics import median
from typing import Dict, List, Any, Optional
from pyln.client import Plugin
from .config import ChainCostDefaults
from .demand_flow import DemandFlowClassifier
from .utils import parse_msat, base_to_sats_floor, base_to_sats_ceil, sats_to_base


# Loser severity ranking for sorting (higher = worse)
_LOSER_SEVERITY = {
    "ZOMBIE": 3,
    "FIRE SALE": 2,
    "DEAD_CAPITAL": 2,
    "STAGNANT+HARD_REBAL": 2,
    "STAGNANT": 1,
}

_CLOSE_TX_VBYTES = 200
_DEFAULT_CLOSE_FEE_RESERVE_MULTIPLIER = 2.0

# F1 (audit CRITICAL): the legacy open-EV forecast assumed 30% of capacity
# turning over daily at 150 ppm — i.e. 45 ppm of capacity per DAY, which is
# 16,425 ppm/year gross, 10-50x observed reality. It approved nearly every
# open at max size and channel sizing was effectively decided by this
# constant. The forecast is now anchored to the node's realized
# revenue-per-capacity; the old constant is retained ONLY as a hard ceiling
# on that anchor and as a bootstrap when the node has no routing history.
LEGACY_FORECAST_DAILY_PPM_CEILING = 45.0

# Discount applied to the node's observed median per-day revenue ppm when
# forecasting a brand-new peer with no track record of its own.
NEW_PEER_DISCOUNT = 0.5

# Assumed average fee rate used to convert forecast revenue into forecast
# volume. The revenue side and the rebalance-cost side of the open-EV model
# must share one volume model (audit: they previously diverged).
ASSUMED_AVG_FEE_PPM = 150.0

# Strategy weights for score normalization (higher = more trusted signal)
STRATEGY_WEIGHTS = {
    "winner": 1.0,
    "demand_flow": 1.0,
    "hive": 0.9,
    "route_pair": 0.85,
    "graph": 0.8,
    "neighbor": 0.9,
}


class CapacityPlanner:
    """
    Identifies capital redeployment opportunities to maximize yield.
    """

    def __init__(self, plugin: Plugin, profitability_analyzer, flow_analyzer, policy_manager=None, config=None):
        self.plugin = plugin
        self.profitability = profitability_analyzer
        self.flow = flow_analyzer
        self.policy_manager = policy_manager
        self.config = config
        self.rebalancer = None
        self.hive_hints = None  # Injected by main plugin
        self.data_service = None  # Injected by main plugin
        # Unified budget provider: injected by main plugin for cross-module budget gating
        self.global_budget_limit_provider = None
        self.external_liquidity_cost_provider = None
        self._capex_engine = None
        self._capital_efficiency = None
        # Coordination signals: cached from last execute_cycle for Boltz integration
        self._last_loser_scids: set = set()        # SCIDs with action=CLOSE
        self._last_funding_deficit_sats: int = 0    # On-chain shortfall for next open
        self._last_best_candidate: dict = {}        # Top candidate that needs funding
        self._last_cycle_ts: int = 0
        # Demand-flow classifier state (populated per cycle)
        self._demand_flow_profiles: Dict[str, Any] = {}
        self._demand_flow_sink_adjacent: set = set()
        # Capital recycling state
        self._last_preferred_loop_out_scid: str | None = None
        self._last_preferred_loop_out_reason: str = ""
        self._pending_recycle: dict | None = None
        # Per-cycle gossip cache (cleared at start of each execute_cycle)
        self._cycle_nodes_by_id: Dict[str, dict] = {}
        self._cycle_channels_dest: Dict[str, list] = {}
        self._cycle_channels_source: Dict[str, list] = {}
        # F1: per-cycle cached revenue anchor (1-tuple when computed, else None)
        self._observed_daily_ppm_cache: tuple | None = None

    def set_capex_engine(self, engine: 'CapexBudgetEngine'):
        """Inject the unified capex budget engine."""
        self._capex_engine = engine

    def set_capital_efficiency(self, analyzer) -> None:
        """Inject the capital-efficiency analyzer."""
        self._capital_efficiency = analyzer

    def set_rebalancer(self, rebalancer) -> None:
        """Inject the rebalance engine used for diagnostic shocks."""
        self.rebalancer = rebalancer

    def get_boltz_coordination(self) -> Dict[str, Any]:
        """Return signals for Boltz planner coordination.

        Called by the Boltz auto-cycle to avoid working against the capacity planner.
        """
        return {
            "loser_scids": set(self._last_loser_scids),
            "funding_deficit_sats": self._last_funding_deficit_sats,
            "best_candidate": dict(self._last_best_candidate) if self._last_best_candidate else None,
            "cycle_ts": self._last_cycle_ts,
            "preferred_loop_out_scid": self._last_preferred_loop_out_scid,
            "preferred_loop_out_reason": self._last_preferred_loop_out_reason,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return planner status for RPC query."""
        db = self.profitability.database if self.profitability else None
        cfg = self.config
        hive_open_count = 0
        if self.hive_hints is not None:
            try:
                hive_open_count = len(self.hive_hints.get_open_candidates())
            except Exception:
                pass
        return {
            "enabled": getattr(cfg, 'planner_enabled', False) if cfg else False,
            "dry_run": getattr(cfg, 'planner_dry_run', False) if cfg else False,
            "execute_closes": getattr(cfg, 'planner_execute_closes', False) if cfg else False,
            "candidate_pool_size": len(db.get_planner_candidates()) if db else 0,
            "recent_actions": db.get_planner_actions(limit=5) if db else [],
            "hive_open_candidates": hive_open_count,
            "metabolic_planner_influence": self.get_metabolic_planner_influence_debug(),
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a strategic redeployment report.
        """
        mempool_rec = self._get_mempool_recommendation()

        # Fetch analyses once and pass to both identification methods
        all_profitability = self.profitability.analyze_all_channels()
        all_flow = self.flow.analyze_all_channels()
        self._seed_revenue_anchor(all_profitability)

        winners = self._identify_winners(all_profitability, all_flow)
        losers = self._identify_losers(all_profitability, all_flow)

        recommendations = self._generate_recommendations(winners, losers)

        summary = {
            "winner_count": len(winners),
            "loser_count": len(losers),
            "recommendation_count": len(recommendations),
            "total_winner_capacity_sats": sum(w.get("capacity", 0) for w in winners),
            "total_loser_capacity_sats": sum(l.get("capacity", 0) for l in losers),
            "actionable_closures": sum(1 for l in losers if l.get("action") == "CLOSE"),
            "pending_defibrillation": sum(1 for l in losers if l.get("action") == "DEFIBRILLATE"),
        }

        return {
            "timestamp": int(time.time()),
            "mempool_recommendation": mempool_rec,
            "summary": summary,
            "winners": winners,
            "losers": losers,
            "recommendations": recommendations,
        }

    def _close_execution_enabled(self, cfg) -> bool:
        """Return True only when live close execution is explicitly allowed."""
        if not getattr(cfg, "planner_execute_closes", False):
            return False
        return getattr(cfg, "planner_max_closes_per_cycle", 1) > 0

    def _defibrillation_limit(self, cfg) -> int:
        """Return the per-cycle diagnostic rebalance limit."""
        raw_limit = getattr(cfg, "planner_max_defibrillations_per_cycle", None)
        if isinstance(raw_limit, bool):
            return int(raw_limit)
        if isinstance(raw_limit, (int, float)):
            return max(0, int(raw_limit))
        return 1

    def _init_cycle_cache(self):
        """Clear per-cycle caches. Node info is fetched lazily via _get_cached_node()."""
        self._cycle_channels_dest.clear()
        self._cycle_channels_source.clear()
        self._cycle_nodes_by_id.clear()
        self._observed_daily_ppm_cache = None

    def _get_cached_channels(self, peer_id: str, direction: str = "destination") -> list:
        """Get listchannels result, cached for this cycle."""
        cache = self._cycle_channels_dest if direction == "destination" else self._cycle_channels_source
        if peer_id in cache:
            return cache[peer_id]
        try:
            if direction == "destination":
                result = (self.data_service.get_channels(destination=peer_id) if self.data_service else self.plugin.rpc.listchannels(destination=peer_id)).get("channels", [])
            else:
                result = (self.data_service.get_channels(source=peer_id) if self.data_service else self.plugin.rpc.listchannels(source=peer_id)).get("channels", [])
        except Exception:
            result = []
        cache[peer_id] = result
        return result

    def _get_cached_node(self, peer_id: str) -> dict | None:
        """Get node info, preferring preloaded dict, falling back to RPC."""
        if peer_id in self._cycle_nodes_by_id:
            return self._cycle_nodes_by_id[peer_id]
        try:
            rpc_result = self.data_service.get_node_info(peer_id) if self.data_service else self.plugin.rpc.listnodes(id=peer_id)
            nodes = rpc_result.get("nodes", [])
            if nodes:
                self._cycle_nodes_by_id[peer_id] = nodes[0]
                return nodes[0]
        except Exception:
            pass
        return None

    def _check_portfolio_balance_gate(self, channels: list) -> str:
        """Check aggregate fleet liquidity balance. Returns state string.

        States:
            healthy     (< 70% local)  — all opens permitted
            watch       (70-85% local) — opens permitted, log warning
            constrained (85-95% local) — only dual-funded or sink-targeting opens
            blocked     (>= 95% local) — no new outbound opens
        """
        total_local = 0
        total_capacity = 0
        for ch in channels:
            if ch.get("state") != "CHANNELD_NORMAL":
                continue
            to_us = parse_msat(ch.get("to_us_msat", 0))
            total = parse_msat(ch.get("total_msat", 0))
            total_local += to_us
            total_capacity += total

        if total_capacity == 0:
            return "healthy"

        local_pct = (total_local / total_capacity) * 100

        if local_pct >= 95:
            return "blocked"
        elif local_pct >= 85:
            return "constrained"
        elif local_pct >= 70:
            return "watch"
        else:
            return "healthy"

    def execute_cycle(self, cfg=None) -> Dict[str, Any]:
        """Main timer-driven cycle. Evaluates and executes open/close decisions."""
        if cfg is None:
            cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        if not cfg.planner_enabled:
            return {"skipped": True, "reason": "planner disabled"}

        self._init_cycle_cache()

        # Compute capex allocations for this cycle
        if self._capex_engine:
            try:
                self._capex_engine.compute_allocations()
            except Exception as e:
                self.plugin.log(f"Capex engine allocation failed: {e}", level='warn')

        summary = {
            "opens": [],
            "defibrillations": [],
            "closes": [],
            "skipped_reasons": [],
            "evaluated_open_candidates": [],
        }

        # 1. Check fee gate
        fee_ok, fee_reason = self._check_fee_gate(cfg)
        if not fee_ok:
            summary["skipped_reasons"].append(fee_reason)

        # 2. Fetch analysis data
        all_profitability = self.profitability.analyze_all_channels()
        all_flow = self.flow.analyze_all_channels()
        self._seed_revenue_anchor(all_profitability)

        # 3. Identify winners and losers
        winners = self._identify_winners(all_profitability, all_flow)
        losers = self._identify_losers(all_profitability, all_flow)

        # Publish loser SCIDs for Boltz coordination (avoid loop-out through doomed channels)
        self._last_loser_scids = {
            l.get("scid") for l in losers
            if l.get("action") == "CLOSE" and l.get("scid")
        }
        self._last_cycle_ts = int(time.time())

        # 4. Defibrillate stagnant losers (bounded; budget/capex enforced in rebalancer)
        defibrillations_this_cycle = 0
        defibrillation_limit = self._defibrillation_limit(cfg)
        defibrillate = [l for l in losers if l.get("action") == "DEFIBRILLATE"]
        sorted_defibrillate = sorted(defibrillate, key=lambda x: x.get("marginal_roi", 0))
        for loser in sorted_defibrillate:
            if defibrillations_this_cycle >= defibrillation_limit:
                break

            scid = loser.get("scid")
            peer_id = loser.get("peer_id")

            cooldown_ok, cooldown_reason = self._check_cooldown(peer_id)
            if not cooldown_ok:
                summary["skipped_reasons"].append(
                    f"Defibrillation cooldown for {scid}: {cooldown_reason}"
                )
                continue

            result = self._execute_defibrillation(scid, peer_id, cfg, loser.get("reason", ""))
            summary["defibrillations"].append({
                "scid": scid,
                "peer_id": peer_id,
                "reason": loser.get("reason", ""),
                "action_id": result.get("action_id"),
                "status": result.get("status", "unknown"),
            })
            defibrillations_this_cycle += 1

        # 5. Close worst losers (up to max_closes_per_cycle)
        close_execution_enabled = self._close_execution_enabled(cfg)
        closes_this_cycle = 0
        configured_close_limit = getattr(cfg, "planner_max_closes_per_cycle", 0)
        close_limit = configured_close_limit if configured_close_limit > 0 else None
        closeable = [l for l in losers if l.get("action") == "CLOSE"]
        sorted_closeable = sorted(closeable, key=lambda x: x.get("marginal_roi", 0))
        for loser in sorted_closeable:
            # Recommendation-only close logging should not be suppressed by the
            # executed-close budget. A zero limit means "no close execution";
            # recommendation mode still surfaces recommendations in that case.
            if close_limit is not None and closes_this_cycle >= close_limit:
                break
            scid = loser.get("scid")
            peer_id = loser.get("peer_id")

            # Check policy allows close
            close_ok, close_reason = self._check_close_allowed(peer_id)
            if not close_ok:
                summary["skipped_reasons"].append(
                    f"Close blocked for {scid}: {close_reason}"
                )
                continue

            if close_execution_enabled:
                guards_ok, guards_reason = self._check_safety_guards(
                    cfg, "close", peer_id
                )
                if not guards_ok:
                    summary["skipped_reasons"].append(
                        f"Close guard failed for {scid}: {guards_reason}"
                    )
                    continue
            else:
                # Recommendation-only close logging still respects cooldowns.
                cooldown_ok, cooldown_reason = self._check_cooldown(peer_id)
                if not cooldown_ok:
                    summary["skipped_reasons"].append(
                        f"Close cooldown for {scid}: {cooldown_reason}"
                    )
                    continue

            result = self._execute_close(scid, peer_id, cfg, loser.get("reason", ""))
            summary["closes"].append({
                "scid": scid,
                "peer_id": peer_id,
                "reason": loser.get("reason", ""),
                "action_id": result.get("action_id"),
                "status": result.get("status", "unknown"),
            })
            closes_this_cycle += 1

        # 6. Check portfolio balance before opening
        portfolio_state = "healthy"
        try:
            peer_channels = (self.data_service.get_peer_channels() if self.data_service else self.plugin.rpc.listpeerchannels()).get("channels", [])
            portfolio_state = self._check_portfolio_balance_gate(peer_channels)
            if portfolio_state == "watch":
                self.plugin.log(f"Portfolio balance governor: {portfolio_state} — opens permitted with warning", level='info')
            elif portfolio_state in ("constrained", "blocked"):
                self.plugin.log(f"Portfolio balance governor: {portfolio_state} — restricting opens", level='info')
        except Exception as e:
            self.plugin.log(f"Portfolio balance check failed: {e}", level='debug')

        # 7. Discover and open channels (up to max_opens_per_cycle)
        candidates = []
        # Reset funding coordination before evaluating opens
        self._last_funding_deficit_sats = 0
        self._last_best_candidate = {}
        self._last_preferred_loop_out_scid = None
        self._last_preferred_loop_out_reason = ""
        if fee_ok and portfolio_state != "blocked":
            candidates = self._discover_peers(winners, all_profitability, all_flow)

            # Get available funds for sizing
            available_sats = 0
            try:
                funds = self.data_service.get_funds() if self.data_service else self.plugin.rpc.listfunds()
                confirmed = sum(
                    base_to_sats_floor(parse_msat(o.get("amount_msat", 0)))
                    for o in funds.get("outputs", [])
                    if o.get("status") == "confirmed"
                )
                min_reserve = getattr(cfg, 'min_wallet_reserve', 500000)
                available_sats = max(0, confirmed - min_reserve)
            except Exception as e:
                self.plugin.log(f"Cannot determine available funds: {e}", level='debug')

            # Detect funding deficit: best candidate needs more than available
            if candidates:
                top = max(candidates, key=lambda c: c.get("score", 0))
                min_open = getattr(cfg, 'planner_min_channel_sats', 500000)
                if available_sats < min_open:
                    self._last_funding_deficit_sats = max(0, min_open - available_sats)
                    self._last_best_candidate = {
                        "peer_id": top.get("peer_id"),
                        "score": top.get("score"),
                        "source": top.get("source"),
                        "min_amount_sats": min_open,
                    }

            # Exploration budget gate: check if budget can fund this cycle's opens
            exploration_budget_sats = 0
            estimated_open_cost = self._estimate_open_cost()
            if self._capex_engine:
                exploration_budget_sats = self._capex_engine.get_fleet_exploration_budget()
            max_opens = cfg.planner_max_opens_per_cycle
            remaining_exploration_budget_sats = exploration_budget_sats
            if remaining_exploration_budget_sats < estimated_open_cost and self._capex_engine:
                self.plugin.log(
                    f"PLANNER: Exploration budget {remaining_exploration_budget_sats} sats "
                    f"< open cost {estimated_open_cost} sats — skipping opens",
                    level='info',
                )

            opens_this_cycle = 0
            evaluated_candidates = []
            score_ranked_candidates = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
            sizing_slots = max(1, min(max_opens, len(score_ranked_candidates)))

            def sizing_pool_for(candidate):
                """Size against executable slots, not every discovered alternative."""
                peer_id = candidate.get("peer_id")
                pool = [candidate]
                for other in score_ranked_candidates:
                    if other.get("peer_id") == peer_id:
                        continue
                    pool.append(other)
                    if len(pool) >= sizing_slots:
                        break
                return pool

            for candidate in sorted(candidates, key=lambda c: c.get("score", 0), reverse=True):
                peer_id = candidate["peer_id"]
                channel_size = self._size_channel(candidate, sizing_pool_for(candidate), available_sats, cfg)
                ev = self._calculate_open_ev(peer_id, channel_size, cfg)
                summary["evaluated_open_candidates"].append({
                    "peer_id": peer_id,
                    "source": candidate.get("source", ""),
                    "score": round(float(candidate.get("score", 0) or 0), 6),
                    "amount_sats": channel_size,
                    "ev": round(ev, 0),
                })
                if ev <= 0:
                    summary["skipped_reasons"].append(
                        f"Negative EV ({ev:.0f}) for {peer_id[:16]}..."
                    )
                    continue
                evaluated = dict(candidate)
                evaluated["_planned_channel_size"] = channel_size
                evaluated["_planned_ev"] = ev
                evaluated_candidates.append(evaluated)

            ranked_candidates = sorted(
                evaluated_candidates,
                key=lambda c: (c.get("_planned_ev", 0), c.get("score", 0)),
                reverse=True,
            )
            for candidate in ranked_candidates:
                if opens_this_cycle >= max_opens:
                    break

                peer_id = candidate["peer_id"]

                # Exploration budget: check remaining budget covers the next open
                if self._capex_engine and remaining_exploration_budget_sats < estimated_open_cost:
                    summary["skipped_reasons"].append(
                        f"Exploration budget: {remaining_exploration_budget_sats} < open cost {estimated_open_cost}"
                    )
                    break  # No point checking more candidates

                if opens_this_cycle > 0:
                    # Recompute against updated available funds after prior successful opens.
                    channel_size = self._size_channel(candidate, sizing_pool_for(candidate), available_sats, cfg)
                    ev = self._calculate_open_ev(peer_id, channel_size, cfg)
                else:
                    channel_size = candidate.get("_planned_channel_size", 0)
                    ev = candidate.get("_planned_ev", 0)
                if ev <= 0:
                    summary["skipped_reasons"].append(
                        f"Negative EV ({ev:.0f}) for {peer_id[:16]}..."
                    )
                    continue

                # Constrained portfolio: only allow sink-adjacent or dual-fund candidates
                if portfolio_state == "constrained":
                    is_sink_adj = peer_id in self._demand_flow_sink_adjacent
                    node_info = self._get_cached_node(peer_id)
                    has_lads = bool(node_info and node_info.get("option_will_fund"))
                    if not is_sink_adj and not has_lads:
                        summary["skipped_reasons"].append(
                            f"Constrained: {peer_id[:16]}... not sink-adjacent or dual-fund"
                        )
                        continue

                # Check safety guards
                guards_ok, guards_reason = self._check_safety_guards(
                    cfg, "open", peer_id, amount_sats=channel_size
                )
                if not guards_ok:
                    summary["skipped_reasons"].append(
                        f"Guard failed for {peer_id[:16]}...: {guards_reason}"
                    )
                    continue

                # Execute open
                result = self._execute_open(
                    peer_id, channel_size, cfg,
                    reason=candidate.get("reason", "Planner cycle open"),
                )
                summary["opens"].append({
                    "peer_id": peer_id,
                    "amount_sats": channel_size,
                    "ev": round(ev, 0),
                    "result": result.get("status", "unknown"),
                    "action_id": result.get("action_id"),
                })
                status = result.get("status", "unknown")
                if status in ("completed", "dry_run"):
                    opens_this_cycle += 1
                    available_sats = max(0, available_sats - channel_size)
                    if self._capex_engine:
                        remaining_exploration_budget_sats = max(
                            0, remaining_exploration_budget_sats - estimated_open_cost
                        )

        # 8. Update candidate pool
        self._update_candidate_pool(candidates if fee_ok else [])

        # 9. Evaluate recycling opportunities (works even when blocked)
        if candidates and losers:
            recycle_plan = self._evaluate_recycle_opportunities(losers, candidates, cfg)
            if recycle_plan:
                summary["recycle_opportunity"] = {
                    "close_scid": recycle_plan["loser"]["scid"],
                    "close_peer": recycle_plan["loser"]["peer_id"][:16],
                    "open_peer": recycle_plan["candidate"]["peer_id"][:16],
                    "recycle_ev": round(recycle_plan["recycle_ev"]),
                }
                self.plugin.log(
                    f"Recycle opportunity: close {recycle_plan['loser']['scid']} → "
                    f"open {recycle_plan['candidate']['peer_id'][:16]}... "
                    f"(EV={recycle_plan['recycle_ev']:.0f} sats)",
                    level='info',
                )

        summary["portfolio_state"] = portfolio_state
        summary["winner_count"] = len(winners)
        summary["loser_count"] = len(losers)
        summary["candidate_count"] = len(candidates)
        summary["timestamp"] = int(time.time())

        # Log skip reasons so we can diagnose why opens/closes aren't happening
        if summary["skipped_reasons"]:
            for reason in summary["skipped_reasons"]:
                self.plugin.log(f"PLANNER SKIP: {reason}", level='info')
        if not candidates and fee_ok and portfolio_state != "blocked":
            self.plugin.log(
                f"PLANNER: No open candidates discovered "
                f"(winners={len(winners)}, losers={len(losers)})",
                level='info',
            )

        return summary

    def _get_mempool_recommendation(self) -> str:
        """Query feerates and return a graduated recommendation based on opening costs."""
        try:
            feerates = self.data_service.get_feerates(style="perkb") if self.data_service else self.plugin.rpc.feerates(style="perkb")
            perkb = feerates.get("perkb", {})
            # opening fee in perkb, we want sat/vB (divide by 1000)
            opening_kvb = perkb.get("opening", 1000)
            sat_per_vb = opening_kvb / 1000.0

            if sat_per_vb > 100:
                return f"HOLD: On-chain fees too high for efficient channel operations ({sat_per_vb:.0f} sat/vB)."
            elif sat_per_vb > 50:
                return f"CAUTION: On-chain fees are elevated ({sat_per_vb:.0f} sat/vB). Consider waiting for lower fees."
            return f"PROCEED: Fee environment is favorable ({sat_per_vb:.0f} sat/vB)."
        except Exception as e:
            self.plugin.log(f"Error checking mempool for capacity report: {e}", level='debug')
            return "UNKNOWN: Could not fetch feerates."

    def _identify_winners(self, all_profitability, all_flow) -> List[Dict[str, Any]]:
        """
        Identify high-performing channels that are capacity-constrained.

        Enriches each winner with additional data signals for downstream
        prioritization (peer discovery, channel sizing):
        - velocity_urgency: Kalman velocity > 0.1 indicates rapid draining
        - congestion_urgent: HTLC slots saturated (>80%)
        - sourced_fee_contribution_sats: Inbound fee contribution to other channels
        - channel_role: Source/sink/router classification
        - dts_posterior_mean: DTS optimal fee estimate (proven fee-earner signal)
        """
        winners = []

        for scid, prof in all_profitability.items():
            flow_metrics = all_flow.get(scid)
            if not flow_metrics:
                continue

            # SCID formatting check - ensure 'x' separator
            scid_display = scid.replace(':', 'x')

            # Logic: (Marginal ROI > 20%) AND (Turnover > 0.5) AND (Flow Ratio > 0.8 OR Flow Ratio < -0.8)
            # Safe turnover calculation to prevent ZeroDivisionError
            capacity = prof.capacity_sats or 0
            turnover = flow_metrics.daily_volume / capacity if capacity > 0 else 0

            # Rebalance difficulty penalty: low success rate reduces effective ROI
            success_data = self.profitability.database.get_channel_rebalance_success_rate(scid, 30)
            rebal_penalty = 0.0
            # M9 FIX: Use .get() to prevent KeyError if dict missing expected keys
            if success_data and success_data.get('total', 0) >= 3:
                sr = success_data.get('success_rate', 1.0)
                if sr < 0.5:
                    rebal_penalty = (0.5 - sr) * 50  # Up to 25% ROI penalty

            effective_roi = prof.marginal_roi_percent - rebal_penalty
            sr_val = success_data.get('success_rate', 1.0) if success_data and success_data.get('total', 0) >= 3 else 1.0
            rebal_difficulty = round(1.0 - sr_val, 2)

            # Kalman velocity urgency: positive velocity on source = draining faster
            kalman_velocity = getattr(flow_metrics, 'kalman_velocity', 0.0) or 0.0
            velocity_urgency = kalman_velocity > 0.1

            # Congestion urgency: HTLC slots saturated (>80%)
            congestion_urgent = bool(getattr(flow_metrics, 'is_congested', False))

            # Sourced fee contribution: inbound channels generating fees for others
            sourced_contribution = 0
            if hasattr(prof, 'revenue') and hasattr(prof.revenue, 'sourced_fee_contribution_sats'):
                sourced_contribution = prof.revenue.sourced_fee_contribution_sats or 0

            # Channel role for downstream prioritization
            channel_role = getattr(prof, 'channel_role', None)
            channel_role_str = channel_role.value if hasattr(channel_role, 'value') else str(channel_role) if channel_role else None

            # DTS posterior mean: high mean with low variance = proven fee-earner
            dts_mean = None
            try:
                fee_state = self.profitability.database.get_fee_strategy_state(scid)
                if fee_state:
                    v2_json_str = fee_state.get('v2_state_json', '{}') or '{}'
                    v2_data = json.loads(v2_json_str) if isinstance(v2_json_str, str) else v2_json_str
                    thompson_state = v2_data.get('thompson_state', {})
                    if thompson_state:
                        dts_mean = thompson_state.get('posterior_mean')
            except Exception:
                pass

            if (effective_roi > 20.0 and
                turnover > 0.5 and
                (flow_metrics.flow_ratio > 0.8 or flow_metrics.flow_ratio < -0.8)):

                winners.append({
                    "scid": scid_display,
                    "peer_id": prof.peer_id,
                    "roi": round(effective_roi, 2),
                    "flow_ratio": round(flow_metrics.flow_ratio, 4),
                    "turnover": round(turnover, 4),
                    "capacity": prof.capacity_sats,
                    "rebal_difficulty": rebal_difficulty,
                    "velocity_urgency": velocity_urgency,
                    "congestion_urgent": congestion_urgent,
                    "sourced_fee_contribution_sats": sourced_contribution,
                    "channel_role": channel_role_str,
                    "dts_posterior_mean": round(dts_mean, 1) if dts_mean is not None else None,
                })

        return winners

    def _identify_losers(self, all_profitability, all_flow) -> List[Dict[str, Any]]:
        """
        Identify poor-performing channels for capital extraction.

        Enriched with additional data signals for smarter closure decisions:
        - Hard bleeder bypass: structurally unprofitable channels skip defibrillation gate
        - Channel role protection: INBOUND_GATEWAYs require much worse ROI for closure
        - Kalman regime change deferral: demotes CLOSE to DEFIBRILLATE if behavior shifting
        - Kalman confidence gate: skips closure for channels with unreliable flow data
        - Peer uptime: added to loser dict for downstream decision-making
        """
        losers = []

        from .profitability_analyzer import ProfitabilityClass, ChannelRole

        # Route-pair protection: channels on top revenue routes get higher closure bar
        route_pair_channels = set()
        db = self.profitability.database if self.profitability else None
        dead_capital_stages = {}
        channel_efficiencies = {}
        if self._capital_efficiency is not None:
            try:
                fleet_efficiency = self._capital_efficiency.analyze()
                channel_efficiencies = getattr(fleet_efficiency, "channel_efficiencies", {}) or {}
            except Exception:
                channel_efficiencies = {}
            if db is not None:
                try:
                    dead_capital_stages = db.get_dead_capital_stages() or {}
                except Exception:
                    dead_capital_stages = {}
        if db:
            try:
                pairs = db.get_top_route_pairs(days=30, min_forwards=3, limit=10)
                for p in pairs:
                    for key in ("in_channel", "out_channel"):
                        ch = str(p.get(key, "")).replace(":", "x")
                        if ch:
                            route_pair_channels.add(ch)
            except Exception:
                pass

        # Get bleeder classification for all channels
        bleeders = {}
        try:
            bleeder_list = self.profitability.identify_bleeders_v2() or []
            # Convert list to dict keyed by channel_id for O(1) lookup
            for b in bleeder_list:
                cid = getattr(b, 'channel_id', None)
                if cid:
                    bleeders[cid] = b
        except Exception:
            pass

        for scid, prof in all_profitability.items():
            flow_metrics = all_flow.get(scid)
            scid_display = scid.replace(':', 'x')
            channel_efficiency = channel_efficiencies.get(scid) or channel_efficiencies.get(scid_display)

            if dead_capital_stages.get(scid_display) and not getattr(channel_efficiency, "is_dead_capital", False):
                try:
                    db.delete_dead_capital_stage(scid_display)
                except Exception:
                    pass

            dead_capital_loser = self._build_dead_capital_loser(
                scid_display,
                prof,
                channel_efficiency,
                dead_capital_stages,
                flow_metrics=flow_metrics,
                route_pair_channels=route_pair_channels,
            )
            if dead_capital_loser is not None:
                losers.append(dead_capital_loser)
                continue

            # Fetch diagnostic stats from DB
            diag_stats = self.profitability.database.get_diagnostic_rebalance_stats(scid, days=14)
            attempt_count = diag_stats.get("attempt_count", 0)

            # Rebalance difficulty scoring from success rate history
            success_data = self.profitability.database.get_channel_rebalance_success_rate(scid, 30)
            rebal_difficulty = 0.0
            if success_data and success_data.get('total', 0) >= 3:
                rebal_difficulty = 1.0 - success_data.get('success_rate', 1.0)  # 0=easy, 1=impossible

            # Hive member protection -- never recommend closing fleet channels
            if self.hive_hints is not None:
                try:
                    if self.hive_hints.is_hive_member(prof.peer_id):
                        continue
                except Exception:
                    pass

            # Protective closure gates (Kalman confidence, inbound gateway,
            # sourced-fee contribution, route-pair). Shared with the
            # dead-capital staging pipeline via _close_protection_reason.
            if self._close_protection_reason(scid_display, prof, flow_metrics, route_pair_channels):
                continue

            # Hard bleeder bypass -- skip defibrillation gate
            bleeder_info = bleeders.get(scid)
            is_hard_bleeder = False
            if bleeder_info is not None:
                if hasattr(bleeder_info, 'is_hard_bleeder'):
                    is_hard_bleeder = bool(bleeder_info.is_hard_bleeder)
                elif isinstance(bleeder_info, dict):
                    is_hard_bleeder = bool(bleeder_info.get('is_hard_bleeder', False))

            # Hive reputation closure signal — peer flagged by fleet intelligence
            hive_closure_flagged = False
            if self.hive_hints is not None:
                try:
                    checker = getattr(self.hive_hints, "is_closure_recommended_fresh", None)
                    if not callable(checker):
                        checker = getattr(self.hive_hints, "is_closure_recommended", None)
                    hive_closure_flagged = bool(checker(prof.peer_id)) if callable(checker) else False
                except Exception:
                    pass

            # Logic 1: FIRE SALE mode (Zombie or Deeply Underwater)
            # Guard: require flow data before recommending closure (matches _identify_winners).
            # Without flow data we can't assess channel viability, so demote to DEFIBRILLATE at most.
            is_fire_sale = False
            fire_sale_reason = None
            if prof.days_open > 90 and flow_metrics is not None:
                if prof.classification == ProfitabilityClass.ZOMBIE:
                    is_fire_sale = True
                    fire_sale_reason = "ZOMBIE"
                elif prof.classification == ProfitabilityClass.UNDERWATER and prof.marginal_roi_percent < -50.0:
                    # Use marginal ROI (operational, 30-day trailing) to avoid sunk cost fallacy.
                    # A channel covering its rebalance costs shouldn't be closed just because
                    # the opening fee hasn't been recouped yet.
                    is_fire_sale = True
                    fire_sale_reason = "FIRE SALE"

            # Logic 2: Stagnant balanced channels (turnover < 0.0015)
            # PROTECTION: Only a loser if stagnant AND marginal_roi_percent < 10.0%
            is_stagnant = False
            if flow_metrics:
                # Safe ratio calculations to prevent ZeroDivisionError
                cap = flow_metrics.capacity or 0
                turnover = flow_metrics.daily_volume / cap if cap > 0 else 0
                # A channel with near-zero flow_ratio has balanced flow direction;
                # combined with low turnover this signals stagnancy.
                is_balanced = abs(flow_metrics.flow_ratio) < 0.2
                if is_balanced and (turnover < 0.0015):
                    if prof.marginal_roi_percent < 10.0:
                        is_stagnant = True

            # High rebalance difficulty makes losers worse — harder to recover
            if rebal_difficulty > 0.7 and not is_fire_sale and is_stagnant:
                is_fire_sale = True
                fire_sale_reason = "STAGNANT+HARD_REBAL"

            # Hive reputation closure: peer has dangerous behavior (force closes,
            # low HTLC success). Combined with underwater/stagnant, escalate.
            # On its own (profitable channel), flag but don't force close.
            if hive_closure_flagged and not is_fire_sale:
                if prof.marginal_roi_percent < 0:
                    is_fire_sale = True
                    fire_sale_reason = "REPUTATION+UNDERWATER"
                elif is_stagnant:
                    is_fire_sale = True
                    fire_sale_reason = "REPUTATION+STAGNANT"

            # Remote-opened channels are "free" capacity — raise the bar for closing.
            # They cost us nothing to acquire, so only close if deeply underwater.
            opener = getattr(prof, 'opener', 'local')
            if opener == 'remote' and is_fire_sale and not is_stagnant:
                if prof.marginal_roi_percent > -75.0:
                    # Skip close recommendation for remote channels unless deeply underwater
                    continue

            # Kalman regime change deferral -- if the channel's flow behavior
            # has fundamentally shifted, it may be improving.
            regime_change = False
            if flow_metrics:
                rc = getattr(flow_metrics, 'kalman_regime_change', False)
                if isinstance(rc, bool):
                    regime_change = rc
                else:
                    regime_change = bool(rc) if rc is not None else False

            # Peer uptime -- low uptime + poor ROI strengthens close signal
            uptime_pct = None
            try:
                uptime_pct = self.profitability.database.get_peer_uptime_percent(
                    prof.peer_id, duration_seconds=168 * 3600)
            except Exception:
                pass

            if is_fire_sale or is_stagnant:
                # PROTECTION: A channel cannot be recommended for "Close"
                # until the diagnostic_rebalance has been attempted at least twice in the last 14 days.
                # EXCEPTION: Hard bleeders bypass the defibrillation gate -- they are
                # structurally unprofitable (rebalance cost > 2x revenue AND net loss > 1000 sats).
                # Accounting v2.0: Include estimated closure cost
                estimated_closure_cost = self._estimate_close_cost()
                reason = fire_sale_reason if is_fire_sale else "STAGNANT"

                if is_hard_bleeder or attempt_count >= 2:
                    action = "CLOSE"
                    # Regime change demotes CLOSE to DEFIBRILLATE (unless hard bleeder,
                    # which is structurally unprofitable regardless of regime shifts)
                    if regime_change and not is_hard_bleeder:
                        action = "DEFIBRILLATE"
                        reason = f"{reason} (REGIME CHANGE)"
                else:
                    action = "DEFIBRILLATE"
                    reason = f"{reason} (NEEDS DEFIBRILLATOR)"

                losers.append({
                    "scid": scid_display,
                    "peer_id": prof.peer_id,
                    "reason": reason,
                    "roi": round(prof.roi_percent, 2),
                    "marginal_roi": round(prof.marginal_roi_percent, 2),
                    "classification": prof.classification.value if hasattr(prof.classification, 'value') else str(prof.classification),
                    "capacity": prof.capacity_sats,
                    "estimated_closure_cost_sats": estimated_closure_cost,
                    "rebal_difficulty": round(rebal_difficulty, 2),
                    "opener": opener,
                    "action": action,
                    # Enrichment fields
                    "is_hard_bleeder": is_hard_bleeder,
                    "hive_closure_flagged": hive_closure_flagged,
                    "uptime_pct": round(uptime_pct, 1) if isinstance(uptime_pct, (int, float)) else None,
                    "regime_change": regime_change,
                    "is_fire_sale": is_fire_sale,
                    "marginal_profit_30d_sats": prof.marginal_profit_30d_sats,
                })

        return losers

    def _close_protection_reason(
        self,
        scid_display: str,
        prof,
        flow_metrics,
        route_pair_channels,
    ) -> Optional[str]:
        """Return why a channel must not be recommended for closure, or None.

        Single source of truth for the protective closure gates used by both
        the regular loser pipeline and the dead-capital staging pipeline:
        - Kalman confidence gate (unreliable flow data)
        - Inbound-gateway ROI protection
        - Sourced-fee-contribution protection
        - Route-pair (revenue route) protection
        """
        from .profitability_analyzer import ChannelRole

        # Hive member protection -- never recommend closing fleet channels
        if self.hive_hints is not None:
            try:
                if self.hive_hints.is_hive_member(prof.peer_id):
                    return "HIVE_MEMBER"
            except Exception:
                pass

        # Kalman confidence gate -- skip closure if data unreliable
        if flow_metrics:
            confidence = getattr(flow_metrics, 'confidence', 1.0)
            # Guard against non-numeric (e.g. MagicMock) or None
            if not isinstance(confidence, (int, float)):
                confidence = 1.0
            confidence = confidence or 1.0
            if confidence < 0.5:
                return "KALMAN_LOW_CONFIDENCE"  # Don't recommend closure with unreliable data

        # Channel role protection -- INBOUND_GATEWAYs source volume for all
        # outbound channels; closing one has outsized negative impact.
        channel_role = getattr(prof, 'channel_role', None)
        is_inbound_gateway = False
        try:
            if channel_role is not None:
                if isinstance(channel_role, ChannelRole):
                    is_inbound_gateway = channel_role == ChannelRole.INBOUND_GATEWAY
                elif hasattr(channel_role, 'value'):
                    is_inbound_gateway = channel_role.value in ('INBOUND_GATEWAY', 'inbound_gateway')
                else:
                    is_inbound_gateway = str(channel_role) in ('INBOUND_GATEWAY', 'inbound_gateway')
        except Exception:
            pass

        # Protect inbound gateways — they source traffic for the fleet
        if is_inbound_gateway and prof.marginal_roi_percent >= -30.0:
            return "INBOUND_GATEWAY"  # Protect inbound gateways (tighter threshold)

        # Protect channels that source significant fee contribution
        # regardless of their own ROI — they enable other channels' revenue.
        # Uses all-time sourced contribution from ChannelRevenue (not 30d window)
        # because channels with a history of sourcing are worth protecting even
        # if recent volume is lower. Avoids modifying ChannelProfitability dataclass.
        sourced_fee_sats = getattr(getattr(prof, 'revenue', None), 'sourced_fee_contribution_sats', 0)
        try:
            sourced_fee_sats = int(sourced_fee_sats)
        except (TypeError, ValueError):
            sourced_fee_sats = 0
        if sourced_fee_sats > 100 and prof.marginal_roi_percent > -50.0:
            return "SOURCED_FEE_CONTRIBUTION"  # Protect significant inbound sources

        # Route-pair protection: channels on proven revenue routes have network value
        # beyond their individual ROI.  Closing one kills the paired channel's revenue.
        # Hive corridor channels on revenue routes get extra protection.
        is_on_revenue_route = scid_display in (route_pair_channels or set())
        is_hive_corridor = False
        if self.hive_hints is not None:
            try:
                corridor_role = self.hive_hints.get_corridor_role(prof.peer_id)
                is_hive_corridor = corridor_role in {"owner", "secondary", "contested"}
            except Exception:
                pass
        if is_on_revenue_route:
            # Hive corridor channels on revenue routes: protect until -50% ROI
            # Regular revenue route channels: protect until -30% ROI
            route_close_threshold = -50.0 if is_hive_corridor else -30.0
            if prof.marginal_roi_percent > route_close_threshold:
                return "REVENUE_ROUTE"  # Protect revenue route channels

        return None

    def _build_dead_capital_loser(
        self,
        scid: str,
        prof,
        channel_efficiency,
        dead_capital_stages: Dict[str, Dict[str, Any]],
        flow_metrics=None,
        route_pair_channels=None,
    ) -> Dict[str, Any] | None:
        """Return staged dead-capital action when the analyzer flags a channel."""
        if channel_efficiency is None or not getattr(channel_efficiency, "is_dead_capital", False):
            return None

        # Protective closure gates apply to dead-capital staging too: a
        # historically valuable channel that merely went quiet must not be
        # staged into CLOSE on timeouts alone. FEE_REDUCE/DEFIBRILLATE stages
        # remain allowed for protected channels.
        close_protection = self._close_protection_reason(
            scid, prof, flow_metrics, route_pair_channels or set()
        )

        db = self.profitability.database if self.profitability else None
        now = int(time.time())
        opener = getattr(prof, "opener", "local")
        stage_timeout = 48 * 3600 if opener == "remote" else 24 * 3600
        stage_row = dead_capital_stages.get(scid, {})
        stage = str(
            getattr(channel_efficiency, "dead_capital_stage", "none")
            or stage_row.get("stage")
            or "none"
        )
        entered_at = int(stage_row.get("entered_at", 0) or 0)
        persist_stage = None

        close_blocked = False
        if stage == "none":
            action = "FEE_REDUCE"
            stage = "fee_reduction"
            entered_at = now
            persist_stage = stage
        elif stage == "fee_reduction" and entered_at and now - entered_at >= stage_timeout:
            action = "DEFIBRILLATE"
            stage = "defibrillation"
            entered_at = now
            persist_stage = stage
        elif stage == "fee_reduction":
            action = "FEE_REDUCE"
        elif stage == "defibrillation" and entered_at and now - entered_at >= stage_timeout:
            if close_protection:
                # Protections forbid CLOSE staging: hold at defibrillation
                # instead of advancing (and re-evaluate next cycle).
                action = "DEFIBRILLATE"
                close_blocked = True
            else:
                action = "CLOSE"
                stage = "close"
                entered_at = now
                persist_stage = stage
        elif stage == "defibrillation":
            action = "DEFIBRILLATE"
        else:
            if close_protection:
                # A previously persisted close stage is demoted back to
                # defibrillation while the channel is protected.
                action = "DEFIBRILLATE"
                stage = "defibrillation"
                entered_at = now
                persist_stage = stage
                close_blocked = True
            else:
                action = "CLOSE"
                stage = "close"

        if close_blocked:
            try:
                self.plugin.log(
                    f"DEAD CAPITAL: close staging blocked for {scid} "
                    f"(protection: {close_protection}); holding at DEFIBRILLATE",
                    level="info",
                )
            except Exception:
                pass

        if persist_stage is not None and db is not None:
            try:
                db.upsert_dead_capital_stage(scid, persist_stage, entered_at)
            except Exception:
                pass

        uptime_pct = None
        if db is not None:
            try:
                uptime_pct = db.get_peer_uptime_percent(
                    prof.peer_id, duration_seconds=168 * 3600
                )
            except Exception:
                pass

        classification = prof.classification.value if hasattr(prof.classification, "value") else str(prof.classification)
        return {
            "scid": scid,
            "peer_id": prof.peer_id,
            "reason": "DEAD_CAPITAL",
            "roi": round(getattr(prof, "roi_percent", 0.0), 2),
            "marginal_roi": round(getattr(prof, "marginal_roi_percent", 0.0), 2),
            "classification": classification,
            "capacity": getattr(prof, "capacity_sats", 0),
            "estimated_closure_cost_sats": self._estimate_close_cost(),
            "rebal_difficulty": 0.0,
            "opener": opener,
            "action": action,
            "is_hard_bleeder": False,
            "hive_closure_flagged": False,
            "uptime_pct": round(uptime_pct, 1) if isinstance(uptime_pct, (int, float)) else None,
            "regime_change": False,
            "is_fire_sale": False,
            "marginal_profit_30d_sats": getattr(prof, "marginal_profit_30d_sats", 0),
            "dead_capital_stage": stage,
            "close_protection": close_protection if close_blocked else None,
        }

    def _generate_recommendations(self, winners: List[Dict], losers: List[Dict]) -> List[str]:
        """
        Create actionable recommendations pairing winners and losers.

        Non-fire-sale, non-hard-bleeder CLOSE losers are demoted to DEFIBRILLATE
        if redeployment EV is negative (capital better off staying).
        """
        recommendations = []

        # Sort winners by ROI (descending)
        sorted_winners = sorted(winners, key=lambda x: x['roi'], reverse=True)

        # EV-based demotion: check each CLOSE loser
        for loser in losers:
            if loser.get("action") != "CLOSE":
                continue
            # Fire sale and hard bleeders bypass EV check
            if loser.get("is_fire_sale", False) or loser.get("is_hard_bleeder", False):
                continue
            # Compute redeployment EV
            try:
                ev, best_peer, winner_ev = self._calculate_redeployment_ev(
                    loser, sorted_winners, self.config
                )
                loser["redeployment_ev"] = round(ev, 0)
                loser["best_winner_peer"] = best_peer
                loser["winner_ev"] = round(winner_ev, 0)

                if ev <= 0:
                    loser["action"] = "DEFIBRILLATE"
                    loser["reason"] = f"{loser['reason']} (NO PROFITABLE REDEPLOYMENT)"
            except Exception:
                pass  # On error, keep original action

        # Re-separate after demotion
        fee_reduce = [l for l in losers if l.get("action") == "FEE_REDUCE"]
        defibrillate = [l for l in losers if l.get("action") == "DEFIBRILLATE"]
        closeable = [l for l in losers if l.get("action") == "CLOSE"]

        # Sort closeable losers by severity then worst ROI first
        sorted_closeable = sorted(
            closeable,
            key=lambda x: (_LOSER_SEVERITY.get(x.get('reason', ''), 0), -x.get('roi', 0)),
            reverse=True,
        )

        # Pair winners with closeable losers for capital redeployment
        closeable_idx = 0
        for winner in sorted_winners:
            if closeable_idx < len(sorted_closeable):
                loser = sorted_closeable[closeable_idx]
                closeable_idx += 1

                ev_note = ""
                if "redeployment_ev" in loser:
                    ev_note = f" EV: {loser['redeployment_ev']:.0f} sats."

                recommendations.append(
                    f"REDEPLOYMENT: Close channel {loser['scid']} ({loser['reason']}) "
                    f"and redeploy funds to {winner['scid']} (ROI: {winner['roi']:.1f}%).{ev_note}"
                )
            else:
                recommendations.append(
                    f"GROWTH POTENTIAL: {winner['scid']} is a high ROI winner ({winner['roi']:.1f}% ROI). "
                    f"Consider adding more capital."
                )

        # Recommend unpaired closeable losers
        for loser in sorted_closeable[closeable_idx:]:
            recommendations.append(
                f"CLOSE CANDIDATE: {loser['scid']} ({loser['reason']}, {loser['roi']:.1f}% ROI). "
                f"No winner available for pairing — consider closing to free capital."
            )

        # Defibrillation alerts
        for loser in defibrillate:
            recommendations.append(
                f"DEFIBRILLATE: {loser['scid']} ({loser['reason']}, {loser['roi']:.1f}% ROI). "
                f"Diagnostic rebalance required before closure can be recommended."
            )

        for loser in fee_reduce:
            recommendations.append(
                f"FEE REDUCE: {loser['scid']} ({loser['reason']}, {loser['roi']:.1f}% ROI). "
                f"Lower fees to the floor and wait one cycle for recovery."
            )

        return recommendations

    def _discover_from_winners(self, winners: List[Dict]) -> List[Dict]:
        """Strategy 1: Propose existing winners for additional channel opens."""
        candidates = []
        for winner in winners:
            if winner["roi"] > 30.0:  # Only very strong winners
                candidates.append({
                    "peer_id": winner["peer_id"],
                    "source": "winner",
                    "score": winner["roi"] / 100.0,
                    "reason": f"Existing winner with {winner['roi']:.1f}% ROI",
                    "scid": winner.get("scid"),
                })
        return candidates

    def _discover_from_neighbors(self, all_profitability) -> List[Dict]:
        """Strategy 2: Find neighbors of top-earning peers (CLBOSS-inspired)."""
        candidates = []

        # Get our own node ID
        try:
            our_node_id = self.data_service.get_node_id() if self.data_service else self.plugin.rpc.getinfo().get("id")
        except Exception:
            return []

        # Get existing peer_ids to exclude
        existing_peers = set()
        for prof in all_profitability.values():
            if hasattr(prof, 'peer_id') and prof.peer_id:
                existing_peers.add(prof.peer_id)

        if self._capital_efficiency is None:
            # Sort channels by marginal ROI, take top 3
            sorted_channels = sorted(
                all_profitability.values(),
                key=lambda p: getattr(p, 'marginal_roi_percent', 0),
                reverse=True
            )[:3]

            for patron in sorted_channels:
                patron_peer_id = getattr(patron, 'peer_id', None)
                if not patron_peer_id:
                    continue

                try:
                    channels = {"channels": self._get_cached_channels(patron_peer_id, "source")}
                    patron_roi = getattr(patron, 'marginal_roi_percent', 0)
                    # Score neighbors by capacity and fee competitiveness, not random order
                    scored_neighbors = []
                    for ch in channels.get("channels", []):
                        dest = ch.get("destination")
                        if not dest or dest == our_node_id or dest in existing_peers:
                            continue
                        cap = base_to_sats_floor(parse_msat(ch.get("amount_msat", 0)))
                        fee_ppm = ch.get("fee_per_millionth", 0) or 0
                        # Skip expensive or tiny channels (only filter when data is available)
                        if fee_ppm > 1500:
                            continue
                        if cap > 0 and cap < 200000:
                            continue
                        # Score: patron ROI scaled, boosted by capacity and low fees
                        base = max(patron_roi / 200.0, 0.1)
                        if cap > 5_000_000:
                            base *= 1.15
                        if 0 < fee_ppm < 100:
                            base *= 1.10
                        scored_neighbors.append((dest, base))
                    scored_neighbors.sort(key=lambda x: x[1], reverse=True)

                    # Take top 5 neighbors per patron
                    for neighbor_id, neighbor_score in scored_neighbors[:5]:
                        candidates.append({
                            "peer_id": neighbor_id,
                            "source": "neighbor",
                            "score": neighbor_score,
                            "reason": f"Neighbor of top earner {patron_peer_id[:12]}...",
                            "patron_peer_id": patron_peer_id,
                        })
                except Exception as e:
                    self.plugin.log(f"Error discovering neighbors of {patron_peer_id[:12]}: {e}", level='debug')
                    continue

            return candidates[:10]  # Max 10 total from this strategy

        patron_pool = self._build_neighbor_patron_pool(all_profitability)
        first_degree: Dict[str, Dict[str, Any]] = {}
        for patron in patron_pool:
            patron_peer_id = patron.get("peer_id")
            if not patron_peer_id:
                continue

            try:
                channels = self._get_cached_channels(patron_peer_id, "source")
            except Exception as e:
                self.plugin.log(f"Error discovering neighbors of {patron_peer_id[:12]}: {e}", level='debug')
                continue

            for ch in channels:
                neighbor = self._build_neighbor_candidate(
                    ch,
                    patron,
                    existing_peers,
                    our_node_id,
                    degree=1,
                )
                if neighbor is None:
                    continue
                current = first_degree.get(neighbor["peer_id"])
                if current is None:
                    neighbor["_patron_ids"] = {patron_peer_id}
                    first_degree[neighbor["peer_id"]] = neighbor
                    continue
                current["_patron_ids"].add(patron_peer_id)
                if neighbor["score"] > current["score"]:
                    current.update({
                        "score": neighbor["score"],
                        "reason": neighbor["reason"],
                        "patron_peer_id": patron_peer_id,
                    })

        first_degree_list = list(first_degree.values())
        for candidate in first_degree_list:
            patron_count = len(candidate.pop("_patron_ids", set()))
            candidate["score"] += 0.15 * patron_count

        first_degree_list.sort(key=lambda item: item["score"], reverse=True)
        second_degree = self._discover_second_degree_neighbors(
            first_degree_list[:3],
            existing_peers,
            our_node_id,
        )

        merged: Dict[str, Dict[str, Any]] = {}
        for candidate in first_degree_list + second_degree:
            existing = merged.get(candidate["peer_id"])
            if existing is None or candidate["score"] > existing["score"]:
                merged[candidate["peer_id"]] = candidate

        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:10]

    def _build_neighbor_patron_pool(self, all_profitability) -> List[Dict[str, Any]]:
        """Build the combined patron pool for efficiency-aware neighbor discovery."""
        try:
            fleet_efficiency = self._capital_efficiency.analyze()
        except Exception:
            fleet_efficiency = None

        channel_efficiencies = getattr(fleet_efficiency, "channel_efficiencies", {}) or {}
        entries = []
        for scid, prof in all_profitability.items():
            revenue = getattr(prof, "revenue", None)
            volume_routed_sats = getattr(revenue, "volume_routed_sats", 0)
            try:
                volume_routed_sats = int(volume_routed_sats or 0)
            except (TypeError, ValueError):
                volume_routed_sats = 0
            channel_eff = channel_efficiencies.get(scid)
            entries.append({
                "channel_id": scid,
                "peer_id": getattr(prof, "peer_id", None),
                "patron_score": float(getattr(channel_eff, "efficiency_rank", 0.1) or 0.1),
                "volume_routed_sats": volume_routed_sats,
                "marginal_roi_percent": float(getattr(prof, "marginal_roi_percent", 0.0) or 0.0),
            })

        selected = []
        selected.extend(sorted(entries, key=lambda item: item["patron_score"], reverse=True)[:5])
        selected.extend(sorted(entries, key=lambda item: item["volume_routed_sats"], reverse=True)[:5])
        selected.extend(sorted(entries, key=lambda item: item["marginal_roi_percent"], reverse=True)[:3])

        deduped = {}
        for entry in selected:
            peer_id = entry.get("peer_id")
            if not peer_id:
                continue
            current = deduped.get(peer_id)
            if current is None or entry["patron_score"] > current["patron_score"]:
                deduped[peer_id] = entry

        return list(deduped.values())[:10]

    def _build_neighbor_candidate(
        self,
        channel: Dict[str, Any],
        patron: Dict[str, Any],
        existing_peers: set,
        our_node_id: str,
        degree: int,
    ) -> Dict[str, Any] | None:
        """Score a first- or second-degree neighbor candidate."""
        peer_id = channel.get("destination")
        if not peer_id or peer_id == our_node_id or peer_id in existing_peers:
            return None

        capacity_sats = base_to_sats_floor(parse_msat(channel.get("amount_msat", 0)))
        fee_ppm = int(channel.get("fee_per_millionth", 0) or 0)
        if fee_ppm > 1500:
            return None
        if 0 < capacity_sats < 200000:
            return None

        base_score = float(patron.get("patron_score", 0.1) or 0.1)
        if degree == 2:
            base_score *= 0.5
        capacity_bonus = min(capacity_sats / 5_000_000, 1.0) * 0.4 if capacity_sats > 0 else 0.0
        if fee_ppm < 200:
            fee_bonus = 0.2
        elif fee_ppm < 500:
            fee_bonus = 0.1
        else:
            fee_bonus = 0.0

        score = base_score + capacity_bonus + fee_bonus
        if degree == 2:
            score *= 0.5

        return {
            "peer_id": peer_id,
            "source": "neighbor",
            "score": score,
            "degree": degree,
            "reason": f"Neighbor of capital-efficient patron {str(patron.get('peer_id', ''))[:12]}...",
            "patron_peer_id": patron.get("peer_id"),
        }

    def _discover_second_degree_neighbors(
        self,
        first_degree_candidates: List[Dict[str, Any]],
        existing_peers: set,
        our_node_id: str,
    ) -> List[Dict[str, Any]]:
        """Explore the top first-degree candidates one hop deeper."""
        second_degree = []
        for candidate in first_degree_candidates:
            try:
                channels = self._get_cached_channels(candidate["peer_id"], "source")
            except Exception:
                continue

            active_channels = [ch for ch in channels if ch.get("destination")]
            capacities = [
                base_to_sats_floor(parse_msat(ch.get("amount_msat", 0)))
                for ch in active_channels
                if base_to_sats_floor(parse_msat(ch.get("amount_msat", 0))) > 0
            ]
            if active_channels:
                channel_count_bonus = min(len(active_channels) / 20.0, 1.0) * 0.3
                median_size_bonus = min((median(capacities) if capacities else 0) / 5_000_000, 1.0) * 0.4
                fees = [int(ch.get("fee_per_millionth", 0) or 0) for ch in active_channels]
                avg_fee_ppm = sum(fees) / len(fees) if fees else 0.0
                if avg_fee_ppm < 200:
                    fee_bonus = 0.2
                elif avg_fee_ppm < 500:
                    fee_bonus = 0.1
                else:
                    fee_bonus = 0.0
                candidate["score"] += channel_count_bonus + median_size_bonus + fee_bonus

            patron = {
                "peer_id": candidate["peer_id"],
                "patron_score": candidate["score"],
            }
            scored = []
            for channel in active_channels:
                child = self._build_neighbor_candidate(
                    channel,
                    patron,
                    existing_peers,
                    our_node_id,
                    degree=2,
                )
                if child is not None:
                    scored.append(child)
            scored.sort(key=lambda item: item["score"], reverse=True)
            second_degree.extend(scored[:3])

        return second_degree

    def _discover_from_graph(self, existing_peer_ids: set) -> List[Dict]:
        """Strategy 3: Network centrality scoring from cached channel data.

        Scores nodes whose channel data was already cached by earlier strategies
        (neighbor, route-pair, demand-flow lookups populate _cycle_channels_source).
        No bulk gossip fetches — only uses data already in memory.
        """
        try:
            our_node_id = self.data_service.get_node_id() if self.data_service else self.plugin.rpc.getinfo().get("id")
        except Exception:
            return []

        scored = []

        # Score nodes already in the channel cache (populated by earlier strategies).
        # This avoids any additional RPC calls — graph discovery is purely opportunistic
        # over data that was already fetched for other purposes.
        for node_id, channels in self._cycle_channels_source.items():
            if node_id == our_node_id or node_id in existing_peer_ids:
                continue

            active_channels = [ch for ch in channels if ch.get("active", False)]
            channel_count = len(active_channels)
            if channel_count < 5:
                continue

            total_capacity_sats = sum(
                base_to_sats_floor(parse_msat(ch.get("amount_msat", 0)))
                for ch in active_channels
            )

            capacity_btc = total_capacity_sats / 100_000_000 if total_capacity_sats > 0 else 0.001
            score = channel_count * math.sqrt(capacity_btc)

            scored.append({
                "peer_id": node_id,
                "source": "graph",
                "score": score,
                "reason": f"Graph centrality: {channel_count} channels, {total_capacity_sats:,} sat",
                "channel_count": channel_count,
                "total_capacity": total_capacity_sats,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:10]

    def _discover_from_route_pairs(self, all_profitability) -> List[Dict]:
        """Strategy 5: Analyze actual forward pairs to find high-revenue routes.

        Instead of guessing which neighbors are valuable from gossip topology,
        look at which (in_channel, out_channel) pairs generate the most fees.
        The peers on the other end of profitable routes — and their neighbors —
        are strong candidates for new channels.
        """
        db = self.profitability.database if self.profitability else None
        if not db:
            return []

        try:
            our_node_id = self.data_service.get_node_id() if self.data_service else self.plugin.rpc.getinfo().get("id")
        except Exception:
            return []

        # Get existing peers to exclude
        existing_peers = set()
        # Map channel_id → peer_id for route pair lookups
        channel_to_peer = {}
        for prof in all_profitability.values():
            pid = getattr(prof, 'peer_id', None)
            if pid:
                existing_peers.add(pid)
            scid = getattr(prof, 'channel_id', None) or getattr(prof, 'scid', None)
            if scid and pid:
                channel_to_peer[str(scid).replace(':', 'x')] = pid

        # Query top revenue-generating (in_channel, out_channel) pairs from last 30 days
        try:
            rows = db.get_top_route_pairs(days=30, min_forwards=3, limit=20)
        except Exception:
            return []

        if not rows:
            return []

        # For the top pairs, find which peers are involved and look for their neighbors
        candidates = []
        route_peer_scores = {}
        for row in rows:
            in_ch = str(row['in_channel']).replace(':', 'x')
            out_ch = str(row['out_channel']).replace(':', 'x')
            total_fee_sats = base_to_sats_ceil(parse_msat(row['total_fee_msat']))

            in_peer = channel_to_peer.get(in_ch)
            out_peer = channel_to_peer.get(out_ch)
            if in_peer:
                route_peer_scores[in_peer] = route_peer_scores.get(in_peer, 0) + total_fee_sats
            if out_peer:
                route_peer_scores[out_peer] = route_peer_scores.get(out_peer, 0) + total_fee_sats

        # For each peer on a profitable route, find THEIR neighbors as candidates
        # The logic: if peer A → us → peer B is a profitable route,
        # then A's other neighbors and B's other neighbors are likely also valuable
        ranked_route_peers = [
            peer_id for peer_id, _score in
            sorted(route_peer_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        for route_peer in ranked_route_peers:
            try:
                channels = {"channels": self._get_cached_channels(route_peer, "source")}
                for ch in channels.get("channels", []):
                    dest = ch.get("destination")
                    if not dest or dest == our_node_id or dest in existing_peers:
                        continue
                    # Score based on fee capacity of the route
                    capacity = base_to_sats_floor(parse_msat(ch.get("amount_msat", 0)))
                    fee_ppm = ch.get("fee_per_millionth", 0)
                    # Prefer neighbors with reasonable fees and decent capacity
                    if fee_ppm > 1000 or capacity < 500000:
                        continue
                    score = 0.3  # Base score for route-pair discovery
                    if capacity > 5_000_000:
                        score *= 1.2
                    if fee_ppm < 200:
                        score *= 1.1
                    candidates.append({
                        "peer_id": dest,
                        "source": "route_pair",
                        "score": score,
                        "reason": f"Neighbor of route-pair peer {route_peer[:12]}...",
                        "route_peer_id": route_peer,
                    })
            except Exception:
                continue

        # Deduplicate and limit
        seen = {}
        for c in candidates:
            pid = c["peer_id"]
            if pid not in seen or c["score"] > seen[pid]["score"]:
                seen[pid] = c
        ranked = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
        return ranked[:10]

    def _discover_from_hive(self) -> List[Dict]:
        """Strategy 4: Propose peers where cl_hive recommends channel opening."""
        if self.hive_hints is None:
            self.plugin.log("Hive discovery: hive_hints not injected, skipping", level='debug')
            return []
        try:
            open_candidates = self.hive_hints.get_open_candidates()
        except Exception as e:
            self.plugin.log(f"Hive discovery: get_open_candidates() failed: {e}", level='info')
            open_candidates = []
        if not open_candidates:
            self.plugin.log("Hive discovery: get_open_candidates() returned empty", level='debug')
        candidates = []
        for peer_id, hint in open_candidates:
            if self._has_direct_peer_channel(peer_id):
                self.plugin.log(
                    f"Hive discovery: ignoring open hint for existing direct peer {str(peer_id)[:12]}...",
                    level='debug',
                )
                continue
            confidence = hint.get("topology_confidence", 0.0)
            reason_str = hint.get("reason", "hive_topology")
            candidates.append({
                "peer_id": peer_id,
                "source": "hive",
                "score": 0.3 * max(confidence, 0.1),
                "reason": f"Hive: {reason_str}",
                "hive_open_hint": hint,
            })
        candidates.extend(self._discover_from_hive_member_topology())
        candidates = self._dedupe_hive_candidates(candidates)
        self.plugin.log(f"Hive discovery: produced {len(candidates)} candidates", level='debug')
        return candidates

    def _dedupe_hive_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Deduplicate hive-discovered open candidates by peer id.

        Explicit channel-open hints are authoritative when the same peer is also
        found through fleet topology fallback. Topology metadata is retained as
        supplemental context, but it must not replace the explicit hint payload.
        """
        deduped: Dict[str, Dict] = {}
        for candidate in candidates:
            peer_id = str(candidate.get("peer_id") or "")
            if not peer_id or self._has_direct_peer_channel(peer_id):
                continue

            candidate = dict(candidate)
            candidate["peer_id"] = peer_id
            existing = deduped.get(peer_id)
            if existing is None:
                deduped[peer_id] = candidate
                continue

            existing_has_hint = existing.get("hive_open_hint") is not None
            candidate_has_hint = candidate.get("hive_open_hint") is not None
            if candidate_has_hint and not existing_has_hint:
                replacement = candidate
                if existing.get("hive_topology_witnesses") and not replacement.get("hive_topology_witnesses"):
                    replacement["hive_topology_witnesses"] = existing["hive_topology_witnesses"]
                replacement["score"] = max(
                    float(replacement.get("score", 0.0)),
                    float(existing.get("score", 0.0)),
                )
                deduped[peer_id] = replacement
                continue

            keeper = existing
            if candidate.get("hive_topology_witnesses") and not keeper.get("hive_topology_witnesses"):
                keeper["hive_topology_witnesses"] = candidate["hive_topology_witnesses"]
            keeper["score"] = max(
                float(keeper.get("score", 0.0)),
                float(candidate.get("score", 0.0)),
            )

        return list(deduped.values())

    def _get_our_node_id(self) -> str:
        try:
            if self.data_service is not None:
                return str(self.data_service.get_node_id() or "")
            return str(self.plugin.rpc.getinfo().get("id") or "")
        except Exception:
            return ""

    def _has_direct_peer_channel(self, peer_id: str) -> bool:
        """Return True when we already have a normal direct channel to peer_id."""
        if not peer_id:
            return False
        try:
            if self.data_service is not None:
                result = self.data_service.get_peer_channels(peer_id=peer_id)
            else:
                result = self.plugin.rpc.call("listpeerchannels", {"id": peer_id})
        except Exception:
            return False

        channels = result.get("channels", []) if isinstance(result, dict) else []
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            if channel.get("state") == "CHANNELD_NORMAL":
                return True
        return False

    def _is_peer_connected(self, peer_id: str) -> bool:
        """Return True when peer_id is currently connected, channel or not."""
        if not peer_id:
            return False
        try:
            if self.data_service is not None:
                result = self.data_service.get_peers()
            else:
                result = self.plugin.rpc.call("listpeers", {"id": peer_id})
        except Exception:
            return False

        peers = result.get("peers", []) if isinstance(result, dict) else []
        for peer in peers:
            if not isinstance(peer, dict):
                continue
            if str(peer.get("id") or "") == peer_id and bool(peer.get("connected", False)):
                return True
        return False

    def _is_hive_topology_witness(self, peer_id: str) -> bool:
        """A hive witness only needs an active peer connection, not a channel."""
        return self._is_peer_connected(peer_id) or self._has_direct_peer_channel(peer_id)

    def _discover_from_hive_member_topology(self) -> List[Dict]:
        """Discover non-direct hive members advertised by direct hive peers.

        cl-hive member hints can carry each member's known topology. When a
        connected hive peer reports other hive members in that topology, those are
        useful mesh-expansion candidates for faster hive gossip and coordination.
        """
        if self.hive_hints is None:
            return []

        try:
            member_ids = self.hive_hints.get_member_peer_ids()
        except Exception:
            return []
        if not isinstance(member_ids, list):
            return []

        our_node_id = self._get_our_node_id()
        member_set = {str(peer_id) for peer_id in member_ids if peer_id}
        direct_members = [
            peer_id
            for peer_id in sorted(member_set)
            if peer_id != our_node_id and self._is_hive_topology_witness(peer_id)
        ]
        if not direct_members:
            return []

        witnesses_by_target: Dict[str, set] = {}
        for witness in direct_members:
            try:
                topology = self.hive_hints.get_fleet_topology(witness)
            except Exception:
                topology = []
            if not isinstance(topology, list):
                continue
            for target in topology:
                target_id = str(target or "")
                if (
                    not target_id
                    or target_id == our_node_id
                    or target_id == witness
                    or target_id not in member_set
                    or self._has_direct_peer_channel(target_id)
                ):
                    continue
                witnesses_by_target.setdefault(target_id, set()).add(witness)

        candidates = []
        for target_id, witnesses in witnesses_by_target.items():
            witness_count = len(witnesses)
            candidates.append({
                "peer_id": target_id,
                "source": "hive",
                "score": min(0.30, 0.14 + (0.04 * witness_count)),
                "reason": f"Hive topology: advertised by {witness_count} connected hive member(s)",
                "hive_topology_witnesses": sorted(witnesses),
            })
        return sorted(candidates, key=lambda c: c["score"], reverse=True)

    def _discover_from_demand_flow(self, all_flow, existing_peers: set) -> List[Dict]:
        """Strategy 6: Find candidates adjacent to our sink peers."""
        try:
            classifier = DemandFlowClassifier()

            # Classify all current peers from flow data
            self._demand_flow_profiles = classifier.classify_peers(all_flow)

            # Find sink peers
            sink_profiles = {
                pid: profile for pid, profile in self._demand_flow_profiles.items()
                if profile.role == "sink"
            }
            if not sink_profiles:
                return []

            # Get channels for each sink peer (from cache)
            sink_channels = {}
            for pid in sink_profiles:
                sink_channels[pid] = self._get_cached_channels(pid, "source")

            # Track which candidates are sink-adjacent for scoring
            candidates = classifier.find_sink_adjacent_candidates(
                sink_profiles, sink_channels, existing_peers
            )
            self._demand_flow_sink_adjacent = {c["peer_id"] for c in candidates}

            return candidates
        except Exception as e:
            self.plugin.log(f"Demand-flow discovery failed: {e}", level='info')
            return []

    def _is_recycle_eligible(
        self, loser: Dict, protected_peers: set, route_pair_scids: set
    ) -> tuple:
        """Check if a channel is eligible for capital recycling.

        Returns (eligible: bool, reason: str)
        """
        scid = loser.get("scid", "")
        peer_id = loser.get("peer_id", "")

        # Age gate: >60 days
        try:
            open_block = int(scid.split("x")[0])
            current_block = self.data_service.get_block_height() if self.data_service else self.plugin.rpc.getinfo().get("blockheight", 0)
            age_days = (current_block - open_block) * 10 / 1440
            if age_days < 60:
                return False, f"Channel age {age_days:.0f}d < 60d minimum"
        except Exception:
            return False, "Cannot determine channel age"

        # Marginal ROI must be negative
        if loser.get("marginal_roi", 0) >= 0:
            return False, f"Marginal ROI {loser.get('marginal_roi', 0):.1f}% >= 0"

        # Hive member protection
        if self.hive_hints is not None:
            try:
                if self.hive_hints.is_hive_member(peer_id):
                    return False, "Hive member protected"
            except Exception:
                pass

        # Policy protection
        if peer_id in protected_peers:
            return False, "Policy-protected peer"

        # Route pair protection
        if scid in route_pair_scids:
            return False, "On revenue route pair"

        return True, "Eligible for recycling"

    def _calculate_recycle_ev(
        self, loser: Dict, candidate: Dict, cfg
    ) -> float:
        """Calculate EV of closing loser and opening to candidate.

        Returns recycle_ev in sats.
        """
        capacity = loser.get("capacity", 0)

        # Candidate EV
        candidate_ev = self._calculate_open_ev(candidate["peer_id"], capacity, cfg)

        # Residual value: credit the loser for its remaining earnings (90-day horizon)
        marginal_30d = loser.get("marginal_profit_30d_sats", 0)
        residual_value = marginal_30d * 3

        # On-chain costs (dynamic via feerates RPC, legacy static fallback)
        open_cost = self._estimate_open_cost()
        close_cost = self._estimate_close_cost()

        recycle_ev = candidate_ev - residual_value - close_cost - open_cost
        return recycle_ev

    RECYCLE_MIN_EV_SATS = 5000

    def _evaluate_recycle_opportunities(
        self, losers: List[Dict], candidates: List[Dict], cfg
    ) -> Dict[str, Any] | None:
        """Find the best recycle opportunity: close a loser, open to a candidate.

        Returns a recycle plan dict or None if no viable opportunity exists.
        """
        if not losers or not candidates:
            return None

        db = self.profitability.database if self.profitability else None
        if not db:
            return None

        # Build protection sets
        protected_peers = set()
        if self.policy_manager:
            try:
                for peer_id, policy in self.policy_manager.get_all_policies().items():
                    tags = getattr(policy, 'tags', []) or []
                    if "protect" in tags or "no_close" in tags:
                        protected_peers.add(peer_id)
            except Exception:
                pass

        route_pair_scids = set()
        try:
            pairs = db.get_top_route_pairs(days=30, min_forwards=3, limit=10)
            for p in pairs:
                for key in ("in_channel", "out_channel"):
                    ch = str(p.get(key, "")).replace(":", "x")
                    if ch:
                        route_pair_scids.add(ch)
        except Exception:
            pass

        # Find eligible losers
        eligible_losers = []
        for loser in losers:
            ok, reason = self._is_recycle_eligible(loser, protected_peers, route_pair_scids)
            if ok:
                eligible_losers.append(loser)

        if not eligible_losers:
            return None

        # Sort candidates by score (best first)
        sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)

        # Find best recycle pair
        best_ev = 0
        best_pair = None
        for candidate in sorted_candidates[:5]:
            for loser in eligible_losers:
                ev = self._calculate_recycle_ev(loser, candidate, cfg)
                if ev > self.RECYCLE_MIN_EV_SATS and ev > best_ev:
                    best_ev = ev
                    best_pair = {
                        "loser": loser,
                        "candidate": candidate,
                        "recycle_ev": ev,
                    }

        if best_pair:
            self._last_preferred_loop_out_scid = best_pair["loser"]["scid"]
            self._last_preferred_loop_out_reason = (
                f"Recycle target: EV={best_pair['recycle_ev']:.0f} sats, "
                f"candidate={best_pair['candidate']['peer_id'][:12]}..."
            )

        return best_pair

    def _score_candidate(self, peer_id: str, base_score: float) -> float:
        """Enrich candidate score with reputation, uptime, and profit history."""
        score = base_score

        # Peer reputation (Laplace-smoothed success rate)
        try:
            rep = self.profitability.database.get_peer_reputation(peer_id)
            if rep:
                rep_score = (rep.get('successes', 0) + 1) / (
                    rep.get('successes', 0) + rep.get('failures', 0) + 2
                )
                score *= rep_score  # 0.0-1.0 multiplier
        except Exception:
            pass

        # Profit inheritance from closed channels
        try:
            closed_summary = self.profitability.database.get_peer_closed_channel_profit_summary(peer_id)
            if closed_summary and closed_summary.get('marginal_roi_proxy', 0) > 0:
                score *= 1.5  # Boost for proven profitable peer
        except Exception:
            pass

        # Peer uptime (if available)
        try:
            uptime = self.profitability.database.get_peer_uptime_percent(peer_id, duration_seconds=604800)
            if uptime is not None and uptime < 90.0:
                score *= (uptime / 100.0)  # Penalize low uptime
        except Exception:
            pass

        # Prefer clearnet peers (more reliable, lower latency)
        try:
            node = self._get_cached_node(peer_id)
            if node:
                addresses = node.get("addresses", [])
                has_clearnet = any(
                    a.get("type") in ("ipv4", "ipv6")
                    for a in addresses
                )
                if has_clearnet:
                    score *= 1.25  # 25% boost for clearnet-reachable peers
        except Exception:
            pass

        # Hive channel-open hint bias (bounded ±30%)
        if self.hive_hints is not None:
            try:
                hint = self.hive_hints.get_channel_open_hint(peer_id)
                pref = hint.get("open_preference")
                conf = hint.get("topology_confidence", 0.0)
                reason = hint.get("reason")
                if pref == "open" and reason == "member_connectivity" and self._has_direct_peer_channel(peer_id):
                    pref = None
                if pref == "open":
                    score *= 1.0 + (0.20 * conf)
                elif pref == "avoid":
                    score *= 1.0 - (0.30 * conf)
            except Exception:
                pass

            try:
                score *= self._get_hive_open_score_multiplier(peer_id)
            except Exception:
                pass

        # Inbound fee penalty: expensive-to-rebalance peers are less valuable
        try:
            inbound_data = self.profitability.database.get_historical_inbound_fee_ppm(peer_id)
            if inbound_data and isinstance(inbound_data, dict):
                median_inbound = inbound_data.get('median_fee_ppm', 0)
                if isinstance(median_inbound, (int, float)) and median_inbound > 200:
                    # Penalize: >200 PPM inbound = 10-40% penalty
                    penalty = min(0.4, (median_inbound - 200) / 1000)
                    score *= (1.0 - penalty)
        except Exception:
            pass

        # Large-channel peer bonus: peers whose existing channels are large
        # serve high-value payment traffic (bigger HTLCs). Opening a channel
        # to them captures more of the payment size distribution.
        try:
            peer_channels = self._get_cached_channels(peer_id, "destination")
            capacities = [
                ch.get("satoshis", 0)
                for ch in peer_channels
                if ch.get("active", False) and ch.get("satoshis", 0) > 0
            ]
            if capacities:
                median_cap = sorted(capacities)[len(capacities) // 2]
                if median_cap >= 10_000_000:  # 10M+ sat channels = large-payment peer
                    score *= 1.2  # 20% boost
                elif median_cap >= 5_000_000:  # 5M+ = medium-large
                    score *= 1.1  # 10% boost
        except Exception:
            pass

        # Demand-flow boost: sink-adjacent candidates get a scoring bonus
        if peer_id in self._demand_flow_sink_adjacent:
            score *= 1.4  # 40% boost for sink-adjacent
        elif peer_id in self._demand_flow_profiles:
            profile = self._demand_flow_profiles[peer_id]
            if profile.role == "sink":
                score *= 1.3
            elif profile.role == "source":
                score *= 1.2
            elif profile.role == "unknown":
                score *= 0.9

        return score

    def _get_hive_open_score_multiplier(self, peer_id: str) -> float:
        """Return bounded open-candidate multiplier from hive routing intelligence."""
        if self.hive_hints is None:
            return 1.0

        multiplier = 1.0

        try:
            multiplier *= max(
                0.9,
                min(1.1, float(self.hive_hints.get_corridor_utilization_bias(peer_id))),
            )
        except Exception:
            pass

        try:
            reputation = self.hive_hints.get_reputation_score(peer_id)
            if isinstance(reputation, (int, float)):
                reputation = max(0.0, min(100.0, float(reputation)))
                multiplier *= 1.0 + (((reputation - 50.0) / 50.0) * 0.1)
        except Exception:
            pass

        try:
            getter = getattr(self.hive_hints, "get_metabolic_open_bias", None)
            if callable(getter):
                multiplier *= max(0.85, min(1.10, float(getter(peer_id))))
        except Exception:
            pass

        try:
            getter = getattr(self.hive_hints, "get_immune_open_bias", None)
            if callable(getter):
                multiplier *= max(0.85, min(1.10, float(getter(peer_id))))
        except Exception:
            pass

        return max(0.75, min(1.25, multiplier))

    def get_metabolic_planner_influence_debug(self) -> Dict[str, Any]:
        """Return read-only metabolic influence diagnostics for planner surfaces."""
        if self.hive_hints is None:
            return {
                "seen": False,
                "usable": False,
                "reason": "no_hive_hint_adapter",
            }
        try:
            status_getter = getattr(self.hive_hints, "get_metabolic_status", None)
            status = status_getter() if callable(status_getter) else {}
            if not isinstance(status, dict):
                status = {}
            if not status.get("present", False):
                return {"seen": False, "usable": False, "reason": "missing"}
            return {
                "seen": True,
                "usable": bool(status.get("usable", False)),
                "m2_scope": status.get("m2_scope"),
                "confidence": status.get("confidence", "unknown"),
                "coverage": dict(status.get("coverage", {}) or {}),
                "reason": status.get("reason") or (None if status.get("usable") else "unusable"),
            }
        except Exception as e:
            return {"seen": False, "usable": False, "reason": f"error: {e}"}

    def _update_candidate_pool(self, candidates: List[Dict]):
        """Persist scored candidates to database."""
        db = self.profitability.database if self.profitability else None
        if not db:
            return
        for candidate in candidates:
            try:
                db.record_planner_candidate(
                    peer_id=candidate["peer_id"],
                    score=candidate["score"],
                    source=candidate["source"],
                    capacity_recommendation_sats=candidate.get("capacity_recommendation_sats"),
                )
            except Exception:
                pass
        # Prune pool: remove candidates with score < -3.0
        try:
            all_candidates = db.get_planner_candidates(min_score=-999.0, limit=100)
            for c in all_candidates:
                if c["score"] < -3.0:
                    db.delete_planner_candidate(c["peer_id"])
            # If pool > 32, remove lowest scored
            if len(all_candidates) > 32:
                to_remove = sorted(all_candidates, key=lambda x: x["score"])[:len(all_candidates) - 32]
                for c in to_remove:
                    db.delete_planner_candidate(c["peer_id"])
        except Exception:
            pass

    def _normalize_candidate_scores(self, candidates: List[Dict]) -> List[Dict]:
        """Normalize candidate scores within each strategy to 0-1, then apply strategy weights."""
        by_source: Dict[str, List[Dict]] = {}
        for c in candidates:
            by_source.setdefault(c["source"], []).append(c)

        for source, group in by_source.items():
            max_score = max((c["score"] for c in group), default=0)
            weight = STRATEGY_WEIGHTS.get(source, 1.0)
            for c in group:
                if max_score > 0:
                    c["score"] = (c["score"] / max_score) * weight
                else:
                    c["score"] = 0.0

        return candidates

    def _apply_pool_quotas(self, candidates: List[Dict], max_pool: int = 32) -> List[Dict]:
        """Apply reserved slot quotas per strategy, then fill remaining slots by score."""
        RESERVED = {"hive": 4, "graph": 4, "route_pair": 4}
        reserved_total = sum(RESERVED.values())
        open_slots = max_pool - reserved_total

        by_source: Dict[str, List[Dict]] = {}
        for c in candidates:
            by_source.setdefault(c["source"], []).append(c)

        for group in by_source.values():
            group.sort(key=lambda c: c["score"], reverse=True)

        result = []
        used_ids = set()
        unfilled_slots = 0

        for source, quota in RESERVED.items():
            group = by_source.get(source, [])
            filled = 0
            for c in group:
                if filled >= quota:
                    break
                if c["peer_id"] not in used_ids:
                    result.append(c)
                    used_ids.add(c["peer_id"])
                    filled += 1
            unfilled_slots += quota - filled

        open_slots += unfilled_slots
        all_remaining = [c for c in candidates if c["peer_id"] not in used_ids]
        all_remaining.sort(key=lambda c: c["score"], reverse=True)
        for c in all_remaining[:open_slots]:
            if c["peer_id"] not in used_ids:
                result.append(c)
                used_ids.add(c["peer_id"])

        return result

    def get_candidate_sources(self) -> Dict[str, Any]:
        """Return strategy distribution of the current candidate pool."""
        db = self.profitability.database if self.profitability else None
        if not db:
            return {"error": "no database"}
        try:
            candidates = db.get_planner_candidates(min_score=-999.0, limit=100)
        except Exception as e:
            return {"error": str(e)}

        by_source: Dict[str, int] = {}
        for c in candidates:
            src = c.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1

        return {
            "total": len(candidates),
            "by_source": by_source,
            "candidates": [
                {"peer_id": c["peer_id"], "score": round(c["score"], 4), "source": c["source"]}
                for c in sorted(candidates, key=lambda x: x["score"], reverse=True)
            ],
        }

    def _check_fee_gate(self, cfg) -> tuple:
        """Check on-chain fee rate is acceptable. Returns (ok, reason)."""
        try:
            feerates = self.data_service.get_feerates(style="perkb") if self.data_service else self.plugin.rpc.feerates(style="perkb")
            opening_kvb = feerates.get("perkb", {}).get("opening", 1000)
            sat_per_vb = opening_kvb / 1000.0
            if sat_per_vb > cfg.planner_max_fee_rate_sat_vb:
                return False, f"Fee rate {sat_per_vb:.0f} sat/vB exceeds max {cfg.planner_max_fee_rate_sat_vb}"
            return True, f"Fee rate {sat_per_vb:.0f} sat/vB acceptable"
        except Exception as e:
            return False, f"Cannot check feerates: {e}"

    def _check_reserve(self, cfg, required_sats: int) -> tuple:
        """Check on-chain balance has sufficient reserve after operation."""
        try:
            funds = self.data_service.get_funds() if self.data_service else self.plugin.rpc.listfunds()
            confirmed = sum(
                base_to_sats_floor(parse_msat(o.get("amount_msat", 0)))
                for o in funds.get("outputs", [])
                if o.get("status") == "confirmed"
            )
            min_reserve = getattr(cfg, 'min_wallet_reserve', 500000)
            available = confirmed - min_reserve
            if available < required_sats:
                return False, f"Insufficient funds: {available} available < {required_sats} required (reserve: {min_reserve})"
            return True, f"Available: {available} sats (reserve: {min_reserve})"
        except Exception as e:
            return False, f"Cannot check funds: {e}"

    def _check_cooldown(self, peer_id: str) -> tuple:
        """Check 24h cooldown per peer."""
        db = self.profitability.database if self.profitability else None
        if not db:
            return True, "No database available for cooldown check"
        try:
            recent = db.get_recent_planner_actions(peer_id, hours=24)
            recent = [
                action for action in recent
                if str(action.get("status", "")).lower() not in ("dry_run", "failed")
            ]
            if recent:
                return False, f"Cooldown: {len(recent)} action(s) for peer in last 24h"
            return True, "No recent actions for peer"
        except Exception as e:
            return False, f"Cooldown check failed: {e}"

    @staticmethod
    def _close_fee_reserve_multiplier(cfg) -> float:
        raw = getattr(cfg, "planner_close_fee_reserve_multiplier", _DEFAULT_CLOSE_FEE_RESERVE_MULTIPLIER)
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            return _DEFAULT_CLOSE_FEE_RESERVE_MULTIPLIER
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = _DEFAULT_CLOSE_FEE_RESERVE_MULTIPLIER
        if math.isnan(value) or math.isinf(value):
            value = _DEFAULT_CLOSE_FEE_RESERVE_MULTIPLIER
        return max(1.0, value)

    @staticmethod
    def _configured_close_fee_cap_sats(cfg) -> int:
        raw = getattr(cfg, "planner_close_fee_cap_sats", 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            return 0
        try:
            return max(0, int(raw or 0))
        except (TypeError, ValueError):
            return 0

    def _close_fee_plan(self, cfg) -> dict:
        estimated = max(1, int(self._estimate_close_cost() or 0))
        configured_cap = self._configured_close_fee_cap_sats(cfg)
        if configured_cap > 0:
            if configured_cap < estimated:
                return {
                    "ok": False,
                    "estimated_cost_sats": estimated,
                    "reserve_sats": configured_cap,
                    "fee_cap_sats": configured_cap,
                    "source": "fixed_cap",
                    "error": (
                        f"Configured close fee cap {configured_cap} sats is below "
                        f"estimated close cost {estimated} sats"
                    ),
                }
            reserve_sats = configured_cap
            source = "fixed_cap"
        else:
            multiplier = self._close_fee_reserve_multiplier(cfg)
            reserve_sats = max(estimated, int(math.ceil(estimated * multiplier)))
            source = "multiplier"
        return {
            "ok": True,
            "estimated_cost_sats": estimated,
            "reserve_sats": reserve_sats,
            "fee_cap_sats": reserve_sats,
            "source": source,
            "feerange": self._close_feerange(cfg, reserve_sats),
        }

    @staticmethod
    def _close_feerange(cfg, fee_cap_sats: int) -> list | None:
        raw_enabled = getattr(cfg, "planner_close_feerange_enabled", False)
        if isinstance(raw_enabled, str):
            enabled = raw_enabled.strip().lower() in ("true", "1", "yes")
        else:
            enabled = raw_enabled is True
        if not enabled:
            return None
        try:
            cap = int(fee_cap_sats or 0)
        except (TypeError, ValueError):
            return None
        if cap <= 0:
            return None
        max_perkb = max(1, int(math.floor((cap * 1000) / _CLOSE_TX_VBYTES)))
        return ["slow", f"{max_perkb}perkb"]

    @staticmethod
    def _extract_actual_close_fee_sats(result) -> int | None:
        if not isinstance(result, dict):
            return None
        for key in ("actual_fee_sats", "close_fee_sats", "closing_fee_sats", "fee_sats"):
            value = result.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value >= 0:
                return int(value)
            if isinstance(value, str):
                stripped = value.strip().lower().removesuffix("sat").strip()
                if stripped.isdigit():
                    return int(stripped)
        for key in ("actual_fee_msat", "close_fee_msat", "closing_fee_msat", "fee_msat", "fees_msat"):
            value = result.get(key)
            parsed = parse_msat(value)
            if parsed is not None and parsed >= 0:
                return int(math.ceil(parsed / 1000.0))
        return None

    def _check_unified_budget(self, estimated_cost_sats: int, cfg=None) -> tuple:
        """Check unified liquidity budget has room for this operation."""
        provider = getattr(self, "global_budget_limit_provider", None)
        if not callable(provider):
            raw_budget = getattr(cfg, "daily_budget_sats", None) if cfg is not None else None
            if isinstance(raw_budget, (int, float)):
                budget = max(0, int(raw_budget or 0))
                if budget <= 0:
                    return False, "Unified budget provider returned zero limit"
                if int(estimated_cost_sats or 0) > budget:
                    return False, (
                        f"Unified budget: estimated cost {estimated_cost_sats} sats "
                        f"exceeds remaining {budget} sats (budget: {budget})"
                    )
                return True, f"Unified budget OK: {budget} remaining of {budget}"
            return True, "No unified budget provider"
        try:
            budget_info = provider()
            if not isinstance(budget_info, dict):
                return False, "Budget provider returned non-dict"
            budget = int(budget_info.get("effective_budget_sats", 0) or 0)
            if budget <= 0:
                return False, "Unified budget provider returned zero limit"
            remaining_raw = budget_info.get("remaining_sats", None)
            if remaining_raw is None:
                remaining = budget
                ext_provider = getattr(self, "external_liquidity_cost_provider", None)
                if callable(ext_provider):
                    try:
                        ext = ext_provider()
                        if isinstance(ext, dict):
                            remaining -= int(ext.get("spent_24h_sats", 0) or 0)
                            remaining -= int(ext.get("reserved_24h_sats", 0) or 0)
                    except Exception:
                        pass
            else:
                remaining = int(remaining_raw or 0)
            if estimated_cost_sats > max(0, remaining):
                return False, (
                    f"Unified budget: estimated cost {estimated_cost_sats} sats "
                    f"exceeds remaining {remaining} sats (budget: {budget})"
                )
            return True, f"Unified budget OK: {remaining} remaining of {budget}"
        except Exception as e:
            return False, f"Budget check failed: {e}"

    def _check_safety_guards(self, cfg, action_type: str, peer_id: str,
                              amount_sats: int = 0) -> tuple:
        """Run all safety checks. Returns (ok, reason)."""
        # Fee gate applies to all actions
        fee_ok, fee_reason = self._check_fee_gate(cfg)
        if not fee_ok:
            return False, fee_reason

        # Reserve check for opens
        if action_type == "open":
            reserve_ok, reserve_reason = self._check_reserve(cfg, amount_sats)
            if not reserve_ok:
                return False, reserve_reason
            # Unified budget gates the immediate on-chain spend for the open.
            budget_ok, budget_reason = self._check_unified_budget(self._estimate_open_cost(), cfg)
            if not budget_ok:
                return False, budget_reason

        if action_type == "close":
            # Close fees are on-chain liquidity costs too; live closes must not
            # bypass the same unified budget surface used by opens.
            fee_plan = self._close_fee_plan(cfg)
            if not fee_plan.get("ok", False):
                return False, str(fee_plan.get("error") or "Close fee cap invalid")
            budget_ok, budget_reason = self._check_unified_budget(int(fee_plan["reserve_sats"]), cfg)
            if not budget_ok:
                return False, budget_reason

        # Cooldown check
        cooldown_ok, cooldown_reason = self._check_cooldown(peer_id)
        if not cooldown_ok:
            return False, cooldown_reason

        return True, "All guards passed"

    def _discover_peers(self, winners: List[Dict], all_profitability, all_flow) -> List[Dict]:
        """Run all discovery strategies and merge candidates."""
        candidates = []
        candidates.extend(self._discover_from_winners(winners))
        candidates.extend(self._discover_from_neighbors(all_profitability))

        # Strategy 3: graph centrality
        existing_peers = set()
        for prof in all_profitability.values():
            pid = getattr(prof, 'peer_id', None)
            if pid:
                existing_peers.add(pid)
        candidates.extend(self._discover_from_graph(existing_peers))

        # Strategy 4: hive topology hints
        candidates.extend(self._discover_from_hive())

        # Strategy 5: route-pair analysis (neighbors of peers on profitable routes)
        candidates.extend(self._discover_from_route_pairs(all_profitability))

        # Strategy 6: demand-flow (sink-adjacent candidates)
        candidates.extend(self._discover_from_demand_flow(all_flow, existing_peers))

        # Normalize scores within each strategy before dedup
        candidates = self._normalize_candidate_scores(candidates)

        # Deduplicate by peer_id, keeping highest score
        seen = {}
        for c in candidates:
            pid = c["peer_id"]
            if pid not in seen or c["score"] > seen[pid]["score"]:
                seen[pid] = c
        merged = list(seen.values())

        # Enrich scores with reputation, uptime, profit history
        for c in merged:
            c["score"] = self._score_candidate(c["peer_id"], c["score"])

        # Apply pool slot quotas to ensure strategy diversity
        merged = self._apply_pool_quotas(merged)

        # Persist to candidate pool
        self._update_candidate_pool(merged)

        return merged

    def _size_channel(self, candidate: Dict, all_candidates: List[Dict],
                       available_sats: int, cfg) -> int:
        """ROI-proportional channel sizing with competitive awareness.

        Allocates available funds proportionally to candidate scores.
        Biases toward larger channels when the peer's other channels are large
        (to be competitive for large-payment routing).
        Clamps to [min_channel, max_channel] and never exceeds 50% of available.
        """
        if not all_candidates:
            return cfg.planner_min_channel_sats

        total_score = sum(max(c.get("score", 0.01), 0.01) for c in all_candidates)
        candidate_score = max(candidate.get("score", 0.01), 0.01)
        roi_weight = candidate_score / total_score

        raw_size = int(available_sats * roi_weight)

        # Never more than half remaining
        raw_size = min(raw_size, available_sats // 2)

        # Competitive sizing: if peer's other channels are large, size up
        # to be competitive for large-payment routing (HTLC max scales with
        # channel capacity). A 2M sat channel among 10M+ neighbors misses
        # all the large payment traffic.
        peer_id = candidate.get("peer_id")
        if peer_id:
            try:
                peer_channels = self._get_cached_channels(peer_id, "destination")
                capacities = [
                    ch.get("satoshis", 0)
                    for ch in peer_channels
                    if ch.get("active", False) and ch.get("satoshis", 0) > 0
                ]
                if capacities:
                    median_cap = sorted(capacities)[len(capacities) // 2]
                    # If peer's median channel is larger than our planned size,
                    # try to match at least 50% of their median
                    competitive_floor = median_cap // 2
                    if competitive_floor > raw_size and competitive_floor <= available_sats // 2:
                        raw_size = min(competitive_floor, cfg.planner_max_channel_sats)
            except Exception:
                pass

        # Clamp to config bounds
        return max(cfg.planner_min_channel_sats,
                   min(raw_size, cfg.planner_max_channel_sats))

    def _seed_revenue_anchor(self, all_profitability) -> None:
        """Cache the observed revenue anchor from profitability data in hand.

        Called by execute_cycle/generate_report with the analyze_all_channels
        result they already fetched, so per-candidate EV calls do not
        re-derive the anchor.
        """
        self._observed_daily_ppm_cache = (
            self._compute_observed_daily_ppm(all_profitability),
        )

    def _observed_node_daily_ppm(self) -> float | None:
        """Return the node's observed median per-day revenue ppm (cached)."""
        cache = self._observed_daily_ppm_cache
        if cache is not None:
            return cache[0]
        try:
            all_profitability = self.profitability.analyze_all_channels()
        except Exception:
            all_profitability = None
        value = self._compute_observed_daily_ppm(all_profitability)
        self._observed_daily_ppm_cache = (value,)
        return value

    @staticmethod
    def _compute_observed_daily_ppm(all_profitability) -> float | None:
        """Median realized (fees_earned/days_open)/capacity across channels, in ppm/day.

        Returns None when no channel exposes usable numeric data (e.g. a
        brand-new node), letting the caller fall back to the bootstrap rate.
        """
        try:
            profs = list(all_profitability.values())
        except Exception:
            return None
        samples = []
        for prof in profs:
            try:
                capacity_sats = int(getattr(prof, "capacity_sats", 0) or 0)
                days_open = int(getattr(prof, "days_open", 0) or 0)
                revenue = getattr(prof, "revenue", None)
                fees_sats = float(getattr(revenue, "fees_earned_msat", 0) or 0) / 1000.0
            except (TypeError, ValueError):
                continue
            if capacity_sats <= 0 or days_open <= 0 or fees_sats < 0:
                continue
            samples.append((fees_sats / days_open) / capacity_sats * 1_000_000.0)
        if not samples:
            return None
        return float(median(samples))

    def _calculate_open_ev(self, peer_id: str, channel_size_sats: int, cfg) -> float:
        """EV-based channel open decision. Returns expected profit in sats.

        EV = expected_lifetime_revenue - on_chain_cost - expected_rebalance_costs
        Only open when EV > 0.

        F1 (audit): the revenue forecast is anchored to the node's OBSERVED
        median revenue-per-capacity (discounted for new peers and clamped to
        the legacy constant as a ceiling), not a fixed utilization guess.
        """
        # Estimate daily revenue
        daily_revenue = 0.0

        # Try profit inheritance from closed channels
        try:
            closed_summary = self.profitability.database.get_peer_closed_channel_profit_summary(peer_id)
            if closed_summary and closed_summary.get('daily_net_est_sats', 0) > 0:
                daily_revenue = closed_summary['daily_net_est_sats']
        except Exception:
            pass

        # Fallback: anchor to this node's realized revenue-per-capacity.
        if daily_revenue <= 0:
            observed_daily_ppm = self._observed_node_daily_ppm()
            if observed_daily_ppm is not None:
                forecast_daily_ppm = min(
                    observed_daily_ppm * NEW_PEER_DISCOUNT,
                    LEGACY_FORECAST_DAILY_PPM_CEILING,
                )
            else:
                # Bootstrap: no routing history at all — use the legacy rate.
                forecast_daily_ppm = LEGACY_FORECAST_DAILY_PPM_CEILING
            daily_revenue = channel_size_sats * forecast_daily_ppm / 1_000_000.0

        # Hive demand signal: apply bounded, confidence-weighted fleet bias
        if self.hive_hints:
            try:
                daily_revenue *= self.hive_hints.get_rebalance_bias(peer_id)
            except Exception:
                pass

        # Estimate on-chain costs (dynamic via feerates RPC, legacy fallback)
        open_cost = self._estimate_open_cost()
        close_cost = self._estimate_close_cost()

        on_chain_cost = open_cost + close_cost

        # Estimate rebalance costs using actual peer inbound fee history
        rebal_cost_per_day = daily_revenue * 0.1  # Default: 10% of revenue
        try:
            inbound_data = self.profitability.database.get_historical_inbound_fee_ppm(peer_id)
            if inbound_data and isinstance(inbound_data, dict):
                median_inbound_ppm = inbound_data.get('median_fee_ppm', 0)
                if median_inbound_ppm > 0:
                    # Daily rebalance cost = daily_volume * inbound_fee_rate.
                    # The volume estimate is derived from the SAME forecast as
                    # the revenue side (audit: it previously assumed 30%
                    # turnover regardless of the revenue model).
                    daily_volume_est = daily_revenue * 1_000_000.0 / ASSUMED_AVG_FEE_PPM
                    rebal_cost_per_day = (daily_volume_est * median_inbound_ppm) / 1_000_000
        except Exception:
            pass  # Fallback to 10% default

        # Conservative 6-month lifetime estimate
        lifetime_days = 180

        expected_revenue = daily_revenue * lifetime_days
        expected_rebal_cost = rebal_cost_per_day * lifetime_days

        min_annual_roi_pct = getattr(cfg, "planner_min_annual_roi_pct", 1.0)
        if isinstance(min_annual_roi_pct, bool) or not isinstance(min_annual_roi_pct, (int, float)):
            min_annual_roi_pct = 1.0
        min_annual_roi_pct = max(0.0, float(min_annual_roi_pct))
        capital_hurdle = channel_size_sats * (min_annual_roi_pct / 100.0) * (lifetime_days / 365.0)

        ev = expected_revenue - on_chain_cost - expected_rebal_cost - capital_hurdle
        return ev

    def _calculate_redeployment_ev(
        self, loser: Dict[str, Any], winners: List[Dict[str, Any]], cfg
    ) -> tuple:
        """Compute the EV of closing a loser and redeploying capital to the best winner.

        Returns:
            (redeployment_ev, best_winner_peer_id, winner_ev)
            redeployment_ev = winner_ev - loser_ongoing_cost - closure_cost
            If no winners, returns (negative_ev, None, 0)
        """
        closure_cost = self._estimate_close_cost()
        loser_capacity = loser.get("capacity", 0)

        # Ongoing cost: projected 6-month loss (0 if channel is profitable)
        marginal_30d = loser.get("marginal_profit_30d_sats", 0)
        ongoing_cost = max(0, -marginal_30d * 6)

        # Find the best winner by EV
        best_ev = 0
        best_peer = None
        for winner in winners:
            try:
                ev = self._calculate_open_ev(winner["peer_id"], loser_capacity, cfg)
                if ev > best_ev:
                    best_ev = ev
                    best_peer = winner["peer_id"]
            except Exception:
                continue

        redeployment_ev = best_ev - ongoing_cost - closure_cost
        return (redeployment_ev, best_peer, best_ev)

    def _estimate_open_cost(self) -> int:
        """Estimate the on-chain cost of opening a channel."""
        try:
            feerates = self.data_service.get_feerates(style="perkb") if self.data_service else self.plugin.rpc.feerates(style="perkb")
            opening_perkb = self._extract_opening_feerate_perkb(feerates)
            if opening_perkb is None:
                return ChainCostDefaults.CHANNEL_OPEN_COST_SATS
            sat_per_vb = opening_perkb / 1000.0
            return int(sat_per_vb * 140)  # ~140 vbytes for funding tx
        except Exception:
            return ChainCostDefaults.CHANNEL_OPEN_COST_SATS

    def _estimate_close_cost(self) -> int:
        """Estimate the on-chain cost of closing a channel.

        Uses the same feerate signal the planner's open-cost and recycle-EV
        paths already use — falls back to the legacy static default only
        when the feerate RPC fails. 200 vbytes matches the inline closure
        size used at capacity_planner lines 1638 and 2193.
        """
        try:
            feerates = self.data_service.get_feerates(style="perkb") if self.data_service else self.plugin.rpc.feerates(style="perkb")
            opening_perkb = self._extract_opening_feerate_perkb(feerates)
            if opening_perkb is None:
                return ChainCostDefaults.CHANNEL_CLOSE_COST_SATS
            sat_per_vb = opening_perkb / 1000.0
            return int(sat_per_vb * 200)  # ~200 vbytes for close tx
        except Exception:
            return ChainCostDefaults.CHANNEL_CLOSE_COST_SATS

    @staticmethod
    def _extract_opening_feerate_perkb(feerates) -> float | None:
        """Return CLN's opening feerate or None for malformed RPC responses."""
        if not isinstance(feerates, dict):
            return None
        perkb = feerates.get("perkb", {})
        if not isinstance(perkb, dict):
            return None
        value = perkb.get("opening", 1000)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value <= 0:
            return None
        return float(value)

    @staticmethod
    def _parse_min_chan_size_error(error_msg: str) -> int:
        """Extract peer's minimum channel size from a 'below min chan size' error.

        Returns the minimum in sats, or 0 if the error doesn't match.
        """
        import re
        match = re.search(r'min chan size of ([0-9.]+) BTC', error_msg)
        if match:
            try:
                return int(float(match.group(1)) * 100_000_000)
            except (ValueError, OverflowError):
                pass
        return 0

    def _rpc_fundchannel(self, peer_id: str, amount_sats: int,
                         request_amt: int = None, compact_lease: str = None) -> Dict[str, Any]:
        params = {"id": peer_id, "amount": amount_sats, "announce": True}
        if request_amt and compact_lease:
            params["request_amt"] = request_amt
            params["compact_lease"] = compact_lease
        return self.data_service.fund_channel(**params) if self.data_service else self.plugin.rpc.call("fundchannel", params)

    def _execute_open(self, peer_id: str, amount_sats: int, cfg, reason: str) -> Dict:
        """Execute a channel open via fundchannel RPC.

        Flow:
        1. Record planned action in database
        2. If dry_run, log and return early
        3. Reserve budget via generic spend ledger
        4. Call fundchannel (auto-connects via gossip)
        5. On success: update action, mark spend
        6. On failure: update action, release reservation
        """
        db = self.profitability.database if self.profitability else None

        estimated_cost = self._estimate_open_cost()

        # Record the planned action
        action_id = None
        if db:
            try:
                action_id = db.record_planner_action(
                    action_type="open",
                    peer_id=peer_id,
                    amount_sats=amount_sats,
                    estimated_cost_sats=estimated_cost,
                    reason=reason,
                )
            except Exception as e:
                self.plugin.log(f"Failed to record planner action: {e}", level='warn')

        # Dry run mode: log but don't execute
        if cfg.planner_dry_run:
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="dry_run")
                except Exception:
                    pass
            self.plugin.log(
                f"[DRY RUN] Would open {amount_sats} sat channel to {peer_id[:16]}... "
                f"(estimated cost: {estimated_cost} sats, reason: {reason})",
                level='info'
            )
            return {"action_id": action_id, "status": "dry_run", "peer_id": peer_id, "amount_sats": amount_sats}

        # Reserve budget
        reservation_id = f"planner-open-{peer_id[:16]}-{int(time.time())}"
        reservation_active = False
        if db:
            try:
                reservation_active = bool(db.reserve_spend(
                    reservation_id=reservation_id,
                    amount_sats=estimated_cost,
                    category="channel_open",
                    subcategory="automated",
                    metadata={"peer_id": peer_id, "amount_sats": amount_sats},
                ))
            except Exception as e:
                self.plugin.log(f"Budget reservation failed: {e}", level='warn')
                reservation_active = False

            if not reservation_active:
                if action_id:
                    try:
                        db.update_planner_action(action_id, status="failed")
                    except Exception:
                        pass
                return {
                    "action_id": action_id,
                    "status": "failed",
                    "peer_id": peer_id,
                    "error": "Budget reservation failed",
                }

        try:
            # Detect dual-fund opportunity
            request_amt = None
            compact_lease = None
            node_info = self._get_cached_node(peer_id)
            if node_info and node_info.get("option_will_fund"):
                owf = node_info["option_will_fund"]
                compact_lease = owf.get("compact_lease")
                if compact_lease:
                    request_amt = amount_sats
                    self.plugin.log(
                        f"Dual-fund available for {peer_id[:16]}..., requesting {request_amt} sats",
                        level='info',
                    )

            # fundchannel auto-connects if CLN knows the peer address (from gossip).
            result = self._rpc_fundchannel(peer_id, amount_sats, request_amt, compact_lease)

            channel_id = result.get("channel_id") or result.get("channelid")

            # Success: update action and mark spend
            if db and action_id:
                try:
                    db.update_planner_action(
                        action_id,
                        status="completed",
                        channel_id=channel_id,
                    )
                except Exception:
                    pass
            if db:
                try:
                    db.mark_spend_reservation_spent(
                        reservation_id=reservation_id,
                        actual_spent_sats=estimated_cost,
                        source="capacity_planner",
                        record_event=True,
                    )
                except Exception:
                    pass

            self.plugin.log(
                f"Channel opened: {channel_id} to {peer_id[:16]}... ({amount_sats} sats)",
                level='info'
            )
            return {"action_id": action_id, "status": "completed", "channel_id": channel_id, "peer_id": peer_id}

        except Exception as e:
            error_msg = str(e)

            # Retry with peer's minimum if we tried too small
            retry_amount = self._parse_min_chan_size_error(error_msg)
            if retry_amount and retry_amount > amount_sats and retry_amount <= cfg.planner_max_channel_sats:
                # Check if we can afford the retry
                try:
                    reserve_ok, _ = self._check_reserve(cfg, retry_amount)
                    if reserve_ok:
                        self.plugin.log(
                            f"Retrying channel open to {peer_id[:16]}... at peer minimum "
                            f"{retry_amount} sats (was {amount_sats})",
                            level='info'
                        )
                        # Release old reservation, retry will make a new one
                        if db and reservation_active:
                            try:
                                db.release_spend_reservation(reservation_id)
                            except Exception:
                                pass
                        return self._execute_open(peer_id, retry_amount, cfg, reason=f"retry: peer min {retry_amount}")
                except Exception as retry_err:
                    self.plugin.log(f"Retry check failed: {retry_err}", level='debug')

            # Failure: update action and release reservation
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="failed")
                except Exception:
                    pass
            if db and reservation_active:
                try:
                    db.release_spend_reservation(reservation_id)
                except Exception:
                    pass

            self.plugin.log(f"Channel open failed for {peer_id[:16]}...: {e}", level='error')
            return {"action_id": action_id, "status": "failed", "error": error_msg, "peer_id": peer_id}

    def _check_close_allowed(self, peer_id: str) -> tuple:
        """Check if a channel close is allowed for this peer.

        Channels with static or passive policy, or tagged
        'protect'/'no_close', must never be auto-closed.

        Returns:
            (allowed, reason) tuple.
        """
        if not self.policy_manager:
            return True, "No policy manager configured"

        try:
            policy = self.policy_manager.get_policy(peer_id)

            # Static and passive policy channels are never auto-closed
            strategy = policy.strategy
            strategy_str = strategy.value if hasattr(strategy, 'value') else str(strategy)
            if strategy_str == "static":
                return False, "Channel has static policy — close blocked"
            if strategy_str == "passive":
                return False, "Channel has passive policy — close blocked"

            # Protected channels are never closed
            if hasattr(policy, 'has_tag'):
                if policy.has_tag("protect") or policy.has_tag("no_close"):
                    tag = "protect" if policy.has_tag("protect") else "no_close"
                    return False, f"Channel tagged '{tag}' — close blocked"
            elif hasattr(policy, 'tags') and policy.tags:
                for tag in ("protect", "no_close"):
                    if tag in policy.tags:
                        return False, f"Channel tagged '{tag}' — close blocked"

        except Exception as e:
            self.plugin.log(f"Policy check failed for {peer_id[:12]}...: {e}", level='warn')
            return False, "Policy unavailable"

        return True, "Close allowed"

    def _execute_defibrillation(self, channel_id: str, peer_id: str, cfg, reason: str = "") -> Dict[str, Any]:
        """Run a bounded diagnostic rebalance through the rebalancer."""
        db = self.profitability.database if self.profitability else None
        action_id = None
        if db:
            try:
                action_id = db.record_planner_action(
                    action_type="defibrillate",
                    peer_id=peer_id,
                    channel_id=channel_id,
                    amount_sats=50_000,
                    estimated_cost_sats=100,
                    reason=reason,
                )
            except Exception as e:
                self.plugin.log(f"Failed to record defibrillation action: {e}", level='warn')

        if cfg.planner_dry_run:
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="dry_run")
                except Exception:
                    pass
            self.plugin.log(
                f"[DRY RUN] Would defibrillate {channel_id} "
                f"(peer: {peer_id[:16]}..., reason: {reason})",
                level='info',
            )
            return {
                "action_id": action_id,
                "status": "dry_run",
                "channel_id": channel_id,
                "peer_id": peer_id,
            }

        if not self.rebalancer or not hasattr(self.rebalancer, "diagnostic_rebalance"):
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="failed")
                except Exception:
                    pass
            return {
                "action_id": action_id,
                "status": "failed",
                "channel_id": channel_id,
                "peer_id": peer_id,
                "error": "rebalancer unavailable",
            }

        try:
            result = self.rebalancer.diagnostic_rebalance(channel_id)
            success = bool(result.get("success")) if isinstance(result, dict) else False
            status = "completed" if success else "failed"
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status=status)
                except Exception:
                    pass
            return {
                "action_id": action_id,
                "status": status,
                "channel_id": channel_id,
                "peer_id": peer_id,
                "message": result.get("message", "") if isinstance(result, dict) else "",
            }
        except Exception as e:
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="failed")
                except Exception:
                    pass
            self.plugin.log(f"Defibrillation failed for {channel_id}: {e}", level='warn')
            return {
                "action_id": action_id,
                "status": "failed",
                "channel_id": channel_id,
                "peer_id": peer_id,
                "error": str(e),
            }

    def _execute_close(self, channel_id: str, peer_id: str, cfg, reason: str = "") -> Dict:
        """Close a channel: record action, gate on dry_run/recommendation mode, stop rebalancer, call close RPC."""
        db = self.profitability.database if self.profitability else None

        fee_plan = self._close_fee_plan(cfg)
        estimated_cost = int(fee_plan.get("estimated_cost_sats", self._estimate_close_cost()) or 0)
        reserved_close_fee_sats = int(fee_plan.get("reserve_sats", estimated_cost) or estimated_cost)

        # Record action
        action_id = None
        if db:
            try:
                action_id = db.record_planner_action(
                    action_type="close",
                    peer_id=peer_id,
                    channel_id=channel_id,
                    estimated_cost_sats=estimated_cost,
                    reason=reason,
                )
            except Exception as e:
                self.plugin.log(f"Failed to record close action: {e}", level='warn')

        # Dry run gate
        if cfg.planner_dry_run:
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="dry_run")
                except Exception:
                    pass
            self.plugin.log(
                f"[DRY RUN] Would close {channel_id} "
                f"(peer: {peer_id[:16]}..., reason: {reason})",
                level='info',
            )
            return {"action_id": action_id, "status": "dry_run"}

        if not self._close_execution_enabled(cfg):
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="recommended")
                except Exception:
                    pass
            self.plugin.log(
                f"[RECOMMEND] Close {channel_id} (peer: {peer_id[:16]}..., reason: {reason})",
                level='info',
            )
            return {
                "action_id": action_id,
                "status": "recommended",
                "channel_id": channel_id,
                "peer_id": peer_id,
            }

        if not fee_plan.get("ok", False):
            budget_ok = False
            budget_reason = str(fee_plan.get("error") or "Close fee cap invalid")
        else:
            budget_ok, budget_reason = self._check_unified_budget(reserved_close_fee_sats, cfg)
        if not budget_ok:
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="failed")
                except Exception:
                    pass
            self.plugin.log(
                f"Planner close blocked by unified budget for {channel_id}: {budget_reason}",
                level='warn',
            )
            return {
                "action_id": action_id,
                "status": "failed",
                "channel_id": channel_id,
                "peer_id": peer_id,
                "error": budget_reason,
            }

        # Stop rebalancer jobs if any
        if self.rebalancer and hasattr(self.rebalancer, 'job_manager'):
            try:
                if self.rebalancer.job_manager.has_active_job(channel_id):
                    self.rebalancer.job_manager.stop_job(channel_id, reason="planner_close")
            except Exception as e:
                self.plugin.log(
                    f"Error stopping rebalancer job for {channel_id}: {e}",
                    level='warn',
                )

        try:
            result = self._rpc_close(channel_id, feerange=fee_plan.get("feerange"))

            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="completed")
                except Exception:
                    pass

            # Record close cost in spend_events (matches open cost tracking pattern)
            if db:
                try:
                    actual_close_fee_sats = self._extract_actual_close_fee_sats(result)
                    close_cost = actual_close_fee_sats if actual_close_fee_sats is not None else reserved_close_fee_sats
                    close_reservation_id = f"planner-close-{channel_id}-{int(time.time())}"
                    reserved = db.reserve_spend(
                        reservation_id=close_reservation_id,
                        amount_sats=reserved_close_fee_sats,
                        category="channel_close",
                        subcategory=reason if reason else "automated",
                        metadata={
                            "channel_id": channel_id,
                            "peer_id": peer_id,
                            "estimated_close_cost_sats": estimated_cost,
                            "reserved_close_fee_sats": reserved_close_fee_sats,
                            "close_fee_cap_source": fee_plan.get("source"),
                            "actual_close_fee_source": "close_rpc" if actual_close_fee_sats is not None else "reserved_cap_fallback",
                        },
                    )
                    if reserved:
                        db.mark_spend_reservation_spent(
                            reservation_id=close_reservation_id,
                            actual_spent_sats=close_cost,
                            source="capacity_planner",
                            record_event=True,
                        )
                except Exception as e:
                    self.plugin.log(f"Failed to record close cost: {e}", level='debug')

            self.plugin.log(
                f"Channel closed: {channel_id} (peer: {peer_id[:16]}...)",
                level='info',
            )
            actual_close_fee_sats = self._extract_actual_close_fee_sats(result)
            return {
                "action_id": action_id,
                "status": "completed",
                "result": result,
                "estimated_close_cost_sats": estimated_cost,
                "reserved_close_fee_sats": reserved_close_fee_sats,
                "actual_close_fee_sats": actual_close_fee_sats,
                "close_fee_cap_source": fee_plan.get("source"),
                "close_feerange": fee_plan.get("feerange"),
            }

        except Exception as e:
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="failed")
                except Exception:
                    pass
            self.plugin.log(
                f"Channel close failed for {channel_id}: {e}",
                level='error',
            )
            return {"action_id": action_id, "status": "failed", "error": str(e)}

    def _rpc_close(self, channel_id: str, feerange: list | None = None) -> Dict[str, Any]:
        params = {"id": channel_id}
        if feerange:
            params["feerange"] = feerange
        return self.data_service.close_channel(**params) if self.data_service else self.plugin.rpc.call("close", params)
