"""
Capacity Planner Module for cl-revenue-ops

This module identifies "Winner" channels for capital injection
and "Loser" channels for capital redeployment (Close).
"""

import json
import time
from typing import Dict, List, Any
from pyln.client import Plugin
from .config import ChainCostDefaults


# Loser severity ranking for sorting (higher = worse)
_LOSER_SEVERITY = {
    "ZOMBIE": 3,
    "FIRE SALE": 2,
    "STAGNANT+HARD_REBAL": 2,
    "STAGNANT": 1,
}


class CapacityPlanner:
    """
    Identifies capital redeployment opportunities to maximize yield.
    """

    def __init__(self, plugin: Plugin, profitability_analyzer, flow_analyzer, policy_manager=None):
        self.plugin = plugin
        self.profitability = profitability_analyzer
        self.flow = flow_analyzer
        self.policy_manager = policy_manager

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a strategic redeployment report.
        """
        mempool_rec = self._get_mempool_recommendation()

        # Fetch analyses once and pass to both identification methods
        all_profitability = self.profitability.analyze_all_channels()
        all_flow = self.flow.analyze_all_channels()

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

    def _get_mempool_recommendation(self) -> str:
        """Query feerates and return a graduated recommendation based on opening costs."""
        try:
            feerates = self.plugin.rpc.feerates(style="perkb")
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
        """
        losers = []

        from .profitability_analyzer import ProfitabilityClass

        for scid, prof in all_profitability.items():
            flow_metrics = all_flow.get(scid)

            # Fetch diagnostic stats from DB
            diag_stats = self.profitability.database.get_diagnostic_rebalance_stats(scid, days=14)
            attempt_count = diag_stats.get("attempt_count", 0)

            # Rebalance difficulty scoring from success rate history
            success_data = self.profitability.database.get_channel_rebalance_success_rate(scid, 30)
            rebal_difficulty = 0.0
            if success_data and success_data.get('total', 0) >= 3:
                rebal_difficulty = 1.0 - success_data.get('success_rate', 1.0)  # 0=easy, 1=impossible

            # SCID formatting check - ensure 'x' separator
            scid_display = scid.replace(':', 'x')

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

            # Remote-opened channels are "free" capacity — raise the bar for closing.
            # They cost us nothing to acquire, so only close if deeply underwater.
            opener = getattr(prof, 'opener', 'local')
            if opener == 'remote' and is_fire_sale and not is_stagnant:
                if prof.marginal_roi_percent > -75.0:
                    # Skip close recommendation for remote channels unless deeply underwater
                    continue

            if is_fire_sale or is_stagnant:
                # PROTECTION: A channel cannot be recommended for "Close"
                # until the diagnostic_rebalance has been attempted at least twice in the last 14 days.
                # Accounting v2.0: Include estimated closure cost
                estimated_closure_cost = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS
                reason = fire_sale_reason if is_fire_sale else "STAGNANT"

                if attempt_count < 2:
                    losers.append({
                        "scid": scid_display,
                        "peer_id": prof.peer_id,
                        "reason": f"{reason} (NEEDS DEFIBRILLATOR)",
                        "roi": round(prof.roi_percent, 2),
                        "marginal_roi": round(prof.marginal_roi_percent, 2),
                        "classification": prof.classification.value if hasattr(prof.classification, 'value') else str(prof.classification),
                        "capacity": prof.capacity_sats,
                        "estimated_closure_cost_sats": estimated_closure_cost,
                        "rebal_difficulty": round(rebal_difficulty, 2),
                        "opener": opener,
                        "action": "DEFIBRILLATE"
                    })
                else:
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
                        "action": "CLOSE"
                    })

        return losers

    def _generate_recommendations(self, winners: List[Dict], losers: List[Dict]) -> List[str]:
        """
        Create actionable recommendations pairing winners and losers.
        """
        recommendations = []

        # Sort winners by ROI (descending)
        sorted_winners = sorted(winners, key=lambda x: x['roi'], reverse=True)

        # Separate closeable losers from defibrillation candidates
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

                recommendations.append(
                    f"REDEPLOYMENT: Close channel {loser['scid']} ({loser['reason']}) "
                    f"and redeploy funds to {winner['scid']} (ROI: {winner['roi']:.1f}%)."
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

        # Defibrillation alerts are always separate — they don't consume winner slots
        for loser in defibrillate:
            recommendations.append(
                f"DEFIBRILLATE: {loser['scid']} ({loser['reason']}, {loser['roi']:.1f}% ROI). "
                f"Diagnostic rebalance required before closure can be recommended."
            )

        return recommendations
