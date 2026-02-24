"""
Capacity Planner Module for cl-revenue-ops

This module identifies "Winner" channels for capital injection (Splice-In)
and "Loser" channels for capital redeployment (Splice-Out/Close).
"""

import time
from typing import Dict, List, Any, Optional, Tuple
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

    def __init__(self, plugin: Plugin, config, profitability_analyzer, flow_analyzer):
        self.plugin = plugin
        self.config = config
        self.profitability = profitability_analyzer
        self.flow = flow_analyzer

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a strategic redeployment report.
        """
        mempool_rec = self._get_mempool_recommendation()

        # Fetch analyses once and pass to both identification methods
        all_profitability = self.profitability.analyze_all_channels()
        all_flow = self.flow.analyze_all_channels()
        peer_splice_map = self._get_peer_splice_map()

        winners = self._identify_winners(all_profitability, all_flow, peer_splice_map)
        losers = self._identify_losers(all_profitability, all_flow, peer_splice_map)

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
            "recommendations": recommendations
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
                return f"HOLD: On-chain fees too high for efficient splicing ({sat_per_vb:.0f} sat/vB)."
            elif sat_per_vb > 50:
                return f"CAUTION: On-chain fees are elevated ({sat_per_vb:.0f} sat/vB). Consider waiting for lower fees."
            return f"PROCEED: Fee environment is favorable ({sat_per_vb:.0f} sat/vB)."
        except Exception as e:
            self.plugin.log(f"Error checking mempool for capacity report: {e}", level='debug')
            return "UNKNOWN: Could not fetch feerates."

    def _get_peer_splice_map(self) -> Dict[str, bool]:
        """Identify which peers support splicing (bits 62/63 for option_splice)."""
        splice_map = {}
        try:
            peers = self.plugin.rpc.listpeers().get("peers", [])
            for peer in peers:
                peer_id = peer.get("id")
                features = peer.get("features", "")
                # BOLT 9: option_splice uses feature bits 62 (even=required) / 63 (odd=optional)
                # CLN provides 'features' as a hex string.
                # We convert hex to int and check the appropriate bits.
                if not features:
                    splice_map[peer_id] = False
                    continue

                has_splice = False
                try:
                    feat_int = int(features, 16)
                    if (feat_int & (1 << 62)) or (feat_int & (1 << 63)):
                        has_splice = True
                except Exception as e:
                    self.plugin.log(f"Splice feature parse error for {peer_id[:12]}...: {e}", level='debug')

                splice_map[peer_id] = has_splice
        except Exception as e:
            self.plugin.log(f"Error mapping peer splice support: {e}", level='debug')

        return splice_map

    def _identify_winners(self, all_profitability, all_flow, peer_splice_map) -> List[Dict[str, Any]]:
        """
        Identify high-performing channels that are capacity-constrained.
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
            if success_data and success_data['total'] >= 3:
                if success_data['success_rate'] < 0.5:
                    rebal_penalty = (0.5 - success_data['success_rate']) * 50  # Up to 25% ROI penalty

            effective_roi = prof.marginal_roi_percent - rebal_penalty
            rebal_difficulty = round(1.0 - (success_data['success_rate'] if success_data and success_data['total'] >= 3 else 1.0), 2)

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
                    "peer_supports_splice": peer_splice_map.get(prof.peer_id, False),
                    "rebal_difficulty": rebal_difficulty,
                })

        return winners

    def _identify_losers(self, all_profitability, all_flow, peer_splice_map) -> List[Dict[str, Any]]:
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
            if success_data and success_data['total'] >= 3:
                rebal_difficulty = 1.0 - success_data['success_rate']  # 0=easy, 1=impossible

            # SCID formatting check - ensure 'x' separator
            scid_display = scid.replace(':', 'x')

            # Logic 1: FIRE SALE mode (Zombie or Deeply Underwater)
            is_fire_sale = False
            fire_sale_reason = None
            if prof.days_open > 90:
                if prof.classification == ProfitabilityClass.ZOMBIE:
                    is_fire_sale = True
                    fire_sale_reason = "ZOMBIE"
                elif prof.classification == ProfitabilityClass.UNDERWATER and prof.roi_percent < -50.0:
                    is_fire_sale = True
                    fire_sale_reason = "FIRE SALE"

            # Logic 2: Stagnant balanced channels (turnover < 0.0015)
            # PROTECTION: Only a loser if stagnant AND marginal_roi_percent < 10.0%
            is_stagnant = False
            if flow_metrics:
                # Safe ratio calculations to prevent ZeroDivisionError
                cap = flow_metrics.capacity or 0
                outbound_ratio = flow_metrics.our_balance / cap if cap > 0 else 0
                turnover = flow_metrics.daily_volume / cap if cap > 0 else 0
                if (0.4 <= outbound_ratio <= 0.6) and (turnover < 0.0015):
                    if prof.marginal_roi_percent < 10.0:
                        is_stagnant = True

            # High rebalance difficulty makes losers worse — harder to recover
            if rebal_difficulty > 0.7 and not is_fire_sale and is_stagnant:
                is_fire_sale = True
                fire_sale_reason = "STAGNANT+HARD_REBAL"

            if is_fire_sale or is_stagnant:
                # PROTECTION: A channel cannot be recommended for "Close" or "Splice-out"
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
                        "peer_supports_splice": peer_splice_map.get(prof.peer_id, False),
                        "rebal_difficulty": round(rebal_difficulty, 2),
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
                        "peer_supports_splice": peer_splice_map.get(prof.peer_id, False),
                        "rebal_difficulty": round(rebal_difficulty, 2),
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
            has_splice = winner.get('peer_supports_splice', False)

            if closeable_idx < len(sorted_closeable):
                loser = sorted_closeable[closeable_idx]
                closeable_idx += 1

                if has_splice:
                    recommendations.append(
                        f"STRATEGIC REDEPLOYMENT: Close channel {loser['scid']} ({loser['reason']}) "
                        f"and splice the funds into {winner['scid']} (ROI: {winner['roi']:.1f}%)."
                    )
                else:
                    recommendations.append(
                        f"REDEPLOYMENT: Close channel {loser['scid']} ({loser['reason']}) "
                        f"and re-open larger with {winner['scid']} (ROI: {winner['roi']:.1f}%, peer lacks splice support)."
                    )
            else:
                if has_splice:
                    recommendations.append(
                        f"GROWTH POTENTIAL: {winner['scid']} is a high ROI winner ({winner['roi']:.1f}% ROI). "
                        f"Consider splicing in more capital."
                    )
                else:
                    recommendations.append(
                        f"GROWTH POTENTIAL: {winner['scid']} is a winner ({winner['roi']:.1f}% ROI) "
                        f"but peer lacks splice support. Consider manual close/re-open larger."
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
