"""
Capacity Planner Module for cl-revenue-ops

This module identifies "Winner" channels for capital injection (Splice-In)
and "Loser" channels for capital redeployment (Splice-Out/Close).
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from pyln.client import Plugin

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
        winners = self._identify_winners()
        losers = self._identify_losers()
        
        recommendations = self._generate_recommendations(winners, losers)
        
        return {
            "timestamp": int(time.time()),
            "mempool_recommendation": mempool_rec,
            "winners": winners,
            "losers": losers,
            "recommendations": recommendations
        }

    def _get_mempool_recommendation(self) -> str:
        """Query feerates and return a recommendation based on opening costs."""
        try:
            feerates = self.plugin.rpc.feerates(style="perkb")
            perkb = feerates.get("perkb", {})
            # opening fee in perkb, we want sat/vB (divide by 1000)
            opening_kvb = perkb.get("opening", 1000)
            sat_per_vb = opening_kvb / 1000.0
            
            if sat_per_vb > 100:
                return "HOLD: On-chain fees are too high for efficient splicing."
            return "PROCEED: Fee environment is optimal."
        except Exception as e:
            self.plugin.log(f"Error checking mempool for capacity report: {e}", level='debug')
            return "UNKNOWN: Could not fetch feerates."

    def _get_peer_splice_map(self) -> Dict[str, bool]:
        """Identify which peers support splicing (bits 161/162)."""
        splice_map = {}
        try:
            peers = self.plugin.rpc.listpeers().get("peers", [])
            for peer in peers:
                peer_id = peer.get("id")
                features = peer.get("features", "")
                # Bits 161/162 for splicing
                # features is a hex string in CLN listpeers
                # We can check for 'a1' (161 set) etc but simplest is checking feature bits if available
                # or just parsing the hex.
                
                # Check for feature bits 161 or 162
                # features hex: 0-indexed from right? No, left-to-right hex.
                # CLN actually provides 'features' as hex string.
                # We can convert hex to int and check bits.
                if not features:
                    splice_map[peer_id] = False
                    continue
                    
                has_splice = False
                try:
                    feat_int = int(features, 16)
                    if (feat_int & (1 << 160)) or (feat_int & (1 << 161)):
                        has_splice = True
                except:
                    pass
                
                splice_map[peer_id] = has_splice
        except Exception as e:
            self.plugin.log(f"Error mapping peer splice support: {e}", level='debug')
            
        return splice_map

    def _identify_winners(self) -> List[Dict[str, Any]]:
        """
        Identify high-performing channels that are capacity-constrained.
        """
        winners = []
        all_profitability = self.profitability.analyze_all_channels()
        all_flow = self.flow.analyze_all_channels()
        peer_splice_map = self._get_peer_splice_map()
        
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
            
            if (prof.marginal_roi_percent > 20.0 and 
                turnover > 0.5 and 
                (flow_metrics.flow_ratio > 0.8 or flow_metrics.flow_ratio < -0.8)):
                
                winners.append({
                    "scid": scid_display,
                    "peer_id": prof.peer_id,
                    "roi": round(prof.marginal_roi_percent, 2),
                    "flow_ratio": round(flow_metrics.flow_ratio, 4),
                    "turnover": round(turnover, 4),
                    "capacity": prof.capacity_sats,
                    "peer_supports_splice": peer_splice_map.get(prof.peer_id, False)
                })
        
        return winners

    def _identify_losers(self) -> List[Dict[str, Any]]:
        """
        Identify poor-performing channels for capital extraction.
        """
        losers = []
        all_profitability = self.profitability.analyze_all_channels()
        all_flow = self.flow.analyze_all_channels()
        
        from .profitability_analyzer import ProfitabilityClass

        for scid, prof in all_profitability.items():
            flow_metrics = all_flow.get(scid)
            
            # Fetch diagnostic stats from DB
            diag_stats = self.profitability.database.get_diagnostic_rebalance_stats(scid, days=14)
            attempt_count = diag_stats["attempt_count"]
            
            # SCID formatting check - ensure 'x' separator
            scid_display = scid.replace(':', 'x')
            
            # Logic 1: FIRE SALE mode (Zombie or Deeply Underwater)
            is_fire_sale = False
            if prof.days_open > 90:
                if prof.classification == ProfitabilityClass.ZOMBIE:
                    is_fire_sale = True
                elif prof.classification == ProfitabilityClass.UNDERWATER and prof.roi_percent < -50.0:
                    is_fire_sale = True
            
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
            
            if is_fire_sale or is_stagnant:
                # PROTECTION: A channel cannot be recommended for "Close" or "Splice-out"
                # until the diagnostic_rebalance has been attempted at least twice in the last 14 days.
                if attempt_count < 2:
                    losers.append({
                        "scid": scid_display,
                        "peer_id": prof.peer_id,
                        "reason": "STAGNANT (NEEDS DEFIBRILLATOR)",
                        "roi": round(prof.roi_percent, 2),
                        "marginal_roi": round(prof.marginal_roi_percent, 2),
                        "classification": prof.classification.value if hasattr(prof.classification, 'value') else str(prof.classification),
                        "capacity": prof.capacity_sats,
                        "action": "DEFIBRILLATE"
                    })
                else:
                    losers.append({
                        "scid": scid_display,
                        "peer_id": prof.peer_id,
                        "reason": "FIRE SALE" if is_fire_sale else "STAGNANT",
                        "roi": round(prof.roi_percent, 2),
                        "marginal_roi": round(prof.marginal_roi_percent, 2),
                        "classification": prof.classification.value if hasattr(prof.classification, 'value') else str(prof.classification),
                        "capacity": prof.capacity_sats,
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
        # Sort losers by capacity (descending)
        sorted_losers = sorted(losers, key=lambda x: x['capacity'], reverse=True)
        
        for winner in sorted_winners:
            # Splicing support check
            splice_action = "STRATEGIC REDEPLOYMENT: Close"
            splice_target = f"Splice the funds into {winner['scid']}"
            
            if not winner.get('peer_supports_splice', False):
                splice_action = "UPGRADE REQUIRED: "
                splice_target = f"{winner['scid']} is a winner but peer lacks splice support. Consider manual Close/Re-open larger."

            if not sorted_losers:
                if winner.get('peer_supports_splice', False):
                    recommendations.append(
                        f"GROWTH POTENTIAL: {winner['scid']} is a high ROI winner ({winner['roi']:.1f}% ROI). "
                        f"Consider splicing in more capital."
                    )
                else:
                    recommendations.append(
                        f"UPGRADE REQUIRED: {winner['scid']} is a winner but peer lacks splice support. "
                        f"Consider manual Close/Re-open larger to increase capacity."
                    )
                continue
                
            # Try to pair with a loser
            loser = sorted_losers.pop(0)
            
            if loser.get("action") == "DEFIBRILLATE":
                recommendations.append(
                    f"STAGNANT ALERT: {loser['scid']} is stagnant. "
                    f"Diagnostic rebalance required before extraction ({loser['roi']:.1f}% ROI)."
                )
            elif winner.get('peer_supports_splice', False):
                recommendations.append(
                    f"STRATEGIC REDEPLOYMENT: Close channel {loser['scid']} ({loser['reason']}) "
                    f"and Splice the funds into {winner['scid']} (ROI: {winner['roi']:.1f}%)."
                )
            else:
                recommendations.append(
                    f"UPGRADE REQUIRED: {winner['scid']} is a winner ({winner['roi']:.1f}% ROI) but peer lacks splice support. "
                    f"Pair with closure of {loser['scid']} ({loser['reason']}) for manual Re-open larger."
                )
            
        return recommendations
