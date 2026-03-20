"""
Capacity Planner Module for cl-revenue-ops

This module identifies "Winner" channels for capital injection
and "Loser" channels for capital redeployment (Close).
"""

import json
import math
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

    def __init__(self, plugin: Plugin, profitability_analyzer, flow_analyzer, policy_manager=None, config=None):
        self.plugin = plugin
        self.profitability = profitability_analyzer
        self.flow = flow_analyzer
        self.policy_manager = policy_manager
        self.config = config
        self.rebalancer = None
        self._pending_closes: Dict[str, int] = {}

    def is_pending_close(self, channel_id: str) -> bool:
        """Check if a channel is pending close by the planner."""
        return channel_id in self._pending_closes

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

    def execute_cycle(self, cfg=None) -> Dict[str, Any]:
        """Main timer-driven cycle. Evaluates and executes open/close decisions."""
        if cfg is None:
            cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        if not cfg.planner_enabled:
            return {"skipped": True, "reason": "planner disabled"}

        summary = {
            "opens": [],
            "closes": [],
            "drains_progressed": [],
            "skipped_reasons": [],
        }

        # 1. Check fee gate
        fee_ok, fee_reason = self._check_fee_gate(cfg)
        if not fee_ok:
            summary["skipped_reasons"].append(fee_reason)

        # 2. Fetch analysis data
        all_profitability = self.profitability.analyze_all_channels()
        all_flow = self.flow.analyze_all_channels()

        # 3. Identify winners and losers
        winners = self._identify_winners(all_profitability, all_flow)
        losers = self._identify_losers(all_profitability, all_flow)

        # 4. Progress existing drains
        db = self.profitability.database if self.profitability else None
        for channel_id in list(self._pending_closes.keys()):
            if self._check_drain_complete(channel_id, cfg):
                # Find the matching draining action from DB
                action_id = None
                peer_id = None
                if db:
                    try:
                        draining_actions = db.get_planner_actions(status='draining')
                        for action in draining_actions:
                            if action.get('channel_id') == channel_id:
                                action_id = action.get('id')
                                peer_id = action.get('peer_id')
                                break
                    except Exception as e:
                        self.plugin.log(
                            f"Error fetching draining actions for {channel_id}: {e}",
                            level='debug',
                        )

                if action_id and peer_id:
                    result = self._execute_close(channel_id, peer_id, action_id, cfg)
                    summary["drains_progressed"].append({
                        "channel_id": channel_id,
                        "peer_id": peer_id,
                        "action_id": action_id,
                        "result": result.get("status", "unknown"),
                    })
                else:
                    # Drain complete but no matching DB record; clean up tracking
                    del self._pending_closes[channel_id]
                    summary["drains_progressed"].append({
                        "channel_id": channel_id,
                        "result": "orphaned_drain_cleaned",
                    })

        # 5. Initiate new closes (up to max_closes_per_cycle)
        closes_this_cycle = 0
        closeable = [l for l in losers if l.get("action") == "CLOSE"]
        sorted_closeable = sorted(closeable, key=lambda x: x.get("marginal_roi", 0))
        for loser in sorted_closeable:
            if closes_this_cycle >= cfg.planner_max_closes_per_cycle:
                break
            scid = loser.get("scid")
            peer_id = loser.get("peer_id")
            if scid in self._pending_closes:
                continue

            # Check policy allows close
            close_ok, close_reason = self._check_close_allowed(peer_id)
            if not close_ok:
                summary["skipped_reasons"].append(
                    f"Close blocked for {scid}: {close_reason}"
                )
                continue

            # Check cooldown
            cooldown_ok, cooldown_reason = self._check_cooldown(peer_id)
            if not cooldown_ok:
                summary["skipped_reasons"].append(
                    f"Close cooldown for {scid}: {cooldown_reason}"
                )
                continue

            action_id = self._initiate_drain(peer_id, scid, cfg, loser.get("reason", ""))
            summary["closes"].append({
                "scid": scid,
                "peer_id": peer_id,
                "reason": loser.get("reason", ""),
                "action_id": action_id,
            })
            closes_this_cycle += 1

        # 6. Discover and open channels (up to max_opens_per_cycle)
        candidates = []
        if fee_ok:
            candidates = self._discover_peers(winners, all_profitability, all_flow)

            # Get available funds for sizing
            available_sats = 0
            try:
                funds = self.plugin.rpc.listfunds()
                confirmed = sum(
                    o.get("amount_msat", 0) // 1000
                    for o in funds.get("outputs", [])
                    if o.get("status") == "confirmed"
                )
                min_reserve = getattr(cfg, 'min_wallet_reserve', 500000)
                available_sats = max(0, confirmed - min_reserve)
            except Exception as e:
                self.plugin.log(f"Cannot determine available funds: {e}", level='debug')

            opens_this_cycle = 0
            for candidate in sorted(candidates, key=lambda c: c.get("score", 0), reverse=True):
                if opens_this_cycle >= cfg.planner_max_opens_per_cycle:
                    break

                peer_id = candidate["peer_id"]

                # Size channel
                channel_size = self._size_channel(candidate, candidates, available_sats, cfg)

                # Check EV
                ev = self._calculate_open_ev(peer_id, channel_size, cfg)
                if ev <= 0:
                    summary["skipped_reasons"].append(
                        f"Negative EV ({ev:.0f}) for {peer_id[:16]}..."
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
                opens_this_cycle += 1

                # Reduce available sats for next candidate
                available_sats = max(0, available_sats - channel_size)

        # 7. Update candidate pool
        self._update_candidate_pool(candidates if fee_ok else [])

        summary["timestamp"] = int(time.time())
        return summary

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

        Enriched with additional data signals for smarter closure decisions:
        - Hard bleeder bypass: structurally unprofitable channels skip defibrillation gate
        - Channel role protection: INBOUND_GATEWAYs require much worse ROI for closure
        - Kalman regime change deferral: demotes CLOSE to DEFIBRILLATE if behavior shifting
        - Kalman confidence gate: skips closure for channels with unreliable flow data
        - Peer uptime: added to loser dict for downstream decision-making
        """
        losers = []

        from .profitability_analyzer import ProfitabilityClass, ChannelRole

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

            # Kalman confidence gate -- skip closure if data unreliable
            if flow_metrics:
                confidence = getattr(flow_metrics, 'confidence', 1.0)
                # Guard against non-numeric (e.g. MagicMock) or None
                if not isinstance(confidence, (int, float)):
                    confidence = 1.0
                confidence = confidence or 1.0
                if confidence < 0.5:
                    continue  # Don't recommend closure with unreliable data

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

            # If inbound gateway, require much worse marginal ROI before closure
            if is_inbound_gateway and prof.marginal_roi_percent > -50.0:
                continue  # Protect inbound gateways

            # Hard bleeder bypass -- skip defibrillation gate
            bleeder_info = bleeders.get(scid)
            is_hard_bleeder = False
            if bleeder_info is not None:
                if hasattr(bleeder_info, 'is_hard_bleeder'):
                    is_hard_bleeder = bool(bleeder_info.is_hard_bleeder)
                elif isinstance(bleeder_info, dict):
                    is_hard_bleeder = bool(bleeder_info.get('is_hard_bleeder', False))

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
                estimated_closure_cost = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS
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
                    "uptime_pct": round(uptime_pct, 1) if isinstance(uptime_pct, (int, float)) else None,
                    "regime_change": regime_change,
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
            info = self.plugin.rpc.getinfo()
            our_node_id = info.get("id")
        except Exception:
            return []

        # Sort channels by marginal ROI, take top 3
        sorted_channels = sorted(
            all_profitability.values(),
            key=lambda p: getattr(p, 'marginal_roi_percent', 0),
            reverse=True
        )[:3]

        # Get existing peer_ids to exclude
        existing_peers = set()
        for prof in all_profitability.values():
            if hasattr(prof, 'peer_id') and prof.peer_id:
                existing_peers.add(prof.peer_id)

        for patron in sorted_channels:
            patron_peer_id = getattr(patron, 'peer_id', None)
            if not patron_peer_id:
                continue

            try:
                channels = self.plugin.rpc.listchannels(source=patron_peer_id)
                neighbor_ids = set()
                for ch in channels.get("channels", []):
                    dest = ch.get("destination")
                    if dest and dest != our_node_id and dest not in existing_peers:
                        neighbor_ids.add(dest)

                # Take up to 5 neighbors per patron
                patron_roi = getattr(patron, 'marginal_roi_percent', 0)
                for neighbor_id in list(neighbor_ids)[:5]:
                    candidates.append({
                        "peer_id": neighbor_id,
                        "source": "neighbor",
                        "score": max(patron_roi / 200.0, 0.1),  # Scale patron ROI
                        "reason": f"Neighbor of top earner {patron_peer_id[:12]}...",
                        "patron_peer_id": patron_peer_id,
                    })
            except Exception as e:
                self.plugin.log(f"Error discovering neighbors of {patron_peer_id[:12]}: {e}", level='debug')
                continue

        return candidates[:10]  # Max 10 total from this strategy

    def _discover_from_graph(self, existing_peer_ids: set) -> List[Dict]:
        """Strategy 3: Network centrality scoring via listnodes."""
        try:
            nodes = self.plugin.rpc.listnodes().get("nodes", [])
        except Exception:
            return []

        if len(nodes) < 800:
            self.plugin.log(
                f"Insufficient graph knowledge ({len(nodes)} nodes, need 800+)",
                level='debug',
            )
            return []

        # Get our own node ID
        try:
            our_node_id = self.plugin.rpc.getinfo().get("id")
        except Exception:
            return []

        scored = []
        for node in nodes:
            node_id = node.get("nodeid")
            if not node_id or node_id == our_node_id or node_id in existing_peer_ids:
                continue

            # Channel count from the node's channel_count field if available
            channel_count = node.get("channel_count", 0)
            if channel_count < 5:
                continue  # Skip poorly connected nodes

            # Total capacity (if available)
            total_capacity = node.get("total_capacity", 0)
            if isinstance(total_capacity, str):
                total_capacity = (
                    int(total_capacity.replace("msat", "")) // 1000
                    if "msat" in total_capacity
                    else int(total_capacity)
                )

            # Compute centrality score = channel_count * sqrt(capacity_btc)
            capacity_btc = total_capacity / 100_000_000 if total_capacity > 0 else 0.001
            score = channel_count * math.sqrt(capacity_btc)

            scored.append({
                "peer_id": node_id,
                "source": "graph",
                "score": score,
                "reason": f"Graph centrality: {channel_count} channels, {total_capacity} sat capacity",
                "channel_count": channel_count,
                "total_capacity": total_capacity,
            })

        # Sort by score descending, take top 10
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:10]

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

        return score

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

    def _check_fee_gate(self, cfg) -> tuple:
        """Check on-chain fee rate is acceptable. Returns (ok, reason)."""
        try:
            feerates = self.plugin.rpc.feerates(style="perkb")
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
            funds = self.plugin.rpc.listfunds()
            confirmed = sum(
                o.get("amount_msat", 0) // 1000
                for o in funds.get("outputs", [])
                if o.get("status") == "confirmed"
            )
            # Handle msat values that might be strings
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
            if recent:
                return False, f"Cooldown: {len(recent)} action(s) for peer in last 24h"
            return True, "No recent actions for peer"
        except Exception as e:
            return True, f"Cooldown check failed (allowing): {e}"

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

        # Persist to candidate pool
        self._update_candidate_pool(merged)

        return merged

    def _size_channel(self, candidate: Dict, all_candidates: List[Dict],
                       available_sats: int, cfg) -> int:
        """ROI-proportional channel sizing.

        Allocates available funds proportionally to candidate scores.
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

        # Clamp to config bounds
        return max(cfg.planner_min_channel_sats,
                   min(raw_size, cfg.planner_max_channel_sats))

    def _calculate_open_ev(self, peer_id: str, channel_size_sats: int, cfg) -> float:
        """EV-based channel open decision. Returns expected profit in sats.

        EV = expected_lifetime_revenue - on_chain_cost - expected_rebalance_costs
        Only open when EV > 0.
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

        # Fallback: estimate from channel size assuming modest utilization
        if daily_revenue <= 0:
            # Assume 30% utilization and 150 PPM average fee
            daily_revenue = channel_size_sats * 0.3 * 150 / 1_000_000

        # Estimate on-chain costs
        try:
            feerates = self.plugin.rpc.feerates(style="perkb")
            sat_per_vb = feerates.get("perkb", {}).get("opening", 1000) / 1000.0
            open_cost = int(sat_per_vb * 140)   # ~140 vbytes for open tx
            close_cost = int(sat_per_vb * 200)  # ~200 vbytes for close tx
        except Exception:
            open_cost = ChainCostDefaults.CHANNEL_OPEN_COST_SATS
            close_cost = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS

        on_chain_cost = open_cost + close_cost

        # Estimate rebalance costs
        # Conservative: assume 10% of revenue goes to rebalancing
        rebal_cost_per_day = daily_revenue * 0.1

        # Conservative 6-month lifetime estimate
        lifetime_days = 180

        expected_revenue = daily_revenue * lifetime_days
        expected_rebal_cost = rebal_cost_per_day * lifetime_days

        ev = expected_revenue - on_chain_cost - expected_rebal_cost
        return ev

    def _estimate_open_cost(self) -> int:
        """Estimate the on-chain cost of opening a channel."""
        try:
            feerates = self.plugin.rpc.feerates(style="perkb")
            sat_per_vb = feerates.get("perkb", {}).get("opening", 1000) / 1000.0
            return int(sat_per_vb * 140)  # ~140 vbytes for funding tx
        except Exception:
            return ChainCostDefaults.CHANNEL_OPEN_COST_SATS

    def _execute_open(self, peer_id: str, amount_sats: int, cfg, reason: str) -> Dict:
        """Execute a channel open via fundchannel RPC.

        Flow:
        1. Record planned action in database
        2. If dry_run, log and return early
        3. Reserve budget via generic spend ledger
        4. Connect to peer
        5. Call fundchannel
        6. On success: update action, mark spend
        7. On failure: update action, release reservation
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
        if db:
            try:
                db.reserve_spend(
                    reservation_id=reservation_id,
                    amount_sats=estimated_cost,
                    category="channel_open",
                    subcategory="automated",
                    metadata={"peer_id": peer_id, "amount_sats": amount_sats},
                )
            except Exception as e:
                self.plugin.log(f"Budget reservation failed: {e}", level='warn')

        try:
            # Connect first (may already be connected)
            try:
                self.plugin.rpc.connect(peer_id)
            except Exception:
                pass  # Connection may already exist

            # Fund channel
            result = self.plugin.rpc.fundchannel(
                id=peer_id,
                amount=amount_sats,
                announce=True,
            )

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
                    )
                except Exception:
                    pass

            self.plugin.log(
                f"Channel opened: {channel_id} to {peer_id[:16]}... ({amount_sats} sats)",
                level='info'
            )
            return {"action_id": action_id, "status": "completed", "channel_id": channel_id, "peer_id": peer_id}

        except Exception as e:
            # Failure: update action and release reservation
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="failed")
                except Exception:
                    pass
            if db:
                try:
                    db.release_spend_reservation(reservation_id)
                except Exception:
                    pass

            self.plugin.log(f"Channel open failed for {peer_id[:16]}...: {e}", level='error')
            return {"action_id": action_id, "status": "failed", "error": str(e), "peer_id": peer_id}

    def _check_close_allowed(self, peer_id: str) -> tuple:
        """Check if a channel close is allowed for this peer.

        Channels with static policy or tagged 'protect'/'no_close' must
        never be closed.

        Returns:
            (allowed, reason) tuple.
        """
        if not self.policy_manager:
            return True, "No policy manager configured"

        try:
            policy = self.policy_manager.get_policy(peer_id)

            # Static policy channels are never closed
            strategy = policy.strategy
            strategy_str = strategy.value if hasattr(strategy, 'value') else str(strategy)
            if strategy_str == "static":
                return False, "Channel has static policy — close blocked"

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

        return True, "Close allowed"

    def _initiate_drain(self, peer_id: str, channel_id: str, cfg, reason: str) -> int:
        """Phase 1: Set policy to drain, record action.

        Sets the channel to passive+source_only so it drains naturally
        before close, and records a draining action in the database.

        Returns:
            The action_id from the database record.
        """
        db = self.profitability.database if self.profitability else None

        action_id = None
        if db:
            try:
                action_id = db.record_planner_action(
                    action_type="close",
                    peer_id=peer_id,
                    channel_id=channel_id,
                    estimated_cost_sats=ChainCostDefaults.CHANNEL_CLOSE_COST_SATS,
                    reason=reason,
                )
            except Exception as e:
                self.plugin.log(f"Failed to record drain action: {e}", level='warn')

        if cfg.planner_dry_run:
            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="dry_run")
                except Exception:
                    pass
            self.plugin.log(
                f"[DRY RUN] Would drain and close {channel_id} "
                f"(peer: {peer_id[:16]}..., reason: {reason})",
                level='info',
            )
            return action_id

        # Set drain policy
        if self.policy_manager:
            try:
                self.policy_manager.set_policy(
                    peer_id,
                    strategy="passive",
                    rebalance_mode="source_only",
                    tags=["closing", "drain_phase"],
                    expires_in_hours=cfg.planner_drain_timeout_hours,
                )
            except Exception as e:
                self.plugin.log(
                    f"Failed to set drain policy for {peer_id[:12]}...: {e}",
                    level='warn',
                )

        if db and action_id:
            try:
                db.update_planner_action(action_id, status="draining")
            except Exception:
                pass

        self._pending_closes[channel_id] = int(time.time())
        return action_id

    def _check_drain_complete(self, channel_id: str, cfg) -> bool:
        """Check if drain phase is complete.

        Drain is complete when either:
        - Local balance is less than 10% of channel capacity, or
        - The drain timeout has elapsed.
        """
        # Check timeout
        drain_start = self._pending_closes.get(channel_id)
        if drain_start:
            elapsed_hours = (time.time() - drain_start) / 3600
            if elapsed_hours > cfg.planner_drain_timeout_hours:
                return True

        # Check balance
        try:
            channels = self.plugin.rpc.listpeerchannels()
            for ch in channels.get("channels", []):
                if ch.get("short_channel_id") == channel_id or ch.get("channel_id") == channel_id:
                    local = ch.get("spendable_msat", 0)
                    if isinstance(local, str):
                        local = int(local.replace("msat", ""))
                    capacity = ch.get("total_msat", 1)
                    if isinstance(capacity, str):
                        capacity = int(capacity.replace("msat", ""))
                    if capacity > 0 and (local / capacity) < 0.1:
                        return True
        except Exception as e:
            self.plugin.log(
                f"Error checking drain status for {channel_id}: {e}",
                level='debug',
            )

        return False

    def _execute_close(self, channel_id: str, peer_id: str, action_id: int, cfg) -> Dict:
        """Phase 2: Execute channel close.

        Stops any active rebalancer jobs on the channel, then calls
        the close RPC. Updates the planner action record with the result.
        """
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

        db = self.profitability.database if self.profitability else None

        try:
            result = self.plugin.rpc.close(id=channel_id)

            if db and action_id:
                try:
                    db.update_planner_action(action_id, status="completed")
                except Exception:
                    pass

            if channel_id in self._pending_closes:
                del self._pending_closes[channel_id]

            self.plugin.log(
                f"Channel closed: {channel_id} (peer: {peer_id[:16]}...)",
                level='info',
            )
            return {"action_id": action_id, "status": "completed", "result": result}

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
