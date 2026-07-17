"""
Demand Flow Classifier for capacity planner.

Classifies network nodes as sources, sinks, or routers based on
internal flow data and gossip heuristics. Used to improve channel
opening candidate scoring.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .utils import parse_msat


EXCHANGE_KEYWORDS = [
    "kraken", "coinbase", "okx", "bitfinex", "binance", "bitstamp",
    "nicehash", "river", "strike", "cashapp", "robinhood", "gemini",
    "bitget", "bybit", "kucoin", "huobi", "gate.io",
]

SINK_KEYWORDS = [
    "wallet", "pay", "shop", "store", "merchant", "pos",
    "btcpay", "coinos", "zebedee", "fountain", "stacker",
]

LSP_KEYWORDS = [
    "lnbig", "lqwd", "acinq", "breez", "phoenix", "muun",
    "olympus", "voltage", "greenlight",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert loose RPC values to float without letting bad gossip abort scoring."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class NodeFlowProfile:
    node_id: str
    role: str = "unknown"
    confidence: float = 0.0
    net_flow_ratio: Optional[float] = None
    gossip_signals: Dict[str, float] = field(default_factory=dict)
    has_liquidity_ads: bool = False


class DemandFlowClassifier:
    """Classifies nodes by payment flow role using internal data and gossip."""

    def classify_peers(self, all_flow: Dict[str, Any]) -> Dict[str, NodeFlowProfile]:
        """Aggregate per-channel FlowMetrics to per-peer flow profiles."""
        peer_in: Dict[str, int] = {}
        peer_out: Dict[str, int] = {}
        for flow in all_flow.values():
            pid = getattr(flow, 'peer_id', None)
            if not pid:
                continue
            peer_in[pid] = peer_in.get(pid, 0) + getattr(flow, 'sats_in', 0)
            peer_out[pid] = peer_out.get(pid, 0) + getattr(flow, 'sats_out', 0)

        profiles: Dict[str, NodeFlowProfile] = {}
        for pid in sorted(set(peer_in) | set(peer_out)):
            total_in = peer_in.get(pid, 0)
            total_out = peer_out.get(pid, 0)
            total = total_in + total_out

            if total == 0:
                profiles[pid] = NodeFlowProfile(node_id=pid)
                continue

            ratio = (total_in - total_out) / total
            if ratio > 0.3:
                role = "source"
            elif ratio < -0.3:
                role = "sink"
            else:
                role = "router"

            confidence = min(0.9, 0.3 * math.log10(max(total, 1)) / math.log10(1_000_000))
            confidence = max(0.1, confidence)

            profiles[pid] = NodeFlowProfile(
                node_id=pid,
                role=role,
                confidence=round(confidence, 3),
                net_flow_ratio=round(ratio, 4),
            )

        return profiles

    def classify_candidate(
        self,
        node_id: str,
        node_info: Optional[Dict] = None,
        channels: Optional[List[Dict]] = None,
    ) -> NodeFlowProfile:
        """Classify a candidate node using gossip heuristics."""
        node_info = node_info or {}
        channels = channels or []
        alias = str(node_info.get("alias", "") or "")
        alias_lower = alias.lower()

        signals: Dict[str, float] = {}
        source_score = 0.0
        sink_score = 0.0
        router_score = 0.0

        # Heuristic 1: Alias pattern matching
        if any(kw in alias_lower for kw in EXCHANGE_KEYWORDS):
            signals["alias_exchange"] = 0.6
            source_score += 0.6

        if any(kw in alias_lower for kw in SINK_KEYWORDS):
            signals["alias_sink"] = 0.5
            sink_score += 0.5

        if any(kw in alias_lower for kw in LSP_KEYWORDS):
            signals["alias_lsp"] = 0.4
            router_score += 0.4

        # Heuristic 2: Channel structure analysis
        active = [ch for ch in channels if isinstance(ch, dict) and ch.get("active", False)]
        if active:
            count = len(active)
            total_cap_msat = sum(
                max(0, parse_msat(ch.get("amount_msat", 0)))
                for ch in active
            )
            total_cap_btc = total_cap_msat / 100_000_000_000
            avg_cap = total_cap_msat // count if count else 0

            if count > 100 and total_cap_btc > 5:
                signals["structure_hub"] = 0.5
                router_score += 0.5
            elif count > 30 and avg_cap < 500_000_000:
                signals["structure_sink"] = 0.4
                sink_score += 0.4
            elif count < 10 and avg_cap > 5_000_000_000:
                signals["structure_source"] = 0.3
                source_score += 0.3

        # Heuristic 3: Fee policy
        if active:
            low_fee_count = sum(
                1 for ch in active
                if parse_msat(ch.get("base_fee_millisatoshi", 1000)) == 0
                and _safe_float(ch.get("fee_per_millionth", 1000), 1000.0) < 50
            )
            if low_fee_count > len(active) * 0.5:
                signals["fee_sink"] = 0.3
                sink_score += 0.3

            high_fee_count = sum(
                1 for ch in active
                if _safe_float(ch.get("fee_per_millionth", 0), 0.0) > 500
            )
            if high_fee_count > len(active) * 0.5:
                signals["fee_extractive"] = -0.2

        # Heuristic 4: Liquidity ads
        has_lads = "option_will_fund" in node_info and bool(node_info["option_will_fund"])

        # Combine signals
        total = source_score + sink_score + router_score
        if total == 0:
            role = "unknown"
            confidence = 0.0
        else:
            if source_score >= sink_score and source_score >= router_score:
                role = "source"
                confidence = source_score / total
            elif sink_score >= source_score and sink_score >= router_score:
                role = "sink"
                confidence = sink_score / total
            else:
                role = "router"
                confidence = router_score / total

        return NodeFlowProfile(
            node_id=node_id,
            role=role,
            confidence=round(confidence, 3),
            gossip_signals=signals,
            has_liquidity_ads=has_lads,
        )

    def find_sink_adjacent_candidates(
        self,
        sink_profiles: Dict[str, NodeFlowProfile],
        sink_channels: Dict[str, List[Dict]],
        existing_peers: set,
    ) -> List[Dict]:
        """Find candidates adjacent to our known sink peers."""
        if not sink_profiles:
            return []

        ranked_sinks = sorted(
            sink_profiles.values(),
            key=lambda p: abs(p.net_flow_ratio or 0),
            reverse=True,
        )[:5]

        candidates = []
        seen = set()

        for rank, sink in enumerate(ranked_sinks):
            channels = sink_channels.get(sink.node_id, [])
            for ch in channels:
                dest = ch.get("destination")
                if not dest or dest in existing_peers or dest in seen:
                    continue
                if not ch.get("active", False):
                    continue

                score = 0.4 * sink.confidence * (1 + (len(ranked_sinks) - rank) / len(ranked_sinks))
                candidates.append({
                    "peer_id": dest,
                    "source": "demand_flow",
                    "score": round(score, 4),
                    "reason": f"Adjacent to sink {sink.node_id[:12]}... (conf={sink.confidence})",
                    "sink_peer_id": sink.node_id,
                    "is_sink_adjacent": True,
                })
                seen.add(dest)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:10]
