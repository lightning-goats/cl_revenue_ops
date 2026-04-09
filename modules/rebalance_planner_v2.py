"""Rebalance planner v2 — unified pair-based planning.

No separate hive push / hive equalization / capex paths. One planner
that pairs over-local channels with over-remote channels, scores them
by value and imbalance, and emits explicit skip reasons for everything
it doesn't select.
"""

from __future__ import annotations

from typing import List, Tuple

from .rebalance_state_v2 import RebalanceStateV2Channel, RebalanceStateV2Snapshot
from .rebalance_types_v2 import V2PairCandidate, V2PlanResult, V2SkipRecord


# Value class scores — hive channels win when routes are cheap, not via fake bonuses
_VALUE_SCORES = {
    "hive": 3,
    "profitable": 2,
    "active": 1,
    "neutral": 0,
}


class RebalancePlannerV2:
    """Pair-based rebalance planner using actual fees and budgets."""

    def __init__(
        self,
        target_band_low: float = 0.35,
        target_band_high: float = 0.65,
        max_chunk_sats: int = 2_000_000,
        max_pairs: int = 10,
    ):
        self.target_band_low = target_band_low
        self.target_band_high = target_band_high
        self.max_chunk_sats = max_chunk_sats
        self.max_pairs = max_pairs

    def plan(self, snapshot: RebalanceStateV2Snapshot) -> V2PlanResult:
        """Classify channels, generate pairs, score and select."""
        over_local: List[RebalanceStateV2Channel] = []
        over_remote: List[RebalanceStateV2Channel] = []
        skipped: List[V2SkipRecord] = []

        # Phase 1: classify every channel
        for ch in snapshot.channels:
            skip = self._check_skip(ch)
            if skip is not None:
                skipped.append(skip)
                continue

            if ch.local_ratio > self.target_band_high:
                over_local.append(ch)
            elif ch.local_ratio < self.target_band_low:
                over_remote.append(ch)
            else:
                skipped.append(V2SkipRecord(
                    channel_id=ch.channel_id,
                    reason="inside_band",
                    value_class=ch.value_class,
                    remaining_budget_sats=ch.remaining_budget_sats,
                ))

        # Phase 2: generate candidate pairs
        paired_sources = set()
        paired_dests = set()
        candidates: List[V2PairCandidate] = []

        pairs = self._generate_pairs(over_local, over_remote)
        pairs.sort(key=lambda p: p.score, reverse=True)

        for pair in pairs:
            if len(candidates) >= self.max_pairs:
                break
            if pair.source_channel_id in paired_sources:
                continue
            if pair.dest_channel_id in paired_dests:
                continue
            candidates.append(pair)
            paired_sources.add(pair.source_channel_id)
            paired_dests.add(pair.dest_channel_id)

        # Phase 3: emit no_partner skips for unpaired valuable channels
        for ch in over_local:
            if ch.channel_id not in paired_sources:
                if not over_remote:
                    skipped.append(V2SkipRecord(
                        channel_id=ch.channel_id,
                        reason="no_partner",
                        value_class=ch.value_class,
                        remaining_budget_sats=ch.remaining_budget_sats,
                        detail="no over-remote channels available",
                    ))

        for ch in over_remote:
            if ch.channel_id not in paired_dests:
                if not over_local:
                    skipped.append(V2SkipRecord(
                        channel_id=ch.channel_id,
                        reason="no_partner",
                        value_class=ch.value_class,
                        remaining_budget_sats=ch.remaining_budget_sats,
                        detail="no over-local channels available",
                    ))

        return V2PlanResult(selected=candidates, skipped=skipped)

    def _check_skip(self, ch: RebalanceStateV2Channel):
        """Return a V2SkipRecord if the channel should be skipped pre-pairing."""
        if not ch.is_valuable:
            return V2SkipRecord(
                channel_id=ch.channel_id,
                reason="not_valuable",
                value_class=ch.value_class,
                remaining_budget_sats=ch.remaining_budget_sats,
            )
        if ch.cooldown_active:
            return V2SkipRecord(
                channel_id=ch.channel_id,
                reason="cooldown",
                value_class=ch.value_class,
                remaining_budget_sats=ch.remaining_budget_sats,
            )
        if ch.remaining_budget_sats <= 0:
            return V2SkipRecord(
                channel_id=ch.channel_id,
                reason="no_budget",
                value_class=ch.value_class,
                remaining_budget_sats=0,
            )
        return None

    def _generate_pairs(
        self,
        over_local: List[RebalanceStateV2Channel],
        over_remote: List[RebalanceStateV2Channel],
    ) -> List[V2PairCandidate]:
        """Generate all valid candidate pairs with scores."""
        pairs = []
        for src in over_local:
            for dest in over_remote:
                if src.peer_id == dest.peer_id:
                    continue

                source_excess = int(
                    (src.local_ratio - self.target_band_high) * src.capacity_sats
                )
                dest_need = int(
                    (self.target_band_low - dest.local_ratio) * dest.capacity_sats
                )
                amount = min(
                    max(0, source_excess),
                    max(0, dest_need),
                    self.max_chunk_sats,
                )
                if amount <= 0:
                    continue

                pair_budget = max(src.remaining_budget_sats, dest.remaining_budget_sats)

                # Score: value * imbalance
                src_value = _VALUE_SCORES.get(src.value_class, 0)
                dest_value = _VALUE_SCORES.get(dest.value_class, 0)
                value_score = max(src_value, dest_value)

                src_imbalance = max(0.0, src.local_ratio - self.target_band_high)
                dest_imbalance = max(0.0, self.target_band_low - dest.local_ratio)
                imbalance_score = (src_imbalance + dest_imbalance) / 2.0

                score = value_score * imbalance_score

                pairs.append(V2PairCandidate(
                    source_channel_id=src.channel_id,
                    dest_channel_id=dest.channel_id,
                    source_peer_id=src.peer_id,
                    dest_peer_id=dest.peer_id,
                    amount_sats=amount,
                    pair_budget_sats=pair_budget,
                    score=score,
                    source_local_ratio=src.local_ratio,
                    dest_local_ratio=dest.local_ratio,
                ))

        return pairs
