"""Rebalance planner v2 — unified pair-based planning.

No separate hive push / hive equalization / capex paths. One planner
that pairs over-local channels with over-remote channels, scores them
by value and imbalance, and emits explicit skip reasons for everything
it doesn't select.
"""

from __future__ import annotations

from typing import List, Tuple

from .rebalance_state_v2 import ChannelState, StateSnapshot
from .rebalance_types_v2 import PairCandidate, PlanResult, SkipRecord


# Value class scores — hive channels win when routes are cheap, not via fake bonuses
_VALUE_SCORES = {
    "hive": 3,
    "profitable": 2,
    "active": 1,
    "funded": 1,  # capex budget approved — bootstrap inventory
    "neutral": 0,
}


def _bootstrap_score_decomposition(
    *,
    value_score: int,
    imbalance_score: float,
    pair_score: float,
    amount_sats: int,
    pair_budget_sats: int,
    source_local_ratio: float,
    dest_local_ratio: float,
) -> dict:
    """Return the initial explicit score breakdown for a planned pair.

    This is intentionally lightweight and descriptive. It exposes the current
    heuristic planner state before route pricing or empirical learning updates
    are applied by the engine.
    """
    p_success = 0.5
    expected_future_value = round(float(pair_score), 6)
    final_score = round(p_success * expected_future_value, 6)
    return {
        "model_version": "v2-bootstrap-explainability",
        "score_units": "planner_score_minus_budget_share",
        "stage": "planner_pre_route",
        "p_success": p_success,
        "expected_future_value": expected_future_value,
        "expected_fee": 0.0,
        "source_opportunity_cost": 0.0,
        "failure_penalty": 0.0,
        "capital_risk_penalty": 0.0,
        "do_nothing_score": 0.0,
        "final_score": final_score,
        "beats_do_nothing": final_score > 0.0,
        "rejection_reason": "",
        "inputs": {
            "value_score": int(value_score),
            "imbalance_score": round(float(imbalance_score), 6),
            "amount_sats": int(amount_sats),
            "pair_budget_sats": int(pair_budget_sats),
            "source_local_ratio": round(float(source_local_ratio), 6),
            "dest_local_ratio": round(float(dest_local_ratio), 6),
        },
    }


class RebalancePlanner:
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

    def plan(self, snapshot: StateSnapshot) -> PlanResult:
        """Classify channels, generate pairs, score and select."""
        over_local: List[ChannelState] = []
        over_remote: List[ChannelState] = []
        skipped: List[SkipRecord] = []

        # Phase 1: classify by band first, then apply role-specific eligibility.
        # Phase 2 of the post-Polar remediation: a neutral over-local channel is
        # a valid drain source even though it can never be a refill destination.
        for ch in snapshot.channels:
            if ch.local_ratio > self.target_band_high:
                if ch.source_eligible:
                    over_local.append(ch)
                else:
                    skipped.append(SkipRecord(
                        channel_id=ch.channel_id,
                        reason=ch.source_reason or "source_ineligible",
                        value_class=ch.value_class,
                        remaining_budget_sats=ch.remaining_budget_sats,
                    ))
            elif ch.local_ratio < self.target_band_low:
                if ch.dest_eligible:
                    over_remote.append(ch)
                else:
                    skipped.append(SkipRecord(
                        channel_id=ch.channel_id,
                        reason=ch.dest_reason or "dest_ineligible",
                        value_class=ch.value_class,
                        remaining_budget_sats=ch.remaining_budget_sats,
                    ))
            else:
                skipped.append(SkipRecord(
                    channel_id=ch.channel_id,
                    reason="inside_band",
                    value_class=ch.value_class,
                    remaining_budget_sats=ch.remaining_budget_sats,
                ))

        # Phase 2: generate candidate pairs
        paired_sources = set()
        paired_dests = set()
        candidates: List[PairCandidate] = []

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

        # Phase 3: emit skip records for ALL unpaired valuable channels.
        # Design rule 6: every eligible channel must be explained.
        at_capacity = len(candidates) >= self.max_pairs

        for ch in over_local:
            if ch.channel_id not in paired_sources:
                if not over_remote:
                    reason, detail = "no_partner", "no over-remote channels available"
                elif at_capacity:
                    reason, detail = "max_pairs_reached", f"limit={self.max_pairs}"
                else:
                    reason, detail = "outcompeted", "lower-scoring pairs selected"
                skipped.append(SkipRecord(
                    channel_id=ch.channel_id,
                    reason=reason,
                    value_class=ch.value_class,
                    remaining_budget_sats=ch.remaining_budget_sats,
                    detail=detail,
                ))

        for ch in over_remote:
            if ch.channel_id not in paired_dests:
                if not over_local:
                    reason, detail = "no_partner", "no over-local channels available"
                elif at_capacity:
                    reason, detail = "max_pairs_reached", f"limit={self.max_pairs}"
                else:
                    reason, detail = "outcompeted", "lower-scoring pairs selected"
                skipped.append(SkipRecord(
                    channel_id=ch.channel_id,
                    reason=reason,
                    value_class=ch.value_class,
                    remaining_budget_sats=ch.remaining_budget_sats,
                    detail=detail,
                ))

        return PlanResult(selected=candidates, skipped=skipped)

    def _generate_pairs(
        self,
        over_local: List[ChannelState],
        over_remote: List[ChannelState],
    ) -> List[PairCandidate]:
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

                # Score: value * imbalance + Phase 2.3 source-side terms
                src_value = _VALUE_SCORES.get(src.value_class, 0)
                dest_value = _VALUE_SCORES.get(dest.value_class, 0)
                value_score = max(src_value, dest_value)

                src_imbalance = max(0.0, src.local_ratio - self.target_band_high)
                dest_imbalance = max(0.0, self.target_band_low - dest.local_ratio)
                imbalance_score = (src_imbalance + dest_imbalance) / 2.0

                # Phase 2.3: explicit source-side preference. The drain term
                # nudges the planner toward stagnant overfull channels, and
                # the cheap-return term prefers low-fee inbound paths so the
                # circular route is cheaper to settle. Both terms are small
                # additive nudges -- Phase 4 replaces this with the additive
                # role-aware utility model.
                drain_bonus = float(src.source_drain_score or 0.0) * 0.10
                inbound_fee_ppm = max(0, int(src.actual_inbound_fee_ppm or 0))
                cheap_return_bonus = max(0.0, (5_000 - min(5_000, inbound_fee_ppm)) / 50_000.0)

                score = (
                    value_score * imbalance_score
                    + drain_bonus
                    + cheap_return_bonus
                )

                pairs.append(PairCandidate(
                    source_channel_id=src.channel_id,
                    dest_channel_id=dest.channel_id,
                    source_peer_id=src.peer_id,
                    dest_peer_id=dest.peer_id,
                    amount_sats=amount,
                    pair_budget_sats=pair_budget,
                    source_capacity_sats=src.capacity_sats,
                    dest_capacity_sats=dest.capacity_sats,
                    source_value_class=src.value_class,
                    dest_value_class=dest.value_class,
                    score=score,
                    source_local_ratio=src.local_ratio,
                    dest_local_ratio=dest.local_ratio,
                    score_decomposition=_bootstrap_score_decomposition(
                        value_score=value_score,
                        imbalance_score=imbalance_score,
                        pair_score=score,
                        amount_sats=amount,
                        pair_budget_sats=pair_budget,
                        source_local_ratio=src.local_ratio,
                        dest_local_ratio=dest.local_ratio,
                    ),
                ))

        return pairs
