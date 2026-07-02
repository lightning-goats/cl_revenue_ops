"""Rebalance planner v2 — unified pair-based planning.

No separate hive push / hive equalization / capex paths. One planner
that pairs over-local channels with over-remote channels, scores them
by value and imbalance, and emits explicit skip reasons for everything
it doesn't select.
"""

from __future__ import annotations

from typing import List

from .rebalance_state_v2 import ChannelState, StateSnapshot
from .rebalance_types_v2 import (
    DrainDemand,
    DrainDemandEntry,
    PairCandidate,
    PlanResult,
    SkipRecord,
)


# Value class scores — hive channels win when routes are cheap, not via fake bonuses
_VALUE_SCORES = {
    "hive": 3,
    "profitable": 2,
    "active": 1,
    "funded": 1,  # capex budget approved — bootstrap inventory
    "neutral": 0,
}


def _sats_from_ratio_delta(ratio_delta: float, capacity_sats: int) -> int:
    return max(0, int(round(float(ratio_delta) * max(0, int(capacity_sats)))))


def _bootstrap_score_decomposition(
    *,
    value_score: int,
    imbalance_score: float,
    pair_score: float,
    amount_sats: int,
    pair_budget_sats: int,
    source_local_ratio: float,
    dest_local_ratio: float,
    dest_urgency_term: float = 0.0,
    source_drain_term: float = 0.0,
    dest_value_term: float = 0.0,
    cheap_return_term: float = 0.0,
    pre_hint_pair_score: float = 0.0,
    hive_source_rebalance_bias: float = 1.0,
    hive_dest_rebalance_bias: float = 1.0,
    hive_hint_score_multiplier: float = 1.0,
) -> dict:
    """Return the initial explicit score breakdown for a planned pair.

    Phase 4.1 exposes the additive role-aware terms (dest_urgency,
    source_drain, dest_value, cheap_return) alongside the legacy
    value_score/imbalance_score diagnostics. The engine layers route
    success and fee on top via _build_score_decomposition.
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
            "dest_urgency_term": round(float(dest_urgency_term), 6),
            "source_drain_term": round(float(source_drain_term), 6),
            "dest_value_term": round(float(dest_value_term), 6),
            "cheap_return_term": round(float(cheap_return_term), 6),
            "pre_hint_pair_score": round(float(pre_hint_pair_score), 6),
            "hive_source_rebalance_bias": round(float(hive_source_rebalance_bias), 6),
            "hive_dest_rebalance_bias": round(float(hive_dest_rebalance_bias), 6),
            "hive_hint_score_multiplier": round(float(hive_hint_score_multiplier), 6),
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
        pair_fee_cap_ppm: int = 0,
    ):
        self.target_band_low = target_band_low
        self.target_band_high = target_band_high
        self.max_chunk_sats = max_chunk_sats
        self.max_pairs = max_pairs
        # Iter1: rebalance fee budget per pair = max(dest_capex_budget,
        # ceil(amount * pair_fee_cap_ppm / 1_000_000)). Decouples per-rebalance
        # fee envelope from capex bootstrap (which is meant for channel
        # opens, not routing fees). 0 disables and keeps legacy behavior.
        self.pair_fee_cap_ppm = max(0, int(pair_fee_cap_ppm))

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

        # Phase 4: publish the residual the circular path cannot place.
        # Boltz structural loop-outs may ONLY act on this residual — the
        # rebalancer keeps first claim on internally placeable excess.
        demand_entries = [
            DrainDemandEntry(
                channel_id=ch.channel_id,
                peer_id=ch.peer_id,
                excess_sats=_sats_from_ratio_delta(
                    ch.local_ratio - self.target_band_high, ch.capacity_sats
                ),
                drain_score=float(ch.source_drain_score),
                value_class=ch.value_class,
            )
            for ch in over_local
            if ch.channel_id not in paired_sources
        ]
        demand_entries.sort(key=lambda e: e.drain_score, reverse=True)
        drain_demand = DrainDemand(
            entries=demand_entries,
            total_excess_sats=sum(e.excess_sats for e in demand_entries),
            over_local_count=len(over_local),
            paired_count=len(paired_sources),
        )

        return PlanResult(selected=candidates, skipped=skipped,
                          drain_demand=drain_demand)

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

                source_excess = _sats_from_ratio_delta(
                    src.local_ratio - self.target_band_high,
                    src.capacity_sats,
                )
                dest_need = _sats_from_ratio_delta(
                    self.target_band_low - dest.local_ratio,
                    dest.capacity_sats,
                )
                amount = min(
                    max(0, source_excess),
                    max(0, dest_need),
                    self.max_chunk_sats,
                )
                if amount <= 0:
                    continue

                # Phase 5.1+5.2: destination authorizes spend. The source's
                # remaining capex budget never enters pair_budget -- we are
                # not opening a channel to the source, just draining it.
                # Iter1: layer a fee-cap-on-amount on top of the capex
                # budget so a small bootstrap channel can still pay enough
                # routing fee to find a path.
                pair_budget = dest.remaining_budget_sats
                if self.pair_fee_cap_ppm > 0 and amount > 0:
                    fee_cap_from_amount = (
                        amount * self.pair_fee_cap_ppm + 999_999
                    ) // 1_000_000
                    pair_budget = max(pair_budget, fee_cap_from_amount)

                # Phase 4.1: explicit additive role-aware planner score.
                # Each term carries a clear meaning instead of a single
                # value*imbalance scalar.
                dest_value = _VALUE_SCORES.get(dest.value_class, 0)

                # Destination value drives the investment decision. Phase 5
                # makes this fully destination-led; Phase 4 already stops
                # mixing source and destination value classes.
                dest_value_term = float(dest_value) * 0.20

                # Refill urgency -- how badly the destination needs sats.
                dest_urgency_term = float(dest.dest_urgency or 0.0) * 0.30

                # Drain benefit -- how stagnant/overfull the source is.
                source_drain_term = float(src.source_drain_score or 0.0) * 0.20

                # Cheap-return bonus -- low inbound fee shortens the circular
                # route cost. Capped so it can't dominate the urgency term.
                inbound_fee_ppm = max(0, int(src.actual_inbound_fee_ppm or 0))
                cheap_return_term = max(
                    0.0, (5_000 - min(5_000, inbound_fee_ppm)) / 50_000.0
                )

                # Imbalance retained as derived diagnostic, not a scoring
                # input -- urgency + drain already encode it.
                src_imbalance = max(0.0, src.local_ratio - self.target_band_high)
                dest_imbalance = max(0.0, self.target_band_low - dest.local_ratio)
                imbalance_score = (src_imbalance + dest_imbalance) / 2.0

                score = (
                    dest_urgency_term
                    + source_drain_term
                    + dest_value_term
                    + cheap_return_term
                )
                pre_hint_score = score
                source_rebalance_bias = self._normalize_rebalance_bias(
                    getattr(src, "rebalance_bias", 1.0)
                )
                dest_rebalance_bias = self._normalize_rebalance_bias(
                    getattr(dest, "rebalance_bias", 1.0)
                )
                hint_multiplier = self._pair_hint_multiplier(
                    source_rebalance_bias,
                    dest_rebalance_bias,
                )
                score *= hint_multiplier

                # Backwards-compat: keep value_score reference for callers and
                # tests that still inspect it via the bootstrap decomposition.
                value_score = dest_value

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
                    source_budget_source=src.budget_source,
                    dest_budget_source=dest.budget_source,
                    source_out_fee_ppm=max(
                        0, int(getattr(src, "local_out_fee_ppm", 0) or 0)
                    ),
                    dest_out_fee_ppm=max(
                        0, int(getattr(dest, "local_out_fee_ppm", 0) or 0)
                    ),
                    source_historical_direct_fee_ppm=max(
                        0.0, float(getattr(src, "historical_direct_fee_ppm", 0.0) or 0.0)
                    ),
                    source_historical_sourced_fee_ppm=max(
                        0.0, float(getattr(src, "historical_sourced_fee_ppm", 0.0) or 0.0)
                    ),
                    dest_historical_direct_fee_ppm=max(
                        0.0, float(getattr(dest, "historical_direct_fee_ppm", 0.0) or 0.0)
                    ),
                    dest_historical_sourced_fee_ppm=max(
                        0.0, float(getattr(dest, "historical_sourced_fee_ppm", 0.0) or 0.0)
                    ),
                    dest_realized_utilization=float(
                        getattr(dest, "realized_utilization", 0.5) or 0.5
                    ),
                    source_realized_utilization=float(
                        getattr(src, "realized_utilization", 0.5) or 0.5
                    ),
                    dest_utilization_is_realized=bool(
                        getattr(dest, "utilization_is_realized", False)
                    ),
                    source_utilization_is_realized=bool(
                        getattr(src, "utilization_is_realized", False)
                    ),
                    hive_source_rebalance_bias=source_rebalance_bias,
                    hive_dest_rebalance_bias=dest_rebalance_bias,
                    hive_hint_score_multiplier=hint_multiplier,
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
                        dest_urgency_term=dest_urgency_term,
                        source_drain_term=source_drain_term,
                        dest_value_term=dest_value_term,
                        cheap_return_term=cheap_return_term,
                        pre_hint_pair_score=pre_hint_score,
                        hive_source_rebalance_bias=source_rebalance_bias,
                        hive_dest_rebalance_bias=dest_rebalance_bias,
                        hive_hint_score_multiplier=hint_multiplier,
                    ),
                ))

        return pairs

    @staticmethod
    def _normalize_rebalance_bias(value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 1.0
        return max(0.85, min(1.15, parsed))

    @classmethod
    def _pair_hint_multiplier(cls, source_bias: float, dest_bias: float) -> float:
        """Convert peer rebalance preferences into a pair-role score multiplier.

        HiveHintAdapter defines sink preference as >1.0 because the peer is a
        better refill destination, and source preference as <1.0 because the
        peer is a worse refill destination. A pair has both roles, so the
        source side is intentionally inverted: source-pref peers are better
        drain sources, while sink-pref peers are worse drain sources.
        """
        source_bias = cls._normalize_rebalance_bias(source_bias)
        dest_bias = cls._normalize_rebalance_bias(dest_bias)
        multiplier = 1.0 + (dest_bias - 1.0) - (source_bias - 1.0)
        return max(0.85, min(1.15, multiplier))
