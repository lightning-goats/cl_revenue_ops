"""Tests for the v2 rebalance planner."""

import pytest

from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_state_v2 import (
    ChannelState,
    StateSnapshot,
    compute_size_tiered_bands,
)


def _ch(
    channel_id="100x1x0",
    peer_id="02" + "aa" * 32,
    capacity_sats=1_000_000,
    local_ratio=0.50,
    actual_inbound_fee_ppm=200,
    value_class="active",
    is_valuable=True,
    remaining_budget_sats=500,
    cooldown_active=False,
    target_band_low=0.35,
    target_band_high=0.65,
):
    # Mirror the Phase 2 role-aware eligibility derived in build_state_snapshot
    # so planner unit tests describe the same semantics as the live snapshot.
    source_eligible = not cooldown_active
    source_reason = "" if source_eligible else "cooldown"
    if not is_valuable:
        dest_eligible, dest_reason = False, "not_valuable"
    elif cooldown_active:
        dest_eligible, dest_reason = False, "cooldown"
    elif remaining_budget_sats <= 0:
        dest_eligible, dest_reason = False, "no_budget"
    else:
        dest_eligible, dest_reason = True, ""
    # Mirror build_state_snapshot's drain/urgency math against default bands.
    drain_headroom = max(0.0, 1.0 - 0.65)
    drain_excess = max(0.0, local_ratio - 0.65)
    source_drain_score = (
        round(min(1.0, drain_excess / drain_headroom), 6) if drain_headroom > 0 else 0.0
    )
    urgency_band = 0.35
    dest_urgency = (
        round(min(1.0, max(0.0, urgency_band - local_ratio) / urgency_band), 6)
        if urgency_band > 0
        else 0.0
    )
    return ChannelState(
        channel_id=channel_id,
        peer_id=peer_id,
        capacity_sats=capacity_sats,
        local_ratio=local_ratio,
        actual_inbound_fee_ppm=actual_inbound_fee_ppm,
        value_class=value_class,
        is_valuable=is_valuable,
        remaining_budget_sats=remaining_budget_sats,
        cooldown_active=cooldown_active,
        source_eligible=source_eligible,
        dest_eligible=dest_eligible,
        source_reason=source_reason,
        dest_reason=dest_reason,
        dest_urgency=dest_urgency,
        source_drain_score=source_drain_score,
        target_band_low=target_band_low,
        target_band_high=target_band_high,
    )


def _snap(*channels):
    return StateSnapshot(
        channels=tuple(channels),
        total_capacity_sats=sum(c.capacity_sats for c in channels),
        total_remaining_budget_sats=sum(c.remaining_budget_sats for c in channels),
        valuable_channel_count=sum(1 for c in channels if c.is_valuable),
    )


class TestPairGeneration:
    def test_builds_pairs_between_over_local_and_over_remote(self):
        planner = RebalancePlanner()
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32, local_ratio=0.90)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32, local_ratio=0.10)
        snap = _snap(src, dest)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        assert result.selected[0].source_channel_id == "src"
        assert result.selected[0].dest_channel_id == "dest"

    def test_computes_correct_transfer_amount(self):
        planner = RebalancePlanner(target_band_low=0.35, target_band_high=0.65)
        # source excess: (0.90 - 0.65) * 1M = 250k
        # dest need: (0.35 - 0.10) * 500k = 125k
        # min(250k, 125k, 2M) = 125k
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  capacity_sats=1_000_000, local_ratio=0.90)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   capacity_sats=500_000, local_ratio=0.10)
        snap = _snap(src, dest)

        result = planner.plan(snap)

        # Float arithmetic may be off by 1 sat from int truncation
        assert abs(result.selected[0].amount_sats - 125_000) <= 1

    def test_pair_budget_uses_fee_cap_ppm_when_capex_too_low(self):
        """Iter1: capex bootstrap budget (e.g. 200 sats) is meant for channel
        opens, not per-rebalance fees. The planner should layer a fee cap
        derived from the rebalance amount on top, so a small capex channel
        can still pay enough route fee for a priced path."""
        planner = RebalancePlanner(pair_fee_cap_ppm=1000)  # 0.1% of amount
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.90, remaining_budget_sats=0)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, remaining_budget_sats=200)
        snap = _snap(src, dest)

        result = planner.plan(snap)

        # amount = min((0.90-0.65)*1M, (0.35-0.10)*1M, max_chunk) = 250k sats
        # fee_cap = ceil(250000 * 1000 / 1_000_000) = 250 sats
        # pair_budget should be max(capex=200, fee_cap=250) = 250
        pair = result.selected[0]
        assert pair.amount_sats == pytest.approx(250_000, abs=1)
        assert pair.pair_budget_sats == 250

    def test_pair_budget_keeps_capex_when_higher_than_fee_cap(self):
        """Iter1: when the channel's earned capex budget exceeds the
        fee-cap-on-amount, keep using the earned budget."""
        planner = RebalancePlanner(pair_fee_cap_ppm=1000)
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.90, remaining_budget_sats=0)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, remaining_budget_sats=5000)
        snap = _snap(src, dest)

        result = planner.plan(snap)

        # fee_cap = 250 sats, capex = 5000 -> use 5000
        assert result.selected[0].pair_budget_sats == 5000

    def test_pair_budget_zero_fee_cap_keeps_legacy_capex_only_behavior(self):
        """Iter1 backwards-compat: pair_fee_cap_ppm=0 disables the fee cap
        and reverts to the Phase 5 destination-led capex-only behavior."""
        planner = RebalancePlanner(pair_fee_cap_ppm=0)
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.90, remaining_budget_sats=0)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, remaining_budget_sats=200)
        snap = _snap(src, dest)

        result = planner.plan(snap)
        assert result.selected[0].pair_budget_sats == 200

    def test_source_safety_controls_survive_destination_led_budget(self):
        """Phase 5.3: even though source budget no longer gates pair spend,
        source-side safety controls (cooldown protection, opportunity-cost
        signals via the score) still apply to prevent re-draining a
        recovering source or burning a hot channel."""
        planner = RebalancePlanner()
        # Cooldown-protected over-local source -> still ineligible.
        protected = _ch(channel_id="protected", peer_id="02" + "aa" * 32,
                        local_ratio=0.92, value_class="active",
                        cooldown_active=True, remaining_budget_sats=10_000)
        # Funded depleted dest -> would otherwise be a happy refill target.
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, value_class="profitable",
                   remaining_budget_sats=2000)
        snap = _snap(protected, dest)

        result = planner.plan(snap)

        assert result.selected == []
        skipped = {s.channel_id: s.reason for s in result.skipped}
        # Source still cooldown-protected -- destination-led budget does
        # not bypass source-side safety.
        assert skipped.get("protected") == "cooldown"
        assert skipped.get("dest") == "no_partner"

    def test_destination_budget_authorizes_pair_spend(self):
        """Phase 5.1+5.2: the destination's remaining capex budget alone
        authorizes pair spend. Source budget does not enter pair_budget --
        we are not opening a channel to the source, we are draining it."""
        planner = RebalancePlanner()
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.90, remaining_budget_sats=5000)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, remaining_budget_sats=500)
        snap = _snap(src, dest)

        result = planner.plan(snap)

        # Destination budget caps spend even when source has more.
        assert result.selected[0].pair_budget_sats == 500

    def test_generate_pairs_threads_realized_utilization_without_transpose(self):
        """Upstream pattern #2 regression guard: _generate_pairs must map
        src.realized_utilization -> PairCandidate.source_realized_utilization
        and dest.realized_utilization -> PairCandidate.dest_realized_utilization
        (and the matching *_is_realized flags) WITHOUT swapping src/dest. Use
        distinct, asymmetric values on each side so a transpose bug (source
        and dest utilization silently swapped) would fail this assertion."""
        import dataclasses

        planner = RebalancePlanner()
        src = dataclasses.replace(
            _ch(channel_id="src", peer_id="02" + "aa" * 32, local_ratio=0.90),
            realized_utilization=0.7,
            utilization_is_realized=True,
        )
        dest = dataclasses.replace(
            _ch(channel_id="dest", peer_id="02" + "bb" * 32, local_ratio=0.10),
            realized_utilization=0.3,
            utilization_is_realized=False,
        )
        snap = _snap(src, dest)

        result = planner.plan(snap)

        pair = result.selected[0]
        assert pair.source_channel_id == "src"
        assert pair.dest_channel_id == "dest"
        assert pair.source_realized_utilization == pytest.approx(0.7)
        assert pair.source_utilization_is_realized is True
        assert pair.dest_realized_utilization == pytest.approx(0.3)
        assert pair.dest_utilization_is_realized is False

    def test_generate_pairs_threads_activity_sats_without_transpose(self):
        """Task 8 coverage gap: _generate_pairs must map
        src.activity_out_sats -> PairCandidate.source_activity_out_sats and
        dest.activity_in_sats -> PairCandidate.dest_activity_in_sats WITHOUT
        reading the wrong attribute (activity_out vs activity_in) or
        swapping src/dest. Use distinct, asymmetric values on both fields of
        both channels so either mistake would fail this assertion."""
        import dataclasses

        planner = RebalancePlanner()
        src = dataclasses.replace(
            _ch(channel_id="src", peer_id="02" + "aa" * 32, local_ratio=0.90),
            activity_out_sats=700_000,
            activity_in_sats=111,
        )
        dest = dataclasses.replace(
            _ch(channel_id="dest", peer_id="02" + "bb" * 32, local_ratio=0.10),
            activity_in_sats=900_000,
            activity_out_sats=222,
        )
        snap = _snap(src, dest)

        result = planner.plan(snap)

        pair = result.selected[0]
        assert pair.source_channel_id == "src"
        assert pair.dest_channel_id == "dest"
        assert pair.source_activity_out_sats == 700_000
        assert pair.dest_activity_in_sats == 900_000


class TestSkipReasons:
    def test_destination_skipped_when_not_valuable(self):
        """Phase 2: an over-remote neutral channel still cannot be a destination."""
        planner = RebalancePlanner()
        ch = _ch(channel_id="neutral_dest", local_ratio=0.10,
                 value_class="neutral", is_valuable=False)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        assert any(s.reason == "not_valuable" for s in result.skipped)

    def test_over_local_neutral_is_eligible_source(self):
        """Phase 2 unstick: an over-local neutral channel is a valid drain
        source. It does not get skipped as not_valuable -- it becomes a source
        that no_partner only when no eligible destination exists."""
        planner = RebalancePlanner()
        ch = _ch(channel_id="neutral_src", local_ratio=0.90,
                 value_class="neutral", is_valuable=False)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        skipped_for_src = [s for s in result.skipped if s.channel_id == "neutral_src"]
        assert len(skipped_for_src) == 1
        assert skipped_for_src[0].reason == "no_partner"

    def test_skips_inside_band_channels(self):
        planner = RebalancePlanner()
        ch = _ch(channel_id="balanced", local_ratio=0.50)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        assert any(s.reason == "inside_band" for s in result.skipped)

    def test_over_local_in_cooldown_skipped_as_protected_source(self):
        planner = RebalancePlanner()
        ch = _ch(channel_id="cooling", local_ratio=0.90, cooldown_active=True)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        assert any(s.reason == "cooldown" for s in result.skipped)

    def test_over_local_no_budget_is_eligible_source(self):
        """Phase 2: sources do not consume capex budget. An over-local channel
        with zero budget can still drain into a funded depleted destination."""
        planner = RebalancePlanner()
        ch = _ch(channel_id="broke_src", local_ratio=0.90, remaining_budget_sats=0)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        skipped_for_src = [s for s in result.skipped if s.channel_id == "broke_src"]
        assert skipped_for_src[0].reason == "no_partner"

    def test_destination_skipped_when_no_budget(self):
        planner = RebalancePlanner()
        ch = _ch(channel_id="broke_dest", local_ratio=0.10, remaining_budget_sats=0)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        assert any(s.reason == "no_budget" for s in result.skipped)

    def test_emits_no_partner_skip_when_no_opposite_side(self):
        planner = RebalancePlanner()
        # Only over-local channels, no over-remote
        src1 = _ch(channel_id="src1", peer_id="02" + "aa" * 32, local_ratio=0.90)
        src2 = _ch(channel_id="src2", peer_id="02" + "bb" * 32, local_ratio=0.85)
        snap = _snap(src1, src2)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        no_partner = [s for s in result.skipped if s.reason == "no_partner"]
        assert len(no_partner) == 2


    def test_emits_outcompeted_skip_for_losing_sources(self):
        planner = RebalancePlanner()
        # Two sources, one dest — loser gets outcompeted skip
        winner = _ch(channel_id="winner", peer_id="02" + "aa" * 32,
                     local_ratio=0.90, value_class="hive")
        loser = _ch(channel_id="loser", peer_id="02" + "bb" * 32,
                    local_ratio=0.80, value_class="active")
        dest = _ch(channel_id="dest", peer_id="02" + "cc" * 32, local_ratio=0.10)
        snap = _snap(winner, loser, dest)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        outcompeted = [s for s in result.skipped if s.reason == "outcompeted"]
        assert len(outcompeted) == 1
        assert outcompeted[0].channel_id == "loser"

    def test_emits_max_pairs_reached_skip(self):
        planner = RebalancePlanner(max_pairs=1)
        src1 = _ch(channel_id="src1", peer_id="02" + "aa" * 32, local_ratio=0.90)
        src2 = _ch(channel_id="src2", peer_id="02" + "bb" * 32, local_ratio=0.85)
        dest1 = _ch(channel_id="dest1", peer_id="02" + "cc" * 32, local_ratio=0.10)
        dest2 = _ch(channel_id="dest2", peer_id="02" + "dd" * 32, local_ratio=0.15)
        snap = _snap(src1, src2, dest1, dest2)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        max_reached = [s for s in result.skipped if s.reason == "max_pairs_reached"]
        assert len(max_reached) >= 1


class TestScoring:
    def test_scores_hive_channels_higher(self):
        planner = RebalancePlanner()
        # Two sources competing for one dest
        hive_src = _ch(channel_id="hive_src", peer_id="02" + "aa" * 32,
                       local_ratio=0.80, value_class="hive")
        active_src = _ch(channel_id="active_src", peer_id="02" + "bb" * 32,
                         local_ratio=0.80, value_class="active")
        dest = _ch(channel_id="dest", peer_id="02" + "cc" * 32, local_ratio=0.10)
        snap = _snap(hive_src, active_src, dest)

        result = planner.plan(snap)

        # Only one pair should be selected (dest can't be paired twice)
        assert len(result.selected) == 1
        # Hive source should win due to higher value score
        assert result.selected[0].source_channel_id == "hive_src"

    def test_score_is_strictly_positive_for_valid_pair(self):
        """Phase 4: the additive role-aware score is always positive for a
        well-formed candidate, but its magnitude depends on urgency/drain
        rather than the legacy value*imbalance product."""
        planner = RebalancePlanner(target_band_low=0.35, target_band_high=0.65)
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.85, value_class="profitable")
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, value_class="active")
        snap = _snap(src, dest)

        result = planner.plan(snap)

        pair = result.selected[0]
        assert pair.score > 0.0

    def test_cheaper_return_dest_wins_when_other_terms_equal(self):
        """Audit wave2 FIX 5(a): the return leg the circular route actually
        traverses is the DEST channel's inbound edge (dest_peer -> us), so
        when two destinations have the same value class and the same local
        ratio, the one with the cheaper inbound return path should win —
        it lowers expected circular route cost. (The source peer's inbound
        edge is never on the route and must not influence the score.)"""
        planner = RebalancePlanner()
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.90, value_class="active",
                  actual_inbound_fee_ppm=5_000)
        # Order with expensive first to prove insertion order doesn't decide.
        expensive = _ch(channel_id="expensive_dest", peer_id="02" + "bb" * 32,
                        local_ratio=0.10, value_class="active",
                        actual_inbound_fee_ppm=5_000)
        cheap = _ch(channel_id="cheap_dest", peer_id="02" + "cc" * 32,
                    local_ratio=0.10, value_class="active",
                    actual_inbound_fee_ppm=10)
        snap = _snap(src, expensive, cheap)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        assert result.selected[0].dest_channel_id == "cheap_dest"

    def test_polar_s2_shape_forms_a_pair(self):
        """Phase 2.4 regression: S2 had a depleted profitable destination plus
        two extreme over-local neutral channels. Phase 2 must let one of those
        neutrals serve as the drain source so a pair forms."""
        planner = RebalancePlanner()
        depleted = _ch(channel_id="159x1x0", peer_id="02" + "1" * 64,
                       local_ratio=0.10, value_class="profitable",
                       remaining_budget_sats=1000)
        neutral_a = _ch(channel_id="243x1x0", peer_id="02" + "2" * 64,
                        local_ratio=0.95, value_class="neutral",
                        is_valuable=False, remaining_budget_sats=0)
        neutral_b = _ch(channel_id="255x1x0", peer_id="02" + "3" * 64,
                        local_ratio=0.95, value_class="neutral",
                        is_valuable=False, remaining_budget_sats=0)
        snap = _snap(depleted, neutral_a, neutral_b)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        pair = result.selected[0]
        assert pair.source_channel_id in {"243x1x0", "255x1x0"}
        assert pair.dest_channel_id == "159x1x0"

    def test_polar_s9_shape_recognizes_neutral_sources(self):
        """Phase 2.4 regression: S9 had two 100%-local neutral channels and a
        cooldown-blocked depleted destination. Sources must be eligible (no
        source_rejected_neutral); the cooldown blocker stays for Phase 3."""
        planner = RebalancePlanner()
        cooldown_dest = _ch(channel_id="123x1x0", peer_id="02" + "9" * 64,
                            local_ratio=0.066, value_class="profitable",
                            remaining_budget_sats=1000, cooldown_active=True)
        neutral_a = _ch(channel_id="200x2x0", peer_id="02" + "8" * 64,
                        local_ratio=1.0, value_class="neutral",
                        is_valuable=False, remaining_budget_sats=0)
        neutral_b = _ch(channel_id="201x2x0", peer_id="02" + "7" * 64,
                        local_ratio=1.0, value_class="neutral",
                        is_valuable=False, remaining_budget_sats=0)
        snap = _snap(cooldown_dest, neutral_a, neutral_b)

        result = planner.plan(snap)

        assert len(result.selected) == 0  # cooldown-blocked destination
        skipped = {s.channel_id: s.reason for s in result.skipped}
        # Neutral sources are eligible -- they hit no_partner because the
        # only depleted destination is in cooldown.
        assert skipped.get("200x2x0") == "no_partner"
        assert skipped.get("201x2x0") == "no_partner"
        assert skipped.get("123x1x0") == "cooldown"

    def test_additive_score_decomposition_exposes_role_terms(self):
        """Phase 4.1: planner pairs carry an explicit additive decomposition
        with explicit destination urgency, source drain, dest value, and
        cheap-return terms. The summed terms equal the pair score."""
        planner = RebalancePlanner()
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.95, value_class="active",
                  actual_inbound_fee_ppm=200)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.05, value_class="profitable")
        snap = _snap(src, dest)

        result = planner.plan(snap)
        pair = result.selected[0]
        decomp = pair.score_decomposition

        # The additive role-aware terms must be exposed by name.
        for term in (
            "dest_urgency_term",
            "source_drain_term",
            "dest_value_term",
            "cheap_return_term",
        ):
            assert term in decomp["inputs"], f"missing term {term}"
            assert decomp["inputs"][term] >= 0.0

        # Score equals the sum of the additive terms (within float tolerance).
        expected = (
            decomp["inputs"]["dest_urgency_term"]
            + decomp["inputs"]["source_drain_term"]
            + decomp["inputs"]["dest_value_term"]
            + decomp["inputs"]["cheap_return_term"]
        )
        assert abs(pair.score - expected) < 1e-6

    def test_more_drained_source_wins_when_value_and_return_tied(self):
        """Phase 2.3: explicit drain preference -- a more over-local source is
        preferred even at the same value class and same return cost."""
        planner = RebalancePlanner()
        very_full = _ch(channel_id="very_full", peer_id="02" + "aa" * 32,
                        local_ratio=0.95, value_class="active",
                        actual_inbound_fee_ppm=100)
        slightly = _ch(channel_id="slightly_full", peer_id="02" + "bb" * 32,
                       local_ratio=0.70, value_class="active",
                       actual_inbound_fee_ppm=100)
        dest = _ch(channel_id="dest", peer_id="02" + "cc" * 32, local_ratio=0.10)
        snap = _snap(very_full, slightly, dest)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        assert result.selected[0].source_channel_id == "very_full"


def test_small_channel_keeps_flat_band():
    caps = {"s1": 1_000_000, "s2": 1_000_000, "s3": 1_000_000, "big": 20_000_000}
    bands = compute_size_tiered_bands(caps, percentile=0.5, small_half_width=0.15, flat_low=0.35, flat_high=0.65)
    assert bands["s1"] == (0.35, 0.65)


def test_large_channel_gets_wider_band():
    caps = {"s1": 1_000_000, "s2": 1_000_000, "s3": 1_000_000, "big": 20_000_000}
    bands = compute_size_tiered_bands(caps, percentile=0.5, small_half_width=0.15, flat_low=0.35, flat_high=0.65)
    low, high = bands["big"]
    assert low < 0.35 and high > 0.65
    assert 0.0 < low < high < 1.0
    # Symmetric widening: the distance below 0.5 must equal the distance
    # above 0.5 (big channels tolerate more imbalance in EITHER direction,
    # they are not skewed to tolerate more remote-heaviness than local).
    assert abs((high - 0.5) - (0.5 - low)) < 1e-9


def test_empty_capacities_returns_empty():
    assert compute_size_tiered_bands({}, percentile=0.5, small_half_width=0.15, flat_low=0.35, flat_high=0.65) == {}


class TestSizeTieredPlannerClassification:
    """Task 7: the planner must classify each channel against ITS OWN
    per-channel target_band_low/target_band_high, not the planner's flat
    self.target_band_low/high scalars."""

    def test_large_channel_classified_by_its_own_wider_band_not_flat(self):
        """A large channel sitting at local_ratio=0.70 would be classified
        over_local under the flat 0.65 band, but its own asymmetric
        per-channel band (e.g. high=0.80) says it's still inside band --
        the planner must honor the per-channel band."""
        planner = RebalancePlanner(target_band_low=0.35, target_band_high=0.65)

        # Big channel: local_ratio 0.70 is > flat 0.65 (would be over_local
        # under the OLD flat-band logic) but its own per-channel band
        # (0.20, 0.80) says it is still comfortably inside band.
        big = _ch(
            channel_id="big",
            peer_id="02" + "aa" * 32,
            capacity_sats=20_000_000,
            local_ratio=0.70,
            value_class="active",
            target_band_low=0.20,
            target_band_high=0.80,
        )
        # Small channel at the SAME 0.70 ratio, flat band -- over_local.
        small = _ch(
            channel_id="small",
            peer_id="02" + "bb" * 32,
            capacity_sats=1_000_000,
            local_ratio=0.70,
            value_class="active",
            target_band_low=0.35,
            target_band_high=0.65,
        )
        dest = _ch(
            channel_id="dest",
            peer_id="02" + "cc" * 32,
            local_ratio=0.10,
            target_band_low=0.35,
            target_band_high=0.65,
        )
        snap = _snap(big, small, dest)

        result = planner.plan(snap)

        # Only the small channel should be selected as an over_local source;
        # the big channel is inside its own (wider) band and should be
        # skipped as "inside_band", not paired as a drain source.
        assert len(result.selected) == 1
        assert result.selected[0].source_channel_id == "small"
        skip_reasons = {s.channel_id: s.reason for s in result.skipped}
        assert skip_reasons.get("big") == "inside_band"

    def test_toggle_off_matches_flat_band_behavior(self):
        """When rebalance_size_tiered_targets is False, the engine builds no
        target_bands map, so build_state_snapshot defaults every channel's
        per-channel band to the flat cfg/default band. Classification with
        all channels carrying (0.35, 0.65) must be identical to the
        pre-feature flat-band behavior."""
        planner = RebalancePlanner(target_band_low=0.35, target_band_high=0.65)

        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.90, target_band_low=0.35, target_band_high=0.65)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, target_band_low=0.35, target_band_high=0.65)
        snap = _snap(src, dest)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        assert result.selected[0].source_channel_id == "src"
        assert result.selected[0].dest_channel_id == "dest"


class TestSizeTieredPlannerSizingAndScoring:
    """FIX 2(a): sizing (source_excess/dest_need) and diagnostics must use
    each channel's OWN per-channel band, not the planner's flat
    self.target_band_low/high scalars -- coherent with classification."""


    def test_pair_amount_uses_source_own_high_and_dest_own_low(self):
        """A big source/dest pair whose per-channel bands are wider than the
        flat planner scalars must have their sizing capped by the OWN
        bands, not the flat 0.35/0.65 -- giving a smaller amount than the
        pre-fix flat-band math would."""
        planner = RebalancePlanner(
            target_band_low=0.35, target_band_high=0.65, max_chunk_sats=10_000_000,
        )

        src = _ch(
            channel_id="big_src",
            peer_id="02" + "aa" * 32,
            capacity_sats=20_000_000,
            local_ratio=0.85,
            value_class="active",
            target_band_low=0.20,
            target_band_high=0.80,
        )
        dest = _ch(
            channel_id="big_dest",
            peer_id="02" + "bb" * 32,
            capacity_sats=20_000_000,
            local_ratio=0.05,
            value_class="active",
            target_band_low=0.35,
            target_band_high=0.65,
        )
        snap = _snap(src, dest)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        pair = result.selected[0]
        assert pair.source_channel_id == "big_src"
        assert pair.dest_channel_id == "big_dest"
        # source_excess (own high 0.80): (0.85-0.80)*20_000_000 = 1_000_000.
        # dest_need (own low 0.35, unchanged): (0.35-0.05)*20_000_000 = 6_000_000.
        # amount = min(source_excess, dest_need, max_chunk) = 1_000_000.
        # The pre-fix flat-band bug would have used source high=0.65,
        # giving source_excess=4_000_000 and thus amount=4_000_000.
        assert pair.amount_sats == 1_000_000

        # Diagnostics (score_decomposition inputs) must also reflect the
        # source's own band, not the flat one.
        imbalance_score = pair.score_decomposition["inputs"]["imbalance_score"]
        # src_imbalance = 0.85-0.80=0.05; dest_imbalance = 0.35-0.05=0.30.
        assert imbalance_score == pytest.approx((0.05 + 0.30) / 2.0)


def test_all_three_features_thread_onto_one_big_channel_pair():
    """Composition guard (final review): the branch ships live default-ON
    and every existing threading test above proves #2 (realized
    utilization) or #1 (activity sats) thread onto a PairCandidate without
    transposition IN ISOLATION, and TestSizeTieredPlannerSizingAndScoring
    proves #3 (size-tiered per-channel band) drives sizing IN ISOLATION.
    Nothing proves a single big channel can carry all three at once and
    have every field land correctly on the SAME resulting PairCandidate --
    e.g. that adding realized-utilization/activity fields to a ChannelState
    doesn't disturb the per-channel-band sizing math, or vice versa.

    Reuses the exact big_src/big_dest wide-band scenario from
    test_pair_amount_uses_source_own_high_and_dest_own_low (already proven
    to size to 1_000_000 via the channels' OWN (0.20, 0.80)/(0.35, 0.65)
    bands, not the planner's flat scalars) and additionally decorates both
    channels with realized utilization and live-activity facts, asserting
    ALL THREE thread onto the one resulting pair with nothing swapped or
    dropped."""
    import dataclasses

    planner = RebalancePlanner(
        target_band_low=0.35, target_band_high=0.65, max_chunk_sats=10_000_000,
    )

    src = dataclasses.replace(
        _ch(
            channel_id="big_src",
            peer_id="02" + "aa" * 32,
            capacity_sats=20_000_000,
            local_ratio=0.85,
            value_class="active",
            target_band_low=0.20,
            target_band_high=0.80,
        ),
        realized_utilization=0.8,
        utilization_is_realized=True,
        activity_out_sats=150_000,
        activity_in_sats=10,
    )
    dest = dataclasses.replace(
        _ch(
            channel_id="big_dest",
            peer_id="02" + "bb" * 32,
            capacity_sats=20_000_000,
            local_ratio=0.05,
            value_class="active",
            target_band_low=0.35,
            target_band_high=0.65,
        ),
        realized_utilization=0.3,
        utilization_is_realized=False,
        activity_in_sats=200_000,
        activity_out_sats=20,
    )
    snap = _snap(src, dest)

    result = planner.plan(snap)

    assert len(result.selected) == 1
    pair = result.selected[0]
    assert pair.source_channel_id == "big_src"
    assert pair.dest_channel_id == "big_dest"

    # --- #3 (size-tiered band): sizing still uses the channels' OWN wide
    # bands, not the planner's flat 0.35/0.65 -- identical to the
    # single-feature test this scenario is copied from. ---
    # source_excess (own high 0.80): (0.85-0.80)*20_000_000 = 1_000_000.
    # dest_need (own low 0.35, unchanged): (0.35-0.05)*20_000_000 = 6_000_000.
    # amount = min(source_excess, dest_need, max_chunk) = 1_000_000.
    assert pair.amount_sats == 1_000_000

    # --- #2 (realized utilization): threaded without transpose. ---
    assert pair.source_realized_utilization == pytest.approx(0.8)
    assert pair.source_utilization_is_realized is True
    assert pair.dest_realized_utilization == pytest.approx(0.3)
    assert pair.dest_utilization_is_realized is False

    # --- #1 (activity): threaded without transpose. ---
    assert pair.source_activity_out_sats == 150_000
    assert pair.dest_activity_in_sats == 200_000



class TestGeneratedPlannerUniverse:
    def _four_pair_snapshot(self):
        """Two sources x two destinations keeps score ties in input order."""
        return _snap(
            _ch(channel_id="src-1", peer_id="02" + "a1" * 32, local_ratio=0.90),
            _ch(channel_id="src-2", peer_id="02" + "a2" * 32, local_ratio=0.90),
            _ch(channel_id="dest-1", peer_id="02" + "b1" * 32, local_ratio=0.10),
            _ch(channel_id="dest-2", peer_id="02" + "b2" * 32, local_ratio=0.10),
        )

    def test_retains_complete_ranked_universe_without_changing_greedy_selection(self):
        planner = RebalancePlanner(max_pairs=2)

        result = planner.plan(self._four_pair_snapshot())

        assert [pair.cheap_rank for pair in result.generated] == [1, 2, 3, 4]
        assert len(result.generated) == 4
        assert [
            (pair.source_channel_id, pair.dest_channel_id)
            for pair in result.selected
        ] == [("src-1", "dest-1"), ("src-2", "dest-2")]
        assert all(pair.source_excess_sats > 0 for pair in result.generated)
        assert all(pair.dest_need_sats > 0 for pair in result.generated)
        assert all(pair.max_chunk_sats == planner.max_chunk_sats for pair in result.generated)
        assert {
            pair.planner_rejection_reason
            for pair in result.generated
            if not pair.planner_selected
        } <= {
            "source_already_paired",
            "dest_already_paired",
            "max_pairs_reached",
        }
        assert [
            (pair.source_channel_id, pair.dest_channel_id)
            for pair in result.generated
            if pair.planner_selected
        ] == [
            (pair.source_channel_id, pair.dest_channel_id)
            for pair in result.selected
        ]

    def test_same_captured_snapshot_order_replays_same_generated_ranks(self):
        planner = RebalancePlanner(max_pairs=2)
        snapshot = self._four_pair_snapshot()

        first = planner.plan(snapshot)
        second = planner.plan(snapshot)

        assert [
            (pair.source_channel_id, pair.dest_channel_id, pair.cheap_rank)
            for pair in first.generated
        ] == [
            (pair.source_channel_id, pair.dest_channel_id, pair.cheap_rank)
            for pair in second.generated
        ]

    def test_empty_snapshot_has_a_neutral_generated_universe(self):
        result = RebalancePlanner().plan(_snap())

        assert result.selected == []
        assert result.generated == []
