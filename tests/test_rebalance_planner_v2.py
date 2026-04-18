"""Tests for the v2 rebalance planner."""

from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_state_v2 import ChannelState, StateSnapshot


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

    def test_uses_max_budget_from_either_channel(self):
        planner = RebalancePlanner()
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.90, remaining_budget_sats=100)
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, remaining_budget_sats=500)
        snap = _snap(src, dest)

        result = planner.plan(snap)

        assert result.selected[0].pair_budget_sats == 500


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

    def test_cheaper_return_source_wins_when_other_terms_equal(self):
        """Phase 2.3: when two sources have the same value class and the same
        local ratio, the one offering a cheaper inbound return path should win
        because it lowers expected circular route cost."""
        planner = RebalancePlanner()
        # Order with expensive first to prove insertion order doesn't decide.
        expensive = _ch(channel_id="expensive_src", peer_id="02" + "bb" * 32,
                        local_ratio=0.90, value_class="active",
                        actual_inbound_fee_ppm=5_000)
        cheap = _ch(channel_id="cheap_src", peer_id="02" + "aa" * 32,
                    local_ratio=0.90, value_class="active",
                    actual_inbound_fee_ppm=10)
        dest = _ch(channel_id="dest", peer_id="02" + "cc" * 32, local_ratio=0.10)
        snap = _snap(expensive, cheap, dest)

        result = planner.plan(snap)

        assert len(result.selected) == 1
        assert result.selected[0].source_channel_id == "cheap_src"

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

    def test_destination_drives_value_term_not_source(self):
        """Phase 4.1: under the additive model the destination's value class
        drives the value term -- a hive source paired with an active dest
        should NOT outrank an active source paired with a hive dest."""
        planner = RebalancePlanner()
        # Pair A: hive source -> active dest. Source value should not beat dest.
        hive_src = _ch(channel_id="hive_src", peer_id="02" + "aa" * 32,
                       local_ratio=0.85, value_class="hive",
                       actual_inbound_fee_ppm=100)
        active_dest = _ch(channel_id="active_dest", peer_id="02" + "bb" * 32,
                          local_ratio=0.15, value_class="active")
        # Pair B: active source -> hive dest.
        active_src = _ch(channel_id="active_src", peer_id="02" + "cc" * 32,
                         local_ratio=0.85, value_class="active",
                         actual_inbound_fee_ppm=100)
        hive_dest = _ch(channel_id="hive_dest", peer_id="02" + "dd" * 32,
                        local_ratio=0.15, value_class="hive")
        snap = _snap(hive_src, active_dest, active_src, hive_dest)

        result = planner.plan(snap)

        # Highest-scoring pair should have the hive *destination*.
        sorted_pairs = sorted(result.selected, key=lambda p: p.score, reverse=True)
        assert sorted_pairs[0].dest_channel_id == "hive_dest"

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
