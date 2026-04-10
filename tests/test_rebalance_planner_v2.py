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
    def test_skips_non_valuable_channels(self):
        planner = RebalancePlanner()
        ch = _ch(channel_id="neutral", local_ratio=0.90,
                 value_class="neutral", is_valuable=False)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        assert any(s.reason == "not_valuable" for s in result.skipped)

    def test_skips_inside_band_channels(self):
        planner = RebalancePlanner()
        ch = _ch(channel_id="balanced", local_ratio=0.50)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        assert any(s.reason == "inside_band" for s in result.skipped)

    def test_skips_cooldown_channels(self):
        planner = RebalancePlanner()
        ch = _ch(channel_id="cooling", local_ratio=0.90, cooldown_active=True)
        snap = _snap(ch)

        result = planner.plan(snap)

        assert len(result.selected) == 0
        assert any(s.reason == "cooldown" for s in result.skipped)

    def test_skips_no_budget_channels(self):
        planner = RebalancePlanner()
        ch = _ch(channel_id="broke", local_ratio=0.90, remaining_budget_sats=0)
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

    def test_score_is_value_times_imbalance(self):
        planner = RebalancePlanner(target_band_low=0.35, target_band_high=0.65)
        src = _ch(channel_id="src", peer_id="02" + "aa" * 32,
                  local_ratio=0.85, value_class="profitable")  # value=2, imbalance=0.20
        dest = _ch(channel_id="dest", peer_id="02" + "bb" * 32,
                   local_ratio=0.10, value_class="active")  # value=1, imbalance=0.25
        snap = _snap(src, dest)

        result = planner.plan(snap)

        pair = result.selected[0]
        # max(2, 1) = 2, avg(0.20, 0.25) = 0.225, score = 0.45
        assert abs(pair.score - 0.45) < 0.001
