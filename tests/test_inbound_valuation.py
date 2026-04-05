"""Tests for inbound source valuation fixes."""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import ChannelRevenue


class TestChannelRevenueMsat:
    """ChannelRevenue stores msat internally, exposes sats via properties."""

    def test_fees_earned_msat_stored_directly(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5500,
            volume_routed_msat=1_000_000_000,
            forward_count=10,
        )
        assert rev.fees_earned_msat == 5500

    def test_fees_earned_sats_truncates_correctly(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5500,
            volume_routed_msat=1_000_000_000,
            forward_count=10,
        )
        assert rev.fees_earned_sats == 5

    def test_sub_sat_fee_rounds_up_to_1(self):
        """A channel earning 50 msat should show 1 sat, not 0."""
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=50,
            volume_routed_msat=500_000,
            forward_count=1,
        )
        assert rev.fees_earned_sats == 1

    def test_zero_fee_stays_zero(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
        )
        assert rev.fees_earned_sats == 0

    def test_sourced_fee_sub_sat_rounds_up(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_fee_contribution_msat=200,
            sourced_forward_count=3,
        )
        assert rev.sourced_fee_contribution_sats == 1

    def test_volume_sats_truncates_no_ceiling(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=500,
            forward_count=0,
        )
        assert rev.volume_routed_sats == 0


class TestTotalContributionMax:
    """total_contribution uses max(earned, sourced) for valuation."""

    def test_exit_dominant_channel(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=10_000,
            volume_routed_msat=5_000_000,
            forward_count=50,
            sourced_fee_contribution_msat=2_000,
            sourced_forward_count=5,
        )
        assert rev.total_contribution_msat == 10_000
        assert rev.total_contribution_sats == 10

    def test_inbound_dominant_channel(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=1_000,
            volume_routed_msat=500_000,
            forward_count=2,
            sourced_fee_contribution_msat=8_000,
            sourced_forward_count=40,
        )
        assert rev.total_contribution_msat == 8_000
        assert rev.total_contribution_sats == 8

    def test_pure_inbound_gateway(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_fee_contribution_msat=120_000,
            sourced_forward_count=63,
        )
        assert rev.total_contribution_msat == 120_000
        assert rev.total_contribution_sats == 120

    def test_sub_sat_sourced_fee_valued(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=0,
            volume_routed_msat=0,
            forward_count=0,
            sourced_fee_contribution_msat=500,
            sourced_forward_count=5,
        )
        assert rev.total_contribution_msat == 500
        assert rev.total_contribution_sats == 1

    def test_total_forward_count(self):
        rev = ChannelRevenue(
            channel_id="100x1x0",
            fees_earned_msat=5_000,
            volume_routed_msat=1_000_000,
            forward_count=10,
            sourced_forward_count=50,
        )
        assert rev.total_forward_count == 60
