"""Tests for capital-efficiency analysis."""

import os
import sys
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capital_efficiency import CapitalEfficiencyAnalyzer
from modules.profitability_analyzer import ChannelRevenue


def _make_profitability(
    channel_id: str,
    peer_id: str,
    capacity_sats: int,
    fees_earned_sats: int,
    days_open: int,
):
    revenue = SimpleNamespace(
        fees_earned_sats=fees_earned_sats,
        fees_earned_msat=fees_earned_sats * 1000,
    )
    return SimpleNamespace(
        channel_id=channel_id,
        peer_id=peer_id,
        capacity_sats=capacity_sats,
        days_open=days_open,
        revenue=revenue,
    )


def _make_flow(channel_id: str, peer_id: str, capacity: int, forward_count: int, daily_volume: int):
    return SimpleNamespace(
        channel_id=channel_id,
        peer_id=peer_id,
        capacity=capacity,
        forward_count=forward_count,
        daily_volume=daily_volume,
    )


def _make_analyzer(profitability, flow, stages=None, hive_members=None, grace_days=14, flow_window_days=7):
    profitability_analyzer = MagicMock()
    profitability_analyzer.analyze_all_channels.return_value = profitability

    flow_analyzer = MagicMock()
    flow_analyzer.analyze_all_channels.return_value = flow
    flow_analyzer.config = SimpleNamespace(flow_window_days=flow_window_days)

    database = MagicMock()
    database.get_dead_capital_stages.return_value = stages or {}

    hive_hints = MagicMock()
    hive_hints.is_hive_member.side_effect = lambda peer_id: peer_id in (hive_members or set())

    config = SimpleNamespace(capex_grace_days=grace_days)
    return CapitalEfficiencyAnalyzer(
        profitability_analyzer=profitability_analyzer,
        flow_analyzer=flow_analyzer,
        database=database,
        hive_hints=hive_hints,
        config=config,
    )


class TestCapitalEfficiencyAnalyzer:
    """Capital-efficiency snapshots are computed from profitability and flow data."""

    def test_analyze_computes_rpsd_rank_and_fleet_summary(self):
        profitability = {
            "100x1x0": _make_profitability("100x1x0", "peer-a", 10_000_000, 500, 30),
            "100x2x0": _make_profitability("100x2x0", "peer-b", 500_000, 100, 30),
            "100x3x0": _make_profitability("100x3x0", "peer-c", 2_000_000, 0, 30),
        }
        flow = {
            "100x1x0": _make_flow("100x1x0", "peer-a", 10_000_000, 21, 500_000),
            "100x2x0": _make_flow("100x2x0", "peer-b", 500_000, 7, 100_000),
            "100x3x0": _make_flow("100x3x0", "peer-c", 2_000_000, 0, 0),
        }
        analyzer = _make_analyzer(
            profitability,
            flow,
            stages={"100x3x0": {"stage": "fee_reduction", "entered_at": 123}},
        )

        fleet = analyzer.analyze()

        assert fleet.median_rpsd == 50.0
        assert fleet.dead_capital_count == 1
        assert fleet.dead_capital_sats == 2_000_000
        assert fleet.total_deployed_sats == 12_500_000

        efficient = fleet.channel_efficiencies["100x2x0"]
        middle = fleet.channel_efficiencies["100x1x0"]
        dead = fleet.channel_efficiencies["100x3x0"]

        assert efficient.rpsd == 200.0
        assert efficient.efficiency_rank == 1.0
        assert efficient.forward_velocity == 1.0

        assert middle.rpsd == 50.0
        assert middle.efficiency_rank == 0.5
        assert middle.forward_velocity == 3.0

        assert dead.is_dead_capital is True
        assert dead.dead_capital_stage == "fee_reduction"
        assert dead.efficiency_rank == 0.0

    def test_dead_capital_excludes_young_channels_and_hive_members(self):
        profitability = {
            "young": _make_profitability("young", "peer-young", 1_000_000, 0, 7),
            "member": _make_profitability("member", "peer-member", 1_500_000, 0, 30),
        }
        flow = {
            "young": _make_flow("young", "peer-young", 1_000_000, 0, 0),
            "member": _make_flow("member", "peer-member", 1_500_000, 0, 0),
        }
        analyzer = _make_analyzer(
            profitability,
            flow,
            hive_members={"peer-member"},
        )

        fleet = analyzer.analyze()

        assert fleet.channel_efficiencies["young"].is_dead_capital is False
        assert fleet.channel_efficiencies["young"].dead_capital_stage == "none"
        assert fleet.channel_efficiencies["member"].is_dead_capital is False
        assert fleet.dead_capital_count == 0

    def test_missing_stage_defaults_to_none(self):
        profitability = {
            "100x1x0": _make_profitability("100x1x0", "peer-a", 2_000_000, 0, 30),
        }
        flow = {
            "100x1x0": _make_flow("100x1x0", "peer-a", 2_000_000, 0, 0),
        }
        analyzer = _make_analyzer(profitability, flow)

        fleet = analyzer.analyze()

        assert fleet.channel_efficiencies["100x1x0"].dead_capital_stage == "none"

    def test_rpsd_uses_msat_precision_not_ceiling_sat_boundary(self):
        analyzer = _make_analyzer({}, {})
        prof = SimpleNamespace(
            capacity_sats=1_000_000,
            revenue=ChannelRevenue(
                channel_id="100x1x0",
                fees_earned_msat=1,
                volume_routed_msat=0,
                forward_count=0,
            ),
        )

        assert analyzer._calculate_rpsd(prof) == pytest.approx(0.001)

    def test_equal_rpsd_channels_share_same_percentile_rank(self):
        profitability = {
            "100x1x0": _make_profitability("100x1x0", "peer-a", 1_000_000, 100, 30),
            "100x2x0": _make_profitability("100x2x0", "peer-b", 1_000_000, 100, 30),
            "100x3x0": _make_profitability("100x3x0", "peer-c", 1_000_000, 0, 30),
        }
        flow = {
            "100x1x0": _make_flow("100x1x0", "peer-a", 1_000_000, 5, 100_000),
            "100x2x0": _make_flow("100x2x0", "peer-b", 1_000_000, 5, 100_000),
            "100x3x0": _make_flow("100x3x0", "peer-c", 1_000_000, 0, 0),
        }
        analyzer = _make_analyzer(profitability, flow)

        fleet = analyzer.analyze()

        assert fleet.channel_efficiencies["100x1x0"].efficiency_rank == pytest.approx(0.75)
        assert fleet.channel_efficiencies["100x2x0"].efficiency_rank == pytest.approx(0.75)

    def test_stage_lookup_normalizes_scid_keys(self):
        profitability = {
            "100:1:0": _make_profitability("100:1:0", "peer-a", 2_000_000, 0, 30),
        }
        flow = {
            "100x1x0": _make_flow("100x1x0", "peer-a", 2_000_000, 0, 0),
        }
        analyzer = _make_analyzer(profitability, flow, stages={"100x1x0": {"stage": "fee_reduction", "entered_at": 123}})

        fleet = analyzer.analyze()

        assert fleet.channel_efficiencies["100:1:0"].dead_capital_stage == "fee_reduction"

    def test_stage_lookup_degrades_cleanly_on_db_failure(self):
        profitability = {
            "100x1x0": _make_profitability("100x1x0", "peer-a", 2_000_000, 0, 30),
        }
        flow = {
            "100x1x0": _make_flow("100x1x0", "peer-a", 2_000_000, 0, 0),
        }
        analyzer = _make_analyzer(profitability, flow)
        analyzer._database.get_dead_capital_stages.side_effect = RuntimeError("db unavailable")

        fleet = analyzer.analyze()

        assert fleet.channel_efficiencies["100x1x0"].dead_capital_stage == "none"
