"""Tests for hive hint bias integration in rebalancer."""

import time
import pytest
from unittest.mock import MagicMock

from modules.rebalancer import EVRebalancer
from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_config():
    c = MagicMock()
    c.max_concurrent_jobs = 3
    c.sling_job_timeout_seconds = 300
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


class TestRebalanceHiveBias:
    def test_get_hive_rebalance_bias_with_adapter(self, mock_plugin, mock_config, mock_database):
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02aabb": {
                    "rebalance_preference": "sink",
                    "peer_quality_score": 0.9,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        reb.hive_hints = adapter
        bias = reb._get_hive_rebalance_bias("02aabb")
        assert bias > 1.0
        assert bias <= 1.15

    def test_get_hive_rebalance_bias_no_adapter(self, mock_plugin, mock_config, mock_database):
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        reb.hive_hints = None
        assert reb._get_hive_rebalance_bias("02aabb") == 1.0

    def test_get_hive_rebalance_bias_exception_returns_neutral(self, mock_plugin, mock_config, mock_database):
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_rebalance_bias.side_effect = Exception("boom")
        reb.hive_hints = adapter
        assert reb._get_hive_rebalance_bias("02aabb") == 1.0

    def test_bias_within_hard_cap(self, mock_plugin, mock_config, mock_database):
        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_rebalance_bias.return_value = 2.0
        reb.hive_hints = adapter
        bias = reb._get_hive_rebalance_bias("02aabb")
        assert 0.85 <= bias <= 1.15

    def test_select_source_candidates_uses_public_rebalance_bias(self, mock_plugin, mock_config, mock_database):
        mock_config.rebalance_min_profit_ppm = 0
        mock_config.rebalance_min_profit = 0

        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        reb.job_manager.get_source_failure_count = MagicMock(return_value=0)
        reb.policy_manager = None
        reb._calculate_turnover_rate = MagicMock(return_value=0.05)

        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_channel_state.return_value = {"state": "balanced"}

        reb.hive_hints = MagicMock(spec=["get_rebalance_bias", "is_hive_member"])
        reb.hive_hints.get_rebalance_bias.side_effect = (
            lambda peer_id: {"peer_a": 1.15, "peer_b": 0.85}[peer_id]
        )
        reb.hive_hints.is_hive_member.return_value = False

        sources = [
            ("200x1x0", {"peer_id": "peer_b", "spendable_sats": 1_000_000, "capacity": 5_000_000, "fee_ppm": 50}, 0.9),
            ("100x1x0", {"peer_id": "peer_a", "spendable_sats": 1_000_000, "capacity": 5_000_000, "fee_ppm": 50}, 0.9),
        ]
        peer_status = {
            "peer_a": {"connected": True},
            "peer_b": {"connected": True},
        }

        candidates = reb._select_source_candidates(
            sources=sources,
            amount_needed=500_000,
            dest_channel="300x1x0",
            dest_outbound_fee_ppm=1000,
            dest_inbound_fee_ppm=100,
            peer_status=peer_status,
        )

        assert [cid for cid, *_ in candidates][:2] == ["100x1x0", "200x1x0"]

    def test_analyze_rebalance_ev_uses_public_corridor_utilization_bias(
        self, mock_plugin, mock_config, mock_database
    ):
        mock_config.flow_window_days = 7
        mock_config.enable_velocity_gate = False
        mock_config.new_channel_grace_days = 7
        mock_config.min_velocity_threshold = 0.01
        mock_config.rebalance_min_amount = 100_000
        mock_config.rebalance_max_amount = 1_000_000
        mock_config.sling_chunk_size_sats = 500_000
        mock_config.enable_kelly = False
        mock_config.rebalance_cooldown_hours = 24
        mock_config.hot_channel_protection_max_rebalance_fee_ppm = 2000
        mock_config.rebalance_min_profit_ppm = 0
        mock_config.rebalance_min_profit = 0

        reb = EVRebalancer(mock_plugin, mock_config, mock_database)
        reb._profitability_analyzer = None
        reb._estimate_inbound_fee = MagicMock(return_value=50)
        reb._get_channel_age_days = MagicMock(return_value=30)
        reb._get_kalman_metrics = MagicMock(return_value={"velocity": 0.0, "uncertainty": 0.0})
        reb._compute_hot_channel_protection = MagicMock(return_value={})
        reb._estimate_expected_fee_sats = MagicMock(return_value=10)
        reb._calculate_turnover_rate = MagicMock(return_value=0.1)
        reb.job_manager.get_source_failure_count = MagicMock(return_value=0)
        reb.database.get_channel_state.return_value = {
            "state": "balanced",
            "sats_in": 7_000_000,
            "sats_out": 7_000_000,
        }
        reb.database.get_fee_strategy_state.return_value = {"last_broadcast_fee_ppm": 500}
        reb.database.get_peer_uptime_percent.return_value = 100.0
        reb.database.get_failure_count.return_value = (0, None)
        reb.database.get_failure_metadata.return_value = {"last_attempted_ppm": 0}
        reb.database.list_hot_channel_protection_override_peers.return_value = []
        reb._select_source_candidates = MagicMock(
            return_value=[
                (
                    "111x1x0",
                    {
                        "peer_id": "source_peer",
                        "capacity": 5_000_000,
                        "spendable_sats": 4_000_000,
                        "fee_ppm": 50,
                    },
                    1.0,
                    50,
                )
            ]
        )

        dest_info = {
            "peer_id": "dest_peer",
            "capacity": 5_000_000,
            "spendable_sats": 1_000_000,
            "fee_ppm": 500,
        }
        sources = [("111x1x0", {"peer_id": "source_peer", "capacity": 5_000_000, "spendable_sats": 4_000_000, "fee_ppm": 50}, 0.8)]

        baseline = reb._analyze_rebalance_ev(
            dest_channel="222x1x0",
            dest_info=dest_info,
            dest_ratio=0.2,
            sources=sources,
            peer_status={"dest_peer": {"connected": True}, "source_peer": {"connected": True}},
            cfg=mock_config,
        )

        reb.hive_hints = MagicMock(spec=["get_corridor_utilization_bias", "is_hive_member"])
        reb.hive_hints.get_corridor_utilization_bias.return_value = 1.10
        reb.hive_hints.is_hive_member.return_value = False
        with_bias = reb._analyze_rebalance_ev(
            dest_channel="222x1x0",
            dest_info=dest_info,
            dest_ratio=0.2,
            sources=sources,
            peer_status={"dest_peer": {"connected": True}, "source_peer": {"connected": True}},
            cfg=mock_config,
        )

        assert baseline is not None
        assert with_bias is not None
        assert with_bias.expected_profit_sats > baseline.expected_profit_sats
