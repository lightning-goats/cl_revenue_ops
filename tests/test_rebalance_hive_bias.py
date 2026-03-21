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
