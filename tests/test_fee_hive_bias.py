"""Tests for hive hint bias integration in fee controller."""

import time
import pytest
from unittest.mock import MagicMock

from modules.fee_controller import FeeController
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
    c.min_fee_ppm = 10
    c.max_fee_ppm = 5000
    c.vegas_decay_rate = 0.95
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


class TestFeeHiveBias:
    def test_get_hive_fee_bias_with_adapter(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02aabb": {
                    "corridor_role": "owner",
                    "competition_bias": 1,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        fc.hive_hints = adapter
        bias = fc._get_hive_fee_bias("02aabb")
        assert bias > 1.0
        assert bias <= 1.1

    def test_get_hive_fee_bias_no_adapter(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = None
        assert fc._get_hive_fee_bias("02aabb") == 1.0

    def test_get_hive_fee_bias_exception_returns_neutral(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.side_effect = Exception("boom")
        fc.hive_hints = adapter
        assert fc._get_hive_fee_bias("02aabb") == 1.0

    def test_bias_within_hard_cap(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.return_value = 1.5
        fc.hive_hints = adapter
        bias = fc._get_hive_fee_bias("02aabb")
        assert 0.9 <= bias <= 1.1
